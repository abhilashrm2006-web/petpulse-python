"""Ports the 3 standalone cron workflows (spec §9), plus the T-3..T-0
vaccination countdown and the 48h re-engagement nudge added later. These are
all proactive batch sends, not reactions to an incoming query, so they stay
outside the agent loop — the user's "LLM decides actions" directive is about
inbound messages, not scheduled reminders (matches n8n, which also sends
these as plain templated text, not LLM-composed)."""

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

from app.deps import AppContext
from app.integrations.supabase_client import get_pet_member_contacts, is_active_subscriber

logger = logging.getLogger(__name__)

REENGAGEMENT_SILENCE_THRESHOLD = timedelta(hours=48)
REENGAGEMENT_COOLDOWN = timedelta(days=7)
# How many profiles' "last active" lookup to run concurrently -- one query per
# customer would take far too long run one-at-a-time at real scale (tens of
# thousands of customers), but hammering Supabase with all of them at once
# isn't safe either; a bounded batch is the middle ground.
REENGAGEMENT_LOOKUP_CONCURRENCY = 20

_COUNTDOWN_PHRASES = {3: "is due in 3 days", 2: "is due in 2 days", 1: "is due tomorrow", 0: "is due today"}


async def send_vaccination_reminders(ctx: AppContext) -> None:
    """Two independent concerns, kept separate on purpose:
    1. A daily countdown (T-3, T-2, T-1, T-0) leading up to the due date, so
       the reminder isn't just a single easy-to-miss ping — tracked via
       vaccinations.last_reminder_offset_sent (the smallest countdown offset
       already sent; the next offset must be strictly smaller, since offsets
       only ever count down as days pass).
    2. The existing one-shot "this is now overdue" ping once the due date has
       actually passed, tracked via the original reminder_sent boolean,
       unchanged from before."""
    client = ctx.supabase
    today = date.today()
    today_iso = today.isoformat()

    await _send_countdown_reminders(ctx, client, today)
    await _send_overdue_reminders(ctx, client, today_iso)


async def _send_countdown_reminders(ctx: AppContext, client, today: date) -> None:
    horizon = (today + timedelta(days=3)).isoformat()
    due = (
        client.table("vaccinations")
        .select("*, pets(name, id)")
        .gte("next_due_date", today.isoformat())
        .lte("next_due_date", horizon)
        .in_("status", ["scheduled", "completed"])
        .execute()
        .data
        or []
    )

    for vax in due:
        offset = (date.fromisoformat(vax["next_due_date"]) - today).days
        if offset not in _COUNTDOWN_PHRASES:
            continue  # defensive -- query bounds should already guarantee 0-3

        # Atomic claim BEFORE sending, same reasoning as the payment-webhook
        # and overdue-reminder idempotency fixes: this cron fires once per
        # worker process (in-process APScheduler) and, if ever run on more
        # than one Railway replica, once per replica too. The claim's WHERE
        # clause (offset must be strictly smaller than whatever was last
        # sent, since T-3/T-2/T-1/T-0 only ever counts down) makes Postgres
        # the single arbiter, and also naturally prevents re-sending the same
        # day's offset if the job somehow ran twice today.
        claimed = (
            client.table("vaccinations")
            .update({"last_reminder_offset_sent": offset})
            .eq("id", vax["id"])
            .or_(f"last_reminder_offset_sent.is.null,last_reminder_offset_sent.gt.{offset}")
            .execute()
            .data
        )
        if not claimed:
            continue

        pet = vax.get("pets") or {}
        pet_name = pet.get("name", "Your pet")
        phrase = _COUNTDOWN_PHRASES[offset]
        basic_text = f"Reminder: {pet_name}'s {vax['vaccine_name']} vaccination {phrase} ({vax.get('next_due_date')})."
        detail_parts = [p for p in (vax.get("manufacturer"), f"Batch/Lot: {vax['batch_number']}" if vax.get("batch_number") else None) if p]
        full_text = basic_text + (f" ({' | '.join(detail_parts)})" if detail_parts else "")

        members = get_pet_member_contacts(client, vax["pet_id"])
        any_sent = False
        for member in members:
            phone = member.get("phone_number")
            profile_id = member.get("profile_id")
            if not phone or not profile_id:
                continue
            text = full_text if is_active_subscriber(client, profile_id) else basic_text
            try:
                await ctx.whatsapp.send_text(phone, text)
                any_sent = True
            except Exception:
                logger.exception("Failed to send vaccination countdown reminder to %s", phone)

        if not any_sent and members:
            # A transient WhatsApp outage must not silently skip this day's
            # reminder forever -- revert to the previous value (or None) so
            # this SAME offset is retried on the next run today/soon, rather
            # than being permanently marked as sent when it never went out.
            previous = vax.get("last_reminder_offset_sent")
            client.table("vaccinations").update({"last_reminder_offset_sent": previous}).eq("id", vax["id"]).execute()


async def _send_overdue_reminders(ctx: AppContext, client, today_iso: str) -> None:
    due = (
        client.table("vaccinations")
        .select("*, pets(name, id)")
        .lt("next_due_date", today_iso)
        .neq("reminder_sent", True)
        .in_("status", ["scheduled", "completed", "overdue"])
        .execute()
        .data
        or []
    )

    for vax in due:
        # Atomic claim BEFORE sending: see _send_countdown_reminders for the
        # full reasoning (same multi-worker/replica idempotency concern).
        claimed = (
            client.table("vaccinations")
            .update({"reminder_sent": True})
            .eq("id", vax["id"])
            .eq("reminder_sent", False)
            .execute()
            .data
        )
        if not claimed:
            continue

        pet = vax.get("pets") or {}
        pet_name = pet.get("name", "Your pet")
        basic_text = f"Reminder: {pet_name}'s {vax['vaccine_name']} vaccination is overdue ({vax.get('next_due_date')})."
        detail_parts = [p for p in (vax.get("manufacturer"), f"Batch/Lot: {vax['batch_number']}" if vax.get("batch_number") else None) if p]
        full_text = basic_text + (f" ({' | '.join(detail_parts)})" if detail_parts else "")

        members = get_pet_member_contacts(client, vax["pet_id"])
        any_sent = False
        for member in members:
            phone = member.get("phone_number")
            profile_id = member.get("profile_id")
            if not phone or not profile_id:
                continue
            text = full_text if is_active_subscriber(client, profile_id) else basic_text
            try:
                await ctx.whatsapp.send_text(phone, text)
                any_sent = True
            except Exception:
                logger.exception("Failed to send overdue vaccination reminder to %s", phone)

        if any_sent or not members:
            client.table("vaccinations").update({"status": "overdue"}).eq("id", vax["id"]).execute()
        else:
            # Same reasoning as before: a total send failure must not
            # permanently exclude this vaccination from every future run.
            client.table("vaccinations").update({"reminder_sent": False}).eq("id", vax["id"]).execute()


async def retain_chat_history(ctx: AppContext, keep_per_session: int = 60) -> None:
    """Ports the weekly `ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY
    id DESC)` delete, keeping the newest N rows per session. Implemented as
    plain PostgREST calls (per-session id lookup + delete) rather than a raw
    SQL statement, since PostgREST has no generic "run this SQL" endpoint."""
    client = ctx.supabase
    sessions = client.table("n8n_chat_history_petpulse").select("session_id").execute().data or []
    distinct_sessions = {row["session_id"] for row in sessions}

    for session_id in distinct_sessions:
        ids = (
            client.table("n8n_chat_history_petpulse")
            .select("id")
            .eq("session_id", session_id)
            .order("id", desc=True)
            .execute()
            .data
            or []
        )
        stale_ids = [row["id"] for row in ids[keep_per_session:]]
        if stale_ids:
            client.table("n8n_chat_history_petpulse").delete().in_("id", stale_ids).execute()


async def send_new_parent_followups(ctx: AppContext) -> None:
    client = ctx.supabase
    now_iso = datetime.utcnow().isoformat()

    due = (
        client.table("new_parent_followups")
        .select("*")
        .eq("status", "pending")
        .lte("due_at", now_iso)
        .execute()
        .data
        or []
    )

    for followup in due:
        # Same atomic-claim reasoning as send_vaccination_reminders -- prevents
        # the same followup being sent twice if this job ever runs on more than
        # one worker/replica.
        claimed = (
            client.table("new_parent_followups")
            .update({"status": "sending"})
            .eq("id", followup["id"])
            .eq("status", "pending")
            .execute()
            .data
        )
        if not claimed:
            continue

        profile_rows = client.table("profiles").select("phone_number").eq("id", followup["profile_id"]).limit(1).execute().data
        phone = profile_rows[0]["phone_number"] if profile_rows else None
        if phone:
            try:
                await ctx.whatsapp.send_text(phone, followup["message_text"])
                client.table("new_parent_followups").update({"status": "sent"}).eq("id", followup["id"]).execute()
            except Exception:
                logger.exception("Failed to send new-parent followup to %s", phone)
                # Revert the claim so this isn't silently lost forever -- next run retries it.
                client.table("new_parent_followups").update({"status": "pending"}).eq("id", followup["id"]).execute()
        else:
            client.table("new_parent_followups").update({"status": "pending"}).eq("id", followup["id"]).execute()


def _reengagement_text(profile: dict) -> str:
    name = profile.get("full_name")
    greeting = f"Hey {name.split()[0]}!" if name else "Hey there!"
    return (
        f"{greeting} Haven't heard from you in a couple of days — just checking in. "
        "Anything going on with your pet I can help with, or any questions on your mind? "
        "I'm here 24/7 whenever you need me 🐾"
    )


async def _last_customer_activity(client, profile_id: str) -> str | None:
    """Derived from existing message history (no extra column needed for
    this half) -- the most recent inbound ("user") message this profile
    sent, across any conversation. None means they've never sent a message
    at all (a brand-new/never-onboarded number), which this job treats as
    out of scope -- there's nothing to "re-engage" if they never engaged."""
    rows = (
        client.table("messages")
        .select("created_at")
        .eq("profile_id", profile_id)
        .eq("sender_type", "user")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    return rows[0]["created_at"] if rows else None


async def send_reengagement_nudges(ctx: AppContext) -> None:
    """A customer who's gone quiet for 48h+ gets one proactive check-in, then
    at most one more every REENGAGEMENT_COOLDOWN (7 days) if they're still
    silent -- never every single day, which would just read as spam.
    Deliberately excludes vets (this is a customer-retention nudge, not
    something a working vet on the scheduling/relay line needs) and anyone
    who's never sent a single message (see _last_customer_activity)."""
    client = ctx.supabase
    now = datetime.now(tz=timezone.utc)
    silence_cutoff = now - REENGAGEMENT_SILENCE_THRESHOLD
    cooldown_cutoff = now - REENGAGEMENT_COOLDOWN

    profiles = client.table("profiles").select("*").eq("role", "customer").execute().data or []
    # Skip anyone nudged within the cooldown window before even checking their
    # activity -- avoids an unnecessary messages-table lookup for the (likely
    # common, once this job has been running a while) case of "already
    # nudged recently".
    candidates = [
        p for p in profiles
        if not p.get("last_reengagement_sent_at") or p["last_reengagement_sent_at"] < cooldown_cutoff.isoformat()
    ]

    semaphore = asyncio.Semaphore(REENGAGEMENT_LOOKUP_CONCURRENCY)

    async def _check_and_nudge(profile: dict) -> None:
        async with semaphore:
            last_active = await _last_customer_activity(client, profile["id"])
        if last_active is None or last_active >= silence_cutoff.isoformat():
            return  # never messaged at all, or still within the silence window

        # Atomic claim BEFORE sending, same idempotency reasoning as every
        # other job here -- prevents a double-nudge if this ever runs on
        # more than one worker/replica.
        claimed = (
            client.table("profiles")
            .update({"last_reengagement_sent_at": now.isoformat()})
            .eq("id", profile["id"])
            .or_(f"last_reengagement_sent_at.is.null,last_reengagement_sent_at.lt.{cooldown_cutoff.isoformat()}")
            .execute()
            .data
        )
        if not claimed:
            return

        phone = profile.get("phone_number")
        if not phone:
            return
        try:
            await ctx.whatsapp.send_text(phone, _reengagement_text(profile))
        except Exception:
            logger.exception("Failed to send re-engagement nudge to %s", phone)
            # Revert the claim so a transient send failure doesn't silently
            # block this customer from ever being nudged again.
            client.table("profiles").update({"last_reengagement_sent_at": profile.get("last_reengagement_sent_at")}).eq("id", profile["id"]).execute()

    await asyncio.gather(*(_check_and_nudge(p) for p in candidates))
