"""Ports the 3 standalone cron workflows (spec §9), plus the T-3..T-0
vaccination countdown and the 48h re-engagement nudge added later. These are
all proactive batch sends, not reactions to an incoming query, so they stay
outside the agent loop — the user's "LLM decides actions" directive is about
inbound messages, not scheduled reminders (matches n8n, which also sends
these as plain templated text, not LLM-composed)."""

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

from app.availability.slots import IST
from app.deps import AppContext
from app.integrations import google_drive
from app.integrations.supabase_client import get_pet_member_contacts, is_active_subscriber, is_unique_violation
from app.media_pipeline.doctor_documents import extract_doctor_fields

logger = logging.getLogger(__name__)

DOCTOR_DRIVE_SYNC_MIME_ALLOWLIST = {"application/pdf", "image/jpeg", "image/png", "image/heic", "image/webp"}

REENGAGEMENT_SILENCE_THRESHOLD = timedelta(hours=48)
REENGAGEMENT_COOLDOWN = timedelta(days=7)
ONBOARDING_STUCK_THRESHOLD = timedelta(hours=24)
ONBOARDING_REMINDER_COOLDOWN = timedelta(days=1)
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


def _onboarding_reminder_text(profile: dict) -> str:
    name = profile.get("full_name")
    greeting = f"Hey {name.split()[0]}!" if name else "Hey there!"
    return (
        f"{greeting} Looks like you didn't quite finish setting up your pet's profile with us. "
        "Just reply here whenever you're ready and we'll pick up right where you left off — "
        "only takes a couple of minutes 🐾"
    )


async def send_onboarding_reminders(ctx: AppContext) -> None:
    """A customer who started the signup flow (registration_step is set,
    e.g. by app.ingestion.registration) but has gone quiet mid-flow for
    ONBOARDING_STUCK_THRESHOLD (24h) gets one nudge to come back and finish,
    then at most one more every ONBOARDING_REMINDER_COOLDOWN (1 day) if
    they're still stuck -- same claim-before-send shape as
    send_reengagement_nudges, but gated on "mid-signup", not "already
    onboarded and gone silent". Unlike that job, a null last_active_at here
    is NOT "never engaged" (they had to message at least once to get a
    registration_step at all) -- it just means they started before
    profiles.last_active_at existed, so they're treated as stuck too."""
    client = ctx.supabase
    now = datetime.now(tz=timezone.utc)
    stuck_cutoff = now - ONBOARDING_STUCK_THRESHOLD
    cooldown_cutoff = now - ONBOARDING_REMINDER_COOLDOWN

    profiles = client.table("profiles").select("*").eq("role", "customer").execute().data or []
    candidates = [
        p for p in profiles
        if p.get("registration_step") and p["registration_step"] != "completed"
        and (not p.get("last_active_at") or p["last_active_at"] < stuck_cutoff.isoformat())
        and (not p.get("last_onboarding_reminder_sent_at") or p["last_onboarding_reminder_sent_at"] < cooldown_cutoff.isoformat())
    ]

    async def _remind(profile: dict) -> None:
        # Atomic claim BEFORE sending, same idempotency reasoning as every
        # other job here -- prevents a double-nudge if this ever runs on
        # more than one worker/replica.
        claimed = (
            client.table("profiles")
            .update({"last_onboarding_reminder_sent_at": now.isoformat()})
            .eq("id", profile["id"])
            .or_(f"last_onboarding_reminder_sent_at.is.null,last_onboarding_reminder_sent_at.lt.{cooldown_cutoff.isoformat()}")
            .execute()
            .data
        )
        if not claimed:
            return
        phone = profile.get("phone_number")
        if not phone:
            return
        try:
            await ctx.whatsapp.send_text(phone, _onboarding_reminder_text(profile))
        except Exception:
            logger.exception("Failed to send onboarding reminder to %s", phone)
            # Revert the claim so a transient send failure doesn't silently
            # block this customer from ever being reminded again.
            client.table("profiles").update(
                {"last_onboarding_reminder_sent_at": profile.get("last_onboarding_reminder_sent_at")}
            ).eq("id", profile["id"]).execute()

    await asyncio.gather(*(_remind(p) for p in candidates))


def _folder_signature(files: list[dict]) -> str:
    """A cheap "did anything in this folder change since last sync" check --
    the max modifiedTime across its files, string-compared (ISO 8601 sorts
    correctly as a string) against what was stored on the draft last time,
    so an unchanged folder doesn't get re-extracted (a real cost: a vision
    LLM call) on every single run."""
    times = [f.get("modifiedTime", "") for f in files]
    return max(times) if times else ""


async def sync_doctor_onboarding_drafts(ctx: AppContext) -> None:
    """Watches a Google Drive folder of per-doctor subfolders (onboarding
    documents a vet has sent over) and turns each into a draft an admin can
    review, edit, and approve in the admin dashboard (app/admin/routes.py) --
    never creates a real, active doctor account directly. See
    app/media_pipeline/doctor_documents.py for the extraction itself and
    app/integrations/google_drive.py for the read-only Drive access (a
    service account shared on just this one folder)."""
    if not ctx.settings.google_service_account_json or not ctx.settings.doctor_drive_folder_id:
        return  # ops-level: feature not configured yet, skip silently

    client = ctx.supabase
    try:
        subfolders = await google_drive.list_subfolders(ctx.settings, ctx.http, ctx.settings.doctor_drive_folder_id)
    except Exception:
        logger.exception("Failed to list doctor-onboarding Drive subfolders")
        return

    for folder in subfolders:
        try:
            await _sync_one_doctor_folder(ctx, client, folder)
        except Exception:
            logger.exception("Failed to sync doctor-onboarding folder %s (%s)", folder.get("name"), folder.get("id"))


async def _sync_one_doctor_folder(ctx: AppContext, client, folder: dict) -> None:
    folder_id = folder["id"]
    existing_rows = (
        client.table("doctor_onboarding_drafts").select("*").eq("drive_folder_id", folder_id).limit(1).execute().data
    )
    existing = existing_rows[0] if existing_rows else None
    if existing and existing["status"] in ("approved", "rejected"):
        return  # final state -- an admin already acted on this folder, never touch it again

    files = await google_drive.list_files(ctx.settings, ctx.http, folder_id)
    if not files:
        return  # doctor hasn't uploaded anything yet

    signature = _folder_signature(files)
    if existing and existing.get("drive_folder_modified_time") == signature:
        return  # nothing changed since last extraction, skip the LLM call

    downloadable = [f for f in files if f.get("mimeType") in DOCTOR_DRIVE_SYNC_MIME_ALLOWLIST]
    downloaded: list[tuple[bytes, str, str]] = []
    for f in downloadable:
        try:
            data = await google_drive.download_file(ctx.settings, ctx.http, f["id"])
            downloaded.append((data, f["mimeType"], f["name"]))
        except Exception:
            logger.exception("Failed to download doctor-onboarding file %s (%s)", f.get("name"), f.get("id"))

    fields = await extract_doctor_fields(ctx.openai, ctx.settings, downloaded)
    source_filenames = fields.pop("_source_filenames", [])

    payload = {
        "drive_folder_id": folder_id,
        "drive_folder_name": folder["name"],
        "drive_folder_modified_time": signature,
        "status": "pending_review",
        "extracted_full_name": fields.get("full_name"),
        "extracted_phone_number": fields.get("phone_number"),
        "extracted_email": fields.get("email"),
        "extracted_qualification": fields.get("qualification"),
        "extracted_registration_number": fields.get("registration_number"),
        "extracted_specialization": fields.get("specialization"),
        "extracted_gender": fields.get("gender"),
        "extracted_date_of_birth": fields.get("date_of_birth"),
        "extracted_city": fields.get("city"),
        "extracted_area": fields.get("area"),
        "source_files": [{"id": f["id"], "name": f["name"]} for f in files],
        "extraction_notes": fields.get("notes"),
    }
    client.table("doctor_onboarding_drafts").upsert(payload, on_conflict="drive_folder_id").execute()


REMINDER_TYPE_LABELS = {"day_before": "Tomorrow's Schedule", "day_of": "Today's Schedule"}


def _format_appointment_line(row: dict, pet_name: str, customer_name: str) -> str:
    start = datetime.fromisoformat(row["preferred_time"])
    if start.tzinfo is None:
        start = start.replace(tzinfo=IST)
    time_str = start.astimezone(IST).strftime("%I:%M %p").lstrip("0")
    line = f"• {time_str} — {pet_name} ({customer_name})"
    if row.get("case_summary"):
        line += f" — {row['case_summary']}"
    return line


async def send_doctor_schedule_reminders(ctx: AppContext, reminder_type: str) -> None:
    """T-1 (evening before) and T-0 (morning of) appointment-schedule pushes
    for doctors, so a vet has their day planned without needing to message
    the bot first -- unlike the customer-facing reminders above, this is
    purely a proactive push (no "did they reply recently" gating), since a
    working vet needs their schedule regardless of how active they've been
    in chat. Only sent to a doctor who actually has at least one accepted
    session that day -- no "you have nothing scheduled" noise for every
    idle vet, every single day. `reminder_type` is "day_before" (target
    date = tomorrow, meant to run in the evening) or "day_of" (target date
    = today, meant to run in the morning) -- two separate scheduler
    triggers call this with different values (see app/scheduler/runner.py)."""
    client = ctx.supabase
    today = date.today()
    target_date = today + timedelta(days=1) if reminder_type == "day_before" else today
    window_start = datetime.combine(target_date, datetime.min.time(), tzinfo=IST).isoformat()
    window_end = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=IST).isoformat()

    rows = (
        client.table("doctor_sessions")
        .select("id, doctor_phone, preferred_time, case_summary, profile_id, pet_id")
        .eq("status", "accepted")
        .gte("preferred_time", window_start)
        .lt("preferred_time", window_end)
        .order("preferred_time")
        .execute()
        .data
        or []
    )
    if not rows:
        return

    by_doctor: dict[str, list[dict]] = {}
    for row in rows:
        by_doctor.setdefault(row["doctor_phone"], []).append(row)

    for doctor_phone, sessions in by_doctor.items():
        # Atomic claim-by-insert BEFORE sending: this cron fires once per worker
        # process (in-process APScheduler) and once per any future horizontal
        # replica, so without claiming first, every worker/replica would
        # independently see the same day's sessions and send the doctor the
        # same schedule N times. Unlike the vaccination/re-engagement jobs'
        # conditional-UPDATE claim, this is a pure insert-or-conflict claim
        # (same idiom as claim_message_id) since there's no pre-existing row
        # to update -- this table only ever records "was this specific
        # reminder already sent".
        try:
            client.table("doctor_schedule_reminders_sent").insert(
                {"doctor_phone": doctor_phone, "reminder_type": reminder_type, "reminder_date": target_date.isoformat()}
            ).execute()
        except Exception as exc:
            if is_unique_violation(exc):
                continue  # already sent for this doctor/date/type
            logger.exception("Failed to claim schedule reminder for doctor=%s", doctor_phone)
            continue

        lines = []
        for row in sessions:
            profile_rows = client.table("profiles").select("full_name").eq("id", row["profile_id"]).limit(1).execute().data
            customer_name = (profile_rows[0].get("full_name") if profile_rows else None) or "Pet Parent"
            pet_rows = client.table("pets").select("name").eq("id", row["pet_id"]).limit(1).execute().data if row.get("pet_id") else []
            pet_name = (pet_rows[0].get("name") if pet_rows else None) or "Pet"
            lines.append(_format_appointment_line(row, pet_name, customer_name))

        day_label = target_date.strftime("%a %d %b")
        header = f"📅 *{REMINDER_TYPE_LABELS[reminder_type]} — {day_label}*\n({len(sessions)} appointment{'s' if len(sessions) != 1 else ''})\n"
        text = header + "\n".join(lines)

        try:
            await ctx.whatsapp.send_text(doctor_phone, text)
        except Exception:
            logger.exception("Failed to send schedule reminder to doctor=%s", doctor_phone)
            # Revert the claim so a transient send failure doesn't silently
            # skip this doctor's reminder forever -- if this job somehow runs
            # again the same day (e.g. a mid-day restart), it gets retried.
            client.table("doctor_schedule_reminders_sent").delete().eq("doctor_phone", doctor_phone).eq(
                "reminder_type", reminder_type
            ).eq("reminder_date", target_date.isoformat()).execute()
