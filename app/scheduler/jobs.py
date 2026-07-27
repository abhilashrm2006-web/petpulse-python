"""Ports the 3 standalone cron workflows (spec §9). These are proactive
batch sends, not reactions to an incoming query, so they stay outside the
agent loop — the user's "LLM decides actions" directive is about inbound
messages, not scheduled reminders (matches n8n, which also sends these as
plain templated text, not LLM-composed)."""

import logging
from datetime import date, datetime, timedelta

from app.deps import AppContext
from app.integrations.supabase_client import get_pet_member_contacts, is_active_subscriber

logger = logging.getLogger(__name__)


async def send_vaccination_reminders(ctx: AppContext) -> None:
    client = ctx.supabase
    horizon = (date.today() + timedelta(days=7)).isoformat()
    today = date.today().isoformat()

    due = (
        client.table("vaccinations")
        .select("*, pets(name, id)")
        .lte("next_due_date", horizon)
        .neq("reminder_sent", True)
        .in_("status", ["scheduled", "completed", "overdue"])
        .execute()
        .data
        or []
    )

    for vax in due:
        pet = vax.get("pets") or {}
        # pet_members has two FKs into profiles (profile_id and added_by) --
        # get_pet_member_contacts already disambiguates that embed correctly
        # (confirmed live); the raw query this used to run here doesn't and
        # made this entire job crash for every vaccination.
        members = get_pet_member_contacts(client, vax["pet_id"])
        overdue = vax.get("next_due_date") and vax["next_due_date"] < today
        status_word = "overdue" if overdue else "due soon"
        pet_name = pet.get("name", "Your pet")
        basic_text = f"Reminder: {pet_name}'s {vax['vaccine_name']} vaccination is {status_word} ({vax.get('next_due_date')})."
        # Subscriber reminder carries the extra detail the full passport has
        # (manufacturer/batch) when on file -- Free gets the basic due-date
        # ping either way, this is a content upgrade, not a send gate.
        detail_parts = [p for p in (vax.get("manufacturer"), f"Batch/Lot: {vax['batch_number']}" if vax.get("batch_number") else None) if p]
        full_text = basic_text + (f" ({' | '.join(detail_parts)})" if detail_parts else "")

        for member in members:
            phone = member.get("phone_number")
            profile_id = member.get("profile_id")
            if not phone or not profile_id:
                continue
            text = full_text if is_active_subscriber(client, profile_id) else basic_text
            try:
                await ctx.whatsapp.send_text(phone, text)
            except Exception:
                logger.exception("Failed to send vaccination reminder to %s", phone)

        client.table("vaccinations").update(
            {"reminder_sent": True, "status": "overdue" if overdue else vax["status"]}
        ).eq("id", vax["id"]).execute()


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
        profile_rows = client.table("profiles").select("phone_number").eq("id", followup["profile_id"]).limit(1).execute().data
        phone = profile_rows[0]["phone_number"] if profile_rows else None
        if phone:
            try:
                await ctx.whatsapp.send_text(phone, followup["message_text"])
                client.table("new_parent_followups").update({"status": "sent"}).eq("id", followup["id"]).execute()
            except Exception:
                logger.exception("Failed to send new-parent followup to %s", phone)
