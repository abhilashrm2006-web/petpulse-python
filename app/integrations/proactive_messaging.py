"""Confirmed live (2026-08-27): every scheduler job that proactively
messages a customer (vaccination reminders, new-parent followups, the 48h
gone-quiet reengagement nudge, the 24h stuck-onboarding reminder, the
price-objection-silence nudge) sent plain free-form text via
WhatsAppClient.send_text -- which WhatsApp silently accepts, then fails to
actually deliver (error 131047, "more than 24 hours have passed since the
customer last replied") for any customer outside its 24h customer-service
session window. Since these are all cron-fired, not replies to something
the customer just said, that's the common case, not an edge case.

send_proactive_message is the single place this is handled: free-form text
when the customer is still inside the window, the approved generic
wrapper template (Settings.whatsapp_generic_nudge_template_name) otherwise.
Falls back to free-form text if that template isn't configured yet, same
as before this existed -- still undeliverable outside the window, but no
worse than the prior behavior, and every call site gets the fix for free
the moment the template is set, with no further code changes."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

SESSION_WINDOW = timedelta(hours=24)


def _is_within_session_window(client, profile_id: str) -> bool:
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
    if not rows:
        return False
    try:
        last = datetime.fromisoformat(rows[0]["created_at"].replace("Z", "+00:00"))
    except (ValueError, AttributeError, KeyError):
        return False
    return (datetime.now(tz=timezone.utc) - last) < SESSION_WINDOW


async def send_proactive_message(ctx: Any, profile_id: str, phone: str, text: str) -> None:
    settings = getattr(ctx, "settings", None)
    template_name = getattr(settings, "whatsapp_generic_nudge_template_name", "") or ""

    # Cheap check first -- skips the messages-table lookup entirely when no
    # template is configured, and keeps every call site working exactly as
    # before (free-form only) until the template is actually set.
    if not template_name or _is_within_session_window(ctx.supabase, profile_id):
        await ctx.whatsapp.send_text(phone, text)
        return

    template_language = getattr(settings, "whatsapp_generic_nudge_template_language", "en") or "en"
    try:
        await ctx.whatsapp.send_template(phone, template_name, template_language, [text])
    except Exception:
        logger.exception("Proactive template send failed for phone=%s, falling back to free-form text", phone)
        await ctx.whatsapp.send_text(phone, text)
