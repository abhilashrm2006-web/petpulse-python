"""Sends a personal thank-you (from Abhilash) to ONE specific customer who
submitted the early-user feedback survey. Not a broadcast: the survey Form
itself collects no name/email/phone, so there is no way to automatically
tell which customer a given response came from -- the admin identifies the
respondent some other way (recognizing their story, a separate message,
etc.) and passes their phone number here explicitly, one at a time.

Uses the approved petpulse_feedback_thankyou template when the customer is
outside WhatsApp's 24h session window, free-form text otherwise (see
app.integrations.proactive_messaging.send_proactive_message) -- same
window-aware behavior as every other proactive send in this codebase.

Usage:
    python -m scripts.send_feedback_thankyou 919876543210 --dry-run
    python -m scripts.send_feedback_thankyou 919876543210            # live send
"""

import argparse
import asyncio
import sys

import httpx
from openai import AsyncOpenAI

from app.config import get_settings
from app.deps import AppContext
from app.integrations.proactive_messaging import send_proactive_message
from app.integrations.supabase_client import get_profile_by_phone, make_supabase_client
from app.integrations.whatsapp import WhatsAppClient

# Kept in sync with the exact message the user asked to send -- do not
# reword without also resubmitting/re-approving the template.
THANKYOU_MESSAGE = (
    "Hey! Thank you so much for taking the time to share that — I read through it myself.\n\n"
    "Feedback like yours is exactly what helps us fix Pulsy for real, not just guess at what's wrong. "
    "I can't promise everything changes overnight, but I promise it's going into how we build the next "
    "version.\n\n"
    "If anything specific comes up with your pet in the meantime, feel free to just message here — I'd "
    "genuinely like Pulsy to be useful to you again.\n\n"
    "Thanks again for being one of our first users \U0001F43E"
)


async def main(phone: str, dry_run: bool) -> None:
    settings = get_settings()
    supabase = make_supabase_client(settings)

    profile = get_profile_by_phone(supabase, phone)
    if not profile:
        print(f"No profile found for phone={phone!r} -- check the number (with country code, no +/spaces).", file=sys.stderr)
        raise SystemExit(1)

    if dry_run:
        print(f"[DRY RUN] would send to {phone} ({profile.get('full_name') or 'no name on file'}):\n{THANKYOU_MESSAGE}")
        return

    if not settings.whatsapp_feedback_thankyou_template_name:
        print(
            "No approved WhatsApp template configured -- if this customer is outside the 24h session "
            "window the send will fail silently on WhatsApp's end. Set "
            "WHATSAPP_FEEDBACK_THANKYOU_TEMPLATE_NAME once petpulse_feedback_thankyou is approved, "
            "or re-run with --dry-run.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        whatsapp = WhatsAppClient(settings, http_client)
        ctx = AppContext(settings=settings, http=http_client, whatsapp=whatsapp, supabase=supabase, openai=AsyncOpenAI(api_key=settings.openai_api_key))
        await send_proactive_message(ctx, profile["id"], phone, THANKYOU_MESSAGE)
    print(f"[SENT] {phone}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone", help="Customer phone number with country code, e.g. 919876543210")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.phone, dry_run=args.dry_run))
