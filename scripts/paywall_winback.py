"""Targeted win-back for the 2 customers identified in the 2026-08-04
root-cause CSV who went quiet after hitting a paywall that no longer exists:

  - 917575067677 ("In hindi ?"): told Hindi needed a subscription.
  - 919674590793 (Amrapali Roy): told consultations "aren't free on the
    Free plan".

Both restrictions were removed on 2026-08-03 (see the subscription-tier
removal commit). Per item 3 of the workstream: verify against LIVE config
before messaging, never claim consultations are free (they're a flat fee,
pulled from settings so this can't go stale), and reuse the same
dry-run/template policy as scripts/reengage_stranded_onboarding.py.

Usage:
    python -m scripts.paywall_winback --dry-run
    python -m scripts.paywall_winback            # live send
"""

import argparse
import asyncio
import sys

import httpx

from app.config import get_settings
from app.integrations.supabase_client import get_profile_by_phone
from app.integrations.whatsapp import WhatsAppClient

TARGET_PHONES = ["917575067677", "919674590793"]


def build_winback_message(settings, first_name: str) -> str:
    """Verified against live config (app.config.Settings), not the CSV
    notes -- consultation price is read from
    settings.razorpay_consult_fee_inr so this message can never go stale
    if pricing changes later. Never states or implies consultations are
    free."""
    fee = settings.razorpay_consult_fee_inr
    return (
        f"Hi {first_name}! Quick update since we last spoke — PetPulse is now fully free for everyday use: "
        "unlimited AI health chats, symptom checks, emergency triage, document vault, multiple pets, and the "
        "vet finder (including Hindi and other regional languages) all work with no plan or subscription needed. "
        f"The only paid part is booking an actual doctor consultation, a flat ₹{fee}/visit, no plan confusion. "
        "Jump back in and ask me anything about your pet's health!"
    )


async def main(dry_run: bool) -> None:
    settings = get_settings()
    from app.integrations.supabase_client import make_supabase_client

    supabase = make_supabase_client(settings)

    targets = []
    for phone in TARGET_PHONES:
        profile = get_profile_by_phone(supabase, phone)
        if not profile:
            print(f"[SKIP] {phone}: no profile found")
            continue
        first_name = (profile.get("full_name") or "").split()[0] if profile.get("full_name") else "there"
        targets.append((profile, build_winback_message(settings, first_name)))

    if dry_run:
        for profile, message in targets:
            print(f"[DRY RUN] would send to {profile['phone_number']}:\n{message}\n")
        return

    if not settings.whatsapp_reengagement_template_name:
        print(
            "No approved WhatsApp template configured -- refusing to send live "
            "(same policy as scripts/reengage_stranded_onboarding.py). "
            "Set WHATSAPP_REENGAGEMENT_TEMPLATE_NAME once one is approved, or re-run with --dry-run.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        whatsapp = WhatsAppClient(settings, http_client)
        for profile, message in targets:
            try:
                await whatsapp.send_template(
                    profile["phone_number"],
                    settings.whatsapp_reengagement_template_name,
                    settings.whatsapp_reengagement_template_language,
                    [message],
                )
                print(f"[SENT] {profile['phone_number']}")
            except Exception as exc:
                print(f"[FAILED] {profile['phone_number']}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
