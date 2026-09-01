"""One-time broadcast: a personal feedback-survey ask (from Abhilash) sent
to every customer profile that has ever messaged the bot, regardless of
onboarding status or activity recency. Uses a dedicated, fully static
template (petpulse_feedback_survey_template_name -- no {{n}} variables,
same exact wording for every recipient) submitted 2026-09-01, since almost
all of this list is outside WhatsApp's 24h free-text session window.

Usage:
    python -m scripts.send_feedback_survey --dry-run
    python -m scripts.send_feedback_survey            # live send
"""

import argparse
import asyncio
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import get_settings
from app.integrations.supabase_client import make_supabase_client
from app.integrations.whatsapp import WhatsAppClient

# Kept in sync with the exact message the user asked to send -- do not
# reword without also resubmitting/re-approving the template.
SURVEY_MESSAGE = (
    "Hi! This is Abhilash from PetPulse \U0001F43E\n\n"
    "You were one of our very first users when we launched Pulsy — genuinely means a lot that you gave "
    "it a shot early on.\n\n"
    "I'm not writing to sell you anything. I just want to make Pulsy actually useful for pet parents like "
    "you, and the only way to do that is to hear honestly what worked and what didn't for you.\n\n"
    "Would you mind sharing a few minutes? No wrong answers, just your real experience:\n"
    "https://forms.gle/rXm1aFZTxPaogr4o7\n\n"
    "Whatever you share goes straight into fixing the product — and I really appreciate you being one of "
    "the first to try it."
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def main(dry_run: bool) -> None:
    settings = get_settings()
    supabase = make_supabase_client(settings)

    profiles = supabase.table("profiles").select("id,phone_number,full_name").eq("role", "customer").execute().data or []
    print(f"Recipients (every customer profile that has ever messaged the bot): {len(profiles)}")

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    if dry_run:
        preview = [{"profile_id": p["id"], "phone_number": p["phone_number"], "message": SURVEY_MESSAGE} for p in profiles]
        _write_csv(out_dir / "feedback_survey_dry_run_preview.csv", preview)
        print(f"[DRY RUN] would send to {len(preview)} recipients. Sample message:\n{SURVEY_MESSAGE}\n")
        print(f"Full recipient list: {out_dir / 'feedback_survey_dry_run_preview.csv'}")
        return

    if not settings.whatsapp_feedback_survey_template_name:
        print(
            "No approved WhatsApp template configured -- refusing to send live. "
            "Set WHATSAPP_FEEDBACK_SURVEY_TEMPLATE_NAME once petpulse_early_user_feedback_survey is "
            "approved in Meta Business Manager, or re-run with --dry-run.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    results = []
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        whatsapp = WhatsAppClient(settings, http_client)
        for profile in profiles:
            phone = profile["phone_number"]
            try:
                await whatsapp.send_template(
                    phone,
                    settings.whatsapp_feedback_survey_template_name,
                    settings.whatsapp_feedback_survey_template_language,
                    [],
                )
                print(f"[SENT] {phone}")
                results.append({"profile_id": profile["id"], "phone_number": phone, "status": "sent"})
            except Exception as exc:
                print(f"[FAILED] {phone}: {exc}")
                results.append({"profile_id": profile["id"], "phone_number": phone, "status": "failed", "error": str(exc)[:300]})

    _write_csv(out_dir / f"feedback_survey_results_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}.csv", results)
    sent = sum(1 for r in results if r["status"] == "sent")
    print(f"\nDone: {sent}/{len(results)} sent successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
