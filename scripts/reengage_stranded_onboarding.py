"""One-time broadcast: nudges customers stranded in the pre-2026-08-03
longer onboarding wizard (registration_step in a step that no longer exists
in the current 3-question flow) to send any message and auto-resume. See
app/reengagement/stranded_onboarding.py for the segmentation/filtering
logic (unit-tested there) -- this script is the thin I/O shell: query,
segment, send-or-preview, record results.

Usage:
    python -m scripts.reengage_stranded_onboarding --dry-run
    python -m scripts.reengage_stranded_onboarding            # live send

Live sending REQUIRES settings.whatsapp_reengagement_template_name to be
set to an approved Meta template -- see the docstring on
app.integrations.whatsapp.WhatsAppClient.send_template. Free-form text is
not used here: many of these customers haven't messaged in days, well
outside WhatsApp's 24h customer-service session window, so a template is
the only reliably deliverable option (see the workstream's item 1).

Suggested template body text to submit for approval (one template covers
both the "has a pet on file" and "no pet yet" cases via {{2}}):

    Hi {{1}}! We've made signing up for PetPulse quicker — just 3 short
    questions now. Reply anything to pick up right where you left off
    with {{2}}. Takes under a minute!

Call with body_params=[first_name, "Bruno's profile"] or
body_params=[first_name, "your profile"] to reproduce Message A / Message B
from the workstream spec exactly.
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import get_settings
from app.integrations.supabase_client import make_supabase_client
from app.integrations.whatsapp import WhatsAppClient
from app.reengagement.stranded_onboarding import Candidate, build_message, segment

logger = logging.getLogger(__name__)


def _pet_clause(candidate: Candidate) -> str:
    return f"{candidate.pet_name}'s profile" if candidate.pet_name else "your profile"


async def _send_one(whatsapp: WhatsAppClient, settings, candidate: Candidate) -> dict:
    try:
        await whatsapp.send_template(
            candidate.phone_number,
            settings.whatsapp_reengagement_template_name,
            settings.whatsapp_reengagement_template_language,
            [candidate.full_name.split()[0] if candidate.full_name else "there", _pet_clause(candidate)],
        )
        return {"profile_id": candidate.profile_id, "phone_number": candidate.phone_number, "status": "sent"}
    except Exception as exc:
        logger.exception("Failed to send stranded-onboarding nudge to %s", candidate.phone_number)
        return {"profile_id": candidate.profile_id, "phone_number": candidate.phone_number, "status": "failed", "error": str(exc)[:300]}


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

    profiles = (
        supabase.table("profiles")
        # Disambiguated FK name -- profiles<->pets has two possible embed
        # paths (direct pets.profile_id, and via pet_members), same
        # ambiguity noted in app/scheduler/jobs.py's module docstring.
        .select("id,full_name,phone_number,registration_step,onboarding_migration_nudge_sent_at,pets!pets_profile_id_fkey(name)")
        .eq("role", "customer")
        .execute()
        .data
        or []
    )

    candidates, manual_review = segment(profiles)

    print(f"Stranded pre-migration candidates: {len(candidates)}")
    print(f"Flagged for manual review (junk name / missing phone): {len(manual_review)}")

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    _write_csv(out_dir / "stranded_onboarding_manual_review.csv", manual_review)

    if dry_run:
        preview = [{"profile_id": c.profile_id, "phone_number": c.phone_number, "message": build_message(c)} for c in candidates]
        _write_csv(out_dir / "stranded_onboarding_dry_run_preview.csv", preview)
        for row in preview[:10]:
            print(f"[DRY RUN] would send to {row['phone_number']}: {row['message']}")
        if len(preview) > 10:
            print(f"... and {len(preview) - 10} more. Full preview: {out_dir / 'stranded_onboarding_dry_run_preview.csv'}")
        print("Manual review list:", out_dir / "stranded_onboarding_manual_review.csv")
        return

    if not settings.whatsapp_reengagement_template_name:
        print(
            "No approved WhatsApp template configured -- refusing to send live.\n"
            "Set WHATSAPP_REENGAGEMENT_TEMPLATE_NAME once a template is approved in "
            "Meta Business Manager (see this script's module docstring for suggested "
            "template body text), or re-run with --dry-run to preview.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        whatsapp = WhatsAppClient(settings, http_client)
        results = []
        now = datetime.now(tz=timezone.utc).isoformat()
        for candidate in candidates:
            result = await _send_one(whatsapp, settings, candidate)
            results.append(result)
            # Atomic-claim-after-send (not before): this is a one-time manual
            # run, not a recurring job racing itself, so the simpler
            # stamp-after-a-confirmed-result is enough -- no revert-on-
            # failure dance needed since a "failed" row is never stamped and
            # can just be re-run next time this script is invoked.
            if result["status"] == "sent":
                supabase.table("profiles").update(
                    {"onboarding_migration_nudge_sent_at": now, "onboarding_migration_nudge_status": "sent"}
                ).eq("id", candidate.profile_id).execute()
            else:
                supabase.table("profiles").update(
                    {"onboarding_migration_nudge_status": f"failed: {result.get('error', '')[:200]}"}
                ).eq("id", candidate.profile_id).execute()

    sent = sum(1 for r in results if r["status"] == "sent")
    failed = sum(1 for r in results if r["status"] == "failed")
    _write_csv(out_dir / "stranded_onboarding_send_results.csv", results)
    print(f"Sent: {sent}, Failed: {failed}, Total: {len(results)}")
    print("Full results:", out_dir / "stranded_onboarding_send_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview messages/segments without sending or writing to the DB.")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
