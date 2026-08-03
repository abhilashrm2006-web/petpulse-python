"""Segmentation + message-building for the stranded-pre-migration-signup
broadcast (Priority 1 of the 2026-08-04 re-engagement workstream). Pure,
DB-agnostic functions so the filtering logic is unit-testable without a real
Supabase client -- app/scripts/reengage_stranded_onboarding.py does the
actual querying/sending and is the thin, mostly-untested I/O shell around
this module."""

from dataclasses import dataclass
from typing import Any

# Registration steps that only existed in the pre-2026-08-03 longer wizard
# (see git history of app/ingestion/registration.py) -- a profile parked here
# auto-recovers into the new 3-question flow the instant it sends any
# message; it does NOT need a code fix, only outreach.
DEPRECATED_ONBOARDING_STEPS = {
    "awaiting_member_type",
    "awaiting_pet_dob",
    "awaiting_pet_age",
    "awaiting_pet_weight",
    "awaiting_kci_status",
    "awaiting_vaccination_status",
    "awaiting_microchip_status",
    "awaiting_microchip_number",
    "awaiting_tier_choice",
    "awaiting_free_subchoice",
    "awaiting_existing_phone",
    "awaiting_existing_verify",
}

# The two new-flow steps this broadcast must NOT target (Priority 2 territory
# -- see app/reengagement/onboarding_events.py; those 10 users stay out until
# validator-rejection data has been reviewed).
NEW_FLOW_ONBOARDING_STEPS = {"awaiting_customer_name", "awaiting_pet_name"}

BUSINESS_NAME_KEYWORDS = (
    "pvt", "ltd", "llp", "enterprises", "traders", "trading", "hardware",
    "driving school", "electricals", "electronics", "constructions",
    "properties", "consultancy", "industries", "stores", "store", "shop",
    "services", "solutions", "company", "co.", "& sons", "& co", "agency",
    "agencies", "distributors", "suppliers", "exports", "imports",
)


def is_junk_name(name: str | None) -> bool:
    """True for a name that's emoji-only, a bare single character, reads
    like a sentence/complaint, or matches common business-name patterns --
    these predate the name validator added to registration.py and must be
    flagged for manual review, not auto-messaged with a broken-looking
    greeting like "Hi 🙂!" or "Hi Pipe & Hardware!"."""
    if not name:
        return True
    text = name.strip()
    if not text:
        return True
    if len(text) <= 1:
        return True
    if not any(ch.isalpha() for ch in text):
        return True
    lowered = text.lower()
    if any(kw in lowered for kw in BUSINESS_NAME_KEYWORDS):
        return True
    if "&" in text:
        return True
    if "?" in text:
        return True
    if len(text.split()) > 3:
        return True
    return False


def real_pet_name(pets: list[dict[str, Any]] | None) -> str | None:
    """Returns the first non-junk pet name on file, or None -- used to pick
    between Message A (mentions the pet) and Message B (doesn't), and to
    avoid dropping a garbage pet name (e.g. "So many we have 12 dogs",
    pre-dating the pet-name validator) straight into an outbound message."""
    for pet in pets or []:
        name = (pet or {}).get("name")
        if name and not is_junk_name(name):
            return name.strip()
    return None


@dataclass
class Candidate:
    profile_id: str
    phone_number: str
    full_name: str
    pet_name: str | None


def is_stranded_pre_migration(profile: dict[str, Any]) -> bool:
    """True only for a customer parked in a step that no longer exists in
    the current (post-2026-08-03) registration wizard -- never for the two
    steps the new flow itself still uses (those are Priority 2, not this
    broadcast)."""
    step = profile.get("registration_step")
    return step in DEPRECATED_ONBOARDING_STEPS


def already_nudged(profile: dict[str, Any]) -> bool:
    return bool(profile.get("onboarding_migration_nudge_sent_at"))


def segment(profiles: list[dict[str, Any]]) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Splits the input into (sendable candidates, rows needing manual
    review) -- never both. A profile with a junk full_name is excluded from
    the sendable list entirely, even if it would otherwise qualify."""
    sendable: list[Candidate] = []
    manual_review: list[dict[str, Any]] = []

    for profile in profiles:
        if not is_stranded_pre_migration(profile):
            continue
        if already_nudged(profile):
            continue

        full_name = (profile.get("full_name") or "").strip()
        phone = profile.get("phone_number")
        if not phone:
            manual_review.append({**profile, "review_reason": "missing_phone_number"})
            continue
        if is_junk_name(full_name):
            manual_review.append({**profile, "review_reason": "junk_or_placeholder_name"})
            continue

        sendable.append(
            Candidate(
                profile_id=profile["id"],
                phone_number=phone,
                full_name=full_name,
                pet_name=real_pet_name(profile.get("pets")),
            )
        )

    return sendable, manual_review


def build_message(candidate: Candidate) -> str:
    first_name = candidate.full_name.split()[0] if candidate.full_name else "there"
    if candidate.pet_name:
        return (
            f"Hi {first_name}! We've made signing up for PetPulse quicker — just 3 short questions now. "
            f"Reply anything to pick up right where you left off with {candidate.pet_name}'s profile. "
            "Takes under a minute!"
        )
    return (
        f"Hi {first_name}! We've made signing up for PetPulse quicker — just 3 short questions now. "
        "Reply anything to pick up right where you left off. Takes under a minute!"
    )
