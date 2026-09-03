"""Covers the segmentation/filtering logic for the stranded-pre-migration-
onboarding broadcast (app/reengagement/stranded_onboarding.py): who counts
as "stuck in the deprecated flow" vs the new flow, junk-name/business-name
filtering, already-nudged exclusion, and message-variant selection."""

from app.reengagement.stranded_onboarding import (
    Candidate,
    build_message,
    is_junk_name,
    is_stranded_pre_migration,
    real_pet_name,
    segment,
)


def _profile(**overrides):
    base = {
        "id": "p1", "full_name": "Anudeep Reddy", "phone_number": "919876543210",
        "registration_step": "awaiting_member_type", "pets": [],
    }
    base.update(overrides)
    return base


# --- deprecated-step filter -----------------------------------------------

def test_deprecated_old_flow_steps_are_stranded():
    for step in ("awaiting_member_type", "awaiting_pet_dob", "awaiting_tier_choice", "awaiting_microchip_number"):
        assert is_stranded_pre_migration(_profile(registration_step=step)) is True


def test_new_flow_steps_are_never_stranded():
    for step in ("awaiting_customer_name", "awaiting_pet_name"):
        assert is_stranded_pre_migration(_profile(registration_step=step)) is False


def test_completed_registration_is_not_stranded():
    assert is_stranded_pre_migration(_profile(registration_step="completed")) is False


def test_unknown_step_is_not_stranded():
    assert is_stranded_pre_migration(_profile(registration_step="awaiting_something_new")) is False


# --- junk / business name filter -------------------------------------------

def test_normal_names_are_not_junk():
    for name in ("Anudeep Reddy", "Priya", "Md Shahjad Alam"):
        assert is_junk_name(name) is False


def test_empty_or_single_char_names_are_junk():
    assert is_junk_name(None) is True
    assert is_junk_name("") is True
    assert is_junk_name("   ") is True
    assert is_junk_name("A") is True


def test_emoji_only_names_are_junk():
    assert is_junk_name("🙂") is True
    assert is_junk_name("😂😂😂") is True


def test_business_names_are_junk():
    for name in ("Pipe & Hardware", "SmART Driving School", "Sharma Enterprises Pvt Ltd", "Kumar Traders"):
        assert is_junk_name(name) is True


def test_sentence_like_names_are_junk():
    assert is_junk_name("Sorry im a veterinarian") is True
    assert is_junk_name("No Hindi language?") is True
    assert is_junk_name("So many we have 12 dogs") is True


def test_email_addresses_are_junk():
    """Live data-quality bug (2026-09-04): full_name held an email address
    for a stranded profile -- "Hi grm679@gmail.com!" is as broken-looking a
    greeting as "Hi Pipe & Hardware!"."""
    assert is_junk_name("grm679@gmail.com") is True


# --- pet name resolution ----------------------------------------------------

def test_real_pet_name_returns_first_non_junk():
    assert real_pet_name([{"name": "Bruno"}]) == "Bruno"


def test_real_pet_name_skips_junk_and_falls_through():
    assert real_pet_name([{"name": "So many we have 12 dogs"}, {"name": "Bobby"}]) == "Bobby"


def test_real_pet_name_is_none_when_no_pets_or_all_junk():
    assert real_pet_name([]) is None
    assert real_pet_name(None) is None
    assert real_pet_name([{"name": "🙂"}]) is None


# --- segment(): the full pipeline -------------------------------------------

def test_segment_sends_to_a_clean_stranded_customer_with_a_pet():
    profiles = [_profile(id="p1", full_name="Anudeep Reddy", pets=[{"name": "Bruno"}])]
    sendable, manual_review = segment(profiles)

    assert len(sendable) == 1
    assert manual_review == []
    assert sendable[0].pet_name == "Bruno"


def test_segment_excludes_new_flow_customers():
    profiles = [_profile(id="p1", registration_step="awaiting_customer_name")]
    sendable, manual_review = segment(profiles)

    assert sendable == []
    assert manual_review == []


def test_segment_excludes_completed_customers():
    profiles = [_profile(id="p1", registration_step="completed")]
    sendable, manual_review = segment(profiles)

    assert sendable == []
    assert manual_review == []


def test_segment_excludes_already_nudged_customers():
    profiles = [_profile(id="p1", onboarding_migration_nudge_sent_at="2026-08-04T00:00:00+00:00")]
    sendable, manual_review = segment(profiles)

    assert sendable == []
    assert manual_review == []


def test_segment_flags_junk_names_for_manual_review_not_auto_send():
    profiles = [_profile(id="p1", full_name="Pipe & Hardware")]
    sendable, manual_review = segment(profiles)

    assert sendable == []
    assert len(manual_review) == 1
    assert manual_review[0]["review_reason"] == "junk_or_placeholder_name"


def test_segment_flags_missing_phone_for_manual_review():
    profiles = [_profile(id="p1", phone_number=None)]
    sendable, manual_review = segment(profiles)

    assert sendable == []
    assert len(manual_review) == 1
    assert manual_review[0]["review_reason"] == "missing_phone_number"


def test_segment_handles_a_realistic_mixed_batch():
    profiles = [
        _profile(id="p1", full_name="Anudeep Reddy", pets=[{"name": "Bruno"}]),   # sendable, with pet
        _profile(id="p2", full_name="Priya", pets=[]),                             # sendable, no pet
        _profile(id="p3", registration_step="awaiting_customer_name"),             # excluded: new flow
        _profile(id="p4", registration_step="completed"),                         # excluded: done
        _profile(id="p5", onboarding_migration_nudge_sent_at="2026-08-01T00:00:00+00:00"),  # excluded: already nudged
        _profile(id="p6", full_name="Kumar Traders"),                             # manual review: business name
        _profile(id="p7", phone_number=""),                                       # manual review: no phone
    ]

    sendable, manual_review = segment(profiles)

    assert {c.profile_id for c in sendable} == {"p1", "p2"}
    assert {r["id"] for r in manual_review} == {"p6", "p7"}


# --- message building --------------------------------------------------------

def test_message_a_mentions_the_pet_by_name():
    candidate = Candidate(profile_id="p1", phone_number="919876543210", full_name="Anudeep Reddy", pet_name="Bruno")
    message = build_message(candidate)

    assert "Anudeep" in message
    assert "Bruno's profile" in message
    assert "3 short questions" in message


def test_message_b_omits_pet_mention_when_none_on_file():
    candidate = Candidate(profile_id="p1", phone_number="919876543210", full_name="Priya", pet_name=None)
    message = build_message(candidate)

    assert "Priya" in message
    assert "'s profile" not in message
    assert "3 short questions" in message


def test_message_falls_back_to_generic_greeting_without_a_name():
    candidate = Candidate(profile_id="p1", phone_number="919876543210", full_name="", pet_name=None)
    message = build_message(candidate)

    assert message.startswith("Hi there!")
