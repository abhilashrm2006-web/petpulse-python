"""_load_awaiting_prescription_session must find a completed session still
awaiting_from="doctor_prescription" for a vet -- without this, a vet's
follow-up document upload has no deterministic way to know which session
it belongs to (open-session queries elsewhere only cover
pending/negotiating/accepted, not completed)."""

import pytest

from app.ingestion.context import _load_awaiting_prescription_session
from tests.fake_supabase import FakeSupabaseClient


def test_finds_completed_session_awaiting_prescription():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "doctor_phone": "919000000001", "status": "completed", "awaiting_from": "doctor_prescription"},
            ],
        }
    )

    result = _load_awaiting_prescription_session(supabase, "919000000001")

    assert result is not None
    assert result["id"] == "session-a"


def test_ignores_sessions_not_awaiting_prescription():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "doctor_phone": "919000000001", "status": "accepted", "awaiting_from": None},
                {"id": "session-b", "doctor_phone": "919000000001", "status": "completed", "awaiting_from": None},
            ],
        }
    )

    result = _load_awaiting_prescription_session(supabase, "919000000001")

    assert result is None


def test_ignores_other_doctors_sessions():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "doctor_phone": "919111111111", "status": "completed", "awaiting_from": "doctor_prescription"},
            ],
        }
    )

    result = _load_awaiting_prescription_session(supabase, "919000000001")

    assert result is None


@pytest.mark.asyncio
async def test_build_context_sets_is_subscriber_from_active_subscription():
    """build_context should surface is_subscriber=True only when the profile
    has a subscriptions row with status="active" -- "trial" (signup started,
    not yet confirmed by Razorpay's webhook) must not count."""
    from app.ingestion.context import build_context
    from app.ingestion.webhook import ExtractedMessage

    client = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane", "role": "customer"}],
            "subscriptions": [{"id": "sub-1", "profile_id": "profile-1", "status": "active"}],
        }
    )
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.1",
        timestamp="1700000000", message_type="text", text="hi",
    )

    ctx = await build_context(client, extracted)

    assert ctx.is_subscriber is True


@pytest.mark.asyncio
async def test_build_context_trial_subscription_does_not_grant_subscriber_status():
    from app.ingestion.context import build_context
    from app.ingestion.webhook import ExtractedMessage

    client = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane", "role": "customer"}],
            "subscriptions": [{"id": "sub-1", "profile_id": "profile-1", "status": "trial"}],
        }
    )
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.2",
        timestamp="1700000001", message_type="text", text="hi",
    )

    ctx = await build_context(client, extracted)

    assert ctx.is_subscriber is False


@pytest.mark.asyncio
async def test_build_context_vet_role_is_never_marked_subscriber():
    from app.ingestion.context import build_context
    from app.ingestion.webhook import ExtractedMessage

    client = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"}],
            "subscriptions": [{"id": "sub-1", "profile_id": "vet-1", "status": "active"}],
        }
    )
    extracted = ExtractedMessage(
        phone_number="919000000001", sender_name="Dr. Rao", message_id="wamid.3",
        timestamp="1700000002", message_type="text", text="hi",
    )

    ctx = await build_context(client, extracted)

    assert ctx.is_subscriber is False
