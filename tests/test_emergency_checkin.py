"""Covers flag_emergency_checkins (2026-08-04 re-engagement workstream,
item 4): a RED/emergency check_symptoms escalation (health_logs row with
ai_risk_score >= 80) gets flagged for a human check-in only if the customer
never sent a single follow-up message within the review window."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.scheduler.jobs import flag_emergency_checkins
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    return SimpleNamespace(supabase=supabase)


def _hours_ago(hours: float) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()


@pytest.mark.asyncio
async def test_flags_a_red_escalation_with_no_followup_after_48h():
    supabase = FakeSupabaseClient(
        initial={
            "health_logs": [
                {
                    "id": "log-1", "profile_id": "p1", "pet_id": "pet-1",
                    "ai_risk_score": 80, "created_at": _hours_ago(50),
                    "needs_human_checkin": False, "human_checkin_flagged_at": None,
                }
            ],
            "messages": [],
        }
    )
    ctx = _make_ctx(supabase)

    await flag_emergency_checkins(ctx)

    log = supabase.rows("health_logs")[0]
    assert log["needs_human_checkin"] is True
    assert log["human_checkin_flagged_at"] is not None


@pytest.mark.asyncio
async def test_does_not_flag_within_the_48h_review_window():
    supabase = FakeSupabaseClient(
        initial={
            "health_logs": [
                {
                    "id": "log-1", "profile_id": "p1", "pet_id": "pet-1",
                    "ai_risk_score": 100, "created_at": _hours_ago(10),
                    "needs_human_checkin": False, "human_checkin_flagged_at": None,
                }
            ],
            "messages": [],
        }
    )
    ctx = _make_ctx(supabase)

    await flag_emergency_checkins(ctx)

    assert supabase.rows("health_logs")[0]["needs_human_checkin"] is False


@pytest.mark.asyncio
async def test_does_not_flag_below_the_red_severity_threshold():
    """ai_risk_score 60 == severity 3 (Yellow), not RED -- must not be
    treated as a silent emergency drop-off."""
    supabase = FakeSupabaseClient(
        initial={
            "health_logs": [
                {
                    "id": "log-1", "profile_id": "p1", "pet_id": "pet-1",
                    "ai_risk_score": 60, "created_at": _hours_ago(72),
                    "needs_human_checkin": False, "human_checkin_flagged_at": None,
                }
            ],
            "messages": [],
        }
    )
    ctx = _make_ctx(supabase)

    await flag_emergency_checkins(ctx)

    assert supabase.rows("health_logs")[0]["needs_human_checkin"] is False


@pytest.mark.asyncio
async def test_does_not_flag_when_the_customer_followed_up():
    supabase = FakeSupabaseClient(
        initial={
            "health_logs": [
                {
                    "id": "log-1", "profile_id": "p1", "pet_id": "pet-1",
                    "ai_risk_score": 100, "created_at": _hours_ago(60),
                    "needs_human_checkin": False, "human_checkin_flagged_at": None,
                }
            ],
            "messages": [
                {"id": "m1", "profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(55)},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await flag_emergency_checkins(ctx)

    assert supabase.rows("health_logs")[0]["needs_human_checkin"] is False


@pytest.mark.asyncio
async def test_a_followup_message_before_the_escalation_does_not_count():
    """The follow-up must come AFTER the escalation, not just exist at some
    point in the profile's history."""
    supabase = FakeSupabaseClient(
        initial={
            "health_logs": [
                {
                    "id": "log-1", "profile_id": "p1", "pet_id": "pet-1",
                    "ai_risk_score": 100, "created_at": _hours_ago(60),
                    "needs_human_checkin": False, "human_checkin_flagged_at": None,
                }
            ],
            "messages": [
                {"id": "m1", "profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(70)},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await flag_emergency_checkins(ctx)

    assert supabase.rows("health_logs")[0]["needs_human_checkin"] is True


@pytest.mark.asyncio
async def test_already_flagged_rows_are_skipped():
    supabase = FakeSupabaseClient(
        initial={
            "health_logs": [
                {
                    "id": "log-1", "profile_id": "p1", "pet_id": "pet-1",
                    "ai_risk_score": 100, "created_at": _hours_ago(60),
                    "needs_human_checkin": False, "human_checkin_flagged_at": _hours_ago(1),
                }
            ],
            "messages": [],
        }
    )
    ctx = _make_ctx(supabase)

    await flag_emergency_checkins(ctx)

    # needs_human_checkin stays False in this test's setup on purpose --
    # the point is that the job's query excludes rows with a
    # human_checkin_flagged_at already set, so it must leave this row
    # completely untouched (not toggle needs_human_checkin either way).
    assert supabase.rows("health_logs")[0]["needs_human_checkin"] is False
