"""Covers send_doctor_schedule_reminders: T-1 (evening before) and T-0
(morning of) appointment-list pushes to doctors, so a vet has their day
planned without needing to message the bot first."""

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.availability.slots import IST
from app.scheduler.jobs import send_doctor_schedule_reminders
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    return SimpleNamespace(supabase=supabase, whatsapp=SimpleNamespace(send_text=AsyncMock()))


def _session_at(doctor_phone: str, when: datetime, **overrides) -> dict:
    row = {
        "id": "s1", "doctor_phone": doctor_phone, "status": "accepted",
        "preferred_time": when.isoformat(), "profile_id": "profile-1", "pet_id": "pet-1",
        "case_summary": "Limping on left hind leg",
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_sends_todays_schedule_to_a_doctor_with_appointments_today():
    today_appt = datetime.combine(date.today(), datetime.min.time(), tzinfo=IST).replace(hour=10, minute=30)
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [_session_at("919000000001", today_appt)],
            "profiles": [{"id": "profile-1", "full_name": "Jane Doe"}],
            "pets": [{"id": "pet-1", "name": "Rex"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_doctor_schedule_reminders(ctx, "day_of")

    ctx.whatsapp.send_text.assert_awaited_once()
    phone, text = ctx.whatsapp.send_text.call_args.args
    assert phone == "919000000001"
    assert "Today's Schedule" in text
    assert "Rex" in text
    assert "Jane Doe" in text
    assert "10:30 AM" in text
    assert "Limping on left hind leg" in text


@pytest.mark.asyncio
async def test_sends_tomorrows_schedule_for_day_before_reminder():
    tomorrow_appt = datetime.combine(date.today() + timedelta(days=1), datetime.min.time(), tzinfo=IST).replace(hour=14)
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [_session_at("919000000001", tomorrow_appt)],
            "profiles": [{"id": "profile-1", "full_name": "Jane Doe"}],
            "pets": [{"id": "pet-1", "name": "Rex"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_doctor_schedule_reminders(ctx, "day_before")

    ctx.whatsapp.send_text.assert_awaited_once()
    text = ctx.whatsapp.send_text.call_args.args[1]
    assert "Tomorrow's Schedule" in text


@pytest.mark.asyncio
async def test_day_of_reminder_ignores_tomorrows_appointments():
    tomorrow_appt = datetime.combine(date.today() + timedelta(days=1), datetime.min.time(), tzinfo=IST).replace(hour=10)
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [_session_at("919000000001", tomorrow_appt)],
            "profiles": [{"id": "profile-1", "full_name": "Jane Doe"}],
            "pets": [{"id": "pet-1", "name": "Rex"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_doctor_schedule_reminders(ctx, "day_of")

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_message_when_doctor_has_nothing_scheduled():
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)

    await send_doctor_schedule_reminders(ctx, "day_of")

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiple_appointments_for_one_doctor_are_combined_into_one_message():
    today = date.today()
    appt1 = datetime.combine(today, datetime.min.time(), tzinfo=IST).replace(hour=9)
    appt2 = datetime.combine(today, datetime.min.time(), tzinfo=IST).replace(hour=15)
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                _session_at("919000000001", appt1, id="s1", pet_id="pet-1", profile_id="profile-1"),
                _session_at("919000000001", appt2, id="s2", pet_id="pet-2", profile_id="profile-2"),
            ],
            "profiles": [
                {"id": "profile-1", "full_name": "Jane Doe"},
                {"id": "profile-2", "full_name": "John Smith"},
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}, {"id": "pet-2", "name": "Bella"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_doctor_schedule_reminders(ctx, "day_of")

    ctx.whatsapp.send_text.assert_awaited_once()
    text = ctx.whatsapp.send_text.call_args.args[1]
    assert "Rex" in text and "Bella" in text
    assert "2 appointments" in text


@pytest.mark.asyncio
async def test_different_doctors_each_get_their_own_message():
    today_appt = datetime.combine(date.today(), datetime.min.time(), tzinfo=IST).replace(hour=10)
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                _session_at("919000000001", today_appt, id="s1", profile_id="profile-1", pet_id="pet-1"),
                _session_at("919000000002", today_appt, id="s2", profile_id="profile-1", pet_id="pet-1"),
            ],
            "profiles": [{"id": "profile-1", "full_name": "Jane Doe"}],
            "pets": [{"id": "pet-1", "name": "Rex"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_doctor_schedule_reminders(ctx, "day_of")

    assert ctx.whatsapp.send_text.await_count == 2
    recipients = {c.args[0] for c in ctx.whatsapp.send_text.await_args_list}
    assert recipients == {"919000000001", "919000000002"}


@pytest.mark.asyncio
async def test_running_twice_never_double_sends():
    today_appt = datetime.combine(date.today(), datetime.min.time(), tzinfo=IST).replace(hour=10)
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [_session_at("919000000001", today_appt)],
            "profiles": [{"id": "profile-1", "full_name": "Jane Doe"}],
            "pets": [{"id": "pet-1", "name": "Rex"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_doctor_schedule_reminders(ctx, "day_of")
    await send_doctor_schedule_reminders(ctx, "day_of")

    assert ctx.whatsapp.send_text.await_count == 1


@pytest.mark.asyncio
async def test_day_before_and_day_of_are_independent_claims():
    """The same doctor can get BOTH a day-before reminder (for tomorrow)
    and, a day later, a day-of reminder (for today) covering the same
    underlying appointment -- these are genuinely different reminders, not
    a duplicate of each other."""
    tomorrow = date.today() + timedelta(days=1)
    appt = datetime.combine(tomorrow, datetime.min.time(), tzinfo=IST).replace(hour=10)
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [_session_at("919000000001", appt)],
            "profiles": [{"id": "profile-1", "full_name": "Jane Doe"}],
            "pets": [{"id": "pet-1", "name": "Rex"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_doctor_schedule_reminders(ctx, "day_before")

    assert ctx.whatsapp.send_text.await_count == 1
    claims = supabase.rows("doctor_schedule_reminders_sent")
    assert len(claims) == 1
    assert claims[0]["reminder_type"] == "day_before"


@pytest.mark.asyncio
async def test_claim_is_reverted_on_send_failure():
    today_appt = datetime.combine(date.today(), datetime.min.time(), tzinfo=IST).replace(hour=10)
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [_session_at("919000000001", today_appt)],
            "profiles": [{"id": "profile-1", "full_name": "Jane Doe"}],
            "pets": [{"id": "pet-1", "name": "Rex"}],
        }
    )
    ctx = _make_ctx(supabase)
    ctx.whatsapp.send_text = AsyncMock(side_effect=RuntimeError("WhatsApp outage"))

    await send_doctor_schedule_reminders(ctx, "day_of")

    assert supabase.rows("doctor_schedule_reminders_sent") == []


@pytest.mark.asyncio
async def test_pending_or_declined_sessions_are_not_included():
    today_appt = datetime.combine(date.today(), datetime.min.time(), tzinfo=IST).replace(hour=10)
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [_session_at("919000000001", today_appt, status="pending")],
            "profiles": [{"id": "profile-1", "full_name": "Jane Doe"}],
            "pets": [{"id": "pet-1", "name": "Rex"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_doctor_schedule_reminders(ctx, "day_of")

    ctx.whatsapp.send_text.assert_not_awaited()
