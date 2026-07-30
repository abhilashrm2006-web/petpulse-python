"""Covers send_onboarding_reminders: a customer who started signup but has
gone quiet mid-flow (stuck at some registration_step) for 24h+ gets a
proactive nudge to come back and finish, then at most one more every day
if still stuck -- never a nudge every single run, which would just read
as spam."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.scheduler.jobs import send_onboarding_reminders
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    return SimpleNamespace(supabase=supabase, whatsapp=SimpleNamespace(send_text=AsyncMock()))


def _hours_ago(hours: float) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()


@pytest.mark.asyncio
async def test_reminds_a_customer_stuck_onboarding_for_over_24_hours():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane Doe",
                    "registration_step": "awaiting_pet_weight", "last_active_at": _hours_ago(30),
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_onboarding_reminders(ctx)

    ctx.whatsapp.send_text.assert_awaited_once()
    assert ctx.whatsapp.send_text.await_args.args[0] == "919000000001"
    assert "Jane" in ctx.whatsapp.send_text.await_args.args[1]
    assert supabase.rows("profiles")[0]["last_onboarding_reminder_sent_at"] is not None


@pytest.mark.asyncio
async def test_does_not_remind_a_customer_active_within_24_hours():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "registration_step": "awaiting_pet_weight", "last_active_at": _hours_ago(2),
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_onboarding_reminders(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_reminds_a_customer_with_no_last_active_at_since_it_predates_that_column():
    """A null last_active_at here is NOT "never engaged" -- unlike the
    reengagement job, this customer had to message at least once to get a
    registration_step at all. It just means they started before
    profiles.last_active_at existed, so they're treated as stuck too."""
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "registration_step": "awaiting_city", "last_active_at": None,
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_onboarding_reminders(ctx)

    ctx.whatsapp.send_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_does_not_remind_a_customer_who_completed_onboarding():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "registration_step": "completed", "last_active_at": _hours_ago(200),
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_onboarding_reminders(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_remind_a_customer_with_no_registration_step_at_all():
    """No registration_step means they predate the onboarding wizard, or
    are otherwise not mid-signup -- nothing to remind them to finish."""
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "registration_step": None, "last_active_at": _hours_ago(200),
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_onboarding_reminders(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_never_reminds_a_vet():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "v1", "role": "vet", "phone_number": "919000000002", "full_name": "Dr. Rao",
                    "registration_step": "awaiting_city", "last_active_at": _hours_ago(200),
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_onboarding_reminders(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_re_remind_within_the_1_day_cooldown():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "registration_step": "awaiting_city", "last_active_at": _hours_ago(100),
                    "last_onboarding_reminder_sent_at": _hours_ago(5),
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_onboarding_reminders(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_re_reminds_after_the_1_day_cooldown_expires():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "registration_step": "awaiting_city", "last_active_at": _hours_ago(100),
                    "last_onboarding_reminder_sent_at": _hours_ago(30),
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_onboarding_reminders(ctx)

    ctx.whatsapp.send_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_running_twice_never_double_reminds():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "registration_step": "awaiting_city", "last_active_at": _hours_ago(30),
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_onboarding_reminders(ctx)
    await send_onboarding_reminders(ctx)

    assert ctx.whatsapp.send_text.await_count == 1


@pytest.mark.asyncio
async def test_claim_is_reverted_on_send_failure():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "registration_step": "awaiting_city", "last_active_at": _hours_ago(30),
                }
            ],
        }
    )
    ctx = _make_ctx(supabase)
    ctx.whatsapp.send_text = AsyncMock(side_effect=RuntimeError("WhatsApp outage"))

    await send_onboarding_reminders(ctx)

    assert supabase.rows("profiles")[0].get("last_onboarding_reminder_sent_at") is None
