"""Covers send_reengagement_nudges: a customer silent for 48h+ gets a
proactive check-in, then at most one more every 7 days if still silent --
never a nudge every single run, which would just read as spam."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.scheduler.jobs import send_reengagement_nudges
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    return SimpleNamespace(supabase=supabase, whatsapp=SimpleNamespace(send_text=AsyncMock()))


def _hours_ago(hours: float) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()


@pytest.mark.asyncio
async def test_nudges_a_customer_silent_for_over_48_hours():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane Doe"}],
            "messages": [{"id": "m1", "profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(50)}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_reengagement_nudges(ctx)

    ctx.whatsapp.send_text.assert_awaited_once()
    assert ctx.whatsapp.send_text.await_args.args[0] == "919000000001"
    assert "Jane" in ctx.whatsapp.send_text.await_args.args[1]
    assert supabase.rows("profiles")[0]["last_reengagement_sent_at"] is not None


@pytest.mark.asyncio
async def test_does_not_nudge_a_customer_active_within_48_hours():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane"}],
            "messages": [{"id": "m1", "profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(10)}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_reengagement_nudges(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_never_nudges_a_customer_who_has_never_sent_a_message():
    """No message history at all means they never actually engaged in the
    first place (a brand-new/never-onboarded number) -- there's nothing to
    "re-engage" and this job must not treat that as 48h of silence."""
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane"}],
            "messages": [],
        }
    )
    ctx = _make_ctx(supabase)

    await send_reengagement_nudges(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_never_nudges_a_vet():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "v1", "role": "vet", "phone_number": "919000000002", "full_name": "Dr. Rao"}],
            "messages": [{"id": "m1", "profile_id": "v1", "sender_type": "user", "created_at": _hours_ago(100)}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_reengagement_nudges(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_re_nudge_within_the_7_day_cooldown():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "last_reengagement_sent_at": _hours_ago(24),  # nudged yesterday
                }
            ],
            "messages": [{"id": "m1", "profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(100)}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_reengagement_nudges(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_re_nudges_after_the_7_day_cooldown_expires():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane",
                    "last_reengagement_sent_at": (datetime.now(tz=timezone.utc) - timedelta(days=8)).isoformat(),
                }
            ],
            "messages": [{"id": "m1", "profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(200)}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_reengagement_nudges(ctx)

    ctx.whatsapp.send_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_running_twice_never_double_nudges():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane"}],
            "messages": [{"id": "m1", "profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(50)}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_reengagement_nudges(ctx)
    await send_reengagement_nudges(ctx)

    assert ctx.whatsapp.send_text.await_count == 1


@pytest.mark.asyncio
async def test_claim_is_reverted_on_send_failure():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "p1", "role": "customer", "phone_number": "919000000001", "full_name": "Jane"}],
            "messages": [{"id": "m1", "profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(50)}],
        }
    )
    ctx = _make_ctx(supabase)
    ctx.whatsapp.send_text = AsyncMock(side_effect=RuntimeError("WhatsApp outage"))

    await send_reengagement_nudges(ctx)

    assert supabase.rows("profiles")[0].get("last_reengagement_sent_at") is None
