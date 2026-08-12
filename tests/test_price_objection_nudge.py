"""Covers send_price_objection_nudges (2026-08 root-cause spec, item #3
paywall sequencing): a customer whose most recent message in the
conversation is the BOT mentioning pricing, with no reply since, gets one
automatic clarifying nudge past the silence threshold, then at most one
more per cooldown window if still silent."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.scheduler.jobs import send_price_objection_nudges
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    return SimpleNamespace(supabase=supabase, whatsapp=SimpleNamespace(send_text=AsyncMock()))


def _hours_ago(hours: float) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()


def _profile(**overrides):
    base = {"id": "p1", "role": "customer", "phone_number": "919876543210", "full_name": "Amrapali Roy"}
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_nudges_after_silence_following_a_pricing_message():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [_profile()],
            "messages": [
                {"profile_id": "p1", "sender_type": "assistant", "content": "Consultations are ₹399 per visit.", "created_at": _hours_ago(10)},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_price_objection_nudges(ctx)

    ctx.whatsapp.send_text.assert_awaited_once()
    assert supabase.rows("profiles")[0]["last_price_objection_nudge_sent_at"] is not None


@pytest.mark.asyncio
async def test_does_not_nudge_within_the_silence_threshold():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [_profile()],
            "messages": [
                {"profile_id": "p1", "sender_type": "assistant", "content": "It's ₹399 per consultation.", "created_at": _hours_ago(1)},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_price_objection_nudges(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_nudge_when_customer_already_replied():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [_profile()],
            "messages": [
                {"profile_id": "p1", "sender_type": "assistant", "content": "Consultations cost ₹399.", "created_at": _hours_ago(20)},
                {"profile_id": "p1", "sender_type": "user", "content": "ok thanks", "created_at": _hours_ago(10)},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_price_objection_nudges(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_nudge_when_last_bot_message_wasnt_about_price():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [_profile()],
            "messages": [
                {"profile_id": "p1", "sender_type": "assistant", "content": "Hope Bobby feels better soon!", "created_at": _hours_ago(20)},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_price_objection_nudges(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_respects_cooldown_after_already_nudged():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [_profile(last_price_objection_nudge_sent_at=_hours_ago(24))],
            "messages": [
                {"profile_id": "p1", "sender_type": "assistant", "content": "It's a flat ₹399 fee.", "created_at": _hours_ago(10)},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_price_objection_nudges(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()
