"""Covers rate_customer_intents: the bulk scheduled job that LLM-rates any
customer who's never been rated, or has new messages since their last
rating, capped at INTENT_RATING_BATCH_SIZE per run."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.scheduler.jobs import rate_customer_intents
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    return SimpleNamespace(supabase=supabase, openai=SimpleNamespace(), settings=SimpleNamespace())


@pytest.mark.asyncio
async def test_rates_a_never_rated_customer_with_messages():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "c1", "role": "customer", "phone_number": "919000000001"}],
            "messages": [{"id": "m1", "profile_id": "c1", "sender_type": "user", "content": "book a vet", "created_at": "2026-06-01T10:00:00"}],
        }
    )
    ctx = _make_ctx(supabase)

    with patch(
        "app.scheduler.jobs.rate_customer_intent",
        AsyncMock(return_value={"rating": "High", "reason": "wants a vet"}),
    ):
        await rate_customer_intents(ctx)

    profile = supabase.rows("profiles")[0]
    assert profile["intent_rating"] == "High"
    assert profile["intent_rating_message_count"] == 1


@pytest.mark.asyncio
async def test_skips_customer_with_no_messages():
    supabase = FakeSupabaseClient(initial={"profiles": [{"id": "c1", "role": "customer", "phone_number": "919000000001"}]})
    ctx = _make_ctx(supabase)

    with patch("app.scheduler.jobs.rate_customer_intent", AsyncMock()) as mock_rate:
        await rate_customer_intents(ctx)

    mock_rate.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_customer_already_rated_with_no_new_messages():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "c1", "role": "customer", "phone_number": "919000000001", "intent_rating_message_count": 2}],
            "messages": [
                {"id": "m1", "profile_id": "c1", "sender_type": "user", "content": "hi", "created_at": "2026-06-01T10:00:00"},
                {"id": "m2", "profile_id": "c1", "sender_type": "assistant", "content": "hello", "created_at": "2026-06-01T10:00:05"},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    with patch("app.scheduler.jobs.rate_customer_intent", AsyncMock()) as mock_rate:
        await rate_customer_intents(ctx)

    mock_rate.assert_not_awaited()


@pytest.mark.asyncio
async def test_re_rates_customer_with_new_messages_since_last_rating():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "c1", "role": "customer", "phone_number": "919000000001", "intent_rating_message_count": 1}],
            "messages": [
                {"id": "m1", "profile_id": "c1", "sender_type": "user", "content": "hi", "created_at": "2026-06-01T10:00:00"},
                {"id": "m2", "profile_id": "c1", "sender_type": "user", "content": "book a vet now", "created_at": "2026-06-01T10:05:00"},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    with patch(
        "app.scheduler.jobs.rate_customer_intent",
        AsyncMock(return_value={"rating": "High", "reason": "wants a vet"}),
    ) as mock_rate:
        await rate_customer_intents(ctx)

    mock_rate.assert_awaited_once()
    assert supabase.rows("profiles")[0]["intent_rating_message_count"] == 2


@pytest.mark.asyncio
async def test_caps_at_batch_size_per_run():
    profiles = [{"id": f"c{i}", "role": "customer", "phone_number": f"91900000{i:04d}"} for i in range(25)]
    messages = [{"id": f"m{i}", "profile_id": f"c{i}", "sender_type": "user", "content": "hi", "created_at": "2026-06-01T10:00:00"} for i in range(25)]
    supabase = FakeSupabaseClient(initial={"profiles": profiles, "messages": messages})
    ctx = _make_ctx(supabase)

    with patch(
        "app.scheduler.jobs.rate_customer_intent",
        AsyncMock(return_value={"rating": "Low", "reason": "quiet"}),
    ) as mock_rate:
        await rate_customer_intents(ctx)

    assert mock_rate.await_count == 20


@pytest.mark.asyncio
async def test_continues_past_a_failed_rating():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "phone_number": "919000000001"},
                {"id": "c2", "role": "customer", "phone_number": "919000000002"},
            ],
            "messages": [
                {"id": "m1", "profile_id": "c1", "sender_type": "user", "content": "hi", "created_at": "2026-06-01T10:00:00"},
                {"id": "m2", "profile_id": "c2", "sender_type": "user", "content": "hi", "created_at": "2026-06-01T10:00:00"},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    with patch(
        "app.scheduler.jobs.rate_customer_intent",
        AsyncMock(side_effect=[RuntimeError("outage"), {"rating": "Low", "reason": "quiet"}]),
    ):
        await rate_customer_intents(ctx)

    ratings = {p["id"]: p.get("intent_rating") for p in supabase.rows("profiles")}
    assert ratings["c2"] == "Low"
    assert ratings["c1"] is None
