"""Covers the Founding Member cohort logic in create_and_send_subscription:
capped by signup count (not a date cutoff), reuses plan_name="Premium" so
is_active_subscriber needs no separate branch -- only amount/plan_id differ
from a standard ₹399 Subscriber."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.subscriptions import FOUNDING_MEMBER_AMOUNT, FOUNDING_MEMBER_CAP, STANDARD_AMOUNT, create_and_send_subscription
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    settings = SimpleNamespace(razorpay_founding_plan_id="plan_FOUNDING", razorpay_subscription_plan_id="plan_STANDARD")
    whatsapp = SimpleNamespace(send_text=AsyncMock())
    return SimpleNamespace(supabase=supabase, whatsapp=whatsapp, settings=settings)


@pytest.mark.asyncio
async def test_first_signup_gets_founding_member_pricing(monkeypatch):
    supabase = FakeSupabaseClient()
    create_subscription = AsyncMock(return_value={"id": "sub_1", "short_url": "https://rzp.io/x"})
    monkeypatch.setattr("app.agent.tools.subscriptions.razorpay_client.create_subscription", create_subscription)
    ctx = _make_ctx(supabase)
    profile = {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}

    result = await create_and_send_subscription(ctx, profile)

    assert result["success"] is True
    assert result["is_founding_member"] is True
    assert create_subscription.call_args.kwargs["plan_id"] == "plan_FOUNDING"
    sub_row = supabase.rows("subscriptions")[0]
    assert sub_row["amount"] == FOUNDING_MEMBER_AMOUNT
    assert sub_row["plan_name"] == "Premium"
    assert "Founding Member" in ctx.whatsapp.send_text.call_args.args[1]


@pytest.mark.asyncio
async def test_501st_signup_falls_back_to_standard_pricing(monkeypatch):
    existing_founders = [{"id": f"p-{i}", "is_founding_member": True} for i in range(FOUNDING_MEMBER_CAP)]
    supabase = FakeSupabaseClient(initial={"profiles": existing_founders})
    create_subscription = AsyncMock(return_value={"id": "sub_2", "short_url": "https://rzp.io/y"})
    monkeypatch.setattr("app.agent.tools.subscriptions.razorpay_client.create_subscription", create_subscription)
    ctx = _make_ctx(supabase)
    profile = {"id": "profile-new", "phone_number": "919876543211", "full_name": "Late Signup"}

    result = await create_and_send_subscription(ctx, profile)

    assert result["is_founding_member"] is False
    assert create_subscription.call_args.kwargs["plan_id"] is None  # falls back to settings.razorpay_subscription_plan_id
    sub_row = [r for r in supabase.rows("subscriptions") if r["profile_id"] == "profile-new"][0]
    assert sub_row["amount"] == STANDARD_AMOUNT
    assert "Founding Member" not in ctx.whatsapp.send_text.call_args.args[1]


@pytest.mark.asyncio
async def test_founding_member_flag_set_on_profile(monkeypatch):
    supabase = FakeSupabaseClient(initial={"profiles": [{"id": "profile-1", "is_founding_member": False}]})
    monkeypatch.setattr(
        "app.agent.tools.subscriptions.razorpay_client.create_subscription",
        AsyncMock(return_value={"id": "sub_3", "short_url": "https://rzp.io/z"}),
    )
    ctx = _make_ctx(supabase)
    profile = {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}

    await create_and_send_subscription(ctx, profile)

    assert supabase.rows("profiles")[0]["is_founding_member"] is True


@pytest.mark.asyncio
async def test_founding_member_gets_identical_subscriptions_shape_to_standard(monkeypatch):
    """Founding Members get the exact same is_active_subscriber path --
    status='active' after webhook confirmation, no separate access branch."""
    supabase = FakeSupabaseClient()
    monkeypatch.setattr(
        "app.agent.tools.subscriptions.razorpay_client.create_subscription",
        AsyncMock(return_value={"id": "sub_4", "short_url": "https://rzp.io/w"}),
    )
    ctx = _make_ctx(supabase)
    profile = {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}

    await create_and_send_subscription(ctx, profile)

    sub_row = supabase.rows("subscriptions")[0]
    assert sub_row["status"] == "trial"
    assert sub_row["billing_cycle"] == "Monthly"
    assert sub_row["payment_provider"] == "razorpay"
