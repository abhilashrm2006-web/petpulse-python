"""Covers the deterministic first-contact registration wizard: greeting,
then three questions (owner name, pet name, city) -- identical for every
customer, no new-vs-existing branching. Runs entirely outside the LLM
agent loop -- every assertion here is about direct state transitions and
WhatsApp sends, not model behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ingestion.registration import handle_registration
from app.ingestion.webhook import ExtractedMessage
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase=None, settings=None):
    whatsapp = SimpleNamespace(send_text=AsyncMock(), send_interactive_buttons=AsyncMock())
    default_settings = SimpleNamespace(razorpay_founding_plan_id="plan_FOUNDING_TEST", razorpay_subscription_plan_id="plan_STANDARD_TEST")
    return SimpleNamespace(supabase=supabase or FakeSupabaseClient(), whatsapp=whatsapp, settings=settings or default_settings)


def _msg(phone="919876543210", text="", button_reply_id=None, sender_name="Jane"):
    return ExtractedMessage(
        phone_number=phone, sender_name=sender_name, message_id="wamid.1", timestamp="1700000000",
        message_type="text", text=text, button_reply_id=button_reply_id,
    )


def _profile(**overrides):
    base = {"id": "profile-1", "phone_number": "919876543210", "full_name": "", "role": "customer", "registration_step": "awaiting_customer_name"}
    base.update(overrides)
    return base


# --- entry / gating ---------------------------------------------------

@pytest.mark.asyncio
async def test_brand_new_number_gets_greeting_and_name_prompt():
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="hi"))

    assert handled is True
    assert ctx.whatsapp.send_text.await_count == 2
    assert "Pulsy" in ctx.whatsapp.send_text.call_args_list[0].args[1]
    assert "full name" in ctx.whatsapp.send_text.call_args_list[1].args[1]
    ctx.whatsapp.send_interactive_buttons.assert_not_awaited()

    profile = supabase.rows("profiles")[0]
    assert profile["registration_step"] == "awaiting_customer_name"
    assert profile["role"] == "customer"


@pytest.mark.asyncio
async def test_already_completed_profile_is_never_intercepted():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="completed")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="anything"))

    assert handled is False
    ctx.whatsapp.send_text.assert_not_awaited()
    ctx.whatsapp.send_interactive_buttons.assert_not_awaited()


@pytest.mark.asyncio
async def test_vet_profile_is_never_intercepted_even_without_registration_step():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(role="vet", registration_step=None)]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="hello"))

    assert handled is False
    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_registration_step_clears_it_and_falls_through():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_something_removed")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="hello"))

    assert handled is False
    assert supabase.rows("profiles")[0]["registration_step"] is None


# --- name / pet name / city --------------------------------------------

@pytest.mark.asyncio
async def test_customer_name_step_saves_name_and_asks_pet_name():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="Anudeep Reddy"))

    assert handled is True
    profile = supabase.rows("profiles")[0]
    assert profile["full_name"] == "Anudeep Reddy"
    assert profile["registration_step"] == "awaiting_pet_name"
    assert "Anudeep Reddy" in ctx.whatsapp.send_text.call_args.args[1]


@pytest.mark.asyncio
async def test_customer_name_step_rejects_junk_and_reprompts():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="Sorry im a veterinarian"))

    assert handled is True
    profile = supabase.rows("profiles")[0]
    assert profile["registration_step"] == "awaiting_customer_name"
    assert profile.get("full_name") != "Sorry im a veterinarian"
    ctx.whatsapp.send_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_pet_name_step_rejects_junk_and_reprompts():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_pet_name")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="So many we have 12 dogs"))

    assert handled is True
    assert supabase.rows("pets") == []
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_pet_name"


@pytest.mark.asyncio
async def test_pet_name_step_creates_pet_and_asks_city():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_pet_name")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="Bobby"))

    assert handled is True
    pet = supabase.rows("pets")[0]
    assert pet["name"] == "Bobby"
    assert pet["profile_id"] == "profile-1"
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_city"
    assert "city" in ctx.whatsapp.send_text.call_args.args[1].lower()


@pytest.mark.asyncio
async def test_pet_name_step_survives_a_db_trigger_already_creating_pet_members():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_pet_name")]})
    supabase.force_conflict_on_insert("pet_members")
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="Bobby"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_city"


@pytest.mark.asyncio
async def test_city_step_empty_reply_reprompts():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_city")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="   "))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_city"


@pytest.mark.asyncio
async def test_city_step_saves_city_and_completes_registration():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [_profile(registration_step="awaiting_city", full_name="Anudeep Reddy")],
            "pets": [{"id": "pet-1", "profile_id": "profile-1", "name": "Bobby", "created_at": "2026-01-01"}],
        }
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="Chennai"))

    assert handled is True
    profile = supabase.rows("profiles")[0]
    assert profile["city"] == "Chennai"
    assert profile["registration_step"] == "completed"
    ctx.whatsapp.send_text.assert_awaited_once()
    ctx.whatsapp.send_interactive_buttons.assert_not_awaited()
    message = ctx.whatsapp.send_text.call_args.args[1]
    assert "Anudeep" in message
    assert "Bobby" in message
    assert "free" in message.lower()
    assert "₹399" in message
