"""Covers the deterministic first-contact registration wizard: greeting,
New/Existing Member choice, structured intake (name/pet details/KCI/
vaccination/microchip/city), membership tier choice, and the Razorpay
Subscriptions flow for Subscriber. Runs entirely outside the LLM agent
loop -- every assertion here is about direct state transitions and
WhatsApp sends, not model behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ingestion.registration import handle_registration, handle_subscription_webhook
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
    base = {"id": "profile-1", "phone_number": "919876543210", "full_name": "", "role": "customer", "registration_step": "awaiting_member_type"}
    base.update(overrides)
    return base


# --- entry / gating ---------------------------------------------------

@pytest.mark.asyncio
async def test_brand_new_number_gets_greeting_and_member_type_buttons():
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="hi"))

    assert handled is True
    ctx.whatsapp.send_text.assert_awaited_once()
    assert "Pulsy" in ctx.whatsapp.send_text.call_args.args[1]
    ctx.whatsapp.send_interactive_buttons.assert_awaited_once()
    button_ids = {b["id"] for b in ctx.whatsapp.send_interactive_buttons.call_args.args[2]}
    assert button_ids == {"member_type|new", "member_type|existing"}

    profile = supabase.rows("profiles")[0]
    assert profile["registration_step"] == "awaiting_member_type"
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


# --- New Member path ----------------------------------------------------

@pytest.mark.asyncio
async def test_member_type_new_asks_for_name():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile()]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(button_reply_id="member_type|new"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_customer_name"
    ctx.whatsapp.send_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_member_type_invalid_reply_reshows_buttons():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile()]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="huh?"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_member_type"
    ctx.whatsapp.send_interactive_buttons.assert_awaited_once()


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
async def test_pet_name_step_creates_pet_and_asks_dob():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_pet_name")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="Bobby"))

    assert handled is True
    pet = supabase.rows("pets")[0]
    assert pet["name"] == "Bobby"
    assert pet["profile_id"] == "profile-1"
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_pet_dob"


@pytest.mark.asyncio
async def test_pet_name_step_survives_a_db_trigger_already_creating_pet_members():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_pet_name")]})
    supabase.force_conflict_on_insert("pet_members")
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="Bobby"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_pet_dob"


@pytest.mark.asyncio
@pytest.mark.parametrize("typed,expected", [("2020-05-01", "2020-05-01"), ("01/05/2020", "2020-05-01"), ("2020", "2020-01-01")])
async def test_pet_dob_accepts_several_formats(typed, expected):
    supabase = FakeSupabaseClient(
        initial={"profiles": [_profile(registration_step="awaiting_pet_dob")], "pets": [{"id": "pet-1", "profile_id": "profile-1", "name": "Bobby"}]}
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text=typed))

    assert handled is True
    assert supabase.rows("pets")[0]["date_of_birth"] == expected
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_pet_age"


@pytest.mark.asyncio
async def test_pet_dob_rejects_garbage():
    supabase = FakeSupabaseClient(
        initial={"profiles": [_profile(registration_step="awaiting_pet_dob")], "pets": [{"id": "pet-1", "profile_id": "profile-1", "name": "Bobby"}]}
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="whenever"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_pet_dob"
    assert "date_of_birth" not in supabase.rows("pets")[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("typed,expected", [("2", 2), ("2.5", 3), ("1.5", 2)])
async def test_pet_age_rounds_half_up_consistently(typed, expected):
    supabase = FakeSupabaseClient(
        initial={"profiles": [_profile(registration_step="awaiting_pet_age")], "pets": [{"id": "pet-1", "profile_id": "profile-1", "name": "Bobby"}]}
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text=typed))

    assert handled is True
    assert supabase.rows("pets")[0]["age"] == expected
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_pet_weight"


@pytest.mark.asyncio
async def test_pet_weight_converts_lbs_to_kg():
    supabase = FakeSupabaseClient(
        initial={"profiles": [_profile(registration_step="awaiting_pet_weight")], "pets": [{"id": "pet-1", "profile_id": "profile-1", "name": "Bobby"}]}
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="20 lbs"))

    assert handled is True
    assert supabase.rows("pets")[0]["weight"] == pytest.approx(9.1, abs=0.05)
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_kci_status"
    button_ids = {b["id"] for b in ctx.whatsapp.send_interactive_buttons.call_args.args[2]}
    assert button_ids == {"kci_status|yes", "kci_status|no"}


@pytest.mark.asyncio
async def test_kci_status_yes_sets_certificate_and_asks_vaccination():
    supabase = FakeSupabaseClient(
        initial={"profiles": [_profile(registration_step="awaiting_kci_status")], "pets": [{"id": "pet-1", "profile_id": "profile-1", "name": "Bobby"}]}
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(button_reply_id="kci_status|yes"))

    assert handled is True
    assert supabase.rows("pets")[0]["has_kci_certificate"] is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_vaccination_status"


@pytest.mark.asyncio
async def test_vaccination_status_no_sets_false_and_asks_microchip():
    supabase = FakeSupabaseClient(
        initial={"profiles": [_profile(registration_step="awaiting_vaccination_status")], "pets": [{"id": "pet-1", "profile_id": "profile-1", "name": "Bobby"}]}
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(button_reply_id="vaccination_status|no"))

    assert handled is True
    assert supabase.rows("pets")[0]["vaccination_confirmed"] is False
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_microchip_status"


@pytest.mark.asyncio
async def test_microchip_status_yes_asks_for_number():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_microchip_status")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(button_reply_id="microchip_status|yes"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_microchip_number"


@pytest.mark.asyncio
async def test_microchip_status_no_skips_straight_to_city():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_microchip_status")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(button_reply_id="microchip_status|no"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_city"


@pytest.mark.asyncio
async def test_microchip_number_step_saves_number_and_asks_city():
    supabase = FakeSupabaseClient(
        initial={"profiles": [_profile(registration_step="awaiting_microchip_number")], "pets": [{"id": "pet-1", "profile_id": "profile-1", "name": "Bobby"}]}
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="985 123 456 789 012"))

    assert handled is True
    assert supabase.rows("pets")[0]["microchip_number"] == "985 123 456 789 012"
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_city"


@pytest.mark.asyncio
async def test_city_step_saves_city_and_sends_tier_choice():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_city")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="Chennai"))

    assert handled is True
    assert supabase.rows("profiles")[0]["city"] == "Chennai"
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_tier_choice"
    assert ctx.whatsapp.send_text.await_count == 2
    all_texts = " ".join(call.args[1] for call in ctx.whatsapp.send_text.await_args_list)
    assert "Welcome to PetPulse AI World" in all_texts
    # Feature pitch leads with the passport/vault/multi-pet anchors, not the AI chat/triage --
    # matches the product's "between vet visits" positioning, not "AI vet in your pocket."
    assert all_texts.index("vaccination passport") < all_texts.index("records vault") < all_texts.index("pets on one account")
    assert all_texts.index("pets on one account") < all_texts.index("AI health chats") < all_texts.index("emergency triage")
    button_ids = {b["id"] for b in ctx.whatsapp.send_interactive_buttons.call_args.args[2]}
    assert button_ids == {"tier_choice|free", "tier_choice|subscriber"}


@pytest.mark.asyncio
async def test_tier_choice_free_asks_continue_or_subscribe():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_tier_choice")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(button_reply_id="tier_choice|free"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_free_subchoice"
    button_ids = {b["id"] for b in ctx.whatsapp.send_interactive_buttons.call_args.args[2]}
    assert button_ids == {"free_subchoice|continue", "free_subchoice|subscribe"}


@pytest.mark.asyncio
async def test_free_subchoice_continue_completes_registration():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_free_subchoice")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(button_reply_id="free_subchoice|continue"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "completed"


@pytest.mark.asyncio
async def test_tier_choice_subscriber_creates_a_real_subscription(monkeypatch):
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_tier_choice", full_name="Anudeep")]})
    ctx = _make_ctx(supabase)
    create_subscription = AsyncMock(return_value={"id": "sub_123", "short_url": "https://rzp.io/sub/abc"})
    monkeypatch.setattr("app.agent.tools.subscriptions.razorpay_client.create_subscription", create_subscription)

    handled = await handle_registration(ctx, _msg(button_reply_id="tier_choice|subscriber"))

    assert handled is True
    create_subscription.assert_awaited_once()
    sub_row = supabase.rows("subscriptions")[0]
    assert sub_row["provider_subscription_id"] == "sub_123"
    assert sub_row["status"] == "trial"
    assert sub_row["plan_name"] == "Premium"
    assert sub_row["billing_cycle"] == "Monthly"
    assert sub_row["start_date"]
    assert supabase.rows("profiles")[0]["registration_step"] == "completed"
    assert "https://rzp.io/sub/abc" in ctx.whatsapp.send_text.call_args.args[1]


@pytest.mark.asyncio
async def test_free_subchoice_subscribe_also_creates_a_subscription(monkeypatch):
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_free_subchoice")]})
    ctx = _make_ctx(supabase)
    create_subscription = AsyncMock(return_value={"id": "sub_456", "short_url": "https://rzp.io/sub/def"})
    monkeypatch.setattr("app.agent.tools.subscriptions.razorpay_client.create_subscription", create_subscription)

    handled = await handle_registration(ctx, _msg(button_reply_id="free_subchoice|subscribe"))

    assert handled is True
    create_subscription.assert_awaited_once()
    assert supabase.rows("subscriptions")[0]["provider_subscription_id"] == "sub_456"


@pytest.mark.asyncio
async def test_subscription_creation_failure_sends_friendly_error_not_a_crash(monkeypatch):
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_tier_choice")]})
    ctx = _make_ctx(supabase)
    monkeypatch.setattr(
        "app.agent.tools.subscriptions.razorpay_client.create_subscription", AsyncMock(side_effect=RuntimeError("razorpay down"))
    )

    handled = await handle_registration(ctx, _msg(button_reply_id="tier_choice|subscriber"))

    assert handled is True
    assert supabase.rows("subscriptions") == []
    ctx.whatsapp.send_text.assert_awaited_once()
    assert "went wrong" in ctx.whatsapp.send_text.call_args.args[1].lower()
    # Being stuck re-showing the tier-choice buttons forever would be worse
    # than completing the wizard with a failed subscribe attempt they can
    # retry later via the start_subscription agent tool.
    assert supabase.rows("profiles")[0]["registration_step"] == "completed"


# --- Existing Member path -----------------------------------------------

@pytest.mark.asyncio
async def test_existing_member_unrecognized_phone_falls_back_to_new_member():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_existing_phone")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="919000000000"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_customer_name"
    assert ctx.whatsapp.send_text.await_count == 2


@pytest.mark.asyncio
async def test_existing_member_recognized_phone_shows_summary_with_embedded_id():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                _profile(registration_step="awaiting_existing_phone"),
                {"id": "old-profile", "phone_number": "919111111111", "full_name": "Anudeep", "role": "customer", "city": "Chennai"},
            ],
            "pets": [{"id": "pet-old", "profile_id": "old-profile", "name": "Bobby", "date_of_birth": "2023-01-01", "age": 2, "weight": 12, "has_kci_certificate": True, "vaccination_confirmed": True, "microchip_number": "12345"}],
        }
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="919111111111"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_existing_verify"
    call = ctx.whatsapp.send_interactive_buttons.call_args
    assert "Bobby" in call.args[1]
    button_ids = {b["id"] for b in call.args[2]}
    assert button_ids == {"existing_verify|old-profile|yes", "existing_verify|old-profile|no"}


@pytest.mark.asyncio
async def test_existing_verify_yes_links_number_and_completes_matched_profile():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                _profile(registration_step="awaiting_existing_verify"),  # placeholder row for the new number, id=profile-1
                {"id": "old-profile", "phone_number": "919111111111", "full_name": "Anudeep", "role": "customer", "registration_step": "completed"},
            ]
        }
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(button_reply_id="existing_verify|old-profile|yes"))

    assert handled is True
    remaining = supabase.rows("profiles")
    assert len(remaining) == 1  # placeholder row deleted
    assert remaining[0]["id"] == "old-profile"
    assert remaining[0]["phone_number"] == "919876543210"
    assert remaining[0]["registration_step"] == "completed"
    assert "Welcome back" in ctx.whatsapp.send_text.call_args.args[1]


@pytest.mark.asyncio
async def test_existing_verify_no_returns_to_member_type_choice():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                _profile(registration_step="awaiting_existing_verify"),
                {"id": "old-profile", "phone_number": "919111111111", "full_name": "Anudeep", "role": "customer"},
            ]
        }
    )
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(button_reply_id="existing_verify|old-profile|no"))

    assert handled is True
    assert len(supabase.rows("profiles")) == 2  # nothing deleted
    profile_1 = next(p for p in supabase.rows("profiles") if p["id"] == "profile-1")
    assert profile_1["registration_step"] == "awaiting_member_type"
    ctx.whatsapp.send_interactive_buttons.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_verify_requires_a_button_tap():
    """The matched profile's id lives only in the button id -- a typed
    "yes" can't be resolved back to it, so this must ask them to tap again
    rather than silently doing nothing or crashing."""
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_existing_verify")]})
    ctx = _make_ctx(supabase)

    handled = await handle_registration(ctx, _msg(text="yes"))

    assert handled is True
    assert supabase.rows("profiles")[0]["registration_step"] == "awaiting_existing_verify"
    ctx.whatsapp.send_text.assert_awaited_once()


# --- subscription webhook ------------------------------------------------

@pytest.mark.asyncio
async def test_subscription_webhook_activated_marks_active():
    supabase = FakeSupabaseClient(initial={"subscriptions": [{"id": "row-1", "provider_subscription_id": "sub_123", "status": "trial"}]})
    ctx = _make_ctx(supabase)
    event = {"event": "subscription.activated", "payload": {"subscription": {"entity": {"id": "sub_123", "notes": {}}}}}

    handled = await handle_subscription_webhook(ctx, event)

    assert handled is True
    assert supabase.rows("subscriptions")[0]["status"] == "active"


@pytest.mark.asyncio
async def test_subscription_webhook_cancelled_marks_cancelled():
    supabase = FakeSupabaseClient(initial={"subscriptions": [{"id": "row-1", "provider_subscription_id": "sub_123", "status": "active"}]})
    ctx = _make_ctx(supabase)
    event = {"event": "subscription.cancelled", "payload": {"subscription": {"entity": {"id": "sub_123", "notes": {}}}}}

    handled = await handle_subscription_webhook(ctx, event)

    assert handled is True
    assert supabase.rows("subscriptions")[0]["status"] == "cancelled"
    assert supabase.rows("subscriptions")[0]["cancelled_at"] is not None


@pytest.mark.asyncio
async def test_subscription_webhook_unknown_subscription_id_returns_false():
    supabase = FakeSupabaseClient(initial={"subscriptions": []})
    ctx = _make_ctx(supabase)
    event = {"event": "subscription.activated", "payload": {"subscription": {"entity": {"id": "sub_999", "notes": {}}}}}

    handled = await handle_subscription_webhook(ctx, event)

    assert handled is False


@pytest.mark.asyncio
async def test_subscription_webhook_ignores_non_subscription_events():
    ctx = _make_ctx(FakeSupabaseClient())

    handled = await handle_subscription_webhook(ctx, {"event": "payment_link.paid", "payload": {}})

    assert handled is False
