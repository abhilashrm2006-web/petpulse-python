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


# --- onboarding_events instrumentation ----------------------------------

@pytest.mark.asyncio
async def test_accepted_customer_name_logs_an_onboarding_event():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    await handle_registration(ctx, _msg(text="Anudeep Reddy"))

    events = supabase.rows("onboarding_events")
    assert len(events) == 1
    assert events[0]["registration_step"] == "awaiting_customer_name"
    assert events[0]["validator_result"] == "accepted"
    assert events[0]["raw_input"] == "Anudeep Reddy"
    assert events[0]["rejection_reason"] is None


@pytest.mark.asyncio
async def test_rejected_customer_name_logs_an_onboarding_event_with_a_reason():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    await handle_registration(ctx, _msg(text="Sorry im a veterinarian"))

    events = supabase.rows("onboarding_events")
    assert len(events) == 1
    assert events[0]["registration_step"] == "awaiting_customer_name"
    assert events[0]["validator_result"] == "rejected"
    assert events[0]["rejection_reason"] == "not_name_like"


@pytest.mark.asyncio
async def test_pet_name_step_logs_accepted_and_rejected_events():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_pet_name")]})
    ctx = _make_ctx(supabase)

    await handle_registration(ctx, _msg(text="So many we have 12 dogs"))
    await handle_registration(ctx, _msg(text="Bobby"))

    events = supabase.rows("onboarding_events")
    assert len(events) == 2
    assert events[0]["registration_step"] == "awaiting_pet_name"
    assert events[0]["validator_result"] == "rejected"
    assert events[1]["validator_result"] == "accepted"


@pytest.mark.asyncio
async def test_missing_onboarding_events_table_never_breaks_registration():
    """Logging is best-effort observability, not load-bearing -- if the
    onboarding_events table doesn't exist yet (e.g. migration not applied),
    the actual registration step must still succeed."""
    from app.ingestion.registration import _log_onboarding_event

    class _BoomClient:
        def table(self, name):
            if name == "onboarding_events":
                raise Exception('relation "onboarding_events" does not exist')
            raise AssertionError(f"unexpected table access: {name}")

    try:
        _log_onboarding_event(_BoomClient(), "profile-1", "awaiting_customer_name", "Anudeep Reddy", accepted=True)
    except Exception:
        pytest.fail("_log_onboarding_event must swallow its own exceptions")


@pytest.mark.asyncio
async def test_handle_registration_succeeds_even_if_event_logging_fails():
    """End-to-end: the actual wizard step (saving the name, advancing
    registration_step, replying) must complete normally even when
    onboarding_events logging is broken."""
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    real_table = supabase.table

    def _table(name):
        if name == "onboarding_events":
            raise Exception('relation "onboarding_events" does not exist')
        return real_table(name)

    supabase.table = _table

    handled = await handle_registration(ctx, _msg(text="Anudeep Reddy"))

    assert handled is True
    profile = supabase.rows("profiles")[0]
    assert profile["full_name"] == "Anudeep Reddy"
    assert profile["registration_step"] == "awaiting_pet_name"


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


# --- funnel visibility: conversations/messages + step history --------------

@pytest.mark.asyncio
async def test_wizard_handled_turn_still_logs_a_conversation_and_messages():
    """The exact bug this fixes: 59/106 stuck profiles had zero
    conversations/messages rows because handle_registration returns before
    run_agent_turn (which used to own all message-logging) ever runs."""
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    await handle_registration(ctx, _msg(text="Anudeep Reddy"))

    conversations = supabase.rows("conversations")
    messages = supabase.rows("messages")
    assert len(conversations) == 1
    assert conversations[0]["profile_id"] == "profile-1"
    assert any(m["sender_type"] == "user" and m["content"] == "Anudeep Reddy" for m in messages)
    assert any(m["sender_type"] == "assistant" for m in messages)


@pytest.mark.asyncio
async def test_a_rejected_reprompt_also_logs_a_conversation_turn():
    """Not just accepted steps -- a rejected/re-prompted reply is exactly
    the ambiguous "did they reply at all" case the spec calls out."""
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    await handle_registration(ctx, _msg(text="Sorry im a veterinarian"))

    messages = supabase.rows("messages")
    assert any(m["content"] == "Sorry im a veterinarian" for m in messages)


@pytest.mark.asyncio
async def test_step_transitions_are_recorded_in_history():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    await handle_registration(ctx, _msg(text="Anudeep Reddy"))

    history = supabase.rows("registration_step_history")
    assert len(history) == 1
    assert history[0]["from_step"] == "awaiting_customer_name"
    assert history[0]["to_step"] == "awaiting_pet_name"


@pytest.mark.asyncio
async def test_brand_new_number_logs_a_step_history_row_from_none():
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)

    await handle_registration(ctx, _msg(text="hi"))

    history = supabase.rows("registration_step_history")
    assert len(history) == 1
    assert history[0]["from_step"] is None
    assert history[0]["to_step"] == "awaiting_customer_name"


@pytest.mark.asyncio
async def test_missing_instrumentation_tables_never_break_the_wizard():
    """Same best-effort guarantee as onboarding_events -- registration_step_history
    and conversations/messages logging must never block the actual wizard step."""
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    real_table = supabase.table

    def _table(name):
        if name in ("registration_step_history", "conversations", "messages"):
            raise Exception(f'relation "{name}" does not exist')
        return real_table(name)

    supabase.table = _table

    handled = await handle_registration(ctx, _msg(text="Anudeep Reddy"))

    assert handled is True
    profile = supabase.rows("profiles")[0]
    assert profile["full_name"] == "Anudeep Reddy"
    assert profile["registration_step"] == "awaiting_pet_name"


# --- bot-echo loop circuit breaker --------------------------------------

@pytest.mark.asyncio
async def test_repeated_identical_rejection_goes_silent_after_threshold():
    """Live bug (2026-09-04): a business auto-responder bounced the exact
    same canned reply back at us 187 times -- our re-prompt triggered their
    auto-reply, which triggered our next re-prompt, forever. The 3rd
    identical rejection in a row must not send a reply at all."""
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)
    canned = "Hi, welcome to Asset Tree Homes! Our executives will contact you shortly. :)"

    await handle_registration(ctx, _msg(text=canned))
    await handle_registration(ctx, _msg(text=canned))
    assert ctx.whatsapp.send_text.await_count == 2  # first two still get the normal re-prompt

    ctx.whatsapp.send_text.reset_mock()
    handled = await handle_registration(ctx, _msg(text=canned))

    assert handled is True
    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_rejection_counter_keeps_climbing_once_tripped():
    """Once tripped, later identical repeats stay silent too -- it's a
    permanent breaker for that exact text, not a one-time skip."""
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)
    canned = "Hi, welcome to Asset Tree Homes! Our executives will contact you shortly. :)"

    for _ in range(6):
        await handle_registration(ctx, _msg(text=canned))

    ctx.whatsapp.send_text.reset_mock()
    await handle_registration(ctx, _msg(text=canned))
    ctx.whatsapp.send_text.assert_not_awaited()
    assert supabase.rows("profiles")[0]["repeated_rejection_count"] == 7


@pytest.mark.asyncio
async def test_a_real_human_varying_their_wording_never_gets_silenced():
    """A genuinely persistent human who keeps trying different wrong
    answers must keep getting re-prompted -- only an IDENTICAL repeat
    trips the breaker."""
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    await handle_registration(ctx, _msg(text="Sorry im a veterinarian"))
    await handle_registration(ctx, _msg(text="What is this"))
    await handle_registration(ctx, _msg(text="No Hindi language?"))

    assert ctx.whatsapp.send_text.await_count == 3


@pytest.mark.asyncio
async def test_a_different_text_after_a_repeat_resets_the_streak():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)
    canned = "Hi, welcome to Asset Tree Homes!"

    await handle_registration(ctx, _msg(text=canned))
    await handle_registration(ctx, _msg(text=canned))
    await handle_registration(ctx, _msg(text="Priya Sharma"))  # a real name, accepted -- resets streak

    profile = supabase.rows("profiles")[0]
    assert profile["repeated_rejection_count"] == 0
    assert profile["registration_step"] == "awaiting_pet_name"


@pytest.mark.asyncio
async def test_missing_rejection_streak_columns_never_break_the_wizard():
    supabase = FakeSupabaseClient(initial={"profiles": [_profile(registration_step="awaiting_customer_name")]})
    ctx = _make_ctx(supabase)

    real_table = supabase.table
    calls = {"n": 0}

    def _table(name):
        if name == "profiles":
            calls["n"] += 1
            if calls["n"] == 2:  # the rejection-streak update specifically
                raise Exception('column "repeated_rejection_count" does not exist')
        return real_table(name)

    supabase.table = _table

    handled = await handle_registration(ctx, _msg(text="Sorry im a veterinarian"))

    assert handled is True
    ctx.whatsapp.send_text.assert_awaited_once()
