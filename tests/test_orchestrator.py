"""Exercises the tool-calling loop end-to-end with OpenAI, Supabase, and
WhatsApp all mocked — asserts the loop calls the right tool for a given
turn and stops correctly once the model returns plain text."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent import orchestrator
from app.config import Settings
from app.deps import AppContext
from app.ingestion.context import AgentContext
from app.ingestion.webhook import ExtractedMessage


def _fake_tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _fake_response(content: str | None, tool_calls: list | None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _make_supabase_mock() -> MagicMock:
    client = MagicMock()
    table_mock = MagicMock()
    execute_result = MagicMock()
    execute_result.data = [{"id": "conv-1"}]
    # every chained call (.insert/.select/.eq/.order/.limit/...) returns the same mock,
    # and .execute() always returns a result carrying .data — enough for the orchestrator's needs.
    table_mock.insert.return_value = table_mock
    table_mock.select.return_value = table_mock
    table_mock.eq.return_value = table_mock
    table_mock.order.return_value = table_mock
    table_mock.limit.return_value = table_mock
    table_mock.update.return_value = table_mock
    table_mock.execute.return_value = execute_result
    client.table.return_value = table_mock
    return client


def _make_agent_ctx(role: str = "customer", is_subscriber: bool = False) -> AgentContext:
    return AgentContext(
        profile={"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane", "onboarding_completed": True, "role": role},
        role=role,
        is_new_profile=False,
        pets=[{"id": "pet-1", "name": "Rex", "created_at": "2024-01-01"}],
        active_pet={"id": "pet-1", "name": "Rex"},
        active_pet_matched_from_message=True,
        memory_context=[],
        medical_context={},
        knowledge_base=[],
        open_session=None,
        pending_negotiation=None,
        onboarding={"complete": True, "missing_fields": []},
        is_subscriber=is_subscriber,
    )


@pytest.mark.asyncio
async def test_agent_calls_the_tool_the_model_picks_then_stops(monkeypatch):
    tool_calls_made = []

    async def fake_tool(ctx, agent_ctx, **kwargs):
        tool_calls_made.append(kwargs)
        return {"success": True, "severity": 4}

    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [{"type": "function", "function": {"name": "check_symptoms"}}])
    monkeypatch.setattr(orchestrator, "get_tool_fn", lambda name: fake_tool)
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)

    responses = [
        _fake_response(None, [_fake_tool_call("call-1", "check_symptoms", {"symptoms": "vomiting"})]),
        _fake_response("Thanks, I've logged that — please monitor closely.", None),
    ]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings,
        http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock()),
        supabase=_make_supabase_mock(),
        openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx()  # check_symptoms is open to Free customers too
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.1",
        timestamp="1700000000", message_type="text", text="Rex has been vomiting",
    )

    result = await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    assert tool_calls_made == [{"symptoms": "vomiting"}]
    assert result == "Thanks, I've logged that — please monitor closely."
    ctx.whatsapp.send_reply_and_chunk.assert_awaited_once_with("919876543210", result)


@pytest.mark.asyncio
async def test_self_messaging_tool_suppresses_final_text(monkeypatch):
    async def fake_tool(ctx, agent_ctx, **kwargs):
        return {"success": True, "mode": "doctor_catalogue_sent", "session_id": "s1"}

    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [{"type": "function", "function": {"name": "request_doctor_session"}}])
    monkeypatch.setattr(orchestrator, "get_tool_fn", lambda name: fake_tool)
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)

    responses = [
        _fake_response(None, [_fake_tool_call("call-1", "request_doctor_session", {"case_summary": "routine checkup"})]),
        _fake_response("", None),
    ]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings,
        http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock()),
        supabase=_make_supabase_mock(),
        openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=True)  # request_doctor_session is subscriber_only
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.2",
        timestamp="1700000001", message_type="text", text="I'd like to book a vet",
    )

    result = await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    assert result == ""
    ctx.whatsapp.send_reply_and_chunk.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["booked", "rescheduled", "prescription_delivered", "declined", "deletion_confirmation_sent", "deletion_done", "deletion_declined"],
)
async def test_booking_confirmation_suppresses_duplicate_agent_reply(monkeypatch, mode):
    """Reproduces a real reported bug: _finalize_booking already sends the
    full confirmation (date/time + Meet link) directly to both parties, but
    its mode wasn't in SELF_MESSAGING_MODES — so the agent composed a
    SECOND, redundant confirmation on top, and the customer saw the same
    date/time and Meet link twice across multiple WhatsApp bubbles.
    "prescription_delivered" is the same class of bug: deliver_prescription
    already sends the text/PDF directly. "declined" is the same class again,
    found via a repeated-message audit: decline_session unconditionally
    messages BOTH the customer and the doctor directly, so whichever one is
    the current actor already got the tool's own message too. The
    "deletion_*" modes are the chat-history-deletion flow (app/agent/tools/
    account.py), which also messages the customer directly at every step."""

    async def fake_tool(ctx, agent_ctx, **kwargs):
        return {"success": True, "mode": mode, "session_id": "s1", "when": "Tue 28 Jul, 11:30 AM IST", "meet_link": "https://meet.google.com/abc"}

    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [{"type": "function", "function": {"name": "book_slot"}}])
    monkeypatch.setattr(orchestrator, "get_tool_fn", lambda name: fake_tool)
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)

    responses = [
        _fake_response(None, [_fake_tool_call("call-1", "book_slot", {"session_id": "s1", "slot_start": "2026-07-28T11:30:00+05:30"})]),
        _fake_response("Booked — Tue 28 Jul, 11:30 AM IST.\n\nMeet link: https://meet.google.com/abc", None),
    ]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings,
        http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock()),
        supabase=_make_supabase_mock(),
        openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=True)  # book_slot is subscriber_only
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.3",
        timestamp="1700000002", message_type="text", text="Tuesday 11:30am works",
    )

    result = await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    assert result == ""
    ctx.whatsapp.send_reply_and_chunk.assert_not_awaited()


@pytest.mark.asyncio
async def test_free_customer_blocked_from_subscriber_only_tool_with_upsell(monkeypatch):
    """A Free customer calling a Subscriber-only tool (book_slot) must never
    reach the real tool function -- the orchestrator should short-circuit
    with a subscriber_only_feature error, and the agent's composed reply
    (relaying that error, per the system prompt) should still go out."""
    tool_calls_made = []

    async def fake_tool(ctx, agent_ctx, **kwargs):
        tool_calls_made.append(kwargs)
        return {"success": True, "mode": "booked"}

    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [{"type": "function", "function": {"name": "book_slot"}}])
    monkeypatch.setattr(orchestrator, "get_tool_fn", lambda name: fake_tool)
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)

    responses = [
        _fake_response(None, [_fake_tool_call("call-1", "book_slot", {"session_id": "s1", "slot_start": "2026-07-28T11:30:00+05:30"})]),
        _fake_response("That's a Subscriber feature — want the subscribe link?", None),
    ]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings,
        http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock()),
        supabase=_make_supabase_mock(),
        openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=False)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.4",
        timestamp="1700000003", message_type="text", text="Book me a vet slot",
    )

    result = await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    assert tool_calls_made == []  # the real tool must never have been called
    assert result == "That's a Subscriber feature — want the subscribe link?"
    ctx.whatsapp.send_reply_and_chunk.assert_awaited_once_with("919876543210", result)


@pytest.mark.asyncio
async def test_free_customer_can_still_reach_check_symptoms(monkeypatch):
    """The triage assessment itself is never paywalled -- a Free customer
    reporting a symptom must reach the real check_symptoms call. Only
    booking an actual consultation afterward is Subscriber-gated (covered
    by the book_slot test above)."""
    tool_calls_made = []

    async def fake_tool(ctx, agent_ctx, **kwargs):
        tool_calls_made.append(kwargs)
        return {"success": True, "severity": 4, "severity_display": "🟠 Urgent (4/5)"}

    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [{"type": "function", "function": {"name": "check_symptoms"}}])
    monkeypatch.setattr(orchestrator, "get_tool_fn", lambda name: fake_tool)
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)

    responses = [
        _fake_response(None, [_fake_tool_call("call-1", "check_symptoms", {"symptoms": "vomiting"})]),
        _fake_response("🟠 Urgent (4/5) — would you like to book a vet consultation for this?", None),
    ]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings,
        http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock()),
        supabase=_make_supabase_mock(),
        openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=False)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.5",
        timestamp="1700000004", message_type="text", text="Rex has been vomiting",
    )

    result = await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    assert tool_calls_made == [{"symptoms": "vomiting"}]  # the real tool WAS called for a Free customer
    assert result == "🟠 Urgent (4/5) — would you like to book a vet consultation for this?"
    ctx.whatsapp.send_reply_and_chunk.assert_awaited_once_with("919876543210", result)


@pytest.mark.asyncio
async def test_free_customer_gets_no_persistent_memory(monkeypatch):
    """Free customers start fresh every turn -- load_chat_history must not
    even be called, and nothing gets appended/extracted afterward."""
    load_calls = []
    append_calls = []

    async def fake_tool(ctx, agent_ctx, **kwargs):
        return {"success": True}

    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "get_tool_fn", lambda name: fake_tool)
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: load_calls.append(pet_id) or [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: append_calls.append(k))
    extract_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", extract_mock)

    responses = [_fake_response("Just a friendly reply.", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_sticker=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=False)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.6",
        timestamp="1700000005", message_type="text", text="hi",
    )

    await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    assert load_calls == []
    assert append_calls == []
    extract_mock.assert_not_awaited()
    ctx.whatsapp.send_sticker.assert_awaited_once_with("919876543210", settings.pulsy_welcome_sticker_url)


@pytest.mark.asyncio
async def test_subscriber_gets_pet_scoped_memory(monkeypatch):
    """Subscriber history is scoped to the active pet, not just the phone
    number, so a multi-pet account doesn't blend context across pets."""
    load_calls = []
    append_calls = []

    async def fake_tool(ctx, agent_ctx, **kwargs):
        return {"success": True}

    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "get_tool_fn", lambda name: fake_tool)
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: load_calls.append(pet_id) or [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: append_calls.append(k))
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    responses = [_fake_response("Sure, here's what I know about Rex.", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings, http=None, whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=True)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.7",
        timestamp="1700000006", message_type="text", text="tell me about Rex",
    )

    await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    assert load_calls == ["pet-1"]
    assert append_calls and append_calls[0].get("pet_id") == "pet-1"


@pytest.mark.asyncio
async def test_voice_reply_sent_for_subscriber_voice_note_in_regional_language(monkeypatch):
    """The actual feature: a Subscriber who sends a voice note gets a spoken
    reply back in the language they spoke, on top of the usual text reply."""
    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    detect_mock = AsyncMock(return_value="ta")
    synthesize_mock = AsyncMock(return_value=b"fake-mp3-bytes")
    monkeypatch.setattr(orchestrator, "detect_regional_language", detect_mock)
    monkeypatch.setattr(orchestrator, "synthesize_speech", synthesize_mock)
    monkeypatch.setattr(orchestrator, "upload_to_storage", MagicMock())
    monkeypatch.setattr(orchestrator, "sign_storage_url", MagicMock(return_value="https://signed.example/reply.mp3"))

    responses = [_fake_response("Namaste, Rex ஆரோக்கியமாக இருக்கிறார்.", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10, voice_replies_enabled=True)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_audio=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=True)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.voice1",
        timestamp="1700000010", message_type="audio", text="",
    )

    result = await orchestrator.run_agent_turn(ctx, agent_ctx, extracted, media_context="[Voice note analysis] என் நாய் சரியில்லை")

    detect_mock.assert_awaited_once()
    synthesize_mock.assert_awaited_once_with(ctx.openai, settings, result, language="Tamil")
    ctx.whatsapp.send_audio.assert_awaited_once_with("919876543210", "https://signed.example/reply.mp3")


@pytest.mark.asyncio
async def test_voice_reply_not_sent_for_text_message(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    detect_mock = AsyncMock(return_value="ta")
    monkeypatch.setattr(orchestrator, "detect_regional_language", detect_mock)
    monkeypatch.setattr(orchestrator, "synthesize_speech", AsyncMock(return_value=b"bytes"))

    responses = [_fake_response("Sure thing.", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10, voice_replies_enabled=True)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_audio=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=True)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.text1",
        timestamp="1700000011", message_type="text", text="how is Rex doing",
    )

    await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    detect_mock.assert_not_awaited()
    ctx.whatsapp.send_audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_reply_not_sent_for_free_tier_voice_note(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)

    detect_mock = AsyncMock(return_value="ta")
    monkeypatch.setattr(orchestrator, "detect_regional_language", detect_mock)
    monkeypatch.setattr(orchestrator, "synthesize_speech", AsyncMock(return_value=b"bytes"))

    responses = [_fake_response("Here's some general advice.", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10, voice_replies_enabled=True)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_audio=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=False)  # Free tier
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.voice2",
        timestamp="1700000012", message_type="audio", text="",
    )

    await orchestrator.run_agent_turn(ctx, agent_ctx, extracted, media_context="[Voice note analysis] hello")

    detect_mock.assert_not_awaited()
    ctx.whatsapp.send_audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_reply_not_sent_when_voice_replies_disabled(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    detect_mock = AsyncMock(return_value="ta")
    monkeypatch.setattr(orchestrator, "detect_regional_language", detect_mock)
    monkeypatch.setattr(orchestrator, "synthesize_speech", AsyncMock(return_value=b"bytes"))

    responses = [_fake_response("Sure thing.", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10, voice_replies_enabled=False)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_audio=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=True)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.voice3",
        timestamp="1700000013", message_type="audio", text="",
    )

    await orchestrator.run_agent_turn(ctx, agent_ctx, extracted, media_context="[Voice note analysis] hello")

    detect_mock.assert_not_awaited()
    ctx.whatsapp.send_audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_reply_not_sent_when_detected_language_is_english(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    monkeypatch.setattr(orchestrator, "detect_regional_language", AsyncMock(return_value="en"))
    synthesize_mock = AsyncMock(return_value=b"bytes")
    monkeypatch.setattr(orchestrator, "synthesize_speech", synthesize_mock)

    responses = [_fake_response("Sure thing.", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10, voice_replies_enabled=True)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_audio=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=True)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.voice4",
        timestamp="1700000014", message_type="audio", text="",
    )

    await orchestrator.run_agent_turn(ctx, agent_ctx, extracted, media_context="[Voice note analysis] hello there")

    synthesize_mock.assert_not_awaited()
    ctx.whatsapp.send_audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_reply_failure_does_not_break_the_turn(monkeypatch):
    """synthesize_speech returning None (any TTS/API failure) must still
    leave the already-sent text reply intact -- no exception, no crash."""
    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    monkeypatch.setattr(orchestrator, "detect_regional_language", AsyncMock(return_value="hi"))
    monkeypatch.setattr(orchestrator, "synthesize_speech", AsyncMock(return_value=None))

    responses = [_fake_response("Sure thing.", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10, voice_replies_enabled=True)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_audio=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=True)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.voice5",
        timestamp="1700000015", message_type="audio", text="",
    )

    result = await orchestrator.run_agent_turn(ctx, agent_ctx, extracted, media_context="[Voice note analysis] namaste")

    assert result == "Sure thing."
    ctx.whatsapp.send_reply_and_chunk.assert_awaited_once()
    ctx.whatsapp.send_audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_welcome_image_not_sent_for_non_greeting_message(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    responses = [_fake_response("Sorry to hear that — tell me more.", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_sticker=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=False)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.greet2",
        timestamp="1700000016", message_type="text", text="hi, my dog is limping",
    )

    await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    ctx.whatsapp.send_sticker.assert_not_awaited()


@pytest.mark.asyncio
async def test_welcome_image_not_sent_for_vet_greeting(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    responses = [_fake_response("Hey doc, what can I help with?", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_sticker=AsyncMock()),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(role="vet")
    extracted = ExtractedMessage(
        phone_number="919000000001", sender_name="Dr. Rao", message_id="wamid.greet3",
        timestamp="1700000017", message_type="text", text="hello",
    )

    await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    ctx.whatsapp.send_sticker.assert_not_awaited()


@pytest.mark.asyncio
async def test_welcome_image_send_failure_does_not_break_the_turn(monkeypatch):
    """A WhatsApp send failure for the image must never take down the rest
    of the turn -- the customer still gets their text reply."""
    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [])
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone, pet_id=None: [])
    monkeypatch.setattr(orchestrator.memory, "append_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.memory, "extract_and_update_memory", AsyncMock(return_value=None))

    responses = [_fake_response("Pulsy — Your Pet's Health Copilot, 24/7\nHey there!", None)]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)

    settings = Settings(openai_agent_max_iterations=10)
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_reply_and_chunk=AsyncMock(), send_sticker=AsyncMock(side_effect=RuntimeError("boom"))),
        supabase=_make_supabase_mock(), openai=MagicMock(),
    )
    agent_ctx = _make_agent_ctx(is_subscriber=False)
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.greet4",
        timestamp="1700000018", message_type="text", text="hey",
    )

    result = await orchestrator.run_agent_turn(ctx, agent_ctx, extracted)

    assert result == "Pulsy — Your Pet's Health Copilot, 24/7\nHey there!"
    ctx.whatsapp.send_reply_and_chunk.assert_awaited_once()
