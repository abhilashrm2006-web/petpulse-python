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
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone: [])
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
    agent_ctx = _make_agent_ctx(is_subscriber=True)  # check_symptoms is subscriber_only
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
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone: [])
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
@pytest.mark.parametrize("mode", ["booked", "rescheduled", "prescription_delivered"])
async def test_booking_confirmation_suppresses_duplicate_agent_reply(monkeypatch, mode):
    """Reproduces a real reported bug: _finalize_booking already sends the
    full confirmation (date/time + Meet link) directly to both parties, but
    its mode wasn't in SELF_MESSAGING_MODES — so the agent composed a
    SECOND, redundant confirmation on top, and the customer saw the same
    date/time and Meet link twice across multiple WhatsApp bubbles.
    "prescription_delivered" is the same class of bug: deliver_prescription
    already sends the text/PDF directly."""

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
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone: [])
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
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone: [])
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
async def test_free_customer_blocked_from_check_symptoms_with_upsell(monkeypatch):
    """The structured triage tool (check_symptoms) is now Subscriber-only --
    a Free customer reporting a new symptom must never reach the real
    triage call; the orchestrator should short-circuit with a
    subscriber_only_feature error instead."""
    tool_calls_made = []

    async def fake_tool(ctx, agent_ctx, **kwargs):
        tool_calls_made.append(kwargs)
        return {"success": True, "severity": 4}

    monkeypatch.setattr(orchestrator, "get_tool_schemas", lambda role: [{"type": "function", "function": {"name": "check_symptoms"}}])
    monkeypatch.setattr(orchestrator, "get_tool_fn", lambda name: fake_tool)
    monkeypatch.setattr(orchestrator, "is_tool_allowed_for_role", lambda name, role: True)

    responses = [
        _fake_response(None, [_fake_tool_call("call-1", "check_symptoms", {"symptoms": "vomiting"})]),
        _fake_response("That's a Subscriber feature — want the subscribe link?", None),
    ]

    async def fake_chat_with_tools(client, settings, messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(orchestrator, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(orchestrator.memory, "load_chat_history", lambda client, phone: [])
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

    assert tool_calls_made == []
    assert result == "That's a Subscriber feature — want the subscribe link?"
    ctx.whatsapp.send_reply_and_chunk.assert_awaited_once_with("919876543210", result)
