"""Covers the customer-initiated chat-history reset flow: a customer asking
to "delete my profile"/"clear my data" gets a Yes/No confirmation first,
"Yes" only clears chat history/memory (never real account/pet/medical
records), and "No" asks for a reason which gets logged as feedback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.account import (
    record_deletion_feedback,
    request_data_deletion,
    respond_to_deletion_confirmation,
)
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase=None):
    whatsapp = SimpleNamespace(send_text=AsyncMock(), send_interactive_buttons=AsyncMock())
    return SimpleNamespace(supabase=supabase or FakeSupabaseClient(), whatsapp=whatsapp, settings=object())


def _make_agent_ctx(profile_id="profile-1"):
    return SimpleNamespace(profile={"id": profile_id, "phone_number": "919876543210", "full_name": "Jane"})


@pytest.mark.asyncio
async def test_request_data_deletion_sends_yes_no_confirmation():
    ctx = _make_ctx()
    agent_ctx = _make_agent_ctx()

    result = await request_data_deletion(ctx, agent_ctx)

    assert result["success"] is True
    assert result["mode"] == "deletion_confirmation_sent"
    ctx.whatsapp.send_interactive_buttons.assert_awaited_once()
    call = ctx.whatsapp.send_interactive_buttons.call_args
    assert call.args[0] == "919876543210"
    button_ids = {b["id"] for b in call.args[2]}
    assert button_ids == {"delete_chat|yes", "delete_chat|no"}


@pytest.mark.asyncio
async def test_confirming_yes_clears_chat_history_and_memory_only():
    """Real account/pet/medical data must never be touched -- only the
    session-scoped chat history and this profile's memory facts."""
    supabase = FakeSupabaseClient(
        initial={
            "n8n_chat_history_petpulse": [
                {"id": "h1", "session_id": "919876543210", "message": {"type": "human", "data": {"content": "hi"}}},
                {"id": "h2", "session_id": "919000000002", "message": {"type": "human", "data": {"content": "other customer"}}},
            ],
            "memory": [
                {"id": "m1", "profile_id": "profile-1", "memory_type": "Fact", "title": "x"},
                {"id": "m2", "profile_id": "profile-2", "memory_type": "Fact", "title": "other customer"},
            ],
            "pets": [{"id": "pet-1", "profile_id": "profile-1", "name": "Rex"}],
            "medical_records": [{"id": "mr1", "profile_id": "profile-1"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx()

    result = await respond_to_deletion_confirmation(ctx, agent_ctx, confirm=True)

    assert result["success"] is True
    assert result["mode"] == "deletion_done"
    remaining_history = [r["session_id"] for r in supabase.rows("n8n_chat_history_petpulse")]
    assert remaining_history == ["919000000002"]
    remaining_memory = [r["profile_id"] for r in supabase.rows("memory")]
    assert remaining_memory == ["profile-2"]
    # untouched
    assert len(supabase.rows("pets")) == 1
    assert len(supabase.rows("medical_records")) == 1
    ctx.whatsapp.send_text.assert_awaited_once()
    assert "chat history" in ctx.whatsapp.send_text.call_args.args[1].lower()


@pytest.mark.asyncio
async def test_declining_asks_for_a_reason_and_does_not_delete_anything():
    supabase = FakeSupabaseClient(
        initial={
            "n8n_chat_history_petpulse": [{"id": "h1", "session_id": "919876543210", "message": {}}],
            "memory": [{"id": "m1", "profile_id": "profile-1", "memory_type": "Fact"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx()

    result = await respond_to_deletion_confirmation(ctx, agent_ctx, confirm=False)

    assert result["success"] is True
    assert result["mode"] == "deletion_declined"
    assert len(supabase.rows("n8n_chat_history_petpulse")) == 1
    assert len(supabase.rows("memory")) == 1
    ctx.whatsapp.send_text.assert_awaited_once()
    assert "why" in ctx.whatsapp.send_text.call_args.args[1].lower() or "sharing" in ctx.whatsapp.send_text.call_args.args[1].lower()


@pytest.mark.asyncio
async def test_record_deletion_feedback_logs_the_reason():
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx()

    result = await record_deletion_feedback(ctx, agent_ctx, reason="too many messages")

    assert result["success"] is True
    row = supabase.rows("memory")[0]
    assert row["profile_id"] == "profile-1"
    assert row["memory_type"] == "Feedback"
    assert row["memory_text"] == "too many messages"


@pytest.mark.asyncio
async def test_record_deletion_feedback_rejects_empty_reason():
    ctx = _make_ctx()
    agent_ctx = _make_agent_ctx()

    result = await record_deletion_feedback(ctx, agent_ctx, reason="   ")

    assert result["success"] is False
