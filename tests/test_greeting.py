"""Pulsy mascot welcome image — sent deterministically in code for a bare
greeting (see app/agent/orchestrator.py), not via an LLM tool call. A
first version routed this through the tool-calling loop and the model
just didn't reliably call it turn to turn (confirmed live)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.greeting import is_bare_greeting, send_welcome_character
from app.config import Settings
from app.deps import AppContext


@pytest.mark.parametrize(
    "text",
    ["hi", "Hi", "HII", "hello", "Hello!", "hey", "heyy", "yo", "hola", "namaste", "sup", "good morning", "Good Evening!"],
)
def test_recognizes_bare_greetings(text):
    assert is_bare_greeting(text) is True


@pytest.mark.parametrize(
    "text",
    ["hi, my dog is limping", "hello can you help me", "hey how much does a consult cost", "", "   ", "history of present illness"],
)
def test_does_not_flag_non_bare_greetings(text):
    assert is_bare_greeting(text) is False


@pytest.mark.asyncio
async def test_send_welcome_character_sends_the_mascot_image():
    settings = Settings(pulsy_welcome_image_url="https://example.com/pulsy.png")
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_image=AsyncMock()),
        supabase=None, openai=None,
    )
    agent_ctx = SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210"})

    await send_welcome_character(ctx, agent_ctx)

    ctx.whatsapp.send_image.assert_awaited_once_with("919876543210", "https://example.com/pulsy.png")
