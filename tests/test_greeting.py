"""Pulsy mascot welcome image -- sent as its own WhatsApp image message
with no caption, distinct from the agent's own text reply (see
GREETING_RULE in system_prompt.py)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.greeting import send_welcome_character
from app.config import Settings
from app.deps import AppContext


@pytest.mark.asyncio
async def test_send_welcome_character_sends_the_mascot_image():
    settings = Settings(pulsy_welcome_image_url="https://example.com/pulsy.png")
    ctx = AppContext(
        settings=settings, http=None,
        whatsapp=SimpleNamespace(send_image=AsyncMock()),
        supabase=None, openai=None,
    )
    agent_ctx = SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210"})

    result = await send_welcome_character(ctx, agent_ctx)

    assert result == {"success": True, "mode": "welcome_character_sent"}
    ctx.whatsapp.send_image.assert_awaited_once_with("919876543210", "https://example.com/pulsy.png")
