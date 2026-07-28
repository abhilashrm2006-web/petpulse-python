"""OpenAI TTS (regional-language voice replies) — must degrade to None on
any failure (disabled, blank text, API error) since a voice reply is
always a best-effort addition on top of a text reply already sent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.integrations.openai_client import synthesize_speech


def _fake_client(audio_bytes: bytes | None = None, exc: Exception | None = None):
    client = SimpleNamespace()
    if exc is not None:
        create = AsyncMock(side_effect=exc)
    else:
        create = AsyncMock(return_value=SimpleNamespace(content=audio_bytes))
    client.audio = SimpleNamespace(speech=SimpleNamespace(create=create))
    return client, create


@pytest.mark.asyncio
async def test_synthesize_speech_returns_bytes_on_success():
    client, create = _fake_client(audio_bytes=b"fake-mp3-bytes")
    settings = Settings(openai_tts_model="tts-1", openai_tts_voice="alloy")

    result = await synthesize_speech(client, settings, "*Namaste!* Rex is doing well.")

    assert result == b"fake-mp3-bytes"
    create.assert_awaited_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "tts-1"
    assert kwargs["voice"] == "alloy"
    assert kwargs["response_format"] == "mp3"
    # markdown stripped before being sent for synthesis
    assert "*" not in kwargs["input"]


@pytest.mark.asyncio
async def test_synthesize_speech_returns_none_when_disabled():
    client, create = _fake_client(audio_bytes=b"irrelevant")
    settings = Settings(voice_replies_enabled=False)

    result = await synthesize_speech(client, settings, "Hello")

    assert result is None
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_synthesize_speech_returns_none_for_blank_text():
    client, create = _fake_client(audio_bytes=b"irrelevant")
    settings = Settings()

    result = await synthesize_speech(client, settings, "   ")

    assert result is None
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_synthesize_speech_returns_none_on_api_error():
    client, create = _fake_client(exc=RuntimeError("rate limited"))
    settings = Settings()

    result = await synthesize_speech(client, settings, "Hello there")

    assert result is None


@pytest.mark.asyncio
async def test_synthesize_speech_truncates_long_text():
    client, create = _fake_client(audio_bytes=b"bytes")
    settings = Settings()
    long_text = "a" * 5000

    await synthesize_speech(client, settings, long_text)

    sent_input = create.call_args.kwargs["input"]
    assert len(sent_input) <= 4000
