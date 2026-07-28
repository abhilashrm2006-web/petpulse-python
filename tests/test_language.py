"""Regional-language detection for voice-note replies — must default to
"en" (no voice-language override) on any failure, since callers treat that
identically to "no regional language detected" (see orchestrator.py)."""

import pytest

from app.agent.language import detect_regional_language
from app.config import Settings


class _FakeOpenAI:
    pass  # detect_regional_language only ever passes this through to json_completion


@pytest.mark.asyncio
async def test_detects_a_supported_language(monkeypatch):
    async def fake_json_completion(client, settings, system_prompt, user_prompt, reasoning_effort="low"):
        return '{"language": "ta"}'

    monkeypatch.setattr("app.agent.language.json_completion", fake_json_completion)

    result = await detect_regional_language(_FakeOpenAI(), Settings(), "வணக்கம், என் நாய் சரியில்லை")

    assert result == "ta"


@pytest.mark.asyncio
async def test_unsupported_code_defaults_to_english(monkeypatch):
    async def fake_json_completion(client, settings, system_prompt, user_prompt, reasoning_effort="low"):
        return '{"language": "fr"}'  # not in SUPPORTED_LANGUAGES

    monkeypatch.setattr("app.agent.language.json_completion", fake_json_completion)

    result = await detect_regional_language(_FakeOpenAI(), Settings(), "bonjour")

    assert result == "en"


@pytest.mark.asyncio
async def test_malformed_json_defaults_to_english(monkeypatch):
    async def fake_json_completion(client, settings, system_prompt, user_prompt, reasoning_effort="low"):
        return "not json"

    monkeypatch.setattr("app.agent.language.json_completion", fake_json_completion)

    result = await detect_regional_language(_FakeOpenAI(), Settings(), "some transcript")

    assert result == "en"


@pytest.mark.asyncio
async def test_api_failure_defaults_to_english(monkeypatch):
    async def fake_json_completion(client, settings, system_prompt, user_prompt, reasoning_effort="low"):
        raise RuntimeError("openai down")

    monkeypatch.setattr("app.agent.language.json_completion", fake_json_completion)

    result = await detect_regional_language(_FakeOpenAI(), Settings(), "some transcript")

    assert result == "en"


@pytest.mark.asyncio
async def test_blank_transcript_short_circuits_without_calling_the_api(monkeypatch):
    calls = []

    async def fake_json_completion(client, settings, system_prompt, user_prompt, reasoning_effort="low"):
        calls.append(user_prompt)
        return '{"language": "hi"}'

    monkeypatch.setattr("app.agent.language.json_completion", fake_json_completion)

    result = await detect_regional_language(_FakeOpenAI(), Settings(), "   ")

    assert result == "en"
    assert calls == []
