"""Google Cloud TTS REST client — must degrade to None on any failure
(missing key, unsupported language, network/API error) since a voice reply
is always a best-effort addition on top of a text reply already sent."""

import base64

import httpx
import pytest

from app.config import Settings
from app.integrations.google_tts import strip_for_speech, synthesize_speech


class _FakeHttpClient:
    def __init__(self, response):
        self._response = response
        self.last_request = None

    async def post(self, url, params, json):
        self.last_request = {"url": url, "params": params, "json": json}
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _ok_response(audio_b64: str):
    request = httpx.Request("POST", "https://texttospeech.googleapis.com/v1/text:synthesize")
    return httpx.Response(200, json={"audioContent": audio_b64}, request=request)


def _error_response():
    request = httpx.Request("POST", "https://texttospeech.googleapis.com/v1/text:synthesize")
    return httpx.Response(400, json={"error": {"message": "bad request"}}, request=request)


def test_strip_for_speech_removes_whatsapp_markdown():
    assert strip_for_speech("*Seriousness:* 🟡 Moderate (3/5)") == "Seriousness: 🟡 Moderate (3/5)"
    assert strip_for_speech("some `code` and _emphasis_ and ~strike~") == "some code and emphasis and strike"


@pytest.mark.asyncio
async def test_synthesize_speech_returns_none_when_not_configured():
    fake = _FakeHttpClient(_ok_response("irrelevant"))
    settings = Settings(google_tts_api_key="")

    result = await synthesize_speech(fake, settings, "Hello", "hi")

    assert result is None
    assert fake.last_request is None  # never even attempted the call


@pytest.mark.asyncio
async def test_synthesize_speech_returns_none_for_unsupported_language():
    fake = _FakeHttpClient(_ok_response("irrelevant"))
    settings = Settings(google_tts_api_key="test-key")

    result = await synthesize_speech(fake, settings, "Hello", "fr")

    assert result is None
    assert fake.last_request is None


@pytest.mark.asyncio
async def test_synthesize_speech_decodes_audio_content():
    audio_bytes = b"fake-mp3-bytes"
    audio_b64 = base64.b64encode(audio_bytes).decode()
    fake = _FakeHttpClient(_ok_response(audio_b64))
    settings = Settings(google_tts_api_key="test-key")

    result = await synthesize_speech(fake, settings, "*नमस्ते* आपका पालतू कैसा है?", "hi")

    assert result == audio_bytes
    assert fake.last_request["params"] == {"key": "test-key"}
    assert fake.last_request["json"]["voice"] == {"languageCode": "hi-IN", "name": "hi-IN-Standard-A"}
    # sent to the API with markdown stripped, not the raw asterisk-laden reply text
    assert "*" not in fake.last_request["json"]["input"]["text"]


@pytest.mark.asyncio
async def test_synthesize_speech_returns_none_on_api_error():
    fake = _FakeHttpClient(_error_response())
    settings = Settings(google_tts_api_key="test-key")

    result = await synthesize_speech(fake, settings, "Hello", "ta")

    assert result is None


@pytest.mark.asyncio
async def test_synthesize_speech_returns_none_on_transport_error():
    fake = _FakeHttpClient(httpx.ConnectError("network down"))
    settings = Settings(google_tts_api_key="test-key")

    result = await synthesize_speech(fake, settings, "Hello", "te")

    assert result is None


@pytest.mark.asyncio
async def test_synthesize_speech_returns_none_for_blank_text():
    fake = _FakeHttpClient(_ok_response("irrelevant"))
    settings = Settings(google_tts_api_key="test-key")

    result = await synthesize_speech(fake, settings, "   ", "hi")

    assert result is None
    assert fake.last_request is None
