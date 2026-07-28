"""transcribe_speech (voice notes + video audio tracks) uses the dedicated
/v1/audio/transcriptions endpoint, NOT chat.completions with an attached
audio block. Real bug found live: the chat.completions approach
(gpt-audio + input_audio content) silently ignored the attached audio and
asked for it again on ~80-90% of real, otherwise-valid clean audio clips,
non-deterministically -- confirmed by repeated live testing against actual
generated speech, including a realistic low-bitrate WhatsApp-style OGG.
The dedicated transcription endpoint was 100% reliable across the same
clips. This is the direct cause of "bot doesn't understand voice notes"
reports."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.integrations.openai_client import transcribe_speech
from app.media_pipeline.audio import analyze_voice_note


def _fake_client(text: str = "transcribed text"):
    create = AsyncMock(return_value=SimpleNamespace(text=text))
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    return client, create


@pytest.mark.asyncio
async def test_transcribe_speech_uses_the_dedicated_transcription_endpoint():
    client, create = _fake_client("My dog Bruno has been vomiting since this morning.")
    settings = Settings(openai_transcription_model="gpt-4o-transcribe")

    result = await transcribe_speech(client, settings, b"fake-audio-bytes", filename="voice_note.ogg", prompt="Pet: Bruno")

    assert result == "My dog Bruno has been vomiting since this morning."
    create.assert_awaited_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-transcribe"
    assert kwargs["file"] == ("voice_note.ogg", b"fake-audio-bytes")
    assert kwargs["prompt"] == "Pet: Bruno"


@pytest.mark.asyncio
async def test_analyze_voice_note_passes_raw_bytes_through_with_no_conversion():
    """No ffmpeg OGG->MP3 conversion step anymore -- confirmed live the
    transcription endpoint handles WhatsApp's native OGG/Opus directly, and
    the conversion step was never the actual problem (removing it also
    removes one more point of failure)."""
    client, create = _fake_client("Bruno seems fine now")
    settings = Settings()

    result = await analyze_voice_note(client, settings, b"raw-ogg-bytes", pet_context="Pet: Bruno, Dog")

    assert result == "Bruno seems fine now"
    kwargs = create.call_args.kwargs
    assert kwargs["file"][1] == b"raw-ogg-bytes"  # unmodified, no conversion
    assert kwargs["prompt"] == "Pet: Bruno, Dog"
