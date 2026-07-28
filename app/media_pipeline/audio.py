"""Ports `Transcribe Audio` (spec §2) -- transcribes a WhatsApp voice note
via the dedicated /v1/audio/transcriptions endpoint (gpt-4o-transcribe), not
a chat.completions call with an attached audio block. The earlier approach
(gpt-audio, "hear speech AND describe non-speech sounds like coughing")
sounded better on paper but was confirmed live to silently ignore the
attached audio and ask for it again on the large majority of real,
otherwise-valid clean audio clips -- the direct cause of "bot doesn't
understand voice notes" reports. Transcription-only trades away the
non-speech-sound-description ambition, but reliably hearing the words said
is worth far more than an unreliable bonus feature."""

from openai import AsyncOpenAI

from app.config import Settings
from app.integrations.openai_client import transcribe_speech


async def analyze_voice_note(
    client: AsyncOpenAI, settings: Settings, audio_bytes: bytes, pet_context: str = ""
) -> str:
    return await transcribe_speech(client, settings, audio_bytes, filename="voice_note.ogg", prompt=pet_context)
