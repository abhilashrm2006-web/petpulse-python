"""OpenAI client wrapper: the agent tool-calling loop's model, plus the
smaller reasoning/vision/audio calls used by the media pipeline and document
classifier (spec §1, §2, §5). Model names are configurable but default to
what the live n8n system already uses successfully."""

import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.utils.formatting import strip_for_speech

logger = logging.getLogger(__name__)

# OpenAI's /v1/audio/speech input cap is 4096 characters.
MAX_TTS_INPUT_CHARS = 4000


def make_openai_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def chat_with_tools(
    client: AsyncOpenAI,
    settings: Settings,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> Any:
    return await client.chat.completions.create(
        model=settings.openai_agent_model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_completion_tokens=settings.openai_agent_max_tokens,
        # gpt-5.4/gpt-5.4-mini reject function tools + reasoning_effort together on
        # /v1/chat/completions ("use /v1/responses or set reasoning_effort to
        # 'none'") — this loop always passes tools, so it's pinned to "none".
        reasoning_effort="none",
    )


async def json_completion(
    client: AsyncOpenAI,
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = "low",
    model: str | None = None,
) -> str:
    """Used by the document classifier, symptom checker, and medical-record
    extraction — all of which expect strict JSON back (spec §2, §3.3)."""
    resp = await client.chat.completions.create(
        model=model or settings.openai_reasoning_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        reasoning_effort=reasoning_effort,
    )
    return resp.choices[0].message.content or "{}"


async def text_completion(
    client: AsyncOpenAI,
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = "low",
    model: str | None = None,
) -> str:
    """Same shape as json_completion but for plain prose output — used where
    the result is a message meant to be read as-is (e.g. a reformatted
    prescription), not parsed as JSON."""
    resp = await client.chat.completions.create(
        model=model or settings.openai_reasoning_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort=reasoning_effort,
    )
    return resp.choices[0].message.content or ""


async def vision_completion(
    client: AsyncOpenAI,
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
    mime_type: str = "image/jpeg",
    max_tokens: int = 1600,
) -> str:
    resp = await client.chat.completions.create(
        model=settings.openai_reasoning_model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                ],
            },
        ],
    )
    return resp.choices[0].message.content or ""


TTS_ACCENT_INSTRUCTIONS_TEMPLATE = (
    "Speak as a warm, friendly native {language} speaker from India, in natural conversational {language}. "
    "Use authentic Indian pronunciation and intonation throughout, the way a caring Indian pet owner or vet "
    "assistant would actually speak to a friend over the phone — not an anglicized or foreign accent."
)


async def synthesize_speech(client: AsyncOpenAI, settings: Settings, text: str, language: str = "Hindi") -> bytes | None:
    """Text-to-speech for regional-language voice replies (see
    app/agent/orchestrator.py) -- reuses the OpenAI account already paid for
    rather than a separate provider. OpenAI's TTS model reads the input in
    whatever language it's written in; the caller is responsible for having
    already generated `text` in the customer's language (see
    app/agent/system_prompt.py VOICE_REPLY_LANGUAGE_RULE_TEMPLATE).

    `language` steers accent/delivery via gpt-4o-mini-tts's `instructions`
    param -- confirmed live (user-verified A/B sample) that without this, the
    older tts-1 voices (and gpt-4o-mini-tts with no instructions) read Indian
    regional-language text with a distinctly foreign/anglicized accent; the
    instructions steer it to sound like an actual native speaker instead.
    `voice`/`model` picked the same way (nova was the user's pick among 5
    A/B samples). Returns None on any failure -- a voice reply is always
    best-effort on top of the text reply that's already been sent, never
    something that can break the turn."""
    if not settings.voice_replies_enabled:
        return None
    spoken_text = strip_for_speech(text)[:MAX_TTS_INPUT_CHARS]
    if not spoken_text:
        return None
    try:
        resp = await client.audio.speech.create(
            model=settings.openai_tts_model,
            voice=settings.openai_tts_voice,
            input=spoken_text,
            instructions=TTS_ACCENT_INSTRUCTIONS_TEMPLATE.format(language=language),
            response_format="mp3",
        )
        return resp.content
    except Exception:
        logger.exception("OpenAI TTS synthesis failed")
        return None


async def transcribe_audio(client: AsyncOpenAI, audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    resp = await client.audio.transcriptions.create(model="whisper-1", file=(filename, audio_bytes))
    return resp.text


async def analyze_audio(
    client: AsyncOpenAI,
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    audio_base64: str,
    audio_format: str = "mp3",
) -> str:
    """gpt-audio: transcribes speech AND describes non-speech sounds
    (wheezing, coughing) for respiratory-symptom video/voice notes (spec
    §5). Deliberately neutral prompt to avoid false-positive alarms."""
    resp = await client.chat.completions.create(
        model=settings.openai_audio_model,
        modalities=["text"],
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_base64, "format": audio_format}},
                ],
            },
        ],
    )
    return resp.choices[0].message.content or ""
