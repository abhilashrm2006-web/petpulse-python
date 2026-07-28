"""Detects which regional Indian language a customer's voice note was
spoken in, so the agent can be told to reply in that language and the reply
can then be spoken back in the same language (see orchestrator.py). Kept as
a separate, cheap classification call rather than trusting the main agent
loop to self-report its input language — a dedicated JSON call is far more
reliable than parsing free-form model output for this."""

import json
import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.integrations.openai_client import json_completion

logger = logging.getLogger(__name__)

# The 10 major Indian regional languages this feature supports, beyond the
# Hindi-only text personalization that existed before it (see
# app/agent/system_prompt.py SUBSCRIBER_PERSONALIZATION_RULE).
SUPPORTED_LANGUAGES: dict[str, str] = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
}

_LANGUAGE_LIST = ", ".join(f'"{code}" ({name})' for code, name in SUPPORTED_LANGUAGES.items())

_SYSTEM_PROMPT = (
    "Identify which language the following transcript of a customer's spoken voice note is in. "
    f"Respond with strict JSON: {{\"language\": \"<code>\"}}. The code must be one of: "
    f'{_LANGUAGE_LIST}, or "en" for English or any language not in that list, or if you cannot '
    "tell. Judge by the actual words spoken, not the script used to write them down (a transcript "
    "written in Latin/Roman letters can still be Hindi, Tamil, etc. spoken phonetically)."
)


async def detect_regional_language(client: AsyncOpenAI, settings: Settings, transcript: str) -> str:
    """Returns a code from SUPPORTED_LANGUAGES, or "en". Never raises --
    callers treat "en" and a failed call identically (no voice-language
    override), so defaulting to "en" on any error is always safe."""
    if not transcript.strip():
        return "en"
    try:
        raw = await json_completion(client, settings, _SYSTEM_PROMPT, transcript, reasoning_effort="low")
        code = json.loads(raw).get("language", "en")
        return code if code in SUPPORTED_LANGUAGES else "en"
    except Exception:
        logger.exception("Language detection failed, defaulting to English")
        return "en"
