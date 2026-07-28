"""Google Cloud Text-to-Speech REST client — regional-language voice replies
for Subscribers who send a WhatsApp voice note (see app/agent/orchestrator.py).
Plain REST + API key, matching this codebase's convention (razorpay_client.py,
whatsapp.py) of raw httpx calls over a full SDK, and keeping setup to "paste
one API key" rather than a service-account JSON file.

Google Cloud TTS has a genuinely free monthly quota (Standard voices: 4
million characters/month) comfortably covering this bot's volume — Standard
voices are used deliberately over Wavenet/Neural2 (1 million free chars/month)
to maximize that headroom, per the user's "free version" request."""

import base64
import logging
import re

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"

# Google's synthesis input cap is 5000 bytes; WhatsApp replies are chunked to
# ~300 chars already (see app/utils/formatting.py) but a voice reply is built
# from the full final_text before chunking, so this guards the rare long reply.
MAX_INPUT_CHARS = 4500

# One human name + one Standard voice per supported language. `ur` (Urdu) is
# the least-verified entry in Google's locale catalog for this key type --
# synthesize_speech() degrades to returning None (never raises) if a given
# language/voice combination isn't actually available on the account.
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

_VOICE_BY_LANGUAGE: dict[str, tuple[str, str]] = {
    # code: (languageCode, voice name)
    "hi": ("hi-IN", "hi-IN-Standard-A"),
    "ta": ("ta-IN", "ta-IN-Standard-A"),
    "te": ("te-IN", "te-IN-Standard-A"),
    "kn": ("kn-IN", "kn-IN-Standard-A"),
    "ml": ("ml-IN", "ml-IN-Standard-A"),
    "bn": ("bn-IN", "bn-IN-Standard-A"),
    "mr": ("mr-IN", "mr-IN-Standard-A"),
    "gu": ("gu-IN", "gu-IN-Standard-A"),
    "pa": ("pa-IN", "pa-IN-Standard-A"),
    "ur": ("ur-IN", "ur-IN-Standard-A"),
}


def strip_for_speech(text: str) -> str:
    """WhatsApp markdown (*bold*) and stray formatting reads badly aloud --
    strip it before synthesis rather than speaking literal asterisks."""
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"[_~`]", "", text)
    return text.strip()


async def synthesize_speech(http: httpx.AsyncClient, settings: Settings, text: str, language_code: str) -> bytes | None:
    """Returns MP3 bytes, or None if the feature isn't configured, the
    language isn't supported, or the call fails for any reason -- this must
    never raise, since a voice reply is always a best-effort addition on top
    of the text reply that's already been sent (see orchestrator.py)."""
    if not settings.google_tts_api_key:
        return None
    voice = _VOICE_BY_LANGUAGE.get(language_code)
    if not voice:
        return None

    spoken_text = strip_for_speech(text)[:MAX_INPUT_CHARS]
    if not spoken_text:
        return None

    google_language_code, voice_name = voice
    try:
        resp = await http.post(
            TTS_ENDPOINT,
            params={"key": settings.google_tts_api_key},
            json={
                "input": {"text": spoken_text},
                "voice": {"languageCode": google_language_code, "name": voice_name},
                "audioConfig": {"audioEncoding": "MP3"},
            },
        )
        resp.raise_for_status()
        audio_content_b64 = resp.json().get("audioContent")
        if not audio_content_b64:
            return None
        return base64.b64decode(audio_content_b64)
    except Exception:
        logger.exception("Google TTS synthesis failed for language=%s", language_code)
        return None
