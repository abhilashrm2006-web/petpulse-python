"""Ports `Detect Message Type` -> `Media Router` (spec §2): routes an
inbound message to the right media_pipeline analyzer and produces both a
plain-text `media_context` for the agent prompt and, for image/document
messages, a `DocumentClassification` the agent can act on via the
`file_document` tool.

Every branch is wrapped so a failure anywhere in it (WhatsApp media
download, ffmpeg frame/audio extraction, PyMuPDF PDF rendering, an OpenAI
vision/audio/whisper call) degrades to a fallback media_context instead of
raising — matching the n8n original, where every media node had
`continueOnFail` set for exactly this reason (spec §5). Without this, one
bad video/image/document silently kills the ENTIRE turn (main.py's webhook
handler catches and swallows at the top level), leaving the customer with
no reply at all instead of a real answer telling them what happened."""

import base64
import logging
from dataclasses import dataclass

from app.deps import AppContext
from app.ingestion.webhook import ExtractedMessage
from app.media_pipeline import audio as audio_pipeline
from app.media_pipeline import document as document_pipeline
from app.media_pipeline import image as image_pipeline
from app.media_pipeline import video as video_pipeline
from app.media_pipeline.classify import DocumentClassification, classify_document
from app.media_pipeline.pet_context import build_pet_background_note

logger = logging.getLogger(__name__)

FALLBACK_MEDIA_CONTEXT = {
    "image": "[Image could not be analyzed due to a technical problem. Ask the customer to describe what "
    "they wanted to show, or ask them to resend the photo.]",
    "audio": "[Voice note could not be transcribed due to a technical problem. Ask the customer to type out "
    "what they said.]",
    "document": "[Document could not be analyzed due to a technical problem. Ask the customer to describe "
    "what the document is, or ask them to resend it.]",
    "video": "[Video could not be analyzed due to a technical problem. Ask the customer to describe what's "
    "happening, or ask them to resend as a photo or in words.]",
}


@dataclass
class MediaResult:
    media_context: str = ""
    document_classification: DocumentClassification | None = None
    document_bytes: bytes | None = None
    document_mime_type: str | None = None


async def process_media(
    ctx: AppContext, extracted: ExtractedMessage, pets: list[dict], active_pet: dict | None = None
) -> MediaResult:
    pet_context = build_pet_background_note(active_pet)
    active_pet_name = active_pet.get("name") if active_pet else None

    if extracted.image_media_id:
        try:
            data, mime = await ctx.whatsapp.download_media_bytes(extracted.image_media_id)
            analysis = await image_pipeline.analyze_image(
                ctx.openai, ctx.settings, base64.b64encode(data).decode(), mime, extracted.text, pet_context
            )
            classification = await classify_document(
                ctx.openai, ctx.settings, "image", mime, analysis, extracted.text, pets, active_pet_name
            )
            return MediaResult(
                media_context=f"[Image analysis] {analysis}",
                document_classification=classification,
                document_bytes=data,
                document_mime_type=mime,
            )
        except Exception:
            logger.exception("Image processing failed for media %s", extracted.image_media_id)
            return MediaResult(media_context=FALLBACK_MEDIA_CONTEXT["image"])

    if extracted.audio_media_id:
        try:
            data, mime = await ctx.whatsapp.download_media_bytes(extracted.audio_media_id)
            transcript = await audio_pipeline.analyze_voice_note(ctx.openai, ctx.settings, data, pet_context)
            return MediaResult(media_context=f"[Voice note analysis] {transcript}")
        except Exception:
            logger.exception("Audio processing failed for media %s", extracted.audio_media_id)
            return MediaResult(media_context=FALLBACK_MEDIA_CONTEXT["audio"])

    if extracted.document_media_id:
        try:
            data, mime = await ctx.whatsapp.download_media_bytes(extracted.document_media_id)
            mime = extracted.document_mime_type or mime
            analysis = await document_pipeline.analyze_document(ctx.openai, ctx.settings, data, mime, extracted.text)
            classification = await classify_document(
                ctx.openai, ctx.settings, "document", mime, analysis, extracted.text, pets, active_pet_name
            )
            return MediaResult(
                media_context=f"[Document analysis] {analysis}",
                document_classification=classification,
                document_bytes=data,
                document_mime_type=mime,
            )
        except Exception:
            logger.exception("Document processing failed for media %s", extracted.document_media_id)
            return MediaResult(media_context=FALLBACK_MEDIA_CONTEXT["document"])

    if extracted.video_media_id:
        try:
            data, mime = await ctx.whatsapp.download_media_bytes(extracted.video_media_id)
            analysis = await video_pipeline.analyze_video(ctx.openai, ctx.settings, data, extracted.text, pet_context)
            classification = await classify_document(
                ctx.openai, ctx.settings, "video", mime, analysis, extracted.text, pets, active_pet_name
            )
            return MediaResult(
                media_context=analysis,
                document_classification=classification,
                document_bytes=data,
                document_mime_type=mime,
            )
        except Exception:
            logger.exception("Video processing failed for media %s", extracted.video_media_id)
            return MediaResult(media_context=FALLBACK_MEDIA_CONTEXT["video"])

    return MediaResult()
