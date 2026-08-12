"""Proves the media pipeline degrades gracefully instead of killing the
whole turn — reproduces the real bug found in production where a video
ffmpeg couldn't process (or any media analysis failure) raised uncaught,
and since main.py's webhook handler swallows exceptions at the top level,
the customer got zero reply. Each modality must catch its own failures and
fall back to a message the agent can still respond around."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ingestion.media import FALLBACK_MEDIA_CONTEXT, process_media
from app.ingestion.webhook import ExtractedMessage


def _make_ctx(download_side_effect=None, download_return=None):
    whatsapp = SimpleNamespace(download_media_bytes=AsyncMock())
    if download_side_effect is not None:
        whatsapp.download_media_bytes.side_effect = download_side_effect
    else:
        whatsapp.download_media_bytes.return_value = download_return or (b"fake-bytes", "video/mp4")
    return SimpleNamespace(whatsapp=whatsapp, openai=object(), settings=object())


def _video_message() -> ExtractedMessage:
    return ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.v1",
        timestamp="1700000000", message_type="video", video_media_id="media-1",
    )


@pytest.mark.asyncio
async def test_video_analysis_failure_falls_back_instead_of_raising(monkeypatch):
    ctx = _make_ctx(download_return=(b"corrupted-not-a-real-video", "video/mp4"))

    async def boom(*args, **kwargs):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr("app.ingestion.media.video_pipeline.analyze_video", boom)

    result = await process_media(ctx, _video_message(), pets=[])

    assert result.media_context == FALLBACK_MEDIA_CONTEXT["video"]
    assert result.document_classification is None


@pytest.mark.asyncio
async def test_video_download_failure_falls_back_instead_of_raising():
    ctx = _make_ctx(download_side_effect=ConnectionError("WhatsApp media fetch timed out"))

    result = await process_media(ctx, _video_message(), pets=[])

    assert result.media_context == FALLBACK_MEDIA_CONTEXT["video"]


@pytest.mark.asyncio
async def test_video_success_path_still_returns_real_analysis(monkeypatch):
    ctx = _make_ctx(download_return=(b"real-video-bytes", "video/mp4"))

    async def fake_analyze(openai_client, settings, data, caption, pet_context=""):
        return "[Video frame analysis] Dog appears to be limping on its left hind leg."

    async def fake_classify(*args, **kwargs):
        return None

    monkeypatch.setattr("app.ingestion.media.video_pipeline.analyze_video", fake_analyze)
    monkeypatch.setattr("app.ingestion.media.classify_document", fake_classify)

    result = await process_media(ctx, _video_message(), pets=[])

    assert "limping" in result.media_context
    assert result.document_bytes == b"real-video-bytes"


@pytest.mark.asyncio
async def test_active_pet_name_is_threaded_into_classify_document(monkeypatch):
    """Confirmed live bug: classify_document used to re-guess pet identity
    from the media alone, ignoring whichever pet the rest of the
    conversation already had as "active" -- this is what fixes the
    "misidentified him in the video" class of bug."""
    ctx = _make_ctx(download_return=(b"real-video-bytes", "video/mp4"))
    captured = {}

    async def fake_analyze(openai_client, settings, data, caption, pet_context=""):
        return "[Video frame analysis] A dog running in a yard."

    async def fake_classify(openai_client, settings, kind, mime, analysis, caption, pets, active_pet_name=None):
        captured["active_pet_name"] = active_pet_name
        return None

    monkeypatch.setattr("app.ingestion.media.video_pipeline.analyze_video", fake_analyze)
    monkeypatch.setattr("app.ingestion.media.classify_document", fake_classify)

    await process_media(ctx, _video_message(), pets=[{"id": "pet-1", "name": "Bobby"}], active_pet={"id": "pet-1", "name": "Bobby"})

    assert captured["active_pet_name"] == "Bobby"
