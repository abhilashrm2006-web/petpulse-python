"""Ports the video branch (spec §2/§5): frame + audio-track extraction
(locally, via ffmpeg — see app.integrations.media_processing) feeding two
parallel analyses, both merged into `video_transcript` for the agent's
context."""

import base64

from openai import AsyncOpenAI

from app.config import Settings
from app.integrations.media_processing import extract_video_audio, extract_video_frame
from app.integrations.openai_client import transcribe_speech, vision_completion

FRAME_SYSTEM_PROMPT = (
    "You are looking at a single still frame extracted from a video sent by a pet owner or vet. "
    "Describe what's visible relevant to the pet's health in 2-4 sentences. This is only one frame "
    "(the very start of the clip) — do not claim to describe motion or the full video, only what's "
    "visible in this still image. You may be given background on the pet's known chronic "
    "conditions/allergies — use it only to sharpen your description, never to claim something isn't "
    "actually visible."
)


async def analyze_video(
    openai_client: AsyncOpenAI,
    settings: Settings,
    video_bytes: bytes,
    caption: str,
    pet_context: str = "",
) -> str:
    frame_bytes = await extract_video_frame(video_bytes)
    frame_b64 = base64.b64encode(frame_bytes).decode()
    frame_analysis = await vision_completion(
        openai_client, settings, FRAME_SYSTEM_PROMPT, f"{pet_context}Caption: {caption or '(none)'}", frame_b64, "image/jpeg"
    )

    audio_bytes = await extract_video_audio(video_bytes)
    if audio_bytes is None:
        return f"[Video frame analysis] {frame_analysis}\n\n[Video audio analysis] (no audio track)"

    # Dedicated transcription endpoint, not chat.completions with an attached
    # audio block -- see app/media_pipeline/audio.py's module docstring for
    # why (confirmed live: the chat.completions approach silently ignored
    # the audio ~80-90% of the time on real, otherwise-valid clips).
    audio_analysis = await transcribe_speech(openai_client, settings, audio_bytes, filename="video_audio.mp3", prompt=pet_context)

    return f"[Video frame analysis] {frame_analysis}\n\n[Video audio analysis] {audio_analysis}"
