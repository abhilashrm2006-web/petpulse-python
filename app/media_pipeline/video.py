"""Ports the video branch (spec §2/§5): frame + audio-track extraction
(locally, via ffmpeg — see app.integrations.media_processing) feeding two
parallel analyses, both merged into `video_transcript` for the agent's
context."""

import base64

from openai import AsyncOpenAI

from app.config import Settings
from app.integrations.media_processing import extract_video_audio, extract_video_frames
from app.integrations.openai_client import multi_image_completion, transcribe_speech

FRAME_SYSTEM_PROMPT = (
    "You are looking at several still frames sampled evenly across a video sent by a pet owner or vet, "
    "in chronological order (first frame = start of the clip, last = end). Describe what's visible relevant "
    "to the pet's health in 2-5 sentences. Unlike a single photo, you CAN describe motion/change across the "
    "frames if it's actually visible (e.g. a limp that appears in some frames but not others, labored "
    "breathing, a head tilt that changes) — but only claim motion/change you can actually see by comparing "
    "the frames, never invent it. If the frames look essentially identical, say so and describe the static "
    "pose/scene instead. You may be given background on the pet's known chronic conditions/allergies — use "
    "it only to sharpen your description, never to claim something isn't actually visible."
)


async def analyze_video(
    openai_client: AsyncOpenAI,
    settings: Settings,
    video_bytes: bytes,
    caption: str,
    pet_context: str = "",
) -> str:
    frames = await extract_video_frames(video_bytes)
    images = [(base64.b64encode(frame).decode(), "image/jpeg") for frame in frames]
    frame_analysis = await multi_image_completion(
        openai_client, settings, FRAME_SYSTEM_PROMPT, f"{pet_context}Caption: {caption or '(none)'}", images
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
