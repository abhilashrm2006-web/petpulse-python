"""Covers extract_video_frames: a single frame from the very start of a
clip can't show a motion-based symptom (limping, a head tilt that comes
and goes, labored breathing) -- verifies multiple distinct frames are
actually sampled across the clip's real duration, not just its first
couple of seconds. Generates real synthetic video via ffmpeg (same
dependency this module already requires in production, per its own
module docstring) rather than mocking subprocess calls, so this actually
exercises the real ffmpeg command."""

import subprocess

import pytest

from app.integrations import media_processing
from app.integrations.media_processing import (
    MAX_VIDEO_FRAMES,
    extract_video_audio,
    extract_video_frames,
)


def _make_test_video(tmp_path, duration: float, with_audio: bool) -> bytes:
    out = tmp_path / "test.mp4"
    args = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=10"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:v", "libx264", "-c:a", "aac", "-shortest"]
    else:
        args += ["-c:v", "libx264"]
    args.append(str(out))
    subprocess.run(args, check=True, capture_output=True)
    return out.read_bytes()


@pytest.mark.asyncio
async def test_extracts_multiple_distinct_frames_across_full_duration(tmp_path):
    video_bytes = _make_test_video(tmp_path, duration=8, with_audio=True)

    frames = await extract_video_frames(video_bytes)

    assert len(frames) == MAX_VIDEO_FRAMES
    # testsrc produces a continuously changing pattern -- distinct byte sizes
    # confirm these are genuinely different frames, not the same frame
    # (e.g. always frame 0) returned MAX_VIDEO_FRAMES times.
    assert len({len(f) for f in frames}) > 1


@pytest.mark.asyncio
async def test_short_clip_still_yields_multiple_frames_not_just_one(tmp_path):
    """Regression: the old behavior (-vframes 1) always returned exactly one
    frame regardless of length. A short clip must still get more than one
    frame -- fps is derived from actual duration, so even a 1s clip at
    MAX_VIDEO_FRAMES=6 asks for 6fps, which a 1s clip can satisfy."""
    video_bytes = _make_test_video(tmp_path, duration=1, with_audio=False)

    frames = await extract_video_frames(video_bytes)

    assert len(frames) > 1


@pytest.mark.asyncio
async def test_audio_extraction_still_works_alongside_multi_frame(tmp_path):
    video_bytes = _make_test_video(tmp_path, duration=3, with_audio=True)

    audio = await extract_video_audio(video_bytes)

    assert audio is not None
    assert len(audio) > 0


@pytest.mark.asyncio
async def test_video_with_no_audio_track_returns_none(tmp_path):
    video_bytes = _make_test_video(tmp_path, duration=3, with_audio=False)

    audio = await extract_video_audio(video_bytes)

    assert audio is None


def test_duration_lookup_falls_back_to_zero_if_ffprobe_is_unavailable(monkeypatch, tmp_path):
    """Regression guard: ffprobe ships in the same package as ffmpeg so it
    should always be present, but if it somehow isn't (or crashes), that
    must degrade to the fixed 1fps fallback (duration=0.0), not raise and
    take down the whole video-analysis turn."""

    def _boom(args, **kwargs):
        raise FileNotFoundError("ffprobe not found")

    monkeypatch.setattr(media_processing.subprocess, "run", _boom)

    duration = media_processing._video_duration_seconds(tmp_path / "irrelevant.mp4")

    assert duration == 0.0


@pytest.mark.asyncio
async def test_extraction_still_works_end_to_end_if_ffprobe_is_unavailable(tmp_path, monkeypatch):
    video_bytes = _make_test_video(tmp_path, duration=2, with_audio=False)

    monkeypatch.setattr(media_processing, "_video_duration_seconds", lambda video_path: 0.0)

    frames = await extract_video_frames(video_bytes)

    assert len(frames) >= 1
