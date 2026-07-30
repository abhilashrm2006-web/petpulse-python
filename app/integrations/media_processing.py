"""Local video-frame / PDF-page extraction — replaces the earlier
Cloudinary-based pipeline (spec §5). Video frame + audio-track extraction
via ffmpeg (subprocess); PDF page-1 rendering via PyMuPDF (pure-Python
wheel, no extra system dependency). No external media service or
credential needed; ffmpeg must be present on the host (see Dockerfile)."""

import asyncio
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF


MAX_VIDEO_FRAMES = 6


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", *args], check=True, capture_output=True)


def _video_duration_seconds(video_path: Path) -> float:
    # Broad except (not just ValueError on a bad parse) -- ffprobe ships in
    # the same apt/homebrew package as ffmpeg so it should always be present,
    # but a crashed/unavailable probe must degrade to the fixed-rate fallback
    # below, not take down the whole video-analysis turn.
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except (OSError, ValueError):
        return 0.0  # duration unknown -- caller falls back to a fixed sample rate


def _extract_video_frames_sync(video_bytes: bytes, max_frames: int = MAX_VIDEO_FRAMES) -> list[bytes]:
    """A single frame from the very start of a clip can't show a
    motion-based symptom (limping, a head tilt that comes and goes, labored
    breathing) -- sample up to max_frames frames spread evenly across the
    WHOLE clip instead. fps is derived from the actual duration (not a fixed
    rate) so a 3s clip and a 30s clip both get frames spanning their full
    length rather than all clustered in the first few seconds; unknown/zero
    duration falls back to a fixed 1fps, which max_frames still caps."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "input.mp4"
        video_path.write_bytes(video_bytes)
        duration = _video_duration_seconds(video_path)
        fps = max_frames / duration if duration > 0 else 1.0
        pattern = Path(tmp) / "frame_%02d.jpg"
        _run_ffmpeg(["-i", str(video_path), "-vf", f"fps={fps}", "-frames:v", str(max_frames), "-f", "image2", str(pattern)])
        frame_paths = sorted(Path(tmp).glob("frame_*.jpg"))
        if not frame_paths:
            # Extremely short/unusual clip where the fps filter yielded
            # nothing -- fall back to one frame rather than returning empty.
            single_path = Path(tmp) / "frame_fallback.jpg"
            _run_ffmpeg(["-i", str(video_path), "-vframes", "1", "-f", "image2", str(single_path)])
            return [single_path.read_bytes()]
        return [p.read_bytes() for p in frame_paths]


def _extract_video_audio_sync(video_bytes: bytes) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "input.mp4"
        audio_path = Path(tmp) / "audio.mp3"
        video_path.write_bytes(video_bytes)
        try:
            _run_ffmpeg(["-i", str(video_path), "-vn", "-acodec", "libmp3lame", str(audio_path)])
        except subprocess.CalledProcessError:
            return None  # video has no audio track
        return audio_path.read_bytes()


def _convert_audio_to_mp3_sync(audio_bytes: bytes) -> bytes:
    """WhatsApp voice notes arrive as OGG/Opus — gpt-audio's input_audio
    format only documents wav/mp3 support, so convert rather than guess."""
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.ogg"
        output_path = Path(tmp) / "output.mp3"
        input_path.write_bytes(audio_bytes)
        _run_ffmpeg(["-i", str(input_path), "-acodec", "libmp3lame", str(output_path)])
        return output_path.read_bytes()


def _render_pdf_first_page_sync(pdf_bytes: bytes) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[0]
        pixmap = page.get_pixmap(dpi=200)
        return pixmap.tobytes("jpg")
    finally:
        doc.close()


async def extract_video_frames(video_bytes: bytes, max_frames: int = MAX_VIDEO_FRAMES) -> list[bytes]:
    return await asyncio.to_thread(_extract_video_frames_sync, video_bytes, max_frames)


async def extract_video_audio(video_bytes: bytes) -> bytes | None:
    return await asyncio.to_thread(_extract_video_audio_sync, video_bytes)


async def render_pdf_first_page(pdf_bytes: bytes) -> bytes:
    return await asyncio.to_thread(_render_pdf_first_page_sync, pdf_bytes)


async def convert_audio_to_mp3(audio_bytes: bytes) -> bytes:
    return await asyncio.to_thread(_convert_audio_to_mp3_sync, audio_bytes)
