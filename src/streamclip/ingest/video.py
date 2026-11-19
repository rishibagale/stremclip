"""Video and stream ingest utilities for streamclip."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from streamclip.models import VodMeta


class VideoIngestError(Exception):
    """Base exception for video ingest errors."""


class DownloadError(VideoIngestError):
    """Raised when video download via yt-dlp fails."""


class ProbeError(VideoIngestError):
    """Raised when media inspection via ffprobe fails."""


def download(
    url: str,
    dest: Path | str,
    quality: str = "bestvideo+bestaudio/best",
) -> Path:
    """Download livestream VOD from URL using yt-dlp.

    Args:
        url: URL of the livestream or VOD to download.
        dest: Destination file path for the downloaded video.
        quality: Format specification selector passed to yt-dlp -f.

    Returns:
        Path to the downloaded file.

    Raises:
        DownloadError: If yt-dlp fails or is not found.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        quality,
        "-o",
        str(dest_path),
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as err:
        raise DownloadError(f"yt-dlp executable not found on PATH: {err}") from err
    except OSError as err:
        raise DownloadError(f"Failed to execute yt-dlp process: {err}") from err

    if result.returncode != 0:
        err_msg = result.stderr.strip() if result.stderr else f"exit code {result.returncode}"
        raise DownloadError(f"yt-dlp failed: {err_msg}")

    return dest_path


def _parse_fps(rate_str: str) -> float:
    """Parse a fractional or decimal frame rate string (e.g. '30/1' or '29.97')."""
    if not rate_str or rate_str == "0/0":
        return 0.0

    if "/" in rate_str:
        num_str, den_str = rate_str.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
            return (num / den) if den > 0 else 0.0
        except ValueError:
            return 0.0

    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def probe(path: Path | str) -> VodMeta:
    """Probe a media file using ffprobe and return VodMeta.

    Args:
        path: Path to the video file to probe.

    Returns:
        VodMeta instance populated with probed video metadata.

    Raises:
        ProbeError: If ffprobe fails, the file does not exist, or required streams are missing.
    """
    target_path = Path(path)
    if not target_path.exists():
        raise ProbeError(f"Video file does not exist: {target_path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(target_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as err:
        raise ProbeError(f"ffprobe executable not found on PATH: {err}") from err
    except OSError as err:
        raise ProbeError(f"Failed to execute ffprobe: {err}") from err

    if result.returncode != 0:
        err_msg = result.stderr.strip() if result.stderr else f"exit code {result.returncode}"
        raise ProbeError(f"ffprobe failed: {err_msg}")

    try:
        data: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as err:
        raise ProbeError(f"Failed to parse ffprobe output as JSON: {err}") from err

    streams: list[dict[str, Any]] = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise ProbeError(f"No video stream found in {target_path}")

    v_stream = video_streams[0]

    try:
        width = int(v_stream.get("width", 0))
        height = int(v_stream.get("height", 0))
    except (ValueError, TypeError) as err:
        raise ProbeError(f"Invalid dimensions in video stream: {err}") from err

    if width <= 0 or height <= 0:
        raise ProbeError(f"Invalid video dimensions: {width}x{height}")

    fps = _parse_fps(str(v_stream.get("avg_frame_rate", "")))
    if fps <= 0.0:
        fps = _parse_fps(str(v_stream.get("r_frame_rate", "")))
    if fps <= 0.0:
        raise ProbeError(f"Invalid or undetectable frame rate in {target_path}")

    # Duration: prefer video stream duration, fallback to format duration
    duration_str = v_stream.get("duration") or data.get("format", {}).get("duration")
    if not duration_str:
        raise ProbeError(f"Could not determine duration for {target_path}")

    try:
        duration_s = float(duration_str)
    except (ValueError, TypeError) as err:
        raise ProbeError(f"Invalid duration value '{duration_str}': {err}") from err

    if duration_s <= 0.0:
        raise ProbeError(f"Media duration must be positive, got {duration_s}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    return VodMeta(
        path=target_path,
        duration_s=duration_s,
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
    )
