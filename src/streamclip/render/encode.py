"""Video and audio encoding: x264/NVENC, AAC, loudness normalization, and faststart."""

from __future__ import annotations

import functools
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from streamclip.ingest.video import probe
from streamclip.models import RenderSpec, VodMeta
from streamclip.render.captions import generate_ass_subtitles
from streamclip.render.layout import build_filtergraph


class RenderError(Exception):
    """Raised when clip encoding or filtergraph execution fails."""


@functools.lru_cache(maxsize=1)
def is_nvenc_available() -> bool:
    """Test whether NVIDIA NVENC hardware acceleration (h264_nvenc) is available."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=160x120:rate=1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=5.0)
        return res.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def select_video_encoder(prefer_nvenc: bool = True) -> str:
    """Select appropriate video encoder: h264_nvenc if requested and available, else libx264."""
    if prefer_nvenc and is_nvenc_available():
        return "h264_nvenc"
    return "libx264"


@dataclass
class RenderJob:
    """Specification of a clip render task."""

    vod: VodMeta | Path | str
    start: float
    end: float
    output_path: Path | str
    spec: RenderSpec | None = None
    prefer_nvenc: bool = True
    mask_profanity: bool = True
    config: Any | None = None


def render_clip(job: RenderJob) -> Path:
    """Render a single highlight clip to 1080x1920 vertical video.

    Applies:
      - Layout filtergraph (stack, gameplay_only, facecam_only).
      - Subtitle burn-in with word-level highlight and mobile-safe margins (if captions present).
      - Video encoding (x264 or NVENC).
      - Audio loudness normalization (EBU R128 / loudnorm=I=-16:TP=-1.5:LRA=11).
      - Audio encoding (AAC 192k).
      - Streaming optimization (-movflags +faststart).

    Args:
        job: RenderJob instance detailing VOD source, time window, spec, and destination.

    Returns:
        Path to rendered MP4 video file.

    Raises:
        RenderError: If ffmpeg execution fails.
        ValueError: If start >= end or invalid inputs.
    """
    start = max(0.0, float(job.start))
    end = max(start, float(job.end))
    duration = end - start
    if duration <= 0.0:
        raise ValueError(f"Clip duration must be positive, got start={start}, end={end}")

    output_path = Path(job.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = job.spec or RenderSpec(layout="gameplay_only")

    # Resolve VOD path and audio capability
    if isinstance(job.vod, VodMeta):
        vod_path = Path(job.vod.path)
        has_audio = job.vod.has_audio
    else:
        vod_path = Path(job.vod)
        try:
            meta = probe(vod_path)
            has_audio = meta.has_audio
        except Exception:
            has_audio = True

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RenderError("ffmpeg executable not found on system PATH")

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        ass_path: Path | None = None

        if spec.captions:
            ass_path = temp_dir / "captions.ass"
            generate_ass_subtitles(
                spec.captions,
                out_path=ass_path,
                clip_start=start,
                mask_profanity=job.mask_profanity,
            )

        v_filter = build_filtergraph(spec, config=job.config, ass_path=ass_path)
        a_filter = "loudnorm=I=-16:TP=-1.5:LRA=11"

        cmd: list[str] = [
            ffmpeg_bin,
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(vod_path),
        ]

        if has_audio:
            filter_complex = f"{v_filter};[0:a]{a_filter}[a]"
        else:
            # Generate null stereo audio so output file always includes AAC audio stream
            cmd.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=44100",
                ]
            )
            filter_complex = f"{v_filter};[1:a]{a_filter}[a]"

        cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "[a]",
            ]
        )

        encoder = select_video_encoder(prefer_nvenc=job.prefer_nvenc)
        if encoder == "h264_nvenc":
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-pix_fmt", "yuv420p"])
        else:
            cmd.extend(
                ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
            )

        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-y",
                str(output_path),
            ]
        )

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as err:
            raise RenderError(f"FFmpeg render failed for {output_path}: {err.stderr}") from err

    return output_path


def render_clips(
    jobs: list[RenderJob],
    max_workers: int = 2,
) -> list[Path]:
    """Render multiple highlight clips in parallel with a worker limit.

    Args:
        jobs: List of RenderJob specifications.
        max_workers: Maximum number of concurrent ffmpeg render processes.

    Returns:
        List of paths to rendered MP4 videos, in the same order as jobs.
    """
    if not jobs:
        return []

    workers = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(render_clip, job) for job in jobs]
        return [f.result() for f in futures]
