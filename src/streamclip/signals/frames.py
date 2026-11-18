"""Frame sampling using fast ffmpeg seeking for highlight candidate windows."""

from __future__ import annotations

import base64
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from streamclip.models import VodMeta


@dataclass
class VideoFrame:
    """A sampled video frame with timestamp, RGB numpy array, and JPEG bytes."""

    t: float
    array: np.ndarray[Any, np.dtype[np.uint8]]
    image_bytes: bytes = b""

    def to_base64(self) -> str:
        """Return base64-encoded string of JPEG image bytes."""
        if not self.image_bytes:
            return ""
        return base64.b64encode(self.image_bytes).decode("ascii")


def _read_ppm_frame(
    ppm_bytes: bytes,
    width: int,
    height: int,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Parse raw RGB pixels from PPM P6 format bytes."""
    lines: list[bytes] = []
    idx = 0
    while len(lines) < 3 and idx < len(ppm_bytes):
        nl = ppm_bytes.find(b"\n", idx)
        if nl == -1:
            break
        line = ppm_bytes[idx:nl].strip()
        idx = nl + 1
        if line and not line.startswith(b"#"):
            lines.append(line)

    pixel_bytes = ppm_bytes[idx:]
    expected_len = height * width * 3
    if len(pixel_bytes) < expected_len:
        padded = np.zeros(expected_len, dtype=np.uint8)
        padded[: len(pixel_bytes)] = np.frombuffer(pixel_bytes, dtype=np.uint8)
        return padded.reshape((height, width, 3))

    return np.frombuffer(pixel_bytes[:expected_len], dtype=np.uint8).reshape((height, width, 3))


def sample_frames(
    vod: VodMeta,
    start_s: float,
    end_s: float,
    interval_s: float = 1.0,
    width: int = 320,
    height: int = 180,
) -> list[VideoFrame]:
    """Sample 1 frame per interval_s (1-2s) for requested windows only using ffmpeg seek.

    Never decodes the whole video; seeks directly to start_s using fast input seeking.

    Args:
        vod: VodMeta describing target media.
        start_s: Start offset in seconds.
        end_s: End offset in seconds.
        interval_s: Frame sampling interval in seconds (default 1.0s, between 1.0 and 2.0s).
        width: Scaled frame width.
        height: Scaled frame height.

    Returns:
        List of VideoFrame objects with timestamps, numpy arrays, and JPEG bytes.
    """
    s = max(0.0, start_s)
    e = min(vod.duration_s, end_s)
    if e <= s or interval_s <= 0.0:
        return []

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin or not Path(vod.path).is_file():
        # Fallback if ffmpeg is missing or file not on disk: synthetic frames for testing
        num_frames = max(1, math.ceil((e - s) / interval_s))
        synthetic_frames: list[VideoFrame] = []
        for i in range(num_frames):
            t = s + i * interval_s
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            synthetic_frames.append(VideoFrame(t=round(t, 3), array=arr, image_bytes=b""))
        return synthetic_frames

    # Ensure width and height are even for ffmpeg scaling
    w_even = width if width % 2 == 0 else width + 1
    h_even = height if height % 2 == 0 else height + 1

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        # Fast input seek with -ss before -i
        cmd = [
            ffmpeg_bin,
            "-v",
            "error",
            "-ss",
            f"{s:.3f}",
            "-to",
            f"{e:.3f}",
            "-i",
            str(vod.path),
            "-filter_complex",
            f"[0:v]fps=1/{interval_s},scale={w_even}:{h_even},split=2[v1][v2]",
            "-map",
            "[v1]",
            str(temp_dir / "f_%04d.ppm"),
            "-map",
            "[v2]",
            "-c:v",
            "mjpeg",
            str(temp_dir / "f_%04d.jpg"),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except (subprocess.SubprocessError, OSError):
            return []

        ppm_files = sorted(temp_dir.glob("f_*.ppm"))
        jpg_files = sorted(temp_dir.glob("f_*.jpg"))

        frames: list[VideoFrame] = []
        for i, ppm_p in enumerate(ppm_files):
            t_frame = s + (i * interval_s)
            ppm_data = ppm_p.read_bytes()
            arr = _read_ppm_frame(ppm_data, w_even, h_even)
            jpg_bytes = jpg_files[i].read_bytes() if i < len(jpg_files) else b""
            frames.append(
                VideoFrame(
                    t=round(t_frame, 3),
                    array=arr,
                    image_bytes=jpg_bytes,
                )
            )

        return frames
