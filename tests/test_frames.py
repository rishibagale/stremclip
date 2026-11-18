"""Tests for frame sampling via ffmpeg seeking."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from streamclip.models import VodMeta
from streamclip.signals.frames import VideoFrame, sample_frames

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_video_frame_base64() -> None:
    """Test VideoFrame base64 serialization."""
    raw_bytes = b"fake_jpeg_image_bytes"
    frame = VideoFrame(t=12.5, array=None, image_bytes=raw_bytes)  # type: ignore[arg-type]
    b64 = frame.to_base64()
    assert b64 == base64.b64encode(raw_bytes).decode("ascii")

    empty_frame = VideoFrame(t=0.0, array=None, image_bytes=b"")  # type: ignore[arg-type]
    assert empty_frame.to_base64() == ""


def test_sample_frames_bounds() -> None:
    """Test frame sampling boundary checks."""
    vod = VodMeta(
        path=Path("mock.mp4"),
        duration_s=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    # Start >= end returns empty list
    assert sample_frames(vod, start_s=50.0, end_s=40.0) == []
    assert sample_frames(vod, start_s=10.0, end_s=10.0) == []
    assert sample_frames(vod, start_s=10.0, end_s=20.0, interval_s=0.0) == []


def test_sample_frames_missing_file_fallback(tmp_path: Path) -> None:
    """Test fallback when media file is not on disk."""
    vod = VodMeta(
        path=tmp_path / "non_existent.mp4",
        duration_s=10.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    frames = sample_frames(vod, start_s=0.0, end_s=5.0, interval_s=1.0)
    assert len(frames) == 5
    assert frames[0].t == 0.0
    assert frames[0].array.shape == (180, 320, 3)


@pytest.mark.skipif(
    not (FIXTURES_DIR / "video.mp4").exists(),
    reason="Synthetic fixtures not available",
)
def test_sample_frames_with_ffmpeg_seek() -> None:
    """Test fast ffmpeg seek frame sampling on fixture video."""
    video_path = FIXTURES_DIR / "video.mp4"
    vod = VodMeta(
        path=video_path,
        duration_s=600.0,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=True,
    )

    # Sample window [50.0, 54.0] at 1 frame per second
    frames = sample_frames(
        vod,
        start_s=50.0,
        end_s=54.0,
        interval_s=1.0,
        width=320,
        height=180,
    )

    assert len(frames) >= 4
    for f in frames:
        assert 50.0 <= f.t <= 55.0
        assert f.array.shape == (180, 320, 3)
        assert len(f.image_bytes) > 0  # JPEG bytes present
        assert len(f.to_base64()) > 0
