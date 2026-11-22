"""Unit tests for automatic facecam detection and skin-chroma clustering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from streamclip.models import FacecamBox, VodMeta
from streamclip.render.face_detection import (
    DetectedFacecam,
    _detect_skin_mask,
    detect_facecam_in_frame,
    detect_facecam_region,
)
from streamclip.signals.frames import VideoFrame


def test_detect_skin_mask_rgb() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:60, 40:60] = [180, 120, 90]

    mask = _detect_skin_mask(frame)
    assert mask.shape == (100, 100)
    assert np.all(mask[40:60, 40:60])
    assert not np.any(mask[0:30, 0:30])


def test_detect_facecam_in_frame_synthetic() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[400:650, 50:350] = [190, 130, 100]

    detection = detect_facecam_in_frame(frame)
    assert detection is not None
    assert isinstance(detection.box, FacecamBox)
    assert detection.w % 2 == 0
    assert detection.h % 2 == 0
    assert 0 <= detection.x < 1280
    assert 0 <= detection.y < 720
    assert detection.x + detection.w <= 1280
    assert detection.y + detection.h <= 720


def test_detect_facecam_in_frame_empty_returns_fallback() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detection = detect_facecam_in_frame(frame)
    assert detection is not None
    # No face or skin clustered, so is_detected is False
    assert not detection.is_detected
    assert detection.preset_name == "bottom-left"


def test_detect_facecam_region_with_detection() -> None:
    vod = VodMeta(
        path=Path("dummy.mp4"),
        duration_s=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )

    detected = DetectedFacecam(
        box=FacecamBox(x=100, y=500, w=380, h=480),
        confidence=0.9,
        preset_name="detected-cv2",
        is_detected=True,
    )

    fake_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    sampled = [VideoFrame(t=10.0, array=fake_frame)]
    with (
        patch("streamclip.render.face_detection.sample_frames", return_value=sampled),
        patch(
            "streamclip.render.face_detection.detect_facecam_in_frame",
            return_value=detected,
        ),
    ):
        result = detect_facecam_region(vod, num_samples=3)
        assert result.x == 100
        assert result.y == 500
        assert result.w == 380
        assert result.h == 480
        assert result.is_detected


def test_detect_facecam_region_fallback_preset() -> None:
    vod = VodMeta(
        path=Path("dummy.mp4"),
        duration_s=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )

    with patch("streamclip.render.face_detection.sample_frames", return_value=[]):
        result = detect_facecam_region(vod, num_samples=2)
        assert result.w % 2 == 0
        assert result.h % 2 == 0
        assert result.x == round(1920 * 0.01)
        assert result.y == round(1080 * 0.44)
        assert not result.is_detected
