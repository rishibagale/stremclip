"""Unit tests for audio signal extraction, silence points, and burst peaks."""

from pathlib import Path

import numpy as np
import pytest

from streamclip.interfaces import SignalExtractor
from streamclip.models import VodMeta
from streamclip.signals.audio import (
    AudioSignalExtractor,
    classify_yamnet,
    extract_audio_signal,
    silence_points,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_vod() -> VodMeta:
    return VodMeta(
        path=FIXTURES_DIR / "video.mp4",
        duration_s=600.0,
        fps=1.0,
        width=160,
        height=120,
        has_audio=True,
    )


@pytest.fixture
def sample_no_audio_vod() -> VodMeta:
    return VodMeta(
        path=FIXTURES_DIR / "video_no_audio.mp4",
        duration_s=600.0,
        fps=1.0,
        width=160,
        height=120,
        has_audio=False,
    )


def test_audio_signal_extractor_implements_protocol() -> None:
    extractor = AudioSignalExtractor()
    assert isinstance(extractor, SignalExtractor)
    assert extractor.name == "audio"


def test_audio_signal_peaks_on_fixture_bursts(sample_vod: VodMeta) -> None:
    track = extract_audio_signal(sample_vod)
    assert track.name == "audio"
    assert track.hz == 1.0
    assert len(track.values) == 600
    assert track.matches_vod(sample_vod)
    assert track.meta["has_audio"] is True

    # Invariant: the 5 injected burst windows (around 60, 180, 300, 420, 540)
    # must show prominent peaks compared to background sine tone
    burst_centers = [60, 180, 300, 420, 540]
    background_indices = [15, 30, 110, 230, 350, 470]

    bg_level = float(np.mean([track.values[idx] for idx in background_indices]))
    assert bg_level > 0.0, "Background tone should have non-zero energy"

    for center in burst_centers:
        peak_level = max(track.values[center - 4 : center + 4])
        assert peak_level > 3.0 * bg_level, (
            f"Expected audio burst peak near {center}s, got {peak_level} vs bg {bg_level}"
        )


def test_no_audio_signal(sample_no_audio_vod: VodMeta) -> None:
    track = extract_audio_signal(sample_no_audio_vod)
    assert track.name == "audio"
    assert len(track.values) == 600
    assert all(v == 0.0 for v in track.values)
    assert track.meta["has_audio"] is False


def test_silence_points_detection(sample_vod: VodMeta, sample_no_audio_vod: VodMeta) -> None:
    # 1. On video with audio, background (-47 dB) is quiet compared to bursts (-21 dB)
    quiet_pts = silence_points(sample_vod, threshold_db=-30.0)
    assert len(quiet_pts) > 0

    # Background seconds should be in quiet_pts
    assert 10.0 in quiet_pts
    assert 20.0 in quiet_pts

    # The burst centers (-21 dB) should NOT be classified as quiet/silent (< -30 dB)
    for burst_center in [60.0, 180.0, 300.0, 420.0, 540.0]:
        assert burst_center not in quiet_pts

    # 2. On video with no audio, all timestamps are silence points
    no_audio_quiet = silence_points(sample_no_audio_vod)
    assert len(no_audio_quiet) == 600


def test_yamnet_fallback() -> None:
    dummy_audio = np.zeros(16000 * 2, dtype=np.float32)
    res = classify_yamnet(dummy_audio)
    assert "scream" in res
    assert "laughter" in res
    assert "available" in res
    assert isinstance(res["scream"], float)
    assert isinstance(res["laughter"], float)
