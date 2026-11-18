"""Tests for multimodal signal fusion and peak detection."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from streamclip.config import Config, FusionWeights
from streamclip.fusion import (
    FusedSignal,
    compute_rolling_zscore,
    fuse_signals,
)
from streamclip.ingest.chat import parse_chat
from streamclip.ingest.video import probe
from streamclip.models import SignalTrack
from streamclip.signals.audio import extract_audio_signal
from streamclip.signals.chat import extract_chat_signal

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_compute_rolling_zscore_mean_std() -> None:
    """Test rolling baseline z-score with mean and standard deviation."""
    # Flat signal
    flat = np.ones(100, dtype=np.float64) * 5.0
    z_flat = compute_rolling_zscore(flat, window_size=20, use_mad=False)
    assert len(z_flat) == 100
    assert np.all(z_flat >= 0.0)
    assert np.all(z_flat < 0.1)

    # Signal with a single high spike at index 50
    spike_arr = np.ones(100, dtype=np.float64) * 2.0
    spike_arr[50] = 50.0
    z_spike = compute_rolling_zscore(spike_arr, window_size=30, use_mad=False)
    assert z_spike[50] > 5.0
    assert np.argmax(z_spike) == 50


def test_compute_rolling_zscore_median_mad() -> None:
    """Test rolling baseline z-score with median and MAD."""
    spike_arr = np.ones(100, dtype=np.float64) * 2.0
    spike_arr[30] = 40.0
    z_mad = compute_rolling_zscore(spike_arr, window_size=30, use_mad=True)
    assert len(z_mad) == 100
    assert np.all(z_mad >= 0.0)
    assert z_mad[30] > 5.0
    assert np.argmax(z_mad) == 30


def test_compute_rolling_zscore_empty() -> None:
    """Test edge cases with empty array or very small array."""
    empty = np.zeros(0, dtype=np.float64)
    z_empty = compute_rolling_zscore(empty)
    assert len(z_empty) == 0

    tiny = np.array([4.0], dtype=np.float64)
    z_tiny = compute_rolling_zscore(tiny, window_size=10)
    assert len(z_tiny) == 1
    assert z_tiny[0] >= 0.0


def test_fuse_signals_empty() -> None:
    """Test fusion with empty track list."""
    res = fuse_signals([])
    assert isinstance(res, FusedSignal)
    assert res.track.name == "fused"
    assert len(res.track.values) == 0
    assert len(res.peaks) == 0
    assert res.track.meta.get("empty") is True


def test_fuse_signals_confidence_weighting() -> None:
    """Test that zero-confidence or empty tracks do not contribute to fused signal."""
    # Track 1: active chat with spike at t=30
    vals1 = [1.0] * 100
    vals1[30] = 20.0
    t1 = SignalTrack(
        name="chat",
        hz=1.0,
        values=vals1,
        meta={"activity_confidence": 1.0},
    )

    # Track 2: empty or silent track with confidence 0.0
    vals2 = [5.0] * 100
    vals2[70] = 50.0  # Would be a massive spike if considered
    t2 = SignalTrack(
        name="audio",
        hz=1.0,
        values=vals2,
        meta={"has_audio": False, "activity_confidence": 0.0},
    )

    cfg = Config(
        fusion_weights=FusionWeights(chat=0.7, audio=0.3, vision=0.0),
        peak_height=1.0,
        min_peak_gap_s=10.0,
    )
    res = fuse_signals([t1, t2], config=cfg)
    assert len(res.peaks) > 0
    # Peak at 30 should be detected; no peak at 70 because audio confidence is 0.0
    peak_times = [p.peak_t for p in res.peaks]
    assert any(28.0 <= pt <= 32.0 for pt in peak_times)
    assert not any(68.0 <= pt <= 72.0 for pt in peak_times)


def test_fuse_signals_components_breakdown() -> None:
    """Test that detected peaks have per-track component breakdowns."""
    # Spikes at t=50 for both tracks
    vals1 = [1.0] * 100
    vals1[50] = 25.0
    vals2 = [2.0] * 100
    vals2[50] = 30.0

    t1 = SignalTrack(name="chat", hz=1.0, values=vals1, meta={"activity_confidence": 1.0})
    t2 = SignalTrack(name="audio", hz=1.0, values=vals2, meta={"activity_confidence": 0.8})

    cfg = Config(
        fusion_weights=FusionWeights(chat=0.5, audio=0.5, vision=0.0),
        peak_height=1.0,
        min_peak_gap_s=10.0,
    )
    res = fuse_signals({"chat": t1, "audio": t2}, config=cfg)
    assert len(res.peaks) >= 1
    top_peak = res.peaks[0]
    assert 48.0 <= top_peak.peak_t <= 52.0
    assert "chat" in top_peak.components
    assert "audio" in top_peak.components
    assert top_peak.components["chat"] > 0.0
    assert top_peak.components["audio"] > 0.0


def test_fuse_signals_top_k_and_min_gap() -> None:
    """Test top_k capping and min_peak_gap_s suppression."""
    vals = [1.0] * 200
    # Create 5 spikes at 20, 25 (close), 80, 120, 160
    vals[20] = 40.0
    vals[25] = 50.0  # within 15s of 20
    vals[80] = 45.0
    vals[120] = 48.0
    vals[160] = 60.0

    track = SignalTrack(name="chat", hz=1.0, values=vals, meta={"activity_confidence": 1.0})
    cfg = Config(
        fusion_weights=FusionWeights(chat=1.0, audio=0.0, vision=0.0),
        peak_height=1.0,
        min_peak_gap_s=15.0,
        top_k=2,
    )
    res = fuse_signals([track], config=cfg)
    assert len(res.peaks) == 2  # capped by top_k
    # Highest peaks were at 160 and 25
    peak_times = [p.peak_t for p in res.peaks]
    assert any(158.0 <= pt <= 162.0 for pt in peak_times)


def test_fuse_signals_deterministic() -> None:
    """Verify that fusion runs deterministically."""
    np.random.seed(42)
    vals = (np.random.rand(300) * 10.0).tolist()
    track = SignalTrack(name="audio", hz=1.0, values=vals, meta={"confidence": 1.0})

    res1 = fuse_signals([track], use_mad=False)
    res2 = fuse_signals([track], use_mad=False)

    assert res1.track.values == res2.track.values
    assert len(res1.peaks) == len(res2.peaks)
    for p1, p2 in zip(res1.peaks, res2.peaks, strict=True):
        assert p1.peak_t == p2.peak_t
        assert p1.score == p2.score
        assert p1.components == p2.components


def test_fuse_signals_8hr_benchmark() -> None:
    """Benchmark test: an 8-hour track must process in under 2 seconds."""
    # 8 hours = 8 * 3600 = 28,800 seconds
    seconds_8hr = 8 * 3600
    rng = np.random.default_rng(12345)

    # Generate synthetic 8-hour signals
    chat_vals = rng.poisson(lam=2.0, size=seconds_8hr).astype(np.float64)
    audio_vals = rng.normal(loc=0.05, scale=0.02, size=seconds_8hr).astype(np.float64)
    audio_vals = np.maximum(0.0, audio_vals)

    # Invert a few big spikes
    chat_vals[1800] = 50.0
    chat_vals[7200] = 80.0
    audio_vals[1800] = 0.8
    audio_vals[14400] = 0.9

    track_chat = SignalTrack(
        name="chat",
        hz=1.0,
        values=[float(x) for x in chat_vals.tolist()],
        meta={"activity_confidence": 1.0},
    )
    track_audio = SignalTrack(
        name="audio",
        hz=1.0,
        values=[float(x) for x in audio_vals.tolist()],
        meta={"activity_confidence": 1.0},
    )

    cfg = Config(
        fusion_weights=FusionWeights(chat=0.6, audio=0.4, vision=0.0),
        peak_height=1.0,
        min_peak_gap_s=20.0,
        top_k=20,
    )

    t0 = time.perf_counter()
    res = fuse_signals([track_chat, track_audio], config=cfg, use_mad=False)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"8-hour fusion took {elapsed:.3f}s, expected < 2.0s"
    assert len(res.track.values) == seconds_8hr
    assert len(res.peaks) > 0


@pytest.mark.skipif(
    not (FIXTURES_DIR / "video.mp4").exists(),
    reason="Synthetic fixtures not available",
)
def test_fuse_signals_with_fixtures() -> None:
    """Test multimodal fusion on synthetic fixtures containing 5 injected bursts."""
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"
    labels_path = FIXTURES_DIR / "labels.json"

    vod = probe(video_path)
    messages = parse_chat(chat_path)
    labels: list[dict[str, float]] = json.loads(labels_path.read_text(encoding="utf-8"))

    chat_track = extract_chat_signal(vod, messages)
    audio_track = extract_audio_signal(vod)

    cfg = Config(
        fusion_weights=FusionWeights(chat=0.6, audio=0.4, vision=0.0),
        peak_height=1.0,
        min_peak_gap_s=20.0,
        top_k=10,
    )

    fused = fuse_signals([chat_track, audio_track], config=cfg)
    assert len(fused.peaks) >= 5

    # Check that each of the 5 ground-truth moments has a corresponding peak
    detected_peak_times = [p.peak_t for p in fused.peaks]
    for label in labels:
        t_start = label["t_start"]
        t_end = label["t_end"]
        # There must be a peak near [t_start - 5, t_end + 5]
        matching = [pt for pt in detected_peak_times if (t_start - 5.0) <= pt <= (t_end + 5.0)]
        assert len(matching) >= 1, (
            f"Ground truth window [{t_start}, {t_end}] missed in peaks: {detected_peak_times}"
        )
