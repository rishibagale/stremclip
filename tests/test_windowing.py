"""Tests for candidate windowing, snapping, and overlap merging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from streamclip.config import Config, FusionWeights
from streamclip.fusion import Peak, fuse_signals
from streamclip.ingest.chat import parse_chat
from streamclip.ingest.video import probe
from streamclip.models import Candidate, Word
from streamclip.signals.audio import extract_audio_signal, silence_points
from streamclip.signals.chat import extract_chat_signal
from streamclip.windowing import (
    build_boundary_points,
    clamp_window,
    create_candidates,
    find_decay_end,
    merge_overlapping_candidates,
    snap_to_boundary,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_build_boundary_points() -> None:
    """Test boundary point aggregation and sorting from silence and words."""
    silence = [12.4567, 45.1, 12.4561]
    words = [
        Word(start=10.0, end=11.5, text="hello"),
        Word(start=45.1, end=46.0, text="world"),
    ]
    pts = build_boundary_points(silence, words)
    assert pts == sorted(set(pts))
    assert 10.0 in pts
    assert 11.5 in pts
    assert 45.1 in pts
    assert 46.0 in pts


def test_snap_to_boundary_within_tolerance() -> None:
    """Test snapping to closest boundary point within tolerance window."""
    boundaries = [40.0, 44.2, 50.0]
    # Target 45.0 is 0.8s away from 44.2 (within 3.0s)
    snapped = snap_to_boundary(45.0, boundaries, tolerance_s=3.0)
    assert snapped == 44.2


def test_snap_to_boundary_outside_tolerance() -> None:
    """Test that timestamp remains unchanged if outside tolerance window."""
    boundaries = [35.0, 55.0]
    # Target 45.0 is 10s away from both boundaries
    snapped = snap_to_boundary(45.0, boundaries, tolerance_s=3.0)
    assert snapped == 45.0


def test_find_decay_end() -> None:
    """Test decay end search with post_roll."""
    scores = np.zeros(100, dtype=np.float64)
    # Peak at index 50 with value 10.0
    scores[50] = 10.0
    scores[51] = 8.0
    scores[52] = 5.0
    scores[53] = 1.0  # Decays below 0.25 * 10 = 2.5
    scores[54] = 0.5

    # With post_roll_s = 4.0, end should be 53 + 4 = 57.0
    end_t = find_decay_end(
        peak_t=50.0,
        fused_scores=scores,
        peak_score=10.0,
        baseline_ratio=0.25,
        post_roll_s=4.0,
    )
    assert end_t == 57.0


def test_clamp_window_duration() -> None:
    """Test minimum and maximum duration clamping."""
    # Too short: 50 to 55 (5s) -> clamped to min 15s
    s1, e1 = clamp_window(
        start=50.0,
        end=55.0,
        peak_t=52.0,
        clip_min_s=15.0,
        clip_max_s=60.0,
        vod_duration_s=600.0,
    )
    assert (e1 - s1) >= 15.0

    # Too long: 10 to 100 (90s) -> clamped to max 60s
    s2, e2 = clamp_window(
        start=10.0,
        end=100.0,
        peak_t=30.0,
        clip_min_s=15.0,
        clip_max_s=60.0,
        vod_duration_s=600.0,
    )
    assert (e2 - s2) <= 60.0
    assert s2 <= 30.0 <= e2


def test_clamp_window_vod_bounds() -> None:
    """Test clamping to VOD bounds [0.0, duration_s]."""
    # Start before 0
    s1, e1 = clamp_window(
        start=-10.0,
        end=10.0,
        peak_t=5.0,
        clip_min_s=15.0,
        clip_max_s=60.0,
        vod_duration_s=100.0,
    )
    assert s1 == 0.0
    assert e1 >= 15.0

    # End after duration
    s2, e2 = clamp_window(
        start=90.0,
        end=110.0,
        peak_t=95.0,
        clip_min_s=15.0,
        clip_max_s=60.0,
        vod_duration_s=100.0,
    )
    assert e2 == 100.0
    assert s2 <= 85.0


def test_merge_overlapping_candidates_keeps_higher_score() -> None:
    """Test that overlapping candidates merge and keep the higher score and its peak."""
    c1 = Candidate(
        peak_t=60.0,
        start=45.0,
        end=75.0,
        score=12.0,
        components={"chat": 8.0, "audio": 4.0},
    )
    c2 = Candidate(
        peak_t=70.0,
        start=55.0,
        end=85.0,
        score=5.0,
        components={"chat": 2.0, "audio": 3.0},
    )

    merged = merge_overlapping_candidates([c1, c2], clip_min_s=15.0, clip_max_s=60.0)
    assert len(merged) == 1
    m = merged[0]
    assert m.score == 12.0
    assert m.peak_t == 60.0
    assert m.start == 45.0
    assert m.end == 85.0
    assert m.components["chat"] == 8.0


def test_merge_overlapping_candidates_chain() -> None:
    """Test multi-candidate overlapping chain."""
    c1 = Candidate(peak_t=20.0, start=10.0, end=30.0, score=3.0)
    c2 = Candidate(peak_t=32.0, start=25.0, end=45.0, score=9.0)  # Highest score
    c3 = Candidate(peak_t=48.0, start=40.0, end=55.0, score=4.0)

    merged = merge_overlapping_candidates([c1, c2, c3], clip_min_s=15.0, clip_max_s=60.0)
    assert len(merged) == 1
    assert merged[0].score == 9.0
    assert merged[0].peak_t == 32.0
    assert merged[0].start == 10.0
    assert merged[0].end == 55.0


def test_non_overlapping_candidates_preserved() -> None:
    """Test that non-overlapping candidates remain separate."""
    c1 = Candidate(peak_t=60.0, start=45.0, end=75.0, score=10.0)
    c2 = Candidate(peak_t=300.0, start=285.0, end=315.0, score=8.0)

    merged = merge_overlapping_candidates([c1, c2], clip_min_s=15.0, clip_max_s=60.0)
    assert len(merged) == 2


def test_create_candidates_basic() -> None:
    """Test create_candidates with synthetic peak and score signals."""
    peaks = [
        Peak(peak_t=60.0, score=10.0, components={"chat": 6.0, "audio": 4.0}),
        Peak(peak_t=200.0, score=8.0, components={"chat": 5.0, "audio": 3.0}),
    ]
    scores = np.zeros(300, dtype=np.float64)
    scores[55:65] = 10.0
    scores[195:205] = 8.0

    silence = [44.5, 184.0]
    cfg = Config(pre_roll_s=15.0, post_roll_s=4.0, clip_min_s=15.0, clip_max_s=60.0)

    candidates = create_candidates(
        peaks=peaks,
        fused_scores=scores,
        vod=300.0,
        config=cfg,
        silence_points=silence,
    )

    assert len(candidates) == 2
    # Verify first candidate snapped to 44.5 (since 60 - 15 = 45 is within 3.0s of 44.5)
    c1 = next(c for c in candidates if 50.0 <= c.peak_t <= 70.0)
    assert c1.start == 44.5
    assert c1.score == 10.0
    assert c1.duration >= 15.0
    assert c1.duration <= 60.0


@pytest.mark.skipif(
    not (FIXTURES_DIR / "video.mp4").exists(),
    reason="Synthetic fixtures not available",
)
def test_create_candidates_with_fixtures() -> None:
    """End-to-end integration test of signal fusion and candidate windowing with fixtures."""
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"
    labels_path = FIXTURES_DIR / "labels.json"

    vod = probe(video_path)
    messages = parse_chat(chat_path)
    labels: list[dict[str, float]] = json.loads(labels_path.read_text(encoding="utf-8"))

    # Extract signals and silence points
    chat_track = extract_chat_signal(vod, messages)
    audio_track = extract_audio_signal(vod)
    silence_pts = silence_points(vod)

    cfg = Config(
        fusion_weights=FusionWeights(chat=0.6, audio=0.4, vision=0.0),
        peak_height=1.0,
        min_peak_gap_s=20.0,
        pre_roll_s=15.0,
        post_roll_s=4.0,
        clip_min_s=15.0,
        clip_max_s=60.0,
        top_k=10,
    )

    fused = fuse_signals([chat_track, audio_track], config=cfg)
    candidates = create_candidates(
        peaks=fused.peaks,
        fused_scores=fused,
        vod=vod,
        config=cfg,
        silence_points=silence_pts,
    )

    assert len(candidates) >= 5

    # Check validity of every candidate window
    for cand in candidates:
        assert 0.0 <= cand.start < cand.end <= vod.duration_s
        assert cfg.clip_min_s <= cand.duration <= cfg.clip_max_s
        assert cand.start <= cand.peak_t <= cand.end
        assert cand.score > 0.0

    # Ensure no two candidates overlap
    sorted_cands = sorted(candidates, key=lambda c: c.start)
    for i in range(len(sorted_cands) - 1):
        assert sorted_cands[i].end <= sorted_cands[i + 1].start

    # Check coverage of the 5 ground-truth moments
    for label in labels:
        t_start = label["t_start"]
        t_end = label["t_end"]
        # Candidate window should intersect or contain ground truth moment
        matching = [c for c in candidates if not (c.end < t_start or c.start > t_end)]
        cand_ranges = [(c.start, c.end) for c in candidates]
        assert len(matching) >= 1, (
            f"Ground truth window [{t_start}, {t_end}] not covered by candidates: {cand_ranges}"
        )
