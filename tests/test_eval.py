"""Tests for evaluation metrics, label loading, hyperparameter tuning, and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from streamclip.cli import app
from streamclip.config import Config, FusionWeights
from streamclip.eval.labels import GroundTruthLabel, load_labels, save_labels
from streamclip.eval.metrics import (
    EvalMetrics,
    compute_overlap_ratio,
    evaluate,
    is_hit,
)
from streamclip.eval.tune import (
    compute_optimization_score,
    generate_grid_configs,
    generate_random_configs,
    tune_hyperparameters,
)
from streamclip.ingest.chat import parse_chat
from streamclip.ingest.video import probe
from streamclip.models import Candidate, SignalTrack
from streamclip.signals.audio import extract_audio_signal, silence_points
from streamclip.signals.chat import extract_chat_signal

FIXTURES_DIR = Path(__file__).parent / "fixtures"
runner = CliRunner()


def test_ground_truth_label_properties() -> None:
    """Test GroundTruthLabel attributes and bounds."""
    lbl = GroundTruthLabel(t_start=50.0, t_end=70.0)
    assert lbl.duration == 20.0
    assert lbl.center_t == 60.0
    assert lbl.contains(60.0) is True
    assert lbl.contains(50.0) is True
    assert lbl.contains(70.0) is True
    assert lbl.contains(49.9) is False
    assert lbl.contains(70.1) is False

    with pytest.raises(ValueError, match="t_end"):
        GroundTruthLabel(t_start=80.0, t_end=70.0)


def test_load_and_save_labels(tmp_path: Path) -> None:
    """Test loading and saving ground truth labels."""
    labels = [
        GroundTruthLabel(t_start=10.0, t_end=25.0),
        GroundTruthLabel(t_start=100.0, t_end=120.0),
    ]
    target_file = tmp_path / "labels.json"
    save_labels(labels, target_file)

    loaded = load_labels(target_file)
    assert len(loaded) == 2
    assert loaded[0].t_start == 10.0
    assert loaded[0].t_end == 25.0
    assert loaded[1].t_start == 100.0

    # Also test loading from JSON string
    json_str = '[{"t_start": 30.0, "t_end": 45.0}]'
    loaded_str = load_labels(json_str)
    assert len(loaded_str) == 1
    assert loaded_str[0].t_start == 30.0


def test_is_hit_logic() -> None:
    """Test hit criteria: peak inside window OR >= 50% window overlap."""
    lbl = GroundTruthLabel(t_start=50.0, t_end=70.0)  # center=60.0, duration=20.0

    # Case 1: Peak inside ground truth (even if overlap < 50%)
    c1 = Candidate(peak_t=60.0, start=58.0, end=65.0, score=10.0)
    assert is_hit(c1, lbl) is True

    # Case 2: Peak outside, but window overlaps by >= 50%
    # Candidate [40.0, 65.0] (25s duration). Overlap with [50, 70] is [50, 65] = 15s.
    # 15s / 20s (label duration) = 75% >= 50%.
    c2 = Candidate(peak_t=45.0, start=40.0, end=65.0, score=8.0)
    assert is_hit(c2, lbl) is True

    # Case 3: Peak outside and overlap < 50%
    # Candidate [40.0, 52.0] (12s duration). Overlap with [50, 70] is [50, 52] = 2s.
    # 2s / 20s = 10%, 2s / 12s = 16.7%.
    c3 = Candidate(peak_t=45.0, start=40.0, end=52.0, score=7.0)
    assert is_hit(c3, lbl) is False

    # Case 4: Completely disjoint
    c4 = Candidate(peak_t=10.0, start=5.0, end=20.0, score=6.0)
    assert is_hit(c4, lbl) is False
    assert compute_overlap_ratio(c4, lbl) == 0.0


def test_evaluate_metrics() -> None:
    """Test calculation of Recall@K, Precision@K, and mean time offset."""
    labels = [
        GroundTruthLabel(t_start=50.0, t_end=70.0),  # center 60
        GroundTruthLabel(t_start=150.0, t_end=170.0),  # center 160
    ]

    # 3 candidates: 2 hits (t=62, t=161), 1 false positive (t=300)
    candidates = [
        Candidate(peak_t=62.0, start=50.0, end=75.0, score=10.0),
        Candidate(peak_t=161.0, start=150.0, end=175.0, score=9.0),
        Candidate(peak_t=300.0, start=290.0, end=310.0, score=5.0),
    ]

    # Evaluate at K=2 (top 2 candidates hit both labels)
    m2 = evaluate(candidates, labels, k=2)
    assert m2.k == 2
    assert m2.hits == 2
    assert m2.recall == 1.0  # 2 hits / 2 labels
    assert m2.precision == 1.0  # 2 hits / 2 evaluated candidates
    # Offsets: |62 - 60| = 2.0, |161 - 160| = 1.0 -> mean = 1.5
    assert abs(m2.mean_time_offset - 1.5) < 1e-4

    # Evaluate at K=3 (2 hits, 1 miss)
    m3 = evaluate(candidates, labels, k=3)
    assert m3.hits == 2
    assert m3.recall == 1.0  # 2 hits / 2 labels
    assert abs(m3.precision - (2 / 3)) < 1e-4  # 2 hits / 3 evaluated candidates
    assert abs(m3.mean_time_offset - 1.5) < 1e-4

    # Empty candidates
    m_empty = evaluate([], labels, k=5)
    assert m_empty.hits == 0
    assert m_empty.recall == 0.0
    assert m_empty.precision == 0.0


def test_optimization_score_ordering() -> None:
    """Verify that compute_optimization_score strictly enforces precision floor."""
    # Meeting precision floor beats not meeting it
    m_high_rec_low_prec = EvalMetrics(
        recall=1.0,
        precision=0.4,
        mean_time_offset=1.0,
        k=5,
        hits=5,
        total_labels=5,
        total_candidates=5,
    )
    m_lower_rec_high_prec = EvalMetrics(
        recall=0.8,
        precision=0.8,
        mean_time_offset=1.0,
        k=5,
        hits=4,
        total_labels=5,
        total_candidates=5,
    )

    score_low_prec = compute_optimization_score(m_high_rec_low_prec, precision_floor=0.5)
    score_high_prec = compute_optimization_score(m_lower_rec_high_prec, precision_floor=0.5)

    assert score_high_prec > score_low_prec
    assert score_high_prec[0] == 1  # met floor
    assert score_low_prec[0] == 0  # failed floor


def test_config_generators() -> None:
    """Test grid and seeded random configuration generators."""
    grid_cfgs = generate_grid_configs()
    assert len(grid_cfgs) > 10

    # Test seeded random reproducibility
    r_cfgs1 = generate_random_configs(n_trials=10, seed=123)
    r_cfgs2 = generate_random_configs(n_trials=10, seed=123)
    assert len(r_cfgs1) == 10
    for c1, c2 in zip(r_cfgs1, r_cfgs2, strict=True):
        assert c1.fusion_weights.chat == c2.fusion_weights.chat
        assert c1.peak_height == c2.peak_height
        assert c1.min_peak_gap_s == c2.min_peak_gap_s


def test_tune_hyperparameters(tmp_path: Path) -> None:
    """Test hyperparameter tuning pipeline."""
    # Create synthetic signals with peaks at 60, 180
    vals1 = [1.0] * 300
    vals1[60] = 50.0
    vals1[180] = 60.0
    track1 = SignalTrack(name="chat", hz=1.0, values=vals1, meta={"activity_confidence": 1.0})

    vals2 = [1.0] * 300
    vals2[60] = 40.0
    vals2[180] = 55.0
    track2 = SignalTrack(name="audio", hz=1.0, values=vals2, meta={"activity_confidence": 1.0})

    labels = [
        GroundTruthLabel(t_start=50.0, t_end=70.0),
        GroundTruthLabel(t_start=170.0, t_end=190.0),
    ]

    out_yaml = tmp_path / "tuned.yaml"
    res = tune_hyperparameters(
        tracks=[track1, track2],
        vod=300.0,
        labels=labels,
        method="grid",
        k=2,
        precision_floor=0.5,
        out_path=out_yaml,
        param_grid={
            "fusion_weights": [FusionWeights(chat=0.6, audio=0.4, vision=0.0)],
            "peak_height": [1.0, 2.0],
            "min_peak_gap_s": [15.0, 20.0],
        },
    )

    assert out_yaml.exists()
    assert res.best_metrics.recall == 1.0
    assert res.best_metrics.precision == 1.0
    assert len(res.trials) == 4

    # Verify saved YAML loads cleanly
    loaded_cfg = Config.from_yaml(out_yaml)
    assert loaded_cfg.peak_height == res.best_config.peak_height


@pytest.mark.skipif(
    not (FIXTURES_DIR / "video.mp4").exists(),
    reason="Synthetic fixtures not available",
)
def test_tune_with_fixtures(tmp_path: Path) -> None:
    """Test tuning against synthetic video and chat fixtures."""
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"
    labels_path = FIXTURES_DIR / "labels.json"

    vod = probe(video_path)
    msgs = parse_chat(chat_path)
    labels = load_labels(labels_path)

    chat_track = extract_chat_signal(vod, msgs)
    audio_track = extract_audio_signal(vod)
    silence_pts = silence_points(vod)

    out_yaml = tmp_path / "fixtures_tuned.yaml"
    res = tune_hyperparameters(
        tracks=[chat_track, audio_track],
        vod=vod,
        labels=labels,
        silence_points=silence_pts,
        method="random",
        k=5,
        precision_floor=0.5,
        n_trials=8,
        seed=42,
        out_path=out_yaml,
    )

    assert out_yaml.exists()
    assert res.best_metrics.recall >= 0.8
    assert res.best_metrics.precision >= 0.5


def test_cli_eval_with_candidates(tmp_path: Path) -> None:
    """Test CLI eval command with precomputed candidates file."""
    labels_file = tmp_path / "labels.json"
    save_labels([GroundTruthLabel(t_start=50.0, t_end=70.0)], labels_file)

    cands_file = tmp_path / "candidates.json"
    cands_data = [
        Candidate(
            peak_t=60.0,
            start=45.0,
            end=75.0,
            score=10.0,
            components={"chat": 5.0},
        ).model_dump(mode="json")
    ]
    cands_file.write_text(json.dumps(cands_data), encoding="utf-8")

    result = runner.invoke(app, ["eval", "-l", str(labels_file), "--candidates", str(cands_file)])
    assert result.exit_code == 0
    assert "Highlight Evaluation Results" in result.output
    assert "Recall@5: 1.0000" in result.output
    assert "Precision@5: 1.0000" in result.output


@pytest.mark.skipif(
    not (FIXTURES_DIR / "video.mp4").exists(),
    reason="Synthetic fixtures not available",
)
def test_cli_eval_and_tune_with_video(tmp_path: Path) -> None:
    """Test CLI eval and tune commands end-to-end with fixture video and chat."""
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"
    labels_path = FIXTURES_DIR / "labels.json"
    out_yaml = tmp_path / "cli_tuned.yaml"

    # 1. Run eval
    eval_res = runner.invoke(
        app,
        [
            "eval",
            "-l",
            str(labels_path),
            "-v",
            str(video_path),
            "-c",
            str(chat_path),
            "-k",
            "5",
        ],
    )
    assert eval_res.exit_code == 0
    assert "Highlight Evaluation Results (K=5):" in eval_res.output
    assert "Recall@5:" in eval_res.output

    # 2. Run tune
    tune_res = runner.invoke(
        app,
        [
            "tune",
            "-l",
            str(labels_path),
            "-v",
            str(video_path),
            "-c",
            str(chat_path),
            "-o",
            str(out_yaml),
            "-m",
            "random",
            "--trials",
            "4",
            "-k",
            "5",
        ],
    )
    assert tune_res.exit_code == 0
    assert "Tuning complete!" in tune_res.output
    assert out_yaml.exists()
