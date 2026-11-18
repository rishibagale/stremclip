"""Hyperparameter tuning for multimodal signal fusion and peak detection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Literal

from streamclip.config import Config, FusionWeights
from streamclip.eval.labels import GroundTruthLabel
from streamclip.eval.metrics import EvalMetrics, evaluate
from streamclip.fusion import fuse_signals
from streamclip.models import SignalTrack, VodMeta, Word
from streamclip.windowing import create_candidates


@dataclass
class TrialResult:
    """Outcome of a single parameter configuration evaluation trial."""

    trial_index: int
    config: Config
    metrics: EvalMetrics
    score: tuple[int, float, float, float]


@dataclass
class TuningResult:
    """Aggregated tuning search results and best discovered configuration."""

    best_config: Config
    best_metrics: EvalMetrics
    best_trial: TrialResult
    trials: list[TrialResult] = field(default_factory=list)
    out_path: Path | None = None


def compute_optimization_score(
    metrics: EvalMetrics,
    precision_floor: float = 0.5,
) -> tuple[int, float, float, float]:
    """Compute lexicographical optimization score for trial ranking.

    Objective: Maximize Recall@K above a precision floor.
    Hierarchy:
      1. Precision floor compliance (1 if precision >= floor, else 0).
      2. Recall@K (higher is better).
      3. Negative mean_time_offset (lower offset is better).
      4. Precision@K (higher precision as tie-breaker).
    """
    meets_floor = 1 if metrics.precision >= precision_floor else 0
    return (
        meets_floor,
        metrics.recall,
        -metrics.mean_time_offset,
        metrics.precision,
    )


def generate_grid_configs(
    base_config: Config | None = None,
    param_grid: dict[str, list[Any]] | None = None,
) -> list[Config]:
    """Generate exhaustive grid search configurations."""
    base = base_config or Config()

    default_weights: list[FusionWeights] = [
        FusionWeights(chat=0.6, audio=0.4, vision=0.0),
        FusionWeights(chat=0.5, audio=0.5, vision=0.0),
        FusionWeights(chat=0.7, audio=0.3, vision=0.0),
        FusionWeights(chat=0.4, audio=0.4, vision=0.2),
        FusionWeights(chat=0.3, audio=0.7, vision=0.0),
    ]
    default_heights: list[float] = [0.5, 1.0, 1.5, 2.0]
    default_gaps: list[float] = [10.0, 15.0, 20.0, 30.0]

    grid = param_grid or {}
    weights_list: list[FusionWeights] = grid.get("fusion_weights", default_weights)
    heights: list[float] = grid.get("peak_height", default_heights)
    gaps: list[float] = grid.get("min_peak_gap_s", default_gaps)

    configs: list[Config] = []
    for w, h, g in product(weights_list, heights, gaps):
        cfg = base.model_copy(deep=True)
        cfg.fusion_weights = w
        cfg.peak_height = float(h)
        cfg.min_peak_gap_s = float(g)
        configs.append(cfg)

    return configs


def generate_random_configs(
    n_trials: int = 30,
    seed: int = 42,
    base_config: Config | None = None,
) -> list[Config]:
    """Generate seeded pseudo-random parameter configurations."""
    base = base_config or Config()
    rng = random.Random(seed)

    gap_choices = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
    configs: list[Config] = []

    for _ in range(n_trials):
        chat_w = round(rng.uniform(0.2, 0.9), 2)
        audio_w = round(rng.uniform(0.1, 0.8), 2)
        vision_w = round(rng.uniform(0.0, 0.4), 2)
        height = round(rng.uniform(0.5, 2.5), 2)
        gap = float(rng.choice(gap_choices))

        cfg = base.model_copy(deep=True)
        cfg.fusion_weights = FusionWeights(chat=chat_w, audio=audio_w, vision=vision_w)
        cfg.peak_height = height
        cfg.min_peak_gap_s = gap
        configs.append(cfg)

    return configs


def tune_hyperparameters(
    tracks: list[SignalTrack] | dict[str, SignalTrack],
    vod: VodMeta | float,
    labels: list[GroundTruthLabel],
    silence_points: list[float] | None = None,
    words: list[Word] | None = None,
    method: Literal["grid", "random"] = "grid",
    k: int = 5,
    precision_floor: float = 0.5,
    n_trials: int = 30,
    seed: int = 42,
    out_path: Path | str | None = Path("configs/tuned.yaml"),
    base_config: Config | None = None,
    param_grid: dict[str, list[Any]] | None = None,
) -> TuningResult:
    """Tune fusion weights, peak height, and min gap to maximize recall@K above precision floor.

    Args:
        tracks: Multimodal SignalTracks to fuse.
        vod: VodMeta or duration in seconds.
        labels: Ground truth highlight annotations.
        silence_points: Optional silence points for candidate start snapping.
        words: Optional timed words for speech boundary snapping.
        method: Search strategy ('grid' or 'random').
        k: Evaluation cutoff rank for Recall@K and Precision@K.
        precision_floor: Minimum precision threshold to qualify.
        n_trials: Number of trials for random search.
        seed: Random seed for reproducible search.
        out_path: File path to save the best configuration YAML (default 'configs/tuned.yaml').
        base_config: Optional base configuration template.
        param_grid: Optional custom grid space for grid search.

    Returns:
        TuningResult holding best config, evaluation metrics, and all trial outcomes.
    """
    if method == "grid":
        configs = generate_grid_configs(base_config=base_config, param_grid=param_grid)
    elif method == "random":
        configs = generate_random_configs(n_trials=n_trials, seed=seed, base_config=base_config)
    else:
        raise ValueError(f"Unknown search method: '{method}'. Expected 'grid' or 'random'.")

    trials: list[TrialResult] = []

    for idx, cfg in enumerate(configs, start=1):
        # 1. Run signal fusion
        fused = fuse_signals(tracks, config=cfg)

        # 2. Window candidates
        candidates = create_candidates(
            peaks=fused.peaks,
            fused_scores=fused,
            vod=vod,
            config=cfg,
            silence_points=silence_points,
            words=words,
        )

        # 3. Evaluate against ground truth
        metrics = evaluate(candidates, labels, k=k)

        # 4. Compute optimization score
        score = compute_optimization_score(metrics, precision_floor=precision_floor)
        trials.append(
            TrialResult(
                trial_index=idx,
                config=cfg,
                metrics=metrics,
                score=score,
            )
        )

    # Sort trials by score descending (tie-break by trial_index ascending)
    trials.sort(key=lambda t: (t.score, -t.trial_index), reverse=True)
    best_trial = trials[0]
    best_config = best_trial.config

    target_path: Path | None = None
    if out_path:
        target_path = Path(out_path)
        best_config.save_yaml(target_path)

    return TuningResult(
        best_config=best_config,
        best_metrics=best_trial.metrics,
        best_trial=best_trial,
        trials=trials,
        out_path=target_path,
    )
