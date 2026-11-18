"""Evaluation and hyperparameter tuning for streamclip."""

from __future__ import annotations

from streamclip.eval.labels import GroundTruthLabel, load_labels, save_labels
from streamclip.eval.metrics import EvalMetrics, evaluate, is_hit
from streamclip.eval.tune import TrialResult, TuningResult, tune_hyperparameters

__all__ = [
    "EvalMetrics",
    "GroundTruthLabel",
    "TrialResult",
    "TuningResult",
    "evaluate",
    "is_hit",
    "load_labels",
    "save_labels",
    "tune_hyperparameters",
]
