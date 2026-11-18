"""Evaluation metrics calculation for highlight candidate detections."""

from __future__ import annotations

from dataclasses import dataclass, field

from streamclip.eval.labels import GroundTruthLabel
from streamclip.models import Candidate


def compute_overlap_ratio(
    candidate: Candidate,
    label: GroundTruthLabel,
) -> float:
    """Compute the maximum temporal overlap ratio between candidate and ground truth.

    Returns the maximum of IoU, intersection/label_duration, and intersection/candidate_duration.
    """
    inter_start = max(candidate.start, label.t_start)
    inter_end = min(candidate.end, label.t_end)
    inter = max(0.0, inter_end - inter_start)
    if inter <= 0.0:
        return 0.0

    union = max(candidate.end, label.t_end) - min(candidate.start, label.t_start)
    iou = inter / union if union > 0 else 0.0

    label_dur = label.duration
    cand_dur = candidate.duration

    ratio_label = inter / label_dur if label_dur > 0 else 0.0
    ratio_cand = inter / cand_dur if cand_dur > 0 else 0.0

    return max(iou, ratio_label, ratio_cand)


def is_hit(
    candidate: Candidate,
    label: GroundTruthLabel,
    min_overlap: float = 0.5,
) -> bool:
    """Determine whether a candidate detection hits a ground truth label.

    A detection hits a label if its window overlaps by 50% or more,
    or its peak is inside it.
    """
    # 1. Peak is inside ground truth window
    if label.contains(candidate.peak_t):
        return True

    # 2. Window overlaps by min_overlap (default 50%) or more
    overlap = compute_overlap_ratio(candidate, label)
    return overlap >= min_overlap


@dataclass
class EvalMetrics:
    """Evaluation summary metrics for highlight detection."""

    recall: float
    precision: float
    mean_time_offset: float
    k: int
    hits: int
    total_labels: int
    total_candidates: int
    offsets: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary."""
        return {
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "mean_time_offset": round(self.mean_time_offset, 4),
            "k": self.k,
            "hits": self.hits,
            "total_labels": self.total_labels,
            "total_candidates": self.total_candidates,
        }


def evaluate(
    candidates: list[Candidate],
    labels: list[GroundTruthLabel],
    k: int = 5,
    min_overlap: float = 0.5,
) -> EvalMetrics:
    """Evaluate candidate highlights against ground truth annotations.

    Candidates should be ranked by score descending.

    Args:
        candidates: Detected highlight candidates.
        labels: Ground truth highlight annotations.
        k: Cutoff rank for evaluation (Recall@K, Precision@K).
        min_overlap: Minimum temporal overlap ratio to qualify as a hit.

    Returns:
        EvalMetrics object containing recall@K, precision@K, and mean time offset.
    """
    if not labels:
        return EvalMetrics(
            recall=1.0 if not candidates else 0.0,
            precision=0.0,
            mean_time_offset=0.0,
            k=k,
            hits=0,
            total_labels=0,
            total_candidates=min(k, len(candidates)),
        )

    eval_cands = candidates[:k]
    total_cands = len(eval_cands)
    total_labels = len(labels)

    matched_label_indices: set[int] = set()
    offsets: list[float] = []

    for cand in eval_cands:
        # Find unmatched labels that this candidate hits
        candidates_for_hit: list[tuple[float, int]] = []
        for idx, lbl in enumerate(labels):
            if idx in matched_label_indices:
                continue
            if is_hit(cand, lbl, min_overlap=min_overlap):
                # Calculate time offset from candidate peak to label center
                offset = abs(cand.peak_t - lbl.center_t)
                candidates_for_hit.append((offset, idx))

        if candidates_for_hit:
            # Match to the label with smallest time offset
            candidates_for_hit.sort(key=lambda x: x[0])
            best_offset, best_idx = candidates_for_hit[0]
            matched_label_indices.add(best_idx)
            offsets.append(best_offset)

    hits = len(matched_label_indices)
    recall = hits / total_labels if total_labels > 0 else 0.0
    precision = hits / total_cands if total_cands > 0 else 0.0
    mean_offset = sum(offsets) / len(offsets) if offsets else 0.0

    return EvalMetrics(
        recall=recall,
        precision=precision,
        mean_time_offset=mean_offset,
        k=k,
        hits=hits,
        total_labels=total_labels,
        total_candidates=total_cands,
        offsets=offsets,
    )
