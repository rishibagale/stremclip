"""Candidate window generation, boundary snapping, and overlap merging."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from streamclip.config import Config
from streamclip.fusion import FusedSignal, Peak
from streamclip.models import Candidate, SignalTrack, VodMeta, Word


def build_boundary_points(
    silence_points: list[float] | None = None,
    words: list[Word] | None = None,
) -> list[float]:
    """Collect and sort unique boundary points from silence points and word boundaries."""
    points: set[float] = set()
    if silence_points:
        for p in silence_points:
            points.add(round(float(p), 3))
    if words:
        for w in words:
            points.add(round(float(w.start), 3))
            points.add(round(float(w.end), 3))
    return sorted(points)


def snap_to_boundary(
    t: float,
    boundaries: list[float],
    tolerance_s: float = 3.0,
) -> float:
    """Snap a timestamp to the closest boundary point within tolerance_s.

    If no boundary point is within tolerance_s, returns t unchanged.
    """
    if not boundaries or tolerance_s <= 0.0:
        return t

    best_p = t
    min_dist = float("inf")
    for p in boundaries:
        dist = abs(p - t)
        if dist < min_dist:
            min_dist = dist
            best_p = p

    if min_dist <= tolerance_s:
        return best_p
    return t


def find_decay_end(
    peak_t: float,
    fused_scores: np.ndarray[Any, np.dtype[np.float64]],
    peak_score: float,
    baseline_ratio: float = 0.25,
    post_roll_s: float = 4.0,
    max_search_s: float = 60.0,
) -> float:
    """Find the point where fused score decays back near baseline, plus post_roll."""
    n = len(fused_scores)
    if n == 0:
        return peak_t + post_roll_s

    threshold = max(0.5, baseline_ratio * peak_score)
    start_idx = max(0, min(n - 1, math.floor(peak_t) + 1))
    max_idx = min(n, math.ceil(peak_t + max_search_s))

    decay_idx = max_idx
    for idx in range(start_idx, max_idx):
        if fused_scores[idx] <= threshold:
            decay_idx = idx
            break

    return float(decay_idx) + post_roll_s


def clamp_window(
    start: float,
    end: float,
    peak_t: float,
    clip_min_s: float,
    clip_max_s: float,
    vod_duration_s: float,
) -> tuple[float, float]:
    """Clamp candidate window duration to [clip_min_s, clip_max_s] and keep within VOD bounds."""
    # 1. Basic VOD bounds
    start = max(0.0, min(start, vod_duration_s))
    end = max(start, min(end, vod_duration_s))

    # 2. Minimum duration clamping
    if end - start < clip_min_s:
        end = min(vod_duration_s, start + clip_min_s)
        if end - start < clip_min_s:
            start = max(0.0, end - clip_min_s)

    # 3. Maximum duration clamping
    if end - start > clip_max_s:
        # Keep peak_t inside window with a small post-margin
        if peak_t - start > clip_max_s - 4.0:
            start = max(0.0, peak_t - clip_max_s + 4.0)
            end = min(vod_duration_s, start + clip_max_s)
        else:
            end = min(vod_duration_s, start + clip_max_s)

    # 4. Final safety clamp to VOD bounds
    start = max(0.0, min(start, vod_duration_s))
    end = max(start, min(end, vod_duration_s))
    return round(start, 3), round(end, 3)


def merge_overlapping_candidates(
    candidates: list[Candidate],
    clip_min_s: float = 15.0,
    clip_max_s: float = 60.0,
    vod_duration_s: float | None = None,
) -> list[Candidate]:
    """Merge overlapping candidates into unified windows, keeping the higher score."""
    if not candidates:
        return []

    # Sort candidates by start ascending, tie-breaking by higher score
    sorted_cands = sorted(candidates, key=lambda c: (c.start, -c.score, c.peak_t))
    merged: list[Candidate] = []

    vod_dur = vod_duration_s if vod_duration_s is not None else float("inf")

    for cand in sorted_cands:
        if not merged:
            merged.append(cand)
            continue

        prev = merged[-1]
        # Overlap condition: start of current candidate <= end of previous candidate
        if cand.start <= prev.end:
            winner = cand if cand.score > prev.score else prev
            raw_start = min(prev.start, cand.start)
            raw_end = max(prev.end, cand.end)

            if vod_dur < float("inf"):
                s, e = clamp_window(
                    raw_start,
                    raw_end,
                    winner.peak_t,
                    clip_min_s,
                    clip_max_s,
                    vod_dur,
                )
            else:
                s = raw_start
                e = raw_end
                if e - s > clip_max_s:
                    if winner.peak_t - s > clip_max_s - 4.0:
                        s = max(0.0, winner.peak_t - clip_max_s + 4.0)
                        e = s + clip_max_s
                    else:
                        e = s + clip_max_s
                s, e = round(s, 3), round(e, 3)

            merged[-1] = Candidate(
                peak_t=winner.peak_t,
                start=s,
                end=e,
                score=winner.score,
                components=dict(winner.components),
            )
        else:
            merged.append(cand)

    return merged


def create_candidates(
    peaks: list[Peak],
    fused_scores: list[float] | np.ndarray[Any, Any] | SignalTrack | FusedSignal,
    vod: VodMeta | float,
    config: Config | None = None,
    silence_points: list[float] | None = None,
    words: list[Word] | None = None,
    snap_tolerance_s: float = 3.0,
    baseline_ratio: float = 0.25,
) -> list[Candidate]:
    """Turn detected peaks into validated Candidate highlight windows.

    Args:
        peaks: List of Peak instances.
        fused_scores: Fused signal timeseries values or container.
        vod: VodMeta instance or duration in seconds.
        config: Configuration containing roll and clip length constraints.
        silence_points: Optional silence timestamps for start snapping.
        words: Optional timed words for start snapping to word boundaries.
        snap_tolerance_s: Maximum window around pre-roll start to snap.
        baseline_ratio: Ratio of peak score used as decay threshold.

    Returns:
        List of Candidate windows with overlaps merged and bounds clamped.
    """
    cfg = config or Config()
    vod_dur = vod.duration_s if isinstance(vod, VodMeta) else float(vod)

    # Extract score array
    scores_arr: np.ndarray[Any, np.dtype[np.float64]]
    if isinstance(fused_scores, FusedSignal):
        scores_arr = np.asarray(fused_scores.track.values, dtype=np.float64)
        if not peaks and fused_scores.peaks:
            peaks = fused_scores.peaks
    elif isinstance(fused_scores, SignalTrack):
        scores_arr = np.asarray(fused_scores.values, dtype=np.float64)
    elif isinstance(fused_scores, np.ndarray):
        scores_arr = fused_scores.astype(np.float64)
    else:
        scores_arr = np.asarray(fused_scores, dtype=np.float64)

    if not peaks:
        return []

    boundaries = build_boundary_points(silence_points, words)
    raw_cands: list[Candidate] = []

    for peak in peaks:
        # 1. Pre-roll before the peak
        raw_start = max(0.0, peak.peak_t - cfg.pre_roll_s)

        # 2. Snap start to boundary (silence point or word boundary)
        snapped_start = snap_to_boundary(raw_start, boundaries, tolerance_s=snap_tolerance_s)
        if snapped_start > peak.peak_t:
            snapped_start = raw_start
        snapped_start = max(0.0, snapped_start)

        # 3. Decay end: when fused score returns near baseline plus post-roll
        decay_end = find_decay_end(
            peak.peak_t,
            scores_arr,
            peak.score,
            baseline_ratio=baseline_ratio,
            post_roll_s=cfg.post_roll_s,
        )

        # 4. Clamp duration and VOD bounds
        s, e = clamp_window(
            snapped_start,
            decay_end,
            peak.peak_t,
            cfg.clip_min_s,
            cfg.clip_max_s,
            vod_dur,
        )

        raw_cands.append(
            Candidate(
                peak_t=peak.peak_t,
                start=s,
                end=e,
                score=peak.score,
                components=dict(peak.components),
            )
        )

    # 5. Merge overlaps (keeping higher score)
    merged = merge_overlapping_candidates(
        raw_cands,
        clip_min_s=cfg.clip_min_s,
        clip_max_s=cfg.clip_max_s,
        vod_duration_s=vod_dur,
    )

    # 6. Sort candidates by score descending
    merged.sort(key=lambda c: (-c.score, c.start))
    return merged


def refine_candidate_with_transcript(
    cand: Candidate,
    words: list[Word],
    max_expansion_s: float = 3.5,
    min_clip_s: float = 15.0,
    max_clip_s: float = 60.0,
    snap_window_s: float = 3.0,
) -> Candidate:
    """Refine candidate start and end to natural sentence/speech boundaries using Whisper words.

    Prevents mid-sentence cutoffs and clipped words by aligning boundaries to pauses
    and punctuation marks (inspired by SupoClip's boundary alignment).
    """
    if not words:
        return cand

    refined_start = cand.start
    refined_end = cand.end

    # 1. Refine start boundary
    expansion = max(max_expansion_s, snap_window_s)
    nearby_start_words = [
        w for w in words if (cand.start - expansion) <= w.start <= (cand.start + 1.5)
    ]
    if nearby_start_words:
        valid_starts: list[tuple[float, float]] = []
        for i, w in enumerate(nearby_start_words):
            has_pause_before = False
            if i > 0:
                has_pause_before = (w.start - nearby_start_words[i - 1].end) >= 0.35
            else:
                all_idx = words.index(w) if w in words else -1
                if all_idx > 0:
                    has_pause_before = (w.start - words[all_idx - 1].end) >= 0.35

            is_capitalized = bool(w.text and w.text[0].isupper())
            if has_pause_before or is_capitalized:
                if w.start <= cand.peak_t:
                    valid_starts.append((abs(w.start - cand.start), w.start))

        if valid_starts:
            valid_starts.sort(key=lambda item: item[0])
            refined_start = max(0.0, valid_starts[0][1])

    # 2. Refine end boundary
    nearby_end_words = [w for w in words if (cand.end - 1.5) <= w.end <= (cand.end + expansion)]
    if nearby_end_words:
        valid_ends: list[tuple[float, float]] = []
        for i, w in enumerate(nearby_end_words):
            ends_with_punct = any(w.text.endswith(p) for p in (".", "!", "?", "...", "…"))
            has_pause_after = False
            if i < len(nearby_end_words) - 1:
                has_pause_after = (nearby_end_words[i + 1].start - w.end) >= 0.35
            else:
                all_idx = words.index(w) if w in words else -1
                if 0 <= all_idx < len(words) - 1:
                    has_pause_after = (words[all_idx + 1].start - w.end) >= 0.35

            if ends_with_punct or has_pause_after:
                if w.end >= cand.peak_t and (w.end - refined_start) <= max_clip_s:
                    valid_ends.append((abs(w.end - cand.end), w.end))

        if valid_ends:
            valid_ends.sort(key=lambda item: item[0])
            refined_end = valid_ends[0][1]

    # Enforce min_clip_s: do not let refinement shrink below min(cand.duration, min_clip_s)
    target_min_dur = min(cand.duration, min_clip_s)
    if (refined_end - refined_start) < target_min_dur:
        refined_start = cand.start
        refined_end = cand.end

    return Candidate(
        peak_t=cand.peak_t,
        start=round(refined_start, 2),
        end=round(refined_end, 2),
        score=cand.score,
        components=dict(cand.components),
    )


def refine_candidates_with_transcript(
    candidates: list[Candidate],
    words: list[Word],
    max_expansion_s: float = 3.5,
    max_clip_s: float = 60.0,
) -> list[Candidate]:
    """Refine a list of candidates against the word transcript."""
    return [
        refine_candidate_with_transcript(
            c, words, max_expansion_s=max_expansion_s, max_clip_s=max_clip_s
        )
        for c in candidates
    ]
