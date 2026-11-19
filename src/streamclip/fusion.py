"""Multimodal signal fusion and peak detection for highlight extraction."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from streamclip.config import Config
from streamclip.models import SignalTrack

# Dynamically import scipy modules to ensure strict mypy typing compatibility
_scipy_signal: Any = importlib.import_module("scipy.signal")
_scipy_ndimage: Any = importlib.import_module("scipy.ndimage")
find_peaks: Any = _scipy_signal.find_peaks


@dataclass
class Peak:
    """Detected highlight peak with timestamp, score, and per-track component breakdown."""

    peak_t: float
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class FusedSignal:
    """Container holding the fused timeseries signal, detected peaks, and per-track z-scores."""

    track: SignalTrack
    peaks: list[Peak]
    track_zscores: dict[str, list[float]] = field(default_factory=dict)


def _smooth_1d(
    arr: np.ndarray[Any, np.dtype[np.float64]],
    sigma: float = 2.0,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Apply 1D Gaussian smoothing using vectorized numpy convolution."""
    if sigma <= 0.0 or len(arr) == 0:
        return arr
    radius = math.ceil(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum > 0:
        kernel /= kernel_sum
    res = np.convolve(arr, kernel, mode="same")
    return np.asarray(res, dtype=np.float64)


def compute_rolling_zscore(
    values: np.ndarray[Any, np.dtype[np.float64]],
    window_size: int = 300,
    use_mad: bool = False,
    eps: float = 1e-6,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute rolling baseline z-score for a timeseries.

    Args:
        values: 1D numpy array of raw signal values.
        window_size: Size of the rolling window in seconds.
        use_mad: If True, uses rolling median and Median Absolute Deviation (MAD).
                 If False, uses rolling mean and standard deviation.
        eps: Small constant for numerical stability.

    Returns:
        1D numpy array of z-scores, clipped to >= 0.0.
    """
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    w = max(3, min(window_size, n))

    if use_mad:
        # Rolling median and MAD via scipy.ndimage.median_filter
        med = np.asarray(
            _scipy_ndimage.median_filter(values, size=w, mode="nearest"), dtype=np.float64
        )
        abs_diff = np.abs(values - med)
        mad = np.asarray(
            _scipy_ndimage.median_filter(abs_diff, size=w, mode="nearest"), dtype=np.float64
        )
        scale = np.maximum(eps, 1.4826 * mad)
        z = np.maximum(0.0, (values - med) / scale)
        return np.asarray(z, dtype=np.float64)

    # Vectorized rolling mean and standard deviation using scipy.ndimage.uniform_filter1d
    mean = np.asarray(
        _scipy_ndimage.uniform_filter1d(values, size=w, mode="nearest"), dtype=np.float64
    )
    mean_sq = np.asarray(
        _scipy_ndimage.uniform_filter1d(values**2, size=w, mode="nearest"), dtype=np.float64
    )
    var = np.maximum(0.0, mean_sq - mean**2)
    std = np.sqrt(var)
    scale = np.maximum(eps, std)
    z = np.maximum(0.0, (values - mean) / scale)
    return np.asarray(z, dtype=np.float64)


def fuse_signals(
    tracks: list[SignalTrack] | dict[str, SignalTrack],
    config: Config | None = None,
    use_mad: bool = False,
    window_size: int = 300,
    smooth_sigma: float = 2.0,
) -> FusedSignal:
    """Fuse multiple SignalTracks into a unified highlight signal and extract top peaks.

    Args:
        tracks: List or mapping of SignalTracks to fuse.
        config: Configuration containing weights, peak_height, min_peak_gap_s, and top_k.
        use_mad: Whether to use rolling median/MAD instead of mean/std for baseline.
        window_size: Window size in seconds for rolling baseline calculation.
        smooth_sigma: Sigma for Gaussian smoothing of the fused score.

    Returns:
        FusedSignal instance containing fused track, detected peaks, and component z-scores.
    """
    cfg = config or Config()

    track_list = list(tracks.values()) if isinstance(tracks, dict) else tracks
    if not track_list:
        empty_track = SignalTrack(name="fused", hz=1.0, values=[], meta={"empty": True})
        return FusedSignal(track=empty_track, peaks=[], track_zscores={})

    total_seconds = max(len(t.values) for t in track_list)
    fused_raw = np.zeros(total_seconds, dtype=np.float64)
    components_by_track: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    zscores_by_track: dict[str, list[float]] = {}

    weights_dict = cfg.fusion_weights.as_dict()

    # Compute raw confidence and effective weights per track
    active_weights: dict[str, float] = {}
    for track in track_list:
        if track.meta.get("empty", False) or track.meta.get("has_audio") is False:
            conf = 0.0
        else:
            conf = float(track.meta.get("activity_confidence", track.meta.get("confidence", 1.0)))
        base_w = float(weights_dict.get(track.name, 1.0))
        active_weights[track.name] = base_w * conf

    sum_effective = sum(active_weights.values())
    sum_base = sum(float(weights_dict.get(t.name, 1.0)) for t in track_list)
    # Renormalize active tracks when inactive tracks are absent to maintain dynamic range
    norm_factor = (sum_base / sum_effective) if sum_effective > 0 else 1.0

    for track in track_list:
        raw_vals = np.zeros(total_seconds, dtype=np.float64)
        raw_vals[: len(track.values)] = track.values

        effective_weight = active_weights[track.name] * norm_factor

        # Rolling baseline z-score
        z = compute_rolling_zscore(raw_vals, window_size=window_size, use_mad=use_mad)
        zscores_by_track[track.name] = [float(x) for x in z.tolist()]

        # Weighted component contribution
        component = effective_weight * z
        components_by_track[track.name] = component
        fused_raw += component

    # 5. Smooth the fused signal
    fused_smoothed = _smooth_1d(fused_raw, sigma=smooth_sigma)
    fused_smoothed = np.maximum(0.0, fused_smoothed)

    # 6. Peak detection via scipy find_peaks
    peak_indices, _ = find_peaks(
        fused_smoothed,
        height=cfg.peak_height,
        distance=cfg.min_peak_gap_s,
    )

    # 7. Collect Peak objects with per-track component breakdown
    peaks: list[Peak] = []
    for idx in peak_indices:
        t_peak = float(idx)
        score = float(fused_smoothed[idx])
        comp_breakdown = {name: float(comp[idx]) for name, comp in components_by_track.items()}
        peaks.append(Peak(peak_t=t_peak, score=score, components=comp_breakdown))

    # 8. Sort peaks by score descending (tie-break by earlier peak_t)
    peaks.sort(key=lambda p: (-p.score, p.peak_t))

    # Keep top_k
    peaks = peaks[: cfg.top_k]

    fused_track = SignalTrack(
        name="fused",
        hz=1.0,
        values=[float(x) for x in fused_smoothed.tolist()],
        meta={
            "track_count": len(track_list),
            "peak_count": len(peaks),
            "use_mad": use_mad,
        },
    )

    return FusedSignal(
        track=fused_track,
        peaks=peaks,
        track_zscores=zscores_by_track,
    )
