"""Audio signal extraction for streamclip."""

from __future__ import annotations

import importlib
import math
import shutil
import subprocess
from typing import Any

import numpy as np

from streamclip.config import Config
from streamclip.interfaces import SignalExtractor
from streamclip.models import SignalTrack, VodMeta

# Check librosa availability dynamically
try:
    librosa: Any = importlib.import_module("librosa")
    HAS_LIBROSA = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    librosa = None
    HAS_LIBROSA = False

# Check torch / torchaudio availability for YAMNet
try:
    torch: Any = importlib.import_module("torch")
    torchaudio: Any = importlib.import_module("torchaudio")
    HAS_AUDIO_ML = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    torch = None
    torchaudio = None
    HAS_AUDIO_ML = False


def extract_audio_pcm(
    vod: VodMeta,
    sample_rate: int = 16000,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Extract mono audio from VOD at the specified sample rate using ffmpeg.

    Args:
        vod: VodMeta describing target media.
        sample_rate: Audio sampling frequency in Hz (default 16 kHz).

    Returns:
        1D numpy float32 array containing raw audio PCM waveform.
    """
    total_seconds = math.ceil(vod.duration_s)
    expected_samples = total_seconds * sample_rate

    if not vod.has_audio:
        return np.zeros(expected_samples, dtype=np.float32)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return np.zeros(expected_samples, dtype=np.float32)

    cmd = [
        ffmpeg_bin,
        "-v",
        "quiet",
        "-i",
        str(vod.path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not proc.stdout:
            return np.zeros(expected_samples, dtype=np.float32)
        raw_pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    except (OSError, ValueError):
        return np.zeros(expected_samples, dtype=np.float32)

    # Pad or truncate to expected length
    if len(raw_pcm) < expected_samples:
        padded = np.zeros(expected_samples, dtype=np.float32)
        padded[: len(raw_pcm)] = raw_pcm
        return padded
    return raw_pcm[:expected_samples]


def compute_per_second_rms(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int = 16000,
    total_seconds: int = 600,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute root-mean-square (RMS) energy for each 1-second interval."""
    rms_vals = np.zeros(total_seconds, dtype=np.float64)

    if len(audio) == 0:
        return rms_vals

    if HAS_LIBROSA:
        # Compute RMS via librosa with 1-second frames
        hop = sample_rate
        raw_rms = librosa.feature.rms(
            y=audio,
            frame_length=sample_rate,
            hop_length=hop,
            center=False,
        )[0]
        count = min(len(raw_rms), total_seconds)
        rms_vals[:count] = raw_rms[:count]
    else:
        # Fallback numpy per-second calculation
        for s in range(total_seconds):
            start = s * sample_rate
            end = min(len(audio), (s + 1) * sample_rate)
            if start < len(audio):
                chunk = audio[start:end]
                if len(chunk) > 0:
                    rms_vals[s] = float(np.sqrt(np.mean(chunk**2)))

    return rms_vals


def compute_per_second_spectral_flux(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int = 16000,
    total_seconds: int = 600,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute spectral flux (onset strength / energy dynamics) per second."""
    flux_vals = np.zeros(total_seconds, dtype=np.float64)

    if len(audio) == 0:
        return flux_vals

    if HAS_LIBROSA:
        hop_length = 4000
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=sample_rate,
            hop_length=hop_length,
        )
        frames_per_sec = sample_rate / hop_length

        for s in range(total_seconds):
            f_start = int(s * frames_per_sec)
            f_end = min(len(onset_env), int((s + 1) * frames_per_sec))
            if f_start < len(onset_env) and f_end > f_start:
                flux_vals[s] = float(np.mean(onset_env[f_start:f_end]))
    else:
        # Fallback: compute magnitude spectrum frame differences
        window_size = 512
        hop = 4000
        frames = len(audio) // hop
        prev_mag: np.ndarray[Any, np.dtype[np.float32]] | None = None
        per_frame_flux = np.zeros(frames, dtype=np.float64)

        for i in range(frames):
            chunk = audio[i * hop : i * hop + window_size]
            if len(chunk) == window_size:
                mag = np.abs(np.fft.rfft(chunk))
                if prev_mag is not None:
                    per_frame_flux[i] = float(np.sum(np.maximum(0.0, mag - prev_mag)))
                prev_mag = mag

        frames_per_sec = sample_rate / hop
        for s in range(total_seconds):
            f_start = int(s * frames_per_sec)
            f_end = min(len(per_frame_flux), int((s + 1) * frames_per_sec))
            if f_start < len(per_frame_flux) and f_end > f_start:
                flux_vals[s] = float(np.mean(per_frame_flux[f_start:f_end]))

    return flux_vals


def classify_yamnet(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int = 16000,
    model: Any | None = None,
) -> dict[str, float]:
    """Optional scream/laughter audio event classifier behind [audio-ml].

    Falls back gracefully when YAMNet model weights or torchaudio are not present.
    """
    if not HAS_AUDIO_ML or len(audio) == 0:
        return {"scream": 0.0, "laughter": 0.0, "available": False}

    if model is not None:
        try:
            waveform = torch.from_numpy(audio).unsqueeze(0)
            with torch.no_grad():
                probs = model(waveform)
            return {
                "scream": (
                    float(torch.max(probs[0, :, 68]).item()) if probs.shape[-1] > 68 else 0.0
                ),
                "laughter": (
                    float(torch.max(probs[0, :, 16]).item()) if probs.shape[-1] > 16 else 0.0
                ),
                "available": True,
            }
        except Exception:
            pass

    return {"scream": 0.0, "laughter": 0.0, "available": False}


def extract_audio_signal(
    vod: VodMeta,
    config: Config | None = None,
) -> SignalTrack:
    """Extract a 1 Hz raw unnormalized SignalTrack from video audio.

    Computes per-second RMS and spectral-flux energy using librosa.

    Args:
        vod: VodMeta describing target VOD.
        config: Optional Config instance.

    Returns:
        SignalTrack named 'audio' of length ceil(vod.duration_s).
    """
    total_seconds = math.ceil(vod.duration_s)

    if not vod.has_audio:
        return SignalTrack(
            name="audio",
            hz=1.0,
            values=[0.0] * total_seconds,
            meta={
                "has_audio": False,
                "rms_mean": 0.0,
                "yamnet_available": False,
            },
        )

    # 1. Extract 16 kHz mono audio via ffmpeg
    sample_rate = 16000
    pcm = extract_audio_pcm(vod, sample_rate=sample_rate)

    # 2. Compute per-second features
    rms_vals = compute_per_second_rms(pcm, sample_rate=sample_rate, total_seconds=total_seconds)
    flux_vals = compute_per_second_spectral_flux(
        pcm, sample_rate=sample_rate, total_seconds=total_seconds
    )

    # 3. Combine raw unnormalized RMS and spectral flux
    combined_signal = rms_vals + (0.5 * flux_vals)

    # 4. Optional YAMNet classification
    yamnet_res = classify_yamnet(pcm[: sample_rate * 30], sample_rate=sample_rate)

    return SignalTrack(
        name="audio",
        hz=1.0,
        values=[float(x) for x in combined_signal.tolist()],
        meta={
            "has_audio": True,
            "rms_mean": float(np.mean(rms_vals)),
            "rms_peak": float(np.max(rms_vals)),
            "yamnet_available": yamnet_res.get("available", False),
            "scream_score": yamnet_res.get("scream", 0.0),
            "laughter_score": yamnet_res.get("laughter", 0.0),
        },
    )


def silence_points(
    vod: VodMeta,
    threshold_db: float = -35.0,
    min_duration_s: float = 1.0,
) -> list[float]:
    """Identify quiet or silent timestamps in the VOD suitable for clip boundaries.

    Args:
        vod: VodMeta describing target media.
        threshold_db: Audio level threshold in decibels below which audio is considered quiet.
        min_duration_s: Minimum duration required for silence consideration.

    Returns:
        List of timestamp points in seconds where audio is quiet.
    """
    total_seconds = math.ceil(vod.duration_s)

    if not vod.has_audio:
        return [float(s) for s in range(total_seconds)]

    sample_rate = 16000
    pcm = extract_audio_pcm(vod, sample_rate=sample_rate)
    rms_vals = compute_per_second_rms(pcm, sample_rate=sample_rate, total_seconds=total_seconds)

    # Convert to dB relative to 1.0
    safe_rms = np.maximum(rms_vals, 1e-6)
    db_vals = 20.0 * np.log10(safe_rms)

    # Detect quiet seconds
    quiet_seconds: list[float] = []
    for s in range(total_seconds):
        if db_vals[s] <= threshold_db:
            quiet_seconds.append(float(s))

    # If no strict threshold matched, find local minima
    if not quiet_seconds:
        mean_db = float(np.mean(db_vals))
        quiet_seconds = [float(s) for s in range(total_seconds) if db_vals[s] < mean_db]

    return quiet_seconds


class AudioSignalExtractor(SignalExtractor):
    """SignalExtractor protocol implementation for VOD audio track."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    @property
    def name(self) -> str:
        return "audio"

    def extract(self, vod: VodMeta, **kwargs: Any) -> SignalTrack:
        cfg: Config = kwargs.get("config") or self.config
        return extract_audio_signal(vod, config=cfg)
