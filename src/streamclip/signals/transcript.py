"""Speech-to-text audio transcription wrapping faster-whisper."""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from streamclip.cache import DiskCache
from streamclip.interfaces import Transcriber
from streamclip.models import Candidate, VodMeta, Word

# Dynamic import for faster-whisper optional dependency
try:
    faster_whisper: Any = importlib.import_module("faster_whisper")
    HAS_FASTER_WHISPER = True
except (ImportError, ModuleNotFoundError):
    faster_whisper = None
    HAS_FASTER_WHISPER = False


def extract_audio_slice(
    vod: VodMeta,
    start_s: float,
    end_s: float,
    sample_rate: int = 16000,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Extract a 16 kHz mono PCM float32 audio slice using ffmpeg.

    Args:
        vod: VodMeta describing target media.
        start_s: Start offset in seconds.
        end_s: End offset in seconds.
        sample_rate: Target sampling rate (default 16 kHz).

    Returns:
        1D numpy float32 array normalized to [-1.0, 1.0].
    """
    duration = max(0.0, end_s - start_s)
    expected_samples = int(duration * sample_rate)

    if not vod.has_audio or duration <= 0.0:
        return np.zeros(expected_samples, dtype=np.float32)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return np.zeros(expected_samples, dtype=np.float32)

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(vod.path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
        audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    except (subprocess.SubprocessError, OSError):
        return np.zeros(expected_samples, dtype=np.float32)


def merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping time interval windows into disjoint sorted intervals."""
    if not windows:
        return []

    sorted_windows = sorted(windows, key=lambda w: (w[0], w[1]))
    merged: list[tuple[float, float]] = []

    for start, end in sorted_windows:
        if end <= start:
            continue
        if not merged:
            merged.append((start, end))
            continue

        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def generate_chunks(
    start_s: float,
    end_s: float,
    chunk_size_s: float = 300.0,
) -> list[tuple[float, float]]:
    """Partition a time interval [start_s, end_s] into contiguous chunks of chunk_size_s."""
    chunks: list[tuple[float, float]] = []
    cur = start_s
    while cur < end_s:
        next_cur = min(end_s, cur + chunk_size_s)
        chunks.append((round(cur, 3), round(next_cur, 3)))
        cur = next_cur
    return chunks


class WhisperTranscriber(Transcriber):
    """faster-whisper speech transcriber implementing the Transcriber protocol."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        model: Any | None = None,
        cache: DiskCache | None = None,
        chunk_size_s: float = 300.0,
        transcriber: Transcriber | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = model
        self.cache = cache or DiskCache()
        self.chunk_size_s = chunk_size_s
        self._delegate = transcriber

    def _get_model(self) -> Any:
        """Lazily load faster-whisper model or return injected model."""
        if self.model is not None:
            return self.model

        if not HAS_FASTER_WHISPER or faster_whisper is None:
            raise RuntimeError(
                "faster-whisper is not installed. Install streamclip with the "
                "[whisper] extra: pip install 'streamclip[whisper]'"
            )

        self.model = faster_whisper.WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self.model

    def transcribe(
        self,
        vod: VodMeta,
        start: float | None = None,
        end: float | None = None,
    ) -> list[Word]:
        """Transcribe speech in the given time window into timed words."""
        if not vod.has_audio:
            return []

        s = 0.0 if start is None else max(0.0, start)
        e = vod.duration_s if end is None else min(vod.duration_s, end)
        if e <= s:
            return []

        # If a delegate (e.g. FakeTranscriber) is injected, defer directly
        if self._delegate is not None:
            return self._delegate.transcribe(vod, start=s, end=e)

        # Extract audio waveform slice
        audio_data = extract_audio_slice(vod, start_s=s, end_s=e)
        if len(audio_data) == 0:
            return []

        model = self._get_model()
        segments, _ = model.transcribe(
            audio_data,
            word_timestamps=True,
            vad_filter=True,
        )

        words: list[Word] = []
        for segment in segments:
            if hasattr(segment, "words") and segment.words:
                for w in segment.words:
                    words.append(
                        Word(
                            start=round(float(w.start) + s, 3),
                            end=round(float(w.end) + s, 3),
                            text=str(w.word).strip(),
                        )
                    )
            elif hasattr(segment, "text") and segment.text:
                # Fallback: segment without word timestamps
                seg_text = str(segment.text).strip()
                tokens = seg_text.split()
                if tokens:
                    seg_s = float(segment.start) + s
                    seg_e = float(segment.end) + s
                    seg_dur = max(0.1, seg_e - seg_s)
                    token_dur = seg_dur / len(tokens)
                    for i, tok in enumerate(tokens):
                        words.append(
                            Word(
                                start=round(seg_s + i * token_dur, 3),
                                end=round(seg_s + (i + 1) * token_dur, 3),
                                text=tok,
                            )
                        )

        return words

    def transcribe_candidates(
        self,
        vod: VodMeta,
        candidates: list[Candidate] | None = None,
        margin_s: float = 2.0,
        full: bool = False,
    ) -> list[Word]:
        """Transcribe speech specifically covering candidate windows or full VOD."""
        return transcribe_vod(
            vod=vod,
            candidates=candidates,
            transcriber=self,
            margin_s=margin_s,
            full=full,
            cache=self.cache,
            chunk_size_s=self.chunk_size_s,
            model_size=self.model_size,
        )


def transcribe_vod(
    vod: VodMeta,
    candidates: list[Candidate] | None = None,
    transcriber: Transcriber | None = None,
    margin_s: float = 2.0,
    full: bool = False,
    cache: DiskCache | None = None,
    chunk_size_s: float = 300.0,
    model_size: str = "base",
) -> list[Word]:
    """Transcribe speech for VOD highlight candidates or full VOD with chunking and disk cache.

    By default, transcribes candidate windows expanded by margin_s.
    If full=True, transcribes the entire VOD.

    Args:
        vod: VodMeta describing target video and duration.
        candidates: Optional detected highlight candidates.
        transcriber: Transcriber implementation (defaults to WhisperTranscriber).
        margin_s: Extra margin in seconds added before/after each candidate window.
        full: If True, transcribes the entire VOD instead of candidate windows.
        cache: DiskCache instance for resumability.
        chunk_size_s: Maximum chunk size in seconds for long audio chunking.
        model_size: Whisper model identifier for cache keying.

    Returns:
        List of timed Word objects sorted by start timestamp.
    """
    if not vod.has_audio or vod.duration_s <= 0.0:
        return []

    # 1. Determine windows to transcribe
    raw_windows: list[tuple[float, float]] = []
    if full:
        raw_windows.append((0.0, vod.duration_s))
    elif candidates:
        for cand in candidates:
            w_start = max(0.0, cand.start - margin_s)
            w_end = min(vod.duration_s, cand.end + margin_s)
            if w_end > w_start:
                raw_windows.append((w_start, w_end))
    else:
        # Candidate-only mode with no candidates provided -> return empty
        return []

    merged_windows = merge_windows(raw_windows)
    if not merged_windows:
        return []

    # 2. Partition into chunks
    all_chunks: list[tuple[float, float]] = []
    for w_start, w_end in merged_windows:
        chunks = generate_chunks(w_start, w_end, chunk_size_s=chunk_size_s)
        all_chunks.extend(chunks)

    # 3. Setup cache and transcriber
    disk_cache = cache or DiskCache()
    worker = transcriber or WhisperTranscriber(
        model_size=model_size,
        cache=disk_cache,
        chunk_size_s=chunk_size_s,
    )

    all_words: list[Word] = []

    # 4. Process each chunk (resumable via cache)
    is_cached_file = Path(vod.path).is_file()

    for c_start, c_end in all_chunks:
        config_slice = {
            "start": c_start,
            "end": c_end,
            "model": model_size,
        }
        cached = disk_cache.get(vod.path, config_slice) if is_cached_file else None
        if cached is not None and isinstance(cached, list):
            chunk_words = [Word.model_validate(w) for w in cached]
        else:
            chunk_words = worker.transcribe(vod, start=c_start, end=c_end)
            if is_cached_file:
                disk_cache.set(vod.path, config_slice, chunk_words)

        all_words.extend(chunk_words)

    # 5. Sort and deduplicate any overlapping words at chunk boundaries
    all_words.sort(key=lambda w: (w.start, w.end))
    unique_words: list[Word] = []
    seen: set[tuple[float, float, str]] = set()
    for w in all_words:
        key = (round(w.start, 2), round(w.end, 2), w.text.strip().lower())
        if key not in seen:
            seen.add(key)
            unique_words.append(w)

    return unique_words
