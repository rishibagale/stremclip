"""Tests for faster-whisper transcription wrapper, chunking, caching, and candidate modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from streamclip.cache import DiskCache
from streamclip.interfaces import Transcriber
from streamclip.models import Candidate, VodMeta, Word
from streamclip.signals.transcript import (
    WhisperTranscriber,
    generate_chunks,
    merge_windows,
    transcribe_vod,
)
from tests.fakes import FakeTranscriber


@dataclass
class MockWord:
    start: float
    end: float
    word: str


@dataclass
class MockSegment:
    start: float
    end: float
    text: str
    words: list[MockWord]


class MockWhisperModel:
    """Mock faster-whisper model returning synthetic segments and words without network/weights."""

    def __init__(self, segments: list[MockSegment] | None = None) -> None:
        self.segments = segments or [
            MockSegment(
                start=0.5,
                end=2.0,
                text="Incredible play",
                words=[
                    MockWord(start=0.5, end=1.2, word=" Incredible "),
                    MockWord(start=1.3, end=2.0, word="play"),
                ],
            )
        ]
        self.call_count = 0

    def transcribe(
        self,
        audio: Any,
        word_timestamps: bool = True,
        **kwargs: Any,
    ) -> tuple[list[MockSegment], dict[str, Any]]:
        self.call_count += 1
        return self.segments, {"language": "en"}


class CountingFakeTranscriber(Transcriber):
    """Fake transcriber tracking how many times transcribe() is executed."""

    def __init__(self) -> None:
        self.call_count = 0

    def transcribe(
        self,
        vod: VodMeta,
        start: float | None = None,
        end: float | None = None,
    ) -> list[Word]:
        self.call_count += 1
        s = start if start is not None else 0.0
        return [
            Word(start=s + 1.0, end=s + 1.5, text="test"),
            Word(start=s + 1.6, end=s + 2.2, text="speech"),
        ]


def test_whisper_transcriber_satisfies_protocol() -> None:
    """Verify that WhisperTranscriber satisfies the Transcriber Protocol."""
    transcriber = WhisperTranscriber()
    assert isinstance(transcriber, Transcriber)


def test_merge_windows() -> None:
    """Test merging overlapping and contiguous windows."""
    # Disjoint windows
    w1 = [(10.0, 20.0), (30.0, 40.0)]
    assert merge_windows(w1) == [(10.0, 20.0), (30.0, 40.0)]

    # Overlapping windows
    w2 = [(10.0, 25.0), (20.0, 35.0), (50.0, 60.0)]
    assert merge_windows(w2) == [(10.0, 35.0), (50.0, 60.0)]

    # Nested windows
    w3 = [(5.0, 50.0), (10.0, 20.0)]
    assert merge_windows(w3) == [(5.0, 50.0)]

    # Empty and degenerate
    assert merge_windows([]) == []
    assert merge_windows([(10.0, 10.0), (20.0, 15.0)]) == []


def test_generate_chunks() -> None:
    """Test chunking of intervals."""
    # Small interval (fits in one chunk)
    chunks_small = generate_chunks(10.0, 50.0, chunk_size_s=300.0)
    assert chunks_small == [(10.0, 50.0)]

    # Multi-chunk interval (700s split into 300s chunks)
    chunks_large = generate_chunks(0.0, 750.0, chunk_size_s=300.0)
    assert chunks_large == [(0.0, 300.0), (300.0, 600.0), (600.0, 750.0)]


def test_transcribe_vod_candidate_mode_with_margin(tmp_path: Path) -> None:
    """Test candidate-only transcription with extra margin."""
    mock_file = tmp_path / "mock.mp4"
    mock_file.write_bytes(b"dummy video data")
    vod = VodMeta(
        path=mock_file,
        duration_s=120.0,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=True,
    )
    # Candidate window [10.0, 15.0]
    candidate = Candidate(peak_t=12.0, start=10.0, end=15.0, score=8.0)
    fake = FakeTranscriber(
        words=[
            Word(start=5.0, end=6.0, text="ignored_early"),
            Word(start=10.0, end=10.5, text="Oh"),
            Word(start=10.6, end=11.2, text="my"),
            Word(start=11.3, end=12.0, text="god!"),
            Word(start=90.0, end=91.0, text="ignored_late"),
        ]
    )

    cache = DiskCache(tmp_path / "cache")
    # Transcribe candidate with margin_s=2.0 -> window [8.0, 17.0]
    words = transcribe_vod(
        vod=vod,
        candidates=[candidate],
        transcriber=fake,
        margin_s=2.0,
        full=False,
        cache=cache,
    )

    # Only words within [8.0, 17.0] should be returned
    assert len(words) == 3
    assert [w.text for w in words] == ["Oh", "my", "god!"]


def test_transcribe_vod_candidate_window_merging(tmp_path: Path) -> None:
    """Test that overlapping candidate windows are merged before transcription."""
    mock_file = tmp_path / "mock.mp4"
    mock_file.write_bytes(b"dummy video data")
    vod = VodMeta(
        path=mock_file,
        duration_s=200.0,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=True,
    )
    # Two overlapping candidates: [10, 20] and [18, 30]
    c1 = Candidate(peak_t=15.0, start=10.0, end=20.0, score=7.0)
    c2 = Candidate(peak_t=25.0, start=18.0, end=30.0, score=6.0)

    tracker = CountingFakeTranscriber()
    cache = DiskCache(tmp_path / "cache")

    transcribe_vod(
        vod=vod,
        candidates=[c1, c2],
        transcriber=tracker,
        margin_s=2.0,
        full=False,
        cache=cache,
        chunk_size_s=300.0,
    )

    # Should merge [8, 22] and [16, 32] into a single window [8, 32], so transcribe() runs once
    assert tracker.call_count == 1


def test_transcribe_vod_full_mode(tmp_path: Path) -> None:
    """Test full VOD transcription mode."""
    mock_file = tmp_path / "mock.mp4"
    mock_file.write_bytes(b"dummy video data")
    vod = VodMeta(
        path=mock_file,
        duration_s=600.0,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=True,
    )
    tracker = CountingFakeTranscriber()
    cache = DiskCache(tmp_path / "cache")

    words = transcribe_vod(
        vod=vod,
        candidates=None,
        transcriber=tracker,
        full=True,
        cache=cache,
        chunk_size_s=300.0,
    )

    # 600s with 300s chunk size -> 2 chunks -> 2 calls to transcribe
    assert tracker.call_count == 2
    assert len(words) > 0


def test_transcribe_resumable_via_cache(tmp_path: Path) -> None:
    """Test that cached chunks are skipped on subsequent runs."""
    vod_file = tmp_path / "test_vod.mp4"
    vod_file.write_bytes(b"mock video data for hashing")
    vod = VodMeta(
        path=vod_file,
        duration_s=100.0,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=True,
    )

    tracker = CountingFakeTranscriber()
    cache = DiskCache(tmp_path / "cache")
    cand = Candidate(peak_t=20.0, start=10.0, end=30.0, score=5.0)

    # First run: executes transcription and writes to cache
    words1 = transcribe_vod(vod, candidates=[cand], transcriber=tracker, cache=cache)
    assert tracker.call_count == 1
    assert len(words1) == 2

    # Second run: hits cache, does not increment call_count
    words2 = transcribe_vod(vod, candidates=[cand], transcriber=tracker, cache=cache)
    assert tracker.call_count == 1  # Still 1!
    assert len(words2) == 2
    assert words1 == words2


def test_transcribe_empty_and_no_audio(tmp_path: Path) -> None:
    """Test edge cases: no audio or empty candidates in non-full mode."""
    vod_no_audio = VodMeta(
        path=tmp_path / "mock.mp4",
        duration_s=100.0,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=False,
    )
    fake = FakeTranscriber()
    assert transcribe_vod(vod_no_audio, transcriber=fake, full=True) == []

    vod_with_audio = VodMeta(
        path=tmp_path / "mock.mp4",
        duration_s=100.0,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=True,
    )
    # candidates=None and full=False -> returns empty
    assert transcribe_vod(vod_with_audio, candidates=None, transcriber=fake, full=False) == []
    assert transcribe_vod(vod_with_audio, candidates=[], transcriber=fake, full=False) == []


def test_mock_whisper_model_transcription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test transcription using an injected mock faster-whisper model."""
    mock_file = tmp_path / "mock.mp4"
    mock_file.write_bytes(b"dummy video data")
    vod = VodMeta(
        path=mock_file,
        duration_s=60.0,
        fps=30.0,
        width=1920,
        height=1080,
        has_audio=True,
    )
    monkeypatch.setattr(
        "streamclip.signals.transcript.extract_audio_slice",
        lambda vod, start_s, end_s: np.ones(16000, dtype=np.float32) * 0.1,
    )
    mock_model = MockWhisperModel()
    cache = DiskCache(tmp_path / "cache")
    transcriber = WhisperTranscriber(model=mock_model, cache=cache)

    cand = Candidate(peak_t=10.0, start=5.0, end=15.0, score=9.0)
    words = transcriber.transcribe_candidates(vod, candidates=[cand], margin_s=1.0)

    assert mock_model.call_count == 1
    assert len(words) == 2
    assert words[0].text == "Incredible"
    assert words[1].text == "play"
    # Offset checked: window start was max(0, 5 - 1) = 4.0
    # Words in mock model were start=0.5 and start=1.3 -> 4.5 and 5.3
    assert words[0].start == 4.5
    assert words[1].start == 5.3


def test_missing_whisper_error(tmp_path: Path) -> None:
    """Test that missing faster-whisper dependency raises helpful RuntimeError."""
    transcriber = WhisperTranscriber()
    # Force HAS_FASTER_WHISPER check
    from streamclip.signals import transcript

    orig_has = transcript.HAS_FASTER_WHISPER
    try:
        transcript.HAS_FASTER_WHISPER = False
        with pytest.raises(RuntimeError, match=r"\[whisper\]"):
            transcriber._get_model()
    finally:
        transcript.HAS_FASTER_WHISPER = orig_has
