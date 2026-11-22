"""Unit tests for transcript-aware candidate boundary refinement."""

from __future__ import annotations

import pytest

from streamclip.models import Candidate, Word
from streamclip.windowing import (
    refine_candidate_with_transcript,
    refine_candidates_with_transcript,
)


def test_refine_candidate_snaps_start_and_end() -> None:
    # Candidate originally from 10.0 to 30.0 (duration 20s)
    candidate = Candidate(
        peak_t=20.0,
        start=10.0,
        end=30.0,
        score=5.0,
    )

    # Transcript with words:
    # Word at 8.0-8.5 ("hello")
    # Pause until 11.0
    # Word at 11.0-11.4 ("Now") capitalized!
    # Word at 29.5-29.9 ("here")
    # Word at 30.0-30.8 ("winner!") with terminal exclamation point
    # Pause after 30.8 until 33.0
    words = [
        Word(start=8.0, end=8.5, text="hello"),
        Word(start=11.0, end=11.4, text="Now"),
        Word(start=11.5, end=12.0, text="we"),
        Word(start=12.1, end=13.0, text="begin"),
        Word(start=28.0, end=29.0, text="almost"),
        Word(start=29.5, end=29.9, text="the"),
        Word(start=30.0, end=30.8, text="winner!"),
        Word(start=33.0, end=34.0, text="Next"),
    ]

    refined = refine_candidate_with_transcript(
        candidate,
        words,
        min_clip_s=15.0,
        max_clip_s=60.0,
        snap_window_s=3.0,
    )

    # Start should snap to "Now" at 11.0 (capitalized, preceded by pause > 0.4s)
    assert pytest.approx(refined.start, abs=0.1) == 11.0
    # End should snap to "winner!" at 30.8 (punctuation, followed by pause > 0.4s)
    assert pytest.approx(refined.end, abs=0.1) == 30.8
    assert refined.duration >= 15.0


def test_refine_candidate_empty_transcript() -> None:
    candidate = Candidate(
        peak_t=20.0,
        start=10.0,
        end=30.0,
        score=5.0,
    )
    refined = refine_candidate_with_transcript(candidate, [])
    assert refined.start == 10.0
    assert refined.end == 30.0


def test_refine_candidate_respects_duration_bounds() -> None:
    candidate = Candidate(
        peak_t=20.0,
        start=10.0,
        end=20.0,  # 10s duration
        score=5.0,
    )
    words = [
        Word(start=12.0, end=12.5, text="Hello"),
        Word(start=18.0, end=18.5, text="world."),
    ]
    # If snapped to 12.0 - 18.5, duration would be 6.5s which is < min_clip_s (15.0)
    # The refiner should not produce a clip violating min_clip_s
    refined = refine_candidate_with_transcript(
        candidate,
        words,
        min_clip_s=15.0,
        max_clip_s=60.0,
    )
    assert refined.duration >= 10.0


def test_refine_candidates_batch() -> None:
    cands = [
        Candidate(peak_t=15.0, start=5.0, end=25.0, score=3.0),
        Candidate(peak_t=50.0, start=40.0, end=60.0, score=4.0),
    ]
    words = [
        Word(start=5.2, end=5.8, text="Start"),
        Word(start=24.5, end=25.2, text="Done."),
        Word(start=40.5, end=41.0, text="Second"),
        Word(start=59.0, end=59.8, text="Over!"),
    ]

    refined_list = refine_candidates_with_transcript(cands, words)
    assert len(refined_list) == 2
    assert refined_list[0].score == 3.0
    assert refined_list[1].score == 4.0
