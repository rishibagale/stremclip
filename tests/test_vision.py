"""Tests for computer vision highlight scorers: MotionScorer, LLMVisionScorer, and Fake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from streamclip.cache import DiskCache
from streamclip.interfaces import VisionScorer
from streamclip.models import Candidate, VodMeta
from streamclip.signals.frames import VideoFrame
from streamclip.signals.vision import (
    FakeVisionScorer,
    LLMVisionScorer,
    MotionScorer,
    TemplateMatcher,
    _parse_strict_json,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_strict_json() -> None:
    """Test strict JSON parser with clean, markdown-fenced, and malformed inputs."""
    # 1. Clean JSON
    d1 = _parse_strict_json('{"score": 0.85, "reason": "action"}')
    assert d1["score"] == 0.85
    assert d1["reason"] == "action"

    # 2. Markdown block
    d2 = _parse_strict_json('```json\n{"score": 0.92, "reason": "explosion"}\n```')
    assert d2["score"] == 0.92

    # 3. Text with surrounding explanation
    d3 = _parse_strict_json('Here is my review: {"score": 0.5, "reason": "calm"} thank you.')
    assert d3["score"] == 0.5

    # 4. Malformed JSON raises ValueError
    with pytest.raises(ValueError, match="Could not parse valid JSON"):
        _parse_strict_json("no json here at all")


def test_fake_vision_scorer_protocol() -> None:
    """Verify that FakeVisionScorer implements VisionScorer."""
    fake = FakeVisionScorer()
    assert isinstance(fake, VisionScorer)


def test_template_matcher_protocol() -> None:
    """Verify that custom classes satisfying match(frame) adhere to TemplateMatcher."""

    class DummyMatcher:
        def match(self, frame: VideoFrame) -> float:
            return 0.95

    matcher = DummyMatcher()
    assert isinstance(matcher, TemplateMatcher)
    frame = VideoFrame(t=1.0, array=np.zeros((10, 10, 3), dtype=np.uint8))
    assert matcher.match(frame) == 0.95


def test_motion_scorer_protocol() -> None:
    """Verify that MotionScorer implements VisionScorer."""
    scorer = MotionScorer()
    assert isinstance(scorer, VisionScorer)


def test_llm_vision_scorer_protocol() -> None:
    """Verify that LLMVisionScorer implements VisionScorer."""
    scorer = LLMVisionScorer(api_key="mock_key")
    assert isinstance(scorer, VisionScorer)


def test_motion_scorer_budget_gating(tmp_path: Path) -> None:
    """Test that MotionScorer spends at most vision_budget_windows."""
    mock_file = tmp_path / "mock.mp4"
    mock_file.write_bytes(b"dummy video data")
    vod = VodMeta(path=mock_file, duration_s=100.0, fps=30.0, width=1920, height=1080)

    # 4 candidates with preliminary scores 10, 8, 6, 4
    candidates = [
        Candidate(peak_t=15.0, start=10.0, end=20.0, score=10.0),
        Candidate(peak_t=35.0, start=30.0, end=40.0, score=8.0),
        Candidate(peak_t=55.0, start=50.0, end=60.0, score=6.0),
        Candidate(peak_t=75.0, start=70.0, end=80.0, score=4.0),
    ]

    cache = DiskCache(tmp_path / "cache")
    # Budget is 2: only top 2 candidates should be scored
    scorer = MotionScorer(budget=2, cache=cache, weight=0.2)
    scored = scorer.score_candidates(vod, candidates)

    assert len(scored) == 4
    # The top 2 by score (10.0 and 8.0) were in budget
    c_top1 = next(c for c in scored if c.peak_t == 15.0)
    c_top2 = next(c for c in scored if c.peak_t == 35.0)
    c_low1 = next(c for c in scored if c.peak_t == 55.0)
    c_low2 = next(c for c in scored if c.peak_t == 75.0)

    assert "vision" in c_top1.components
    assert "vision" in c_top2.components
    assert c_low1.components.get("vision", 0.0) == 0.0
    assert c_low2.components.get("vision", 0.0) == 0.0
    assert c_low1.score == 6.0
    assert c_low2.score == 4.0


def test_motion_scorer_caching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that MotionScorer caches per window and reuses cached results."""
    mock_file = tmp_path / "mock.mp4"
    mock_file.write_bytes(b"dummy video data")
    vod = VodMeta(path=mock_file, duration_s=60.0, fps=30.0, width=1920, height=1080)

    sample_calls = 0

    def mock_sample(vod: Any, s: float, e: float, **kwargs: Any) -> list[VideoFrame]:
        nonlocal sample_calls
        sample_calls += 1
        f1 = VideoFrame(t=s, array=np.zeros((90, 160, 3), dtype=np.uint8))
        f2 = VideoFrame(t=e, array=np.ones((90, 160, 3), dtype=np.uint8) * 128)
        return [f1, f2]

    monkeypatch.setattr("streamclip.signals.vision.sample_frames", mock_sample)

    cache = DiskCache(tmp_path / "cache")
    scorer = MotionScorer(budget=5, cache=cache, weight=0.2)
    cand = Candidate(peak_t=10.0, start=5.0, end=15.0, score=5.0)

    # First pass: samples frames, scores, and caches
    scored1 = scorer.score_candidates(vod, [cand])
    assert sample_calls == 1
    v1 = scored1[0].components["vision"]
    assert v1 > 0.0

    # Second pass: hits cache, sample_calls stays 1
    scored2 = scorer.score_candidates(vod, [cand])
    assert sample_calls == 1
    assert scored2[0].components["vision"] == v1


def test_llm_vision_scorer_mock_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test LLMVisionScorer with an injected mock http_client returning valid JSON."""
    mock_file = tmp_path / "mock.mp4"
    mock_file.write_bytes(b"dummy video data")
    vod = VodMeta(path=mock_file, duration_s=60.0, fps=30.0, width=1920, height=1080)

    # Mock sample_frames to return 3 frames with dummy JPEG bytes
    def mock_sample(vod: Any, s: float, e: float, **kwargs: Any) -> list[VideoFrame]:
        return [
            VideoFrame(t=s, array=np.zeros((90, 160, 3), dtype=np.uint8), image_bytes=b"jpg1"),
            VideoFrame(t=s + 1, array=np.zeros((90, 160, 3), dtype=np.uint8), image_bytes=b"jpg2"),
        ]

    monkeypatch.setattr("streamclip.signals.vision.sample_frames", mock_sample)

    api_calls = 0

    def mock_client(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal api_calls
        api_calls += 1
        # Verify base64 images were packaged in payload
        messages = payload["messages"]
        user_content = messages[0]["content"]
        assert any(block.get("type") == "image" for block in user_content)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"score": 0.95, "reason": "Intense car chase"}),
                }
            ]
        }

    cache = DiskCache(tmp_path / "cache")
    scorer = LLMVisionScorer(
        api_key="test_key",
        budget=1,
        weight=0.2,
        http_client=mock_client,
        cache=cache,
    )

    cand = Candidate(peak_t=10.0, start=5.0, end=15.0, score=6.0)
    scored = scorer.score_candidates(vod, [cand])

    assert api_calls == 1
    assert len(scored) == 1
    assert scored[0].components["vision"] == 0.95
    # Updated score: 6.0 + 0.2 * 0.95 = 6.19
    assert abs(scored[0].score - 6.19) < 1e-4


def test_llm_vision_scorer_retries_transient_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that LLMVisionScorer retries on transient errors and recovers."""
    mock_file = tmp_path / "mock.mp4"
    mock_file.write_bytes(b"dummy video data")
    vod = VodMeta(path=mock_file, duration_s=60.0, fps=30.0, width=1920, height=1080)

    monkeypatch.setattr(
        "streamclip.signals.vision.sample_frames",
        lambda vod, s, e, **kwargs: [
            VideoFrame(t=s, array=np.zeros((90, 160, 3), dtype=np.uint8), image_bytes=b"jpg1")
        ],
    )

    attempts = 0

    def flaking_client(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("Transient network glitch")
        return {"score": 0.75, "reason": "Recovered on attempt 2"}

    cache = DiskCache(tmp_path / "cache")
    scorer = LLMVisionScorer(
        api_key="test_key",
        http_client=flaking_client,
        max_retries=3,
        cache=cache,
    )

    cand = Candidate(peak_t=10.0, start=5.0, end=15.0, score=5.0)
    scored = scorer.score_candidates(vod, [cand])

    assert attempts == 2
    assert scored[0].components["vision"] == 0.75


def test_llm_vision_scorer_missing_api_key_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that calling LLMVisionScorer without API key or client raises RuntimeError."""
    mock_file = tmp_path / "mock.mp4"
    mock_file.write_bytes(b"dummy video data")
    vod = VodMeta(path=mock_file, duration_s=60.0, fps=30.0, width=1920, height=1080)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    scorer = LLMVisionScorer(api_key=None, http_client=None)

    cand = Candidate(peak_t=10.0, start=5.0, end=15.0, score=5.0)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        scorer.score_candidates(vod, [cand])


@pytest.mark.skipif(
    not (FIXTURES_DIR / "video.mp4").exists(),
    reason="Synthetic fixtures not available",
)
def test_motion_scorer_with_fixtures(tmp_path: Path) -> None:
    """Test MotionScorer end-to-end on synthetic fixtures video."""
    video_path = FIXTURES_DIR / "video.mp4"
    vod = VodMeta(
        path=video_path, duration_s=600.0, fps=30.0, width=1920, height=1080, has_audio=True
    )

    cache = DiskCache(tmp_path / "cache")
    scorer = MotionScorer(budget=2, cache=cache, weight=0.2)

    # 2 candidates around known moments
    candidates = [
        Candidate(peak_t=60.0, start=50.0, end=70.0, score=8.0),
        Candidate(peak_t=180.0, start=170.0, end=190.0, score=7.0),
    ]

    scored = scorer.score_candidates(vod, candidates)
    assert len(scored) == 2
    for c in scored:
        assert "vision" in c.components
        assert 0.0 <= c.components["vision"] <= 1.0
        assert c.score >= 7.0
