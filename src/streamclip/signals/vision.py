"""Computer vision highlight candidate scorers: MotionScorer, LLMVisionScorer, and Fake."""

from __future__ import annotations

import itertools
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from streamclip.cache import DiskCache
from streamclip.interfaces import VisionScorer
from streamclip.models import Candidate, VodMeta
from streamclip.signals.frames import VideoFrame, sample_frames


def _parse_strict_json(text: str) -> dict[str, Any]:
    """Parse strict JSON from LLM response text with markdown and regex fallbacks."""
    cleaned = text.strip()
    # 1. Direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Extract from markdown block or curly braces
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(f"Could not parse valid JSON from LLM response: {cleaned[:120]}")


class MotionScorer(VisionScorer):
    """Free baseline vision scorer using frame-difference magnitude and scene-change rate."""

    def __init__(
        self,
        budget: int = 40,
        interval_s: float = 1.0,
        cut_threshold: float = 0.35,
        weight: float = 0.2,
        cache: DiskCache | None = None,
    ) -> None:
        self.budget = budget
        self.interval_s = interval_s
        self.cut_threshold = cut_threshold
        self.weight = weight
        self.cache = cache or DiskCache()

    def _score_window(
        self,
        vod: VodMeta,
        start_s: float,
        end_s: float,
    ) -> float:
        """Compute motion intensity and scene cut score for a time window."""
        frames = sample_frames(vod, start_s, end_s, interval_s=self.interval_s)
        if len(frames) < 2:
            return 0.0

        # Compute consecutive frame differences in grayscale
        diffs: list[float] = []
        for f1, f2 in itertools.pairwise(frames):
            arr1 = f1.array.astype(np.float64)
            arr2 = f2.array.astype(np.float64)
            # Grayscale mean if RGB
            g1 = arr1.mean(axis=2) if arr1.ndim == 3 else arr1
            g2 = arr2.mean(axis=2) if arr2.ndim == 3 else arr2
            diff = float(np.mean(np.abs(g2 - g1)) / 255.0)
            diffs.append(diff)

        if not diffs:
            return 0.0

        avg_motion = float(np.mean(diffs))
        cut_count = sum(1 for d in diffs if d >= self.cut_threshold)
        cut_rate = cut_count / len(diffs)

        # Composite score normalized to [0.0, 1.0]
        # High frame-to-frame change (action) + high scene cut rate (action transitions)
        raw_score = 0.65 * (avg_motion * 3.5) + 0.35 * (cut_rate * 2.5)
        return min(1.0, max(0.0, float(raw_score)))

    def score_candidates(
        self,
        vod: VodMeta,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        """Score candidate windows within budget, caching results per window."""
        if not candidates:
            return []

        # 1. Gate candidates by budget (score top candidates first)
        sorted_cands = sorted(candidates, key=lambda c: -c.score)
        scored_results: list[Candidate] = []
        is_cached_file = Path(vod.path).is_file()

        for idx, cand in enumerate(sorted_cands):
            if idx < self.budget:
                # Within budget: check cache or compute
                config_slice = {
                    "start": round(cand.start, 2),
                    "end": round(cand.end, 2),
                    "scorer": "motion",
                }
                cached = self.cache.get(vod.path, config_slice) if is_cached_file else None
                if cached is not None and isinstance(cached, (int, float)):
                    vision_score = float(cached)
                else:
                    vision_score = self._score_window(vod, cand.start, cand.end)
                    if is_cached_file:
                        self.cache.set(vod.path, config_slice, vision_score)

                comps = dict(cand.components)
                comps["vision"] = round(vision_score, 4)
                updated_cand = Candidate(
                    peak_t=cand.peak_t,
                    start=cand.start,
                    end=cand.end,
                    score=cand.score + (self.weight * vision_score),
                    components=comps,
                )
                scored_results.append(updated_cand)
            else:
                # Exceeded budget: keep existing score and component
                comps = dict(cand.components)
                comps.setdefault("vision", 0.0)
                scored_results.append(
                    Candidate(
                        peak_t=cand.peak_t,
                        start=cand.start,
                        end=cand.end,
                        score=cand.score,
                        components=comps,
                    )
                )

        # Re-sort by updated score descending
        scored_results.sort(key=lambda c: (-c.score, c.peak_t))
        return scored_results


class LLMVisionScorer(VisionScorer):
    """Visual action scorer calling Anthropic API on base64 images with retries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        budget: int = 40,
        interval_s: float = 1.5,
        max_frames: int = 5,
        weight: float = 0.2,
        max_retries: int = 3,
        http_client: Any | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.budget = budget
        self.interval_s = interval_s
        self.max_frames = max_frames
        self.weight = weight
        self.max_retries = max_retries
        self.http_client = http_client
        self.cache = cache or DiskCache()

    def _call_api_with_retries(self, frames: list[VideoFrame]) -> tuple[float, str]:
        """Send base64 frames to Anthropic API and return (score, reason) with retries."""
        # Check authentication if no mock client is injected
        if not self.api_key and self.http_client is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Provide an API key or set ANTHROPIC_API_KEY to use LLMVisionScorer."
            )

        # Prepare image content blocks
        content_blocks: list[dict[str, Any]] = []
        for frame in frames:
            b64 = frame.to_base64()
            if b64:
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    }
                )

        content_blocks.append(
            {
                "type": "text",
                "text": (
                    "Analyze these sequential frames from a livestream video moment. "
                    "Rate the dramatic highlight potential (action, climax, funny glitch, "
                    "major gameplay milestone, crash/death) on a scale from 0.0 to 1.0. "
                    'Return ONLY a valid JSON object: {"score": <float 0.0 to 1.0>, '
                    '"reason": "<short 1-sentence explanation>"}.'
                ),
            }
        )

        payload = {
            "model": self.model,
            "max_tokens": 150,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": content_blocks}],
        }

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.http_client is not None:
                    # Injected mock or custom client
                    result = self.http_client(payload)
                    if isinstance(result, dict) and "score" in result:
                        score = float(result["score"])
                        reason = str(result.get("reason", ""))
                        return min(1.0, max(0.0, score)), reason
                    if isinstance(result, str):
                        data = _parse_strict_json(result)
                        score = float(data.get("score", 0.0))
                        reason = str(data.get("reason", ""))
                        return min(1.0, max(0.0, score)), reason
                    if isinstance(result, dict) and "content" in result:
                        text_resp = result["content"][0]["text"]
                        data = _parse_strict_json(text_resp)
                        score = float(data.get("score", 0.0))
                        reason = str(data.get("reason", ""))
                        return min(1.0, max(0.0, score)), reason
                    raise ValueError(f"Unexpected response format from http_client: {result}")

                # Live Anthropic API request via urllib
                headers = {
                    "x-api-key": str(self.api_key),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    resp_body = resp.read().decode("utf-8")
                    resp_json = json.loads(resp_body)
                    text_resp = resp_json["content"][0]["text"]
                    data = _parse_strict_json(text_resp)
                    score = float(data.get("score", 0.0))
                    reason = str(data.get("reason", ""))
                    return min(1.0, max(0.0, score)), reason

            except Exception as err:
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(0.2 * attempt)
                else:
                    raise last_err from None

        return 0.0, "Failed after retries"

    def _score_window(
        self,
        vod: VodMeta,
        start_s: float,
        end_s: float,
    ) -> float:
        """Extract frames and score with Anthropic API."""
        frames = sample_frames(vod, start_s, end_s, interval_s=self.interval_s)
        if not frames:
            return 0.0

        # Subsample if more frames than max_frames
        if len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            sampled = [frames[i] for i in indices]
        else:
            sampled = frames

        score, _ = self._call_api_with_retries(sampled)
        return score

    def score_candidates(
        self,
        vod: VodMeta,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        """Score candidate windows within budget using LLM Vision."""
        if not candidates:
            return []

        if not self.api_key and self.http_client is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Provide an API key or set ANTHROPIC_API_KEY to use LLMVisionScorer."
            )

        sorted_cands = sorted(candidates, key=lambda c: -c.score)
        scored_results: list[Candidate] = []
        is_cached_file = Path(vod.path).is_file()

        for idx, cand in enumerate(sorted_cands):
            if idx < self.budget:
                config_slice = {
                    "start": round(cand.start, 2),
                    "end": round(cand.end, 2),
                    "scorer": "llm_vision",
                    "model": self.model,
                }
                cached = self.cache.get(vod.path, config_slice) if is_cached_file else None
                if cached is not None and isinstance(cached, (int, float)):
                    vision_score = float(cached)
                else:
                    vision_score = self._score_window(vod, cand.start, cand.end)
                    if is_cached_file:
                        self.cache.set(vod.path, config_slice, vision_score)

                comps = dict(cand.components)
                comps["vision"] = round(vision_score, 4)
                updated_cand = Candidate(
                    peak_t=cand.peak_t,
                    start=cand.start,
                    end=cand.end,
                    score=cand.score + (self.weight * vision_score),
                    components=comps,
                )
                scored_results.append(updated_cand)
            else:
                comps = dict(cand.components)
                comps.setdefault("vision", 0.0)
                scored_results.append(
                    Candidate(
                        peak_t=cand.peak_t,
                        start=cand.start,
                        end=cand.end,
                        score=cand.score,
                        components=comps,
                    )
                )

        scored_results.sort(key=lambda c: (-c.score, c.peak_t))
        return scored_results


class FakeVisionScorer(VisionScorer):
    """In-memory fake vision scorer with gating and caching for unit tests."""

    def __init__(
        self,
        default_score: float = 0.85,
        budget: int = 40,
        weight: float = 0.2,
        cache: DiskCache | None = None,
    ) -> None:
        self.default_score = default_score
        self.budget = budget
        self.weight = weight
        self.cache = cache or DiskCache()
        self.call_count = 0

    def score_candidates(
        self,
        vod: VodMeta,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        """Score candidate windows with mock action values respecting budget and cache."""
        if not candidates:
            return []

        sorted_cands = sorted(candidates, key=lambda c: -c.score)
        scored_results: list[Candidate] = []
        is_cached_file = Path(vod.path).is_file()

        for idx, cand in enumerate(sorted_cands):
            if idx < self.budget:
                config_slice = {
                    "start": round(cand.start, 2),
                    "end": round(cand.end, 2),
                    "scorer": "fake_vision",
                }
                cached = self.cache.get(vod.path, config_slice) if is_cached_file else None
                if cached is not None and isinstance(cached, (int, float)):
                    vision_score = float(cached)
                else:
                    self.call_count += 1
                    vision_score = self.default_score
                    if is_cached_file:
                        self.cache.set(vod.path, config_slice, vision_score)

                comps = dict(cand.components)
                comps["vision"] = round(vision_score, 4)
                updated_cand = Candidate(
                    peak_t=cand.peak_t,
                    start=cand.start,
                    end=cand.end,
                    score=cand.score + (self.weight * vision_score),
                    components=comps,
                )
                scored_results.append(updated_cand)
            else:
                comps = dict(cand.components)
                comps.setdefault("vision", 0.0)
                scored_results.append(
                    Candidate(
                        peak_t=cand.peak_t,
                        start=cand.start,
                        end=cand.end,
                        score=cand.score,
                        components=comps,
                    )
                )

        scored_results.sort(key=lambda c: (-c.score, c.peak_t))
        return scored_results
