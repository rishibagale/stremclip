"""Performance profiling and call budget verification for simulated long runs (3-hour VOD)."""

from __future__ import annotations

import time
import unittest.mock
from pathlib import Path
from typing import Any

import numpy as np

from streamclip.config import Config
from streamclip.fusion import fuse_signals
from streamclip.models import Candidate, ChatMessage, SignalTrack, VodMeta
from streamclip.rerank import LLMReranker
from streamclip.signals.chat import extract_chat_signal
from streamclip.signals.vision import MotionScorer
from streamclip.windowing import create_candidates


def test_profile_simulated_3hour_signals_and_fusion() -> None:
    """Profile simulated 3-hour run (10,800s, 50,000 chat messages).

    Ensures signal extraction, multimodal fusion, and windowing complete in under 2.0s.
    """
    three_hours_s = 10800.0
    vod = VodMeta(
        path=Path("simulated_3hr.mp4"),
        duration_s=three_hours_s,
        fps=30.0,
        width=1920,
        height=1080,
    )

    # 1. Synthesize 50,000 chat messages with 10 bursts
    rng = np.random.default_rng(42)
    burst_centers = [600, 1500, 2700, 3900, 5100, 6300, 7500, 8700, 9600, 10200]
    all_timestamps: list[float] = []

    # 35,000 background messages uniformly distributed
    all_timestamps.extend(rng.uniform(0.0, three_hours_s, size=35000).tolist())

    # 15,000 burst messages clustered around burst centers
    for center in burst_centers:
        burst_ts = rng.normal(loc=center, scale=10.0, size=1500)
        clipped_ts = np.clip(burst_ts, 0.0, three_hours_s)
        all_timestamps.extend(clipped_ts.tolist())

    all_timestamps.sort()
    chat_messages = [
        ChatMessage(
            t=round(t, 2),
            user=f"user_{i % 500}",
            text="LUL KEKW THAT WAS INSANE" if i % 5 == 0 else "gg",
        )
        for i, t in enumerate(all_timestamps)
    ]
    assert len(chat_messages) == 50000

    # 2. Time Chat Signal Extraction
    t0 = time.perf_counter()
    chat_track = extract_chat_signal(vod, chat_messages)
    t_chat = time.perf_counter() - t0

    # 3. Time Audio Signal Simulation (10,800s array)
    audio_vals = rng.normal(loc=0.05, scale=0.02, size=int(three_hours_s))
    for center in burst_centers:
        audio_vals[center - 5 : center + 5] += 0.8
    audio_track = SignalTrack(
        name="audio",
        hz=1.0,
        values=audio_vals.tolist(),
        meta={"has_audio": True},
    )

    # 4. Time Multimodal Fusion
    t1 = time.perf_counter()
    fused = fuse_signals([chat_track, audio_track], config=Config(top_k=25))
    t_fusion = time.perf_counter() - t1

    # 5. Time Candidate Windowing
    t2 = time.perf_counter()
    candidates = create_candidates(
        peaks=fused.peaks,
        fused_scores=fused,
        vod=vod,
        config=Config(clip_min_s=15.0, clip_max_s=60.0),
    )
    t_windowing = time.perf_counter() - t2

    total_pipeline_time = t_chat + t_fusion + t_windowing

    # Assertions
    assert len(fused.peaks) >= 10
    assert len(candidates) >= 10
    assert total_pipeline_time < 2.0, (
        f"Simulated 3-hour signal processing took {total_pipeline_time:.3f}s (budget: < 2.0s). "
        f"Breakdown: chat={t_chat:.3f}s, fusion={t_fusion:.3f}s, windowing={t_windowing:.3f}s"
    )


def test_vision_budget_is_strictly_respected() -> None:
    """Confirm MotionScorer and LLMVisionScorer never exceed vision_budget_windows."""
    vod = VodMeta(
        path=Path("fake_vod.mp4"),
        duration_s=3600.0,
        fps=30.0,
        width=1920,
        height=1080,
    )

    # 60 candidate windows, budget = 25
    budget = 25
    candidates = [
        Candidate(
            peak_t=float(i * 50),
            start=float(i * 50 - 15),
            end=float(i * 50 + 15),
            score=float(100 - i),
            components={"chat": float(100 - i)},
        )
        for i in range(60)
    ]

    scorer = MotionScorer(budget=budget)
    with unittest.mock.patch.object(scorer, "_score_window", return_value=0.75):
        scored = scorer.score_candidates(vod, candidates)

    assert len(scored) == 60
    # Top 25 candidates must have received a computed vision score
    candidates_with_vision = [c for c in scored if c.components.get("vision", 0.0) > 0.0]
    candidates_without_vision = [c for c in scored if c.components.get("vision", 0.0) == 0.0]

    # Exactly budget items were evaluated
    assert len(candidates_with_vision) == budget
    assert len(candidates_without_vision) == 60 - budget


def test_llm_cost_budget_is_strictly_respected() -> None:
    """Confirm LLMReranker strictly gates candidate count by cost_cap."""
    cost_cap = 15
    llm_calls = 0

    def mock_http_client(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal llm_calls
        llm_calls += 1
        return {
            "score": 90.0,
            "hook_strength": 85.0,
            "title": f"Highlight #{llm_calls}",
            "reason": "Top action",
        }

    reranker = LLMReranker(
        api_key="mock-key",
        http_client=mock_http_client,
        cost_cap=cost_cap,
    )

    # 50 candidate windows
    candidates = [
        Candidate(
            peak_t=float(i * 60),
            start=float(i * 60 - 15),
            end=float(i * 60 + 15),
            score=float(50 - i),
        )
        for i in range(50)
    ]

    ranked = reranker.rerank(candidates)
    # The HTTP client should have been called exactly cost_cap times (15)
    assert llm_calls == cost_cap, f"Expected {cost_cap} LLM calls, but got {llm_calls}"
    assert len(ranked) == 50
