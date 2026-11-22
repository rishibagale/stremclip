"""Unit tests for 4D virality scoring (Hook, Engagement, Value, Shareability) and hook titling."""

from __future__ import annotations

from typing import Any

from streamclip.config import Config
from streamclip.models import Candidate, Word
from streamclip.rerank import HeuristicReranker, LLMReranker


def test_heuristic_reranker_computes_virality_scores() -> None:
    cfg = Config()
    reranker = HeuristicReranker(config=cfg)

    cand = Candidate(
        peak_t=25.0,
        start=15.0,
        end=45.0,
        score=8.5,
        components={"chat": 0.8, "audio": 0.9, "motion": 0.7},
    )

    words = [
        Word(start=16.0, end=17.0, text="Wait"),
        Word(start=17.5, end=18.5, text="what"),
        Word(start=19.0, end=20.0, text="happened?!"),
    ]

    ranked = reranker.rerank([cand], transcript=words)
    assert len(ranked) == 1
    clip = ranked[0]

    # Verify 4D virality scores are computed in [0, 100] range
    assert 0.0 <= clip.hook_score <= 100.0
    assert 0.0 <= clip.engagement_score <= 100.0
    assert 0.0 <= clip.value_score <= 100.0
    assert 0.0 <= clip.shareability_score <= 100.0

    # Verify viral hook title is generated
    assert clip.hook_title is not None
    assert len(clip.hook_title) > 0
    assert len(clip.hook_title) <= 40
    assert clip.hook_title == clip.hook_title.upper()


def test_llm_reranker_parses_virality_and_hook() -> None:
    cfg = Config()

    def mock_http_client(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": 94.0,
            "hook_strength": 96.0,
            "engagement_score": 90.0,
            "value_score": 85.0,
            "shareability_score": 94.0,
            "hook_title": "BRO CANNOT BELIEVE THIS",
            "title": "Unbelievable Clutch",
            "reason": "Insane reaction to sudden win",
        }

    reranker = LLMReranker(http_client=mock_http_client, config=cfg)
    cand = Candidate(peak_t=10.0, start=0.0, end=25.0, score=5.0)

    ranked = reranker.rerank([cand], transcript=[])
    assert len(ranked) == 1
    clip = ranked[0]

    assert clip.hook_score == 96.0
    assert clip.engagement_score == 90.0
    assert clip.value_score == 85.0
    assert clip.shareability_score == 94.0
    assert clip.hook_title == "BRO CANNOT BELIEVE THIS"
    assert clip.title == "Unbelievable Clutch"
