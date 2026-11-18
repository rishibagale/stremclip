"""Unit tests for highlight candidate reranking, LLM scoring, and deduplication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streamclip.cache import DiskCache
from streamclip.config import Config
from streamclip.interfaces import Reranker
from streamclip.models import Candidate, ChatMessage, RankedClip, VodMeta, Word
from streamclip.rerank import (
    HeuristicReranker,
    LLMReranker,
    compute_clip_iou,
    dedupe_clips,
    extract_chat_excerpt,
    extract_transcript_excerpt,
)


def test_extract_transcript_excerpt() -> None:
    words = [
        Word(start=1.0, end=2.0, text="Welcome"),
        Word(start=2.5, end=3.5, text="to"),
        Word(start=4.0, end=5.0, text="the"),
        Word(start=10.0, end=11.0, text="stream"),
        Word(start=20.0, end=21.0, text="everyone"),
    ]

    # Matching window [2.0, 10.5]
    res = extract_transcript_excerpt(words, 2.0, 10.5)
    assert res == "to the stream"

    # Truncation via max_words
    res_trunc = extract_transcript_excerpt(words, 0.0, 30.0, max_words=2)
    assert res_trunc == "Welcome to"

    # No words in range
    res_empty = extract_transcript_excerpt(words, 50.0, 60.0)
    assert res_empty == "(No speech detected)"

    # None transcript
    assert extract_transcript_excerpt(None, 0.0, 10.0) == "(No speech detected)"


def test_extract_chat_excerpt() -> None:
    msgs = [
        ChatMessage(t=5.0, user="alice", text="hello"),
        ChatMessage(t=12.0, user="bob", text="POGGERS"),
        ChatMessage(t=14.5, user="carol", text="NO WAY"),
        ChatMessage(t=30.0, user="dave", text="gg"),
    ]

    # Matching window [10.0, 15.0]
    res = extract_chat_excerpt(msgs, 10.0, 15.0)
    assert "[+2.0s] bob: POGGERS" in res
    assert "[+4.5s] carol: NO WAY" in res
    assert "alice" not in res
    assert "dave" not in res

    # No messages in range
    assert extract_chat_excerpt(msgs, 40.0, 50.0) == "(No chat activity)"

    # None chat
    assert extract_chat_excerpt(None, 0.0, 10.0) == "(No chat activity)"


def test_compute_clip_iou() -> None:
    c1 = Candidate(peak_t=10.0, start=5.0, end=15.0, score=2.0)
    c2 = Candidate(peak_t=12.0, start=10.0, end=20.0, score=1.8)
    # Inter: [10, 15] = 5.0. Union: 10 + 10 - 5 = 15.0. IoU = 5 / 15 = 0.333...
    assert round(compute_clip_iou(c1, c2), 3) == 0.333

    # Identical
    assert compute_clip_iou(c1, c1) == 1.0

    # No overlap
    c3 = Candidate(peak_t=30.0, start=25.0, end=35.0, score=1.0)
    assert compute_clip_iou(c1, c3) == 0.0


def test_dedupe_clips() -> None:
    c1 = Candidate(peak_t=100.0, start=85.0, end=115.0, score=5.0)
    c2 = Candidate(peak_t=102.0, start=87.0, end=117.0, score=4.5)  # Peak gap 2s (<10s), IoU > 0.5
    c3 = Candidate(peak_t=200.0, start=185.0, end=215.0, score=4.0)  # Far away

    r1 = RankedClip(
        candidate=c1, llm_score=90.0, title="Top Clip", reason="Peak 1", final_score=8.5
    )
    r2 = RankedClip(
        candidate=c2, llm_score=80.0, title="Dup Clip", reason="Peak 2", final_score=7.5
    )
    r3 = RankedClip(
        candidate=c3, llm_score=85.0, title="Second Clip", reason="Peak 3", final_score=8.0
    )

    deduped = dedupe_clips([r2, r1, r3], min_iou=0.5, min_gap_s=10.0)
    assert len(deduped) == 2
    assert deduped[0].title == "Top Clip"
    assert deduped[1].title == "Second Clip"
    assert all(c.title != "Dup Clip" for c in deduped)

    # Empty list
    assert dedupe_clips([]) == []


def test_heuristic_reranker_protocol_and_ranking() -> None:
    reranker = HeuristicReranker()
    assert isinstance(reranker, Reranker)

    c_chat = Candidate(
        peak_t=50.0,
        start=35.0,
        end=65.0,
        score=4.5,
        components={"chat": 5.0, "audio": 1.2, "vision": 0.5},
    )
    c_audio = Candidate(
        peak_t=150.0,
        start=135.0,
        end=165.0,
        score=3.8,
        components={"chat": 0.8, "audio": 4.5, "vision": 0.4},
    )
    c_vision = Candidate(
        peak_t=250.0,
        start=235.0,
        end=265.0,
        score=3.2,
        components={"chat": 0.2, "audio": 0.3, "vision": 4.0},
    )
    c_fused = Candidate(
        peak_t=350.0,
        start=335.0,
        end=365.0,
        score=2.0,
        components={},
    )

    ranked = reranker.rerank([c_vision, c_chat, c_fused, c_audio])
    assert len(ranked) == 4
    # Highest score first: c_chat (4.5), c_audio (3.8), c_vision (3.2), c_fused (2.0)
    assert ranked[0].candidate.peak_t == 50.0
    assert "Chat Surge" in ranked[0].title
    assert "chat score" in ranked[0].reason

    assert ranked[1].candidate.peak_t == 150.0
    assert "Audio Climax" in ranked[1].title
    assert "audio score" in ranked[1].reason

    assert ranked[2].candidate.peak_t == 250.0
    assert "Action Climax" in ranked[2].title
    assert "vision score" in ranked[2].reason

    assert ranked[3].candidate.peak_t == 350.0
    assert "Highlight Moment" in ranked[3].title

    # Empty input
    assert reranker.rerank([]) == []


def test_llm_reranker_protocol_and_default_fallback_without_key() -> None:
    # No api_key and no http_client -> gracefully falls back to HeuristicReranker
    reranker = LLMReranker(api_key=None, http_client=None)
    assert isinstance(reranker, Reranker)

    cand = Candidate(
        peak_t=60.0,
        start=45.0,
        end=75.0,
        score=3.5,
        components={"chat": 4.0, "audio": 1.0},
    )
    ranked = reranker.rerank([cand])
    assert len(ranked) == 1
    assert "Chat Surge" in ranked[0].title
    assert ranked[0].final_score == 3.5


def test_llm_reranker_with_mock_client() -> None:
    payloads_received: list[dict[str, Any]] = []

    def mock_http_client(payload: dict[str, Any]) -> str:
        payloads_received.append(payload)
        # Return strict JSON inside markdown block
        return json.dumps(
            {
                "score": 88.5,
                "hook_strength": 92.0,
                "title": "Insane Streamer Clutch",
                "reason": "Massive chat explosion and epic reaction",
            }
        )

    reranker = LLMReranker(
        api_key="mock-key",
        http_client=mock_http_client,
    )

    cand = Candidate(
        peak_t=120.0,
        start=105.0,
        end=135.0,
        score=4.0,
        components={"chat": 4.5, "audio": 2.0},
    )
    chat = [ChatMessage(t=120.5, user="speed", text="OMG CLIP THIS")]
    words = [Word(start=119.0, end=121.0, text="We did it!")]

    ranked = reranker.rerank([cand], chat=chat, transcript=words)
    assert len(ranked) == 1
    clip = ranked[0]
    assert clip.title == "Insane Streamer Clutch"
    assert clip.reason == "Massive chat explosion and epic reaction"
    assert clip.llm_score == 88.5
    # final_score = 4.0 * 0.4 + 8.85 * 0.4 + 9.2 * 0.2 = 1.6 + 3.54 + 1.84 = 6.98
    assert clip.final_score == 6.98

    # Verify payload contents
    assert len(payloads_received) == 1
    content_blocks = payloads_received[0]["messages"][0]["content"]
    text_block = next(b for b in content_blocks if b["type"] == "text")
    assert "We did it!" in text_block["text"]
    assert "OMG CLIP THIS" in text_block["text"]


def test_llm_reranker_retries_and_heuristic_fallback_on_failure() -> None:
    calls = 0

    def failing_http_client(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise ConnectionResetError("API network error")

    reranker = LLMReranker(
        api_key="mock-key",
        http_client=failing_http_client,
        max_retries=2,
    )

    cand = Candidate(
        peak_t=80.0,
        start=65.0,
        end=95.0,
        score=3.0,
        components={"audio": 3.5},
    )

    ranked = reranker.rerank([cand])
    assert len(ranked) == 1
    assert calls == 2  # Retried twice
    # Gracefully fell back to heuristic for this candidate
    assert "Audio Climax" in ranked[0].title
    assert ranked[0].final_score == 3.0


def test_llm_reranker_cost_cap() -> None:
    calls = 0

    def mock_http_client(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "score": 90.0,
            "hook_strength": 85.0,
            "title": f"LLM Highlight #{calls}",
            "reason": "Top tier highlight",
        }

    cfg = Config(top_k=2)
    reranker = LLMReranker(
        api_key="mock-key",
        http_client=mock_http_client,
        config=cfg,
        cost_cap=2,
    )

    # 4 distinct candidates
    cands = [
        Candidate(peak_t=50.0, start=35.0, end=65.0, score=5.0),
        Candidate(peak_t=150.0, start=135.0, end=165.0, score=4.0),
        Candidate(peak_t=250.0, start=235.0, end=265.0, score=3.0),
        Candidate(peak_t=350.0, start=335.0, end=365.0, score=2.0),
    ]

    ranked = reranker.rerank(cands)
    assert len(ranked) == 4
    # Only top 2 candidates were sent to LLM
    assert calls == 2
    assert "LLM Highlight" in ranked[0].title
    assert "LLM Highlight" in ranked[1].title
    assert "Highlight Moment" in ranked[2].title
    assert "Highlight Moment" in ranked[3].title


def test_llm_reranker_caching(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "rerank_cache")
    dummy_video = tmp_path / "stream.mp4"
    dummy_video.write_bytes(b"dummy video data")

    vod = VodMeta(
        path=dummy_video,
        duration_s=300.0,
        fps=30.0,
        width=1920,
        height=1080,
    )

    calls = 0

    def mock_http_client(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "score": 95.0,
            "hook_strength": 90.0,
            "title": "Cached Stunt",
            "reason": "Flawless execution",
        }

    reranker = LLMReranker(
        api_key="mock-key",
        http_client=mock_http_client,
        cache=cache,
        vod=vod,
    )

    cand = Candidate(peak_t=100.0, start=85.0, end=115.0, score=4.0)

    # First call -> hits http_client
    ranked1 = reranker.rerank([cand], vod=vod)
    assert calls == 1
    assert ranked1[0].title == "Cached Stunt"

    # Second call -> cache hit, http_client not invoked
    ranked2 = reranker.rerank([cand], vod=vod)
    assert calls == 1
    assert ranked2[0].title == "Cached Stunt"
