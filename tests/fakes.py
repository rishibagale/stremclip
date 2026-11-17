"""In-memory test doubles (fakes) implementing streamclip protocols."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from streamclip.interfaces import ChatSource, Reranker, SignalExtractor, Transcriber, VisionScorer
from streamclip.models import Candidate, ChatMessage, RankedClip, SignalTrack, VodMeta, Word


class FakeChatSource(ChatSource):
    """Fake chat source returning in-memory chat messages."""

    def __init__(self, messages: list[ChatMessage] | None = None) -> None:
        self.messages = messages or [
            ChatMessage(
                t=12.0, user="viewer1", text="OMEGALUL that was insane", emotes=["OMEGALUL"]
            ),
            ChatMessage(t=13.5, user="viewer2", text="W gameplay", emotes=[]),
            ChatMessage(t=14.0, user="viewer3", text="clip that right now!", emotes=[]),
        ]

    def load_messages(self, source: Path | str) -> list[ChatMessage]:
        return list(self.messages)


class FakeSignalExtractor(SignalExtractor):
    """Fake signal extractor returning raw 1 Hz unnormalized float arrays."""

    def __init__(self, name: str = "fake_signal", constant_value: float = 1.0) -> None:
        self._name = name
        self.constant_value = constant_value

    @property
    def name(self) -> str:
        return self._name

    def extract(self, vod: VodMeta, **kwargs: Any) -> SignalTrack:
        length = math.ceil(vod.duration_s)
        # Raw unnormalized values of length ceil(duration_s)
        values = [float(self.constant_value) for _ in range(length)]
        return SignalTrack(
            name=self.name,
            hz=1.0,
            values=values,
            meta={"extractor": "FakeSignalExtractor", "length": length},
        )


class FakeVisionScorer(VisionScorer):
    """Fake vision scorer assigning synthetic visual action scores."""

    def __init__(self, default_score: float = 0.85) -> None:
        self.default_score = default_score

    def score_candidates(self, vod: VodMeta, candidates: list[Candidate]) -> list[Candidate]:
        updated: list[Candidate] = []
        for cand in candidates:
            comps = dict(cand.components)
            comps["vision"] = self.default_score
            updated.append(
                Candidate(
                    peak_t=cand.peak_t,
                    start=cand.start,
                    end=cand.end,
                    score=cand.score + (0.2 * self.default_score),
                    components=comps,
                )
            )
        return updated


class FakeReranker(Reranker):
    """Fake reranker generating ranked highlight clips."""

    def __init__(self, default_llm_score: float = 9.0) -> None:
        self.default_llm_score = default_llm_score

    def rerank(
        self,
        candidates: list[Candidate],
        chat: list[ChatMessage] | None = None,
        transcript: list[Word] | None = None,
    ) -> list[RankedClip]:
        ranked: list[RankedClip] = []
        for i, cand in enumerate(candidates, start=1):
            final = cand.score * 0.5 + (self.default_llm_score / 10.0) * 0.5
            ranked.append(
                RankedClip(
                    candidate=cand,
                    llm_score=self.default_llm_score,
                    title=f"Highlight #{i} at {cand.peak_t:.1f}s",
                    reason="High chat activity and intense gameplay action",
                    final_score=final,
                )
            )
        ranked.sort(key=lambda c: c.final_score, reverse=True)
        return ranked


class FakeTranscriber(Transcriber):
    """Fake speech-to-text transcriber returning mock timed words."""

    def __init__(self, words: list[Word] | None = None) -> None:
        self.words = words or [
            Word(start=10.0, end=10.5, text="Oh"),
            Word(start=10.6, end=11.2, text="my"),
            Word(start=11.3, end=12.0, text="god!"),
        ]

    def transcribe(
        self,
        vod: VodMeta,
        start: float | None = None,
        end: float | None = None,
    ) -> list[Word]:
        if start is None and end is None:
            return list(self.words)
        s = start if start is not None else 0.0
        e = end if end is not None else vod.duration_s
        return [w for w in self.words if w.start >= s and w.end <= e]
