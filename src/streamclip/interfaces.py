"""Protocol definitions for extensible streamclip components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from streamclip.models import Candidate, ChatMessage, RankedClip, SignalTrack, VodMeta, Word


@runtime_checkable
class ChatSource(Protocol):
    """Protocol for reading livestream chat replay sources."""

    def load_messages(self, source: Path | str) -> list[ChatMessage]:
        """Load chat messages from a file or replay source."""
        ...


@runtime_checkable
class SignalExtractor(Protocol):
    """Protocol for extracting raw 1 Hz timeseries signals from VODs or chat."""

    @property
    def name(self) -> str:
        """Name of the signal track produced by this extractor."""
        ...

    def extract(self, vod: VodMeta, **kwargs: Any) -> SignalTrack:
        """Extract a raw, unnormalized SignalTrack with one value per second."""
        ...


@runtime_checkable
class VisionScorer(Protocol):
    """Protocol for scoring candidate windows with computer vision models."""

    def score_candidates(self, vod: VodMeta, candidates: list[Candidate]) -> list[Candidate]:
        """Compute vision scores for candidate windows within a budget."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for LLM or heuristic candidate reranking and title generation."""

    def rerank(
        self,
        candidates: list[Candidate],
        chat: list[ChatMessage] | None = None,
        transcript: list[Word] | None = None,
    ) -> list[RankedClip]:
        """Score, filter, and rank highlight candidates."""
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for speech-to-text audio transcription."""

    def transcribe(
        self,
        vod: VodMeta,
        start: float | None = None,
        end: float | None = None,
    ) -> list[Word]:
        """Transcribe speech in the given time window into timed words."""
        ...
