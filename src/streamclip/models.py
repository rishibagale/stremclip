"""Domain models for streamclip."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class VodMeta(BaseModel):
    """Metadata describing a livestream VOD."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool = True

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v  # type: ignore[no-any-return]

    @property
    def expected_signal_length(self) -> int:
        """Expected length of a 1 Hz raw signal track for this VOD."""
        return math.ceil(self.duration_s)


class ChatMessage(BaseModel):
    """A single chat replay message."""

    t: float
    user: str
    text: str
    emotes: list[str] = Field(default_factory=list)


class Word(BaseModel):
    """A timed word transcription segment."""

    start: float
    end: float
    text: str


class SignalTrack(BaseModel):
    """A 1D timeseries signal track.

    Every signal is a SignalTrack with one float per second (length ceil(duration_s)),
    raw and unnormalized.
    """

    name: str
    hz: float = 1.0
    values: list[float] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, v: Any) -> list[float]:
        if isinstance(v, np.ndarray):
            return [float(x) for x in v.tolist()]
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        return v  # type: ignore[no-any-return]

    @property
    def array(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return the values as a numpy float64 array."""
        return np.asarray(self.values, dtype=np.float64)

    def matches_vod(self, vod: VodMeta) -> bool:
        """Check if signal matches VOD duration requirements (1Hz and ceil(duration_s))."""
        return self.hz == 1.0 and len(self.values) == vod.expected_signal_length


class Candidate(BaseModel):
    """A detected highlight clip candidate window."""

    peak_t: float
    start: float
    end: float
    score: float
    components: dict[str, float] = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration of candidate window in seconds."""
        return self.end - self.start


class RankedClip(BaseModel):
    """A highlight candidate that has been scored and annotated by an LLM/reranker."""

    candidate: Candidate
    llm_score: float
    title: str
    reason: str
    final_score: float


class FacecamBox(BaseModel):
    """Bounding box coordinates for facecam region."""

    x: int
    y: int
    w: int
    h: int


class RenderSpec(BaseModel):
    """Specification for rendering a vertical highlight clip."""

    layout: Literal["stack", "gameplay_only", "facecam_only"]
    facecam: FacecamBox | None = None
    captions: list[Word] = Field(default_factory=list)
    out_w: int = 1080
    out_h: int = 1920
