"""Candidate reranking, LLM scoring, title generation, and clip deduplication."""

from __future__ import annotations

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
from streamclip.config import Config
from streamclip.interfaces import Reranker
from streamclip.models import Candidate, ChatMessage, RankedClip, VodMeta, Word
from streamclip.signals.frames import VideoFrame, sample_frames


def extract_transcript_excerpt(
    transcript: list[Word] | None,
    start: float,
    end: float,
    max_words: int = 100,
) -> str:
    """Extract and format timed speech words falling within or overlapping [start, end]."""
    if not transcript:
        return "(No speech detected)"

    matching = [w for w in transcript if w.end > start and w.start < end]
    if not matching:
        return "(No speech detected)"

    selected = matching[:max_words]
    text = " ".join(w.text for w in selected).strip()
    return text if text else "(No speech detected)"


def extract_chat_excerpt(
    chat: list[ChatMessage] | None,
    start: float,
    end: float,
    max_messages: int = 25,
) -> str:
    """Extract and format chat replay messages falling within [start, end]."""
    if not chat:
        return "(No chat activity)"

    matching = [m for m in chat if start <= m.t <= end]
    if not matching:
        return "(No chat activity)"

    selected = matching[:max_messages]
    lines: list[str] = []
    for m in selected:
        offset = m.t - start
        lines.append(f"[{offset:+.1f}s] {m.user}: {m.text}")

    return "\n".join(lines)


def compute_clip_iou(cand1: Candidate, cand2: Candidate) -> float:
    """Compute temporal Intersection-over-Union (IoU) between two candidate windows."""
    inter = max(0.0, min(cand1.end, cand2.end) - max(cand1.start, cand2.start))
    if inter <= 0.0:
        return 0.0
    union = (cand1.end - cand1.start) + (cand2.end - cand2.start) - inter
    return float(inter / union) if union > 0.0 else 0.0


def dedupe_clips(
    clips: list[RankedClip],
    min_iou: float = 0.5,
    min_gap_s: float = 10.0,
) -> list[RankedClip]:
    """Deduplicate near-duplicate clips using greedy Non-Maximum Suppression (NMS).

    Clips are kept if they do not have significant temporal overlap (IoU >= min_iou)
    or close peak timestamps (|peak1 - peak2| < min_gap_s) with any higher-scoring clip.
    """
    if not clips:
        return []

    # Sort descending by final_score, then llm_score, then earlier peak
    sorted_clips = sorted(
        clips,
        key=lambda c: (-c.final_score, -c.llm_score, c.candidate.peak_t),
    )

    accepted: list[RankedClip] = []
    for clip in sorted_clips:
        cand = clip.candidate
        is_duplicate = False
        for acc in accepted:
            acc_cand = acc.candidate
            iou = compute_clip_iou(cand, acc_cand)
            peak_diff = abs(cand.peak_t - acc_cand.peak_t)
            if iou >= min_iou or peak_diff < min_gap_s:
                is_duplicate = True
                break
        if not is_duplicate:
            accepted.append(clip)

    return accepted


def _parse_strict_json(text: str) -> dict[str, Any]:
    """Parse strict JSON from LLM response text with markdown and regex extraction fallbacks."""
    cleaned = text.strip()
    # 1. Direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Extract from markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Extract curly braces block
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(f"Could not parse valid JSON from LLM response: {cleaned[:120]}")


class HeuristicReranker(Reranker):
    """Component-score-based highlight reranker used as baseline and fallback."""

    def __init__(
        self,
        config: Config | None = None,
        min_iou: float = 0.5,
        min_gap_s: float = 10.0,
    ) -> None:
        self.config = config or Config()
        self.min_iou = min_iou
        self.min_gap_s = min_gap_s

    def _score_single(self, cand: Candidate) -> RankedClip:
        """Derive score, title, and reason for a single candidate using component scores."""
        comps = cand.components
        if comps:
            dominant_name, dominant_val = max(comps.items(), key=lambda kv: kv[1])
        else:
            dominant_name, dominant_val = "fused", cand.score

        if dominant_name == "chat":
            title = f"Chat Surge at {cand.peak_t:.0f}s"
            reason = f"High chat message velocity and emote spikes (chat score: {dominant_val:.2f})"
        elif dominant_name == "audio":
            title = f"Audio Climax at {cand.peak_t:.0f}s"
            reason = f"Loud volume and vocal intensity (audio score: {dominant_val:.2f})"
        elif dominant_name == "vision":
            title = f"Action Climax at {cand.peak_t:.0f}s"
            reason = (
                f"High visual movement and scene transitions (vision score: {dominant_val:.2f})"
            )
        else:
            title = f"Highlight Moment at {cand.peak_t:.0f}s"
            reason = f"Multimodal signal peak detected (score: {cand.score:.2f})"

        llm_score = min(100.0, max(0.0, round(cand.score * 10.0, 1)))
        final_score = round(cand.score, 3)

        return RankedClip(
            candidate=cand,
            llm_score=llm_score,
            title=title,
            reason=reason,
            final_score=final_score,
        )

    def rerank(
        self,
        candidates: list[Candidate],
        chat: list[ChatMessage] | None = None,
        transcript: list[Word] | None = None,
        vod: VodMeta | None = None,
        **kwargs: Any,
    ) -> list[RankedClip]:
        """Score and deduplicate candidates using component scores only."""
        if not candidates:
            return []

        ranked: list[RankedClip] = [self._score_single(c) for c in candidates]
        return dedupe_clips(ranked, min_iou=self.min_iou, min_gap_s=self.min_gap_s)


class LLMReranker(Reranker):
    """LLM-powered highlight reranker using Anthropic API, transcript, chat, and frames."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        cost_cap: int | None = None,
        max_frames: int = 4,
        max_retries: int = 3,
        http_client: Any | None = None,
        config: Config | None = None,
        vod: VodMeta | None = None,
        min_iou: float = 0.5,
        min_gap_s: float = 10.0,
        cache: DiskCache | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.cost_cap = cost_cap
        self.max_frames = max_frames
        self.max_retries = max_retries
        self.http_client = http_client
        self.config = config or Config()
        self.vod = vod
        self.min_iou = min_iou
        self.min_gap_s = min_gap_s
        self.cache = cache or DiskCache()
        self._heuristic_reranker = HeuristicReranker(
            config=self.config,
            min_iou=self.min_iou,
            min_gap_s=self.min_gap_s,
        )

    def _call_llm_with_retries(
        self,
        cand: Candidate,
        transcript_excerpt: str,
        chat_excerpt: str,
        frames: list[VideoFrame],
    ) -> dict[str, Any]:
        """Call Anthropic API with multimodal payload and return parsed response dict."""
        content_blocks: list[dict[str, Any]] = []

        # 1. Base64 JPEG frames
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

        # 2. Text instructions and candidate context
        prompt = (
            "You are an expert video editor producing high-retention vertical highlights "
            "(TikTok, Shorts, Reels) from livestreams.\n\n"
            f"Candidate highlight window: {cand.start:.1f}s to {cand.end:.1f}s "
            f"(duration: {cand.duration:.1f}s, peak action at {cand.peak_t:.1f}s)\n"
            f"Detector component scores: {cand.components}\n\n"
            f"=== TRANSCRIPT EXCERPT ===\n{transcript_excerpt}\n\n"
            f"=== CHAT REPLAY EXCERPT ===\n{chat_excerpt}\n\n"
            "Evaluate this candidate moment for a standalone short-form highlight.\n"
            "Return ONLY a valid JSON object with the following keys:\n"
            '- "score": float between 0.0 and 100.0 (overall standalone quality)\n'
            '- "hook_strength": float between 0.0 and 100.0 (opening 3-5 seconds hook potential)\n'
            '- "title": concise, engaging viral title (under 60 characters)\n'
            '- "reason": one-line explanation of why this moment is highlight-worthy\n'
        )
        content_blocks.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "max_tokens": 250,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": content_blocks}],
        }

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.http_client is not None:
                    result = self.http_client(payload)
                    if isinstance(result, dict) and "score" in result:
                        parsed = result
                    elif isinstance(result, str):
                        parsed = _parse_strict_json(result)
                    elif isinstance(result, dict) and "content" in result:
                        text_resp = result["content"][0]["text"]
                        parsed = _parse_strict_json(text_resp)
                    else:
                        raise ValueError(f"Unexpected response format from http_client: {result}")
                else:
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
                        parsed = _parse_strict_json(text_resp)

                score = float(parsed["score"])
                hook = float(parsed.get("hook_strength", score))
                title = str(parsed.get("title", "")).strip()
                reason = str(parsed.get("reason", "")).strip()
                if not title:
                    title = f"Highlight at {cand.peak_t:.0f}s"
                if not reason:
                    reason = "High viral potential and engagement"

                return {
                    "score": min(100.0, max(0.0, score)),
                    "hook_strength": min(100.0, max(0.0, hook)),
                    "title": title,
                    "reason": reason,
                }
            except Exception as err:
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(0.1 * attempt)
                else:
                    raise last_err from None

        raise RuntimeError("Failed after retries")

    def rerank(
        self,
        candidates: list[Candidate],
        chat: list[ChatMessage] | None = None,
        transcript: list[Word] | None = None,
        vod: VodMeta | None = None,
        **kwargs: Any,
    ) -> list[RankedClip]:
        """Score, filter, deduplicate, and rank highlight candidates."""
        if not candidates:
            return []

        # If no API key and no test mock client, fallback to heuristic reranker
        if not self.api_key and self.http_client is None:
            return self._heuristic_reranker.rerank(
                candidates, chat=chat, transcript=transcript, vod=vod, **kwargs
            )

        target_vod = vod or self.vod
        effective_cap = self.cost_cap if self.cost_cap is not None else self.config.top_k

        sorted_candidates = sorted(candidates, key=lambda c: (-c.score, c.peak_t))
        ranked_clips: list[RankedClip] = []

        for idx, cand in enumerate(sorted_candidates):
            if idx < effective_cap:
                try:
                    transcript_text = extract_transcript_excerpt(transcript, cand.start, cand.end)
                    chat_text = extract_chat_excerpt(chat, cand.start, cand.end)

                    frames: list[VideoFrame] = []
                    if target_vod is not None:
                        sampled = sample_frames(
                            target_vod,
                            cand.start,
                            cand.end,
                            interval_s=max(1.0, cand.duration / max(1, self.max_frames)),
                        )
                        if len(sampled) > self.max_frames:
                            indices = np.linspace(0, len(sampled) - 1, self.max_frames, dtype=int)
                            frames = [sampled[i] for i in indices]
                        else:
                            frames = sampled

                    is_cached_file = target_vod is not None and Path(target_vod.path).is_file()
                    cached_data = None
                    config_slice = {
                        "start": round(cand.start, 2),
                        "end": round(cand.end, 2),
                        "peak_t": round(cand.peak_t, 2),
                        "model": self.model,
                        "task": "rerank",
                    }
                    if is_cached_file and target_vod is not None:
                        cached_data = self.cache.get(target_vod.path, config_slice)

                    if isinstance(cached_data, dict) and "score" in cached_data:
                        llm_out = cached_data
                    else:
                        llm_out = self._call_llm_with_retries(
                            cand, transcript_text, chat_text, frames
                        )
                        if is_cached_file and target_vod is not None:
                            self.cache.set(target_vod.path, config_slice, llm_out)

                    llm_score = float(llm_out["score"])
                    hook_strength = float(llm_out.get("hook_strength", llm_score))
                    title = str(llm_out["title"])
                    reason = str(llm_out["reason"])

                    final_score = round(
                        (cand.score * 0.4)
                        + ((llm_score / 10.0) * 0.4)
                        + ((hook_strength / 10.0) * 0.2),
                        3,
                    )

                    ranked_clips.append(
                        RankedClip(
                            candidate=cand,
                            llm_score=llm_score,
                            title=title,
                            reason=reason,
                            final_score=final_score,
                        )
                    )
                except Exception:
                    ranked_clips.append(self._heuristic_reranker._score_single(cand))
            else:
                ranked_clips.append(self._heuristic_reranker._score_single(cand))

        return dedupe_clips(ranked_clips, min_iou=self.min_iou, min_gap_s=self.min_gap_s)
