"""Chat signal extraction for streamclip."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from streamclip.config import Config
from streamclip.interfaces import SignalExtractor
from streamclip.models import ChatMessage, SignalTrack, VodMeta

# Known hype keywords (case-insensitive)
HYPE_KEYWORDS: set[str] = {
    "lul",
    "kekw",
    "omegalul",
    "wtf",
    "no way",
    "clip it",
    "pog",
    "poggers",
    "pogchamp",
    "holy shit",
    "holy crap",
    "sheeesh",
    "sheesh",
    "insane",
}

# Regex for repeated characters (e.g. 'omggggg', 'lmaoooo', 'wwwwww')
REPEATED_CHARS_PATTERN: re.Pattern[str] = re.compile(r"(.)\1{3,}", re.IGNORECASE)


def _smooth_1d(
    arr: np.ndarray[Any, np.dtype[np.float64]],
    sigma: float = 2.0,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """1D Gaussian smoothing using numpy convolution."""
    radius = math.ceil(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum > 0:
        kernel /= kernel_sum
    res = np.convolve(arr, kernel, mode="same")
    return np.asarray(res, dtype=np.float64)


def _compute_message_hype(msg: ChatMessage) -> float:
    """Score a single chat message for hype indicators."""
    text_lower = msg.text.lower()
    score = 0.0

    # 1. Keyword and emote matches
    for kw in HYPE_KEYWORDS:
        if kw in text_lower:
            score += 1.5

    for emote in msg.emotes:
        if emote.lower() in HYPE_KEYWORDS:
            score += 1.5
        elif ord(emote[0]) > 127:  # Unicode / emoji emote
            score += 0.8

    # 2. Punctuation excitement (e.g. ??? or !!!)
    if "???" in msg.text or "!!!" in msg.text:
        score += 1.0

    # 3. Capitalization ratio (shouting)
    letters = [c for c in msg.text if c.isalpha()]
    if len(letters) >= 4:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio >= 0.7:
            score += 1.0

    # 4. Repeated character excitement (e.g. LMAOOOO)
    if REPEATED_CHARS_PATTERN.search(msg.text):
        score += 0.8

    # 5. Repeated word spam within message
    words = text_lower.split()
    if len(words) >= 3 and len(set(words)) == 1:
        score += 1.0

    return score


def extract_chat_signal(
    vod: VodMeta,
    messages: list[ChatMessage],
    config: Config | None = None,
) -> SignalTrack:
    """Extract a 1 Hz raw unnormalized SignalTrack from chat replay messages.

    Signal = smoothed messages/sec + hype-weighted component.

    Args:
        vod: VodMeta describing target VOD.
        messages: Time-sorted list of ChatMessage instances.
        config: Optional configuration specifying weights and thresholds.

    Returns:
        SignalTrack named 'chat' of exact length ceil(vod.duration_s).
    """
    total_seconds = math.ceil(vod.duration_s)
    cfg = config or Config()

    # Handle empty chat
    if not messages:
        return SignalTrack(
            name="chat",
            hz=1.0,
            values=[0.0] * total_seconds,
            meta={
                "empty": True,
                "activity_confidence": 0.0,
                "total_messages": 0,
            },
        )

    # 1. Bin messages and hype scores into 1-second intervals
    raw_counts = np.zeros(total_seconds, dtype=np.float64)
    raw_hype = np.zeros(total_seconds, dtype=np.float64)

    # Track recent message spam across users
    recent_messages: list[tuple[float, str]] = []

    for msg in messages:
        sec = math.floor(msg.t)
        if 0 <= sec < total_seconds:
            raw_counts[sec] += 1.0

            hype = _compute_message_hype(msg)

            # Check cross-user message repetition spam within 3 seconds
            norm_text = msg.text.strip().lower()
            if norm_text:
                is_spam = any(
                    abs(msg.t - prev_t) <= 3.0 and prev_txt == norm_text
                    for prev_t, prev_txt in recent_messages
                )
                if is_spam:
                    hype += 1.0
                recent_messages.append((msg.t, norm_text))
                # Keep recent buffer compact
                if len(recent_messages) > 30:
                    recent_messages.pop(0)

            raw_hype[sec] += hype

    # 2. Smooth signal with gaussian filter
    smoothed_rate = _smooth_1d(raw_counts, sigma=2.0)
    smoothed_hype = _smooth_1d(raw_hype, sigma=2.0)

    # 3. Combine message rate and hype component using config weights
    chat_weight = cfg.fusion_weights.chat
    combined_signal = smoothed_rate + (chat_weight * smoothed_hype)

    # Ensure strictly non-negative values
    combined_signal = np.maximum(0.0, combined_signal)

    # 4. Activity confidence metric (low volume means unreliable)
    avg_rate = len(messages) / max(1.0, vod.duration_s)
    # Reaches ~0.9+ when rate >= 0.5 msgs/sec or total >= 100 messages
    confidence = float(
        np.clip(
            (len(messages) / 100.0) * min(1.0, avg_rate / 0.2),
            0.0,
            1.0,
        )
    )

    return SignalTrack(
        name="chat",
        hz=1.0,
        values=[float(x) for x in combined_signal.tolist()],
        meta={
            "empty": False,
            "activity_confidence": round(confidence, 3),
            "total_messages": len(messages),
            "peak_msg_rate": float(np.max(raw_counts)),
        },
    )


class ChatSignalExtractor(SignalExtractor):
    """SignalExtractor protocol implementation for livestream chat."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    @property
    def name(self) -> str:
        return "chat"

    def extract(self, vod: VodMeta, **kwargs: Any) -> SignalTrack:
        messages: list[ChatMessage] = kwargs.get("messages") or kwargs.get("chat") or []
        cfg: Config = kwargs.get("config") or self.config
        return extract_chat_signal(vod, messages, config=cfg)
