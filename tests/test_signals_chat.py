"""Unit tests for chat signal extraction and hype weighting."""

from pathlib import Path

import numpy as np
import pytest

from streamclip.config import Config
from streamclip.ingest.chat import parse_chat
from streamclip.interfaces import SignalExtractor
from streamclip.models import ChatMessage, VodMeta
from streamclip.signals.chat import ChatSignalExtractor, extract_chat_signal

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_vod() -> VodMeta:
    return VodMeta(
        path=FIXTURES_DIR / "video.mp4",
        duration_s=600.0,
        fps=1.0,
        width=160,
        height=120,
        has_audio=True,
    )


def test_chat_signal_extractor_implements_protocol() -> None:
    extractor = ChatSignalExtractor()
    assert isinstance(extractor, SignalExtractor)
    assert extractor.name == "chat"


def test_chat_signal_peaks_on_fixture_messages(sample_vod: VodMeta) -> None:
    messages = parse_chat(FIXTURES_DIR / "chat.csv")
    assert len(messages) > 0

    track = extract_chat_signal(sample_vod, messages)
    assert track.name == "chat"
    assert track.hz == 1.0
    assert len(track.values) == 600
    assert track.matches_vod(sample_vod)

    # Invariant: the 5 injected burst timestamps (around 60, 180, 300, 420, 540)
    # must show prominent peaks compared to background
    burst_centers = [60, 180, 300, 420, 540]
    background_indices = [15, 30, 110, 230, 350, 470]

    bg_level = float(np.mean([track.values[idx] for idx in background_indices]))

    for center in burst_centers:
        peak_level = max(track.values[center - 5 : center + 5])
        assert peak_level > 2.0 * bg_level, (
            f"Expected peak near {center}s, got {peak_level} vs bg {bg_level}"
        )

    assert track.meta["empty"] is False
    assert track.meta["activity_confidence"] > 0.5


def test_empty_chat_signal(sample_vod: VodMeta) -> None:
    track = extract_chat_signal(sample_vod, [])
    assert track.name == "chat"
    assert len(track.values) == 600
    assert all(v == 0.0 for v in track.values)
    assert track.meta["empty"] is True
    assert track.meta["activity_confidence"] == 0.0


def test_activity_confidence_scaling(sample_vod: VodMeta) -> None:
    # Very few messages -> low confidence
    sparse_messages = [
        ChatMessage(t=10.0, user="u1", text="hi"),
        ChatMessage(t=200.0, user="u2", text="hello"),
    ]
    track_sparse = extract_chat_signal(sample_vod, sparse_messages)
    assert track_sparse.meta["activity_confidence"] < 0.2

    # High volume chat -> high confidence
    dense_messages = [ChatMessage(t=float(i), user=f"u{i}", text=f"msg {i}") for i in range(150)]
    track_dense = extract_chat_signal(sample_vod, dense_messages)
    assert track_dense.meta["activity_confidence"] >= 0.9


def test_hype_weighting_impact(sample_vod: VodMeta) -> None:
    # Compare plain text vs hype text
    plain_msgs = [
        ChatMessage(t=50.0 + i, user=f"u{i}", text="this is normal text") for i in range(5)
    ]
    hype_msgs = [
        ChatMessage(
            t=50.0 + i,
            user=f"u{i}",
            text="OMG LUL CLIP IT NO WAY WTF???",
            emotes=["LUL", "POG"],
        )
        for i in range(5)
    ]

    cfg = Config()
    track_plain = extract_chat_signal(sample_vod, plain_msgs, config=cfg)
    track_hype = extract_chat_signal(sample_vod, hype_msgs, config=cfg)

    plain_max = max(track_plain.values[45:55])
    hype_max = max(track_hype.values[45:55])
    assert hype_max > 1.5 * plain_max
