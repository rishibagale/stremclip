"""Unit tests for streamclip data models."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from streamclip.models import (
    Candidate,
    ChatMessage,
    FacecamBox,
    RankedClip,
    RenderSpec,
    SignalTrack,
    VodMeta,
    Word,
)


def test_vod_meta_initialization() -> None:
    meta = VodMeta(
        path=Path("videos/gta6_vod.mp4"),
        duration_s=120.4,
        fps=60.0,
        width=1920,
        height=1080,
        has_audio=True,
    )
    assert isinstance(meta.path, Path)
    assert meta.duration_s == 120.4
    assert meta.expected_signal_length == math.ceil(120.4)
    assert meta.expected_signal_length == 121

    # Also verify validation from dict with string path
    meta_from_dict = VodMeta.model_validate(
        {
            "path": "videos/gta6_vod.mp4",
            "duration_s": 10.0,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
        }
    )
    assert isinstance(meta_from_dict.path, Path)


def test_chat_message() -> None:
    msg = ChatMessage(t=45.2, user="streamfan", text="INSANE SHOT!", emotes=["LUL"])
    assert msg.t == 45.2
    assert msg.user == "streamfan"
    assert msg.emotes == ["LUL"]


def test_word() -> None:
    word = Word(start=1.2, end=1.8, text="welcome")
    assert word.start == 1.2
    assert word.end == 1.8
    assert word.text == "welcome"


def test_signal_track_1hz_invariant() -> None:
    vod = VodMeta(
        path=Path("sample.mp4"),
        duration_s=10.2,
        fps=30.0,
        width=1920,
        height=1080,
    )
    expected_len = math.ceil(vod.duration_s)
    assert expected_len == 11

    raw_values = [float(i) for i in range(expected_len)]
    track = SignalTrack(name="chat_rate", hz=1.0, values=raw_values)
    assert track.hz == 1.0
    assert len(track.values) == 11
    assert track.matches_vod(vod)

    # Check numpy conversion
    np_arr = track.array
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.shape == (11,)
    assert np_arr[5] == 5.0

    # Invariant failure when length does not match ceil(duration_s)
    short_track = SignalTrack(name="chat_rate", hz=1.0, values=[1.0, 2.0])
    assert not short_track.matches_vod(vod)


def test_signal_track_coerces_numpy_input() -> None:
    arr = np.array([0.1, 0.4, 0.9], dtype=np.float32)
    track = SignalTrack(name="audio_energy", hz=1.0, values=arr)  # type: ignore[arg-type]
    assert isinstance(track.values, list)
    assert len(track.values) == 3
    assert pytest.approx(track.values[1]) == 0.4


def test_candidate_and_ranked_clip() -> None:
    cand = Candidate(
        peak_t=50.0,
        start=35.0,
        end=54.0,
        score=2.5,
        components={"chat": 1.5, "audio": 1.0},
    )
    assert cand.duration == 19.0

    ranked = RankedClip(
        candidate=cand,
        llm_score=8.5,
        title="High Speed Chase in Vice City",
        reason="Rapid chat spike paired with loud siren audio",
        final_score=5.5,
    )
    assert ranked.title.startswith("High Speed")
    assert ranked.candidate.peak_t == 50.0


def test_facecam_box() -> None:
    box = FacecamBox(x=1440, y=0, w=480, h=360)
    assert box.x == 1440
    assert box.h == 360


def test_render_spec_valid_layouts() -> None:
    spec = RenderSpec(
        layout="stack",
        facecam=FacecamBox(x=0, y=0, w=300, h=200),
        captions=[Word(start=0.0, end=0.5, text="Look!")],
    )
    assert spec.layout == "stack"
    assert spec.out_w == 1080
    assert spec.out_h == 1920
    assert len(spec.captions) == 1

    spec2 = RenderSpec(layout="gameplay_only")
    assert spec2.layout == "gameplay_only"
    assert spec2.facecam is None


def test_render_spec_invalid_layout() -> None:
    with pytest.raises(ValidationError):
        RenderSpec(layout="picture_in_picture")  # type: ignore[arg-type]
