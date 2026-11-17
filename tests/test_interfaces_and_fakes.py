"""Unit tests verifying interfaces protocols and fake implementations."""

import math
from pathlib import Path

from streamclip.interfaces import ChatSource, Reranker, SignalExtractor, Transcriber, VisionScorer
from streamclip.models import Candidate, VodMeta
from tests.fakes import (
    FakeChatSource,
    FakeReranker,
    FakeSignalExtractor,
    FakeTranscriber,
    FakeVisionScorer,
)


def test_fake_chat_source_implements_protocol() -> None:
    fake = FakeChatSource()
    assert isinstance(fake, ChatSource)

    messages = fake.load_messages("dummy_path.json")
    assert len(messages) >= 3
    assert messages[0].user == "viewer1"


def test_fake_signal_extractor_implements_protocol_and_invariant() -> None:
    extractor = FakeSignalExtractor(name="fake_audio", constant_value=42.0)
    assert isinstance(extractor, SignalExtractor)
    assert extractor.name == "fake_audio"

    vod = VodMeta(
        path=Path("fake_gta.mp4"),
        duration_s=63.2,
        fps=60.0,
        width=1920,
        height=1080,
    )
    signal = extractor.extract(vod)

    # Invariant: length must be ceil(duration_s) and hz=1.0
    expected_len = math.ceil(vod.duration_s)
    assert expected_len == 64
    assert signal.hz == 1.0
    assert len(signal.values) == 64
    assert signal.values[0] == 42.0
    assert signal.matches_vod(vod)


def test_fake_vision_scorer_implements_protocol() -> None:
    scorer = FakeVisionScorer(default_score=0.9)
    assert isinstance(scorer, VisionScorer)

    vod = VodMeta(path=Path("v.mp4"), duration_s=100.0, fps=30.0, width=1920, height=1080)
    cand = Candidate(peak_t=50.0, start=35.0, end=55.0, score=2.0)
    scored = scorer.score_candidates(vod, [cand])

    assert len(scored) == 1
    assert scored[0].components["vision"] == 0.9
    assert scored[0].score > 2.0


def test_fake_reranker_implements_protocol() -> None:
    reranker = FakeReranker(default_llm_score=9.5)
    assert isinstance(reranker, Reranker)

    cand = Candidate(peak_t=30.0, start=15.0, end=35.0, score=3.0)
    ranked = reranker.rerank([cand])

    assert len(ranked) == 1
    assert ranked[0].llm_score == 9.5
    assert "Highlight #1" in ranked[0].title
    assert ranked[0].final_score > 0.0


def test_fake_transcriber_implements_protocol() -> None:
    transcriber = FakeTranscriber()
    assert isinstance(transcriber, Transcriber)

    vod = VodMeta(path=Path("v.mp4"), duration_s=60.0, fps=30.0, width=1920, height=1080)
    all_words = transcriber.transcribe(vod)
    assert len(all_words) == 3

    window_words = transcriber.transcribe(vod, start=10.5, end=11.5)
    assert len(window_words) == 1
    assert window_words[0].text == "my"
