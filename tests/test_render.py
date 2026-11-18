"""Unit and integration tests for rendering: layout filtergraphs, captions, and encoding."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from streamclip.ingest.video import probe
from streamclip.models import FacecamBox, RenderSpec, Word
from streamclip.render import (
    RenderError,
    RenderJob,
    build_filtergraph,
    generate_ass_subtitles,
    mask_profane_word,
    render_clip,
    render_clips,
    resolve_facecam,
    select_video_encoder,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Layout Tests
# ---------------------------------------------------------------------------


def test_resolve_facecam() -> None:
    # 1. From spec
    spec_with_fc = RenderSpec(
        layout="stack",
        facecam=FacecamBox(x=100, y=200, w=300, h=400),
    )
    assert resolve_facecam(spec_with_fc) == FacecamBox(x=100, y=200, w=300, h=400)

    # 2. From config object
    class MockConfig:
        facecam = FacecamBox(x=10, y=20, w=30, h=40)

    spec_no_fc = RenderSpec(layout="stack")
    assert resolve_facecam(spec_no_fc, config=MockConfig()) == FacecamBox(x=10, y=20, w=30, h=40)

    # 3. Default fallback
    assert resolve_facecam(spec_no_fc, config=None) == FacecamBox(x=0, y=0, w=480, h=360)


def test_build_filtergraph_gameplay_only() -> None:
    spec = RenderSpec(layout="gameplay_only", out_w=1080, out_h=1920)
    fg = build_filtergraph(spec)
    assert "[0:v]scale=" in fg
    assert "crop=1080:1920" in fg
    assert "setsar=1[v]" in fg


def test_build_filtergraph_facecam_only() -> None:
    spec = RenderSpec(
        layout="facecam_only",
        facecam=FacecamBox(x=10, y=20, w=320, h=240),
        out_w=1080,
        out_h=1920,
    )
    fg = build_filtergraph(spec)
    assert "crop=320:240:10:20" in fg
    assert "crop=1080:1920" in fg
    assert "setsar=1[v]" in fg


def test_build_filtergraph_stack() -> None:
    spec = RenderSpec(
        layout="stack",
        facecam=FacecamBox(x=50, y=50, w=400, h=300),
        out_w=1080,
        out_h=1920,
    )
    fg = build_filtergraph(spec, top_split_ratio=0.5)
    assert "[0:v]split=2[v_game][v_cam]" in fg
    assert "crop=1080:960,setsar=1[top]" in fg
    assert "crop=400:300:50:50" in fg
    assert "crop=1080:960,setsar=1[bottom]" in fg
    assert "[top][bottom]vstack=inputs=2[v]" in fg


def test_build_filtergraph_with_ass_subtitles(tmp_path: Path) -> None:
    sub_file = tmp_path / "subs.ass"
    sub_file.write_text("dummy", encoding="utf-8")
    spec = RenderSpec(layout="gameplay_only", out_w=1080, out_h=1920)
    fg = build_filtergraph(spec, ass_path=sub_file)
    assert "ass=filename=" in fg
    assert "[v]" in fg


def test_build_filtergraph_invalid_layout() -> None:
    spec = RenderSpec(layout="gameplay_only")
    spec.layout = "invalid_layout"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown render layout"):
        build_filtergraph(spec)


# ---------------------------------------------------------------------------
# Captions Tests
# ---------------------------------------------------------------------------


def test_mask_profane_word() -> None:
    assert mask_profane_word("fuck") == "f**k"
    assert mask_profane_word("Fucking!") == "F*****g!"
    assert mask_profane_word("shit") == "s**t"
    assert mask_profane_word("bitch") == "b***h"
    assert mask_profane_word("awesome") == "awesome"
    assert mask_profane_word("hello,") == "hello,"


def test_generate_ass_subtitles_empty() -> None:
    ass_content = generate_ass_subtitles([])
    assert "[Script Info]" in ass_content
    assert "PlayResX: 1080" in ass_content
    assert "PlayResY: 1920" in ass_content
    assert "MarginV" in ass_content


def test_generate_ass_subtitles_with_words_and_highlight(tmp_path: Path) -> None:
    words = [
        Word(start=1.0, end=1.5, text="Look"),
        Word(start=1.6, end=2.0, text="at"),
        Word(start=2.1, end=2.8, text="that"),
        Word(start=3.0, end=3.5, text="fuck!"),
    ]

    out_file = tmp_path / "output.ass"
    content = generate_ass_subtitles(
        words,
        out_path=out_file,
        clip_start=1.0,
        mask_profanity=True,
    )
    assert out_file.is_file()

    # Mobile safe margins
    assert "420,1" in content  # MarginV=420

    # Word highlight tags present
    assert r"{\c&H0000FFFF&}" in content
    assert r"{\c&H00FFFFFF&}" in content

    # Profanity masked
    assert "f**k!" in content
    assert "fuck!" not in content

    # Normalized clip start offset: word 1 starts at 0.0s (1.0 - 1.0)
    assert "Dialogue: 0,0:00:00.00,0:00:00.50" in content


# ---------------------------------------------------------------------------
# Encoder Tests & Hardware Detection
# ---------------------------------------------------------------------------


def test_select_video_encoder() -> None:
    # When NVENC is mocked as True
    with patch("streamclip.render.encode.is_nvenc_available", return_value=True):
        assert select_video_encoder(prefer_nvenc=True) == "h264_nvenc"
        assert select_video_encoder(prefer_nvenc=False) == "libx264"

    # When NVENC is mocked as False
    with patch("streamclip.render.encode.is_nvenc_available", return_value=False):
        assert select_video_encoder(prefer_nvenc=True) == "libx264"
        assert select_video_encoder(prefer_nvenc=False) == "libx264"


def test_render_clip_invalid_duration(tmp_path: Path) -> None:
    job = RenderJob(
        vod=FIXTURES_DIR / "video.mp4",
        start=10.0,
        end=10.0,  # Zero duration
        output_path=tmp_path / "out.mp4",
    )
    with pytest.raises(ValueError, match="duration must be positive"):
        render_clip(job)


# ---------------------------------------------------------------------------
# Integration Tests with Fixture Video & ffprobe Inspection
# ---------------------------------------------------------------------------


def test_render_fixture_gameplay_only_with_probe(tmp_path: Path) -> None:
    """Render fixture clip, checking resolution, duration (+-0.1s), and audio via ffprobe."""
    video_path = FIXTURES_DIR / "video.mp4"
    assert video_path.is_file(), "Fixture video.mp4 must exist"

    out_mp4 = tmp_path / "rendered_gameplay.mp4"
    job = RenderJob(
        vod=video_path,
        start=10.0,
        end=15.0,  # 5.0 seconds
        output_path=out_mp4,
        spec=RenderSpec(layout="gameplay_only", out_w=1080, out_h=1920),
        prefer_nvenc=False,  # Force libx264 for deterministic test execution
    )

    res_path = render_clip(job)
    assert res_path.is_file()

    # Verify via ffprobe probe
    meta = probe(res_path)
    assert meta.width == 1080
    assert meta.height == 1920
    assert abs(meta.duration_s - 5.0) <= 0.1
    assert meta.has_audio is True


def test_render_fixture_stack_with_captions_and_probe(tmp_path: Path) -> None:
    """Render vertical stack layout with facecam and burned-in word-level captions."""
    video_path = FIXTURES_DIR / "video.mp4"
    out_mp4 = tmp_path / "rendered_stack.mp4"

    captions = [
        Word(start=20.0, end=21.0, text="Incredible"),
        Word(start=21.2, end=22.5, text="moment!"),
    ]

    job = RenderJob(
        vod=video_path,
        start=20.0,
        end=24.0,  # 4.0 seconds
        output_path=out_mp4,
        spec=RenderSpec(
            layout="stack",
            facecam=FacecamBox(x=10, y=10, w=60, h=40),
            captions=captions,
            out_w=1080,
            out_h=1920,
        ),
        prefer_nvenc=False,
    )

    res_path = render_clip(job)
    assert res_path.is_file()

    meta = probe(res_path)
    assert meta.width == 1080
    assert meta.height == 1920
    assert abs(meta.duration_s - 4.0) <= 0.1
    assert meta.has_audio is True


def test_render_fixture_no_audio_synthesizes_audio(tmp_path: Path) -> None:
    """Render fixture lacking audio, verifying synthesized AAC audio stream in output."""
    no_audio_video = FIXTURES_DIR / "video_no_audio.mp4"
    assert no_audio_video.is_file()

    out_mp4 = tmp_path / "rendered_no_audio.mp4"
    job = RenderJob(
        vod=no_audio_video,
        start=5.0,
        end=8.0,  # 3.0 seconds
        output_path=out_mp4,
        spec=RenderSpec(layout="gameplay_only", out_w=1080, out_h=1920),
        prefer_nvenc=False,
    )

    res_path = render_clip(job)
    assert res_path.is_file()

    meta = probe(res_path)
    assert meta.width == 1080
    assert meta.height == 1920
    assert abs(meta.duration_s - 3.0) <= 0.1
    assert meta.has_audio is True


def test_render_clips_parallel(tmp_path: Path) -> None:
    """Render multiple clips concurrently with worker limit."""
    video_path = FIXTURES_DIR / "video.mp4"

    job1 = RenderJob(
        vod=video_path,
        start=1.0,
        end=3.0,  # 2.0s
        output_path=tmp_path / "clip1.mp4",
        spec=RenderSpec(layout="gameplay_only", out_w=1080, out_h=1920),
        prefer_nvenc=False,
    )
    job2 = RenderJob(
        vod=video_path,
        start=4.0,
        end=6.0,  # 2.0s
        output_path=tmp_path / "clip2.mp4",
        spec=RenderSpec(layout="gameplay_only", out_w=1080, out_h=1920),
        prefer_nvenc=False,
    )

    results = render_clips([job1, job2], max_workers=2)
    assert len(results) == 2
    for p in results:
        assert p.is_file()
        m = probe(p)
        assert m.width == 1080
        assert m.height == 1920
        assert abs(m.duration_s - 2.0) <= 0.1
        assert m.has_audio is True

    # Empty list
    assert render_clips([]) == []


def test_render_clip_ffmpeg_error_raises_render_error(tmp_path: Path) -> None:
    """Test that failed ffmpeg execution raises typed RenderError."""
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"not a valid video")

    job = RenderJob(
        vod=corrupt_file,
        start=0.0,
        end=2.0,
        output_path=tmp_path / "out.mp4",
        prefer_nvenc=False,
    )
    with pytest.raises(RenderError, match="FFmpeg render failed"):
        render_clip(job)
