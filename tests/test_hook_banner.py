"""Unit tests for ASS subtitle generation with dynamic hook title banner and layout margins."""

from __future__ import annotations

from pathlib import Path

from streamclip.models import Word
from streamclip.render.captions import generate_ass_subtitles


def test_generate_ass_with_hook_banner(tmp_path: Path) -> None:
    words = [
        Word(start=0.5, end=1.0, text="Hello"),
        Word(start=1.2, end=1.8, text="chat"),
    ]
    out_file = tmp_path / "banner.ass"

    content = generate_ass_subtitles(
        words,
        out_path=out_file,
        clip_start=0.0,
        hook_title="YOU WON'T BELIEVE THIS",
        layout="stack",
    )

    assert out_file.is_file()

    # Verify HookBanner style definition
    assert "Style: HookBanner" in content
    # Top-center alignment is 8 in ASS style
    assert ",8,80,80,140,1" in content

    # Verify HookBanner dialogue event (0.0s to 3.5s)
    assert "0:00:00.00,0:00:03.50,HookBanner,,0,0,0,,{\\b1}YOU WON'T BELIEVE THIS{\\b0}" in content

    # Verify stack layout vertical margin for speech captions (above the 50/50 split)
    assert ",920,1" in content


def test_generate_ass_without_hook_banner() -> None:
    words = [Word(start=1.0, end=1.5, text="test")]
    content = generate_ass_subtitles(words, hook_title=None, layout="gameplay_only")

    assert "HookBanner" not in content
    # In gameplay_only, captions are in the lower third (MarginV=420)
    assert ",420,1" in content


def test_hook_banner_with_profanity_masking() -> None:
    words = [Word(start=0.5, end=1.0, text="clean")]
    content = generate_ass_subtitles(
        words,
        hook_title="WHAT THE FUCK HAPPENED",
        mask_profanity=True,
    )
    assert "WHAT THE F**K HAPPENED" in content
    assert "WHAT THE FUCK HAPPENED" not in content
