"""Tests verifying synthetic fixtures and deterministic regeneration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from streamclip.models import ChatMessage, VodMeta
from tests.fixtures.make_fixtures import (
    GROUND_TRUTH_MOMENTS,
    generate_all_fixtures,
)

FIXTURES_DIR = Path(__file__).parent


def test_labels_structure() -> None:
    labels_file = FIXTURES_DIR / "labels.json"
    assert labels_file.is_file(), "labels.json must exist in fixtures"

    data = json.loads(labels_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 5

    prev_end = 0.0
    for item in data:
        assert "t_start" in item and "t_end" in item
        assert item["t_start"] < item["t_end"]
        assert item["t_start"] >= prev_end
        prev_end = item["t_end"]

    assert data == GROUND_TRUTH_MOMENTS


def test_deterministic_regeneration(tmp_path: Path) -> None:
    """Verify that regenerating fixtures with seed=42 produces bit-for-bit identical files."""
    # Generate into a temporary directory without videos
    generate_all_fixtures(output_dir=tmp_path, seed=42, generate_videos=False)

    files_to_check = [
        "labels.json",
        "chat_twitch.json",
        "chat_downloader.json",
        "chat.csv",
        "variants/no_chat_twitch.json",
        "variants/no_chat_downloader.json",
        "variants/no_chat.csv",
        "variants/chat_long_gaps_twitch.json",
        "variants/chat_long_gaps_downloader.json",
        "variants/chat_long_gaps.csv",
        "variants/chat_non_ascii_twitch.json",
        "variants/chat_non_ascii_downloader.json",
        "variants/chat_non_ascii.csv",
    ]

    for rel_path in files_to_check:
        generated_file = tmp_path / rel_path
        committed_file = FIXTURES_DIR / rel_path

        assert generated_file.is_file(), f"Missing generated file: {rel_path}"
        assert committed_file.is_file(), f"Missing committed fixture: {rel_path}"

        gen_bytes = generated_file.read_bytes()
        com_bytes = committed_file.read_bytes()
        assert gen_bytes == com_bytes, f"Mismatch in deterministically regenerated {rel_path}"


def test_chat_csv_format() -> None:
    csv_file = FIXTURES_DIR / "chat.csv"
    assert csv_file.is_file()

    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["t", "user", "text"]

        rows = list(reader)
        assert len(rows) > 0

        # Verify rows convert to ChatMessage models cleanly
        for row in rows[:50]:
            t_val = float(row[0])
            msg = ChatMessage(t=t_val, user=row[1], text=row[2], emotes=[])
            assert 0.0 <= msg.t <= 600.0


def test_twitch_downloader_format() -> None:
    json_file = FIXTURES_DIR / "chat_twitch.json"
    assert json_file.is_file()

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "streamer" in data
    assert "comments" in data
    comments = data["comments"]
    assert len(comments) > 0

    first = comments[0]
    assert "content_offset_seconds" in first
    assert "commenter" in first
    assert "message" in first

    msg = ChatMessage(
        t=float(first["content_offset_seconds"]),
        user=str(first["commenter"]["display_name"]),
        text=str(first["message"]["body"]),
        emotes=[e["_id"] for e in first["message"].get("emoticons", [])],
    )
    assert msg.t >= 0.0


def test_chat_downloader_format() -> None:
    json_file = FIXTURES_DIR / "chat_downloader.json"
    assert json_file.is_file()

    items = json.loads(json_file.read_text(encoding="utf-8"))
    assert isinstance(items, list)
    assert len(items) > 0

    first = items[0]
    assert "time_in_seconds" in first
    assert "author" in first
    assert "message" in first

    msg = ChatMessage(
        t=float(first["time_in_seconds"]),
        user=str(first["author"]["name"]),
        text=str(first["message"]),
        emotes=[e["name"] for e in first.get("emotes", [])],
    )
    assert msg.t >= 0.0


def test_video_fixtures_metadata() -> None:
    video_file = FIXTURES_DIR / "video.mp4"
    assert video_file.is_file(), "video.mp4 must exist in fixtures"

    vod = VodMeta(
        path=video_file,
        duration_s=600.0,
        fps=1.0,
        width=160,
        height=120,
        has_audio=True,
    )
    assert vod.expected_signal_length == 600
    assert vod.has_audio is True

    no_audio_file = FIXTURES_DIR / "video_no_audio.mp4"
    assert no_audio_file.is_file(), "video_no_audio.mp4 must exist in fixtures"

    vod_no_audio = VodMeta(
        path=no_audio_file,
        duration_s=600.0,
        fps=1.0,
        width=160,
        height=120,
        has_audio=False,
    )
    assert vod_no_audio.has_audio is False


def test_variants_structure() -> None:
    variants_dir = FIXTURES_DIR / "variants"

    # Empty chat variant
    empty_csv = variants_dir / "no_chat.csv"
    assert empty_csv.is_file()
    assert empty_csv.read_text(encoding="utf-8").strip() == "t,user,text"

    empty_twitch = json.loads((variants_dir / "no_chat_twitch.json").read_text(encoding="utf-8"))
    assert len(empty_twitch["comments"]) == 0

    empty_downloader = json.loads(
        (variants_dir / "no_chat_downloader.json").read_text(encoding="utf-8")
    )
    assert len(empty_downloader) == 0

    # Long gaps variant
    gap_file = variants_dir / "chat_long_gaps_downloader.json"
    gap_data = json.loads(gap_file.read_text(encoding="utf-8"))
    timestamps = [item["time_in_seconds"] for item in gap_data]
    diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    # Verify that a long gap (> 100s) exists
    assert any(d > 100.0 for d in diffs)

    # Non-ASCII variant
    unicode_csv = (variants_dir / "chat_non_ascii.csv").read_text(encoding="utf-8")
    # Verify presence of non-ASCII characters
    assert any(ord(c) > 127 for c in unicode_csv)
