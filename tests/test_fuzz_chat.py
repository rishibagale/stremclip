"""Fuzz and edge-case regression tests for chat replay parsing and CLI inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from streamclip.cli import app
from streamclip.ingest.chat import ChatParseError, parse_chat

runner = CliRunner()


def test_fuzz_chat_nan_inf_offsets(tmp_path: Path) -> None:
    """Ensure NaN and Inf offsets in Twitch, chat-downloader, and CSV do not crash parser."""
    # 1. Twitch format with NaN and Inf
    twitch_json = {
        "comments": [
            {"content_offset_seconds": "nan", "message": {"body": "nan msg"}},
            {"content_offset_seconds": float("nan"), "message": {"body": "nan float"}},
            {"content_offset_seconds": "inf", "message": {"body": "inf msg"}},
            {"content_offset_seconds": float("inf"), "message": {"body": "inf float"}},
            {"content_offset_seconds": "-inf", "message": {"body": "-inf msg"}},
            {"content_offset_seconds": -5.0, "message": {"body": "negative offset"}},
            {"content_offset_seconds": 12.5, "message": {"body": "valid message"}},
        ]
    }
    f_twitch = tmp_path / "twitch_fuzz.json"
    import json

    f_twitch.write_text(json.dumps(twitch_json), encoding="utf-8")
    msgs = parse_chat(f_twitch)
    assert len(msgs) == 1
    assert msgs[0].t == 12.5
    assert msgs[0].text == "valid message"

    # 2. Chat-downloader format with NaN and Inf
    cd_json = [
        {"time_in_seconds": "NaN", "message": "skip me"},
        {"time_in_seconds": float("inf"), "message": "skip me too"},
        {"time_in_seconds": 42.0, "message": "keep me", "author": "tester"},
    ]
    f_cd = tmp_path / "cd_fuzz.json"
    f_cd.write_text(json.dumps(cd_json), encoding="utf-8")
    msgs_cd = parse_chat(f_cd)
    assert len(msgs_cd) == 1
    assert msgs_cd[0].t == 42.0

    # 3. CSV with NaN and Inf
    csv_content = "t,user,text\nnan,user1,bad\ninf,user2,bad\n-10,user3,neg\n15.5,user4,good\n"
    f_csv = tmp_path / "fuzz.csv"
    f_csv.write_text(csv_content, encoding="utf-8")
    msgs_csv = parse_chat(f_csv)
    assert len(msgs_csv) == 1
    assert msgs_csv[0].t == 15.5


def test_fuzz_chat_null_bytes_and_surrogates(tmp_path: Path) -> None:
    """Ensure NUL bytes, malformed characters, and surrogates don't crash parser."""
    csv_null = "t,user,text\n1.0,alice,hello\x00world\n2.0,bob,foo\x00\x00bar\n"
    f_null = tmp_path / "null_bytes.csv"
    f_null.write_text(csv_null, encoding="utf-8")
    msgs = parse_chat(f_null)
    assert len(msgs) == 2
    assert "world" in msgs[0].text


def test_fuzz_chat_malformed_json_and_csv(tmp_path: Path) -> None:
    """Test handling of empty, truncated, or corrupted files."""
    # Completely empty file
    f_empty = tmp_path / "empty.json"
    f_empty.write_text("", encoding="utf-8")
    assert parse_chat(f_empty) == []

    # Whitespace only
    f_ws = tmp_path / "ws.json"
    f_ws.write_text("   \n\t  ", encoding="utf-8")
    assert parse_chat(f_ws) == []

    # Truncated JSON
    f_trunc = tmp_path / "trunc.json"
    f_trunc.write_text('{"comments": [{"content_offset_seconds": 12.0', encoding="utf-8")
    with pytest.raises(ChatParseError, match="Failed to parse chat source as JSON"):
        parse_chat(f_trunc)

    # Missing file
    with pytest.raises(ChatParseError, match="not found"):
        parse_chat(tmp_path / "does_not_exist.json")


def test_fuzz_cli_missing_files_and_invalid_inputs(tmp_path: Path) -> None:
    """Test CLI commands resilience to missing files, negative K, and invalid config."""
    # 1. run with missing video
    res = runner.invoke(app, ["run", str(tmp_path / "non_existent.mp4")])
    assert res.exit_code == 1
    assert "not found" in res.output

    # 2. detect with missing video
    res_det = runner.invoke(app, ["detect", str(tmp_path / "non_existent.mp4")])
    assert res_det.exit_code == 1
    assert "not found" in res_det.output

    # 3. render with missing candidates
    res_rnd = runner.invoke(app, ["render", str(tmp_path / "no_cands.json"), "-v", "test.mp4"])
    assert res_rnd.exit_code == 1
    assert "not found" in res_rnd.output

    # 4. eval with missing labels
    res_ev = runner.invoke(app, ["eval", "-l", str(tmp_path / "no_labels.json")])
    assert res_ev.exit_code == 1
    assert "not found" in res_ev.output

    # 5. tune with missing video
    res_tn = runner.invoke(
        app,
        [
            "tune",
            "-l",
            str(tmp_path / "labels.json"),
            "-v",
            str(tmp_path / "no_vid.mp4"),
        ],
    )
    assert res_tn.exit_code == 1
    assert "not found" in res_tn.output
