"""Unit tests for chat replay parsing and format auto-detection."""

import json
from pathlib import Path

import pytest

from streamclip.ingest.chat import ChatParseError, ChatParser, parse_chat
from streamclip.interfaces import ChatSource

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_chat_parser_conforms_to_protocol() -> None:
    parser = ChatParser()
    assert isinstance(parser, ChatSource)


def test_parse_twitch_downloader_fixture() -> None:
    path = FIXTURES_DIR / "chat_twitch.json"
    messages = parse_chat(path)
    assert len(messages) > 100

    # Ensure strictly sorted by timestamp
    for i in range(len(messages) - 1):
        assert messages[i].t <= messages[i + 1].t

    # Verify attributes
    first = messages[0]
    assert first.t >= 0.0
    assert first.user != ""
    assert first.text != ""


def test_parse_chat_downloader_fixture() -> None:
    path = FIXTURES_DIR / "chat_downloader.json"
    messages = parse_chat(path)
    assert len(messages) > 100

    for i in range(len(messages) - 1):
        assert messages[i].t <= messages[i + 1].t

    first = messages[0]
    assert first.t >= 0.0
    assert first.user != ""


def test_parse_csv_fixture() -> None:
    path = FIXTURES_DIR / "chat.csv"
    messages = parse_chat(path)
    assert len(messages) > 100

    for i in range(len(messages) - 1):
        assert messages[i].t <= messages[i + 1].t


def test_parse_empty_chat_variants() -> None:
    variants_dir = FIXTURES_DIR / "variants"
    assert parse_chat(variants_dir / "no_chat_twitch.json") == []
    assert parse_chat(variants_dir / "no_chat_downloader.json") == []
    assert parse_chat(variants_dir / "no_chat.csv") == []
    assert parse_chat("") == []
    assert parse_chat("   \n\t  ") == []


def test_parse_chat_with_long_gaps_variant() -> None:
    variants_dir = FIXTURES_DIR / "variants"
    messages = parse_chat(variants_dir / "chat_long_gaps_downloader.json")
    assert len(messages) > 0

    # Verify time sorted
    for i in range(len(messages) - 1):
        assert messages[i].t <= messages[i + 1].t

    # Verify a long gap (>100s) exists
    diffs = [messages[i + 1].t - messages[i].t for i in range(len(messages) - 1)]
    assert any(d > 100.0 for d in diffs)


def test_parse_non_ascii_variant() -> None:
    variants_dir = FIXTURES_DIR / "variants"
    messages = parse_chat(variants_dir / "chat_non_ascii_twitch.json")
    assert len(messages) > 0

    # Ensure unicode emotes / text are retained
    has_emoji = any(any(ord(c) > 127 for c in m.text) for m in messages)
    assert has_emoji


def test_negative_offsets_filtered() -> None:
    raw_twitch = {
        "comments": [
            {"content_offset_seconds": -15.0, "message": {"body": "pre-stream"}},
            {"content_offset_seconds": -0.1, "message": {"body": "still pre-stream"}},
            {"content_offset_seconds": 0.0, "message": {"body": "stream start"}},
            {"content_offset_seconds": 10.5, "message": {"body": "gameplay"}},
        ]
    }
    messages = parse_chat(json.dumps(raw_twitch))
    assert len(messages) == 2
    assert messages[0].t == 0.0
    assert messages[0].text == "stream start"
    assert messages[1].t == 10.5


def test_missing_offsets_skipped() -> None:
    raw_downloader = [
        {"time_in_seconds": None, "message": "no offset"},
        {"time_in_seconds": "invalid_num", "message": "bad offset"},
        {"time_in_seconds": 5.0, "message": "valid offset"},
    ]
    messages = parse_chat(json.dumps(raw_downloader))
    assert len(messages) == 1
    assert messages[0].t == 5.0
    assert messages[0].text == "valid offset"


def test_duplicates_deduplicated() -> None:
    csv_content = """t,user,text
10.0,user1,duplicate message
10.0,user1,duplicate message
10.0,user1,duplicate message
10.0,user2,different user message
"""
    messages = parse_chat(csv_content)
    assert len(messages) == 2
    assert messages[0].user == "user1"
    assert messages[1].user == "user2"


def test_bursts_order_preserved() -> None:
    csv_content = """t,user,text
50.0,userA,first in burst
50.0,userB,second in burst
50.0,userC,third in burst
50.0,userD,fourth in burst
"""
    messages = parse_chat(csv_content)
    assert len(messages) == 4
    users = [m.user for m in messages]
    assert users == ["userA", "userB", "userC", "userD"]


def test_invalid_json_raises_error() -> None:
    with pytest.raises(ChatParseError, match="Failed to parse chat source as JSON"):
        parse_chat("{ invalid json content [")


def test_non_existent_file_raises_error() -> None:
    with pytest.raises(ChatParseError, match="Failed to read chat source"):
        parse_chat("non_existent_file.json")
