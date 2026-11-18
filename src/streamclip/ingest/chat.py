"""Chat replay parser and ingest for streamclip."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

from streamclip.interfaces import ChatSource
from streamclip.models import ChatMessage


class ChatParseError(Exception):
    """Raised when chat replay parsing fails."""


def _parse_emotes(raw_emotes: Any) -> list[str]:
    """Extract emote names or IDs from heterogeneous emote structures."""
    if not raw_emotes or not isinstance(raw_emotes, (list, tuple)):
        return []

    emotes: list[str] = []
    for item in raw_emotes:
        if isinstance(item, str) and item.strip():
            emotes.append(item.strip())
        elif isinstance(item, dict):
            name = item.get("name") or item.get("_id") or item.get("id") or item.get("code")
            if name and isinstance(name, str) and name.strip():
                emotes.append(name.strip())
    # Deduplicate while preserving order
    return list(dict.fromkeys(emotes))


def _parse_twitch_downloader_dict(data: dict[str, Any]) -> list[ChatMessage]:
    """Parse comments from a TwitchDownloader JSON payload."""
    comments = data.get("comments", [])
    if not isinstance(comments, list):
        return []

    messages: list[ChatMessage] = []
    seen: set[tuple[float, str, str]] = set()

    for item in comments:
        if not isinstance(item, dict):
            continue

        raw_offset = item.get("content_offset_seconds")
        if raw_offset is None:
            continue

        try:
            t = float(raw_offset)
        except (ValueError, TypeError):
            continue

        if math.isnan(t) or math.isinf(t) or t < 0.0:
            continue

        commenter = item.get("commenter")
        user = "anonymous"
        if isinstance(commenter, dict):
            user = (
                commenter.get("display_name")
                or commenter.get("name")
                or commenter.get("_id")
                or "anonymous"
            )

        msg_obj = item.get("message")
        text = ""
        emotes: list[str] = []
        if isinstance(msg_obj, dict):
            text = str(msg_obj.get("body", ""))
            emotes = _parse_emotes(msg_obj.get("emoticons"))
            # Fallback to fragments if emoticons is empty
            if not emotes:
                fragments = msg_obj.get("fragments", [])
                if isinstance(fragments, list):
                    frag_emotes: list[str] = []
                    for frag in fragments:
                        if isinstance(frag, dict) and "emoticon" in frag:
                            emo = frag["emoticon"]
                            if isinstance(emo, dict):
                                eid = emo.get("emoticon_id") or emo.get("id")
                                if eid:
                                    frag_emotes.append(str(eid))
                    emotes = list(dict.fromkeys(frag_emotes))

        key = (round(t, 4), user, text)
        if key in seen:
            continue
        seen.add(key)

        messages.append(ChatMessage(t=t, user=user, text=text, emotes=emotes))

    return messages


def _parse_chat_downloader_list(items: list[Any]) -> list[ChatMessage]:
    """Parse messages from a chat-downloader JSON array payload."""
    messages: list[ChatMessage] = []
    seen: set[tuple[float, str, str]] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        raw_t = item.get("time_in_seconds")
        if raw_t is None:
            raw_t = item.get("offset") or item.get("timestamp")
        if raw_t is None:
            continue

        try:
            t = float(raw_t)
        except (ValueError, TypeError):
            continue

        if math.isnan(t) or math.isinf(t) or t < 0.0:
            continue

        author = item.get("author")
        user = "anonymous"
        if isinstance(author, dict):
            user = author.get("name") or author.get("display_name") or "anonymous"
        elif isinstance(author, str) and author.strip():
            user = author.strip()

        text = str(item.get("message", ""))
        emotes = _parse_emotes(item.get("emotes"))

        key = (round(t, 4), user, text)
        if key in seen:
            continue
        seen.add(key)

        messages.append(ChatMessage(t=t, user=user, text=text, emotes=emotes))

    return messages


def _parse_csv_content(content: str) -> list[ChatMessage]:
    """Parse messages from CSV format: t,user,text."""
    messages: list[ChatMessage] = []
    seen: set[tuple[float, str, str]] = set()

    reader = csv.reader(io.StringIO(content))
    header = next(reader, None)
    if not header:
        return []

    # Map column indices with explicit integer defaults
    col_map: dict[str, int] = {col.strip().lower(): idx for idx, col in enumerate(header)}
    t_idx = col_map.get("t", col_map.get("time", 0))
    user_idx = col_map.get("user", col_map.get("author", 1))
    text_idx = col_map.get("text", col_map.get("message", 2))

    for row in reader:
        if not row or len(row) <= max(t_idx, user_idx, text_idx):
            continue

        raw_t = row[t_idx].strip()
        if not raw_t:
            continue

        try:
            t = float(raw_t)
        except ValueError:
            continue

        if math.isnan(t) or math.isinf(t) or t < 0.0:
            continue

        user = row[user_idx].strip() or "anonymous"
        text = row[text_idx]

        key = (round(t, 4), user, text)
        if key in seen:
            continue
        seen.add(key)

        messages.append(ChatMessage(t=t, user=user, text=text, emotes=[]))

    return messages


def parse_chat(source: Path | str) -> list[ChatMessage]:
    """Parse chat replay from TwitchDownloader JSON, chat-downloader JSON, or CSV.

    Auto-detects the format from file content and extension.
    Handles negative offsets (filtered), missing offsets (skipped),
    duplicates (deduplicated), bursts (order preserved), and empty chat.

    Args:
        source: File path or raw string content containing chat log.

    Returns:
        A time-sorted list of ChatMessage instances.

    Raises:
        ChatParseError: If the source cannot be read or parsed.
    """
    if isinstance(source, str) and not source.strip():
        return []

    content: str
    is_file_target = isinstance(source, Path) or (
        isinstance(source, str)
        and "\n" not in source
        and bool(source.strip())
        and (
            source.endswith((".json", ".csv", ".txt", ".log"))
            or "/" in source
            or "\\" in source
            or Path(source).is_file()
        )
    )

    if is_file_target:
        source_path = Path(source)
        if not source_path.is_file():
            raise ChatParseError(f"Failed to read chat source file: {source} not found")
        try:
            content = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError as err:
            raise ChatParseError(f"Failed to read chat source file {source}: {err}") from err
    else:
        content = str(source)

    content = content.replace("\x00", "")
    stripped = content.strip()
    if not stripped:
        return []

    messages: list[ChatMessage] = []

    # Format auto-detection
    is_json = stripped.startswith("{") or stripped.startswith("[")
    if is_json:
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as err:
            raise ChatParseError(f"Failed to parse chat source as JSON: {err}") from err

        if isinstance(parsed, dict):
            # TwitchDownloader format
            messages = _parse_twitch_downloader_dict(parsed)
        elif isinstance(parsed, list):
            # chat-downloader format
            messages = _parse_chat_downloader_list(parsed)
        else:
            raise ChatParseError(f"Unexpected JSON root type: {type(parsed).__name__}")
    else:
        # Fallback to CSV parser
        try:
            messages = _parse_csv_content(content)
        except Exception as err:
            raise ChatParseError(f"Failed to parse chat source as CSV: {err}") from err

    # Stable sort by timestamp
    messages.sort(key=lambda m: m.t)
    return messages


class ChatParser(ChatSource):
    """ChatSource protocol implementation wrapping parse_chat."""

    def load_messages(self, source: Path | str) -> list[ChatMessage]:
        """Load and parse chat messages from file or content."""
        return parse_chat(source)
