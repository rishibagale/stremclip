"""ASS subtitle generation with word-level highlight, mobile-safe margins, and profanity masking."""

from __future__ import annotations

import re
from pathlib import Path

from streamclip.models import Word

PROFANITY_WORDS = {
    "fuck",
    "fucking",
    "fucked",
    "fucker",
    "fuckers",
    "fucks",
    "shit",
    "shits",
    "shitting",
    "bullshit",
    "bitch",
    "bitches",
    "bitching",
    "asshole",
    "assholes",
    "cunt",
    "cunts",
    "dick",
    "dicks",
    "pussy",
    "pussies",
    "bastard",
    "bastards",
    "damn",
    "cock",
    "cocks",
    "nigger",
    "nigga",
    "retard",
    "retarded",
    "fag",
    "faggot",
}


def mask_profane_word(text: str) -> str:
    """Mask profane word with asterisks, preserving punctuation and case.

    Examples:
        'fuck' -> 'f**k'
        'Fucking!' -> 'F*****g!'
        'shit' -> 'sh*t'
    """
    match = re.match(r"^([^\w]*)([\w]+)([^\w]*)$", text)
    if not match:
        return text

    prefix, core, suffix = match.groups()
    lower_core = core.lower()

    if lower_core in PROFANITY_WORDS:
        if len(core) <= 2:
            masked_core = core[0] + "*"
        elif len(core) == 3:
            masked_core = core[0] + "*" + core[-1]
        else:
            masked_core = core[0] + ("*" * (len(core) - 2)) + core[-1]
        return f"{prefix}{masked_core}{suffix}"

    return text


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds into ASS timestamp string: H:MM:SS.cs."""
    s = max(0.0, seconds)
    hrs = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    cs = round((s - int(s)) * 100)
    if cs >= 100:
        cs = 0
        secs += 1
        if secs >= 60:
            secs = 0
            mins += 1
            if mins >= 60:
                mins = 0
                hrs += 1
    return f"{hrs}:{mins:02d}:{secs:02d}.{cs:02d}"


def chunk_words(
    words: list[Word],
    max_words: int = 5,
    max_duration_s: float = 2.5,
    min_pause_s: float = 0.5,
) -> list[list[Word]]:
    """Group timed words into short, legible lines for vertical video screens."""
    if not words:
        return []

    chunks: list[list[Word]] = []
    current_chunk: list[Word] = []

    for word in words:
        if not current_chunk:
            current_chunk.append(word)
            continue

        prev_word = current_chunk[-1]
        pause = word.start - prev_word.end
        duration = word.end - current_chunk[0].start

        if len(current_chunk) >= max_words or pause >= min_pause_s or duration > max_duration_s:
            chunks.append(current_chunk)
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def generate_ass_subtitles(
    words: list[Word],
    out_path: Path | str | None = None,
    clip_start: float = 0.0,
    mask_profanity: bool = True,
    highlight_color: str = "&H0000FFFF&",
    default_color: str = "&H00FFFFFF&",
    font_size: int = 68,
    margin_v: int = 420,
    margin_lr: int = 100,
    hook_title: str | None = None,
    hook_duration_s: float = 3.5,
    layout: str = "gameplay_only",
) -> str:
    """Generate an Advanced SubStation Alpha (.ass) subtitle script with word-level highlight.

    Designed for 1080x1920 vertical video with mobile-safe margins (safe zone above
    TikTok/Reels/Shorts navigation and comment overlays) and optional viral hook banner.

    Args:
        words: List of timed Word objects.
        out_path: Optional file path to write the .ass file.
        clip_start: Timestamp offset of clip start in seconds (normalized to 0.0).
        mask_profanity: Whether to mask swear words (e.g. 'fuck' -> 'f**k').
        highlight_color: ASS color tag for active word (default yellow: &H0000FFFF&).
        default_color: ASS color tag for non-active words (default white: &H00FFFFFF&).
        font_size: Font size in pixels (default 68 for 1080x1920).
        margin_v: Vertical margin from bottom in pixels (default 420px; 920px for stack).
        margin_lr: Left and right margins in pixels (default 100px).
        hook_title: Optional 3-5 word attention hook title banner for first few seconds.
        hook_duration_s: Duration in seconds to display the hook banner (default 3.5s).
        layout: Vertical layout type ('gameplay_only', 'stack', 'facecam_only').

    Returns:
        Full text content of the generated .ass subtitle file.
    """
    # Smart vertical margin: stack layout places subtitles near split boundary
    if layout == "stack" and margin_v == 420:
        margin_v = 920

    styles = [
        f"Style: Default,Arial,{font_size},{default_color},{highlight_color},&H00000000,"
        f"&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,{margin_lr},{margin_lr},{margin_v},1\n"
    ]
    if hook_title:
        styles.append(
            "Style: HookBanner,Arial,64,&H00FFFFFF&,&H0000FFFF&,&H00000000,"
            "&H90000000,-1,0,0,0,100,100,0,0,1,6,4,8,80,80,140,1\n"
        )

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + "".join(styles)
        + "\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines: list[str] = [header]

    # Optional viral hook title banner at top of frame for first 3.5 seconds
    if hook_title:
        clean_hook = hook_title.strip()
        if mask_profanity:
            clean_hook = " ".join(mask_profane_word(w) for w in clean_hook.split())
        hook_end_str = format_ass_timestamp(max(0.5, hook_duration_s))
        lines.append(
            f"Dialogue: 1,0:00:00.00,{hook_end_str},HookBanner,,0,0,0,,"
            f"{{\\b1}}{clean_hook}{{\\b0}}\n"
        )

    if not words:
        full_empty = "".join(lines)
        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(full_empty, encoding="utf-8")
        return full_empty
    chunks = chunk_words(words)

    for chunk in chunks:
        # Prepare processed word texts (with optional profanity masking)
        word_texts: list[str] = []
        for w in chunk:
            t = mask_profane_word(w.text) if mask_profanity else w.text
            word_texts.append(t)

        for i, active_word in enumerate(chunk):
            w_start = max(0.0, active_word.start - clip_start)
            w_end = max(w_start, active_word.end - clip_start)

            # Highlight the i-th word
            formatted_words: list[str] = []
            for j, text in enumerate(word_texts):
                if j == i:
                    formatted_words.append(f"{{\\c{highlight_color}}}{text}{{\\c{default_color}}}")
                else:
                    formatted_words.append(text)

            line_text = " ".join(formatted_words)
            start_str = format_ass_timestamp(w_start)
            end_str = format_ass_timestamp(w_end)
            event_line = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{line_text}\n"
            lines.append(event_line)

    full_content = "".join(lines)
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(full_content, encoding="utf-8")

    return full_content
