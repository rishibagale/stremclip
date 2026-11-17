"""Synthetic test fixtures generator for streamclip.

Generates:
- 10-minute synthetic ffmpeg testsrc video with sine-tone audio and 5 loud bursts.
- Synthetic chat replay logs with message rate spikes at the same 5 moments.
  Formatted as:
    1. TwitchDownloader JSON
    2. chat-downloader JSON
    3. CSV (t,user,text)
- labels.json listing ground-truth moments as [{t_start, t_end}].
- Variants:
    1. no chat (empty logs across formats)
    2. no audio (video without audio track)
    3. chat with long gaps
    4. non-ASCII / emoji emotes
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

# 5 known ground-truth moments (in seconds) within the 10-minute (600s) window
GROUND_TRUTH_MOMENTS: list[dict[str, float]] = [
    {"t_start": 50.0, "t_end": 70.0},
    {"t_start": 170.0, "t_end": 190.0},
    {"t_start": 290.0, "t_end": 310.0},
    {"t_start": 410.0, "t_end": 430.0},
    {"t_start": 530.0, "t_end": 550.0},
]

# Audio burst windows corresponding to the 5 moments
AUDIO_BURST_WINDOWS: list[tuple[float, float]] = [
    (55.0, 65.0),
    (175.0, 185.0),
    (295.0, 305.0),
    (415.0, 425.0),
    (535.0, 545.0),
]

HYPE_TEXTS: list[str] = [
    "OMG WHAT A PLAY",
    "INSANE MOVE!!",
    "CLIP THAT RIGHT NOW",
    "LUL THAT WAS CRAZY",
    "W GAMEPLAY",
    "POG CHAT",
    "NO WAY HE SURVIVED THAT",
    "GTA VI MOMENT",
    "HOLY CRAP",
    "SHEEEESH",
]

NORMAL_TEXTS: list[str] = [
    "hey chat",
    "game looks clean",
    "what car is that?",
    "where on the map is this",
    "good stream today",
    "nice graphics",
    "is this vice city?",
    "driving mechanics look nice",
    "lol nice",
    "vibes are good",
]

NON_ASCII_TEXTS: list[str] = [
    "OMG 🔥🔥🔥",
    "THAT WAS DEADLY 💀💀💀",
    "W STREAM 🚀🎮",
    "EXPLOSION 💥💥",
    "(╯°□°)╯ AMAZING!",
    "STREAM IS INSANE ✨✨",
    "BOOM 🚗💨",
    "POG CHAMP 🎉🎉",
    "LET'S GOOO 🏆",
    "CHOP CHOP 🐶",
]


def generate_video(
    output_path: Path,
    duration_s: float = 600.0,
    with_audio: bool = True,
) -> None:
    """Generate a lightweight synthetic 10-minute video using ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if with_audio:
        # Build volume expression: loud (1.0) during bursts, quiet (0.05) elsewhere
        burst_conditions = [
            f"between(t,{start:.1f},{end:.1f})" for start, end in AUDIO_BURST_WINDOWS
        ]
        vol_expr = "+".join(burst_conditions)
        filter_str = f"volume='if({vol_expr}, 1.0, 0.05)':eval=frame"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_s:.1f}:size=160x120:rate=1",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration_s:.1f}",
            "-af",
            filter_str,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "51",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-b:a",
            "8k",
            str(output_path),
        ]
    else:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_s:.1f}:size=160x120:rate=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "51",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with code {result.returncode}:\n{result.stderr}")


def generate_chat_messages(
    seed: int = 42,
    duration_s: float = 600.0,
    long_gaps: bool = False,
    non_ascii: bool = False,
) -> list[dict[str, Any]]:
    """Deterministically generate synthetic chat messages with spikes matching the 5 moments."""
    rng = np.random.default_rng(seed)
    messages: list[dict[str, Any]] = []

    # Choose text pool
    hype_pool = NON_ASCII_TEXTS if non_ascii else HYPE_TEXTS
    normal_pool = NON_ASCII_TEXTS if non_ascii else NORMAL_TEXTS

    # 1. Background chat (unless long_gaps variant is enabled)
    if not long_gaps:
        # Uniform rate of ~0.3 msgs/sec
        bg_times = np.sort(rng.uniform(0.0, duration_s, size=int(duration_s * 0.3)))
        for t in bg_times:
            # Skip if inside a spike window to avoid overlapping background noise
            if any(m["t_start"] <= t <= m["t_end"] for m in GROUND_TRUTH_MOMENTS):
                continue
            u_id = int(rng.integers(1, 100))
            txt = str(rng.choice(normal_pool))
            bg_emotes = ["LUL"] if ("LUL" in txt or "lol" in txt) else []
            messages.append(
                {
                    "t": round(float(t), 3),
                    "user": f"viewer_{u_id:03d}",
                    "text": txt,
                    "emotes": bg_emotes,
                }
            )

    # 2. Message rate spikes at the 5 ground-truth moments
    active_moments = (
        [GROUND_TRUTH_MOMENTS[0], GROUND_TRUTH_MOMENTS[3]] if long_gaps else GROUND_TRUTH_MOMENTS
    )

    for moment in active_moments:
        # 80 messages concentrated in each 20s highlight window (~4 msgs/sec)
        spike_times = np.sort(rng.uniform(moment["t_start"], moment["t_end"], size=80))
        for t in spike_times:
            u_id = int(rng.integers(1, 200))
            txt = str(rng.choice(hype_pool))
            spike_emotes: list[str] = []
            if "LUL" in txt:
                spike_emotes.append("LUL")
            if "POG" in txt:
                spike_emotes.append("POG")
            if "W" in txt:
                spike_emotes.append("W")
            if non_ascii:
                spike_emotes.append("🔥")

            messages.append(
                {
                    "t": round(float(t), 3),
                    "user": f"viewer_{u_id:03d}",
                    "text": txt,
                    "emotes": spike_emotes,
                }
            )

    # Sort all messages by timestamp
    messages.sort(key=lambda m: float(m["t"]))
    return messages


def export_twitch_json(messages: list[dict[str, Any]], output_path: Path) -> None:
    """Export messages in TwitchDownloader JSON format."""
    comments: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages, start=1):
        comments.append(
            {
                "_id": f"c_{idx:06d}",
                "channel_id": "123456789",
                "content_id": "v987654321",
                "content_offset_seconds": msg["t"],
                "content_type": "video",
                "created_at": "2026-01-01T00:00:00.000Z",
                "source": "chat",
                "state": "published",
                "commenter": {
                    "_id": f"u_{msg['user']}",
                    "display_name": msg["user"],
                    "name": msg["user"],
                },
                "message": {
                    "body": msg["text"],
                    "emoticons": [
                        {"_id": emote, "begin": 0, "end": len(emote)} for emote in msg["emotes"]
                    ],
                },
            }
        )

    payload = {
        "streamer": {"id": 123456789, "name": "gtavi_streamer"},
        "comments": comments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_chat_downloader_json(messages: list[dict[str, Any]], output_path: Path) -> None:
    """Export messages in chat-downloader JSON format (array of messages)."""
    items: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages, start=1):
        minutes = int(msg["t"] // 60)
        seconds = int(msg["t"] % 60)
        items.append(
            {
                "message_id": f"msg_{idx:06d}",
                "message": msg["text"],
                "message_type": "text_message",
                "time_in_seconds": msg["t"],
                "time_text": f"{minutes:02d}:{seconds:02d}",
                "author": {
                    "id": f"auth_{msg['user']}",
                    "name": msg["user"],
                    "display_name": msg["user"],
                },
                "emotes": [{"name": emote} for emote in msg["emotes"]],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def export_csv(messages: list[dict[str, Any]], output_path: Path) -> None:
    """Export messages in CSV format: t,user,text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "user", "text"])
        for msg in messages:
            writer.writerow([msg["t"], msg["user"], msg["text"]])


def export_labels(moments: list[dict[str, float]], output_path: Path) -> None:
    """Export ground-truth moments to labels.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(moments, indent=2), encoding="utf-8")


def generate_all_fixtures(
    output_dir: Path,
    seed: int = 42,
    generate_videos: bool = True,
) -> None:
    """Generate all synthetic test fixtures and variants."""
    output_dir.mkdir(parents=True, exist_ok=True)
    variants_dir = output_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ground truth labels
    export_labels(GROUND_TRUTH_MOMENTS, output_dir / "labels.json")

    # 2. Main synthetic chat (at same 5 timestamps) in 3 formats
    main_messages = generate_chat_messages(seed=seed, duration_s=600.0)
    export_twitch_json(main_messages, output_dir / "chat_twitch.json")
    export_chat_downloader_json(main_messages, output_dir / "chat_downloader.json")
    export_csv(main_messages, output_dir / "chat.csv")

    # 3. Variants
    # Variant A: No chat
    export_twitch_json([], variants_dir / "no_chat_twitch.json")
    export_chat_downloader_json([], variants_dir / "no_chat_downloader.json")
    export_csv([], variants_dir / "no_chat.csv")

    # Variant B: Chat with long gaps
    gap_messages = generate_chat_messages(seed=seed, duration_s=600.0, long_gaps=True)
    export_twitch_json(gap_messages, variants_dir / "chat_long_gaps_twitch.json")
    export_chat_downloader_json(gap_messages, variants_dir / "chat_long_gaps_downloader.json")
    export_csv(gap_messages, variants_dir / "chat_long_gaps.csv")

    # Variant C: Non-ASCII emotes
    unicode_messages = generate_chat_messages(seed=seed, duration_s=600.0, non_ascii=True)
    export_twitch_json(unicode_messages, variants_dir / "chat_non_ascii_twitch.json")
    export_chat_downloader_json(unicode_messages, variants_dir / "chat_non_ascii_downloader.json")
    export_csv(unicode_messages, variants_dir / "chat_non_ascii.csv")

    # 4. Videos with ffmpeg
    if generate_videos and shutil.which("ffmpeg"):
        generate_video(output_dir / "video.mp4", duration_s=600.0, with_audio=True)
        generate_video(output_dir / "video_no_audio.mp4", duration_s=600.0, with_audio=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fixtures for streamclip.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Target output directory for fixtures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video generation via ffmpeg",
    )
    args = parser.parse_args()
    generate_all_fixtures(
        output_dir=args.output_dir,
        seed=args.seed,
        generate_videos=not args.no_video,
    )


if __name__ == "__main__":
    main()
