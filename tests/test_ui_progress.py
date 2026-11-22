"""Unit tests for video download progress parsing and telemetry."""

from __future__ import annotations

from streamclip.ingest.video import DownloadProgress, parse_ytdlp_progress_line


def test_parse_ytdlp_standard_progress() -> None:
    line = "[download]  45.2% of  712.50MiB at 12.45MiB/s ETA 00:32"
    prog = parse_ytdlp_progress_line(line)
    assert prog is not None
    assert prog.percent == 45.2
    assert prog.total_str == "712.50MiB"
    assert prog.speed_str == "12.45MiB/s"
    assert prog.eta_str == "00:32"
    assert prog.status == "downloading"


def test_parse_ytdlp_approx_total() -> None:
    line = "[download]   0.5% of ~  1.50GiB at  8.20MiB/s ETA 03:12"
    prog = parse_ytdlp_progress_line(line)
    assert prog is not None
    assert prog.percent == 0.5
    assert "1.50GiB" in prog.total_str
    assert "8.20MiB/s" in prog.speed_str
    assert prog.eta_str == "03:12"


def test_parse_ytdlp_completed_line() -> None:
    line = "[download] 100% of 50.00MiB in 00:04"
    prog = parse_ytdlp_progress_line(line)
    assert prog is not None
    assert prog.percent == 100.0
    assert "50.00MiB" in prog.total_str


def test_parse_non_progress_line() -> None:
    assert parse_ytdlp_progress_line("[download] Destination: carry.mp4") is None
    assert parse_ytdlp_progress_line("[info] Downloading video metadata...") is None
    assert parse_ytdlp_progress_line("") is None


def test_download_progress_defaults() -> None:
    p = DownloadProgress()
    assert p.percent == 0.0
    assert p.speed_str == "0.0 MB/s"
    assert p.status == "downloading"
