"""Unit tests for video ingest and ffprobe inspection."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from streamclip.ingest.video import DownloadError, ProbeError, download, probe

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_probe_fixtures_with_audio() -> None:
    video_path = FIXTURES_DIR / "video.mp4"
    assert video_path.is_file(), "Fixture video.mp4 must exist"

    meta = probe(video_path)
    assert meta.path == video_path
    assert meta.width == 160
    assert meta.height == 120
    assert meta.duration_s == 600.0
    assert meta.fps == 1.0
    assert meta.has_audio is True
    assert meta.expected_signal_length == 600


def test_probe_fixtures_no_audio() -> None:
    video_path = FIXTURES_DIR / "video_no_audio.mp4"
    assert video_path.is_file(), "Fixture video_no_audio.mp4 must exist"

    meta = probe(video_path)
    assert meta.path == video_path
    assert meta.width == 160
    assert meta.height == 120
    assert meta.duration_s == 600.0
    assert meta.fps == 1.0
    assert meta.has_audio is False
    assert meta.expected_signal_length == 600


def test_probe_non_existent_file() -> None:
    with pytest.raises(ProbeError, match="does not exist"):
        probe("non_existent_path.mp4")


@patch("subprocess.run")
def test_probe_ffprobe_missing(mock_run: MagicMock) -> None:
    mock_run.side_effect = FileNotFoundError("ffprobe not found")
    with pytest.raises(ProbeError, match="ffprobe executable not found"):
        probe(FIXTURES_DIR / "video.mp4")


@patch("subprocess.run")
def test_probe_ffprobe_failure(mock_run: MagicMock) -> None:
    mock_run.return_value = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=1,
        stdout="",
        stderr="Corrupt input file",
    )
    with pytest.raises(ProbeError, match="ffprobe failed"):
        probe(FIXTURES_DIR / "video.mp4")


@patch("subprocess.run")
def test_probe_invalid_json(mock_run: MagicMock) -> None:
    mock_run.return_value = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout="not a valid json output",
        stderr="",
    )
    with pytest.raises(ProbeError, match="Failed to parse ffprobe output as JSON"):
        probe(FIXTURES_DIR / "video.mp4")


@patch("subprocess.run")
def test_probe_no_video_stream(mock_run: MagicMock) -> None:
    mock_run.return_value = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout='{"streams": [{"codec_type": "audio"}], "format": {"duration": "10.0"}}',
        stderr="",
    )
    with pytest.raises(ProbeError, match="No video stream found"):
        probe(FIXTURES_DIR / "video.mp4")


@patch("subprocess.run")
def test_probe_zero_duration(mock_run: MagicMock) -> None:
    zero_duration_json = (
        '{"streams": [{"codec_type": "video", "width": 1920, "height": 1080, '
        '"avg_frame_rate": "30/1", "duration": "0.0"}], "format": {}}'
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout=zero_duration_json,
        stderr="",
    )
    with pytest.raises(ProbeError, match="positive"):
        probe(FIXTURES_DIR / "video.mp4")


@patch("subprocess.run")
def test_download_success(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = subprocess.CompletedProcess(
        args=["yt-dlp"],
        returncode=0,
        stdout="[download] 100% of 10.0MiB",
        stderr="",
    )
    dest = tmp_path / "out.mp4"
    res = download("https://example.com/vod/123", dest)
    assert res == dest

    mock_run.assert_called_once()
    called_cmd = mock_run.call_args[0][0]
    assert called_cmd[0] == "yt-dlp"
    assert "https://example.com/vod/123" in called_cmd
    assert str(dest) in called_cmd


@patch("subprocess.run")
def test_download_custom_quality(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = subprocess.CompletedProcess(
        args=["yt-dlp"],
        returncode=0,
        stdout="[download] 100%",
        stderr="",
    )
    dest = tmp_path / "out_1080.mp4"
    res = download("https://example.com/vod/456", dest, quality="bestvideo[height<=1080]+bestaudio")
    assert res == dest

    mock_run.assert_called_once()
    called_cmd = mock_run.call_args[0][0]
    assert "-f" in called_cmd
    f_idx = called_cmd.index("-f")
    assert called_cmd[f_idx + 1] == "bestvideo[height<=1080]+bestaudio"


@patch("subprocess.run")
def test_download_failure(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = subprocess.CompletedProcess(
        args=["yt-dlp"],
        returncode=1,
        stdout="",
        stderr="HTTP Error 404: Not Found",
    )
    dest = tmp_path / "out.mp4"
    with pytest.raises(DownloadError, match="yt-dlp failed: HTTP Error 404"):
        download("https://example.com/vod/invalid", dest)


@patch("subprocess.run")
def test_download_ytdlp_not_found(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.side_effect = FileNotFoundError("yt-dlp not found")
    dest = tmp_path / "out.mp4"
    with pytest.raises(DownloadError, match="yt-dlp executable not found"):
        download("https://example.com/vod/123", dest)
