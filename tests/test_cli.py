"""Unit tests for the streamclip Typer CLI: run, detect, render, eval, and tune."""

from pathlib import Path

from typer.testing import CliRunner

from streamclip.cli import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Livestream VOD highlight clips extractor" in result.stdout


def test_cli_commands_exist() -> None:
    commands = ["run", "detect", "render", "eval", "tune"]
    for cmd in commands:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"Command {cmd} failed --help"

        # Test command execution without arguments
        exec_result = runner.invoke(app, [cmd])
        assert exec_result.exit_code == 0, f"Command {cmd} execution failed"


def test_cli_run_dry_run(tmp_path: Path) -> None:
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"

    result = runner.invoke(
        app,
        [
            "run",
            str(video_path),
            "-c",
            str(chat_path),
            "-o",
            str(tmp_path / "out"),
            "--dry-run",
            "--top-k",
            "5",
        ],
    )
    assert result.exit_code == 0
    assert "Dry Run Plan" in result.output
    assert "Planned Highlight Clips" in result.output
    assert "Estimated LLM Cost" in result.output


def test_cli_detect_command(tmp_path: Path) -> None:
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"
    cands_file = tmp_path / "detected_candidates.json"

    result = runner.invoke(
        app,
        [
            "detect",
            str(video_path),
            "-c",
            str(chat_path),
            "-o",
            str(cands_file),
            "--no-vision",
        ],
    )
    assert result.exit_code == 0
    assert "highlight candidates" in result.output
    assert cands_file.is_file()


def test_cli_render_command(tmp_path: Path) -> None:
    video_path = FIXTURES_DIR / "video.mp4"
    cands_file = tmp_path / "candidates.json"
    cands_file.write_text(
        '[{"peak_t": 60.0, "start": 50.0, "end": 65.0, "score": 5.0, "components": {}}]',
        encoding="utf-8",
    )
    out_dir = tmp_path / "rendered"

    result = runner.invoke(
        app,
        [
            "render",
            str(cands_file),
            "-v",
            str(video_path),
            "-o",
            str(out_dir),
            "--top-k",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "Rendered 1 clips successfully" in result.output
    rendered_clips = list(out_dir.glob("*.mp4"))
    assert len(rendered_clips) == 1


def test_parse_facecam_arg() -> None:
    from streamclip.cli import parse_facecam_arg

    # Auto or None returns None
    assert parse_facecam_arg(None) is None
    assert parse_facecam_arg("auto") is None
    assert parse_facecam_arg("detect") is None

    # Presets return valid FacecamBox with even dimensions
    bl = parse_facecam_arg("bottom-left")
    assert bl is not None
    assert bl.w % 2 == 0
    assert bl.h % 2 == 0

    # Custom coordinate string
    custom = parse_facecam_arg("100, 200, 300, 400")
    assert custom is not None
    assert custom.x == 100
    assert custom.y == 200
    assert custom.w == 300
    assert custom.h == 400


def test_cli_run_with_virality_and_hook_banner(tmp_path: Path) -> None:
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"

    result = runner.invoke(
        app,
        [
            "run",
            str(video_path),
            "-c",
            str(chat_path),
            "-o",
            str(tmp_path / "out"),
            "--dry-run",
            "--hook-banner",
            "--top-k",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Virality" in result.output
