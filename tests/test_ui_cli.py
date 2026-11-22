"""Unit tests for the streamclip Web UI CLI command and helper functions."""

from typer.testing import CliRunner

from streamclip.cli import app
from streamclip.ui.app import check_dependencies

runner = CliRunner()


def test_cli_ui_help() -> None:
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--host" in result.output
    assert "--no-browser" in result.output


def test_cli_root_lists_ui() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ui" in result.output


def test_check_dependencies() -> None:
    deps = check_dependencies()
    assert isinstance(deps, dict)
    assert "ffmpeg" in deps
    assert "ffprobe" in deps
