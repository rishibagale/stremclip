"""Unit tests for the streamclip Typer CLI skeleton."""

from typer.testing import CliRunner

from streamclip.cli import app

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

        # Test command execution
        exec_result = runner.invoke(app, [cmd])
        assert exec_result.exit_code == 0, f"Command {cmd} execution failed"
