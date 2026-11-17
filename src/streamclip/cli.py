"""Command-line interface skeleton for streamclip."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="streamclip",
    help="Livestream VOD highlight clips extractor targeting vertical 1080x1920 video.",
    no_args_is_help=True,
)


@app.command()
def run() -> None:
    """Run end-to-end highlight extraction pipeline."""


@app.command()
def detect() -> None:
    """Detect highlight candidates from extracted signals."""


@app.command()
def render() -> None:
    """Render highlight clips into vertical video layout."""


@app.command()
def eval() -> None:
    """Evaluate candidate highlights against ground truth annotations."""


@app.command()
def tune() -> None:
    """Tune fusion weights and peak detection parameters."""


if __name__ == "__main__":
    app()
