"""Command-line interface skeleton for streamclip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from streamclip.config import Config
from streamclip.eval.labels import load_labels
from streamclip.eval.metrics import evaluate
from streamclip.eval.tune import tune_hyperparameters
from streamclip.fusion import fuse_signals
from streamclip.ingest.chat import parse_chat
from streamclip.ingest.video import probe
from streamclip.models import Candidate
from streamclip.signals.audio import extract_audio_signal, silence_points
from streamclip.signals.chat import extract_chat_signal

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
def eval(
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            "-l",
            help="Path to ground truth labels JSON.",
            dir_okay=False,
        ),
    ] = None,
    candidates: Annotated[
        Path | None,
        typer.Option(
            "--candidates",
            help="Path to precomputed candidates JSON.",
            dir_okay=False,
        ),
    ] = None,
    video: Annotated[
        Path | None,
        typer.Option(
            "--video",
            "-v",
            help="Path to VOD video file.",
            dir_okay=False,
        ),
    ] = None,
    chat: Annotated[
        Path | None,
        typer.Option(
            "--chat",
            "-c",
            help="Path to chat replay file.",
            dir_okay=False,
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to config YAML file.",
            dir_okay=False,
        ),
    ] = None,
    k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="K cutoff for Recall@K and Precision@K.",
        ),
    ] = 5,
) -> None:
    """Evaluate candidate highlights against ground truth annotations."""
    if labels is None:
        typer.echo("Error: --labels is required.")
        return

    cands: list[Candidate] = []

    if candidates is not None:
        raw_cands = json.loads(candidates.read_text(encoding="utf-8"))
        if not isinstance(raw_cands, list):
            raise typer.BadParameter("Candidates JSON must contain a list.")
        cands = [Candidate.model_validate(c) for c in raw_cands]
    elif video is not None:
        vod = probe(video)
        cfg = Config.from_yaml(config) if config else Config()
        tracks = [extract_audio_signal(vod)]
        if chat is not None:
            msgs = parse_chat(chat)
            tracks.append(extract_chat_signal(vod, msgs, config=cfg))
        fused = fuse_signals(tracks, config=cfg)
        silence_pts = silence_points(vod)
        from streamclip.windowing import create_candidates

        cands = create_candidates(
            peaks=fused.peaks,
            fused_scores=fused,
            vod=vod,
            config=cfg,
            silence_points=silence_pts,
        )
    else:
        typer.echo("Error: Please provide either --candidates or --video.", err=True)
        return

    gt_labels = load_labels(labels)
    metrics = evaluate(cands, gt_labels, k=k)

    typer.echo(f"Highlight Evaluation Results (K={k}):")
    typer.echo(f"  Recall@{k}: {metrics.recall:.4f}")
    typer.echo(f"  Precision@{k}: {metrics.precision:.4f}")
    typer.echo(f"  Mean Time Offset: {metrics.mean_time_offset:.2f}s")
    typer.echo(
        f"  Hits: {metrics.hits}/{metrics.total_labels} "
        f"(from {metrics.total_candidates} candidates evaluated)"
    )


@app.command()
def tune(
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            "-l",
            help="Path to ground truth labels JSON.",
            dir_okay=False,
        ),
    ] = None,
    video: Annotated[
        Path | None,
        typer.Option(
            "--video",
            "-v",
            help="Path to VOD video file.",
            dir_okay=False,
        ),
    ] = None,
    chat: Annotated[
        Path | None,
        typer.Option(
            "--chat",
            "-c",
            help="Path to chat replay file.",
            dir_okay=False,
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to save best tuned config YAML.",
        ),
    ] = Path("configs/tuned.yaml"),
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help="Search method: 'grid' or 'random'.",
        ),
    ] = "grid",
    k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="K cutoff for Recall@K and Precision@K.",
        ),
    ] = 5,
    precision_floor: Annotated[
        float,
        typer.Option(
            "--precision-floor",
            help="Minimum precision threshold.",
        ),
    ] = 0.5,
    trials: Annotated[
        int,
        typer.Option(
            "--trials",
            "-n",
            help="Number of trials if using random search.",
        ),
    ] = 30,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for search.",
        ),
    ] = 42,
) -> None:
    """Tune fusion weights and peak detection parameters."""
    if labels is None or video is None:
        typer.echo("Error: --labels and --video are required.")
        return

    gt_labels = load_labels(labels)
    vod = probe(video)
    tracks = [extract_audio_signal(vod)]
    if chat is not None:
        msgs = parse_chat(chat)
        tracks.append(extract_chat_signal(vod, msgs))
    silence_pts = silence_points(vod)

    typer.echo(f"Starting {method} search tuning over multimodal signals...")
    search_method: Literal["grid", "random"] = "random" if method == "random" else "grid"
    res = tune_hyperparameters(
        tracks=tracks,
        vod=vod,
        labels=gt_labels,
        silence_points=silence_pts,
        method=search_method,
        k=k,
        precision_floor=precision_floor,
        n_trials=trials,
        seed=seed,
        out_path=output,
    )
    typer.echo(f"Tuning complete! Best config written to {output}")
    typer.echo(f"  Best Recall@{k}: {res.best_metrics.recall:.4f}")
    typer.echo(f"  Best Precision@{k}: {res.best_metrics.precision:.4f}")
    typer.echo(f"  Mean Time Offset: {res.best_metrics.mean_time_offset:.2f}s")
    typer.echo(f"  Weights: {res.best_config.fusion_weights.as_dict()}")
    typer.echo(
        f"  Peak Height: {res.best_config.peak_height}, Min Gap: {res.best_config.min_peak_gap_s}s"
    )


if __name__ == "__main__":
    app()
