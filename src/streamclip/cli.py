"""Command-line interface for streamclip: run, detect, render, eval, and tune."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from streamclip.config import Config
from streamclip.eval.labels import load_labels
from streamclip.eval.metrics import evaluate
from streamclip.eval.tune import tune_hyperparameters
from streamclip.fusion import fuse_signals
from streamclip.ingest.chat import parse_chat
from streamclip.ingest.video import probe
from streamclip.models import Candidate, FacecamBox
from streamclip.pipeline import (
    PipelineError,
    PipelineOptions,
    detect_candidates,
    render_from_candidates,
    run_pipeline,
)
from streamclip.signals.audio import extract_audio_signal, silence_points
from streamclip.signals.chat import extract_chat_signal

app = typer.Typer(
    name="streamclip",
    help="Livestream VOD highlight clips extractor targeting vertical 1080x1920 video.",
    no_args_is_help=True,
)
console = Console()


def parse_facecam_arg(
    val: str | None,
    vod_w: int = 1920,
    vod_h: int = 1080,
) -> FacecamBox | None:
    """Parse --facecam CLI argument supporting presets and explicit coordinates."""
    if not val:
        return None
    cleaned = val.strip().lower()

    if cleaned in ("bottom-left", "bl"):
        w = round(vod_w * 0.22)
        h = round(vod_h * 0.26)
        x = round(vod_w * 0.01)
        y = round(vod_h * 0.44)
        return FacecamBox(
            x=x,
            y=y,
            w=w if w % 2 == 0 else w + 1,
            h=h if h % 2 == 0 else h + 1,
        )

    if cleaned in ("bottom-right", "br"):
        w = round(vod_w * 0.22)
        h = round(vod_h * 0.26)
        x = round(vod_w * 0.77)
        y = round(vod_h * 0.73)
        return FacecamBox(
            x=x,
            y=y,
            w=w if w % 2 == 0 else w + 1,
            h=h if h % 2 == 0 else h + 1,
        )

    if cleaned in ("top-left", "tl"):
        w = round(vod_w * 0.22)
        h = round(vod_h * 0.26)
        x = round(vod_w * 0.01)
        y = round(vod_w * 0.01)
        return FacecamBox(
            x=x,
            y=y,
            w=w if w % 2 == 0 else w + 1,
            h=h if h % 2 == 0 else h + 1,
        )

    if cleaned in ("top-right", "tr"):
        w = round(vod_w * 0.22)
        h = round(vod_h * 0.26)
        x = round(vod_w * 0.77)
        y = round(vod_h * 0.01)
        return FacecamBox(
            x=x,
            y=y,
            w=w if w % 2 == 0 else w + 1,
            h=h if h % 2 == 0 else h + 1,
        )

    if "," in cleaned:
        try:
            parts = [int(p.strip()) for p in cleaned.split(",")]
            if len(parts) == 4:
                return FacecamBox(x=parts[0], y=parts[1], w=parts[2], h=parts[3])
        except ValueError:
            return None

    return None


@app.command()
def run(
    video: Annotated[
        str | None,
        typer.Argument(
            help="Path or URL to input livestream VOD.",
        ),
    ] = None,
    chat: Annotated[
        Path | None,
        typer.Option(
            "--chat",
            "-c",
            help="Path to chat replay file (Twitch JSON, chat-downloader, or CSV).",
            dir_okay=False,
        ),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            "-o",
            help="Output directory for rendered clips and pipeline artifacts.",
        ),
    ] = Path("output"),
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to configuration YAML file.",
            dir_okay=False,
        ),
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option(
            "--no-llm",
            help="Disable LLM reranking and use fast heuristic scoring.",
        ),
    ] = False,
    no_vision: Annotated[
        bool,
        typer.Option(
            "--no-vision",
            help="Disable visual motion/action scoring.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print plan and estimated LLM cost without encoding.",
        ),
    ] = False,
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of top highlight clips to extract.",
        ),
    ] = None,
    layout: Annotated[
        str,
        typer.Option(
            "--layout",
            help="Vertical layout: 'gameplay_only', 'stack', or 'facecam_only'.",
        ),
    ] = "gameplay_only",
    facecam: Annotated[
        str | None,
        typer.Option(
            "--facecam",
            help="Facecam position preset ('bottom-left', 'bottom-right', etc.) or 'X,Y,W,H'.",
        ),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option(
            "--max-workers",
            "-w",
            help="Maximum concurrent rendering workers.",
        ),
    ] = 2,
) -> None:
    """Run end-to-end highlight extraction, reranking, and vertical rendering pipeline."""
    if video is None:
        console.print("[yellow]Usage: streamclip run <video_path_or_url> [options][/yellow]")
        console.print("Run 'streamclip run --help' for all available options.")
        return

    cfg = Config.from_yaml(config) if config else Config()
    parsed_facecam = parse_facecam_arg(facecam)
    options = PipelineOptions(
        video=video,
        chat=chat,
        out_dir=out_dir,
        config=cfg,
        no_llm=no_llm,
        no_vision=no_vision,
        dry_run=dry_run,
        top_k=top_k,
        layout=layout,
        facecam=parsed_facecam,
        max_workers=max_workers,
    )

    if dry_run:
        console.print(
            Panel.fit("[bold cyan]streamclip Dry Run Plan[/bold cyan]", border_style="cyan")
        )

    try:
        with console.status("[bold green]Executing streamclip pipeline...[/bold green]"):
            result = run_pipeline(options)
    except PipelineError as err:
        console.print(f"[bold red]Pipeline Error:[/bold red] {err}")
        raise typer.Exit(code=1) from None
    except Exception as err:
        console.print(f"[bold red]Unexpected Error:[/bold red] {err}")
        raise typer.Exit(code=1) from None

    if dry_run and result.dry_run_plan:
        plan = result.dry_run_plan
        console.print(f"VOD Source: [bold]{plan['vod']['path']}[/bold]")
        console.print(f"Duration: {plan['vod']['duration_s']}s ({plan['vod']['resolution']})")
        console.print(f"Candidates Detected: {plan['candidates_detected']}")
        console.print(f"Top-K Highlights: {plan['top_k_selected']}")
        console.print(f"LLM Enabled: {plan['llm_eval_enabled']}")
        cost_str = f"${plan['estimated_llm_cost_usd']:.4f}"
        console.print(f"Estimated LLM Cost: [bold green]{cost_str}[/bold green]")
        console.print(f"Layout: {plan['render_layout']} ({plan['target_resolution']})")

        table = Table(title="Planned Highlight Clips")
        table.add_column("Rank", justify="right")
        table.add_column("Title")
        table.add_column("Start", justify="right")
        table.add_column("End", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Score", justify="right")
        for clip_info in plan["clips"]:
            table.add_row(
                str(clip_info["rank"]),
                clip_info["title"],
                f"{clip_info['start']}s",
                f"{clip_info['end']}s",
                f"{clip_info['duration']}s",
                f"{clip_info['score']}",
            )
        console.print(table)
        return

    console.print(f"[bold green]Pipeline completed successfully![/bold green] Output: {out_dir}")
    table = Table(title=f"Extracted Highlights (Top {len(result.ranked_clips)})")
    table.add_column("Rank", justify="right")
    table.add_column("Title")
    table.add_column("Window", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("File")

    for i, clip in enumerate(result.ranked_clips):
        filename = result.rendered_paths[i].name if i < len(result.rendered_paths) else "-"
        table.add_row(
            str(i + 1),
            clip.title,
            f"{clip.candidate.start:.1f}s - {clip.candidate.end:.1f}s",
            f"{clip.final_score:.2f}",
            filename,
        )
    console.print(table)


@app.command()
def detect(
    video: Annotated[
        str | None,
        typer.Argument(
            help="Path or URL to input livestream VOD.",
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
            help="Destination path for detected candidates JSON file.",
        ),
    ] = Path("candidates.json"),
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to configuration YAML file.",
            dir_okay=False,
        ),
    ] = None,
    no_vision: Annotated[
        bool,
        typer.Option(
            "--no-vision",
            help="Disable visual motion/action scoring.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print detection plan without executing.",
        ),
    ] = False,
) -> None:
    """Detect highlight candidate windows from video and chat signals."""
    if video is None:
        console.print("[yellow]Usage: streamclip detect <video_path_or_url> [options][/yellow]")
        console.print("Run 'streamclip detect --help' for all available options.")
        return

    cfg = Config.from_yaml(config) if config else Config()
    options = PipelineOptions(
        video=video,
        chat=chat,
        config=cfg,
        no_vision=no_vision,
        dry_run=dry_run,
    )

    if dry_run:
        console.print(
            Panel.fit("[bold cyan]streamclip Detect Plan[/bold cyan]", border_style="cyan")
        )
        console.print(f"VOD Input: [bold]{video}[/bold]")
        console.print(f"Chat Source: {chat or '(None)'}")
        console.print(f"Vision Motion Scoring: {not no_vision}")
        console.print(f"Target Output: {output}")
        return

    try:
        with console.status("[bold green]Detecting highlight candidates...[/bold green]"):
            candidates = detect_candidates(options)
    except PipelineError as err:
        console.print(f"[bold red]Detection Error:[/bold red] {err}")
        raise typer.Exit(code=1) from None
    except Exception as err:
        console.print(f"[bold red]Unexpected Error:[/bold red] {err}")
        raise typer.Exit(code=1) from None

    output.parent.mkdir(parents=True, exist_ok=True)
    cands_json = [c.model_dump(mode="json") for c in candidates]
    output.write_text(json.dumps(cands_json, indent=2), encoding="utf-8")

    console.print(
        f"[bold green]Detected {len(candidates)} highlight candidates.[/bold green] "
        f"Written to {output}"
    )


@app.command()
def render(
    candidates: Annotated[
        Path | None,
        typer.Argument(
            help="Path to candidates.json containing detected candidate windows.",
        ),
    ] = None,
    video: Annotated[
        str | None,
        typer.Option(
            "--video",
            "-v",
            help="Path or URL to VOD video file.",
        ),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            "-o",
            help="Output directory for rendered clips.",
        ),
    ] = Path("output"),
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to configuration YAML file.",
            dir_okay=False,
        ),
    ] = None,
    layout: Annotated[
        str,
        typer.Option(
            "--layout",
            help="Vertical layout: 'gameplay_only', 'stack', or 'facecam_only'.",
        ),
    ] = "gameplay_only",
    facecam: Annotated[
        str | None,
        typer.Option(
            "--facecam",
            help="Facecam position preset ('bottom-left', 'bottom-right', etc.) or 'X,Y,W,H'.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print rendering plan without encoding.",
        ),
    ] = False,
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            "-k",
            help="Maximum number of clips to render.",
        ),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option(
            "--max-workers",
            "-w",
            help="Maximum concurrent rendering workers.",
        ),
    ] = 2,
) -> None:
    """Render vertical video clips directly from precomputed candidate windows."""
    if candidates is None or video is None:
        console.print(
            "[yellow]Usage: streamclip render <candidates.json> --video <video> [options][/yellow]"
        )
        console.print("Run 'streamclip render --help' for all available options.")
        return

    if not candidates.is_file():
        console.print(f"[bold red]Error:[/bold red] Candidates file '{candidates}' not found.")
        raise typer.Exit(code=1)

    cfg = Config.from_yaml(config) if config else Config()
    parsed_facecam = parse_facecam_arg(facecam)
    options = PipelineOptions(
        video=video,
        out_dir=out_dir,
        config=cfg,
        layout=layout,
        facecam=parsed_facecam,
        dry_run=dry_run,
        top_k=top_k,
        max_workers=max_workers,
    )

    if dry_run:
        console.print(
            Panel.fit("[bold cyan]streamclip Render Plan[/bold cyan]", border_style="cyan")
        )
        console.print(f"Candidates File: {candidates}")
        console.print(f"Video Source: {video}")
        console.print(f"Layout: {layout}")
        console.print(f"Output Directory: {out_dir}")
        return

    try:
        with console.status("[bold green]Rendering highlight clips...[/bold green]"):
            rendered_paths = render_from_candidates(candidates, video, out_dir, options=options)
    except Exception as err:
        console.print(f"[bold red]Render Error:[/bold red] {err}")
        raise typer.Exit(code=1) from None

    console.print(
        f"[bold green]Rendered {len(rendered_paths)} clips successfully![/bold green] "
        f"Saved in {out_dir}"
    )


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
        console.print("[yellow]Error: --labels is required.[/yellow]")
        return
    if not labels.is_file():
        console.print(f"[bold red]Error:[/bold red] Labels file '{labels}' not found.")
        raise typer.Exit(code=1)

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
        console.print("[bold red]Error:[/bold red] Please provide either --candidates or --video.")
        return

    gt_labels = load_labels(labels)
    metrics = evaluate(cands, gt_labels, k=k)

    console.print(f"Highlight Evaluation Results (K={k}):")
    console.print(f"  Recall@{k}: {metrics.recall:.4f}")
    console.print(f"  Precision@{k}: {metrics.precision:.4f}")
    console.print(f"  Mean Time Offset: {metrics.mean_time_offset:.2f}s")
    console.print(
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
        console.print("[yellow]Error: --labels and --video are required.[/yellow]")
        return
    if not labels.is_file():
        console.print(f"[bold red]Error:[/bold red] Labels file '{labels}' not found.")
        raise typer.Exit(code=1)
    if not video.is_file():
        console.print(f"[bold red]Error:[/bold red] Video file '{video}' not found.")
        raise typer.Exit(code=1)

    gt_labels = load_labels(labels)
    vod = probe(video)
    tracks = [extract_audio_signal(vod)]
    if chat is not None:
        msgs = parse_chat(chat)
        tracks.append(extract_chat_signal(vod, msgs))
    silence_pts = silence_points(vod)

    console.print(f"Starting {method} search tuning over multimodal signals...")
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
    console.print(f"[bold green]Tuning complete![/bold green] Best config written to {output}")
    console.print(f"  Best Recall@{k}: {res.best_metrics.recall:.4f}")
    console.print(f"  Best Precision@{k}: {res.best_metrics.precision:.4f}")
    console.print(f"  Mean Time Offset: {res.best_metrics.mean_time_offset:.2f}s")
    console.print(f"  Weights: {res.best_config.fusion_weights.as_dict()}")
    console.print(
        f"  Peak Height: {res.best_config.peak_height}, Min Gap: {res.best_config.min_peak_gap_s}s"
    )


if __name__ == "__main__":
    app()
