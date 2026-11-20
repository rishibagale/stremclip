"""End-to-end streamclip pipeline: ingest, signals, fusion, windowing, rerank, render."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from streamclip.cache import DiskCache
from streamclip.config import Config
from streamclip.fusion import fuse_signals
from streamclip.ingest.chat import parse_chat
from streamclip.ingest.video import download, probe
from streamclip.interfaces import Reranker, Transcriber
from streamclip.models import (
    Candidate,
    ChatMessage,
    FacecamBox,
    RankedClip,
    RenderSpec,
    VodMeta,
    Word,
)
from streamclip.render import RenderJob, render_clips
from streamclip.rerank import HeuristicReranker, LLMReranker
from streamclip.signals.audio import extract_audio_signal, silence_points
from streamclip.signals.chat import extract_chat_signal
from streamclip.signals.transcript import HAS_FASTER_WHISPER, transcribe_vod
from streamclip.signals.vision import MotionScorer
from streamclip.windowing import create_candidates


class PipelineError(Exception):
    """Raised when pipeline execution fails."""


def slugify_title(title: str) -> str:
    """Convert clip title to a clean, safe filename slug."""
    cleaned = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "_", cleaned).strip("_")
    return slug[:40] if slug else "highlight"


def estimate_llm_cost(num_candidates: int, cost_per_candidate: float = 0.02) -> float:
    """Estimate total LLM API cost in USD based on evaluated candidate count."""
    return round(max(0, num_candidates) * cost_per_candidate, 4)


@dataclass
class PipelineOptions:
    """Configuration options for pipeline execution."""

    video: str | Path
    chat: str | Path | None = None
    out_dir: Path = field(default_factory=lambda: Path("output"))
    config: Config | None = None
    no_llm: bool = False
    no_vision: bool = False
    dry_run: bool = False
    cache_dir: Path = field(default_factory=lambda: Path(".streamclip_cache"))
    max_workers: int = 2
    layout: str = "gameplay_only"
    facecam: FacecamBox | None = None
    top_k: int | None = None
    mask_profanity: bool = True
    candidates_path: Path | None = None
    transcriber: Transcriber | None = None
    captions_path: Path | None = None


@dataclass
class PipelineResult:
    """Results returned from end-to-end pipeline execution."""

    vod: VodMeta
    candidates: list[Candidate]
    ranked_clips: list[RankedClip]
    rendered_paths: list[Path]
    manifest_path: Path | None = None
    dry_run_plan: dict[str, Any] | None = None


def ingest_inputs(
    video_source: str | Path,
    chat_source: str | Path | None = None,
    download_dir: Path | None = None,
) -> tuple[VodMeta, list[ChatMessage]]:
    """Ingest VOD (from local file or remote URL) and chat replay messages."""
    video_str = str(video_source).strip()
    is_url = video_str.startswith("http://") or video_str.startswith("https://")

    if is_url:
        dl_dir = download_dir or Path(".streamclip_cache") / "downloads"
        dl_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dl_dir / "downloaded_vod.mp4"
        try:
            video_file = download(video_str, dest_path)
        except Exception as err:
            raise PipelineError(f"Failed to download video from URL '{video_str}': {err}") from err
    else:
        video_file = Path(video_source)
        if not video_file.is_file():
            raise PipelineError(f"Input video file not found: {video_file}")

    try:
        vod = probe(video_file)
    except Exception as err:
        raise PipelineError(f"Failed to probe video file '{video_file}': {err}") from err

    chat_messages: list[ChatMessage] = []
    if chat_source:
        chat_path = Path(chat_source)
        if not chat_path.is_file():
            raise PipelineError(f"Chat replay file not found: {chat_path}")
        try:
            chat_messages = parse_chat(chat_path)
        except Exception as err:
            raise PipelineError(f"Failed to parse chat file '{chat_path}': {err}") from err

    return vod, chat_messages


def extract_signals_parallel(
    vod: VodMeta,
    chat_messages: list[ChatMessage],
    config: Config,
    cache: DiskCache | None = None,
) -> tuple[list[Any], list[float]]:
    """Extract audio and chat timeseries signals in parallel with disk caching."""
    is_cached_file = cache is not None and Path(vod.path).is_file()

    def get_audio() -> Any:
        cfg_slice = {"task": "audio_signal"}
        if is_cached_file and cache is not None:
            cached = cache.get(vod.path, cfg_slice)
            if isinstance(cached, dict):
                from streamclip.models import SignalTrack

                return SignalTrack.model_validate(cached)
        track = extract_audio_signal(vod, config=config)
        if is_cached_file and cache is not None:
            cache.set(vod.path, cfg_slice, track.model_dump(mode="json"))
        return track

    def get_silence(audio: Any | None = None) -> list[float]:
        cfg_slice = {"task": "silence_points"}
        if is_cached_file and cache is not None:
            cached = cache.get(vod.path, cfg_slice)
            if isinstance(cached, list):
                return [float(x) for x in cached]
        rms_data = None
        if audio is not None and hasattr(audio, "meta"):
            rms_data = audio.meta.get("_rms_vals") or getattr(audio, "values", None)
        pts = silence_points(vod, rms_vals=rms_data)
        if is_cached_file and cache is not None:
            cache.set(vod.path, cfg_slice, pts)
        return pts

    def get_chat() -> Any:
        cfg_slice = {"task": "chat_signal", "msg_count": len(chat_messages)}
        if is_cached_file and cache is not None:
            cached = cache.get(vod.path, cfg_slice)
            if isinstance(cached, dict):
                from streamclip.models import SignalTrack

                return SignalTrack.model_validate(cached)
        track = extract_chat_signal(vod, chat_messages, config=config)
        if is_cached_file and cache is not None:
            cache.set(vod.path, cfg_slice, track.model_dump(mode="json"))
        return track

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_audio = executor.submit(get_audio)
        fut_chat = executor.submit(get_chat)

        audio_track = fut_audio.result()
        chat_track = fut_chat.result()

    silence_pts = get_silence(audio_track)

    return [audio_track, chat_track], silence_pts


def detect_candidates(options: PipelineOptions) -> list[Candidate]:
    """Detect highlight candidate windows from video and chat without reranking or rendering."""
    cfg = options.config or Config()
    cache = DiskCache(options.cache_dir)

    vod, chat_messages = ingest_inputs(
        options.video, options.chat, download_dir=options.cache_dir / "downloads"
    )
    tracks, silence_pts = extract_signals_parallel(vod, chat_messages, cfg, cache)

    fused = fuse_signals(tracks, config=cfg)
    candidates = create_candidates(
        peaks=fused.peaks,
        fused_scores=fused,
        vod=vod,
        config=cfg,
        silence_points=silence_pts,
    )

    if not options.no_vision and candidates:
        vision_scorer = MotionScorer(budget=cfg.vision_budget_windows, cache=cache)
        candidates = vision_scorer.score_candidates(vod, candidates)

    return candidates


def run_pipeline(options: PipelineOptions) -> PipelineResult:
    """Execute the end-to-end livestream highlight extraction pipeline.

    Stages:
      1. Ingest: Download (if URL) and probe video, parse chat replay.
      2. Signals: Parallel audio (RMS/spectral-flux/silence) and chat signal extraction.
      3. Fusion: Rolling baseline z-score normalization and peak detection.
      4. Windowing: Candidate boundary snapping and overlap merging.
      5. Multimodal Enrichment: Motion scoring and Whisper transcription on candidate windows only.
      6. Rerank: LLM or heuristic standalone highlight scoring and deduplication.
      7. Render: Vertical layout filtergraph encoding (x264/AAC/loudnorm/faststart).

    Args:
        options: PipelineOptions configuring inputs, outputs, models, and flags.

    Returns:
        PipelineResult containing VOD metadata, candidates, ranked clips, and rendered video paths.
    """
    cfg = options.config or Config()
    cache = DiskCache(options.cache_dir)
    out_dir = Path(options.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest
    vod, chat_messages = ingest_inputs(
        options.video, options.chat, download_dir=options.cache_dir / "downloads"
    )

    # 2. Candidate Detection or Loading
    if options.candidates_path and Path(options.candidates_path).is_file():
        raw_cands = json.loads(Path(options.candidates_path).read_text(encoding="utf-8"))
        candidates = [Candidate.model_validate(c) for c in raw_cands]
    else:
        tracks, silence_pts = extract_signals_parallel(vod, chat_messages, cfg, cache)
        fused = fuse_signals(tracks, config=cfg)
        candidates = create_candidates(
            peaks=fused.peaks,
            fused_scores=fused,
            vod=vod,
            config=cfg,
            silence_points=silence_pts,
        )

        if not options.no_vision and candidates:
            vision_scorer = MotionScorer(budget=cfg.vision_budget_windows, cache=cache)
            candidates = vision_scorer.score_candidates(vod, candidates)

    # Save candidates JSON
    candidates_file = out_dir / "candidates.json"
    candidates_file.write_text(
        json.dumps([c.model_dump(mode="json") for c in candidates], indent=2),
        encoding="utf-8",
    )

    # 3. Targeted Transcription on Candidate Windows Only
    words: list[Word] = []
    if options.captions_path and Path(options.captions_path).is_file():
        try:
            raw_c = json.loads(Path(options.captions_path).read_text(encoding="utf-8"))
            if isinstance(raw_c, list):
                words = [Word.model_validate(w) for w in raw_c]
        except Exception:
            words = []
    if not words and candidates and vod.has_audio:
        if options.transcriber is not None or HAS_FASTER_WHISPER:
            try:
                words = transcribe_vod(
                    vod=vod,
                    candidates=candidates,
                    transcriber=options.transcriber,
                    cache=cache,
                )
            except Exception:
                words = []

    # 4. Reranking
    effective_top_k = options.top_k if options.top_k is not None else cfg.top_k
    reranker: Reranker
    if options.no_llm:
        reranker = HeuristicReranker(config=cfg)
    else:
        reranker = LLMReranker(config=cfg, vod=vod, cache=cache, cost_cap=effective_top_k)

    ranked_clips = reranker.rerank(candidates, chat=chat_messages, transcript=words, vod=vod)
    selected_clips = ranked_clips[:effective_top_k]

    # 5. Handle Dry Run
    if options.dry_run:
        llm_candidate_count = 0 if options.no_llm else min(len(candidates), effective_top_k)
        est_cost = estimate_llm_cost(llm_candidate_count)
        plan = {
            "vod": {
                "path": str(vod.path),
                "duration_s": vod.duration_s,
                "resolution": f"{vod.width}x{vod.height}",
                "fps": vod.fps,
                "has_audio": vod.has_audio,
            },
            "chat_messages_count": len(chat_messages),
            "candidates_detected": len(candidates),
            "top_k_selected": len(selected_clips),
            "llm_eval_enabled": not options.no_llm,
            "estimated_llm_cost_usd": est_cost,
            "render_layout": options.layout,
            "target_resolution": "1080x1920",
            "output_directory": str(out_dir),
            "clips": [
                {
                    "rank": i + 1,
                    "title": c.title,
                    "start": round(c.candidate.start, 2),
                    "end": round(c.candidate.end, 2),
                    "duration": round(c.candidate.duration, 2),
                    "score": round(c.final_score, 3),
                    "reason": c.reason,
                }
                for i, c in enumerate(selected_clips)
            ],
        }
        return PipelineResult(
            vod=vod,
            candidates=candidates,
            ranked_clips=selected_clips,
            rendered_paths=[],
            dry_run_plan=plan,
        )

    # 6. Render Highlight Clips
    render_jobs: list[RenderJob] = []
    rendered_paths: list[Path] = []

    for i, clip in enumerate(selected_clips, start=1):
        slug = slugify_title(clip.title)
        out_filename = f"clip_{i:02d}_{slug}.mp4"
        clip_path = out_dir / out_filename

        # Filter transcript words strictly inside this clip's window
        cand = clip.candidate
        clip_words = [w for w in words if w.end > cand.start and w.start < cand.end]

        spec_layout = (
            options.layout
            if options.layout in ("stack", "gameplay_only", "facecam_only")
            else "gameplay_only"
        )
        resolved_facecam = options.facecam or getattr(cfg, "facecam", None)
        if spec_layout == "stack" and resolved_facecam is None:
            from streamclip.render.layout import suggest_facecam

            resolved_facecam = suggest_facecam(vod)

        spec = RenderSpec(
            layout=spec_layout,  # type: ignore[arg-type]
            facecam=resolved_facecam,
            captions=clip_words,
            out_w=1080,
            out_h=1920,
        )

        # Check if already rendered (cache resumability)
        if clip_path.is_file() and clip_path.stat().st_size > 1024:
            rendered_paths.append(clip_path)
            continue

        job = RenderJob(
            vod=vod,
            start=cand.start,
            end=cand.end,
            output_path=clip_path,
            spec=spec,
            mask_profanity=options.mask_profanity,
            config=cfg,
        )
        render_jobs.append(job)

    if render_jobs:
        newly_rendered = render_clips(render_jobs, max_workers=options.max_workers)
        rendered_paths.extend(newly_rendered)

    # Sort rendered paths to match rank order
    rendered_paths.sort(key=lambda p: p.name)

    # 7. Write Manifest Summary
    manifest_file = out_dir / "manifest.json"
    manifest_data = {
        "vod": {
            "path": str(vod.path),
            "duration_s": vod.duration_s,
            "resolution": f"{vod.width}x{vod.height}",
        },
        "clips": [
            {
                "rank": i + 1,
                "file": (out_dir / f"clip_{i + 1:02d}_{slugify_title(c.title)}.mp4").name,
                "title": c.title,
                "reason": c.reason,
                "start": c.candidate.start,
                "end": c.candidate.end,
                "duration": c.candidate.duration,
                "final_score": c.final_score,
                "llm_score": c.llm_score,
            }
            for i, c in enumerate(selected_clips)
        ],
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return PipelineResult(
        vod=vod,
        candidates=candidates,
        ranked_clips=selected_clips,
        rendered_paths=rendered_paths,
        manifest_path=manifest_file,
    )


def render_from_candidates(
    candidates_source: Path | list[Candidate],
    vod_source: str | Path,
    out_dir: Path,
    options: PipelineOptions | None = None,
) -> list[Path]:
    """Render vertical video highlight clips directly from a pre-computed
    candidates list or JSON file.
    """
    opts = options or PipelineOptions(video=vod_source, out_dir=out_dir)
    cfg = opts.config or Config()
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(candidates_source, Path):
        raw = json.loads(candidates_source.read_text(encoding="utf-8"))
        candidates = [Candidate.model_validate(c) for c in raw]
    else:
        candidates = list(candidates_source)

    vod, _ = ingest_inputs(vod_source, download_dir=opts.cache_dir / "downloads")

    # Transcribe candidate windows for captions if audio is available
    cache = DiskCache(opts.cache_dir)
    words: list[Word] = []
    if opts.captions_path and opts.captions_path.is_file():
        try:
            raw_c = json.loads(opts.captions_path.read_text(encoding="utf-8"))
            if isinstance(raw_c, list):
                words = [Word.model_validate(w) for w in raw_c]
        except Exception:
            words = []
    if not words and vod.has_audio and (opts.transcriber is not None or HAS_FASTER_WHISPER):
        try:
            words = transcribe_vod(
                vod=vod,
                candidates=candidates,
                transcriber=opts.transcriber,
                cache=cache,
            )
        except Exception:
            words = []

    # Heuristic titling and deduplication
    reranker = HeuristicReranker(config=cfg)
    ranked = reranker.rerank(candidates, transcript=words, vod=vod)
    effective_top_k = opts.top_k if opts.top_k is not None else cfg.top_k
    selected = ranked[:effective_top_k]

    if opts.dry_run:
        return []

    jobs: list[RenderJob] = []
    rendered_paths: list[Path] = []

    for i, clip in enumerate(selected, start=1):
        slug = slugify_title(clip.title)
        out_filename = f"clip_{i:02d}_{slug}.mp4"
        clip_path = out_dir / out_filename

        cand = clip.candidate
        clip_words = [w for w in words if w.end > cand.start and w.start < cand.end]

        spec_layout = (
            opts.layout
            if opts.layout in ("stack", "gameplay_only", "facecam_only")
            else "gameplay_only"
        )
        resolved_facecam = opts.facecam or getattr(cfg, "facecam", None)
        if spec_layout == "stack" and resolved_facecam is None:
            from streamclip.render.layout import suggest_facecam

            resolved_facecam = suggest_facecam(vod)

        spec = RenderSpec(
            layout=spec_layout,  # type: ignore[arg-type]
            facecam=resolved_facecam,
            captions=clip_words,
            out_w=1080,
            out_h=1920,
        )

        if clip_path.is_file() and clip_path.stat().st_size > 1024:
            rendered_paths.append(clip_path)
            continue

        job = RenderJob(
            vod=vod,
            start=cand.start,
            end=cand.end,
            output_path=clip_path,
            spec=spec,
            mask_profanity=opts.mask_profanity,
            config=cfg,
        )
        jobs.append(job)

    if jobs:
        newly_rendered = render_clips(jobs, max_workers=opts.max_workers)
        rendered_paths.extend(newly_rendered)

    rendered_paths.sort(key=lambda p: p.name)
    return rendered_paths
