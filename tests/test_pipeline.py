"""Tests for end-to-end highlight pipeline, dry run, and acceptance criteria."""

from __future__ import annotations

import json
from pathlib import Path

from streamclip.config import Config
from streamclip.eval.labels import load_labels
from streamclip.eval.metrics import evaluate
from streamclip.models import Candidate
from streamclip.pipeline import (
    PipelineOptions,
    detect_candidates,
    estimate_llm_cost,
    render_from_candidates,
    run_pipeline,
    slugify_title,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_estimate_llm_cost() -> None:
    assert estimate_llm_cost(0) == 0.0
    assert estimate_llm_cost(10, cost_per_candidate=0.02) == 0.2
    assert estimate_llm_cost(20, cost_per_candidate=0.02) == 0.4


def test_slugify_title() -> None:
    assert slugify_title("Chat Surge at 60s") == "chat_surge_at_60s"
    assert slugify_title("INSANE CLUTCH!! (Must See)") == "insane_clutch_must_see"
    assert slugify_title("") == "highlight"


def test_pipeline_dry_run(tmp_path: Path) -> None:
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"
    assert video_path.is_file()
    assert chat_path.is_file()

    options = PipelineOptions(
        video=video_path,
        chat=chat_path,
        out_dir=tmp_path / "dry_run_out",
        cache_dir=tmp_path / "cache",
        dry_run=True,
        no_llm=False,
        top_k=5,
    )

    result = run_pipeline(options)
    assert result.dry_run_plan is not None
    plan = result.dry_run_plan
    assert plan["vod"]["duration_s"] == 600.0
    assert plan["candidates_detected"] >= 5
    assert plan["top_k_selected"] == 5
    assert plan["llm_eval_enabled"] is True
    assert plan["estimated_llm_cost_usd"] > 0.0
    assert len(plan["clips"]) == 5
    # In dry-run mode, no videos are rendered
    assert len(result.rendered_paths) == 0


def test_detect_candidates_fixture(tmp_path: Path) -> None:
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"

    options = PipelineOptions(
        video=video_path,
        chat=chat_path,
        cache_dir=tmp_path / "cache",
        no_vision=True,
    )

    candidates = detect_candidates(options)
    assert len(candidates) >= 5
    assert all(isinstance(c, Candidate) for c in candidates)


def test_render_from_candidates_fixture(tmp_path: Path) -> None:
    video_path = FIXTURES_DIR / "video.mp4"
    cands_file = tmp_path / "cands.json"
    cands = [
        Candidate(peak_t=60.0, start=50.0, end=65.0, score=5.0),
        Candidate(peak_t=180.0, start=170.0, end=185.0, score=4.5),
    ]
    cands_file.write_text(json.dumps([c.model_dump(mode="json") for c in cands]), encoding="utf-8")

    out_dir = tmp_path / "rendered_from_cands"
    rendered = render_from_candidates(cands_file, video_path, out_dir=out_dir)
    assert len(rendered) == 2
    for p in rendered:
        assert p.is_file()
        assert p.stat().st_size > 1000


def test_acceptance_five_moments_recall_and_determinism(tmp_path: Path) -> None:
    """Acceptance test: On synthetic fixture with no API key, all 5 injected moments

    appear in top 5 clips (recall@5 = 1.0). Output must be identical across two runs.
    """
    video_path = FIXTURES_DIR / "video.mp4"
    chat_path = FIXTURES_DIR / "chat.csv"
    labels_path = FIXTURES_DIR / "labels.json"

    assert video_path.is_file()
    assert chat_path.is_file()
    assert labels_path.is_file()

    gt_labels = load_labels(labels_path)
    assert len(gt_labels) == 5

    out_dir_1 = tmp_path / "run_1"
    cache_dir = tmp_path / "cache"
    cfg = Config(top_k=5)

    options_1 = PipelineOptions(
        video=video_path,
        chat=chat_path,
        out_dir=out_dir_1,
        cache_dir=cache_dir,
        config=cfg,
        no_llm=True,  # No API key
        top_k=5,
    )

    # Run 1
    result_1 = run_pipeline(options_1)
    assert len(result_1.ranked_clips) == 5

    # Check Recall@5 = 1.0 against ground truth labels
    metrics_1 = evaluate([rc.candidate for rc in result_1.ranked_clips], gt_labels, k=5)
    assert metrics_1.recall == 1.0, f"Expected Recall@5 == 1.0, got {metrics_1.recall}"
    assert metrics_1.hits == 5

    # Run 2 into a distinct output directory
    out_dir_2 = tmp_path / "run_2"
    options_2 = PipelineOptions(
        video=video_path,
        chat=chat_path,
        out_dir=out_dir_2,
        cache_dir=cache_dir,
        config=cfg,
        no_llm=True,
        top_k=5,
    )

    result_2 = run_pipeline(options_2)
    assert len(result_2.ranked_clips) == 5

    metrics_2 = evaluate([rc.candidate for rc in result_2.ranked_clips], gt_labels, k=5)
    assert metrics_2.recall == 1.0
    assert metrics_2.hits == 5

    # Verify identical output across both runs
    manifest_1 = json.loads((out_dir_1 / "manifest.json").read_text(encoding="utf-8"))
    manifest_2 = json.loads((out_dir_2 / "manifest.json").read_text(encoding="utf-8"))

    assert manifest_1["clips"] == manifest_2["clips"]

    for c1, c2 in zip(result_1.ranked_clips, result_2.ranked_clips, strict=True):
        assert c1.candidate.peak_t == c2.candidate.peak_t
        assert c1.candidate.start == c2.candidate.start
        assert c1.candidate.end == c2.candidate.end
        assert c1.final_score == c2.final_score
        assert c1.title == c2.title
        assert c1.reason == c2.reason
