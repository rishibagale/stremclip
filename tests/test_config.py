"""Unit tests for configuration loading and serialization."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from streamclip.config import Config, FusionWeights


def test_default_config() -> None:
    cfg = Config()
    assert cfg.fusion_weights.chat == 0.5
    assert cfg.fusion_weights.audio == 0.3
    assert cfg.fusion_weights.vision == 0.2
    assert cfg.peak_height == 1.0
    assert cfg.min_peak_gap_s == 20.0
    assert cfg.clip_min_s == 15.0
    assert cfg.clip_max_s == 60.0
    assert cfg.pre_roll_s == 15.0
    assert cfg.post_roll_s == 4.0
    assert cfg.top_k == 20
    assert cfg.vision_budget_windows == 40


def test_fusion_weights_dict() -> None:
    weights = FusionWeights(chat=0.6, audio=0.3, vision=0.1)
    assert weights.as_dict() == {"chat": 0.6, "audio": 0.3, "vision": 0.1}


def test_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = Config(
        peak_height=2.5,
        clip_min_s=20.0,
        fusion_weights=FusionWeights(chat=0.7, audio=0.2, vision=0.1),
    )
    yaml_str = cfg.to_yaml()
    loaded = Config.from_yaml(yaml_str)
    assert loaded.peak_height == 2.5
    assert loaded.clip_min_s == 20.0
    assert loaded.fusion_weights.chat == 0.7

    file_path = tmp_path / "config.yaml"
    cfg.save_yaml(file_path)
    assert file_path.exists()

    loaded_from_file = Config.from_yaml(file_path)
    assert loaded_from_file.peak_height == 2.5
    assert loaded_from_file.fusion_weights.audio == 0.2


def test_from_empty_yaml() -> None:
    cfg = Config.from_yaml("")
    assert cfg.top_k == 20


def test_from_invalid_yaml() -> None:
    with pytest.raises(ValueError, match="must deserialize to a mapping"):
        Config.from_yaml("- item1\n- item2")

    with pytest.raises(ValidationError):
        Config.from_yaml("top_k: not_a_number")
