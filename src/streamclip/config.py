"""Configuration management for streamclip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class FusionWeights(BaseModel):
    """Weights for multimodal signal fusion."""

    chat: float = 0.5
    audio: float = 0.3
    vision: float = 0.2

    def as_dict(self) -> dict[str, float]:
        """Return weights as a plain dictionary."""
        return {"chat": self.chat, "audio": self.audio, "vision": self.vision}


class Config(BaseModel):
    """Pipeline configuration for highlight extraction and scoring."""

    fusion_weights: FusionWeights = Field(default_factory=FusionWeights)
    peak_height: float = 1.0
    min_peak_gap_s: float = 20.0
    clip_min_s: float = 15.0
    clip_max_s: float = 60.0
    pre_roll_s: float = 15.0
    post_roll_s: float = 4.0
    top_k: int = 20
    vision_budget_windows: int = 40

    @classmethod
    def from_yaml(cls, path_or_yaml: Path | str) -> Config:
        """Load configuration from a YAML file path or YAML content string."""
        content: str
        if isinstance(path_or_yaml, Path) or (
            isinstance(path_or_yaml, str)
            and "\n" not in path_or_yaml
            and Path(path_or_yaml).is_file()
        ):
            content = Path(path_or_yaml).read_text(encoding="utf-8")
        else:
            content = str(path_or_yaml)

        data = yaml.safe_load(content)
        if not data:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML content must deserialize to a mapping, got {type(data).__name__}"
            )
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize configuration to a YAML formatted string."""
        dumped: Any = self.model_dump(mode="json")
        return yaml.dump(dumped, sort_keys=False)

    def save_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_yaml(), encoding="utf-8")
