"""Ground truth label representations and loaders for highlight evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class GroundTruthLabel(BaseModel):
    """Ground truth highlight segment annotation."""

    t_start: float
    t_end: float
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("t_end")
    @classmethod
    def _validate_bounds(cls, v: float, info: Any) -> float:
        t_start = info.data.get("t_start")
        if t_start is not None and v < t_start:
            raise ValueError(f"t_end ({v}) cannot be less than t_start ({t_start})")
        return v

    @property
    def duration(self) -> float:
        """Duration of ground truth window in seconds."""
        return self.t_end - self.t_start

    @property
    def center_t(self) -> float:
        """Center timestamp of ground truth window."""
        return (self.t_start + self.t_end) / 2.0

    def contains(self, t: float) -> bool:
        """Check if timestamp falls within this ground truth window."""
        return self.t_start <= t <= self.t_end


def load_labels(path_or_json: Path | str) -> list[GroundTruthLabel]:
    """Load ground truth labels from a JSON file path or raw JSON string."""
    content: str
    if isinstance(path_or_json, Path) or (
        isinstance(path_or_json, str)
        and "\n" not in path_or_json
        and not path_or_json.strip().startswith("[")
        and Path(path_or_json).is_file()
    ):
        content = Path(path_or_json).read_text(encoding="utf-8")
    else:
        content = str(path_or_json)

    raw_data = json.loads(content)
    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a JSON list of labels, got {type(raw_data).__name__}")

    return [GroundTruthLabel.model_validate(item) for item in raw_data]


def save_labels(labels: list[GroundTruthLabel], path: Path | str) -> None:
    """Save ground truth labels to a formatted JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = [
        lbl.model_dump(mode="json", exclude={"meta"} if not lbl.meta else None) for lbl in labels
    ]
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
