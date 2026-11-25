"""Unit tests for open-source LICENSE file and pyproject.toml metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_license_file_exists_and_valid() -> None:
    lic_path = ROOT_DIR / "LICENSE"
    assert lic_path.is_file(), "LICENSE file must exist in repository root"
    content = lic_path.read_text(encoding="utf-8")
    assert "MIT License" in content
    assert "Rishi Bagale" in content


def test_pyproject_license_metadata() -> None:
    pp_path = ROOT_DIR / "pyproject.toml"
    assert pp_path.is_file()
    with open(pp_path, "rb") as f:
        data = tomllib.load(f)

    proj = data.get("project", {})
    assert proj.get("license") == "MIT"
    assert "LICENSE" in proj.get("license-files", [])
    authors = proj.get("authors", [])
    assert any(a.get("name") == "Rishi Bagale" for a in authors)
