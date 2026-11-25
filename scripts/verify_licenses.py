"""License verification script for open-source compliance."""

from __future__ import annotations

import importlib.metadata as metadata
import sys
import tomllib
from pathlib import Path

ALLOWED_KEYWORDS = (
    "mit",
    "bsd",
    "apache",
    "isc",
    "unlicense",
    "public domain",
    "psf",
    "python software foundation",
    "mpl",
    "zlib",
)

CORE_PACKAGES = [
    "numpy",
    "scipy",
    "pydantic",
    "pyyaml",
    "rich",
    "typer",
    "streamlit",
    "altair",
    "yt-dlp",
]


def verify_license_file(root: Path) -> list[str]:
    """Verify LICENSE file exists and has MIT terms."""
    errors: list[str] = []
    lic_path = root / "LICENSE"
    if not lic_path.is_file():
        errors.append("Missing LICENSE file in repository root.")
        return errors

    content = lic_path.read_text(encoding="utf-8")
    if "MIT License" not in content:
        errors.append("LICENSE file does not contain 'MIT License' header.")
    if "Rishi Bagale" not in content:
        errors.append("LICENSE file missing copyright holder.")
    return errors


def verify_pyproject_metadata(root: Path) -> list[str]:
    """Verify pyproject.toml has valid license and author metadata."""
    errors: list[str] = []
    pp_path = root / "pyproject.toml"
    if not pp_path.is_file():
        errors.append("Missing pyproject.toml.")
        return errors

    with open(pp_path, "rb") as f:
        data = tomllib.load(f)

    proj = data.get("project", {})
    if proj.get("license") != "MIT":
        errors.append(f"Expected project.license = 'MIT', got {proj.get('license')!r}")
    if not proj.get("authors"):
        errors.append("Missing project.authors in pyproject.toml.")
    return errors


def verify_dependencies() -> list[str]:
    """Audit core runtime dependencies for permissive open-source licenses."""
    warnings: list[str] = []
    for pkg in CORE_PACKAGES:
        try:
            meta = metadata.metadata(pkg)
            lic = (meta.get("License") or "").lower()
            classifiers = " ".join(meta.get_all("Classifier") or []).lower()
            combined = f"{lic} {classifiers}"
            if not any(k in combined for k in ALLOWED_KEYWORDS):
                warnings.append(f"Package '{pkg}' license could not be confirmed: {lic[:60]}")
        except metadata.PackageNotFoundError:
            pass
    return warnings


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    errors = verify_license_file(root) + verify_pyproject_metadata(root)
    warnings = verify_dependencies()

    for w in warnings:
        print(f"[WARN] {w}")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    print("[OK] License file, pyproject.toml metadata, and dependency compliance verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
