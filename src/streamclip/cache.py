"""Disk cache keyed by file hash and configuration slice."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def compute_file_hash(path: Path | str, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hash of a file on disk."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot compute hash for non-existent file: {file_path}")

    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_config_hash(config_slice: Any) -> str:
    """Compute a deterministic SHA-256 hash for a config slice or dictionary."""
    if isinstance(config_slice, BaseModel):
        data = config_slice.model_dump(mode="json")
    elif isinstance(config_slice, dict):
        data = config_slice
    elif hasattr(config_slice, "__dict__"):
        data = config_slice.__dict__
    else:
        data = {"value": config_slice}

    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DiskCache:
    """Persistent on-disk cache keyed by source file hash and configuration slice."""

    def __init__(self, cache_dir: Path | str = ".streamclip_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_key(self, file_path: Path | str, config_slice: Any) -> str:
        """Generate a composite cache key from file content and config slice."""
        f_hash = compute_file_hash(file_path)
        c_hash = compute_config_hash(config_slice)
        return f"{f_hash[:16]}_{c_hash[:16]}"

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def has(self, file_path: Path | str, config_slice: Any) -> bool:
        """Check if an entry exists for the given file and config slice."""
        try:
            key = self.get_key(file_path, config_slice)
            return self._path_for_key(key).is_file()
        except FileNotFoundError:
            return False

    def get(self, file_path: Path | str, config_slice: Any) -> Any | None:
        """Retrieve cached value if available, else return None."""
        try:
            key = self.get_key(file_path, config_slice)
        except FileNotFoundError:
            return None

        entry_path = self._path_for_key(key)
        if not entry_path.is_file():
            return None

        try:
            raw = entry_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            return payload.get("data")
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, file_path: Path | str, config_slice: Any, value: Any) -> Path:
        """Store a value in disk cache for the given file and config slice."""
        key = self.get_key(file_path, config_slice)
        entry_path = self._path_for_key(key)

        serializable_data: Any
        if isinstance(value, BaseModel):
            serializable_data = value.model_dump(mode="json")
        elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
            serializable_data = [item.model_dump(mode="json") for item in value]
        else:
            serializable_data = value

        payload = {
            "key": key,
            "source_path": str(file_path),
            "data": serializable_data,
        }

        entry_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return entry_path

    def delete(self, file_path: Path | str, config_slice: Any) -> bool:
        """Delete a cache entry if present."""
        try:
            key = self.get_key(file_path, config_slice)
            path = self._path_for_key(key)
            if path.is_file():
                path.unlink()
                return True
        except FileNotFoundError:
            pass
        return False

    def clear(self) -> int:
        """Remove all cache files in cache directory. Return count of removed files."""
        count = 0
        if self.cache_dir.is_dir():
            for p in self.cache_dir.glob("*.json"):
                if p.is_file():
                    p.unlink()
                    count += 1
        return count
