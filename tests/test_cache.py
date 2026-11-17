"""Unit tests for file + config slice disk caching."""

from pathlib import Path

import pytest

from streamclip.cache import DiskCache, compute_config_hash, compute_file_hash
from streamclip.config import Config
from streamclip.models import SignalTrack


def test_compute_hashes(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_text("streamclip data", encoding="utf-8")

    h1 = compute_file_hash(f)
    assert isinstance(h1, str)
    assert len(h1) == 64

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        compute_file_hash(tmp_path / "does_not_exist.txt")

    # Config hash
    cfg = Config()
    c_hash = compute_config_hash(cfg.fusion_weights)
    assert isinstance(c_hash, str)
    assert len(c_hash) == 64


def test_disk_cache_get_set(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = DiskCache(cache_dir=cache_dir)

    source_file = tmp_path / "video.mp4"
    source_file.write_bytes(b"dummy video bytes 123456789")

    config_slice = {"fusion": "chat_only", "hz": 1.0}
    track = SignalTrack(name="chat_count", hz=1.0, values=[1.0, 5.0, 2.0])

    assert not cache.has(source_file, config_slice)
    assert cache.get(source_file, config_slice) is None

    # Set value
    cache.set(source_file, config_slice, track)
    assert cache.has(source_file, config_slice)

    # Retrieve value
    cached_val = cache.get(source_file, config_slice)
    assert cached_val is not None
    assert cached_val["name"] == "chat_count"
    assert cached_val["values"] == [1.0, 5.0, 2.0]


def test_disk_cache_key_invalidation_on_change(tmp_path: Path) -> None:
    cache = DiskCache(cache_dir=tmp_path / "cache")
    source_file = tmp_path / "chat.json"
    source_file.write_text("initial chat", encoding="utf-8")

    config_1 = {"threshold": 2.0}
    config_2 = {"threshold": 3.0}

    cache.set(source_file, config_1, {"result": 42})
    assert cache.has(source_file, config_1)
    assert not cache.has(source_file, config_2)

    # Modify file content -> hash changes -> cache miss
    source_file.write_text("updated chat with new messages", encoding="utf-8")
    assert not cache.has(source_file, config_1)


def test_disk_cache_clear(tmp_path: Path) -> None:
    cache = DiskCache(cache_dir=tmp_path / "cache")
    source_file = tmp_path / "video.mp4"
    source_file.write_bytes(b"test data")

    cache.set(source_file, {"k": 1}, "val1")
    cache.set(source_file, {"k": 2}, "val2")

    cleared = cache.clear()
    assert cleared == 2
    assert not cache.has(source_file, {"k": 1})
    assert not cache.has(source_file, {"k": 2})
