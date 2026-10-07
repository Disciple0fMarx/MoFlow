"""Regression tests for the self-healing norm-stats cache loader.

The cache file ``results_sdd/cor_fm/_norm_stats_ho<scene>.npz`` is a *transient*
artifact written with ``np.savez``. A hard crash mid-save leaves it truncated;
``np.load`` then dies with a cryptic ``EOFError: No data left in file`` and
every downstream tool (attribution, transfer matrix, gain geometry) aborts.
These tests pin the recovery contract of ``_load_norm_stats_cache``: corrupt ->
return ``None`` **and** delete the file so the caller recomputes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.visualize_trajectory_comparison import (
    _load_norm_stats_cache,
    _save_norm_stats_cache,
)


def _valid_cache(path: Path, offset: float = 0.0) -> None:
    np.savez(
        path,
        past_min=1.0 + offset,
        past_max=2.0 + offset,
        fut_min=3.0 + offset,
        fut_max=4.0 + offset,
    )


def test_loads_valid_cache(tmp_path: Path) -> None:
    p = tmp_path / "cache.npz"
    _valid_cache(p)
    assert _load_norm_stats_cache(p) == {
        "past_min": 1.0,
        "past_max": 2.0,
        "fut_min": 3.0,
        "fut_max": 4.0,
    }
    assert p.exists()  # healthy cache must be left in place


def test_truncated_cache_is_deleted_and_none_returned(tmp_path: Path) -> None:
    p = tmp_path / "cache.npz"
    _valid_cache(p)
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])  # simulate a crash mid-savez
    assert _load_norm_stats_cache(p) is None
    assert not p.exists()  # deleted so the caller recomputes, not re-crashes


def test_garbage_cache_is_deleted_and_none_returned(tmp_path: Path) -> None:
    p = tmp_path / "cache.npz"
    p.write_bytes(b"\x00\x01not a zip at all\xff\xfe")
    assert _load_norm_stats_cache(p) is None
    assert not p.exists()


def test_malformed_zip_missing_key_is_deleted(tmp_path: Path) -> None:
    p = tmp_path / "cache.npz"
    np.savez(p, past_min=1.0, fut_min=3.0)  # omits past_max / fut_max
    assert _load_norm_stats_cache(p) is None
    assert not p.exists()


def test_empty_file_is_deleted_and_none_returned(tmp_path: Path) -> None:
    p = tmp_path / "cache.npz"
    p.touch()
    assert _load_norm_stats_cache(p) is None
    assert not p.exists()


def test_atomic_save_writes_readable_cache(tmp_path: Path) -> None:
    """Regression: np.savez auto-appends .npz to names lacking it, so the temp
    name must already end in .npz or os.replace dies with ENOENT."""
    p = tmp_path / "_norm_stats_hobookstore.npz"
    _save_norm_stats_cache(p, (1.0, 2.0, 3.0, 4.0))
    assert p.exists()
    # no stray temp file left behind, and no <name>.tmp.npz
    leftovers = [f.name for f in tmp_path.iterdir()]
    assert leftovers == [p.name]
    data = np.load(p)
    assert (float(data["past_min"]), float(data["past_max"])) == (1.0, 2.0)
    assert (float(data["fut_min"]), float(data["fut_max"])) == (3.0, 4.0)


def test_atomic_save_regenerates_over_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "cache.npz"
    p.write_bytes(b"\x00garbage not a zip")
    _save_norm_stats_cache(p, (0.0, 1.0, 2.0, 3.0))
    assert _load_norm_stats_cache(p) == {
        "past_min": 0.0,
        "past_max": 1.0,
        "fut_min": 2.0,
        "fut_max": 3.0,
    }