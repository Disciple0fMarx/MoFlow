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

from tools.visualize_trajectory_comparison import _load_norm_stats_cache


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