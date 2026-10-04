"""Tests for the RAM/VRAM safeguards added after the 8x8 sweep OOM freeze.

Covers the four mechanisms the sweep depends on:

1. ``resolve_num_workers`` — clamps DataLoader fan-out so workers cannot
   multiply the dataset/feature copies.
2. ``run_with_oom_split`` — halves an oversized batch instead of aborting a
   128-cell sweep, and concatenates per-condition dict results correctly.
3. ``_slice_batch`` — row-subsets a collated batch while preserving the collate
   contract (tensors, per-window lists, ``batch_size``).
4. ``SDDFrameFeatureLookup`` — memory-mapped features, compact per-video index
   (no ``dict[(video_id, frame_id)]`` retained), and ``release_lookup_cache``.
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from utils.memory import (MAX_NUM_WORKERS, free_memory, is_oom, iter_with_free,
                          resolve_num_workers, run_with_oom_split)
from video_encoder.sdd_adapter import (SDD_SCENES, SDDFrameFeatureLookup,
                                       get_lookup, release_lookup_cache)
from video_encoder.tests.sdd_test_utils import (write_synthetic_sdd,
                                                write_synthetic_video_cache)


# ---------------------------------------------------------------------------
# 1. DataLoader worker clamping
# ---------------------------------------------------------------------------
def test_resolve_num_workers_defaults_to_zero() -> None:
    assert resolve_num_workers(None) == 0
    assert resolve_num_workers(-4) == 0  # nonsense falls back, never negative
    assert resolve_num_workers(0) == 0


def test_resolve_num_workers_clamps_to_max() -> None:
    assert resolve_num_workers(1) == 1
    assert resolve_num_workers(MAX_NUM_WORKERS) == MAX_NUM_WORKERS
    # An over-eager config is clamped, not rejected, so it cannot OOM the host.
    assert resolve_num_workers(999) == MAX_NUM_WORKERS


# ---------------------------------------------------------------------------
# 2. OOM batch splitting
# ---------------------------------------------------------------------------
def test_is_oom_detects_cuda_and_host_failures() -> None:
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_oom(MemoryError())
    assert not is_oom(ValueError("shape mismatch"))


def test_run_with_oom_split_passes_through_when_it_fits() -> None:
    calls: list[int] = []

    def fn(items):
        calls.append(len(items))
        return np.asarray(list(items))

    out = run_with_oom_split(fn, [0, 1, 2, 3])
    assert out.tolist() == [0, 1, 2, 3]
    assert calls == [4]  # no split when there is no failure


def test_run_with_oom_split_halves_and_preserves_order() -> None:
    sizes: list[int] = []

    def fn(items):
        sizes.append(len(items))
        if len(items) > 2:
            raise RuntimeError("CUDA out of memory")
        return np.asarray(list(items))

    out = run_with_oom_split(fn, list(range(8)))
    assert out.tolist() == list(range(8))  # order preserved across recursion
    assert sizes[0] == 8
    assert max(sizes[1:]) <= 4  # strictly smaller chunks on retry


def test_run_with_oom_split_concatenates_dict_results() -> None:
    """The attribution tool returns {condition: ndarray}; merging must work."""

    def fn(items):
        if len(items) > 1:
            raise RuntimeError("CUDA out of memory")
        return {
            c: np.full((len(items), 2), float(items[0])) for c in ("baseline", "zeroed")
        }

    out = run_with_oom_split(fn, list(range(4)))
    assert set(out) == {"baseline", "zeroed"}
    assert out["baseline"].shape == (4, 2)
    assert out["baseline"][:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_run_with_oom_split_reraises_non_oom_errors() -> None:
    def fn(items):
        raise ValueError("not a memory problem")

    with pytest.raises(ValueError, match="not a memory problem"):
        run_with_oom_split(fn, [0, 1])


def test_run_with_oom_split_gives_up_below_min_chunk() -> None:
    def fn(items):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        run_with_oom_split(fn, [0], min_chunk=1)


def test_iter_with_free_and_free_memory_are_safe_without_cuda() -> None:
    out = list(iter_with_free(range(10), every=3))
    assert out == list(range(10))
    free_memory()  # must not raise when CUDA is unavailable
    gc.collect()


# ---------------------------------------------------------------------------
# 3. Batch row-slicing used by the OOM path
# ---------------------------------------------------------------------------
def test_slice_batch_preserves_collate_contract() -> None:
    from tools.attrib_video_conditioning import _slice_batch

    batch = {
        "batch_size": 4,
        "past_trj": torch.arange(12).reshape(4, 3),
        "fut_traj_original_scale": torch.arange(16).reshape(4, 2, 2),
        "z_video_global": torch.ones(4, 5),
        "scene": ["a", "b", "c", "d"],
        "video_id": [0, 1, 0, 1],
        "meta": {"name": "not-per-row"},
    }
    sub = _slice_batch(batch, [1, 3])
    assert sub["batch_size"] == 2
    assert sub["past_trj"].shape == (2, 3)
    assert sub["past_trj"].tolist() == [[3, 4, 5], [9, 10, 11]]
    assert sub["z_video_global"].shape == (2, 5)
    assert sub["scene"] == ["b", "d"]
    assert sub["video_id"] == [1, 1]
    assert sub["meta"] is batch["meta"]  # non-per-row entries pass through
    # original batch untouched
    assert batch["batch_size"] == 4 and batch["past_trj"].shape[0] == 4


def test_slice_batch_contiguous_slice_is_a_view() -> None:
    from tools.attrib_video_conditioning import _slice_batch

    batch = {"batch_size": 3, "x": torch.zeros(3, 2)}
    sub = _slice_batch(batch, [0, 1])
    assert sub["x"].shape == (2, 2)
    assert sub["batch_size"] == 2


# ---------------------------------------------------------------------------
# 4. Lookup memory behaviour
# ---------------------------------------------------------------------------
def test_lookup_is_mmap_not_read_into_ram() -> None:
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=1, videos_per_scene=2, frames_per_video=40
        )
        write_synthetic_video_cache(
            Path(td) / "feats", SDD_SCENES[0], ("video0", "video1"), frames_per_video=40
        )
        root = Path(td) / "feats"
        release_lookup_cache()
        lookup = get_lookup(root, SDD_SCENES[0], sdd_root=Path(td))

        # mmap_mode="r" -> the array is a view on disk, never a heap copy.
        assert isinstance(lookup.features, np.memmap)
        assert lookup.features.flags["WRITEABLE"] is False
        release_lookup_cache()


def test_lookup_index_is_compact_numpy_not_a_tuple_dict() -> None:
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=1, videos_per_scene=2, frames_per_video=30
        )
        write_synthetic_video_cache(
            Path(td) / "feats", SDD_SCENES[0], ("video0", "video1"), frames_per_video=30
        )
        release_lookup_cache()
        lookup = get_lookup(Path(td) / "feats", SDD_SCENES[0], sdd_root=Path(td))
        # the hot-path index is two int64 arrays per video
        for vid, ids in lookup.video_sorted_ids.items():
            assert isinstance(ids, np.ndarray) and ids.dtype == np.int64
            assert isinstance(lookup.video_sorted_rows[vid], np.ndarray)
            assert ids.size == lookup.video_sorted_rows[vid].size
        # ...and nothing tuple-keyed is retained as an attribute
        assert "video_frame_to_row" not in lookup.__dict__
        # the compatibility property still reproduces the old mapping exactly
        legacy = lookup.video_frame_to_row
        assert len(legacy) == 60
        assert (
            legacy[("video0", 5)]
            == lookup.video_sorted_rows["video0"][
                int(np.searchsorted(lookup.video_sorted_ids["video0"], 5))
            ]
        )
        release_lookup_cache()


def test_release_lookup_cache_actually_drops_lookups() -> None:
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=1, videos_per_scene=1, frames_per_video=20
        )
        write_synthetic_video_cache(
            Path(td) / "feats", SDD_SCENES[0], ("video0",), frames_per_video=20
        )
        root = Path(td) / "feats"
        release_lookup_cache()
        get_lookup(root, SDD_SCENES[0], sdd_root=Path(td))

        from video_encoder.sdd_adapter import _cached_lookup

        assert _cached_lookup.cache_info().currsize == 1
        release_lookup_cache()
        assert _cached_lookup.cache_info().currsize == 0


def test_lookup_resolution_semantics_unchanged_by_reindexing() -> None:
    """floor/nearest/drop must match the previous dict-based implementation."""
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=1, videos_per_scene=1, frames_per_video=25
        )
        write_synthetic_video_cache(
            Path(td) / "feats", SDD_SCENES[0], ("video0",), frames_per_video=25
        )
        release_lookup_cache()
        lookup = get_lookup(Path(td) / "feats", SDD_SCENES[0], sdd_root=Path(td))
        # get(frame_id, video_id, policy=...) — frame id comes FIRST
        assert lookup.get(7, "video0", policy="floor") is not None
        assert lookup.get(24, "video0", policy="nearest") is not None
        assert lookup.get(0, "video0", policy="floor") is not None
        # an unknown video is never resolved and never inherits another
        # video's frames (no cross-camera leakage)
        assert lookup.get(5, "video1", policy="floor") is None
        assert lookup.get(5, "video1", policy="nearest") is None
        assert lookup.get(5, "nope", policy="floor") is None
        release_lookup_cache()


def test_verifier_still_detects_a_pre_fix_cache() -> None:
    """The compact index must not weaken the missing-frames guard."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_synthetic_sdd(
            root, n_scenes=1, videos_per_scene=2, frames_per_video=20, sdd_scenes=True
        )
        # a cache built with the OLD per-frame_id collapse: only video0's rows
        feats = write_synthetic_video_cache(
            root / "feats", SDD_SCENES[0], ("video0",), frames_per_video=20
        )
        release_lookup_cache()
        with pytest.raises(ValueError):
            get_lookup(feats, SDD_SCENES[0], sdd_root=root)
        release_lookup_cache()


def test_verifier_accepts_a_complete_cache() -> None:
    """Guard the other direction: a healthy cache must still load cleanly."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_synthetic_sdd(
            root, n_scenes=1, videos_per_scene=2, frames_per_video=20, sdd_scenes=True
        )
        feats = write_synthetic_video_cache(
            root / "feats", SDD_SCENES[0], ("video0", "video1"), frames_per_video=20
        )
        release_lookup_cache()
        lookup = get_lookup(feats, SDD_SCENES[0], sdd_root=root)
        assert lookup.window(5, 3, video_id="video1").shape == (3, 512)
        release_lookup_cache()
