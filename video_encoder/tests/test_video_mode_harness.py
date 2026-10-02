"""VIDEO_MODE isolation-harness tests on a synthetic SDD tree + cache.

The three modes:
    off    -> zero sentinel (numel==1; collate returns None), no cache touched
    static -> ONE fixed [D_raw] vector for every window of the run
    full   -> per-window mean-pooled lookup (differs across windows' videos)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from data.dataloader_sdd_global import SDDGlobalDataset
from video_encoder.tests.sdd_test_utils import (write_synthetic_sdd,
                                                write_synthetic_video_cache)


class _Cfg:
    class _CE:
        VIDEO_FEATURES_ROOT = None
        VIDEO_DIM_RAW = 512
        AGENTS = None

    MODEL = type("M", (object,), {"CONTEXT_ENCODER": _CE()})
    past_traj_min = past_traj_max = fut_traj_min = fut_traj_max = None


def _build(td: str, mode: str) -> SDDGlobalDataset:
    root = Path(td)
    write_synthetic_sdd(
        root, n_scenes=5, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
    )
    write_synthetic_video_cache(
        root / "feats", "bookstore", ("video0", "video1"), frames_per_video=60
    )
    return SDDGlobalDataset(
        _Cfg(),
        training=True,
        sdd_root=root,
        held_out_scene="coupa",
        use_video=True,
        video_features_root=root / "feats",
        video_mode=mode,
    )


def test_video_mode_off_returns_sentinel() -> None:
    with tempfile.TemporaryDirectory() as td:
        ds = _build(td, "off")
        for i in range(4):
            z = ds[i]["z_video_global"]
            assert z.dim() == 1 and z.numel() == 1


def test_video_mode_static_identical_across_windows() -> None:
    with tempfile.TemporaryDirectory() as td:
        ds = _build(td, "static")
        zs = [ds[i]["z_video_global"] for i in range(4)]
        for z in zs:
            assert z.dim() == 1 and z.numel() == 512
        assert all(torch.equal(zs[0], z) for z in zs)


def test_video_mode_full_varies_by_video() -> None:
    with tempfile.TemporaryDirectory() as td:
        ds = _build(td, "full")
        # bookstore windows (cached): video0 and video1 must resolve differently
        by_vid: dict[str, torch.Tensor] = {}
        for i in range(len(ds.windows)):
            it = ds[i]
            if it["scene"] != "bookstore":
                continue
            vid = it["video_id"]
            z = it["z_video_global"]
            assert z.dim() == 1 and z.numel() == 512
            assert abs(z.mean()) < 1100  # real features, not the zero sentinel
            if vid not in by_vid:
                by_vid[vid] = z
        assert set(by_vid) == {"video0", "video1"}
        assert not torch.equal(by_vid["video0"], by_vid["video1"])


def test_video_mode_invalid_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError):
            _build(td, "wat")


def test_video_id_filter_restricts_and_keeps_index_alignment() -> None:
    # Single-video ablation: --video-id video0 must (a) keep only video0
    # windows of the (LOSO-selected) scene, and (b) preserve the
    # scene_idx/video_idx -> name mapping after filtering (the empty-dict
    # alignment invariant in build_window_index).
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_synthetic_sdd(
            root, n_scenes=5, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
        )
        write_synthetic_video_cache(
            root / "feats", "coupa", ("video0", "video1"), frames_per_video=60
        )
        cfg = _Cfg()
        cfg.past_traj_min = cfg.past_traj_max = -100.0
        cfg.fut_traj_min = cfg.fut_traj_max = -100.0
        ds = SDDGlobalDataset(
            cfg,
            training=False,
            sdd_root=root,
            held_out_scene="coupa",
            split="test",
            use_video=True,
            video_features_root=root / "feats",
            video_mode="full",
            video_ids=["video0"],
        )
        assert len(ds.windows) > 0
        for i in range(len(ds)):
            it = ds[i]
            assert it["scene"] == "coupa"
            assert it["video_id"] == "video0"
            assert ds.windows.video_id(i) == "video0"


def test_video_id_filter_other_video_kills_half_the_windows() -> None:
    # video1 filter on the same scene: every surviving window is video1 —
    # proves the filter selects per-video, not merely shrinks the pool.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_synthetic_sdd(
            root, n_scenes=5, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
        )
        write_synthetic_video_cache(
            root / "feats", "coupa", ("video0", "video1"), frames_per_video=60
        )
        cfg = _Cfg()
        cfg.past_traj_min = cfg.past_traj_max = -100.0
        cfg.fut_traj_min = cfg.fut_traj_max = -100.0
        full = SDDGlobalDataset(
            cfg,
            training=False,
            sdd_root=root,
            held_out_scene="coupa",
            split="test",
            use_video=True,
            video_features_root=root / "feats",
            video_mode="full",
        )
        filtered = SDDGlobalDataset(
            cfg,
            training=False,
            sdd_root=root,
            held_out_scene="coupa",
            split="test",
            use_video=True,
            video_features_root=root / "feats",
            video_mode="full",
            video_ids=["video1"],
        )
        assert len(filtered) < len(full)
        for i in range(len(filtered)):
            assert filtered[i]["video_id"] == "video1"


def test_video_mode_off_ignores_missing_cache() -> None:
    # off must not touch the feature cache at all, even with a bogus root.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_synthetic_sdd(
            root, n_scenes=5, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
        )
        ds = SDDGlobalDataset(
            _Cfg(),
            training=True,
            sdd_root=root,
            held_out_scene="coupa",
            use_video=True,
            video_features_root=root / "nonexistent",
            video_mode="off",
        )
        for i in range(4):
            assert ds[i]["z_video_global"].dim() == 1
