"""Perspective-scope audit: quantifies wrong-camera GV conditioning.

Two cameras in one scene share frame ids; the audit must report that the
pre-fix (frame-id-only) resolution would serve wrong-camera features for a
meaningful fraction of windows, while the post-fix (video-scoped) resolution
serves the correct camera for every frame.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from tools.audit_sdd_perspective import scope_stats_for_scene
from video_encoder.sdd_adapter import SDD_SCENES
from video_encoder.tests.sdd_test_utils import (
    write_synthetic_sdd,
    write_synthetic_video_cache,
)
from video_encoder.global_video_encoder_sdd import build_sdd_frame_index


def _run_scope_stats(td: str, with_cache: bool):
    pd_root = Path(td)
    write_synthetic_sdd(
        pd_root, n_scenes=1, videos_per_scene=2, n_tracks_per_video=1,
        frames_per_video=60, sdd_scenes=True,
    )
    feats = None
    if with_cache:
        write_synthetic_video_cache(
            pd_root / "feats", SDD_SCENES[0], ("video0", "video1"), frames_per_video=60
        )
        feats = pd_root / "feats"
    frame_df = build_sdd_frame_index(sdd_root=pd_root)
    df = scope_stats_for_scene(
        SDD_SCENES[0], frame_df, ["video0", "video1"], feats, pd_root
    )
    return df


def test_scope_stats_reports_wrong_camera_frames() -> None:
    with tempfile.TemporaryDirectory() as td:
        df = _run_scope_stats(td, with_cache=False)
        # every window-frame in this synthetic worst case resolves to the wrong
        # video under old (frame-id-only) semantics: video1 loses all collisions.
        assert len(df) == 2
        v0 = df[df["video_id"] == "video0"].iloc[0]
        v1 = df[df["video_id"] == "video1"].iloc[0]
        assert v0["n_frames_wrong_old"] == 0   # video0 wins frame-id collisions
        assert v1["n_frames_wrong_old"] > 0    # video1 entirely served from video0
        assert v0["n_frames_wrong_new"] == 0
        assert v1["n_frames_wrong_new"] == 0
        # per track there is ONE contiguous 60-frame run -> one 20-frame window
        assert v0["n_windows"] == 1
        assert v1["n_windows"] == 1


def test_scope_stats_with_real_cache_matches_annotation_derived() -> None:
    with tempfile.TemporaryDirectory() as td:
        df_no_f = _run_scope_stats(td, with_cache=False)
        df_f = _run_scope_stats(td, with_cache=True)
        assert (df_no_f["n_windows"] == df_f["n_windows"]).all()
        assert (df_no_f["n_frames_wrong_new"] == df_f["n_frames_wrong_new"]).all()