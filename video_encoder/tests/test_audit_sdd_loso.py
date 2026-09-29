"""Verify the LOSO unit is the *scene category* (all videos/perspectives).

Regression tests for ``tools/audit_sdd_loso.py`` against a synthetic SDD tree:
every held-out scene's videos are excluded from train, window keys are
disjoint, and frame-id overlaps across videos within a scene are surfaced.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from video_encoder.sdd_adapter import SDD_SCENES
from tools.audit_sdd_loso import (
    census_scene_videos,
    check_loso_disjointness,
    frame_id_overlap,
)
from video_encoder.tests.sdd_test_utils import write_synthetic_sdd


def test_loso_checks_pass_on_synthetic_root() -> None:
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=5, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
        )
        checks = check_loso_disjointness(Path(td))
        # synthetic root only has the first 5 canonical scenes
        assert {c.held_out for c in checks} == set(SDD_SCENES[:5])
        for c in checks:
            assert c.ok, f"LOSO check failed for held-out={c.held_out}: {c}"
            assert c.leaked_windows == 0
            assert c.leaked_video_ids == []
            assert c.n_train_scenes == 4
            assert c.n_test_scenes == 1


def test_loso_train_excludes_held_out_videos() -> None:
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=3, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
        )
        checks = check_loso_disjointness(Path(td))
        assert len(checks) == 3
        for c in checks:
            # train videos never include the held-out scene's video ids
            held_scene_videos = [v for v in c.test_videos]
            assert len(held_scene_videos) == 2  # 2 videos per synthetic scene
            assert set(c.test_videos).isdisjoint(c.train_videos)


def test_census_counts_windows_and_videos() -> None:
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=2, videos_per_scene=3, n_tracks_per_video=1,
            frames_per_video=60, sdd_scenes=True,
        )
        census = census_scene_videos(Path(td), list(SDD_SCENES[:2]))
        assert len(census) == 6  # 2 scenes x 3 videos
        # frames_per_video=60 frames form ONE contiguous run -> one 20-frame window
        for row in census:
            assert row.n_windows == 1, f"{row.scene}/{row.video_id}: {row.n_windows} windows"
            assert row.n_frames == 60
            assert row.frame_min == 0
            assert row.frame_max == 59


def test_frame_id_overlap_between_videos_detected() -> None:
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=1, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
        )
        overlaps = frame_id_overlap(Path(td), [SDD_SCENES[0]])
        # both videos share the full 0..59 frame range -> one overlap row
        assert len(overlaps) == 1
        assert overlaps[0].scene == SDD_SCENES[0]
        assert overlaps[0].shared_frame_ids == 60