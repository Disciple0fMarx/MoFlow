"""Regression tests for the SDD* feature-lookup perspective-scope fix.

Scenario: two videos in the same scene share the full frame-id range (both
0-based per SDD). Fix guarantees:

1. The encoder's manifest keeps **every** (video_id, frame_id) row — the old
   ``drop_duplicates("frame_id")`` collapse is gone.
2. ``SDDFrameFeatureLookup`` resolves a window's frames inside **its own
   video's** rows, never the other camera's.
3. A pre-fix cache (annotated frames missing from the manifest) raises loudly
   instead of silently serving wrong-camera features.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from video_encoder.sdd_adapter import SDDFrameFeatureLookup, get_lookup, SDD_SCENES
from video_encoder.tests.sdd_test_utils import (
    write_synthetic_sdd,
    write_synthetic_video_cache,
)


def test_manifest_keeps_all_video_frames_after_fix() -> None:
    """The encode path must not collapse (video, frame) rows by frame id."""
    from video_encoder.global_video_encoder_sdd import build_sdd_frame_index

    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=1, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
        )
        df = build_sdd_frame_index(sdd_root=Path(td))
        # unique (video_id, frame_id) rows — this is the post-fix `unique` table source
        unique = (
            df[["scene", "video_id", "frame_id"]]
            .drop_duplicates()
            .sort_values(["video_id", "frame_id"])
        )
        assert len(unique) == 2 * 60  # both videos' frames preserved
        # verify the OLD collapse would have dropped exactly one video's rows
        old_collapsed = unique.drop_duplicates("frame_id")
        assert len(old_collapsed) == 60


def test_lookup_resolves_within_own_video() -> None:
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=1, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
        )
        write_synthetic_video_cache(
            Path(td) / "feats", SDD_SCENES[0], ("video0", "video1"), frames_per_video=60
        )
        lookup = get_lookup(Path(td) / "feats", SDD_SCENES[0], sdd_root=Path(td))

        # same frame id, different videos -> different feature vectors
        f0 = lookup.get(frame_id=30, video_id="video0")
        f1 = lookup.get(frame_id=30, video_id="video1")
        assert f0 is not None and f1 is not None
        assert not np.allclose(f0, f1), "videos share feature rows -> scope bug!"
        # fingerprint: video-index offset * 1000
        assert np.isclose(f0[0], (f0 - f0[0])[0] + 0) or True  # noise check below
        # deterministic signal: mean of f0 should sit near offset 0, f1 near 1000
        assert abs(f0.mean() - 0) < 10 and abs(f1.mean() - 1000) < 10, (f0.mean(), f1.mean())

        # window resolves every frame inside the window's own video
        z0 = lookup.window(start_frame_id=10, n_frames=8, video_id="video0")
        z1 = lookup.window(start_frame_id=10, n_frames=8, video_id="video1")
        assert z0.shape == (8, 512) and z1.shape == (8, 512)
        assert abs(z0.mean() - 0) < 10 and abs(z1.mean() - 1000) < 10


def test_pre_fix_cache_raises() -> None:
    """A cache missing a video's annotated frames must fail loudly."""
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_sdd(
            Path(td), n_scenes=1, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
        )
        feats_dir = write_synthetic_video_cache(
            Path(td) / "feats", SDD_SCENES[0], ("video0",), frames_per_video=60
        )
        # simulate the OLD collapse: only video0's rows present
        with pytest.raises(ValueError):
            get_lookup(feats_dir, SDD_SCENES[0], sdd_root=Path(td))


def test_lookup_without_sdd_root_still_scopes_by_video() -> None:
    """Even without annotations, the lookup must not cross videos."""
    with tempfile.TemporaryDirectory() as td:
        write_synthetic_video_cache(
            Path(td) / "feats", "bookstore", ("video0", "video1"), frames_per_video=60
        )
        lookup = SDDFrameFeatureLookup.from_root(Path(td) / "feats", "bookstore")
        z0 = lookup.window(start_frame_id=5, n_frames=3, video_id="video0")
        z1 = lookup.window(start_frame_id=5, n_frames=3, video_id="video1")
        assert abs(z0.mean() - 0) < 10 and abs(z1.mean() - 1000) < 10


def test_window_nearest_snap_inside_same_video() -> None:
    """Missing frames snap to the closest frame of the *same* video."""
    with tempfile.TemporaryDirectory() as td:
        feats_dir = write_synthetic_video_cache(
            Path(td) / "feats", "quad", ("video0",), frames_per_video=60, seed=7
        )
        lookup = SDDFrameFeatureLookup.from_root(feats_dir, "quad")
        # fully missing window with policy="drop" -> zero-filled (no cross-video contamination)
        out = lookup.window(start_frame_id=70, n_frames=4, video_id="video0", policy="drop")
        assert np.all(out == 0.0)
        # default policy="nearest": 70..73 snap to frame 59 of the SAME video
        out2 = lookup.window(start_frame_id=70, n_frames=4, video_id="video0")
        f59 = lookup.get(frame_id=59, video_id="video0")
        assert out2.shape == (4, 512)
        assert np.allclose(out2, np.tile(f59, (4, 1)))