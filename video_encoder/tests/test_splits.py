"""Unit tests for the temporal split + LOSO logic. Uses a synthetic data_trajpred tree."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from video_encoder.scenes import CANONICAL_SCENES, scene_to_folder, scene_to_raw_txt
from video_encoder.splits import build_loso_splits, temporal_split


def _make_synthetic_root(root: Path, n_frames: int = 100, n_peds: int = 5):
    raw_dir = root / "raw" / "all_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for scene in CANONICAL_SCENES:
        # trajectories: every frame is multiple of 10 (matches Introvert stride)
        rows = []
        for f in range(0, n_frames * 10, 10):
            for pid in range(n_peds):
                rows.append((f, pid, rng.normal(), rng.normal()))
        txt = raw_dir / f"{scene_to_raw_txt(scene)}.txt"
        with txt.open("w") as fp:
            for f, pid, x, y in rows:
                fp.write(f"{f} {pid} {x:.4f} {y:.4f}\n")

        # visual_data: one tiny jpg per frame multiple of 10
        vd = root / scene_to_folder(scene) / "visual_data"
        vd.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (8, 8), color=(123, 222, 64))
        for f in range(0, n_frames * 10, 10):
            img.save(vd / f"frame{f:06d}.jpg", "JPEG", quality=50)


@pytest.fixture()
def synth_root(tmp_path):
    root = tmp_path / "data_trajpred"
    _make_synthetic_root(root)
    return root


def test_temporal_split_ratio():
    df = pd.DataFrame({"frame_id": np.repeat(np.arange(10), 3), "scene": "x", "ped_id": 0, "x": 0.0, "y": 0.0})
    sp = temporal_split(df, ratio=0.8)
    assert sp.train["frame_id"].max() < sp.val["frame_id"].min()
    assert sp.cutoff_frame_id == 8  # frame ids 0..7 = train (80%), 8..9 = val


def test_build_loso_disjoint_and_existing_paths(synth_root, tmp_path):
    out = tmp_path / "splits"
    paths = build_loso_splits(synth_root, leave_out="zara1", out_dir=out)
    test_train = pd.read_parquet(paths["test_train"])
    test_val = pd.read_parquet(paths["test_val"])
    train = pd.read_parquet(paths["train"])
    val = pd.read_parquet(paths["val"])

    # The held-out scene only appears in test_* splits.
    assert set(test_train["scene"].unique()) == {"zara1"}
    assert set(test_val["scene"].unique()) == {"zara1"}
    assert "zara1" not in set(train["scene"].unique())
    assert "zara1" not in set(val["scene"].unique())

    # 80/20 boundary per scene.
    for scene, sub in train.groupby("scene"):
        max_train = sub["frame_id"].max()
        min_val = val[val["scene"] == scene]["frame_id"].min()
        assert max_train < min_val

    # All frame_paths exist.
    for df in (test_train, test_val, train, val):
        for p in df["frame_path"].tolist():
            assert os.path.exists(p), p
