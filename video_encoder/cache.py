"""On-disk feature cache: one .npy per scene plus a row-index manifest."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def feature_paths(out_dir: str | Path, backbone: str, scene: str) -> tuple[Path, Path]:
    base = Path(out_dir) / backbone
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{scene}.npy", base / f"{scene}.manifest.parquet"


def write_scene_features(
    out_dir: str | Path,
    backbone: str,
    scene: str,
    frame_ids: list[int],
    features: np.ndarray,
) -> tuple[Path, Path]:
    feats_path, manifest_path = feature_paths(out_dir, backbone, scene)
    if features.shape[0] != len(frame_ids):
        raise ValueError(
            f"features rows ({features.shape[0]}) != frame_ids ({len(frame_ids)})"
        )
    np.save(feats_path, features.astype(np.float32))
    pd.DataFrame(
        {
            "scene": scene,
            "frame_id": frame_ids,
            "row_idx": np.arange(len(frame_ids), dtype=np.int64),
        }
    ).to_parquet(manifest_path, index=False)
    return feats_path, manifest_path


def load_scene_features(
    out_dir: str | Path, backbone: str, scene: str
) -> tuple[np.ndarray, pd.DataFrame]:
    feats_path, manifest_path = feature_paths(out_dir, backbone, scene)
    return np.load(feats_path), pd.read_parquet(manifest_path)
