"""Deterministic 80/20 temporal splits + leave-one-scene-out assembler."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd

from .scenes import CANONICAL_SCENES, others, scene_to_folder
from .trajectories import load_raw_trajectories
from .frames import list_available_frames, snap_frame_id, frame_path


@dataclass
class TemporalSplit:
    train: pd.DataFrame  # earliest 80% of unique frame_ids
    val: pd.DataFrame    # latest 20%
    cutoff_frame_id: int


def temporal_split(df: pd.DataFrame, ratio: float = 0.8) -> TemporalSplit:
    """Split rows by sorted unique frame_id at the `ratio` quantile (no shuffling)."""
    if df.empty:
        return TemporalSplit(df.copy(), df.copy(), 0)
    unique_frames = sorted(df["frame_id"].unique().tolist())
    cut_idx = int(len(unique_frames) * ratio)  # floor
    cut_idx = max(1, min(cut_idx, len(unique_frames) - 1))
    cutoff = unique_frames[cut_idx]  # first frame of the val side
    train = df[df["frame_id"] < cutoff].reset_index(drop=True)
    val = df[df["frame_id"] >= cutoff].reset_index(drop=True)
    return TemporalSplit(train=train, val=val, cutoff_frame_id=int(cutoff))


def _attach_frame_paths(
    df: pd.DataFrame,
    data_root: Path,
    policy: str = "nearest",
) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["snapped_frame_id"] = pd.Series(dtype="int64")
        df["frame_path"] = pd.Series(dtype="object")
        return df
    out_rows = []
    for scene, sub in df.groupby("scene", sort=False):
        available = list_available_frames(str(data_root), scene)
        snapped = []
        paths = []
        keep = []
        for fid in sub["frame_id"].tolist():
            s = snap_frame_id(int(fid), available, policy=policy)  # type: ignore[arg-type]
            if s is None:
                snapped.append(-1)
                paths.append("")
                keep.append(False)
            else:
                snapped.append(s)
                paths.append(str(frame_path(data_root, scene, s)))
                keep.append(True)
        sub = sub.copy()
        sub["snapped_frame_id"] = snapped
        sub["frame_path"] = paths
        sub = sub[keep]
        out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True) if out_rows else df


def build_loso_splits(
    data_root: str | Path,
    leave_out: str,
    out_dir: str | Path,
    snap_policy: str = "nearest",
    ratio: float = 0.8,
) -> dict[str, Path]:
    """Build the leave-one-scene-out splits described in the protocol.

    Writes 4 parquet files under <out_dir>/<leave_out>/:
        test_train.parquet, test_val.parquet  (held-out scene's 80/20)
        train.parquet, val.parquet            (other scenes' 80% / 20%)
    Returns a dict of split_name -> path.
    """
    if leave_out not in CANONICAL_SCENES:
        raise ValueError(f"leave_out must be one of {CANONICAL_SCENES}, got {leave_out!r}")
    data_root = Path(data_root)
    out_dir = Path(out_dir) / leave_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Held-out scene.
    held = load_raw_trajectories(data_root, leave_out)
    held_split = temporal_split(held, ratio=ratio)
    test_train = _attach_frame_paths(held_split.train, data_root, snap_policy)
    test_val = _attach_frame_paths(held_split.val, data_root, snap_policy)

    # Remaining scenes — concat 80% slices into train, 20% slices into val.
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    per_scene_cutoffs: dict[str, int] = {leave_out: held_split.cutoff_frame_id}
    for scene in others(leave_out):
        df = load_raw_trajectories(data_root, scene)
        sp = temporal_split(df, ratio=ratio)
        per_scene_cutoffs[scene] = sp.cutoff_frame_id
        train_parts.append(sp.train)
        val_parts.append(sp.val)

    train = _attach_frame_paths(pd.concat(train_parts, ignore_index=True), data_root, snap_policy)
    val = _attach_frame_paths(pd.concat(val_parts, ignore_index=True), data_root, snap_policy)

    paths = {
        "test_train": out_dir / "test_train.parquet",
        "test_val": out_dir / "test_val.parquet",
        "train": out_dir / "train.parquet",
        "val": out_dir / "val.parquet",
    }
    test_train.to_parquet(paths["test_train"], index=False)
    test_val.to_parquet(paths["test_val"], index=False)
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["val"], index=False)

    # Sidecar manifest with cutoffs + row counts.
    manifest = pd.DataFrame(
        [
            {"split": k, "rows": len(df), "scenes": ",".join(sorted(df["scene"].unique().tolist())) if len(df) else ""}
            for k, df in [
                ("test_train", test_train),
                ("test_val", test_val),
                ("train", train),
                ("val", val),
            ]
        ]
    )
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    pd.DataFrame(
        [{"scene": s, "cutoff_frame_id": c} for s, c in per_scene_cutoffs.items()]
    ).to_csv(out_dir / "cutoffs.csv", index=False)

    return paths


def unique_scene_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique (scene, snapped_frame_id, frame_path) rows for encoding."""
    cols = ["scene", "snapped_frame_id", "frame_path"]
    return df[cols].drop_duplicates().reset_index(drop=True)
