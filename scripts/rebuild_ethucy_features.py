#!/usr/bin/env python3
"""Purge and rebuild ETH/UCY global video feature caches with exact coverage.

The LOSO loaders sample observed histories at ``start + k * annotation_step``
where both the start frames and ``annotation_step == 10`` follow the benchmark
10-frame grid.  This script computes the union of every frame id required by
the committed splits and asks the AVI-backed encoder for precisely that set,
so strict coverage holds even if future splits change.

Usage::

    python scripts/rebuild_ethucy_features.py \
        --frames-root /path/to/tu-clausthal/datasets \
        --out features/resnet18 \
        --purge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.tu_clausthal_eth_ucy_parser import (load_split_ranges,
                                              load_trajectories,
                                              window_trajectories)
from utils.eth_ucy_geometry import observed_frame_ids
from video_encoder.scripts.build_ethucy_features import build_scene

CANONICAL = ("eth", "hotel", "univ", "zara1", "zara2")


def required_ids_for_scene(
    frames_root: Path,
    scene: str,
    past_frames: int,
    future_frames: int,
    split_file: str | None,
) -> list[int]:
    loaded = load_trajectories(str(frames_root), scene)
    entry = load_split_ranges(split_file)[scene]
    ranges = [entry["train"], entry["val"], entry["test"]]
    _, starts, _ = window_trajectories(
        loaded["traj"], past_frames, future_frames, ranges
    )
    ids: set[int] = set()
    for start in starts:
        ids.update(observed_frame_ids(int(start), scene, past_frames))
    if not ids:
        raise RuntimeError(f"No required frame ids computed for scene={scene}")
    return sorted(ids)


def verify_cache(out_dir: Path, scene: str, expected: list[int]) -> dict:
    import pandas as pd

    manifest_path = out_dir / f"{scene}.manifest.parquet"
    features_path = out_dir / f"{scene}.npy"
    if not manifest_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(f"Missing rebuilt cache for {scene}: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    features = np.load(features_path, mmap_mode="r")
    got = set(int(x) for x in manifest["frame_id"].tolist())
    want = set(int(x) for x in expected)
    missing = sorted(want - got)
    extra = sorted(got - want)
    if missing:
        raise RuntimeError(
            f"{scene}: cache misses {len(missing)} required ids: {missing[:10]}"
        )
    if extra:
        raise RuntimeError(
            f"{scene}: cache has {len(extra)} unexpected ids: {extra[:10]}"
        )
    if features.shape[0] != len(manifest):
        raise RuntimeError(
            f"{scene}: feature rows {features.shape[0]} != manifest rows {len(manifest)}"
        )
    if not np.isfinite(np.asarray(features[: min(8, len(features))])).all():
        raise RuntimeError(f"{scene}: non-finite feature values in rebuilt cache")
    return {
        "scene": scene,
        "frames": int(len(manifest)),
        "dim": int(features.shape[1]),
        "min_frame": int(min(want)),
        "max_frame": int(max(want)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-root", required=True)
    p.add_argument("--out", default=str(_REPO_ROOT / "features" / "resnet18"))
    p.add_argument("--scenes", nargs="*", default=list(CANONICAL))
    p.add_argument("--past-frames", type=int, default=8)
    p.add_argument("--future-frames", type=int, default=12)
    p.add_argument("--split-file", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--purge",
        action="store_true",
        help="Delete the selected scenes' .npy/.manifest.parquet before rebuilding.",
    )
    p.add_argument(
        "--dry-run-ids",
        action="store_true",
        help="Only compute and print required frame ids; do not encode.",
    )
    args = p.parse_args()

    frames_root = Path(args.frames_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not frames_root.is_dir():
        raise FileNotFoundError(f"--frames-root not found: {frames_root}")

    ids_map = {
        scene: required_ids_for_scene(
            frames_root,
            scene,
            args.past_frames,
            args.future_frames,
            args.split_file,
        )
        for scene in args.scenes
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = out_dir / "required_frame_ids.json"
    with open(ids_path, "w", encoding="utf-8") as fh:
        json.dump(ids_map, fh, indent=2, sort_keys=True)
    print(f"[rebuild] required ids -> {ids_path}", flush=True)
    for scene, ids in ids_map.items():
        print(
            f"[rebuild] {scene}: {len(ids)} unique frames " f"({min(ids)}..{max(ids)})",
            flush=True,
        )
    if args.dry_run_ids:
        return

    if args.purge:
        for scene in args.scenes:
            for suffix in (".npy", ".manifest.parquet"):
                target = out_dir / f"{scene}{suffix}"
                if target.exists():
                    target.unlink()
                    print(f"[rebuild] purged {target}", flush=True)

    reports = []
    for scene in args.scenes:
        n, dim = build_scene(
            frames_root=frames_root,
            scene=scene,
            out_dir=out_dir,
            stride=None,
            device=args.device,
            batch_size=args.batch_size,
            frame_ids=ids_map[scene],
        )
        report = verify_cache(out_dir, scene, ids_map[scene])
        report["encoded"] = int(n)
        report["dim"] = int(dim)
        reports.append(report)
        print(f"[rebuild] verified {scene}: {report}", flush=True)

    summary = out_dir / "rebuild_summary.json"
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"scenes": reports}, fh, indent=2, sort_keys=True)
    print(f"[rebuild] summary -> {summary}", flush=True)


if __name__ == "__main__":
    main()
