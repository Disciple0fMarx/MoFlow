"""Build full-range ETH/UCY global video features from raw videos.

The legacy cache in ``features/resnet18`` only covers the validation segment of
each scene. Full raw training/test windows need every frame in the scene's
video timeline, so this script encodes each raw TU-Clausthal video at the
dataset stride and writes the standard per-scene cache consumed by
``FrameFeatureLookup``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data.tu_clausthal_eth_ucy_parser import load_scene_data
from video_encoder.custom_video import encode_custom_video


def _scene_video(data_root: str, scene: str) -> Path:
    info = load_scene_data(data_root, scene)
    if info["video_path"] is None:
        raise FileNotFoundError(
            f"No raw video found for scene '{scene}' under {info['scene_path']}"
        )
    return Path(info["video_path"])


def build_scene(
    data_root: str,
    scene: str,
    out_dir: Path,
    backbone: str,
    stride: int,
    device: str | None,
    batch_size: int,
) -> tuple[int, int]:
    video = _scene_video(data_root, scene)
    with tempfile.TemporaryDirectory(prefix=f"ethucy_feat_{scene}_") as tmp:
        res = encode_custom_video(
            video=video,
            out_dir=tmp,
            backbone=backbone,
            stride=stride,
            device=device,
            batch_size=batch_size,
        )
        feats = np.load(res["features"])
        manifest = pd.read_parquet(res["manifest"])

    manifest = manifest.rename(columns={"frame_idx": "frame_id"}).copy()
    manifest["scene"] = scene
    manifest["frame_id"] = manifest["frame_id"].astype(np.int64)
    manifest["row_idx"] = np.arange(len(manifest), dtype=np.int64)
    manifest = manifest[["scene", "frame_id", "row_idx"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{scene}.npy", feats.astype(np.float32))
    manifest.to_parquet(out_dir / f"{scene}.manifest.parquet", index=False)
    return feats.shape[0], feats.shape[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root", required=True, help="Strict TU-Clausthal datasets root"
    )
    p.add_argument("--out", default="features/resnet18")
    p.add_argument(
        "--scenes",
        nargs="*",
        default=["eth", "hotel", "univ", "zara1", "zara2"],
    )
    p.add_argument("--backbone", default="resnet18", choices=["resnet18"])
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    out_dir = Path(args.out)
    for scene in args.scenes:
        n, dim = build_scene(
            data_root=args.data_root,
            scene=scene,
            out_dir=out_dir,
            backbone=args.backbone,
            stride=args.stride,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"[ethucy-features] {scene}: {n} frames x {dim} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
