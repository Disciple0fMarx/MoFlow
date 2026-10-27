"""Build full-range ETH/UCY global video features from ``visual_data`` frames.

The legacy cache in ``features/resnet18`` only covers the validation segment of
each scene. Full raw training/test windows need every frame in the scene's
timeline. This script encodes the repo-local frame images (which cover the full
timelines) at the dataset stride and writes the standard per-scene cache
consumed by ``FrameFeatureLookup``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from video_encoder.backbones import build_backbone


def _frame_files(frames_root: Path, scene: str, stride: int) -> list[tuple[int, Path]]:
    scene_dir = frames_root / scene / "visual_data"
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Missing frame directory: {scene_dir}")
    frames = []
    for path in scene_dir.glob("frame*.jpg"):
        fid = int(path.stem.replace("frame", ""))
        if fid % stride == 0:
            frames.append((fid, path))
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise FileNotFoundError(f"No frame images found under {scene_dir}")
    return frames


@torch.inference_mode()
def build_scene(
    frames_root: Path,
    scene: str,
    out_dir: Path,
    stride: int,
    device: str | None,
    batch_size: int,
) -> tuple[int, int]:
    model, dim, preprocess = build_backbone("resnet18")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    frames = _frame_files(frames_root, scene, stride)
    feats: list[np.ndarray] = []
    buf: list[torch.Tensor] = []
    for _, path in tqdm(frames, desc=f"encode {scene}", unit="frame"):
        buf.append(preprocess(Image.open(path).convert("RGB")))
        if len(buf) >= batch_size:
            x = torch.stack(buf, dim=0).to(dev, non_blocking=True)
            feats.append(model(x).detach().cpu().numpy().astype(np.float32))
            buf.clear()
    if buf:
        x = torch.stack(buf, dim=0).to(dev, non_blocking=True)
        feats.append(model(x).detach().cpu().numpy().astype(np.float32))

    features = np.concatenate(feats, axis=0)
    manifest = pd.DataFrame(
        {
            "scene": scene,
            "frame_id": np.asarray([fid for fid, _ in frames], dtype=np.int64),
            "row_idx": np.arange(len(frames), dtype=np.int64),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{scene}.npy", features)
    manifest.to_parquet(out_dir / f"{scene}.manifest.parquet", index=False)
    return features.shape[0], dim


def main() -> None:
    default_frames = Path(__file__).resolve().parents[2] / "data" / "eth_ucy"
    p = argparse.ArgumentParser()
    p.add_argument("--frames-root", default=str(default_frames))
    p.add_argument("--out", default="features/resnet18")
    p.add_argument(
        "--scenes",
        nargs="*",
        default=["eth", "hotel", "univ", "zara1", "zara2"],
    )
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    out_dir = Path(args.out)
    frames_root = Path(args.frames_root)
    for scene in args.scenes:
        n, dim = build_scene(
            frames_root=frames_root,
            scene=scene,
            out_dir=out_dir,
            stride=args.stride,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"[ethucy-features] {scene}: {n} frames x {dim} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
