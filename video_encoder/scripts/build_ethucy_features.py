"""Build full-range ETH/UCY global video features from TU-Clausthal layout.

Supports nested TU-Clausthal paths (ETH/seq_eth, ETH/seq_hotel, UCY/students03,
UCY/zara01, UCY/zara02). If <scene_dir>/visual_data is missing/empty, auto-extracts
frames from video.avi using cv2.VideoCapture into visual_data/frame{idx:06d}.jpg,
then encodes at dataset stride and writes the standard per-scene cache consumed by
FrameFeatureLookup. Strict external path binding (--frames-root).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Ensure repo root on path for data.* imports when run as script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.tu_clausthal_eth_ucy_parser import TU_CLAUSTHAL_MAP
from video_encoder.backbones import build_backbone


def _resolve_scene_dir(frames_root: Path, scene: str) -> Path:
    if scene not in TU_CLAUSTHAL_MAP:
        raise ValueError(f"Unknown scene {scene}, valid: {sorted(set(TU_CLAUSTHAL_MAP.keys()))}")
    group, scene_dir = TU_CLAUSTHAL_MAP[scene]
    path = frames_root / group / scene_dir
    if not path.is_dir():
        raise FileNotFoundError(
            f"TU-Clausthal scene directory not found: {path}. "
            f"Expected nested layout under --frames-root ({frames_root}) with {group}/{scene_dir}."
        )
    return path


def _extract_frames_from_video(video_path: Path, visual_data_dir: Path) -> list[tuple[int, Path]]:
    visual_data_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video for auto-extraction: {video_path}")
    frames: list[tuple[int, Path]] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # Use 0-based frame numbering to match obsmat frame identifiers
            fid = idx
            out_path = visual_data_dir / f"frame{fid:06d}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise IOError(f"Failed to write extracted frame: {out_path}")
            frames.append((fid, out_path))
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frames


def _ensure_visual_data(scene_dir: Path) -> Path:
    visual_data_dir = scene_dir / "visual_data"
    has_frames = visual_data_dir.is_dir() and any(visual_data_dir.glob("frame*.jpg"))
    if has_frames:
        return visual_data_dir
    # Auto-extract from video.avi in the TU-Clausthal scene folder
    video_path = scene_dir / "video.avi"
    if not video_path.is_file():
        raise FileNotFoundError(
            f"visual_data missing/empty and video.avi not found for auto-extraction in: {scene_dir}. "
            "Strict external TU-Clausthal layout requires video.avi or pre-extracted visual_data/frame*.jpg."
        )
    print(f"[ethucy-features] auto-extracting frames from {video_path} -> {visual_data_dir}", flush=True)
    _extract_frames_from_video(video_path, visual_data_dir)
    return visual_data_dir


def _frame_files(frames_root: Path, scene: str, stride: int) -> list[tuple[int, Path]]:
    scene_dir = _resolve_scene_dir(frames_root, scene)
    visual_data_dir = _ensure_visual_data(scene_dir)
    frames = []
    for path in visual_data_dir.glob("frame*.jpg"):
        try:
            fid = int(path.stem.replace("frame", ""))
        except ValueError:
            continue
        if stride <= 1 or fid % stride == 0:
            frames.append((fid, path))
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise FileNotFoundError(f"No frame images found under {visual_data_dir} (stride={stride})")
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
    default_frames = Path("/home/efrei_stage/Desktop/tu-clausthal/datasets")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frames-root",
        default=str(default_frames),
        help="External TU-Clausthal datasets root (contains ETH/ and UCY/).",
    )
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
    if not frames_root.is_dir():
        raise FileNotFoundError(
            f"External --frames-root not found: {frames_root}. "
            "Expected TU-Clausthal datasets root (e.g. /home/efrei_stage/Desktop/tu-clausthal/datasets)."
        )
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
