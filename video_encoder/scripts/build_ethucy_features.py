"""Build ETH/UCY ResNet18 global video feature caches (strict external-only).

Invariants:
- Read source videos EXCLUSIVELY from external TU-Clausthal mirror:
  <frames-root>/(ETH|UCY)/<scene_dir>/video.avi (resolved via TU_CLAUSTHAL_MAP).
- Write ONLY precomputed feature caches (<scene>.npy, <scene>.manifest.parquet)
  to the repository output directory (features/resnet18). No dataset assets,
  extracted frames, or visual_data folders are created/read inside _REPO_ROOT.
- 0-based frame indexing for obsmat parity: decoded frame idx -> frame_id = idx
  (covers fid 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, Tuple

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


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _resolve_video_path(frames_root: Path, scene: str) -> Path:
    frames_root = frames_root.resolve()
    if _is_under(frames_root, _REPO_ROOT):
        raise ValueError(
            f"--frames-root must be an external TU-Clausthal path, not inside the repo: {frames_root} (repo={_REPO_ROOT})"
        )
    if scene not in TU_CLAUSTHAL_MAP:
        raise ValueError(f"Unknown scene {scene}, valid: {sorted(set(TU_CLAUSTHAL_MAP.keys()))}")
    group, scene_dir = TU_CLAUSTHAL_MAP[scene]
    scene_path = frames_root / group / scene_dir
    if not scene_path.is_dir():
        raise FileNotFoundError(
            f"TU-Clausthal scene directory not found (external-only): {scene_path}. "
            f"Expected nested layout under --frames-root ({frames_root}) with {group}/{scene_dir}."
        )
    # Never read/create visual_data inside repo; do not use repo paths.
    if _is_under(scene_path, _REPO_ROOT):
        raise ValueError(f"Resolved scene path is inside repo (forbidden): {scene_path}")
    video_path = scene_path / "video.avi"
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Source video not found (external-only): {video_path}. "
            "build_ethucy_features reads exclusively from external TU-Clausthal video.avi."
        )
    return video_path


def _iter_video_frames(video_path: Path, stride: int) -> Iterator[Tuple[int, np.ndarray]]:
    """Stream frames from external video.avi with 0-based obsmat-parity indexing."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open external video: {video_path}")
    idx = 0
    yielded = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            fid = idx  # 0-based for obsmat parity
            if stride <= 1 or fid % stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield fid, frame_rgb
                yielded += 1
            idx += 1
    finally:
        cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames decoded from external video: {video_path}")
    if yielded == 0:
        raise RuntimeError(f"No frames selected at stride={stride} from external video: {video_path}")


@torch.inference_mode()
def build_scene(
    frames_root: Path,
    scene: str,
    out_dir: Path,
    stride: int,
    device: str | None,
    batch_size: int,
) -> tuple[int, int]:
    out_dir = out_dir.resolve()
    # Cache output is allowed in repo (features/resnet18); ensure we don't write dataset assets elsewhere.
    if not _is_under(out_dir, _REPO_ROOT):
        # allow external out if explicitly requested, but primary invariant is repo features output
        pass

    video_path = _resolve_video_path(frames_root, scene)
    model, dim, preprocess = build_backbone("resnet18")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    frame_ids: list[int] = []
    feats: list[np.ndarray] = []
    buf: list[torch.Tensor] = []
    buf_fids: list[int] = []

    for fid, frame_rgb in tqdm(_iter_video_frames(video_path, stride), desc=f"encode {scene}", unit="frame"):
        buf.append(preprocess(Image.fromarray(frame_rgb)))
        buf_fids.append(fid)
        if len(buf) >= batch_size:
            x = torch.stack(buf, dim=0).to(dev, non_blocking=True)
            feats.append(model(x).detach().cpu().numpy().astype(np.float32))
            frame_ids.extend(buf_fids)
            buf.clear()
            buf_fids.clear()
    if buf:
        x = torch.stack(buf, dim=0).to(dev, non_blocking=True)
        feats.append(model(x).detach().cpu().numpy().astype(np.float32))
        frame_ids.extend(buf_fids)

    features = np.concatenate(feats, axis=0)
    if features.shape[0] != len(frame_ids):
        raise RuntimeError(f"Feature/manifest size mismatch for {scene}: {features.shape[0]} vs {len(frame_ids)}")
    manifest = pd.DataFrame(
        {
            "scene": scene,
            "frame_id": np.asarray(frame_ids, dtype=np.int64),
            "row_idx": np.arange(len(frame_ids), dtype=np.int64),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{scene}.npy", features)
    manifest.to_parquet(out_dir / f"{scene}.manifest.parquet", index=False)
    return features.shape[0], dim


def main() -> None:
    default_frames = Path("/home/efrei_stage/Desktop/tu-clausthal/datasets")
    default_out = _REPO_ROOT / "features" / "resnet18"
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frames-root",
        default=str(default_frames),
        help="External TU-Clausthal datasets root (contains ETH/ and UCY/). Must not be inside repo.",
    )
    p.add_argument("--out", default=str(default_out), help="Output feature cache dir (repo-root features/resnet18).")
    p.add_argument(
        "--scenes",
        nargs="*",
        default=["eth", "hotel", "univ", "zara1", "zara2"],
    )
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    out_dir = Path(args.out).resolve()
    frames_root = Path(args.frames_root).resolve()
    if not frames_root.is_dir():
        raise FileNotFoundError(
            f"External --frames-root not found: {frames_root}. "
            "Expected TU-Clausthal datasets root (e.g. /home/efrei_stage/Desktop/tu-clausthal/datasets)."
        )
    if _is_under(frames_root, _REPO_ROOT):
        raise ValueError(f"--frames-root must not be inside the repo (_REPO_ROOT={_REPO_ROOT}): {frames_root}")
    # Guard: do not write dataset assets into repo except intended cache output
    if _is_under(out_dir, _REPO_ROOT) and not str(out_dir).endswith("features/resnet18") and not str(out_dir).endswith("features\\resnet18"):
        # allow subdirs under features; be permissive but explicit
        if "features" not in out_dir.parts:
            raise ValueError(f"--out inside repo must target a features cache path, got: {out_dir}")

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
