"""Build ETH/UCY ResNet18 global video feature caches (strict external-only).

Invariants:
- Read source videos EXCLUSIVELY from external TU-Clausthal mirror:
  <frames-root>/(ETH|UCY)/<scene_dir>/video.avi (resolved via TU_CLAUSTHAL_MAP).
- Write ONLY precomputed feature caches (<scene>.npy, <scene>.manifest.parquet)
  to the repository output directory (features/resnet18). No dataset assets or
  pre-extracted frame directories are created/read inside _REPO_ROOT.
- 0-based frame indexing for obsmat parity: decoded frame idx -> frame_id = idx
  (covers fid 0).
"""

from __future__ import annotations

import argparse
import json
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
from utils.eth_ucy_geometry import annotation_step
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
    # Never read/create frame directories inside the repo; use native video only.
    if _is_under(scene_path, _REPO_ROOT):
        raise ValueError(f"Resolved scene path is inside repo (forbidden): {scene_path}")
    video_path = scene_path / "video.avi"
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Source video not found (external-only): {video_path}. "
            "build_ethucy_features reads exclusively from external TU-Clausthal video.avi."
        )
    return video_path


def _iter_video_frames(
    video_path: Path,
    stride: int,
    keep_ids: set[int] | None = None,
) -> Iterator[Tuple[int, np.ndarray]]:
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
            selected = fid in keep_ids if keep_ids is not None else (
                stride <= 1 or fid % stride == 0
            )
            if selected:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield fid, frame_rgb
                yielded += 1
            idx += 1
    finally:
        cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames decoded from external video: {video_path}")
    if yielded == 0:
        if keep_ids is not None:
            raise RuntimeError(
                f"None of the {len(keep_ids)} requested frame ids exist in "
                f"external video: {video_path}"
            )
        raise RuntimeError(f"No frames selected at stride={stride} from external video: {video_path}")


@torch.inference_mode()
def build_scene(
    frames_root: Path,
    scene: str,
    out_dir: Path,
    stride: int | None,
    device: str | None,
    batch_size: int,
    frame_ids: list[int] | None = None,
) -> tuple[int, int]:
    out_dir = out_dir.resolve()
    # Cache output is allowed in repo (features/resnet18); ensure we don't write dataset assets elsewhere.
    if not _is_under(out_dir, _REPO_ROOT):
        # allow external out if explicitly requested, but primary invariant is repo features output
        pass

    video_path = _resolve_video_path(frames_root, scene)
    keep_ids = None if frame_ids is None else {int(f) for f in frame_ids}
    step = int(stride) if stride is not None else annotation_step(scene)
    if keep_ids is None and step <= 0:
        raise ValueError(f"stride must be positive for scene={scene}, got {step}")
    model, dim, preprocess = build_backbone("resnet18")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    encoded_ids: list[int] = []
    feats: list[np.ndarray] = []
    buf: list[torch.Tensor] = []
    buf_fids: list[int] = []

    for fid, frame_rgb in tqdm(
        _iter_video_frames(video_path, step, keep_ids=keep_ids),
        desc=f"encode {scene}",
        unit="frame",
    ):
        buf.append(preprocess(Image.fromarray(frame_rgb)))
        buf_fids.append(fid)
        if len(buf) >= batch_size:
            x = torch.stack(buf, dim=0).to(dev, non_blocking=True)
            feats.append(model(x).detach().cpu().numpy().astype(np.float32))
            encoded_ids.extend(buf_fids)
            buf.clear()
            buf_fids.clear()
    if buf:
        x = torch.stack(buf, dim=0).to(dev, non_blocking=True)
        feats.append(model(x).detach().cpu().numpy().astype(np.float32))
        encoded_ids.extend(buf_fids)

    features = np.concatenate(feats, axis=0)
    if features.shape[0] != len(encoded_ids):
        raise RuntimeError(
            f"Feature/manifest size mismatch for {scene}: "
            f"{features.shape[0]} vs {len(encoded_ids)}"
        )
    if keep_ids is not None:
        missing = sorted(keep_ids - set(encoded_ids))
        if missing:
            raise RuntimeError(
                f"Requested frame ids were not decoded for {scene}: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''} "
                f"({len(missing)} missing) in {video_path}"
            )
    manifest = pd.DataFrame(
        {
            "scene": scene,
            "frame_id": np.asarray(encoded_ids, dtype=np.int64),
            "row_idx": np.arange(len(encoded_ids), dtype=np.int64),
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
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Frame stride; default is the benchmark 10-frame grid used by "
            "every TU-Clausthal scene. Ignored when --frame-ids-json supplies "
            "explicit ids for the scene."
        ),
    )
    p.add_argument(
        "--frame-ids-json",
        default=None,
        help=(
            "JSON file mapping scene -> explicit frame ids to encode. Use this "
            "to guarantee exact coverage of the LOSO start+step windows."
        ),
    )
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

    ids_map = None
    if args.frame_ids_json:
        with open(args.frame_ids_json, "r", encoding="utf-8") as fh:
            ids_map = json.load(fh)
        if not isinstance(ids_map, dict):
            raise ValueError("--frame-ids-json must contain a JSON object of scene -> ids")

    for scene in args.scenes:
        ids = None if ids_map is None else ids_map.get(scene)
        if ids is not None:
            ids = [int(x) for x in ids]
        n, dim = build_scene(
            frames_root=frames_root,
            scene=scene,
            out_dir=out_dir,
            stride=args.stride,
            device=args.device,
            batch_size=args.batch_size,
            frame_ids=ids,
        )
        print(f"[ethucy-features] {scene}: {n} frames x {dim} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
