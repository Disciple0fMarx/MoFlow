"""Global Video Encoder for SDD — implements Note Technique §5.2.

Pipeline (per the technical note, §3 / §5.2):

    Observed clip V = {I_{t-T_p+1}, ..., I_t}  (T_obs frames)
        |
    CNN backbone (frame-by-frame, no 3D conv)           ← CLAUDE.md: strictly 2D
        |
    Per-frame feature z_t in R^D_raw
        |
    Light temporal Transformer / pooling                ← L_global tokens
        |
    Projection to D = 128                                ← cfg.MODEL.CONTEXT_ENCODER.D_MODEL
        |
    z_video_global in R^{T_obs x D}                     ← matches ETH/UCY pipeline shape

The encoder runs **once at frame-index build time** to populate a per-scene
``.npy`` cache. At training time the dataloader only does O(1) lookups into
this cache (see ``SDDFrameFeatureLookup.window`` in :mod:`sdd_adapter`).

Two public entry points:

* :class:`SDDGlobalVideoEncoder` — streams each SDD video **once**, encodes the
  unique annotated frame ids in batches, and writes per-scene caches via
  :meth:`SDDGlobalVideoEncoder.encode_split_to_cache`.
* :func:`build_sdd_frame_index` — parses ``<sdd_root>/annotations/**`` and
  returns the canonical (scene, video_id, track_id, frame_id, ...) DataFrame.

CLI (see :mod:`video_encoder.cli`)::

    python -m video_encoder encode-sdd \
        --sdd-root /home/efrei_stage/Desktop/Datasets/SDD \
        --out ./features/resnet18
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

from .sdd_adapter import (
    SDD_SCENES,
    annotation_path,
    COL_FRAME,
    COL_TRACK_ID,
    COL_XMAX,
    COL_XMIN,
    COL_YMAX,
    COL_YMIN,
    DEFAULT_SDD_ROOT,
    expand_sdd_root,
    video_mov_path,
)


# ---------------------------------------------------------------------------
# Frame index
# ---------------------------------------------------------------------------

def build_sdd_frame_index(
    sdd_root: str | Path | None = None,
    scenes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Parse every ``annotations.txt`` under ``~/Desktop/Datasets/SDD`` and return a
    flat DataFrame with one row per (scene, video_id, track_id, frame_id).

    Columns: ``scene, video_id, track_id, frame_id, xmin, ymin, xmax, ymax``.

    This function is the **only** place that touches ``annotations.txt`` —
    it runs once at index-build time, never during training.
    """
    root = expand_sdd_root(sdd_root)
    scenes = list(scenes) if scenes is not None else list(SDD_SCENES)

    rows: list[dict] = []
    for scene in scenes:
        ann_dir = root / "annotations" / scene
        if not ann_dir.is_dir():
            print(f"[sdd_index] skip missing scene dir: {ann_dir}")
            continue
        for video_id in sorted(p for p in ann_dir.iterdir() if p.is_dir()):
            txt = annotation_path(root, scene, video_id.name)
            if not txt.exists():
                continue
            try:
                # Use only the columns we need: track_id, xmin, ymin, xmax, ymax, frame.
                arr = np.loadtxt(
                    txt,
                    usecols=(COL_TRACK_ID, COL_XMIN, COL_YMIN, COL_XMAX, COL_YMAX, COL_FRAME),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[sdd_index] failed to parse {txt}: {exc}")
                continue
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for row in arr:
                rows.append(
                    {
                        "scene": scene,
                        "video_id": video_id.name,
                        "track_id": int(row[0]),
                        "frame_id": int(row[5]),
                        "xmin": float(row[1]),
                        "ymin": float(row[2]),
                        "xmax": float(row[3]),
                        "ymax": float(row[4]),
                    }
                )
    df = pd.DataFrame(rows)
    df.sort_values(["scene", "video_id", "track_id", "frame_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Backbone: 2D ResNet-18 (CLAUDE.md non-negotiable)
# ---------------------------------------------------------------------------

def _build_resnet18(device: str | torch.device) -> tuple[torch.nn.Module, int, transforms.Compose]:
    """Return ``(model, feature_dim, preprocess)``.

    Delegates to :func:`backbones.build_backbone` so SDD features are computed
    with the exact same frozen ResNet-18 + ImageNet preprocessing as the
    ETH/UCY pipeline. Strictly 2D backbone (no temporal conv).
    """
    from .backbones import build_backbone

    model, dim, preprocess = build_backbone("resnet18")
    return model.to(device), dim, preprocess


# ---------------------------------------------------------------------------
# Streaming video reader (sequential decode, no full-video RAM load)
# ---------------------------------------------------------------------------

def _open_video_reader(video_path: Path):
    """Yield ``(frame_idx, PIL RGB Image)`` for every frame, sequentially.

    Tries ``decord`` first (mirrors :mod:`video_encoder.custom_video`), then
    falls back to OpenCV. Only one decoded frame is held at a time — unlike
    ``torchvision.io.read_video`` which materializes the entire clip.
    """
    try:  # preferred backend
        import decord  # type: ignore

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(video_path))
        n = len(vr)
    except Exception:
        vr, n = None, None
    if vr is not None:
        for i in range(n):
            yield i, Image.fromarray(vr[i].asnumpy())
        return

    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield idx, Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Frame dataset: reads JPEGs from <sdd_root>/videos/<scene>/<video_id>/frames/...
# ---------------------------------------------------------------------------

class SDDFrameDataset(Dataset):
    """Yield individual frames extracted from SDD ``video.mov`` files.

    For memory efficiency this class assumes the user has already extracted
    individual JPEGs to::

        <sdd_root>/videos/<scene>/<video_id>/frames/frame000001.jpg

    Use ``scripts/extract_sdd_frames.py`` to materialize them once.
    Each ``__getitem__`` returns a preprocessed tensor ready for ResNet-18.
    """

    def __init__(
        self,
        frames_df: pd.DataFrame,
        sdd_root: Path,
        preprocess: transforms.Compose,
    ) -> None:
        self.df = frames_df.reset_index(drop=True)
        self.sdd_root = Path(sdd_root)
        self.preprocess = preprocess
        # Required columns: scene, video_id, frame_id
        assert {"scene", "video_id", "frame_id"}.issubset(self.df.columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        scene = row["scene"]
        video_id = row["video_id"]
        frame_id = int(row["frame_id"])
        path = self.sdd_root / "videos" / scene / video_id / "frames" / f"frame{frame_id:06d}.jpg"
        if not path.exists():
            # Fallback: load the .mov and decode the specific frame. Slow path.
            from torchvision.io import read_video

            mov = video_mov_path(self.sdd_root, scene, video_id)
            reader = read_video(str(mov), pts_unit="sec")
            # `reader` is a (video, audio, info) namedtuple. We pick frames by index.
            # SDD .mov FPS varies (~29.97); we assume frame_id matches the file order.
            video_tensor = reader[0]  # [T, H, W, C] uint8
            if frame_id < video_tensor.shape[0]:
                img_tensor = video_tensor[frame_id]
            else:
                img_tensor = video_tensor[-1]
            from PIL import Image

            img = Image.fromarray(img_tensor.numpy())
        else:
            from PIL import Image

            img = Image.open(path).convert("RGB")
        return {
            "image": self.preprocess(img),
            "scene": scene,
            "video_id": video_id,
            "frame_id": frame_id,
        }


def _collate_frames(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    return {
        "image": images,
        "scene": [b["scene"] for b in batch],
        "video_id": [b["video_id"] for b in batch],
        "frame_id": [b["frame_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

@dataclass
class SDDGlobalVideoEncoder:
    """Encode unique (scene, frame_id) pairs into per-frame feature vectors."""

    backbone: str = "resnet18"
    device: str | None = None
    batch_size: int = 32
    num_workers: int = 2

    def __post_init__(self) -> None:
        if self.backbone != "resnet18":
            raise ValueError("Only resnet18 supported in v1 (CLAUDE.md: 2D backbone).")
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.feature_dim, self.preprocess = _build_resnet18(self.device)

    @torch.inference_mode()
    def encode(
        self,
        frames_df: pd.DataFrame,
        sdd_root: str | Path | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Encode every row of ``frames_df`` and return (features [N,D], manifest_df)."""
        root = expand_sdd_root(sdd_root)
        ds = SDDFrameDataset(frames_df, root, self.preprocess)
        if len(ds) == 0:
            return (
                np.zeros((0, self.feature_dim), dtype=np.float32),
                pd.DataFrame(columns=["scene", "video_id", "frame_id", "row_idx"]),
            )
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_frames,
            pin_memory=(self.device == "cuda"),
        )
        feats: list[np.ndarray] = []
        manifest_rows: list[dict] = []
        row_offset = 0
        for batch in tqdm(loader, desc=f"sdd-encode[{self.backbone}]"):
            x = batch["image"].to(self.device, non_blocking=True)
            y = self.model(x).detach().cpu().numpy().astype(np.float32)
            feats.append(y)
            for scene, video_id, frame_id in zip(
                batch["scene"], batch["video_id"], batch["frame_id"]
            ):
                manifest_rows.append(
                    {
                        "scene": scene,
                        "video_id": video_id,
                        "frame_id": int(frame_id),
                        "row_idx": row_offset,
                    }
                )
                row_offset += 1
        features = np.concatenate(feats, axis=0) if feats else np.zeros((0, self.feature_dim), dtype=np.float32)
        manifest = pd.DataFrame(manifest_rows)
        return features, manifest

    @torch.inference_mode()
    def _encode_frames_stream(
        self,
        needed_by_video: dict[str, list[int]],
        root: Path,
        scene: str,
    ) -> dict[int, np.ndarray]:
        """Stream each SDD video once and encode only its annotated frame ids.

        Returns ``{frame_id: feature_vec[D]}``. Frame ids requested beyond the
        decoded length are reported and omitted (the lookup snaps to nearest).
        """
        feats_by_fid: dict[int, np.ndarray] = {}
        for video_id in sorted(needed_by_video):
            wanted = set(int(f) for f in needed_by_video[video_id])
            vid_path = video_mov_path(root, scene, video_id)
            if not vid_path.exists():
                print(f"[sdd-encode] {scene}/{video_id}: missing video {vid_path}, skipping")
                continue

            buf_imgs: list[torch.Tensor] = []
            buf_fids: list[int] = []

            def flush() -> None:
                if not buf_imgs:
                    return
                x = torch.stack(buf_imgs, dim=0).to(self.device, non_blocking=True)
                y = self.model(x).detach().cpu().numpy().astype(np.float32)
                for fid, vec in zip(buf_fids, y):
                    feats_by_fid[fid] = vec
                buf_imgs.clear()
                buf_fids.clear()

            n_decoded = 0
            for idx, img in _open_video_reader(vid_path):
                n_decoded = idx + 1
                if idx in wanted:
                    buf_imgs.append(self.preprocess(img))
                    buf_fids.append(idx)
                    if len(buf_imgs) >= self.batch_size:
                        flush()
            flush()

            missing = wanted - feats_by_fid.keys()
            if missing:
                print(
                    f"[sdd-encode] {scene}/{video_id}: {len(missing)} annotated "
                    f"frame(s) beyond decoded length {n_decoded} "
                    f"(e.g. {min(missing)}); they will nearest-snap at lookup time."
                )
        return feats_by_fid

    def encode_split_to_cache(
        self,
        sdd_root: str | Path | None,
        scenes: Iterable[str] | None,
        out_dir: str | Path = "./features/resnet18",
    ) -> dict[str, tuple[Path, Path]]:
        """For each scene, encode every *unique* annotated frame and write cache.

        Output layout (consumed by ``SDDFrameFeatureLookup``)::

            <out_dir>/<scene>.npy               # [N_frames, D] float32
            <out_dir>/<scene>.manifest.parquet  # scene, video_id, frame_id, row_idx

        ``out_dir`` defaults to the repo-local ``./features/resnet18`` because
        the production SDD dataset directory is read-only.

        Returns ``{scene: (npy_path, parquet_path)}``.
        """
        root = expand_sdd_root(sdd_root)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scenes = list(scenes) if scenes is not None else list(SDD_SCENES)

        written: dict[str, tuple[Path, Path]] = {}
        for scene in scenes:
            ann_dir = root / "annotations" / scene
            if not ann_dir.is_dir():
                print(f"[sdd-encode] skip {scene}: no annotations dir {ann_dir}")
                continue
            # Unique frames across all (video_id, track_id) for this scene.
            df = build_sdd_frame_index(sdd_root=root, scenes=[scene])
            unique = (
                df[["scene", "video_id", "frame_id"]]
                .drop_duplicates()
                .sort_values(["video_id", "frame_id"])
                .drop_duplicates("frame_id")  # a frame shared by 2 videos → first video wins
                .reset_index(drop=True)
            )
            if unique.empty:
                print(f"[sdd-encode] {scene}: no frames to encode, skipping")
                continue

            needed: dict[str, list[int]] = {
                vid: sorted(g["frame_id"].astype(int))
                for vid, g in unique.groupby("video_id", sort=False)
            }
            vid_of = dict(zip(unique["frame_id"].astype(int), unique["video_id"]))

            feats_by_fid = self._encode_frames_stream(needed, root, scene)
            if not feats_by_fid:
                print(f"[sdd-encode] {scene}: nothing decodable, skipping")
                continue

            fid_sorted = sorted(feats_by_fid)
            features = np.stack([feats_by_fid[f] for f in fid_sorted]).astype(np.float32)
            manifest = pd.DataFrame(
                {
                    "scene": scene,
                    "video_id": [vid_of[f] for f in fid_sorted],
                    "frame_id": np.asarray(fid_sorted, dtype=np.int64),
                    "row_idx": np.arange(len(fid_sorted), dtype=np.int64),
                }
            )

            npy_path = out_dir / f"{scene}.npy"
            parquet_path = out_dir / f"{scene}.manifest.parquet"
            np.save(npy_path, features)
            manifest.to_parquet(parquet_path, index=False)
            written[scene] = (npy_path, parquet_path)
            print(
                f"[sdd-encode] {scene}: {features.shape} from "
                f"{len(needed)} video(s) -> {npy_path}"
            )
        return written


__all__ = [
    "SDD_SCENES",
    "DEFAULT_SDD_ROOT",
    "expand_sdd_root",
    "build_sdd_frame_index",
    "SDDFrameDataset",
    "SDDGlobalVideoEncoder",
]
