"""SDD-specific adapter for the global video encoder (note §5.2).

Stanford Drone Dataset lives on the **remote lab machine** at ``~/Datasets/SDD``
and uses a layout completely different from ETH/UCY:

    ~/Datasets/SDD/
        annotations/<scene>/<videoX>/annotations.txt   # TrackID,xmin,ymin,xmax,ymax,frame,lost,occ,gen,label
        videos/<scene>/<videoX>/video.mov

This module exposes the same ``FrameFeatureLookup``-style API that the
MoFlow data layer expects, but resolves frame paths against this layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Eight canonical SDD scenes. Order matches the LOSO protocol described in
# SDD_READY_FOR_TRAINING.md (which we deliberately did NOT delete).
SDD_SCENES: tuple[str, ...] = (
    "bookstore",
    "coupa",
    "deathCircle",
    "gates",
    "hyang",
    "little",
    "nexus",
    "quad",
)

# Default remote path. Override via the ``--sdd_root`` CLI flag or the
# ``cfg.MODEL.CONTEXT_ENCODER.SDD_ROOT`` YAML key.
DEFAULT_SDD_ROOT = Path("~/Datasets/SDD").expanduser()

SnapPolicy = Literal["nearest", "floor", "drop"]


def expand_sdd_root(maybe_root: str | Path | None) -> Path:
    """Resolve the SDD root directory, defaulting to ``~/Datasets/SDD``.

    The constant is intentionally *hardcoded* — local repositories must NOT
    ship a copy of SDD. Anything that bypasses this helper is a bug.
    """
    if maybe_root is None:
        return DEFAULT_SDD_ROOT
    p = Path(maybe_root).expanduser()
    return p


def scene_annotations_dir(sdd_root: Path, scene: str) -> Path:
    return sdd_root / "annotations" / scene


def scene_videos_dir(sdd_root: Path, scene: str) -> Path:
    return sdd_root / "videos" / scene


def video_mov_path(sdd_root: Path, scene: str, video_id: str) -> Path:
    """Return the canonical ``video.mov`` path for a given (scene, video_id)."""
    return scene_videos_dir(sdd_root, scene) / video_id / "video.mov"


def annotation_path(sdd_root: Path, scene: str, video_id: str) -> Path:
    return scene_annotations_dir(sdd_root, scene) / video_id / "annotations.txt"


# Annotation column indices (whitespace-separated).
COL_TRACK_ID = 0
COL_XMIN = 1
COL_YMIN = 2
COL_XMAX = 3
COL_YMAX = 4
COL_FRAME = 5


@dataclass
class SDDFrameFeatureLookup:
    """Per-scene cached feature lookup, identical semantics to the ETH/UCY
    ``FrameFeatureLookup`` but for SDD scene names.

    The cache layout is::

        <features_root>/<scene>.npy              # [N_frames, D] float32
        <features_root>/<scene>.manifest.parquet  # columns: frame_id, row_idx, path
    """

    scene: str
    features: np.ndarray            # [N, D]
    frame_to_row: dict[int, int]    # frame_id -> row index
    sorted_frame_ids: np.ndarray    # ascending

    @classmethod
    def from_root(
        cls, root: str | Path, scene: str, mmap: bool = True
    ) -> "SDDFrameFeatureLookup":
        root = Path(root)
        npy = root / f"{scene}.npy"
        manifest = root / f"{scene}.manifest.parquet"
        if not npy.exists() or not manifest.exists():
            raise FileNotFoundError(
                f"Missing cached features for SDD scene={scene!r}: "
                f"expected {npy} and {manifest}. Run frame encoding first."
            )
        feats = np.load(npy, mmap_mode="r" if mmap else None)
        man = pd.read_parquet(manifest)
        fmap = dict(zip(man["frame_id"].astype(int), man["row_idx"].astype(int)))
        sorted_ids = np.sort(np.fromiter(fmap.keys(), dtype=np.int64))
        return cls(scene=scene, features=feats, frame_to_row=fmap, sorted_frame_ids=sorted_ids)

    # ---- single-frame access ------------------------------------------------
    def get(self, frame_id: int, policy: SnapPolicy = "nearest") -> np.ndarray | None:
        row = self._resolve(int(frame_id), policy)
        if row is None:
            return None
        return np.asarray(self.features[row])

    # ---- windowed access ----------------------------------------------------
    def window(
        self,
        start_frame_id: int,
        n_frames: int,
        stride: int = 1,
        policy: SnapPolicy = "nearest",
    ) -> np.ndarray:
        """Return ``[n_frames, D]`` for ``[start, start+stride, ..., start+(n-1)*stride]``.

        Unresolvable frames are zero-filled (matches the ETH/UCY helper).
        """
        D = int(self.features.shape[1])
        out = np.zeros((n_frames, D), dtype=np.float32)
        for i in range(n_frames):
            fid = int(start_frame_id) + i * int(stride)
            row = self._resolve(fid, policy)
            if row is not None:
                out[i] = np.asarray(self.features[row])
        return out

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    # ---- internal -----------------------------------------------------------
    def _resolve(self, frame_id: int, policy: SnapPolicy) -> int | None:
        if frame_id in self.frame_to_row:
            return self.frame_to_row[frame_id]
        if policy == "drop":
            return None
        ids = self.sorted_frame_ids
        if ids.size == 0:
            return None
        idx = int(np.searchsorted(ids, frame_id))
        if policy == "floor":
            if idx == 0:
                return None
            return self.frame_to_row[int(ids[idx - 1])]
        # nearest
        candidates: list[int] = []
        if idx > 0:
            candidates.append(int(ids[idx - 1]))
        if idx < ids.size:
            candidates.append(int(ids[idx]))
        if not candidates:
            return None
        best = min(candidates, key=lambda f: abs(f - frame_id))
        return self.frame_to_row[best]


@lru_cache(maxsize=64)
def _cached_lookup(features_root: str, scene: str) -> SDDFrameFeatureLookup:
    """Memoize feature lookups across ``__getitem__`` calls.

    The cache lives for the lifetime of the DataLoader worker process.
    Memory cost: one mmap'd ``.npy`` per scene (~5–20 MB).
    """
    return SDDFrameFeatureLookup.from_root(features_root, scene)


def get_lookup(features_root: str | Path, scene: str) -> SDDFrameFeatureLookup:
    """Public, lru_cache-wrapped accessor (one mmap per process per scene)."""
    return _cached_lookup(str(Path(features_root)), scene)
