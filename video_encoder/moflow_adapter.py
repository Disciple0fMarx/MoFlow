"""
MoFlow adapter for the global video encoder (note §5.2).

Exposes per-(scene, frame_id) global video features to MoFlow's training loop
without taking over MoFlow's splits, dataloader, or model definition.

Typical use inside `data/dataloader_eth_ucy.py`:

    from video_encoder.moflow_adapter import FrameFeatureLookup
    lookup = FrameFeatureLookup.from_root("features/resnet18", scene="eth")
    z = lookup.window(start_frame_id, n_frames=8, stride=10)  # [T_obs, D]

If a frame_id is not in the manifest (because the cached encoder used a
different stride), the lookup snaps to the nearest available frame; pass
`policy="floor"` or `policy="drop"` to change that.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

SnapPolicy = Literal["nearest", "floor", "drop"]


@dataclass
class FrameFeatureLookup:
    scene: str
    features: np.ndarray            # [N, D]
    frame_to_row: dict[int, int]    # frame_id -> row index in `features`
    sorted_frame_ids: np.ndarray    # ascending

    @classmethod
    def from_root(cls, root: str | Path, scene: str) -> "FrameFeatureLookup":
        root = Path(root)
        npy = root / f"{scene}.npy"
        manifest = root / f"{scene}.manifest.parquet"
        if not npy.exists() or not manifest.exists():
            raise FileNotFoundError(
                f"Missing cached features for scene={scene!r}: "
                f"expected {npy} and {manifest}. Run `python -m video_encoder encode ...` first."
            )
        feats = np.load(npy)
        man = pd.read_parquet(manifest)
        # Manifest columns: scene, frame_id, path, idx
        fmap = dict(zip(man["frame_id"].astype(int), man["idx"].astype(int)))
        sorted_ids = np.sort(np.fromiter(fmap.keys(), dtype=np.int64))
        return cls(scene=scene, features=feats, frame_to_row=fmap, sorted_frame_ids=sorted_ids)

    # ---- single-frame access ----------------------------------------------
    def __call__(self, frame_id: int, policy: SnapPolicy = "nearest") -> np.ndarray:
        row = self._resolve(int(frame_id), policy)
        if row is None:
            raise KeyError(f"frame_id={frame_id} not available (policy={policy}, scene={self.scene})")
        return self.features[row]

    # ---- windowed access (matches MoFlow's T_obs observation window) ------
    def window(
        self,
        start_frame_id: int,
        n_frames: int,
        stride: int = 1,
        policy: SnapPolicy = "nearest",
    ) -> np.ndarray:
        """Return [n_frames, D] for frames [start, start+stride, ..., start+(n-1)*stride].

        Missing frames are zero-filled when policy=='drop'; otherwise snapped.
        """
        D = self.features.shape[1]
        out = np.zeros((n_frames, D), dtype=self.features.dtype)
        for i in range(n_frames):
            fid = int(start_frame_id) + i * int(stride)
            row = self._resolve(fid, policy)
            if row is not None:
                out[i] = self.features[row]
        return out

    # ---- internal ----------------------------------------------------------
    def _resolve(self, frame_id: int, policy: SnapPolicy) -> int | None:
        if frame_id in self.frame_to_row:
            return self.frame_to_row[frame_id]
        if policy == "drop":
            return None
        ids = self.sorted_frame_ids
        if ids.size == 0:
            return None
        idx = np.searchsorted(ids, frame_id)
        if policy == "floor":
            if idx == 0:
                return None
            return self.frame_to_row[int(ids[idx - 1])]
        # nearest
        candidates = []
        if idx > 0:
            candidates.append(int(ids[idx - 1]))
        if idx < ids.size:
            candidates.append(int(ids[idx]))
        best = min(candidates, key=lambda f: abs(f - frame_id))
        return self.frame_to_row[best]


def feature_dim(root: str | Path, scene: str) -> int:
    """Inspect the cached feature dim without loading the full array."""
    arr = np.load(Path(root) / f"{scene}.npy", mmap_mode="r")
    return int(arr.shape[1])
