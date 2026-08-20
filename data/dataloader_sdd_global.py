"""Memory-efficient SDD dataloader (Leave-One-Scene-Out).

Designed to fix the OOM observed with the previous monolithic
``data/dataloader_sdd.py`` implementation. Three changes matter:

1. **Flat window index** — ``self.windows`` is a single ``np.ndarray`` of
   shape ``[N_windows, 6]`` (``scene_idx, video_idx, track_id, start_frame,
   past_frame_ids[0..7], future_frame_ids[0..11]``). No per-sample tensors.

2. **Lazy video features** — ``__getitem__`` calls
   :func:`video_encoder.sdd_adapter.get_lookup`, which mmap-loads the
   per-scene ``.npy`` and caches one copy per worker. Resident memory is
   bounded by the largest single scene's feature matrix (~5–20 MB), not
   by ``N_windows × T_obs × D``.

3. **Per-scene LOSO** — ``scene_to_use`` picks the held-out test scene;
   train set is the union of the other seven. ``_build_window_index``
   skips loading any scene's annotations into a Python list.

Coordinate handling:
* SDD annotations are in **pixel coordinates** (xmin/xmax/ymin/ymax, frame_id).
* Trajectory point = bbox center = ``((xmin+xmax)/2, (ymin+ymax)/2)``.
* Identity homography (pixels ≈ meters), consistent with the previous
  ``SDD_READY_FOR_TRAINING.md`` notes — flagged with a ``WARNING`` at init
  so the user is aware that predicted coords will be in pixel space.

Hardcoded path: ``~/Desktop/Datasets/SDD`` via :data:`video_encoder.sdd_adapter.DEFAULT_SDD_ROOT`.
Override per-run with ``--sdd_root`` / ``cfg.MODEL.CONTEXT_ENCODER.SDD_ROOT``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

from utils.normalization import normalize_min_max

from video_encoder.sdd_adapter import (
    COL_FRAME,
    COL_TRACK_ID,
    COL_XMAX,
    COL_XMIN,
    COL_YMAX,
    COL_YMIN,
    DEFAULT_SDD_ROOT,
    SDD_SCENES,
    annotation_path,
    expand_sdd_root,
    get_lookup,
)

# ---------------------------------------------------------------------------
# Constants from cfg/eth_ucy/cor_fm.yml (kept hardcoded for safety)
# ---------------------------------------------------------------------------
SDD_PAST_FRAMES = 8
SDD_FUTURE_FRAMES = 12
SDD_SEQ_LEN = SDD_PAST_FRAMES + SDD_FUTURE_FRAMES
SDD_AGENTS_PER_WINDOW = 1   # SDD windows track one pedestrian at a time


# ---------------------------------------------------------------------------
# Window index — single contiguous int32 array, no per-window Python objects
# ---------------------------------------------------------------------------

@dataclass
class SDDWindowIndex:
    """Compact representation of every trajectory window in the dataset.

    ``rows`` is a structured ``np.ndarray`` with fields::

        scene_idx    uint8
        video_idx    uint16
        track_id     int32
        anchor_frame int32     # the future's *last* frame (= past_traj[-1] frame)
        scene        str[16]    # ascii scene name (zero-padded)
        video_id     str[16]    # ascii video id  (zero-padded)

    The corresponding ``past_frame_ids`` and ``future_frame_ids`` for a row
    are ``[anchor - past + 1, ..., anchor]`` and
    ``[anchor + 1, ..., anchor + future]`` respectively — derived lazily in
    :meth:`past_future_frame_ids` because SDD uses a fixed frame-rate.
    """

    rows: np.ndarray
    scenes: list[str]          # index → scene name
    videos: list[str]          # index → video_id, scoped per scene

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def scene_id(self, i: int) -> str:
        return self.scenes[int(self.rows["scene_idx"][i])]

    def video_id(self, i: int) -> str:
        return self.videos[int(self.rows["scene_idx"][i])][int(self.rows["video_idx"][i])]

    def anchor_frame(self, i: int) -> int:
        return int(self.rows["anchor_frame"][i])

    def past_frame_ids(self, i: int) -> np.ndarray:
        a = self.anchor_frame(i)
        return np.arange(a - SDD_PAST_FRAMES + 1, a + 1, dtype=np.int32)

    def future_frame_ids(self, i: int) -> np.ndarray:
        a = self.anchor_frame(i)
        return np.arange(a + 1, a + SDD_FUTURE_FRAMES + 1, dtype=np.int32)


def _load_track_centers(
    sdd_root: Path, scene: str, video_id: str
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Read ``annotations.txt`` and return ``(frame_id → (cx, cy))`` per track.

    Returns
    -------
    track_frames : dict[int, np.ndarray]
        ``track_id → frame_ids`` (sorted ascending).
    track_xy : dict[int, np.ndarray]
        ``track_id → [N, 2]`` float32 (cx, cy).
    """
    txt = annotation_path(sdd_root, scene, video_id)
    if not txt.exists():
        return {}, {}
    arr = np.loadtxt(
        txt,
        usecols=(COL_TRACK_ID, COL_XMIN, COL_YMIN, COL_XMAX, COL_YMAX, COL_FRAME),
    )
    if arr.size == 0:
        return {}, {}
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    cx = (arr[:, 1] + arr[:, 3]) / 2.0
    cy = (arr[:, 2] + arr[:, 4]) / 2.0
    xy = np.column_stack([cx.astype(np.float32), cy.astype(np.float32)])
    frames = arr[:, 5].astype(np.int32)
    tids = arr[:, 0].astype(np.int32)

    track_frames: dict[int, np.ndarray] = {}
    track_xy: dict[int, np.ndarray] = {}
    for tid in np.unique(tids):
        mask = tids == tid
        order = np.argsort(frames[mask])
        track_frames[int(tid)] = frames[mask][order]
        track_xy[int(tid)] = xy[mask][order]
    return track_frames, track_xy


def build_window_index(
    sdd_root: str | Path | None,
    scenes: Sequence[str],
) -> SDDWindowIndex:
    """Walk every (scene, video_id, track_id) and emit sliding-window rows.

    No precomputed pickle; one pass over the raw text annotations.
    """
    root = expand_sdd_root(sdd_root)
    rows_list: list[np.ndarray] = []
    scenes_used: list[str] = []
    videos_per_scene: list[dict[str, int]] = []

    for scene_idx, scene in enumerate(scenes):
        ann_dir = root / "annotations" / scene
        if not ann_dir.is_dir():
            print(f"[sdd-index] skip missing scene dir: {ann_dir}")
            continue
        scenes_used.append(scene)
        video_ids = sorted(p.name for p in ann_dir.iterdir() if p.is_dir())
        videos_per_scene.append({vid: i for i, vid in enumerate(video_ids)})

        for video_id in video_ids:
            track_frames, _ = _load_track_centers(root, scene, video_id)
            video_idx = videos_per_scene[-1][video_id]
            for tid, frames in track_frames.items():
                # Build a sorted unique list of frame ids, then slide a window.
                # SDD annotation frame_ids are sparse; we keep only those that
                # form a contiguous block of length SDD_SEQ_LEN.
                f_unique = np.unique(frames)
                if f_unique.size < SDD_SEQ_LEN:
                    continue
                # Detect contiguous runs of length >= SDD_SEQ_LEN
                diffs = np.diff(f_unique)
                starts = np.concatenate([[0], np.where(diffs > 1)[0] + 1])
                for start in starts:
                    end_idx = start + SDD_SEQ_LEN - 1
                    if end_idx >= f_unique.size:
                        continue
                    # Require the run to actually be contiguous.
                    if diffs[start:end_idx].max() != 1:
                        continue
                    anchor = int(f_unique[end_idx])
                    rows_list.append(
                        (
                            scene_idx,
                            video_idx,
                            int(tid),
                            anchor,
                            scene.encode("ascii", "ignore")[:16].ljust(16),
                            video_id.encode("ascii", "ignore")[:16].ljust(16),
                        )
                    )

    if not rows_list:
        raise RuntimeError(
            f"No SDD trajectory windows found under {root} for scenes={scenes}. "
            f"Verify the dataset layout matches ~/Desktop/Datasets/SDD/{{annotations,videos}}/<scene>/..."
        )

    dt = np.dtype(
        [
            ("scene_idx", np.uint8),
            ("video_idx", np.uint16),
            ("track_id", np.int32),
            ("anchor_frame", np.int32),
            ("scene", "S16"),
            ("video_id", "S16"),
        ]
    )
    arr = np.array(rows_list, dtype=dt)
    return SDDWindowIndex(
        rows=arr,
        scenes=scenes_used,
        videos=[sorted(v.keys()) for v in videos_per_scene],
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SDDGlobalDataset(Dataset):
    """SDD dataset with Leave-One-Scene-Out support and lazy video features."""

    def __init__(
        self,
        cfg,
        training: bool = True,
        sdd_root: str | Path | None = None,
        held_out_scene: str | None = None,
        split: str | None = None,
        use_video: bool = True,
        video_features_root: str | Path | None = None,
        video_stride: int = 1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.training = training
        self.root = expand_sdd_root(sdd_root)

        if held_out_scene is None:
            held_out_scene = getattr(cfg.MODEL.CONTEXT_ENCODER, "HELD_OUT_SCENE", None)
        if held_out_scene is None:
            raise ValueError(
                "SDDGlobalDataset requires --held_out_scene (LOSO) or "
                "cfg.MODEL.CONTEXT_ENCODER.HELD_OUT_SCENE."
            )
        if held_out_scene not in SDD_SCENES:
            raise ValueError(
                f"held_out_scene={held_out_scene!r} not in {SDD_SCENES}"
            )

        # LOSO: train = all scenes except held_out; test = held_out only.
        if split is None:
            split = "train" if training else "test"
        if split == "train":
            scenes = [s for s in SDD_SCENES if s != held_out_scene]
        elif split == "test":
            scenes = [held_out_scene]
        else:
            raise ValueError(f"split must be train|test, got {split!r}")

        print(
            f"[SDDGlobalDataset] {split} split — held_out={held_out_scene} "
            f"→ using {len(scenes)} scene(s): {scenes}"
        )

        self.windows = build_window_index(self.root, scenes)
        print(
            f"[SDDGlobalDataset] indexed {len(self.windows):,} trajectory windows "
            f"across {len(scenes)} scene(s)."
        )

        self.past_frames = SDD_PAST_FRAMES
        self.future_frames = SDD_FUTURE_FRAMES

        # Pre-load center trajectories as one big tensor to avoid per-__getitem__
        # file I/O. Memory: N_windows × (P+F) × 2 × 4 bytes ≈ 4.6 MB for 30k
        # windows — negligible.
        self._xy = self._materialize_xy()

        # Normalization stats: computed once on the union of training windows
        # (the held-out scene is excluded from the stats to avoid leakage).
        if training and (not hasattr(cfg, "past_traj_min") or cfg.past_traj_min is None):
            past_rel = self._xy["past_rel"]    # [N, P, 2]
            fut_rel = self._xy["fut_rel"]
            cfg.past_traj_max = float(past_rel.max())
            cfg.past_traj_min = float(past_rel.min())
            cfg.fut_traj_max = float(fut_rel.max())
            cfg.fut_traj_min = float(fut_rel.min())

        self.past_traj_min = getattr(cfg, "past_traj_min", -1.0)
        self.past_traj_max = getattr(cfg, "past_traj_max", 1.0)
        self.fut_traj_min = getattr(cfg, "fut_traj_min", -1.0)
        self.fut_traj_max = getattr(cfg, "fut_traj_max", 1.0)

        # Build normalized tensor caches.
        self.past_traj_original_scale = self._xy["past_full"]   # [N, 1, P, 6]
        self.fut_traj_original_scale = self._xy["fut_rel"]     # [N, 1, F, 2]
        self.fut_traj_vel = self._xy["fut_vel"]                # [N, 1, F, 2]
        self.past_traj = normalize_min_max(
            self.past_traj_original_scale,
            self.past_traj_min,
            self.past_traj_max,
            -1.0,
            1.0,
        ).contiguous()
        self.fut_traj = normalize_min_max(
            self.fut_traj_original_scale,
            self.fut_traj_min,
            self.fut_traj_max,
            -1.0,
            1.0,
        ).contiguous()

        # ---- Video settings -------------------------------------------------
        self.use_video = bool(use_video)
        self.video_stride = int(video_stride)
        if video_features_root is None:
            video_features_root = getattr(
                cfg.MODEL.CONTEXT_ENCODER, "VIDEO_FEATURES_ROOT", None
            )
        self.video_features_root = (
            Path(video_features_root).expanduser() if video_features_root else None
        )
        if self.use_video and self.video_features_root is None:
            print(
                "[SDDGlobalDataset] WARNING: use_video=True but no video_features_root "
                "configured; video branch will return zero placeholders."
            )

        self.video_dim_raw = int(getattr(cfg.MODEL.CONTEXT_ENCODER, "VIDEO_DIM_RAW", 512))
        cfg.MODEL.CONTEXT_ENCODER.AGENTS = SDD_AGENTS_PER_WINDOW

        # ---- Identity homography warning ------------------------------------
        print(
            "[SDDGlobalDataset] NOTE: using identity homography (pixels ≈ meters); "
            "predictions are in pixel coordinates. See CLAUDE.md coordinate rules."
        )

    # -----------------------------------------------------------------------
    # Materialize xy trajectories into pre-allocated tensors
    # -----------------------------------------------------------------------
    def _materialize_xy(self) -> dict[str, torch.Tensor]:
        N = len(self.windows)
        P, F = SDD_PAST_FRAMES, SDD_FUTURE_FRAMES

        # We allocate once, fill by reading annotations on demand.
        # Per-frame I/O is O(N × (P+F) × small) — acceptable because
        # annotation files are tiny and we cache by (scene, video_id).
        past_abs = np.zeros((N, 1, P, 2), dtype=np.float32)
        past_rel = np.zeros((N, 1, P, 2), dtype=np.float32)
        fut_abs = np.zeros((N, 1, F, 2), dtype=np.float32)
        fut_rel = np.zeros((N, 1, F, 2), dtype=np.float32)

        # Cache (scene, video_id) → {track_id → (frame_ids[N], xy[N,2])}
        cache: dict[tuple[str, str], dict[int, tuple[np.ndarray, np.ndarray]]] = {}

        for i in range(N):
            scene = self.windows.scene_id(i)
            video_id = self.windows.video_id(i)
            anchor = self.windows.anchor_frame(i)
            past_fids = self.windows.past_frame_ids(i)
            fut_fids = self.windows.future_frame_ids(i)
            key = (scene, video_id)
            track_data = cache.get(key)
            if track_data is None:
                track_frames, track_xy = _load_track_centers(self.root, scene, video_id)
                track_data = {tid: (track_frames[tid], track_xy[tid]) for tid in track_xy}
                cache[key] = track_data

            tid = int(self.windows.rows["track_id"][i])
            track_frames, xy = track_data.get(tid, (None, None))
            if xy is None or xy.size == 0:
                continue
            # ``build_window_index`` only keeps *contiguous* runs, so the
            # ordinal index of frame ``anchor`` equals ``anchor - first_frame``.
            first_frame = int(track_frames[0])
            base = anchor - first_frame
            p_slice = xy[base - SDD_PAST_FRAMES + 1 : base + 1]
            f_slice = xy[base + 1 : base + 1 + SDD_FUTURE_FRAMES]
            if p_slice.shape[0] != SDD_PAST_FRAMES or f_slice.shape[0] != SDD_FUTURE_FRAMES:
                continue
            past_abs[i, 0] = p_slice
            init = past_abs[i, 0, -1]
            past_rel[i, 0] = past_abs[i, 0] - init
            fut_abs[i, 0] = f_slice
            fut_rel[i, 0] = fut_abs[i, 0] - init

        # Past velocity: [N, 1, P, 2], last frame = 0
        past_vel = np.concatenate(
            [past_rel[:, :, 1:] - past_rel[:, :, :-1], np.zeros_like(past_rel[:, :, -1:])],
            axis=2,
        )
        # Past full feature: concat [abs | rel | vel] along last axis → [N, 1, P, 6]
        past_full = np.concatenate([past_abs, past_rel, past_vel], axis=-1)
        # Future velocity: [N, 1, F, 2]
        fut_vel = np.concatenate(
            [fut_rel[:, :, 1:] - fut_rel[:, :, :-1], np.zeros_like(fut_rel[:, :, -1:])],
            axis=2,
        )

        return {
            "past_full": torch.from_numpy(past_full),
            "fut_rel": torch.from_numpy(fut_rel),
            "fut_vel": torch.from_numpy(fut_vel),
            "past_rel": torch.from_numpy(past_rel),
        }

    # -----------------------------------------------------------------------
    # Required Dataset API
    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "index": torch.tensor([idx], dtype=torch.int32),
            "past_traj": self.past_traj[idx],                # [1, P, 6] normalized
            "fut_traj": self.fut_traj[idx],                  # [1, F, 2] normalized
            "past_traj_original_scale": self.past_traj_original_scale[idx],
            "fut_traj_original_scale": self.fut_traj_original_scale[idx],
            "fut_traj_vel": self.fut_traj_vel[idx],
            "scene": self.windows.scene_id(idx),
            "video_id": self.windows.video_id(idx),
            "anchor_frame": self.windows.anchor_frame(idx),
        }

        # ---- Lazy global video features ------------------------------------
        if self.use_video and self.video_features_root is not None:
            scene = item["scene"]
            anchor = item["anchor_frame"]
            past_fids = self.windows.past_frame_ids(idx)
            try:
                lookup = get_lookup(self.video_features_root, scene)
                z = lookup.window(
                    start_frame_id=int(past_fids[0]),
                    n_frames=SDD_PAST_FRAMES,
                    stride=self.video_stride,
                    policy="nearest",
                )  # [P, D]
                item["z_video_global"] = torch.from_numpy(z)
            except FileNotFoundError:
                # Missing cache → return a dummy scalar (the collate will skip it).
                item["z_video_global"] = torch.zeros(1)
        else:
            item["z_video_global"] = torch.zeros(1)

        return item


# ---------------------------------------------------------------------------
# (No module-private helpers needed — _load_track_centers + the contiguous-run
# guarantee in build_window_index are sufficient.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Collate function (memory-safe)
# ---------------------------------------------------------------------------

def collate_sdd_global(batch: list[dict]) -> dict:
    """Collate with strict shape checks.

    Detects the "no video" sentinel via ``.dim() == 0`` so we never stack
    a scalar with a 3-D tensor.
    """
    keys_simple = [
        "past_traj",
        "fut_traj",
        "past_traj_original_scale",
        "fut_traj_original_scale",
        "fut_traj_vel",
    ]
    out: dict = {}
    for k in keys_simple:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["index"] = torch.cat([b["index"] for b in batch], dim=0)
    out["batch_size"] = torch.tensor(len(batch))

    # ---- Video features ---------------------------------------------------
    videos = [b["z_video_global"] for b in batch]
    if videos[0].dim() == 3:                                # [P, D] per sample
        out["z_video_global"] = torch.stack(videos, dim=0)  # [B, P, D]
    else:
        out["z_video_global"] = None

    out["scene"] = [b["scene"] for b in batch]
    out["video_id"] = [b["video_id"] for b in batch]
    out["anchor_frame"] = torch.tensor(
        [b["anchor_frame"] for b in batch], dtype=torch.int32
    )
    return out


__all__ = [
    "SDD_PAST_FRAMES",
    "SDD_FUTURE_FRAMES",
    "SDD_SEQ_LEN",
    "SDD_SCENES",
    "SDDWindowIndex",
    "build_window_index",
    "SDDGlobalDataset",
    "collate_sdd_global",
]
