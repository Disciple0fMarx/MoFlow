"""Resolve trajectory frame_ids to actual JPEGs in <scene>/visual_data/."""
from __future__ import annotations

from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Literal
import re

from .scenes import scene_to_folder

_FRAME_RE = re.compile(r"frame(\d+)\.jpg$", re.IGNORECASE)


@lru_cache(maxsize=None)
def list_available_frames(data_root: str, scene: str) -> tuple[int, ...]:
    """Return sorted tuple of integer frame indices available on disk for `scene`."""
    folder = Path(data_root) / scene_to_folder(scene) / "visual_data"
    if not folder.exists():
        raise FileNotFoundError(f"Missing visual_data folder: {folder}")
    out = []
    for p in folder.iterdir():
        m = _FRAME_RE.search(p.name)
        if m:
            out.append(int(m.group(1)))
    if not out:
        raise FileNotFoundError(f"No frame*.jpg files found in {folder}")
    out.sort()
    return tuple(out)


def detect_stride(frames: tuple[int, ...]) -> int:
    """GCD of frame-index deltas — Introvert dumps every 10th frame, so this is usually 10."""
    if len(frames) < 2:
        return 1
    g = 0
    for a, b in zip(frames[:-1], frames[1:]):
        g = gcd(g, b - a)
    return max(g, 1)


def frame_path(data_root: str | Path, scene: str, frame_idx: int) -> Path:
    folder = Path(data_root) / scene_to_folder(scene) / "visual_data"
    # Introvert uses 6-digit zero-padded names. Fall back to a directory scan if needed.
    p = folder / f"frame{frame_idx:06d}.jpg"
    if p.exists():
        return p
    # Tolerate other widths.
    for width in (5, 7, 8):
        alt = folder / f"frame{frame_idx:0{width}d}.jpg"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"No file for {scene} frame {frame_idx} under {folder}")


def snap_frame_id(
    frame_id: int,
    available: tuple[int, ...],
    policy: Literal["nearest", "floor", "drop"] = "nearest",
) -> int | None:
    """Map a trajectory frame_id to a frame_id that exists on disk."""
    if frame_id in available:
        return frame_id
    if policy == "drop":
        return None
    # binary search
    import bisect

    i = bisect.bisect_left(available, frame_id)
    if policy == "floor":
        if i == 0:
            return None
        return available[i - 1]
    # nearest
    candidates = []
    if i > 0:
        candidates.append(available[i - 1])
    if i < len(available):
        candidates.append(available[i])
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs(f - frame_id))


def resolve_frame_paths(
    data_root: str | Path,
    scene: str,
    frame_ids,
    policy: Literal["nearest", "floor", "drop"] = "nearest",
):
    """Return list of (orig_frame_id, snapped_frame_id, Path) skipping unresolvable ones."""
    available = list_available_frames(str(data_root), scene)
    out = []
    for fid in frame_ids:
        snapped = snap_frame_id(int(fid), available, policy=policy)
        if snapped is None:
            continue
        out.append((int(fid), snapped, frame_path(data_root, scene, snapped)))
    return out
