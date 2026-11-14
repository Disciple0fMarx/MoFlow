"""Authoritative ETH/UCY geometry, temporal, and video-index conventions.

This module is the single source of truth for every world/pixel conversion,
annotation stride, and temporal anchor used by the ETH/UCY pipeline.

Coordinate conventions (empirically validated against the TU-Clausthal
``video.avi`` + ``obsmat.txt`` / ``annotation.vsp`` pairs):

* ``H.txt`` maps image pixels -> world meters.  ``inv(H) @ [x, y, 1]`` gives a
  normalized pixel pair ``(a, b)``.
* ``eth`` / ``hotel``: ``pixel = (b, a)`` (row/column order is swapped).
* ``univ`` / ``zara1`` / ``zara2``: VSP/annotation coordinates are centered
  pixels, so ``pixel = (a + W/2, H/2 - b)`` (no swap, vertical flip).

Temporal convention: the benchmark ``.txt`` files and their pre-extracted frame assets
sample every scene at a fixed 10-video-frame grid (25 fps -> 2.5 Hz).  The
native VSP control-point spacing is discarded when those files are resampled,
so the history stride is 10 for all five scenes.  The final observed frame of
an 8-frame history is ``start_frame + (past_frames - 1) * annotation_step``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

CANONICAL_SCENES: tuple[str, ...] = ("eth", "hotel", "univ", "zara1", "zara2")

#: Benchmark frame stride (video frames between consecutive window samples).
#: All TU-Clausthal/SocialGAN benchmark files use the same 10-frame grid, so
#: this must stay scene-uniform even though the native VSP annotations have
#: scene-specific control-point spacing.
ANNOTATION_STEP: dict[str, int] = {
    "eth": 10,
    "hotel": 10,
    "univ": 10,
    "zara1": 10,
    "zara2": 10,
}

#: Native video resolution as (width, height).
SCENE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "eth": (640, 480),
    "hotel": (720, 576),
    "univ": (720, 576),
    "zara1": (720, 576),
    "zara2": (720, 576),
}

_ALIASES: dict[str, str] = {
    "students03": "univ",
    "zara01": "zara1",
    "zara02": "zara2",
}


def canonical_scene(scene: str) -> str:
    """Return the canonical scene name or raise ``ValueError``."""
    key = str(scene).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in CANONICAL_SCENES:
        raise ValueError(
            f"Unknown ETH/UCY scene '{scene}'. Expected one of {CANONICAL_SCENES} "
            f"(aliases: {sorted(_ALIASES)})."
        )
    return key


def annotation_step(scene: str) -> int:
    """Scene-specific annotation stride in video frames."""
    return int(ANNOTATION_STEP[canonical_scene(scene)])


def scene_resolution(scene: str) -> tuple[int, int]:
    """Native ``(width, height)`` for a scene."""
    return SCENE_RESOLUTIONS[canonical_scene(scene)]


def _as_xy(pts: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(-1, 2), arr.shape
    if arr.ndim == 1:
        if arr.shape[0] < 2:
            raise ValueError(f"Expected at least 2 coordinates, got shape {arr.shape}")
        orig = arr.shape
        arr = arr.reshape(1, -1)
    else:
        orig = arr.shape
        if arr.ndim == 2:
            arr = arr
        else:
            arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] < 2:
        raise ValueError(f"Expected last dimension >= 2, got shape {orig}")
    return arr, orig


def world_to_pixel(
    pts_xy,
    H: np.ndarray,
    scene: str,
    orig_size: Sequence[int] | None = None,
    target_res: Sequence[int] | None = None,
) -> np.ndarray:
    """Project world-meter points onto native image pixels for an ETH/UCY scene.

    Parameters
    ----------
    pts_xy:
        ``[..., 2]`` world coordinates in meters (``(x, y)`` as stored in the
        annotation/obsmat files).
    H:
        3x3 homography from ``H.txt`` (image pixels -> world meters).
    scene:
        Canonical scene name (aliases accepted).
    orig_size:
        Native ``(width, height)``; defaults to :data:`SCENE_RESOLUTIONS`.
    target_res:
        Optional ``(width, height)`` of a resized image.  Projection is done in
        native pixels first and then scaled linearly, matching ``imshow``.
    """
    sc = canonical_scene(scene)
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H must be 3x3 for scene={sc}, got shape {H.shape}")

    pts, orig_shape = _as_xy(pts_xy)
    if pts.size == 0:
        return pts.reshape(orig_shape)

    W, Hh = (
        (float(orig_size[0]), float(orig_size[1]))
        if orig_size is not None
        else scene_resolution(sc)
    )

    homog = np.hstack([pts[:, :2], np.ones((len(pts), 1), dtype=np.float64)])
    proj = homog @ np.linalg.inv(H).T
    denom = proj[:, 2]
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError(f"Degenerate homography projection for scene={sc}")
    a = proj[:, 0] / denom
    b = proj[:, 1] / denom

    if sc in ("eth", "hotel"):
        u, v = b, a
    else:  # univ, zara1, zara2
        u = a + W / 2.0
        v = Hh / 2.0 - b

    out = np.column_stack([u, v])

    if target_res is not None:
        tw, th = int(target_res[0]), int(target_res[1])
        if (tw, th) != (int(W), int(Hh)):
            out = out.copy()
            out[:, 0] *= tw / W
            out[:, 1] *= th / Hh

    if len(orig_shape) == 1:
        return out.reshape(-1)
    return out.reshape(orig_shape[:-1] + (2,))


def observed_frame_ids(start_frame: int, scene: str, past_frames: int) -> list[int]:
    """Return the annotation frame ids for an observed history window."""
    step = annotation_step(scene)
    past = int(past_frames)
    if past <= 0:
        raise ValueError(f"past_frames must be positive, got {past}")
    start = int(start_frame)
    return [start + i * step for i in range(past)]


def temporal_anchor_frame(start_frame: int, scene: str, past_frames: int) -> int:
    """Final observed frame of the history window (background/Grad-CAM anchor)."""
    return int(observed_frame_ids(start_frame, scene, past_frames)[-1])


def video_index_for_frame(scene: str, frame_id: int) -> int:
    """Map an annotation frame id to a 0-based ``video.avi`` frame index.

    Validated as identity for all five TU-Clausthal scenes (annotation ids are
    within the native video frame count).
    """
    canonical_scene(scene)
    idx = int(frame_id)
    if idx < 0:
        raise ValueError(f"Frame id must be non-negative, got {idx}")
    return idx


def assert_temporal_alignment(
    start_frame: int,
    scene: str,
    past_frames: int,
    frame_ids: Iterable[int] | None = None,
) -> int:
    """Assert scene stride and history/background alignment; return the anchor.

    Raises ``AssertionError`` if an explicit ``frame_ids`` sequence does not
    match ``start + i * step`` or if the final observed frame is not the
    expected temporal anchor.
    """
    sc = canonical_scene(scene)
    step = annotation_step(sc)
    expected = observed_frame_ids(start_frame, sc, past_frames)
    anchor = expected[-1]
    if frame_ids is not None:
        got = [int(x) for x in frame_ids]
        if got != expected:
            raise AssertionError(
                f"Temporal misalignment for scene={sc}: expected frames {expected}, got {got}"
            )
    if anchor != int(start_frame) + (int(past_frames) - 1) * step:
        raise AssertionError(
            f"Anchor formula mismatch for scene={sc}: anchor={anchor}, "
            f"start={start_frame}, past_frames={past_frames}, step={step}"
        )
    return anchor


def assert_pixel_frame_alignment(
    obs_px: np.ndarray,
    gt_abs_px: np.ndarray,
    pred_abs_px: np.ndarray,
) -> None:
    """Assert prediction and ground-truth overlays share one pixel frame.

    All three arrays are projected with the same homography/scene mapping; this
    guard rejects non-finite coordinates and shape mismatches before plotting.
    """
    for name, arr in (("obs", obs_px), ("gt", gt_abs_px), ("pred", pred_abs_px)):
        a = np.asarray(arr)
        if a.size and not np.isfinite(a).all():
            raise AssertionError(f"Non-finite pixel coordinates in {name} overlay")
    gt = np.asarray(gt_abs_px).reshape(-1, 2)
    pred = np.asarray(pred_abs_px).reshape(-1, 2)
    if gt.shape[-1] != 2 or pred.shape[-1] != 2:
        raise AssertionError(
            f"Overlay coordinate mismatch: gt={gt.shape}, pred={pred.shape} (expected [...,2])"
        )


def scene_video_path(data_root: str | Path, scene: str) -> Path:
    """Resolve ``<tu_clausthal_root>/<group>/<scene>/video.avi`` strictly.

    ``data_root`` may be the TU-Clausthal root, a group directory, or a scene
    directory.  JPEG frame dumps are never considered.
    """
    from data.tu_clausthal_eth_ucy_parser import TU_CLAUSTHAL_MAP

    sc = canonical_scene(scene)
    group, scene_dir = TU_CLAUSTHAL_MAP[sc]
    root = Path(data_root).expanduser()
    candidates = [root / group / scene_dir, root / scene_dir, root]
    for candidate in candidates:
        video = candidate / "video.avi"
        if video.is_file():
            return video
    expected = root / group / scene_dir / "video.avi"
    raise FileNotFoundError(
        f"Native video not found for scene '{sc}'. Looked for: "
        + ", ".join(str(c / "video.avi") for c in candidates)
        + f". Expected TU-Clausthal layout at {expected}."
    )
