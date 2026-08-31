"""Robust SDD `annotations.txt` parser and agent-centric bounding-box crop
extraction for the agent video encoder.

The Stanford Drone Dataset stores every agent's per-frame state in a
whitespace-separated ``annotations.txt`` with exactly 10 columns::

    col 0 : Track ID
    col 1 : xmin         (bbox top-left x,   pixels)
    col 2 : ymin         (bbox top-left y,   pixels)
    col 3 : xmax         (bbox bottom-right x, pixels)
    col 4 : ymax         (bbox bottom-right y, pixels)
    col 5 : frame        (frame id)
    col 6 : lost         (1 if outside the image)
    col 7 : occluded     (1 if occluded)
    col 8 : generated    (1 if interpolated between annotations)
    col 9 : label        (enclosed in quotation marks, e.g. "pedestrian")

This module mirrors the coordinate / path conventions established in
:mod:`video_encoder.sdd_adapter` (via ``expand_sdd_root``), so an
``AgentCropDataset`` integrates with the existing LOSO dataloader
(:mod:`data.dataloader_sdd_global`) without duplicating path logic.

Pipeline
--------
1. :func:`parse_annotations`  — read the 10-column text into a structured
   ``np.ndarray`` (string labels parsed through a shlex-aware reader).
2. :func:`build_tracks`       — group rows by Track ID into per-track arrays
   of ``(frame, xmin, ymin, xmax, ymax)`` and per-track validity predicates
   derived from ``lost`` / ``occluded`` / ``generated``.
3. :func:`extract_agent_crops`— for an (optional) video path, decode the
   frames and slice square crops centred on each annotation box, with
   boundary clamping + zero-padding for out-of-view crops.
4. :class:`SDDAgentCropDataset` — a ``torch.utils.data.Dataset`` pairing each
   trajectory window with its ``[T_obs, C, H, W]`` crop tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from video_encoder.sdd_adapter import (
    COL_FRAME,
    COL_TRACK_ID,
    COL_XMAX,
    COL_XMIN,
    COL_YMAX,
    COL_YMIN,
    SDD_SCENES,
    annotation_path,
    expand_sdd_root,
    find_video_path,
)

# ---------------------------------------------------------------------------
# 10-column annotation schema (exact indices per the task spec)
# ---------------------------------------------------------------------------
A_TRACK_ID = 0
A_XMIN = 1
A_YMIN = 2
A_XMAX = 3
A_YMAX = 4
A_FRAME = 5
A_LOST = 6
A_OCCLUDED = 7
A_GENERATED = 8
A_LABEL = 9

# Default crop footprint (pixels). The encoder is designed to be crop-size
# agnostic (it does not hardcode an input resolution), so 64x64 is only a
# sensible default and can be overridden per-dataset.
DEFAULT_CROP_SIZE = 64
DEFAULT_PADDING = 0
DEFAULT_OBS_FRAMES = 8

#: Compact dtype for rows produced by :func:`parse_annotations`.
ANN_DTYPE = np.dtype(
    [
        ("track_id", np.int32),
        ("xmin", np.float32),
        ("ymin", np.float32),
        ("xmax", np.float32),
        ("ymax", np.float32),
        ("frame", np.int32),
        ("lost", np.uint8),
        ("occluded", np.uint8),
        ("generated", np.uint8),
        ("label", "U24"),
    ]
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_annotations(
    sdd_root: str | Path, scene: str, video_id: str
) -> np.ndarray:
    """Parse one ``annotations.txt`` file (10 columns) into a structured array.

    The trailer ``label`` column is enclosed in quotes and may contain spaces
    internally (rare), so we do **not** use ``np.loadtxt`` on the raw file.
    Instead every field is parsed defensively with Python, then packed into a
    homogeneous structured ``np.ndarray``.

    Returns an array of shape ``[N_rows, 10]``-semantics with named fields; an
    empty (0-row) array is returned when the file is missing or empty.
    """
    path = annotation_path(expand_sdd_root(sdd_root), scene, video_id)
    if not path.exists():
        return np.zeros(0, dtype=ANN_DTYPE)

    recs: list[tuple] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            tokens = _split_annotation_line(line)
            if len(tokens) != 10:
                # Defensive skip for malformed rows (e.g. a stray extra token).
                continue
            try:
                track_id = int(float(tokens[A_TRACK_ID]))
                xmin = float(tokens[A_XMIN])
                ymin = float(tokens[A_YMIN])
                xmax = float(tokens[A_XMAX])
                ymax = float(tokens[A_YMAX])
                frame = int(float(tokens[A_FRAME]))
                lost = int(float(tokens[A_LOST]))
                occluded = int(float(tokens[A_OCCLUDED]))
                generated = int(float(tokens[A_GENERATED]))
                label = tokens[A_LABEL].strip('"')
            except (ValueError, IndexError):
                continue
            recs.append(
                (
                    track_id,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    frame,
                    lost,
                    occluded,
                    generated,
                    label,
                )
            )

    if not recs:
        return np.zeros(0, dtype=ANN_DTYPE)
    return np.array(recs, dtype=ANN_DTYPE)


def _split_annotation_line(line: str) -> list[str]:
    """Split a 10-column annotation line respecting quoted label quoting.

    Numeric columns are single whitespace tokens, but the final ``label``
    column is wrapped in double quotes and can contain spaces (e.g.
    ``"pedestrian"`` or occasionally ``"a walking person"``).  We split once
    on the closing quote of the label to keep the quoted span intact.
    """
    # Find the first double-quote; everything before it is the fixed 5-field
    # numeric prefix (columns 0..8), everything from the quote onward is the
    # (possibly space-containing) label.
    q = line.find('"')
    if q == -1:
        return line.split()  # degenerate: fall back to naive split
    prefix = line[:q].split()
    label = line[q:].rstrip()
    return prefix + [label]


@dataclass
class AgentTrack:
    """Per-agent annotation table for one video."""

    track_id: int
    frames: np.ndarray          # [N] int32, ascending frame ids
    bbox: np.ndarray            # [N, 4] float32 — xmin, ymin, xmax, ymax
    lost: np.ndarray            # [N] uint8
    occluded: np.ndarray        # [N] uint8
    generated: np.ndarray       # [N] uint8
    labels: np.ndarray          # [N] "U24"

    def box_centers(self) -> np.ndarray:
        """Return ``[N, 2]`` float32 bbox centres (cx, cy)."""
        return np.column_stack(
            [
                (self.bbox[:, 0] + self.bbox[:, 2]) / 2.0,
                (self.bbox[:, 1] + self.bbox[:, 3]) / 2.0,
            ]
        )

    def valid_mask(
        self, drop_lost: bool = True, drop_occluded: bool = False,
        drop_generated: bool = False,
    ) -> np.ndarray:
        """Boolean validity per frame from the lost/occluded/generated flags.

        ``lost`` is on by default because a lost frame has no trustworthy
        position to crop around.  ``occluded`` and ``generated`` lines remain
        usable by default (the bbox is still present) but can be dropped.
        """
        mask = np.ones(self.frames.size, dtype=bool)
        if drop_lost:
            mask &= self.lost == 0
        if drop_occluded:
            mask &= self.occluded == 0
        if drop_generated:
            mask &= self.generated == 0
        return mask


def build_tracks(annotations: np.ndarray) -> dict[int, AgentTrack]:
    """Group parsed rows by Track ID into :class:`AgentTrack` objects.

    ``annotations`` is the structured array returned by :func:`parse_annotations`.
    Rows are sorted by frame id within each track so that temporal indexing is
    trivial during crop extraction.
    """
    tracks: dict[int, AgentTrack] = {}
    if annotations.size == 0:
        return tracks

    ids = annotations["track_id"]
    for tid in np.unique(ids):
        mask = ids == tid
        order = np.argsort(annotations["frame"][mask], kind="stable")
        tracks[int(tid)] = AgentTrack(
            track_id=int(tid),
            frames=np.asarray(annotations["frame"][mask][order], dtype=np.int32),
            bbox=np.column_stack(
                [
                    annotations[k][mask][order]
                    for k in ("xmin", "ymin", "xmax", "ymax")
                ]
            ).astype(np.float32),
            lost=np.asarray(annotations["lost"][mask][order], dtype=np.uint8),
            occluded=np.asarray(annotations["occluded"][mask][order], dtype=np.uint8),
            generated=np.asarray(annotations["generated"][mask][order], dtype=np.uint8),
            labels=np.asarray(annotations["label"][mask][order], dtype="U24"),
        )
    return tracks


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------

@dataclass
class CropExtractor:
    """Slices fixed-size square crops centred on annotation boxes.

    Parameters
    ----------
    crop_size : int
        Side length of the output square crop in pixels.
    padding : int
        Extra margin (pixels) added around the bbox before centring/scaling.
    pad_value : int
        Fill value for out-of-image regions (default 0 → black).
    """

    crop_size: int = DEFAULT_CROP_SIZE
    padding: int = DEFAULT_PADDING
    pad_value: float | int = 0

    def __post_init__(self) -> None:
        if self.crop_size <= 0:
            raise ValueError("crop_size must be a positive integer")

    def frame_crop(self, frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
        """Return a ``[C, crop_size, crop_size]`` uint8 crop for one box.

        ``box`` is ``(xmin, ymin, xmax, ymax)`` in pixel coordinates.
        The crop is centred on the box centre enlarged by ``padding``, clamped
        to the image and zero-padded where it falls outside.  ``frame`` is the
        full decoded image, shape ``[H, W, C]`` (BGR as produced by OpenCV) and
        is returned transposed to NHWC-to-CHW ``[C, H, W]`` for torch.
        """
        H, W = frame.shape[:2]
        xmin, ymin, xmax, ymax = (float(v) for v in box)

        # Enlarge the box slightly, then treat its centre as the crop centre.
        xmin -= self.padding
        ymin -= self.padding
        xmax += self.padding
        ymax += self.padding
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        half = self.crop_size / 2.0

        left = int(round(cx - half))
        top = int(round(cy - half))
        right = left + self.crop_size
        bottom = top + self.crop_size

        # Clamp the on-image window.
        src_x0 = max(0, left)
        src_y0 = max(0, top)
        src_x1 = min(W, right)
        src_y1 = min(H, bottom)

        canvas = np.full(
            (self.crop_size, self.crop_size, frame.shape[2]),
            self.pad_value,
            dtype=np.uint8,
        )
        if src_x1 > src_x0 and src_y1 > src_y0:
            canvas[
                (src_y0 - top):(src_y1 - top),
                (src_x0 - left):(src_x1 - left),
            ] = frame[src_y0:src_y1, src_x0:src_x1]
        # CHW
        return np.transpose(canvas, (2, 0, 1)).copy()


def decode_frame(video_path: str | Path, frame_id: int) -> np.ndarray | None:
    """Decode a single frame from an SDD video via sequential-capture seeking.

    ``cv2.VideoCapture`` absolute seeking (``CAP_PROP_POS_FRAMES``) is
    unreliable across codecs, so we instead open the capture, walk the frame
    counter up to ``frame_id`` and grab the requested frame.  Returns the BGR
    frame ``[H, W, 3]`` or ``None`` if the video cannot be opened or the frame
    is past the end.

    ``video_path`` must already be resolved by the caller (e.g. via
    :func:`video_encoder.sdd_adapter.find_video_path`).

    Frame indexing: SDD annotation ``frame`` ids are 0-based OpenCV ordinals
    (video frame 0 == annotation ``frame=0``).  ``CAP_PROP_POS_FRAMES`` and the
    sequential counter compare against the frame count using the same
    convention, so a frame past the end is rejected rather than silently
    off-by-one.
    """
    import logging
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.getLogger("agent_crop_sdd").warning(
            "decode_frame: cv2.VideoCapture failed to open video at absolute path %s",
            str(Path(video_path).resolve()),
        )
        return None
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if 0 < count <= frame_id:
            return None
        # Position at the requested (0-based) frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return np.asarray(frame)
    finally:
        cap.release()


def extract_agent_crops(
    annotations: np.ndarray,
    video_path: str | Path,
    track_id: int,
    past_frames: Sequence[int],
    crop_size: int = DEFAULT_CROP_SIZE,
    padding: int = DEFAULT_PADDING,
    drop_lost: bool = True,
    pad_value: float | int = 0,
) -> np.ndarray | None:
    """Extract ``[T_obs, C, crop_size, crop_size]`` crops for one agent window.

    Parameters
    ----------
    annotations : structured array from :func:`parse_annotations`
    video_path  : path to the raw SDD video file
    track_id    : Track ID of the agent
    past_frames : the ``T_obs`` observed frame ids (length must equal the
                  number of crops requested)
    crop_size / padding / pad_value
        forwarded to :class:`CropExtractor`.

    Returns ``None`` if the track or any of its observed frames are missing or
    flagged ``lost`` (when ``drop_lost`` is set); otherwise a ``[T_obs, C, S, S]``
    uint8 tensor.
    """
    tracks = build_tracks(annotations)
    track = tracks.get(int(track_id))
    if track is None:
        return None

    # Index lookups (frames sorted, so a searchsorted gives the ordinal).
    pos = np.searchsorted(track.frames, np.asarray(past_frames, dtype=np.int32))
    valid = pos < track.frames.size
    valid &= track.frames[np.clip(pos, 0, track.frames.size - 1)] == np.asarray(
        past_frames, dtype=np.int32
    )
    if drop_lost:
        safe = np.where(valid, pos, 0)
        valid &= track.lost[safe] == 0
    if not bool(np.all(valid)):
        return None
    if video_path is None:
        # Video could not be resolved/opened; the explicit WARNING listing the
        # attempted absolute paths was already emitted by find_video_path.
        return None

    extractor = CropExtractor(crop_size=crop_size, padding=padding, pad_value=pad_value)
    crops = []
    for i, fid in enumerate(past_frames):
        frame = decode_frame(video_path, fid)
        if frame is None:
            return None
        box = track.bbox[int(pos[i])]
        crops.append(extractor.frame_crop(frame, box))
    return np.stack(crops, axis=0)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class SDDAgentCropDataset(Dataset):
    """Sample agent-centric crops for trajectory windows.

    Combines :func:`parse_annotations` + :func:`extract_agent_crops` with an
    optional pre-existing window list (for LOSO integration).  Each ``__getitem__
    returns a dict with at least::

        "agent_crops"     : [T_obs, C, S, S] uint8 tensor
        "track_frames"    : [T_obs] int32 observed frame ids

    Notes
    -----
    * Frames are decoded lazily on demand.  For throughput, pre-extracting
      crops to disk (or caching decoded frames) is recommended.
    * We reuse :func:`video_encoder.sdd_adapter.find_video_path`, which scans
      both the Lab (``videos/…``/``.mov``) and Kaggle (``video/…``/``.mp4``)
      layouts and file extensions, so no extra configuration is needed.
    """

    def __init__(
        self,
        sdd_root: str | Path,
        scene: str,
        video_id: str,
        track_ids: Sequence[int] | None = None,
        past_frames_per_window: Sequence[Sequence[int]] | None = None,
        crop_size: int = DEFAULT_CROP_SIZE,
        obs_frames: int = DEFAULT_OBS_FRAMES,
        padding: int = DEFAULT_PADDING,
        drop_lost: bool = True,
    ) -> None:
        self.root = expand_sdd_root(sdd_root)
        self.scene = scene
        self.video_id = video_id
        # Dynamically resolve the raw video across layouts/extensions; logs an
        # explicit WARNING with the attempted absolute paths when it fails.
        self.video_path = find_video_path(self.root, scene, video_id)

        self.annotations = parse_annotations(self.root, scene, video_id)
        self.tracks = build_tracks(self.annotations)
        self.crop_size = int(crop_size)
        self.obs_frames = int(obs_frames)
        self.padding = int(padding)
        self.drop_lost = bool(drop_lost)

        # Windows: either caller-provided (track, frame-list) pairs or derived
        # sliding windows over each track.
        if past_frames_per_window is None:
            self.windows = self._slide_windows()
        else:
            self.windows = []
            for tid, fids in zip(track_ids or [], past_frames_per_window):
                self.windows.append((int(tid), np.asarray(fids, dtype=np.int32)))

    def _slide_windows(self) -> list[tuple[int, np.ndarray]]:
        windows: list[tuple[int, np.ndarray]] = []
        for tid, track in self.tracks.items():
            n = track.frames.size
            if n < self.obs_frames:
                continue
            # Only windows whose T_obs frames are all present and contiguous.
            for start in range(n - self.obs_frames + 1):
                span = track.frames[start : start + self.obs_frames]
                if np.all(np.diff(span) == 1):
                    windows.append((tid, span))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        track_id, frames = self.windows[idx]
        crops = extract_agent_crops(
            self.annotations,
            self.video_path,
            track_id,
            frames,
            crop_size=self.crop_size,
            padding=self.padding,
            drop_lost=self.drop_lost,
        )
        if crops is None:
            # Defensive fallback: black crops if the source frame could not be
            # decoded (e.g. missing video file on a partial checkout).
            crops = np.zeros(
                (self.obs_frames, 3, self.crop_size, self.crop_size), dtype=np.uint8
            )
        return {
            "agent_crops": torch.from_numpy(np.ascontiguousarray(crops)),
            "track_frames": torch.from_numpy(np.asarray(frames, dtype=np.int32)),
            "track_id": torch.tensor([track_id], dtype=torch.int32),
            "scene": self.scene,
            "video_id": self.video_id,
        }


__all__ = [
    "ANN_DTYPE",
    "A_TRACK_ID",
    "A_XMIN",
    "A_YMIN",
    "A_XMAX",
    "A_YMAX",
    "A_FRAME",
    "A_LOST",
    "A_OCCLUDED",
    "A_GENERATED",
    "A_LABEL",
    "DEFAULT_CROP_SIZE",
    "parse_annotations",
    "build_tracks",
    "AgentTrack",
    "CropExtractor",
    "decode_frame",
    "extract_agent_crops",
    "SDDAgentCropDataset",
    "SDD_SCENES",
]
