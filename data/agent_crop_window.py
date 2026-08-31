"""Efficient agent-centric crop fetching aligned with LOSO trajectory windows.

This module bridges the flat trajectory window index (``SDDGlobalDataset`` /
``SDDWindowIndex`` in ``data/dataloader_sdd_global.py``) and the agent-centric
crop machinery (``data/agent_crop_sdd.py``).

Why a dedicated fetcher instead of ``SDDAgentCropDataset``?
------------------------------------------------------------
``SDDAgentCropDataset`` re-derives its own sliding windows and decodes *each*
window with its own sequential-video seek.  For training over tens of
thousands of windows this is prohibitively I/O-bound.  Here we instead:

* reuse the *exact* ``SDDWindowIndex`` rows (scene / video / track / anchor),
  so crop windows are perfectly aligned with the trajectory windows that feed
  the CFM pipeline (strict coordinate synchronization, see CLAUDE.md);
* decode each video with a *single* forward pass (cheap ``grab()`` stepping +
  ``retrieve()`` only on the frames a window actually needs), caching decoded
  frames per ``(scene, video_id, frame_id)`` key.

The fetched tensor for a window is ``[T_obs, C, crop_size, crop_size]`` uint8,
identical to what ``SDDAgentCropDataset`` returns, so ``CropExtractor`` /
``AgentVideoEncoder`` expectations are preserved.  The crop bbox is taken from
the track's own annotations (centered on the SDD box, see CLAUDE.md crop rules);
``lost`` frames are dropped (their position is untrustworthy — same policy as
``extract_agent_crops``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from video_encoder.sdd_adapter import expand_sdd_root, video_mov_path
from data.agent_crop_sdd import (
    CropExtractor,
    DEFAULT_CROP_SIZE,
    DEFAULT_OBS_FRAMES,
    DEFAULT_PADDING,
    build_tracks,
    parse_annotations,
)


def _ordinal_for_frames(track: AgentTrack, frame_ids: np.ndarray) -> np.ndarray | None:
    """Return the ordinal index into ``track.frames`` for each observed frame.

    Returns ``None`` if a requested frame id is absent from the track or is
    flagged ``lost`` (crop would be centered on an untrustworthy box).
    """
    pos = np.searchsorted(track.frames, frame_ids)
    found = pos < track.frames.size
    found &= track.frames[np.clip(pos, 0, track.frames.size - 1)] == frame_ids
    safe = np.where(found, pos, 0)
    found &= track.lost[safe] == 0
    if not bool(np.all(found)):
        return None
    return pos


class SDDAgentWindowDataset:
    """Pre-materialized agent crops for every window in an ``SDDWindowIndex``.

    Crops are built eagerly (per ``(scene, video_id)`` group, one sequential
    video pass each) and stored as a single ``[N, T_obs, C, S, S]`` uint8 array
    plus a validity mask.  ``__getitem__`` returns ``(crops, valid)``.

    Parameters
    ----------
    sdd_root : root of the SDD dataset (Lab or Kaggle layout, auto-detected).
    windows : an ``SDDWindowIndex`` from ``data.dataloader_sdd_global``.
    crop_size : side length of the square crop in pixels.
    obs_frames : number of observed frames per window (must equal ``T_obs``).
    padding : extra margin added around the bbox before centering.
    drop_lost : when True, windows containing a ``lost`` frame get ``valid=False``.
    """

    def __init__(
        self,
        sdd_root: str | Path | None,
        windows,
        crop_size: int = DEFAULT_CROP_SIZE,
        obs_frames: int = DEFAULT_OBS_FRAMES,
        padding: int = DEFAULT_PADDING,
        drop_lost: bool = True,
    ) -> None:
        self.root = expand_sdd_root(sdd_root)
        self.windows = windows
        self.crop_size = int(crop_size)
        self.obs_frames = int(obs_frames)
        self.padding = int(padding)
        self.drop_lost = bool(drop_lost)
        self.extractor = CropExtractor(
            crop_size=self.crop_size, padding=self.padding, pad_value=0
        )

        self.crops, self.valid = self._build_all()

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def _build_all(self) -> tuple[np.ndarray, np.ndarray]:
        N = len(self.windows)
        crops = np.zeros(
            (N, self.obs_frames, 3, self.crop_size, self.crop_size),
            dtype=np.uint8,
        )
        valid = np.zeros(N, dtype=bool)

        # Group window indices by (scene, video_id) for single-pass decoding.
        groups: dict[tuple[str, str], list[int]] = {}
        for i in range(N):
            groups.setdefault(
                (self.windows.scene_id(i), self.windows.video_id(i)), []
            ).append(i)

        for (scene, video_id), idx_ls in groups.items():
            annotations = parse_annotations(self.root, scene, video_id)
            tracks = build_tracks(annotations)
            video_path = video_mov_path(self.root, scene, video_id)

            # Collect the union of frames this group's windows need, decode each
            # exactly once per group with a single forward scan of the video.
            needed = set()
            for i in idx_ls:
                needed.update(self.windows.past_frame_ids(i).tolist())
            frames_cache: dict[int, np.ndarray] = {}
            self._decode_needed(
                video_path, sorted(needed), frames_cache
            )

            for i in idx_ls:
                tid = int(self.windows.rows["track_id"][i])
                track = tracks.get(tid)
                if track is None:
                    valid[i] = False
                    continue
                past_fids = self.windows.past_frame_ids(i).astype(np.int32)
                if past_fids.size != self.obs_frames:
                    valid[i] = False
                    continue

                ords = _ordinal_for_frames(track, past_fids)
                if ords is None:
                    valid[i] = False
                    continue

                stack = np.empty(
                    (self.obs_frames, 3, self.crop_size, self.crop_size),
                    dtype=np.uint8,
                )
                for t, (fid, pos) in enumerate(zip(past_fids.tolist(), ords.tolist())):
                    frame = frames_cache.get(fid)
                    if frame is None:
                        valid[i] = False
                        break
                    box = track.bbox[pos]
                    stack[t] = self.extractor.frame_crop(frame, box)
                else:
                    crops[i] = stack
                    valid[i] = True

        return crops, valid

    def _decode_needed(
        self,
        video_path: Path,
        needed: Sequence[int],
        cache: dict[int, np.ndarray],
    ) -> None:
        """Walk a single VideoCapture forward, decoding exactly the frames in
        ``needed`` (ascending) and caching them in ``cache``."""
        import cv2

        if not needed:
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            # Missing/unreadable video → leave cache empty (valid=False downstream).
            cap.release()
            return
        try:
            cursor = -1
            for target in needed:
                # Forward-step the capture to ``target`` with cheap grab().
                while cursor < target:
                    ok = cap.grab()
                    cursor += 1
                    if not ok:
                        break
                if cursor == target:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        cache[target] = np.asarray(frame)
        finally:
            cap.release()

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, bool]:
        """Return ``(crops [T_obs, C, S, S] uint8, valid bool)`` for ``idx``."""
        return self.crops[idx], bool(self.valid[idx])


__all__ = ["SDDAgentWindowDataset", "_ordinal_for_frames"]
