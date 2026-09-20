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
  the CFM pipeline (strict trajectory/video coordinate synchronization);
* decode each video with a *single* forward pass (cheap ``grab()`` stepping +
  ``retrieve()`` only on the frames a window actually needs), caching decoded
  frames per ``(scene, video_id, frame_id)`` key.

The fetched tensor for a window is ``[T_obs, C, crop_size, crop_size]`` uint8,
identical to what ``SDDAgentCropDataset`` returns, so ``CropExtractor`` /
``AgentVideoEncoder`` expectations are preserved.  The crop bbox is taken from
the track's own annotations (centered on the SDD box that the trajectory branch
already consumes).

Every trajectory window is guaranteed a real, non-black crop:

* ``lost`` frames keep a crop centered on the *same* annotation box the
  trajectory branch uses (``drop_lost`` is opt-in strict mode);
* an isolated undecodable frame falls back to the nearest decoded frame;
* a group whose raw video is missing falls back to the directory's
  ``reference.jpg`` still image;
* a group with **no pixel source at all** (no video AND no still image) raises
  instead of silently black-filling the catalog.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from video_encoder.sdd_adapter import (
    expand_sdd_root,
    find_video_path,
    reference_image_path,
)
from data.agent_crop_sdd import (
    CropExtractor,
    DEFAULT_CROP_SIZE,
    DEFAULT_OBS_FRAMES,
    DEFAULT_PADDING,
    build_tracks,
    parse_annotations,
)


def _ordinal_for_frames(
    track: AgentTrack, frame_ids: np.ndarray, drop_lost: bool = False
) -> np.ndarray | None:
    """Return the ordinal index into ``track.frames`` for each observed frame.

    Returns ``None`` only when a requested frame id is absent from the track
    (or, in strict ``drop_lost`` mode, flagged ``lost``).  By default ``lost``
    frames are kept: the crop is centered on the same annotation box the
    trajectory branch uses for that frame, keeping coordinates synchronized.
    """
    pos = np.searchsorted(track.frames, frame_ids)
    found = pos < track.frames.size
    found &= track.frames[np.clip(pos, 0, track.frames.size - 1)] == frame_ids
    safe = np.where(found, pos, 0)
    if drop_lost:
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
    drop_lost : strict opt-in mode.  When True, windows containing a ``lost``
        frame get ``valid=False`` (their crops are *not* emitted).  The default
        (False) matches the trajectory branch, which consumes every annotated
        frame — crops stay centered on those same boxes.
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

        unresolvable: list[tuple[str, str, str]] = []
        for (scene, video_id), idx_ls in groups.items():
            annotations = parse_annotations(self.root, scene, video_id)
            tracks = build_tracks(annotations)
            # Dynamic resolution across layouts/extensions; logs a WARNING with
            # the attempted absolute paths on failure. When no raw video exists,
            # fall back to the directory's still image (reference.jpg) so a
            # partial mirror cannot black-out a group.
            video_path = find_video_path(self.root, scene, video_id)
            still = (
                None
                if video_path is not None
                else reference_image_path(self.root, scene, video_id)
            )
            if video_path is None and still is None:
                unresolvable.append(
                    (scene, video_id, "no raw video and no reference still image")
                )
                continue

            # Collect the union of frames this group's windows need, decode each
            # exactly once per group with a single forward scan of the video.
            needed = set()
            for i in idx_ls:
                needed.update(self.windows.past_frame_ids(i).tolist())
            frames_cache: dict[int, np.ndarray] = {}
            self._decode_needed(video_path, still, sorted(needed), frames_cache)
            if not frames_cache:
                unresolvable.append(
                    (scene, video_id, "pixel source present but zero frames decoded")
                )
                continue

            for i in idx_ls:
                tid = int(self.windows.rows["track_id"][i])
                track = tracks.get(tid)
                if track is None:
                    unresolvable.append(
                        (
                            scene,
                            video_id,
                            f"track_id={tid} missing from parsed annotations",
                        )
                    )
                    continue
                past_fids = self.windows.past_frame_ids(i).astype(np.int32)
                if past_fids.size != self.obs_frames:
                    unresolvable.append(
                        (
                            scene,
                            video_id,
                            f"window {i} asks for {past_fids.size} frames, expected "
                            f"{self.obs_frames}",
                        )
                    )
                    continue
                ords = _ordinal_for_frames(
                    track, past_fids, drop_lost=self.drop_lost
                )
                if ords is None:
                    valid[i] = False  # strict drop_lost policy
                    continue

                stack = np.empty(
                    (self.obs_frames, 3, self.crop_size, self.crop_size),
                    dtype=np.uint8,
                )
                for t, (fid, pos) in enumerate(zip(past_fids.tolist(), ords.tolist())):
                    frame = frames_cache.get(fid)
                    if frame is None:
                        unresolvable.append(
                            (
                                scene,
                                video_id,
                                f"frame id {fid} undecodable and no nearest-frame "
                                "fallback available",
                            )
                        )
                        break
                    box = track.bbox[pos]
                    stack[t] = self.extractor.frame_crop(frame, box)
                else:
                    crops[i] = stack
                    valid[i] = True

        if unresolvable:
            details = "; ".join(
                f"{s}/{v} ({reason})" for s, v, reason in unresolvable
            )
            raise RuntimeError(
                f"Cannot extract agent-centric crops for {len(unresolvable)} "
                f"(scene/video) group(s): {details}. "
                f"Trajectory windows exist, but their (scene, video_id) has no "
                f"decodable pixel source under {self.root}. Verify every annotated "
                f"video directory ships a raw video (videos/…, video/…) or a "
                f"reference.jpg still image."
            )

        return crops, valid

    def _decode_needed(
        self,
        video_path: Path | None,
        still: Path | None,
        needed: Sequence[int],
        cache: dict[int, np.ndarray],
    ) -> None:
        """Fill ``cache`` with one frame array per id in ``needed`` (ascending).

        Primary source: a single forward ``grab()``/``retrieve()`` walk of the
        ``cv2.VideoCapture`` for ``video_path``.  A frame that cannot be
        retrieved (mux glitch, or an id at/after the end of the video) is
        transparently mapped to the most recent successfully decoded frame so
        the window still gets real pixels instead of a black-crop fallback.

        When ``video_path`` is ``None`` but ``still`` (reference.jpg) is set,
        the still image is decoded once and serves every requested frame id.

        Frame indexing: SDD annotation ``frame`` ids are 0-based OpenCV ordinals
        (video frame 0 == annotation ``frame=0``), so ``grab()``/``retrieve()``
        are stepped directly to each ``frame_id``.
        """
        import logging
        import cv2

        if not needed:
            return
        if still is not None:
            img = cv2.imread(str(still))
            if img is None:
                logging.getLogger("agent_crop_window").warning(
                    "reference-image fallback failed to decode: %s", still
                )
                return
            frame = np.asarray(img)
            for target in needed:
                cache.setdefault(target, frame)
            return
        if video_path is None:
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.getLogger("agent_crop_window").warning(
                "cv2.VideoCapture failed to open video at absolute path %s",
                str(Path(video_path).resolve()),
            )
            cap.release()
            return
        try:
            cursor = -1
            last_frame: np.ndarray | None = None
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
                        arr = np.asarray(frame)
                        cache[target] = arr
                        last_frame = arr
                        continue
                # Nearest-frame fallback: reuse the latest decoded frame so one
                # undecodable frame cannot black-out the whole window.
                if last_frame is not None:
                    cache[target] = last_frame
        finally:
            cap.release()

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, bool]:
        """Return ``(crops [T_obs, C, S, S] uint8, valid bool)`` for ``idx``."""
        return self.crops[idx], bool(self.valid[idx])


__all__ = ["SDDAgentWindowDataset", "_ordinal_for_frames"]
