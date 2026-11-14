"""Cached, thread-safe OpenCV reader for native ETH/UCY ``video.avi`` files.

``video.avi`` is the only visual source of truth for ETH/UCY.  Pre-extracted
frame files and nearest-frame substitutions are not supported: a missing file
or undecodable exact frame raises immediately.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np


class VideoDecodeError(ValueError):
    """Raised when an exact frame index cannot be decoded from a video."""


class VideoFrameReader:
    """Exact-index frame reader over a single video file.

    Instances are reused through :func:`get_video_reader`; each reader owns one
    ``cv2.VideoCapture`` and serializes access with a lock.  The cache is only
    an optimization -- the file on disk remains the source of truth.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.RLock()
        self._frame_count: int | None = None
        self._fps: float | None = None
        self._size: tuple[int, int] | None = None
        self._next_index: int | None = None

    # ------------------------------------------------------------------ props
    def _ensure_open(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise FileNotFoundError(f"OpenCV failed to open video: {self.path}")
            self._cap = cap
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._frame_count = total if total > 0 else None
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self._fps = fps if fps > 0 else None
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self._size = (w, h) if w > 0 and h > 0 else None
            self._next_index = 0
        return self._cap

    @property
    def frame_count(self) -> int:
        with self._lock:
            self._ensure_open()
            if self._frame_count is None:
                raise VideoDecodeError(f"Video reports no frame count: {self.path}")
            return int(self._frame_count)

    @property
    def fps(self) -> float | None:
        with self._lock:
            self._ensure_open()
            return self._fps

    @property
    def size(self) -> tuple[int, int] | None:
        """Native ``(width, height)``."""
        with self._lock:
            self._ensure_open()
            return self._size

    # ------------------------------------------------------------------- read
    def _read_sequential(self, index: int) -> np.ndarray:
        """Decode exactly ``index`` frames from the start (slow but exact)."""
        cap = self._cap
        if cap is None:
            raise VideoDecodeError(f"Video reader is closed: {self.path}")
        cap.release()
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"OpenCV failed to reopen video: {self.path}")
        self._cap = cap
        frame = None
        for i in range(index + 1):
            ok, fr = cap.read()
            if not ok or fr is None:
                self._next_index = None
                raise VideoDecodeError(
                    f"Failed to decode frame {index} from {self.path} "
                    f"(video ended at frame {i})."
                )
            frame = fr
        self._next_index = index + 1
        return frame

    def read(self, index: int, *, color: str = "rgb") -> np.ndarray:
        """Decode exactly ``index`` (0-based) and return an RGB or BGR array."""
        idx = int(index)
        if idx < 0:
            raise ValueError(f"Frame index must be non-negative, got {idx}")
        if color not in ("rgb", "bgr"):
            raise ValueError(f"color must be 'rgb' or 'bgr', got {color!r}")

        with self._lock:
            cap = self._ensure_open()
            if self._frame_count is not None and idx >= self._frame_count:
                raise VideoDecodeError(
                    f"Frame index {idx} out of range for {self.path} "
                    f"({self._frame_count} frames)."
                )

            frame: np.ndarray | None = None
            if self._next_index == idx:
                # Fast path: sequential forward read.
                ok, fr = cap.read()
                if ok and fr is not None:
                    frame = fr
                    self._next_index = idx + 1
                else:
                    self._next_index = None
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = cap.read()
                if ok and fr is not None:
                    frame = fr
                    self._next_index = idx + 1
                else:
                    self._next_index = None

            if frame is None:
                # Exact sequential fallback (never substitutes a different frame).
                frame = self._read_sequential(idx)

            if color == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                finally:
                    self._cap = None
                    self._next_index = None

    def __enter__(self) -> "VideoFrameReader":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_CACHE: dict[str, VideoFrameReader] = {}
_CACHE_LOCK = threading.RLock()


def get_video_reader(path: str | Path) -> VideoFrameReader:
    """Return a process-wide cached :class:`VideoFrameReader` for ``path``."""
    key = str(Path(path).resolve())
    with _CACHE_LOCK:
        reader = _CACHE.get(key)
        if reader is None:
            reader = VideoFrameReader(key)
            _CACHE[key] = reader
        return reader


def clear_video_reader_cache() -> None:
    """Release and drop every cached video reader (used by tests)."""
    with _CACHE_LOCK:
        readers = list(_CACHE.values())
        _CACHE.clear()
    for reader in readers:
        reader.close()
