"""SDD-specific adapter for the global video encoder (note §5.2).

Dual-infrastructure environment matrix:

A. **Remote Lab Machine** (default / production — full multi-epoch training)::

    /home/efrei_stage/Desktop/Datasets/SDD/
        annotations/<scene>/<videoX>/annotations.txt  # TrackID,xmin,...,frame,...
        videos/<scene>/<videoX>/video.mov             # subdir ``videos/``, ext .mov

B. **Kaggle** (compute proxy — VRAM profiling with dummy caches)::

    /kaggle/input/datasets/aryashah2k/stanford-drone-dataset/
        annotations/<scene>/<videoX>/annotations.txt
        video/<scene>/<videoX>/video.mp4              # subdir ``video/``, ext .mp4

Resolution order: explicit argument > Kaggle auto-detection > Lab default.
All CLI parsers and function signatures default to the **Lab machine values**;
Kaggle is handled via :func:`is_kaggle` auto-detection or explicit overrides.

Raw-video resolution (:func:`find_video_path`) combines a fixed layout grid
(``videos/`` vs ``video/``, ``.mov/.MOV/.mp4/.MP4/.avi/.AVI``) with a **live
directory scan**, so annotation/video folder-name mismatches (``video0`` vs
``0`` vs ``video_0``, zero-padding, casing) do not black-out crops.
"""

from __future__ import annotations

import math
import os
import re
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

# ---------------------------------------------------------------------------
# Environment matrix (defaults are the Remote Lab Machine values)
# ---------------------------------------------------------------------------
LAB_SDD_ROOT = Path("/home/efrei_stage/Desktop/Datasets/SDD")
KAGGLE_SDD_ROOT = Path("/kaggle/input/datasets/aryashah2k/stanford-drone-dataset")

#: Signature-level default: the Remote Lab Machine root.
DEFAULT_SDD_ROOT = LAB_SDD_ROOT


def is_kaggle() -> bool:
    """True when executing inside a Kaggle notebook container."""
    return os.path.exists("/kaggle")


def detect_sdd_root() -> Path:
    """Auto-detect the dataset root for the current environment.

    Returns the Kaggle mount when ``/kaggle`` exists, otherwise the Remote
    Lab Machine root (production default).
    """
    return KAGGLE_SDD_ROOT if is_kaggle() else LAB_SDD_ROOT


def _is_kaggle_root(root: Path) -> bool:
    """Classify a resolved root so the video layout follows its environment."""
    return str(root).startswith("/kaggle")


def expand_sdd_root(maybe_root: str | Path | None) -> Path:
    """Resolve the SDD root directory.

    Resolution order: explicit argument > Kaggle auto-detection > Lab default.
    The constant is intentionally *hardcoded* — local repositories must NOT
    ship a copy of SDD. Anything that bypasses this helper is a bug.
    """
    if maybe_root is None:
        return detect_sdd_root()
    return Path(maybe_root).expanduser()


#: Video subdirectory name per environment (Lab: ``videos/``, Kaggle: ``video/``).
VIDEO_DIR_LAB = "videos"
VIDEO_DIR_KAGGLE = "video"
#: Video file extension per environment (Lab: ``.mov``, Kaggle: ``.mp4``).
VIDEO_EXT_LAB = ".mov"
VIDEO_EXT_KAGGLE = ".mp4"


def scene_annotations_dir(sdd_root: Path, scene: str) -> Path:
    return sdd_root / "annotations" / scene


def scene_videos_dir(sdd_root: Path, scene: str) -> Path:
    subdir = VIDEO_DIR_KAGGLE if _is_kaggle_root(Path(sdd_root)) else VIDEO_DIR_LAB
    return Path(sdd_root) / subdir / scene


def video_mov_path(sdd_root: Path, scene: str, video_id: str) -> Path:
    """Return the canonical raw-video path for a given (scene, video_id).

    Lab layout: ``<sdd_root>/videos/<scene>/<video_id>/video.mov``;
    Kaggle layout: ``<sdd_root>/video/<scene>/<video_id>/video.mp4``.
    """
    ext = VIDEO_EXT_KAGGLE if _is_kaggle_root(Path(sdd_root)) else VIDEO_EXT_LAB
    return scene_videos_dir(sdd_root, scene) / video_id / f"video{ext}"


#: Directory layout names to try, in order (Lab ``videos/`` then Kaggle ``video/``).
VIDEO_DIR_CANDIDATES = ("videos", "video")
#: File extensions to try, in order, covering common raw-drone wrappers.
VIDEO_EXT_CANDIDATES = (".mov", ".MOV", ".mp4", ".MP4", ".avi", ".AVI")
#: File name stem for the raw video within each ``<scene>/<video_id>/`` folder.
VIDEO_FILE_STEM = "video"
#: Fallback still image used by some SDD mirrors when the raw video is absent.
REFERENCE_IMAGE_NAME = "reference.jpg"


def _normalize_video_key(name: str) -> str:
    """Return a comparable numeric core from a video folder name.

    Strips every non-digit, so ``video001``, ``0001``, ``video_1`` and ``1`` all
    reduce to ``1``.  Used to match an annotation ``video_id`` (e.g. ``video0``)
    against the actual on-disk video subdirectory name (e.g. ``0`` or ``video_0``).
    """
    digits = re.sub(r"\D", "", str(name))
    return digits.lstrip("0") or "0"


def _video_dir_by_listing(video_scene_dir: Path, video_id: str) -> Path | None:
    """Locate the video subdir matching ``video_id`` via a **live listing**.

    Tries the exact directory name first, then falls back to a numeric fuzzy
    match against every subdirectory of ``video_scene_dir``.  This tolerates the
    lab-machine naming mismatches called out in the debug notes (``video0`` vs
    ``0`` vs ``video_0``, zero-padding, casing).
    """
    exact = video_scene_dir / video_id
    if exact.is_dir():
        return exact
    if not video_scene_dir.is_dir():
        return None
    want = _normalize_video_key(video_id)
    for p in sorted(video_scene_dir.iterdir()):
        if p.is_dir() and _normalize_video_key(p.name) == want:
            return p
    return None


def _existing_video_files_in_dir(video_dir: Path) -> list[Path]:
    """Return the existing, non-empty raw-video files inside ``video_dir``.

    Prefers the canonical ``video<ext>`` name, then falls back to any
    ``.mov/.MOV/.mp4/.MP4/.avi/.AVI`` glob so differently-named raw videos
    (``raw.mov`` etc.) still resolve.  Does not consider ``reference.jpg``
    (see :data:`REFERENCE_IMAGE_NAME`) a *video* — it is only surfaced for
    diagnostics.
    """
    outs: list[Path] = []
    for ext in VIDEO_EXT_CANDIDATES:
        p = video_dir / f"{VIDEO_FILE_STEM}{ext}"
        if p.is_file() and p.stat().st_size > 0 and p not in outs:
            outs.append(p)
    for pattern in ("*.mov", "*.MOV", "*.mp4", "*.MP4", "*.avi", "*.AVI"):
        outs.extend(
            p
            for p in video_dir.glob(pattern)
            if p.is_file() and p.stat().st_size > 0 and p not in outs
        )
    return outs


def _listing_context(sdd_root: Path, scene: str) -> str:
    """Build a diagnostic string of what actually exists under the scene video dirs."""
    parts = []
    for layout in VIDEO_DIR_CANDIDATES:
        d = Path(sdd_root) / layout / scene
        if d.is_dir():
            entries = sorted(str(p.name) for p in d.iterdir())
            parts.append(f"{d} -> {entries}")
        else:
            parts.append(f"{d} -> MISSING")
    return "; ".join(parts)


def _resolve_video_candidates(sdd_root: Path, scene: str, video_id: str) -> list[Path]:
    """Yield every plausible absolute raw-video path for ``(scene, video_id)``.

    Combines the fixed ``<layout>/<scene>/<video_id>/video<ext>`` grid with a
    **live directory scan** of ``videos/<scene>/`` and ``video/<scene>/`` so a
    mismatch between the annotation subdir name (``video0``) and the video
    subdir name (``0``, ``video_0``, ``0001``) still resolves.  Also tries the
    user-documented ``video<video_id>`` folder spelling as a second grid row.

    Duplicate absolute paths are de-duplicated (in order).
    """
    cur_dir = Path(sdd_root) if sdd_root else Path.cwd()
    video_scene_dirs = [cur_dir / layout / scene for layout in VIDEO_DIR_CANDIDATES]

    candidates: list[Path] = []
    for vsc in video_scene_dirs:
        # 1. Fixed grid: <scene>/<video_id>/video<ext> and <scene>/video<video_id>/video<ext>
        for vdir_name in (video_id, f"{VIDEO_FILE_STEM}{video_id}"):
            for ext in VIDEO_EXT_CANDIDATES:
                candidates.append(
                    (vsc / vdir_name / f"{VIDEO_FILE_STEM}{ext}").resolve()
                )
        # 2. Live listing: exact-name or numeric-fuzzy-matched subdir -> its files
        matched = _video_dir_by_listing(vsc, video_id)
        if matched is not None:
            for f in _existing_video_files_in_dir(matched):
                candidates.append(f.resolve())

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def find_video_path(
    sdd_root: str | Path | None,
    scene: str,
    video_id: str,
    verify_open: bool = True,
) -> Path | None:
    """Dynamically resolve the raw video for ``(scene, video_id)``.

    Unlike the fixed-layout :func:`video_mov_path`, this scans **both** the Lab
    (``videos/``) and Kaggle (``video/``) directory layouts across the common
    extensions (``.mov/.MOV/.mp4/.MP4/.avi/.AVI``), and additionally performs a
    **live directory listing** so annotation/video folder name mismatches
    (``video0`` vs ``0`` vs ``video_0``, zero-padding, casing) do not silently
    fall through to the black-crop path.

    When ``verify_open`` is set, a candidate must open with ``cv2.VideoCapture``
    to be accepted; but if *every* existing file fails the open probe, the
    largest existing non-empty file is returned as a best-effort fallback so a
    transient codec-probe false-negative cannot black-out an entire scene.  The
    decode path is the source of truth for actual decodability and logs its own
    absolute-path warning on failure.

    Returns ``None`` only when no candidate file exists at all — and logs an
    explicit ``WARNING`` containing the attempted absolute paths *and* a live
    listing of the scene's ``videos/`` + ``video/`` directories so the lab
    layout can be traced.
    """
    import logging

    logger = logging.getLogger("sdd_adapter")
    root = expand_sdd_root(sdd_root)
    candidates = _resolve_video_candidates(root, scene, video_id)

    existing = [p for p in candidates if p.is_file()]
    if existing:
        import cv2

        if verify_open:
            for path in existing:
                cap = cv2.VideoCapture(str(path))
                ok = cap.isOpened()
                cap.release()
                if ok:
                    return path
            # Every existing file failed the open probe: fall back to the
            # largest (most likely to be a genuine, decodable raw video).  The
            # decode step re-verifies and emits its own absolute-path warning.
            best = max(existing, key=lambda p: p.stat().st_size)
            logger.warning(
                "find_video_path: scene=%r video_id=%r — %d existing candidate(s) "
                "failed the cv2.VideoCapture open probe; falling back to the "
                "largest file %s for decode.",
                scene,
                video_id,
                len(existing),
                best,
            )
            return best
        return existing[0]

    attempted = ", ".join(str(p) for p in candidates)
    logger.warning(
        "find_video_path: could not resolve a raw video for scene=%r video_id=%r. "
        "Paths attempted (absolute): %s. "
        "Live listing: %s",
        scene,
        video_id,
        attempted,
        _listing_context(root, scene),
    )
    return None


def annotation_path(sdd_root: Path, scene: str, video_id: str) -> Path:
    return scene_annotations_dir(sdd_root, scene) / video_id / "annotations.txt"


#: Image file extensions probed by reference_image_path (still-image fallback).
IMAGE_EXT_CANDIDATES = (".jpg", ".jpeg", ".png", ".bmp")


def reference_image_path(
    sdd_root: str | Path | None,
    scene: str,
    video_id: str,
) -> Path | None:
    """Return the still-image fallback for ``(scene, video_id)``, if present.

    Some SDD mirrors ship ``reference.jpg`` in each video directory when the raw
    video itself is absent.  Preferring ``reference.jpg`` over any other image,
    this scans both layout grids (``videos/`` and ``video/``) and applies the
    same numeric folder-name normalization as :func:`find_video_path`, so a
    ``video0`` vs ``0`` mismatch still resolves.

    Returns ``None`` when no video directory (or no image inside it) exists —
    such groups have no pixel source at all and must be surfaced, not black-filled.
    """
    root = expand_sdd_root(sdd_root)
    for layout in VIDEO_DIR_CANDIDATES:
        vsc = Path(root) / layout / scene
        d = _video_dir_by_listing(vsc, video_id)
        if d is None:
            continue
        canonical = d / REFERENCE_IMAGE_NAME
        if canonical.is_file() and canonical.stat().st_size > 0:
            return canonical
        for p in sorted(d.iterdir()):
            if (
                p.is_file()
                and p.stat().st_size > 0
                and p.suffix.lower() in IMAGE_EXT_CANDIDATES
            ):
                return p
    return None


# Annotation column indices (whitespace-separated).
COL_TRACK_ID = 0
COL_XMIN = 1
COL_YMIN = 2
COL_XMAX = 3
COL_YMAX = 4
COL_FRAME = 5


@dataclass
class SDDFrameFeatureLookup:
    """Video-scoped cached feature lookup for SDD.

    The cache layout is::

        <features_root>/<scene>.npy              # [N_frames, D] float32
        <features_root>/<scene>.manifest.parquet  # columns: scene, video_id, frame_id, row_idx

    Because SDD ``frame_id`` values are 0-based *per video*, lookups are keyed
    by ``(video_id, frame_id)`` — never by ``frame_id`` alone. A window from
    video B must resolve its frames inside **B's own** rows; resolving against
    a frame-only map would silently serve video A's features whenever the two
    videos' frame-id ranges overlap.

    A cache built by the *pre-fix* encoder collapsed ``drop_duplicates("frame_id")``,
    so whole rows for later-sorted videos are missing. When ``sdd_root`` is
    provided, :meth:`from_root` cross-checks the manifest against the per-video
    annotation frame sets and raises if any annotated frame is missing — such a
    cache must be re-encoded before use.
    """

    scene: str
    features: np.ndarray  # [N, D] — mmap'd when mmap=True, so RSS stays flat
    video_sorted_ids: dict[str, np.ndarray]  # video_id -> ascending frame ids
    # video_id -> feature row indices ALIGNED with video_sorted_ids. These two
    # parallel int64 arrays replace the previous dict[(video_id, frame_id)] ->
    # row mapping: a Python dict of N tuple keys costs ~100 B/entry and was the
    # single largest RAM consumer of the lookup (built once per scene per
    # process, then retained by the lru_cache).
    video_sorted_rows: dict[str, np.ndarray]

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        scene: str,
        sdd_root: str | Path | None = None,
        mmap: bool = True,
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

        if "video_id" not in man.columns:
            raise ValueError(
                f"SDD feature manifest for scene={scene!r} has no 'video_id' column "
                f"({man.shape[0]} rows). This is a pre-scope cache — re-encode with "
                "`python -m video_encoder encode-sdd`."
            )

        # Build the per-video index vectorized: two int64 arrays per video
        # instead of a Python dict of tuples. Nothing list-shaped proportional
        # to the number of frames is ever materialized.
        vids = man["video_id"].astype(str).to_numpy()
        fids = man["frame_id"].astype(np.int64).to_numpy()
        ridx = man["row_idx"].astype(np.int64).to_numpy()
        n_manifest_rows = int(man.shape[0])
        del man  # the DataFrame is not needed after the index is built

        video_sorted_ids: dict[str, np.ndarray] = {}
        video_sorted_rows: dict[str, np.ndarray] = {}
        for v in np.unique(vids):
            sel = vids == v
            f = fids[sel]
            r = ridx[sel]
            order = np.argsort(f, kind="stable")
            video_sorted_ids[str(v)] = f[order]
            video_sorted_rows[str(v)] = r[order]
        del vids, fids, ridx

        # Pre-fix detection: the old encoder dropped duplicate frame ids across
        # videos, so a whole losing video's annotated frames are absent. Compare
        # the manifest against the annotations per (scene, video).
        if sdd_root is not None:
            _verify_manifest_vs_annotations(
                Path(sdd_root),
                scene,
                video_sorted_ids,
                npy.stat().st_size,
                n_manifest_rows,
            )

        return cls(
            scene=scene,
            features=feats,
            video_sorted_ids=video_sorted_ids,
            video_sorted_rows=video_sorted_rows,
        )

    # ---- single-frame access ------------------------------------------------
    def get(
        self, frame_id: int, video_id: str, policy: SnapPolicy = "nearest"
    ) -> np.ndarray | None:
        row = self._resolve(video_id, int(frame_id), policy)
        if row is None:
            return None
        return np.asarray(self.features[row])

    # ---- windowed access ----------------------------------------------------
    def window(
        self,
        start_frame_id: int,
        n_frames: int,
        video_id: str,
        stride: int = 1,
        policy: SnapPolicy = "nearest",
    ) -> np.ndarray:
        """Return ``[n_frames, D]`` for ``[start, start+stride, ..., start+(n-1)*stride]``.

        All frames resolve inside ``video_id``'s own rows only. Unresolvable
        frames are zero-filled (matches the ETH/UCY helper semantics for
        missing rows).
        """
        D = int(self.features.shape[1])
        out = np.zeros((n_frames, D), dtype=np.float32)
        for i in range(n_frames):
            fid = int(start_frame_id) + i * int(stride)
            row = self._resolve(video_id, fid, policy)
            if row is not None:
                out[i] = np.asarray(self.features[row])
        return out

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    # ---- internal -----------------------------------------------------------
    def _resolve(self, video_id: str, frame_id: int, policy: SnapPolicy) -> int | None:
        """Resolve ``frame_id`` inside ``video_id``'s own rows only.

        ``side="right" - 1`` lands on the exact frame when present (and on the
        LAST duplicate, matching the dict-based behaviour this replaced); when
        absent, ``pos`` is the floor candidate.
        """
        ids = self.video_sorted_ids.get(video_id)
        if ids is None or ids.size == 0:
            return None
        rows = self.video_sorted_rows[video_id]
        pos = int(np.searchsorted(ids, frame_id, side="right")) - 1
        if pos >= 0 and int(ids[pos]) == frame_id:
            return int(rows[pos])
        if policy == "drop":
            return None
        if policy == "floor":
            return int(rows[pos]) if pos >= 0 else None
        # nearest: floor candidate (pos) vs the next frame (pos + 1); ties keep
        # the floor, as the previous implementation did.
        best = pos
        if best < 0 or (
            pos + 1 < ids.size
            and abs(int(ids[pos + 1]) - frame_id) < abs(int(ids[best]) - frame_id)
        ):
            best = pos + 1
        return int(rows[best]) if best >= 0 else None

    @property
    def video_frame_to_row(self) -> dict[tuple[str, int], int]:
        """Materialize ``(video_id, frame_id) -> row`` (debug/tests only).

        Kept for backwards compatibility. Building this dict costs ~100 B per
        frame, so hot loops must use :meth:`get` / :meth:`window` instead.
        """
        out: dict[tuple[str, int], int] = {}
        for vid, ids in self.video_sorted_ids.items():
            rows = self.video_sorted_rows[vid]
            for f, r in zip(ids.tolist(), rows.tolist()):
                out[(vid, f)] = r
        return out


@lru_cache(maxsize=64)
def _cached_lookup(
    features_root: str, scene: str, sdd_root: str | None = None
) -> SDDFrameFeatureLookup:
    """Memoize feature lookups across ``__getitem__`` calls.

    The cache lives for the lifetime of the DataLoader worker process.
    Memory cost: one mmap'd ``.npy`` per scene (~5–20 MB).
    """
    return SDDFrameFeatureLookup.from_root(features_root, scene, sdd_root=sdd_root)


def release_lookup_cache() -> None:
    """Drop every memoized :class:`SDDFrameFeatureLookup` in this process.

    Each cached lookup pins one mmap'd ``.npy`` plus its per-video index. A
    long-running process that walks several scenes (or a sweep cell that finishes
    a scene) should call this between scenes so the previous scene's mapping is
    unmapped immediately instead of waiting for the process to exit.
    """
    _cached_lookup.cache_clear()


def get_lookup(
    features_root: str | Path,
    scene: str,
    sdd_root: str | Path | None = None,
) -> SDDFrameFeatureLookup:
    """Public, lru_cache-wrapped accessor (one mmap per process per scene).

    ``sdd_root`` is optional but **strongly recommended**: when provided, the
    per-video manifest is cross-checked against the annotation frame sets so a
    pre-fix (frame-collapsed) cache fails loudly instead of silently serving
    wrong-camera features.
    """
    return _cached_lookup(
        str(Path(features_root)),
        scene,
        str(Path(sdd_root)) if sdd_root is not None else None,
    )


def temporal_smooth(features: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Gaussian temporal smoothing along the time axis of ``[T, D]`` features.

    ResNet-18 per-frame features are noisy; a light Gaussian blur over the
    observation window decorrelates that noise before the dataloader mean-pools
    ``[T, D] -> [D]`` (see ``data/dataloader_sdd_global.py``). ``sigma`` is in
    **frames** (e.g. ``1.0`` blurs ±3 frames at ``3σ``).

    * ``sigma <= 0``  → identity (returned as-is, no copy).
    * Kernel radius is ``ceil(3σ)``; edges are padded by reflection so the
      window's first/last frames are not zero-damped.
    * Output dtype == input dtype; shape is preserved ``[T, D]``.
    """
    if sigma is None or sigma <= 0 or features.shape[0] <= 1:
        return features
    if features.ndim != 2:
        raise ValueError(f"temporal_smooth expects [T, D], got shape {features.shape}")

    radius = int(math.ceil(3.0 * sigma))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(features, ((radius, radius), (0, 0)), mode="reflect")
    out = np.empty_like(features, dtype=np.float64)
    for d in range(features.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return out.astype(features.dtype, copy=False)


def _verify_manifest_vs_annotations(
    sdd_root: Path,
    scene: str,
    cached_ids: dict[str, np.ndarray],
    npy_bytes: int,
    n_manifest_rows: int,
) -> None:
    """Raise if any annotated (video, frame) row for ``scene`` is missing from the manifest.

    This is the authoritative pre-fix detector: the old encoder ran
    ``drop_duplicates("frame_id")`` across videos, deleting an entire losing
    video's frames wherever two videos' frame-id ranges overlapped. A
    correctly encoded cache contains **every** video's annotated frames.

    ``cached_ids`` is the per-video ascending frame-id array already built by
    :meth:`SDDFrameFeatureLookup.from_root`, so this check no longer allocates a
    second Python ``dict[str, set[int]]`` copy of the whole manifest.
    """
    ann_dir = scene_annotations_dir(sdd_root, scene)
    if not ann_dir.is_dir():
        return  # no annotations -> nothing to check (defensive)

    missing_total = 0
    per_video = []
    for vdir in sorted(p for p in ann_dir.iterdir() if p.is_dir()):
        txt = vdir / "annotations.txt"
        if not txt.exists():
            continue
        arr = np.loadtxt(
            txt,
            usecols=(COL_FRAME,),
        )
        annotated = np.atleast_1d(arr).astype(np.int64, copy=False)
        if annotated.size == 0:
            continue
        annotated = np.unique(annotated)
        have = cached_ids.get(vdir.name)
        if have is None or have.size == 0:
            n_missing = int(annotated.size)
        else:
            n_missing = int(
                annotated.size - np.isin(annotated, have, assume_unique=True).sum()
            )
        if n_missing:
            missing_total += n_missing
            per_video.append(f"{vdir.name}:{n_missing}")
    if missing_total:
        sample = ", ".join(per_video[:8])
        raise ValueError(
            f"SDD feature cache for scene={scene!r} is PRE-FIX / INCOMPLETE: "
            f"{missing_total} annotated (video, frame) rows missing across {len(per_video)} "
            f"video(s) ({sample}). Manifest has {n_manifest_rows} rows for a "
            f"{npy_bytes}-byte feature file. The pre-fix encoder collapsed frame ids shared "
            "across videos. RE-ENCODE with `python -m video_encoder encode-sdd` before "
            "running video-conditioned training."
        )
