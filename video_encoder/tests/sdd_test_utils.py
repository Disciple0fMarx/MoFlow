"""Synthetic SDD fixtures for offline testing (no video decode required).

Creates a fake ``annotations/<scene>/<video_id>/annotations.txt`` tree whose
rows match the schema consumed by
:func:`video_encoder.global_video_encoder_sdd.build_sdd_frame_index` and the
window index builder (whitespace columns: track_id, xmin, ymin, xmax, ymax,
frame_id).

Key for the perspective-scope audit: SDD frame ids are 0-based *per video*,
so two videos in the same scene are constructed to share frame-id ranges —
exactly the precondition of the frame-collapse bug.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from video_encoder.sdd_adapter import SDD_SCENES


def _ann_line(tid: int, x: float, y: float, frame: int) -> str:
    # full bbox around point (x, y); center recovered by the parsers
    return f"{tid} {x - 1.0:.3f} {y - 1.0:.3f} {x + 1.0:.3f} {y + 1.0:.3f} {frame}\n"


def write_synthetic_sdd(
    root: Path,
    n_scenes: int = 2,
    videos_per_scene: int = 2,
    n_tracks_per_video: int = 1,
    frames_per_video: int = 60,
    sdd_scenes: bool = False,
    rng_seed: int = 0,
) -> Path:
    """Write a synthetic SDD annotations tree and return ``root``.

    ``sdd_scenes=True`` names the scenes with the real canonical ids so the
    LOSO checks and ``SDD_SCENES``-bound code paths run verbatim.
    """
    rng = np.random.default_rng(rng_seed)
    scenes = list(SDD_SCENES[:n_scenes]) if sdd_scenes else [f"scene{i}" for i in range(n_scenes)]

    for scene in scenes:
        for vi in range(videos_per_scene):
            vid = f"video{vi}"
            ann_dir = root / "annotations" / scene / vid
            ann_dir.mkdir(parents=True, exist_ok=True)
            with (ann_dir / "annotations.txt").open("w") as fp:
                x0, y0 = rng.uniform(20, 200, size=2)
                for ti in range(n_tracks_per_video):
                    vx, vy = rng.uniform(-0.5, 0.5, size=2)
                    x, y = x0, y0
                    for f in range(frames_per_video):
                        fp.write(_ann_line(ti, x, y, f))
                        x += vx
                        y += vy
    return root


def write_synthetic_video_cache(
    features_root: Path,
    scene: str,
    videos: tuple[str, ...],
    frames_per_video: int,
    dim: int = 512,
    seed: int = 1,
) -> Path:
    """Write a video-scoped feature cache (fixed / the 'new' format).

    The manifest lists every ``(video_id, frame_id)`` row, and the ``.npy``
    holds one feature vector per row. Distinct videos get **distinct** feature
    values so tests can assert *which* video's pixels a lookup returned.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    features_root.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int]] = []
    feats: list[np.ndarray] = []
    for vi, vid in enumerate(videos):
        for f in range(frames_per_video):
            rows.append((vid, f))
            base = rng.uniform(-10, 10, size=dim).astype(np.float32)
            feats.append(base + vi * 1000.0)  # video-index offset: unambiguous fingerprints
    features = np.stack(feats).astype(np.float32)
    np.save(features_root / f"{scene}.npy", features)
    pd.DataFrame(
        {
            "scene": scene,
            "video_id": [r[0] for r in rows],
            "frame_id": [r[1] for r in rows],
            "row_idx": np.arange(len(rows), dtype=np.int64),
        }
    ).to_parquet(features_root / f"{scene}.manifest.parquet", index=False)
    return features_root