#!/usr/bin/env python3
"""Per-scene scene-complexity statistics for the Stanford Drone Dataset.

Computes, for each SDD scene, quantitative statistics used in the progress
report (report/bilan.tex, Section "Interpretation of Results") to explain the
per-scene behaviour of the video-conditioned MoFlow Teacher:

  1. Pedestrian density
       n_tracks          number of distinct pedestrian tracks in the scene
       mean_active       mean number of active pedestrians per frame
       max_active        maximum number of active pedestrians in a single frame
       density_px2       mean_active / scene area (pixels^2); uses
                         --frame-w/--frame-h when provided, otherwise the
                         bounding box of all annotations in the scene

  2. Interactions (social complexity), for each threshold D
       pairs_mean_t<D>   mean number of simultaneously-visible pedestrian pairs
                         whose bbox centres are closer than D pixels
       agents_mean_t<D>  mean number of pedestrians involved in >= 1 such pair
       frac_frames_t<D>  fraction of frames containing at least one such pair
       frac_pairs_t<D>   fraction of all simultaneously-visible pairs that are
                         closer than D pixels at least once

  3. Obstacle proxy (the static scene is not annotated in SDD text files)
       non_ped_tracks        distinct non-pedestrian tracks (biker, skater,
                             cart, car, bus, ...)
       non_ped_mean_active   mean active non-pedestrian agents per frame
       non_ped_classes       class -> distinct track count breakdown
       coverage              mean fraction of the scene footprint covered by
                             pedestrian bounding boxes

  4. Motion (informational)
       motion_mean / motion_std   mean / std of per-frame per-agent step size
       motion_frac_still          fraction of steps < 1 px

Only the standard library and numpy are required. The annotation files follow
the 10-column SDD layout documented in data/agent_crop_sdd.py
(track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label).

Usage (run on the lab machine, where the raw SDD dataset lives)::

    python3 sdd_scene_complexity.py --data-root ~/Desktop/Datasets/SDD \\
        --out scene_complexity.csv --json scene_complexity.json --thresholds 25 50
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_SCENES = [
    "bookstore",
    "coupa",
    "deathCircle",
    "gates",
    "hyang",
    "little",
    "nexus",
    "quad",
]


def _column_indices():
    """Return indices of (track, xmin, ymin, xmax, ymax, frame) in the lines.

    Prefer the constants used by the MoFlow codebase when it is importable;
    fall back to the documented 10-column order otherwise.
    """
    try:
        from video_encoder.sdd_adapter import (  # type: ignore
            COL_FRAME,
            COL_TRACK_ID,
            COL_XMAX,
            COL_XMIN,
            COL_YMAX,
            COL_YMIN,
        )

        return COL_TRACK_ID, COL_XMIN, COL_YMIN, COL_XMAX, COL_YMAX, COL_FRAME
    except Exception:
        return 0, 1, 2, 3, 4, 5


COL_TRACK_ID, COL_XMIN, COL_YMIN, COL_XMAX, COL_YMAX, COL_FRAME = _column_indices()


def iter_annotation_files(data_root: Path, scene: str):
    """Yield `annotations.txt` paths for one scene.

    Tolerates both ``<root>/annotations/<scene>/<video>/annotations.txt`` and
    ``<root>/<scene>/annotations/<video>/annotations.txt`` layouts.
    """
    seen = set()
    for base in (data_root / "annotations" / scene, data_root / scene):
        if not base.is_dir():
            continue
        for ann in sorted(base.rglob("annotations.txt")):
            if ann not in seen:
                seen.add(ann)
                yield ann


def parse_annotations(path: Path):
    """Parse one ``annotations.txt`` into numpy arrays.

    Returns ``(frames, track_id, cx, cy, w, h, labels)`` or ``None`` if empty.
    Rows that cannot be interpreted are skipped.
    """
    frames, track_ids, cx, cy, w, h, labels = [], [], [], [], [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                track_id = int(float(parts[COL_TRACK_ID]))
                xmin = float(parts[COL_XMIN])
                ymin = float(parts[COL_YMIN])
                xmax = float(parts[COL_XMAX])
                ymax = float(parts[COL_YMAX])
                frame = int(float(parts[COL_FRAME]))
            except (ValueError, IndexError):
                continue
            if not (xmax > xmin and ymax > ymin):
                continue
            label = parts[9].strip("\"'") if len(parts) > 9 else ""
            frames.append(frame)
            track_ids.append(track_id)
            cx.append(0.5 * (xmin + xmax))
            cy.append(0.5 * (ymin + ymax))
            w.append(xmax - xmin)
            h.append(ymax - ymin)
            labels.append(label)
    if not frames:
        return None
    return (
        np.asarray(frames, dtype=np.int64),
        np.asarray(track_ids, dtype=np.int64),
        np.asarray(cx, dtype=np.float64),
        np.asarray(cy, dtype=np.float64),
        np.asarray(w, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
        labels,
    )


def is_pedestrian(label: str) -> bool:
    label = label.lower()
    return "pedestrian" in label or label in ("", "person", "ped", "walking")


def scene_statistics(data_root: Path, scene: str, thresholds, frame_w, frame_h):
    """Aggregate complexity statistics over all videos of one scene."""
    per_frame_ped = {}      # frame -> [(track, cx, cy)]
    per_frame_nonped = {}   # frame -> [(track, cx, cy)]
    ped_tracks = set()
    non_ped = defaultdict(set)  # class label -> distinct track ids
    bbox_areas_ped = []
    wx, wy = [], []
    n_videos = 0

    for ann in iter_annotation_files(data_root, scene):
        parsed = parse_annotations(ann)
        if parsed is None:
            continue
        frames, track_ids, cx, cy, w, h, labels = parsed
        n_videos += 1
        for i, tid in enumerate(track_ids):
            if is_pedestrian(labels[i]):
                ped_tracks.add(tid)
                bbox_areas_ped.append(w[i] * h[i])
                per_frame_ped.setdefault(int(frames[i]), []).append((tid, cx[i], cy[i]))
            else:
                cls = labels[i].lower() or "unknown"
                non_ped[cls].add(tid)
                per_frame_nonped.setdefault(int(frames[i]), []).append((tid, cx[i], cy[i]))
        wx.extend(cx.tolist())
        wy.extend(cy.tolist())

    if not per_frame_ped:
        return None

    wx = np.asarray(wx)
    wy = np.asarray(wy)
    scene_w = float(wx.max() - wx.min())
    scene_h = float(wy.max() - wy.min())
    scene_area = (
        float(frame_w * frame_h)
        if (frame_w and frame_h)
        else max(scene_w * scene_h, 1.0)
    )
    mean_bbox_area = float(np.mean(bbox_areas_ped)) if bbox_areas_ped else 0.0

    active = np.asarray([len(v) for v in per_frame_ped.values()], dtype=float)
    n_frames = active.size
    mean_active = float(active.mean())
    max_active = int(active.max())

    # Per-frame pair statistics
    pair_acc = {t: 0.0 for t in thresholds}
    ag_acc = {t: 0.0 for t in thresholds}
    frame_acc = {t: 0 for t in thresholds}
    close_distinct = {t: set() for t in thresholds}
    total_visible_pairs = 0

    for agents in per_frame_ped.values():
        n = len(agents)
        if n < 2:
            continue
        pos = np.asarray([[a[1], a[2]] for a in agents])
        ids = [a[0] for a in agents]
        total_visible_pairs += n * (n - 1) / 2
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        tri = np.triu(np.ones((n, n), dtype=bool), k=1)
        for t in thresholds:
            close = (d < t) & tri
            npairs = int(close.sum())
            pair_acc[t] += npairs
            ag_acc[t] += int(np.unique(np.nonzero(close)[0]).size)
            if npairs:
                frame_acc[t] += 1
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if close[i, j]:
                            close_distinct[t].add(tuple(sorted((ids[i], ids[j]))))

    clusters = {
        t: {
            "pairs_mean": pair_acc[t] / n_frames,
            "agents_mean": ag_acc[t] / n_frames,
            "frac_frames": frame_acc[t] / n_frames,
            "frac_pairs": (
                len(close_distinct[t]) / total_visible_pairs
                if total_visible_pairs > 0
                else 0.0
            ),
        }
        for t in thresholds
    }

    non_ped_active = (
        np.asarray([len(v) for v in per_frame_nonped.values()], dtype=float)
        if per_frame_nonped
        else np.zeros(0)
    )

    # Motion: step size between consecutive observations of the same track
    steps = []
    for ann in iter_annotation_files(data_root, scene):
        parsed = parse_annotations(ann)
        if parsed is None:
            continue
        frames, track_ids, cx, cy, _, _, _ = parsed
        order = np.argsort(frames, kind="stable")
        frames, track_ids, cx, cy = frames[order], track_ids[order], cx[order], cy[order]
        for tid in np.unique(track_ids):
            m = track_ids == tid
            f, x, y = frames[m], cx[m], cy[m]
            if f.size < 2:
                continue
            steps.extend(np.hypot(np.diff(x), np.diff(y)).tolist())
    if steps:
        all_steps = np.asarray(steps)
        motion_mean = float(all_steps.mean())
        motion_std = float(all_steps.std())
        motion_frac_still = float((all_steps < 1.0).mean())
    else:
        motion_mean = motion_std = motion_frac_still = float("nan")

    row = {
        "scene": scene,
        "n_videos": n_videos,
        "n_frames": n_frames,
        "n_ped_tracks": len(ped_tracks),
        "ped_mean_active": round(mean_active, 3),
        "ped_max_active": max_active,
        "ped_density_px2": round(mean_active / scene_area * 1e3, 4),
        "scene_w_px": round(scene_w, 1),
        "scene_h_px": round(scene_h, 1),
        "scene_area_px2": round(scene_area, 1),
        "coverage": round(mean_bbox_area * mean_active / scene_area, 4),
        "non_ped_tracks": sum(len(v) for v in non_ped.values()),
        "non_ped_mean_active": (
            round(float(non_ped_active.mean()), 3) if non_ped_active.size else 0.0
        ),
        "non_ped_classes": dict(sorted((k, len(v)) for k, v in non_ped.items())),
        "motion_mean": round(motion_mean, 3),
        "motion_std": round(motion_std, 3),
        "motion_frac_still": round(motion_frac_still, 3),
    }
    return row, clusters


def to_flat(row, clusters, thresholds):
    """Flatten the stats dict into a CSV-friendly row."""
    flat = dict(row)
    for t in thresholds:
        c = clusters[t]
        flat[f"pairs_mean_t{t}"] = round(c["pairs_mean"], 3)
        flat[f"agents_mean_t{t}"] = round(c["agents_mean"], 3)
        flat[f"frac_frames_t{t}"] = round(c["frac_frames"], 4)
        flat[f"frac_pairs_t{t}"] = round(c["frac_pairs"], 4)
    flat.pop("non_ped_classes", None)
    flat.pop("interactions", None)
    return flat


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(Path.home() / "Desktop" / "Datasets" / "SDD"))
    p.add_argument("--scenes", nargs="*", default=None, help="default: all eight SDD scenes")
    p.add_argument("--thresholds", nargs="+", type=float, default=[25.0, 50.0])
    p.add_argument("--frame-w", type=float, default=0.0,
                   help="video width in px (for density normalization)")
    p.add_argument("--frame-h", type=float, default=0.0,
                   help="video height in px (for density normalization)")
    p.add_argument("--out", default="scene_complexity.csv")
    p.add_argument("--json", default=None, help="also write a JSON dump")
    args = p.parse_args(argv)

    root = Path(args.data_root).expanduser()
    scenes = args.scenes or DEFAULT_SCENES
    thresholds = sorted(args.thresholds)

    rows = []
    flat_rows = []
    for scene in scenes:
        result = scene_statistics(root, scene, thresholds, args.frame_w, args.frame_h)
        print(f"scene={scene:12s}  missing/empty: {result is None}")
        if result is None:
            continue
        row, clusters = result
        row = dict(row)
        row["interactions"] = {
            f"t{t}": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in clusters[t].items()}
            for t in thresholds
        }
        rows.append(row)
        flat_rows.append(to_flat(row, clusters, thresholds))

    if not flat_rows:
        sys.stdout.write("No annotations found under %s\n" % root)
        return 1

    order = sorted(flat_rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=order)
        w.writeheader()
        w.writerows(flat_rows)
    print(f"Wrote {args.out} ({len(flat_rows)} scenes).")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"Wrote {args.json}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())