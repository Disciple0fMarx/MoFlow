"""Standalone audit: homography conventions, preprocessing consistency, and motion lift.
Zero-modification of core repo modules. Uses synthetic fixtures for tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def homogeneous_transform(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pts_xy = pts[:, :2]
    h = np.hstack([pts_xy, np.ones((len(pts_xy), 1))])
    c = H @ h.T
    return (c / c[2:3]).T[:, :2]


def apply_scene_convention(img_pts: np.ndarray, scene: str, w: int, h: int) -> np.ndarray:
    out = img_pts.copy()
    if scene in ("univ", "zara1", "zara2"):
        out[:, 0] = out[:, 0] + w / 2.0
        out[:, 1] = h / 2.0 - out[:, 1]
    return out


def world_to_pixel_audit(world_pts, H, scene, orig_res=(720, 576), target_res=None, use_inv=True):
    M = np.linalg.inv(H) if use_inv else H
    img_pts = homogeneous_transform(M, np.asarray(world_pts))
    img_pts = apply_scene_convention(img_pts, scene, orig_res[0], orig_res[1])
    if target_res is not None and tuple(orig_res) != tuple(target_res):
        s = np.array([target_res[0] / orig_res[0], target_res[1] / orig_res[1]])
        img_pts = img_pts * s
    return img_pts


def motion_lift_score(frames_bgr: List[np.ndarray], traj_pixels: np.ndarray, radius: int = 8) -> Dict[str, float]:
    if len(frames_bgr) < 2:
        return {"hit_rate": 0.0, "bg_rate": 0.0, "lift": 0.0}
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames_bgr], axis=0)
    med = np.median(stack, axis=0)
    diffs = np.abs(stack - med[None, ...]).mean(axis=0)
    mask = diffs > np.percentile(diffs, 70)
    H, W = mask.shape
    hit, total = 0, 0
    for pt in np.asarray(traj_pixels).reshape(-1, 2):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < W and 0 <= y < H:
            total += 1
            x0, x1 = max(0, x - radius), min(W, x + radius + 1)
            y0, y1 = max(0, y - radius), min(H, y + radius + 1)
            if mask[y0:y1, x0:x1].mean() > 0.1:
                hit += 1
    hit_rate = hit / max(total, 1)
    bg_rate = float(mask.mean())
    lift = hit_rate - bg_rate
    return {"hit_rate": float(hit_rate), "bg_rate": float(bg_rate), "lift": float(lift)}


def grid_search_projection(world_pts, pixel_pts_gt, H, scene, orig_res, target_res=None):
    candidates = []
    for use_inv in [True, False]:
        for flip_y in [False, True]:
            for swap_xy in [False, True]:
                pts = np.asarray(world_pts)
                if swap_xy:
                    pts = pts[:, [1, 0]]
                proj = homogeneous_transform(np.linalg.inv(H) if use_inv else H, pts)
                proj = apply_scene_convention(proj, scene, orig_res[0], orig_res[1])
                if flip_y and not (scene in ("univ", "zara1", "zara2") and use_inv):
                    proj[:, 1] = orig_res[1] - proj[:, 1]
                if target_res and tuple(orig_res) != tuple(target_res):
                    s = np.array([target_res[0] / orig_res[0], target_res[1] / orig_res[1]])
                    proj = proj * s
                err = np.linalg.norm(proj - np.asarray(pixel_pts_gt), axis=-1)
                candidates.append({
                    "use_inv": use_inv, "flip_y": flip_y, "swap_xy": swap_xy,
                    "mean_err_px": float(err.mean()), "max_err_px": float(err.max()),
                })
    candidates.sort(key=lambda x: x["mean_err_px"])
    return candidates


def synthetic_homography():
    # simple affine-like H for tests
    H = np.array([[1.2, 0.1, 10.0], [0.05, -1.1, 400.0], [0.0001, 0.0002, 1.0]], dtype=np.float64)
    return H


def run_audit(args=None):
    H = synthetic_homography()
    world = np.array([[0, 0], [1, 2], [3, 4], [5, 1]], dtype=np.float64)
    pix = world_to_pixel_audit(world, H, "eth", orig_res=(640, 480), target_res=(640, 480), use_inv=True)
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    for f in frames:
        f[200:220, 300:320] = 255
    lift = motion_lift_score(frames, np.array([[310, 210]]))
    grid = grid_search_projection(world, pix, H, "eth", (640, 480), (640, 480))
    report = {
        "preprocessing_audit": {
            "note": "Standalone audit: verify training uses consistent H/inv(H) convention per scene.",
            "best_convention": grid[0],
        },
        "visualization_grid_search": grid[:5],
        "motion_lift": lift,
    }
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    report = run_audit(args)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
