"""FastAPI dashboard server for the video_encoder module.

Run:
    pip install fastapi uvicorn scikit-learn
    VE_DATA_ROOT=./data_trajpred python -m video_encoder.api
    # or: uvicorn video_encoder.api:app --reload --port 8000

Endpoints
---------
GET  /api/scenes                          -> available canonical scenes (existence-checked)
GET  /api/scenes/{scene}/info             -> counts, frame range, stride, world bbox, split cutoff
GET  /api/scenes/{scene}/trajectories     -> all rows {frame_id, ped_id, x, y}
GET  /api/scenes/{scene}/frames           -> sorted list of available frame indices
GET  /api/scenes/{scene}/frame/{fid}.jpg  -> JPEG bytes (snapped to nearest dumped frame)
GET  /api/scenes/{scene}/frame/{fid}/agent_crops -> agent crops for that frame (JSON with base64)
GET  /api/scenes/{scene}/stats            -> density-over-time, speed-distribution, per-ped counts
GET  /api/loso/{held_out}                 -> train/val row counts per training scene, cutoffs
POST /api/features                        -> body: {npy_path}; returns 2-D PCA + indices
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

from .scenes import CANONICAL_SCENES, scene_to_folder, others
from .trajectories import load_raw_trajectories
from .splits import temporal_split
from .frames import list_available_frames, detect_stride, snap_frame_id, frame_path


DATA_ROOT = Path(os.environ.get("VE_DATA_ROOT", "./data_trajpred")).resolve()

app = FastAPI(title="Video Encoder Dashboard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- caches ----------
_traj_cache: dict[str, pd.DataFrame] = {}

def _traj(scene: str) -> pd.DataFrame:
    if scene not in _traj_cache:
        _traj_cache[scene] = load_raw_trajectories(DATA_ROOT, scene)
    return _traj_cache[scene]

def _scene_exists(scene: str) -> bool:
    try:
        load_raw_trajectories(DATA_ROOT, scene)
    except FileNotFoundError:
        return False
    return True

# Scene-specific constants (copied from dataloader_eth_ucy.py)
SCENE_RESOLUTIONS = {
    'eth':   (640, 480),
    'hotel': (720, 576),
    'univ':  (720, 576),
    'zara1': (720, 576),
    'zara2': (720, 576),
    # SDD scenes - SDD videos are typically 1280x720 (720p)
    'bookstore': (1280, 720),
    'coupa': (1280, 720),
    'deathCircle': (1280, 720),
    'gates': (1280, 720),
    'hyang': (1280, 720),
    'little': (1280, 720),
    'nexus': (1280, 720),
    'quad': (1280, 720),
}

SCENE_WORLD_BOUNDS = {
    'eth':   (-7.69, 13.89,  -1.81, 12.67),
    'hotel': (-10.31, 4.31, -2.77,  4.04),
    'univ':  (-0.46, 15.47,  -0.32, 13.89),
    'zara1': (-0.14, 15.48,  -0.37, 12.39),
    'zara2': (-0.36, 15.56,  -0.19, 13.48),
    # SDD scenes - set to make world_to_pixel approximately identity for visualization
    # Derived from: xmin=pad, xmax=width-pad, ymin=height-pad, ymax=pad (with pad=10)
    # This ensures that when trajectory data is in pixel coordinates matching the video,
    # the world_to_pixel function returns approximately the same coordinates for visualization
    'bookstore': (10, 1270, 710, 10),
    'coupa': (10, 1270, 710, 10),
    'deathCircle': (10, 1270, 710, 10),
    'gates': (10, 1270, 710, 10),
    'hyang': (10, 1270, 710, 10),
    'little': (10, 1270, 710, 10),
    'nexus': (10, 1270, 710, 10),
    'quad': (10, 1270, 710, 10),
}

def world_to_pixel(traj_pts, scene: str, target_res=(64, 64)):
    """Convert world coordinates to pixel coordinates using scene bounds.
    Mirrors ETHDataset.world_to_pixel (with 10-pixel padding).
    """
    traj_w = np.array(traj_pts, dtype=np.float64)
    orig_w, orig_h = SCENE_RESOLUTIONS[scene]
    x_min, x_max, y_min, y_max = SCENE_WORLD_BOUNDS[scene]
    pad = 10
    img_pts = np.zeros_like(traj_w)
    img_pts[:, 0] = (traj_w[:, 0] - x_min) / (x_max - x_min) * (orig_w - 2*pad) + pad
    img_pts[:, 1] = (traj_w[:, 1] - y_min) / (y_max - y_min) * (orig_h - 2*pad) + pad
    # Y-axis is inverted relative to image coordinates in all ETH-UCY scenes
    img_pts[:, 1] = orig_h - img_pts[:, 1]
    img_pts[:, 0] = img_pts[:, 0] / orig_w * target_res[0]
    img_pts[:, 1] = img_pts[:, 1] / orig_h * target_res[1]
    return img_pts

# ---------- routes ----------
@app.get("/api/health")
def health():
    return {"ok": True, "data_root": str(DATA_ROOT), "exists": DATA_ROOT.exists()}


@app.get("/api/scenes")
def scenes():
    out = []
    for s in CANONICAL_SCENES:
        try:
            df = _traj(s)
            try:
                frames = list_available_frames(str(DATA_ROOT), s)
                has_frames = True
                n_frames = len(frames)
            except FileNotFoundError:
                has_frames = False
                n_frames = 0
            out.append({
                "scene": s,
                "folder": scene_to_folder(s),
                "rows": int(len(df)),
                "pedestrians": int(df["ped_id"].nunique()),
                "frames_on_disk": n_frames,
                "has_frames": has_frames,
            })
        except FileNotFoundError:
            out.append({"scene": s, "folder": scene_to_folder(s), "missing": True})
    return {"scenes": out, "data_root": str(DATA_ROOT)}


@app.get("/api/scenes/{scene}/info")
def scene_info(scene: str):
    if scene not in CANONICAL_SCENES:
        raise HTTPException(404, "unknown scene")
    df = _traj(scene)
    split = temporal_split(df)
    try:
        frames = list_available_frames(str(DATA_ROOT), scene)
        stride = detect_stride(frames)
        frame_min, frame_max = int(frames[0]), int(frames[-1])
        has_frames = True
    except FileNotFoundError:
        frames = ()
        stride = 10
        frame_min = int(df["frame_id"].min())
        frame_max = int(df["frame_id"].max())
        has_frames = False
    return {
        "scene": scene,
        "rows": int(len(df)),
        "pedestrians": int(df["ped_id"].nunique()),
        "frame_min": frame_min,
        "frame_max": frame_max,
        "traj_frame_min": int(df["frame_id"].min()),
        "traj_frame_max": int(df["frame_id"].max()),
        "stride": int(stride),
        "world_bbox": {
            "xmin": float(df["x"].min()),
            "xmax": float(df["x"].max()),
            "ymin": float(df["y"].min()),
            "ymax": float(df["y"].max()),
        },
        "split_cutoff_frame_id": int(split.cutoff_frame_id),
        "train_rows": int(len(split.train)),
        "val_rows": int(len(split.val)),
        "has_frames": has_frames,
    }


@app.get("/api/scenes/{scene}/trajectories")
def trajectories(scene: str):
    if scene not in CANONICAL_SCENES:
        raise HTTPException(404, "unknown scene")
    df = _traj(scene)[["frame_id", "ped_id", "x", "y"]]
    return JSONResponse({
        "frame_ids": df["frame_id"].astype(int).tolist(),
        "ped_ids": df["ped_id"].astype(int).tolist(),
        "xs": df["x"].astype(float).round(4).tolist(),
        "ys": df["y"].astype(float).round(4).tolist(),
    })


@app.get("/api/scenes/{scene}/frames")
def frames(scene: str):
    if scene not in CANONICAL_SCENES:
        raise HTTPException(404, "unknown scene")
    try:
        fs = list_available_frames(str(DATA_ROOT), scene)
    except FileNotFoundError:
        return {"frames": [], "stride": 10}
    return {"frames": list(fs), "stride": detect_stride(fs)}


@app.get("/api/scenes/{scene}/frame/{fid}.jpg")
def frame(scene: str, fid: int):
    if scene not in CANONICAL_SCENES:
        raise HTTPException(404, "unknown scene")
    try:
        fs = list_available_frames(str(DATA_ROOT), scene)
        snapped = snap_frame_id(fid, fs, policy="nearest")
        p = frame_path(DATA_ROOT, scene, snapped)
        if not p.exists():
            raise FileNotFoundError(p)
        return FileResponse(str(p), media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(404, f"frame {fid} not available")


@app.get("/api/scenes/{scene}/frame/{fid}/agent_crops")
def agent_crops(
    scene: str,
    fid: int,
    size: int = Query(64, gt=0, le=256),
):
    """Return agent crops for the given scene and frame ID as base64 JPEGs."""
    if scene not in CANONICAL_SCENES:
        raise HTTPException(404, "unknown scene")
    # Snap frame ID to nearest available frame
    try:
        available_frames = list_available_frames(str(DATA_ROOT), scene)
    except FileNotFoundError:
        raise HTTPException(404, f"No frames found for scene {scene}")
    snapped = snap_frame_id(fid, available_frames, policy="nearest")
    if snapped is None:
        raise HTTPException(404, f"Frame {fid} not available for scene {scene}")
    # Load trajectories
    df = _traj(scene)
    # Filter rows for the snapped frame
    rows = df[df["frame_id"] == snapped]
    if rows.empty:
        # No agents at this frame
        return JSONResponse({"frame_id": snapped, "crops": []})
    # Load the full frame image
    img_path = frame_path(DATA_ROOT, scene, snapped)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(500, f"Failed to load frame image: {e}")
    width, height = img.size
    crops_b64 = []
    for _, row in rows.iterrows():
        world_xy = np.array([[float(row["x"]), float(row["y"])]])
        pixel_xy = world_to_pixel(world_xy, scene, target_res=(size, size))[0]
        px, py = int(round(pixel_xy[0])), int(round(pixel_xy[1]))
        # Calculate crop boundaries
        x1 = px - size // 2
        y1 = py - size // 2
        x2 = x1 + size
        y2 = y1 + size
        # Compute padding needed
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - width)
        pad_bottom = max(0, y2 - height)
        # Adjust crop coordinates to image bounds
        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x2_clip = min(width, x2)
        y2_clip = min(height, y2)
        # Extract region
        if x2_clip > x1_clip and y2_clip > y1_clip:
            crop_img = img.crop((x1_clip, y1_clip, x2_clip, y2_clip))
        else:
            # No overlap -> black image
            crop_img = Image.new("RGB", (size, size), (0, 0, 0))
        # Apply padding if needed
        if pad_left or pad_top or pad_right or pad_bottom:
            padded = Image.new("RGB", (size, size), (0, 0, 0))
            padded.paste(crop_img, (pad_left, pad_top))
            crop_img = padded
        # Encode to JPEG base64
        buffer = io.BytesIO()
        crop_img.save(buffer, format="JPEG")
        jpeg_bytes = buffer.getvalue()
        import base64
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        crops_b64.append({
            "ped_id": int(row["ped_id"]),
            "crop_base64": b64,
        })
    return JSONResponse({
        "scene": scene,
        "frame_id": snapped,
        "requested_frame_id": fid,
        "size": size,
        "crops": crops_b64,
    })


@app.get("/api/scenes/{scene}/stats")
def stats(scene: str):
    if scene not in CANONICAL_SCENES:
        raise HTTPException(404, "unknown scene")
    df = _traj(scene).sort_values(["ped_id", "frame_id"]).reset_index(drop=True)

    # density: peds per frame
    density = df.groupby("frame_id").size().reset_index(name="count")
    # downsample to ~200 points for the chart
    if len(density) > 200:
        density = density.iloc[:: max(1, len(density) // 200)]

    # per-pedestrian speed: euclidean distance / frame-time (~0.4s @ stride 10)
    g = df.groupby("ped_id")
    dx = g["x"].diff()
    dy = g["y"].diff()
    dt_frames = g["frame_id"].diff()
    dt_seconds = dt_frames * (0.4 / 10.0)  # ETH/UCY 25 fps -> stride 10 = 0.4 s
    speeds = np.sqrt(dx.pow(2) + dy.pow(2)) / dt_seconds.replace(0, np.nan)
    speeds = speeds.replace([np.inf, -np.inf], np.nan).dropna()
    speeds = speeds[(speeds >= 0) & (speeds < 10)]  # plausible peds < 10 m/s

    # speed histogram (30 bins, 0–4 m/s typical)
    hist, edges = np.histogram(speeds.values, bins=30, range=(0, 4))

    return {
        "density": {
            "frame_ids": density["frame_id"].astype(int).tolist(),
            "counts": density["count"].astype(int).tolist(),
            "mean": float(density["count"].mean()),
            "max": int(density["count"].max()),
        },
        "speeds": {
            "bins": edges.round(3).tolist(),
            "counts": hist.astype(int).tolist(),
            "mean": float(speeds.mean()) if len(speeds) else 0.0,
            "median": float(speeds.median()) if len(speeds) else 0.0,
        },
        "per_ped": {
            "count": int(df["ped_id"].nunique()),
            "mean_track_length": float(df.groupby("ped_id").size().mean()),
        },
    }


@app.get("/api/loso/{held_out}")
def loso(held_out: str):
    if held_out not in CANONICAL_SCENES:
        raise HTTPException(404, "unknown held-out scene")
    train_scenes = others(held_out)
    train_info = []
    total_train, total_val = 0, 0
    for s in train_scenes:
        try:
            df = _traj(s)
        except FileNotFoundError:
            continue
        sp = temporal_split(df)
        train_info.append({
            "scene": s,
            "rows": int(len(df)),
            "train_rows": int(len(sp.train)),
            "val_rows": int(len(sp.val)),
            "cutoff": int(sp.cutoff_frame_id),
            "frame_min": int(df["frame_id"].min()),
            "frame_max": int(df["frame_id"].max()),
        })
        total_train += len(sp.train)
        total_val += len(sp.val)
    try:
        held = _traj(held_out)
        test = {"scene": held_out, "rows": int(len(held))}
    except FileNotFoundError:
        test = {"scene": held_out, "rows": 0, "missing": True}
    return {
        "held_out": held_out,
        "train_scenes": list(train_scenes),
        "train": train_info,
        "test": test,
        "totals": {"train": total_train, "val": total_val},
    }


class FeatureReq(BaseModel):
    npy_path: str
    max_points: int = 3000


@app.post("/api/features")
def features(req: FeatureReq):
    p = Path(req.npy_path)
    if not p.is_absolute():
        p = (DATA_ROOT.parent / p).resolve()
    if not p.exists():
        raise HTTPException(404, f"file not found: {p}")
    arr = np.load(p)
    if arr.ndim != 2:
        raise HTTPException(400, f"expected 2-D (N, D) array, got shape {arr.shape}")
    n = arr.shape[0]
    idx = np.arange(n)
    if n > req.max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, req.max_points, replace=False))
        arr = arr[idx]
    # cheap PCA via SVD on centered data (no sklearn required)
    x = arr - arr.mean(axis=0, keepdims=True)
    # truncate to first 2 components
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    proj = (u[:, :2] * s[:2])
    explained = (s[:2] ** 2) / (s ** 2).sum()
    return {
        "n": int(n),
        "n_used": int(arr.shape[0]),
        "dim": int(arr.shape[1]),
        "indices": idx.astype(int).tolist(),
        "x": proj[:, 0].astype(float).round(4).tolist(),
        "y": proj[:, 1].astype(float).round(4).tolist(),
        "explained_variance": [float(e) for e in explained],
    }


def main():
    import uvicorn
    uvicorn.run("video_encoder.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()