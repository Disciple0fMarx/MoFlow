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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

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
