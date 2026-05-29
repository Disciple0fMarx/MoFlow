"""Parse ETH/UCY raw trajectory .txt files.

Format (whitespace-separated): frame_id  ped_id  x  y
Frame ids are integers; coordinates are floats in world space.

Axis convention
---------------
The BIWI scenes (`eth`, `hotel`) store world coordinates with axes that are
rotated 90° relative to the camera image plane. To get a world frame whose
(x, y) axes line up with the (horizontal, vertical) of the recorded video --
which is what every downstream consumer (visualizer, encoder concatenation,
agent-centric crops) implicitly assumes -- we apply a per-scene 90° rotation:

    (x', y') = (y, -x)   for `eth` and `hotel`

This is the same correction applied by Social-GAN / Trajectron++ / Y-Net
preprocessing pipelines. UCY scenes (`univ`, `zara1`, `zara2`) are already
image-aligned and pass through unchanged.

Set the env var `VE_NO_AXIS_FIX=1` to disable the rotation if you ever need
the raw on-disk values (e.g. for round-trip checks against the .txt files).
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from .scenes import scene_to_raw_txt

# scenes that need 90° CW rotation: (x, y) -> (y, -x)
_AXIS_ROTATE_SCENES = {"eth", "hotel"}


def load_raw_trajectories(data_root: str | Path, scene: str) -> pd.DataFrame:
    """Load a scene's full trajectories from data_trajpred/raw/all_data/<name>.txt."""
    data_root = Path(data_root)
    txt = data_root / "raw" / "all_data" / f"{scene_to_raw_txt(scene)}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Missing raw trajectory file: {txt}")

    df = pd.read_csv(
        txt,
        sep=r"\s+",
        header=None,
        names=["frame_id", "ped_id", "x", "y"],
        comment="#",
        engine="python",
    )
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
    df["ped_id"] = pd.to_numeric(df["ped_id"], errors="coerce").astype("Int64")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().astype({"frame_id": "int64", "ped_id": "int64"})

    if scene in _AXIS_ROTATE_SCENES and os.environ.get("VE_NO_AXIS_FIX") != "1":
        # 90° CW rotation so world axes match the camera image plane.
        new_x = df["y"].copy()
        new_y = -df["x"].copy()
        df["x"] = new_x
        df["y"] = new_y

    df["scene"] = scene
    return df.reset_index(drop=True)
