"""
TU-Clausthal ETH/UCY raw parser (Kaggle-ready).
Parses obsmat.txt (and optionally obsmat_px.txt, annotation.vsp) from the
official TU-Clausthal layout:
- ETH/seq_eth, ETH/seq_hotel
- UCY/students03, UCY/zara01, UCY/zara02

Uses inv(H) for world->pixel as validated. All paths are resolved relative
to data_root which defaults to Kaggle mounts (/kaggle/input/...) or can be
overridden via --data_root.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Mapping from our subset names to TU-Clausthal layout paths
TU_CLAUSTHAL_MAP = {
    "eth": ("ETH", "seq_eth"),
    "hotel": ("ETH", "seq_hotel"),
    "univ": ("UCY", "students03"),
    "students03": ("UCY", "students03"),
    "zara1": ("UCY", "zara01"),
    "zara01": ("UCY", "zara01"),
    "zara2": ("UCY", "zara02"),
    "zara02": ("UCY", "zara02"),
}

# Kaggle input search paths (in order of preference)
KAGGLE_SEARCH_PATHS = [
    "/kaggle/input/ethucy",
    "/kaggle/input/eth-ucy",
    "/kaggle/input/trajectory-prediction-pedestrian",
    "/kaggle/input/tu-clausthal-ethucy",
    "/kaggle/input/datasets/menhari/ethucyjaad-data-set-with-labels",
]


REQUIRED_SCENES = (
    Path("ETH/seq_eth"),
    Path("ETH/seq_hotel"),
    Path("UCY/students03"),
    Path("UCY/zara01"),
    Path("UCY/zara02"),
)


def _is_valid_root(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    return all(
        (root / rel).is_dir()
        and (root / rel / "obsmat.txt").is_file()
        and (root / rel / "H.txt").is_file()
        for rel in REQUIRED_SCENES
    )


def find_tu_clausthal_root(data_root: Optional[str] = None) -> Path:
    """Locate the strict TU-Clausthal ETH/UCY raw root.

    Accepted roots contain the full strict layout::

        ETH/seq_eth, ETH/seq_hotel, UCY/students03, UCY/zara01, UCY/zara02

    Each scene must contain ``obsmat.txt`` and ``H.txt``. No alternative
    flat/processed layouts are accepted.
    """
    candidates = []
    if data_root is not None:
        candidates.append(Path(data_root))

    for p in KAGGLE_SEARCH_PATHS:
        candidates.append(Path(p))

    # Common Kaggle clone locations.
    for p in (
        Path("/kaggle/working/tu-clausthal/datasets"),
        Path("/kaggle/working/tu-clausthal"),
        Path("/kaggle/input/tu-clausthal/Trajectory-Prediction-Pedestrian/datasets"),
    ):
        candidates.append(p)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for item in kaggle_input.iterdir():
            if item.is_dir():
                candidates.append(item)
                candidates.append(item / "datasets")
                for sub in item.iterdir():
                    if sub.is_dir():
                        candidates.append(sub)
                        candidates.append(sub / "datasets")

    for cand in candidates:
        if _is_valid_root(cand):
            return cand
        if _is_valid_root(cand / "datasets"):
            return cand / "datasets"

    checked = [str(c) for c in candidates]
    raise FileNotFoundError(
        "Could not find the strict TU-Clausthal raw ETH/UCY layout. "
        f"Required: {[str(r) for r in REQUIRED_SCENES]}. "
        f"Checked: {checked}. Pass --data_dir/--data_root pointing at the "
        "directory containing ETH/ and UCY/."
    )


def load_homography(h_path: Path) -> np.ndarray:
    """Load homography matrix from txt file."""
    H = np.loadtxt(h_path)
    if H.shape != (3, 3):
        raise ValueError(f"Invalid homography shape {H.shape} in {h_path}")
    return H


def parse_obsmat(obsmat_path: Path) -> np.ndarray:
    """
    Parse obsmat.txt from TU-Clausthal format.
    Format: [frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y]
    pos_x, pos_y in meters (world coordinates), obtained with homography.
    """
    data = np.loadtxt(obsmat_path)
    # Columns: frame, id, x, z, y, vx, vz, vy (z ignored)
    return data


def load_scene_data(data_root: Optional[str], subset: str) -> Dict:
    """Load raw scene data from the strict TU-Clausthal layout."""
    root = find_tu_clausthal_root(data_root)

    if subset not in TU_CLAUSTHAL_MAP:
        raise ValueError(
            f"Unknown subset {subset}, valid: {list(TU_CLAUSTHAL_MAP.keys())}"
        )

    group, scene_dir = TU_CLAUSTHAL_MAP[subset]
    scene_path = root / group / scene_dir
    if not scene_path.is_dir():
        raise FileNotFoundError(f"Strict scene path not found: {scene_path}")

    h_path = scene_path / "H.txt"
    if not h_path.is_file():
        raise FileNotFoundError(f"Missing raw homography: {h_path}")
    H = load_homography(h_path)
    H_inv = np.linalg.inv(H)

    obsmat_path = scene_path / "obsmat.txt"
    if not obsmat_path.is_file():
        raise FileNotFoundError(f"Missing raw obsmat: {obsmat_path}")
    traj_data = parse_obsmat(obsmat_path)

    annotation_path = scene_path / "annotation.vsp"
    if not annotation_path.is_file():
        annotation_path = None

    video_path = None
    for vname in ("video.avi", f"{scene_dir}.avi", f"{subset}.avi"):
        vpath = scene_path / vname
        if vpath.is_file():
            video_path = vpath
            break

    return {
        "root": root,
        "scene_path": scene_path,
        "H": H,
        "H_inv": H_inv,
        "traj_data": traj_data,
        "annotation_path": annotation_path,
        "video_path": video_path,
        "group": group,
        "scene_dir": scene_dir,
    }


def _interp_track(
    frames: np.ndarray, values: np.ndarray, target_frames: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a single pedestrian track onto target frames."""
    order = np.argsort(frames)
    frames = frames[order]
    values = values[order]
    if len(frames) < 2:
        return np.empty((0, 2), dtype=np.float64)
    lo, hi = frames[0], frames[-1]
    target_frames = target_frames[(target_frames >= lo) & (target_frames <= hi)]
    if target_frames.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    out = np.empty((target_frames.size, 2), dtype=np.float64)
    out[:, 0] = np.interp(target_frames, frames, values[:, 0])
    out[:, 1] = np.interp(target_frames, frames, values[:, 1])
    return out


def load_trajectories(data_root: Optional[str], subset: str) -> Dict:
    """Load benchmark-grid world trajectories from raw files only.

    ETH (``seq_eth``) uses ``obsmat.txt`` and interpolates its native 6-frame
    sampling onto the benchmark 10-frame grid. Hotel already uses the 10-frame
    grid. UCY scenes use ``annotation.vsp`` + ``H.txt`` because their
    ``obsmat.txt`` was produced with the legacy ``H-old``/``H-cam`` homography
    and is not aligned with the video/``H.txt`` frame.
    """
    scene = load_scene_data(data_root, subset)
    scene_dir = scene["scene_dir"]
    H = scene["H"]

    if scene_dir == "seq_eth":
        obs = scene["traj_data"]
        rows = []
        frames_all = np.unique(obs[:, 0])
        target = np.arange(frames_all.min(), frames_all.max() + 1, 10, dtype=np.float64)
        for ped_id in np.unique(obs[:, 1]):
            track = obs[obs[:, 1] == ped_id]
            vals = _interp_track(track[:, 0], track[:, 2:5:2], target)
            valid_target = target[
                (target >= track[:, 0].min()) & (target <= track[:, 0].max())
            ]
            if len(valid_target) != len(vals):
                raise RuntimeError(
                    f"ETH interpolation mismatch for ped {ped_id}: "
                    f"{len(valid_target)} frames vs {len(vals)} values"
                )
            for f, (x, y) in zip(valid_target, vals):
                if x == 0 and y == 0:
                    continue
                rows.append((float(f), float(ped_id), float(x), float(y)))
        traj = np.asarray(rows, dtype=np.float64)
        source = "obsmat.txt+interp10"
    elif scene_dir == "seq_hotel":
        obs = scene["traj_data"]
        # TU-Clausthal hotel obsmat frames are 1,11,21,...; benchmark uses
        # 0,10,20,... . Shift by the observed offset (exactly -1).
        offset = -1.0
        traj = obs[:, [0, 1, 2, 4]].copy()
        traj[:, 0] += offset
        source = "obsmat.txt"
    else:
        vsp_path = scene["annotation_path"]
        if vsp_path is None:
            raise FileNotFoundError(
                f"UCY scene '{subset}' requires annotation.vsp, but none was found "
                f"under {scene['scene_path']}"
            )
        traj = _parse_vsp_to_world(vsp_path, H)
        source = "annotation.vsp+H.txt"

    if traj.size == 0:
        raise ValueError(f"No raw trajectories loaded for subset '{subset}'")
    return {
        **scene,
        "traj": traj,
        "trajectory_source": source,
    }


def _parse_vsp_control_points(vsp_path: Path) -> List[np.ndarray]:
    """Parse VSP splines into lists of ``[x_px, y_px, frame]`` control points."""
    pedestrians = []
    current = []
    with open(vsp_path, "r") as f:
        for line in f:
            if "- Num of control points" in line or "- the number of splines" in line:
                if current:
                    pedestrians.append(np.asarray(current, dtype=np.float64))
                current = []
                continue
            if " - " in line:
                line = line[: line.find(" - ")]
            entries = [e for e in line.split(" ") if e]
            if len(entries) != 4:
                continue
            x, y, frame, _ = entries
            current.append([float(x), float(y), int(frame)])
    if current:
        pedestrians.append(np.asarray(current, dtype=np.float64))
    return pedestrians


def _project_pixels(H: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    hom = np.hstack([pixels, np.ones((len(pixels), 1))])
    out = (H @ hom.T).T
    return out[:, :2] / out[:, 2:3]


def _parse_vsp_to_world(vsp_path: Path, H: np.ndarray) -> np.ndarray:
    """Convert VSP splines to world-frame 10-frame trajectories.

    Mirrors the TU-Clausthal OpenTraj ``loader_crowds`` semantics: linear
    interpolation of each spline's control points, sampled every 10 frames.
    """
    peds = _parse_vsp_control_points(vsp_path)
    rows = []
    for ped_idx, ped in enumerate(peds, start=1):
        frames = ped[:, 2]
        if len(frames) < 2 or frames.max() <= frames.min():
            continue
        world = _project_pixels(H, ped[:, :2])
        target = np.arange(
            (int(frames.min()) // 10) * 10 + 10, int(frames.max()), 10, dtype=np.float64
        )
        if target.size == 0:
            continue
        for axis in range(2):
            vals = np.interp(target, frames, world[:, axis])
            if axis == 0:
                xs = vals
            else:
                ys = vals
        for f, x, y in zip(target, xs, ys):
            rows.append((float(f), float(ped_idx), float(x), float(y)))
    return np.asarray(rows, dtype=np.float64)


def load_split_ranges(
    split_file: Optional[str] = None,
) -> Dict[str, Dict[str, List[List[int]]]]:
    """Load the committed raw split frame ranges (JSON, never pickle)."""
    if split_file is None:
        here = Path(__file__).resolve().parent
        split_file = here / "eth_ucy" / "splits" / "tu_clausthal_splits.json"
    split_path = Path(split_file)
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")
    with open(split_path, "r") as f:
        return json.load(f)


def window_trajectories(
    traj: np.ndarray,
    past_frames: int,
    future_frames: int,
    frame_ranges: List[List[int]],
    min_ped: int = 1,
    skip: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window raw ``[frame,id,x,y]`` rows into ``[N,T,2]`` samples.

    Returns ``(windows, start_frames, ped_ids)``. Windows follow the standard
    SocialGAN rule: 8 observed + 12 future frames, every pedestrian present in
    all 20 frames is emitted if the frame has more than ``min_ped`` pedestrians.
    """
    seq_len = past_frames + future_frames
    if traj.ndim != 2 or traj.shape[1] < 4:
        raise ValueError(f"Expected [N,>=4] raw trajectories, got {traj.shape}")

    mask = np.zeros(len(traj), dtype=bool)
    for lo, hi in frame_ranges:
        mask |= (traj[:, 0] >= lo) & (traj[:, 0] <= hi)
    data = traj[mask]
    if len(data) == 0:
        return (
            np.empty((0, seq_len, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    frames = np.unique(data[:, 0]).tolist()
    frame_to_rows = {f: data[data[:, 0] == f] for f in frames}
    windows = []
    start_frames = []
    ped_ids = []

    num_sequences = int(np.ceil((len(frames) - seq_len + 1) / skip))
    for idx in range(0, num_sequences * skip + 1, skip):
        if idx + seq_len > len(frames):
            break
        curr_frames = frames[idx : idx + seq_len]
        curr_rows = np.concatenate([frame_to_rows[f] for f in curr_frames], axis=0)
        peds = np.unique(curr_rows[:, 1])
        frame_index = {f: i for i, f in enumerate(curr_frames)}
        accepted = []
        for ped_id in peds:
            ped_rows = curr_rows[curr_rows[:, 1] == ped_id]
            if len(ped_rows) != seq_len:
                continue
            first = frame_index.get(ped_rows[0, 0])
            last = frame_index.get(ped_rows[-1, 0])
            if first is None or last is None or last - first + 1 != seq_len:
                continue
            seq = ped_rows[np.argsort(ped_rows[:, 0]), 2:4]
            accepted.append((int(ped_id), seq))
        if len(accepted) > min_ped:
            for ped_id, seq in accepted:
                windows.append(seq)
                start_frames.append(int(curr_frames[0]))
                ped_ids.append(ped_id)

    if not windows:
        return (
            np.empty((0, seq_len, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(start_frames, dtype=np.int64),
        np.asarray(ped_ids, dtype=np.int64),
    )


def load_loso_windows(
    data_root: Optional[str],
    held_out: str,
    past_frames: int,
    future_frames: int,
    split_file: Optional[str] = None,
) -> Dict:
    """Build strict leave-one-scene-out train/test windows from raw data.

    Training uses every scene except ``held_out`` and the union of that
    scene's train+val frame ranges. Test uses only ``held_out`` and its full
    test range.
    """
    scenes = ["eth", "hotel", "univ", "zara1", "zara2"]
    if held_out not in scenes:
        raise ValueError(f"Unknown held_out scene {held_out}; valid: {scenes}")
    splits = load_split_ranges(split_file)

    def ranges_for(scene: str, mode: str) -> List[List[int]]:
        entry = splits[scene]
        if mode == "train":
            return [entry["train"], entry["val"]]
        return [entry["test"]]

    train_windows, train_frames, train_ids, train_scenes = [], [], [], []
    for scene in scenes:
        if scene == held_out:
            continue
        loaded = load_trajectories(data_root, scene)
        w, f, p = window_trajectories(
            loaded["traj"], past_frames, future_frames, ranges_for(scene, "train")
        )
        if len(w) == 0:
            raise ValueError(f"No training windows produced for raw scene '{scene}'")
        train_windows.append(w)
        train_frames.append(f)
        train_ids.append(p)
        train_scenes.extend([scene] * len(w))

    test_loaded = load_trajectories(data_root, held_out)
    test_w, test_f, test_p = window_trajectories(
        test_loaded["traj"], past_frames, future_frames, ranges_for(held_out, "test")
    )
    if len(test_w) == 0:
        raise ValueError(f"No test windows produced for raw scene '{held_out}'")

    return {
        "train": {
            "windows": np.concatenate(train_windows, axis=0),
            "start_frames": np.concatenate(train_frames, axis=0),
            "ped_ids": np.concatenate(train_ids, axis=0),
            "scenes": np.asarray(train_scenes, dtype=object),
        },
        "test": {
            "windows": test_w,
            "start_frames": test_f,
            "ped_ids": test_p,
            "scenes": np.asarray([held_out] * len(test_w), dtype=object),
        },
        "homographies": {s: load_scene_data(data_root, s)["H"] for s in scenes},
    }
