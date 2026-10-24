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

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Mapping from our subset names to TU-Clausthal layout paths
TU_CLAUSTHAL_MAP = {
    'eth': ('ETH', 'seq_eth'),
    'hotel': ('ETH', 'seq_hotel'),
    'univ': ('UCY', 'students03'),
    'students03': ('UCY', 'students03'),
    'zara1': ('UCY', 'zara01'),
    'zara01': ('UCY', 'zara01'),
    'zara2': ('UCY', 'zara02'),
    'zara02': ('UCY', 'zara02'),
}

# Kaggle input search paths (in order of preference)
KAGGLE_SEARCH_PATHS = [
    '/kaggle/input/ethucy',
    '/kaggle/input/eth-ucy',
    '/kaggle/input/trajectory-prediction-pedestrian',
    '/kaggle/input/tu-clausthal-ethucy',
    '/kaggle/input/datasets/menhari/ethucyjaad-data-set-with-labels',
]


def find_tu_clausthal_root(data_root: Optional[str] = None) -> Path:
    """Find TU-Clausthal dataset root in Kaggle or provided path."""
    candidates = []
    if data_root is not None:
        candidates.append(Path(data_root))
    
    # Check common Kaggle input locations
    for p in KAGGLE_SEARCH_PATHS:
        candidates.append(Path(p))
    
    # Also check if datasets are directly under /kaggle/input
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists():
        for item in kaggle_input.iterdir():
            if item.is_dir():
                candidates.append(item)
                # Check subdirectories
                for sub in item.iterdir():
                    if sub.is_dir():
                        candidates.append(sub)
    
    for cand in candidates:
        if not cand.exists():
            continue
        # Check if it has ETH/seq_eth or datasets/ETH structure
        if (cand / 'ETH' / 'seq_eth').exists() or (cand / 'datasets' / 'ETH' / 'seq_eth').exists():
            if (cand / 'datasets' / 'ETH').exists():
                return cand / 'datasets'
            return cand
        if (cand / 'seq_eth').exists() and (cand / 'seq_hotel').exists():
            # Maybe ETH and UCY at top level
            return cand
    
    # Fallback to first candidate if exists
    for cand in candidates:
        if cand.exists():
            return cand
    
    raise FileNotFoundError(
        f"Could not find TU-Clausthal dataset. Checked: {[str(c) for c in candidates]}. "
        "Ensure dataset is attached in Kaggle input."
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
    """Load all raw data for a scene from TU-Clausthal layout."""
    root = find_tu_clausthal_root(data_root)
    
    if subset not in TU_CLAUSTHAL_MAP:
        raise ValueError(f"Unknown subset {subset}, valid: {list(TU_CLAUSTHAL_MAP.keys())}")
    
    group, scene_dir = TU_CLAUSTHAL_MAP[subset]
    scene_path = root / group / scene_dir
    
    if not scene_path.exists():
        # Try alternative layout
        scene_path = root / scene_dir
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene path not found: {scene_path}")
    
    # Load homography
    h_path = scene_path / 'H.txt'
    if not h_path.exists():
        h_path = scene_path / 'H-old.txt'
    H = load_homography(h_path)
    H_inv = np.linalg.inv(H)
    
    # Load trajectories
    obsmat_path = scene_path / 'obsmat.txt'
    traj_data = parse_obsmat(obsmat_path)
    
    # Load video path if exists
    video_path = None
    for vname in ['video.avi', f'{scene_dir}.avi', f'{subset}.avi']:
        vpath = scene_path / vname
        if vpath.exists():
            video_path = vpath
            break
    
    return {
        'root': root,
        'scene_path': scene_path,
        'H': H,
        'H_inv': H_inv,
        'traj_data': traj_data,
        'video_path': video_path,
        'group': group,
        'scene_dir': scene_dir,
    }
