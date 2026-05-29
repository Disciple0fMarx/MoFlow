"""Build a per-sample (start_frame_id) sidecar aligned to MoFlow's pickle.

MoFlow's canonical `<subset>_<split>.pkl` stores `traj` as `[N, T_total, 2]`
without frame metadata. This script re-derives the start_frame_id of each
sample by replaying the SAME windowing logic used to build the pickle, and
emits `<subset>_<split>_frame_index.pkl` next to it with:

    {
        "scene":            <subset>,
        "stride":           <int>,          # frame-id stride between consecutive obs steps
        "start_frame_id":   np.ndarray[N],  # row-aligned with traj
    }

Usage:
    python -m video_encoder.scripts.build_frame_index \
        --data-root data/eth_ucy/raw/all_data \
        --pkl-dir   data/eth_ucy/original \
        --scene     eth \
        --split     train \
        --past-frames 8 --future-frames 12 --skip 1

Assumes Introvert-style raw txt: `<scene>.txt` with columns
`frame_id  ped_id  x  y` (whitespace-separated, frame_id is integer).
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np


# Mapping from MoFlow's subset name to the raw filename Introvert uses.
_SCENE_TO_RAW = {
    "eth":   "biwi_eth.txt",
    "hotel": "biwi_hotel.txt",
    "univ":  "students003.txt",
    "zara1": "crowds_zara01.txt",
    "zara2": "crowds_zara02.txt",
}


def load_raw(data_root: Path, scene: str) -> np.ndarray:
    filename = _SCENE_TO_RAW.get(scene, f"{scene}.txt")
    path = data_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw trajectory file not found: {path}")
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Unexpected shape {arr.shape} in {path}; expected (N, >=4).")
    return arr


def build_frame_index(scene: str, raw: np.ndarray, past_frames: int,
                      future_frames: int, skip: int) -> dict:
    """Replicates the windowing logic of `data/store_pickle_eth_files.py`:

        seq_len      = past_frames + future_frames
        for each contiguous window of `seq_len` distinct frames (stepped by `skip`),
            we emit one sample whose start_frame_id is the FIRST frame of the
            OBSERVATION window (i.e. frames[idx]).

    Then per agent inside the window, MoFlow appends a row to `traj`. The
    order is: outer loop = window index, inner loop = pedestrian index
    (whatever order they appear in `peds_in_curr_seq`). We replicate that
    order so the sidecar aligns 1:1 with axis-0 of the pickle.
    """
    seq_len = past_frames + future_frames
    frames = np.unique(raw[:, 0]).astype(np.int64).tolist()
    frame_data = {f: raw[raw[:, 0] == f] for f in frames}

    # Detect stride between successive frames (Introvert: usually 10).
    if len(frames) >= 2:
        stride = int(np.gcd.reduce(np.diff(frames).astype(np.int64)))
        stride = max(stride, 1)
    else:
        stride = 1

    num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))
    start_frame_ids: list[int] = []

    for idx in range(0, num_sequences * skip + 1, skip):
        if idx + seq_len > len(frames):
            break
        window_frames = frames[idx: idx + seq_len]
        curr_seq_data = np.concatenate([frame_data[f] for f in window_frames], axis=0)
        peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
        for ped_id in peds_in_curr_seq:
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id]
            # Mirror MoFlow's pad-front / pad-end filter: only count pedestrians
            # present for the entire window.
            pad_front = window_frames.index(curr_ped_seq[0, 0]) - idx
            pad_end   = window_frames.index(curr_ped_seq[-1, 0]) - idx + 1
            if pad_end - pad_front != seq_len:
                continue
            start_frame_ids.append(int(window_frames[0]))

    return {
        "scene": scene,
        "stride": stride,
        "start_frame_id": np.asarray(start_frame_ids, dtype=np.int64),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="Folder with raw <scene>.txt files.")
    ap.add_argument("--pkl-dir", required=True,
                    help="MoFlow pickle dir, e.g. data/eth_ucy/original (sidecar lands in <pkl-dir>/<scene>/).")
    ap.add_argument("--scene", required=True, choices=list(_SCENE_TO_RAW.keys()))
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--past-frames", type=int, default=8)
    ap.add_argument("--future-frames", type=int, default=12)
    ap.add_argument("--skip", type=int, default=1)
    args = ap.parse_args()

    raw = load_raw(Path(args.data_root), args.scene)
    index = build_frame_index(args.scene, raw,
                              args.past_frames, args.future_frames, args.skip)

    out_dir = Path(args.pkl_dir) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.scene}_{args.split}_frame_index.pkl"

    # Sanity-check against the canonical pickle if present.
    pkl_path = out_dir / f"{args.scene}_{args.split}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
        n_pkl = pkl["traj"].shape[0]
        n_idx = index["start_frame_id"].shape[0]
        if n_pkl != n_idx:
            print(f"[WARN] sample count mismatch: pickle={n_pkl}, sidecar={n_idx}")
            print("       The pickle in your fork may have been built with different "
                  "(past_frames, future_frames, skip) than the defaults here. Re-run "
                  "with matching args.")

    with open(out_path, "wb") as f:
        pickle.dump(index, f)
    print(f"Wrote {out_path}  (N={index['start_frame_id'].shape[0]}, stride={index['stride']})")


if __name__ == "__main__":
    main()
