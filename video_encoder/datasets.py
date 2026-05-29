"""Torch datasets over resolved (scene, frame_id, path) triples."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class GlobalFrameDataset(Dataset):
    """Iterates unique (scene, snapped_frame_id, frame_path) rows for encoding.

    Returns dict with keys: 'image' (tensor), 'scene' (str), 'frame_id' (int).
    """

    def __init__(self, frames_df: pd.DataFrame, preprocess: Callable):
        required = {"scene", "snapped_frame_id", "frame_path"}
        missing = required - set(frames_df.columns)
        if missing:
            raise ValueError(f"frames_df missing columns: {missing}")
        self.df = frames_df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        with Image.open(row["frame_path"]) as img:
            img = img.convert("RGB")
            tensor = self.preprocess(img)
        return {
            "image": tensor,
            "scene": row["scene"],
            "frame_id": int(row["snapped_frame_id"]),
        }


def collate(batch: Sequence[dict]):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "scene": [b["scene"] for b in batch],
        "frame_id": [b["frame_id"] for b in batch],
    }
