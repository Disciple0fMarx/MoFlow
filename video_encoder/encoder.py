"""Global video encoder: batched forward pass over GlobalFrameDataset."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .backbones import build_backbone
from .datasets import GlobalFrameDataset, collate
from .cache import write_scene_features


class BaseFrameEncoder(ABC):
    """Abstract hook. The future agent/crop encoder will subclass this."""

    feature_dim: int

    @abstractmethod
    def encode_dataframe(self, frames_df: pd.DataFrame) -> tuple[np.ndarray, list[int], list[str]]:
        ...


class GlobalVideoEncoder(BaseFrameEncoder):
    def __init__(
        self,
        backbone: str = "resnet18",
        device: str | None = None,
        batch_size: int = 64,
        num_workers: int = 2,
    ):
        self.backbone_name = backbone
        self.model, self.feature_dim, self.preprocess = build_backbone(backbone)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.inference_mode()
    def encode_dataframe(
        self, frames_df: pd.DataFrame
    ) -> tuple[np.ndarray, list[int], list[str]]:
        """Encode every row of `frames_df`. Returns (features [N,D], frame_ids, scenes)."""
        ds = GlobalFrameDataset(frames_df, preprocess=self.preprocess)
        if len(ds) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32), [], []
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=(self.device == "cuda"),
        )
        feats: list[np.ndarray] = []
        frame_ids: list[int] = []
        scenes: list[str] = []
        for batch in tqdm(loader, desc=f"encode[{self.backbone_name}]"):
            x = batch["image"].to(self.device, non_blocking=True)
            y = self.model(x)
            feats.append(y.detach().cpu().numpy().astype(np.float32))
            frame_ids.extend(batch["frame_id"])
            scenes.extend(batch["scene"])
        return np.concatenate(feats, axis=0), frame_ids, scenes

    def encode_split_to_cache(
        self,
        split_parquet: str | Path,
        out_dir: str | Path,
        limit: int | None = None,
    ) -> dict[str, Path]:
        """Read a split parquet, dedupe (scene, frame), encode, write per-scene cache."""
        df = pd.read_parquet(split_parquet)
        unique = (
            df[["scene", "snapped_frame_id", "frame_path"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if limit is not None:
            unique = unique.head(limit)
        written: dict[str, Path] = {}
        for scene, sub in unique.groupby("scene", sort=False):
            sub = sub.reset_index(drop=True)
            features, frame_ids, _ = self.encode_dataframe(sub)
            feats_path, _ = write_scene_features(
                out_dir, self.backbone_name, scene, frame_ids, features
            )
            written[scene] = feats_path
        return written
