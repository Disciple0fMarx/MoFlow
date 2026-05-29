"""Custom-video pipeline: decode any video file and encode sampled frames."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .backbones import build_backbone


def _iter_frames_decord(video: str, stride: int) -> Iterator[tuple[int, float, Image.Image]]:
    import decord  # type: ignore

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video)
    fps = float(vr.get_avg_fps()) or 1.0
    for idx in range(0, len(vr), stride):
        arr = vr[idx].asnumpy()
        yield idx, idx / fps, Image.fromarray(arr)


def _iter_frames_opencv(video: str, stride: int) -> Iterator[tuple[int, float, Image.Image]]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield idx, idx / fps, Image.fromarray(rgb)
            idx += 1
    finally:
        cap.release()


def _iter_frames(video: str, stride: int):
    try:
        yield from _iter_frames_decord(video, stride)
        return
    except Exception:
        pass
    yield from _iter_frames_opencv(video, stride)


@torch.inference_mode()
def encode_custom_video(
    video: str | Path,
    out_dir: str | Path,
    backbone: str = "resnet18",
    stride: int = 10,
    device: str | None = None,
    batch_size: int = 32,
) -> dict[str, Path]:
    """Encode a single video into <out_dir>/<backbone>/custom.npy + manifest."""
    model, dim, preprocess = build_backbone(backbone)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    feats: list[np.ndarray] = []
    rows: list[dict] = []
    buf_tensors: list[torch.Tensor] = []
    buf_meta: list[dict] = []

    def flush():
        if not buf_tensors:
            return
        x = torch.stack(buf_tensors, dim=0).to(dev, non_blocking=True)
        y = model(x).detach().cpu().numpy().astype(np.float32)
        feats.append(y)
        rows.extend(buf_meta)
        buf_tensors.clear()
        buf_meta.clear()

    for frame_idx, t_sec, img in tqdm(_iter_frames(str(video), stride), desc="decode"):
        buf_tensors.append(preprocess(img))
        buf_meta.append({"frame_idx": frame_idx, "t_seconds": float(t_sec)})
        if len(buf_tensors) >= batch_size:
            flush()
    flush()

    out_base = Path(out_dir) / backbone
    out_base.mkdir(parents=True, exist_ok=True)
    feats_arr = (
        np.concatenate(feats, axis=0) if feats else np.zeros((0, dim), dtype=np.float32)
    )
    feats_path = out_base / "custom.npy"
    manifest_path = out_base / "custom.manifest.parquet"
    np.save(feats_path, feats_arr)
    pd.DataFrame(rows).assign(
        row_idx=np.arange(len(rows), dtype=np.int64),
        source=str(video),
    ).to_parquet(manifest_path, index=False)
    return {"features": feats_path, "manifest": manifest_path}
