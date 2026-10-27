"""Evaluate an ETH/UCY checkpoint on another scene's test split.

This is the ETH/UCY analogue of ``tools/attrib_video_conditioning.py``'s
``baseline`` arm, used by the Q2 transfer matrix. It deliberately decouples the
normalization statistics from the evaluation scene: inputs are normalized with
``--norm-scene``'s LOSO train-split statistics (the statistics the checkpoint
was trained with), while the test windows come from ``--eval-scene``.

Output: one CSV row per run with ``condition,ade_min,fde_min,n_agents`` in
original (meter) scale.

Example::

    python tools/eval_eth_transfer.py \
        --cfg cfg/eth_ucy/cor_fm.yml \
        --ckpt checkpoints/eth_full.pt \
        --data-dir /kaggle/working/tu-clausthal/datasets \
        --features-root /kaggle/input/moflow-ethucy-processed/features/resnet18 \
        --norm-scene eth --eval-scene hotel \
        --out q2/eth__eval_hotel__full
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from tools.visualize_trajectory_comparison import load_checkpoint_state
from utils.config import Config
from utils.normalization import unnormalize_min_max


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="cfg/eth_ucy/cor_fm.yml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features-root", default=None)
    p.add_argument("--norm-scene", required=True)
    p.add_argument("--eval-scene", required=True)
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=0)
    p.add_argument("--sampling-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    cfg = Config(args.cfg, "eth-transfer")
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.subset = args.eval_scene
    cfg.rotate = False
    cfg.rotate_aug = False
    cfg.data_norm = "min_max"
    cfg.denoising_head_preds = getattr(cfg, "denoising_head_preds", 20)
    cfg.k_preds = getattr(cfg, "k_preds", 20)
    if args.sampling_steps is not None:
        cfg.sampling_steps = args.sampling_steps
    cfg.tied_noise = False
    cfg.fm_rew_sqrt = False
    cfg.fm_in_scaling = False
    cfg.fm_wrapper = "direct"
    cfg.perturb_ctx = 0.0
    cfg.drop_method = "emb"
    cfg.drop_logi_k = 20.0
    cfg.drop_logi_m = 0.5
    cfg.MODEL.USE_PRE_NORM = False
    cfg.LOSS_NN_MODE = "agent"
    cfg.LOSS_REG_REDUCTION = "sum"
    cfg.LOSS_REG_SQUARED = False
    cfg.LOSS_VELOCITY = False

    state = load_checkpoint_state(Path(args.ckpt), use_ema=args.use_ema)
    has_video = any(
        isinstance(k, str) and k.startswith("model.context_encoder.video_proj.")
        for k in state
    )
    video_mode = "full" if has_video else "off"
    if args.features_root:
        cfg.MODEL.CONTEXT_ENCODER.VIDEO_FEATURES_ROOT = args.features_root

    # Compute LOSO train-split statistics from raw data (no pickle).
    _ = ETHDataset(
        cfg=cfg,
        training=True,
        data_dir=str(data_dir),
        subset=args.norm_scene,
        type="original",
        video_mode="off",
    )
    past_min, past_max = float(cfg.past_traj_min), float(cfg.past_traj_max)
    fut_min, fut_max = float(cfg.fut_traj_min), float(cfg.fut_traj_max)

    # Restore the checkpoint's video mode after the stats-only dataset.
    cfg.VIDEO_MODE = video_mode
    cfg.MODEL.CONTEXT_ENCODER.VIDEO_MODE = video_mode
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = has_video
    print(
        f"[eth-transfer] norm={args.norm_scene} eval={args.eval_scene} "
        f"past=[{past_min:.3f},{past_max:.3f}] fut=[{fut_min:.3f},{fut_max:.3f}]",
        flush=True,
    )
    print(f"[eth-transfer] video_mode={video_mode} ckpt={args.ckpt}", flush=True)

    if has_video:
        feat_root = Path(cfg.MODEL.CONTEXT_ENCODER.VIDEO_FEATURES_ROOT)
        npy = feat_root / f"{args.eval_scene}.npy"
        man = feat_root / f"{args.eval_scene}.manifest.parquet"
        if not npy.exists() or not man.exists():
            raise SystemExit(
                f"[eth-transfer] video features missing for eval scene "
                f"'{args.eval_scene}': expected {npy} and {man}. "
                "Pass --features-root or disable the full checkpoint."
            )
    dset = ETHDataset(
        cfg=cfg,
        training=False,
        data_dir=str(data_dir),
        subset=args.eval_scene,
        type="original",
        video_mode=video_mode,
    )
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate_eth,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ETHMotionTransformer(
        model_config=cfg.MODEL, logger=_NullLogger(), config=cfg
    )
    denoiser = FlowMatcher(cfg, model, logger=_NullLogger())
    denoiser.load_state_dict(state)
    denoiser.to(device)
    denoiser.eval()

    ade_sum = 0.0
    fde_sum = 0.0
    n_agents = 0
    for i, batch in enumerate(loader):
        if args.n_batches and i >= args.n_batches:
            break
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        with torch.no_grad():
            pred, *_ = denoiser.sample(
                batch, num_trajs=int(cfg.denoising_head_preds), return_all_states=False
            )
        pred = rearrange(pred, "b k a (f d) -> (b a) k f d", f=int(cfg.future_frames))[
            ..., :2
        ]
        pred = unnormalize_min_max(pred, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
        gt = rearrange(batch["fut_traj_original_scale"], "b a f d -> (b a) f d")
        dist = torch.linalg.norm(pred - gt[:, None], dim=-1)  # [BA, K, F]
        ade_sum += dist.mean(dim=-1).min(dim=-1).values.sum().item()
        fde_sum += dist[..., -1].min(dim=-1).values.sum().item()
        n_agents += dist.shape[0]
        if (i + 1) % 10 == 0:
            print(f"[eth-transfer] batch {i + 1}/{len(loader)}", flush=True)

    if n_agents == 0:
        raise SystemExit("No test agents evaluated.")

    ade = ade_sum / n_agents
    fde = fde_sum / n_agents
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ade_min", "fde_min", "n_agents"])
        w.writerow(["baseline", f"{ade:.6f}", f"{fde:.6f}", n_agents])
    print(
        f"[eth-transfer] scene={args.eval_scene} norm={args.norm_scene} "
        f"ADE_min={ade:.4f} FDE_min={fde:.4f} n={n_agents}",
        flush=True,
    )


if __name__ == "__main__":
    main()
