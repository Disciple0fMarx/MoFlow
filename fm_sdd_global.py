"""Training entry point for SDD with the global video encoder (LOSO).

Aligned with Note Technique §5.2 (Global Video Encoder) and §9 (Étape 2).
Reuses the existing ``ETHEncoder`` + ``ETHMotionTransformer`` stack; only the
dataloader and CLI flags are SDD-specific.

Dual-infrastructure defaults (see ``video_encoder.sdd_adapter``):
the SDD root defaults to the Remote Lab Machine
(`/home/efrei_stage/Desktop/Datasets/SDD`) and auto-switches to the Kaggle
mount (`/kaggle/input/datasets/aryashah2k/stanford-drone-dataset`) when
``/kaggle`` exists. An explicit ``--sdd_root`` always wins.

Usage on the remote lab machine:

    python fm_sdd_global.py \\
        --cfg cfg/sdd/cor_fm.yml \\
        --held_out_scene coupa \\
        --video_features_root /home/efrei_stage/Desktop/Datasets/SDD/features/resnet18

    # Evaluation
    python fm_sdd_global.py --cfg cfg/sdd/cor_fm.yml --held_out_scene coupa --eval
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data.dataloader_sdd_global import (
    SDD_SCENES,
    SDDGlobalDataset,
    collate_sdd_global,
)

from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from trainer.denoising_model_trainers import Trainer
from utils.config import Config
from utils.utils import back_up_code_git, log_config_to_file, set_random_seed
from video_encoder.sdd_adapter import (
    DEFAULT_SDD_ROOT,
    expand_sdd_root,
    is_kaggle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="cfg/eth_ucy/cor_fm.yml", type=str)
    p.add_argument("--exp", default="", type=str)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--eval_on_train", action="store_true")

    # ---- SDD-specific -------------------------------------------------------
    p.add_argument(
        "--sdd_root",
        default=None,
        type=str,
        help=(
            "SDD dataset root. Default: the Remote Lab Machine root "
            f"({DEFAULT_SDD_ROOT}); auto-switches to the Kaggle mount when "
            "/kaggle exists. An explicit value always wins."
        ),
    )
    p.add_argument(
        "--held_out_scene",
        required=False,
        type=str,
        choices=SDD_SCENES,
        help="LOSO held-out scene. Required at runtime.",
    )
    p.add_argument(
        "--video_features_root",
        default=None,
        type=str,
        help="Directory with <scene>.npy + <scene>.manifest.parquet caches.",
    )
    p.add_argument("--video_stride", default=1, type=int)
    p.add_argument(
        "--no_video",
        action="store_true",
        help="Disable the global video branch (baseline trajectory-only ablation).",
    )

    # ---- Standard overrides (mirror fm_eth.py) -------------------------------
    p.add_argument("--epochs", default=None, type=int)
    p.add_argument("--batch_size", default=None, type=int)
    p.add_argument("--n_train", default=None, type=int)
    p.add_argument("--n_test", default=None, type=int)
    p.add_argument("--data_norm", default="min_max", choices=["min_max", "original"])
    p.add_argument("--rotate", action="store_true")
    p.add_argument("--rotate_time_frame", default=0, type=int)
    p.add_argument("--rotate_aug", action="store_true")
    p.add_argument("--checkpt_freq", default=5, type=int)
    p.add_argument("--max_num_ckpts", default=5, type=int)
    p.add_argument("--fix_random_seed", action="store_true")
    p.add_argument("--seed", default=42, type=int)

    # ---- FM / arch / loss / optimization (subset of fm_eth.py flags) --------
    p.add_argument("--sampling_steps", default=10, type=int)
    p.add_argument("--t_schedule", default="logit_normal", choices=["uniform", "logit_normal"])
    p.add_argument("--logit_norm_mean", default=-0.5, type=float)
    p.add_argument("--logit_norm_std", default=1.5, type=float)
    p.add_argument("--fm_wrapper", default="direct", choices=["direct", "velocity", "precond"])
    p.add_argument("--fm_rew_sqrt", action="store_true")
    p.add_argument("--fm_in_scaling", action="store_true")
    p.add_argument("--perturb_ctx", default=0.0, type=float)
    p.add_argument("--drop_method", default="emb", choices=["None", "input", "emb"])
    p.add_argument("--drop_logi_k", default=20.0, type=float)
    p.add_argument("--drop_logi_m", default=0.5, type=float)
    p.add_argument("--use_pre_norm", action="store_true")
    p.add_argument("--num_layers", default=None, type=int)
    p.add_argument("--dropout", default=None, type=float)
    p.add_argument("--tied_noise", action="store_true")
    p.add_argument("--loss_nn_mode", default="agent", choices=["agent", "scene"])
    p.add_argument("--loss_reg_reduction", default="sum", choices=["mean", "sum"])
    p.add_argument("--loss_reg_squared", action="store_true")
    p.add_argument("--loss_velocity", action="store_true")
    p.add_argument("--loss_cls_weight", default=1.0, type=float)
    p.add_argument("--init_lr", default=None, type=float)
    p.add_argument("--weight_decay", default=None, type=float)
    return p.parse_args()


def init_basics(args: argparse.Namespace) -> tuple[Config, object, SummaryWriter]:
    cfg = Config(args.cfg, args.exp)
    tag = "_"

    cfg.denoising_method = "fm"
    cfg.sampling_steps = args.sampling_steps
    cfg.t_schedule = args.t_schedule
    cfg.fm_wrapper = args.fm_wrapper
    cfg.fm_rew_sqrt = args.fm_rew_sqrt
    cfg.fm_in_scaling = args.fm_in_scaling
    if args.t_schedule == "logit_normal":
        cfg.logit_norm_mean = args.logit_norm_mean
        cfg.logit_norm_std = args.logit_norm_std
    cfg.perturb_ctx = args.perturb_ctx
    cfg.drop_method = args.drop_method
    cfg.drop_logi_k = args.drop_logi_k
    cfg.drop_logi_m = args.drop_logi_m
    cfg.MODEL.USE_PRE_NORM = args.use_pre_norm

    if args.num_layers is not None:
        cfg.MODEL.NUM_LAYERS = args.num_layers
        cfg.MODEL.CONTEXT_ENCODER.NUM_ATTN_LAYERS = args.num_layers
        cfg.MODEL.MOTION_DECODER.NUM_DECODER_BLOCKS = args.num_layers
    if args.dropout is not None:
        cfg.MODEL.DROPOUT = args.dropout
        cfg.MODEL.CONTEXT_ENCODER.DROPOUT_OF_ATTN = args.dropout
        cfg.MODEL.MOTION_DECODER.DROPOUT_OF_ATTN = args.dropout

    cfg.tied_noise = args.tied_noise
    cfg.LOSS_NN_MODE = args.loss_nn_mode
    cfg.LOSS_REG_REDUCTION = args.loss_reg_reduction
    cfg.LOSS_REG_SQUARED = args.loss_reg_squared
    cfg.LOSS_VELOCITY = args.loss_velocity

    cfg.rotate = args.rotate
    cfg.rotate_aug = args.rotate_aug
    cfg.data_norm = args.data_norm

    if args.init_lr is not None:
        cfg.OPTIMIZATION.LR = args.init_lr
    if args.weight_decay is not None:
        cfg.OPTIMIZATION.WEIGHT_DECAY = args.weight_decay
    cfg.OPTIMIZATION.LOSS_WEIGHTS["cls"] = args.loss_cls_weight

    if args.epochs is not None:
        cfg.OPTIMIZATION.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.train_batch_size = args.batch_size
        cfg.test_batch_size = args.batch_size * 2
    cfg.checkpt_freq = args.checkpt_freq
    cfg.max_num_ckpts = args.max_num_ckpts

    # ---- SDD knobs injected into cfg ----------------------------------------
    cfg.MODEL.CONTEXT_ENCODER.SDD_ROOT = str(expand_sdd_root(args.sdd_root))
    if is_kaggle() and args.sdd_root is None:
        print(f"[fm_sdd_global] Kaggle detected → SDD root: {cfg.MODEL.CONTEXT_ENCODER.SDD_ROOT}")
    cfg.MODEL.CONTEXT_ENCODER.HELD_OUT_SCENE = args.held_out_scene
    if args.video_features_root is not None:
        cfg.MODEL.CONTEXT_ENCODER.VIDEO_FEATURES_ROOT = str(
            Path(args.video_features_root).expanduser()
        )
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = (not args.no_video) and bool(
        getattr(cfg.MODEL.CONTEXT_ENCODER, "USE_VIDEO", False)
    )

    tag += f"SDD_ho{args.held_out_scene}"
    if cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO:
        tag += "_vid"
    else:
        tag += "_novid"
    tag = tag.replace("__", "_")

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = cfg.create_dirs(tag_suffix=tag)
    if args.fix_random_seed:
        set_random_seed(args.seed)

    tb_dir = os.path.abspath(os.path.join(cfg.log_dir, "../tb"))
    os.makedirs(tb_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_dir)

    back_up_code_git(cfg, logger=logger)
    log_config_to_file(cfg.yml_dict, logger=logger)
    return cfg, logger, tb_log


def build_data_loaders(cfg, args):
    common = dict(
        cfg=cfg,
        sdd_root=args.sdd_root,
        held_out_scene=args.held_out_scene,
        use_video=cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO,
        video_features_root=args.video_features_root,
        video_stride=args.video_stride,
    )
    train_dset = SDDGlobalDataset(training=True, split="train", **common)
    test_dset = SDDGlobalDataset(training=False, split="test", **common)

    # Note: num_workers=0 keeps the per-worker feature mmap unique to the
    # main process — multi-worker would duplicate the (already mmap'd) .npy
    # buffers in RAM.  Acceptable tradeoff for SDD where annotation parsing
    # is cheap and the bottleneck is GPU forward pass, not data loading.
    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_sdd_global,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sdd_global,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_network(cfg, logger):
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=logger, config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=logger)
    return denoiser


def main() -> None:
    args = parse_args()
    if not args.eval and args.held_out_scene is None:
        raise SystemExit(
            "--held_out_scene is required for training. "
            f"Choose one of: {SDD_SCENES}"
        )

    cfg, logger, tb_log = init_basics(args)
    train_loader, test_loader = build_data_loaders(cfg, args)
    denoiser = build_network(cfg, logger)

    trainer = Trainer(
        cfg,
        denoiser,
        train_loader,
        test_loader,
        tb_log=tb_log,
        logger=logger,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        ema_update_every=1,
    )
    if args.eval:
        trainer.test(mode="best", eval_on_train=args.eval_on_train)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
