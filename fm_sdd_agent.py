"""Training entry point for SDD with the agent-centric video encoder (LOSO).

Companion to ``fm_sdd_global.py``: the global variant conditions on scene-level
features; this variant drives the **agent-centric** branch.  It reuses the exact
same proven stack (``ETHMotionTransformer`` + ``FlowMatcher`` + ``Trainer``) so
metrics (minADE/minFDE), CFM loss, logging, evaluation (10-step Euler) and the
checkpoint layout are byte-for-byte compatible with the global run.  The only
architectural difference is the conditioning source:

* ``USE_VIDEO = False``         → no scene-level global features;
* ``USE_AGENT_VIDEO = True``    → encode per-agent ``[B, A, T_obs, C, H, W]``
                                  crops with ``AgentVideoEncoder``;
* ``USE_TRI_MODAL_FUSION = True`` → identity-masked cascaded cross-attention
                                  (exactly the ``ETHEncoder`` fusion path).

Datasets reuse ``SDDGlobalDataset`` (identical LOSO window index, trajectory
normalization and ``past_traj_original_scale``/``fut_traj_original_scale``
contracts) and decorate each window with its agent-centric crops fetched by
``SDDAgentWindowDataset`` (one sequential video pass per ``(scene, video_id)``,
see ``data/agent_crop_window.py``).

VRAM safeguards layered in on top of the Trainer's Accelerator:
* ``--grad_accum_steps`` (Native: passed to ``Trainer(gradient_accumulate_every=)``);
* AMP via the Trainer's Accelerator ``mixed_precision`` (f16 on CUDA);
* DataLoader ``pin_memory`` + ``prefetch_factor``/worker tuning;
* ``torch.cuda.empty_cache()`` at evaluation boundaries.

Usage on the remote lab machine:

    python fm_sdd_agent.py \\
        --cfg cfg/sdd/cor_fm.yml \\
        --held_out_scene coupa

    # Evaluation
    python fm_sdd_agent.py --cfg cfg/sdd/cor_fm.yml --held_out_scene coupa --eval
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data.agent_crop_window import SDDAgentWindowDataset
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


# ---------------------------------------------------------------------------
# Dataset: SDDGlobalDataset + aligned agent-centric crops
# ---------------------------------------------------------------------------
class SDDAgentCentricDataset(SDDGlobalDataset):
    """``SDDGlobalDataset`` decorated with per-window agent-centric crops.

    Crops are fetched by :class:`~data.agent_crop_window.SDDAgentWindowDataset`
    using the *same* ``SDDWindowIndex`` rows, so the trajectory features and the
    visual crops for a given ``idx`` refer to the exact same (scene, video,
    track, anchor) — preserving the strict coordinate synchronization required
    by CLAUDE.md.  Each ``__getitem__`` additionally returns::

        "agent_crops"        : [T_obs, C, S, S] uint8 tensor
        "agent_crops_valid"  : bool (False when a source frame/lost flag rules
                               the window out)
    """

    def __init__(
        self,
        cfg,
        training: bool = True,
        sdd_root=None,
        held_out_scene: str | None = None,
        split: str | None = None,
        crop_size: int = 64,
        padding: int = 0,
        drop_lost: bool = True,
    ) -> None:
        # The agent-centric branch is trajectory-only at the dataloader level
        # (no scene-level feature cache); enable the visual branch downstream.
        super().__init__(
            cfg=cfg,
            training=training,
            sdd_root=sdd_root,
            held_out_scene=held_out_scene,
            split=split,
            use_video=False,
        )
        self._crop_ds = SDDAgentWindowDataset(
            sdd_root,
            self.windows,
            crop_size=int(crop_size),
            obs_frames=self.past_frames,
            padding=int(padding),
            drop_lost=bool(drop_lost),
        )
        n_invalid = int((~self._crop_ds.valid).sum())
        if n_invalid:
            print(
                f"[fm_sdd_agent] {n_invalid:,}/{len(self._crop_ds):,} windows lack a "
                "usable crop (missing video or lost frame) → black crop fallback."
            )

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        crops, valid = self._crop_ds[idx]
        item["agent_crops"] = torch.from_numpy(crops)          # [T_obs, C, S, S] uint8
        item["agent_crops_valid"] = bool(valid)
        return item


def collate_sdd_agent(batch: list[dict]) -> dict:
    """Collate that also stacks the agent-centric crops.

    Reuses the global trajectory/auxiliary packing (batch_size, index, scene,
    video_id, anchor_frame, z_video_global=…) and adds::

        "agent_crops" : [B, A=1, T_obs, C, S, S] float32  (the model expects
                        ``[B, A, P, 3, h, w]`` and normalizes internally)
    """
    out = collate_sdd_global(batch)
    crops = torch.stack([b["agent_crops"] for b in batch], dim=0)  # [B, T, C, S, S]
    # Add the agents dimension (SDD: one pedestrian per window).
    out["agent_crops"] = crops.unsqueeze(1).to(dtype=torch.float32)  # [B, 1, T, C, S, S]
    out["agent_crops_valid"] = torch.tensor(
        [b["agent_crops_valid"] for b in batch], dtype=torch.bool
    )
    return out


# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="cfg/sdd/cor_fm.yml", type=str)
    p.add_argument("--exp", default="", type=str)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--eval_on_train", action="store_true")

    # ---- SDD-specific -------------------------------------------------------
    p.add_argument(
        "--sdd_root",
        default=None,
        type=str,
        help=f"SDD root (default: {DEFAULT_SDD_ROOT}; Kaggle auto-detected).",
    )
    p.add_argument(
        "--held_out_scene",
        required=False,
        type=str,
        choices=SDD_SCENES,
        help="LOSO held-out scene. Required at runtime.",
    )
    p.add_argument("--crop_size", default=64, type=int)
    p.add_argument("--padding", default=0, type=int)
    p.add_argument("--no_drop_lost", action="store_true")

    # ---- Standard overrides (mirror fm_sdd_global.py / fm_eth.py) -----------
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

    # ---- FM / arch / loss / optimization (subset mirror) ---------------------
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

    # ---- VRAM / throughput (agent-centric specific) ---------------------------
    p.add_argument(
        "--grad_accum_steps",
        default=1,
        type=int,
        help="Gradient accumulation steps (native Trainer gradient_accumulate_every).",
    )
    p.add_argument(
        "--amp",
        default="f16",
        choices=["no", "f16", "bf16"],
        help="Accelerator mixed_precision for the Trainer's AMP (f16 on CUDA).",
    )
    p.add_argument("--num_workers", default=2, type=int)
    p.add_argument("--prefetch_factor", default=2, type=int)
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
        print(f"[fm_sdd_agent] Kaggle detected → SDD root: {cfg.MODEL.CONTEXT_ENCODER.SDD_ROOT}")
    cfg.MODEL.CONTEXT_ENCODER.HELD_OUT_SCENE = args.held_out_scene

    # ---- Agent-centric branch ------------------------------------------------
    # Scene-level video OFF; agent-centric ON with tri-modal cascaded fusion.
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = False
    cfg.MODEL.CONTEXT_ENCODER.USE_AGENT_VIDEO = True
    cfg.MODEL.CONTEXT_ENCODER.USE_TRI_MODAL_FUSION = True
    cfg.MODEL.CONTEXT_ENCODER.AGENT_CROP_SIZE = [args.crop_size, args.crop_size]

    tag += f"SDD_ho{args.held_out_scene}_agent"
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


# ---------------------------------------------------------------------------
# DataLoaders / network
# ---------------------------------------------------------------------------
def build_data_loaders(cfg, args):
    common = dict(
        cfg=cfg,
        sdd_root=args.sdd_root,
        held_out_scene=args.held_out_scene,
        crop_size=args.crop_size,
        padding=args.padding,
        drop_lost=not args.no_drop_lost,
    )
    train_dset = SDDAgentCentricDataset(training=True, split="train", **common)
    test_dset = SDDAgentCentricDataset(training=False, split="test", **common)

    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_sdd_agent,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,  # eager crops → safe with workers.
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_sdd_agent,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, test_loader


def build_network(cfg, logger):
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=logger, config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=logger)
    return denoiser


# ---------------------------------------------------------------------------
# Trainer sub-class: VRAM guards + AMP
# ---------------------------------------------------------------------------
class AgentTrainer(Trainer):
    """Add explicit AMP and eval-boundary memory cleanup to the base Trainer.

    The base ``Trainer`` owns the HuggingFace Accelerator, constructed inside
    ``__init__`` without exposing its ``mixed_precision``.  We monkey-patch the
    module-level ``Accelerator`` symbol *before* ``super().__init__`` so the base
    builds its accelerator with the CLI-selected precision (f16/bf16 on CUDA),
    then restore the original symbol.  This is the requested AMP safeguard and
    avoids re-preparing model/opt/loaders with a second accelerator.

    ``test`` flushes the CUDA cache at evaluation boundaries (the requested
    ``torch.cuda.empty_cache()`` safeguard).
    """

    def __init__(self, *args, mixed_precision: str = "no", **kwargs):
        import trainer.denoising_model_trainers as tdt
        from accelerate import Accelerator as _Accelerator

        orig = tdt.Accelerator
        tdt.Accelerator = lambda *a, _mp=mixed_precision, **k: _Accelerator(
            *a, **{**k, "mixed_precision": _mp}
        )
        try:
            super().__init__(*args, **kwargs)
        finally:
            tdt.Accelerator = orig

    @staticmethod
    def _empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test(self, mode="best", eval_on_train=False):
        self._empty_cache()
        try:
            super().test(mode=mode, eval_on_train=eval_on_train)
        finally:
            self._empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
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

    # AMP: default f16 on CUDA, bf16 requested explicitly, no on CPU.
    if args.amp != "no" and cfg.device == "cpu":
        print("[fm_sdd_agent] --amp ignored (no CUDA detected).")
        amp = "no"
    else:
        amp = args.amp

    trainer = AgentTrainer(
        cfg,
        denoiser,
        train_loader,
        test_loader,
        tb_log=tb_log,
        logger=logger,
        gradient_accumulate_every=max(1, args.grad_accum_steps),
        ema_decay=0.995,
        ema_update_every=1,
        mixed_precision=amp,
    )
    if args.eval:
        trainer.test(mode="best", eval_on_train=args.eval_on_train)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
