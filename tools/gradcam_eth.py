"""Grad-CAM attribution for the ETH/UCY ResNet-18 scene video encoder.

For a trained ETH/UCY FlowMatching denoiser conditioned on z_video_global,
compute per-frame Grad-CAM heatmaps over ResNet-18 layer4. Saves overlays
to results_eth/gradcam_visualizations/<subset>/w<idx>/.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from utils.normalization import unnormalize_min_max
from utils.config import Config


class _NullLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _make_backbone(
    name: str = "resnet18",
    pretrained: bool = True,
    device: str = "cpu",
) -> tuple[nn.Module, int, Callable]:
    import torchvision
    from torchvision import transforms

    if name == "resnet18":
        m = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        dim = 512
    elif name == "resnet50":
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        dim = 2048
    else:
        raise ValueError(f"Unsupported backbone '{name}'")
    m.fc = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    m = m.to(device)
    return m, dim, preprocess


def euler_sample_predictions(
    denoiser: FlowMatcher,
    cfg,
    x_data: dict,
    num_frames: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    K = int(cfg.denoising_head_preds)
    S = int(cfg.sampling_steps)
    dt = 1.0 / S
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    y_t = torch.randn((1, K, 1, num_frames * 2), device=device, dtype=torch.float)
    y_t.requires_grad_(True)
    for idx in range(S):
        cur_t = dt * idx
        batched_t = torch.full((1,), float(cur_t), device=device)
        preds = denoiser.model_predictions(y_t, x_data, batched_t, flag_print=False)
        y_t = y_t + preds.pred_vel * dt
    return y_t


def best_of_k_target_score(pred: torch.Tensor, gt: torch.Tensor, cfg):
    pred_metric = rearrange(pred, "b k a (f d) -> b k a f d", f=cfg.future_frames)[..., :2]
    pred_metric = unnormalize_min_max(pred_metric, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
    gt_metric = rearrange(gt, "b a (f d) -> b a f d", f=cfg.future_frames)[..., :2]
    gt_metric = unnormalize_min_max(gt_metric, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
    diff = pred_metric - gt_metric.unsqueeze(1)
    ade = torch.norm(diff, dim=-1).mean(dim=-1).mean(dim=-1)
    ade_min, best_idx = ade.min(dim=1)
    score = -ade_min
    return score, best_idx, ade[0]


def gradcam_pass(
    backbone,
    denoiser,
    cfg,
    x_data: dict,
    frames_t: torch.Tensor,
    gt: torch.Tensor,
    num_frames: int,
    device: str,
    seed: int,
):
    feats = []
    grads = []

    def f_hook(m, inp, out):
        feats.append(out)

    def b_hook(m, gin, gout):
        grads.append(gout[0])

    h1 = backbone.layer4.register_forward_hook(f_hook)
    h2 = backbone.layer4.register_full_backward_hook(b_hook)
    try:
        z_list = []
        for i in range(frames_t.shape[0]):
            with torch.enable_grad():
                ft = frames_t[i : i + 1]
                ft.requires_grad_(True)
                rep = backbone(ft)  # [1,512]
                z_list.append(rep)
        z_cat = torch.cat(z_list, dim=0)  # [T,512]
        z_mean = z_cat.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,512]
        x_data_gc = dict(x_data)
        x_data_gc["z_video_global"] = z_mean
        pred = euler_sample_predictions(denoiser, cfg, x_data_gc, num_frames, device, seed)
        score, best_idx, per_hyp_ade = best_of_k_target_score(pred, gt, cfg)
        score = score.mean()
        denoiser.model.zero_grad()
        if z_mean.grad is not None:
            z_mean.grad.zero_()
        score.backward()
        A = feats[-1]  # [T,512,7,7]
        G = grads[-1]  # [T,512,7,7]
        alpha = G.mean(dim=(-2, -1), keepdim=True)
        cam = torch.relu((alpha * A).sum(dim=1))
        cam = cam - cam.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        cam = cam / (cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-12)
        return {
            "cam": cam.detach().cpu().numpy(),
            "alphas": alpha.detach().cpu().numpy(),
            "activations": A.detach().cpu().numpy(),
            "z_video_global": z_mean.detach().cpu().numpy(),
            "per_hyp_ade": per_hyp_ade.detach().cpu().numpy(),
            "score": float(score.detach().cpu().numpy()),
            "best_idx": int(best_idx.detach().cpu().numpy()[0]),
        }
    finally:
        h1.remove()
        h2.remove()


def _apply_jet(cam2d: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    cam_u8 = np.uint8(255 * np.clip(cam2d, 0, 1))
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (heat.astype(np.float32) / 255.0, alpha)


def _save_single(out_dir: Path, fid: int, frame_rgb: np.ndarray, cam2d: np.ndarray, alpha: float = 0.4):
    H, W = frame_rgb.shape[:2]
    cam_up = cv2.resize(cam2d, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_rgb, a = _apply_jet(cam_up, alpha)
    base = frame_rgb.astype(np.float32) / 255.0
    overlay = (1 - a) * base + a * heat_rgb
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"frame_f{fid:06d}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"frame_f{fid:06d}_cam.jpg"), np.uint8(255 * cam_up))


def load_ckpt(ckpt_path: str, model):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt
    # strip module if present
    from collections import OrderedDict
    nd = OrderedDict()
    for kk, vv in sd.items():
        if kk.startswith("module."):
            nd[kk[7:]] = vv
        else:
            nd[kk] = vv
    missing, unexpected = model.load_state_dict(nd, strict=False)
    return model


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="cfg/eth_ucy/cor_fm.yml")
    p.add_argument("--subset", required=True, choices=["eth", "hotel", "univ", "zara1", "zara2"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="data/eth_ucy")
    p.add_argument("--video-features-root", default="features/resnet18")
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    cfg = Config(args.cfg, f"gradcam_eth_{args.subset}")
    cfg.device = args.device
    try:
        cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = True
        cfg.MODEL.CONTEXT_ENCODER.VIDEO_MODE = "full"
        cfg.MODEL.CONTEXT_ENCODER.VIDEO_FEATURES_ROOT = args.video_features_root
    except Exception:
        pass

    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "results_eth" / "gradcam_visualizations" / args.subset / f"w{args.window_index:05d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ETHDataset(cfg=cfg, training=False, data_dir=args.data_dir, subset=args.subset, rotate_time_frame=0, type="original", video_mode="full")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=seq_collate_eth)
    it = iter(loader)
    batch = None
    for i in range(args.batch_index + 1):
        batch = next(it)

    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=_NullLogger(), config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=_NullLogger())
    denoiser = load_ckpt(args.ckpt, denoiser)
    denoiser = denoiser.to(args.device)
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad_(False)

    backbone, dim, preprocess = _make_backbone(args.backbone, pretrained=args.pretrained, device=args.device)
    backbone.eval()

    # load frames for observed window
    frames_t_list = []
    frames_rgb_list = []
    vis_dir = os.path.join(args.data_dir, args.subset, "visual_data")
    # try to get frame ids if available
    frame_ids = getattr(ds, "frame_ids", None)
    base_fid = int(frame_ids[args.batch_index]) if frame_ids is not None else 0
    for t in range(cfg.past_frames):
        fid = base_fid - (cfg.past_frames - 1 - t) * 10  # approximate
        # find nearest
        fpath = os.path.join(vis_dir, f"frame{fid:06d}.jpg")
        if not os.path.exists(fpath):
            # search
            import glob
            cand = sorted(glob.glob(os.path.join(vis_dir, "frame*.jpg")))
            if not cand:
                raise FileNotFoundError(f"No frames in {vis_dir}")
            # pick nearest by index
            fpath = cand[min(max(0, len(cand)//2), len(cand)-1)]
        img = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames_rgb_list.append(img_rgb)
        from PIL import Image
        pil = Image.fromarray(img_rgb)
        frames_t_list.append(preprocess(pil))
    frames_t = torch.stack(frames_t_list, dim=0).to(args.device)

    x_data = {
        "past_traj_original_scale": batch["past_traj_original_scale"].to(args.device),
        "fut_traj": batch["fut_traj"].to(args.device),
    }
    if "z_video_global" in batch:
        x_data["z_video_global"] = batch["z_video_global"].to(args.device)
    gt = batch["fut_traj_original_scale"].to(args.device).float()

    res = gradcam_pass(backbone, denoiser, cfg, x_data, frames_t, gt, cfg.future_frames, args.device, args.seed)
    cam = np.asarray(res["cam"])
    for i in range(min(cam.shape[0], len(frames_rgb_list))):
        _save_single(out_dir, base_fid - (cfg.past_frames - 1 - i) * 10, frames_rgb_list[i], cam[i], args.alpha)
    np.savez(out_dir / "_gradcam.npz", cam=cam, **{k: res[k] for k in ["alphas", "activations", "z_video_global", "per_hyp_ade", "score", "best_idx"]})
    print(f"[gradcam_eth:{args.subset}] saved to {out_dir}")


if __name__ == "__main__":
    main()
