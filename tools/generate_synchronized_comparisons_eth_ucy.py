#!/usr/bin/env python3
"""Synchronized 4-panel qualitative comparison for ETH/UCY video-conditioning ablation.

Direct ETH/UCY equivalent of tools/generate_synchronized_comparisons.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

from einops import rearrange

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
from data.tu_clausthal_eth_ucy_parser import TU_CLAUSTHAL_MAP
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from utils.config import Config
from utils.eth_ucy_geometry import (annotation_step,
                                    assert_pixel_frame_alignment,
                                    observed_frame_ids, scene_video_path,
                                    temporal_anchor_frame,
                                    video_index_for_frame)
from utils.eth_ucy_geometry import world_to_pixel as geo_world_to_pixel
from utils.norm_bounds import (NormBoundsError, apply_bounds_to_cfg,
                               bounds_from_cfg, load_norm_bounds_from_run,
                               resolve_run_dir_from_checkpoint)
from utils.normalization import unnormalize_min_max
from utils.video_reader import get_video_reader
from video_encoder.moflow_adapter import FrameFeatureLookup

# Optional Grad-CAM support
try:
    from tools import gradcam_eth as gcam_eth
    from tools.gradcam_eth import _apply_jet, _make_backbone, gradcam_pass
    _HAS_GCAM = True
except Exception:
    gcam_eth = None
    _HAS_GCAM = False

SCENES = ("eth", "hotel", "univ", "zara1", "zara2")
VARIANTS = ("off", "static", "full")
PANEL_TITLES = {
    "off": "off  ($z_{vid}=0$)",
    "static": "static  (fixed prior, $P=1$)",
    "full": "full  ($P=8$ frames)",
}
PRED_COLOR = "#d62728"

class NullLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass

def to_numpy(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.array(x)

def infer_agents_from_ckpt(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = None
        if isinstance(ckpt, dict):
            for k in ["state_dict", "model_state_dict", "model", "ema_state_dict"]:
                if k in ckpt and isinstance(ckpt[k], dict):
                    sd = ckpt[k]
                    break
            if sd is None and not any(isinstance(v, dict) for v in ckpt.values()):
                sd = ckpt
        else: sd = ckpt
        if sd is None: return None
        for key in sd.keys():
            if 'agent_query_embedding.weight' in key: return int(sd[key].shape[0])
            if 'agent_order_embedding.weight' in key: return int(sd[key].shape[0])
    except: pass
    return None

def force_set_agents(cfg_m, agents):
    for fn in [lambda c,a:setattr(c.MODEL.CONTEXT_ENCODER,'AGENTS',a), lambda c,a:setattr(c,'AGENTS',a)]:
        try: fn(cfg_m,agents)
        except: pass

def _apply_run_bounds(cfg, run_dir=None, ckpt=None):
    """Load exact training bounds from a run artifact onto ``cfg`` (strict)."""
    target = run_dir or (resolve_run_dir_from_checkpoint(ckpt) if ckpt else None)
    if target is None:
        raise NormBoundsError(
            "Cannot locate a training run directory for normalization bounds. "
            "Pass a checkpoint inside a run dir or provide --norm-bounds-dir."
        )
    bounds = load_norm_bounds_from_run(target)
    apply_bounds_to_cfg(cfg, bounds)
    return bounds

def load_ckpt_flexible(ckpt_path, model):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = None
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "ema_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict): sd = ckpt[k]; break
        if sd is None: sd = ckpt
    else: sd = ckpt
    from collections import OrderedDict
    nd = OrderedDict()
    for kk,vv in sd.items():
        kk2 = kk[7:] if kk.startswith("module.") else kk
        nd[kk2]=vv
    # no fragile reshaping: instantiate model from the correct run/checkpoint config so
    # embedding shapes (e.g. agent_query_embedding/agent_order_embedding) match natively.
    incompatible = model.load_state_dict(nd, strict=False)
    try:
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing or unexpected:
            print(f"[ckpt] loaded with missing={len(missing)} unexpected={len(unexpected)} (strict=False)")
    except Exception:
        pass
    return model

def find_checkpoint(scene, mode, cor_fm_root="results_eth_ucy/cor_fm", flat_dir="results_eth_ucy/loso_full_matrix/checkpoints"):
    flat = os.path.join(flat_dir, f"{scene}_{mode}.pt")
    if os.path.exists(flat): return flat, None
    vm = f"vm{mode}"
    import glob
    cands = []
    for pat in [os.path.join(cor_fm_root,f"*_{scene}_{vm}*"),os.path.join(cor_fm_root,f"*{scene}*{vm}*")]:
        cands.extend(glob.glob(pat))
    for c in cands:
        if os.path.isdir(c):
            bp=os.path.join(c,'models','checkpoint_best.pt'); lp=os.path.join(c,'models','checkpoint_last.pt')
            ck=bp if os.path.exists(bp) else (lp if os.path.exists(lp) else None)
            if ck: return ck,c
        elif c.endswith('.pt'): return c,None
    return None,None

def load_run_config(run_dir, default_cfg='cfg/eth_ucy/cor_fm.yml'):
    if run_dir and os.path.isdir(run_dir):
        for name in ['cor_fm_updated.yml','cor_fm.yml','config.yml']:
            p=os.path.join(run_dir,name)
            if os.path.exists(p): return p
            p2=os.path.join(run_dir,'code_backup',name)
            if os.path.exists(p2): return p2
    return default_cfg

def _decode_frame_from_video(video_path, frame_index, scene=None):
    """Decode an exact frame from ``video.avi`` (BGR, no offset, no clamping)."""
    reader = get_video_reader(video_path)
    if scene is None:
        for name, (_group, scene_dir) in TU_CLAUSTHAL_MAP.items():
            if Path(video_path).parent.name == scene_dir:
                scene = name
                break
    index = video_index_for_frame(scene, frame_index) if scene else int(frame_index)
    return reader.read(index, color="bgr")


def _resolve_scene_path(data_dir, scene):
    data_dir = Path(data_dir)
    if scene not in TU_CLAUSTHAL_MAP:
        raise ValueError(f"Unknown scene {scene}, valid: {sorted(set(TU_CLAUSTHAL_MAP.keys()))}")
    group, scene_dir = TU_CLAUSTHAL_MAP[scene]
    # nested TU-Clausthal layout
    nested = data_dir / group / scene_dir
    if nested.is_dir():
        return nested
    # compatibility: data_dir already points at group or scene_dir
    if (data_dir / scene_dir).is_dir():
        return data_dir / scene_dir
    if data_dir.name == group and (data_dir / scene_dir).is_dir():
        return data_dir / scene_dir
    raise FileNotFoundError(
        f"TU-Clausthal scene path not found for scene={scene} under external --data-dir={data_dir}. "
        f"Expected nested layout: {group}/{scene_dir} (e.g. /home/efrei_stage/Desktop/tu-clausthal/datasets/{group}/{scene_dir})."
    )


def _decode_scene_frame(data_dir, scene, frame_index):
    vpath = scene_video_path(data_dir, scene)
    frame_bgr = _decode_frame_from_video(vpath, int(frame_index), scene=scene)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _sample_start_frame_ids(ds, batch, chosen_bi):
    """Resolve the first observed frame id for a dataset sample."""
    try:
        fids = getattr(ds, 'frame_ids', None)
        if fids is not None and len(fids) > int(chosen_bi):
            return int(np.asarray(fids).reshape(-1)[int(chosen_bi)])
    except Exception:
        pass
    try:
        if isinstance(batch, dict) and 'index' in batch:
            idx = batch['index']
            if torch.is_tensor(idx):
                idx = int(idx.detach().cpu().reshape(-1)[0].item())
            else:
                idx = int(np.asarray(idx).reshape(-1)[0])
            fids = getattr(ds, 'frame_ids', None)
            if fids is not None and 0 <= idx < len(fids):
                return int(np.asarray(fids).reshape(-1)[idx])
    except Exception:
        pass
    raise ValueError(
        "Cannot resolve the start frame for this sample; refusing to guess a temporal anchor."
    )


def _anchor_frame_id(ds, batch, chosen_bi, cfg, data_dir, scene):
    """Final observed frame: start + (past_frames - 1) * scene annotation step."""
    start = _sample_start_frame_ids(ds, batch, chosen_bi)
    past = int(getattr(cfg, 'past_frames', 8) or 8)
    return temporal_anchor_frame(start, scene, past)


def find_ref_img(data_dir, scene, anchor_frame=None):
    frame = 0 if anchor_frame is None else int(anchor_frame)
    return f"video.avi#frame={max(0, frame)}"


def load_img(p, data_dir=None, scene=None, anchor_frame=None):
    if anchor_frame is not None and data_dir is not None and scene is not None:
        return _decode_scene_frame(data_dir, scene, anchor_frame)
    if isinstance(p, str) and p.startswith('video.avi#frame='):
        fi = int(p.split('=')[-1])
        if data_dir is not None and scene is not None:
            return _decode_scene_frame(data_dir, scene, fi)
    raise FileNotFoundError(
        f"Cannot resolve frame (path={p!r}, data_dir={data_dir}, scene={scene}); "
        "only native video.avi decoding is supported."
    )

def _audit_verified_world_to_pixel(pts_xy, H, scene, orig_size, target_res=None):
    """Delegate to the single authoritative world->pixel mapping.

    eth/hotel use ``(u, v) = (b, a)``; univ/zara1/zara2 use no swap plus the
    UCY optical-centre convention ``(a + W/2, H/2 - b)``.  Empirically verified
    against moving-pixel distance on all five scenes.
    """
    pts = np.asarray(pts_xy, dtype=np.float64)
    if pts.size == 0:
        return pts
    return geo_world_to_pixel(
        pts, H, scene=scene, orig_size=orig_size, target_res=target_res
    )


def world_to_pix_native(ds, pts, scene=None, target_res=None):
    if pts is None or ds is None:
        return None
    try:
        pts_arr = np.asarray(pts, dtype=float)
        if pts_arr.size == 0:
            return pts_arr
        if pts_arr.ndim >= 2 and pts_arr.shape[-1] > 2:
            pts_arr = pts_arr[..., :2]
        elif pts_arr.ndim == 1:
            pts_arr = pts_arr.reshape(-1, pts_arr.shape[-1] if pts_arr.ndim > 1 else 1)
            if pts_arr.shape[-1] < 2:
                return None
        original_shape = pts_arr.shape
        flat = pts_arr.reshape(-1, 2)
        sc = scene or getattr(ds, 'subset', None)
        H = None
        try:
            if hasattr(ds, 'scene_H') and sc in ds.scene_H:
                H = ds.scene_H[sc]
            elif hasattr(ds, 'H'):
                H = ds.H
        except Exception:
            H = None
        orig_size = getattr(ds, 'orig_res', (720, 576))
        try:
            if hasattr(ds, 'SCENE_RESOLUTIONS'):
                orig_size = ds.SCENE_RESOLUTIONS.get(sc, orig_size)
        except Exception:
            pass
        if H is not None:
            res = _audit_verified_world_to_pixel(flat, H, sc, orig_size, target_res=target_res)
        else:
            pts_t = torch.tensor(flat, dtype=torch.float32)
            try:
                res = ds.world_to_pixel(pts_t, scene=sc, target_res=target_res) if target_res is not None else ds.world_to_pixel(pts_t, scene=sc)
            except TypeError:
                res = ds.world_to_pixel(pts_t, scene=sc)
            if isinstance(res, torch.Tensor):
                res = res.detach().cpu().numpy()
        if res is None:
            return None
        res = np.asarray(res)
        if res.ndim == 2 and res.shape[1] >= 2:
            res = res[:, :2]
        if len(original_shape) > 2:
            res = res.reshape(original_shape[:-1] + (2,))
        return res
    except Exception as e:
        print(f'[viz] world_to_pixel failed: {e}')
        traceback.print_exc()
        return None

def ensure_2d_coords(arr):
    if arr is None: return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if len(arr) >= 2:
            return arr.reshape(-1, 2)
        return None
    if arr.ndim == 2:
        if arr.shape[1] < 2:
            return None
        return arr[..., :2]
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[1] < 2: return None
        return arr[..., :2]
    return None


def shared_limits_eth(bundles, background, scene, ds_ref):
    arrays = []
    for b in bundles.values():
        arrays.extend([
            np.asarray(b['obs']).reshape(-1,2),
            np.asarray(b['gt']).reshape(-1,2),
            np.asarray(b['preds']).reshape(-1,2),
            np.asarray(b['best']).reshape(-1,2),
        ])
    pts = np.concatenate(arrays,axis=0)
    if background is not None and ds_ref is not None:
        lo,hi = pts.min(0), pts.max(0)
        margin=1.0
        x0,x1 = float(lo[0]-margin), float(hi[0]+margin)
        y0,y1 = float(lo[1]-margin), float(hi[1]+margin)
        # get image extent by projecting corners
        c1 = world_to_pix_native(ds_ref, np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]), scene, target_res=(bg.shape[1],bg.shape[0]) if bg is not None else None)
        if c1 is not None and np.isfinite(c1).all():
            cx0,cx1=np.min(c1[:,0]),np.max(c1[:,0]); cy0,cy1=np.min(c1[:,1]),np.max(c1[:,1])
            return (cx0,cx1),(cy1,cy0)
    lo,hi = pts.min(0), pts.max(0); sp=np.maximum(hi-lo,1e-6); pad=0.08*sp
    return (lo[0]-pad[0],hi[0]+pad[0]),(lo[1]-pad[1],hi[1]+pad[1])

def _legend_handles():
    return [
        Line2D([0],[0],color='black',linestyle='--',linewidth=1.6,marker='o',markersize=3,label='Observed history'),
        Line2D([0],[0],color='green',linewidth=1.8,marker='o',markersize=3,label='Ground truth'),
        Line2D([0],[0],color=PRED_COLOR,linewidth=1.0,alpha=0.5,label='Predicted (K=20)'),
        Line2D([0],[0],color=PRED_COLOR,linewidth=2.6,label='Best-of-K (min-ADE)'),
    ]

def euler_sample(denoiser,cfg,x_data,num_frames,device,seed):
    K=int(getattr(cfg,'denoising_head_preds',20)); S=int(getattr(cfg,'sampling_steps',10))
    dt=1.0/S
    torch.manual_seed(seed); np.random.seed(seed)
    y_t=torch.randn((1,K,1,num_frames*2),device=device)
    for idx in range(S):
        cur_t=dt*idx; batched_t=torch.full((1,),float(cur_t),device=device)
        preds=denoiser.model_predictions(y_t,x_data,batched_t,flag_print=False)
        y_t=y_t+preds.pred_vel*dt
    return y_t

def best_of_k_eth(pred,gt,cfg,ds=None,init_pos=None):
    bounds = bounds_from_cfg(cfg, source="best_of_k_eth")
    fut_min, fut_max = bounds.fut_min, bounds.fut_max
    future_frames=int(getattr(cfg,'future_frames',12))
    gt_arr=to_numpy(gt)
    if gt_arr.ndim==4:  # [B,A,F,D] or [B,A,F*D]? normalize below
        if gt_arr.shape[-1]==future_frames*2:
            gt_m=rearrange(torch.tensor(gt_arr).float(),'b a (f d)->b a f d',f=future_frames)[0,...,:2]
        else:
            gt_m=torch.tensor(gt_arr[0,...,:future_frames,:2]).float()
    elif gt_arr.ndim==3:
        if gt_arr.shape[-1]==future_frames*2:
            gt_m=rearrange(torch.tensor(gt_arr).float(),'b (f d)->b f d',f=future_frames)[0,...,:2].unsqueeze(0)
        else:
            gt_m=torch.tensor(gt_arr[0,:future_frames,:2]).float().unsqueeze(0)
    else:
        gt_m=torch.tensor(gt_arr).float().reshape(-1,future_frames,2)
    # Ground truth is already raw metres (relative to the last observed pose).
    # Unnormalize predictions only.
    gt_t=gt_m  # [A,F,2]
    pred_arr=to_numpy(pred)
    pred_t=torch.tensor(pred_arr).float()
    if pred_t.ndim==4 and pred_t.shape[-1]==future_frames*2:
        pred_r=rearrange(pred_t,'b k a (f d)->b k a f d',f=future_frames)[...,:2]
    elif pred_t.ndim==4 and pred_t.shape[-1]==2:
        pred_r=pred_t  # [B,K,F,2] -> add agent dim
        if pred_r.shape[2] == future_frames:
            pred_r=pred_r.unsqueeze(2)
    elif pred_t.ndim==5 and pred_t.shape[-1]==2:
        pred_r=pred_t
    elif pred_t.ndim==3 and pred_t.shape[-1]==future_frames*2:
        pred_r=rearrange(pred_t,'b k (f d)->b k f d',f=future_frames)[...,:2].unsqueeze(2)
    else:
        pred_r=pred_t.reshape(pred_t.shape[0], pred_t.shape[1], -1, future_frames, 2)
    pred_r=unnormalize_min_max(pred_r,fut_min,fut_max,-1,1)
    # Convert relative to absolute by anchoring to each agent's last observed absolute position
    if init_pos is not None:
        init_a = np.asarray(init_pos)
        if init_a.ndim==1:
            init_a = init_a.reshape(1,2)
        elif init_a.ndim==2 and init_a.shape[0] != gt_t.shape[0]:
            # if [A,2] mismatched, keep as-is for broadcasting attempt
            pass
        init_t = torch.tensor(init_a, dtype=torch.float32)
        # align to [A,2] when possible
        if init_t.ndim==2 and init_t.shape[0]==gt_t.shape[0]:
            anchor = init_t.reshape(1,1,gt_t.shape[0],1,2)
            gt_abs = gt_t.unsqueeze(0) + anchor[0,0]
            pred_abs = pred_r + anchor
        else:
            anchor = init_t.reshape(1,1,-1,1,2)
            gt_abs = gt_t.unsqueeze(0) + anchor[0,0]
            pred_abs = pred_r + anchor
        gt_t = gt_abs
        pred_r = pred_abs
    diff=pred_r[0]-gt_t.unsqueeze(0)  # [K,A,F,2]
    ade_k = torch.norm(diff,-1).mean(-1)  # [K,A]
    ade_k = ade_k.mean(-1) if ade_k.ndim>1 else ade_k
    ade_min,bi=ade_k.min(0)
    return float(ade_min.detach().cpu().reshape(()).item()), int(bi.detach().cpu().reshape(()).item()), pred_r[0].detach().cpu(), gt_t.unsqueeze(0).detach().cpu()

def _draw_bundle_eth(ax,bundle,background,extent,ds_ref,scene,xlim,ylim):
    if background is not None and ds_ref is not None:
        ax.imshow(background,extent=extent,origin='upper',zorder=0,aspect='equal',interpolation='bilinear')
    obs_p=world_to_pix_native(ds_ref,bundle['obs'],scene,target_res=(w,h) if 'w' in locals() and bg is not None else None)
    if obs_p is not None and np.isfinite(obs_p).all(): ax.plot(obs_p[:,0],obs_p[:,1],'--',color='black',linewidth=1.6,marker='o',markersize=3,zorder=4)
    else:
        o=np.array(bundle['obs']).reshape(-1,2); ax.plot(o[:,0],o[:,1],'--',color='black',linewidth=1.6,zorder=4)
    for hyp in np.asarray(bundle['preds']):
        hp=world_to_pix_native(ds_ref,hyp,scene,target_res=(w,h) if 'w' in locals() and bg is not None else None)
        if hp is not None and np.isfinite(hp).all(): ax.plot(hp[:,0],hp[:,1],color=PRED_COLOR,alpha=0.12,zorder=2)
    best_p=world_to_pix_native(ds_ref,bundle['best'],scene)
    if best_p is not None and np.isfinite(best_p).all(): ax.plot(best_p[:,0],best_p[:,1],color=PRED_COLOR,linewidth=2.6,alpha=0.9,marker='o',markersize=3,zorder=5)
    gt_p=world_to_pix_native(ds_ref,bundle['gt'],scene)
    if gt_p is not None and np.isfinite(gt_p).all(): ax.plot(gt_p[:,0],gt_p[:,1],color='green',linewidth=1.8,marker='o',markersize=3,zorder=4)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

def _draw_agent_crops(ax, agent_crops, scene_name=None):
    """Disabled for ETH/UCY synchronized comparisons: show full uncropped scene context (no agent zoom inset)."""
    return


def _resolve_dataset_agents(ckpt_path, cfg_m, batch, ds_fallback):
    ac = infer_agents_from_ckpt(ckpt_path)
    if ac is not None:
        return int(ac)
    try:
        po = batch.get('past_traj_original_scale') if isinstance(batch, dict) else None
        if po is not None and torch.is_tensor(po):
            if po.ndim == 4:
                return int(po.shape[1])
            if po.ndim == 3:
                return int(po.shape[0])
    except Exception:
        pass
    try:
        return int(getattr(ds_fallback, 'past_traj_original_scale').shape[1])
    except Exception:
        return int(getattr(cfg_m.MODEL.CONTEXT_ENCODER, 'AGENTS', 2))


def _dry_run_mode_dataset(scene, mode, data_dir, base_cfg_path, device):
    cfg = Config(base_cfg_path, f"dry_{scene}_{mode}")
    cfg.device = device
    cfg.data_norm = "original"
    try:
        cfg.MODEL.CONTEXT_ENCODER.VIDEO_MODE = mode
        cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = mode != "off"
    except Exception:
        traceback.print_exc()
    ds = ETHDataset(
        cfg=cfg,
        training=False,
        data_dir=data_dir,
        subset=scene,
        rotate_time_frame=0,
        type="original",
        video_mode=mode,
    )
    return cfg, ds


def _batch_at(ds, chosen_bi):
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=seq_collate_eth
    )
    batch = None
    for _ in range(int(chosen_bi) + 1):
        batch = next(iter(loader))
    if batch is None:
        raise RuntimeError(f"Cannot load batch {chosen_bi} for dry run")
    return batch


def _anchor_overlay_metric(data_dir, scene, anchor_fid, world_xy, ds_ref, bg_shape):
    prev_fid = max(0, int(anchor_fid) - annotation_step(scene))
    curr = _decode_scene_frame(data_dir, scene, int(anchor_fid))
    prev = _decode_scene_frame(data_dir, scene, prev_fid)
    diff = cv2.absdiff(
        cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY),
    )
    mask = cv2.dilate((diff > 25).astype(np.uint8), np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(mask)
    px = world_to_pix_native(
        ds_ref, world_xy, scene, target_res=(bg_shape[1], bg_shape[0])
    )
    px = ensure_2d_coords(px)
    finite = px is not None and np.isfinite(px).all()
    if not finite or len(xs) == 0:
        return {
            "finite": bool(finite),
            "moving_pixels": int(len(xs)),
            "median_px": None,
            "frac_within_10px": None,
        }
    d = np.sqrt(
        (px[:, 0, None] - xs[None, :]) ** 2 + (px[:, 1, None] - ys[None, :]) ** 2
    ).min(axis=1)
    width, height = bg_shape[1], bg_shape[0]
    in_bounds = (
        (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    )
    return {
        "finite": True,
        "moving_pixels": int(len(xs)),
        "median_px": float(np.median(d)),
        "frac_within_10px": float((d < 10.0).mean()),
        "in_bounds_fraction": float(in_bounds.mean()),
        "anchor_frame": int(anchor_fid),
        "previous_frame": int(prev_fid),
        "background_shape": [int(bg_shape[0]), int(bg_shape[1])],
    }


def _render_dry_run_alignment(
    scene, args, out_dir, data_dir, ds_ref, batch, chosen_bi, anchor_fid, bg, cfg_ref
):
    """Validate caches/overlays without model checkpoints and save a figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_fid = _sample_start_frame_ids(ds_ref, batch, chosen_bi)
    past = int(getattr(cfg_ref, "past_frames", 8) or 8)
    obs_raw = to_numpy(batch["past_traj_original_scale"])
    gt_raw = to_numpy(batch["fut_traj_original_scale"])
    obs_xy = obs_raw[0, ..., :2] if obs_raw.ndim >= 4 else obs_raw[..., :2]
    gt_xy = gt_raw[0, ..., :2] if gt_raw.ndim >= 4 else gt_raw[..., :2]
    obs_flat = np.asarray(obs_xy).reshape(-1, 2)
    gt_flat = np.asarray(gt_xy).reshape(-1, 2)

    mode_reports = {}
    for mode in VARIANTS:
        cfg_m, ds_m = _dry_run_mode_dataset(
            scene, mode, data_dir, args.cfg, args.device
        )
        batch_m = _batch_at(ds_m, chosen_bi)
        mode_start = _sample_start_frame_ids(ds_m, batch_m, chosen_bi)
        report = {
            "mode": mode,
            "start_frame": int(mode_start),
            "past_frames": int(getattr(cfg_m, "past_frames", 8)),
            "use_video": bool(getattr(ds_m, "use_video", False)),
        }
        if mode != "off":
            feats_root = getattr(
                cfg_m.MODEL.CONTEXT_ENCODER,
                "VIDEO_FEATURES_ROOT",
                "features/resnet18",
            )
            lookup = FrameFeatureLookup.from_root(feats_root, scene=scene)
            step = annotation_step(scene)
            frame_ids = observed_frame_ids(mode_start, scene, report["past_frames"])
            missing = [int(f) for f in frame_ids if int(f) not in lookup.frame_to_row]
            window = lookup.window(
                mode_start, report["past_frames"], stride=step, policy="nearest"
            )
            exact = all(int(f) in lookup.frame_to_row for f in frame_ids)
            z = to_numpy(batch_m.get("z_video_global"))
            if z is None:
                raise RuntimeError(f"{scene}/{mode}: dataset did not emit z_video_global")
            z = z[0]
            if z.shape != window.shape:
                raise RuntimeError(
                    f"{scene}/{mode}: z_video_global shape {z.shape} != lookup {window.shape}"
                )
            if not np.isfinite(z).all():
                raise RuntimeError(f"{scene}/{mode}: non-finite z_video_global")
            if mode == "full":
                if not np.allclose(z, window, atol=1e-6):
                    raise RuntimeError(
                        f"{scene}/{mode}: z_video_global does not match exact frame lookup"
                    )
            else:
                ids = lookup.sorted_frame_ids
                sample = ids[:: max(1, ids.size // 8)][:8]
                static = np.stack(
                    [lookup(int(f)) for f in sample], axis=0
                ).mean(axis=0)
                expected = np.repeat(static[None, :], z.shape[0], axis=0)
                if not np.allclose(z, expected, atol=1e-6):
                    raise RuntimeError(
                        f"{scene}/{mode}: z_video_global does not match cached static mean"
                    )
            report.update(
                {
                    "cache_root": str(feats_root),
                    "observed_frame_ids": [int(f) for f in frame_ids],
                    "frame_ids_exact": bool(exact),
                    "missing_frame_ids": missing,
                    "features_shape": list(z.shape),
                    "features_finite": bool(np.isfinite(z).all()),
                    "features_nonzero": bool(np.abs(z).sum() > 0),
                    "timesteps_identical": bool(
                        np.allclose(z, np.repeat(z[:1], z.shape[0], axis=0), atol=1e-6)
                    ),
                }
            )
        mode_reports[mode] = report

    anchor_world = np.asarray(obs_xy).reshape(-1, np.asarray(obs_xy).shape[-1])[
        -1, :2
    ].reshape(1, 2)
    alignment = _anchor_overlay_metric(
        data_dir, scene, anchor_fid, anchor_world, ds_ref, bg.shape
    )

    width, height = bg.shape[1], bg.shape[0]
    obs_p = ensure_2d_coords(
        world_to_pix_native(ds_ref, obs_flat, scene, target_res=(width, height))
    )
    gt_p = ensure_2d_coords(
        world_to_pix_native(ds_ref, gt_flat, scene, target_res=(width, height))
    )
    if obs_p is None or gt_p is None or not np.isfinite(obs_p).all():
        raise RuntimeError(f"{scene}: projection produced non-finite overlay points")

    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.6))
    extent = (0, width, height, 0)
    for ax, mode in zip(axes[:3], VARIANTS):
        ax.imshow(bg, extent=extent, origin="upper", zorder=0, aspect="equal")
        ax.plot(
            obs_p[:, 0],
            obs_p[:, 1],
            "--",
            color="black",
            linewidth=1.6,
            marker="o",
            markersize=3,
            zorder=4,
        )
        ax.plot(
            gt_p[:, 0],
            gt_p[:, 1],
            color="green",
            linewidth=1.8,
            marker="o",
            markersize=3,
            zorder=4,
        )
        ap = ensure_2d_coords(
            world_to_pix_native(
                ds_ref, anchor_world, scene, target_res=(width, height)
            )
        )
        if ap is not None:
            ax.plot(
                ap[:, 0],
                ap[:, 1],
                "x",
                color="blue",
                markersize=10,
                mew=2.2,
                zorder=6,
            )
        rep = mode_reports[mode]
        cache_txt = (
            "no cache"
            if mode == "off"
            else f"frames={rep['features_shape'][0]} exact={rep['frame_ids_exact']}"
        )
        ax.set_title(f"{mode} | {cache_txt}")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        assert_pixel_frame_alignment(obs_p, gt_p, obs_p)
    axes[3].axis("off")
    report = {
        "scene": scene,
        "dry_run": True,
        "start_frame": int(start_fid),
        "anchor_frame": int(anchor_fid),
        "background_frame": int(anchor_fid),
        "background_video": str(scene_video_path(data_dir, scene)),
        "modes": mode_reports,
        "alignment": alignment,
        "overlay_points": {
            "observed": int(len(obs_p)),
            "future": int(len(gt_p)),
            "obs_finite": bool(np.isfinite(obs_p).all()),
            "gt_finite": bool(np.isfinite(gt_p).all()),
        },
    }
    axes[3].text(
        0.0,
        0.98,
        json.dumps(report, indent=2, sort_keys=True)[:4000],
        family="monospace",
        fontsize=7,
        va="top",
    )
    fig.tight_layout()
    png_path = out_dir / f"{scene}_synchronized_comparison_dry_run.png"
    json_path = out_dir / f"{scene}_synchronized_comparison_dry_run.json"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"[dry-run] {scene}: saved {png_path} and {json_path}", flush=True)
    print(f"[dry-run] {scene}: alignment={alignment}", flush=True)


def render_scene_eth(scene,args,out_dir,data_dir,device):
    cfg_ref=Config(args.cfg,f'sync_{scene}'); cfg_ref.device=device
    # Reference dataset is used for geometry/frame ids only; keep raw metres.
    cfg_ref.data_norm='original'
    ds_ref=ETHDataset(cfg=cfg_ref,training=False,data_dir=data_dir,subset=scene,rotate_time_frame=0,type='original',video_mode='full')
    loader=DataLoader(ds_ref,batch_size=1,shuffle=False,num_workers=0,collate_fn=seq_collate_eth)
    batch=None
    chosen_bi = args.batch_idx
    if getattr(args, 'batch_idx', 0) == 0:
        # scan for valid window with motion
        for bi in range(min(500,len(ds_ref))):
            try:
                it=iter(loader)
                for _ in range(bi+1): batch=next(it)
                po=batch.get('past_traj_original_scale')
                if po is not None and torch.is_tensor(po):
                    po_n = po.detach().cpu().numpy()
                    if po_n.shape[1]>0 and po_n.reshape(-1,2).std()>1e-6:
                        chosen_bi = bi
                        break
            except Exception:
                traceback.print_exc()
                continue
    args.batch_idx = chosen_bi
    # reload batch at chosen_bi
    try:
        it=iter(loader)
        for _ in range(chosen_bi+1): batch=next(it)
    except Exception:
        traceback.print_exc()
    if batch is None: print(f'[ETH-UCY] {scene}: no valid batch'); return
    anchor_fid = _anchor_frame_id(ds_ref, batch, chosen_bi, cfg_ref, data_dir, scene)
    bg = load_img(None, data_dir=data_dir, scene=scene, anchor_frame=anchor_fid)
    if getattr(args, "dry_run", False):
        _render_dry_run_alignment(
            scene,
            args,
            out_dir,
            data_dir,
            ds_ref,
            batch,
            chosen_bi,
            anchor_fid,
            bg,
            cfg_ref,
        )
        return
    bundles={}
    for mode in VARIANTS:
        ckpt,run_dir=find_checkpoint(scene,mode,args.cor_fm_root,args.ckpt_dir)
        if not ckpt: print(f'  SKIP {scene}/{mode} missing'); continue
        # Build config from the run that produced this checkpoint so embedding shapes match natively.
        cfg_path=load_run_config(run_dir,args.cfg); cfg_m=Config(cfg_path,f'sync_{scene}_{mode}'); cfg_m.device=device
        _apply_run_bounds(cfg_m, run_dir=run_dir, ckpt=ckpt)
        try: cfg_m.MODEL.CONTEXT_ENCODER.VIDEO_MODE=mode; cfg_m.MODEL.CONTEXT_ENCODER.USE_VIDEO=(mode!='off')
        except: pass
        ac=_resolve_dataset_agents(ckpt,cfg_m,batch,ds_ref)
        force_set_agents(cfg_m,ac)
        # Variant conditioning isolation: load per-mode dataset/batch
        ds_m=ETHDataset(cfg=cfg_m,training=False,data_dir=data_dir,subset=scene,rotate_time_frame=0,type='original',video_mode=mode)
        loader_m=DataLoader(ds_m,batch_size=1,shuffle=False,num_workers=0,collate_fn=seq_collate_eth)
        it_m=iter(loader_m); batch_m=None
        for _ in range(chosen_bi+1): batch_m=next(it_m)
        try:
            model=ETHMotionTransformer(model_config=cfg_m.MODEL,logger=NullLogger(),config=cfg_m)
            denoiser=FlowMatcher(cfg_m,model,logger=NullLogger())
            denoiser=load_ckpt_flexible(ckpt,denoiser).to(device).eval()
            for p in denoiser.parameters(): p.requires_grad_(False)
            x_data={k:v.to(device) for k,v in (batch_m or batch).items() if k in ['past_traj_original_scale','fut_traj','agent_mask','z_video_global']}
            gt=(batch_m or batch)['fut_traj_original_scale'].to(device).float()
            pred=euler_sample(denoiser,cfg_m,x_data,getattr(cfg_m,'future_frames',12),device,args.seed)
            try:
                print(f'[DIAG] {scene}/{mode} raw_pred min/max: {pred.min().item():.6f}/{pred.max().item():.6f}')
            except Exception:
                traceback.print_exc()
            # get init pos (use mode batch): last observed absolute world position for anchoring
            init_pos=None
            try:
                po_b = (batch_m or batch).get('past_traj_original_scale')
                if po_b is not None:
                    po_n = to_numpy(po_b)
                    if po_n.ndim==4:  # [B,A,P,D]
                        init_pos = po_n[0, :, -1, :2]
                    elif po_n.ndim==3:  # [A,P,D] or [B,P,D]
                        init_pos = po_n[:, -1, :2]
            except Exception:
                traceback.print_exc()
            ade_min,bi,pred_m,gt_m=best_of_k_eth(pred,gt,cfg_m,ds_m,init_pos=init_pos)
            try:
                print(f'[DIAG] {scene}/{mode} unnorm_pred_m min/max: {np.asarray(pred_m).min():.6f}/{np.asarray(pred_m).max():.6f}, ade_min={float(np.asarray(ade_min).reshape(())):.6f}')
            except Exception:
                traceback.print_exc()
            pm = to_numpy(pred_m)
            bi_i = int(np.asarray(bi).reshape(()))
            btm = to_numpy(pred_m[bi_i]) if np.asarray(pred_m).ndim >= 1 else to_numpy(pred_m)
            gtm = to_numpy(gt_m)
            # normalize shapes for plotting (keep all agents as flattened trajs)
            if gtm.ndim == 4:  # [1,A,F,2]
                gtm = gtm[0]
            if gtm.ndim == 3 and gtm.shape[-1] != 2:
                gtm = gtm.reshape(-1, 2)
            if btm.ndim == 3 and btm.shape[-1] == 2:
                pass  # [A,F,2]
            bundles[mode]={'preds':pm,'best':np.asarray(btm).reshape(-1,2),'gt':np.asarray(gtm).reshape(-1,2),'ade':float(np.asarray(ade_min).reshape(())), 'batch': batch_m or batch, 'ds': ds_m}
            del model,denoiser; torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            traceback.print_exc(); print(f'  {mode} err: {e}')
    if not bundles: return
    obs_raw=batch['past_traj_original_scale']
    obs_xy=to_numpy(obs_raw)
    if obs_xy.ndim==4: obs_xy=obs_xy[0,0]
    elif obs_xy.ndim==3: obs_xy=obs_xy[0]
    # past_traj_original_scale already stores absolute world metres: never
    # unnormalize it (that was the source of the ~1/11 ground-truth collapse).
    bundles_list=bundles
    # attach obs to all
    for k in bundles_list: bundles_list[k]['obs']=obs_xy
    # world limits
    arrs=[np.array(bundles_list[k]['obs']).reshape(-1,2) for k in bundles_list]+[np.array(bundles_list[k]['gt']).reshape(-1,2) for k in bundles_list]+[np.array(bundles_list[k]['preds']).reshape(-1,2) for k in bundles_list]+[np.array(bundles_list[k]['best']).reshape(-1,2) for k in bundles_list]
    ptsw=np.concatenate(arrs,0); low,high=ptsw.min(0),ptsw.max(0); sp=np.maximum(high-low,1e-6); pad=0.08*sp; xlimw=(low[0]-pad[0],high[0]+pad[0]); ylimw=(low[1]-pad[1],high[1]+pad[1])
    # pix limits via native
    xlimp,ylimp=xlimw,ylimw
    if bg is not None and ds_ref is not None:
        c1=world_to_pix_native(ds_ref,np.array([[xlimw[0],ylimw[0]],[xlimw[1],ylimw[0]],[xlimw[1],ylimw[1]],[xlimw[0],ylimw[1]]]),scene)
        if c1 is not None and np.isfinite(c1).all(): cx0,cx1=np.min(c1[:,0]),np.max(c1[:,0]); cy0,cy1=np.min(c1[:,1]),np.max(c1[:,1]); xlimp=(cx0,cx1); ylimp=(cy1,cy0)
    
    order=[v for v in VARIANTS if v in bundles_list]
    fig,axes=plt.subplots(1,4,figsize=(22.0,5.6))
    extent=None
    if bg is not None:
        h,w=bg.shape[:2]; extent=(0,w,h,0)
    for ax,variant in zip(axes[:3],order):
        b=bundles_list[variant]
        if bg is not None and ds_ref is not None:
            ax.imshow(bg,extent=extent,origin='upper',zorder=0,aspect='equal',interpolation='bilinear')
        # draw in pixel space for overlay
        obs_p=world_to_pix_native(ds_ref,b['obs'],scene,target_res=(w,h) if bg is not None else None)
        obs_p=ensure_2d_coords(obs_p)
        if obs_p is not None and np.isfinite(obs_p).all(): ax.plot(obs_p[:,0],obs_p[:,1],'--',color='black',linewidth=1.6,marker='o',markersize=3,zorder=4)
        preds_b = np.asarray(b['preds'])
        # normalize preds_b to list of [T,2]
        if preds_b.ndim == 3:  # [K,T,2]
            hyps = preds_b
        elif preds_b.ndim == 4:  # [B,K,T,2] or similar
            hyps = preds_b.reshape(-1, preds_b.shape[-2], preds_b.shape[-1])
        elif preds_b.ndim == 2:  # [T,2] single
            hyps = preds_b[None,...]
        else:
            hyps = preds_b.reshape(-1, preds_b.shape[-2], preds_b.shape[-1]) if preds_b.shape[-1]==2 else preds_b
        for hyp in hyps:
            hyp_arr = np.asarray(hyp).reshape(-1, 2)
            hp=world_to_pix_native(ds_ref,hyp_arr,scene,target_res=(w,h) if bg is not None else None)
            hp=ensure_2d_coords(hp)
            if hp is not None and np.isfinite(hp).all() and hp.shape[0]>1: ax.plot(hp[:,0],hp[:,1],color=PRED_COLOR,alpha=0.12,zorder=2)
        best_arr = np.asarray(b['best'])
        gt_arr = np.asarray(b['gt'])
        if best_arr.ndim==3: best_arr = best_arr.reshape(-1,2)
        else: best_arr = best_arr.reshape(-1,2)
        if gt_arr.ndim==3: gt_arr = gt_arr.reshape(-1,2)
        else: gt_arr = gt_arr.reshape(-1,2)
        best_p=world_to_pix_native(ds_ref,best_arr,scene,target_res=(w,h) if bg is not None else None)
        best_p=ensure_2d_coords(best_p)
        if best_p is not None and np.isfinite(best_p).all() and best_p.shape[0]>1: ax.plot(best_p[:,0],best_p[:,1],color=PRED_COLOR,linewidth=2.6,alpha=0.9,marker='o',markersize=3,zorder=5)
        gt_p=world_to_pix_native(ds_ref,gt_arr,scene,target_res=(w,h) if bg is not None else None)
        gt_p=ensure_2d_coords(gt_p)
        if obs_p is not None and gt_p is not None and best_p is not None:
            assert_pixel_frame_alignment(obs_p, gt_p, best_p)
        if gt_p is not None and np.isfinite(gt_p).all() and gt_p.shape[0]>1: ax.plot(gt_p[:,0],gt_p[:,1],color='green',linewidth=1.8,marker='o',markersize=3,zorder=4)
        if bg is not None:
            ax.set_xlim(0, w); ax.set_ylim(h, 0)
        else:
            ax.set_xlim(*xlimp); ax.set_ylim(*ylimp)
        ax.set_title(f"{PANEL_TITLES[variant]}\nADE$_{{min}}$ = {b['ade']:.2f}",fontsize=11)
        ax.set_xlabel('x [px]'); ax.set_ylabel('y [px]'); ax.tick_params(labelsize=8)
        try:
            _draw_agent_crops(ax, ((bundles_list.get('full') or {}).get('batch') or batch).get('agent_crops'))
        except Exception:
            traceback.print_exc()
    # panel 4: Grad-CAM
    ax=axes[3]
    cam_overlay_done=False
    gcam_err=None
    if bg is not None and _HAS_GCAM and ds_ref is not None:
        try:
            import torchvision
            from PIL import Image
            from torchvision import transforms

            # load backbone
            backbone, dim, preprocess = _make_backbone('resnet18', pretrained=True, device=device)
            backbone.eval()
            # load observed frames directly from video.avi at scene annotation step
            gcam_ds = (bundles_list.get('full') or {}).get('ds') or ds_ref
            gcam_batch = (bundles_list.get('full') or {}).get('batch') or batch
            bidx = locals().get('chosen_bi', args.batch_idx)
            try:
                if isinstance(gcam_batch, dict) and 'index' in gcam_batch:
                    idx_v = gcam_batch['index']
                    bidx_cand = int(idx_v.detach().cpu().numpy().reshape(-1)[0]) if hasattr(idx_v, 'detach') else int(idx_v)
                    if 0 <= bidx_cand < len(gcam_ds):
                        bidx = bidx_cand
            except Exception:
                traceback.print_exc()
            start_fid = _sample_start_frame_ids(gcam_ds, gcam_batch, bidx)
            frame_ids_obs = observed_frame_ids(start_fid, scene, int(cfg_ref.past_frames))
            if frame_ids_obs[-1] != int(anchor_fid):
                raise AssertionError(
                    f"Grad-CAM anchor {anchor_fid} != final observed frame {frame_ids_obs[-1]}"
                )
            frames_t_list=[]; frames_rgb_list=[]
            for fid in frame_ids_obs:
                img_rgb = _decode_scene_frame(data_dir, scene, fid)
                frames_rgb_list.append(img_rgb)
                pil=Image.fromarray(img_rgb); frames_t_list.append(preprocess(pil))
            if frames_t_list:
                frames_t=torch.stack(frames_t_list,0).to(device)
                # get ckpt for full (instantiate with checkpoint/run config to match embedding dims)
                ckpt_f,run_f=find_checkpoint(scene,'full',args.cor_fm_root,args.ckpt_dir)
                if ckpt_f:
                    cfg_f=Config(load_run_config(run_f,args.cfg),f'gcam_{scene}')
                    cfg_f.device=device
                    _apply_run_bounds(cfg_f, run_dir=run_f, ckpt=ckpt_f)
                    try: cfg_f.MODEL.CONTEXT_ENCODER.VIDEO_MODE='full'; cfg_f.MODEL.CONTEXT_ENCODER.USE_VIDEO=True
                    except: pass
                    ds_full = gcam_ds; ac_f=_resolve_dataset_agents(ckpt_f,cfg_f,gcam_batch,ds_full)
                    force_set_agents(cfg_f,ac_f)
                    model_f=ETHMotionTransformer(model_config=cfg_f.MODEL,logger=NullLogger(),config=cfg_f)
                    den_f=FlowMatcher(cfg_f,model_f,logger=NullLogger())
                    den_f=load_ckpt_flexible(ckpt_f,den_f).to(device).eval()
                for p in den_f.parameters(): p.requires_grad_(False)
                gcam_batch = (bundles_list.get('full') or {}).get('batch') or batch; x_gc={k:v.to(device) for k,v in gcam_batch.items() if k in ['past_traj_original_scale','fut_traj','agent_mask','z_video_global']}
                # ensure z_video_global present for gradcam if expected
                if 'z_video_global' not in x_gc and getattr(ds_full,'z_video_global',None) is not None:
                    try:
                        bidx_cand = locals().get('chosen_bi', args.batch_idx)
                        x_gc['z_video_global'] = ds_full.z_video_global[bidx_cand:bidx_cand+1].to(device)
                    except Exception:
                        traceback.print_exc()
                gcam_batch = (bundles_list.get('full') or {}).get('batch') or batch; gt_gc=gcam_batch['fut_traj_original_scale'].to(device).float()
                res_gc=gradcam_pass(backbone,den_f,cfg_f,x_gc,frames_t,gt_gc,cfg_f.future_frames,device,args.seed)
                cam=res_gc['cam']
                if cam is not None and len(cam)>0 and len(frames_rgb_list)>0:
                    cam2d=cam[-1]
                    H,W=frames_rgb_list[-1].shape[:2]
                    cam_up=cv2.resize(cam2d,(W,H),interpolation=cv2.INTER_LINEAR)
                    heat_rgb,_=_apply_jet(cam_up,0.4)
                    base=frames_rgb_list[-1].astype(np.float32)/255.0
                    overlay=(1-0.4)*base+0.4*heat_rgb
                    overlay=np.clip(overlay*255,0,255).astype(np.uint8)
                    hbg,wbg=overlay.shape[:2]
                    extent=(0,wbg,hbg,0)
                    ax.imshow(overlay,extent=extent,origin='upper',zorder=0,aspect='equal',interpolation='bilinear')
                    cam_overlay_done=True
                    # draw trajs
                    fb=bundles_list.get('full') or list(bundles_list.values())[0]
                    obs_p=world_to_pix_native(ds_ref,fb['obs'],scene,target_res=(w,h) if 'w' in locals() and bg is not None else None); obs_p=ensure_2d_coords(obs_p)
                    if obs_p is not None and np.isfinite(obs_p).all(): ax.plot(obs_p[:,0],obs_p[:,1],'--',color='black',linewidth=1.6,marker='o',markersize=3,zorder=4)
                    gt_p=world_to_pix_native(ds_ref,fb['gt'],scene,target_res=(w,h) if 'w' in locals() and bg is not None else None); gt_p=ensure_2d_coords(gt_p)
                    if gt_p is not None and np.isfinite(gt_p).all(): ax.plot(gt_p[:,0],gt_p[:,1],color='lime',linewidth=1.8,marker='o',markersize=3,zorder=4)
                    best_p=world_to_pix_native(ds_ref,fb['best'],scene,target_res=(w,h) if 'w' in locals() and bg is not None else None); best_p=ensure_2d_coords(best_p)
                    if best_p is not None and np.isfinite(best_p).all(): ax.plot(best_p[:,0],best_p[:,1],color='white',linewidth=1.8,alpha=0.95,linestyle='--',zorder=5)
                    ax.set_xlim(*xlimp); ax.set_ylim(*ylimp)
                    ax.set_title(f"Grad-CAM (full)\nanchor frame",fontsize=11)
                    ax.set_xlabel('x [px]'); ax.set_ylabel('y [px]'); ax.tick_params(labelsize=8)
                    try:
                        _draw_agent_crops(ax, ((bundles_list.get('full') or {}).get('batch') or batch).get('agent_crops'))
                    except Exception:
                        traceback.print_exc()
        except Exception as e:
            gcam_err = e
            traceback.print_exc()
            print(f'  gradcam err: {e}')
    if cam_overlay_done:
        pass
    else:
        if bg is not None:
            ax.imshow(bg,extent=extent,origin='upper',zorder=0,aspect='equal',interpolation='bilinear',alpha=0.3)
        err_msg = f"Grad-CAM FAILED\n{gcam_err}" if 'gcam_err' in locals() and gcam_err is not None else "Grad-CAM unavailable"
        ax.text(0.5,0.5,err_msg,ha='center',va='center',fontsize=8,color='red',transform=ax.transAxes,wrap=True)
        ax.set_xlim(*xlimp); ax.set_ylim(*ylimp)
        ax.set_title(f"Grad-CAM (full)",fontsize=11)
        ax.set_xlabel('x [px]'); ax.set_ylabel('y [px]'); ax.tick_params(labelsize=8)
    fig.suptitle(f'ETH/UCY synchronized comparison | {scene.upper()} | window idx {args.batch_idx}',fontsize=13)
    fig.legend(handles=_legend_handles(),loc='lower center',ncol=4,frameon=False,fontsize=10,bbox_to_anchor=(0.5,0.0))
    fig.tight_layout(rect=(0,0.06,1,0.94))
    out_dir.mkdir(parents=True,exist_ok=True)
    save_path=out_dir/f'{scene}_synchronized_comparison.png'
    fig.savefig(save_path,dpi=300,bbox_inches='tight'); plt.close(fig)
    print(f'  saved {save_path}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',default='cfg/eth_ucy/cor_fm.yml')
    p.add_argument('--ckpt-dir',default='results_eth_ucy/loso_full_matrix/checkpoints')
    p.add_argument('--cor-fm-root',default='results_eth_ucy/cor_fm')
    p.add_argument('--data-dir',default='/home/efrei_stage/Desktop/tu-clausthal/datasets')
    p.add_argument('--out-dir',default='results_eth_ucy/synchronized_comparisons')
    p.add_argument('--scenes',nargs='+',default=list(SCENES))
    p.add_argument('--batch-idx',type=int,default=0)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate feature caches and projected overlays without checkpoints; saves *_dry_run.{png,json}.',
    )
    args=p.parse_args()
    data_dir=args.data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"External --data-dir not found: {data_dir}. "
            "Expected TU-Clausthal datasets root (e.g. /home/efrei_stage/Desktop/tu-clausthal/datasets). "
            "No repo-internal fallback is used."
        )
    out_dir=Path(args.out_dir)
    for scene in args.scenes: render_scene_eth(scene,args,out_dir,data_dir,args.device)
    print('Done.')

if __name__=='__main__': main()
