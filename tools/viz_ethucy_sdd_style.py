#!/usr/bin/env python3
"""
ETH/UCY visualization in SDD-style report format.
Compares off/static/full modes with Best-of-K trajectories and Grad-CAM overlays.
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
from pathlib import Path

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

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from einops import rearrange
from torch.utils.data import DataLoader
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from utils.normalization import unnormalize_min_max
from utils.config import Config

class NullLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass

def to_numpy(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.array(x)

def extract_traj_xy(t, past_frames=None, future_frames=None):
    """Robustly extract [T, 2] or [A, T, 2] from various shapes."""
    if t is None: return None
    if isinstance(t, torch.Tensor): t = t.detach().cpu()
    t = np.array(t)
    # [B, A, T, D]
    if t.ndim == 4:
        if t.shape[-1] >= 2:
            return t[..., :2]
        # maybe flattened
    if t.ndim == 3:
        if t.shape[-1] >= 2:
            return t[..., :2]
        # try to infer
        if past_frames and future_frames and t.shape[-1] == (past_frames + future_frames)*2:
            return t.reshape(t.shape[0], past_frames+future_frames, 2)[..., :2]
    if t.ndim == 2:
        if t.shape[-1] >= 2:
            # maybe [T, D] or [A, D]
            if t.shape[-1] % 2 == 0 and past_frames and t.shape[-1]//2 == past_frames:
                return t.reshape(t.shape[0], past_frames, 2)[..., :2]
            return t.reshape(-1, t.shape[-1])[..., :2]
    return None

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
        else:
            sd = ckpt
        if sd is None: return None
        for key in sd.keys():
            if 'agent_query_embedding.weight' in key:
                return int(sd[key].shape[0])
        for key in sd.keys():
            if 'agent_order_embedding.weight' in key:
                return int(sd[key].shape[0])
    except Exception:
        pass
    return None

def force_set_agents(cfg_m, agents):
    try:
        cfg_m.MODEL.CONTEXT_ENCODER.AGENTS = agents
    except Exception:
        pass
    try:
        cfg_m.MODEL.AGENTS = agents
    except Exception:
        pass
    try:
        cfg_m.AGENTS = agents
    except Exception:
        pass
    try:
        if hasattr(cfg_m, 'yml_dict'):
            yd = cfg_m.yml_dict
            if 'MODEL' in yd and isinstance(yd['MODEL'], dict):
                if 'CONTEXT_ENCODER' in yd['MODEL'] and isinstance(yd['MODEL']['CONTEXT_ENCODER'], dict):
                    yd['MODEL']['CONTEXT_ENCODER']['AGENTS'] = agents
                yd['MODEL']['AGENTS'] = agents
            yd['AGENTS'] = agents
    except Exception:
        pass

def copy_norm_stats(src, dst):
    for k in ['fut_traj_min','fut_traj_max','past_traj_min','past_traj_max','norm_mode']:
        if hasattr(src,k):
            try:
                setattr(dst,k,getattr(src,k))
            except Exception:
                pass
    try:
        if hasattr(dst,'yml_dict'):
            yd = dst.yml_dict
            for k in ['fut_traj_min','fut_traj_max','past_traj_min','past_traj_max','norm_mode']:
                if hasattr(src,k): yd[k]=getattr(src,k)
    except Exception:
        pass

def load_ckpt_flexible(ckpt_path, model):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = None
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "ema_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
        if sd is None:
            sd = ckpt
    else:
        sd = ckpt
    from collections import OrderedDict
    nd = OrderedDict()
    for kk, vv in sd.items():
        kk2 = kk[7:] if kk.startswith("module.") else kk
        nd[kk2] = vv
    missing, unexpected = model.load_state_dict(nd, strict=False)
    return model

def find_checkpoint_info(scene, mode, cor_fm_root="results_eth_ucy/cor_fm", flat_dir="results_eth_ucy/loso_full_matrix/checkpoints"):
    flat = os.path.join(flat_dir, f"{scene}_{mode}.pt")
    if os.path.exists(flat): 
        return flat, None
    vm = f"vm{mode}"
    candidates = []
    for pat in [os.path.join(cor_fm_root, f"*_{scene}_{vm}*"), os.path.join(cor_fm_root, f"*{scene}*{vm}*")]:
        candidates.extend(glob.glob(pat))
    if not candidates:
        for d in os.listdir(cor_fm_root):
            dp = os.path.join(cor_fm_root,d)
            if os.path.isdir(dp) and scene in d and vm in d:
                candidates.append(dp)
    for c in candidates:
        if os.path.isdir(c):
            bp = os.path.join(c,'models','checkpoint_best.pt')
            lp = os.path.join(c,'models','checkpoint_last.pt')
            ckpt = bp if os.path.exists(bp) else (lp if os.path.exists(lp) else None)
            if ckpt:
                return ckpt, c
        elif c.endswith('.pt'):
            return c, None
    return None, None

def load_run_config(run_dir, default_cfg='cfg/eth_ucy/cor_fm.yml'):
    if run_dir and os.path.isdir(run_dir):
        for name in ['cor_fm_updated.yml','config_updated.yml','cor_fm.yml','config.yml']:
            p = os.path.join(run_dir, name)
            if os.path.exists(p): return p
            p2 = os.path.join(run_dir, 'code_backup', name)
            if os.path.exists(p2): return p2
    return default_cfg

def euler_sample(denoiser, cfg, x_data, num_frames, device, seed):
    K = int(getattr(cfg,'denoising_head_preds',20))
    S = int(getattr(cfg,'sampling_steps',10))
    dt = 1.0/S
    torch.manual_seed(seed)
    if device.startswith("cuda"): torch.cuda.manual_seed_all(seed)
    y_t = torch.randn((1,K,1,num_frames*2), device=device)
    for idx in range(S):
        cur_t = dt*idx
        batched_t = torch.full((1,), float(cur_t), device=device)
        preds = denoiser.model_predictions(y_t, x_data, batched_t, flag_print=False)
        y_t = y_t + preds.pred_vel * dt
    return y_t

def best_of_k(pred, gt, cfg, ds=None):
    fut_min = getattr(cfg,'fut_traj_min',None)
    fut_max = getattr(cfg,'fut_traj_max',None)
    if ds is not None:
        fut_min = getattr(ds,'fut_traj_min',fut_min)
        fut_max = getattr(ds,'fut_traj_max',fut_max)
    if fut_min is None: fut_min = -1.0
    if fut_max is None: fut_max = 1.0
    future_frames = int(getattr(cfg,'future_frames',12))
    
    # gt
    if isinstance(gt, torch.Tensor):
        if gt.dim() == 4:
            gt_m = gt[0,0,...]
        elif gt.dim() == 3:
            gt_m = gt[0,...]
            if gt_m.dim()==2 and gt_m.shape[0]>future_frames and gt_m.shape[0]<100:
                gt_m = gt_m[:future_frames]
            elif gt_m.dim()==2 and gt_m.shape[1]==future_frames*2:
                gt_m = gt_m.reshape(future_frames,2)
            elif gt_m.dim()==3:
                gt_m = gt_m[0]
        else:
            gt_m = gt
    else:
        gt_m = np.array(gt)
        if gt_m.ndim==4: gt_m = gt_m[0,0]
        elif gt_m.ndim==3: gt_m = gt_m[0,0] if gt_m.shape[0]==1 else gt_m[0]
    gt_m_t = torch.tensor(gt_m).float()
    if gt_m_t.dim()==2 and gt_m_t.shape[1]==2:
        gt_m_t = gt_m_t.unsqueeze(0).unsqueeze(0)
    elif gt_m_t.dim()==2 and gt_m_t.shape[1]==future_frames*2:
        gt_m_t = rearrange(gt_m_t.unsqueeze(0),'b a (f d)->b a f d',f=future_frames)[...,:2]
    gt_m_u = unnormalize_min_max(gt_m_t[...,:2], fut_min, fut_max, -1, 1)[0,0]
    
    # pred
    if isinstance(pred, torch.Tensor):
        if pred.dim()==4 and pred.shape[-1]==future_frames*2:
            pred_r = rearrange(pred,'b k a (f d)->b k a f d',f=future_frames)[...,:2]
        elif pred.dim()==3 and pred.shape[-1]==future_frames*2:
            pred_r = rearrange(pred,'b k (f d)->b k f d',f=future_frames)[...,:2].unsqueeze(2)
        elif pred.dim()==4 and pred.shape[-1]==2:
            pred_r = pred
        else:
            pred_r = pred
    else:
        pred_r = torch.tensor(np.array(pred)).float()
        if pred_r.dim()==4 and pred_r.shape[-1]==future_frames*2:
            pred_r = rearrange(pred_r,'b k a (f d)->b k a f d',f=future_frames)[...,:2]
    pred_r = unnormalize_min_max(pred_r, fut_min, fut_max, -1, 1)
    diff = pred_r[0] - gt_m_u.unsqueeze(0).unsqueeze(0)
    ade = torch.norm(diff, dim=-1).mean(dim=-1)
    if ade.dim()==2: ade = ade.mean(dim=1)
    ade_min, best_idx = ade.min(dim=0)
    return ade_min, int(best_idx.item()), pred_r[0], gt_m_u.unsqueeze(0).unsqueeze(0)

def load_h(H_path):
    if not os.path.exists(H_path): return None
    try: return np.loadtxt(H_path).astype(float)
    except: return None

def world_to_pix(H, pts):
    if H is None or pts is None: return None
    pts = np.asarray(pts).reshape(-1,2)
    ph = np.c_[pts, np.ones((len(pts),1))] @ H.T
    w = ph[:,2:3]; w[np.abs(w)<1e-12]=np.nan
    out = (ph[:,:2]/w)
    # reshape back
    try:
        return out.reshape(np.asarray(pts).shape[:-1]+(2,))
    except:
        return out

def find_ref_img(data_dir, subset):
    for base in [os.path.join(data_dir,subset,'visual_data'), os.path.join(_REPO_ROOT,'data','eth_ucy',subset,'visual_data')]:
        if os.path.exists(base):
            for f in sorted(os.listdir(base)):
                if f.lower().endswith(('.jpg','.png','.jpeg')):
                    return os.path.join(base,f)
    return None

def load_img(p):
    if not p or not os.path.exists(p): return None
    try:
        img = Image.open(p).convert('RGB'); return np.array(img)
    except:
        try:
            img=cv2.imread(p); return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        except: return None

def make_backbone(device='cpu'):
    import torchvision
    from torchvision import transforms
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return m.to(device), preprocess

def gradcam_for_full(ckpt_path, cfg, batch, data_dir, subset, device, seed=42, ds=None):
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=NullLogger(), config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=NullLogger())
    denoiser = load_ckpt_flexible(ckpt_path, denoiser).to(device).eval()
    for p in denoiser.parameters(): p.requires_grad_(False)
    backbone, preprocess = make_backbone(device)
    
    feats=[]; grads=[]
    h1 = backbone.layer4.register_forward_hook(lambda m,i,o: feats.append(o))
    h2 = backbone.layer4.register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
    
    try:
        vis_dir = os.path.join(data_dir,subset,'visual_data')
        if not os.path.exists(vis_dir):
            vis_dir = os.path.join(_REPO_ROOT,'data','eth_ucy',subset,'visual_data')
        cand = sorted([os.path.join(vis_dir,f) for f in os.listdir(vis_dir) if f.lower().endswith('.jpg')])[:getattr(cfg,'past_frames',8)]
        if len(cand)<getattr(cfg,'past_frames',8) and cand: cand = cand*getattr(cfg,'past_frames',8)
        if not cand: raise FileNotFoundError(f'No frames in {vis_dir}')
        frames_t=[]
        for fp in cand[:getattr(cfg,'past_frames',8)]:
            pil = Image.open(fp).convert('RGB'); frames_t.append(preprocess(pil))
        frames_t = torch.stack(frames_t).to(device)
        
        z_list=[]
        for i in range(frames_t.shape[0]):
            ft = frames_t[i:i+1].requires_grad_(True)
            rep = backbone(ft)
            z_list.append(rep)
        z_cat = torch.cat(z_list,0)
        z_mean = z_cat.mean(0,keepdim=True).unsqueeze(0)
        
        x_data = {k:v.to(device) for k,v in batch.items() if k in ['past_traj_original_scale','fut_traj','agent_mask']}
        x_data['z_video_global']=z_mean
        gt = batch['fut_traj_original_scale'].to(device).float()
        
        pred = euler_sample(denoiser,cfg,x_data,getattr(cfg,'future_frames',12),device,seed)
        future_frames = getattr(cfg,'future_frames',12)
        if pred.dim() == 4 and pred.shape[-1] == future_frames*2:
            pred_m = rearrange(pred,'b k a (f d)->b k a f d',f=future_frames)[...,:2]
        elif pred.dim()==3 and pred.shape[-1]==future_frames*2:
            pred_m = rearrange(pred,'b k (f d)->b k f d',f=future_frames)[...,:2].unsqueeze(2)
        else:
            pred_m = pred
        fut_min = getattr(cfg,'fut_traj_min',getattr(ds,'fut_traj_min',-1.0) if ds else -1.0)
        fut_max = getattr(cfg,'fut_traj_max',getattr(ds,'fut_traj_max',1.0) if ds else 1.0)
        pred_m = unnormalize_min_max(pred_m,fut_min,fut_max,-1,1)
        if gt.dim()==4: gt_m = gt[0,0,...][:future_frames,:2].unsqueeze(0).unsqueeze(0)
        elif gt.dim()==3: gt_m = gt[0,...][:future_frames,:2].unsqueeze(0).unsqueeze(0)
        else: gt_m = gt
        gt_m = unnormalize_min_max(gt_m,fut_min,fut_max,-1,1)
        diff = pred_m - gt_m.unsqueeze(1)
        ade = torch.norm(diff,dim=-1).mean(-1).mean(-1)
        score = -ade.mean()
        denoiser.model.zero_grad()
        score.backward()
        A=feats[-1]; G=grads[-1]
        alpha = G.mean(dim=(-2,-1),keepdim=True)
        cam = torch.relu((alpha*A).sum(1))
        cam = cam - cam.amin(dim=(-2,-1),keepdim=True)
        cam = cam / (cam.amax(dim=(-2,-1),keepdim=True)+1e-12)
        return cam.detach().cpu().numpy(), cand[0]
    finally:
        h1.remove(); h2.remove()

def overlay_cam(frame_rgb, cam2d, alpha=0.45):
    h,w = frame_rgb.shape[:2]
    cam_u8 = np.uint8(255*np.clip(cam2d,0,1))
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)
    heat_up = cv2.resize(heat,(w,h),interpolation=cv2.INTER_LINEAR)
    base = frame_rgb.astype(np.float32)/255.0
    heatf = heat_up.astype(np.float32)/255.0
    out = (1-alpha)*base + alpha*heatf
    return np.clip(out*255,0,255).astype(np.uint8)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='cfg/eth_ucy/cor_fm.yml')
    p.add_argument('--ckpt-dir', default='results_eth_ucy/loso_full_matrix/checkpoints')
    p.add_argument('--cor-fm-root', default='results_eth_ucy/cor_fm')
    p.add_argument('--data-dir', default='/home/efrei_stage/Desktop/tu-clausthal/datasets')
    p.add_argument('--out-dir', default='results_eth_ucy/loso_full_matrix/viz')
    p.add_argument('--scenes', nargs='+', default=['eth','hotel','univ','zara1','zara2'])
    p.add_argument('--modes', nargs='+', default=['off','static','full'])
    p.add_argument('--batch-idx', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    if not os.path.exists(args.data_dir):
        args.data_dir = str(_REPO_ROOT/'data/eth_ucy')
    
    for scene in args.scenes:
        print(f'\n=== {scene.upper()} ===')
        cfg_ref = Config(args.cfg, f'viz_{scene}')
        cfg_ref.device = args.device
        try:
            ds_ref = ETHDataset(cfg=cfg_ref, training=False, data_dir=args.data_dir, subset=scene, rotate_time_frame=0, type='original', video_mode='full')
            copy_norm_stats(ds_ref, cfg_ref)
            loader = DataLoader(ds_ref,batch_size=1,shuffle=False,num_workers=0,collate_fn=seq_collate_eth)
            it=iter(loader)
            for i in range(args.batch_idx+1): batch=next(it)
        except Exception as e:
            print(f'[{scene}] data err: {e}'); continue
        
        H = load_h(os.path.join(args.data_dir,scene,'H.txt')) or load_h(os.path.join(_REPO_ROOT,'data','eth_ucy',scene,'H.txt'))
        bg = load_img(find_ref_img(args.data_dir,scene))
        
        preds_all={}
        for mode in args.modes:
            ckpt, run_dir = find_checkpoint_info(scene, mode, args.cor_fm_root, args.ckpt_dir)
            if not ckpt:
                print(f'  missing checkpoint for {scene}/{mode}'); continue
            print(f'  [{mode}] {ckpt}')
            cfg_path = load_run_config(run_dir, args.cfg)
            print(f'    using config: {cfg_path}')
            cfg_m = Config(cfg_path, f'viz_{scene}_{mode}')
            cfg_m.device=args.device
            try:
                cfg_m.MODEL.CONTEXT_ENCODER.VIDEO_MODE = mode
                cfg_m.MODEL.CONTEXT_ENCODER.USE_VIDEO = mode != 'off'
            except: pass
            agents_ckpt = infer_agents_from_ckpt(ckpt)
            if agents_ckpt is not None:
                print(f'    overriding MODEL.AGENTS to {agents_ckpt} from checkpoint')
                force_set_agents(cfg_m, agents_ckpt)
            try:
                ds_m = ETHDataset(cfg=cfg_m, training=False, data_dir=args.data_dir, subset=scene, rotate_time_frame=0, type='original', video_mode=mode)
                copy_norm_stats(ds_m, cfg_m)
            except Exception as e:
                print(f'    ds_m err: {e}')
                ds_m = ds_ref
            try:
                model = ETHMotionTransformer(model_config=cfg_m.MODEL, logger=NullLogger(), config=cfg_m)
                denoiser = FlowMatcher(cfg_m, model, logger=NullLogger())
                denoiser = load_ckpt_flexible(ckpt, denoiser).to(args.device).eval()
                for p in denoiser.parameters(): p.requires_grad_(False)
                x_data = {k:v.to(args.device) for k,v in batch.items() if k in ['past_traj_original_scale','fut_traj','agent_mask','z_video_global']}
                gt = batch['fut_traj_original_scale'].to(args.device).float()
                pred = euler_sample(denoiser,cfg_m,x_data,getattr(cfg_m,'future_frames',12),args.device,1234)
                _,best_idx,pred_m,gt_m = best_of_k(pred,gt,cfg_m,ds_m)
                if pred_m is None or gt_m is None: 
                    print(f'    skipped {mode} due to shape'); continue
                bi = int(best_idx)
                try:
                    preds_all[mode] = {'best': pred_m[bi].detach().cpu().numpy(), 'gt': gt_m[0,0].detach().cpu().numpy()}
                except:
                    preds_all[mode] = {'best': np.array(pred_m)[bi], 'gt': np.array(gt_m).reshape(-1,2)}
                del model,denoiser; torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                print(f'  [{mode}] err: {e}')
                import traceback; traceback.print_exc()
        
        if not preds_all: continue
        obs_raw = batch.get('past_traj_original_scale')
        obs_m = None
        if obs_raw is not None:
            obs_xy = extract_traj_xy(obs_raw, getattr(cfg_ref,'past_frames',8))
            if obs_xy is not None:
                if obs_xy.ndim==4: obs_xy = obs_xy[0]
                if obs_xy.ndim==3: obs_xy = obs_xy[0]
                past_min = getattr(cfg_ref,'past_traj_min',getattr(ds_ref,'past_traj_min',-1.0))
                past_max = getattr(cfg_ref,'past_traj_max',getattr(ds_ref,'past_traj_max',1.0))
                obs_m_t = unnormalize_min_max(torch.tensor(obs_xy).float().unsqueeze(0).unsqueeze(0), past_min, past_max, -1, 1)[0,0].numpy()
                obs_m = obs_m_t
        
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,9))
        fig.suptitle(f'{scene.upper()} - ETH/UCY Trajectory Comparison (Best-of-K)',fontsize=16,fontweight='bold')
        for ax in [ax1,ax2]:
            if bg is not None:
                ax.imshow(bg); ax.set_xlim(0,bg.shape[1]); ax.set_ylim(bg.shape[0],0)
            else: ax.grid(True,ls='--',alpha=0.5)
            ax.axis('off')
        ax=ax1
        if bg is not None and H is not None and obs_m is not None and len(obs_m)>0 and 'full' in preds_all:
            try:
                obsp = world_to_pix(H, obs_m)
                if obsp is not None and obsp.ndim==2:
                    ax.plot(obsp[:,0],obsp[:,1],'--',color='black',lw=2,label='History'); ax.scatter(obsp[-1,0],obsp[-1,1],c='black',s=60)
                gtp = world_to_pix(H,preds_all['full']['gt'])
                if gtp is not None:
                    gtp2 = gtp.reshape(-1,2)
                    ax.plot(gtp2[:,0],gtp2[:,1],'-',color='green',lw=2,label='GT'); ax.scatter(gtp2[-1,0],gtp2[-1,1],c='green',s=60)
                for mode,col in [('off','blue'),('static','orange'),('full','red')]:
                    if mode in preds_all:
                        bp = world_to_pix(H,preds_all[mode]['best'])
                        if bp is not None:
                            bp2 = bp.reshape(-1,2)
                            ax.plot(bp2[:,0],bp2[:,1],'-',color=col,lw=2.5,label=f'Best-of-K {mode}')
                ax.set_title('Pixel Space (projected via H.txt)',fontsize=14)
            except Exception as e:
                print(e)
        else:
            if obs_m is not None and len(obs_m)>0:
                om = np.array(obs_m).reshape(-1,2)
                ax.plot(om[:,0],om[:,1],'--',lw=2,label='History')
            if 'full' in preds_all:
                gt2 = np.array(preds_all['full']['gt']).reshape(-1,2)
                ax.plot(gt2[:,0],gt2[:,1],'-',color='green',lw=2,label='GT')
            for mode,col in [('off','blue'),('static','orange'),('full','red')]:
                if mode in preds_all:
                    bp2 = np.array(preds_all[mode]['best']).reshape(-1,2)
                    ax.plot(bp2[:,0],bp2[:,1],'-',color=col,lw=2.5,label=f'Best-of-K {mode}')
            ax.set_title('World Coordinates',fontsize=14)
        ax.legend(loc='best',framealpha=0.9)
        
        ax=ax2
        if obs_m is not None and len(obs_m)>0:
            om = np.array(obs_m).reshape(-1,2)
            ax.plot(om[:,0],om[:,1],'--',lw=2,label='History')
        if 'full' in preds_all:
            gt2 = np.array(preds_all['full']['gt']).reshape(-1,2)
            ax.plot(gt2[:,0],gt2[:,1],'-',color='green',lw=2,label='GT')
        for mode,col in [('off','blue'),('static','orange'),('full','red')]:
            if mode in preds_all:
                bp2 = np.array(preds_all[mode]['best']).reshape(-1,2)
                ax.plot(bp2[:,0],bp2[:,1],'-',color=col,lw=2.5,label=f'Best-of-K {mode}')
        ax.set_title('World Coordinates (meters)',fontsize=14)
        ax.grid(True,ls='--',alpha=0.5)
        ax.axis('equal')
        ax.legend(loc='best',framealpha=0.9)
        plt.tight_layout()
        plt.savefig(out_dir/f'{scene}_trajectory_comparison_sdd_style.png',dpi=300)
        plt.close()
        print(f'  saved trajectory comparison')
        
        if 'full' in preds_all:
            try:
                ckpt_f, run_dir_f = find_checkpoint_info(scene,'full',args.cor_fm_root,args.ckpt_dir)
                if not ckpt_f: raise FileNotFoundError('no ckpt')
                cfg_path_f = load_run_config(run_dir_f, args.cfg)
                cfg_f = Config(cfg_path_f,f'gc_{scene}'); cfg_f.device=args.device
                try:
                    cfg_f.MODEL.CONTEXT_ENCODER.VIDEO_MODE='full'; cfg_f.MODEL.CONTEXT_ENCODER.USE_VIDEO=True
                except: pass
                agents_ckpt_f = infer_agents_from_ckpt(ckpt_f)
                if agents_ckpt_f is not None:
                    force_set_agents(cfg_f, agents_ckpt_f)
                try:
                    ds_f = ETHDataset(cfg=cfg_f, training=False, data_dir=args.data_dir, subset=scene, rotate_time_frame=0, type='original', video_mode='full')
                    copy_norm_stats(ds_f, cfg_f)
                except Exception:
                    ds_f = ds_ref
                cam,cand_fp = gradcam_for_full(ckpt_f,cfg_f,batch,args.data_dir,scene,args.device,1234,ds_f)
                frm = load_img(cand_fp)
                if frm is not None and len(cam)>0:
                    fig,axes=plt.subplots(2,4,figsize=(20,10))
                    fig.suptitle(f'{scene.upper()} - ResNet-18 Grad-CAM (full mode)',fontsize=16,fontweight='bold')
                    for i in range(min(8,len(cam))):
                        ax=axes[i//4,i%4]
                        ov = overlay_cam(frm, cam[i], 0.45)
                        ax.imshow(ov); ax.set_title(f'Obs frame {i+1}',fontsize=11); ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(out_dir/f'{scene}_gradcam_full_sdd_style.png',dpi=300)
                    plt.close()
                    print(f'  saved gradcam')
            except Exception as e:
                print(f'  gradcam err: {e}')
    
    print(f'\nDone. Outputs in {out_dir}')

if __name__=='__main__': main()
