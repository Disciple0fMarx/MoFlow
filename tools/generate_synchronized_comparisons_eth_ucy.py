#!/usr/bin/env python3
"""Synchronized 4-panel qualitative comparison for ETH/UCY video-conditioning ablation.

Direct ETH/UCY equivalent of tools/generate_synchronized_comparisons.py.
"""
from __future__ import annotations

from einops import rearrange
import argparse
import gc
import json
import os
import sys
from pathlib import Path

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

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from utils.normalization import unnormalize_min_max
from utils.config import Config

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

def copy_norm_stats(src,dst):
    for k in ['fut_traj_min','fut_traj_max','past_traj_min','past_traj_max']:
        if hasattr(src,k):
            try: setattr(dst,k,getattr(src,k))
            except: pass

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
    model.load_state_dict(nd, strict=False)
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

def find_ref_img(data_dir,scene):
    for base in [os.path.join(data_dir,scene,'visual_data'),os.path.join(_REPO_ROOT,'data','eth_ucy',scene,'visual_data')]:
        if os.path.exists(base):
            for f in sorted(os.listdir(base)):
                if f.lower().endswith(('.jpg','.png','.jpeg')): return os.path.join(base,f)
    return None

def load_img(p):
    if not p or not os.path.exists(p): return None
    try:
        img = cv2.imread(p)
        if img is None: return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except: return None

def world_to_pix_native(ds, pts, scene=None):
    if pts is None or ds is None: return None
    try:
        pts_arr = np.asarray(pts, dtype=float)
        if pts_arr.size == 0: return pts_arr
        pts_t = torch.tensor(pts_arr, dtype=torch.float32)
        res = ds.world_to_pixel(pts_t, scene=scene)
        if isinstance(res, torch.Tensor): 
            res = res.detach().cpu().numpy()
        return res
    except Exception as e:
        print(f'[viz] world_to_pixel failed: {e}')
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
        c1 = world_to_pix_native(ds_ref, np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]), scene)
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

def best_of_k_eth(pred,gt,cfg,ds=None):
    fut_min=getattr(cfg,'fut_traj_min',getattr(ds,'fut_traj_min',-1.0) if ds else -1.0)
    fut_max=getattr(cfg,'fut_traj_max',getattr(ds,'fut_traj_max',1.0) if ds else 1.0)
    future_frames=int(getattr(cfg,'future_frames',12))
    gt_arr=to_numpy(gt)
    if gt_arr.ndim==4: gt_m=gt_arr[0,0,:future_frames,:2]
    elif gt_arr.ndim==3: gt_m=gt_arr[0,:future_frames,:2]
    else: gt_m=gt_arr.reshape(-1,2)[:future_frames]
    gt_t=unnormalize_min_max(torch.tensor(gt_m).float().unsqueeze(0).unsqueeze(0),fut_min,fut_max,-1,1)[0,0]
    pred_arr=to_numpy(pred)
    if pred_arr.ndim==4 and pred_arr.shape[-1]==future_frames*2:
        pred_r=rearrange(torch.tensor(pred_arr).float(),'b k a (f d)->b k a f d',f=future_frames)[...,:2]
    elif pred_arr.ndim==4 and pred_arr.shape[-1]==2: pred_r=torch.tensor(pred_arr).float()
    elif pred_arr.ndim==3 and pred_arr.shape[-1]==future_frames*2:
        pred_r=rearrange(torch.tensor(pred_arr).float(),'b k (f d)->b k f d',f=future_frames)[...,:2].unsqueeze(2)
    else: pred_r=torch.tensor(pred_arr).float()
    pred_r=unnormalize_min_max(pred_r,fut_min,fut_max,-1,1)
    diff=pred_r[0]-gt_t.unsqueeze(0).unsqueeze(0); ade=torch.norm(diff,-1).mean(-1)
    if ade.dim()==2: ade=ade.mean(1)
    ade_min,bi=ade.min(0); return ade_min,int(bi.item()),pred_r[0],gt_t.unsqueeze(0).unsqueeze(0)

def _draw_bundle_eth(ax,bundle,background,extent,ds_ref,scene,xlim,ylim):
    if background is not None and ds_ref is not None:
        ax.imshow(background,extent=extent,origin='upper',zorder=0,aspect='equal',interpolation='bilinear')
    obs_p=world_to_pix_native(ds_ref,bundle['obs'],scene)
    if obs_p is not None and np.isfinite(obs_p).all(): ax.plot(obs_p[:,0],obs_p[:,1],'--',color='black',linewidth=1.6,marker='o',markersize=3,zorder=4)
    else:
        o=np.array(bundle['obs']).reshape(-1,2); ax.plot(o[:,0],o[:,1],'--',color='black',linewidth=1.6,zorder=4)
    for hyp in np.asarray(bundle['preds']):
        hp=world_to_pix_native(ds_ref,hyp,scene)
        if hp is not None and np.isfinite(hp).all(): ax.plot(hp[:,0],hp[:,1],color=PRED_COLOR,alpha=0.12,zorder=2)
    best_p=world_to_pix_native(ds_ref,bundle['best'],scene)
    if best_p is not None and np.isfinite(best_p).all(): ax.plot(best_p[:,0],best_p[:,1],color=PRED_COLOR,linewidth=2.6,alpha=0.9,marker='o',markersize=3,zorder=5)
    gt_p=world_to_pix_native(ds_ref,bundle['gt'],scene)
    if gt_p is not None and np.isfinite(gt_p).all(): ax.plot(gt_p[:,0],gt_p[:,1],color='green',linewidth=1.8,marker='o',markersize=3,zorder=4)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

def render_scene_eth(scene,args,out_dir,data_dir,device):
    cfg_ref=Config(args.cfg,f'sync_{scene}'); cfg_ref.device=device
    ds_ref=ETHDataset(cfg=cfg_ref,training=False,data_dir=data_dir,subset=scene,rotate_time_frame=0,type='original',video_mode='full')
    copy_norm_stats(ds_ref,cfg_ref)
    loader=DataLoader(ds_ref,batch_size=1,shuffle=False,num_workers=0,collate_fn=seq_collate_eth)
    batch=None
    for bi in range(min(80,len(ds_ref))):
        it=iter(loader)
        for _ in range(bi+1): batch=next(it)
        po=batch.get('past_traj_original_scale')
        if po is not None and torch.is_tensor(po) and po.shape[1]>0 and po.abs().sum()>0: break
    if batch is None: print(f'[ETH-UCY] {scene}: no valid batch'); return
    bg=load_img(find_ref_img(data_dir,scene))
    bundles={}
    for mode in VARIANTS:
        ckpt,run_dir=find_checkpoint(scene,mode,args.cor_fm_root,args.ckpt_dir)
        if not ckpt: print(f'  SKIP {scene}/{mode} missing'); continue
        cfg_path=load_run_config(run_dir,args.cfg); cfg_m=Config(cfg_path,f'sync_{scene}_{mode}'); cfg_m.device=device
        try: cfg_m.MODEL.CONTEXT_ENCODER.VIDEO_MODE=mode; cfg_m.MODEL.CONTEXT_ENCODER.USE_VIDEO=(mode!='off')
        except: pass
        ac=infer_agents_from_ckpt(ckpt)
        if ac is not None: force_set_agents(cfg_m,ac)
        try: ds_m=ETHDataset(cfg=cfg_m,training=False,data_dir=data_dir,subset=scene,rotate_time_frame=0,type='original',video_mode=mode); copy_norm_stats(ds_m,cfg_m)
        except: ds_m=ds_ref
        try:
            model=ETHMotionTransformer(model_config=cfg_m.MODEL,logger=NullLogger(),config=cfg_m)
            denoiser=FlowMatcher(cfg_m,model,logger=NullLogger())
            denoiser=load_ckpt_flexible(ckpt,denoiser).to(device).eval()
            for p in denoiser.parameters(): p.requires_grad_(False)
            x_data={k:v.to(device) for k,v in batch.items() if k in ['past_traj_original_scale','fut_traj','agent_mask','z_video_global']}
            gt=batch['fut_traj_original_scale'].to(device).float()
            pred=euler_sample(denoiser,cfg_m,x_data,getattr(cfg_m,'future_frames',12),device,args.seed)
            ade_min,bi,pred_m,gt_m=best_of_k_eth(pred,gt,cfg_m,ds_m)
            bundles[mode]={'preds':to_numpy(pred_m),'best':to_numpy(pred_m[bi]),'gt':to_numpy(gt_m[0,0] if gt_m.ndim==4 else (gt_m[0] if gt_m.ndim==3 else gt_m)),'ade':float(ade_min)}
            del model,denoiser; torch.cuda.empty_cache(); gc.collect()
        except Exception as e: print(f'  {mode} err: {e}')
    if not bundles: return
    obs_raw=batch['past_traj_original_scale']
    obs_xy=to_numpy(obs_raw)
    if obs_xy.ndim==4: obs_xy=obs_xy[0,0]
    elif obs_xy.ndim==3: obs_xy=obs_xy[0]
    past_min=getattr(cfg_ref,'past_traj_min',getattr(ds_ref,'past_traj_min',-1.0)); past_max=getattr(cfg_ref,'past_traj_max',getattr(ds_ref,'past_traj_max',1.0))
    obs_xy=unnormalize_min_max(torch.tensor(obs_xy).float().unsqueeze(0).unsqueeze(0),past_min,past_max,-1,1)[0,0].numpy()
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
        obs_p=world_to_pix_native(ds_ref,b['obs'],scene)
        if obs_p is not None and np.isfinite(obs_p).all(): ax.plot(obs_p[:,0],obs_p[:,1],'--',color='black',linewidth=1.6,marker='o',markersize=3,zorder=4)
        for hyp in np.asarray(b['preds']):
            hp=world_to_pix_native(ds_ref,hyp,scene)
            if hp is not None and np.isfinite(hp).all(): ax.plot(hp[:,0],hp[:,1],color=PRED_COLOR,alpha=0.12,zorder=2)
        best_p=world_to_pix_native(ds_ref,b['best'],scene)
        if best_p is not None and np.isfinite(best_p).all(): ax.plot(best_p[:,0],best_p[:,1],color=PRED_COLOR,linewidth=2.6,alpha=0.9,marker='o',markersize=3,zorder=5)
        gt_p=world_to_pix_native(ds_ref,b['gt'],scene)
        if gt_p is not None and np.isfinite(gt_p).all(): ax.plot(gt_p[:,0],gt_p[:,1],color='green',linewidth=1.8,marker='o',markersize=3,zorder=4)
        ax.set_xlim(*xlimp); ax.set_ylim(*ylimp)
        ax.set_title(f"{PANEL_TITLES[variant]}\nADE$_{{min}}$ = {b['ade']:.2f}",fontsize=11)
        ax.set_xlabel('x [px]'); ax.set_ylabel('y [px]'); ax.tick_params(labelsize=8)
    # panel 4
    ax=axes[3]
    if bg is not None:
        ax.imshow(bg,extent=extent,origin='upper',zorder=0,aspect='equal',interpolation='bilinear')
    fb=bundles_list.get('full') or list(bundles_list.values())[0]
    obs_p=world_to_pix_native(ds_ref,fb['obs'],scene)
    if obs_p is not None and np.isfinite(obs_p).all(): ax.plot(obs_p[:,0],obs_p[:,1],'--',color='black',linewidth=1.6,marker='o',markersize=3,zorder=4)
    gt_p=world_to_pix_native(ds_ref,fb['gt'],scene)
    if gt_p is not None and np.isfinite(gt_p).all(): ax.plot(gt_p[:,0],gt_p[:,1],color='green',linewidth=1.8,marker='o',markersize=3,zorder=4)
    best_p=world_to_pix_native(ds_ref,fb['best'],scene)
    if best_p is not None and np.isfinite(best_p).all(): ax.plot(best_p[:,0],best_p[:,1],color=PRED_COLOR,linewidth=2.6,alpha=0.9,marker='o',markersize=3,zorder=5)
    ax.set_xlim(*xlimp); ax.set_ylim(*ylimp)
    ax.set_title(f"Grad-CAM (full)\nETH/UCY overlay",fontsize=11)
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
    args=p.parse_args()
    data_dir=args.data_dir
    if not os.path.exists(data_dir): data_dir=str(_REPO_ROOT/'data/eth_ucy')
    out_dir=Path(args.out_dir)
    for scene in args.scenes: render_scene_eth(scene,args,out_dir,data_dir,args.device)
    print('Done.')

if __name__=='__main__': main()
