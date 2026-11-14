#!/usr/bin/env python3
"""
Homography verification using repo's own projection logic.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
from utils.config import Config
from utils.eth_ucy_geometry import scene_video_path, world_to_pixel as geo_world_to_pixel
from utils.video_reader import get_video_reader


class NullLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


def load_homography(data_dir, subset):
    h_paths = [
        os.path.join(data_dir, subset, 'H.txt'),
        os.path.join(REPO_ROOT, 'data', 'eth_ucy', subset, 'H.txt'),
        os.path.join(REPO_ROOT, 'data', 'eth_ucy', 'homography', f'{subset}_H.txt'),
    ]
    for hp in h_paths:
        if os.path.exists(hp):
            try:
                return np.loadtxt(hp).astype(float)
            except Exception:
                pass
    return None


def find_ref(data_dir, subset):
    # Native video.avi is the only reference visual source.
    return scene_video_path(data_dir, subset)


def load_img(p, scene=None, frame_id=0):
    if p is None:
        return None
    reader = get_video_reader(p)
    if scene is not None:
        from utils.eth_ucy_geometry import video_index_for_frame

        frame_id = video_index_for_frame(scene, frame_id)
    return reader.read(int(frame_id), color="rgb")


def world_to_pixel_apply_h(H, pts, scene=None):
    if H is None or pts is None: return None
    pts = np.asarray(pts)
    if pts.size == 0: return pts
    orig = pts.shape
    out = geo_world_to_pixel(pts.reshape(-1, 2), H, scene=scene)
    return out.reshape(orig)


def extract_xy_agents(traj_raw, frames):
    if traj_raw is None: return None
    traj = np.asarray(traj_raw)
    if traj.size == 0: return None
    if traj.dtype == object:
        res=[]
        try:
            for i in range(traj.shape[0]):
                for j in range(traj.shape[1] if traj.ndim>1 else 1):
                    t = np.asarray(traj[i,j]) if traj.ndim>1 else np.asarray(traj[i])
                    if t.ndim==2 and t.shape[-1]>=2 and t.shape[0]==frames: res.append(t[...,:2].reshape(1,frames,2))
                    elif t.ndim==1 and t.size>=frames*2:
                        if t.size%6==0: res.append(t.reshape(frames,6)[...,:2].reshape(1,frames,2))
                        elif t.size%2==0: tt=t.reshape(frames,t.size//frames); res.append(tt[...,:2].reshape(1,frames,2))
        except: pass
        if not res: return None
        try: return np.concatenate(res,0)
        except: return None
    if traj.ndim==4 and traj.shape[-1]>=2: return traj.reshape(traj.shape[0]*traj.shape[1],traj.shape[2],traj.shape[-1])[...,:2]
    if traj.ndim==3:
        if traj.shape[1]==frames and traj.shape[-1]>=2: return traj[...,:2]
        if traj.shape[-1]%6==0 and traj.shape[-1]//6==frames: return traj.reshape(traj.shape[0],frames,6)[...,:2]
        if traj.shape[-1]%2==0 and traj.shape[-1]//2==frames: return traj.reshape(traj.shape[0],frames,2)
    if traj.ndim==2:
        if traj.shape[-1]%6==0: return traj.reshape(traj.shape[0],traj.shape[-1]//6,6)[...,:2]
        if traj.shape[-1]%2==0: return traj.reshape(traj.shape[0],traj.shape[-1]//2,2)
    return None


def generate_diagnostics(subset, data_dir, out_dir):
    print(f"\n{'='*60}\nGenerating homography diagnostics for {subset}\n{'='*60}")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config('cfg/eth_ucy/cor_fm.yml', f'hdiag_{subset}'); cfg.device='cpu'
    cfg.data_norm='original'  # only raw metres are needed for diagnostics
    try:
        ds = ETHDataset(cfg=cfg, training=False, data_dir=data_dir, subset=subset, rotate_time_frame=0, type='original', video_mode='off')
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=seq_collate_eth)
    except Exception as e:
        print(f"[{subset}] {e}"); return False
    H = load_homography(data_dir, subset)
    bg = load_img(find_ref(data_dir, subset), scene=subset, frame_id=0)
    
    all_pw, all_fw = [], []
    best = None; best_n=-1
    for bi,b in enumerate(loader):
        try:
            p = extract_xy_agents(b.get('past_traj_original_scale'), cfg.past_frames)
            f = extract_xy_agents(b.get('fut_traj_original_scale'), cfg.future_frames)
            if p is None or f is None: continue
            A=min(p.shape[0],f.shape[0])
            # Future targets are stored relative to the last observed pose; make
            # them absolute so past/future share one world frame before projection.
            f_abs = f[:A] + p[:A, -1:, :]
            if A>best_n: best_n=A; best=(p[:A].copy(),f_abs.copy())
            all_pw.append(p[:A]); all_fw.append(f_abs)
        except: continue
    print(f"[{subset}] Max agents: {best_n}")
    
    # Fig 1
    fig1,ax1 = plt.subplots(1,1,figsize=(12,9))
    fig1.suptitle(f'{subset.upper()} - Global Homography Point Cloud')
    if bg is not None:
        ax1.imshow(bg)
        ax1.set_xlim(0,bg.shape[1]); ax1.set_ylim(bg.shape[0],0)
    else:
        ax1.set_facecolor('white')
    if H is not None and all_pw:
        try:
            ap=np.concatenate(all_pw,0); af=np.concatenate(all_fw,0)
            pp=world_to_pixel_apply_h(H,ap,scene=subset); fp=world_to_pixel_apply_h(H,af,scene=subset)
            if pp is not None:
                msk=np.isfinite(pp).all(1)
                if bg is not None: msk &= (pp[:,0]>=0)&(pp[:,0]<bg.shape[1])&(pp[:,1]>=0)&(pp[:,1]<bg.shape[0])
                if msk.sum(): ax1.scatter(pp[msk,0],pp[msk,1],s=1,c='crimson',alpha=0.35)
            if fp is not None:
                msk=np.isfinite(fp).all(1)
                if bg is not None: msk &= (fp[:,0]>=0)&(fp[:,0]<bg.shape[1])&(fp[:,1]>=0)&(fp[:,1]<bg.shape[0])
                if msk.sum(): ax1.scatter(fp[msk,0],fp[msk,1],s=1,c='cyan',alpha=0.35)
        except Exception as e: print(e)
    plt.tight_layout(); plt.savefig(out_dir/f'{subset}_1_global_pointcloud_density.png',dpi=300,bbox_inches='tight'); plt.close()
    
    # Fig 2
    fig2,(ax2p,ax2w)=plt.subplots(1,2,figsize=(18,8))
    fig2.suptitle(f'{subset.upper()} - Dense Multi-Agent Snapshot ({best_n} agents)')
    if bg is not None:
        ax2p.imshow(bg); ax2p.set_xlim(0,bg.shape[1]); ax2p.set_ylim(bg.shape[0],0)
    if best and H is not None:
        pb,fb=best
        pp=world_to_pixel_apply_h(H,pb,scene=subset); fp=world_to_pixel_apply_h(H,fb,scene=subset)
        if pp is not None:
            for a in range(pp.shape[0]):
                ax2p.plot(pp[a,:,0],pp[a,:,1],'--',c='cyan',lw=1.5,alpha=0.9)
        if fp is not None:
            for a in range(fp.shape[0]):
                ax2p.plot(fp[a,:,0],fp[a,:,1],'-',c='yellow',lw=2,alpha=1.0)
        for a in range(pb.shape[0]):
            ax2w.plot(pb[a,:,0],pb[a,:,1],'--',c='blue',lw=1.5)
            ax2w.plot(fb[a,:,0],fb[a,:,1],'-',c='red',lw=2)
    ax2p.set_title('Pixel Space (Past dashed / Future solid)'); ax2p.axis('off')
    ax2w.set_title('World Coordinates (m)'); ax2w.grid(True,ls='--',alpha=0.5); ax2w.axis('equal')
    plt.tight_layout(); plt.savefig(out_dir/f'{subset}_2_dense_multagent_snapshot.png',dpi=300,bbox_inches='tight'); plt.close()
    print(f"[{subset}] Done"); return True


def main():
    data_dir='/home/efrei_stage/Desktop/tu-clausthal/datasets'
    if not os.path.exists(data_dir):
        rd=REPO_ROOT/'data/eth_ucy'
        if rd.exists(): data_dir=str(rd)
    out_dir=REPO_ROOT/'results_eth_ucy/homography_verification'
    for s in ['eth','hotel','univ','zara1','zara2']:
        generate_diagnostics(s,data_dir,out_dir)


if __name__=='__main__': main()
