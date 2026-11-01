#!/usr/bin/env python3
"""
Rigorous two-part homography diagnostic (smoke-test style).
1) Global point cloud density map over reference frame
2) Dense multi-agent snapshot in pixel and world coords
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


class NullLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


def load_homography(data_dir, subset):
    h_paths = [
        os.path.join(data_dir, subset, 'H.txt'),
        os.path.join(data_dir, subset, f'{subset}_H.txt'),
        os.path.join(REPO_ROOT, 'data', 'eth_ucy', subset, 'H.txt'),
        os.path.join(REPO_ROOT, 'data', 'eth_ucy', 'homography', f'{subset}_H.txt'),
    ]
    for hp in h_paths:
        if os.path.exists(hp):
            try:
                H = np.loadtxt(hp)
                return H
            except Exception:
                pass
    return None


def find_reference_frame(data_dir, subset):
    candidates = [
        os.path.join(data_dir, subset, 'visual_data'),
        os.path.join(REPO_ROOT, 'data', 'eth_ucy', subset, 'visual_data'),
    ]
    for c in candidates:
        if os.path.exists(c):
            for fn in sorted(os.listdir(c)):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(c, fn)
    return None


def load_image(path):
    if not path or not os.path.exists(path):
        return None
    try:
        from PIL import Image
        return np.array(Image.open(path))
    except Exception:
        try:
            import cv2
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception:
            return None


def world_to_pixel(H, points):
    if H is None or points is None:
        return None
    pts = np.asarray(points)
    if pts.size == 0:
        return pts
    orig_shape = pts.shape
    pts_flat = pts.reshape(-1, 2)
    ones = np.ones((pts_flat.shape[0], 1))
    homo = np.concatenate([pts_flat, ones], axis=1)
    pix = (H @ homo.T).T
    pix = pix[:, :2] / (pix[:, 2:3] + 1e-12)
    return pix.reshape(orig_shape)


def extract_xy_agents(traj_raw, frames):
    if traj_raw is None:
        return None
    traj = np.asarray(traj_raw)
    if traj.size == 0:
        return None
    if traj.dtype == object:
        all_agents = []
        try:
            for i in range(traj.shape[0]):
                for j in range(traj.shape[1] if traj.ndim > 1 else 1):
                    t = np.asarray(traj[i, j]) if traj.ndim > 1 else np.asarray(traj[i])
                    if t.ndim == 2 and t.shape[-1] >= 2 and t.shape[0] == frames:
                        all_agents.append(t[..., :2].reshape(1, frames, 2))
                    elif t.ndim == 1 and t.size >= frames * 2:
                        if t.size % 6 == 0:
                            tt = t.reshape(frames, 6)
                            all_agents.append(tt[..., :2].reshape(1, frames, 2))
                        elif t.size % 2 == 0:
                            tt = t.reshape(frames, t.size // frames)
                            if tt.shape[-1] >= 2:
                                all_agents.append(tt[..., :2].reshape(1, frames, 2))
        except Exception:
            pass
        if all_agents:
            try:
                return np.concatenate(all_agents, axis=0)
            except Exception:
                return None
        return None
    if traj.ndim == 4:
        B, A, T, F = traj.shape
        if F >= 2:
            return traj.reshape(B * A, T, F)[..., :2]
    elif traj.ndim == 3:
        if traj.shape[1] == frames and traj.shape[-1] >= 2:
            return traj[..., :2]
        if traj.shape[-1] % 6 == 0 and traj.shape[-1] // 6 == frames:
            return traj.reshape(traj.shape[0], frames, 6)[..., :2]
        if traj.shape[-1] % 2 == 0 and traj.shape[-1] // 2 == frames:
            return traj.reshape(traj.shape[0], frames, 2)
    elif traj.ndim == 2:
        if traj.shape[-1] % 6 == 0:
            n_t = traj.shape[-1] // 6
            return traj.reshape(traj.shape[0], n_t, 6)[..., :2]
        if traj.shape[-1] % 2 == 0:
            n_t = traj.shape[-1] // 2
            return traj.reshape(traj.shape[0], n_t, 2)
    return None


def generate_diagnostics(subset, data_dir, out_dir):
    print(f"\n{'='*60}")
    print(f"Generating homography diagnostics for {subset}")
    print(f"{'='*60}")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = Config('cfg/eth_ucy/cor_fm.yml', f'hdiag_{subset}')
    cfg.device = 'cpu'
    
    try:
        ds = ETHDataset(cfg=cfg, training=False, data_dir=data_dir, subset=subset,
                        rotate_time_frame=0, type='original', video_mode='off')
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=seq_collate_eth)
    except Exception as e:
        print(f"[{subset}] Failed to load dataset: {e}")
        return False
    
    H = load_homography(data_dir, subset)
    bg_path = find_reference_frame(data_dir, subset)
    bg_img = load_image(bg_path)
    if bg_path:
        print(f"[{subset}] Reference frame: {bg_path}")
    else:
        print(f"[{subset}] WARNING: No reference frame found")
    
    # Collect all points for density map
    all_past_pix = []
    all_fut_pix = []
    best_batch = None
    best_count = -1
    
    for batch_idx, batch in enumerate(loader):
        try:
            past_raw = batch.get('past_traj_original_scale')
            fut_raw = batch.get('fut_traj_original_scale')
            past = extract_xy_agents(past_raw, frames=cfg.past_frames)
            fut = extract_xy_agents(fut_raw, frames=cfg.future_frames)
            if past is None or fut is None:
                continue
            A = min(past.shape[0], fut.shape[0])
            if A > best_count:
                best_count = A
                best_batch = (past[:A].copy(), fut[:A].copy())
            if H is not None:
                pp = world_to_pixel(H, past[:A])
                fp = world_to_pixel(H, fut[:A])
                if pp is not None:
                    all_past_pix.append(pp.reshape(-1, 2))
                if fp is not None:
                    all_fut_pix.append(fp.reshape(-1, 2))
        except Exception:
            continue
    
    # Figure 1: Global point cloud density map
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 9))
    fig1.suptitle(f'{subset.upper()} - Global Homography Point Cloud Density Map', fontsize=14, fontweight='bold')
    if bg_img is not None:
        ax1.imshow(bg_img, alpha=0.7)
        ax1.set_xlim(0, bg_img.shape[1])
        ax1.set_ylim(bg_img.shape[0], 0)
    else:
        ax1.set_facecolor('#f5f5f5')
    
    all_pp = []
    all_fp = []
    if all_past_pix:
        all_pp = np.concatenate(all_past_pix, axis=0)
    if all_fut_pix:
        all_fp = np.concatenate(all_fut_pix, axis=0)
    
    if len(all_pp) > 0:
        ax1.scatter(all_pp[:, 0], all_pp[:, 1], s=1, c='cyan', alpha=0.3, label='Past (all)')
    if len(all_fp) > 0:
        ax1.scatter(all_fp[:, 0], all_fp[:, 1], s=1, c='magenta', alpha=0.3, label='Future (all)')
    ax1.legend(markerscale=10)
    ax1.set_xlabel('u (pixels)')
    ax1.set_ylabel('v (pixels)')
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    out1 = out_dir / f'{subset}_1_global_pointcloud_density.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{subset}] Saved Figure 1: {out1}")
    
    # Figure 2: Dense multi-agent snapshot
    fig2, (ax2p, ax2w) = plt.subplots(1, 2, figsize=(18, 8))
    fig2.suptitle(f'{subset.upper()} - Dense Multi-Agent Snapshot (Max Density: {best_count} agents)', 
                  fontsize=14, fontweight='bold')
    
    if best_batch is None:
        print(f"[{subset}] WARNING: No dense snapshot found")
        plt.close(fig2)
        return False
    
    past_b, fut_b = best_batch
    A = past_b.shape[0]
    
    # Pixel space with background
    if bg_img is not None:
        ax2p.imshow(bg_img, alpha=0.8)
        ax2p.set_xlim(0, bg_img.shape[1])
        ax2p.set_ylim(bg_img.shape[0], 0)
    else:
        ax2p.set_facecolor('#f5f5f5')
    
    if H is not None:
        pp_b = world_to_pixel(H, past_b)
        fp_b = world_to_pixel(H, fut_b)
        if pp_b is not None and fp_b is not None:
            for a in range(A):
                ax2p.plot(pp_b[a, :, 0], pp_b[a, :, 1], '--', color='cyan', alpha=0.8, linewidth=1.5)
                ax2p.scatter(pp_b[a, -1, 0], pp_b[a, -1, 1], c='cyan', s=20)
                ax2p.plot(fp_b[a, :, 0], fp_b[a, :, 1], '-', color='yellow', alpha=0.9, linewidth=2)
                ax2p.scatter(fp_b[a, 0, 0], fp_b[a, 0, 1], c='orange', s=25, marker='s')
    
    ax2p.set_title('Pixel Space (with background)')
    ax2p.set_xlabel('u (pixels)')
    ax2p.set_ylabel('v (pixels)')
    ax2p.grid(True, alpha=0.3)
    
    # World coordinates
    for a in range(A):
        ax2w.plot(past_b[a, :, 0], past_b[a, :, 1], '--', color='blue', alpha=0.7, linewidth=1.5)
        ax2w.scatter(past_b[a, -1, 0], past_b[a, -1, 1], c='blue', s=20)
        ax2w.plot(fut_b[a, :, 0], fut_b[a, :, 1], '-', color='red', alpha=0.9, linewidth=2)
        ax2w.scatter(fut_b[a, 0, 0], fut_b[a, 0, 1], c='darkred', s=25, marker='s')
    
    ax2w.set_title('World Coordinates (meters)')
    ax2w.set_xlabel('X (m)')
    ax2w.set_ylabel('Y (m)')
    ax2w.grid(True, alpha=0.4)
    ax2w.axis('equal')
    
    plt.tight_layout()
    out2 = out_dir / f'{subset}_2_dense_multagent_snapshot.png'
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{subset}] Saved Figure 2: {out2}")
    
    return True


def main():
    data_dir = '/home/efrei_stage/Desktop/tu-clausthal/datasets'
    if not os.path.exists(data_dir):
        repo_data = REPO_ROOT / 'data' / 'eth_ucy'
        if repo_data.exists():
            data_dir = str(repo_data)
    
    out_dir = REPO_ROOT / 'results_eth_ucy' / 'homography_verification'
    
    scenes = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    results = {}
    
    print("Starting rigorous homography diagnostics...")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {out_dir}")
    
    for scene in scenes:
        results[scene] = generate_diagnostics(scene, data_dir, out_dir)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for scene, ok in results.items():
        print(f"{scene:10s}: {'OK' if ok else 'FAILED'}")
    
    print(f"\nFigures saved to: {out_dir}")


if __name__ == '__main__':
    main()
