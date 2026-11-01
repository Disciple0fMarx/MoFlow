#!/usr/bin/env python3
"""
Visual verification of homography alignment for ETH/UCY scenes.
Loads each scene via ETHDataset (strict TU-Clausthal raw) and plots trajectories
in world and pixel spaces to check for axis inversions/scaling issues.
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
                print(f"[{subset}] Loaded H from {hp}, shape={H.shape}")
                return H
            except Exception as e:
                print(f"[{subset}] Failed to load {hp}: {e}")
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


def plot_scene(subset, data_dir, out_dir, max_windows=10):
    print(f"\n{'='*60}")
    print(f"Plotting homography verification for {subset}")
    print(f"{'='*60}")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = Config('cfg/eth_ucy/cor_fm.yml', f'viz_h_{subset}')
    cfg.device = 'cpu'
    
    try:
        ds = ETHDataset(cfg=cfg, training=False, data_dir=data_dir, subset=subset,
                        rotate_time_frame=0, type='original', video_mode='off')
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=seq_collate_eth)
    except Exception as e:
        print(f"[{subset}] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    H = load_homography(data_dir, subset)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'ETH/UCY Homography Alignment Verification - {subset.upper()}', fontsize=16, fontweight='bold')
    
    ax_world = axes[0, 0]
    ax_pixel = axes[0, 1]
    ax_overlay = axes[1, 0]
    ax_detail = axes[1, 1]
    
    count = 0
    for batch_idx, batch in enumerate(loader):
        if count >= max_windows:
            break
        try:
            past = batch['past_traj_original_scale'][0].cpu().numpy()
            fut = batch['fut_traj_original_scale'][0].cpu().numpy()
            A = past.shape[0]
            P = cfg.past_frames
            F = cfg.future_frames
            past = past.reshape(A, P, 2)
            fut = fut.reshape(A, F, 2)
            
            for a in range(min(A, 5)):
                ax_world.plot(past[a, :, 0], past[a, :, 1], 'o-', color='blue', alpha=0.6, markersize=3)
                ax_world.plot(fut[a, :, 0], fut[a, :, 1], 's-', color='red', alpha=0.6, markersize=3)
                ax_world.plot([past[a, -1, 0], fut[a, 0, 0]], [past[a, -1, 1], fut[a, 0, 1]], 
                             'k--', alpha=0.3, linewidth=0.5)
            
            if H is not None:
                past_pix = world_to_pixel(H, past)
                fut_pix = world_to_pixel(H, fut)
                for a in range(min(A, 5)):
                    ax_pixel.plot(past_pix[a, :, 0], past_pix[a, :, 1], 'o-', color='blue', alpha=0.6, markersize=3)
                    ax_pixel.plot(fut_pix[a, :, 0], fut_pix[a, :, 1], 's-', color='red', alpha=0.6, markersize=3)
                    ax_pixel.plot([past_pix[a, -1, 0], fut_pix[a, 0, 0]], 
                                 [past_pix[a, -1, 1], fut_pix[a, 0, 1]], 
                                 'k--', alpha=0.3, linewidth=0.5)
                for a in range(min(A, 5)):
                    ax_overlay.plot(past_pix[a, :, 0], past_pix[a, :, 1], 'o-', color='cyan', alpha=0.7, markersize=4)
                    ax_overlay.plot(fut_pix[a, :, 0], fut_pix[a, :, 1], 's-', color='orange', alpha=0.7, markersize=4)
                    ax_overlay.plot([past_pix[a, -1, 0], fut_pix[a, 0, 0]], 
                                    [past_pix[a, -1, 1], fut_pix[a, 0, 1]], 
                                    'k-', alpha=0.5, linewidth=1)
                a = 0
                ax_detail.plot(past_pix[a, :, 0], past_pix[a, :, 1], 'o-', color='blue', label='History', markersize=6)
                ax_detail.plot(fut_pix[a, :, 0], fut_pix[a, :, 1], 's-', color='red', label='Future', markersize=6)
                ax_detail.annotate('Obs end', xy=(past_pix[a, -1, 0], past_pix[a, -1, 1]), 
                                   xytext=(5, 5), textcoords='offset points')
                ax_detail.legend()
            
            count += 1
        except Exception as e:
            print(f"[{subset}] Error processing batch {batch_idx}: {e}")
            continue
    
    ax_world.set_title('World Coordinates (meters)')
    ax_world.set_xlabel('X (m)')
    ax_world.set_ylabel('Y (m)')
    ax_world.grid(True, alpha=0.3)
    ax_world.axis('equal')
    
    ax_pixel.set_title('Pixel Coordinates (via Homography)')
    ax_pixel.set_xlabel('U (pixels)')
    ax_pixel.set_ylabel('V (pixels)')
    ax_pixel.grid(True, alpha=0.3)
    ax_pixel.invert_yaxis()
    
    ax_overlay.set_title('Overlay View (Pixel Space)')
    ax_overlay.set_xlabel('U (pixels)')
    ax_overlay.set_ylabel('V (pixels)')
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.invert_yaxis()
    ax_overlay.legend(['History', 'Future'], loc='upper right')
    
    ax_detail.set_title('Detail View - Agent Trajectory')
    ax_detail.set_xlabel('U (pixels)')
    ax_detail.set_ylabel('V (pixels)')
    ax_detail.grid(True, alpha=0.3)
    ax_detail.invert_yaxis()
    
    plt.tight_layout()
    out_path = out_dir / f'{subset}_homography_verification.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[{subset}] Saved: {out_path}")
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
    
    print("Starting homography verification...")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {out_dir}")
    
    for scene in scenes:
        results[scene] = plot_scene(scene, data_dir, out_dir)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for scene, ok in results.items():
        print(f"{scene:10s}: {'OK' if ok else 'FAILED'}")
    
    print(f"\nFigures saved to: {out_dir}")


if __name__ == '__main__':
    main()
