#!/usr/bin/env python3
"""
Visual verification of homography alignment for ETH/UCY scenes.
Loads each scene via ETHDataset (strict TU-Clausthal raw) and plots trajectories
in world and pixel spaces with background images when available.
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


def find_background_image(data_dir, subset):
    """Find background/scene image if available."""
    search_paths = [
        os.path.join(data_dir, subset, 'visual_data'),
        os.path.join(data_dir, subset, 'frames'),
        os.path.join(data_dir, subset, 'images'),
        os.path.join(REPO_ROOT, 'data', 'eth_ucy', subset, 'visual_data'),
        os.path.join(REPO_ROOT, 'data', 'eth_ucy', subset, 'frames'),
    ]
    # Try common names
    for sp in search_paths:
        if os.path.exists(sp):
            # look for jpg/png
            for fn in sorted(os.listdir(sp)):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(sp, fn)
    # Also check videos dir
    for cand in [
        os.path.join(data_dir, subset, 'reference.jpg'),
        os.path.join(data_dir, subset, 'background.jpg'),
        os.path.join(REPO_ROOT, 'data', 'eth_ucy', 'videos', f'{subset}.jpg'),
    ]:
        if os.path.exists(cand):
            return cand
    return None


def load_image(path):
    try:
        from PIL import Image
        img = Image.open(path)
        return np.array(img)
    except Exception as e:
        try:
            import cv2
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e2:
            print(f"[WARN] Failed to load image {path}: {e2}")
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


def extract_xy_agents(traj_raw, frames=8):
    if traj_raw is None:
        return None
    traj = np.asarray(traj_raw)
    if traj.size == 0:
        return traj
    if traj.dtype == object:
        all_agents = []
        try:
            for i in range(traj.shape[0]):
                if traj.ndim == 1:
                    t = np.asarray(traj[i])
                else:
                    t = np.asarray(traj[i, 0]) if traj.shape[1] > 0 else np.asarray(traj[i])
                if t.ndim == 2 and t.shape[-1] >= 2:
                    xy = t[..., :2]
                    if xy.shape[0] == frames:
                        all_agents.append(xy.reshape(1, frames, 2))
                    elif xy.shape[-2] == frames:
                        all_agents.append(xy)
                elif t.ndim == 1 and t.size >= frames * 2:
                    # try to find pattern
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


def plot_scene(subset, data_dir, out_dir, max_windows=3):
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
        return False
    
    H = load_homography(data_dir, subset)
    bg_path = find_background_image(data_dir, subset)
    if bg_path:
        print(f"[{subset}] Found background: {bg_path}")
    else:
        print(f"[{subset}] WARNING: No background image found - using blank canvas")
    
    bg_img = load_image(bg_path) if bg_path else None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'ETH/UCY Homography Alignment Verification - {subset.upper()}', fontsize=16, fontweight='bold')
    
    ax_world = axes[0, 0]
    ax_pixel = axes[0, 1]
    ax_overlay = axes[1, 0]
    ax_detail = axes[1, 1]
    
    count = 0
    total_agents = 0
    for batch_idx, batch in enumerate(loader):
        if count >= max_windows:
            break
        try:
            past_raw = batch.get('past_traj_original_scale')
            fut_raw = batch.get('fut_traj_original_scale')
            
            past = extract_xy_agents(past_raw, frames=cfg.past_frames)
            fut = extract_xy_agents(fut_raw, frames=cfg.future_frames)
            
            if past is None or fut is None:
                continue
            
            A = min(past.shape[0], fut.shape[0])
            total_agents += A
            
            for a in range(A):
                ax_world.plot(past[a, :, 0], past[a, :, 1], '-', color='blue', alpha=0.5, linewidth=1)
                ax_world.scatter(past[a, -1, 0], past[a, -1, 1], color='blue', s=8, alpha=0.6)
                ax_world.plot(fut[a, :, 0], fut[a, :, 1], '-', color='red', alpha=0.5, linewidth=1)
            
            if H is not None:
                past_pix = world_to_pixel(H, past[:A])
                fut_pix = world_to_pixel(H, fut[:A])
                if past_pix is not None and fut_pix is not None:
                    for a in range(A):
                        ax_pixel.plot(past_pix[a, :, 0], past_pix[a, :, 1], '-', color='cyan', alpha=0.5, linewidth=1)
                        ax_pixel.plot(fut_pix[a, :, 0], fut_pix[a, :, 1], '-', color='yellow', alpha=0.5, linewidth=1)
                    for a in range(A):
                        ax_overlay.plot(past_pix[a, :, 0], past_pix[a, :, 1], '-', color='lime', alpha=0.6, linewidth=1.2)
                        ax_overlay.plot(fut_pix[a, :, 0], fut_pix[a, :, 1], '-', color='magenta', alpha=0.6, linewidth=1.2)
                    if A > 0:
                        ax_detail.plot(past_pix[0, :, 0], past_pix[0, :, 1], 'o-', color='cyan', label='History', markersize=4)
                        ax_detail.plot(fut_pix[0, :, 0], fut_pix[0, :, 1], 's-', color='yellow', label='Future', markersize=4)
                        ax_detail.legend()
            
            count += 1
        except Exception as e:
            print(f"[{subset}] Error: {e}")
            continue
    
    # Add background to pixel plots
    if bg_img is not None:
        for ax in [ax_pixel, ax_overlay, ax_detail]:
            ax.imshow(bg_img, alpha=0.6, extent=None)
            ax.set_xlim(0, bg_img.shape[1])
            ax.set_ylim(bg_img.shape[0], 0)
    else:
        for ax in [ax_pixel, ax_overlay, ax_detail]:
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.5, 'No background image available', 
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='gray', alpha=0.7)
    
    ax_world.set_title('World Coordinates (meters) - All Agents')
    ax_world.set_xlabel('X (m)')
    ax_world.set_ylabel('Y (m)')
    ax_world.grid(True, alpha=0.3)
    ax_world.axis('equal')
    
    ax_pixel.set_title('Pixel Coordinates (via Homography) + Background')
    ax_pixel.set_xlabel('U (pixels)')
    ax_pixel.set_ylabel('V (pixels)')
    ax_pixel.grid(True, alpha=0.3)
    
    ax_overlay.set_title('Trajectory Overlay on Scene')
    ax_overlay.set_xlabel('U (pixels)')
    ax_overlay.set_ylabel('V (pixels)')
    ax_overlay.grid(True, alpha=0.3)
    
    ax_detail.set_title('Detail View - Sample Agent')
    ax_detail.set_xlabel('U (pixels)')
    ax_detail.set_ylabel('V (pixels)')
    ax_detail.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / f'{subset}_homography_verification.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[{subset}] Saved: {out_path} (agents: {total_agents})")
    return total_agents > 0


def main():
    data_dir = '/home/efrei_stage/Desktop/tu-clausthal/datasets'
    if not os.path.exists(data_dir):
        repo_data = REPO_ROOT / 'data' / 'eth_ucy'
        if repo_data.exists():
            data_dir = str(repo_data)
    
    out_dir = REPO_ROOT / 'results_eth_ucy' / 'homography_verification'
    
    scenes = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    results = {}
    
    print("Starting homography verification with background...")
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
