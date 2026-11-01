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


def extract_xy_agents(traj_raw, frames=8):
    """Extract (A, T, 2) world coordinates for all agents.
    Handles variable A and feature dims [A, T, 6] etc.
    """
    if traj_raw is None:
        return None
    traj = np.asarray(traj_raw)
    if traj.size == 0:
        return traj
    # Try to handle object arrays from variable-length agents
    if traj.dtype == object:
        all_agents = []
        for i in range(traj.shape[0]):
            for j in range(traj.shape[1]) if traj.ndim > 1 else range(1):
                try:
                    t = np.asarray(traj[i] if traj.ndim == 1 else traj[i, j])
                    if t.ndim == 2 and t.shape[-1] >= 2:
                        xy = t[..., :2]
                        if xy.shape[0] == frames or xy.shape[-2] == frames:
                            if xy.shape[-2] == frames and xy.shape[0] != frames:
                                all_agents.append(xy)
                            elif xy.shape[0] == frames:
                                all_agents.append(xy.reshape(1, frames, 2))
                    elif t.ndim == 1 and t.size >= frames * 2:
                        all_agents.append(t.reshape(-1, frames, t.size // frames)[:, :, :2])
                except Exception:
                    continue
        if all_agents:
            try:
                return np.concatenate(all_agents, axis=0)
            except Exception:
                return all_agents
        return None
    if traj.ndim == 4:
        # (B, A, T, F)
        B, A, T, F = traj.shape
        if F >= 2:
            return traj.reshape(B * A, T, F)[..., :2]
    elif traj.ndim == 3:
        # (A, T, F) or (B, something)
        if traj.shape[-1] >= 2:
            # Could be (A, T*F) flattened
            if traj.shape[-1] % 2 == 0 and traj.shape[-1] > 4 and traj.shape[1] * 2 == traj.shape[-1] and traj.shape[-1] % frames == 0:
                # (A, T*F) -> check if F>=2
                n_t = traj.shape[-1] // 2
                if n_t == frames:
                    return traj.reshape(traj.shape[0], frames, 2)
            if traj.shape[1] == frames:
                return traj[..., :2]
            # try to reshape if flattened
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


def plot_scene(subset, data_dir, out_dir, max_windows=5):
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
                print(f"[{subset}] batch {batch_idx}: could not extract coords, shapes: past={getattr(past_raw,'shape',type(past_raw))}, fut={getattr(fut_raw,'shape',type(fut_raw))}")
                continue
            
            # past might be (B*A, P, 2) or (A, P, 2)
            if past.ndim == 3 and past.shape[0] > 1000:  # shouldn't happen
                past = past[:100]
            if fut.ndim == 3 and fut.shape[0] > 1000:
                fut = fut[:100]
            
            A_p = past.shape[0]
            A_f = fut.shape[0]
            A = min(A_p, A_f)
            total_agents += A
            print(f"[{subset}] batch {batch_idx}: plotting {A} agents")
            
            # Plot all agents
            for a in range(A):
                ax_world.plot(past[a, :, 0], past[a, :, 1], '-', color='blue', alpha=0.5, linewidth=1)
                ax_world.scatter(past[a, -1, 0], past[a, -1, 1], color='blue', s=10, alpha=0.6)
                ax_world.plot(fut[a, :, 0], fut[a, :, 1], '-', color='red', alpha=0.5, linewidth=1)
                ax_world.scatter(fut[a, 0, 0], fut[a, 0, 1], color='orange', s=10, alpha=0.6, marker='s')
            
            if H is not None:
                past_pix = world_to_pixel(H, past[:A])
                fut_pix = world_to_pixel(H, fut[:A])
                if past_pix is not None and fut_pix is not None:
                    for a in range(A):
                        ax_pixel.plot(past_pix[a, :, 0], past_pix[a, :, 1], '-', color='blue', alpha=0.5, linewidth=1)
                        ax_pixel.plot(fut_pix[a, :, 0], fut_pix[a, :, 1], '-', color='red', alpha=0.5, linewidth=1)
                    for a in range(min(A, 20)):
                        ax_overlay.plot(past_pix[a, :, 0], past_pix[a, :, 1], '-', color='cyan', alpha=0.6, linewidth=1.2)
                        ax_overlay.plot(fut_pix[a, :, 0], fut_pix[a, :, 1], '-', color='orange', alpha=0.6, linewidth=1.2)
                    # detail
                    a = 0
                    if A > 0:
                        ax_detail.plot(past_pix[a, :, 0], past_pix[a, :, 1], 'o-', color='blue', label='History', markersize=4)
                        ax_detail.plot(fut_pix[a, :, 0], fut_pix[a, :, 1], 's-', color='red', label='Future', markersize=4)
                        ax_detail.legend()
            
            count += 1
        except Exception as e:
            print(f"[{subset}] Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[{subset}] Total agents plotted: {total_agents}")
    
    ax_world.set_title('World Coordinates (meters) - All Agents')
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
    
    ax_detail.set_title('Detail View - Sample Agent')
    ax_detail.set_xlabel('U (pixels)')
    ax_detail.set_ylabel('V (pixels)')
    ax_detail.grid(True, alpha=0.3)
    ax_detail.invert_yaxis()
    
    plt.tight_layout()
    out_path = out_dir / f'{subset}_homography_verification.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[{subset}] Saved: {out_path}")
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
