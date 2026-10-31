#!/usr/bin/env python3
import os, sys, json, csv, shutil, glob, base64, zipfile, subprocess
from pathlib import Path

KAGGLE_WORK = Path("/kaggle/working")
REPO_URL = "https://github.com/Disciple0fMarx/MoFlow.git"
BRANCH = "fix/eth-ucy-alignment"
SUBSETS = ["eth", "hotel", "univ", "zara1", "zara2"]
VIDEO_MODES = ["off", "static", "full"]

def run_cmd(cmd, cwd=None, check=True):
    print(f"[CMD] {' '.join(map(str,cmd))}", flush=True)
    res = subprocess.run(cmd, cwd=cwd, check=False)
    if check and res.returncode != 0:
        raise SystemExit(f"Command failed with {res.returncode}: {' '.join(map(str,cmd))}")
    return res

def clone_repo(repo_dir: Path):
    if not repo_dir.exists():
        run_cmd(["git", "clone", "-b", BRANCH, REPO_URL, str(repo_dir)])
    else:
        run_cmd(["git", "fetch", "origin"], cwd=str(repo_dir))
        run_cmd(["git", "checkout", BRANCH], cwd=str(repo_dir))
        run_cmd(["git", "pull", "origin", BRANCH], cwd=str(repo_dir))

def main():
    repo_dir = KAGGLE_WORK / "MoFlow"
    clone_repo(repo_dir)
    os.chdir(repo_dir)
    sys.path.insert(0, str(repo_dir))
    
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "easydict", "einops", "tensorboardX", "pyyaml", "GitPython", "ema-pytorch", "opencv-python-headless", "scikit-image"], check=False)
    
    # Ensure checkpoints exist
    ckpt_dir = KAGGLE_WORK / "checkpoints"
    if not ckpt_dir.exists() or len(list(ckpt_dir.glob("*.pt"))) == 0:
        # try to find in results
        for p in glob.glob(str(repo_dir / "results_eth_ucy" / "**" / "*.pt"), recursive=True):
            shutil.copy2(p, ckpt_dir / Path(p).name)
    
    out_root = KAGGLE_WORK / "loso_visualizations"
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Generate trajectory visualizations
    print("[viz] Generating trajectory visualizations...", flush=True)
    try:
        run_cmd([sys.executable, "tools/generate_qualitative_figures.py", 
                 "--protocol", "video0", "--num-samples", "3", 
                 "--sdd-root", "/dev/null", "--video-features-root", str(KAGGLE_WORK / "features" / "resnet18") if (KAGGLE_WORK / "features").exists() else "features/resnet18"], check=False)
    except Exception as e:
        print(f"[viz] SDD viz failed (expected): {e}", flush=True)
    
    # Generate ETH/UCY specific visualizations
    viz_script = repo_dir / "tools" / "viz_ethucy_loso.py"
    if not viz_script.exists():
        # Create it
        viz_script.write_text('''
#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from utils.config import Config
from utils.normalization import unnormalize_min_max
from einops import rearrange

class NullLogger:
    def info(self,*a,**k): pass
    def warning(self,*a,**k): pass

def load_ckpt(ckpt_path, model):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        for k in ['state_dict','model_state_dict','model']:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]; break
        else: sd = ckpt
    else: sd = ckpt
    from collections import OrderedDict
    nd = OrderedDict()
    for kk,vv in sd.items():
        if kk.startswith('module.'): nd[kk[7:]]=vv
        else: nd[kk]=vv
    model.load_state_dict(nd, strict=False)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='cfg/eth_ucy/cor_fm.yml')
    p.add_argument('--subset', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--video-mode', default='off')
    p.add_argument('--data-dir', default='data/eth_ucy')
    p.add_argument('--features-root', default=None)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--num-samples', type=int, default=3)
    args = p.parse_args()
    
    cfg = Config(args.cfg, f'viz_{args.subset}_{args.video_mode}')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = args.video_mode != 'off'
        cfg.MODEL.CONTEXT_ENCODER.VIDEO_MODE = args.video_mode
        if args.features_root:
            cfg.MODEL.CONTEXT_ENCODER.VIDEO_FEATURES_ROOT = args.features_root
    except: pass
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ds = ETHDataset(cfg=cfg, training=False, data_dir=args.data_dir, subset=args.subset, rotate_time_frame=0, type='original', video_mode=args.video_mode)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=seq_collate_eth)
    
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=NullLogger(), config=cfg)
    fm = FlowMatcher(cfg, model, logger=NullLogger())
    fm = load_ckpt(args.ckpt, fm)
    fm = fm.to(cfg.device).eval()
    
    K = int(getattr(cfg, 'denoising_head_preds', 20))
    S = int(getattr(cfg, 'sampling_steps', 20))
    dt = 1.0/S
    
    for i, batch in enumerate(loader):
        if i >= args.num_samples: break
        with torch.no_grad():
            x_data = {k:v.to(cfg.device) for k,v in batch.items() if k in ['past_traj_original_scale','fut_traj','z_video_global','agent_video_feats']}
            y_t = torch.randn((1,K,1,cfg.future_frames*2), device=cfg.device)
            for step in range(S):
                t = torch.full((1,), step*dt, device=cfg.device)
                preds = fm.model_predictions(y_t, x_data, t, flag_print=False)
                y_t = y_t + preds.pred_vel * dt
            pred = y_t
        # unnormalize
        pred_traj = rearrange(pred, 'b k a (f d) -> b k a f d', f=cfg.future_frames)[..., :2]
        pred_traj = unnormalize_min_max(pred_traj, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
        gt = rearrange(batch['fut_traj_original_scale'], 'b a (f d) -> b a f d', f=cfg.future_frames)[..., :2]
        hist = batch['past_traj_original_scale'][0].cpu().numpy().reshape(cfg.past_frames, 2)
        gt_np = gt[0].cpu().numpy()
        pred_np = pred_traj[0].cpu().numpy()
        
        fig, ax = plt.subplots(1,1,figsize=(10,8))
        ax.plot(hist[:,0], hist[:,1], 'k--', label='History', linewidth=2)
        ax.plot(gt_np[0,:,0], gt_np[0,:,1], 'g-', label='GT', linewidth=2, marker='o')
        for k in range(min(K,20)):
            ax.plot(pred_np[k,0,:,0], pred_np[k,0,:,1], 'b-', alpha=0.3, linewidth=1)
        # best
        ade = np.mean(np.linalg.norm(pred_np[:,0] - gt_np[0:1], axis=-1), axis=-1)
        best = np.argmin(ade)
        ax.plot(pred_np[best,0,:,0], pred_np[best,0,:,1], 'r-', linewidth=2.5, label=f'Best (ADE={ade[best]:.3f})')
        ax.scatter(hist[0,0], hist[0,1], c='k', s=50)
        ax.legend()
        ax.set_title(f'{args.subset} | {args.video_mode} | sample {i}')
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(out_dir / f'traj_{args.subset}_{args.video_mode}_s{i:03d}.png', dpi=200)
        plt.close()
    print(f'Saved to {out_dir}')

if __name__=='__main__':
    main()
''')
    
    # Run visualizations for all checkpoints
    for subset in SUBSETS:
        for mode in VIDEO_MODES:
            ckpt = ckpt_dir / f"{subset}_{mode}.pt"
            if not ckpt.exists():
                # try other names
                for c in ckpt_dir.glob(f"{subset}_{mode}*"):
                    ckpt = c; break
            if not ckpt.exists():
                print(f"[skip] {subset}/{mode} no checkpoint", flush=True)
                continue
            print(f"[viz] {subset}/{mode}", flush=True)
            out_d = out_root / "trajectories" / subset / mode
            try:
                run_cmd([sys.executable, str(viz_script), 
                         "--subset", subset, "--ckpt", str(ckpt), 
                         "--video-mode", mode, "--out-dir", str(out_d),
                         "--num-samples", "5", 
                         "--features-root", str(KAGGLE_WORK / "features" / "resnet18") if (KAGGLE_WORK / "features").exists() else "features/resnet18",
                         "--data-dir", str(repo_dir / "data" / "eth_ucy") if (repo_dir / "data" / "eth_ucy").exists() else "data/eth_ucy"], check=False)
            except Exception as e:
                print(f"[err] {e}", flush=True)
    
    # Grad-CAM for static and full
    for subset in SUBSETS:
        for mode in ["static", "full"]:
            ckpt = ckpt_dir / f"{subset}_{mode}.pt"
            if not ckpt.exists():
                for c in ckpt_dir.glob(f"{subset}_{mode}*"):
                    ckpt = c; break
            if not ckpt.exists():
                continue
            print(f"[gradcam] {subset}/{mode}", flush=True)
            out_d = out_root / "gradcam" / subset / mode
            try:
                run_cmd([sys.executable, "tools/gradcam_eth.py",
                         "--subset", subset, "--ckpt", str(ckpt),
                         "--video-features-root", str(KAGGLE_WORK / "features" / "resnet18") if (KAGGLE_WORK / "features").exists() else "features/resnet18",
                         "--out-dir", str(out_d), "--window-index", "0"], check=False)
            except Exception as e:
                print(f"[err gradcam] {e}", flush=True)
    
    # Bundle
    zip_path = KAGGLE_WORK / "loso_visualizations.zip"
    print(f"[bundle] Creating {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(out_root):
            for fn in files:
                fp = Path(root)/fn
                zf.write(fp, fp.relative_to(out_root))
    
    # Git push
    try:
        res_dst = repo_dir / "results_eth_ucy" / "loso_visualizations"
        shutil.copytree(out_root, res_dst, dirs_exist_ok=True)
        run_cmd(["git", "add", "-f", "results_eth_ucy/loso_visualizations/*"], cwd=str(repo_dir), check=False)
        run_cmd(["git", "config", "user.email", "kaggle@runner.local"], cwd=str(repo_dir), check=False)
        run_cmd(["git", "config", "user.name", "KaggleRunner"], cwd=str(repo_dir), check=False)
        run_cmd(["git", "commit", "-m", "viz(eth_ucy): add LOSO trajectory plots and gradcam"], cwd=str(repo_dir), check=False)
        run_cmd(["git", "push", "origin", BRANCH], cwd=str(repo_dir), check=False)
    except Exception as e:
        print(e, flush=True)
    
    # Base64
    print("===VIZ_B64_START===", flush=True)
    try:
        data = zip_path.read_bytes()
        # chunk
        b64 = base64.b64encode(data).decode()
        for i in range(0, len(b64), 76800):
            print(f"VIZ:{i//76800}::{b64[i:i+76800]}", flush=True)
    except Exception as e:
        print(f"ERR:{e}", flush=True)
    print("===VIZ_B64_END===", flush=True)
    
    print("DONE", flush=True)

if __name__ == '__main__':
    main()
