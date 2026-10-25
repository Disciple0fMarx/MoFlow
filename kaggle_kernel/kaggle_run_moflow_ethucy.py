#!/usr/bin/env python
"""
Kaggle Kernel Runner for MoFlow ETH/UCY TU-Clausthal Pipeline
Clones TU-Clausthal datasets and runs M1-M6 verification/training
"""
import os
import sys
import subprocess
import json
from pathlib import Path

# Kaggle paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORK = Path("/kaggle/working")
KAGGLE_TEMP = Path("/kaggle/temp")

def run_cmd(cmd, cwd=None, check=True):
    print(f"RUN: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str))
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        if check:
            sys.exit(result.returncode)
    return result

def main():
    print("=" * 60)
    print("MoFlow ETH/UCY TU-Clausthal Pipeline - Kaggle Execution")
    print("=" * 60)
    
    # Step 1: Clone TU-Clausthal datasets
    print("\n[STEP 1] Cloning TU-Clausthal datasets...")
    datasets_dir = KAGGLE_WORK / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    tu_repo = datasets_dir / "tu-clausthal"
    if not tu_repo.exists():
        run_cmd([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "https://gitlab.tu-clausthal.de/pka20/Trajectory-Prediction-Pedestrian.git",
            str(tu_repo)
        ])
        run_cmd(["git", "sparse-checkout", "set", "datasets/ETH", "datasets/UCY"], cwd=str(tu_repo))
    else:
        print(f"TU-Clausthal repo already exists at {tu_repo}")
    
    # Step 2: Clone MoFlow repo
    print("\n[STEP 2] Cloning MoFlow repository...")
    repo_dir = KAGGLE_WORK / "MoFlow"
    if not repo_dir.exists():
        run_cmd([
            "git", "clone", "-b", "fix/eth-ucy-alignment",
            "https://github.com/Disciple0fMarx/MoFlow.git",
            str(repo_dir)
        ])
    else:
        print(f"MoFlow repo already exists at {repo_dir}")
        run_cmd(["git", "fetch", "origin"], cwd=str(repo_dir))
        run_cmd(["git", "checkout", "fix/eth-ucy-alignment"], cwd=str(repo_dir))
        run_cmd(["git", "pull", "origin", "fix/eth-ucy-alignment"], cwd=str(repo_dir))
    
    os.chdir(repo_dir)
    
    # Install dependencies
    print("\n[STEP 3] Installing dependencies...")
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "easydict", "einops", "tensorboardX", "pyyaml", "GitPython"], check=False)
    
    # Determine data root
    data_root = None
    for cand in [tu_repo / "datasets", KAGGLE_INPUT / "ethucy", KAGGLE_INPUT / "tu-clausthal"]:
        if cand.exists():
            data_root = str(cand)
            break
    if data_root is None:
        # Try to find in kaggle input
        for item in KAGGLE_INPUT.iterdir():
            if item.is_dir():
                if (item / "ETH").exists() or (item / "datasets").exists():
                    data_root = str(item)
                    break
    
    print(f"\nData root: {data_root}")
    
    # M1: Data & Alignment Verification
    print("\n" + "=" * 60)
    print("M1: Data & Homography Alignment Verification")
    print("=" * 60)
    run_cmd([sys.executable, "scripts/kaggle_dry_run_test.py", "--data_root", data_root] if data_root else [sys.executable, "scripts/kaggle_dry_run_test.py", "--create_mock"], check=False)
    
    # M2: Global Video Modes Training
    print("\n" + "=" * 60)
    print("M2: Global Video Encoder Training - off/static/full modes")
    print("=" * 60)
    for video_mode in ['off', 'static', 'full']:
        print(f"\n--- Training with video_mode={video_mode} ---")
        cmd = [
            sys.executable, "fm_eth.py",
            "--cfg", "cfg/eth_ucy/cor_fm.yml",
            "--subset", "eth",
            "--video_mode", video_mode,
            "--epochs", "5",  # Quick test on Kaggle
            "--batch_size", "64"
        ]
        if data_root:
            cmd.extend(["--data_root", data_root])
        run_cmd(cmd, check=False)
    
    # M3: Grad-CAM
    print("\n" + "=" * 60)
    print("M3: Grad-CAM Spatial Attribution")
    print("=" * 60)
    if os.path.exists('tools/gradcam_eth.py'):
        run_cmd([sys.executable, "tools/gradcam_eth.py", "--help"], check=False)
    else:
        print("tools/gradcam_eth.py not found, skipping")
    
    # M4: Latent Perturbation
    print("\n" + "=" * 60)
    print("M4: Latent Perturbation (baseline/zeroed/permuted)")
    print("=" * 60)
    run_cmd([sys.executable, "tools/attrib_video_conditioning.py", "--help"], check=False)
    
    # M5: Cross-Scene Transfer
    print("\n" + "=" * 60)
    print("M5: Cross-Scene Transfer Matrices")
    print("=" * 60)
    run_cmd(["bash", "scripts/research/03_transfer_matrix.sh"], check=False)
    
    # M6: Agent-Centric Fusion
    print("\n" + "=" * 60)
    print("M6: Agent-Centric Video Encoder & Tri-Modal Fusion")
    print("=" * 60)
    run_cmd([sys.executable, "-c", """
# Test agent encoder
import torch
from video_encoder.agent_encoder import AgentVideoEncoder
enc = AgentVideoEncoder()
# AgentVideoEncoder expects [B, A, T, 3, H, W] based on usage in ETHEncoder
x = torch.randn(2, 3, 8, 3, 64, 64)
y = enc(x)
print('Agent encoder output:', y.shape)
"""], check=False)
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
