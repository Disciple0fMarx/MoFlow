#!/usr/bin/env python
"""
Kaggle Kernel Runner for MoFlow ETH/UCY M1-M6 pipeline.

The heavy lifting (data parsing / homography / video features) is expected to
be precomputed and attached as a Kaggle dataset with this layout:

    <root>/original/<subset>/<subset>_{train,val,test}.pkl
    <root>/original/<subset>/<subset>_{train,test}_frame_index.pkl
    <root>/homography/<subset>_H.txt
    <root>/features/resnet18/<subset>.npy

If no processed dataset is attached we fall back to cloning the raw
TU-Clausthal repository, but training requires the processed pickles, so the
run is skipped with an explicit error instead of silently "succeeding".
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORK = Path("/kaggle/working")

BRANCH = "fix/eth-ucy-alignment"
REPO_URL = "https://github.com/Disciple0fMarx/MoFlow.git"
SUBSET = os.environ.get("SUBSET", "eth")
VIDEO_MODES = os.environ.get("VIDEO_MODES", "off,static,full").split(",")
EPOCHS = os.environ.get("EPOCHS", "80")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "64")
RUN_M5 = os.environ.get("RUN_M5", "0") == "1"


def run_cmd(cmd, cwd=None, check=True):
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"RUN: {printable}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str))
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}", flush=True)
        raise SystemExit(result.returncode)
    return result


def find_processed_root(subset: str):
    """Return a directory that contains original/<subset>/<subset>_train.pkl."""
    rel = Path("original") / subset / f"{subset}_train.pkl"
    candidates = [
        KAGGLE_INPUT / "moflow-ethucy-processed",
        KAGGLE_INPUT / f"moflow-ethucy-processed" / "data",
        KAGGLE_INPUT / "ethucy" / "processed",
        KAGGLE_INPUT / "ethucy-processed",
    ]
    if KAGGLE_INPUT.is_dir():
        for item in sorted(KAGGLE_INPUT.iterdir()):
            if not item.is_dir():
                continue
            candidates.append(item)
            candidates.extend(sorted(p for p in item.iterdir() if p.is_dir()))
    for cand in candidates:
        if (cand / rel).exists():
            return cand
    return None


def find_video_features(root: Path):
    cand = root / "features" / "resnet18"
    if cand.is_dir():
        return cand
    for path in root.glob("**/features/resnet18"):
        if path.is_dir():
            return path
    return None


def clone_repo(repo_dir: Path):
    if not repo_dir.exists():
        run_cmd(["git", "clone", "-b", BRANCH, REPO_URL, str(repo_dir)])
    else:
        run_cmd(["git", "fetch", "origin"], cwd=str(repo_dir))
        run_cmd(["git", "checkout", BRANCH], cwd=str(repo_dir))
        run_cmd(["git", "pull", "origin", BRANCH], cwd=str(repo_dir))


def copy_run_artifacts(repo_dir: Path, video_mode: str, data_dir: Path):
    """Copy the checkpoint/metrics belonging to THIS run only."""
    out_ckpt_dir = KAGGLE_WORK / "checkpoints"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(
        glob.glob(
            str(repo_dir / "results_eth_ucy" / "cor_fm" / f"*_{SUBSET}_vm{video_mode}*" / "models" / "checkpoint_*.pt")
        ),
        key=os.path.getmtime,
    )
    for ckpt in ckpts:
        name = Path(ckpt).name
        if name == "checkpoint_best.pt":
            shutil.copy2(ckpt, out_ckpt_dir / f"{SUBSET}_{video_mode}.pt")
        elif name == "checkpoint_last.pt":
            shutil.copy2(ckpt, out_ckpt_dir / f"{SUBSET}_{video_mode}_last.pt")
    for pattern in ("metrics*.csv", "*eval*.csv", "*.log"):
        for f in glob.glob(str(repo_dir / "results_eth_ucy" / "cor_fm" / f"*_{SUBSET}_vm{video_mode}*" / "**" / pattern), recursive=True):
            try:
                shutil.copy2(f, KAGGLE_WORK / f"{SUBSET}_{video_mode}_{Path(f).name}")
            except OSError:
                pass
    found = sorted(p.name for p in out_ckpt_dir.glob(f"{SUBSET}_{video_mode}*.pt"))
    print(f"[artifacts] {video_mode}: {found}", flush=True)


def main():
    print("=" * 60)
    print("MoFlow ETH/UCY Pipeline - Kaggle Execution")
    print("=" * 60, flush=True)

    repo_dir = KAGGLE_WORK / "MoFlow"
    clone_repo(repo_dir)
    os.chdir(repo_dir)
    (KAGGLE_WORK / "checkpoints").mkdir(parents=True, exist_ok=True)

    run_cmd(
        [sys.executable, "-m", "pip", "install", "-q",
         "easydict", "einops", "tensorboardX", "pyyaml", "GitPython", "ema-pytorch"],
        check=False,
    )

    data_dir = find_processed_root(SUBSET)
    if data_dir is None:
        raw_dir = KAGGLE_WORK / "datasets" / "tu-clausthal"
        if not raw_dir.exists():
            run_cmd([
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "https://gitlab.tu-clausthal.de/pka20/Trajectory-Prediction-Pedestrian.git",
                str(raw_dir),
            ])
            run_cmd(["git", "sparse-checkout", "set", "datasets/ETH", "datasets/UCY"], cwd=str(raw_dir))
        raise SystemExit(
            "No processed ETH/UCY dataset attached. Expected "
            f"original/{SUBSET}/{SUBSET}_train.pkl under /kaggle/input. "
            "Attach dhyaelbahri/moflow-ethucy-processed to this kernel."
        )

    features_root = find_video_features(data_dir)
    print(f"[data] processed root: {data_dir}", flush=True)
    print(f"[data] video features: {features_root}", flush=True)

    for video_mode in VIDEO_MODES:
        video_mode = video_mode.strip()
        if not video_mode:
            continue
        print(f"\n--- Training subset={SUBSET} video_mode={video_mode} ---", flush=True)
        cmd = [
            sys.executable, "fm_eth.py",
            "--cfg", "cfg/eth_ucy/cor_fm.yml",
            "--subset", SUBSET,
            "--video_mode", video_mode,
            "--epochs", EPOCHS,
            "--batch_size", BATCH_SIZE,
            "--data_dir", str(data_dir),
        ]
        if features_root is not None:
            cmd.extend(["--video_features_root", str(features_root)])
        run_cmd(cmd, check=False)
        copy_run_artifacts(repo_dir, video_mode, data_dir)

    if RUN_M5:
        env = os.environ.copy()
        env["DATASET"] = "eth_ucy"
        env["DATA_ROOT"] = str(data_dir)
        env["REPO_ROOT"] = str(repo_dir)
        env["RESULTS_ROOT"] = str(repo_dir / "results_eth_ucy" / "cor_fm")
        subprocess.run(["bash", "scripts/research/03_transfer_matrix.sh"], cwd=repo_dir, env=env)
    else:
        print("[q2] RUN_M5!=1, skipping transfer matrix", flush=True)

    # Final sweep: make sure any CSV/report generated under the repo is visible.
    for pattern in ("**/q2_transfer*.csv", "**/*transfer*matrix*.csv", "**/metrics*.csv"):
        for f in glob.glob(pattern, recursive=True):
            try:
                shutil.copy2(f, KAGGLE_WORK / Path(f).name)
            except OSError:
                pass

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
