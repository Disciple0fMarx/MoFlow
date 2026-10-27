#!/usr/bin/env python
"""
Kaggle Kernel Runner for MoFlow ETH/UCY M1-M6 pipeline.

Trajectories and homographies come from the strict TU-Clausthal repository
layout (cloned at runtime if needed):

    <raw>/ETH/seq_eth/{obsmat.txt,H.txt}
    <raw>/ETH/seq_hotel/{obsmat.txt,H.txt}
    <raw>/UCY/{students03,zara01,zara02}/{obsmat.txt,H.txt,annotation.vsp}

Video features are the only precomputed artifact and may be attached as:

    <features-root>/<subset>.npy
    <features-root>/<subset>.manifest.parquet

No ETH/UCY pickle files are read.

Set ``BUILD_FEATURES=1`` to rebuild full-range global features from the repo's
``data/eth_ucy/*/visual_data`` frames (default when ``full`` is in
``VIDEO_MODES``). Set ``BUILD_FEATURES=0`` to reuse an attached full-range
feature cache.
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
BUILD_FEATURES = os.environ.get("BUILD_FEATURES", "").strip().lower()


def run_cmd(cmd, cwd=None, check=True):
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"RUN: {printable}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str))
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}", flush=True)
        raise SystemExit(result.returncode)
    return result


def _input_candidates():
    candidates = [
        KAGGLE_INPUT / "moflow-ethucy-processed",
        KAGGLE_INPUT / "datasets" / "dhyaelbahri" / "moflow-ethucy-processed",
        KAGGLE_INPUT / "datasets" / "moflow-ethucy-processed",
        KAGGLE_INPUT / "ethucy" / "processed",
        KAGGLE_INPUT / "ethucy-processed",
    ]
    if KAGGLE_INPUT.is_dir():
        for item in sorted(KAGGLE_INPUT.iterdir()):
            if not item.is_dir():
                continue
            candidates.append(item)
            try:
                children = sorted(p for p in item.iterdir() if p.is_dir())
            except OSError:
                children = []
            candidates.extend(children)
            for child in children:
                try:
                    candidates.extend(sorted(p for p in child.iterdir() if p.is_dir()))
                except OSError:
                    pass
    return candidates


def find_features_root():
    """Return the attached resnet18 feature directory, if present."""
    for cand in _input_candidates():
        feat = cand / "features" / "resnet18"
        if feat.is_dir() and any(feat.glob("*.npy")):
            return feat
    return None


RAW_REPO_URL = (
    "https://gitlab.tu-clausthal.de/pka20/Trajectory-Prediction-Pedestrian.git"
)
RAW_SPARSE_PATTERNS = [
    "/datasets/ETH/seq_eth/obsmat.txt",
    "/datasets/ETH/seq_eth/H.txt",
    "/datasets/ETH/seq_hotel/obsmat.txt",
    "/datasets/ETH/seq_hotel/H.txt",
    "/datasets/UCY/students03/obsmat.txt",
    "/datasets/UCY/students03/H.txt",
    "/datasets/UCY/students03/annotation.vsp",
    "/datasets/UCY/zara01/obsmat.txt",
    "/datasets/UCY/zara01/H.txt",
    "/datasets/UCY/zara01/annotation.vsp",
    "/datasets/UCY/zara02/obsmat.txt",
    "/datasets/UCY/zara02/H.txt",
    "/datasets/UCY/zara02/annotation.vsp",
]


def ensure_raw_data_root():
    """Clone/fetch the strict TU-Clausthal layout and return its datasets dir."""
    raw_repo = KAGGLE_WORK / "tu-clausthal"
    data_root = raw_repo / "datasets"
    required = [
        data_root / "ETH" / "seq_eth" / "obsmat.txt",
        data_root / "ETH" / "seq_hotel" / "obsmat.txt",
        data_root / "UCY" / "students03" / "annotation.vsp",
        data_root / "UCY" / "zara01" / "annotation.vsp",
        data_root / "UCY" / "zara02" / "annotation.vsp",
    ]
    if not all(p.exists() for p in required):
        if not raw_repo.exists():
            run_cmd(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    RAW_REPO_URL,
                    str(raw_repo),
                ]
            )
        run_cmd(
            ["git", "sparse-checkout", "set", "--no-cone", *RAW_SPARSE_PATTERNS],
            cwd=str(raw_repo),
        )
        run_cmd(["git", "checkout"], cwd=str(raw_repo))
    missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise SystemExit(
            "Strict raw TU-Clausthal data incomplete; missing: " + ", ".join(missing)
        )
    return data_root


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
            str(
                repo_dir
                / "results_eth_ucy"
                / "cor_fm"
                / f"*_{SUBSET}_vm{video_mode}*"
                / "models"
                / "checkpoint_*.pt"
            )
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
        for f in glob.glob(
            str(
                repo_dir
                / "results_eth_ucy"
                / "cor_fm"
                / f"*_{SUBSET}_vm{video_mode}*"
                / "**"
                / pattern
            ),
            recursive=True,
        ):
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
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "easydict",
            "einops",
            "tensorboardX",
            "pyyaml",
            "GitPython",
            "ema-pytorch",
        ],
        check=False,
    )

    modes = {m.strip() for m in VIDEO_MODES if m.strip()}
    needs_features = any(m != "off" for m in modes)
    if BUILD_FEATURES == "":
        # The attached feature cache is validation-only; rebuild for full mode.
        build_features = "full" in modes
    else:
        build_features = BUILD_FEATURES in {"1", "true", "yes", "on"}
    data_dir = ensure_raw_data_root()
    features_root = find_features_root()
    if build_features:
        features_root = KAGGLE_WORK / "features" / "resnet18"
        run_cmd(
            [
                sys.executable,
                "-m",
                "video_encoder.scripts.build_ethucy_features",
                "--frames-root",
                str(repo_dir / "data" / "eth_ucy"),
                "--out",
                str(features_root),
                "--stride",
                "10",
            ]
        )
    elif needs_features and features_root is None:
        raise SystemExit(
            "Video modes requested but no features/resnet18/*.npy found under "
            "/kaggle/input. Attach the features artifact, set BUILD_FEATURES=1, "
            "or use VIDEO_MODES=off."
        )
    print(f"[data] raw root: {data_dir}", flush=True)
    print(f"[data] video features: {features_root}", flush=True)

    for video_mode in VIDEO_MODES:
        video_mode = video_mode.strip()
        if not video_mode:
            continue
        print(f"\n--- Training subset={SUBSET} video_mode={video_mode} ---", flush=True)
        cmd = [
            sys.executable,
            "fm_eth.py",
            "--cfg",
            "cfg/eth_ucy/cor_fm.yml",
            "--subset",
            SUBSET,
            "--video_mode",
            video_mode,
            "--epochs",
            EPOCHS,
            "--batch_size",
            BATCH_SIZE,
            "--data_dir",
            str(data_dir),
        ]
        if features_root is not None:
            cmd.extend(["--video_features_root", str(features_root)])
        run_cmd(cmd, check=False)
        copy_run_artifacts(repo_dir, video_mode, data_dir)

    if RUN_M5:
        env = os.environ.copy()
        env["DATASET"] = "eth_ucy"
        env["ETH_DATA_ROOT"] = str(data_dir)
        if features_root is not None:
            env["ETH_FEATURES_ROOT"] = str(features_root)
        env["DATA_ROOT"] = str(data_dir)
        env["REPO_ROOT"] = str(repo_dir)
        env["RESULTS_ROOT"] = str(repo_dir / "results_eth_ucy" / "cor_fm")
        env["BATCH_SIZE"] = os.environ.get("M5_BATCH_SIZE", "128")
        subprocess.run(
            ["bash", "scripts/research/03_transfer_matrix.sh", "--n-batches", "0"],
            cwd=repo_dir,
            env=env,
        )
    else:
        print("[q2] RUN_M5!=1, skipping transfer matrix", flush=True)

    # Final sweep: make sure any CSV/report generated under the repo is visible.
    for pattern in (
        "**/q2_transfer*.csv",
        "**/*transfer*matrix*.csv",
        "**/metrics*.csv",
    ):
        for f in glob.glob(pattern, recursive=True):
            try:
                shutil.copy2(f, KAGGLE_WORK / Path(f).name)
            except OSError:
                pass

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
