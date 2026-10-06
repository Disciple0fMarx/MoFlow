#!/usr/bin/env python
"""Kaggle kernel runner — DEFINITIVE MoFlow/SDD pipeline (both angles).

Uses the NATIVE research scripts only (scripts/research/00..04); no
Kaggle-specific adapter or folder-restructuring logic. The
``brendanalvey/stanford-drone-dataset`` Kaggle dataset has the SAME layout as
the remote lab machine (``annotations/`` + ``videos/`` + ``.mov``), so we just
point ``SDD_ROOT`` at the mount and write everything to ``/kaggle/working``.

Pipeline (per angle):
  00  re-encode per-scene ResNet-18 feature caches   (once, shared)
  01  video-mode ablation off/static/full            (train + eval, LOSO x8)
  02  z_video attribution baseline/zeroed/permuted   (Q1)
  03  8x8 transfer matrix                            (Q2, --n-batches 0)
  04  per-window gain + geometry + renders           (Q5/Q7)

Angle A = geographic LOSO (no --video-id).
Angle B = single-video ``video0`` (--video-id video0, _vid… suffixes).

Scope / knobs (env overrides, read at the start of the run):
  PIPELINE_SCOPE  full | encode | angle1 | angle2   (default: full)
  SCENE           single scene to restrict LOSO loops (default: all 8)
  VIDEO_ID        optional --video-id override (default from PIPELINE_SCOPE)
  N_BATCHES       02/03/04 batch cap for fast smoke runs (default: 0=all)
  BATCH_SIZE      attribution eval batch (default 64)
  TOP_K           04 renders top-K (default 3)

When run on Kaggle the repo is cloned into /kaggle/working/MoFlow at the
branch HEAD (SHA recorded by the provenance files / this log). No config below
pins a SHA so re-clone + re-push-on-failure keeps working.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_URL = os.environ.get("MF_REPO_URL", "https://github.com/Disciple0fMarx/MoFlow.git")
REPO_BRANCH = os.environ.get("MF_REPO_BRANCH", "feature/sdd-phase1-refactor")
KAGGLE_SDD_ROOT = Path("/kaggle/input/stanford-drone-dataset")
KAGGLE_WORK = Path("/kaggle/working")
REPO_DIR = KAGGLE_WORK / "MoFlow"

PIPELINE_SCOPE = os.environ.get("PIPELINE_SCOPE", "full").lower()
SCENE = os.environ.get("SCENE", "")
VIDEO_ID = os.environ.get("VIDEO_ID", "")
N_BATCHES = os.environ.get("N_BATCHES", "0")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "64")
TOP_K = os.environ.get("TOP_K", "3")
SEED = os.environ.get("SEED", "42")
SAMPLING_STEPS = os.environ.get("SAMPLING_STEPS", "10")


def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] {level} {msg}", flush=True)


def die(msg: str) -> None:
    log(msg, "ERROR")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------
def check_env() -> None:
    if not Path("/kaggle").exists():
        die("Not running inside a Kaggle kernel (/kaggle missing).")
    if not KAGGLE_SDD_ROOT.is_dir():
        die(f"Dataset not mounted at {KAGGLE_SDD_ROOT}. Check kernel-metadata.json dataset_sources.")
    ann = KAGGLE_SDD_ROOT / "annotations"
    vid = KAGGLE_SDD_ROOT / "videos"
    if not ann.is_dir() or not vid.is_dir():
        die(f"Expected {KAGGLE_SDD_ROOT}/{{annotations,videos}}: got ann={ann.is_dir()} vid={vid.is_dir()}")
    scenes = sorted(p.name for p in vid.iterdir() if p.is_dir())
    log(f"Dataset OK: {len(scenes)} scenes under {vid}: {scenes}")
    if not shutil.which("nvidia-smi"):
        log("WARNING: nvidia-smi not found — GPU may be unavailable.", "WARN")
    else:
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                       check=False)


def install_deps() -> None:
    """Install project deps WITHOUT forcing the Kaggle image's torch version."""
    log("Installing Python deps (keeping image torch)...")
    reqs = REPO_DIR / "requirements.txt"
    filtered = (
        line.strip()
        for line in reqs.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
        and not line.strip().lower().startswith("torch")
    )
    pinned = [line for line in filtered if "==" in line]
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", *pinned]
    log("pip install: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    # Media + IO extras the image may not have.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
         "decord", "opencv-python-headless", "pyarrow", "pillow"],
        check=True,
    )


def clone_repo() -> None:
    if (REPO_DIR / ".git").exists():
        log(f"Repo already present at {REPO_DIR} — skipping clone.")
        return
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    log(f"Cloning {REPO_URL} [{REPO_BRANCH}] -> {REPO_DIR}")
    subprocess.run(
        ["git", "clone", "--single-branch", "--branch", REPO_BRANCH, "--depth", "1",
         REPO_URL, str(REPO_DIR)],
        check=True,
    )
    sha = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    log(f"Repo HEAD: {sha}")


# ---------------------------------------------------------------------------
# Pipeline drivers
# ---------------------------------------------------------------------------
def build_env() -> dict:
    env = dict(os.environ)
    env.update({
        "SDD_ROOT": str(KAGGLE_SDD_ROOT),
        "PYTHON_BIN": sys.executable,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "MALLOC_ARENA_MAX": "2",
        "CONTINUE_ON_OOM": "1",
        "SEED": SEED,
        "SAMPLING_STEPS": SAMPLING_STEPS,
        "BATCH_SIZE": BATCH_SIZE,
        "FEATURES_ROOT": str(KAGGLE_WORK / "MoFlow" / "features" / "resnet18"),
        "RESULTS_ROOT": str(KAGGLE_WORK / "MoFlow" / "results_sdd" / "cor_fm"),
        "RESEARCH_ROOT": str(KAGGLE_WORK / "MoFlow" / "report" / "research"),
    })
    return env


def run_script(name: str, args: str, env: dict) -> None:
    cmd = ["bash", str(REPO_DIR / "scripts" / "research" / name)] + args.split()
    log(f"=== {name} {' '.join(args)} ===")
    subprocess.run(cmd, cwd=str(REPO_DIR), env=env, check=False)


def pipeline(angle: str, env: dict) -> None:
    video_args = ""
    if angle == "angle2":
        video_args = "--video-id video0"
    scene_args = f"--scene {SCENE}" if SCENE else ""
    nb = "" if N_BATCHES == "0" else f"--n-batches {N_BATCHES}"

    run_script("01_video_mode_ablation.sh", f"{scene_args} {video_args}".strip(), env)
    run_script("02_zvid_attribution.sh", f"{scene_args} {nb} {video_args}".strip(), env)
    run_script("03_transfer_matrix.sh", f"{scene_args} {nb} {video_args}".strip(), env)
    run_script("04_gain_geometry.sh", f"{scene_args} {nb} --top-k {TOP_K} {video_args}".strip(), env)


def audit(env: dict) -> None:
    log("Audit: scanning /kaggle/working for artifacts...")
    manifest = []
    for p in sorted(KAGGLE_WORK.rglob("*")):
        if p.is_file():
            manifest.append(f"{p.relative_to(KAGGLE_WORK)}:{p.stat().st_size}")
    out = KAGGLE_WORK / "MANIFEST.txt"
    out.write_text("\n".join(manifest) + "\n")
    log(f"Manifest written: {out} ({len(manifest)} files)")


def main() -> None:
    log(f"PIPELINE_SCOPE={PIPELINE_SCOPE} SCENE={SCENE or 'all'} "
        f"VIDEO_ID={VIDEO_ID or '(angle default)'} N_BATCHES={N_BATCHES} "
        f"BATCH_SIZE={BATCH_SIZE} TOP_K={TOP_K}")
    check_env()
    clone_repo()
    install_deps()
    env = build_env()

    if PIPELINE_SCOPE in ("full", "encode", "angle1", "angle2"):
        run_script("00_reencode_features.sh", "--scene " + SCENE if SCENE else "", env)
    else:
        die(f"Unknown PIPELINE_SCOPE={PIPELINE_SCOPE}")

    if PIPELINE_SCOPE in ("full", "angle1"):
        pipeline("angle1", env)
    if PIPELINE_SCOPE in ("full", "angle2"):
        pipeline("angle2", env)

    audit(env)
    log("Pipeline DONE.")


if __name__ == "__main__":
    main()