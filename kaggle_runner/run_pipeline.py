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
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORK = Path("/kaggle/working")
REPO_DIR = KAGGLE_WORK / "MoFlow"
RESEARCH_DIR = REPO_DIR / "report" / "research"
RESULTS_DIR = REPO_DIR / "results_sdd" / "cor_fm"
# Set SUPERVISOR_COPY=1 to materialise real files in q3/q5/q7 instead of
# relative symlinks (useful if the output tarball does not preserve symlinks).
SUPERVISOR_COPY = os.environ.get("SUPERVISOR_COPY", "0") == "1"
# Copy the whole results tree is never intended; the final sweep below removes
# every checkpoint once the pipeline is fully done (scope=full only).
PRUNE_CHECKPOINTS = os.environ.get("MF_PRUNE_CHECKPOINTS", "")

PIPELINE_SCOPE = os.environ.get("PIPELINE_SCOPE", "full").lower()
SCENE = os.environ.get("SCENE", "")
VIDEO_ID = os.environ.get("VIDEO_ID", "")
N_BATCHES = os.environ.get("N_BATCHES", "0")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "64")
TOP_K = os.environ.get("TOP_K", "3")
SEED = os.environ.get("SEED", "42")
SAMPLING_STEPS = os.environ.get("SAMPLING_STEPS", "10")


# ---------------------------------------------------------------------------
# Logging (console + durable file so a SIGKILL still leaves a transcript)
# ---------------------------------------------------------------------------
LOG_FILE = Path(os.environ.get(
    "MF_LOG_FILE", str(KAGGLE_WORK / "pipeline.log")))
_log_fh = None


def _ensure_log_fh():
    global _log_fh
    if _log_fh is None:
        try:
            KAGGLE_WORK.mkdir(parents=True, exist_ok=True)
            _log_fh = open(LOG_FILE, "a")
        except Exception:
            _log_fh = None


def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {level} {msg}"
    print(line, flush=True)
    _ensure_log_fh()
    if _log_fh is not None:
        try:
            _log_fh.write(line + "\n")
            _log_fh.flush()
            os.fsync(_log_fh.fileno())
        except Exception:
            pass


def die(msg: str) -> None:
    log(msg, "ERROR")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------
def discover_sdd_root() -> Path:
    """Locate the mounted SDD dataset under /kaggle/input.

    Kaggle mounts a ``dataset_sources`` entry at a path whose spelling depends
    on the dataset/version; the bare-slug form (``/kaggle/input/<slug>``) is
    common but several mirrors appear under ``/kaggle/input/datasets/<owner>/<slug>``.
    We therefore scan /kaggle/input for a directory containing an ``annotations/``
    dir AND a ``videos/`` dir AND an ``annotations/<scene>/<video>/annotations.txt``
    sample — the definitive SDD layout signature. Explicit overrides via
    ``MF_SDD_ROOT`` still win.
    """
    explicit = os.environ.get("MF_SDD_ROOT")
    if explicit:
        p = Path(explicit)
        if (p / "annotations").is_dir() and (p / "videos").is_dir():
            return p
        raise RuntimeError(f"MF_SDD_ROOT set but missing annotations/videos: {p}")

    if not KAGGLE_INPUT.exists():
        raise RuntimeError("/kaggle/input not found — not running in Kaggle")

    # (a) the bare slug the metadata _should_ produce
    for cand in (KAGGLE_INPUT / "stanford-drone-dataset",
                 KAGGLE_INPUT / "brendanalvey/stanford-drone-dataset",
                 KAGGLE_INPUT / "datasets/stanford-drone-dataset",
                 KAGGLE_INPUT / "datasets/brendanalvey/stanford-drone-dataset"):
        if (cand / "annotations").is_dir() and (cand / "videos").is_dir():
            return cand

    # (b) fallback: any subdir (2 levels deep) that carries the layout signature
    for level in (KAGGLE_INPUT.iterdir(), KAGGLE_INPUT.glob("*/")):
        for d in sorted(level if isinstance(level, list) else level):
            if not d.is_dir():
                continue
            if (d / "annotations").is_dir() and (d / "videos").is_dir():
                return d
            for sub in sorted(d.iterdir()):
                if (sub / "annotations").is_dir() and (sub / "videos").is_dir():
                    return sub
        break
    raise RuntimeError(
        f"No SDD dataset layout (annotations/ + videos/) found under {KAGGLE_INPUT}. "
        "Check kernel-metadata.json dataset_sources."
    )


def check_env() -> Path:
    try:
        sdd_root = discover_sdd_root()
    except RuntimeError as exc:
        die(str(exc))
    log(f"SDD dataset resolved to {sdd_root}")
    ann = sdd_root / "annotations"
    vid = sdd_root / "videos"
    scenes = sorted(p.name for p in vid.iterdir() if p.is_dir())
    log(f"Dataset OK: {len(scenes)} scenes under {vid}: {scenes}")
    sample = next((p for p in ann.rglob("annotations.txt")), None)
    if sample is None:
        die("No annotations.txt found under the dataset — wrong mount?")
    log(f"Annotation sample: {sample}")
    if not shutil.which("nvidia-smi"):
        log("WARNING: nvidia-smi not found — GPU may be unavailable.", "WARN")
    else:
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                       check=False)
    return sdd_root


def install_deps() -> None:
    """Install the project deps the Kaggle image lacks, WITHOUT pinning/upgrade.

    The image (py3.13) already ships modern torch, numpy, matplotlib, scipy,
    PyYAML, tqdm, pandas — pinning those to the lab's versions (numpy==2.2.4,
    matplotlib==3.8.3) creates pip resolution conflicts with the image's torch,
    and ``--upgrade`` risks churning the image's pinned stack mid-session.
    We install only the pure-python deps requirements.txt declares that the
    image may not have, letting pip keep satisfying versions already present.
    """
    log("Installing Python deps (keeping image torch/numpy/matplotlib)...")
    project_only = [
        "accelerate", "easydict", "einops", "ema_pytorch", "GitPython",
        "tensorboardX",
    ]
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir",
           "--no-input", "--no-warn-script-location", *project_only]
    log("pip install: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    # Media + IO extras the image may not have.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir",
         "--no-input", "--no-warn-script-location",
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
def build_env(sdd_root: Path) -> dict:
    env = dict(os.environ)
    env.update({
        "SDD_ROOT": str(sdd_root),
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


def _link(src: Path, dst: Path) -> None:
    """Materialise dst as a (relative symlink to src) or a copy."""
    try:
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst, ignore_errors=True)
            else:
                dst.unlink()
    except OSError:
        pass
    if SUPERVISOR_COPY:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(os.path.relpath(src, dst.parent))
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def wire_supervisor_layout() -> None:
    """Populate the supervisor-facing q3/q5/q7 aliases in report/research/.

    Canonical artifacts stay where the research scripts wrote them; q3/q5/q7 are
    thin links so the supervisor's Q-mapping resolves without moving files:
        q3 → features/PROVENANCE*.md + every execution log (provenance trail)
        q5 → q4/q4_scene_gain_summary*.csv + q4/q4_window_gains*.csv
        q7 → q4/figs*/  (roundabout visual renders)
    """
    root = RESEARCH_DIR
    if not root.is_dir():
        log("Supervisor layout: research dir absent — nothing to wire.", "WARN")
        return
    q3, q5, q7 = root / "q3", root / "q5", root / "q7"
    for d in (q3, q5, q7):
        d.mkdir(parents=True, exist_ok=True)

    n3 = n5 = n7 = 0
    for f in sorted((root / "features").glob("PROVENANCE*.md")) + \
             sorted((root / "features").glob("*.log")):
        _link(f, q3 / f.name); n3 += 1
    for f in sorted(root.glob("q*/run.log")) + sorted(root.glob("q*/render.log")):
        _link(f, q3 / f"{f.parent.name}_{f.name}"); n3 += 1
    for pat in ("q4_scene_gain_summary*.csv", "q4_window_gains*.csv"):
        for f in sorted((root / "q4").glob(pat)):
            _link(f, q5 / f.name); n5 += 1
    for d in sorted((root / "q4").glob("figs*")):
        if d.is_dir():
            _link(d, q7 / d.name); n7 += 1
    log(f"Supervisor layout: q3={n3} links, q5={n5} links, q7={n7} links "
        f"(copy={SUPERVISOR_COPY})")


def disk_report(tag: str = "") -> None:
    try:
        total, used, free = shutil.disk_usage(KAGGLE_WORK)
    except OSError:
        return
    log(f"Disk [{tag or '-'}] working: {used/1e9:.2f} GB used / "
        f"{total/1e9:.2f} GB ({free/1e9:.2f} GB free)")


def prune_transients() -> None:
    """Remove large, non-deliverable transient files before packaging.

    TensorBoard event files are the only sizeable throwaway the pipeline writes
    under results_sdd; the code_backup/ dirs are kept for audit provenance.
    """
    freed = n = 0
    for p in RESULTS_DIR.rglob("events.out.tfevents.*"):
        try:
            freed += p.stat().st_size
            p.unlink()
            n += 1
        except OSError:
            pass
    if n:
        log(f"Disk guard: removed {n} TensorBoard event files ({freed/1e6:.1f} MB).")


def prune_checkpoints() -> None:
    """Final sweep: remove every checkpoint once the whole pipeline is done.

    Only safe for PIPELINE_SCOPE=full (both angles) because the angle-2 scripts
    currently reference the unsuffixed angle-1 checkpoint dirs; deleting them
    mid-scope would break a later stage.
    """
    freed = n = 0
    for p in RESULTS_DIR.rglob("checkpoint_*"):
        if p.suffix not in (".pt", ".pth"):
            continue
        try:
            freed += p.stat().st_size
            p.unlink()
            n += 1
        except OSError:
            pass
    log(f"Disk guard: final checkpoint sweep removed {n} files ({freed/1e6:.1f} MB).")


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
    sdd_root = check_env()
    clone_repo()
    install_deps()
    env = build_env(sdd_root)

    if PIPELINE_SCOPE in ("full", "encode", "angle1", "angle2"):
        run_script("00_reencode_features.sh", "--scene " + SCENE if SCENE else "", env)
    else:
        die(f"Unknown PIPELINE_SCOPE={PIPELINE_SCOPE}")
    disk_report("after encode")

    if PIPELINE_SCOPE in ("full", "angle1"):
        pipeline("angle1", env)
    disk_report("after angle1")
    if PIPELINE_SCOPE in ("full", "angle2"):
        pipeline("angle2", env)
    disk_report("after angle2")

    # Free throwaway disk before packaging, then expose the supervisor layout.
    prune_transients()
    wire_supervisor_layout()
    if PIPELINE_SCOPE == "full" and PRUNE_CHECKPOINTS != "0":
        prune_checkpoints()
    disk_report("before audit")

    audit(env)
    log("Pipeline DONE.")


if __name__ == "__main__":
    log("Script started.", "BOOT")
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        log(f"FATAL {type(exc).__name__}: {exc}", "ERROR")
        import traceback
        traceback.print_exc()
        try:
            (KAGGLE_WORK / "FATAL.txt").write_text(
                traceback.format_exc() + f"\n{type(exc).__name__}: {exc}\n")
        except Exception:
            pass
        sys.exit(1)
    else:
        log("main() returned cleanly.")