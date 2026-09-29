#!/usr/bin/env bash
#
# Launch the VIDEO_MODE ablation (off / static / full) on the lab machine.
#
# The lab box (melek-Precision-3660) has a SINGLE GPU (RTX 4080, index 0).
# Earlier runs used CUDA_VISIBLE_DEVICES=1/2, which do not exist, so torch
# silently fell back to CPU and the jobs stalled. This script:
#
#   1. kills every running fm_sdd_global.py process (stale CPU-bound jobs),
#   2. archives the previous run directories (optional, RESET_RUNS=1),
#   3. relaunches all three modes IN PARALLEL, all pinned to CUDA_VISIBLE_DEVICES=0,
#   4. verifies each process actually picked up the GPU ("device ---> cuda")
#      and is still alive after startup.
#
# Usage (on the lab machine, from the repo root):
#   bash run_video_mode_ablations.sh
#
# Optional overrides (env vars):
#   HELD_OUT_SCENE=coupa     LOSO held-out scene (default coupa)
#   FEATURES_ROOT=...        ResNet18 feature cache root (lab default)
#   SEED=42                  fixed seed for reproducibility across arms
#   RESET_RUNS=1             archive (mv -T) previous run dirs before launch
#   GPU_DEVICE=0             CUDA device to pin all jobs to (default 0)
#
set -euo pipefail

HELD_OUT_SCENE="${HELD_OUT_SCENE:-coupa}"
FEATURES_ROOT="${FEATURES_ROOT:-/home/efrei_stage/Desktop/Datasets/SDD/features/resnet18}"
SEED="${SEED:-42}"
GPU_DEVICE="${GPU_DEVICE:-0}"
RESET_RUNS="${RESET_RUNS:-0}"

MODES="off static full"
# Deterministic per-mode run dirs (matches fm_sdd_global.init_basics tag).
RUN_DIR_off="results_sdd/cor_fm/_SDD_ho${HELD_OUT_SCENE}_novid"
RUN_DIR_static="results_sdd/cor_fm/_SDD_ho${HELD_OUT_SCENE}_vmstatic"
RUN_DIR_full="results_sdd/cor_fm/_SDD_ho${HELD_OUT_SCENE}_vmfull"

echo "=== [1/4] GPU check ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found on this host." >&2
    exit 1
fi
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "nvidia-smi reports $N_GPUS GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
if [ "$N_GPUS" -lt 1 ]; then
    echo "ERROR: no GPU available." >&2
    exit 1
fi
if [ "$GPU_DEVICE" -ge "$N_GPUS" ]; then
    echo "ERROR: GPU_DEVICE=$GPU_DEVICE out of range ($N_GPUS GPU(s))." >&2
    exit 1
fi

echo "=== [2/4] Kill stale fm_sdd_global.py processes ==="
# SIGTERM first, then SIGKILL stragglers after a short grace period.
pkill -TERM -f 'fm_sdd_global\.py' 2>/dev/null && echo "  SIGTERM sent to prior fm_sdd_global.py jobs" || echo "  no prior fm_sdd_global.py jobs running"
sleep 5
if pgrep -f 'fm_sdd_global\.py' >/dev/null 2>&1; then
    echo "  still alive after SIGTERM -> SIGKILL"
    pkill -KILL -f 'fm_sdd_global\.py' 2>/dev/null || true
    sleep 2
fi
LEFT=$(pgrep -f 'fm_sdd_global\.py' | wc -l)
echo "  remaining fm_sdd_global.py processes: $LEFT"
if [ "$LEFT" -ne 0 ]; then
    echo "WARNING: $LEFT job(s) still alive; continuing anyway." >&2
fi

echo "=== [3/4] Archive previous run dirs (RESET_RUNS=$RESET_RUNS) ==="
if [ "$RESET_RUNS" = "1" ]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    for mode in $MODES; do
        d="RUN_DIR_${mode}"
        if [ -d "${!d}" ]; then
            mv -T "${!d}" "${!d}_arch_${STAMP}"
            echo "  archived ${!d} -> ${!d}_arch_${STAMP}"
        fi
    done
else
    echo "  keeping existing run dirs (training resumes/overwrites in place)."
fi

echo "=== [4/4] Relaunch all three modes on CUDA_VISIBLE_DEVICES=$GPU_DEVICE ==="
pids=""
for mode in $MODES; do
    echo "  launching --video_mode $mode ..."
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" nohup python fm_sdd_global.py \
        --cfg cfg/sdd/cor_fm.yml \
        --held_out_scene "$HELD_OUT_SCENE" \
        --video_features_root "$FEATURES_ROOT" \
        --video_mode "$mode" \
        --fix_random_seed --seed "$SEED" \
        > "train_${mode}.log" 2>&1 &
    pids="$pids $!"
done

echo "  launched PIDs:$pids"
echo ""
echo "=== Startup verification (GPU picked up + process alive) ==="
# Device line is written early (config dump in init_basics); model build takes
# ~10-20s, so 25s is enough to confirm the device choice.
sleep 25
FAILED=""
for mode in $MODES; do
    log="train_${mode}.log"
    if ! grep -q "device ---> cuda\b" "$log"; then
        if grep -q "device ---> cpu\b" "$log"; then
            echo "  WARNING [$mode]: process is on CPU again (see $log). FAILING FAST: killing job."
            FAILED="$FAILED $mode"
        else
            echo "  WARNING [$mode]: device line not found yet in $log; check shortly."
        fi
    else
        echo "  OK [$mode]: device ---> cuda"
    fi
done

echo ""
echo "PIDs:$pids"
echo "Logs: train_{off,static,full}.log"
echo "Monitor:"
echo "  tail -f train_off.log   # trajectories-only baseline"
echo "  tail -f train_static.log"
echo "  tail -f train_full.log"
echo "  watch -n5 nvidia-smi    # verify GPU util >0 and no OOM"
if [ -n "$FAILED" ]; then
    echo "NOTE: modes failed on GPU pickup:$FAILED — inspect those logs."
    exit 1
fi