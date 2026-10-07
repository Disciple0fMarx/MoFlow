#!/usr/bin/env bash
# 02_zvid_attribution.sh — Q1 evidence: is the gain actually from z_vid?
#
# Runs the cvxp attribution pipeline (tools/attrib_video_conditioning.py) on
# every LOSO scene's `full` VIDEO_MODE checkpoint. For each scene we get three
# statistically controlled condition arms — baseline / zeroed / permuted — on
# the SAME seeded sampling noise, giving:
#
#   baseline vs zeroed : total causal influence of the video channel
#   baseline vs permuted : whether per-window CONTENT matters (vs "any video")
#
# having a zeroed/permuted arm with significant ADE/FDE movement relative to
# baseline + non-zero d_traj is the quantitative answer to "the gain comes
# from z_vid, not from a confound". Per-scene CSV rows are emitted by the tool
# itself (its fixed-RNG design makes them directly comparable).
#
# Usage:
#   scripts/research/02_zvid_attribution.sh [--scene <scene>]
#   scripts/research/02_zvid_attribution.sh --n-batches 0 --batch-size 64
#   scripts/research/02_zvid_attribution.sh --conditions baseline permuted
#
# Flags: --scene <s>, --n-batches <n> (0/None=all), --batch-size <b>,
#        --conditions (subset of: baseline zeroed permuted), --use-ema.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_common.sh
source "${DIR}/00_common.sh"

LABEL="q1-zvid-attribution"
SCENE=""
N_BATCHES=""
BATCH_SIZE="${BATCH_SIZE:-64}"   # env-tunable memory knob (see 00_common.sh)
CONDITIONS=(baseline zeroed permuted)
USE_EMA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --n-batches) N_BATCHES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --conditions) shift; CONDITIONS=(); while [[ $# -gt 0 && "$1" != --* ]]; do CONDITIONS+=("$1"); shift; done ;;
        --use-ema) USE_EMA="--use-ema"; shift ;;
        --video-id) VIDEO_ID="$2"; shift 2 ;;
        --dry-run) RESEARCH_DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 3 ;;
    esac
done
build_video_args
OUTSUF="$(vid_out_suffix)"

OUTDIR="${RESEARCH_ROOT}/q1"
mkdir -p "$OUTDIR"
LOG="${OUTDIR}/run.log"
: > "$LOG"
write_provenance "$OUTDIR" "scripts/research/02_zvid_attribution.sh"
log_to "$LOG" INFO "Q1 starting. scenes=($(loso_scenes "$LABEL" "$SCENE")) conditions=(${CONDITIONS[*]}) n_batches=${N_BATCHES:-all}"

CFG="cfg/sdd/cor_fm.yml"
for scene in $(loso_scenes "$LABEL" "$SCENE"); do
    CKPT="${RESULTS_ROOT}/_SDD_ho${scene}_vmfull${OUTSUF}/models/checkpoint_best.pt"
    require_ckpt "$CKPT" "$LABEL"
    ensure_features "$scene"
    log_to "$LOG" INFO "[${scene}] attribution on ${CKPT}"

    ARGS=(tools/attrib_video_conditioning.py
        --cfg "$CFG"
        --ckpt "$CKPT"
        --sdd-root "$SDD_ROOT"
        --video-features-root "$FEATURES_ROOT"
        --held-out-scene "$scene"
        --split test
        --batch-size "$BATCH_SIZE"
        --conditions "${CONDITIONS[@]}"
        --out "$OUTDIR/${scene}_attrib${OUTSUF}")
    [[ -n "${N_BATCHES}" && "${N_BATCHES}" != "0" ]] && ARGS+=(--n-batches "$N_BATCHES")
    [[ -n "${USE_EMA}" ]] && ARGS+=("$USE_EMA")
    ARGS+=( "${video_args[@]}" )
    run_py "attrib:${scene}" "$LOG" "${ARGS[@]}"
done

build_supervisor_layout 2>/dev/null || true

log_to "$LOG" INFO "Q1 complete. per-scene CSVs in ${OUTDIR} (each *_attrib.csv)"