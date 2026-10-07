#!/usr/bin/env bash
# run_full_suite.sh — run the WHOLE research suite for both angles, then build
# the supervisor Q1–Q7 layout.
#
# Two angles are produced so the supervisor's "loose scene granularity"
# curiosity (Q3) is answered with hard evidence:
#   angle 1 : geographic LOSO, all videos        (report files WITHOUT _vid suffix)
#   angle 2 : single-video filter --video-id video0 (report files WITH _vidvideo0)
# The single-video filter is applied AFTER LOSO scene selection, so geographic
# isolation still governs train/test pooling (see README.md "Defining a Scene").
#
# Order (each stage is its own subprocess; a failure is logged and the run
# continues, with a summary + non-zero exit at the end):
#   00  optional feature-cache re-encode  (REENCODE=auto|always|never)
#   01  Q6 video-mode ablation -> checkpoints + q6 CSVs      (angle1, angle2)
#   02  Q1 z_vid attribution   -> q1 CSVs                    (angle1, angle2)
#   03  Q2 transfer matrix     -> q2 CSVs+8x8                (angle1, angle2)
#   04  Q5/Q7 gain geometry    -> q4 CSVs + figs + roundabout (angle1, angle2)
#   +   build_supervisor_layout -> q3/q5/q7 aliases + q4 design-note snapshots
#
# Usage:
#   scripts/research/run_full_suite.sh                                   # both angles
#   scripts/research/run_full_suite.sh --angle 1                         # angle 1 only
#   scripts/research/run_full_suite.sh --stages "01 02" --angle 2        # subset, angle 2
#   scripts/research/run_full_suite.sh --reencode always                 # force encode
#   scripts/research/run_full_suite.sh --n-batches 3 --top-k 2 --no-render  # quick pass
#   scripts/research/run_full_suite.sh --dry-run                         # print, no exec
#   REENCODE=never CONTINUE_ON_OOM=0 scripts/research/run_full_suite.sh  # env knobs
#
# Env knobs (all overridable): REENCODE (auto|always|never), STAGES ("01 02 03 04"),
#   VIDEO0 (default video0), TRANSFER_NBATCHES (default 0=all cells),
#   TOP_K (default 3), RENDER (default 1), CLEANUP_CHECKPOINTS (default 0),
#   STOP_ON_ERROR (default 0), plus everything in 00_common.sh (SDD_ROOT, SEED, ...).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_common.sh
source "${DIR}/00_common.sh"
set +e   # this orchestrator collects failures instead of aborting (-u/pipefail stay on)

REENCODE="${REENCODE:-auto}"
STAGES="${STAGES:-01 02 03 04}"
VIDEO0="${VIDEO0:-video0}"
TRANSFER_NBATCHES="${TRANSFER_NBATCHES:-0}"
TOP_K="${TOP_K:-3}"
RENDER="${RENDER:-1}"
CLEANUP_CHECKPOINTS="${CLEANUP_CHECKPOINTS:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
ANGLES="1 2"

# Export so every child stage (which sources 00_common.sh fresh) honours it.
export RESEARCH_DRY_RUN="${RESEARCH_DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --angle) ANGLES="$2"; shift 2 ;;
        --reencode) REENCODE="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --video-id) VIDEO0="$2"; shift 2 ;;
        --n-batches) TRANSFER_NBATCHES="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --no-render) RENDER=0; shift ;;
        --cleanup) CLEANUP_CHECKPOINTS=1; shift ;;
        --stop-on-error) STOP_ON_ERROR=1; shift ;;
        --dry-run) RESEARCH_DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 3 ;;
    esac
done

SUCCEEDED=()
FAILED=()
mkdir -p "${RESEARCH_ROOT}"
LOG="${RESEARCH_ROOT}/_full_suite.log"
: > "$LOG"

# usage: run_stage <label> <script> [args...]  -> subprocess; record outcome
run_stage() {
    local label="$1" script="$2"
    shift 2
    # --video-id "" is passed explicitly so an exported VIDEO_ID in the caller's
    # environment can never silently leak into an angle-1 child.
    log_to "$LOG" INFO "[${label}] START: bash ${script} $*"
    local rc=0
    bash "$script" "$@" 2>&1 | tee -a "$LOG"
    rc="${PIPESTATUS[0]}"
    if ((rc == 0)); then
        log_to "$LOG" INFO "[${label}] OK"
        SUCCEEDED+=("${label}")
    else
        log_to "$LOG" ERROR "[${label}] FAILED rc=${rc}"
        FAILED+=("${label}:rc=${rc}")
        if ((STOP_ON_ERROR)); then
            log_to "$LOG" ERROR "STOP_ON_ERROR=1 — aborting"
            exit 1
        fi
    fi
}

# usage: run_angle <video_id>
run_angle() {
    local vid="$1" tag
    tag="angle1"
    [[ -n "$vid" ]] && tag="angle2_${vid}"
    log_to "$LOG" "=============================================================="
    log_to "$LOG" "=== ${tag}: video_id=${vid:-<unfiltered geographic LOSO>} ==="
    local step
    for step in ${STAGES}; do
        case "$step" in
            01) run_stage "${tag}/01" "${DIR}/01_video_mode_ablation.sh" --video-id "$vid" ;;
            02) run_stage "${tag}/02" "${DIR}/02_zvid_attribution.sh" --video-id "$vid" ;;
            03) run_stage "${tag}/03" "${DIR}/03_transfer_matrix.sh" --video-id "$vid" --n-batches "${TRANSFER_NBATCHES}" ;;
            04) run_stage "${tag}/04" "${DIR}/04_gain_geometry.sh" --video-id "$vid" --top-k "${TOP_K}" $([[ "$RENDER" == 1 ]] && echo "" || echo --no-render) ;;
            *) log_to "$LOG" ERROR "unknown stage: ${step}" ;;
        esac
    done
}

log_to "$LOG" "##############################################################"
log_to "$LOG" "full-suite start: angles=(${ANGLES}) stages=(${STAGES}) reencode=${REENCODE}"
log_to "$LOG" "  seed=${SEED} sampling_steps=${SAMPLING_STEPS} cuda=${CUDA_VISIBLE_DEVICES}"
log_to "$LOG" "  sdd_root=${SDD_ROOT} features_root=${FEATURES_ROOT} results_root=${RESULTS_ROOT}"

# Phase 0 — optional feature-cache re-encode (one-time / corrupted-cache fix).
# auto = run 00 only when any of the 8 per-scene caches is missing; a pre-fix
# (frame-collapsed) cache that exists is detected LOUDLY at first use (see
# video_encoder/sdd_adapter.py) — set REENCODE=always to force a clean rebuild.
if [[ "${STAGES}" == *"00"* ]]; then
    case "${REENCODE}" in
        always) run_stage "angle1/00" "${DIR}/00_reencode_features.sh" ;;
        never)  log_to "$LOG" INFO "[00] skipped by REENCODE=never" ;;
        auto)
            MISSING=""
            for s in "${SDD_SCENES[@]}"; do
                [[ -f "${FEATURES_ROOT}/${s}.npy" && -f "${FEATURES_ROOT}/${s}.manifest.parquet" ]] || MISSING+=" ${s}"
            done
            if [[ -n "${MISSING}" ]]; then
                log_to "$LOG" INFO "[00]${MISSING} cache(s) missing — re-encoding"
                run_stage "angle1/00" "${DIR}/00_reencode_features.sh"
            else
                log_to "$LOG" INFO "[00] all 8 feature caches present (REENCODE=auto) — skipping 00"
            fi
            ;;
        *) log_to "$LOG" ERROR "unknown REENCODE=${REENCODE} (auto|always|never)" ;;
    esac
fi

# Angles 1 and 2 — the four Q-evidence scripts each.
if [[ "${ANGLES}" == *"1"* ]]; then
    run_angle ""
fi
if [[ "${ANGLES}" == *"2"* ]]; then
    run_angle "${VIDEO0}"
fi

# Supervisor layout: q3 (provenance+logs), q5 (gain summaries), q7 (figures)
# and the Q4 design-note + config snapshots under q4/.
log_to "$LOG" "building supervisor layout (q3/q5/q7 aliases + q4 design snapshots)"
build_supervisor_layout 2>/dev/null

if [[ "$CLEANUP_CHECKPOINTS" == "1" && "${RESEARCH_DRY_RUN:-0}" != "1" ]]; then
    log_to "$LOG" INFO "[cleanup] CLEANUP_CHECKPOINTS=1 — removing trained checkpoints under results_sdd run dirs"
    for pat in "*_vmfull*" "*_novid*" "*_vmstatic*"; do
        for d in "${RESULTS_ROOT}"/${pat}; do
            [[ -d "$d" && -d "${d}/models" ]] && rm -f "${d}"/models/checkpoint_*.pt "${d}"/models/checkpoint_*.pth
        done
    done
fi

echo
log_to "$LOG" "##############################################################"
if (( ${#FAILED[@]} == 0 )); then
    log_to "$LOG" "full-suite DONE: all ${#SUCCEEDED[@]} stages OK — reports under ${RESEARCH_ROOT}"
    exit 0
fi
for f in "${FAILED[@]}"; do
    log_to "$LOG" ERROR "FAILED stage: ${f}"
done
log_to "$LOG" ERROR "re-run failed stages individually (see per-stage run.log files). Summary: ${LOG}"
exit 1