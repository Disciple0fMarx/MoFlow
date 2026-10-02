#!/usr/bin/env bash
# 00_common.sh — shared environment + helpers for the research script suite.
#
# Sources once from any of the 0X_*.sh entry points. It does NOT execute
# anything itself; it only defines variables and functions the entry points
# call. This keeps the methodological knobs (paths, seeds, LOSO loop) in one
# audit-friendly place.
#
# ==== PyPI-free / no external deps: pure bash + the project venv python. ====
set -euo pipefail

# ---------------------------------------------------------------------------
# Repo / interpreter resolution (works when sourced from scripts/research/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    for cand in \
        "${REPO_ROOT}/.venv/bin/python" \
        "${REPO_ROOT}/venv/bin/python" \
        "$(command -v python3 || true)"; do
        if [[ -n "${cand}" && -x "${cand}" ]]; then
            PYTHON_BIN="${cand}"
            break
        fi
    done
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "[00_common] ERROR: no python found. Set PYTHON_BIN explicitly." >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Dataset / feature / output locations (all overridable via env)
# ---------------------------------------------------------------------------
# Lab machine SDD annotations (annotations/<scene>/<video_id>/annotations.txt).
SDD_ROOT="${SDD_ROOT:-/home/efrei_stage/Desktop/Datasets/SDD}"
# Writable project-local video feature cache (<scene>.npy + <scene>.manifest.parquet).
# NOT the read-only $SDD_ROOT/features/resnet18 — that dir must never be read.
FEATURES_ROOT="${FEATURES_ROOT:-/home/efrei_stage/MoFlow/features/resnet18}"
# Where fm_sdd_global.py writes run dirs (_SDD_ho<scene>_vm{static,full} / _novid).
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results_sdd/cor_fm}"
# Where this suite writes its aggregated reports / renderings.
RESEARCH_ROOT="${RESEARCH_ROOT:-${REPO_ROOT}/report/research}"

# ---------------------------------------------------------------------------
# LOSO protocol configuration
# ---------------------------------------------------------------------------
SDD_SCENES=(bookstore coupa deathCircle gates hyang little nexus quad)
# One NVIDIA RTX 4080 (index 0) on the lab box; overridable for multi-GPU hosts.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Fixed randomness: every experiment in a family shares the same seed so gains
# are attributable to the conditioning channel, not sampling luck.
SEED="${SEED:-42}"
# Sampling schedule shared by every train/eval so results are comparable.
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"
# Single-video ablation: restrict every LOSO-selected split to this video
# folder name (e.g. 'video0'). Empty = all videos of each scene. Geographic
# LOSO still holds — the filter is applied AFTER scene selection.
VIDEO_ID="${VIDEO_ID:-}"
# Rebuild the --video-id CLI args from the current VIDEO_ID. Must be called
# AFTER a research script parses its own args (it sources 00_common first).
# usage: build_video_args; ARGS+=( "${video_args[@]}" )
build_video_args() {
    video_args=()
    if [[ -n "${VIDEO_ID}" ]]; then
        video_args=(--video-id "$VIDEO_ID")
    fi
}
build_video_args  # initial state before scripts override VIDEO_ID

# usage: vid_out_suffix  -> echoes "_vid<name>" (or "" when unfiltered) so a
# single-video run never overwrites the full-scene geographic-LOSO artifacts.
vid_out_suffix() {
    if [[ -n "${VIDEO_ID}" ]]; then
        echo "_vid${VIDEO_ID}"
    fi
}

export CUDA_VISIBLE_DEVICES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# usage: loso_scenes <label> [scene]   -> echoes the space-separated scene list
# A single optional scene arg restricts the LOSO loop; otherwise all 8 scenes
# run. Emits the label (for logs) on stdout together with the scenes.
loso_scenes() {
    local label="$1"
    if [[ $# -ge 2 && -n "$2" ]]; then
        local s="$2" ok=0
        for c in "${SDD_SCENES[@]}"; do
            [[ "$s" == "$c" ]] && ok=1
        done
        if ((ok != 1)); then
            echo "[${label}] ERROR: '$s' is not a valid SDD scene." >&2
            echo "  valid: ${SDD_SCENES[*]}" >&2
            exit 3
        fi
        echo "$s"
    else
        echo "${SDD_SCENES[@]}"
    fi
}

# usage: log_to <file> <level> <message>
log_to() {
    local file="$1" level="$2"
    shift 2
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[${ts}] ${level} $*" | tee -a "$file"
}

# usage: require_ckpt <path> <label>   -> hard-fails if checkpoint missing
require_ckpt() {
    local p="$1" label="$2"
    if [[ "${RESEARCH_DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    if [[ ! -f "$p" ]]; then
        echo "[${label}] ERROR: checkpoint missing: ${p}" >&2
        echo "  → have you run the training step for this scene first?" >&2
        exit 4
    fi
}

# usage: run_py <label> <logfile> -- args...   -> runs the project python
run_py() {
    local label="$1" logfile="$2"
    shift 2
    echo "---------------------------------------------------------------------" >> "$logfile"
    log_to "$logfile" INFO "[${label}] cmd:" "$PYTHON_BIN" "$@"
    if [[ "${RESEARCH_DRY_RUN:-0}" == "1" ]]; then
        log_to "$logfile" INFO "[${label}] (dry-run — command not executed)"
        return 0
    fi
    if ! (cd "$REPO_ROOT" && "$PYTHON_BIN" "$@"); then
        log_to "$logfile" ERROR "[${label}] command failed (rc=$?)"
        exit 5
    fi
    log_to "$logfile" INFO "[${label}] OK"
}

# Write a provenance file describing how a report was produced (audit trail).
write_provenance() {
    local outdir="$1" script="$2"
    mkdir -p "$outdir"
    local git_sha="unknown"
    if (command -v git >/dev/null && cd "$REPO_ROOT" && git rev-parse --short HEAD) >/dev/null 2>&1; then
        git_sha="$(cd "$REPO_ROOT" && git rev-parse --short HEAD)"
    fi
    {
        echo "# Provenance"
        echo ""
        echo "- Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "- Script:    ${script}"
        echo "- Repo SHA:  ${git_sha}"
        echo "- Seed:      ${SEED}"
        echo "- Sampling steps: ${SAMPLING_STEPS}"
        echo "- CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
        echo "- SDD_ROOT:  ${SDD_ROOT}"
        echo "- FEATURES_ROOT: ${FEATURES_ROOT}"
        echo "- RESULTS_ROOT: ${RESULTS_ROOT}"
        echo "- VIDEO_ID:  ${VIDEO_ID:-all}"
    } > "$outdir/PROVENANCE.md"
}