#!/usr/bin/env bash
# 01_video_mode_ablation.sh — Q6 evidence: does video beat a static image?
#
# For every LOSO scene (default: all 8; optional single-scene arg) trains and
# evaluates three VIDEO_MODE arms of the SAME FlowMatching model:
#
#   off     → trajectory-only (video branch never built)     [_novid]
#   static  → one fixed scene vector for every window        [_vmstatic]
#   full    → per-window mean-pooled video lookup            [_vmfull]
#
# Only 'full' carries per-window visual information; 'static' is the
# no-per-window-content image control. Comparing off/static/full isolates:
#   off→full : total visual gain
#   static→full : the gain attributable to per-window content (what a naive
#                 "just use an image" baseline would wrongly claim as video).
#
# The trainer's own test split is used (LOSO held-out scene). Metrics are
# duplicated into report/research/q6/ (CSV + provenance). Every arm uses the
# SAME seed and sampling steps so differences are causal, not sampling luck.
#
# Usage:
#   scripts/research/01_video_mode_ablation.sh [--scene <scene>]
#   scripts/research/01_video_mode_ablation.sh --train-only --eval-only
#   RESEARCH_DRY_RUN=1 scripts/research/01_video_mode_ablation.sh --scene coupa
#
# Flags: --scene <s> (LOSO single scene), --train-only, --eval-only,
#        --modes "off static full" (space-separated subset).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_common.sh
source "${DIR}/00_common.sh"

LABEL="q6-video-mode"
SCENE=""
TRAIN_ONLY=0
EVAL_ONLY=0
MODES="off static full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --train-only) TRAIN_ONLY=1; shift ;;
        --eval-only) EVAL_ONLY=1; shift ;;
        --modes) MODES="$2"; shift 2 ;;
        --video-id) VIDEO_ID="$2"; shift 2 ;;
        --dry-run) RESEARCH_DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 3 ;;
    esac
done
build_video_args
if ((TRAIN_ONLY && EVAL_ONLY)); then
    echo "[${LABEL}] ERROR: --train-only and --eval-only are mutually exclusive." >&2
    exit 3
fi

OUTDIR="${RESEARCH_ROOT}/q6"
mkdir -p "$OUTDIR"
LOG="${OUTDIR}/run.log"
: > "$LOG"
write_provenance "$OUTDIR" "scripts/research/01_video_mode_ablation.sh"
log_to "$LOG" INFO "Q6 starting. scenes=($(loso_scenes "$LABEL" "$SCENE")) modes=(${MODES}) train_only=${TRAIN_ONLY} eval_only=${EVAL_ONLY}"

# Feature caches are needed for the eval scene(s); training additionally needs
# the other 7 scenes' caches. ensure_feature_list auto-encodes any that are missing.
if ((EVAL_ONLY)); then
    ensure_feature_list "$(loso_scenes "$LABEL" "$SCENE")"
else
    ensure_feature_list "${SDD_SCENES[@]}"
fi

CFG="cfg/sdd/cor_fm.yml"

for scene in $(loso_scenes "$LABEL" "$SCENE"); do
    for mode in ${MODES}; do
        log_to "$LOG" INFO "[${scene}/${mode}] begin"
        if ((EVAL_ONLY != 1)); then
            run_py "train:${scene}/${mode}" "$LOG" \
                fm_sdd_global.py \
                --cfg "$CFG" \
                --sdd_root "$SDD_ROOT" \
                --held_out_scene "$scene" \
                --video_mode "$mode" \
                --video_features_root "$FEATURES_ROOT" \
                --fix_random_seed --seed "$SEED" \
                --sampling_steps "$SAMPLING_STEPS" \
                "${video_args[@]}"
        fi
        if ((TRAIN_ONLY != 1)); then
            run_py "eval:${scene}/${mode}" "$LOG" \
                fm_sdd_global.py \
                --cfg "$CFG" \
                --sdd_root "$SDD_ROOT" \
                --held_out_scene "$scene" \
                --video_mode "$mode" \
                --video_features_root "$FEATURES_ROOT" \
                --fix_random_seed --seed "$SEED" \
                --sampling_steps "$SAMPLING_STEPS" \
                --eval \
                "${video_args[@]}"
        fi
    done
done

log_to "$LOG" INFO "aggregating eval metrics -> ${OUTDIR}/q6_video_mode_ade.csv"
SCENE_LIST="$(loso_scenes "$LABEL" "$SCENE")"
if [[ "${RESEARCH_DRY_RUN:-0}" != "1" ]]; then
    SCENE_LIST="$SCENE_LIST" MODES="$MODES" \
        RESULTS_ROOT="$RESULTS_ROOT" RESEARCH_ROOT="$RESEARCH_ROOT" \
        OUTSUF="$(vid_out_suffix)" \
        "$PYTHON_BIN" - <<PYEOF
import csv, os, sys
root_dir = os.environ["RESULTS_ROOT"]
scenes = os.environ["SCENE_LIST"].split()
modes = os.environ["MODES"].split()
outsuf = os.environ.get("OUTSUF", "")
suffix = {"off": "novid", "static": "vmstatic", "full": "vmfull"}
rows, header = [], ["scene", "video_mode"]
# columns: ADE_min / FDE_min at the 4 horizons the trainer reports (1.2..4.8s)
for metric in ("ADE_min", "FDE_min"):
    for t in (1, 2, 3, 4):
        header.append(f"{metric}_{t}s")
rows.append(header)
for scene in scenes:
    for mode in modes:
        run_dir = os.path.join(root_dir, f"_SDD_ho{scene}_{suffix[mode]}{outsuf}")
        csv_path = next(
            (os.path.join(run_dir, *parts)
             for parts in (("eval_test_metrics.csv",), ("log", "eval_test_metrics.csv"))
             if os.path.exists(os.path.join(run_dir, *parts))),
            None,
        )
        if csv_path is None:
            print(f"[q6] MISSING eval csv: {run_dir}", file=sys.stderr)
            rows.append([scene, mode] + ["NA"] * 8)
            continue
        with open(csv_path, newline="") as f:
            lines = [r for r in csv.DictReader(f) if r]
        # prefer the COMPLETE row (partial=0); fall back to last partial
        complete = [r for r in lines if r.get("partial", "").strip() == "0"]
        row = (complete or [lines[-1]])[-1]
        num_trajs = max(1, int(row["num_trajs"]))
        out = [scene, mode]
        for metric in ("ADE_min", "FDE_min"):
            for t in (1, 2, 3, 4):
                key = f"{metric}_{t}s"
                val = row.get(key)
                out.append(f"{float(val)/num_trajs:.4f}" if val else "NA")
        rows.append(out)
with open(os.path.join(os.environ["RESEARCH_ROOT"], "q6", f"q6_video_mode_ade{outsuf}.csv"), "w", newline="") as f:
    csv.writer(f).writerows(rows)
print("[q6] wrote q6_video_mode_ade.csv")
PYEOF
fi
log_to "$LOG" INFO "Q6 complete. reports in ${OUTDIR}"