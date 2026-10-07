#!/usr/bin/env bash
# 03_transfer_matrix.sh — Q2 evidence: is the video gain scene-specific or
#                         transferable?
#
# Builds an 8×8 cross-scene "transferability matrix". Row = the scene held out
# during training (the checkpoint's identity), column = the scene whose TEST
# windows the checkpoint is evaluated on.
#
# The evaluator is the attribution tool restricted to the `baseline` arm
# (real z_video). The crucial methodological point: inputs are normalized with
# the TRAINING scene's LOSO train-split statistics (--norm-scene <row>), never
# with the target column's stats — otherwise a cross-scene ADE would measure
# normalization shift instead of transfer. This decoupling is exactly why the
# attribution tool gained --norm-scene in this branch.
#
# Outputs two CSVs:
#   q2_transfer_ade.csv   : transferability of ADE_min (row=held-out train scene)
#   q2_transfer_gain.csv  : video-vs-novideo gain per (train scene, test scene);
#                           diag = LOSO gain, off-diag = transferable gain.
#
# Usage:
#   scripts/research/03_transfer_matrix.sh                # full 8×8
#   scripts/research/03_transfer_matrix.sh --scene coupa  # only row/col coupa
#   scripts/research/03_transfer_matrix.sh --train coupa
#   scripts/research/03_transfer_matrix.sh --n-batches 3  # quick check
#
# Flags: --scene <s> (evaluator scenes = columns), --train <s> (training
#        scenes = rows; default: all), --n-batches <n>/0, --batch-size <b>.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_common.sh
source "${DIR}/00_common.sh"

LABEL="q2-transfer"
SCENE=""
TRAIN_SCENE=""
N_BATCHES="3"    # quick by default: 3 batches per cell completes fast
BATCH_SIZE="${BATCH_SIZE:-64}"   # env-tunable memory knob (see 00_common.sh)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --train) TRAIN_SCENE="$2"; shift 2 ;;
        --n-batches) N_BATCHES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --video-id) VIDEO_ID="$2"; shift 2 ;;
        --dry-run) RESEARCH_DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 3 ;;
    esac
done
build_video_args
OUTSUF="$(vid_out_suffix)"

OUTDIR="${RESEARCH_ROOT}/q2"
mkdir -p "$OUTDIR"
LOG="${OUTDIR}/run.log"
: > "$LOG"
write_provenance "$OUTDIR" "scripts/research/03_transfer_matrix.sh"

COL_SCENES="$(loso_scenes "$LABEL" "$SCENE")"
ROW_SCENES="$(loso_scenes "$LABEL" "$TRAIN_SCENE")"
ensure_feature_list ${COL_SCENES}
log_to "$LOG" INFO "Q2 starting. rows(train)=(${ROW_SCENES}) cols(eval)=(${COL_SCENES}) n_batches=${N_BATCHES:-all}"

CFG="cfg/sdd/cor_fm.yml"

for train_scene in ${ROW_SCENES}; do
    CKPT_FULL="${RESULTS_ROOT}/_SDD_ho${train_scene}_vmfull/models/checkpoint_best.pt"
    CKPT_NOVID="${RESULTS_ROOT}/_SDD_ho${train_scene}_novid/models/checkpoint_best.pt"
    require_ckpt "$CKPT_FULL" "$LABEL"
    require_ckpt "$CKPT_NOVID" "$LABEL"
    for eval_scene in ${COL_SCENES}; do
        log_to "$LOG" INFO "[train=${train_scene} → eval=${eval_scene}]"
        for variant in full novid; do
            suffix="vmfull"
            [[ "$variant" == "novid" ]] && suffix="novid"
            CKPT="${RESULTS_ROOT}/_SDD_ho${train_scene}_${suffix}/models/checkpoint_best.pt"
            ARGS=(tools/attrib_video_conditioning.py
                --cfg "$CFG"
                --ckpt "$CKPT"
                --sdd-root "$SDD_ROOT"
                --video-features-root "$FEATURES_ROOT"
                --held-out-scene "$eval_scene"
                --norm-scene "$train_scene"      # decode with TRAINER's stats
                --split test
                --conditions baseline
                --batch-size "$BATCH_SIZE"
                --out "$OUTDIR/${train_scene}__eval_${eval_scene}__${variant}${OUTSUF}")
            [[ -n "${N_BATCHES}" && "${N_BATCHES}" != "0" ]] && ARGS+=(--n-batches "$N_BATCHES")
            ARGS+=( "${video_args[@]}" )
            run_py "transfer:${train_scene}->${eval_scene}:${variant}" "$LOG" "${ARGS[@]}"
        done
    done
done

log_to "$LOG" INFO "aggregating cells -> q2_transfer_ade.csv / q2_transfer_gain.csv"
if [[ "${RESEARCH_DRY_RUN:-0}" != "1" ]]; then
    COL_SCENES="$COL_SCENES" ROW_SCENES="$ROW_SCENES" \
        OUTDIR="$OUTDIR" RESEARCH_ROOT="$RESEARCH_ROOT" OUTSUF="$OUTSUF" \
        "$PYTHON_BIN" - <<PYEOF
import csv, os, sys
outdir = os.environ["OUTDIR"]
rows = os.environ["ROW_SCENES"].split()
cols = os.environ["COL_SCENES"].split()
outsuf = os.environ.get("OUTSUF", "")

def cell(train, eval, variant):
    p = os.path.join(outdir, f"{train}__eval_{eval}__{variant}{outsuf}.csv")
    if not os.path.exists(p):
        return None
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("condition") == "baseline":
                return float(r["ade_min"])
    return None

# The aggregate matrices must carry OUTSUF too: the per-cell CSVs are suffixed,
# but these two were not, so a --video-id run silently overwrote the
# geographic-LOSO 8x8 matrices the supervisor's Q2 answer depends on.
outsuf = os.environ.get("OUTSUF", "")

# ADE matrix (ade_min of 'full' checkpoint)
with open(os.path.join(os.environ["RESEARCH_ROOT"], "q2", f"q2_transfer_ade{outsuf}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["train_held_out"] + cols)
    for t in rows:
        w.writerow([t] + ["" if cell(t, c, "full") is None else f"{cell(t, c, 'full'):.4f}" for c in cols])

# GAIN matrix: ADE(novid) - ADE(full)  -> positive = video helps
with open(os.path.join(os.environ["RESEARCH_ROOT"], "q2", f"q2_transfer_gain{outsuf}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["train_held_out"] + cols)
    for t in rows:
        gains = []
        for c in cols:
            a = cell(t, c, "novid")
            b = cell(t, c, "full")
            gains.append("" if (a is None or b is None) else f"{a - b:+.4f}")
        w.writerow([t] + gains)
print(f"[q2] wrote q2_transfer_ade{outsuf}.csv and q2_transfer_gain{outsuf}.csv")
PYEOF
fi
build_supervisor_layout 2>/dev/null || true

log_to "$LOG" INFO "Q2 complete. reports in ${OUTDIR}"