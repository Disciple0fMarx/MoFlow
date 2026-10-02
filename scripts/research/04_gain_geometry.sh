#!/usr/bin/env bash
# 04_gain_geometry.sh — Q5/Q7 evidence: where does video help, and is the
#                       non-linear/roundabout geometry the reason?
#
# Step 1 — per-window attribution. Runs the attribution tool with
# ``--per-window`` (conditions: baseline + zeroed) on every scene's `full`
# checkpoint, giving per-window ADE_min with the real video channel and with
# it zeroed out. Per-window gain = ADE_min(zeroed) - ADE_min(baseline):
#   > 0  → this window's prediction NEEDS the video (video helps)
#   ~ 0  → video is neutral for this window
#   < 0  → video slightly hurts this window (e.g. right of way is ambiguous)
#
# Step 2 — geometry corollary. Aggregates gains per scene and per
# (video_id/anchor_frame) so we can check the Q5 hypothesis "low-gain scenes
# ↔ non-linear motion / complex geometry" and the Q7 hypothesis that the
# roundabout-exit windows are the high-gain ones (exit choice = intent).
#
# Step 3 — publication renders. For each scene renders the top-K highest-
# gain and top-K lowest-gain test windows via visualize_trajectory_comparison.py
# (its --window-index selects the exact window the attribution ranked), plus
# a dedicated multimodal pass over DEATH_CIRCLE roundabout windows.
#
# Usage:
#   scripts/research/04_gain_geometry.sh
#   scripts/research/04_gain_geometry.sh --scene deathCircle
#   scripts/research/04_gain_geometry.sh --n-batches 5 --top-k 2
#   scripts/research/04_gain_geometry.sh --skip-render
#
# Flags: --scene <s>, --n-batches <n>/0, --batch-size <b>, --top-k <k>,
#        --no-render (skip figures; CSV summary only).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_common.sh
source "${DIR}/00_common.sh"

LABEL="q4-gain-geometry"
SCENE=""
N_BATCHES=""
BATCH_SIZE=64
TOP_K=3
RENDER=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --n-batches) N_BATCHES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --video-id) VIDEO_ID="$2"; shift 2 ;;
        --no-render) RENDER=0; shift ;;
        --dry-run) RESEARCH_DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 3 ;;
    esac
done
build_video_args
OUTSUF="$(vid_out_suffix)"

OUTDIR="${RESEARCH_ROOT}/q4"
mkdir -p "$OUTDIR"
LOG="${OUTDIR}/run.log"
: > "$LOG"
write_provenance "$OUTDIR" "scripts/research/04_gain_geometry.sh"
log_to "$LOG" INFO "Q4 starting. scenes=($(loso_scenes "$LABEL" "$SCENE")) n_batches=${N_BATCHES:-all} top_k=${TOP_K} render=${RENDER}"

CFG="cfg/sdd/cor_fm.yml"

# ---- Step 1: per-window attribution for every scene -----------------------
for scene in $(loso_scenes "$LABEL" "$SCENE"); do
    CKPT="${RESULTS_ROOT}/_SDD_ho${scene}_vmfull/models/checkpoint_best.pt"
    require_ckpt "$CKPT" "$LABEL"
    ensure_features "$scene"
    ARGS=(tools/attrib_video_conditioning.py
        --cfg "$CFG"
        --ckpt "$CKPT"
        --sdd-root "$SDD_ROOT"
        --video-features-root "$FEATURES_ROOT"
        --held-out-scene "$scene"
        --split test
        --conditions baseline zeroed
        --batch-size "$BATCH_SIZE"
        --per-window "$OUTDIR/${scene}_windows${OUTSUF}.csv"
        --out "$OUTDIR/${scene}_attrib${OUTSUF}")
    [[ -n "${N_BATCHES}" && "${N_BATCHES}" != "0" ]] && ARGS+=(--n-batches "$N_BATCHES")
    ARGS+=( "${video_args[@]}" )
    run_py "windows:${scene}" "$LOG" "${ARGS[@]}"
done

log_to "$LOG" INFO "aggregating per-window gains -> q4_window_gains.csv"
if [[ "${RESEARCH_DRY_RUN:-0}" != "1" ]]; then
    # Re-scan which scenes actually got a windows CSV (loop may have been dry).
    SCENE_LIST="$(loso_scenes "$LABEL" "$SCENE")"
    SCENE_LIST="$SCENE_LIST" OUTDIR="$OUTDIR" RESEARCH_ROOT="$RESEARCH_ROOT" \
        OUTSUF="$OUTSUF" \
        "$PYTHON_BIN" - <<PYEOF
import csv, os, sys
outdir = os.environ["OUTDIR"]
scenes = os.environ["SCENE_LIST"].split()
outsuf = os.environ.get("OUTSUF", "")
rows = []
for scene in scenes:
    p = os.path.join(outdir, f"{scene}_windows{outsuf}.csv")
    if not os.path.exists(p):
        print(
            f"[q4] MISSING windows csv: {p} — the attribution tool either "
            "failed its --per-window guarantee (check its '[cvxp] batches "
            "processed=/skipped=' and FATAL lines in run.log: the usual cause "
            "is a missing video feature cache for the filtered scene/video, "
            "making every batch a skipped sentinel) or the scene has no "
            "windows under this --video-id.", file=sys.stderr)
        continue
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            try:
                base = float(r["ade_min_baseline"])
                zero = float(r["ade_min_zeroed"])
            except (KeyError, ValueError):
                continue
            fde_b = float(r["fde_min_baseline"]) if r.get("fde_min_baseline") else float("nan")
            fde_z = float(r["fde_min_zeroed"]) if r.get("fde_min_zeroed") else float("nan")
            rows.append({
                "scene": r["scene"], "video_id": r["video_id"],
                "anchor_frame": r["anchor_frame"], "window_index": r.get("window_index", ""),
                "ade_baseline": base, "ade_zeroed": zero,
                "gain_ade": zero - base, "gain_fde": (fde_z - fde_b),
            })
with open(os.path.join(os.environ["RESEARCH_ROOT"], "q4", f"q4_window_gains{outsuf}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scene"])
    w.writeheader()
    w.writerows(rows)
print(f"[q4] aggregate: {len(rows)} windows from {len(scenes)} scene(s)")

# per-scene summary (Q5: rank scenes by mean gain)
agg = {}
for row in rows:
    a = agg.setdefault(row["scene"], {"n": 0, "gain_sum": 0.0, "ade_base": 0.0})
    a["n"] += 1; a["gain_sum"] += row["gain_ade"]; a["ade_base"] += row["ade_baseline"]
with open(os.path.join(os.environ["RESEARCH_ROOT"], "q4", f"q4_scene_gain_summary{outsuf}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "n_windows", "mean_ade_baseline", "mean_gain_ade"])
    for s in sorted(agg, key=lambda s: agg[s]["gain_sum"] / agg[s]["n"], reverse=True):
        a = agg[s]
        w.writerow([s, a["n"], f"{a['ade_base']/a['n']:.4f}", f"{a['gain_sum']/a['n']:+.4f}"])
print("[q4] wrote q4_scene_gain_summary.csv (sorted by mean per-window video gain)")
PYEOF
fi

# ---- Step 3: publication renders (highest & lowest-gain windows + roundabout)
if ((RENDER)); then
    log_to "$LOG" INFO "rendering top/bottom per-scene windows"
    for scene in $(loso_scenes "$LABEL" "$SCENE"); do
        WCSV="${OUTDIR}/${scene}_windows${OUTSUF}.csv"
        [[ -f "$WCSV" ]] || { log_to "$LOG" WARN "[${scene}] no windows CSV — skipping renders"; continue; }

        # highest-gain window indexes (zeroed worse => video most needed)
        TOP_IDX="$(TOP_K="$TOP_K" "$PYTHON_BIN" - "$WCSV" <<'PYEOF'
import csv, sys
path = sys.argv[1]
rows = []
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        try:
            rows.append((float(r["ade_min_zeroed"]) - float(r["ade_min_baseline"]), int(r["window_index"])))
        except (KeyError, ValueError):
            continue
rows.sort(reverse=True)
k = max(1, int(__import__("os").environ.get("TOP_K", 3)))
print(" ".join(str(i) for _, i in rows[:k]))
PYEOF
)"
        # lowest-gain (including negative) window indexes
        BOT_IDX="$(TOP_K="$TOP_K" "$PYTHON_BIN" - "$WCSV" <<'PYEOF'
import csv, sys
path = sys.argv[1]
rows = []
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        try:
            rows.append((float(r["ade_min_zeroed"]) - float(r["ade_min_baseline"]), int(r["window_index"])))
        except (KeyError, ValueError):
            continue
rows.sort()
k = max(1, int(__import__("os").environ.get("TOP_K", 3)))
print(" ".join(str(i) for _, i in rows[:k]))
PYEOF
)"
        for idx in ${TOP_IDX}; do
            run_py "render:${scene}/top-window-${idx}" "$OUTDIR/render.log" \
                tools/visualize_trajectory_comparison.py \
                --scene "$scene" --window-index "$idx" \
                --out-dir "$OUTDIR/figs${OUTSUF}" \
                --sdd-root "$SDD_ROOT" --video_features_root "$FEATURES_ROOT" \
                "${video_args[@]}"
        done
        for idx in ${BOT_IDX}; do
            run_py "render:${scene}/bottom-window-${idx}" "$OUTDIR/render.log" \
                tools/visualize_trajectory_comparison.py \
                --scene "$scene" --window-index "$idx" \
                --out-dir "$OUTDIR/figs${OUTSUF}" \
                --sdd-root "$SDD_ROOT" --video_features_root "$FEATURES_ROOT" \
                "${video_args[@]}"
        done
    done
    # Q7: multimodal roundabout-exit evidence on DEATH_CIRCLE
    if [[ "$(loso_scenes "$LABEL" "$SCENE")" == *deathCircle* ]]; then
        log_to "$LOG" INFO "Q7 multimodal DEATH_CIRCLE pass (every test window)"
        N=$(W="$OUTDIR/deathCircle_windows${OUTSUF}.csv" "$PYTHON_BIN" - <<'PYEOF'
import csv, os, sys
p = os.environ["W"]
with open(p) as f:
    rows = [r for r in csv.DictReader(f)]
print(len(rows))
PYEOF
)
        for ((i = 0; i < N && i < 8; i++)); do
            run_py "render:deathCircle/window-${i}" "$OUTDIR/render.log" \
                tools/visualize_trajectory_comparison.py \
                --scene deathCircle --window-index "$i" \
                --out-dir "$OUTDIR/figs${OUTSUF}" \
                --sdd-root "$SDD_ROOT" --video_features_root "$FEATURES_ROOT" \
                "${video_args[@]}"
        done
    fi
fi

log_to "$LOG" INFO "Q4 complete. CSVs + figures in ${OUTDIR}"