#!/bin/bash
# Full LOSO training + evaluation matrix for ETH/UCY (strict external TU-Clausthal paths)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Optional: activate venv if present (remote lab)
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/.venv/bin/activate"
fi

DATA_DIR="${DATA_DIR:-/home/efrei_stage/Desktop/tu-clausthal/datasets}"
FRAMES_ROOT="${FRAMES_ROOT:-/home/efrei_stage/Desktop/tu-clausthal/datasets}"
FEATURES_ROOT="${FEATURES_ROOT:-$REPO_DIR/features/resnet18}"
CFG="${CFG:-cfg/eth_ucy/cor_fm.yml}"
SKIP_FEATURES="${SKIP_FEATURES:-0}"

for arg in "$@"; do
  case "$arg" in
    --skip-features) SKIP_FEATURES=1 ;;
  esac
done

EPOCHS=${EPOCHS:-64}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}

SUBSETS=("eth" "hotel" "univ" "zara1" "zara2")
MODES=("off" "static" "full")

mkdir -p "$REPO_DIR/results_eth_ucy/loso_full_matrix"/{checkpoints,metrics,logs,viz}
mkdir -p "$REPO_DIR/logs_loso_full"

if [ "$SKIP_FEATURES" != "1" ]; then
  echo "=== Building ETH/UCY video features (external TU-Clausthal layout) ==="
  echo "FRAMES_ROOT=$FRAMES_ROOT"
  echo "FEATURES_ROOT=$FEATURES_ROOT"
  python -m video_encoder.scripts.build_ethucy_features \
    --frames-root /home/efrei_stage/Desktop/tu-clausthal/datasets \
    --out "$FEATURES_ROOT" \
    --scenes eth hotel univ zara1 zara2 \
    --stride 10
else
  echo "=== SKIP_FEATURES enabled: using cached features at $FEATURES_ROOT ==="
fi

# Verify feature cache path is repo-root based (or explicitly overridden) and exists for spawned jobs.
if [ ! -d "$FEATURES_ROOT" ]; then
  echo "[WARN] FEATURES_ROOT does not exist: $FEATURES_ROOT"
fi

for SUBSET in "${SUBSETS[@]}"; do
  for MODE in "${MODES[@]}"; do
    echo "=================================================="
    echo "[$(date)] Training: held-out=$SUBSET video_mode=$MODE"
    echo "=================================================="

    python fm_eth.py \
      --cfg "$CFG" \
      --subset "$SUBSET" \
      --video_mode "$MODE" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --data_dir "$DATA_DIR" \
      --video_features_root "$FEATURES_ROOT" \
      2>&1 | tee "$REPO_DIR/logs_loso_full/${SUBSET}_${MODE}_train.log"

    CKPT=$(find "$REPO_DIR/results_eth_ucy/cor_fm" -name "*_${SUBSET}_vm${MODE}*" -path "*/models/checkpoint_best.pt" | head -1)
    if [ -z "${CKPT:-}" ]; then
      CKPT=$(find "$REPO_DIR/results_eth_ucy/cor_fm" -name "*_${SUBSET}_vm${MODE}*" -path "*/models/checkpoint_last.pt" | head -1)
    fi

    if [ -z "${CKPT:-}" ]; then
      echo "[ERROR] No checkpoint for $SUBSET/$MODE"
      continue
    fi

    cp "$CKPT" "$REPO_DIR/results_eth_ucy/loso_full_matrix/checkpoints/${SUBSET}_${MODE}.pt" 2>/dev/null || true

    echo "=================================================="
    echo "[$(date)] Evaluating: held-out=$SUBSET video_mode=$MODE"
    echo "=================================================="

    OUT_BASE="$REPO_DIR/results_eth_ucy/loso_full_matrix/metrics/${SUBSET}_${MODE}"
    python tools/eval_eth_transfer.py \
      --cfg "$CFG" \
      --ckpt "$CKPT" \
      --data-dir "$DATA_DIR" \
      --norm-scene "$SUBSET" \
      --eval-scene "$SUBSET" \
      --split test \
      --video-mode "$MODE" \
      --batch-size "$EVAL_BATCH_SIZE" \
      --features-root "$FEATURES_ROOT" \
      --out "$OUT_BASE" \
      2>&1 | tee "$REPO_DIR/logs_loso_full/${SUBSET}_${MODE}_eval.log"

    echo "[DONE] $SUBSET/$MODE"
  done
done

echo "Aggregating..."
python3 << 'PY'
import csv
from pathlib import Path
resd = Path('results_eth_ucy/loso_full_matrix/metrics')
out = []
for f in sorted(resd.glob('*.csv')):
    if 'loso_5x5' in f.name: continue
    try:
        with open(f) as fh:
            r = list(csv.DictReader(fh))[0]
        name = f.stem
        parts = name.split('_')
        out.append({
            'held_out_scene': parts[0],
            'video_mode': parts[1],
            'split': 'test',
            'ade_min': r['ade_min'],
            'fde_min': r['fde_min'],
            'ade_avg': r.get('ade_avg',''),
            'fde_avg': r.get('fde_avg',''),
            'n_agents': r.get('n_agents',''),
            'status': 'ok',
            'checkpoint': f'results_eth_ucy/loso_full_matrix/checkpoints/{parts[0]}_{parts[1]}.pt'
        })
    except Exception as e:
        print(e)
fields = ['held_out_scene','video_mode','split','ade_min','fde_min','ade_avg','fde_avg','n_agents','status','checkpoint']
with open(resd.parent / 'loso_5x5_metrics.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(out)
PY

echo "Generating visualizations..."
python tools/plot_homography_verification.py 2>&1 | tail -20 || true

echo "COMPLETE"
