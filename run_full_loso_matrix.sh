#!/bin/bash
# Full LOSO training + evaluation matrix for ETH/UCY
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $REPO_DIR
source .venv/bin/activate

DATA_DIR=/home/efrei_stage/Desktop/tu-clausthal/datasets
if [ ! -d "$DATA_DIR" ]; then
    DATA_DIR=$REPO_DIR/data/eth_ucy
fi
FEATURES_ROOT=$REPO_DIR/features/resnet18
CFG=cfg/eth_ucy/cor_fm.yml

EPOCHS=${EPOCHS:-64}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}

SUBSETS=("eth" "hotel" "univ" "zara1" "zara2")
MODES=("off" "static" "full")

mkdir -p results_eth_ucy/loso_full_matrix/{checkpoints,metrics,logs,viz}
mkdir -p logs_loso_full

echo "=== Building video features ==="
if [ -d "$DATA_DIR/../data/eth_ucy" ] || [ -d "$REPO_DIR/data/eth_ucy" ]; then
    FRAMES_ROOT="$REPO_DIR/data/eth_ucy"
else
    FRAMES_ROOT="$DATA_DIR/../.."
fi
python -m video_encoder.scripts.build_ethucy_features \
  --frames-root $FRAMES_ROOT \
  --out $FEATURES_ROOT \
  --stride 10 2>&1 | tail -30 || true

for SUBSET in "${SUBSETS[@]}"; do
  for MODE in "${MODES[@]}"; do
    echo "=================================================="
    echo "[$(date)] Training: held-out=$SUBSET video_mode=$MODE"
    echo "=================================================="
    
    python fm_eth.py \
      --cfg $CFG \
      --subset $SUBSET \
      --video_mode $MODE \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --data_dir $DATA_DIR \
      --video_features_root $FEATURES_ROOT \
      2>&1 | tee logs_loso_full/${SUBSET}_${MODE}_train.log
    
    CKPT=$(find results_eth_ucy/cor_fm -name "*_${SUBSET}_vm${MODE}*" -path "*/models/checkpoint_best.pt" | head -1)
    if [ -z "$CKPT" ]; then
      CKPT=$(find results_eth_ucy/cor_fm -name "*_${SUBSET}_vm${MODE}*" -path "*/models/checkpoint_last.pt" | head -1)
    fi
    
    if [ -z "$CKPT" ]; then
      echo "[ERROR] No checkpoint for $SUBSET/$MODE"
      continue
    fi
    
    cp "$CKPT" results_eth_ucy/loso_full_matrix/checkpoints/${SUBSET}_${MODE}.pt 2>/dev/null || true
    
    echo "=================================================="
    echo "[$(date)] Evaluating: held-out=$SUBSET video_mode=$MODE"
    echo "=================================================="
    
    OUT_BASE=results_eth_ucy/loso_full_matrix/metrics/${SUBSET}_${MODE}
    python tools/eval_eth_transfer.py \
      --cfg $CFG \
      --ckpt $CKPT \
      --data-dir $DATA_DIR \
      --norm-scene $SUBSET \
      --eval-scene $SUBSET \
      --split test \
      --video-mode $MODE \
      --batch-size $EVAL_BATCH_SIZE \
      --features-root $FEATURES_ROOT \
      --out $OUT_BASE \
      2>&1 | tee logs_loso_full/${SUBSET}_${MODE}_eval.log
    
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
