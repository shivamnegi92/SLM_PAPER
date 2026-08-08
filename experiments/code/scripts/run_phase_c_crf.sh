#!/usr/bin/env bash
# Phase C: CRF slot-head ablation vs the softmax head, at depth 3, on the
# slot-bearing datasets. Softmax depth-3 results already exist; this adds CRF.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results
for ds in atis snips massive; do
  echo "==================== $ds depth3 CRF ===================="
  python scripts/train_pruned.py --dataset "$ds" --depth 3 --epochs 3 \
    --batch-size 16 --max-train 5000 --use-crf \
    --output "results/${ds}_pruned_depth3_crf.json" || echo "FAILED: $ds crf"
done
echo "PHASE C DONE"
