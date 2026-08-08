#!/usr/bin/env bash
# Gate B validation: fine-tune at {probe-recommended depth, full depth} for
# each remaining dataset. Bounded to 5000 train examples for CPU tractability.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results
for ds in snips massive clinc150 banking77; do
  for d in 3 12; do
    echo "==================== $ds depth $d ===================="
    python scripts/train_pruned.py --dataset "$ds" --depth "$d" --epochs 3 \
      --batch-size 16 --max-train 5000 \
      --output "results/${ds}_pruned_depth${d}.json" || echo "FAILED: $ds depth $d"
  done
done
echo "ALL PRUNED SWEEPS DONE"
