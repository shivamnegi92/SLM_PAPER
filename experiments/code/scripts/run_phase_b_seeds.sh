#!/usr/bin/env bash
# Phase B: multi-seed variance on the headline result (ours depth3 vs depth12,
# all 5 datasets). Seed 42 already exists from earlier runs; add 2 more seeds.
# Report mean +/- std over the 3 seeds in the write-up.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results
for seed in 1 2; do
  for ds in atis snips massive clinc150 banking77; do
    for d in 3 12; do
      echo "============ $ds depth $d seed $seed ============"
      python scripts/train_pruned.py --dataset "$ds" --depth "$d" --epochs 3 \
        --batch-size 16 --max-train 5000 --seed "$seed" \
        --output "results/${ds}_pruned_depth${d}_seed${seed}.json" \
        || echo "FAILED: $ds depth $d seed $seed"
    done
  done
done
echo "PHASE B DONE"
