#!/usr/bin/env bash
# Matched 3000-example three-way comparison on ATIS:
#   generative gpt2 (autoregressive)     -- already at 3k
#   discriminative gpt2 depth 12 @ 3k    -- NEW
#   discriminative gpt2 depth  3 @ 3k    -- NEW
# Isolates the generative->discriminative effect from the training-budget
# effect (the reviewer's second-largest concern).
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results

BUDGET=3000
for depth in 1 3 12; do
  out="results/atis_pruned_depth${depth}_budget3k.json"
  if [[ -s "$out" ]]; then
    echo "SKIP $out (exists)"
    continue
  fi
  echo "==================== atis depth${depth} matched budget ${BUDGET} ===================="
  python scripts/train_pruned.py --dataset atis --depth "$depth" --epochs 3 \
    --batch-size 16 --max-train "$BUDGET" \
    --output "$out" || echo "FAILED: depth=${depth} budget=${BUDGET}"
done
echo "MATCHED 3K BUDGET SWEEP DONE"
