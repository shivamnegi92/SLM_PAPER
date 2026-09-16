#!/usr/bin/env bash
# Depth-2 fine-tuning across all 5 datasets at 5000 budget. Fills the gap
# between depth-1 and depth-3 in Table 2 so we can defensibly claim depth 3
# (not depth 2) is the shallowest evaluated depth that preserves quality.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results
for ds in atis snips massive clinc150 banking77; do
  out="results/${ds}_pruned_depth2.json"
  if [[ -s "$out" ]]; then
    echo "SKIP $out (exists)"
    continue
  fi
  echo "==================== $ds depth2 headline budget ===================="
  python scripts/train_pruned.py --dataset "$ds" --depth 2 --epochs 3 \
    --batch-size 16 --max-train 5000 \
    --output "$out" || echo "FAILED: $ds depth2"
done
echo "DEPTH-2 HEADLINE SWEEP DONE"
