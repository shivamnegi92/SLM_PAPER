#!/usr/bin/env bash
# Depth-1 fine-tuning sweep at the same 5000-example budget as the headline
# depth-3 and depth-12 rows, on all five datasets. Motivated by the dense
# probe: layer 1 is competitive on every dataset; if depth-1 fine-tunes to
# near-depth-3 quality, the method actually selects the deployed model.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results
for ds in atis snips massive clinc150 banking77; do
  out="results/${ds}_pruned_depth1.json"
  if [[ -s "$out" ]]; then
    echo "SKIP $out (exists)"
    continue
  fi
  echo "==================== $ds depth1 headline budget ===================="
  python scripts/train_pruned.py --dataset "$ds" --depth 1 --epochs 3 \
    --batch-size 16 --max-train 5000 \
    --output "$out" || echo "FAILED: $ds depth1"
done
echo "DEPTH-1 HEADLINE SWEEP DONE"
