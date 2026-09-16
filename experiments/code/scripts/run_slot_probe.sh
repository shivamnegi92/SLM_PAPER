#!/usr/bin/env bash
# Frozen token-level slot probe on the three slot-bearing datasets, layers 1-12.
# Answers whether slot-level structure becomes linearly accessible at the same
# depth as intent, or requires deeper representations. Motivated by the depth-1
# fine-tune result where intent survives but slot F1 collapses.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results

DEPTHS="1,2,3,4,5,6,7,8,9,10,11,12"
MAXTRAIN=4000
EPOCHS=200
for ds in atis snips massive; do
  out="results/${ds}_slot_probe_dense.json"
  if [[ -s "$out" ]]; then
    echo "SKIP $out (exists)"
    continue
  fi
  echo "==================== $ds slot probe sweep ===================="
  python scripts/slot_probe.py --dataset "$ds" --depths "$DEPTHS" \
    --max-train "$MAXTRAIN" --probe-epochs "$EPOCHS" \
    --output "$out" || echo "FAILED: $ds slot probe"
done
echo "SLOT PROBE SWEEP DONE"
