#!/usr/bin/env bash
# Run the encoder baseline (B2) on all 5 datasets, 3 epochs each.
# Uses cached models/datasets (offline). Writes one JSON per dataset to results/.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p results
for ds in atis snips massive banking77 clinc150; do
  echo "==================== $ds ===================="
  python scripts/train_encoder.py \
    --dataset "$ds" --epochs 3 --batch-size 32 --lr 5e-5 --slot-loss-weight 2.0 \
    --output "results/${ds}_encoder.json" || echo "FAILED: $ds"
done
echo "ALL DONE"
