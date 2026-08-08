#!/usr/bin/env bash
# Phase A: (1) B2 encoder at matched 5000-example budget for a fair head-to-head,
# and (3) fill in depths 6 & 9 for the 4 datasets that only have {3,12}.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results

echo "######## (1) B2 encoder @ matched 5000 budget ########"
for ds in atis snips massive clinc150 banking77; do
  echo "==================== $ds encoder-5k ===================="
  python scripts/train_encoder.py --dataset "$ds" --epochs 3 --batch-size 32 \
    --lr 5e-5 --slot-loss-weight 2.0 --max-train 5000 \
    --output "results/${ds}_encoder_5k.json" || echo "FAILED: $ds encoder-5k"
done

echo "######## (3) Pareto fill-in: depths 6 & 9 for the 4 new datasets ########"
for ds in snips massive clinc150 banking77; do
  for d in 6 9; do
    echo "==================== $ds depth $d ===================="
    python scripts/train_pruned.py --dataset "$ds" --depth "$d" --epochs 3 \
      --batch-size 16 --max-train 5000 \
      --output "results/${ds}_pruned_depth${d}.json" || echo "FAILED: $ds depth $d"
  done
done
echo "PHASE A DONE"
