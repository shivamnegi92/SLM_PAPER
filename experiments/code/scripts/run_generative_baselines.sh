#!/usr/bin/env bash
# Generative baseline (B1) on the key datasets. Bounded train/eval to keep CPU
# autoregressive generation tractable in one session (full-scale = camera-ready).
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results
for ds in atis snips clinc150; do
  echo "==================== $ds (generative) ===================="
  python scripts/train_generative.py --dataset "$ds" --model-path models/gpt2 \
    --epochs 3 --batch-size 8 --lr 5e-5 --max-train 3000 --max-eval 300 \
    --output "results/${ds}_generative.json" || echo "FAILED: $ds"
done
echo "ALL GENERATIVE DONE"
