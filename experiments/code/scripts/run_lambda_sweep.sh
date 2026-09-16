#!/usr/bin/env bash
# lambda (slot-loss weight) sweep on ATIS depth-3, softmax slot head.
# Paper hardcodes lambda=2.0; this sweep supports the stretch ablation in
# camera-ready. Reads a single dataset (ATIS) because it is the fastest
# slot-bearing benchmark; extend with snips/massive if time allows.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results
for lam in 1.0 1.5 2.0 2.5 3.0; do
  tag="${lam/./p}"
  out="results/atis_pruned_depth3_lambda${tag}.json"
  if [[ -s "$out" ]]; then
    echo "SKIP $out (exists)"
    continue
  fi
  echo "==================== atis depth3 lambda=${lam} ===================="
  python scripts/train_pruned.py --dataset atis --depth 3 --epochs 3 \
    --batch-size 16 --max-train 5000 --slot-loss-weight "${lam}" \
    --output "$out" || echo "FAILED: lambda=${lam}"
done
echo "LAMBDA SWEEP DONE"
