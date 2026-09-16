#!/usr/bin/env bash
# Dense per-layer probe sweep (layers 1..L) to answer the reviewer question:
# "Why not layer 1 or 2?" Writes a *_probe_sweep_dense.json alongside the
# existing sparse sweep so the two are not mixed.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results

DEPTHS="1,2,3,4,5,6,7,8,9,10,11,12"
# Match the sparse-probe training budget (max_train=4000, probe_epochs=200)
# so dense per-layer results are comparable with the existing sparse baseline.
MAXTRAIN=4000
EPOCHS=200
for ds in atis snips massive clinc150 banking77; do
  out="results/${ds}_probe_sweep_dense.json"
  if [[ -s "$out" ]]; then
    echo "SKIP $out (exists)"
    continue
  fi
  echo "==================== $ds dense probe sweep ===================="
  python scripts/probe_sweep.py --dataset "$ds" --depths "$DEPTHS" \
    --max-train "$MAXTRAIN" --probe-epochs "$EPOCHS" \
    --output "$out" || echo "FAILED: $ds dense probe"
done
echo "DENSE PROBE SWEEP DONE"
