#!/usr/bin/env bash
# Wait for the seed sweep (Phase B) to finish, then re-run the CRF ablation with
# the length-normalized CRF loss (fixes the intent-head starvation), then
# regenerate the final aggregated tables.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[crf-rerun] waiting for Phase B (seeds) to finish..."
while ! grep -q "ALL PHASES DONE" results/chain_cb.log 2>/dev/null; do
  sleep 20
done
echo "[crf-rerun] seeds done. Re-running CRF ablation (normalized loss)."
for ds in atis snips massive; do
  python scripts/train_pruned.py --dataset "$ds" --depth 3 --epochs 3 \
    --batch-size 16 --max-train 5000 --use-crf \
    --output "results/${ds}_pruned_depth3_crf.json" || echo "FAILED: $ds crf"
done
echo "[crf-rerun] regenerating tables."
python scripts/aggregate_results.py > ../RESULTS_FINAL_TABLES.md 2>&1
echo "[crf-rerun] ALL DONE"
