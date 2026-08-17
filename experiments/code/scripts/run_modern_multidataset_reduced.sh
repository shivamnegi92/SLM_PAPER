#!/usr/bin/env bash
# Reduced-budget cross-dataset modern-backbone matrix (CPU).
#
# Scope: probe sweep + pruned d3 + pruned d12, for the 4 breadth datasets
# (snips, massive, clinc150, banking77) x 2 modern models (Nemotron, Phi).
#
# Budget is DELIBERATELY reduced vs the GPT-2 publication rows because these
# 4B-class models run on CPU: full 5000x3 budget would take months. These runs
# are labeled "cross-dataset preliminary" -- NOT apples-to-apples with the
# GPT-2 headline tables.
#
# Semantics:
#   * run-if-missing: skips any artifact that already exists (resumable)
#   * stop-and-check: exits nonzero on the FIRST hard failure so a human can
#     inspect before more CPU time is spent.
#
# Timings are extrapolated from the ATIS sanity runs; worst case (Phi d12) is
# ~3h per run at max_train=150.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results

NEMO='/Users/s0n0611/Documents/GitHub/SLM_PAPER/nemotron-mini-4b'
PHI='/Users/s0n0611/Documents/GitHub/SLM_PAPER/phi-3.5-mini'

# reduced budget knobs
PROBE_MAX_TRAIN=120
PROBE_EPOCHS=10
PRUNED_MAX_TRAIN=150
PRUNED_EPOCHS=1
PRUNED_BATCH=2

run_if_missing () {
  local out="$1"; shift
  if [ -f "$out" ]; then
    echo "[skip] exists: $out"
    return 0
  fi
  echo "[run ] $out :: $*"
  "$@"
  echo "[done] $out"
}

# args: <tag> <model_path>
run_model () {
  local tag="$1" mp="$2"
  for ds in snips massive clinc150 banking77; do
    echo "==================== $ds :: $tag :: probe ===================="
    run_if_missing "results/${ds}_probe_sweep_usmodern_${tag}.json" \
      python scripts/probe_sweep.py --dataset "$ds" --model-path "$mp" \
        --max-train "$PROBE_MAX_TRAIN" --probe-epochs "$PROBE_EPOCHS" \
        --output "results/${ds}_probe_sweep_usmodern_${tag}.json"

    for d in 3 12; do
      echo "==================== $ds :: $tag :: pruned d$d ===================="
      run_if_missing "results/${ds}_pruned_depth${d}_usmodern_${tag}.json" \
        python scripts/train_pruned.py --dataset "$ds" --depth "$d" --model-path "$mp" \
          --epochs "$PRUNED_EPOCHS" --batch-size "$PRUNED_BATCH" --max-train "$PRUNED_MAX_TRAIN" \
          --output "results/${ds}_pruned_depth${d}_usmodern_${tag}.json"
    done
  done
}

echo "######## Nemotron cross-dataset (reduced) ########"
run_model nemotron "$NEMO"

echo "######## Phi cross-dataset (reduced) ########"
run_model phi "$PHI"

echo "MODERN_MULTIDATASET_REDUCED_DONE"
