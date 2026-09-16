#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

PYTHON="${PROJECT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

export PYTHONPATH="$PROJECT/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_ROOT="$PROJECT/results/local_big_wins_20260908"
CAPABILITY_OUT="$RUN_ROOT/capability_powered"
REPLAY_PARENT="$RUN_ROOT/replay_workspace/SLM_PAPER"
REPLAY_PROJECT="$REPLAY_PARENT/circuit_breakdown"
STATUS="$RUN_ROOT/status.log"
mkdir -p "$RUN_ROOT"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$STATUS"
}

require_path() {
  if [[ ! -e "$1" ]]; then
    log "MISSING $1"
    exit 1
  fi
}

log "Starting local big-win diagnostics"
log "Project: $PROJECT"
log "Python: $PYTHON"

require_path "$PROJECT/data/validated_manifest_v1.json"
require_path "$PROJECT/data_bench/hellaswag_val.jsonl"
require_path "$PROJECT/data_bench/arc_easy_test_200.json"
require_path "$PROJECT/data_bench/tinyshakespeare.txt"
require_path "$PROJECT/results/validated_v1/phi-3.5-mini/intermediate/calibration.json"
require_path "$PROJECT/results/validated_v1/phi-3.5-mini/transfer/calibration.json"
require_path "$PROJECT/results/postrun_audit_v1/reproducibility/manifest.json"
require_path "$PROJECT/../phi-3.5-mini"

log "Compiling local helpers and touched runners"
"$PYTHON" -m py_compile \
  local_big_wins/prepare_large_benchmarks.py \
  src/localize.py \
  src/run_capability_study.py \
  src/run_convergence_pilot.py \
  src/run_guard_rate_pilot.py \
  src/run_swap_format_pilot.py \
  src/run_baseline_context_pilot.py

log "Trying to prepare larger public benchmark caches"
if "$PYTHON" local_big_wins/prepare_large_benchmarks.py --project "$PROJECT" --count 1000 2>&1 | tee -a "$STATUS"; then
  HELLASWAG="$PROJECT/data_bench/hellaswag_val_1000.jsonl"
  ARC="$PROJECT/data_bench/arc_easy_test_1000.json"
  N_HS=1000
  log "Using larger benchmark caches"
else
  HELLASWAG="$PROJECT/data_bench/hellaswag_val.jsonl"
  ARC="$PROJECT/data_bench/arc_easy_test_200.json"
  N_HS=240
  log "Large cache prep failed; falling back to frozen local caches"
fi

log "G4: powered capability check on Phi"
"$PYTHON" src/run_capability_study.py \
  --manifest "$PROJECT/data/validated_manifest_v1.json" \
  --model "$PROJECT/../phi-3.5-mini" \
  --study "$PROJECT/results/validated_v1" \
  --outdir "$CAPABILITY_OUT" \
  --hellaswag "$HELLASWAG" \
  --arc "$ARC" \
  --text "$PROJECT/data_bench/tinyshakespeare.txt" \
  --n-hs "$N_HS" \
  --seeds 0 1 2 \
  --device auto 2>&1 | tee -a "$STATUS"

log "Preparing fresh local replay workspace for fixed-output G1/G2 pilots"
rm -rf "$REPLAY_PARENT"
mkdir -p "$REPLAY_PARENT"
rsync -a \
  --exclude 'results/*' \
  --exclude '.venv' \
  "$PROJECT/" "$REPLAY_PROJECT/"
mkdir -p "$REPLAY_PROJECT/results"
rsync -a "$PROJECT/results/validated_v1" "$REPLAY_PROJECT/results/"
rsync -a "$PROJECT/results/postrun_audit_v1" "$REPLAY_PROJECT/results/"
ln -s "$PROJECT/../phi-3.5-mini" "$REPLAY_PARENT/phi-3.5-mini"
ln -s "$PROJECT/../llama-3.2-3b" "$REPLAY_PARENT/llama-3.2-3b" 2>/dev/null || true
ln -s "$PROJECT/../nemotron-mini-4b" "$REPLAY_PARENT/nemotron-mini-4b" 2>/dev/null || true

log "G1: fresh local convergence replay"
"$PYTHON" "$REPLAY_PROJECT/src/run_convergence_pilot.py" --project "$REPLAY_PROJECT" --device auto 2>&1 | tee -a "$STATUS"

log "G1: fresh local guard/rate replay"
"$PYTHON" "$REPLAY_PROJECT/src/run_guard_rate_pilot.py" --project "$REPLAY_PROJECT" --device auto 2>&1 | tee -a "$STATUS"

log "G2: fresh local swap-format diagnostic replay"
"$PYTHON" "$REPLAY_PROJECT/src/run_swap_format_pilot.py" --project "$REPLAY_PROJECT" --device auto 2>&1 | tee -a "$STATUS"

log "G2: fresh local baseline-context diagnostic replay"
"$PYTHON" "$REPLAY_PROJECT/src/run_baseline_context_pilot.py" --project "$REPLAY_PROJECT" --device auto 2>&1 | tee -a "$STATUS"

log "Local big-win diagnostics complete"
log "Outputs: $RUN_ROOT"
