#!/usr/bin/env bash
# Wait for the in-flight Phase A to finish, then run Phase C (CRF) and Phase B
# (seeds) back-to-back. Lets the whole remaining experiment matrix complete
# autonomously.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[chain] waiting for Phase A to finish..."
while ! grep -q "PHASE A DONE" results/phase_a.log 2>/dev/null; do
  sleep 20
done
echo "[chain] Phase A done. Starting Phase C (CRF)."
bash scripts/run_phase_c_crf.sh
echo "[chain] Phase C done. Starting Phase B (seeds)."
bash scripts/run_phase_b_seeds.sh
echo "[chain] ALL PHASES DONE"
