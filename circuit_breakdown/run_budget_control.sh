#!/bin/bash
set -e
# H1 TEST: is subspace decision-insufficiency budget-dependent?
# Sweep relative budget on BOTH Llama and Phi, tracking-subspace vs full, matched steps.
for B in 0.17 0.30 0.50; do
  echo "### llama rel=$B FULL"
  python src/intervene_pareto.py --model ../llama-3.2-3b --layers 18 20 22 24 \
    --n 126 --seed 0 --stage-a-steps 16 --lr 0.05 --rel-budget $B \
    --max-control-drop 0.20 --bootstrap-iters 2000 --stage-b-steps 0 \
    --out-name bc_llama_full_b$B
  echo "### llama rel=$B TRACK8"
  python src/intervene_pareto.py --model ../llama-3.2-3b --layers 18 20 22 24 \
    --n 126 --seed 0 --stage-a-steps 16 --lr 0.05 --rel-budget $B \
    --max-control-drop 0.20 --bootstrap-iters 2000 --stage-b-steps 0 \
    --subspace-rank 8 --out-name bc_llama_track8_b$B
done
for B in 0.17 0.50; do
  echo "### phi rel=$B FULL"
  python src/intervene_pareto.py --model ../phi-3.5-mini --layers 17 19 21 23 \
    --n 126 --seed 0 --stage-a-steps 16 --lr 0.05 --rel-budget $B \
    --max-control-drop 0.20 --bootstrap-iters 2000 --stage-b-steps 0 \
    --out-name bc_phi_full_b$B
  echo "### phi rel=$B TRACK8"
  python src/intervene_pareto.py --model ../phi-3.5-mini --layers 17 19 21 23 \
    --n 126 --seed 0 --stage-a-steps 16 --lr 0.05 --rel-budget $B \
    --max-control-drop 0.20 --bootstrap-iters 2000 --stage-b-steps 0 \
    --subspace-rank 8 --out-name bc_phi_track8_b$B
done
echo "BUDGET_CONTROL_COMPLETE"
