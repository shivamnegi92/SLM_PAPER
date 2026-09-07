#!/bin/bash
set -e
for B in 0.17 0.30 0.50; do
  echo "### nemotron rel=$B FULL"
  python src/intervene_pareto.py --model ../nemotron-mini-4b --layers 17 19 21 23 \
    --n 126 --seed 0 --stage-a-steps 16 --lr 0.05 --rel-budget $B \
    --max-control-drop 0.20 --bootstrap-iters 2000 --stage-b-steps 0 \
    --out-name bc_nemotron_full_b$B
  echo "### nemotron rel=$B TRACK8"
  python src/intervene_pareto.py --model ../nemotron-mini-4b --layers 17 19 21 23 \
    --n 126 --seed 0 --stage-a-steps 16 --lr 0.05 --rel-budget $B \
    --max-control-drop 0.20 --bootstrap-iters 2000 --stage-b-steps 0 \
    --subspace-rank 8 --out-name bc_nemotron_track8_b$B
done
echo "NEMOTRON_MATCHED_COMPLETE"
