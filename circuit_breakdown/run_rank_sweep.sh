#!/bin/bash
set -e
BASE="--n 126 --stage-a-steps 8 --lr 0.05 --norm-budget 4.0 --max-control-drop 0.20 --bootstrap-iters 4000 --stage-b-steps 0"
# r8 already done at 3 seeds. Fill in r1, r4, r16, r26 at 2 seeds.
# Rank-major order so each rank yields a complete tracking-vs-complement pair early.
for R in 1 4 16 26; do
  for S in 0 1; do
    echo "### rank=$R seed=$S TRACKING"
    python src/intervene_pareto.py $BASE --seed $S --subspace-rank $R --out-name diss_track${R}_s$S
    echo "### rank=$R seed=$S COMPLEMENT"
    python src/intervene_pareto.py $BASE --seed $S --subspace-rank $R --subspace-complement --out-name diss_comp${R}_s$S
  done
done
echo "RANK_SWEEP_COMPLETE"
