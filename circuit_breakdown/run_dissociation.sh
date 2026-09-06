#!/bin/bash
set -e
BASE="--n 126 --stage-a-steps 8 --lr 0.05 --norm-budget 4.0 --max-control-drop 0.20 --bootstrap-iters 4000 --stage-b-steps 0"
for S in 0 1 2; do
  echo "### [$S] full-space unconstrained"
  python src/intervene_pareto.py $BASE --seed $S --out-name diss_full_s$S
  echo "### [$S] TRACKING subspace r8"
  python src/intervene_pareto.py $BASE --seed $S --subspace-rank 8 --out-name diss_track8_s$S
  echo "### [$S] COMPLEMENT r8"
  python src/intervene_pareto.py $BASE --seed $S --subspace-rank 8 --subspace-complement --out-name diss_comp8_s$S
  echo "### [$S] KL lam=10"
  python src/intervene_pareto.py $BASE --seed $S --lam-kl 10.0 --out-name diss_kl10_s$S
done
echo "DISSOCIATION_COMPLETE"
