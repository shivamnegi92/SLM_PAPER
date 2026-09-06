#!/bin/bash
set -e
BASE="--n 42 --stage-a-steps 8 --lr 0.05 --norm-budget 4.0 --max-control-drop 0.20 --bootstrap-iters 2000 --stage-b-steps 0"
for S in 1 2; do
  echo "### baseline seed=$S"
  python src/intervene_pareto.py $BASE --seed $S --out-name pareto_arm1_baseline_s$S
  echo "### kl10 seed=$S"
  python src/intervene_pareto.py $BASE --seed $S --lam-kl 10.0 --out-name pareto_w2c_kl10_s$S
done
echo "SEEDS_COMPLETE"
