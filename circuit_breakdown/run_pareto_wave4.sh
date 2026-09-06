#!/bin/bash
set -e
BASE="--n 42 --stage-a-steps 8 --lr 0.05 --norm-budget 4.0 --max-control-drop 0.20 --bootstrap-iters 2000 --stage-b-steps 0"
for S in 0 1 2; do
  echo "### COMPLEMENT r8 seed=$S"
  python src/intervene_pareto.py $BASE --seed $S --subspace-rank 8 --subspace-complement --out-name pareto_w4_comp8_s$S
done
echo "### TWO-STAGE unclamped (budget 12) seed=0"
python src/intervene_pareto.py --n 42 --seed 0 --stage-a-steps 8 --lr 0.05 --norm-budget 12.0 --max-control-drop 0.20 --bootstrap-iters 2000 --two-stage --stage-b-steps 8 --w-norm 0.05 --w-hinge 20 --out-name pareto_w4_twostage_unclamped
echo "### STAGE-A-ONLY at budget 12 (control for above) seed=0"
python src/intervene_pareto.py --n 42 --seed 0 --stage-a-steps 8 --lr 0.05 --norm-budget 12.0 --max-control-drop 0.20 --bootstrap-iters 2000 --stage-b-steps 0 --out-name pareto_w4_stageA_b12
echo "WAVE4_COMPLETE"
