#!/bin/bash
set -e
COMMON="--n 42 --seed 0 --stage-a-steps 8 --lr 0.05 --norm-budget 4.0 --max-control-drop 0.20 --bootstrap-iters 2000"

echo "### ARM 1: baseline (margin only)"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --out-name pareto_arm1_baseline

echo "### ARM 2: +KL trust region"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --lam-kl 1.0 --out-name pareto_arm2_kl

echo "### ARM 3: +two-stage"
python src/intervene_pareto.py $COMMON --two-stage --stage-b-steps 6 --out-name pareto_arm3_twostage

echo "### ARM 4: +subspace"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --subspace-rank 8 --out-name pareto_arm4_subspace

echo "### ARM 5: ALL THREE"
python src/intervene_pareto.py $COMMON --two-stage --stage-b-steps 6 --lam-kl 1.0 --subspace-rank 8 --out-name pareto_arm5_all

echo "ABLATION_COMPLETE"
