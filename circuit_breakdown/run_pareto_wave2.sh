#!/bin/bash
set -e
COMMON="--n 42 --seed 0 --stage-a-steps 8 --lr 0.05 --norm-budget 4.0 --max-control-drop 0.20 --bootstrap-iters 2000"

echo "### W2-A: two-stage, STRONG norm shrink (w_norm=0.05)"
python src/intervene_pareto.py $COMMON --two-stage --stage-b-steps 6 --w-norm 0.05 --w-hinge 20 --out-name pareto_w2a_shrink05

echo "### W2-B: two-stage, VERY strong norm shrink (w_norm=0.2)"
python src/intervene_pareto.py $COMMON --two-stage --stage-b-steps 6 --w-norm 0.2 --w-hinge 20 --out-name pareto_w2b_shrink20

echo "### W2-C: KL strong (lam=10)"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --lam-kl 10.0 --out-name pareto_w2c_kl10

echo "### W2-D: KL very strong (lam=50)"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --lam-kl 50.0 --out-name pareto_w2d_kl50

echo "### W2-E: subspace rank 32"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --subspace-rank 32 --out-name pareto_w2e_sub32

echo "### W2-F: subspace rank 64, wider budget"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --subspace-rank 64 --norm-budget 8.0 --out-name pareto_w2f_sub64

echo "WAVE2_COMPLETE"
