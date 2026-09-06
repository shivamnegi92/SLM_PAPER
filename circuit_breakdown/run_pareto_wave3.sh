#!/bin/bash
set -e
COMMON="--n 42 --seed 0 --stage-a-steps 8 --lr 0.05 --norm-budget 4.0 --max-control-drop 0.20 --bootstrap-iters 2000"

echo "### W3-A: subspace r8, STEP-MATCHED"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --subspace-rank 8 --out-name pareto_w3a_sub8_matched

echo "### W3-B: subspace r26 (max), STEP-MATCHED"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --subspace-rank 26 --out-name pareto_w3b_sub26_matched

echo "### W3-C: KL sweep lam=3"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --lam-kl 3.0 --out-name pareto_w3c_kl3

echo "### W3-D: KL sweep lam=20"
python src/intervene_pareto.py $COMMON --stage-b-steps 0 --lam-kl 20.0 --out-name pareto_w3d_kl20

echo "### W3-E: BEST COMBO = two-stage + KL10"
python src/intervene_pareto.py $COMMON --two-stage --stage-b-steps 6 --lam-kl 10.0 --out-name pareto_w3e_twostage_kl10

echo "### W3-F: two-stage + KL10 + subspace8 step-matched"
python src/intervene_pareto.py $COMMON --two-stage --stage-b-steps 6 --lam-kl 10.0 --subspace-rank 8 --out-name pareto_w3f_all_matched

echo "WAVE3_COMPLETE"
