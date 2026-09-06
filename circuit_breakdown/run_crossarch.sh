#!/bin/bash
set -e
BASE="--n 126 --stage-a-steps 8 --lr 0.05 --norm-budget 4.0 --max-control-drop 0.20 --bootstrap-iters 4000 --stage-b-steps 0 --layers 17 19 21 23"
for M in phi-3.5-mini nemotron-mini-4b; do
  for S in 0 1; do
    echo "### $M s$S FULL"
    python src/intervene_pareto.py $BASE --model ../$M --seed $S --out-name xarch_${M}_full_s$S
    echo "### $M s$S TRACK8"
    python src/intervene_pareto.py $BASE --model ../$M --seed $S --subspace-rank 8 --out-name xarch_${M}_track8_s$S
    echo "### $M s$S COMP8"
    python src/intervene_pareto.py $BASE --model ../$M --seed $S --subspace-rank 8 --subspace-complement --out-name xarch_${M}_comp8_s$S
  done
done
echo "CROSSARCH_COMPLETE"
