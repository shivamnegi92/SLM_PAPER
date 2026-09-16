# Local Big Wins

This folder is the local replacement for the Colab plan. It keeps the next high-value runs clean and separate from the frozen validated study.

## Files

```text
local_big_wins/
  README.md
  prepare_large_benchmarks.py
  run_local_big_wins.sh
```

## What This Runs

```text
1. G4 powered capability check on Phi
2. G1 local replay: convergence + guard/rate pilots
3. G2 local replay: swap-format + baseline-context diagnostics
```

## Required Local Layout

From the project root, these paths must exist:

```text
circuit_breakdown/
  data/validated_manifest_v1.json
  data_bench/hellaswag_val.jsonl
  data_bench/arc_easy_test_200.json
  data_bench/tinyshakespeare.txt
  results/validated_v1/
  results/postrun_audit_v1/reproducibility/manifest.json
  src/
../phi-3.5-mini/
../llama-3.2-3b/
../nemotron-mini-4b/
```

## Recommended Run

Run this from `circuit_breakdown`:

```bash
bash local_big_wins/run_local_big_wins.sh
```

The script writes new outputs under:

```text
results/local_big_wins_20260908/
```

It does not overwrite the frozen validated study. G4 writes a new capability
folder in the main repo. G1/G2 fixed-output pilots run in a fresh local replay
workspace here:

```text
results/local_big_wins_20260908/replay_workspace/SLM_PAPER/circuit_breakdown/
```

## If You Want Larger Benchmark Caches First

The run script attempts to create larger public benchmark caches if possible:

```text
data_bench/hellaswag_val_1000.jsonl
data_bench/arc_easy_test_1000.json
```

If the network or Hugging Face download fails, the script falls back to the existing local caches:

```text
data_bench/hellaswag_val.jsonl
data_bench/arc_easy_test_200.json
```

## Simple Meaning

```text
G4 asks: does our edit hurt normal model skill?
G1 asks: is rank8 weak because the optimizer/settings are weak?
G2 asks: did the harder task fail because of format/prompt confusion or true weak reasoning?
```

## What To Trust

Treat these as new diagnostic outputs. Do not mix them into the original frozen study until we review the results and update the tracker.
