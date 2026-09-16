# Colab Big Wins

Use this folder as the clean Colab package for GPU-heavy follow-up diagnostics.

## Files

```text
notebooks/colab_big_wins/
  COLAB_BIG_WINS.ipynb
  README.md
```

## Google Drive Layout

Before opening the notebook in Colab, place the repo and local model folders in Drive like this:

```text
MyDrive/SLM_PAPER/
  circuit_breakdown/
    data/
    data_bench/
    protocols/
    results/
    src/
    notebooks/
      colab_big_wins/
        COLAB_BIG_WINS.ipynb
  phi-3.5-mini/
  llama-3.2-3b/
  nemotron-mini-4b/
```

The model folders must be siblings of `circuit_breakdown`, because the existing protocols and scripts expect that layout.

## Colab Run Order

Open this notebook:

```text
MyDrive/SLM_PAPER/circuit_breakdown/notebooks/colab_big_wins/COLAB_BIG_WINS.ipynb
```

Then run cells from top to bottom.

The notebook will create a clean work copy here:

```text
/content/slm_paper_colab/circuit_breakdown
```

It links model folders here:

```text
/content/slm_paper_colab/phi-3.5-mini
/content/slm_paper_colab/llama-3.2-3b
/content/slm_paper_colab/nemotron-mini-4b
```

Fresh Colab outputs are copied back to Drive here:

```text
MyDrive/SLM_PAPER/colab_outputs/20260908_colab_big_wins/
```

## Best Quick Wins

Run these first:

```text
1. Powered capability check on Phi
2. Optional powered capability check on all three models
3. CUDA replay of convergence + guard/rate pilots
4. Fast swap-format + baseline-context diagnostics
```

## What This Helps Prove

```text
G4: Does capability preservation still fail/pass with larger benchmark samples?
G1: Do optimizer/convergence diagnostics reproduce on CUDA?
G2: Are harder-task failures caused by answer format, prompt context, or real weak task competence?
```

Do not overwrite the frozen validated study with these outputs. Treat this folder as a separate diagnostic package until results are reviewed.
