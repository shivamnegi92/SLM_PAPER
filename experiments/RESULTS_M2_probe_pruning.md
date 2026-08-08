# M2 Results — Probe-Guided Depth Selection (C1)

Backbone: `gpt2` (124M, 12 layers), truncated to depth d in {3,6,9,12}, converted
to discriminative (mean-pooled intent head + per-token softmax slot head, same
architecture as `pruned_model.PrunedGenerativeClassifier`). ATIS, seed 42, 3
epochs, batch 16, lr 5e-5, slot-loss-weight 2.0. CPU: Apple Silicon arm64 (see
`CPU_SPEC.md`). Raw JSON: `results/atis_probe_sweep.json`,
`results/atis_pruned_depth{3,6,9,12}.json`.

## Step 1 — the probe sweep (cheap, before any real fine-tuning)
Frozen gpt2 backbone, single forward pass with `output_hidden_states=True`,
mean-pooled per depth, linear probe trained on intent labels only
(4000 train examples, 200 probe epochs, ~18 seconds total wall-clock for all 4
depths -- this cost is what makes C1 useful):

| Depth | Probe val accuracy |
|---:|---:|
| 3 | 0.9500 |
| 6 | 0.9538 |
| 9 | 0.9525 |
| 12 (full) | 0.9600 |

`pick_depth(epsilon=0.02)` -> **recommended depth = 3** (within 1.0pt of the
best/full-depth probe accuracy, at 1/4 the layers).

## Step 2 — does the recommendation hold up after REAL fine-tuning? (Gate B)
Full intent+slot fine-tune (not just a linear probe) at every candidate depth:

| Depth | Intent Acc | Slot F1 | P50 latency | Train time |
|---:|---:|---:|---:|---:|
| **3 (probe pick)** | **0.9795** | 0.9096 | **5.12ms** | 61s |
| 6 | 0.9778 | 0.9227 | 10.91ms | 108s |
| 9 | 0.9812 | 0.9218 | 15.79ms | 156s |
| 12 (full) | 0.9744 | 0.9188 | 19.54ms | 214s |

**Gate B: PASS.** The probe-recommended depth (3) achieves intent accuracy
*higher than* full depth (97.95% vs 97.44%) and slot F1 within 1pt of full
depth (90.96% vs 91.88%) -- statistically indistinguishable at n=586 test
examples, single seed. The extra 9 layers bought nothing on this task, and
the linear probe correctly predicted that *before* paying for 3-4x the
fine-tuning compute.

**Gate A: PASS (trivially).** Every pruned depth crushes the full generative
baseline (see `RESULTS_M1_generative.md`: 48.33% intent / 44.58% slot F1 /
359ms / 46.7% parse-fail) on every axis. Depth 3 alone is +49.6pt intent,
+46.4pt slot F1, ~70x faster, and 0% parse failures by construction.

## Bonus: how does pruned-gpt2-depth3 compare to the B2 encoder baseline?
| Model | Params (active) | Intent Acc | Slot F1 | P50 (ms) |
|---|---:|---:|---:|---:|
| DistilBERT encoder (B2, 6 layers, bidirectional) | 66M | 98.98% | **95.12%** | 6.99 |
| **Pruned gpt2 (ours, depth 3, causal)** | ~30M (est.) | 97.95% | 90.96% | **5.12** |

The pruned causal model is faster and nearly matches intent accuracy, but
trails on slot F1 by ~4pts -- the honest, expected cost of causal (left-only)
attention for token tagging vs a bidirectional encoder (flagged in
`pruned_model.py` docstring and PLAN.md's M4 ablations). This is a genuine
trade-off to report, not a weakness to hide: **"ours" is a recipe for turning
an existing generative decoder into a fast discriminative model without
needing a separate bidirectional encoder**, and the paper should frame the
comparison honestly (vs. generative: unambiguous win; vs. purpose-built
encoder: competitive on intent, a real slot-F1 gap to characterize/close in M4).

## Honest caveats
- **Single seed.** The depth-12 result being slightly *worse* than depth-3/6/9
  could be optimization noise (3 epochs, fixed LR) rather than a real "deeper
  is worse" effect. M4's seed-variance ablation (>=3 seeds) is required before
  this claims robustness in the paper -- currently a single-seed pilot.
  Directionally, though, "shallow is enough" is exactly C1's thesis regardless
  of whether depth 12 is *slightly worse* or merely *equal*.
- Only 1 dataset (ATIS) swept so far. PLAN.md calls for the full sweep across
  all 5 datasets before this becomes a headline table.
- Only 1 backbone (gpt2-124M). A second model size is a stretch goal per
  PLAN.md, not required for the core claim.

## Status
M2 pilot (Gates A & B) PASSES on ATIS, single seed. Next: repeat the sweep on
the remaining 4 datasets, then M4's multi-seed variance + CRF slot head
ablation (may close some of the observed slot-F1 gap vs the encoder baseline).
