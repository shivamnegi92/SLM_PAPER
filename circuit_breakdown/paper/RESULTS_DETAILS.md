# Detailed Validated Results

Generated from the completed frozen study snapshot. Rates include 95% intervals; accuracy differences are percentage points (pp), not relative percent changes.

The CPU-only post-run audit recomputes stored predictions, counts and statistics. It does not rerun model forward passes, reconstruct missing logits, or prove historical weight provenance. See [the audit and reproduction guide](../docs/REPRODUCIBILITY.md).

The two eligible tasks each contain 150 semantic test pairs shared across models and methods, in three disjoint 50-pair seed blocks. Repeated method/model evaluations are not extra independent semantic examples. Full-space edits are optimized separately for each target-informed example; this is not a reusable editor or demonstrated error repair.

![Target override and same-sign damage](../results/final_validated_evidence_v1/validated_control.png)

## Development Competence

| Model | Task | Seed-0 dev pairs | Clean accuracy | Counterfactual accuracy | Gate |
|---|---|---|---|---|---|
| llama-3.2-3b | intermediate | 24 | 95.8% [79.8, 99.3] | 100.0% [86.2, 100.0] | passed |
| llama-3.2-3b | transfer | 24 | 91.7% [74.2, 97.7] | 91.7% [74.2, 97.7] | passed |
| llama-3.2-3b | container_swap | 24 | 12.5% [4.3, 31.0] | 8.3% [2.3, 25.8] | failed; baseline only |
| phi-3.5-mini | intermediate | 24 | 100.0% [86.2, 100.0] | 100.0% [86.2, 100.0] | passed |
| phi-3.5-mini | transfer | 24 | 100.0% [86.2, 100.0] | 100.0% [86.2, 100.0] | passed |
| phi-3.5-mini | container_swap | 24 | 12.5% [4.3, 31.0] | 41.7% [24.5, 61.2] | failed; baseline only |
| nemotron-mini-4b | intermediate | 24 | 100.0% [86.2, 100.0] | 95.8% [79.8, 99.3] | passed |
| nemotron-mini-4b | transfer | 24 | 100.0% [86.2, 100.0] | 95.8% [79.8, 99.3] | passed |
| nemotron-mini-4b | container_swap | 24 | 12.5% [4.3, 31.0] | 20.8% [9.2, 40.5] | failed; baseline only |

The 80% gate is applied to both accuracies on seed-0 development data. Learning-rate selection uses four seed-0 development examples with equal candidate grids. The other development blocks are not additional fitted calibration replicates.

## Hard-Task Baseline Tests

| Model | Task | Seed | Pairs | Clean accuracy | Counterfactual accuracy |
|---|---|---|---|---|---|
| llama-3.2-3b | container_swap | 0 | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| llama-3.2-3b | container_swap | 1 | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| llama-3.2-3b | container_swap | 2 | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| phi-3.5-mini | container_swap | 0 | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| phi-3.5-mini | container_swap | 1 | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| phi-3.5-mini | container_swap | 2 | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| nemotron-mini-4b | container_swap | 0 | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| nemotron-mini-4b | container_swap | 1 | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| nemotron-mini-4b | container_swap | 2 | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |

All container-swap development gates failed. These test rows describe the failed generalization check and cannot support a causal-control conclusion. Baseline-only files store flags but not item IDs; the audit checks their counts and rates, not the original item ordering independently.

## Pooled Interventions

| Model | Task | Method | Pairs | Target override | Baseline-correct n | Same-sign damage | Negative disruption |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b | intermediate | full | 150 | 100.0% [97.5, 100.0] | 144 | 0.0% [0.0, 2.6] | 50.0% [41.9, 58.1] |
| llama-3.2-3b | intermediate | track8 | 150 | 10.0% [6.2, 15.8] | 144 | 2.1% [0.7, 5.9] | 62.5% [54.4, 70.0] |
| llama-3.2-3b | intermediate | comp8 | 150 | 0.0% [0.0, 2.5] | 144 | 3.5% [1.5, 7.9] | 4.9% [2.4, 9.7] |
| llama-3.2-3b | transfer | full | 150 | 100.0% [97.5, 100.0] | 110 | 0.0% [0.0, 3.4] | 84.5% [76.6, 90.1] |
| llama-3.2-3b | transfer | track8 | 150 | 25.3% [19.0, 32.8] | 110 | 5.5% [2.5, 11.4] | 82.7% [74.6, 88.7] |
| llama-3.2-3b | transfer | comp8 | 150 | 0.0% [0.0, 2.5] | 110 | 6.4% [3.1, 12.6] | 12.7% [7.7, 20.2] |
| phi-3.5-mini | intermediate | full | 150 | 100.0% [97.5, 100.0] | 150 | 0.0% [0.0, 2.5] | 13.3% [8.8, 19.7] |
| phi-3.5-mini | intermediate | track8 | 150 | 8.7% [5.1, 14.3] | 150 | 2.7% [1.0, 6.7] | 16.7% [11.6, 23.4] |
| phi-3.5-mini | intermediate | comp8 | 150 | 0.0% [0.0, 2.5] | 150 | 2.0% [0.7, 5.7] | 0.7% [0.1, 3.7] |
| phi-3.5-mini | transfer | full | 150 | 100.0% [97.5, 100.0] | 150 | 0.0% [0.0, 2.5] | 12.7% [8.3, 18.9] |
| phi-3.5-mini | transfer | track8 | 150 | 20.0% [14.4, 27.1] | 150 | 0.7% [0.1, 3.7] | 5.3% [2.7, 10.2] |
| phi-3.5-mini | transfer | comp8 | 150 | 0.0% [0.0, 2.5] | 150 | 0.0% [0.0, 2.5] | 0.0% [0.0, 2.5] |
| nemotron-mini-4b | intermediate | full | 150 | 100.0% [97.5, 100.0] | 137 | 0.0% [0.0, 2.7] | 79.6% [72.0, 85.5] |
| nemotron-mini-4b | intermediate | track8 | 150 | 12.7% [8.3, 18.9] | 137 | 22.6% [16.4, 30.3] | 80.3% [72.8, 86.1] |
| nemotron-mini-4b | intermediate | comp8 | 150 | 0.0% [0.0, 2.5] | 137 | 6.6% [3.5, 12.0] | 5.8% [3.0, 11.1] |
| nemotron-mini-4b | transfer | full | 150 | 100.0% [97.5, 100.0] | 140 | 0.0% [0.0, 2.7] | 63.6% [55.3, 71.1] |
| nemotron-mini-4b | transfer | track8 | 150 | 56.0% [48.0, 63.7] | 140 | 5.0% [2.4, 10.0] | 69.3% [61.2, 76.3] |
| nemotron-mini-4b | transfer | comp8 | 150 | 0.0% [0.0, 2.5] | 140 | 7.1% [3.9, 12.6] | 1.4% [0.4, 5.1] |

Damage and disruption use only baseline-correct clean inputs. Target override uses all counterfactual inputs. A zero observed damage rate is not a guarantee for unrelated inputs.

## Per-Seed Interventions

| Model | Task | Method | Seed / data | Pairs | Target override | Baseline-correct n | Same-sign damage | Negative disruption |
|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b | intermediate | full | [0](../results/validated_v1/llama-3.2-3b/intermediate/diss_full_s0.json) | 50 | 100.0% [92.9, 100.0] | 48 | 0.0% [0.0, 7.4] | 47.9% [34.5, 61.7] |
| llama-3.2-3b | intermediate | full | [1](../results/validated_v1/llama-3.2-3b/intermediate/diss_full_s1.json) | 50 | 100.0% [92.9, 100.0] | 50 | 0.0% [0.0, 7.1] | 40.0% [27.6, 53.8] |
| llama-3.2-3b | intermediate | full | [2](../results/validated_v1/llama-3.2-3b/intermediate/diss_full_s2.json) | 50 | 100.0% [92.9, 100.0] | 46 | 0.0% [0.0, 7.7] | 63.0% [48.6, 75.5] |
| llama-3.2-3b | intermediate | track8 | [0](../results/validated_v1/llama-3.2-3b/intermediate/diss_track8_s0.json) | 50 | 12.0% [5.6, 23.8] | 48 | 2.1% [0.4, 10.9] | 62.5% [48.4, 74.8] |
| llama-3.2-3b | intermediate | track8 | [1](../results/validated_v1/llama-3.2-3b/intermediate/diss_track8_s1.json) | 50 | 10.0% [4.3, 21.4] | 50 | 4.0% [1.1, 13.5] | 62.0% [48.2, 74.1] |
| llama-3.2-3b | intermediate | track8 | [2](../results/validated_v1/llama-3.2-3b/intermediate/diss_track8_s2.json) | 50 | 8.0% [3.2, 18.8] | 46 | 0.0% [0.0, 7.7] | 63.0% [48.6, 75.5] |
| llama-3.2-3b | intermediate | comp8 | [0](../results/validated_v1/llama-3.2-3b/intermediate/diss_comp8_s0.json) | 50 | 0.0% [0.0, 7.1] | 48 | 6.2% [2.1, 16.8] | 8.3% [3.3, 19.6] |
| llama-3.2-3b | intermediate | comp8 | [1](../results/validated_v1/llama-3.2-3b/intermediate/diss_comp8_s1.json) | 50 | 0.0% [0.0, 7.1] | 50 | 0.0% [0.0, 7.1] | 4.0% [1.1, 13.5] |
| llama-3.2-3b | intermediate | comp8 | [2](../results/validated_v1/llama-3.2-3b/intermediate/diss_comp8_s2.json) | 50 | 0.0% [0.0, 7.1] | 46 | 4.3% [1.2, 14.5] | 2.2% [0.4, 11.3] |
| llama-3.2-3b | transfer | full | [0](../results/validated_v1/llama-3.2-3b/transfer/diss_full_s0.json) | 50 | 100.0% [92.9, 100.0] | 44 | 0.0% [0.0, 8.0] | 81.8% [68.0, 90.5] |
| llama-3.2-3b | transfer | full | [1](../results/validated_v1/llama-3.2-3b/transfer/diss_full_s1.json) | 50 | 100.0% [92.9, 100.0] | 41 | 0.0% [0.0, 8.6] | 82.9% [68.7, 91.5] |
| llama-3.2-3b | transfer | full | [2](../results/validated_v1/llama-3.2-3b/transfer/diss_full_s2.json) | 50 | 100.0% [92.9, 100.0] | 25 | 0.0% [0.0, 13.3] | 92.0% [75.0, 97.8] |
| llama-3.2-3b | transfer | track8 | [0](../results/validated_v1/llama-3.2-3b/transfer/diss_track8_s0.json) | 50 | 20.0% [11.2, 33.0] | 44 | 2.3% [0.4, 11.8] | 75.0% [60.6, 85.4] |
| llama-3.2-3b | transfer | track8 | [1](../results/validated_v1/llama-3.2-3b/transfer/diss_track8_s1.json) | 50 | 38.0% [25.9, 51.8] | 41 | 12.2% [5.3, 25.5] | 85.4% [71.6, 93.1] |
| llama-3.2-3b | transfer | track8 | [2](../results/validated_v1/llama-3.2-3b/transfer/diss_track8_s2.json) | 50 | 18.0% [9.8, 30.8] | 25 | 0.0% [0.0, 13.3] | 92.0% [75.0, 97.8] |
| llama-3.2-3b | transfer | comp8 | [0](../results/validated_v1/llama-3.2-3b/transfer/diss_comp8_s0.json) | 50 | 0.0% [0.0, 7.1] | 44 | 0.0% [0.0, 8.0] | 9.1% [3.6, 21.2] |
| llama-3.2-3b | transfer | comp8 | [1](../results/validated_v1/llama-3.2-3b/transfer/diss_comp8_s1.json) | 50 | 0.0% [0.0, 7.1] | 41 | 12.2% [5.3, 25.5] | 17.1% [8.5, 31.3] |
| llama-3.2-3b | transfer | comp8 | [2](../results/validated_v1/llama-3.2-3b/transfer/diss_comp8_s2.json) | 50 | 0.0% [0.0, 7.1] | 25 | 8.0% [2.2, 25.0] | 12.0% [4.2, 30.0] |
| phi-3.5-mini | intermediate | full | [0](../results/validated_v1/phi-3.5-mini/intermediate/diss_full_s0.json) | 50 | 100.0% [92.9, 100.0] | 50 | 0.0% [0.0, 7.1] | 12.0% [5.6, 23.8] |
| phi-3.5-mini | intermediate | full | [1](../results/validated_v1/phi-3.5-mini/intermediate/diss_full_s1.json) | 50 | 100.0% [92.9, 100.0] | 50 | 0.0% [0.0, 7.1] | 14.0% [7.0, 26.2] |
| phi-3.5-mini | intermediate | full | [2](../results/validated_v1/phi-3.5-mini/intermediate/diss_full_s2.json) | 50 | 100.0% [92.9, 100.0] | 50 | 0.0% [0.0, 7.1] | 14.0% [7.0, 26.2] |
| phi-3.5-mini | intermediate | track8 | [0](../results/validated_v1/phi-3.5-mini/intermediate/diss_track8_s0.json) | 50 | 10.0% [4.3, 21.4] | 50 | 2.0% [0.4, 10.5] | 18.0% [9.8, 30.8] |
| phi-3.5-mini | intermediate | track8 | [1](../results/validated_v1/phi-3.5-mini/intermediate/diss_track8_s1.json) | 50 | 6.0% [2.1, 16.2] | 50 | 2.0% [0.4, 10.5] | 20.0% [11.2, 33.0] |
| phi-3.5-mini | intermediate | track8 | [2](../results/validated_v1/phi-3.5-mini/intermediate/diss_track8_s2.json) | 50 | 10.0% [4.3, 21.4] | 50 | 4.0% [1.1, 13.5] | 12.0% [5.6, 23.8] |
| phi-3.5-mini | intermediate | comp8 | [0](../results/validated_v1/phi-3.5-mini/intermediate/diss_comp8_s0.json) | 50 | 0.0% [0.0, 7.1] | 50 | 0.0% [0.0, 7.1] | 2.0% [0.4, 10.5] |
| phi-3.5-mini | intermediate | comp8 | [1](../results/validated_v1/phi-3.5-mini/intermediate/diss_comp8_s1.json) | 50 | 0.0% [0.0, 7.1] | 50 | 4.0% [1.1, 13.5] | 0.0% [0.0, 7.1] |
| phi-3.5-mini | intermediate | comp8 | [2](../results/validated_v1/phi-3.5-mini/intermediate/diss_comp8_s2.json) | 50 | 0.0% [0.0, 7.1] | 50 | 2.0% [0.4, 10.5] | 0.0% [0.0, 7.1] |
| phi-3.5-mini | transfer | full | [0](../results/validated_v1/phi-3.5-mini/transfer/diss_full_s0.json) | 50 | 100.0% [92.9, 100.0] | 50 | 0.0% [0.0, 7.1] | 18.0% [9.8, 30.8] |
| phi-3.5-mini | transfer | full | [1](../results/validated_v1/phi-3.5-mini/transfer/diss_full_s1.json) | 50 | 100.0% [92.9, 100.0] | 50 | 0.0% [0.0, 7.1] | 8.0% [3.2, 18.8] |
| phi-3.5-mini | transfer | full | [2](../results/validated_v1/phi-3.5-mini/transfer/diss_full_s2.json) | 50 | 100.0% [92.9, 100.0] | 50 | 0.0% [0.0, 7.1] | 12.0% [5.6, 23.8] |
| phi-3.5-mini | transfer | track8 | [0](../results/validated_v1/phi-3.5-mini/transfer/diss_track8_s0.json) | 50 | 22.0% [12.8, 35.2] | 50 | 0.0% [0.0, 7.1] | 6.0% [2.1, 16.2] |
| phi-3.5-mini | transfer | track8 | [1](../results/validated_v1/phi-3.5-mini/transfer/diss_track8_s1.json) | 50 | 18.0% [9.8, 30.8] | 50 | 0.0% [0.0, 7.1] | 2.0% [0.4, 10.5] |
| phi-3.5-mini | transfer | track8 | [2](../results/validated_v1/phi-3.5-mini/transfer/diss_track8_s2.json) | 50 | 20.0% [11.2, 33.0] | 50 | 2.0% [0.4, 10.5] | 8.0% [3.2, 18.8] |
| phi-3.5-mini | transfer | comp8 | [0](../results/validated_v1/phi-3.5-mini/transfer/diss_comp8_s0.json) | 50 | 0.0% [0.0, 7.1] | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| phi-3.5-mini | transfer | comp8 | [1](../results/validated_v1/phi-3.5-mini/transfer/diss_comp8_s1.json) | 50 | 0.0% [0.0, 7.1] | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| phi-3.5-mini | transfer | comp8 | [2](../results/validated_v1/phi-3.5-mini/transfer/diss_comp8_s2.json) | 50 | 0.0% [0.0, 7.1] | 50 | 0.0% [0.0, 7.1] | 0.0% [0.0, 7.1] |
| nemotron-mini-4b | intermediate | full | [0](../results/validated_v1/nemotron-mini-4b/intermediate/diss_full_s0.json) | 50 | 100.0% [92.9, 100.0] | 49 | 0.0% [0.0, 7.3] | 71.4% [57.6, 82.2] |
| nemotron-mini-4b | intermediate | full | [1](../results/validated_v1/nemotron-mini-4b/intermediate/diss_full_s1.json) | 50 | 100.0% [92.9, 100.0] | 46 | 0.0% [0.0, 7.7] | 82.6% [69.3, 90.9] |
| nemotron-mini-4b | intermediate | full | [2](../results/validated_v1/nemotron-mini-4b/intermediate/diss_full_s2.json) | 50 | 100.0% [92.9, 100.0] | 42 | 0.0% [0.0, 8.4] | 85.7% [72.2, 93.3] |
| nemotron-mini-4b | intermediate | track8 | [0](../results/validated_v1/nemotron-mini-4b/intermediate/diss_track8_s0.json) | 50 | 14.0% [7.0, 26.2] | 49 | 8.2% [3.2, 19.2] | 73.5% [59.7, 83.8] |
| nemotron-mini-4b | intermediate | track8 | [1](../results/validated_v1/nemotron-mini-4b/intermediate/diss_track8_s1.json) | 50 | 14.0% [7.0, 26.2] | 46 | 28.3% [17.3, 42.5] | 76.1% [62.1, 86.1] |
| nemotron-mini-4b | intermediate | track8 | [2](../results/validated_v1/nemotron-mini-4b/intermediate/diss_track8_s2.json) | 50 | 10.0% [4.3, 21.4] | 42 | 33.3% [21.0, 48.4] | 92.9% [81.0, 97.5] |
| nemotron-mini-4b | intermediate | comp8 | [0](../results/validated_v1/nemotron-mini-4b/intermediate/diss_comp8_s0.json) | 50 | 0.0% [0.0, 7.1] | 49 | 6.1% [2.1, 16.5] | 0.0% [0.0, 7.3] |
| nemotron-mini-4b | intermediate | comp8 | [1](../results/validated_v1/nemotron-mini-4b/intermediate/diss_comp8_s1.json) | 50 | 0.0% [0.0, 7.1] | 46 | 4.3% [1.2, 14.5] | 10.9% [4.7, 23.0] |
| nemotron-mini-4b | intermediate | comp8 | [2](../results/validated_v1/nemotron-mini-4b/intermediate/diss_comp8_s2.json) | 50 | 0.0% [0.0, 7.1] | 42 | 9.5% [3.8, 22.1] | 7.1% [2.5, 19.0] |
| nemotron-mini-4b | transfer | full | [0](../results/validated_v1/nemotron-mini-4b/transfer/diss_full_s0.json) | 50 | 100.0% [92.9, 100.0] | 49 | 0.0% [0.0, 7.3] | 67.3% [53.4, 78.8] |
| nemotron-mini-4b | transfer | full | [1](../results/validated_v1/nemotron-mini-4b/transfer/diss_full_s1.json) | 50 | 100.0% [92.9, 100.0] | 48 | 0.0% [0.0, 7.4] | 60.4% [46.3, 73.0] |
| nemotron-mini-4b | transfer | full | [2](../results/validated_v1/nemotron-mini-4b/transfer/diss_full_s2.json) | 50 | 100.0% [92.9, 100.0] | 43 | 0.0% [0.0, 8.2] | 62.8% [47.9, 75.6] |
| nemotron-mini-4b | transfer | track8 | [0](../results/validated_v1/nemotron-mini-4b/transfer/diss_track8_s0.json) | 50 | 56.0% [42.3, 68.8] | 49 | 6.1% [2.1, 16.5] | 69.4% [55.5, 80.5] |
| nemotron-mini-4b | transfer | track8 | [1](../results/validated_v1/nemotron-mini-4b/transfer/diss_track8_s1.json) | 50 | 54.0% [40.4, 67.0] | 48 | 0.0% [0.0, 7.4] | 72.9% [59.0, 83.4] |
| nemotron-mini-4b | transfer | track8 | [2](../results/validated_v1/nemotron-mini-4b/transfer/diss_track8_s2.json) | 50 | 58.0% [44.2, 70.6] | 43 | 9.3% [3.7, 21.6] | 65.1% [50.2, 77.6] |
| nemotron-mini-4b | transfer | comp8 | [0](../results/validated_v1/nemotron-mini-4b/transfer/diss_comp8_s0.json) | 50 | 0.0% [0.0, 7.1] | 49 | 6.1% [2.1, 16.5] | 2.0% [0.4, 10.7] |
| nemotron-mini-4b | transfer | comp8 | [1](../results/validated_v1/nemotron-mini-4b/transfer/diss_comp8_s1.json) | 50 | 0.0% [0.0, 7.1] | 48 | 6.2% [2.1, 16.8] | 2.1% [0.4, 10.9] |
| nemotron-mini-4b | transfer | comp8 | [2](../results/validated_v1/nemotron-mini-4b/transfer/diss_comp8_s2.json) | 50 | 0.0% [0.0, 7.1] | 43 | 9.3% [3.7, 21.6] | 0.0% [0.0, 8.2] |

## Additional Intervention Metrics

| Model | Task | Method | New-target eligible n | New target rate | Third-token rate | Mean two-way probability change |
|---|---|---|---|---|---|---|
| llama-3.2-3b | intermediate | full | 150 | 100.0% [97.5, 100.0] | 0.0% [0.0, 2.5] | 0.9926 |
| llama-3.2-3b | intermediate | track8 | 150 | 10.0% [6.2, 15.8] | 27.3% [20.8, 35.0] | 0.1773 |
| llama-3.2-3b | intermediate | comp8 | 150 | 0.0% [0.0, 2.5] | 9.3% [5.6, 15.1] | 0.0003 |
| llama-3.2-3b | transfer | full | 150 | 100.0% [97.5, 100.0] | 0.0% [0.0, 2.5] | 0.9910 |
| llama-3.2-3b | transfer | track8 | 150 | 25.3% [19.0, 32.8] | 56.0% [48.0, 63.7] | 0.5126 |
| llama-3.2-3b | transfer | comp8 | 150 | 0.0% [0.0, 2.5] | 34.7% [27.5, 42.6] | 0.0009 |
| phi-3.5-mini | intermediate | full | 150 | 100.0% [97.5, 100.0] | 0.0% [0.0, 2.5] | 0.9987 |
| phi-3.5-mini | intermediate | track8 | 150 | 8.7% [5.1, 14.3] | 10.7% [6.7, 16.6] | 0.1115 |
| phi-3.5-mini | intermediate | comp8 | 150 | 0.0% [0.0, 2.5] | 6.0% [3.2, 11.0] | 0.0001 |
| phi-3.5-mini | transfer | full | 150 | 100.0% [97.5, 100.0] | 0.0% [0.0, 2.5] | 0.9969 |
| phi-3.5-mini | transfer | track8 | 150 | 20.0% [14.4, 27.1] | 4.0% [1.8, 8.5] | 0.2239 |
| phi-3.5-mini | transfer | comp8 | 150 | 0.0% [0.0, 2.5] | 0.7% [0.1, 3.7] | 0.0000 |
| nemotron-mini-4b | intermediate | full | 150 | 100.0% [97.5, 100.0] | 0.0% [0.0, 2.5] | 0.9937 |
| nemotron-mini-4b | intermediate | track8 | 150 | 12.7% [8.3, 18.9] | 54.7% [46.7, 62.4] | 0.3011 |
| nemotron-mini-4b | intermediate | comp8 | 150 | 0.0% [0.0, 2.5] | 22.7% [16.7, 30.0] | 0.0005 |
| nemotron-mini-4b | transfer | full | 150 | 100.0% [97.5, 100.0] | 0.0% [0.0, 2.5] | 0.9985 |
| nemotron-mini-4b | transfer | track8 | 150 | 56.0% [48.0, 63.7] | 27.3% [20.8, 35.0] | 0.6497 |
| nemotron-mini-4b | transfer | comp8 | 150 | 0.0% [0.0, 2.5] | 14.7% [9.9, 21.2] | 0.0004 |

## Paired Contrasts

| Model | Task | First minus second | Metric | Eligible pairs | Difference [95% interval] | p |
|---|---|---|---|---|---|---|
| llama-3.2-3b | intermediate | full_minus_track8 | steer | 150 | 90.0 pp [79.9, 94.3] | 4.592e-41 |
| llama-3.2-3b | intermediate | full_minus_track8 | same_sign_damage | 144 | -2.1 pp [-6.8, 2.8] | 0.25 |
| llama-3.2-3b | intermediate | full_minus_track8 | negative_edit_disruption | 144 | -12.5 pp [-23.4, -0.8] | 0.002102 |
| llama-3.2-3b | intermediate | full_minus_track8 | delta_p2way | 150 | 0.8153 [0.7729, 0.8555] | 0.0001 |
| llama-3.2-3b | intermediate | full_minus_comp8 | steer | 150 | 100.0 pp [93.5, 100.0] | 1.401e-45 |
| llama-3.2-3b | intermediate | full_minus_comp8 | same_sign_damage | 144 | -3.5 pp [-8.8, 2.0] | 0.0625 |
| llama-3.2-3b | intermediate | full_minus_comp8 | negative_edit_disruption | 144 | 45.1 pp [32.8, 54.4] | 5.421e-20 |
| llama-3.2-3b | intermediate | full_minus_comp8 | delta_p2way | 150 | 0.9923 [0.9908, 0.9937] | 0.0001 |
| llama-3.2-3b | intermediate | track8_minus_comp8 | delta_p2way | 150 | 0.1770 [0.1367, 0.2194] | 0.0001 |
| llama-3.2-3b | transfer | full_minus_track8 | steer | 150 | 74.7 pp [62.8, 81.7] | 3.852e-34 |
| llama-3.2-3b | transfer | full_minus_track8 | same_sign_damage | 110 | -5.5 pp [-12.5, 2.1] | 0.03125 |
| llama-3.2-3b | transfer | full_minus_track8 | negative_edit_disruption | 110 | 1.8 pp [-8.5, 12.0] | 0.7744 |
| llama-3.2-3b | transfer | full_minus_track8 | delta_p2way | 150 | 0.4784 [0.4168, 0.5391] | 0.0001 |
| llama-3.2-3b | transfer | full_minus_comp8 | steer | 150 | 100.0 pp [93.5, 100.0] | 1.401e-45 |
| llama-3.2-3b | transfer | full_minus_comp8 | same_sign_damage | 110 | -6.4 pp [-13.7, 1.5] | 0.01562 |
| llama-3.2-3b | transfer | full_minus_comp8 | negative_edit_disruption | 110 | 71.8 pp [57.0, 80.3] | 3.309e-24 |
| llama-3.2-3b | transfer | full_minus_comp8 | delta_p2way | 150 | 0.9901 [0.9834, 0.9949] | 0.0001 |
| llama-3.2-3b | transfer | track8_minus_comp8 | delta_p2way | 150 | 0.5116 [0.4509, 0.5744] | 0.0001 |
| phi-3.5-mini | intermediate | full_minus_track8 | steer | 150 | 91.3 pp [81.5, 95.2] | 1.148e-41 |
| phi-3.5-mini | intermediate | full_minus_track8 | same_sign_damage | 150 | -2.7 pp [-7.5, 2.3] | 0.125 |
| phi-3.5-mini | intermediate | full_minus_track8 | negative_edit_disruption | 150 | -3.3 pp [-12.8, 6.3] | 0.3833 |
| phi-3.5-mini | intermediate | full_minus_track8 | delta_p2way | 150 | 0.8872 [0.8479, 0.9228] | 0.0001 |
| phi-3.5-mini | intermediate | full_minus_comp8 | steer | 150 | 100.0 pp [93.5, 100.0] | 1.401e-45 |
| phi-3.5-mini | intermediate | full_minus_comp8 | same_sign_damage | 150 | -2.0 pp [-6.5, 2.6] | 0.25 |
| phi-3.5-mini | intermediate | full_minus_comp8 | negative_edit_disruption | 150 | 12.7 pp [4.5, 20.0] | 3.815e-06 |
| phi-3.5-mini | intermediate | full_minus_comp8 | delta_p2way | 150 | 0.9985 [0.9969, 0.9996] | 0.0001 |
| phi-3.5-mini | intermediate | track8_minus_comp8 | delta_p2way | 150 | 0.1114 [0.0756, 0.1505] | 0.0001 |
| phi-3.5-mini | transfer | full_minus_track8 | steer | 150 | 80.0 pp [68.5, 86.3] | 1.505e-36 |
| phi-3.5-mini | transfer | full_minus_track8 | same_sign_damage | 150 | -0.7 pp [-4.4, 3.1] | 1 |
| phi-3.5-mini | transfer | full_minus_track8 | negative_edit_disruption | 150 | 7.3 pp [-2.2, 16.4] | 0.0266 |
| phi-3.5-mini | transfer | full_minus_track8 | delta_p2way | 150 | 0.7730 [0.7156, 0.8273] | 0.0001 |
| phi-3.5-mini | transfer | full_minus_comp8 | steer | 150 | 100.0 pp [93.5, 100.0] | 1.401e-45 |
| phi-3.5-mini | transfer | full_minus_comp8 | same_sign_damage | 150 | 0.0 pp [-3.2, 3.2] | 1 |
| phi-3.5-mini | transfer | full_minus_comp8 | negative_edit_disruption | 150 | 12.7 pp [4.5, 20.0] | 3.815e-06 |
| phi-3.5-mini | transfer | full_minus_comp8 | delta_p2way | 150 | 0.9969 [0.9920, 0.9998] | 0.0001 |
| phi-3.5-mini | transfer | track8_minus_comp8 | delta_p2way | 150 | 0.2239 [0.1696, 0.2818] | 0.0001 |
| nemotron-mini-4b | intermediate | full_minus_track8 | steer | 150 | 87.3 pp [76.8, 92.2] | 7.347e-40 |
| nemotron-mini-4b | intermediate | full_minus_track8 | same_sign_damage | 137 | -22.6 pp [-31.5, -12.1] | 9.313e-10 |
| nemotron-mini-4b | intermediate | full_minus_track8 | negative_edit_disruption | 137 | -0.7 pp [-11.9, 10.5] | 1 |
| nemotron-mini-4b | intermediate | full_minus_track8 | delta_p2way | 150 | 0.6926 [0.6367, 0.7474] | 0.0001 |
| nemotron-mini-4b | intermediate | full_minus_comp8 | steer | 150 | 100.0 pp [93.5, 100.0] | 1.401e-45 |
| nemotron-mini-4b | intermediate | full_minus_comp8 | same_sign_damage | 137 | -6.6 pp [-13.0, 0.3] | 0.003906 |
| nemotron-mini-4b | intermediate | full_minus_comp8 | negative_edit_disruption | 137 | 73.7 pp [61.0, 81.2] | 7.889e-31 |
| nemotron-mini-4b | intermediate | full_minus_comp8 | delta_p2way | 150 | 0.9932 [0.9892, 0.9965] | 0.0001 |
| nemotron-mini-4b | intermediate | track8_minus_comp8 | delta_p2way | 150 | 0.3006 [0.2460, 0.3565] | 0.0001 |
| nemotron-mini-4b | transfer | full_minus_track8 | steer | 150 | 44.0 pp [32.0, 53.1] | 2.711e-20 |
| nemotron-mini-4b | transfer | full_minus_track8 | same_sign_damage | 140 | -5.0 pp [-10.9, 1.3] | 0.01562 |
| nemotron-mini-4b | transfer | full_minus_track8 | negative_edit_disruption | 140 | -5.7 pp [-17.6, 6.5] | 0.2153 |
| nemotron-mini-4b | transfer | full_minus_track8 | delta_p2way | 150 | 0.3488 [0.2908, 0.4078] | 0.0001 |
| nemotron-mini-4b | transfer | full_minus_comp8 | steer | 150 | 100.0 pp [93.5, 100.0] | 1.401e-45 |
| nemotron-mini-4b | transfer | full_minus_comp8 | same_sign_damage | 140 | -7.1 pp [-13.6, -0.1] | 0.001953 |
| nemotron-mini-4b | transfer | full_minus_comp8 | negative_edit_disruption | 140 | 62.1 pp [49.2, 70.8] | 1.292e-26 |
| nemotron-mini-4b | transfer | full_minus_comp8 | delta_p2way | 150 | 0.9981 [0.9967, 0.9991] | 0.0001 |
| nemotron-mini-4b | transfer | track8_minus_comp8 | delta_p2way | 150 | 0.6492 [0.5904, 0.7071] | 0.0001 |

Binary contrasts use paired gain/loss Wilson bounds with a within-contrast Bonferroni adjustment and exact McNemar tests. Continuous contrasts use the saved seed-stratified paired bootstrap and sign-permutation procedure. These exploratory contrasts do not have a study-wide multiplicity correction. Intervals are conditional on the observed seed blocks; equal rates or p=1 do not establish equivalence.

## Capability Accuracy and Controls

| Model | Edit seed | Exposure | Benchmark | Items | Baseline | Edited | Accuracy change [95% interval] | 2 pp loss bound |
|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b | 0 | active_prefix | hellaswag | 240 | 57.1% [50.8, 63.2] | 58.3% [52.0, 64.4] | 1.2 pp [-1.7, 4.1] | met |
| llama-3.2-3b | 0 | active_prefix | arc_easy | 200 | 63.0% [56.1, 69.4] | 61.0% [54.1, 67.5] | -2.0 pp [-6.8, 2.9] | not established |
| llama-3.2-3b | 0 | random_prefix | hellaswag | 240 | 57.1% [50.8, 63.2] | 57.5% [51.2, 63.6] | 0.4 pp [-2.0, 2.8] | met |
| llama-3.2-3b | 0 | random_prefix | arc_easy | 200 | 63.0% [56.1, 69.4] | 63.5% [56.6, 69.9] | 0.5 pp [-4.2, 5.2] | not established |
| llama-3.2-3b | 0 | zero_prefix | hellaswag | 240 | 57.1% [50.8, 63.2] | 57.1% [50.8, 63.2] | 0.0 pp [-2.1, 2.1] | not established |
| llama-3.2-3b | 0 | zero_prefix | arc_easy | 200 | 63.0% [56.1, 69.4] | 63.0% [56.1, 69.4] | 0.0 pp [-2.5, 2.5] | not established |
| llama-3.2-3b | 0 | global | hellaswag | 240 | 57.1% [50.8, 63.2] | 54.6% [48.3, 60.8] | -2.5 pp [-7.5, 2.6] | not established |
| llama-3.2-3b | 0 | global | arc_easy | 200 | 63.0% [56.1, 69.4] | 60.0% [53.1, 66.5] | -3.0 pp [-11.1, 5.3] | not established |
| llama-3.2-3b | 1 | active_prefix | hellaswag | 240 | 57.1% [50.8, 63.2] | 59.2% [52.9, 65.2] | 2.1 pp [-1.8, 5.9] | met |
| llama-3.2-3b | 1 | active_prefix | arc_easy | 200 | 63.0% [56.1, 69.4] | 58.5% [51.6, 65.1] | -4.5 pp [-10.9, 2.1] | not established |
| llama-3.2-3b | 1 | random_prefix | hellaswag | 240 | 57.1% [50.8, 63.2] | 57.5% [51.2, 63.6] | 0.4 pp [-2.6, 3.4] | not established |
| llama-3.2-3b | 1 | random_prefix | arc_easy | 200 | 63.0% [56.1, 69.4] | 63.5% [56.6, 69.9] | 0.5 pp [-4.7, 5.7] | not established |
| llama-3.2-3b | 1 | zero_prefix | hellaswag | 240 | 57.1% [50.8, 63.2] | 57.1% [50.8, 63.2] | 0.0 pp [-2.1, 2.1] | not established |
| llama-3.2-3b | 1 | zero_prefix | arc_easy | 200 | 63.0% [56.1, 69.4] | 63.0% [56.1, 69.4] | 0.0 pp [-2.5, 2.5] | not established |
| llama-3.2-3b | 1 | global | hellaswag | 240 | 57.1% [50.8, 63.2] | 56.7% [50.3, 62.8] | -0.4 pp [-6.3, 5.5] | not established |
| llama-3.2-3b | 1 | global | arc_easy | 200 | 63.0% [56.1, 69.4] | 60.0% [53.1, 66.5] | -3.0 pp [-11.9, 6.0] | not established |
| llama-3.2-3b | 2 | active_prefix | hellaswag | 240 | 57.1% [50.8, 63.2] | 57.5% [51.2, 63.6] | 0.4 pp [-2.0, 2.8] | met |
| llama-3.2-3b | 2 | active_prefix | arc_easy | 200 | 63.0% [56.1, 69.4] | 59.0% [52.1, 65.6] | -4.0 pp [-9.0, 1.2] | not established |
| llama-3.2-3b | 2 | random_prefix | hellaswag | 240 | 57.1% [50.8, 63.2] | 57.9% [51.6, 64.0] | 0.8 pp [-1.9, 3.5] | met |
| llama-3.2-3b | 2 | random_prefix | arc_easy | 200 | 63.0% [56.1, 69.4] | 62.0% [55.1, 68.4] | -1.0 pp [-5.4, 3.5] | not established |
| llama-3.2-3b | 2 | zero_prefix | hellaswag | 240 | 57.1% [50.8, 63.2] | 57.1% [50.8, 63.2] | 0.0 pp [-2.1, 2.1] | not established |
| llama-3.2-3b | 2 | zero_prefix | arc_easy | 200 | 63.0% [56.1, 69.4] | 63.0% [56.1, 69.4] | 0.0 pp [-2.5, 2.5] | not established |
| llama-3.2-3b | 2 | global | hellaswag | 240 | 57.1% [50.8, 63.2] | 55.8% [49.5, 62.0] | -1.2 pp [-6.2, 3.8] | not established |
| llama-3.2-3b | 2 | global | arc_easy | 200 | 63.0% [56.1, 69.4] | 58.0% [51.1, 64.6] | -5.0 pp [-11.8, 2.1] | not established |
| phi-3.5-mini | 0 | active_prefix | hellaswag | 240 | 58.8% [52.4, 64.8] | 57.5% [51.2, 63.6] | -1.2 pp [-4.7, 2.2] | not established |
| phi-3.5-mini | 0 | active_prefix | arc_easy | 200 | 74.0% [67.5, 79.6] | 71.5% [64.9, 77.3] | -2.5 pp [-7.5, 2.6] | not established |
| phi-3.5-mini | 0 | random_prefix | hellaswag | 240 | 58.8% [52.4, 64.8] | 57.5% [51.2, 63.6] | -1.2 pp [-4.7, 2.2] | not established |
| phi-3.5-mini | 0 | random_prefix | arc_easy | 200 | 74.0% [67.5, 79.6] | 73.5% [67.0, 79.1] | -0.5 pp [-5.2, 4.2] | not established |
| phi-3.5-mini | 0 | zero_prefix | hellaswag | 240 | 58.8% [52.4, 64.8] | 58.8% [52.4, 64.8] | 0.0 pp [-2.1, 2.1] | not established |
| phi-3.5-mini | 0 | zero_prefix | arc_easy | 200 | 74.0% [67.5, 79.6] | 74.0% [67.5, 79.6] | 0.0 pp [-2.5, 2.5] | not established |
| phi-3.5-mini | 0 | global | hellaswag | 240 | 58.8% [52.4, 64.8] | 56.7% [50.3, 62.8] | -2.1 pp [-9.3, 5.2] | not established |
| phi-3.5-mini | 0 | global | arc_easy | 200 | 74.0% [67.5, 79.6] | 67.0% [60.2, 73.1] | -7.0 pp [-15.3, 1.7] | not established |
| phi-3.5-mini | 1 | active_prefix | hellaswag | 240 | 58.8% [52.4, 64.8] | 60.4% [54.1, 66.4] | 1.7 pp [-3.2, 6.5] | not established |
| phi-3.5-mini | 1 | active_prefix | arc_easy | 200 | 74.0% [67.5, 79.6] | 71.0% [64.4, 76.8] | -3.0 pp [-8.2, 2.3] | not established |
| phi-3.5-mini | 1 | random_prefix | hellaswag | 240 | 58.8% [52.4, 64.8] | 57.1% [50.8, 63.2] | -1.7 pp [-5.3, 2.0] | not established |
| phi-3.5-mini | 1 | random_prefix | arc_easy | 200 | 74.0% [67.5, 79.6] | 74.0% [67.5, 79.6] | 0.0 pp [-3.9, 3.9] | not established |
| phi-3.5-mini | 1 | zero_prefix | hellaswag | 240 | 58.8% [52.4, 64.8] | 58.8% [52.4, 64.8] | 0.0 pp [-2.1, 2.1] | not established |
| phi-3.5-mini | 1 | zero_prefix | arc_easy | 200 | 74.0% [67.5, 79.6] | 74.0% [67.5, 79.6] | 0.0 pp [-2.5, 2.5] | not established |
| phi-3.5-mini | 1 | global | hellaswag | 240 | 58.8% [52.4, 64.8] | 59.6% [53.3, 65.6] | 0.8 pp [-6.5, 8.2] | not established |
| phi-3.5-mini | 1 | global | arc_easy | 200 | 74.0% [67.5, 79.6] | 67.0% [60.2, 73.1] | -7.0 pp [-15.3, 1.7] | not established |
| phi-3.5-mini | 2 | active_prefix | hellaswag | 240 | 58.8% [52.4, 64.8] | 60.8% [54.5, 66.8] | 2.1 pp [-2.2, 6.3] | not established |
| phi-3.5-mini | 2 | active_prefix | arc_easy | 200 | 74.0% [67.5, 79.6] | 73.5% [67.0, 79.1] | -0.5 pp [-6.5, 5.5] | not established |
| phi-3.5-mini | 2 | random_prefix | hellaswag | 240 | 58.8% [52.4, 64.8] | 57.9% [51.6, 64.0] | -0.8 pp [-3.5, 1.9] | not established |
| phi-3.5-mini | 2 | random_prefix | arc_easy | 200 | 74.0% [67.5, 79.6] | 73.0% [66.5, 78.7] | -1.0 pp [-5.4, 3.5] | not established |
| phi-3.5-mini | 2 | zero_prefix | hellaswag | 240 | 58.8% [52.4, 64.8] | 58.8% [52.4, 64.8] | 0.0 pp [-2.1, 2.1] | not established |
| phi-3.5-mini | 2 | zero_prefix | arc_easy | 200 | 74.0% [67.5, 79.6] | 74.0% [67.5, 79.6] | 0.0 pp [-2.5, 2.5] | not established |
| phi-3.5-mini | 2 | global | hellaswag | 240 | 58.8% [52.4, 64.8] | 55.8% [49.5, 62.0] | -2.9 pp [-11.8, 6.1] | not established |
| phi-3.5-mini | 2 | global | arc_easy | 200 | 74.0% [67.5, 79.6] | 69.0% [62.3, 75.0] | -5.0 pp [-14.2, 4.4] | not established |
| nemotron-mini-4b | 0 | active_prefix | hellaswag | 240 | 54.6% [48.3, 60.8] | 54.6% [48.3, 60.8] | 0.0 pp [-4.5, 4.5] | not established |
| nemotron-mini-4b | 0 | active_prefix | arc_easy | 200 | 75.5% [69.1, 80.9] | 71.0% [64.4, 76.8] | -4.5 pp [-9.6, 0.8] | not established |
| nemotron-mini-4b | 0 | random_prefix | hellaswag | 240 | 54.6% [48.3, 60.8] | 55.0% [48.7, 61.2] | 0.4 pp [-3.9, 4.8] | not established |
| nemotron-mini-4b | 0 | random_prefix | arc_easy | 200 | 75.5% [69.1, 80.9] | 74.0% [67.5, 79.6] | -1.5 pp [-7.0, 4.1] | not established |
| nemotron-mini-4b | 0 | zero_prefix | hellaswag | 240 | 54.6% [48.3, 60.8] | 54.6% [48.3, 60.8] | 0.0 pp [-2.1, 2.1] | not established |
| nemotron-mini-4b | 0 | zero_prefix | arc_easy | 200 | 75.5% [69.1, 80.9] | 75.5% [69.1, 80.9] | 0.0 pp [-2.5, 2.5] | not established |
| nemotron-mini-4b | 0 | global | hellaswag | 240 | 54.6% [48.3, 60.8] | 55.0% [48.7, 61.2] | 0.4 pp [-6.9, 7.7] | not established |
| nemotron-mini-4b | 0 | global | arc_easy | 200 | 75.5% [69.1, 80.9] | 73.0% [66.5, 78.7] | -2.5 pp [-10.3, 5.4] | not established |
| nemotron-mini-4b | 1 | active_prefix | hellaswag | 240 | 54.6% [48.3, 60.8] | 55.4% [49.1, 61.6] | 0.8 pp [-3.3, 5.0] | not established |
| nemotron-mini-4b | 1 | active_prefix | arc_easy | 200 | 75.5% [69.1, 80.9] | 75.0% [68.6, 80.5] | -0.5 pp [-5.2, 4.2] | not established |
| nemotron-mini-4b | 1 | random_prefix | hellaswag | 240 | 54.6% [48.3, 60.8] | 55.0% [48.7, 61.2] | 0.4 pp [-3.1, 3.9] | not established |
| nemotron-mini-4b | 1 | random_prefix | arc_easy | 200 | 75.5% [69.1, 80.9] | 75.0% [68.6, 80.5] | -0.5 pp [-5.7, 4.7] | not established |
| nemotron-mini-4b | 1 | zero_prefix | hellaswag | 240 | 54.6% [48.3, 60.8] | 54.6% [48.3, 60.8] | 0.0 pp [-2.1, 2.1] | not established |
| nemotron-mini-4b | 1 | zero_prefix | arc_easy | 200 | 75.5% [69.1, 80.9] | 75.5% [69.1, 80.9] | 0.0 pp [-2.5, 2.5] | not established |
| nemotron-mini-4b | 1 | global | hellaswag | 240 | 54.6% [48.3, 60.8] | 52.1% [45.8, 58.3] | -2.5 pp [-10.9, 6.0] | not established |
| nemotron-mini-4b | 1 | global | arc_easy | 200 | 75.5% [69.1, 80.9] | 71.0% [64.4, 76.8] | -4.5 pp [-13.2, 4.4] | not established |
| nemotron-mini-4b | 2 | active_prefix | hellaswag | 240 | 54.6% [48.3, 60.8] | 53.3% [47.0, 59.5] | -1.2 pp [-4.1, 1.7] | not established |
| nemotron-mini-4b | 2 | active_prefix | arc_easy | 200 | 75.5% [69.1, 80.9] | 77.0% [70.7, 82.3] | 1.5 pp [-2.7, 5.6] | not established |
| nemotron-mini-4b | 2 | random_prefix | hellaswag | 240 | 54.6% [48.3, 60.8] | 54.2% [47.8, 60.4] | -0.4 pp [-3.4, 2.6] | not established |
| nemotron-mini-4b | 2 | random_prefix | arc_easy | 200 | 75.5% [69.1, 80.9] | 75.5% [69.1, 80.9] | 0.0 pp [-3.3, 3.3] | not established |
| nemotron-mini-4b | 2 | zero_prefix | hellaswag | 240 | 54.6% [48.3, 60.8] | 54.6% [48.3, 60.8] | 0.0 pp [-2.1, 2.1] | not established |
| nemotron-mini-4b | 2 | zero_prefix | arc_easy | 200 | 75.5% [69.1, 80.9] | 75.5% [69.1, 80.9] | 0.0 pp [-2.5, 2.5] | not established |
| nemotron-mini-4b | 2 | global | hellaswag | 240 | 54.6% [48.3, 60.8] | 56.7% [50.3, 62.8] | 2.1 pp [-2.9, 7.0] | not established |
| nemotron-mini-4b | 2 | global | arc_easy | 200 | 75.5% [69.1, 80.9] | 74.5% [68.0, 80.0] | -1.0 pp [-8.4, 6.5] | not established |

Each edit reuses the same 240 HellaSwag items and 200 ARC-Easy items. The rows are not independent datasets and their confidence endpoints are not averaged. The accuracy criterion requires the paired lower bound to be strictly above -2 pp. Failure to meet that bound may reflect uncertainty, damage, or both. Global exposure is a stress test, not an asserted upper bound on other deployment policies.

## Text Exposure

| Model | Edit seed | Exposure | Perplexity ratio [95% interval] | 10% ratio bound | Window 1-5 mean NLL changes |
|---|---|---|---|---|---|
| llama-3.2-3b | 0 | active_prefix | 0.9990 [0.9986, 0.9994] | met | -0.00126, -0.00072, -0.00156, -0.00111, -0.00036 |
| llama-3.2-3b | 0 | random_prefix | 0.9996 [0.9987, 1.0001] | met | 0.00013, 0.00016, 0.00012, -0.00049, -0.00201 |
| llama-3.2-3b | 0 | zero_prefix | 1.0000 [1.0000, 1.0000] | met | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| llama-3.2-3b | 0 | global | 1.3922 [1.2926, 1.5424] | not established | 0.28354, 0.33999, 0.53105, 0.23356, 0.26639 |
| llama-3.2-3b | 1 | active_prefix | 0.9999 [0.9984, 1.0016] | met | -0.00134, 0.00146, -0.00206, 0.00266, -0.00132 |
| llama-3.2-3b | 1 | random_prefix | 0.9996 [0.9990, 1.0004] | met | -0.00144, -0.00005, -0.00065, 0.00112, -0.00082 |
| llama-3.2-3b | 1 | zero_prefix | 1.0000 [1.0000, 1.0000] | met | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| llama-3.2-3b | 1 | global | 1.7946 [1.5695, 2.1259] | not established | 0.52250, 0.60491, 0.91872, 0.49232, 0.38554 |
| llama-3.2-3b | 2 | active_prefix | 0.9997 [0.9989, 1.0006] | met | -0.00132, 0.00145, -0.00013, -0.00103, -0.00060 |
| llama-3.2-3b | 2 | random_prefix | 0.9999 [0.9991, 1.0006] | met | -0.00129, -0.00005, 0.00115, 0.00027, -0.00071 |
| llama-3.2-3b | 2 | zero_prefix | 1.0000 [1.0000, 1.0000] | met | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| llama-3.2-3b | 2 | global | 1.4502 [1.3227, 1.5899] | not established | 0.39905, 0.49184, 0.46785, 0.24087, 0.25882 |
| phi-3.5-mini | 0 | active_prefix | 1.0031 [1.0017, 1.0052] | met | 0.00176, 0.00211, 0.00152, 0.00701, 0.00323 |
| phi-3.5-mini | 0 | random_prefix | 0.9998 [0.9983, 1.0010] | met | 0.00021, 0.00171, -0.00014, -0.00292, 0.00004 |
| phi-3.5-mini | 0 | zero_prefix | 1.0000 [1.0000, 1.0000] | met | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| phi-3.5-mini | 0 | global | 1.4991 [1.4152, 1.6698] | not established | 0.61883, 0.35224, 0.34594, 0.36114, 0.34605 |
| phi-3.5-mini | 1 | active_prefix | 1.0018 [1.0008, 1.0034] | met | 0.00175, 0.00103, 0.00093, 0.00490, 0.00047 |
| phi-3.5-mini | 1 | random_prefix | 1.0000 [0.9993, 1.0006] | met | 0.00043, -0.00102, 0.00045, -0.00075, 0.00075 |
| phi-3.5-mini | 1 | zero_prefix | 1.0000 [1.0000, 1.0000] | met | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| phi-3.5-mini | 1 | global | 1.4677 [1.3946, 1.5911] | not established | 0.54009, 0.34384, 0.31577, 0.36789, 0.35095 |
| phi-3.5-mini | 2 | active_prefix | 1.0008 [0.9985, 1.0034] | met | 0.00554, -0.00319, 0.00163, -0.00132, 0.00154 |
| phi-3.5-mini | 2 | random_prefix | 1.0007 [0.9997, 1.0020] | met | -0.00096, 0.00305, 0.00055, 0.00016, 0.00058 |
| phi-3.5-mini | 2 | zero_prefix | 1.0000 [1.0000, 1.0000] | met | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| phi-3.5-mini | 2 | global | 1.8089 [1.6476, 2.0738] | not established | 0.86039, 0.47340, 0.52498, 0.59234, 0.51243 |
| nemotron-mini-4b | 0 | active_prefix | 1.0015 [0.9986, 1.0048] | met | 0.00771, -0.00106, 0.00309, -0.00251, 0.00030 |
| nemotron-mini-4b | 0 | random_prefix | 0.9999 [0.9985, 1.0013] | met | -0.00045, -0.00035, -0.00254, 0.00218, 0.00068 |
| nemotron-mini-4b | 0 | zero_prefix | 1.0000 [1.0000, 1.0000] | met | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| nemotron-mini-4b | 0 | global | 1.6489 [1.4386, 2.0181] | not established | 0.89255, 0.39800, 0.45886, 0.43524, 0.31594 |
| nemotron-mini-4b | 1 | active_prefix | 1.0027 [0.9985, 1.0070] | met | 0.00996, -0.00190, 0.00667, -0.00226, 0.00085 |
| nemotron-mini-4b | 1 | random_prefix | 1.0008 [0.9990, 1.0026] | met | 0.00358, -0.00077, 0.00297, -0.00079, -0.00122 |
| nemotron-mini-4b | 1 | zero_prefix | 1.0000 [1.0000, 1.0000] | met | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| nemotron-mini-4b | 1 | global | 1.8976 [1.6170, 2.3688] | not established | 1.05889, 0.55386, 0.58141, 0.59532, 0.41338 |
| nemotron-mini-4b | 2 | active_prefix | 1.0009 [0.9989, 1.0030] | met | 0.00360, 0.00036, 0.00362, -0.00248, -0.00044 |
| nemotron-mini-4b | 2 | random_prefix | 0.9998 [0.9997, 0.9999] | met | -0.00011, -0.00004, -0.00032, -0.00024, -0.00037 |
| nemotron-mini-4b | 2 | zero_prefix | 1.0000 [1.0000, 1.0000] | met | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| nemotron-mini-4b | 2 | global | 1.1492 [1.0916, 1.2453] | not established | 0.29897, 0.07764, 0.11096, 0.12186, 0.08597 |

The ratio is the exponentiated mean of five paired window-level changes in mean token negative log likelihood (NLL). The 95% interval bootstraps those five fixed windows; it is not population-wide text uncertainty. All windows come from one public text. The declared ratio upper bound must be strictly below 1.10. A result of 1 means unchanged measured perplexity. Baseline and zero-control token arrays remain in the JSON.

## Capability Edit Norms

| Model | Seed | Exposure | Layers | Recorded norms | Per-layer budgets |
|---|---|---|---|---|---|
| llama-3.2-3b | 0 | active_prefix | 18, 20, 22, 24 | 3.1694, 3.4559, 3.5987, 4.4475 | 5.3737, 6.7314, 7.4637, 8.8692 |
| llama-3.2-3b | 0 | random_prefix | 18, 20, 22, 24 | 3.1694, 3.4559, 3.5987, 4.4475 | 5.3737, 6.7314, 7.4637, 8.8692 |
| llama-3.2-3b | 0 | zero_prefix | 18, 20, 22, 24 | 0.0000, 0.0000, 0.0000, 0.0000 | 5.3737, 6.7314, 7.4637, 8.8692 |
| llama-3.2-3b | 0 | global | 18, 20, 22, 24 | 3.1694, 3.4559, 3.5987, 4.4475 | 5.3737, 6.7314, 7.4637, 8.8692 |
| llama-3.2-3b | 1 | active_prefix | 18, 20, 22, 24 | 3.5789, 4.0702, 4.2032, 5.2833 | 5.4283, 6.8783, 7.5715, 8.7730 |
| llama-3.2-3b | 1 | random_prefix | 18, 20, 22, 24 | 3.5789, 4.0702, 4.2032, 5.2833 | 5.4283, 6.8783, 7.5715, 8.7730 |
| llama-3.2-3b | 1 | zero_prefix | 18, 20, 22, 24 | 0.0000, 0.0000, 0.0000, 0.0000 | 5.4283, 6.8783, 7.5715, 8.7730 |
| llama-3.2-3b | 1 | global | 18, 20, 22, 24 | 3.5789, 4.0702, 4.2032, 5.2833 | 5.4283, 6.8783, 7.5715, 8.7730 |
| llama-3.2-3b | 2 | active_prefix | 18, 20, 22, 24 | 3.2404, 3.6201, 3.5159, 4.1584 | 5.3669, 6.7635, 7.4859, 8.8923 |
| llama-3.2-3b | 2 | random_prefix | 18, 20, 22, 24 | 3.2404, 3.6201, 3.5159, 4.1584 | 5.3669, 6.7635, 7.4859, 8.8923 |
| llama-3.2-3b | 2 | zero_prefix | 18, 20, 22, 24 | 0.0000, 0.0000, 0.0000, 0.0000 | 5.3669, 6.7635, 7.4859, 8.8923 |
| llama-3.2-3b | 2 | global | 18, 20, 22, 24 | 3.2404, 3.6201, 3.5159, 4.1584 | 5.3669, 6.7635, 7.4859, 8.8923 |
| phi-3.5-mini | 0 | active_prefix | 21, 23, 25, 28 | 37.2547, 39.4734, 40.5570, 36.9315 | 58.3868, 72.6104, 87.8418, 111.8511 |
| phi-3.5-mini | 0 | random_prefix | 21, 23, 25, 28 | 37.2547, 39.4734, 40.5570, 36.9315 | 58.3868, 72.6104, 87.8418, 111.8511 |
| phi-3.5-mini | 0 | zero_prefix | 21, 23, 25, 28 | 0.0000, 0.0000, 0.0000, 0.0000 | 58.3868, 72.6104, 87.8418, 111.8511 |
| phi-3.5-mini | 0 | global | 21, 23, 25, 28 | 37.2547, 39.4734, 40.5570, 36.9315 | 58.3868, 72.6104, 87.8418, 111.8511 |
| phi-3.5-mini | 1 | active_prefix | 21, 23, 25, 28 | 40.9787, 41.2962, 39.8820, 37.4012 | 58.5281, 72.6551, 86.9724, 111.0814 |
| phi-3.5-mini | 1 | random_prefix | 21, 23, 25, 28 | 40.9787, 41.2962, 39.8820, 37.4012 | 58.5281, 72.6551, 86.9724, 111.0814 |
| phi-3.5-mini | 1 | zero_prefix | 21, 23, 25, 28 | 0.0000, 0.0000, 0.0000, 0.0000 | 58.5281, 72.6551, 86.9724, 111.0814 |
| phi-3.5-mini | 1 | global | 21, 23, 25, 28 | 40.9787, 41.2962, 39.8820, 37.4012 | 58.5281, 72.6551, 86.9724, 111.0814 |
| phi-3.5-mini | 2 | active_prefix | 21, 23, 25, 28 | 47.8872, 50.1479, 50.0165, 54.4698 | 59.1276, 73.3947, 88.0086, 111.0699 |
| phi-3.5-mini | 2 | random_prefix | 21, 23, 25, 28 | 47.8872, 50.1479, 50.0165, 54.4698 | 59.1276, 73.3947, 88.0086, 111.0699 |
| phi-3.5-mini | 2 | zero_prefix | 21, 23, 25, 28 | 0.0000, 0.0000, 0.0000, 0.0000 | 59.1276, 73.3947, 88.0086, 111.0699 |
| phi-3.5-mini | 2 | global | 21, 23, 25, 28 | 47.8872, 50.1479, 50.0165, 54.4698 | 59.1276, 73.3947, 88.0086, 111.0699 |
| nemotron-mini-4b | 0 | active_prefix | 21, 23, 25, 28 | 45.4773, 48.0027, 48.9226, 63.6531 | 45.4773, 53.8489, 62.8510, 76.1916 |
| nemotron-mini-4b | 0 | random_prefix | 21, 23, 25, 28 | 45.4773, 48.0027, 48.9226, 63.6531 | 45.4773, 53.8489, 62.8510, 76.1916 |
| nemotron-mini-4b | 0 | zero_prefix | 21, 23, 25, 28 | 0.0000, 0.0000, 0.0000, 0.0000 | 45.4773, 53.8489, 62.8510, 76.1916 |
| nemotron-mini-4b | 0 | global | 21, 23, 25, 28 | 45.4773, 48.0027, 48.9226, 63.6531 | 45.4773, 53.8489, 62.8510, 76.1916 |
| nemotron-mini-4b | 1 | active_prefix | 21, 23, 25, 28 | 45.3134, 53.2740, 61.5730, 63.2015 | 45.3134, 53.2740, 62.2037, 75.7406 |
| nemotron-mini-4b | 1 | random_prefix | 21, 23, 25, 28 | 45.3134, 53.2740, 61.5730, 63.2015 | 45.3134, 53.2740, 62.2037, 75.7406 |
| nemotron-mini-4b | 1 | zero_prefix | 21, 23, 25, 28 | 0.0000, 0.0000, 0.0000, 0.0000 | 45.3134, 53.2740, 62.2037, 75.7406 |
| nemotron-mini-4b | 1 | global | 21, 23, 25, 28 | 45.3134, 53.2740, 61.5730, 63.2015 | 45.3134, 53.2740, 62.2037, 75.7406 |
| nemotron-mini-4b | 2 | active_prefix | 21, 23, 25, 28 | 14.2148, 14.3009, 15.0971, 19.9185 | 44.8096, 52.8806, 61.5782, 75.3861 |
| nemotron-mini-4b | 2 | random_prefix | 21, 23, 25, 28 | 14.2148, 14.3009, 15.0971, 19.9185 | 44.8096, 52.8806, 61.5782, 75.3861 |
| nemotron-mini-4b | 2 | zero_prefix | 21, 23, 25, 28 | 0.0000, 0.0000, 0.0000, 0.0000 | 44.8096, 52.8806, 61.5782, 75.3861 |
| nemotron-mini-4b | 2 | global | 21, 23, 25, 28 | 14.2148, 14.3009, 15.0971, 19.9185 | 44.8096, 52.8806, 61.5782, 75.3861 |

## Head Diagnostics With Uncertainty

| Model | Diagnostic | Valid paired examples | Difference [95% interval] | p |
|---|---|---|---|---|
| llama-3.2-3b | Top8 minus depth-matched random faithfulness | 72 | 0.7410 [0.7239, 0.7578] | 0.0001 |
| llama-3.2-3b | Selected minus random receiver compensation | 72 | 0.2753 [0.2316, 0.3203] | 0.0001 |
| phi-3.5-mini | Top8 minus depth-matched random faithfulness | 72 | 0.5578 [0.5420, 0.5736] | 0.0001 |
| phi-3.5-mini | Selected minus random receiver compensation | 72 | -0.0327 [-0.0550, -0.0107] | 0.0053 |
| nemotron-mini-4b | Top8 minus depth-matched random faithfulness | 72 | 0.4467 [0.4307, 0.4626] | 0.0001 |
| nemotron-mini-4b | Selected minus random receiver compensation | 72 | 0.0677 [0.0555, 0.0799] | 0.0001 |

Head discovery uses 12 seed-0 training examples. Verification uses 24 held-out examples from each seed, with five depth-matched random top8 sets. Faithfulness is normalized logit-gap recovery, not question-answering accuracy. Compensation is a logit-margin difference for a selected source/receiver intervention, not a full circuit or exhaustive self-repair analysis. The recorded projection check uses natural replacement norms; it is not the same intervention as budgeted additive steering.

## Evidence Limits

The historical model fingerprint used configuration and root weight-file sizes, not hashes of weight contents. Head artifacts also lack run-time source/weight hashes. Post-run checksums identify the files available now, not a retrospectively verified remote revision. Continuous intervention deltas and head faithfulness cannot be reconstructed from missing original full logits. No claim of optimizer convergence, universal steering impossibility, general capability preservation, or publication readiness follows from completion of the consistency audit.
