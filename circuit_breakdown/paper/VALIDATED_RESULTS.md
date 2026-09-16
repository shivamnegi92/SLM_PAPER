# Validated Evidence Ledger

Status: declared matrix complete

Generated from saved artifacts, not inferred from log messages or planned runs.

The [detailed supplement](RESULTS_DETAILS.md) adds per-seed rates and denominators,
paired contrasts, all capability controls, text-window results, and head
confidence intervals. The [post-run audit](../results/postrun_audit_v1/consistency.json)
checks consistency of saved evidence, not historical full logits or weight provenance.

## Task Competence

| Model | Task | Dev clean | Dev counterfactual | Causal-control gate |
|---|---|---:|---:|---|
| llama-3.2-3b | intermediate | 95.8% | 100.0% | passed |
| llama-3.2-3b | transfer | 91.7% | 91.7% | passed |
| llama-3.2-3b | container_swap | 12.5% | 8.3% | failed; baseline-only |
| phi-3.5-mini | intermediate | 100.0% | 100.0% | passed |
| phi-3.5-mini | transfer | 100.0% | 100.0% | passed |
| phi-3.5-mini | container_swap | 12.5% | 41.7% | failed; baseline-only |
| nemotron-mini-4b | intermediate | 100.0% | 95.8% | passed |
| nemotron-mini-4b | transfer | 100.0% | 95.8% | passed |
| nemotron-mini-4b | container_swap | 12.5% | 20.8% | failed; baseline-only |

## Locked Intervention Results

| Model | Task | Condition | Unique test pairs | Target override | Same-sign damage | Negative disruption |
|---|---|---|---:|---:|---:|---:|
| llama-3.2-3b | intermediate | full | 150 | 100.0% | 0.0% | 50.0% |
| llama-3.2-3b | intermediate | track8 | 150 | 10.0% | 2.1% | 62.5% |
| llama-3.2-3b | intermediate | comp8 | 150 | 0.0% | 3.5% | 4.9% |
| llama-3.2-3b | transfer | full | 150 | 100.0% | 0.0% | 84.5% |
| llama-3.2-3b | transfer | track8 | 150 | 25.3% | 5.5% | 82.7% |
| llama-3.2-3b | transfer | comp8 | 150 | 0.0% | 6.4% | 12.7% |
| phi-3.5-mini | intermediate | full | 150 | 100.0% | 0.0% | 13.3% |
| phi-3.5-mini | intermediate | track8 | 150 | 8.7% | 2.7% | 16.7% |
| phi-3.5-mini | intermediate | comp8 | 150 | 0.0% | 2.0% | 0.7% |
| phi-3.5-mini | transfer | full | 150 | 100.0% | 0.0% | 12.7% |
| phi-3.5-mini | transfer | track8 | 150 | 20.0% | 0.7% | 5.3% |
| phi-3.5-mini | transfer | comp8 | 150 | 0.0% | 0.0% | 0.0% |
| nemotron-mini-4b | intermediate | full | 150 | 100.0% | 0.0% | 79.6% |
| nemotron-mini-4b | intermediate | track8 | 150 | 12.7% | 22.6% | 80.3% |
| nemotron-mini-4b | intermediate | comp8 | 150 | 0.0% | 6.6% | 5.8% |
| nemotron-mini-4b | transfer | full | 150 | 100.0% | 0.0% | 63.6% |
| nemotron-mini-4b | transfer | track8 | 150 | 56.0% | 5.0% | 69.3% |
| nemotron-mini-4b | transfer | comp8 | 150 | 0.0% | 7.1% | 1.4% |

Damage and disruption are conditioned on baseline-correct clean answers; denominators and intervals are in the JSON ledger.

## Capability Exposure

| Model | Edit seed | Benchmark | Active-prefix accuracy change (pp) | Paired interval (pp) |
|---|---:|---|---:|---|
| llama-3.2-3b | 0 | hellaswag | 1.2 | [-1.7, 4.1] |
| llama-3.2-3b | 0 | arc_easy | -2.0 | [-6.8, 2.9] |
| llama-3.2-3b | 1 | hellaswag | 2.1 | [-1.8, 5.9] |
| llama-3.2-3b | 1 | arc_easy | -4.5 | [-10.9, 2.1] |
| llama-3.2-3b | 2 | hellaswag | 0.4 | [-2.0, 2.8] |
| llama-3.2-3b | 2 | arc_easy | -4.0 | [-9.0, 1.2] |
| phi-3.5-mini | 0 | hellaswag | -1.2 | [-4.7, 2.2] |
| phi-3.5-mini | 0 | arc_easy | -2.5 | [-7.5, 2.6] |
| phi-3.5-mini | 1 | hellaswag | 1.7 | [-3.2, 6.5] |
| phi-3.5-mini | 1 | arc_easy | -3.0 | [-8.2, 2.3] |
| phi-3.5-mini | 2 | hellaswag | 2.1 | [-2.2, 6.3] |
| phi-3.5-mini | 2 | arc_easy | -0.5 | [-6.5, 5.5] |
| nemotron-mini-4b | 0 | hellaswag | 0.0 | [-4.5, 4.5] |
| nemotron-mini-4b | 0 | arc_easy | -4.5 | [-9.6, 0.8] |
| nemotron-mini-4b | 1 | hellaswag | 0.8 | [-3.3, 5.0] |
| nemotron-mini-4b | 1 | arc_easy | -0.5 | [-5.2, 4.2] |
| nemotron-mini-4b | 2 | hellaswag | -1.2 | [-4.1, 1.7] |
| nemotron-mini-4b | 2 | arc_easy | 1.5 | [-2.7, 5.6] |

Here pp means percentage points, not relative percent change. Each edit reuses
the same benchmark items; rows are not independent datasets. Global exposure,
other controls and per-window text results are in the detailed supplement.

## Mechanism Diagnostics

- llama-3.2-3b: held-out top8-minus-random faithfulness 0.7410; selected receiver compensation minus random 0.2753. This is a bounded diagnostic, not a complete circuit.
- phi-3.5-mini: held-out top8-minus-random faithfulness 0.5578; selected receiver compensation minus random -0.0327. This is a bounded diagnostic, not a complete circuit.
- nemotron-mini-4b: held-out top8-minus-random faithfulness 0.4467; selected receiver compensation minus random 0.0677. This is a bounded diagnostic, not a complete circuit.

## Missing Artifacts

None in the declared matrix. This does not mean every scientific hypothesis was supported.
