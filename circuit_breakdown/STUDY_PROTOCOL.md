# Validated Local Study Protocol

Frozen on 2026-09-06 before the new test-set evaluations. This study replaces
the old inference protocol; historical files remain unchanged. Outcome gates
permit negative findings and do not require the old model-split story to hold.

## Primary Comparison

- Existing local Llama-3.2-3B, Phi-3.5-mini and Nemotron-Mini-4B only; fp32 on
  MPS with one model process at a time. Checkpoint configuration fingerprints
  and weight-file sizes are recorded, not asserted to prove a remote revision.
- Three seeds, 64 train / 24 dev / 50 untouched test pairs per task and seed,
  plus three separate few-shot examples. No prompt reused across splits/seeds.
- Shared tokenizer-compatible manifest: intermediate, transfer, container_swap.
  Container-swap test wording is held out from training/dev/demonstrations.
- Train-only difference bases: rank8 and a dimension-matched random slice of
  its orthogonal complement, plus full-space editing. Numerical rank and
  orthogonality checks are mandatory.
- Per-layer budgets: 0.30 times mean train residual norm at the edited position.
  Match output-depth fractions: Llama [18,20,22,24], 32-layer models [21,23,25,28].
- Target cross-entropy over the full vocabulary, Adam, 32 steps. Equal
  development tuning allowance per method: base learning rates 0.025/0.05/0.1,
  four seed-0 dev examples, highest mean target-vs-best-other margin. Rank-based
  step scaling remains an approximation, not a convergence theorem; traces
  expose the 16-to-32-step behavior. Test outcomes do not select settings.
- Require at least 80% clean and counterfactual development accuracy before
  interpreting intervention outcomes as a tracking mechanism. Models/tasks
  failing this gate receive baseline-only test evaluation and a limitation.

## Metrics and Inference

Report target override, newly achieved override, full-vocabulary target margin,
continuous two-way probability change, third-token predictions, same-sign
damage among initially-correct clean examples, and negative-edit disruption
separately. Optimizer-visible control agreement is not held-out accuracy.

Match method results by stable semantic IDs and token-input hashes. Reject
duplicates, absent seeds and protocol differences. Binary contrasts use paired
gain/loss Wilson bounds and exact McNemar tests; continuous contrasts use
seed-stratified paired bootstrap intervals and sign-permutation p-values with
the Monte Carlo plus-one correction. Report per-seed outcomes. Inference is
conditional on these seeds, not an assertion of population equivalence.

## Mechanism and Capability

Discover mover heads on training prompts and verify on held-out prompts with
depth-matched random heads. Test a selected source-head ablation and clamp
candidate downstream receivers to their unablated baseline to measure a
possible compensation effect; do not call it a complete edge-level circuit.

Capability uses active prefix positions, not the final unscored sequence token.
Use cached HellaSwag and a public ARC-Easy subset if retrievable, multiple
preselected nonoverlapping text windows, and three edits from separate seeds.
Record per-item/per-edit pairs and zero/random/global controls. No safety claim
without a paired lower accuracy bound above -0.02 and a supported perplexity
ratio upper bound below 1.10. Otherwise report uncertainty or measured damage.

## Scope Decisions

The old rank power-law and Stage-B efficiency claims are removed from the
primary paper until independently justified. Their old results remain
exploratory; optional rank top-ups and Stage-B tuning are not prerequisites for
the narrower validated claim. No new models, cloud compute, external publishing
or git commits are part of this protocol. ICLR is the working venue target;
format and submission deadline must be verified separately before submission.