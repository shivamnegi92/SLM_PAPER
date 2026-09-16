# Execution Handoff

Updated 2026-09-07. The declared experiment matrix is complete. No Nemotron
benchmark or original collector job remains to be started.

The active checklist is [PENDING_PLAN.md](PENDING_PLAN.md). The frozen study
specification is [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md); the original
[PLAN.md](PLAN.md) is retained as a historical roadmap.

## Implemented and Verified

- Corrected scored-prefix capability exposure, baseline-aware intervention
  outcomes, paired sample identities and strict result compatibility checks.
- Shared-tokenizer manifests, numerical-rank checks, per-layer budgets,
  full-vocabulary objectives and development-only calibration.
- Simulator-backed container swaps and measured extraction baselines for the
  original tasks; held-out head and projected-subspace diagnostics.
- Public ARC-Easy input, multi-benchmark capability evaluation, durable case
  checkpoints, and a results collector that refuses incomplete matrices.

All 54 eligible method/seed runs, nine baseline-only harder-task runs, nine
capability runs and three head runs are complete. The
[post-run audit](results/postrun_audit_v1/consistency.json) also checks saved
identities and statistics. It does not rerun model forward passes or recover
missing historical provenance. The [log](validated_execution_20260906.log)
is execution history, not the current pending list.

## Current Handoff

1. Read [the manuscript](paper/MANUSCRIPT.md),
  [summary](paper/VALIDATED_RESULTS.md), and [detailed tables](paper/RESULTS_DETAILS.md).
2. Review the audit limitations: original weight content/source provenance is
  incomplete, baseline-only test files lack item IDs, and missing logits cannot
  be reconstructed from summary metrics.
3. Use fresh filenames when reproducing the CPU-only audit and tables. Do not
  restart the model queue or overwrite the original snapshot.
4. Finish submission formatting, anonymity checks, the required AI-use statement
  and author review. The official [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
  specify September 18/25, 2026, 23:59 AoE for abstract/paper respectively.

The full-space advantage is conditional on this protocol, not proof of universal
subspace insufficiency. All harder-task competence gates failed. General
capability preservation was not established. Keep those outcomes visible.

## Optional New Research

A reusable-edit pilot or a fresh development-only harder-task investigation
requires a separate protocol and untouched test data. No such experiment was
started as part of the post-run audit. Additional rank or Stage-B sweeps are
not prerequisites for the narrower paper.

Optional rank-law and Stage-B claims have been removed from the primary paper;
their legacy artifacts remain exploratory. No model downloads, commits, pushes,
external publishing or submission are performed by the queue.