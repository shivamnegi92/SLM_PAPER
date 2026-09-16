# Pending Work — Prioritized Plan

Status as of 2026-09-07. The declared experiment matrix and CPU-only
post-run consistency audit are complete. This plan distinguishes completed
measurements, unsupported claims, and future work; completion is not publication
readiness. Original experiment artifacts and the frozen protocol are unchanged.

**New revision phase, authorized 2026-09-07:** the
[reviewer-gap tracker](REVIEWER_GAP_TRACKER.md) records the expert review and
sequential work queue. G1 optimizer convergence is the first development-only
pilot; the remaining scientific gaps are not closed by the completed audit.
New protocols/results are separate from the frozen study described below.

---

## Claim ledger (what is actually established)

> **Current evidence:** all 54 eligible method/seed runs, nine harder-task
> baseline-only runs, nine capability runs and three head runs are present.
> The [post-run audit](results/postrun_audit_v1/consistency.json) rebuilt saved
> statistics and checked tokenized identities, budgets and checkpoints without
> loading model weights. See [the summary](paper/VALIDATED_RESULTS.md) and
> [detailed tables](paper/RESULTS_DETAILS.md). The legacy zero-loss capability
> result is invalid safety evidence. Corrected capability measurements are
> complete, but their preservation criteria were not all established.

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Tracking and matched-control continuous responses differ in the tested setting | New identity-paired contrasts and intervals are in the detailed tables; old 24-208x ratios remain historical | **NEW STATISTICS RECOMPUTED; not a universal ratio** |
| C2 | Direction matters within the tested ranks | Historical separation at five ranks | **EXPLORATORY; rank law retired from primary paper** |
| C3 | Active intervention preserves general capability | Nine selected edits: active accuracy bound met on HellaSwag 3/9 and ARC-Easy 0/9; text-window ratio bound met 9/9 | **MEASUREMENT COMPLETE; GENERAL PRESERVATION NOT ESTABLISHED** |
| C4 | Discrete and continuous metrics answer different questions | Historical ten-cell top-1 null is not the outcome of the new frozen study | **RETAIN AS MEASUREMENT MOTIVATION, NOT A NEW GENERAL LAW** |
| C5 | KL trust region reduces negative-edit disruption | Reported n=78, p=0.63 at the tested setting | **NO BENEFIT DEMONSTRATED; not proof of no effect** |
| C6 | Full-space editing outperforms the selected rank8 subspace under the frozen settings | All six eligible comparisons: full 100%, track8 8.7%-56%; full-minus-track8 44.0-91.3 percentage points | **AUDITED CONDITIONAL RESULT; architecture is not isolated** |
| C7 | Absolute norm budgets are not comparable without scale context | Historical residual-scale/budget observations motivated train-normalized per-layer budgets | **PROTOCOL MOTIVATION; not a cross-architecture law** |

**Completed runs are not supported hypotheses.** The new study does not reproduce
the old Phi-matches-full pilot story. Full-space advantages remain conditional
on the finite optimization budget, tasks, selected layers and checkpoints.
The audit verifies consistency of stored evidence, not the missing original
logits, optimizer convergence, historical weight content or complete mechanisms.

---

## P0 - Blocking Measurement and Evidence Checks

### P0.0 Measurement hardening *(IMPLEMENTED AND REGRESSION-TESTED)*
- [x] Reproduce the final-token scoring blind spot with a deterministic CPU test.
- [x] Edit the final context token for HellaSwag and a fixed prefix token for
  text, then score the continuation; clean up hooks on errors.
- [x] Store per-item/per-edit outcomes, Wilson accuracy intervals and approximate
  paired gain/loss bounds instead of averaging CI endpoints or declaring safety
  from overlapping intervals. Version outputs and refuse overwrites.
- [x] Pass 15 CPU regression tests, check 2,880 cached continuation boundaries
  across three local tokenizers, and complete a tiny offline Llama/MPS smoke run.
- [x] Separate same-sign collateral damage from legacy negative-edit BREAK,
  condition new breakage on baseline correctness, and record third-token errors.
- [x] Audit sample identities and paired inference in
  [src/analyze_dissociation.py](src/analyze_dissociation.py); validate unique
  examples, compatible configurations, missing-run errors and non-degenerate CIs.

See [CAPABILITY_DEPLOYED.md](CAPABILITY_DEPLOYED.md) for the corrected protocol
and verification commands. The two-item smoke run is integration evidence only.
The earlier measurement-hardening stage recorded 58 passing CPU tests plus a
separately passed MPS conversion regression. The post-run auditor adds checks
without modifying those frozen experiment sources. Empty eligible populations return undefined rates;
legacy array-only artifacts receive descriptive analysis, not inferred pairing.
The frozen [manifest](data/validated_manifest_v1.json) contains 150 unique test
pairs per task, independent of train/dev/few-shot prompts and other seed blocks.

### P0.1 Budget-matched control sweep   **DONE — see `BUDGET_CONTROL_RESULTS.md`**
The seed-0 pilot matches 16 optimizer steps and nominal relative budgets.
Llama retains a gap at 0.17/0.30/0.50; Phi has matching observed rates at
0.17/0.50. This narrows the original budget/step explanation. It does not prove
population equivalence or rule out remaining layer, optimizer-convergence,
tokenizer/example or checkpoint differences. Do not rerun the completed pilot
under the same output names.

### P0.1b Nemotron at matched budget  *(DONE)*
The observed full-space minus track8 gaps are +23.1/+30.8/+30.8 percentage
points at budgets 0.17/0.30/0.50. Preserve these as completed pilot results;
uncertainty and replication requirements are the same as for Llama/Phi.

### P0.2 Replicate under a frozen comparison protocol *(DECLARED PROTOCOL EXECUTED; SEE EVIDENCE)*
Completed seeds 0, 1 and 2 with shared semantic examples, matched layer depths,
per-layer relative budgets of 0.30, full/track8/comp8, and 32 optimizer steps.
Each eligible model/task/method has 150 test pairs across three seed blocks.
The audit recomputed all six paired summaries from 2,700 method/example records
and checked the saved case/configuration identities. Repeated methods/models
do not create additional independent semantic examples.

The executable protocol is [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md), with
case-level resume in [src/run_study_cases.py](src/run_study_cases.py).
Seed-0 development data provided the competence gate; four of those development
examples selected learning rates from the same grid for each method. The
[sequential queue](run_validated_all.sh) is execution history, not the next
command to run. Neither calibration nor this audit proves solver convergence.

### P0.3 Scope the claim to the validated results *(CURRENT CLAIMS RECONCILED)*
The manuscript and current tables use the new protocol. Historical
[dissociation](DISSOCIATION_RESULTS.md) and [budget](BUDGET_CONTROL_RESULTS.md)
observations remain separate from the new comparison. Do not claim universal
prevention, equivalence from `p=1.0`, or architecture as the isolated cause. STEER is
target-informed, per-example counterfactual override, not demonstrated repair
of naturally wrong answers.

### P0.4 Self-repair / backup-head validity check *(DECLARED PROTOCOL EXECUTED; SEE GATES)*
[src/validate_heads.py](src/validate_heads.py) discovers heads on training
examples and verifies top8 patching against five depth-matched random sets on
24 held-out examples per seed. A selected source-head ablation is combined with
baseline clamps of downstream receivers and matched random receivers. Baseline
clamps must be a no-op; unit tests verify slicing, cleanup and activation-only
gradients. This is a bounded compensation diagnostic, not an exhaustive backup
search or a full edge-level circuit.

The same runner compares full residual replacement with projected replacement
inside the learned tracking basis and its matched control. Identity/zero
projection tests pass. All three model artifacts now exist; their paired head
statistics and baseline no-op checks were audited. Confidence intervals are
reported in [the detailed tables](paper/RESULTS_DETAILS.md). The head artifacts
lack run-time source/weight content hashes, and no complete mechanism is claimed.

### P0.5 Corrected capability benchmarks *(DECLARED PROTOCOL EXECUTED; SEE GATES)*
The completed [capability study](src/run_capability_study.py) fixes 240 cached
HellaSwag items, 200 public ARC-Easy test items, five non-overlapping text
windows and three preselected seed edits. ARC provenance and content hash are
in [the cache](data_bench/arc_easy_test_200.json); all 799 ARC continuations
pass context-prefix checks under each of the three tokenizers. Edits and
benchmark blocks checkpoint independently. Baseline, active-prefix, equal-norm
random, zero and global exposure are now reported with paired uncertainty.

All nine final capability files and their intermediate checkpoints passed the
post-run consistency audit. The active-prefix 2-percentage-point accuracy bound
was met for three HellaSwag edits and no ARC edits; the 10% text-window ratio
bound was met for all nine edits. These counts reuse benchmark items and are
not independent datasets. A failed bound is not proof of damage in every case;
neither overlapping intervals nor successful zero controls establish safety.

---

## P1 - Task Validity and Generalization

### P1.1 A simulator-backed compositional task *(IMPLEMENTED AND VERIFIED)*
The existing tasks' query-aware extraction baseline and the simulator-backed
container-swap task were implemented before evaluation. Changing an initial
location changes the final answer through several operations. Simulator checks
and held-out wording constrain specific shortcuts; this remains synthetic,
not evidence of naturalistic generalization.

The frozen test set has 150 unique container-swap pairs. The declared
copy/last-mention rules score 0%; query-aware extraction scores 100% on both
legacy task sets. Test wording is held out from training/dev/demonstrations.
Simulator composition, inversion, target consistency and split integrity tests
pass. These checks rule out those specific shortcuts, not every heuristic.

### P1.2 Cross-arch on the harder task *(EXECUTED WITH COMPETENCE GATE)*
All three checkpoints failed the declared seed-0 development competence gate.
The nine baseline-only test runs are complete; no harder-task causal-control
claim follows. The audit recomputed rates from saved flags, but those files
lack per-item identities. Any future task-format investigation must use fresh
development data and a new protocol, preserving this negative result.

---

## P2 — Strengthens, doesn't block

### P2.1 Two-stage Stage-B claim *(RETIRED FROM PRIMARY PAPER)*
The single-seed norm-shrink observation is retained as historical exploration,
not promoted as a validated contribution. Stage B now has access to the
full-vocabulary margin, but new Stage-B efficacy experiments were not run.
Replication is unnecessary for a claim that the primary paper no longer makes.

### P2.2 Rank power-law claim *(RETIRED FROM PRIMARY PAPER)*
Do not extend the legacy sweep merely to reinforce a power-law narrative from
five ranks. The updated rank analyzer reports the available files descriptively
without unverified p-values; a two-seed audit of all five saved ranks passes.
The rank8 matched control remains part of the primary frozen comparison.

### P2.3 Capability on Phi + Nemotron *(COMPLETED; SEE P0.5)*
Both models have all three corrected capability outputs. No extra run is
required to close the declared matrix, and C3 remains unsupported as a general claim.

### P2.4 Same-sign damage at matched STEER
The single declared budget point has audited same-sign damage estimates.
A matched-success tradeoff curve has not been measured and is optional new
work, not a completed frontier. Legacy negative-edit BREAK remains separate.

---

## P3 — Paper mechanics

- **P3.1 DONE:** corrected summary and detailed control/text tables are linked
  from the paper. Legacy capability observations keep the scoring warning.
- **P3.2:** retain the survey and metric-choice context and cite the verified
  subspace-illusion paper. Do not present the general localization/control
  distinction as a newly discovered principle or infer novelty from a survey.
- **P3.3 DONE:** the existing [figure](results/final_validated_evidence_v1/validated_control.png)
  shows override versus same-sign damage at the declared operating point with
  uncertainty bars. It is not a matched-success frontier or a plot of collateral BREAK.
- **P3.4:** the [manuscript](paper/MANUSCRIPT.md) contains the audited results;
  [detailed tables](paper/RESULTS_DETAILS.md) include per-seed rates, denominators,
  paired contrasts, all capability controls, text windows and head intervals.
- **P3.5** Fix stale cross-model references. `PLAN.md` and `README.md` still say
  "Llama vs Gemma-2-2B"; `RESULTS.md:5` admits Gemma is blocked (gated HF
  download) and never happened. What actually ran — Phi-3.5-mini +
  Nemotron-Mini-4B, 3 families total — is stronger than the original plan, but
  the docs don't say so. 5-minute fix, prevents a reader chasing nonexistent
  Gemma results. **DONE:** README names the actual three models; historical
  roadmap/results banners distinguish unexecuted Gemma plans from measurements.
- **P3.6:** ICLR 2027 remains the working target. Official author guidance was
  checked on 2026-09-07: abstract September 18 and paper September 25, both
  23:59 Anywhere on Earth; initial main text at most nine pages. See the
  [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
  Formatting, anonymous submission materials, an accurate AI-use statement,
  author approval and the submission itself are separate remaining steps.

---

## Suggested execution order

```text
Completed: frozen experiment matrix, original snapshot, figure and results
Completed: post-run consistency audit and detailed reporting
Recorded:  reproduction procedure and verified submission requirements
Remaining: submission formatting, anonymous release copy and author review
Optional:  a separately declared reusable-edit or harder-task pilot
```

**Execution:** do not restart the completed [queue](run_validated_all.sh).
Use [the read-only auditor](src/audit_study.py) and
[detailed reporter](src/report_study.py) to reproduce post-run checks and tables
with fresh output names. The frozen experiment source and original snapshots
must remain unchanged; new interventions require a new declared protocol.

**The priority:** valid measurements and reproducible contrasts, whether or not
new evidence preserves the current model split. Implementation completion is
separate from experiment completion, and an unsupported hypothesis is not
silently converted into a successful claim.

## Completed Evidence Snapshot

The declared matrix passed the missing-artifact gate and the post-run consistency
checks. See [the summary](paper/VALIDATED_RESULTS.md),
[detailed evidence](paper/RESULTS_DETAILS.md), and
[audit scope](results/postrun_audit_v1/consistency.json). Historical missing
runtime provenance and logits cannot be recovered by a successful audit.
No external submission was made.

The [reproduction guide](docs/REPRODUCIBILITY.md) distinguishes current-file
checksums from historical provenance. The [submission checklist](docs/SUBMISSION_CHECKLIST.md)
records official requirements and the author tasks not completed by this audit.
