# Reviewer Gaps and Revision Tracker

Updated 2026-09-08. User authorized documenting the review and starting work
sequentially. This is a new revision phase, not an unfinished portion of the
completed frozen study. Original experiments, source hashes and snapshots stay
unchanged. New measurements use separate protocol/output names.

## Established Evidence

- Completed 54 main method/seed runs, nine harder-task baseline-only runs,
  nine capability runs and three head runs, plus a post-run consistency audit.
- On the two eligible synthetic answer-extraction tasks, full-space target
  override was 100% in all six model/task comparisons; selected rank8 override
  was 8.7%-56%, and the matched complement control achieved 0%.
- Each eligible task has 150 semantic test pairs reused across methods/models.
  The 2,700 method/example evaluations are not independent semantic examples.
- Selected heads outperform depth-matched random head sets on held-out data.
  This is component-level evidence, not a complete circuit.
- Capability measurements are complete but general preservation is unsupported.
  All harder-task development competence gates failed.

Sources: [current summary](paper/VALIDATED_RESULTS.md),
[detailed tables](paper/RESULTS_DETAILS.md), and
[post-run audit](results/postrun_audit_v1/consistency.json).

**Defensible current claim:** under the frozen finite-budget procedure,
target-informed full-space edits outperform the selected rank8
activation-difference subspace on the two tested answer-extraction tasks.
This is not proof of universal insufficiency, reusable editing, error repair,
general reasoning control, safety, or architecture as the cause.

## Work Queue

| ID | Priority | Gap | Status | Next deliverable |
|---|---|---|---|---|
| G1 | High | Predictive tracking subspace versus causal control subspace is not isolated | THREE PILOTS COMPLETE, FOURTH RUNNING; GAP OPEN | v1 (Phi) done: random never succeeds, PCA tracks tracking, projection of v_full fails. v2 (Llama) running with convergence stopping, learned-causal basis and degeneracy diagnostics |
| G2 | High | Harder-task generalization and stable baseline competence | TWO PILOTS COMPLETE; GAP OPEN | Diagnose weak no-swap prediction without assuming context alone explains it; no multistep competence established |
| G3 | High for practical control | Target specificity and reusable edits | QUEUED | Different-answer hard negatives; reusable goal only with an explicit common intervention objective |
| G4 | High for preservation claims | Capability sample size and edit coverage | QUEUED | Paired precision/power plan, then distinct edits and matched-efficacy controls |
| G5 | Medium | Component localization is not a mechanism explanation | QUEUED | Analysis of saved projected replacements, followed by scoped necessity/sufficiency controls |
| G6 | High for publication | Novelty and stronger comparison methods | QUEUED | Verify sparse steering/causal probing literature and complete the predictive-vs-causal comparison matrix |
| G7 | Medium | Historical provenance and independent replication | PARTIAL: all four diagnostic pilots have pre-run provenance | Limited independent replay; historical omissions remain disclosed |

Implementation, pilot execution, and scientific resolution are separate statuses.
A negative or ambiguous pilot is a valid result. A gap is not closed because a
script exists, an output file is present, or the preferred result appears once.

## G1 - Predictive Versus Causal Subspace

**Direction update, 2026-09-09.** The next paper should not claim that
low-dimensional causal control is impossible. Emerging sparse steering and
causal-mediation work makes that broad claim fragile. The sharper hypothesis is
that a compact subspace can make a reasoning state readable without being the
same compact subspace through which the model can be causally controlled:

```text
Predictive subspace != causal subspace
```

This shifts the goal from "rank8 failed" to an equal-dimensional comparison of
tracking, PCA, random and causally selected subspaces. The new planning document
is [predictive-vs-causal subspace plan](PREDICTIVE_VS_CAUSAL_SUBSPACE_PLAN.md),
and the next protocol skeleton is
[predictive_vs_causal_subspace_v1](protocols/predictive_vs_causal_subspace_v1.json).
The first implementation adds reusable subspace utilities,
[the equal-k runner](src/run_predictive_vs_causal_subspace.py), and focused
unit tests. Its causal arm uses gradient-ranked coordinate bases as a cheap
first-order causal selector; learned sparse masks and finite-difference sweeps
remain later variants.

### v1 Pilot Outcome (Phi, transfer, eight development examples)

Completed 2026-09-09. Development-only; not confirmatory. Raw override counts:

| rank | full | tracking | pca | random | gradient-coordinate |
|---|---:|---:|---:|---:|---:|
| 1, 2, 4 | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| 8 | 8/8 | 2/8 | 2/8 | 0/8 | 0/8 |
| 16 | 8/8 | 5/8 | 5/8 | 0/8 | 0/8 |
| 32 | 8/8 | 6/8 | 3/8 | 0/8 | 1/8 |
| 64 | 8/8 | 3/8 | 3/8 | 0/8 | 1/8 |

Three observations, at descending confidence:

1. **Strong.** Random directions never succeeded at any rank up to 64 while
   structured bases eventually did. Arbitrary dimensional capacity alone does
   not explain control.
2. **Suggestive, n=8.** Tracking and PCA behave similarly. PCA does not use the
   clean-versus-corrupt contrast at all, so the claim that tracking isolated a
   special reasoning representation is now under pressure. This is not yet an
   equivalence claim.
3. **Mechanistically interesting.** The successful full-space edit has only
   0.2 to 7.5 percent of its energy inside the tracking subspace, and projecting
   that known-successful vector into tracking produced zero successes at every
   rank.

**Interpretation constraint carried forward.** Observations 2 and 3 together
forbid the conclusion "causal control lies outside the tracking subspace." Two
different questions were conflated:

```text
Subspace CONTAINS the known full-space solution   (projection test)
Subspace PERMITS a different solution to be found (fresh-optimization test)
```

At rank 32, projection of v_full into tracking gave 0/8 while independent
optimization inside tracking gave 6/8. If both hold, there may be multiple
geometrically distinct activation edits producing the same behavioral outcome.
That raises an identifiability question -- are behavioral interventions unique?
-- which may matter more than the original dimensionality hypothesis.

**Optimizer warning.** Tracking success ran 2/8, 5/8, 6/8, then fell to 3/8 at
rank 64. More capacity should not reduce success under a fair comparison. Do
not interpret the rank-64 cell scientifically; it is evidence that the fixed
32-step budget confounds capacity with ease of optimization.

### v2 Pilot Design (Llama, in progress)

Launched 2026-09-09 against
[protocols/predictive_vs_causal_subspace_v2.json](protocols/predictive_vs_causal_subspace_v2.json)
via [the v2 runner](src/run_predictive_vs_causal_subspace_v2.py). Llama, not
Nemotron, because Phi has previously behaved differently from the other two and
a Llama replication tests whether v1's signatures are a phenomenon or a Phi
artifact. Changes:

1. **Convergence-based stopping** replaces the fixed 32 steps. Optimization
   halts when the loss changes by less than 1e-4 for five consecutive steps, or
   at 80 steps. Ranks are compared at equal effort, not equal step count. This
   addresses the rank-64 warning above.
2. **Ranks 8, 16, 32, 64 only.** Ranks 1, 2 and 4 were uninformative in v1.
3. **"causal" renamed to "gradient_coordinate".** Raw dimensions ranked by
   gradient sensitivity test whether INDIVIDUAL coordinates are locally
   sensitive. A distributed causal direction needs no individually sensitive
   coordinate, so this baseline is too weak to carry the name "causal" and its
   v1 near-null must not be read as evidence against sparse causal control.
4. **Added "learned_causal"**, a DAS-lite basis: a shared per-layer orthonormal
   rotation optimized directly against the behavioral objective,
   `U* = argmin_{U,theta} L(h + U theta)` subject to `U^T U = I`, with a QR
   retraction after each step. This is the strong causal comparator that v1
   lacked.
5. **Both experiments retained for every basis** -- projection of the known
   v_full, and fresh optimization within the basis -- so the two questions above
   are never conflated again.
6. **Degeneracy diagnostics** on every example where full and fresh-in-basis
   both succeed: `cos(v_full, v_basis)`, `||v_basis|| / ||v_full||`, and
   `KL(p(y|h+v_full) || p(y|h+v_basis))`. Low cosine with low KL would
   demonstrate intervention degeneracy directly.

Pre-declared outcome readings, recorded before results were seen:

| Outcome | Reading |
|---|---|
| tracking ~ pca << learned_causal | Predictive and variance-aligned subspaces poorly approximate intervention-optimized ones |
| tracking ~ pca ~ learned_causal | Nothing special about the clean-versus-corrupt representation; high-variance geometry dominates |
| tracking > pca > learned_causal | The causal optimization is weak; draw no mechanistic conclusion |
| full >> everything, at converged budget | The causal-dimensionality hypothesis strengthens, but sparse-steering literature demands stronger evidence |

Sequencing: Nemotron runs only if Llama reproduces `pca ~ tracking >> random`.
Development sample grows beyond eight only after that, for confirmation.

### v2 Pilot Outcome (Llama, complete)

Completed 2026-09-09. Fresh-optimization raw counts:

| rank | full | tracking | pca | random | gradient-coordinate | learned_causal |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 8/8 | 1/8 | 0/8 | 0/8 | 0/8 | 2/8 |
| 16 | 8/8 | 4/8 | 6/8 | 0/8 | 2/8 | 5/8 |
| 32 | 8/8 | 5/8 | 6/8 | 0/8 | 5/8 | 6/8 |
| 64 | 8/8 | 6/8 | 4/8 | 0/8 | 8/8 | 5/8 |

Projection of v_full is 0/8 at every one of the 20 basis-by-rank cells, with
no exception -- a cleaner replication than v1's occasional nonzero projection
success. Random stays 0/8 at every rank in a second architecture. Ranking
stability holds: `full >> structured >> random`, `pca ~ tracking` through
rank 32. `learned_causal` never separates cleanly from tracking/PCA at any
rank -- leaning toward "nothing special about the clean-vs-corrupt contrast,
high-variance geometry dominates" rather than "a properly optimized causal
subspace closes the gap." Honest surprise not to be buried: `gradient_coordinate`
reaches 8/8 at rank 64, beating every other basis including learned_causal;
the rename to a weaker-comparator framing does not fully survive this cell.

Degeneracy diagnostics on double-success cases: cosine(v_full, v_basis) stays
flat around 0.08-0.15 REGARDLESS OF RANK (e.g. tracking: 0.090 at k=16, 0.083
at k=32, 0.086 at k=64) while KL(full || basis) generally falls with rank
(tracking: 0.22 -> 0.08 -> 0.18). Geometric divergence with behavioral
convergence -- multiple very different-looking edits produce similar output
distributions. This is the empirical basis for treating intervention
non-uniqueness as a real phenomenon, not a projection artifact.

### Causal Interchange Pilot v1 (Llama, rank 32, in progress)

Launched 2026-09-09 against
[causal_interchange_v1.json](protocols/causal_interchange_v1.json) via
[the interchange runner](src/run_causal_interchange.py), using the new
[interchange dataset generator](src/interchange_dataset.py) and
[interchange mechanism](src/causal_interchange.py). Motivation: the steering
result can be dismissed as "the optimizer was told the answer." An interchange
intervention is determined entirely by a DONOR example's activations, with no
objective that knows the receiver's target -- a substantially harder test of
whether a subspace carries a reusable causal state rather than merely
supporting an optimizer that finds some way to the receiver's own answer.

Design: donor/receiver transfer-task pairs with fully disjoint names and
objects (so a correctly patched receiver emits a token absent from its own
context -- unambiguous), plus a same-answer/different-reasoning-path control
(donor and receiver reach the same holder via different chains, so there is
no answer to flip and the informative quantity is raw-state cosine
similarity). v = P_S(h_donor - h_receiver) reuses the existing additive hook
mechanism exactly; S = full space is an exact replacement (the oracle).

**Two real bugs caught by smoke-testing before the full run, both fixed:**

1. Interchange items were sampled from the raw PEOPLE/OBJECTS pools without
   single-Llama-token restriction, and `from dataset import PEOPLE` bound a
   stale reference that `dataset.restrict_to_single_token`'s mutation
   couldn't reach. Fixed to `import dataset` + `dataset.PEOPLE` at call
   sites, plus a tokenizer-verified generation filter (`generate(...,
   tokenizer=...)`) that validates the ACTUAL rendered prompt rather than a
   generic proxy context.
2. No few-shot prefix was prepended, so the base model could not solve the
   task zero-shot (baseline receiver accuracy 0/8). Fixed by threading the
   same manifest few-shot prefix used everywhere else in this project.

**One genuine methodological finding from smoke-testing, not a tuned
parameter:** patching at the mid-sentence recipient-name position gave oracle
(full-space) interchange 0/8 even with a healthy baseline. Patching at the
FINAL (pre-answer) token position, same 4-layer set, gave oracle 6/8. This was
fixed BEFORE generating the 32 held-out items now being scored, and is
recorded in the protocol's `position_note` for auditability.

Early signal (n=4 smoke, not to be trusted quantitatively): oracle 4/4,
random-donor 3/4, and EVERY basis-restricted interchange (tracking, pca,
random, gradient_coordinate, learned_causal) at 0/4 -- consistent with the
steering pilot's projection failure at every rank. The real n=32 run is in
progress.

### Interchange Audit: a measurement error of mine, found and corrected

The completed n=32 run produced correct-donor IIA 24/32 (75%) and
random-donor IIA 23/32 (72%), which I initially read as "no donor
specificity -- the protocol is hijacking the tail computation
 indiscriminately." **That reading was wrong, and the error was in my
control, not in the protocol.**

What I audited first, and ruled out:
- **Label collision.** The original derangement enforced
  `random_donor_answer != receiver_answer` but NOT
  `random_donor_answer != correct_donor_answer`; with a 24-name pool and 32
  items, 2/32 items collided. Fixed with a collision-free derangement
  ([audit_interchange.py](src/audit_interchange.py)). Result essentially
  unchanged (23/32). Not the driver.
- **Broken patch mechanism.** A true no-op (self-patch, zero delta) scored
  32/32 identical-to-baseline. Mechanism is sound.
- **Baseline conditioning.** Conditioned on the receiver already being
  correct (n=24): correct 17/24, random 19/24. Still no apparent gap.

**The actual error:** comparing `patch D -> P(Y_D)` against
`patch R -> P(Y_R)` is not a specificity test. Both are legitimate
interchange interventions with different donors, and the high-level causal
model predicts BOTH should succeed. Two separate forward passes each hitting
their own target is evidence the intervention WORKS, not that it is
indiscriminate. A "random" donor is not a control that ought to fail.

Donor specificity has to be measured as a CROSS-TERM within a SINGLE patched
pass ([test_cross_term_specificity.py](src/test_cross_term_specificity.py)),
n=16, position -1, layers [18,20,22,24]:

| quantity | baseline | after do(D) |
|---|---:|---:|
| P(Y_donor), the injected answer | 0.000 | **0.718** |
| P(Y_other), a donor NOT injected | 0.066 | **0.000** |
| P(Y_receiver), receiver's own answer | 0.604 | **0.000** |

`DS_cross = +0.718`. Top-1 is the injected donor 13/16; top-1 is a
non-injected donor 0/16. The intervention installs the donor's specific
answer, fully suppresses the receiver's own tracked state, and does not leak
toward arbitrary other names.

**Magnitude sweep** ([sweep_interchange_magnitude.py](src/sweep_interchange_magnitude.py)),
alpha in {0, .125, .25, .5, .75, 1}, n=16: a clean dose-response
(delta_donor = .005 -> .201 -> .601 -> .653 -> .718) with a threshold near
alpha~0.25. My earlier "binary hijack" reading from the position sweep was
also wrong -- the effect is smoothly graded in magnitude.

**Norm-matched random direction control** (a Gaussian direction rescaled to
EXACTLY the donor delta's combined norm): **+0.000 at every alpha.** Same
magnitude, no semantic direction, zero effect. This rules out generic
perturbation-magnitude disruption decisively -- the effect requires a real
model-state direction and tracks precisely which one.

### Open question the validated protocol does NOT yet settle

Specificity is established, but "transports the REASONING STATE" is not yet
distinguished from "transports an already-compressed ANSWER TOKEN." The tell
is the position sweep ([sweep_interchange_position.py](src/sweep_interchange_position.py)),
n=12, layers [18,20,22,24]: position -1 gives correct 9/12, but positions
-2 through -7 give **0/12 for both correct and random donors**. If layers
18-24 carried a distributed "current holder" variable, patching one token
earlier should do something. That it does nothing is consistent with the
final-token residual having already collapsed the computation into a
near-linearly-decodable answer encoding.

This distinction decides what the compact-basis nulls mean:
- reasoning-state transport => "compact predictive subspaces cannot carry
  the causal state" is a strong claim.
- answer-token transport => the nulls only say compact bases cannot carry a
  late answer encoding, which is much weaker.

Decisive next test: **secondary probes.** Transport the donor state, then ask
a DIFFERENT question of the same patched pass ("who ORIGINALLY had the
object?", "how many transfers?"). If the donor's answers to those also
transport, it is a genuine reasoning state; if only "current holder" moves,
it is answer compression. Not yet run.

### Earlier G1 Evidence - Convergence, Basis and Capacity

**Evidence.** The original optimizer used 32 Adam steps and selected a base
learning rate from three values using four seed-0 development examples.
Phi's selected rank8 mean target margin increased by approximately 2.445
(intermediate) and 2.638 (transfer) from step 16 to step 32, at the grid's
largest base rate, 0.1. A descriptive inspection of original test traces found
10/150 Phi transfer examples reached the target before failing at the final
step. Those observations motivate new development work; they do not redefine
the original success metric or show that optimization explains the whole gap.

Sources: [optimizer](src/intervene_pareto.py),
[calibration](src/run_validated_study.py),
[Phi transfer calibration](results/validated_v1/phi-3.5-mini/transfer/calibration.json).

**First pilot.** Use Phi transfer, the original seed-1 training basis and the
first eight seed-1 development pairs. Full and track8 share the same data,
layers, train-normalized 0.30 budgets, objective, guard and 128-step trajectory
length; base rates remain the original seed-0 development selections. Record
post-update checkpoints at 32, 64 and 128, final and best-seen success, full
logits, vectors, per-layer norms, gradient norms and guard interventions.
Do not evaluate test examples or automatically choose a final paper setting.
The unchanged 32-step optimizer is the equivalence reference on a CPU fixture.

**Then.** Based on that diagnosis, predeclare development comparisons with a
second optimizer, restarts, and a bounded learning-rate schedule. Add ordinary
random rank8 and a train-learned alternative rank8 basis to separate basis
choice from dimensional capacity. Use equal tuning allowance, report compute
cost, and confirm a selected protocol on untouched test data.

**Next declared diagnostic.** The
[causal dimensionality protocol](protocols/causal_dimensionality_pilot_v1.json)
uses the saved successful full-space vectors from the completed convergence
pilot and decomposes them into tracking-subspace and orthogonal components. The
[runner](src/run_causal_dimensionality_pilot.py) measures baseline, full,
tracking projection, tracking orthogonal, random projection, coordinate top-k
and permuted-vector controls over ranks 1, 2, 4, 8, 16, 32 and 64. This directly
tests whether the selected tracking SVD subspace captures the causal direction's
energy and behavioral effect. It does not optimize a new low-rank behavior
subspace and therefore cannot prove low-rank causal control is impossible.

**Resolution gate.** A qualified conclusion about this selected subspace needs
stability under credible optimization controls, an equal-k predictive-vs-causal
comparison, and held-out confirmation. If a causally selected low-rank subspace
recovers dense control, revise the paper toward causal subspace discovery rather
than low-rank failure. If no sparse subspace recovers dense control until large
k, frame the result as a measured causal dimensionality curve, not an
impossibility theorem.

## G2 - Task Validity and Competence

**Evidence.** Query-aware extraction solves the two original tasks. Every
container-swap test block scores 0%; development ends with `is now in box`,
while held-out wording ends with `is`. Canonical appended-answer boundaries
passed 1,332 tokenizer checks across three models, so that specific mismatch
was not found. Response formatting versus reasoning failure remains unresolved
because the original baseline-only artifacts did not store predicted text.
Llama transfer clean/counterfactual test accuracies across seed blocks were
88/90%, 82/82%, and 50/60%, despite its seed-0 development gate passing.

Sources: [task generator](src/dataset.py),
[baseline scorer](src/run_validated_study.py), and
[per-seed evidence](paper/RESULTS_DETAILS.md).

**Work.** Generate fresh development-only simulator examples. Cross wording
and chain length while holding answer cues constant; include zero/one-operation
sanity cases. Save generated responses, top-token choices and canonical answer
scores. Separately report formatting errors and wrong locations. Then assess
the original 80% clean/counterfactual competence criterion across development
blocks before selecting any new held-out protocol.

**Resolution gate.** A genuine non-copy task with reliable measured competence
and untouched test examples. Preserve old failing seeds/tasks; never lower the
gate, remove hard cases, or change the old test wording after seeing its result.

## G3 - Specificity and Reuse

**Evidence.** Each target-informed edit is optimized separately. The clean
same-sign input wants the same answer that the edit promotes, so zero damage
there cannot demonstrate entity specificity or protect different answers.

Source: [per-example intervention evaluation](src/intervene_pareto.py).

**Work.** Add same-context different-entity questions, same-entity changed-state
questions and unrelated different-answer controls. Include an explicit
target-token-promotion diagnostic with its extra output access stated. Evaluate
each edit on multiple held-out hard negatives. For reuse, define one common
goal, fit once on training examples, and apply without target-label access or
per-example optimization at evaluation.

**Resolution gate.** Report efficacy and wrong-answer promotion jointly with
paired uncertainty. A reusable editor requires unseen-input success without
refitting. Otherwise retain the narrower per-example override claim.

## G4 - Capability Precision and Coverage

**Evidence.** The nine capability runs use three selected full-space edits per
model from the intermediate task, on reused benchmark items and five windows
from one text. Under the current conservative paired bound, unchanged outcomes
give approximately +/-2.4504 pp at n=200 and +/-2.0504 pp at n=240. This explains
why even the zero control cannot certify a strict 2 pp bound at those sizes;
it does not establish damage. ARC bounds are unsupported for all active edits.

Sources: [paired interval](src/experiment_metrics.py),
[capability sampling](src/run_capability_study.py), and
[control/text tables](paper/RESULTS_DETAILS.md).

**Work.** Use paired gain/loss frequencies to predeclare sample sizes, number
of independent edits and a fixed analysis/stopping rule. Include multiple text
sources and compare methods at matched steering efficacy, with zero/random
controls. Plan uncertainty over both edits and inputs; more repeated rows do
not make the same input examples independent.

**Resolution gate.** Report the declared bounds with adequate precision under
a clearly specified deployment/exposure policy, or explicitly retain an
unsupported-preservation conclusion. Do not keep sampling until the bound passes.

## G5 - Mechanism Explanation

**Evidence.** Held-out top8-minus-random faithfulness intervals are positive
for all models. Selected-receiver compensation differs in sign/magnitude:
Llama 0.2753 [0.2316, 0.3203], Phi -0.0327 [-0.0550, -0.0107], Nemotron
0.0677 [0.0555, 0.0799]. These are logit-margin diagnostics, not accuracies.
Natural projected replacement and budgeted additive edits use different interventions.

Source: [head diagnostics](src/validate_heads.py), [detailed results](paper/RESULTS_DETAILS.md).

**Work.** First summarize the saved full/tracking/control projected-replacement
results with explicit denominators and uncertainty. For a broader mechanism
claim, match sites, token positions and budgets; test necessity/sufficiency and
discovery stability across training samples. A selected ablation/clamp is not
an exhaustive backup-circuit search.

**Resolution gate.** Either support the specified causal mechanism with matched
interventions or keep the component-localization interpretation.

## G6 - Novelty and Comparisons

**Evidence.** [Makelov, Lange and Nanda (2023)](https://arxiv.org/abs/2311.17030)
already distinguish changing behavior from identifying a faithful mechanism,
and include both counterexamples and a success case. The general distinction
cannot be claimed as new. Three checkpoints alone do not establish novelty.

**Work.** Build a primary-source comparison matrix: prior claim, method,
evaluation, limitations, and our actual difference. Select a stronger
established control/baseline compatible with this task and access setting.
Tie the contribution to the explanation supported after G1/G2, not to a
survey's list of open problems or experiment volume.

**Direction update, 2026-09-09.** The comparison matrix must now cover CAA,
ActAdd, representation engineering, DAS, LEACE, causal probing, sparse
autoencoders, sparse causal mediation/GCM, sparse attention-head steering,
CAS/BiPO-style preference steering and control-utility/KL-trust-region work.
Recent claims about 2026 sparse steering effectiveness must be verified from
primary sources before being cited or used to constrain the paper claim.

**Resolution gate.** A precise supported contribution and relevant comparisons.
No acceptance probability or novelty clearance is inferred from this tracker.

## G7 - Provenance and Replication

**Evidence.** Original fingerprints used configuration and root `.safetensors`
sizes; Nemotron's `.bin` weights were outside that weight inventory. Later
content hashes cannot prove historical run-time bytes. Head artifacts lack
run-time source/weight hashes, baseline-only results lack item IDs, and full
logits were not retained for all original metrics.

Source: [reproduction limitations](docs/REPRODUCIBILITY.md).

**Work.** For every new run, verify active model-file content, hash the protocol
and executed source before model loading, and save environment, input IDs,
predictions and relevant logits. Resume only an exactly matching protocol.
A limited separately executed confirmation can strengthen reproducibility;
do not relabel the original data as retroactively authenticated.

**Resolution gate.** New measurements have pre-execution provenance and a
repeatable verification command. Historical omissions remain disclosed.

## Current Execution Log

### Second Diagnostic Wave

- Authorized after the first G1/G2 pilots. New protocols and outputs remain
  separate from all completed experiments.
- [Guard/rate protocol](protocols/guard_rate_pilot_v1.json): four seed-2 Phi
  transfer development pairs, full/track8 crossed with shrink/observe guards
  and original/quarter learning rate. All arms observe the same three guard
  prompts. Fixed checkpoints are 32/64/128; 32 trajectories total. The
  [runner](src/run_guard_rate_pilot.py) uses the unchanged trajectory kernel.
  Eight focused tests passed. All 32 trajectories completed and were
  revalidated on 2026-09-08, including configurations, checkpoint guard scores,
  source hashes and exact report regeneration. See
  [guard/rate results](results/guard_rate_pilot_v1/RESULTS.md). The observe arm
  is not a capability-preserving deployment method.
- [Baseline-context protocol](protocols/baseline_context_pilot_v1.json): 16
  fresh zero/one-swap pairs, each evaluated under none/prior/fresh_a/fresh_b
  demonstration context, with fixed swap wording and in_box cue. It schedules
  128 responses with unchanged generation and parsing. Seven focused tests
  passed; all actual query answer boundaries align. All 128 responses completed
  on 2026-09-08 after guard/rate verification. Saved responses, token identities,
  source hashes and paired summaries passed the runner's verification. See
  [baseline-context results](results/baseline_context_pilot_v1/RESULTS.md).
- CPU-only design validation found too few unused no-swap questions in the
  original six-object vocabulary. Before any new G2 model evaluation, cup/pen
  were added to the declared query vocabulary and deterministic paired-object
  assignment was recorded. Queries and objects stay identical across all
  context arms, labels remain balanced, and prior/test prompt overlap is rejected.
  This is a documented feasibility correction, not model-outcome selection.
- Earlier no-swap swap/in_box output was H in 14/16 responses despite balanced
  targets. The new context comparison tests one explanation for that collapse;
  it does not assume demonstrations caused it.

### Second-Wave Outcomes

At step 128, on the same four Phi transfer development pairs:

| Method | Rate factor | Shrink-enforced target success | Observe-only target success | Guard disagreement: shrink / observe |
|---|---:|---:|---:|---:|
| Full-space | 1.0 | 4/4 | 4/4 | 0.0% / 41.7% |
| Full-space | 0.25 | 4/4 | 4/4 | 0.0% / 16.7% |
| Track8 | 1.0 | 1/4 | 3/4 | 8.3% / 16.7% |
| Track8 | 0.25 | 1/4 | 3/4 | 0.0% / 16.7% |

The paired observe-minus-shrink track8 increase is +50 percentage points at
both rates, with interval [-43.0, 87.3]. This small crossed experiment shows
guard-policy sensitivity on these development pairs, not a population effect,
a convergence proof or a safety improvement. Guard disagreement is measured on
three optimizer-visible prompts reused across the four edits, not held-out accuracy.

Fresh zero/one-swap questions with fixed wording and answer cue:

| Demonstration context | Zero-swap correct /16 | One-swap correct /16 |
|---|---:|---:|
| None | 1 | 16 |
| Earlier prefix | 3 | 15 |
| Fresh set A | 10 | 16 |
| Fresh set B | 2 | 16 |

Strict and parsed correct counts agree in this fixed-cue diagnostic. The earlier
prefix produced H on 15/16 zero-swap responses; fresh set B produced G on 12/16.
Changing context changes behavior, but removing it does not fix zero-swap
performance. Do not conclude that demonstrations alone caused the failure or
select fresh set A as a validated solution after observing its higher score.
The contexts reuse eight semantic pairs per depth and two sides; this is not
128 independent examples or evidence of reliable multistep reasoning.

All 122 subproject tests passed after both runs. The original frozen core
source and evidence ledger still match their recorded hashes. No historical
experiment or first-wave pilot was overwritten.

### First-Wave History

- 2026-09-07: documented all seven reviewer gaps, evidence and resolution gates.
- G1: separate [trajectory kernel](src/convergence_pilot.py) and
  [runner](src/run_convergence_pilot.py) implemented. Eleven focused tests pass,
  including equivalence to the original 32-step optimizer and guard behavior.
  The [declared protocol](protocols/convergence_pilot_v1.json) completed on MPS;
  all 16 saved trajectories, source hashes, summary and report were revalidated.
  Full-space succeeded on 8/8 examples at 32, 64 and 128 steps; track8 succeeded
  on 1/8, 2/8 and 2/8 respectively. See [pilot results](results/convergence_pilot_v1/RESULTS.md).
  Track8 had 118 guard shrink events across its eight trajectories, versus 11
  for full-space. This motivates a separate development-only guard/optimizer
  diagnosis, not a retrospective change to the current protocol or safety rule.
  The small pilot does not close G1 or prove convergence.
- G2 preparation: [fresh-data builder](src/swap_format_data.py) and
  [baseline runner](src/run_swap_format_pilot.py) implemented for the
  [crossed diagnostic](protocols/swap_format_pilot_v1.json). Eleven focused tests
  pass; all 256 prompts are disjoint from the old manifest and have aligned
  canonical answer boundaries. G2 ran after G1 completed and saved all 256
  responses. Case identities, decoded answers, full first-step scores, summary
  and report were independently revalidated. See [G2 results](results/swap_format_pilot_v1/RESULTS.md).
- Original frozen study is complete; no historical result is scheduled for overwrite.
- 2026-09-07 21:20 local check: no active model worker; G1 complete, G2 prepared
  but not yet executed. Sequential work resumes with the declared G2 diagnostic.
- Subsequently completed G2 in one sequential MPS run; no changes to original
  experiment sources or the original evidence ledger. All 107 subproject tests pass.

### First G1 Pilot Outcome

| Method | 32 steps | 64 steps | 128 steps |
|---|---:|---:|---:|
| Full-space | 8/8 | 8/8 | 8/8 |
| Track8 | 1/8 | 2/8 | 2/8 |

For track8, the 128-minus-32 difference is +12.5 percentage points with a
wide paired interval [-36.7, 52.1]. Mean target margin improves from -5.3591
to -3.1163; one example was successful earlier but failed at the final step.
These are descriptive development results, not new test-set evidence. The
next G1 experiment must distinguish guard-triggered shrinking, effective
step size, and basis restrictions before a held-out confirmation is selected.

### First G2 Pilot Outcome

| Example condition | Strict next-token correct | Parsed first answer correct |
|---|---:|---:|
| One swap, exchange wording, `is` cue | 0/16 | 16/16 |
| Three swaps, exchange wording, `is` cue | 0/16 | 7/16 |
| Four swaps, exchange wording, `is` cue | 0/16 | 7/16 |
| Zero swaps, swap wording, `in_box` cue | 4/16 | 4/16 |

Every `is`-cue cell has strict next-token accuracy zero, but some generated
answers correctly start with `box` and a label. A saved example targets D and
returns `box D.`, which is strictly incorrect at the first token but has a
correct parsed location. This directly demonstrates a format-sensitive score
in the new diagnostic, not a proven root cause for every historical test failure.

Parsed accuracy on three/four-swap cells is 6-9/16, below 80%. Zero-swap controls
are also weak (1-9/16 depending on condition); do not infer a simple monotonic
depth effect or reliable baseline competence. Fixed few-shot context, format,
and state reasoning still need to be separated on fresh development data.

All raw responses were parseable under the predeclared parser. The simpler new
diagnostic omits the original irrelevant final swap and reuses eight pairs
across wordings/cues; results are not independent or a replacement test set.

### Next Revision Steps

1. G1: the declared guard/rate comparison is complete. A further protocol needs
  alternative optimizer and equal-rank basis controls before untouched-test
  confirmation. Preserve the target-control versus monitored-disagreement tradeoff;
  observe-only is not a lowered safety gate for deployment.
2. G2: the context comparison is complete. Inspect the saved wrong no-swap
  predictions before selecting another bounded development hypothesis; removing
  demonstrations is not sufficient. Keep the answer cue/parser fixed and do not
  treat a retrospectively selected demonstration prefix as a competence result.
3. Keep G3-G6 queued until the next experimental protocol is explicitly fixed.
  None of the four completed pilots closes its parent scientific gap.

The earlier reproduction capture identifies the document versions at that
capture time. This authorized revision changes tracking documents and adds new
code; it does not rewrite that capture. New-run provenance separately records
the current executed sources and verifies the captured model asset contents.

See [the active task plan](task_plan.md) and [progress log](progress.md).