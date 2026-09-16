# Predictive Subspace vs Causal Subspace Plan

Updated 2026-09-09. This document changes the next research direction from
"rank8 tracking failed" to a sharper, falsifiable question:

```text
Predictive subspace != causal subspace
```

The current validated study remains unchanged. This is a new revision direction
for future development-only protocols and, only after locked confirmation,
paper claims.

## Plain-English Thesis

A representation can make the answer readable without being the place where the
model actually computes or transmits the answer.

So the central question is not:

```text
Can a small subspace control reasoning?
```

The better question is:

```text
When does a subspace that predicts a reasoning state also become causally
sufficient for controlling that reasoning behavior?
```

## Why We Should Change Direction

Existing and emerging work makes a broad claim like this risky:

```text
Low-dimensional causal control of reasoning is impossible.
```

The safer and stronger opportunity is narrower:

```text
Correlational or probe-selected tracking subspaces can be highly predictive,
while equally small causally selected subspaces may be much more effective for
intervention.
```

That converts the paper from a negative steering story into a representation
versus mechanism story.

## Current Evidence From This Repo

The completed study already establishes a useful starting point:

```text
full-space target-informed edits: 100% on eligible comparisons
selected track8 edits: 8.7%-56%
matched comp8 controls: 0%
```

But that does not prove why track8 failed. Possible explanations remain:

```text
1. The tracking subspace is predictive but not causal.
2. The tracking subspace estimator is poor.
3. Rank8 is too small.
4. Optimization, guard policy, or learning rate hurt track8.
5. A different causally selected low-rank subspace could work.
```

## New Experimental Core

Use the same model, task, layers, examples and dimensionality whenever possible.
Compare four equal-dimensional subspaces:

| Subspace | How selected | Question |
|---|---|---|
| Tracking | Current clean-corrupt activation-difference SVD / probe-like tracking basis | Where is the reasoning state readable? |
| PCA | High-variance activation directions | Is generic representation structure enough? |
| Random | Random orthonormal basis | Does dimensionality alone explain control? |
| Causal | Directions/components selected by intervention effect | Where does changing activation actually change behavior? |

For each subspace $S_k$, measure both readout and intervention:

$$
D(S_k) = \mathrm{Accuracy}(\mathrm{probe}(P_{S_k}h))
$$

$$
C(S_k) = P(Y = y_{target} \mid do(P_{S_k}h + \delta)) - P(Y = y_{target})
$$

The paper-worthy table is not just accuracy. It is the mismatch between
`D(S_k)` and `C(S_k)`.

## Step A: Projection Diagnostic Already Implemented

The current new protocol is:

```text
protocols/causal_dimensionality_pilot_v1.json
src/run_causal_dimensionality_pilot.py
```

It uses saved successful full-space vectors from the convergence pilot and
measures:

```text
v_full
P_S(v_full)
v_perp = v_full - P_S(v_full)
random projection
coordinate top-k reconstruction
permuted full-vector control
```

Across:

```text
k = 1, 2, 4, 8, 16, 32, 64
```

This answers the first question:

```text
How much of the successful full control direction is captured by the selected
tracking subspace?
```

It does not yet find a causally optimized low-rank subspace.

## Step B: Equal-k Predictive vs Causal Subspace Test

Next, run a new protocol that constructs all four subspaces at the same rank:

```text
k = 1, 2, 4, 8, 16, 32, 64
```

For each k:

```text
D_tracking(k), C_tracking(k)
D_pca(k),      C_pca(k)
D_random(k),   C_random(k)
D_causal(k),   C_causal(k)
```

Implementation started in:

```text
src/predictive_vs_causal_subspace.py
src/run_predictive_vs_causal_subspace.py
tests/test_predictive_vs_causal_subspace.py
protocols/predictive_vs_causal_subspace_v1.json
```

The first implemented causal arm uses gradient-ranked coordinate bases: for a
fixed intervention site, it backpropagates the target log-probability through a
zero additive edit and selects the highest-magnitude coordinates as a cheap
first-order causal direction set. This is not yet a learned sparse mask or a
full finite-difference sweep, but it creates an equal-rank causal baseline that
is distinct from tracking, PCA and random bases.

Run command, once local execution is available:

```bash
PYTHONPATH=src .venv/bin/python src/run_predictive_vs_causal_subspace.py --device auto
```

Expected output:

```text
results/predictive_vs_causal_subspace_v1/summary.json
results/predictive_vs_causal_subspace_v1/RESULTS.md
```

Decision table:

| Outcome | Meaning | Paper direction |
|---|---|---|
| Tracking decodes high and controls low; causal controls high | Predictive subspace differs from causal subspace | Strong representation-vs-mechanism result |
| Tracking and causal both control high | Original tracking method may be adequate after optimization | Steering/discovery method paper |
| No low-k subspace controls well | Causal control may be distributed for this behavior | Causal dimensionality paper |
| Random controls similarly to causal | Control is not semantically aligned; likely perturbation artifact | Revise or abandon causal claim |

## How To Select a Causal Subspace

Start with a finite-difference directional effect score. For candidate direction
$u_i$:

$$
CE_i = \frac{L(h + \epsilon u_i) - L(h - \epsilon u_i)}{2\epsilon}
$$

Rank by $|CE_i|$ and construct:

$$
S^k_{causal} = \mathrm{span}(u_1, \ldots, u_k)
$$

Candidate direction pools, from cheapest to stronger:

```text
1. Hidden coordinate axes at selected layers
2. PCA directions
3. Tracking SVD directions plus orthogonal-complement directions
4. Attention-head output directions or receiver components
5. Learned sparse mask / optimized low-rank basis
```

The first version should use coordinate/PCA/tracking candidate pools because it
is cheaper and easier to audit.

## Control Versus Utility

Do not call this only safety. Measure control versus utility.

For each method and intervention strength alpha:

```text
Target control
Original task accuracy
Counterfactual task accuracy
KL shift from baseline logits
Text perplexity ratio
General benchmark accuracy if budget allows
Wrong-answer promotion on hard negatives
```

The useful paper result would look like:

```text
Causal sparse subspace approaches dense control while preserving more utility
than full-space steering.
```

But that is a hypothesis, not a current result.

## Literature Mapping Queue

The repo currently contains notes for Schaeffer et al. 2023, Sharkey et al. 2025,
and Naseem 2026. The new claims about sparse causal steering, sparse attention
heads, causal probing, GCM, CAS/BiPO, LEACE, CAA, ActAdd, DAS and sparse
autoencoders still need primary-source verification before use in the paper.

Comparison matrix to fill:

| Area | What prior work likely shows | What we must verify | Our possible remaining gap |
|---|---|---|---|
| CAA / ActAdd / representation engineering | Activation directions can steer behavior | Tasks, models, dimension, utility metrics | Reasoning-state predictiveness versus causal sufficiency |
| DAS / subspace alignment | Distributed subspaces can manipulate outputs or align states | Whether behavior change equals faithful mechanism | Direct decode-control mismatch on reasoning tasks |
| LEACE / causal probing | Linear information removal differs from readout | Exact intervention/removal assumptions | Projection and orthogonal-complement causal test |
| Sparse mediation / GCM | Sparse causal selection can recover dense steering | Claimed dimension fractions and benchmarks | Equal-k predictive-vs-causal comparison |
| Sparse attention/head steering | Causally selected components can beat probe localization | Component type and selection method | Same behavior, same k, same utility protocol |
| Sparse autoencoders / feature discovery | Interpretable features may not steer causally unless evaluated by intervention | Which methods fail/pass causal evaluation | Reasoning-specific causal controllability curve |
| CAS / BiPO / preference steering | Steering can trade control against utility or KL shift | Metrics and normal-manifold constraints | Control-utility frontier for reasoning intervention |

## Claims To Avoid For Now

Do not claim:

```text
Low-rank causal control is impossible.
Tracking subspaces are never causal.
Full-space edits are safe.
The current method finds the reasoning circuit.
Existing literature has not studied this issue.
```

## Better Claim To Test

The next paper hypothesis should be:

```text
Reading is not controlling: a compact subspace can expose a reasoning state to a
probe while failing to causally control that behavior, and causally selected
subspaces can have a different control-utility frontier at the same rank.
```

This is falsifiable. If causally selected low-rank subspaces recover dense
control, the paper becomes a method/comparison paper. If they do not, the paper
becomes a causal dimensionality paper.

## Immediate Next Actions

```text
1. Run causal_dimensionality_pilot_v1 once local execution works.
2. Run predictive_vs_causal_subspace_v1 on development examples.
3. Verify the new literature claims from primary sources.
4. Add a stronger causal selector if gradient-ranked coordinates are promising or ambiguous.
5. Add control-utility alpha sweep only after the core decode-control split is visible.
6. Confirm any selected result on untouched examples and at least one additional model.
```
