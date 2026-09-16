# Budget-Matched Control - Historical Pilot

> **2026-09-07: retired interpretation below.** The original pilot rates are
> retained as history, but its architecture-split narrative, equivalence from
> `p=1`, and claims of identical predictions are not current paper claims.
> The completed frozen study found full-space advantages on all six eligible
> comparisons, including Phi: full 100% versus track8 8.7% on intermediate
> and 20.0% on transfer. See [the current summary](paper/VALIDATED_RESULTS.md)
> and [audited detailed tables](paper/RESULTS_DETAILS.md). The sections below,
> including their historical next steps, are not instructions to restart runs.
> Negative-edit BREAK is not steering's same-sign collateral damage.

> **Pilot caveat:** "resolved" here refers to completion of the seed-0 budget
> experiment, not proof that architecture is the isolated cause. Equal
> observed rates or `p=1.0` do not establish population equivalence, nor does
> equality of rates imply identical per-example predictions. Residual-layer
> windows, optimization behavior and checkpoint differences require separate
> checks. Preserve the measured gaps below as exploratory evidence. The
> authoritative new comparison is [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md).

`run_budget_control.sh`. Llama-3.2-3B and Phi-3.5-mini, matched `stage_a_steps=16`,
`rel_budget` swept at identical values on both models, n=126, seed 0, n_test=26 per cell.

## The question

Cross-arch replication (`CROSSARCH_RESULTS.md`) found the dissociation held on
Nemotron but not on Phi — but at *mismatched* hyperparameters (Llama headline:
17% budget / 8 steps; cross-arch: 30% budget / 16 steps). Before concluding
anything about architecture, the confound had to be removed.

## Result: confound ruled out, on both ends

| model | budget | FULL steer | TRACK8 steer | diff | p | TRACK8 break |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B | 0.17 | 96.2% | 0.0% | +96.2% | <0.0001 | 19.2% |
| Llama-3.2-3B | 0.30 | 92.3% | 0.0% | +92.3% | <0.0001 | 53.8% |
| Llama-3.2-3B | 0.50 | 92.3% | **15.4%** | +76.9% | <0.0001 | 65.4% |
| Phi-3.5-mini | 0.17 | 42.3% | **42.3%** | +0.0% | 1.0000 (ns) | 92.3% |
| Phi-3.5-mini | 0.50 | 42.3% | **42.3%** | +0.0% | 1.0000 (ns) | 92.3% |

**Llama:** the dissociation survives at every budget tested, from the original
17% all the way to 50% (half the residual norm — a huge edit). It narrows as
budget grows (gap 96.2 → 92.3 → 76.9 pts) but never closes; even at 0.50 the
tracking subspace still cracks only to 15.4% while full space reaches 92.3%.

**Phi:** zero dissociation at *both* budgets tested. Full-space and
subspace-confined steering produce **statistically identical predictions**
(p=1.0 — not just non-significant, exactly the same outcomes). This is not
"weaker than Llama's effect," it is **no effect at all**, and it is stable
across a 3× budget range.

## Verdict

This is not a budget confound. It is a genuine cross-architecture difference:

> On Llama-3.2-3B, the tracking subspace is **decision-insufficient** across a
> wide range of edit budgets — the directions that flip the output lie
> substantially outside it. On Phi-3.5-mini, the tracking subspace is
> **decision-sufficient**: confining edits to it costs nothing relative to
> unconstrained editing, at any budget tested.

Notably, Phi's subspace-confined edits are also enormously disruptive
(92.3% BREAK vs Llama's 19.2–65.4%) — so on Phi, controlling the decision
through this subspace comes at a much higher collateral cost than on Llama, even
though it succeeds where Llama's fails.

## Reframed claim for the paper

The universal, architecture-independent finding is **not** "confining edits to
the tracking subspace prevents override." It is the weaker but still novel and
better-supported claim:

> **Whether a causally-identified subspace is decision-sufficient is itself an
> architecture-dependent property.** We demonstrate this with a budget-matched
> control across two architectures: the same subspace-confinement procedure
> produces near-total resistance to override on one model (Llama, gap 76.9–96.2
> pts, p<0.0001 at all tested budgets) and zero resistance on another (Phi,
> exactly matched outcomes, p=1.0), at identical relative edit budgets.

This is a stronger paper than the original, singular claim — it predicts variance
across architectures instead of asserting a universal law, and the variance
itself is a finding (worth investigating: does it correlate with attention head
count, tokenizer, or training data? Nemotron's result, still to be re-run at
matched budget, will help triangulate).

## What's still open

1. **Nemotron at matched budget — DONE.** Confirms Nemotron patterns with
   Llama, not Phi (see `## Three-architecture picture` below).
2. **Second seed for the budget sweep.** All cells above are seed 0 only
   (n=26). Given how clean the pattern is (p=1.0 exactly on Phi, twice; p<0.02
   at worst on Nemotron), this is low-risk, but the plan requires it before
   this becomes a headline claim.
3. **Investigate why Phi differs.** Candidate correlates: attention head count
   (Phi 32 vs Llama 24 vs Nemotron 24), tokenizer (SentencePiece vs BPE),
   training data mix. Not required for the current claim but would strengthen
   the discussion section.

## Three-architecture picture (complete)

| model | budget 0.17 | budget 0.30 | budget 0.50 | pattern |
|---|---:|---:|---:|---|
| Llama-3.2-3B | +96.2% *** | +92.3% *** | +76.9% *** | dissociates, gap narrows with budget |
| Nemotron-Mini-4B | +23.1% *** | +30.8% *** | +30.8% *** | dissociates, gap **stable** across budget |
| Phi-3.5-mini | +0.0% (ns) | — | +0.0% (ns) | **no dissociation at any budget** |

(gap = full-space STEER minus tracking-subspace STEER; *** = p<0.05, all
significant gaps are p<0.02 or better)

**2 of 3 architectures show significant, budget-robust dissociation; 1 does
not, and its null is exact (p=1.0) rather than marginal.** This is not
universal, and it is not noise — it is a genuine, reportable split. Nemotron's
pattern is arguably cleaner than Llama's: the gap does not narrow with budget
at all (23.1% → 30.8% → 30.8%, if anything widening slightly), suggesting its
decision boundary lies more robustly outside the tracking subspace than
Llama's does.

Notable secondary finding: Nemotron's subspace-confined edits are also far more
disruptive where the subspace is engaged (BREAK 92.3→100→100%) even though
STEER stays near zero — consistent with the Phi pattern of "subspace edits cost
a lot of collateral regardless of whether they succeed at steering," but here
they mostly *don't* succeed, unlike Phi where they succeed at full efficiency.
## Reproduction

```bash
./run_budget_control.sh   # 10 runs, ~90 min
```

Artifacts: `results/bc_{llama,phi}_{full,track8}_b{0.17,0.30,0.50}.json`
(Phi only has 0.17 and 0.50).
