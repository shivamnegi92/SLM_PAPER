# Pending Work — Prioritized Plan

Status as of 2026-09-06. Ordered by **risk to the paper's central claim**, not
by effort. Items in P0 can invalidate published claims; P3 items are polish.

---

## Claim ledger (what is actually established)

> **UPDATE:** P0.1 is resolved — see `BUDGET_CONTROL_RESULTS.md`. The confound
> is ruled out on both ends (Llama's gap survives 3 budgets; Phi's null is exact
> at 2 budgets, p=1.0). C6 is reframed from a universal claim to an
> architecture-dependence claim, which is better supported by the data.

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Continuous metric separates tracking vs complement | 3 archs, all p<0.0001, 24–208× | **SOLID** |
| C2 | Gap is direction, not dimensionality | 5 ranks, ratio ~r^0.95, all p<0.0001 | **SOLID** |
| C3 | Deployed intervention costs ~0 general capability | HellaSwag/ppl identical, n=240 | **SOLID** (1 model) |
| C4 | Metric ladder is load-bearing | 10/10 cells null on top-1, all separated continuously | **SOLID** |
| C5 | KL trust region reduces BREAK | n=78, p=0.63 | **FALSIFIED** — reported as negative |
| C6 | Subspace decision-sufficiency is **architecture-dependent** | Llama: gap 76.9-96.2pts, p<0.0001 at budgets 0.17/0.30/0.50. Phi: p=1.0 (exact match) at budgets 0.17/0.50 | **RESOLVED — reframed, stronger** |
| C7 | Absolute norm budgets don't transfer across archs | 7.5× resid spread; Phi 0%→66.7% | **SOLID** (methods contribution) |

**C6 is the paper's headline and it currently fails on 1 of 3 architectures.**
Everything in P0 exists to resolve C6.

---

## P0 — Blocking. Resolves the headline claim.

### P0.1 Budget-matched control sweep   **DONE — see `BUDGET_CONTROL_RESULTS.md`**
Ruled out the confound both directions. Llama's dissociation survives budgets
0.17/0.30/0.50 (all p<0.0001, narrowing but never closing). Phi shows **exact**
equivalence (p=1.0) at 0.17 and 0.50 — not weak, genuinely absent. This is a
real architecture difference, not a hyperparameter artifact. Reframe C6
accordingly (see `BUDGET_CONTROL_RESULTS.md` for the exact wording).

### P0.1b Nemotron at matched budget  *(NEXT — closes the triangle)*
Nemotron showed a significant gap (+21.2%, p<0.0001) but at 30%/16-step
settings only, not the full 0.17/0.30/0.50 sweep. Running it at matched budgets
places it on the same axis as Llama and Phi and tells us whether it patterns
with Llama (subspace matters) or Phi (subspace doesn't). 6 runs, ~55 min.

### P0.2 Seed 2 for cross-architecture AND for the budget-matched sweep
All budget-control cells above are seed 0 only (n=26 each). Given how clean the
pattern is (p=1.0 exactly, twice, on Phi) this is low-risk, but a second seed on
each of the 10 budget-control cells plus the Nemotron sweep is required before
this becomes a headline claim. ~2h.

### P0.3 Rewrite the C6 claim to match whatever P0.1 shows
No new compute. Update `DISSOCIATION_RESULTS.md` §Result 1 and the abstract
framing. **Do not let the Llama-only phrasing survive into the draft.**

---

## P1 — Required for the generalization score (the 4/10)

### P1.1 A harder, non-templated task
Current tasks are synthetic templates ("package moved from X to Y to Z"). This
is the single most-cited weakness and no amount of extra seeds fixes it.

Options, cheapest first:
- **Nested/distractor state tracking** — same generator, add irrelevant entity
  mentions and 5–6 hops. Cheap, tests the same circuit under load.
- **bAbI task 2/3** (two/three supporting facts) — real benchmark, still
  single-token answers, downloadable.
- **Naturalistic pronoun/coreference binding** — closest to real usage.

Recommend bAbI-2: real benchmark, minimal harness change, quotable name.

### P1.2 Cross-arch on the harder task
Only meaningful after P1.1 and P0.1. The pairing "3 architectures × 2 task
families" is what converts generalization 4/10 → 7/10.

---

## P2 — Strengthens, doesn't block

### P2.1 Replicate two-stage Stage-B norm shrink
Only remaining single-seed claim (−28% |edit|, −11pts drift at budget 12).
Seeds 1–2, ~25 min. Either promote or drop it.

### P2.2 Rank sweep top-up
r ∈ {1,4,16,26} have 2 seeds vs 3 for r8. Trend is monotonic and all p<0.0001,
so this is camera-ready polish only.

### P2.3 Capability on Phi + Nemotron
C3 is one model. Cheap to extend now that `capability_deployed.py` exists.

### P2.4 BREAK-at-matched-STEER curve
The fair way to compare interventions. Needed if any BREAK claim returns to the
headline; currently BREAK is reported as an honest cost, so not blocking.

---

## P3 — Paper mechanics

- **P3.1** Replace `RESULTS.md` §6b with the deployed-capability table; relabel
  the global number as an explicit worst-case upper bound.
- **P3.2** Fold `references/NOTES.md` actions into the draft — cite Open Problems
  A.2.2.1a in the abstract, Schaeffer for the metric ladder, Makelov for the
  subspace-illusion framing.
- **P3.3** Frontier figure: STEER vs BREAK scatter, all methods/archs, CI ellipses.
- **P3.4** Consolidate the five results markdown files into one paper-shaped
  narrative.

---

## Suggested execution order

```
P0.1 (running) ──> P0.3 ──> P0.2 ──┐
                                   ├──> P1.1 ──> P1.2 ──> P3.*
P2.1, P2.3 (fill idle compute) ────┘
```

**Next 24h of compute:** finish P0.1, run P0.2 and P2.1 back-to-back, then start
P1.1 harness work while those run.

**The one thing that matters most:** P0.1's outcome determines whether the paper
claims "prevents override" or "requires disproportionate budget." Everything
downstream inherits that wording.
