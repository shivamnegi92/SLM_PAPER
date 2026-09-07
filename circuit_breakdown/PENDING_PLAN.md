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
| C6 | Subspace decision-sufficiency is **architecture-dependent** | Llama: gap 76.9-96.2pts p<0.0001 @ 3 budgets. Nemotron: gap 23.1-30.8pts p≤0.019 @ 3 budgets. Phi: p=1.0 (exact) @ 2 budgets | **RESOLVED — 2-of-3 dissociate** |
| C7 | Absolute norm budgets don't transfer across archs | 7.5× resid spread; Phi 0%→66.7% | **SOLID** (methods contribution) |

**C6 is resolved with a clean 2-of-3 architecture split.** Remaining P0 work is
replication (seeds) and two validity gaps surfaced from re-reading PLAN.md.

---

## P0 — Blocking. Resolves the headline claim.

### P0.1 Budget-matched control sweep   **DONE — see `BUDGET_CONTROL_RESULTS.md`**
Ruled out the confound both directions. Llama's dissociation survives budgets
0.17/0.30/0.50 (all p<0.0001, narrowing but never closing). Phi shows **exact**
equivalence (p=1.0) at 0.17 and 0.50 — not weak, genuinely absent. This is a
real architecture difference, not a hyperparameter artifact. Reframe C6
accordingly (see `BUDGET_CONTROL_RESULTS.md` for the exact wording).

### P0.1b Nemotron at matched budget  *(DONE)*
Nemotron **patterns with Llama, not Phi**, and its pattern is arguably cleaner:
gap is significant at all three budgets (+23.1%/+30.8%/+30.8%, p≤0.019) and
does not narrow with budget the way Llama's does — if anything it widens
slightly. **Final triangle: 2 of 3 architectures dissociate (Llama, Nemotron),
1 does not (Phi), and Phi's null is exact (p=1.0) rather than marginal.** This
is a genuine, reportable variance pattern — see `BUDGET_CONTROL_RESULTS.md`
§Three-architecture picture.

### P0.2 Seed 2 for cross-architecture AND for the budget-matched sweep
All budget-control cells above are seed 0 only (n=26 each). Given how clean the
pattern is (p=1.0 exactly, twice, on Phi) this is low-risk, but a second seed on
each of the 10 budget-control cells plus the Nemotron sweep is required before
this becomes a headline claim. ~2h.

### P0.3 Rewrite the C6 claim to match whatever P0.1 shows
No new compute. Update `DISSOCIATION_RESULTS.md` §Result 1 and the abstract
framing. **Do not let the Llama-only phrasing survive into the draft.**

### P0.4 Self-repair / backup-head validity check  *(NEW — from PLAN.md §4, never done)*
PLAN.md explicitly names this "our main threat to validity": ablating one head
can be silently compensated by a backup head (the Hydra effect, McGrath/Wang),
which corrupts importance estimates from `heads.py`'s AtP + real-patching
pipeline. **No test for this exists anywhere in the repo.** Needs: path patching
on the top-k heads, or at minimum an explicit check that ablating a top head
doesn't get silently absorbed by a next-layer head. This is a paper-shaped gap,
not a compute-shaped one — a reviewer who knows the literature will ask for it
by name.

### P0.5 ARC-Easy / MMLU-subset capability run  *(NEW — PLAN.md §4.1 gate never fully closed)*
`RESULTS.md` itself flags this as pending (lines 169, 240, 253) and PLAN.md's
G3 gate specifically named ARC-Easy/MMLU, not HellaSwag. `capability_deployed.py`
already has the harness shape (baseline/deployed/global/random, bootstrap CIs) —
swapping in an MMLU-subset loader is the same script, new data loader. Needed to
formally close G3 rather than leave it hedged on a proxy benchmark.

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
- **P3.5** Fix stale cross-model references. `PLAN.md` and `README.md` still say
  "Llama vs Gemma-2-2B"; `RESULTS.md:5` admits Gemma is blocked (gated HF
  download) and never happened. What actually ran — Phi-3.5-mini +
  Nemotron-Mini-4B, 3 families total — is stronger than the original plan, but
  the docs don't say so. 5-minute fix, prevents a reader chasing nonexistent
  Gemma results.
- **P3.6** Pick a venue. `PLAN.md` assumes ICLR; the original blueprint said
  ICML. Unresolved, blocks final formatting.

---

## Suggested execution order

```
P0.1 (running) ──> P0.3 ──> P0.2 ──┐
                                   ├──> P1.1 ──> P1.2 ──> P3.*
P2.1, P2.3 (fill idle compute) ────┘
```

**Next 24h of compute:** finish P0.1b (Nemotron, ~25min left), run P0.2 and P2.1
back-to-back, then start P1.1 harness work while those run. P0.4/P0.5 are
documentation/harness work, not compute-bound — can happen in parallel with
anything.

**The one thing that matters most:** P0.1's outcome determines whether the paper
claims "prevents override" or "requires disproportionate budget." Everything
downstream inherits that wording.
