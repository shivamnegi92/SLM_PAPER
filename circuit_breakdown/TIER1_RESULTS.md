# Tier-1 Frontier Experiments — Results Log

Implementation: `src/intervene_pareto.py`. Llama-3.2-3B, `intermediate`, fp32,
MPS, layers [18,20,22,24], n=42 (test=9/seed), inference-time only, no weight
updates. Reference notes: `references/NOTES.md`.

---

## Headline table (seed-pooled where available)

| Method | runs | n_test | STEER (95% CI) | BREAK (95% CI) | Logit lift | ctrl | \|edit\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| unconstrained margin (full space) | 3 | 27 | 66.7% [48.1, 85.2] | 22.2% [7.4, 37.0] | +9.353 ±1.36 | 4.9% | 14.58 |
| + KL trust region (λ=10) | 3 | 27 | 55.6% [37.0, 74.1] | **14.8% [3.7, 29.6]** | +8.398 ±1.10 | 3.7% | 14.93 |
| confined to **tracking subspace** r8 | 1 | 9 | **0.0%** | 22.2% | +1.977 | 0.0% | 16.00 |
| confined to **orthogonal complement** r8 | 3 | 27 | **0.0%** | 0.0% | +0.485 ±0.12 | 1.2% | 15.64 |

---

## Finding 1 — KL trust region reduces BREAK, weakly

λ sweep {1, 3, 10, 20, 50} gives a clean interior optimum near λ≈10: λ=1 is
indistinguishable from baseline, λ=50 destroys the edit (11.1% STEER). Pooled,
KL moves BREAK 22.2% → 14.8% and control-drop 4.9% → 3.7%.

Paired per-sample check (n=27): KL fixed **2** cases the baseline broke and
broke **0** the baseline handled. Directionally strictly-dominant, but 2/27 with
heavily overlapping CIs is **not significant**.

Caveat that must not be glossed: KL also lowered STEER 66.7% → 55.6%, so part of
the BREAK reduction is simply a lower-STEER operating point. The honest
comparison is **BREAK at matched STEER**, which needs the full λ-curve at n≥120.

**Status: promising, unproven.**

## Finding 2 — Two-stage lifts STEER; Stage B shrinks edits only once unclamped

At `norm_budget=4`, Stage B did nothing to edit size: `|edit|` sat at exactly
16.00 = 4 layers × 4.0, and a 40× increase in `w_norm` (0.05 → 0.2) moved it by
0.01. Stage B was entirely clamp-bound — the norm penalty had no room to act.

Re-running at `norm_budget=12` with Stage A held identical isolates the effect:

| | STEER | BREAK | \|edit\| | ctrl |
|---|---:|---:|---:|---:|
| Stage A only | 55.6% | 22.2% | 29.13 | 40.7% |
| Stage A + Stage B | 44.4% | 22.2% | **20.91** | **29.6%** |

Stage B cuts edit norm **−28%** and control-drift **−11 pts** at unchanged BREAK,
costing ~11 pts STEER. So the mechanism does work as designed — the earlier null
was an artifact of the clamp, not a failure of the idea. Single seed; needs
replication.

## Finding 3 (headline) — A causal/control dissociation

This is the most important result and it is a **negative-plus-control**, which
makes it much stronger than the earlier accidental null.

Two confounds were found and eliminated before drawing any conclusion:

1. **Step-size confound.** Adam moves ~`lr` per coordinate, so step norm scales
   as `lr·√(#params)`. Optimizing 8 subspace coefficients rather than 3072 raw
   dims shrank the effective step ~20×. Fixed by `√(hidden/rank)` lr autoscaling
   (`--no-lr-autoscale` to disable). Post-fix, subspace runs reach the full norm
   budget (|edit| ≈ 16.00), so they are genuinely step-matched.
2. **Silent rank cap.** The basis is an SVD over `n_train` difference vectors, so
   rank ≤ n_train+1 = 26. Requests for 32 and 64 both silently became 26 — which
   is exactly why those runs were byte-identical. Now surfaced as
   `subspace_rank_actual`.

With both fixed, and adding the orthogonal-complement control (verified
orthogonal to 1e-8, orthonormal to 2e-7):

- Edits confined to the **tracking subspace** — the directions that causally
  carry the clean/corrupt distinction — achieve **0% STEER**, at every rank
  tried (8, 26), at full edit budget.
- Edits confined to the **orthogonal complement** also achieve 0% STEER, but
  produce **4–7× less logit movement** (+0.485 vs +1.977 at r8).

So the tracking subspace is doing real, measurable work on the continuous
decision variable — it is not inert — yet it is **decision-insufficient**: no
edit within it crosses the decision boundary. The directions that actually flip
the output lie substantially outside the span of the difference vectors.

> Causally valid localization does not imply decision-level control. We show this
> constructively: confining edits to the causally-identified subspace prevents
> override, while unconstrained edits in the *same layers, same budget* succeed
> 66.7% of the time.

Two further points worth making in the paper:

- **This is Schaeffer's effect inside our own method.** Tracking subspace and
  complement are indistinguishable under the discontinuous metric (both 0.0%
  STEER) but differ 4–7× under the continuous one (logit lift). Had we reported
  only STEER we would have concluded the tracking subspace was inert — exactly
  the metric-choice artifact `2304.15004` warns about. Strong argument for
  reporting a continuous decision metric alongside STEER everywhere.
- **It engages the subspace-illusion debate from the other side.** Makelov et al.
  (2023) warn that a *successful* subspace intervention is weak evidence for
  having found the mechanism. Here we have the converse: a subspace with genuine
  causal provenance that nonetheless cannot control behaviour. Both directions
  point at the same conclusion — intervention success/failure alone is not a
  reliable readout of mechanism.

---

## Full single-seed ablation (seed 0 unless noted)

| Arm | Method | STEER | BREAK | dlogit | ctrl | \|edit\| |
|---|---|---:|---:|---:|---:|---:|
| 1 | margin baseline | 44.4% | 11.1% | +8.015 | 11.1% | 13.33 |
| 2 | + KL λ=1 | 44.4% | 11.1% | +7.712 | 11.1% | 13.33 |
| 3 | + two-stage | 55.6% | 11.1% | +6.876 | 14.8% | 16.00 |
| 4 | + subspace r8 (**unmatched lr — invalid**) | 0.0% | 0.0% | +0.572 | 0.0% | 4.46 |
| 5 | all three (**unmatched lr — invalid**) | 0.0% | 0.0% | +0.955 | 0.0% | 7.69 |
| W2-A | two-stage, w_norm=0.05 | 55.6% | 11.1% | +6.922 | 14.8% | 16.00 |
| W2-B | two-stage, w_norm=0.2 | 44.4% | 11.1% | +6.899 | 22.2% | 15.99 |
| W2-C | KL λ=10 | 44.4% | 0.0% | +7.209 | 7.4% | 14.40 |
| W2-D | KL λ=50 | 11.1% | 11.1% | +4.458 | 7.4% | 14.93 |
| W3-A | subspace r8, step-matched | 0.0% | 22.2% | +1.977 | 0.0% | 16.00 |
| W3-B | subspace r26 (max), step-matched | 0.0% | 22.2% | +4.054 | 0.0% | 16.00 |
| W3-C | KL λ=3 | 33.3% | 0.0% | +7.519 | 7.4% | 13.87 |
| W3-D | KL λ=20 | 33.3% | 11.1% | +6.222 | 7.4% | 14.40 |
| W3-E | two-stage + KL10 | 33.3% | 11.1% | +6.236 | 11.1% | 16.00 |
| W3-F | two-stage + KL10 + sub8 matched | 0.0% | 22.2% | +1.959 | 0.0% | 16.00 |
| W4 | complement r8, s0/s1/s2 | 0.0% | 0.0% | +0.318/+0.597/+0.540 | ~1% | 15.6 |

Seed spread on the two pooled arms (STEER / BREAK):

| Method | s0 | s1 | s2 |
|---|---|---|---|
| baseline | 44.4 / 11.1 | 77.8 / 44.4 | 77.8 / 11.1 |
| KL λ=10 | 44.4 / 0.0 | 55.6 / 33.3 | 66.7 / 11.1 |

33-point single-seed swings confirm n_test=9 is below usable resolution
(Schaeffer: resolution = 1/test-set size = 11.1%). **Never report single-seed
numbers from this rig.**

---

## Status vs. the Tier-1 plan

| Method | Implemented | Verified | Outcome |
|---|---|---|---|
| 1. Two-stage Pareto | yes | yes | STEER +11pts; norm-shrink confirmed only after relaxing clamp (−28% edit, −11pts drift) |
| 2. KL trust region | yes | yes | BREAK 22.2%→14.8% pooled; right direction, not significant |
| 3. Subspace projection | yes | yes (2 confounds fixed + complement control) | Fails to steer — **now the strongest result in the set** |

## Next actions

1. **Scale the dissociation to n≥120** and add tracking-subspace seeds 1–2 (only
   s0 exists). This is the headline; it deserves the tightest CIs in the paper.
2. **λ-curve at matched STEER** for the KL arm — the only way to make the BREAK
   claim defensible.
3. **Replicate the Stage B norm-shrink** at budget 12 across seeds 1–2.
4. Add a **continuous decision metric** (margin, or Brier-style
   `p(clean)/(p(clean)+p(corrupt))`) to every table — Finding 3 shows STEER alone
   would have hidden a 4–7× effect.
5. Sweep the **rank** of the complement control to confirm the logit-lift gap is
   about direction, not dimensionality.
