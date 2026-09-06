# Dissociation Experiment — n=126, 3 seeds, 4 conditions

`src/intervene_pareto.py` + `src/analyze_dissociation.py`.
Llama-3.2-3B, `intermediate`, fp32, MPS, layers [18,20,22,24], n=126 per seed
(test=26 seed 0, 26 each seeds 1–2 → **n_test=78 pooled per condition**).
Inference-time only, no weight updates. 12 runs total.

Metric ladder, most → least discontinuous (motivated by Schaeffer et al.
`2304.15004`; see `references/NOTES.md`):

| metric | definition |
|---|---|
| `STEER top1` | `argmax over full vocab == clean` |
| `STEER 2way` | `p(clean) > p(corrupt)` |
| `d p2way` | change in `p(clean)/(p(clean)+p(corrupt))`, bounded |
| `d logit` | change in `logit(clean)`, unbounded |

---

## Main table (3 seeds pooled, n_test=78)

| condition | STEER top1 | STEER 2way | BREAK | d p2way | d logit | ctrl | \|edit\| | pred=corrupt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unconstrained margin (full space) | **62.8%** [53, 73] | 100.0% [100, 100] | 26.9% [18, 37] | +0.9676 | +7.572 | 2.6% | 14.77 | 0.0% |
| + KL trust region (λ=10) | 42.3% [32, 54] | 97.4% [94, 100] | 23.1% [14, 32] | +0.9431 | +6.978 | 1.7% | 15.14 | 1.3% |
| confined to **tracking subspace** r8 | **0.0%** [0, 0] | 0.0% [0, 0] | 33.3% [23, 44] | **+0.0194** | +3.374 | 0.0% | 16.00 | 64.1% |
| confined to **orthogonal complement** r8 | **0.0%** [0, 0] | 0.0% [0, 0] | 5.1% [1, 10] | **+0.0001** | +0.459 | 0.0% | 15.75 | 80.8% |

---

## Result 1 — The dissociation is real and significant

**Same layers, same edit budget, same optimizer, same data.** The only
difference is whether the edit is confined to the causally-identified subspace.

- Unconstrained: **62.8%** top-1 override.
- Confined to the tracking subspace: **0.0%**.
- Difference **+62.8%**, 95% CI [+52.6, +73.1], bootstrap **p < 0.0001**.

The tracking subspace — the span of clean−corrupt activation differences, the
directions that demonstrably *carry* the state distinction — is **causally real
but decision-insufficient**. The directions that actually flip the output lie
substantially outside it.

> Causally valid localization does not imply decision-level control. We show it
> constructively: confining edits to the causally-identified subspace prevents
> override, while unconstrained edits in the same layers at the same budget
> succeed 62.8% of the time.

This engages the Makelov et al. (2023) subspace-illusion debate from the
opposite side. That work warns a *successful* subspace intervention is weak
evidence of having found the mechanism. Here: a subspace with genuine causal
provenance that **cannot** control behaviour. Both directions converge on the
same lesson — intervention success or failure alone is not a reliable readout
of mechanism.

## Result 2 — The metric ladder earns its keep (Schaeffer effect, in-house)

Tracking subspace and complement are **exactly identical** under both
discontinuous metrics (0.0% top-1, 0.0% 2-way). Under the continuous metric they
are cleanly separated:

- `d p2way`: **+0.0194 vs +0.0001**, difference +0.0193, 95% CI [+0.0086, +0.0345], **p < 0.0001**
- `d logit`: **+3.374 vs +0.459** — a **7.3×** gap

Reporting only top-1 accuracy would have produced the false conclusion that the
tracking subspace is inert. It is not: it moves the decision variable ~194×
more than a dimension-matched random control, while still never crossing the
boundary. This is precisely the metric-choice artifact `2304.15004` describes,
reproduced inside our own method — a strong argument for the ladder.

`pred_is_corrupt` adds the mechanism: under subspace confinement the model
mostly still emits the corrupt token (64.1% / 80.8%), i.e. the edit genuinely
fails to move the decision. Under unconstrained editing it is **0.0%** — when
full-space steering "fails," it never falls back to the corrupt answer; it has
overshot into some third token. Those are different failure modes and top-1
accuracy conflates them.

## Result 3 — KL trust region: hypothesis falsified at scale

At n=27 KL looked like a promising BREAK reducer (22.2% → 14.8%). At n=78 it
does not survive:

- BREAK difference **+3.8%**, 95% CI **[−10.3%, +17.9%]**, p = 0.63 — indistinguishable from zero.
- Paired McNemar: 3 fixed / 0 caused, exact **p = 0.25** — not significant.
- STEER cost **−20.5%**, 95% CI [−35.9, −5.1], **p = 0.012** — significant and *harmful*.

**Conclusion: KL(λ=10) costs a statistically significant 20.5 points of STEER
and buys no measurable BREAK reduction.** The earlier n=27 result was
small-sample noise, exactly as the resolution argument predicted. We report this
as a negative result and do not use KL in the headline method.

The honest framing for the paper: the apparent BREAK win at n=27 was mostly an
artifact of KL operating at a lower-STEER point on the same curve, not a genuine
frontier improvement.

## Result 4 — Where BREAK actually comes from

BREAK ordering is informative:
complement 5.1% < KL 23.1% < full 26.9% < tracking subspace 33.3%.

The tracking subspace has the **highest** BREAK despite **zero** steering
ability. So collateral damage is not a byproduct of successful steering — edits
along the tracking directions disrupt the clean-prompt computation while failing
to redirect the corrupt one. Meanwhile the complement barely perturbs anything
(5.1% BREAK, +0.0001 d p2way), confirming it is a genuinely inert control rather
than a differently-damaging one.

---

## Reproduction

```bash
./run_dissociation.sh                     # 12 runs, ~75 min on M4 Pro
python src/analyze_dissociation.py        # pooled CIs + significance tests
```

Artifacts: `results/diss_{full,kl10,track8,comp8}_s{0,1,2}.json`,
`results/diss_pooled_summary.json`.

## Status

| Claim | n | Verdict |
|---|---:|---|
| Localization ≠ control (subspace fails, full space works) | 78 | **Confirmed, p < 0.0001** |
| Continuous metric separates what top-1 cannot | 78 | **Confirmed, p < 0.0001** |
| KL trust region reduces BREAK | 78 | **Falsified** (p = 0.63; costs 20.5 pts STEER, p = 0.012) |
| Two-stage Stage B shrinks edits when unclamped | 9 | Preliminary (−28% \|edit\|, single seed) |

## Next

1. **Rank sweep on the complement** (r ∈ {4, 8, 16, 26}) to confirm the
   `d p2way` gap tracks *direction*, not dimensionality.
2. **Replicate two-stage at budget 12** across seeds 1–2 — the only remaining
   single-seed claim.
3. **Cross-architecture** dissociation on Phi-3.5-mini and Nemotron-Mini-4B;
   pairs with the relative-depth invariance law.
4. Retire KL from the headline; keep it as a documented negative result.
