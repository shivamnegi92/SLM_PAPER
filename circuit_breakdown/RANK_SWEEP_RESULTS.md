# Rank Sweep — Direction vs. Dimensionality

`src/intervene_pareto.py` + `src/analyze_rank_sweep.py`.
Llama-3.2-3B, `intermediate`, fp32, MPS, layers [18,20,22,24], n=126/seed.
r=8 uses 3 seeds (n_test=78); r ∈ {1,4,16,26} use 2 seeds (n_test=52).
20 runs total, ~2h on M4 Pro.

## The objection this answers

> *"Your tracking subspace moves the decision variable more than the random
> complement simply because it is a better-conditioned r-dimensional subspace,
> or because r itself is doing the work. Show the gap is about WHICH directions,
> not HOW MANY."*

Discriminating prediction:

- **Direction** → the gap persists (or widens) across ranks; complement stays near-inert.
- **Dimensionality** → the two converge as r grows and both subspaces span enough space to matter.

---

## Results

| rank | n | TRACK d p2way | COMP d p2way | ratio | p |
|---:|---:|---:|---:|---:|---:|
| 1 | 52 | +0.00083 [+0.00033, +0.00144] | +0.00004 [+0.00001, +0.00009] | **18.5×** | <0.0001 |
| 4 | 52 | +0.00576 [+0.00262, +0.00979] | +0.00007 [+0.00002, +0.00016] | **80.3×** | <0.0001 |
| 8 | 78 | +0.01937 [+0.00865, +0.03466] | +0.00009 [+0.00004, +0.00018] | **207.9×** | <0.0001 |
| 16 | 52 | +0.05948 [+0.03146, +0.09435] | +0.00021 [+0.00007, +0.00045] | **282.0×** | <0.0001 |
| 26 | 52 | +0.13819 [+0.08771, +0.19492] | +0.00036 [+0.00012, +0.00076] | **381.0×** | <0.0001 |

| rank | TRACK steer | COMP steer | TRACK break | COMP break | TRACK d logit | COMP d logit | T\|edit\| | C\|edit\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0% | 0.0% | 26.9% | 9.6% | +0.952 | +0.232 | 16.00 | 15.96 |
| 4 | 0.0% | 0.0% | 34.6% | 9.6% | +2.237 | +0.316 | 16.00 | 15.54 |
| 8 | 0.0% | 0.0% | 33.3% | 5.1% | +3.374 | +0.459 | 16.00 | 15.75 |
| 16 | 0.0% | 0.0% | 42.3% | 9.6% | +5.225 | +0.583 | 15.91 | 16.00 |
| 26 | 0.0% | 0.0% | 48.1% | 9.6% | +6.305 | +0.653 | 15.91 | 15.82 |

Edit norms are matched throughout (15.5–16.0), so no condition is advantaged by budget.

---

## Verdict: DIRECTION, decisively

The gap does not merely persist — it **widens monotonically**, 18.5× → 381×.
This is the *opposite* of the dimensionality prediction.

**Scaling exponents (log-log fit vs rank):**

```
TRACKING    d p2way ~ r^1.58
COMPLEMENT  d p2way ~ r^0.63
ratio                ~ r^0.95
```

Under the dimensionality hypothesis both exponents would match and the ratio
exponent would be ≈ 0. Observed: **0.95**.

**Per-dimension efficiency is the cleanest cut.** If dimensionality were doing
the work, `d p2way / r` would be comparable across conditions. Instead the two
move in *opposite directions*:

| rank | tracking per-dim | complement per-dim |
|---:|---:|---:|
| 1 | 0.000828 | 0.000045 |
| 4 | 0.001440 | 0.000018 |
| 8 | 0.002422 | 0.000012 |
| 16 | 0.003718 | 0.000013 |
| 26 | 0.005315 | 0.000014 |

Tracking dimensions get **6.4× more effective each** as rank grows (they compose
constructively). Complement dimensions get **3.2× less effective each** (they
dilute — random directions partially cancel). Additional random directions add
essentially nothing; additional tracking directions compound.

**Even a single tracking direction beats a 26-dimensional random subspace**:
r=1 tracking `d p2way` = +0.00083 vs r=26 complement = +0.00036 — 2.3× larger
with 1/26th the dimensions. Dimensionality is not what matters.

---

## Secondary finding: BREAK is direction-specific too

| | r=1 → r=26 | corr(rank, BREAK) |
|---|---|---:|
| tracking | 26.9% → 48.1% | **+0.965** |
| complement | 9.6% → 9.6% | +0.166 |

Tracking-subspace BREAK scales almost perfectly with rank while complement BREAK
is flat. So the tracking subspace is *causally potent in both directions* — it
increasingly disrupts clean-prompt computation as more of it is engaged, yet at
**every rank, top-1 STEER remains exactly 0.0%**.

That sharpens the headline considerably:

> Engaging more of the tracking subspace monotonically increases both the
> decision variable (+0.00083 → +0.13819, 167×) and collateral damage
> (26.9% → 48.1%), while top-1 override stays pinned at 0.0% throughout.
> The subspace is causally potent but decision-insufficient — and no amount of
> additional tracking dimensions changes that.

## Tertiary: the metric ladder holds across the whole sweep

Top-1 STEER is 0.0% for **both** conditions at **every** rank — 10 cells, all
identical, all uninformative. The continuous metric separates them at every
single one (5/5 ranks, p<0.0001, ratios 18.5×–381×). A top-1-only analysis would
have reported ten null cells and concluded nothing was happening.

---

## Reproduction

```bash
./run_rank_sweep.sh                    # 16 runs (+ r8 reused from dissociation)
python src/analyze_rank_sweep.py
```

Artifacts: `results/diss_{track,comp}{1,4,8,16,26}_s{0,1,2}.json`,
`results/rank_sweep_summary.json`.

## Status

| Claim | Verdict |
|---|---|
| Gap is about direction, not dimensionality | **Confirmed** — ratio grows r^0.95, 5/5 ranks p<0.0001 |
| Per-dim efficiency diverges between conditions | **Confirmed** — 6.4× up vs 3.2× down |
| One tracking dim > 26 random dims | **Confirmed** — +0.00083 vs +0.00036 |
| Decision-insufficiency holds at every rank | **Confirmed** — 0.0% top-1 in all 10 cells |
| BREAK is direction-specific | **Confirmed** — corr +0.965 vs +0.166 |

## Remaining gaps

1. Two-stage budget-12 norm-shrink is still single-seed (only surviving unreplicated claim).
2. Cross-architecture dissociation (Phi-3.5-mini, Nemotron-Mini-4B) not yet run.
3. r ∈ {1,4,16,26} use 2 seeds vs 3 for r=8 — fine for the trend, worth topping up for camera-ready.
