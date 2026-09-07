# Cross-Architecture Replication — Partial

`run_crossarch2.sh` + `src/intervene_pareto.py`. n=126/seed, 2 seeds for Phi and
Nemotron (n_test=52), 3 seeds for Llama (n_test=78). Layers [17,19,21,23] for the
32-layer models (≈0.66 relative depth), [18,20,22,24] for Llama (28 layers).
`--rel-budget 0.30`, 16 optimizer steps.

## Cross-architecture budget correction (required for any of this to be valid)

Residual-stream norms at the entity position differ by **7.5×** across families:

| model | resid norm | what budget 4.0 means |
|---|---:|---:|
| Llama-3.2-3B | 23.55 | 17% perturbation |
| Nemotron-Mini-4B | 126.36 | 3.2% |
| Phi-3.5-mini | 177.43 | 2.3% |

An absolute norm budget is **not comparable across architectures**. Using
Llama's budget of 4.0 on Phi produced 0% STEER and would have supported a false
"the effect is Llama-specific" conclusion. With `--rel-budget 0.30` (fraction of
measured residual norm) and 16 steps, Phi reaches 40.4% STEER.

**Any cross-model steering comparison that does not normalize by residual norm
is confounded.** This is worth stating in the paper as a methods contribution.

## Results

| model | condition | n | STEER top-1 | STEER 2-way | BREAK | d p2way |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-3B | full | 78 | 62.8% [52.5, 73.1] | 100.0% | 26.9% | +0.9676 |
| Llama-3.2-3B | tracking r8 | 78 | **0.0%** | 0.0% | 33.3% | +0.0194 |
| Llama-3.2-3B | complement r8 | 78 | 0.0% | 0.0% | 5.1% | +0.0001 |
| Phi-3.5-mini | full | 52 | 40.4% [26.9, 53.8] | 98.1% | 26.9% | +0.9742 |
| Phi-3.5-mini | tracking r8 | 52 | **34.6%** [21.2, 48.1] | 67.3% | 90.4% | +0.6337 |
| Phi-3.5-mini | complement r8 | 52 | 0.0% | 1.9% | 1.9% | +0.0268 |
| Nemotron-4B | full | 52 | 23.1% [11.5, 34.6] | 90.4% | 25.0% | +0.8897 |
| Nemotron-4B | tracking r8 | 52 | **1.9%** [0.0, 5.8] | 59.6% | 100.0% | +0.5853 |
| Nemotron-4B | complement r8 | 52 | 0.0% | 0.0% | 1.9% | +0.0029 |

## Claim 1 — Dissociation (full vs tracking subspace): 2 of 3

| model | full | tracking | diff | p | verdict |
|---|---:|---:|---:|---:|---|
| Llama-3.2-3B | 62.8% | 0.0% | +62.8% | <0.0001 | **replicates** |
| Nemotron-4B | 23.1% | 1.9% | +21.2% | 0.0003 | **replicates** |
| Phi-3.5-mini | 40.4% | 34.6% | +5.8% | 0.61 | **does not replicate** |

**On Phi, subspace-confined edits steer nearly as well as unconstrained ones.**
The strong form of the claim — "confining edits to the causally-identified
subspace prevents override" — is false for Phi-3.5-mini.

This must be reported. The defensible claim is now:

> On 2 of 3 architectures, edits confined to the tracking subspace fail to
> override the decision while unconstrained edits in the same layers succeed.
> On Phi-3.5-mini the subspace is decision-*sufficient*, showing that the
> relationship between a causally-identified subspace and decision-level control
> is architecture-dependent rather than universal.

That is a weaker headline but a more interesting paper: the dissociation is a
*property of a model*, not a law of transformers.

### Why Phi may differ (hypotheses, untested)

1. **Larger relative edit.** Phi's rel-budget 0.30 on a norm-177 stream is a much
   larger absolute perturbation; the subspace may simply have enough room.
   *Test: sweep rel_budget on Phi and check whether the dissociation appears at
   lower budgets.* This is the most likely explanation and the cheapest to test.
2. **Rank cap interacts with n_train.** r=8 of a possible 26 may capture a larger
   fraction of Phi's tracking variance.
   *Test: rank sweep on Phi.*
3. **Genuinely different geometry** — Phi is SentencePiece and a different
   pretraining mix; its tracking directions may be better aligned with the
   unembedding.
   *Test: measure cosine between the tracking subspace and the clean−corrupt
   unembedding direction, per model.*

Hypothesis 1 is a confound, not a finding. **It must be ruled out before the
architecture-dependence claim is made.**

## Claim 2 — Tracking ≫ complement on the continuous metric: 3 of 3

| model | tracking | complement | ratio | p |
|---|---:|---:|---:|---:|
| Llama-3.2-3B | +0.01937 | +0.00009 | 207.9× | <0.0001 |
| Phi-3.5-mini | +0.63369 | +0.02677 | 23.7× | <0.0001 |
| Nemotron-4B | +0.58532 | +0.00288 | 203.1× | <0.0001 |

**This replicates everywhere.** The tracking subspace is causally privileged over
a dimension-matched random subspace on every architecture tested, 24–208×,
all p<0.0001. This is now the most robust claim in the paper.

## Claim 3 — Complement is inert: 3 of 3

Complement STEER is 0.0% on all three models; complement BREAK is 5.1% / 1.9% /
1.9%. The control behaves as a control everywhere.

## Claim 4 — BREAK is direction-specific: 3 of 3, and stronger off-Llama

Tracking-subspace BREAK: 33.3% (Llama), 90.4% (Phi), 100.0% (Nemotron) — versus
complement BREAK of 5.1% / 1.9% / 1.9%. On Phi and Nemotron the tracking
subspace is *massively* disruptive. Nemotron tracking edits break the clean
prompt **100% of the time while steering only 1.9%** — the cleanest single
illustration of "causally potent, decision-insufficient" in the whole project.

## Honest status

| claim | replication | verdict |
|---|---|---|
| tracking ≫ complement (continuous) | 3/3, p<0.0001 | **robust** |
| complement is inert | 3/3 | **robust** |
| BREAK is direction-specific | 3/3 | **robust** |
| full ≫ tracking (the dissociation) | **2/3** | **architecture-dependent** |
| rel-budget needed for cross-model comparison | 3/3 | **methods contribution** |

## Reproduction

```bash
./run_crossarch2.sh
```

Artifacts: `results/xa_{phi-3.5-mini,nemotron-mini-4b}_{full,track8,comp8}_s{0,1}.json`.
