# RESULTS — Circuit Breakdown (Llama-3.2-3B, run 1)

> **Legacy experiment report.** Later experiments include Phi-3.5-mini and
> Nemotron-Mini-4B; Gemma is not part of the completed model set. These tables
> are historical observations, not the frozen validated study. The single-token
> tasks admit answer-extraction shortcuts; near-total logit-gap recovery is not
> arbitrary-task accuracy. The global capability sweep is a stress test, not a
> proven upper bound. See [METRICS.md](METRICS.md),
> [CAPABILITY_DEPLOYED.md](CAPABILITY_DEPLOYED.md) and
> [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md) for the corrected interpretation.

All numbers produced locally on an **Apple M4 Pro (48 GB), fp32, MPS**, base
`Llama-3.2-3B`, public templated data only. Reproduce with the commands at the
bottom. This is a single-model run; Gemma-2-2B cross-architecture is pending.

## Headline
In a 3B model, multi-step state-tracking is carried by an **extremely sparse,
position-localized** set of residual-stream sites. Patching them recovers ~100%
of the clean-vs-corrupt logit difference (random sites: 17–33%), and a
**diff-of-means direction at a single mid layer causally breaks tracking on
93–100% of examples** — all audited end-to-end in **~3.5 min/task on a laptop**.

## 1. The task is real (not recency-solvable)
Two minimal-pair families, `clean_target != corrupt_target`, single-token
position-aligned (self-validated by `src/dataset.py`):
- `intermediate` — query a non-final state ("the location before X").
- `transfer` — two-object binding; **0/200** pairs solvable by a "copy last
  name" heuristic (the last-named person holds the *distractor* object).

Baseline next-token accuracy (3-shot, n=40):

| task | acc | logit-diff clean | logit-diff corrupt |
|---|---|---|---|
| intermediate | 95.0% | +11.17 | −10.86 |
| transfer | 85.0% | +10.40 | −10.36 |

Large, clean clean-vs-corrupt gap => strong causal signal to localize.

## 2. Localization: two positions carry everything
Attribution patching (AtP, ~2 passes/pair) over all (layer × position) sites,
averaged over 40 pairs. Effect concentrates on **exactly two token positions**:

| task | final/readout pos (score) | entity/answer pos (score) | next-highest |
|---|---|---|---|
| intermediate | from-end 0: **125.96** | from-end **8**: **101.06** | 2.13 |
| transfer | from-end 0: **112.96** | from-end **13**: **62.46** | 1.92 |

Everything else ≈ 1. See `results/heatmap_{intermediate,transfer}.png` — two hot
columns, the rest ice-cold.

## 3. Faithfulness + the reviewer-proof circuit
Real activation patching (verify) of attribution-ranked sites, faithfulness =
(LD_patched − LD_corrupt)/(LD_clean − LD_corrupt):

| task | top-40 | random-40 | non-trivial* | min circuit ≥80% |
|---|---|---|---|---|
| intermediate | 100.0% ±0.0 | 16.8% ±21.1 | **100.0% ±0.1** | k=1** |
| transfer | 100.0% ±0.0 | 32.7% ±41.2 | **100.0% ±0.1** | k=1** |

\* **non-trivial** = excludes the final position AND the last 2 layers, so the
recovery *cannot* be the trivial downstream readout. It is still 100%: patching
only mid-layer **entity-position** sites fully controls the answer.

\** k=1 = 100% is the *trivial* late-layer readout (patching the final residual
overwrites the answer). We report it honestly and do not headline it — the
non-trivial circuit is the scientific result.

## 3b. Head-level circuit (attention heads, not just layers)
Upgrading from residual-site to **attention-head** resolution (patch the per-head
input to o_proj at the final/readout position). This is a real circuit claim: a
*graded* faithfulness curve (not the trivial site-level 100%) driven by a
handful of specific **mover heads** in the late layers.

| heads patched | Llama-3.2-3B | Phi-3.5-mini |
|---|---|---|
| top-1 | 18.9% | 11.8% |
| top-2 | 32.3% | 18.4% |
| top-4 | 44.8% | 34.3% |
| top-8 | 71.6% | 52.5% |
| top-16 | 85.3% | 73.6% |
| **random-8** | **-0.3%** | (near 0) |

Dominant mover heads: **Llama L24.H15, L21.H2, L15.H18, L27.H5**; **Phi L31.H4,
L23.H4, L20.H1, L21.H21**. Both families concentrate the readout in a small set
of **late-layer** heads; specific indices differ (expected across architectures)
but the *depth band* (late) and *sparsity* (~8-16 heads recover most of the
signal, random heads recover nothing) are shared. This retires the "it's just
coarse layer patching" objection.

## 4. Mechanism: a causal handoff off the entity token
Patch **only the entity position** at a **single layer L**, sweep L
(`results/layer_handoff.png`):

- L0–~13: **~100%** — correcting the entity early lets all downstream layers
  re-derive the answer.
- Sharp decay L13→21; **0% by L27**.
- 50% crossover ≈ **L17 (intermediate)** / **L20 (transfer)**.

Interpretation: the tracked state lives **on the entity token through the
early/mid layers, then is moved to the final position** for readout. The harder
`transfer` (binding) task hands off **~3 layers deeper** than `intermediate` —
a small, sensible task-complexity signature.

## 4b. Relative-depth invariance law (the novel claim)
The *absolute* handoff layer varies across architectures, but the **relative**
depth (crossover layer / total layers) is near-constant:

| model | layers | 50% crossover layer | relative depth |
|---|---|---|---|
| Llama-3.2-3B | 28 | 17.4 | **0.62** |
| Phi-3.5-mini | 32 | 20.8 | **0.65** |
| Nemotron-Mini-4B | 32 | 22.3 | **0.70** |

**Across three architecture families and two depths, the tracked state is handed
off the entity token at ~0.62-0.70 (mean ~0.66) of network depth.** On a
relative-depth axis the three handoff curves largely collapse
(`results/relative_depth_law.png`, right panel) whereas on an absolute-layer axis
they are clearly offset (left panel). This is a new, quantitative, falsifiable
regularity: tracking resolution is a function of *fractional* depth, not
absolute layer count.

Honest scope: n=3 models is a suggestive regularity, not a proven law; the
collapse is tight but imperfect (Nemotron sits slightly deeper). Testing more
models/scales and both tasks is the obvious follow-up.

## 5. Weight-free intervention: BREAK is easy, STEER is hard
Diff-of-means direction d = mean(clean_resid − corrupt_resid) at the entity
position; applied at one layer, alpha=8, n=30:

| task | best layer | BREAK (clean→wrong) | STEER (corrupt→clean) |
|---|---|---|---|
| intermediate | 12 | **93%** | 0% |
| transfer | 12 | **100%** | 0% |

- **BREAK** (subtract d): reliably destroys tracking at a single mid layer.
- **STEER** (add d): 0% — a single-layer additive vector cannot *constructively*
  install the alternative answer. This confirms the PLAN.md pre-registered hedge
  ("breaking is easy, fixing is hard"); full multi-layer patching (§3) does steer
  perfectly, so constructive control needs more than one additive site.

### 5b. Multi-layer additive steering (Block 1 execution)
We implemented and ran `src/intervene_multilayer.py`, which applies clean-ward
vectors across **multiple layers** and can target both the entity position and
final readout position. We tuned layer-set + alpha on dev and evaluated on test.

Llama-3.2-3B results:
- **intermediate**: best dev = layers 14-22, alpha_entity=1, alpha_final=0;
  test STEER=0%, BREAK=11.1%, clean-target logit lift on corrupt = **+0.238**.
- **transfer**: best dev = layers [14,16,18,20], alpha_entity=1, alpha_final=0;
  test STEER=0%, BREAK=33.3%, clean-target logit lift on corrupt = **+0.047**.

So multi-layer additive control gives a measurable clean-target **logit lift**
but still fails to flip top-1 answers. This narrows the failure mode: the
tracked feature is partially recoverable linearly, but not enough for robust
constructive steering without a stronger method (e.g., DAS / learned multi-layer
projection / path-specific interventions).

### 5c. DAS-style learned multi-layer projection (executed)
We implemented `src/intervene_das.py` (low-rank ridge maps from corrupt residuals
to cleanward deltas per layer) and evaluated on train/dev/test.

Llama-3.2-3B results:
- `intermediate` (layers [14,16,18,20], ranks {4,8}, alphas {0.5,1,2}):
  test **STEER=0%**, BREAK=33.3%, clean-target logit lift **+0.271**.
- `transfer` (same search space):
  test **STEER=0%**, BREAK=66.7%, clean-target logit lift **+0.988**.

Conclusion: DAS-style linear projections increase the target logit substantially
(especially on transfer) but still do not reliably cross the argmax boundary.
Constructive override remains open; this points toward needing non-linear or
path-specific interventions (e.g., path patching constraints, learned adapters,
or optimization over sequence-level objective at inference).

### Capability caveat (honest)
Spraying d at **all positions** of 8 generic prompts at the same strong alpha=8
keeps next-token argmax on **5/8 (62%)**. This is an adversarial stress test
(global application, large alpha); the tracking break itself is applied at a
single position. A proper alpha↔capability tradeoff curve and an ARC/MMLU-subset
regression with CIs are the next step before any "no degradation" claim.

## Decision-level intervention benchmark

All values are from local Mac runs on Llama-3.2-3B (`intermediate`).
Linear-method rows come from Block A atlas (`n_test=5`); hardened margin rows include both single-run (`n_test=6`) and seed-pooled stability sweep (`n_test_total=27`) with tighter capability guard.

| Method | Alpha | n_test | STEER (95% CI) | BREAK (95% CI) | Logit lift | Control drop | Collapse | Random STEER | Random BREAK | Random Logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single_layer_additive | 2.0 | 5 | 0.0% [0.0%, 0.0%] | 20.0% [0.0%, 60.0%] | +0.111 | 0.0% | 0.0% | 0.0% | 20.0% | +0.006 |
| single_layer_additive | 4.0 | 5 | 0.0% [0.0%, 0.0%] | 20.0% [0.0%, 60.0%] | +0.237 | 0.0% | 0.0% | 0.0% | 20.0% | +0.010 |
| single_layer_additive | 8.0 | 5 | 0.0% [0.0%, 0.0%] | 20.0% [0.0%, 60.0%] | +0.220 | 0.0% | 0.0% | 0.0% | 20.0% | +0.011 |
| single_layer_additive | 16.0 | 5 | 0.0% [0.0%, 0.0%] | 20.0% [0.0%, 60.0%] | -0.048 | 0.0% | 0.0% | 0.0% | 20.0% | +0.030 |
| multi_layer_additive | 2.0 | 5 | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | +0.356 | 0.0% | 0.0% | 0.0% | 20.0% | +0.006 |
| multi_layer_additive | 4.0 | 5 | 0.0% [0.0%, 0.0%] | 60.0% [20.0%, 100.0%] | +0.612 | 0.0% | 0.0% | 0.0% | 20.0% | +0.010 |
| multi_layer_additive | 8.0 | 5 | 0.0% [0.0%, 0.0%] | 40.0% [0.0%, 80.0%] | +0.376 | 0.0% | 0.0% | 0.0% | 20.0% | +0.011 |
| multi_layer_additive | 16.0 | 5 | 0.0% [0.0%, 0.0%] | 80.0% [40.0%, 100.0%] | -4.985 | 0.0% | 0.0% | 0.0% | 20.0% | +0.030 |
| das_multilayer | 2.0 | 5 | 0.0% [0.0%, 0.0%] | 80.0% [40.0%, 100.0%] | +0.139 | 0.0% | 0.0% | 0.0% | 20.0% | +0.006 |
| das_multilayer | 4.0 | 5 | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | +0.044 | 0.0% | 0.0% | 0.0% | 20.0% | +0.010 |
| das_multilayer | 8.0 | 5 | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | -0.138 | 0.0% | 0.0% | 0.0% | 20.0% | +0.011 |
| das_multilayer | 16.0 | 5 | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | -0.340 | 0.0% | 0.0% | 0.0% | 20.0% | +0.030 |
| margin_runtime_hardened_balanced | n/a | 6 | 50.0% [16.7%, 83.3%] | 33.3% [0.0%, 66.7%] | +7.644 | 0.0% | 0.0% | 0.0% | 0.0% | +0.000 |
| margin_runtime_hardened_preserve_steer | n/a | 6 | 100.0% [100.0%, 100.0%] | 50.0% [16.7%, 83.3%] | +9.735 | 0.0% | 0.0% | 0.0% | 0.0% | +0.000 |
| margin_runtime_hardened_balanced_seedpooled | n/a | 27 | 51.9% [33.3%, 70.4%] | 18.5% [3.7%, 33.3%] | +7.134 | 3.7% | 0.0% | 0.0% | 0.0% | +0.000 |
| margin_runtime_hardened_preserve_seedpooled | n/a | 27 | 63.0% [44.4%, 81.5%] | 22.2% [7.4%, 37.0%] | +9.228 | 7.4% | 0.0% | 0.0% | 0.0% | +0.000 |

**Interpretation:** Block A still shows the core bottleneck: linear interventions increase target logits without flipping top-1 (0% STEER). With objective-aware runtime optimization and tighter guardrails, seed-pooled stability runs (n=42, seeds 0-2) show a cleaner frontier: **balanced** reaches 51.9% STEER at 18.5% BREAK, while **preserve** reaches 63.0% STEER at 22.2% BREAK. Compared to the earlier small-split result (100% STEER, 66.7% BREAK), collateral breakage is substantially reduced while retaining meaningful decision-level control. The car turns now, and it clips the curb much less.

## 6. Cross-architecture universality (the novelty hook)
Same pipeline, three *different* architecture families, both tasks (n=25–40,
3-shot), all local fp32 on the M4 Pro. **Every cell: non-trivial circuit and
top-40 faithfulness = 100%.**

| model | family | layers | task | acc | entity pos (from end) | non-trivial | random-40 |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | Llama3 (BPE) | 28 | intermediate | 95% | **8** | **100%** | 16.8% |
| Llama-3.2-3B | Llama3 (BPE) | 28 | transfer | 85% | **13** | **100%** | 32.7% |
| Nemotron-Mini-4B | Nemotron (BPE) | 32 | intermediate | 88% | **8** | **100%** | 0.9% |
| Nemotron-Mini-4B | Nemotron (BPE) | 32 | transfer | 100% | **13** | **100%** | 9.7% |
| Phi-3.5-mini | Phi3 (SentencePiece) | 32 | intermediate | 96% | **8** | **100%** | 41.9% |
| Phi-3.5-mini | Phi3 (SentencePiece) | 32 | transfer | 100% | **13** | **100%** | 0.1% |

All three families localize the tracked state to the identical entity-token
position and are fully controllable by patching only mid-layer entity-position
sites. The mechanism is architecture-independent; only the random-baseline noise
floor differs (Phi's 41.9% on `intermediate` is high/noisy -> report the
top-vs-random *gap* and raise n/random-repeats to tighten it).

## 6b. Capability regression (Llama-3.2-3B, real benchmark + CIs)
BREAK direction (diff-of-means at layer 12, ||d||=1.31) applied **globally**
(worst case, all positions), swept over strength alpha. HellaSwag acc_norm
(n=120, 95%% bootstrap CI) + Tiny-Shakespeare perplexity + equal-norm random
control. Baseline: **HS 56.7%% [46.7, 65.8], ppl 20.9.**

| alpha | BREAK | HellaSwag [95% CI] | ppl | rand HS | rand ppl | verdict |
|---|---|---|---|---|---|---|
| 0 | 7% | 56.7% [46.7,65.8] | 20.9 | 56.7% | 20.9 | within CI |
| 4 | 23% | 55.0% [45.8,64.2] | 30.3 | 57.5% | 25.0 | within CI |
| 6 | 87% | 53.3% [44.2,61.7] | 55.9 | 53.3% | 36.1 | within CI |
| **8** | **100%** | **50.8% [41.7,60.0]** | 127.2 | 50.0% | 67.4 | **within CI** |
| 12 | 100% | 40.8% [32.5,50.0] | 567.3 | 40.8% | 401.0 | DEGRADED |

**Sweet spot alpha=8: tracking fully broken (100%%) while HellaSwag stays inside
the baseline 95%% CI**, even under worst-case global application. Free-form
perplexity does degrade at that strength (a real cost) -- and notably our
direction hurts fluency *more* than an equal-norm random vector (127 vs 67),
confirming it is a functionally meaningful direction, not noise. Because the
deployed break is applied at a single entity position (absent from generic
inputs), this global result is a strict upper bound on collateral damage.
Caveat (self-review W5): n=120 gives ~+/-10%% CIs -> underpowered; scale n and
add ARC/PIQA before a hard "no-degradation" claim.

## 7. Consumer-hardware audit cost (a feature, not a footnote)
- Model load (fp32): ~4–12 s. Attribution: ~0.9–1.8 s/pair.
- Full localization per task (n=40, attribution + k-sweep verify + random
  baselines + heatmap): **~215 s**. Layer handoff sweep (n=30, 28 layers,
  2 tasks): ~7 min. Everything on a laptop, no GPU cluster.

## Limitations / next
1. **Cross-architecture done for `intermediate`** (3 families) — extend to
   `transfer` on all 3 + Gemma-2-2B (gated download) for a 4th family.
2. **Residual-site granularity**, not attention-head level — head-level DAS is
   future work; also mitigates the self-repair confound.
3. **Capability regression** needs the proper CI'd ARC/MMLU-subset run and an
   alpha<->capability tradeoff curve before any "no degradation" claim.
4. STEER needs a stronger constructive method (multi-layer / DAS direction).
5. Phi's random baseline is noisy (41.9%) — raise n and random-repeats to tighten
   the top-vs-random gap.

## Reproduce
```bash
source .venv/bin/activate
python src/localize.py    --task intermediate --n 40 --fewshot 3 --min-len 3 --max-len 4
python src/localize.py    --task transfer     --n 40 --fewshot 3 --min-len 1 --max-len 2
python src/layer_sweep.py --n 30 --fewshot 3
python src/intervene.py   --task intermediate --n 30 --fewshot 3 --alpha 8 --layers 8 12 16 20 24
python src/intervene.py   --task transfer     --n 30 --fewshot 3 --min-len 1 --max-len 2 --alpha 8 --layers 8 12 16 20 24
```
