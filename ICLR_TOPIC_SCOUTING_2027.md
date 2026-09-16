# ICLR Topic Scouting (CPU-only, no API) — SLM/GenAI/VLM

Context anchored to current repo direction (`Generate Less, Classify More`) and your constraints.

## Hard constraints (non-negotiable gates)
1. **AI/ML core contribution** (not just systems engineering).
2. **No LLM API dependency** (fully local/open models).
3. **No GPU requirement** (research must run on commodity CPU).
4. **Technical GenAI/SLM/VLM focus** with strong inference relevance.

---

## 30 candidate topics (with quick refute/discuss)

| # | Topic | Why it aligns | Refute / risk |
|---|---|---|---|
| 1 | Probe-guided dynamic depth per input | Directly extends your probe work; pure inference win | Could look incremental if not paired with theory/guarantees |
| 2 | Uncertainty-aware early exit for SLM intent+slot | Strong CPU latency story + reliability | Calibration quality may collapse OOD |
| 3 | Layer skipping with confidence envelopes | Inference-only, no API/GPU | Hard to prove robust under class imbalance |
| 4 | Token-level adaptive compute for slot tagging | Very technical and novel for decoder-style SLMs | Implementation complexity vs payoff |
| 5 | Structured discriminative decoding from generative backbone | Your strongest existing signal | Risk: reviewers ask "why not just encoder?" |
| 6 | Parse-failure formalization + elimination guarantees | Reliability contribution, ICLR-friendly framing | Needs tight theorem/empirical bridge |
| 7 | Budget-aware latency controller (ms target -> depth policy) | Practical + principled deployment angle | Might feel too systems unless math-heavy |
| 8 | Distillation-free pruning via probe separability | No retraining-heavy pipeline; CPU friendly | Gains may be smaller on harder datasets |
| 9 | Joint intent-slot confidence decomposition | Better abstention/error handling | Needs careful calibration metrics |
| 10 | Loss-balanced multi-head optimization under shallow depth | Strongly technical and reproducible | Might be seen as optimization tuning |
| 11 | CRF vs non-CRF transition modeling under causal attention | Clean ablation paper | Novelty limited unless generalized |
| 12 | Inference-time ensembling of shallow checkpoints (CPU) | No GPU needed; can improve robustness | Latency overhead could kill narrative |
| 13 | ONNX export robustness for decoder discriminative heads | You already have evidence | Tooling-heavy unless framed algorithmically |
| 14 | Quantization failure taxonomy for small decoders | Honest negative-results paper | Could be workshop-y unless broader theory |
| 15 | Dynamic quantization with confidence fallback | Inference + reliability | Benefit may be marginal on shallow models |
| 16 | CPU cache-aware sequence packing for small SLM inference | Real latency wins possible | Systems-heavy, weak ML novelty risk |
| 17 | Intent granularity vs required depth scaling law | Very ICLR-friendly scientific question | Must show across many datasets/tasks |
| 18 | Label-space complexity as a predictor of compute need | Novel analytical framing | Could be noisy without strong proxies |
| 19 | Task-adaptive router: depth policy from probe features | Strong blend of ML + inference | Router overhead can erase latency gains |
| 20 | Risk-controlled abstaining inference for SLM NLU | Reliability + safety angle | Benchmark choice critical |
| 21 | VLM patch/token pruning for small vision-language encoders (CPU) | Satisfies VLM requirement directly | Need a small, CPU-runnable VLM benchmark suite |
| 22 | Text-conditional visual token dropping in tiny VLMs | Novel and technical | Dataset/tooling setup burden |
| 23 | Dynamic resolution routing for CPU VLM inference | Inference and cost tradeoff | Could drift into CV systems work |
| 24 | Cross-modal early exit in tiny VLMs | Nice novelty if done well | High engineering burden with limited compute |
| 25 | Confidence-calibrated VLM refusal under low visual evidence | Reliability contribution | Requires careful calibration ground truth |
| 26 | Distilling structured outputs into VLM classifiers | GenAI-to-discriminative concept reuse | Distillation setup may quietly need GPU |
| 27 | Prompt-free structured VLM decision heads | Strongly technical, inference-focused | Data labeling complexity |
| 28 | Unified SLM+VLM compute budget scheduler on CPU | Broad and practical | Scope explosion for one ICLR paper |
| 29 | Pareto frontier benchmark suite for CPU tiny GenAI models | Community value | Benchmark paper novelty risk |
| 30 | Negative-results compendium: when optimizations fail on tiny GenAI | High honesty value | Not enough novelty alone for ICLR main |

---

## How we got from 30 -> 20 (conditions for top-20)
A topic stayed in the top-20 only if it passed **all**:
- **C1:** Can be done end-to-end on CPU in <2 weeks of experiment runtime.
- **C2:** Needs no proprietary data and no external API calls.
- **C3:** Has a measurable primary metric pair: *(quality, latency)*.
- **C4:** Has a reliability angle (calibration, failures, validity, abstention, etc.).
- **C5:** Has plausible ICLR novelty beyond pure engineering.

Dropped after gate review: #12, #16, #23, #26, #27, #28, #29, #30, plus two low-novelty variants merged into #11/#14.

---

## Shortlisted 20 topics (post-gating)
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 24

---

## 20-loop alignment/refute cycle (condensed)

| Loop | Stress test used | Leader after loop | Refute verdict |
|---|---|---|---|
| 1 | Constraint fit (no API/no GPU) | #5 | Pass |
| 2 | Incrementality penalty | #17 | #5 still strong, but needs broader science claim |
| 3 | Reproducibility risk | #17 | Pass |
| 4 | Latency-vs-accuracy signal strength | #1 | Pass |
| 5 | Reliability novelty | #6 | Pass |
| 6 | Reviewer objection simulation: "just use encoder" | #17 | #5 vulnerable alone |
| 7 | Cross-dataset generality | #17 | Pass |
| 8 | Mathematical framing potential | #18 | Pass with caveats |
| 9 | CPU implementation burden | #1 | Easier than #21/#24 |
| 10 | Ablation clarity | #1 | Strong |
| 11 | VLM inclusion pressure | #21 | Attractive but high setup cost |
| 12 | Risk of underpowered empirical evidence | #17 | #21 drops |
| 13 | Camera-ready extensibility | #17 | Strong |
| 14 | Negative-results integration value | #14 + #17 combo | #14 not standalone winner |
| 15 | Ethical/safety discussion depth | #20 | Good but weaker core novelty |
| 16 | Surprise factor for ICLR reviewers | #17 | Strong |
| 17 | Alignment with your current codebase assets | #1/#17 tie | Both very practical |
| 18 | Time-to-first-paper-quality result | #1 | Fastest path |
| 19 | Thesis strength (single sentence claim) | #17 | Best scientific framing |
| 20 | Final weighted score (novelty 35, feasibility 25, evidence 25, fit 15) | **#17** | **Winner** |

---

## Best topic recommendation
## **Topic #17: Intent Granularity vs Required Depth Scaling Law for Small Generative Backbones Converted to Discriminative Inference**

Why this is the best pick:
- Naturally extends your strongest empirical story (probe-guided depth).
- Feels like **science**, not just tuning: predicts compute need from task complexity.
- CPU-only feasible using current pipeline and public datasets.
- Lets you include reliability (parse-failure elimination, calibration) as secondary contributions.

### Minimal thesis statement
> "For closed-set GenAI-to-discriminative SLM inference, the minimum useful depth scales with label-space complexity; this relation can be predicted cheaply by frozen probes and used to enforce latency budgets with bounded quality loss."

### Suggested paper skeleton (tight)
1. **Scaling hypothesis:** depth requirement increases with label granularity/ambiguity.
2. **Predictor:** probe separability metrics estimate required depth pre-fine-tune.
3. **Policy:** choose depth under explicit latency budget.
4. **Reliability:** compare failure modes vs generative decoding.
5. **Generalization:** intent-only + intent-slot datasets (+ optional tiny VLM extension if time permits).

---

## If you want pure speed to submission
If timeline gets brutal, go with **#1** (dynamic depth per input) as fallback. It is easiest to execute quickly using your existing artifacts.

Generated by code-puppy for `/SLM_PAPER` topic scouting.