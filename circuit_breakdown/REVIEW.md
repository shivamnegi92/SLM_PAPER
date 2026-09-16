# REVIEW.md — Simulated ICLR review of "Circuit Breakdown"

I am role-playing a **tough-but-fair ICLR area reviewer**. Scores use the ICLR
rubric. This is deliberately harsh where it needs to be — the point is to find
the ceiling and how to raise it. Updated as the work evolves (see CHANGELOG at
bottom).

## Summary of submission
The paper localizes multi-step **state-tracking** in three ≤4B "small" LMs
(Llama-3.2-3B, Nemotron-Mini-4B, Phi-3.5-mini) using attribution + activation
patching on two minimal-pair task families, and intervenes with a diff-of-means
steering direction. Claims: (i) tracking is carried by a sparse, position-local
set of residual sites; (ii) a "non-trivial" circuit (mid-layer entity position,
excluding the final position / last 2 layers) recovers ~100% of the logit diff
vs 0.1–42% random; (iii) the mechanism is architecture-independent; (iv) a
single-layer vector **breaks** tracking (93–100%) but cannot constructively
**steer** (0%); (v) at break-strength, HellaSwag stays within the baseline CI
while free-form perplexity degrades; (vi) all auditable on a laptop in minutes.

## Scores (ICLR rubric)
- **Soundness: 3/4 (good).** Metrics are standard and honestly reported;
  random/non-trivial controls and CIs are present. Dinged for coarse
  (residual-site, not head-level) granularity and toy tasks.
- **Presentation: 3/4.** Clear figures, honest negative results. Needs a crisper
  central claim.
- **Contribution: 2/4 (fair).** This is the crux — see Weakness W1.
- **Overall rating: 5/10 (marginally below acceptance) for main track as-is;
  7/10 as a strong workshop/Findings paper.**
- **Confidence: 4/5.**

## Strengths
- **S1. Clean, honest empiricism.** Non-trivial circuit control (excludes the
  trivial late-layer readout), random baselines, bootstrap CIs, and a reported
  *negative* result (steering fails) — this is the good kind of interp paper.
- **S2. Cross-architecture universality.** The identical localization across
  BPE (Llama/Nemotron) and SentencePiece (Phi) families is genuinely nice and
  rare; most interp papers study one model.
- **S3. Consumer-hardware auditability, quantified.** Full audit in minutes on a
  laptop, fp32, with wall-clock reported. Good for reproducibility/accessibility.
- **S4. A concrete mechanism.** The "state lives on the entity token, then is
  moved to the final position in late layers" handoff curve is interpretable and
  consistent across tasks/models.

## Weaknesses (ranked by how much they cost the score)
- **W1 (decisive). Incremental novelty / the mechanism is already known.**
  "Information written at the entity/subject token and moved to the final
  position by late 'mover' layers" is essentially the IOI *name-mover* picture
  (Wang et al. 2022) and the entity-tracking findings of Kim & Schuster (2024)
  and Feng & Steinhardt (2023), now replicated on 3B models. **What is the NEW
  scientific claim?** As written, the paper is *replication + consolidation +
  engineering*. For main-track ICLR I need one novel, falsifiable claim.
- **W2. "Circuit" is overclaimed.** Patching whole residual *sites* at one
  position is coarse; k=1 → 100% is trivially the downstream readout. A *circuit*
  means specific **components** (attention heads / MLPs) and **edges** (path
  patching). Without head-level localization + path patching, this is
  *position/layer localization*, not a circuit. Retitle or deliver heads.
- **W3. Toy tasks, possibly too easy.** Templated, single-token answers, few-shot
  base models, ~100% faithfulness everywhere. Is the effect informative or just
  "linearly readable at the entity token"? Need a harder/naturalistic tracking
  task or an argument for ecological validity.
- **W4. The "override" half of the thesis is unmet.** Title promises
  *overriding* tracking nodes; you can break but not constructively steer with a
  portable vector (full patching steers, but that's not a portable edit). The
  intervention story is half-complete.
- **W5. Underpowered capability claim.** n=120 HellaSwag → ±10% CI, so "within
  CI" is weak (low power). Perplexity degrades. Need larger n, ≥2 benchmarks,
  and a clean *targeted-deployment* capability number (should be ≈0 change).

## Questions to authors
- Q1. Which **attention heads** implement the move? Are they the same heads
  (functionally) across architectures, or just the same *depth band*?
- Q2. Does the handoff depth obey a **relative-depth law** (constant fraction of
  total layers) across the 3 models? You have 28- and 32-layer models — test it.
- Q3. Can a **multi-layer** or DAS-derived direction achieve constructive
  steering, closing W4?
- Q4. Targeted (entity-position-only) deployment: what is the *exact* capability
  delta (should be ~0)? Report it separately from the global stress test.

## Conceptual verdict: is the work getting better?
**Yes, materially.** Adding (a) cross-architecture universality and (b) a real
CI'd capability regression directly retired two earlier weaknesses. But the
**ceiling is bounded by W1 (novelty) and W2 (granularity).** To cross from
"borderline / strong workshop" to "clear accept," the single highest-leverage
moves are:

1. **Head-level localization + path patching** (kills W2; upgrades "localization"
   to a real "circuit"; likely reveals a small mover-head set).
2. **One novel claim.** Best candidate given the data: a **relative-depth
   invariance law** — the tracking handoff occurs at a near-constant *fraction*
   of network depth across architectures and scales predictably with task
   compositional depth. That is new, quantitative, and testable with what we
   have.
3. **(Stretch, highest payoff) Bridge to the parent SLM_PAPER thesis:**
   mechanistically show that a *complete tool/interface schema* engages this
   tracking circuit while an *underspecified* one does not — turning "interface,
   not capacity, drives SLM failure" from a black-box claim into a mechanistic
   one. That reframes the paper around a genuinely novel contribution and would
   be a top-tier story.

With #1 + #2 done well, I would raise to **6–7 (accept)**. With #3, **7–8**.

## Recommendation
Current: **borderline reject for main track; accept for a strong workshop
(BlackboxNLP/ATTRIB) or Findings.** Actionable path to accept above.

---
## CHANGELOG
- v1: initial review after localization + intervention + capability + 3x2
  cross-arch matrix. Rating 5/10 (main), 7/10 (workshop).
- v2: authors added (a) **head-level circuit** (attention-head attribution +
  patching; ~8-16 late-layer mover heads recover 72-85%, random-8 = -0.3%) and
  (b) a **relative-depth invariance law** (handoff at ~0.62-0.70 of depth across
  3 architectures). See re-assessment below.

## v2 RE-ASSESSMENT (post head-level + relative-depth)
**What changed my mind:**
- **W2 (circuit overclaim) -> largely RESOLVED.** Head-level attribution +
  patching now shows a *graded* faithfulness curve driven by a small set of
  specific late-layer mover heads, with a near-zero random-head baseline
  (-0.3%). This is a legitimate circuit claim, not coarse layer patching.
- **W1 (novelty) -> PARTIALLY RESOLVED.** The relative-depth invariance
  (~2/3 depth across Llama/Nemotron/Phi, 28 vs 32 layers) is a genuinely new,
  quantitative, falsifiable regularity. It elevates the paper from
  "replication" to "replication + a new empirical law." Caveat: n=3 models is
  suggestive, not conclusive -- needs more scales/families and both tasks to be
  a headline claim rather than an observation.
- **W5 (capability) -> PARTIALLY RESOLVED.** Real benchmark (HellaSwag acc_norm)
  with bootstrap CIs + perplexity + random-direction control; a clear sweet spot
  (100% break, HS within base CI). Still underpowered (n=120 -> +/-10% CI) and
  single-benchmark.

**Still open:** W3 (toy tasks / ecological validity) and W4 (constructive
steering unmet -- can break, cannot install an answer with a portable vector).
Authors attempted a stronger multi-layer additive intervention (entity+final,
layer-set+alpha search): it increases clean-target logits but still yields 0%
argmax STEER, so W4 remains open.

**Revised scores:**
- Soundness: 3/4 -> **3/4** (head-level controls strengthen it; tasks still toy).
- Contribution: 2/4 -> **3/4** (a new law + a real circuit + cross-arch).
- Overall rating: 5/10 -> **6.5/10** (leaning accept for main track; solid
  accept for Findings/BlackboxNLP).
- Confidence: 4/5.

**To reach a confident 7-8 (clear accept):**
1. Push the relative-depth law across >=6 models spanning >=2 scale points, both
   tasks -> turn the observation into a headline law (fit + CI on the ratio).
2. Constructive steering via a multi-layer / DAS direction (close W4), so the
   title's "override" is fully earned.
3. Power up capability (n>=500, add ARC/PIQA) and add the *targeted-deployment*
   delta (expected ~0) alongside the global stress test.
4. (Highest payoff) The parent-thesis bridge: show a complete vs underspecified
   interface schema engages vs bypasses this circuit -> mechanistic evidence for
   "interface, not capacity."
