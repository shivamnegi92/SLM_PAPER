# Paper Proposal — Efficient SLMs for Intent + Slot Filling

> Public-benchmark-only. See `COMPLIANCE.md`. No proprietary references anywhere.

## Recommendation summary
- Draft **Paper 1 (primary)** first, then **Paper 2 (secondary)**.
- Prove *every* claim on public benchmarks: ATIS, SNIPS, MASSIVE, CLINC150,
  BANKING77.
- **Venue split is now fixed by the ICLR 1-submission cap** (no team author has a
  prior top-venue paper, decided 2026-08-07):
  - **Paper 1 -> ICLR 2027** (the single allowed submission).
  - **Paper 2 -> AISTATS 2027 (Oct 8) or EMNLP/Findings / *SEM / NER workshop** —
    NOT ICLR 2027.

---

## Paper 1 (PRIMARY)

**Working title:** *Generate Less, Classify More: Probe-Guided Pruning and
Discriminative Heads for Efficient Intent Detection and Slot Filling with Small
Language Models*

**Thesis.** For closed-set intent detection and slot filling, an autoregressive
generative SLM is unnecessarily slow and unreliable. Converting it into a
**discriminative single-pass model** — via probe-guided layer pruning plus a
linear intent head and a CRF BIO slot head — yields large CPU latency reductions,
matches or improves accuracy, and eliminates structured-output parse failures.

**Contributions.**
1. **Probe-guided depth selection.** Use a frozen linear probe layer sweep to
   choose how many transformer layers to retain, quantifying that intent
   detection is a shallow task while slot filling needs more depth.
2. **Generative → discriminative conversion** as a *reliability* result: the
   autoregressive JSON parse-failure mode is removed by construction.
3. **Implicit-slot recovery** (Paper 2's core, summarized here): intent-
   conditioned rules recover non-span slots at negligible cost.
4. **A negative-results study** of common CPU inference optimizations (dynamic
   INT8, rotation-based quantization, ONNX/graph export, graph compilation,
   vendor CPU extensions) with root-cause analysis, on public models.

**Novelty positioning.** Not "we pruned a model." The novelty is (a) a *decision
procedure* for pruning depth via probing, (b) reframing generation removal as a
correctness/reliability guarantee rather than only a speed hack, and (c) the
implicit-slot ceiling analysis.

---

## Paper 2 (SECONDARY)

**Working title:** *Implicit Slots: The Structural Recall Ceiling of Span-Based
Slot Filling*

**Thesis.** A non-trivial fraction of gold slots in task-oriented dialogue
datasets are **not literal text spans** (they are implied by context, pronouns,
or intent). Span-based taggers (BIO/CRF and extractive LLM prompting) therefore
have a hard, measurable recall ceiling. We quantify this per dataset, decompose
false negatives into "implicit" vs "genuine miss," and show intent-conditioned
rules recover implicit slots cheaply.

**Venue fit:** *SEM, an NER / efficient-NLP workshop, or ACL/EMNLP Findings.

---

## What each paper needs before submission
See `experiments/EXPERIMENT_PLAN.md`. In short: full benchmark grid, strong
public baselines (encoder discriminative models + same-size generative model),
ablations (depth sweep, CRF vs softmax, head-LR, slot-loss weighting, seed
variance), and latency measured on a clearly-specified commodity CPU.
