# Paper 1 (PRIMARY) — Full Draft Skeleton

**Working title:** *Generate Less, Classify More: Probe-Guided Pruning and
Discriminative Heads for Efficient Intent Detection and Slot Filling with Small
Language Models*

> Public-benchmark-only. Every result cell is `TBD` until produced on the datasets
> in `../experiments/datasets.md`. No proprietary references (see `../COMPLIANCE.md`).

---

## Abstract (skeleton)
Task-oriented dialogue increasingly uses small language models (SLMs) that emit
structured output (intent + slots) via autoregressive generation. We show that
for *closed-set* intent detection and slot filling this is both slow and brittle.
We present a recipe that converts a generative SLM into a **discriminative
single-pass** model: (1) a frozen-probe layer sweep selects a pruned backbone
depth, (2) a linear intent head and a CRF BIO slot head replace token-by-token
generation, and (3) intent-conditioned rules recover non-span ("implicit") slots.
Across ATIS, SNIPS, MASSIVE, CLINC150, and BANKING77 we obtain [TBD]x CPU speedup
at equal-or-better accuracy while eliminating structured-output parse failures.
We further contribute a negative-results study of common CPU inference
optimizations and their failure root causes.

## 1. Introduction
- Trend: generative SLMs for structured NLU. Cost: autoregressive decode latency,
  and non-zero rate of malformed structured output.
- Claim: closed-set intent + slot filling does not need generation.
- Contributions bullet list (see PROPOSAL.md).

## 2. Related Work
- Intent detection & slot filling (joint models, JointBERT, slot-gated, etc.).
- Constrained / structured decoding for LLMs.
- Model compression: layer pruning, distillation, quantization (GPTQ/AWQ/QuaRot).
- Probing internal representations (linear probes, layer-wise analysis).
- Position vs each: we combine probing→pruning→discriminative heads and study the
  reliability angle, not just FLOPs.

## 3. Method
### 3.1 Problem setup
- Closed intent set; slot set with BIO tagging; single-pass discriminative output.
### 3.2 Probe-guided depth selection
- Freeze base SLM; train linear probe on hidden states at layers {L/4, L/2, 3L/4, L}.
- Pick smallest depth within ε of full-depth probe accuracy → prune to that depth.
### 3.3 Discriminative heads
- Intent: `Linear(d_model → |intents|)`, argmax.
- Slots: `Linear(d_model → |BIO tags|) + CRF`, Viterbi decode.
- Joint loss: `L = L_intent + λ · L_slot(CRF-NLL)`, with slot-token upweighting.
### 3.4 Implicit-slot recovery (summary; full treatment in Paper 2)
- Intent-conditioned deterministic rules for non-span slots; precision-first.
### 3.5 Why generation removal is a reliability win
- Discriminative output is structurally valid by construction → 0% parse failure.

## 4. Experimental Setup
- Datasets: ATIS, SNIPS, MASSIVE, CLINC150 (intent), BANKING77 (intent),
  ATIS/SNIPS/MASSIVE (slots). See `../experiments/datasets.md`.
- Base models (public, licensed): e.g., a small Llama-family / SmolLM / Qwen-small
  checkpoint + an encoder baseline (DistilBERT/MiniLM). Exact list in datasets.md.
- Metrics: intent accuracy, slot F1 (span-level), joint/exact-match, latency P50/P95
  on a specified commodity CPU, model size.
- Hardware fully specified for reproducibility.

## 5. Results
### 5.1 Main comparison (TBD)
| Model | Intent Acc | Slot F1 | Exact Match | Lat P50 (CPU) | Size |
|---|---|---|---|---|---|
| Generative SLM (full depth) | TBD | TBD | TBD | TBD | TBD |
| Encoder discriminative (DistilBERT) | TBD | TBD | TBD | TBD | TBD |
| **Ours: probe-pruned discriminative** | TBD | TBD | TBD | TBD | TBD |

### 5.2 Probe layer sweep (TBD)
- Table: probe accuracy vs depth → justifies chosen pruned depth.

### 5.3 Ablations (TBD)
- CRF vs plain softmax slot head.
- Slot-loss weight λ sweep.
- Pruned depth sweep (quality/latency Pareto).
- Seed variance.
- With / without implicit-slot recovery.

### 5.4 Reliability (TBD)
- Parse-failure rate: generative vs discriminative (expected 0% for ours).

## 6. Negative Results: CPU Inference Optimizations That Didn't Pay Off
For each, report result + root cause on public models:
- Naive dynamic INT8 (activation outliers).
- Rotation-based INT8 (QuaRot/Hadamard) — accuracy recovered, speed needs fused
  kernels.
- ONNX / graph export for decoder SLMs (dynamic KV cache / dynamic shapes).
- Graph compilation (codegen edge cases on decoder architectures).
- Vendor CPU extensions (instruction-set dependence).
- Takeaway: for shallow closed-set tasks on commodity CPUs, depth pruning +
  discriminative heads dominates.

## 7. Discussion & Limitations
- Closed-set assumption; open-set / novel-intent detection out of scope.
- Generation still needed for free-form generation tasks.
- Implicit-slot rules are dataset/intent specific (see Paper 2).

## 8. Conclusion
- Restate: generate less, classify more.

## Appendix
- Full hyperparameters, per-dataset tables, reproduction commands.
