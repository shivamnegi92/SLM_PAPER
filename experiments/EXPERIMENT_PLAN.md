# Experiment Plan (Public Benchmarks Only)

> No proprietary data or references. See `../COMPLIANCE.md`.

## Goal
Produce every number claimed in Paper 1 and Paper 2 on public datasets, with a
fully specified, reproducible setup.

## Phase A — Baselines (Paper 1)
1. **Generative SLM baseline.** Fine-tune a small public decoder SLM to emit
   `{"intent": ..., "slots": {...}}` autoregressively. Measure intent acc, slot F1,
   exact match, latency P50/P95 on the target CPU, and **parse-failure rate**.
2. **Encoder discriminative baseline.** JointBERT-style DistilBERT/MiniLM
   (linear intent head + BIO slot head). Same metrics.

## Phase B — Probe-Guided Pruning (Paper 1 core)
3. Freeze the base decoder SLM; train linear probes on hidden states at depths
   {L/4, L/2, 3L/4, L} for intent. Record probe accuracy vs depth.
4. Select pruned depth = smallest depth within ε (e.g., 1–2 pts) of full-depth
   probe accuracy.
5. Fine-tune the pruned backbone + linear intent head + CRF BIO slot head.

## Phase C — Implicit-Slot Recovery (Papers 1 & 2)
6. Run the implicit-slot detector (value not a substring of the utterance) on each
   dataset → implicit %, theoretical max span recall.
7. Decompose CRF tagger FNs into implicit vs genuine miss.
8. Add intent-conditioned rules; measure ΔF1, rule precision, latency overhead.

## Phase D — Ablations (Paper 1)
9. CRF vs plain softmax slot head.
10. Slot-loss weight λ ∈ {1, 2, 2.5, 3}.
11. Pruned-depth sweep → quality/latency Pareto curve.
12. Seed variance (≥3 seeds) for CRF init stability.

## Phase E — Negative Results (Paper 1)
13. Apply, on the public models, and record result + root cause:
    dynamic INT8, rotation-based INT8 (QuaRot/Hadamard), ONNX/graph export,
    graph compilation, vendor CPU extensions.

## Metrics & Reporting
- Intent accuracy; span-level slot F1 (P/R); joint exact match.
- Latency P50/P95 on a **named** commodity CPU (single thread + fixed thread count),
  warmed up, batch size 1.
- Model size (params + on-disk checkpoint).
- Every table reports mean ± std over ≥3 seeds where applicable.

## Reproducibility
- Pin: dataset versions, model revisions/SHAs, library versions, hardware, threads.
- Ship: training/eval scripts, config files, and exact CLI commands in each paper's
  appendix. Publish code under a permissive license.

## Suggested repo layout for code (to add later)
```
experiments/
  code/
    data/            # public dataset loaders (HF datasets)
    probe/           # linear probe layer sweep
    train/           # prune + heads training
    eval/            # metrics, latency harness
    implicit/        # implicit-slot detector + rules
    negresults/      # quantization/export/compile trials
  configs/
  results/           # generated tables/plots (public data only)
```
