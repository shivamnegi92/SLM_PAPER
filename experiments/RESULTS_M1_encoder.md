# M1 Results — Encoder Baseline (B2)

Model: `distilbert-base-uncased` (66M), JointBERT-style (shared encoder + linear
intent head + softmax BIO slot head). 3 epochs, batch 32, lr 5e-5,
slot-loss-weight 2.0, seed 42. CPU: Apple Silicon arm64, 10 threads (see
`CPU_SPEC.md`). Latency = single-example inference, P50 over 50 runs (5 warmup).

Raw JSON per dataset in `results/<dataset>_encoder.json`. Reproduce:
`bash scripts/run_all_baselines.sh` (uses cached model + local data, offline).

| Dataset | #Intents | #Tags | Intent Acc | Slot F1 | P50 (ms) | Train (s) |
|---|---:|---:|---:|---:|---:|---:|
| ATIS | 17 | 101 | 0.9898 | 0.9512 | 6.99 | 76 |
| SNIPS | 7 | 72 | 0.9900 | 0.9610 | 7.18 | 214 |
| MASSIVE (en) | 60 | 107 | 0.8699 | 0.7857 | 6.69 | 160 |
| CLINC150 | 150 | 1 (intent-only) | 0.9580 | — | 6.70 | 209 |
| BANKING77 | 77 | 1 (intent-only) | 0.9140 | — | 6.74 | 280 |

## Sanity vs published work
- ATIS/SNIPS intent ~99% and slot F1 95-96% are on par with published JointBERT
  (Chen et al. 2019: ATIS 97.5%/96.1%, SNIPS 98.6%/97.0%).
- CLINC150 in-scope ~95.8% matches BERT-family baselines (~96%).
- MASSIVE-en 87.0% intent / 78.6% slot F1 is in the expected range for a
  distilled encoder on the harder 60-intent/55-slot schema.

These are the **discriminative-encoder reference points** the paper compares
against. Next: B1 (generative SLM baseline) for the parse-failure + latency
contrast (C2), then the probe-pruned discriminative model (C1, "ours").
