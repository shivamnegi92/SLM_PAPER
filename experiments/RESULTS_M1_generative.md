# M1 Results — Generative Baseline (B1) & the C2 Contrast

Model: `gpt2` (124M) fine-tuned to emit `{"intent":..., "slots":...}` JSON via
causal LM. Bounded for tractable CPU generation: 3000 train examples, 3 epochs,
batch 8, lr 5e-5, greedy decode, eval on 300 held-out examples (5 warmup).
Full-scale (all train, all eval, stronger SLM) is a camera-ready item.
Raw JSON: `results/<dataset>_generative.json`.

## The headline: Generate Less, Classify More (C2)
Discriminative encoder (B2) vs generative SLM (B1), same datasets:

| Dataset | Model | Intent Acc | Slot F1 | Parse-fail | P50 (ms) |
|---|---|---:|---:|---:|---:|
| ATIS | Encoder (B2) | **0.9898** | **0.9512** (span) | **0.000** | **6.99** |
| ATIS | Generative (B1) | 0.4833 | 0.4458 (value) | 0.467 | 359 |

The discriminative model wins on **all three axes**: accuracy, reliability
(parse-failure rate 0% *by construction* vs 46.7%), and latency (**~51× faster**
on CPU).

## Honest caveats (threats to validity)
- **gpt2 is a weak, non-instruction-tuned base model**, which inflates its
  parse-failure rate. A stronger modern SLM (SmolLM2-135M / Qwen2.5-0.5B) would
  parse-fail less. Those weren't yet pullable through the Walmart Artifactory HF
  proxy (not pre-warmed in the mirror; needs a Remote-Proxy request). The
  **latency gap (~50x) and the structural 0%-parse-failure guarantee are
  model-agnostic** and are the robust part of the claim.
- Generative slot F1 is *value-based* (emitted values); encoder slot F1 is
  *span-based* (token positions). Both are standard; a value-based comparison for
  the encoder can be added for strict apples-to-apples.
- Generative intent accuracy counts parse failures as wrong (a parse failure ==
  no usable intent), which is the honest operational metric.

## Status
- ATIS: complete (above).
- SNIPS, CLINC150: generative runs executing (bounded, same config).
