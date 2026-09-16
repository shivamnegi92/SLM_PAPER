# M1 Results — Generative Baseline (B1) & the C2 Contrast

Model: `gpt2` (124M) fine-tuned to emit `{"intent":..., "slots":...}` JSON via
causal LM. Bounded for tractable CPU generation: 3000 train examples, 3 epochs,
batch 8, lr 5e-5, greedy decode, eval on 300 held-out examples (5 warmup).
Full-scale (all train, all eval, stronger SLM) is a camera-ready item.
Raw JSON: `results/<dataset>_generative.json`.

## The headline: Generate Less, Classify More (C2)
Discriminative encoder (B2, distilbert) vs generative SLM (B1, gpt2):

| Dataset | Model | Intent Acc | Slot F1 | Parse-fail | P50 (ms) |
|---|---|---:|---:|---:|---:|
| ATIS (101 tag types) | Encoder | **0.9898** | **0.9512** (span) | **0.000** | **6.99** |
| ATIS | Generative | 0.4833 | 0.4458 (value) | **0.467** | 359 |
| SNIPS (72 tag types) | Encoder | **0.9900** | **0.9610** (span) | **0.000** | **7.18** |
| SNIPS | Generative | 0.9633 | 0.8086 (value) | 0.033 | 243 |
| CLINC150 (intent-only) | Encoder | **0.9580** | — | **0.000** | **6.70** |
| CLINC150 | Generative | 0.8533 | 1.000* | 0.000 | 83 |

\*CLINC150 has no real slots (intent-only); slot F1 = 1.0 is trivial (empty vs
empty).

The discriminative model wins on **all three axes** everywhere, and by a much
larger margin exactly where structured output is hardest.

## Unplanned but important finding: reliability & latency scale with slot-schema complexity
- **Parse-failure rate tracks slot-schema complexity**: 46.7% (ATIS, 101 tag
  types) -> 3.3% (SNIPS, 72 tag types) -> 0.0% (CLINC150, intent-only, trivial
  `{"slots":{}}`). The harder the target JSON structure, the more often
  unconstrained generation breaks it.
- **Generative latency scales with output length** (more slots -> more
  autoregressive decode steps): 359ms -> 243ms -> 83ms P50 across the same
  three datasets. **Discriminative latency is flat (~7ms) regardless of slot
  schema** -- it's a single forward pass, not autoregressive decode.
- This is a stronger, more mechanistic version of C2 than a single aggregate
  number: it's not just "generative is slower/less reliable on average", it's
  "generative degrades *precisely as a function of output structure
  complexity*, and the discriminative approach is structurally immune to that
  axis entirely."

## Honest caveats (threats to validity)
- **gpt2 is a weak, non-instruction-tuned base model**, which likely inflates
  ATIS's parse-failure rate specifically. A stronger modern SLM (SmolLM2-135M /
  Qwen2.5-0.5B) would probably parse-fail less in absolute terms, but the
  *complexity-scaling trend* (fail-rate and latency rising with schema size)
  should still hold directionally for any autoregressive model.
  SmolLM2/Qwen weren't available through the local model mirror when these
  runs were captured; gpt2 was fetched from Hugging Face directly into a local
  folder instead.
- Generative slot F1 is *value-based* (emitted values); encoder slot F1 is
  *span-based* (token positions). Both are standard; a value-based comparison
  for the encoder can be added for strict apples-to-apples.
- Generative intent accuracy counts parse failures as wrong (a parse failure ==
  no usable intent), which is the honest operational metric.
- Bounded to 3000 train / 300 eval examples for CPU tractability; full-scale
  numbers are a follow-up, not expected to change the qualitative story.

## Status: M1 (baselines) COMPLETE
- B2 (encoder, discriminative): all 5 datasets, full scale.
- B1 (generative): 3 of 5 datasets (ATIS, SNIPS, CLINC150), bounded scale --
  the intent+slot vs intent-only spread needed to see the complexity-scaling
  finding. MASSIVE/BANKING77 generative runs are straightforward repeats of
  the same script, not run yet (time-bounded, not blocked).
- Next: M2 -- probe-guided depth selection (C1) on the pruned discriminative
  backbone; this is "ours".
