# ENTERPRISE_VALUE — Why This Matters Outside a Research Paper

This document translates the real, measured results in `PROJECT_REPORT.md`
into business terms for any organization running natural-language
understanding (NLU) at production scale — chatbots, IVR/voice assistants,
support-ticket routing, search-intent classification, or any system that maps
a user utterance to a closed set of intents + slots.

> Framing note: this stays intentionally generic (no company-specific tie-in),
> consistent with `COMPLIANCE.md` (this repo is public-benchmark-only). The
> numbers below are the same real measurements from `PROJECT_REPORT.md` §4,
> reframed for a business audience instead of a research one.

---

## 1. The business problem in one sentence

If your NLU pipeline uses a generative LLM/SLM to produce structured output
(intent + slots as JSON), you are paying **autoregressive decode cost on every
request** and accepting a **non-zero rate of outright broken responses** —
both of which are avoidable for closed-set classification tasks.

## 2. What changes, in terms that matter to a budget owner

| Lever | What we measured | Business translation |
|---|---|---|
| **Compute cost per request** | ~70× lower CPU latency (359ms → 5.1ms on ATIS) | For a high-volume NLU workload, this is a ~70× reduction in CPU-seconds consumed per request — directly proportional to your compute bill for that service, whether it's cloud instance-hours or CPU quota on shared infra. |
| **Reliability / support burden** | Parse-failure rate 46.7% → **0%**, by construction | Every malformed structured output today is either a silent misroute, a fallback-to-human escalation, or a retry — all of which cost money and erode user trust. Eliminating this *structurally* (not by prompting harder) removes a whole class of production incidents. |
| **Time-to-decision on model sizing** | A 15-second linear probe correctly predicts the right model depth, before any fine-tuning | Model-sizing decisions today are usually made by trial-and-error fine-tuning runs (hours to days of GPU time each). A cheap probe collapses that to seconds, which matters when you're evaluating "how small can we go" across many product surfaces. |
| **Deployment footprint** | 3-of-12-layer model matches full-depth accuracy on 3 of 5 benchmarks; small (~1-2.6pt) tradeoffs on the rest | Smaller models mean cheaper serving hardware (CPU-only viable instead of GPU), easier edge/on-device deployment, and lower memory footprint per replica — relevant for anyone running many concurrent NLU services. |
| **Extra CPU-runtime engineering** | Dynamic INT8 and `torch.compile` gave *no* benefit at this model scale; ONNX Runtime gave a genuine further 4.5× | Don't assume every "make it faster" checkbox pays off — we measured it, and two of the three common tricks were a waste of engineering time at this scale. ONNX conversion is worth the (real) export effort; the others are not, for shallow models. |

## 3. Where this recipe applies (and where it doesn't)

**Applies well:**
- Chatbot/voice-assistant intent routing with a fixed, known intent taxonomy
- Support-ticket / email triage classification
- Search-query intent classification
- Any slot-filling task where the values are literal spans of the input text
  (dates, product names, locations spoken/typed verbatim) — this covers the
  overwhelming majority of production slot-filling schemas we measured (0%
  implicit-slot rate across three real benchmarks, see §4.5 of
  `PROJECT_REPORT.md`)

**Doesn't apply / needs adaptation:**
- Open-set or novel-intent detection (new, unseen categories at inference time)
- Free-form generation tasks (summarization, drafting responses) — generation
  is still the right tool there
- Slot values that must be *inferred* rather than read off the text (e.g.
  computed durations, coreference-resolved entities) — a genuinely different
  and rarer problem than what most production slot-filling systems actually
  need

## 4. A practical adoption checklist for an engineering team

1. **Do you have an existing generative SLM already fine-tuned for structured
   output?** If yes, you don't need a new model — you can convert it.
2. **Is your intent/slot schema closed-set?** (a fixed list, not open
   vocabulary) — if yes, this recipe applies directly.
3. **Run the probe first.** Before committing to a fine-tuning budget, run the
   ~15-second linear-probe sweep to see how shallow you can go. This alone can
   save a fine-tuning cycle.
4. **Measure your own parse-failure rate today.** If you don't already track
   this, you likely don't know your true reliability cost — it's an easy,
   high-signal metric to start logging before and after switching to a
   discriminative head.
5. **Don't reach for quantization or `torch.compile` first.** Measure whether
   they help at your model's specific depth/scale before investing engineering
   time — our data shows they can be net-negative for small models. Evaluate
   ONNX export instead if CPU latency is the binding constraint.

## 5. Honest caveats for a business audience

- This is not "our model beats a state-of-the-art encoder" — it's "you can
  convert what you already have (a generative decoder) into something fast
  and reliable, without standing up a separate encoder architecture."
- The latency/cost numbers above are measured on one backbone size (124M
  parameters) on one CPU class; exact multipliers will vary with your model
  and hardware, but the *qualitative* findings (generation is unnecessarily
  costly for closed-set tasks; common CPU tricks don't automatically help
  small models; a cheap probe can guide sizing) are architecture-general.
- Full measurement methodology, datasets, and reproduction instructions are
  in `PROJECT_REPORT.md` and `experiments/`.
