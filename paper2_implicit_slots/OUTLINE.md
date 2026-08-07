# Paper 2 (SECONDARY) — Full Draft Skeleton

**Working title:** *Implicit Slots: The Structural Recall Ceiling of Span-Based
Slot Filling*

> Public-benchmark-only. Results are `TBD` until measured on public datasets.
> No proprietary references (see `../COMPLIANCE.md`).

---

## Abstract (skeleton)
Span-based slot filling (BIO/CRF taggers and extractive LLM prompting) assumes
every slot value appears as a contiguous span in the input. We show this
assumption is systematically violated in task-oriented dialogue: a measurable
fraction of gold slots are **implicit** — implied by pronouns, context, or the
intent, but absent as literal text. This imposes a hard recall ceiling that no
span tagger can exceed by design. We (1) quantify the implicit-slot rate across
ATIS, SNIPS, MASSIVE, and other public benchmarks, (2) decompose tagger false
negatives into "implicit" vs "genuine miss," and (3) show that lightweight
intent-conditioned rules recover implicit slots at near-zero cost and latency.

## 1. Introduction
- Span assumption is baked into BIO tagging and extractive prompting.
- Observation: some gold slots are never spans (e.g., an implied speaker/self
  slot, an implied time, an inferred role).
- Contribution: name, quantify, decompose, and cheaply mitigate the phenomenon.

## 2. Background
- BIO tagging, CRF decoding, span-level F1.
- Extractive vs generative slot filling.
- Why the span assumption matters for the recall metric.

## 3. Defining Implicit Slots
- Definition: a gold slot whose value string is not a contiguous substring of the
  (normalized) input utterance.
- Taxonomy: pronoun-implied, intent-implied, context/anaphora-implied,
  normalized-form mismatch.

## 4. Measuring the Ceiling (TBD)
- Per dataset: % of gold slots that are implicit → theoretical max span-tagger
  recall.
| Dataset | Total slots | Implicit slots | Implicit % | Max span recall |
|---|---|---|---|---|
| ATIS | TBD | TBD | TBD | TBD |
| SNIPS | TBD | TBD | TBD | TBD |
| MASSIVE | TBD | TBD | TBD | TBD |

## 5. False-Negative Decomposition (TBD)
- Take a strong CRF tagger; split its FNs into implicit vs genuine miss.
- Show what fraction of "errors" are structurally unreachable.

## 6. Cheap Recovery: Intent-Conditioned Rules (TBD)
- Rules fire only for intents known to imply a slot (precision-first).
- Report ΔF1, precision of fired rules, latency overhead.

## 7. Discussion
- Implication for benchmark design and for choosing extractive vs generative
  slot fillers.
- When generation is actually justified (high implicit-slot datasets).

## 8. Conclusion
- Span taggers are not "bad"; they hit a definitional ceiling. Measure it, then
  patch it cheaply or switch paradigms deliberately.

## Appendix
- Detection script for implicit slots, per-type breakdowns, reproduction commands.
