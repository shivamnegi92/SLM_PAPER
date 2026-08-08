# C3 — Implicit-Slot Analysis (real measurement, one-paragraph limitation)

**Method:** `slmpaper.implicit.implicit_slot_rate` marks a gold slot value
"implicit" if it is not a contiguous token span of the (normalized) utterance --
i.e. a span-based tagger cannot recover it by construction, regardless of model
quality. Measured on full train+test of the three slot-filling benchmarks:

| Dataset | Total slot values | Implicit | Rate | Max span-tagger recall |
|---|---:|---:|---:|---:|
| ATIS | 16,350 | 0 | 0.0% | 100% |
| SNIPS | 37,542 | 0 | 0.0% | 100% |
| MASSIVE (en) | 14,159 | 0 | 0.0% | 100% |

**Finding:** the implicit-slot rate is exactly 0% on all three benchmarks --
every gold slot value is a literal substring of its utterance. This means our
span-based (softmax / CRF) slot head has no theoretical recall ceiling on these
datasets; there is no headroom here for a generative or rule-based
non-span-recovery mechanism to add value.

**Why this shelved Paper 2 (`paper2_implicit_slots/`):** the entire premise of
that paper -- quantifying implicit-slot recall loss and recovering it with
intent-conditioned rules -- requires datasets where implicit slots actually
occur. None of ATIS/SNIPS/MASSIVE exhibit this (they are curated,
span-annotation-first benchmarks by design). A real implicit-slot study would
need a different data source (e.g. dialogue-context-dependent slot filling with
coreference/ellipsis, or a schema requiring inferred values like durations
computed from two spans) -- out of scope for this paper.

**Paper 1 framing (limitation, not a workstream):** report the 0% rate as a
scope statement. Our discriminative-heads recipe targets *closed-set,
span-realized* slot filling, which covers these benchmarks completely; tasks
with a meaningful implicit-slot rate would need either a generative fallback or
explicit non-span recovery rules, and are left to future work.
