# SLM_PAPER — Efficient Small Language Models for Intent Detection & Slot Filling

Research paper workspace. **Everything in this folder is public-benchmark-only
and fully self-contained.** No proprietary data, wording, problem framing, or
results are used anywhere. See `COMPLIANCE.md`.

## What's here

```
SLM_PAPER/
├── README.md                        # this file
├── COMPLIANCE.md                    # hard rule: public-only, no internal refs
├── PROPOSAL.md                      # venue strategy + both paper pitches
├── paper1_efficient_slm/
│   └── OUTLINE.md                   # PRIMARY paper — full section-by-section draft skeleton
├── paper2_implicit_slots/
│   └── OUTLINE.md                   # SECONDARY paper — implicit/non-span slot ceiling
└── experiments/
    ├── EXPERIMENT_PLAN.md           # reproducible plan on public benchmarks
    └── datasets.md                  # public datasets, licenses, splits, metrics
```

## Order of work
1. **Paper 1 (primary)** — `paper1_efficient_slm/` — the discriminative-vs-generative
   efficiency result. Draft first.
2. **Paper 2 (secondary)** — `paper2_implicit_slots/` — the implicit-slot recall
   ceiling. Draft after Paper 1's experiments are in.

## Venue strategy (short)
- Primary target: **EMNLP/NAACL main or Findings**, with **ICLR** as a stretch
  once the full public-benchmark suite is in.
- Backup: **NeurIPS ENLSP** (Efficient NLP) or **MLSys** workshop.

## The method (public, generic — no proprietary tie-in)
A recipe for turning a task-specialized generative SLM into a fast, reliable
**discriminative single-pass** intent + slot model:
1. **Probe-guided depth pruning** — a linear-probe layer sweep decides how many
   transformer layers to keep.
2. **Discriminative heads** — a linear intent classifier + a CRF BIO slot tagger
   replace autoregressive JSON generation (removes parse-failure mode entirely).
3. **Implicit-slot recovery** — lightweight, intent-conditioned rules recover
   slots that are not literal text spans.

All claims in the papers must be produced on the public benchmarks listed in
`experiments/datasets.md`. No numbers are asserted until reproduced there.
