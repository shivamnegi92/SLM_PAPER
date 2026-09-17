# SLM_PAPER — Efficient Small Language Models for Intent Detection & Slot Filling

Research paper workspace. **Everything in this folder is public-benchmark-only
and fully self-contained.** No proprietary data, wording, problem framing, or
results are used anywhere. See `COMPLIANCE.md`.

## What's here

```
SLM_PAPER/
├── README.md                        # this file
├── PROJECT_REPORT.md                # in-depth, all-real-numbers project report (start here)
├── ENTERPRISE_VALUE.md              # business-value translation of the results (generic, no company tie-in)
├── COMPLIANCE.md                    # hard rule: public-only, no internal refs
├── PROPOSAL.md                      # venue strategy + both paper pitches
├── paper1_efficient_slm/            # PAPER 1 — probe-guided pruning, intent + slots
│   ├── latex/paper.tex              #   ICLR-format manuscript (+ paper.pdf)
│   ├── OUTLINE.md                   #   section-by-section skeleton
│   └── MODERN_BACKBONE_PARITY_PLAN.md  # plan to add Nemotron/Phi at GPT-2 parity
├── circuit_breakdown/               # PAPER 3 — intervention selectivity (interpretability)
│   ├── README.md                    #   START HERE for this subproject
│   ├── paper/PAPER_DRAFT.md         #   the paper (+ paper.html, figures)
│   └── results/frozen_e91985c/      #   frozen snapshot behind every number
├── paper2_implicit_slots/
│   └── OUTLINE.md                   # SECONDARY paper — implicit/non-span slot ceiling
├── kaggle_gpu_pack/                 # offline GPU runner for the modern-backbone matrix
│   ├── notebooks/                   #   end-to-end Kaggle notebook
│   └── datasets/raw/                #   all 5 datasets bundled, no internet needed
├── experiments/
│   ├── EXPERIMENT_PLAN.md           # reproducible plan on public benchmarks
│   ├── datasets.md                  # public datasets, licenses, splits, metrics
│   └── code/results/                # 118 result artifacts
└── GET_MODELS.ipynb                 # fetch model weights (gitignored, ~29 GB)
```

## The three papers at a glance

| | subject | status |
|---|---|---|
| **Paper 1** `paper1_efficient_slm/` | Probe-guided pruning + discriminative heads for intent/slots | ICLR 2027 target (Sep 25 deadline). **Caveat: headline results are GPT-2 only** — Nemotron/Phi exist but were run at a 53x smaller budget, see `MODERN_BACKBONE_PARITY_PLAN.md` |
| **Paper 2** `paper2_implicit_slots/` | Implicit / non-span slot ceiling | outline only |
| **Paper 3** `circuit_breakdown/` | Activation interventions can be effective, specific and dose-graded yet **non-selective** (0/240, two model families) | draft complete, numbers guarded by a 44-check verifier; ARR Oct 12 2026 target |

### Model weights are NOT in this repo

~29 GB, gitignored. Run `GET_MODELS.ipynb`, or download to the repo root as
`phi-3.5-mini/`, `llama-3.2-3b/`, `nemotron-mini-4b/`. Do not quantize if you
are reproducing `circuit_breakdown` — 4-bit distorts activation geometry.

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
