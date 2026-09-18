# SMOKE TESTS — NOT VALID RESULTS. DO NOT CITE OR PUBLISH.

These 32 JSON files (`*_usmodern_phi.json`, `*_usmodern_nemotron.json`) are
**pipeline smoke tests**, not experiments. They exist only to prove the code
runs end-to-end against Phi-3.5-mini and Nemotron-Mini-4B. Every number in
them is meaningless.

Quarantined 2026-09-17 after they were evaluated as possible evidence for a
multi-backbone claim in Paper 1. They are not.

## Why they are invalid

| Problem | Smoke tests | Real runs (GPT-2, in paper) |
|---|---|---|
| Train examples | **80–150** | 4,274–5,000 |
| Epochs | **1** | 3 |
| Probe validation set | **~11–24 examples** | proper val split |
| ATIS test slice | 483 ex / 9 intents | 586 ex / 17 intents |

Probe accuracies are computed on ~11–24 validation examples, i.e. each single
example moves accuracy by 4–9 points. This is noise, not signal.

## What they would wrongly imply

1. **No depth-3 consensus.** The paper's C1 claim is that all five datasets
   independently select depth 3 on the sparse grid. These sweeps recommend
   **32, 16, 24, and 8** — scattered, with no consensus. On CLINC150/Nemotron
   all four depths score *identically* (0.182), i.e. the probe read pure noise.
   BANKING77 scores 0.000 at some depths.

2. **Near-chance fine-tuned accuracy.** BANKING77 10.8–18.3% (64 classes),
   CLINC150 10.7–20.7% (93 classes), MASSIVE 13.2–24.5% (47 classes). Depth 3
   beats depth 12 on ATIS/Nemotron (0.830 vs 0.741) but loses elsewhere —
   coin-flipping, not a depth effect.

Publishing these would read as: "the authors ran two modern decoders, got no
consensus and chance-level accuracy, and reported only the backbone that
worked." That is worse than declaring the single-backbone limitation openly,
which is what `paper.tex` §7 now does.

## What a real multi-backbone experiment needs

Measured CPU training cost from these very runs:

- Phi-3.5-mini: **14.0 s/example-epoch** → ATIS alone (4,274 × 3ep) ≈ **50 hours**
- Nemotron-Mini-4B: **5.3 s/example-epoch** → ATIS alone ≈ **19 hours**
- GPT-2 124M: 0.005 s/example-epoch → ATIS in **61 seconds**

That is one dataset at one depth. The full grid × 5 datasets × 2 backbones is
months of CPU. **Requires GPU.** No CUDA or MPS was available on the machine
these were run on.

## If you resurrect this

Do not reuse these files. Re-run from scratch on GPU with the same protocol as
the GPT-2 runs: full training split (capped at 5,000), 3 epochs, proper
validation split, and the same `{L/4, L/2, 3L/4, L}` sparse grid scaled to the
backbone's actual layer count.
