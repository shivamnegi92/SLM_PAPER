# Public Datasets & Models

> Public-benchmark-only. Verify current licenses/citations at submission time.

## Datasets

| Dataset | Task | Size (approx) | Notes | Typical citation |
|---|---|---|---|---|
| **ATIS** | Intent + slot filling | ~5k train | Airline travel; classic slot-filling benchmark | Hemphill et al., 1990; Tur et al., 2010 |
| **SNIPS** | Intent + slot filling | ~14k train | 7 intents, voice-assistant style | Coucke et al., 2018 |
| **MASSIVE** | Intent + slot filling | ~1M utts (51 langs) | 60 intents, 55 slot types; multilingual | FitzGerald et al., 2022 (Amazon) |
| **CLINC150** | Intent (incl. OOS) | ~22.5k | 150 intents + out-of-scope; good for closed-set + OOS | Larson et al., 2019 |
| **BANKING77** | Intent (fine-grained) | ~13k | 77 fine-grained banking intents | Casanueva et al., 2020 |

- Intent-only papers/experiments: CLINC150, BANKING77 (+ intent side of ATIS/SNIPS/MASSIVE).
- Slot-filling + implicit-slot analysis: ATIS, SNIPS, MASSIVE.
- All are available via Hugging Face `datasets`. Pin the exact version/revision.

## Candidate public base models (verify license before use)

| Model | Type | Params | License | Use |
|---|---|---|---|---|
| SmolLM2-135M / -360M | Decoder (Llama-arch) | 135M / 360M | Apache-2.0 | Small generative SLM + prune target |
| Qwen2.5-0.5B | Decoder | 0.5B | Apache-2.0 | Alt small SLM |
| Llama-3.2-1B | Decoder | 1B | Llama Community License | Larger SLM comparison (check license terms for publication) |
| DistilBERT-base | Encoder | 66M | Apache-2.0 | JointBERT-style discriminative baseline |
| MiniLM-L6 | Encoder | ~22M | MIT | Lightweight encoder baseline |

- Prefer **Apache-2.0 / MIT** models to avoid license friction in an academic release.
- Record model revision SHA and download source for each.

## Metrics definitions
- **Intent accuracy**: exact intent match.
- **Slot F1**: span-level (exact boundary + type) precision/recall/F1 (seqeval).
- **Exact match (joint)**: intent correct AND all slots correct.
- **Implicit-slot rate**: fraction of gold slot values not a contiguous substring
  of the normalized utterance.
- **Latency**: P50/P95, batch=1, warmed, on a named CPU with fixed thread count.
