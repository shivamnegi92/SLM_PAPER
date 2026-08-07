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

## Verified Hub sources (2026-08-07) -- use these exact repo IDs
| Dataset | Hub repo | Columns used | Status |
|---|---|---|---|
| ATIS | `tuetschek/atis` | `text`, `intent`, `slots` (whitespace-aligned BIO string) | Verified, loads fast, no LFS/Xet |
| CLINC150 | `contemmcm/clinc150` | `text`, `intent`, `split` (single `complete` split, filter by `split` col) | Verified, loads fast |
| BANKING77 | `mteb/banking77` (fallback `legacy-datasets/banking77`) | `text`, `label_text` / `label` | Verified schema; **blob download blocked by some corporate proxies** (Xet/LFS transport returns 407 on the CDN redirect even though API/metadata calls succeed). Works fine off-VPN or on unrestricted networks. |
| SNIPS (with slots) | not yet found on the Hub | -- | **Open**: `benayas/snips` only has intent, no slots. No verified Hub source with BIO slots found yet. Fallback plan: pull the original public-domain SNIPS NLU benchmark files directly (sonos-nlu-benchmark, CC0) and parse into the same `Example` schema via `records_to_examples`. |
| MASSIVE (en) | not a Hub dataset for this path | -- | **SOLVED via a different source** -- see "Datasets acquired" below (direct S3 tarball, no HF Hub involved). |

## Datasets acquired (2026-08-07)
| Dataset | Source | Loader | Verified stats |
|---|---|---|---|
| ATIS | HF `tuetschek/atis` (live) | `hf_loaders.load_atis` / `local_loaders.load_atis_local` | loads fine |
| CLINC150 | HF `contemmcm/clinc150` (live) | `hf_loaders.load_clinc150` / `local_loaders.load_clinc150_local` | loads fine |
| MASSIVE (en) | Direct S3 tarball: `https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz` (linked from `github.com/alexa/massive` README). Plain HTTPS, reachable directly from this sandbox, ~40MB, no proxy/Xet/auth issues at all. | `slmpaper.massive.load_massive_local` (parses the `annot_utt` bracket-slot format) | **11,514 train / 2,974 test, 60 intents, 55 slot types** -- matches the official published MASSIVE stats exactly. Note: implicit-slot rate is trivially 0% here since `annot_utt` only ever brackets literal spans -- MASSIVE isn't useful for the Paper 2 implicit-slot analysis, ATIS/SNIPS are. |
| BANKING77, SNIPS | still manual-download (see below) | `local_loaders.load_banking77_local` (SNIPS loader TBD) | -- |

**Practical note:** set `HF_HUB_DISABLE_XET=1` when downloading, and route through
whatever proxy your network requires via standard `HTTP_PROXY`/`HTTPS_PROXY` env
vars -- no proxy hostnames are hardcoded in `slmpaper/hf_loaders.py` (keeps the
anonymized public release proxy-agnostic).

## Investigated and closed (2026-08-07): why the agent sandbox can't pull BANKING77/SNIPS/MASSIVE
Confirmed via live testing + a GEC code search (`Walmart-Tech-Chile/spec-kit-walmart`
reference scripts) that there are two independent blockers, neither fixable from
this sandbox:
1. **No direct network route to `blob.core.windows.net`** from this environment
   (DNS doesn't even resolve without a proxy) -- that's the CDN Xet/LFS-backed
   repos redirect large files to, and the corporate sysproxy returns 407 on that
   redirect.
2. **Walmart's official HF-via-Artifactory proxy (`HF_ENDPOINT=https://ci.artifacts.walmart.com/artifactory/api/huggingfaceml/hub-huggingfaceml-release-remote`) only mirrors MODEL repos, not dataset repos.** Verified: a model
   download (`sentence-transformers/all-MiniLM-L6-v2`) succeeds through it; every
   dataset repo tried (`repo_type="dataset"`) returns 401, including via the
   `hf_hub_download()` file-level API the reference scripts recommend.

**Conclusion:** this is a sandbox-network limitation, not a code or credentials
problem. The fix is downloading on a real Walmart laptop (full corporate internet
with automatic blob-CDN routing) and handing off the files locally -- see
`../experiments/code/data/README.md` for the manual-download workflow and the
local-file loaders already built for it.

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
