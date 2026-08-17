# Kaggle GPU Pack (Modern Backbone E2E)

This folder is Kaggle-ready and contains:

- `notebooks/modern_backbone_e2e_kaggle.ipynb` — end-to-end runner notebook (markdown-structured)
- `datasets/raw/...` — copied dataset files used by the loaders (fully offline, incl. CLINC150)
- `code/` — copied training scripts and `slmpaper` package (GPU-aware patches applied)

## Pre-flight (READ FIRST)

1. **GPU ON** — Settings → Accelerator → GPU.
2. **Internet ON** — only needed if you let the models download from Hugging Face. Datasets are all local.
3. **Attach models** — the two ~4B backbones are NOT in this pack (too big for git):
   - Attach them as Kaggle Datasets and set paths in the **Model Configuration** cell, or
   - Leave the HF IDs (`nvidia/Nemotron-Mini-4B-Instruct`, `microsoft/Phi-3.5-mini-instruct`) and let them download.

## Kaggle usage

1. Upload this folder as a Kaggle Dataset (or zip + attach in notebook).
2. Enable GPU (and Internet if downloading models).
3. Open and run `notebooks/modern_backbone_e2e_kaggle.ipynb` top to bottom.

## What it runs

- **All 5 datasets**: ATIS, SNIPS, MASSIVE, CLINC150, BANKING77
- **Both models**: Nemotron-Mini-4B, Phi-3.5-mini
- Per dataset: probe sweep + pruned depth-3 + pruned depth-12

## Publication-grade budget (the reason to use a GPU)

Training size scales with intent cardinality so high-cardinality datasets are no longer starved:

- `pruned_max_train = max(400, 20 × num_intents)`
  - ATIS(17)=400, SNIPS(7)=400, MASSIVE(60)=1200, BANKING77(77)=1540, CLINC150(150)=3000
- `probe_max_train = min(3000, max(600, 40 × num_intents))`
- `PRUNED_EPOCHS = 3`, `BATCH_SIZE = 16`, `PROBE_EPOCHS = 50`

Tweak these knobs in the **Run Training Matrix** cell if you want to go bigger.

## Outputs

- Per-run artifacts under `code/results/*_usmodern_*.json`
- Consolidated CSV: `kaggle_run_summary.csv`

## Notes

- CLINC150 loader uses the packaged local `data/raw/clinc150/data_full.json` (15000 train / 4500 test, 150 intents). No internet needed for data.
- Intent labels for CLINC150 are stringified integer IDs — identical mapping to the original HF loader, so results are comparable.
- The notebook is resumable via `run_if_missing` to avoid recomputing finished artifacts.
