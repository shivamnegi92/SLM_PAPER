# Kaggle GPU Pack (Modern Backbone E2E)

This folder is Kaggle-ready and contains:

- `notebooks/modern_backbone_e2e_kaggle.ipynb` — end-to-end runner notebook
- `datasets/raw/...` — copied dataset files used by the loaders
- `code/` — copied training scripts and `slmpaper` package (GPU-aware patches applied)

## Kaggle usage

1. Upload this folder as a Kaggle Dataset (or zip + attach in notebook).
2. Enable **GPU** in notebook settings.
3. Open and run `notebooks/modern_backbone_e2e_kaggle.ipynb` top to bottom.
4. In the **Model Configuration** cell, point model paths to:
   - Kaggle input model folders (recommended), or
   - Hugging Face model IDs (requires internet).

## Outputs

- Per-run artifacts under `code/results/*_usmodern_*.json`
- Consolidated CSV: `kaggle_run_summary.csv`

## Notes

- CLINC150 loader uses local `data/raw/clinc150/data_full.json` if present; otherwise falls back to Hugging Face.
- The notebook is resumable via `run_if_missing` to avoid recomputing finished artifacts.
