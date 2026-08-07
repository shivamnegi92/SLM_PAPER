# Getting Real Data In (Manual Download Path)

Corporate proxy blocks the Xet/LFS CDN redirect from this environment (see
`../datasets.md`), so the path forward is: **you download via browser/git on
your own machine (proper VPN/SSO auth), drop the files here, loaders read
them from disk.** No network calls from the sandbox required.

## Expected layout
```
data/raw/
├── atis/
│   └── train.parquet          # from huggingface.co/datasets/tuetschek/atis
├── clinc150/
│   └── data_full.json         # from github.com/clinc/oos-eval (data/data_full.json)
├── banking77/
│   └── train.csv               # from github.com/PolyAI-LDN/task-specific-datasets
├── snips/                      # format TBD until real files are inspected
└── massive/                    # format TBD until real files are inspected
```

## Loaders (already implemented + tested)
```python
from slmpaper.local_loaders import load_atis_local, load_clinc150_local, load_banking77_local

atis_train = load_atis_local("data/raw/atis/train.parquet")
clinc_train = load_clinc150_local("data/raw/clinc150/data_full.json", split="train")
banking_train = load_banking77_local("data/raw/banking77/train.csv")
```

## SNIPS and MASSIVE
Not implemented yet -- their exact local file shape (JSON per-intent folders for
SNIPS; per-locale JSONL for MASSIVE) needs to be confirmed against the real
downloaded files before writing a parser. Drop the files in `data/raw/snips/`
and `data/raw/massive/` and say so; the loader will be written and tested
against the actual format at that point (no guessing at schemas we haven't
seen).
