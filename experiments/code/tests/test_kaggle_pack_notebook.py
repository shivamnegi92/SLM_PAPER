import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACK_DIR = REPO_ROOT / 'kaggle_gpu_pack'
NB_PATH = PACK_DIR / 'notebooks' / 'modern_backbone_e2e_kaggle.ipynb'


def test_kaggle_pack_notebook_exists():
    assert NB_PATH.exists(), f'Missing notebook: {NB_PATH}'


def test_kaggle_pack_notebook_sections():
    data = json.loads(NB_PATH.read_text())
    cells = data.get('cells', [])
    md = '\n'.join(''.join(c.get('source', [])) for c in cells if c.get('cell_type') == 'markdown')
    required = [
        'Kaggle GPU End-to-End Runner',
        'Environment Setup',
        'Dataset Staging',
        'Run Training Matrix',
        'Results Summary',
    ]
    for section in required:
        assert section in md, f'Missing markdown section: {section}'
