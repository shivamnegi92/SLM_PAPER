import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACK_DIR = REPO_ROOT / 'kaggle_gpu_pack'
NB_PATH = PACK_DIR / 'notebooks' / 'modern_backbone_e2e_kaggle.ipynb'


def _load_cells():
    data = json.loads(NB_PATH.read_text())
    return data.get('cells', [])


def _markdown_text():
    return '\n'.join(
        ''.join(c.get('source', []))
        for c in _load_cells()
        if c.get('cell_type') == 'markdown'
    )


def _code_text():
    return '\n'.join(
        ''.join(c.get('source', []))
        for c in _load_cells()
        if c.get('cell_type') == 'code'
    )


def test_kaggle_pack_notebook_exists():
    assert NB_PATH.exists(), f'Missing notebook: {NB_PATH}'


def test_kaggle_pack_notebook_sections():
    md = _markdown_text()
    required = [
        'Kaggle GPU End-to-End Runner',
        'Pre-flight Checklist',
        'Environment Setup',
        'Dataset Staging',
        'Model Configuration',
        'Run Training Matrix',
        'Results Summary',
    ]
    for section in required:
        assert section in md, f'Missing markdown section: {section}'


def test_preflight_mentions_internet_and_models():
    md = _markdown_text().lower()
    assert 'internet' in md, 'Pre-flight checklist must mention Internet ON'
    assert 'model' in md, 'Pre-flight checklist must mention attaching models'


def test_matrix_includes_atis():
    code = _code_text()
    assert "'atis'" in code, 'ATIS must be in the dataset matrix'


def test_publication_grade_budget_scaling():
    code = _code_text()
    # per-dataset budget scaled by intent count (>= 20x intents)
    assert 'INTENTS' in code, 'Notebook must define per-dataset INTENTS map'
    assert '20 *' in code or '20*' in code, 'Budget must scale >= 20x num intents'


def test_clinc150_local_copy_present():
    clinc = PACK_DIR / 'datasets' / 'raw' / 'clinc150' / 'data_full.json'
    assert clinc.exists(), 'Local CLINC150 data_full.json must be packaged for offline use'
    data = json.loads(clinc.read_text())
    assert 'train' in data and 'test' in data, 'CLINC150 local file needs train/test splits'
