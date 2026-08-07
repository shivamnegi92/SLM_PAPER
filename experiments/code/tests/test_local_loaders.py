"""RED: local-file loaders for downloaded raw datasets (no network needed).

Formats targeted (verified against publicly documented structures):
  - ATIS      : local parquet with columns text/intent/slots (same schema as
                the HF mirror we already verified live).
  - CLINC150  : original oos-eval `data_full.json` -> {"train": [[text, intent], ...], ...}
  - BANKING77 : original PolyAI CSV -> columns text,category
"""
import json

import pandas as pd
import pytest

from slmpaper.local_loaders import (
    load_atis_local,
    load_clinc150_local,
    load_banking77_local,
)


def test_load_atis_local_reads_parquet(tmp_path):
    df = pd.DataFrame({
        "text": ["find a flight to boston", "book denver flight"],
        "intent": ["flight", "flight"],
        "slots": ["O O O O B-toloc.city_name", "O B-toloc.city_name O"],
    })
    pq_path = tmp_path / "atis_train.parquet"
    df.to_parquet(pq_path)

    examples = load_atis_local(pq_path)
    assert len(examples) == 2
    assert examples[0].intent == "flight"
    assert examples[0].slots == {"toloc.city_name": ["boston"]}


def test_load_clinc150_local_reads_split(tmp_path):
    data = {
        "train": [["what is my balance", "balance"], ["play some jazz", "play_music"]],
        "test": [["hello there", "greeting"]],
    }
    json_path = tmp_path / "data_full.json"
    json_path.write_text(json.dumps(data))

    train_examples = load_clinc150_local(json_path, split="train")
    assert len(train_examples) == 2
    assert train_examples[0].intent == "balance"
    assert train_examples[0].slots == {}

    test_examples = load_clinc150_local(json_path, split="test")
    assert len(test_examples) == 1
    assert test_examples[0].intent == "greeting"


def test_load_clinc150_local_unknown_split_raises(tmp_path):
    json_path = tmp_path / "data_full.json"
    json_path.write_text(json.dumps({"train": []}))
    with pytest.raises(KeyError):
        load_clinc150_local(json_path, split="nonexistent")


def test_load_banking77_local_reads_csv(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "text,category\n"
        '"i lost my card",card_lost\n'
        '"what is my balance",balance\n'
    )
    examples = load_banking77_local(csv_path)
    assert len(examples) == 2
    assert examples[0].intent == "card_lost"
    assert examples[0].slots == {}
