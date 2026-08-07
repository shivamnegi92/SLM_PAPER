"""RED: two more local loaders for the manually-downloaded raw datasets.

  - ATIS (IOB CSV): Kaggle-mirrored community version, columns
    id,tokens,slots,intent, with BOS/EOS sentinels wrapping each utterance.
  - SNIPS: real sonos/nlu-benchmark format -- per-intent dir with
    train_<Intent>_full.json / validate_<Intent>.json, each example a list
    of {"text": ..., "entity": <optional>} chunks.
"""
import json

import pytest

from slmpaper.local_loaders import load_atis_iob_local, load_snips_local


def test_load_atis_iob_local_strips_bos_eos(tmp_path):
    csv_path = tmp_path / "atis.train.csv"
    csv_path.write_text(
        "id,tokens,slots,intent\n"
        '"t1","BOS find a flight to boston EOS","O O O O O B-toloc.city_name O","flight"\n'
    )
    examples = load_atis_iob_local(csv_path)
    assert len(examples) == 1
    assert examples[0].tokens == ["find", "a", "flight", "to", "boston"]
    assert examples[0].intent == "flight"
    assert examples[0].slots == {"toloc.city_name": ["boston"]}


def test_load_snips_local_parses_data_chunks(tmp_path):
    intent_dir = tmp_path / "AddToPlaylist"
    intent_dir.mkdir()
    payload = {
        "AddToPlaylist": [
            {"data": [
                {"text": "add "},
                {"text": "Stani", "entity": "entity_name"},
                {"text": " to "},
                {"text": "my playlist", "entity": "playlist"},
            ]}
        ]
    }
    (intent_dir / "train_AddToPlaylist_full.json").write_text(json.dumps(payload))

    examples = load_snips_local(tmp_path, split="train")
    assert len(examples) == 1
    ex = examples[0]
    assert ex.intent == "AddToPlaylist"
    assert ex.tokens == ["add", "Stani", "to", "my", "playlist"]
    assert ex.slots == {"entity_name": ["Stani"], "playlist": ["my playlist"]}


def test_load_snips_local_validate_split(tmp_path):
    intent_dir = tmp_path / "GetWeather"
    intent_dir.mkdir()
    payload = {"GetWeather": [{"data": [{"text": "is it raining"}]}]}
    (intent_dir / "validate_GetWeather.json").write_text(json.dumps(payload))

    examples = load_snips_local(tmp_path, split="validate")
    assert len(examples) == 1
    assert examples[0].intent == "GetWeather"
    assert examples[0].slots == {}


def test_load_snips_local_unknown_split_raises(tmp_path):
    with pytest.raises(ValueError):
        load_snips_local(tmp_path, split="bogus")


def test_load_snips_local_handles_latin1_file(tmp_path):
    """Documented quirk: some original SNIPS files are Latin-1, not UTF-8."""
    intent_dir = tmp_path / "PlayMusic"
    intent_dir.mkdir()
    payload = '{"PlayMusic": [{"data": [{"text": "play caf\xe9 music"}]}]}'
    (intent_dir / "train_PlayMusic_full.json").write_bytes(payload.encode("latin-1"))

    examples = load_snips_local(tmp_path, split="train")
    assert len(examples) == 1
    assert "caf\xe9" in examples[0].text
