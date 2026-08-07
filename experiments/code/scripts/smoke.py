"""Offline end-to-end smoke: load a SNIPS-format JSONL fixture -> Examples -> stats.

Deterministic and network-free (uses a bundled fixture). To run on real SNIPS,
install the `models` extra and use `datasets.load_dataset(...)`, mapping columns
into `records_to_examples(...)` the same way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from slmpaper.pipeline import records_to_examples, dataset_stats

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "snips_like_mini.jsonl"


def main() -> int:
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    examples = records_to_examples(
        records, tokens_key="tokens", tags_key="ner_tags", intent_key="intent",
    )
    stats = dataset_stats(examples)
    print("SNIPS-like smoke test")
    print(f"  examples     : {stats['num_examples']}")
    print(f"  intents      : {stats['num_intents']}")
    print(f"  slot types   : {stats['num_slot_types']}")
    imp = stats["implicit"]
    print(f"  slot values  : {imp['total']}")
    print(f"  implicit     : {imp['implicit']} (rate={imp['rate']:.2%})")
    print(f"  max span rec : {imp['max_span_recall']:.2%}")
    assert stats["num_examples"] == 4, "fixture load failed"
    assert stats["num_intents"] == 4
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
