"""Validated public ARC-Easy input preparation and fixed text windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from study_data import fingerprint, write_new


def arc_item(record):
    labels = [str(label) for label in record["choices"]["label"]]
    endings = record["choices"]["text"]
    answer = str(record["answerKey"])
    if answer not in labels or len(set(labels)) != len(labels) or len(endings) != len(labels):
        raise ValueError("ARC choices/answer labels are inconsistent")
    if not isinstance(record["question"], str) or not all(isinstance(text, str) and text for text in endings):
        raise ValueError("ARC question/answers must be nonempty strings")
    return {"id": record["id"], "context": f"Question: {record['question']}\nAnswer:",
            "endings": endings, "correct_index": labels.index(answer)}


def download_arc(output, count=200):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if count < 1:
        raise ValueError("Positive sample count required")
    items, sources = [], []
    for offset in range(0, count, 100):
        query = urlencode({"dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test",
                           "offset": offset, "length": min(100, count - offset)})
        url = "https://datasets-server.huggingface.co/rows?" + query
        with urlopen(url, timeout=60) as response:
            payload = json.load(response)
        items.extend(arc_item(row["row"]) for row in payload["rows"])
        sources.append(url)
    if len(items) != count or len({item["id"] for item in items}) != count:
        raise ValueError("ARC response is incomplete or duplicated")
    document = {"dataset": "allenai/ai2_arc", "subset": "ARC-Easy", "split": "test",
                "license": "CC-BY-SA-4.0 (dataset card)", "source_urls": sources,
                "selection": f"first {count} test rows, fixed before model evaluation",
                "items": items, "items_sha256": fingerprint(items)}
    write_new(output, document)
    print(f"saved {len(items)} ARC-Easy examples -> {output}; {document['items_sha256']}")


def load_arc(path):
    with Path(path).open() as stream:
        saved = json.load(stream)
    if fingerprint(saved["items"]) != saved["items_sha256"]:
        raise ValueError("ARC cache integrity mismatch")
    return [(item["context"], item["endings"], item["correct_index"]) for item in saved["items"]]


def text_windows(text, count=5, width=2000):
    if min(count, width) < 1 or len(text) < count * width:
        raise ValueError("Insufficient text for non-overlapping windows")
    return [text[index * width:(index + 1) * width] for index in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    download_arc(args.output, args.count)


if __name__ == "__main__":
    main()