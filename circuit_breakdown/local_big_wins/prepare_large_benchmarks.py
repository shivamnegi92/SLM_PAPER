"""Prepare larger public benchmark caches for local capability checks.

This helper writes new files only. It never overwrites the frozen benchmark
caches used by the completed study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def line_count(path: Path) -> int:
    with path.open() as stream:
        return sum(1 for _ in stream)


def ensure_hellaswag(output: Path, count: int) -> None:
    if output.exists():
        print(f"Using existing {output} ({line_count(output)} rows)")
        return
    try:
        datasets = __import__("datasets", fromlist=["load_dataset"])
    except Exception as error:
        raise RuntimeError("Install datasets to build a larger HellaSwag cache") from error
    rows = datasets.load_dataset("Rowan/hellaswag", split="validation")
    selected = rows.select(range(min(count, len(rows))))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("w") as stream:
        for row in selected:
            stream.write(json.dumps({"ctx": row["ctx"], "endings": row["endings"], "label": row["label"]}) + "\n")
    temporary.replace(output)
    print(f"Wrote {output} ({line_count(output)} rows)")


def ensure_arc(project: Path, output: Path, count: int) -> None:
    if output.exists():
        print(f"Using existing {output}")
        return
    command = [sys.executable, str(project / "src/public_benchmarks.py"), "--output", str(output), "--count", str(count)]
    subprocess.run(command, cwd=project, check=True)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    project = args.project.resolve()
    ensure_hellaswag(project / f"data_bench/hellaswag_val_{args.count}.jsonl", args.count)
    ensure_arc(project, project / f"data_bench/arc_easy_test_{args.count}.json", args.count)


if __name__ == "__main__":
    main()
