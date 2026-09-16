"""Frozen public-data manifests shared across local model tokenizers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import dataset as ds
from intervene_margin import encode_pair


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)


def token_aligned(pair, tokenizers):
    for tokenizer in tokenizers:
        clean = tokenizer(pair.clean_prompt, add_special_tokens=True)["input_ids"]
        corrupt = tokenizer(pair.corrupt_prompt, add_special_tokens=True)["input_ids"]
        if len(clean) != len(corrupt) or sum(left != right for left, right in zip(clean, corrupt)) != 1:
            return False
        for target in (pair.clean_target, pair.corrupt_target):
            base = tokenizer("the", add_special_tokens=False)["input_ids"]
            extended = tokenizer("the " + target, add_special_tokens=False)["input_ids"]
            if extended[:len(base)] != base or len(extended) != len(base) + 1:
                return False
    return True


def build_manifest(tokenizers, seeds=(0, 1, 2), train=64, dev=24, test=50, fewshot=3):
    if not seeds or len(set(seeds)) != len(seeds) or min(train, dev, test, fewshot) < 1:
        raise ValueError("Distinct seeds and positive split sizes are required")
    tasks, used_prompts = {}, set()
    for task in ("intermediate", "transfer", "container_swap"):
        tasks[task] = {}
        for seed in seeds:
            splits = {name: [] for name in ("fewshot", "train", "dev", "test")}
            targets = {"fewshot": fewshot, "train": train, "dev": dev, "test": test}
            lengths = (1, 2) if task == "transfer" else (3, 4)
            for batch in range(100):
                candidates = ds.generate(task, 500, seed=100000 * (seed + 1) + batch,
                                         min_len=lengths[0], max_len=lengths[1])
                for pair in candidates:
                    if pair.clean_prompt in used_prompts or pair.corrupt_prompt in used_prompts:
                        continue
                    if not token_aligned(pair, tokenizers):
                        continue
                    for split in splits:
                        if len(splits[split]) >= targets[split]:
                            continue
                        if task == "container_swap" and pair.metadata["template"] != int(split == "test"):
                            continue
                        splits[split].append(asdict(pair))
                        used_prompts.update([pair.clean_prompt, pair.corrupt_prompt])
                        break
                if all(len(splits[name]) == targets[name] for name in splits):
                    break
            else:
                raise RuntimeError(f"Could not fill frozen splits for {task}, seed={seed}")
            tasks[task][str(seed)] = splits
    manifest = {"version": "paired_public_manifest_v1", "seeds": list(seeds),
                "counts": {"train": train, "dev": dev, "test": test, "fewshot": fewshot},
                "tasks": tasks,
                "holdout": "container_swap test uses unseen wording template1; train/dev/fewshot template0"}
    manifest["sha256"] = fingerprint(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    unhashed = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != fingerprint(unhashed):
        raise ValueError("Manifest hash mismatch")
    prompts = set()
    for task, seeds in manifest["tasks"].items():
        for seed, splits in seeds.items():
            if set(splits) != set(manifest["counts"]):
                raise ValueError("Missing manifest split")
            for split, pairs in splits.items():
                if len(pairs) != manifest["counts"][split]:
                    raise ValueError("Manifest count mismatch")
                for saved in pairs:
                    pair = ds.Pair(**saved)
                    if pair.task != task:
                        raise ValueError("Task mismatch in manifest")
                    for prompt in (pair.clean_prompt, pair.corrupt_prompt):
                        if prompt in prompts:
                            raise ValueError("Prompt overlaps between splits, seeds or demonstrations")
                        prompts.add(prompt)
    return manifest


def load_manifest(path):
    with Path(path).open() as stream:
        return validate_manifest(json.load(stream))


def manifest_records(harness, manifest, task, seed):
    splits = manifest["tasks"][task][str(seed)]
    prefix = "".join(f"{pair['clean_prompt']} {pair['clean_target']}.\n"
                     for pair in splits["fewshot"])
    return {name: [encode_pair(harness, ds.Pair(**pair), prefix) for pair in splits[name]]
            for name in ("train", "dev", "test")}


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--dev", type=int, default=24)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--fewshot", type=int, default=3)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    tokenizers = [AutoTokenizer.from_pretrained(model, local_files_only=True) for model in args.models]
    manifest = build_manifest(tokenizers, args.seeds, args.train, args.dev, args.test, args.fewshot)
    write_new(args.output, manifest)
    print(f"saved {args.output}: {manifest['sha256']}, counts={manifest['counts']}")
    for task, seeds in manifest["tasks"].items():
        pairs = [ds.Pair(**pair) for splits in seeds.values() for pair in splits["test"]]
        print(f"{task}: {len(pairs)} unique held-out pairs, shortcuts={ds.shortcut_scores(pairs)}")


if __name__ == "__main__":
    main()