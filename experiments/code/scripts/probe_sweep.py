"""Probe-guided depth selection (C1): sweep candidate depths on the FROZEN gpt2
backbone, train a cheap linear probe per depth on intent labels, and pick the
shallowest depth within epsilon of best probe accuracy -- all *before* paying
for real fine-tuning.

Example:
  python scripts/probe_sweep.py --dataset atis --model-path models/gpt2 \
      --output results/atis_probe_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModel, AutoTokenizer

from slmpaper.backbone import get_num_layers

from slmpaper.data_registry import load_split
from slmpaper.depth_selection import pick_depth
from slmpaper.labels import build_intent_vocab
from slmpaper.probe import pool_hidden_states, train_linear_probe


def extract_features_all_depths(model, tokenizer, examples, depths, batch_size=16, max_length=64):
    """One forward pass per batch (output_hidden_states=True); pool at each depth."""
    feats = {d: [] for d in depths}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            enc = tokenizer([ex.text for ex in chunk], padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt")
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        output_hidden_states=True)
            # hidden_states[0] = embeddings, hidden_states[k] = after block k
            for d in depths:
                pooled = pool_hidden_states(out.hidden_states[d], enc["attention_mask"])
                feats[d].append(pooled)
    return {d: torch.cat(v, dim=0) for d, v in feats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-path", default="models/gpt2")
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--probe-epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depths", default="",
                    help="Comma-separated depths override, e.g. 1,2,3,4,5,6,7,8,9,10,11,12. "
                         "Empty (default) uses {L/4, L/2, 3L/4, L} where L is backbone depth.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    train = load_split(args.dataset, "train")
    import random
    random.Random(args.seed).shuffle(train)
    if args.max_train:
        train = train[:args.max_train]
    n_val = max(1, int(len(train) * args.val_frac))
    val_examples, train_examples = train[:n_val], train[n_val:]

    intent2id = build_intent_vocab(train_examples)
    # drop any val examples with an intent unseen in train (can't score them)
    val_examples = [e for e in val_examples if e.intent in intent2id]
    print(f"[{args.dataset}] probe-train={len(train_examples)} probe-val={len(val_examples)} "
          f"intents={len(intent2id)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.model_path)
    n_layer = get_num_layers(model)
    if args.depths:
        depths = sorted({int(d) for d in args.depths.split(",") if d.strip()})
        # Clamp to a valid range so a bad flag doesn't waste a run.
        depths = [d for d in depths if 1 <= d <= n_layer]
    else:
        depths = sorted(set([max(1, n_layer // 4), max(1, n_layer // 2),
                              max(1, 3 * n_layer // 4), n_layer]))
    print(f"  candidate depths (of {n_layer}): {depths}", flush=True)

    train_labels = torch.tensor([intent2id[e.intent] for e in train_examples])
    val_labels = torch.tensor([intent2id[e.intent] for e in val_examples])

    train_feats = extract_features_all_depths(model, tokenizer, train_examples, depths)
    val_feats = extract_features_all_depths(model, tokenizer, val_examples, depths)

    depth_to_acc = {}
    for d in depths:
        acc = train_linear_probe(
            train_feats[d], train_labels, val_feats[d], val_labels,
            num_classes=len(intent2id), epochs=args.probe_epochs,
        )
        depth_to_acc[d] = acc
        print(f"  depth={d:>3}  probe_val_acc={acc:.4f}", flush=True)

    recommended = pick_depth(depth_to_acc, epsilon=args.epsilon)
    result = {
        "dataset": args.dataset, "model": args.model_path, "n_layer_full": n_layer,
        "config": {"max_train": args.max_train, "val_frac": args.val_frac,
                    "epsilon": args.epsilon, "probe_epochs": args.probe_epochs, "seed": args.seed},
        "depth_to_probe_accuracy": {str(k): v for k, v in depth_to_acc.items()},
        "recommended_depth": recommended,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
