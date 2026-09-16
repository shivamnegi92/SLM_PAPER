"""Token-level slot probe (C1 extension): frozen backbone -> linear BIO-tag
classifier per candidate depth. Answers whether task-level slot information
becomes linearly accessible at the same depth as intent information, or
strictly later. Complements the intent probe in probe_sweep.py.

Example:
  python scripts/slot_probe.py --dataset atis --depths 1,2,3,4,5,6,7,8,9,10,11,12 \\
      --output results/atis_slot_probe_dense.json
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
from slmpaper.batch import prepare_batch
from slmpaper.data_registry import load_split
from slmpaper.depth_selection import pick_depth
from slmpaper.labels import build_intent_vocab, build_tag_vocab
from slmpaper.probe import train_linear_probe


def extract_token_features_all_depths(model, tokenizer, examples, intent2id, tag2id,
                                       depths, batch_size=16, max_length=64):
    """One forward pass per batch; keep only tokens with valid slot labels.
    We filter -100 positions per batch before cross-batch concat so variable
    per-batch padding doesn't break the cat.
    """
    per_depth_feats = {d: [] for d in depths}
    per_batch_labels = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            batch = prepare_batch(chunk, tokenizer, intent2id, tag2id, max_length=max_length)
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                        output_hidden_states=True)
            labels_flat = batch.slot_labels.reshape(-1)
            valid = labels_flat != -100
            per_batch_labels.append(labels_flat[valid])
            for d in depths:
                h = out.hidden_states[d]
                h_flat = h.reshape(-1, h.shape[-1])
                per_depth_feats[d].append(h_flat[valid].float())
    all_labels = torch.cat(per_batch_labels, dim=0)
    result = {d: torch.cat(per_depth_feats[d], dim=0) for d in depths}
    return result, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-path", default="models/gpt2")
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--probe-epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depths", default="",
                    help="Comma-separated depths override; empty = 1..L.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    train = load_split(args.dataset, "train")
    import random
    random.Random(args.seed).shuffle(train)
    if args.max_train:
        train = train[:args.max_train]
    # Build the tag vocabulary from the full slice BEFORE train/val split so
    # rare tags that only fall on the val side don't KeyError the aligner.
    intent2id_full = build_intent_vocab(train)
    tag2id = build_tag_vocab(train)
    n_val = max(1, int(len(train) * args.val_frac))
    val_examples, train_examples = train[:n_val], train[n_val:]

    # Only slot-bearing datasets are meaningful; short-circuit if no non-O tags exist.
    has_slots = any(any(t != "O" for t in ex.bio_tags) for ex in train_examples)
    if not has_slots:
        print(f"[{args.dataset}] intent-only dataset (no BIO slots), skipping slot probe")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "dataset": args.dataset, "skipped": "intent-only dataset"}, indent=2))
        return

    intent2id = build_intent_vocab(train_examples)
    val_examples = [e for e in val_examples if e.intent in intent2id]
    print(f"[{args.dataset}] slot-probe train={len(train_examples)} val={len(val_examples)} "
          f"tags={len(tag2id)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.model_path)
    n_layer = get_num_layers(model)
    if args.depths:
        depths = sorted({int(d) for d in args.depths.split(",") if d.strip()})
        depths = [d for d in depths if 1 <= d <= n_layer]
    else:
        depths = list(range(1, n_layer + 1))
    print(f"  candidate depths (of {n_layer}): {depths}", flush=True)

    train_feats, train_labels = extract_token_features_all_depths(
        model, tokenizer, train_examples, intent2id, tag2id, depths)
    val_feats, val_labels = extract_token_features_all_depths(
        model, tokenizer, val_examples, intent2id, tag2id, depths)

    depth_to_acc = {}
    for d in depths:
        acc = train_linear_probe(
            train_feats[d], train_labels, val_feats[d], val_labels,
            num_classes=len(tag2id), epochs=args.probe_epochs,
        )
        depth_to_acc[d] = acc
        print(f"  depth={d:>3}  slot_tag_probe_val_acc={acc:.4f}", flush=True)

    recommended = pick_depth(depth_to_acc, epsilon=args.epsilon)
    result = {
        "dataset": args.dataset, "model": args.model_path, "n_layer_full": n_layer,
        "config": {"max_train": args.max_train, "val_frac": args.val_frac,
                   "epsilon": args.epsilon, "probe_epochs": args.probe_epochs,
                   "seed": args.seed, "probe_type": "token_slot_tag"},
        "n_tags": len(tag2id),
        "depth_to_slot_probe_accuracy": {str(k): v for k, v in depth_to_acc.items()},
        "recommended_depth": recommended,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
