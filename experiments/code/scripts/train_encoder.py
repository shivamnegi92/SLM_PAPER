"""Train the JointBERT-style encoder baseline (B2) on a public dataset.

Produces real M1 baseline numbers: intent accuracy, slot span-F1, and latency.
Run from experiments/code/ with the models venv active. Model weights are
fetched from the standard Hugging Face endpoint by default; override via the
``HF_ENDPOINT`` environment variable when using an internal mirror.

Example:
  python scripts/train_encoder.py --dataset snips --epochs 3 --output results/snips_encoder.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from slmpaper.batch import prepare_batch
from slmpaper.data_registry import load_split
from slmpaper.evaluation import evaluate_model
from slmpaper.labels import build_intent_vocab, build_tag_vocab


def make_batches(examples, tokenizer, intent2id, tag2id, batch_size):
    batches = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        batches.append(prepare_batch(chunk, tokenizer, intent2id, tag2id))
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--slot-loss-weight", type=float, default=2.0)
    ap.add_argument("--max-train", type=int, default=0, help="0 = use all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"

    print(f"[{args.dataset}] loading data...", flush=True)
    train = load_split(args.dataset, "train")
    test = load_split(args.dataset, "test")
    import random
    random.Random(args.seed).shuffle(train)
    if args.max_train:
        train = train[:args.max_train]
    print(f"  train={len(train)} test={len(test)}", flush=True)

    intent2id = build_intent_vocab(train)
    tag2id = build_tag_vocab(train)
    id2intent = {v: k for k, v in intent2id.items()}
    id2tag = {v: k for k, v in tag2id.items()}
    # unseen test intents/tags map safely: filter test to known intents
    test = [ex for ex in test if ex.intent in intent2id]
    print(f"  intents={len(intent2id)} tags={len(tag2id)} test(kept)={len(test)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointEncoderModel_from(args, intent2id, tag2id).to(device)

    train_batches = make_batches(train, tokenizer, intent2id, tag2id, args.batch_size)
    test_batches = make_batches(test, tokenizer, intent2id, tag2id, args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in train_batches:
            opt.zero_grad()
            out = model(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                intent_labels=batch.intent_labels, slot_labels=batch.slot_labels,
            )
            out.loss.backward()
            opt.step()
            total += out.loss.item()
        print(f"  epoch {epoch+1}/{args.epochs} loss={total/max(len(train_batches),1):.4f}", flush=True)
    train_time = time.time() - t0

    metrics = evaluate_model(model, test_batches, id2intent, id2tag, device=device)

    # latency: single-example inference, P50/P95 (ms), warmup excluded
    from slmpaper.latency import measure_latency
    one = make_batches(test[:1], tokenizer, intent2id, tag2id, 1)[0]

    def infer(_):
        with torch.no_grad():
            model(input_ids=one.input_ids, attention_mask=one.attention_mask)

    lat_s = measure_latency(infer, inputs=list(range(55)), warmup=5)
    lat = {k: (v * 1000 if k in ("mean", "p50", "p95") else v) for k, v in lat_s.items()}

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "config": {"epochs": args.epochs, "batch_size": args.batch_size,
                    "lr": args.lr, "slot_loss_weight": args.slot_loss_weight,
                    "max_train": args.max_train, "seed": args.seed},
        "n_train": len(train), "n_test": len(test),
        "n_intents": len(intent2id), "n_tags": len(tag2id),
        "intent_accuracy": metrics["intent_accuracy"],
        "slot_f1": metrics["slot_f1"],
        "latency_ms": lat,
        "train_time_s": train_time,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def JointEncoderModel_from(args, intent2id, tag2id):
    from slmpaper.encoder_model import JointEncoderModel
    return JointEncoderModel.from_pretrained_backbone(
        args.model, num_intents=len(intent2id), num_tags=len(tag2id),
        slot_loss_weight=args.slot_loss_weight,
    )


if __name__ == "__main__":
    main()
