"""Fine-tune the pruned discriminative classifier (C1 payoff) at a given depth
and evaluate intent+slot accuracy + latency -- validates Gate B: does the
probe-recommended depth match near-best empirical accuracy?

Example:
  python scripts/train_pruned.py --dataset atis --depth 3 --model-path models/gpt2 \
      --output results/atis_pruned_depth3.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer

from slmpaper.batch import prepare_batch
from slmpaper.data_registry import load_split
from slmpaper.evaluation import evaluate_model, evaluate_model_crf
from slmpaper.labels import build_intent_vocab, build_tag_vocab, has_only_known_bio_tags
from slmpaper.pruned_model import PrunedGenerativeClassifier


def make_batches(examples, tokenizer, intent2id, tag2id, batch_size):
    batches = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        batches.append(prepare_batch(chunk, tokenizer, intent2id, tag2id))
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--model-path", default="models/gpt2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--slot-loss-weight", type=float, default=2.0)
    ap.add_argument("--use-crf", action="store_true", help="CRF slot head instead of softmax")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"

    train = load_split(args.dataset, "train")
    test = load_split(args.dataset, "test")
    import random
    random.Random(args.seed).shuffle(train)
    if args.max_train:
        train = train[:args.max_train]

    intent2id = build_intent_vocab(train)
    tag2id = build_tag_vocab(train)
    id2intent = {v: k for k, v in intent2id.items()}
    id2tag = {v: k for k, v in tag2id.items()}
    test = [ex for ex in test if ex.intent in intent2id]
    test = [ex for ex in test if has_only_known_bio_tags(ex, tag2id)]
    print(f"[{args.dataset}] depth={args.depth} train={len(train)} test={len(test)} "
          f"intents={len(intent2id)} tags={len(tag2id)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PrunedGenerativeClassifier.from_pretrained_backbone(
        args.model_path, depth=args.depth, num_intents=len(intent2id), num_tags=len(tag2id),
        slot_loss_weight=args.slot_loss_weight, use_crf=args.use_crf,
    ).to(device)

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

    eval_fn = evaluate_model_crf if args.use_crf else evaluate_model
    metrics = eval_fn(model, test_batches, id2intent, id2tag, device=device)

    from slmpaper.latency import measure_latency
    one = make_batches(test[:1], tokenizer, intent2id, tag2id, 1)[0]

    def infer(_):
        with torch.no_grad():
            model(input_ids=one.input_ids, attention_mask=one.attention_mask)

    lat_s = measure_latency(infer, inputs=list(range(55)), warmup=5)
    lat = {k: (v * 1000 if k in ("mean", "p50", "p95") else v) for k, v in lat_s.items()}

    result = {
        "dataset": args.dataset, "model": args.model_path, "kind": "pruned_discriminative",
        "depth": args.depth, "use_crf": args.use_crf,
        "config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "slot_loss_weight": args.slot_loss_weight, "max_train": args.max_train,
                    "seed": args.seed},
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


if __name__ == "__main__":
    main()
