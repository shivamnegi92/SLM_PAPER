"""Train + eval the generative SLM baseline (B1): emit {"intent","slots"} JSON.

This is the C2 comparison point: generation is slower and can produce malformed
structured output (parse failures) -- both of which the discriminative model
eliminates. Uses a local model folder (see models/gpt2) to avoid the metadata
proxy limitation.

Example:
  python scripts/train_generative.py --dataset atis --model-path models/gpt2 \
      --epochs 3 --max-eval 300 --output results/atis_generative.json
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from slmpaper.data_registry import load_split
from slmpaper.gen_format import build_causal_labels, format_prompt, format_target
from slmpaper.generative import parse_structured_output
from slmpaper.metrics import intent_accuracy
from slmpaper.value_metrics import value_slot_f1


def encode_example(tokenizer, ex, max_length):
    prompt = format_prompt(ex) + " "
    target = format_target(ex) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = build_causal_labels(input_ids, prompt_len=len(prompt_ids))
    return input_ids, labels


def collate(tokenizer, rows, device):
    maxlen = max(len(r[0]) for r in rows)
    pad_id = tokenizer.pad_token_id
    input_ids, attn, labels = [], [], []
    for ids, labs in rows:
        n = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * n)
        attn.append([1] * len(ids) + [0] * n)
        labels.append(labs + [-100] * n)
    return (torch.tensor(input_ids, device=device),
            torch.tensor(attn, device=device),
            torch.tensor(labels, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-path", default="models/gpt2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=48)
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
    if args.max_eval:
        test = test[:args.max_eval]
    print(f"[{args.dataset}] train={len(train)} eval={len(test)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    enc_train = [encode_example(tokenizer, ex, args.max_length) for ex in train]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        for i in range(0, len(enc_train), args.batch_size):
            rows = enc_train[i:i + args.batch_size]
            input_ids, attn, labels = collate(tokenizer, rows, device)
            opt.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            total += out.loss.item()
            n += 1
        print(f"  epoch {epoch+1}/{args.epochs} loss={total/max(n,1):.4f}", flush=True)
    train_time = time.time() - t0

    # ---- generation-based eval ----
    model.eval()
    raw_outputs, pred_intents, gold_intents = [], [], []
    pred_slots, gold_slots = [], []
    latencies = []
    with torch.no_grad():
        for k, ex in enumerate(test):
            prompt = format_prompt(ex) + " "
            ids = tokenizer(prompt, return_tensors="pt").to(device)
            t = time.perf_counter()
            gen = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tokenizer.pad_token_id)
            if k >= 5:  # warmup first 5
                latencies.append((time.perf_counter() - t) * 1000)
            completion = tokenizer.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            raw_outputs.append(completion)
            parsed = parse_structured_output(completion)
            if parsed is None:
                pred_intents.append("__PARSE_FAIL__")
                pred_slots.append({})
            else:
                pred_intents.append(parsed["intent"])
                slots = parsed["slots"]
                # coerce values to list form for value_slot_f1
                norm = {kk: (vv if isinstance(vv, list) else [vv]) for kk, vv in slots.items()}
                pred_slots.append(norm)
            gold_intents.append(ex.intent)
            gold_slots.append(ex.slots)

    from slmpaper.generative import parse_failure_rate
    import statistics
    lat = {
        "n": len(latencies),
        "mean": statistics.mean(latencies) if latencies else 0.0,
        "p50": statistics.median(latencies) if latencies else 0.0,
        "p95": (sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0.0),
    }
    result = {
        "dataset": args.dataset,
        "model": args.model_path,
        "kind": "generative",
        "config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "max_train": args.max_train, "max_eval": args.max_eval,
                    "max_new_tokens": args.max_new_tokens, "seed": args.seed},
        "n_train": len(train), "n_eval": len(test),
        "intent_accuracy": intent_accuracy(pred_intents, gold_intents),
        "slot_value_f1": value_slot_f1(pred_slots, gold_slots),
        "parse_failure_rate": parse_failure_rate(raw_outputs),
        "latency_ms": lat,
        "train_time_s": train_time,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
