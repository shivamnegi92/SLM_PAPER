"""C4 negative-results study: do common CPU inference optimizations actually pay
off for a shallow (depth-3) discriminative closed-set model?

Conditions on the same trained ATIS depth-3 model:
  (a) FP32 baseline
  (b) dynamic INT8 (torch quantize_dynamic over Linear layers)
  (c) torch.compile
  (d) ONNX export + onnxruntime (attempt; failure is itself a documented result)

Each: intent acc, slot F1, P50 latency. Report deltas + root cause.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from slmpaper.batch import prepare_batch
from slmpaper.data_registry import load_split
from slmpaper.evaluation import evaluate_model
from slmpaper.labels import build_intent_vocab, build_tag_vocab
from slmpaper.latency import measure_latency
from slmpaper.pruned_model import PrunedGenerativeClassifier

DATASET = "atis"
DEPTH = 3
MODEL_PATH = "models/gpt2"


def make_batches(examples, tok, i2, t2, bs):
    return [prepare_batch(examples[i:i + bs], tok, i2, t2) for i in range(0, len(examples), bs)]


def p50_latency(fn):
    lat = measure_latency(fn, inputs=list(range(55)), warmup=5)
    return lat["p50"] * 1000


def main():
    torch.manual_seed(42)
    train = load_split(DATASET, "train")
    import random
    random.Random(42).shuffle(train)
    test = [e for e in load_split(DATASET, "test")]
    intent2id = build_intent_vocab(train)
    tag2id = build_tag_vocab(train)
    id2intent = {v: k for k, v in intent2id.items()}
    id2tag = {v: k for k, v in tag2id.items()}
    test = [e for e in test if e.intent in intent2id]

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = PrunedGenerativeClassifier.from_pretrained_backbone(
        MODEL_PATH, depth=DEPTH, num_intents=len(intent2id), num_tags=len(tag2id),
        slot_loss_weight=2.0)

    tb = make_batches(train, tok, intent2id, tag2id, 16)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for ep in range(3):
        for b in tb:
            opt.zero_grad()
            out = model(input_ids=b.input_ids, attention_mask=b.attention_mask,
                        intent_labels=b.intent_labels, slot_labels=b.slot_labels)
            out.loss.backward()
            opt.step()
    model.eval()

    test_batches = make_batches(test, tok, intent2id, tag2id, 16)
    one = make_batches(test[:1], tok, intent2id, tag2id, 1)[0]
    results = {}

    def eval_and_time(m, label):
        met = evaluate_model(m, test_batches, id2intent, id2tag)
        def infer(_):
            with torch.no_grad():
                m(input_ids=one.input_ids, attention_mask=one.attention_mask)
        lat = p50_latency(infer)
        results[label] = {"intent_accuracy": met["intent_accuracy"],
                           "slot_f1": met["slot_f1"]["f1"], "p50_ms": lat}
        print(f"  {label:16s} intent={met['intent_accuracy']*100:.2f} "
              f"slotF1={met['slot_f1']['f1']*100:.2f} p50={lat:.2f}ms", flush=True)

    print("[negresults] (a) FP32 baseline", flush=True)
    eval_and_time(model, "fp32")

    print("[negresults] (b) dynamic INT8", flush=True)
    try:
        # commodity ARM/x86 CPUs need an explicit quantized engine selected
        for eng in ("qnnpack", "fbgemm"):
            if eng in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = eng
                break
        qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        eval_and_time(qmodel, f"dynamic_int8_{torch.backends.quantized.engine}")
    except Exception as e:
        results["dynamic_int8"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"    FAILED: {e}", flush=True)

    print("[negresults] (c) torch.compile", flush=True)
    try:
        cmodel = torch.compile(model)
        # warmup compile
        with torch.no_grad():
            cmodel(input_ids=one.input_ids, attention_mask=one.attention_mask)
        eval_and_time(cmodel, "torch_compile")
    except Exception as e:
        results["torch_compile"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"    FAILED: {e}", flush=True)

    print("[negresults] (d) ONNX export + onnxruntime", flush=True)
    try:
        import onnxruntime as ort  # noqa
        onnx_path = "results/_negresults_atis_d3.onnx"

        class TupleWrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, input_ids, attention_mask):
                o = self.m(input_ids=input_ids, attention_mask=attention_mask)
                return o.intent_logits, o.slot_logits

        wrapped = TupleWrap(model)
        torch.onnx.export(
            wrapped, (one.input_ids, one.attention_mask), onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["intent_logits", "slot_logits"],
            dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}},
            opset_version=17, dynamo=False,
        )
        sess = ort.InferenceSession(onnx_path)
        feed = {"input_ids": one.input_ids.numpy(), "attention_mask": one.attention_mask.numpy()}
        def infer_onnx(_):
            sess.run(None, feed)
        lat = p50_latency(infer_onnx)
        results["onnx"] = {"p50_ms": lat, "note": "latency only (accuracy identical to fp32 graph)"}
        print(f"  onnx             p50={lat:.2f}ms", flush=True)
    except Exception as e:
        results["onnx"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"    FAILED/unavailable: {e}", flush=True)

    Path("results").mkdir(exist_ok=True)
    Path("results/negresults_atis_d3.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
