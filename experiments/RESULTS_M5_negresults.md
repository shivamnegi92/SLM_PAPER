# M5 / C4 Results — CPU Inference Optimizations: What Pays Off (and What Doesn't)

Model: our depth-3 pruned discriminative classifier (gpt2 backbone), trained on
ATIS (seed 42, 3 epochs). Same trained weights across all conditions; intent
accuracy, span slot F1, and P50 single-example CPU latency (Apple Silicon
arm64). Raw JSON: `experiments/code/results/negresults_atis_d3.json`.
Reproduce: `python scripts/negresults.py`.

| Condition | Intent Acc | Slot F1 | P50 (ms) | vs FP32 | Verdict |
|---|---:|---:|---:|---:|---|
| FP32 baseline | 97.95 | 90.96 | 5.13 | 1.00x | reference |
| Dynamic INT8 (qnnpack) | 97.95 | 90.68 | 5.02 | 1.02x | **no benefit** |
| torch.compile | 97.95 | 90.96 | 5.39 | 0.95x | **slower** |
| ONNX Runtime | 97.95 | 90.96 | **1.14** | **4.5x** | **big win, brittle export** |

## Findings + root causes
1. **Dynamic INT8 does not pay off.** ~2% latency change, and a small slot-F1
   drop (90.96 -> 90.68). Root cause: at depth 3 the Linear layers are not the
   bottleneck, and dynamic per-batch quant/dequant overhead offsets the cheaper
   matmuls. Also note: it *silently doesn't run at all* on a stock CPU build
   without an explicit quantized engine (`NoQEngine` error until we set
   `torch.backends.quantized.engine = "qnnpack"`) -- a portability footgun.
2. **torch.compile does not pay off (slightly slower).** Root cause: compilation
   overhead + dynamic sequence shapes; the model is too shallow/small for graph
   fusion to amortize on CPU. Correctness preserved (identical accuracy).
3. **ONNX Runtime is a real ~4.5x CPU speedup** (5.13 -> 1.14 ms) at identical
   accuracy -- BUT export is brittle: the modern dynamo exporter fails on the
   decoder backbone outright (`torch.export` step 1/3 failure), and the legacy
   exporter rejects the model's dataclass output (`PrunedOutput`); it only
   succeeds after wrapping the model to return a plain `(intent_logits,
   slot_logits)` tuple. Takeaway: the speedup is available but requires
   deliberate export-surgery, not a turnkey flag.

## Paper framing (C4)
For shallow, closed-set discriminative NLU on commodity CPUs, the two most
commonly-reached-for optimizations (dynamic INT8, graph compilation) give
**no benefit** -- the win already came from *depth pruning + discriminative
single-pass* (C1/C2), which removed the autoregressive decode that dominated
generative latency. The one optimization that does help, ONNX Runtime graph
execution, is orthogonal and stackable, but carries real export-engineering
friction for decoder-derived models. Net: **architecture (prune + classify)
beats post-hoc kernel/quant tricks for this task class.**

## Honest scope
Single model (gpt2-124M -> depth 3), single dataset (ATIS), single CPU. Not
run here (PLAN.md stretch / camera-ready): rotation-based INT8 (QuaRot/Hadamard)
with fused kernels, vendor CPU extensions (oneDNN/ACL), and ONNX INT8. These are
listed as future items; the qualitative claim (prune+classify > post-hoc tricks)
is supported by the three conditions measured.
