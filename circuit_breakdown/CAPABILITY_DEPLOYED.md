# Capability Under the *Deployed* Intervention

> **CORRECTION - 2026-09-06:** The legacy single-position rows below do not
> establish capability preservation. Their hooks edited the final sequence
> token, whose prediction neither scorer used. The implementation is now
> corrected and tested; a representative corrected capability study is pending.

## What Failed

[src/capability_deployed.py](src/capability_deployed.py) previously edited the
last token of each complete sequence. Both scoring functions in
[src/capability.py](src/capability.py) use logits only through position `T-2`
to predict tokens through `T-1`. Causal attention prevents an edit at `T-1`
from changing any earlier scored logits. Exactly unchanged scores were
therefore expected, including for an arbitrarily strong last-token edit.

There was no entity-trigger detector in this evaluator. The old explanation
that generic text lacked a targeted entity position was incorrect. A global
edit can affect scored predictions, but it is a stress test, not a mathematical
upper bound on damage from a different single-position intervention.

## Historical Results

Preserved for provenance only: Llama-3.2-3B, HellaSwag n=240, one text sample,
layers [18,20,22,24], three evaluated optimized edits (mean total norm 13.60).
The intervals below are the original reported bootstrap intervals.

| condition | HellaSwag (95% CI) | Δ vs base | ppl | ppl × |
|---|---:|---:|---:|---:|
| baseline | 57.1% [50.8, 63.3] | — | 20.92 | 1.00× |
| legacy single position (unscored) | 57.1% [50.8, 63.3] | +0.0% | 20.92 | 1.00× |
| global (all positions) | 55.7% [49.4, 61.9] | −1.4% | 28.84 | 1.38× |
| legacy random position (unscored) | 57.1% [50.8, 63.3] | +0.0% | 20.92 | 1.00× |

The original [JSON artifact](results/capability_deployed_llama-3.2-3b.json)
has not been overwritten. Do not cite the two unscored rows as safety evidence
or replace the older global stress results with them.

## Corrected Protocol

The new `active_prefix_v1` evaluator measures **active exposure on unrelated
inputs**, not a validated deployment policy or entity-triggered intervention.

- HellaSwag: edit the last **context** token for each ending and score its
  continuation. Reject a tokenizer boundary that is not an exact prefix rather
  than silently scoring the wrong tokens.
- Text: edit the last token of a fixed prefix and score only subsequent tokens.
  `--ppl-prefix-tokens` includes BOS and defaults to 32; `--ppl-max-tokens`
  defaults to 768. Baseline and all conditions use the same scored span.
- Conditions: baseline, optimized active-prefix edit, equal-norm random-prefix
  edit, zero-prefix edit, and global exposure. The zero condition must reproduce
  baseline scores within the stated numerical tolerance.
- Record individual correctness flags, token losses, edit positions/norms,
  benchmark item hashes, text hash, optimizer settings, seed and protocol version.
- Use Wilson accuracy intervals and approximate 95% paired bounds formed from
  gain/loss Wilson intervals with a Bonferroni adjustment. Report these per edit.
  Never average interval endpoints or count the same items repeatedly as new
  independent examples. Aggregates across edits are explicitly descriptive.
- Report text loss/perplexity changes descriptively; a single contiguous text
  sample does not support an independent-token CI or a capability-safety claim.

New output defaults to a separate `results/capability_active_prefix_v1/`
directory, with names `capability_active_prefix_<model>_s<seed>.json`.
Existing outputs are rejected; use a fresh `--outdir` for another run.

## Verification Completed

- **15 CPU regression tests pass**, including a known positive control,
  legacy last-token no-op, zero edit, paired interval edge cases, changing
  continuation lengths, hook cleanup on errors and overwrite protection.
- **2,880 continuation boundaries checked:** the first 240 cached HellaSwag
  items (four endings each) across Llama, Phi and Nemotron tokenizers all passed
  the exact-prefix check. This validates those inputs, not every possible text.
- **Offline Llama/MPS smoke passed:** two HellaSwag items, one optimized edit,
  one optimizer step, prefix 16, maximum 96 text tokens (80 scored), four layers.
  Active-prefix token losses changed; the zero-edit losses matched baseline.
  The run completed in approximately 31 seconds including loading.

The [smoke artifact](results/capability_active_prefix_v1/smoke_20260906/capability_active_prefix_llama-3.2-3b_s0.json)
is **integration evidence only**. Its two items and one short text sample are
not a replacement for the n=240 historical study or a meaningful capability
estimate. No full corrected benchmark, ARC/MMLU run or cross-model capability
study has been completed in this implementation stage.

## Remaining Evidence

1. Freeze examples, edit selection and tolerated loss before a representative
   corrected benchmark. Use multiple edits and text windows; retain paired data.
2. Separate same-sign damage from negative-edit BREAK. The current tracking
   BREAK metric uses the opposite-sign edit and includes baseline-wrong cases;
   it cannot be interpreted as the normal edit's collateral error rate.
3. The optimizer guard uses three reused generic prompts, not held-out benchmark
   accuracy. Small `control_drop` does not establish capability preservation.
4. Add ARC-Easy/MMLU only after confirming its scorer uses active positions.
   Overlapping CIs are not an equivalence test. Report uncertainty or measured
   degradation when a predeclared tolerance cannot be established.

The tracked order is in [PENDING_PLAN.md](PENDING_PLAN.md).

## Commands

From the `circuit_breakdown` directory, run the CPU regressions:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests \
    -p 'test_capability_scoring.py' -v
```

A corrected benchmark command for a **future**, predeclared run (not executed
as part of this stage):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python src/capability_deployed.py \
    --model ../llama-3.2-3b --device mps --layers 18 20 22 24 \
    --n-hs 240 --n-edits 4 --n-eval 3 --ppl-prefix-tokens 32 \
    --text data_bench/tinyshakespeare.txt \
    --outdir results/capability_active_prefix_v1/benchmark_001
```
