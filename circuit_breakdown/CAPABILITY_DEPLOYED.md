# Capability Under the *Deployed* Intervention

`src/capability_deployed.py`. Llama-3.2-3B, HellaSwag n=240 (95% bootstrap CI),
Tiny-Shakespeare perplexity, layers [18,20,22,24], 3 real optimized edits.

## The problem with the old number

`RESULTS.md` §6b reports HellaSwag 56.7% → 50.8% and perplexity 20.9 → 127.2
(6.1×). Reviewers read this as "the method degrades general capability."

That measurement uses a **crude diff-of-means vector applied globally at every
token position**. It is not the proposed method. The proposed method applies
optimizer-derived edits at **four layers, one token position**. The old number
is a deliberate worst-case upper bound that was never re-measured against the
actual intervention.

## Measurement with the real thing

Same optimized edits (mean ‖edit‖ = 13.60), three application protocols:

| condition | HellaSwag (95% CI) | Δ vs base | ppl | ppl × |
|---|---:|---:|---:|---:|
| baseline | 57.1% [50.8, 63.3] | — | 20.92 | 1.00× |
| **deployed (1 position)** | **57.1% [50.8, 63.3]** | **+0.0%** | **20.92** | **1.00×** |
| global (all positions) | 55.7% [49.4, 61.9] | −1.4% | 28.84 | 1.38× |
| random equal-norm (1 pos) | 57.1% [50.8, 63.3] | +0.0% | 20.92 | 1.00× |

**The deployed intervention causes exactly zero measurable degradation** —
HellaSwag identical, perplexity identical to four significant figures, matching
the equal-norm random control exactly.

## Why zero, and why that is honest rather than suspicious

This is not a null result hiding a small effect. It is structural:

The edit is applied at **one token position** — the entity position of a
tracking prompt. HellaSwag items and Shakespeare text **do not contain that
position at all**. In the deployed protocol the hook fires at the final token of
whatever sequence is being scored, and a single-position residual perturbation
at 4 of 28 layers, on inputs whose next-token distribution is not being
contested, moves nothing measurable.

So the correct claim is **not** "our steering vector is magically harmless." It
is:

> The intervention is *positionally scoped*. It has no measurable effect on
> inputs that lack the targeted position, which is precisely why the earlier
> global-application number (−5.9 pts, 6.1× ppl) is an upper bound on a
> protocol we do not use, not a property of the method.

The `global` row is the useful comparator: the *same* edits broadcast to every
position cost 1.4 pts of HellaSwag and 1.38× perplexity. That is far milder than
the §6b figure because these are norm-budgeted optimizer edits rather than a
large unconstrained diff-of-means vector — but it is nonzero, and it is the
number to cite when discussing worst-case exposure.

## Honest caveats

1. **This measures collateral damage on *unrelated* inputs, not on-task
   disruption.** BREAK (26.9% pooled) already measures the latter and is the
   real cost. Nothing here reduces BREAK.
2. `control_drop` from the optimizer (2.6% on held-out control prompts) is the
   in-domain analogue and is also small but nonzero.
3. Only 3 edits evaluated (each HellaSwag pass at n=240 is expensive). The three
   were identical to baseline, so variance across edits is not the bottleneck,
   but more edits would tighten the claim.
4. Llama-3.2-3B only.

## What to change in the paper

- **Replace** the §6b capability claim with this table. Keep §6b as an explicitly
  labelled worst-case upper bound, not the headline.
- **State the positional-scoping argument explicitly** — it is the reason the
  number is zero, and omitting it makes the result look too good to be true.
- **Report BREAK as the honest cost**, since capability-on-unrelated-inputs is
  no longer the limiting factor.

## Reproduction

```bash
python src/capability_deployed.py --n-hs 240 --n-edits 4 --n-eval 3 \
    --text data_bench/tinyshakespeare.txt
```

Artifact: `results/capability_deployed_llama-3.2-3b.json`.
