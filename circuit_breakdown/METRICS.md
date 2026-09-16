# METRICS — exact causal criteria (no hand-waving)

These replace the gameable "modify <0.05% of activation vectors" line. Every
number in the paper maps to one of these definitions.

## Notation
- Clean run `x_c` (target `t_c`), corrupt run `x_*` (target `t_*`), with
  `t_c != t_*` (enforced by the dataset generator).
- **Logit difference** on a run: `LD = logit(t_c) - logit(t_*)`.
- `LD_clean` on the clean run (large, positive by construction).
- `LD_corrupt` on the corrupt run (small / negative).

## 1. Normalized logit-difference recovered (faithfulness) — HEADLINE
Patch component set `S` from clean into the corrupt run, giving `LD_patched(S)`.

```
faithfulness(S) = (LD_patched(S) - LD_corrupt) / (LD_clean - LD_corrupt)
```

- `1.0` = patching `S` fully restores clean behavior.
- `0.0` = `S` does nothing (matches corrupt).
- Report averaged over the prompt distribution with 95% CIs (bootstrap).

## 2. Completeness & minimality (Wang et al., 2022)
- **Completeness / sufficiency:** patching only `S` recovers behavior
  (`faithfulness(S)` high, e.g. > 0.7).
- **Minimality:** for every `c in S`, removing `c` measurably drops faithfulness
  (each member earns its place). Report the drop per component.

## 3. KL divergence to clean
`KL( P_clean(vocab) || P_patched(S)(vocab) )` — guards against "restores the
target token but scrambles everything else." Lower is better.

## 4. Circuit size (the honest "small number")
`|S|` = number of attention heads / components in the minimal faithful circuit,
with variance across the prompt distribution. Example honest headline:
"tracking is carried by 4 heads across 2 layers (recovering 82% +/- 5% of the
logit diff), vs 6% +/- 3% for a random 4-head baseline."

## 5. Random-patching baseline
For each reported `|S|`, sample K random component sets of the same size; report
mean +/- CI faithfulness. The real circuit must dominate this.

## 6. Intervention (Phase 3) metrics
- **Target override:** full-vocabulary argmax equals the clean target on the
  counterfactual prompt. The target is known during per-example optimization;
  this is not evidence of repairing naturally wrong answers.
- **New target rate:** override among counterfactual prompts whose baseline
  prediction was not already the clean target.
- **Same-sign damage:** positive edit changes a baseline-correct clean answer
  to a wrong answer, divided by the number of baseline-correct clean examples.
- **Negative-edit disruption:** the same conditioned calculation for a negative
  edit. The legacy `break` field counts unconditional negative-edit errors and
  includes already-wrong examples; do not treat it as collateral damage.
- **No eligible examples:** report `null` and a zero denominator, not 0% damage.
- **Continuous outputs:** target probability, target-vs-best-other margin,
  two-way probability change, and third-token outputs accompany top-1 metrics.
- **Capability regression:** paired accuracy and continuation-loss differences
  when the intervention affects scored positions. A confidence interval that
  overlaps baseline is not an equivalence test. Report per-edit uncertainty and
  compare against a predeclared tolerated loss before claiming preservation.

## 6b. Identity and uncertainty requirements

New runs record semantic sample IDs, token-input hashes, seed and exact
comparison configuration. Analysis rejects missing runs, duplicate examples,
incompatible configurations, and unequal sample sets. Binary rates use Wilson
intervals rather than degenerate bootstrap [0,0] or [1,1] bounds. Paired binary
differences use gain/loss Wilson bounds and exact McNemar tests; continuous
differences use paired seed-stratified bootstrap and sign permutations with a
Monte Carlo plus-one correction. Report per-seed results and distinguish
unique prompts from repeated runs. Equal observed rates or `p=1` are not proof
of equivalence. See [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md).

## 7. Compute-cost metrics (consumer-hardware story)
Report as first-class results: wall-clock per localization pass, total audit
wall-clock, peak RSS/unified-RAM, device (mps/cpu), dtype (fp32).

## Secondary framing line (allowed, de-emphasized)
The "% of activations/params touched" line may appear once, clearly labeled as a
framing statistic with its denominator stated explicitly. It is never a headline.
