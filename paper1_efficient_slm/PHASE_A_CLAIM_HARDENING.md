# Phase A — Claim Hardening (ICLR Paper 1)

## Frozen one-sentence thesis
For closed-set intent detection and slot filling, converting a generative decoder into a probe-pruned discriminative model provides a pre-finetune depth-selection rule (C1), eliminates structured-output parse failures by construction (C2), and delivers a strong latency-quality tradeoff with clearly bounded regimes where deeper models still help (C3/C4 context).

## Claim → Evidence map

| Claim clause | Primary evidence | Backup / robustness evidence | Status |
|---|---|---|---|
| C1: Probe can choose shallow depth before full fine-tuning | `PROJECT_REPORT.md` §4.2 probe sweep; `OUTLINE.md` §5.2 | Multi-seed depth-3 vs depth-12 in `PROJECT_REPORT.md` §4.3 / `OUTLINE.md` §5.3 | Strong, with dataset-specific caveat |
| C2: Discriminative conversion removes parse-failure mode | `PROJECT_REPORT.md` §4.1 and §4.5; `OUTLINE.md` §5.1 and §5.4 | Structural argument in method sections (`PROJECT_REPORT.md` §2.2, `OUTLINE.md` §3.5) | Strong |
| Latency advantage is material, not cosmetic | `PROJECT_REPORT.md` §4.1 (ATIS 359ms→5.1ms), §4.6 ONNX/CPU | Pareto depth curves in `OUTLINE.md` §5.3 and figure refs | Strong |
| Tradeoff boundaries are explicitly acknowledged | `PROJECT_REPORT.md` §4.3/§4.4 and §5 positioning | `OUTLINE.md` seed table + fair head-to-head table | Strong and honest |
| C3 scope: implicit-slot premise does not hold on current benchmarks | `PROJECT_REPORT.md` §4.5 | `OUTLINE.md` §7 limitation statement | Strong negative-result framing |
| C4: common CPU tricks often underperform here | `PROJECT_REPORT.md` §4.6 | `OUTLINE.md` §6 | Strong for tested stack |

## Explicit null hypotheses and fail conditions

### C1 (probe-guided depth selection)
- **H0-C1:** Probe-selected depth is not predictive of best/near-best fine-tuned depth.
- **Fail C1 if:** probe-chosen depth misses best depth by >1 layer on majority datasets, OR incurs >2pt intent drop without a latency justification.
- **Downgrade action:** reframe C1 as "empirical depth characterization" instead of predictive procedure.

### C2 (reliability via discriminative conversion)
- **H0-C2:** Discriminative conversion does not improve output reliability versus generative decoding.
- **Fail C2 if:** parse-failure is non-zero in discriminative path due to schema/postprocess faults, OR generative baseline parse-failure is negligible across all relevant settings.
- **Downgrade action:** keep speed claim; remove reliability headline.

### C3 (implicit-slot scope finding)
- **H0-C3:** Benchmarks contain non-trivial implicit slot rates that span-taggers cannot capture.
- **Fail C3 if:** measured implicit-slot rate is materially >0 and no recovery path is evaluated.
- **Downgrade action:** present as limitation requiring new datasets / follow-up paper.

### C4 (negative-results on CPU optimizations)
- **H0-C4:** Dynamic INT8/compile/ONNX provide no differential impact in this setup.
- **Fail C4 if:** measurements are inconsistent across reruns/hardware or not reproducible from scripts.
- **Downgrade action:** move C4 to appendix as environment-specific observation.

## Immediate Phase A completion checklist
- [ ] Embed this claim sentence in abstract/introduction verbatim (or tighter variant).
- [ ] Add one "What we do not claim" paragraph in introduction/discussion.
- [ ] Ensure every headline claim in `paper.tex` cites one concrete table/figure.
- [ ] Mark C1 as conditional by dataset granularity (already observed on MASSIVE/BANKING77).
- [ ] Keep matched-budget fairness table in main text (not appendix).
