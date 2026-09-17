# Circuit Breakdown — Auditing & Overriding Tracking Circuits in SLMs

> Subproject of `SLM_PAPER`. **Public-benchmark / public-model only, fully
> self-contained.** No proprietary data, framing, or results. See the parent
> `../COMPLIANCE.md`.

---

## CURRENT PAPER (2026-09-17) — start here

The active paper is **[`paper/PAPER_DRAFT.md`](paper/PAPER_DRAFT.md)**
(rendered: [`paper/paper.html`](paper/paper.html)). Everything below the
"Historical context" divider predates it and points at a superseded
manuscript; read this section first.

**Question.** Can an activation intervention be behaviourally effective,
donor-specific, norm-controlled and dose-graded, yet still fail to control the
reasoning variable it appears to manipulate?

**Answer, in this setting: yes.** Selectivity is **0/240** in two model
families, 95% CI [0.0%, 1.6%].

| | Phi-3.5-mini (primary) | Llama-3.2-3B |
|---|---|---|
| completeness | 71.7% [65.7, 77.0] | 44.2% [35.6, 53.1] |
| **selectivity** | **0/240 = 0.0%** [0.0, 1.6] | **0/240 = 0.0%** [0.0, 1.6] |
| graded probes | 4 | 3 |

The clearest single result: both models answer *"what object did X have at the
start?"* with **100% accuracy unintervened**, and with a **person's name**
after an intervention that was supposed to set only *who currently holds* the
object.

### Verify without models or a GPU (under 1 second)

```bash
python src/verify_paper_numbers.py   # 44 checks against the frozen snapshot
./scripts/build_paper.sh             # verify, then render paper/paper.html
```

`build_paper.sh` refuses to render if any quoted number disagrees with
`results/frozen_e91985c/`.

### Reproducing (models required; ~25 min/model for the main panel)

```bash
python src/test_cross_question_panel.py --model phi-3.5-mini --n 40 --seeds 21 22 23
python src/validate_selectivity_metric.py --model phi-3.5-mini --n 40
python src/make_paper_figures.py && python src/make_selectivity_explainer.py
```

Seeds resample items *and* the few-shot prefix, so re-runs are statistically
equivalent, **not identical**. Compare against `results/frozen_e91985c/`.

Model weights are gitignored (29 GB) and resolve to the parent directory:
`../phi-3.5-mini`, `../llama-3.2-3b`, `../nemotron-mini-4b`. Fetch via
`../GET_MODELS.ipynb`. **Do not quantize** — 4-bit distorts activation
geometry and invalidates these measurements.

### Three caveats to read before quoting any number

1. **Phi is primary; Llama corroborates selectivity only.** Llama's
   `current_holder` competence is 46.7% at n=120, below the gate, so its
   completeness sits on three probes against Phi's four. The two completeness
   figures are **not comparable** and the paper does not compare them.
2. **The cross-question diagnostic is not novel.** RAVEL (arXiv:2402.17700)
   established effectiveness-vs-selectivity for static entity attributes; MIB
   (arXiv:2504.13151) reports full-vector interventions failing it. The
   contribution is the extension to a *sequentially computed* state.
3. **Three hypotheses were tested and falsified** and are kept for provenance,
   not hidden: the compact-basis mechanistic reading (it was answer-token
   transport), the novelty of the panel, and non-uniqueness of successful
   edits (median pairwise cosine +0.840/+0.912/+0.948 — they converge). See
   [`REVIEWER_GAP_TRACKER.md`](REVIEWER_GAP_TRACKER.md).

### Key paths

| path | what |
|---|---|
| `paper/PAPER_DRAFT.md` | the paper |
| `paper/NEXT_STEPS_PAPER.md` | venue decision + roadmap |
| `results/frozen_e91985c/` | **the snapshot behind every number; cannot be regenerated** |
| `results/grid_pilot/` | layer x position viability pilot — verdict: not viable |
| `src/verify_paper_numbers.py` | the 44-check guard |
| `REVIEWER_GAP_TRACKER.md` | full provenance incl. falsified claims |

---

## Historical context (pre-2026-09-07)

*The material below describes the earlier study phase and refers to
`paper/MANUSCRIPT.md`, which the current draft supersedes. Retained for
provenance.*

**Current research question:** When do answer-sensitive activation differences
also support target-informed control, and what damage does that edit cause?
The existing local models are **Llama-3.2-3B, Phi-3.5-mini, and
Nemotron-Mini-4B**. Gemma was part of the historical proposal, not a completed
experiment. The current fp32 study uses shared examples, per-layer relative
budgets and corrected active-position capability scoring.

No universal control limitation, negligible-capability-loss claim, or acceptance
forecast is established by the legacy tables. See the corrected measurement
definitions and frozen protocol before interpreting results.

**2026-09-07 status:** the declared matrix and post-run consistency audit are
complete. [Detailed results](paper/RESULTS_DETAILS.md) include all per-seed
tables and capability controls. Full-space target override was 100% in the
six eligible comparisons, with track8 at 8.7%-56%. These are per-example,
target-informed edits, not a reusable editor. General capability preservation
was not established, and all harder-task competence gates failed.

See [reproduction commands and limits](docs/REPRODUCIBILITY.md) and
[verified submission requirements](docs/SUBMISSION_CHECKLIST.md) before sharing
the paper or artifacts. Original local outputs are not automatically anonymous.

## Read these in order
1. [PENDING_PLAN.md](PENDING_PLAN.md) - completed evidence and genuinely remaining work.
2. [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md) - frozen samples, objectives and comparisons.
3. [METRICS.md](METRICS.md) and [CAPABILITY_DEPLOYED.md](CAPABILITY_DEPLOYED.md) -
  baseline-aware outcomes and the correction to the old capability test.
4. [paper/MANUSCRIPT.md](paper/MANUSCRIPT.md) - manuscript draft with scoped claims.
5. [PLAN.md](PLAN.md) - historical 15-week roadmap, retained for context.

## Why this exists (the bug we fixed)
The original blueprint's minimal pair had the **same answer on both sides**
(`...London→Paris→Berlin` and `...London→Tokyo→Berlin` both answer "Berlin"),
so activation patching would have measured **noise**. We replaced it with two
task families whose corruption genuinely changes the target:
- **`intermediate`** — query a non-final state ("the location *before* Berlin").
- **`transfer`** - object-passing text with a final distractor transfer. A
  query-aware last-recipient rule solves it; it is not proof of composition.
- **`container_swap`** - simulator-backed box swaps with changed initial
  locations, at least two relevant moves and held-out test wording. Simple
  copy/last-mention baselines do not solve the frozen examples.

## Quickstart
```bash
cd SLM_PAPER/circuit_breakdown
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 1) generate + self-validate the datasets (no model needed)
python src/dataset.py --out data/tracking.jsonl --n 200 --task transfer --self-check
python src/dataset.py --out data/tracking_intermediate.jsonl --n 200 --task intermediate --self-check
```

## Historical experiment inventory

The checked entries below describe completed legacy experiments, not validated
universal or safety claims. The active status is in the linked pending plan.
- [x] Scaffold + hardened plan
- [x] Corrected minimal-pair generator w/ self-validation (0/200 recency-solvable)
- [x] Baseline forward pass + residual-stream extraction (fp32, MPS, no swap)
- [x] Attribution patching localization (both tasks) -> `results/heatmap_*.png`
- [x] Verified patching of top-k vs random baseline + non-trivial circuit (100%)
- [x] Layer-handoff sweep -> `results/layer_handoff.png`
- [x] Steering-vector BREAK (93-100%) / STEER (hard, as predicted)
- [x] Cross-model localization: Llama-3.2-3B, Nemotron-Mini-4B, Phi-3.5-mini
      all localize to the same entity-token site (non-trivial circuit = 100%)
- [x] Head-level circuit: ~8-16 late-layer mover heads (72-85%), random-8 ~= 0
- [x] Relative-depth observation: handoff at ~0.62-0.70 in three nearby-depth models
- [x] Legacy global capability stress test; unscored single-position test superseded
- [x] Historical simulated review (not external validation or publication probability)

## Validated workflow

The [shared manifest](data/validated_manifest_v1.json) fixes train/dev/test
examples before evaluation. Each run saves identities, protocol configuration,
per-example results and separate same-sign/negative-edit metrics. Completed
artifacts are resumed only if their configuration matches; do not change a
frozen source or write over old results to force a rerun.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python src/audit_study.py \
  --snapshot results/final_validated_evidence_v1 \
  --output results/my_recheck/audit.json
.venv/bin/python src/report_study.py \
  --snapshot results/final_validated_evidence_v1 \
  --audit results/my_recheck/audit.json --output paper/my_recheck_details.md
```

Choose fresh output names; the audit/reporter refuse overwrites. The audit loads
local tokenizers and small saved edit tensors on CPU, not model weights. Its
passed consistency checks are distinct from supported scientific hypotheses.

The [original queue](run_validated_all.sh) is retained for execution history.
Do not rerun it in this completed checkout: its snapshot/finalizer use exclusive
creation. The [collector](src/collect_study.py) checks missing artifacts;
[audit_study.py](src/audit_study.py) separately rebuilds saved statistics and
checks identities, configurations, budgets, controls and snapshot consistency.
