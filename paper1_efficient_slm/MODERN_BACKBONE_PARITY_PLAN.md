# Plan: make Nemotron and Phi comparable to GPT-2

Goal: replace "GPT-2 only" with a side-by-side table across three backbones,
so a reviewer cannot dismiss the paper as a result about one 124M model from
2019.

---

## 1. The actual gap

Not "modern backbones are missing" — they exist. The problem is that they were
run at a **53x smaller training budget**, so the numbers cannot legally sit in
the same table.

| run | max_train | epochs | batch | ATIS intent acc | ATIS slot F1 |
|---|---:|---:|---:|---:|---:|
| **GPT-2** d3 | **4274** (all) | 3 | 16 | **0.980** | **0.910** |
| Nemotron d3 | 80 | 1 | 2 | 0.830 | 0.677 |
| Phi d3 | 80 | 1 | 2 | 0.770 | 0.466 |
| **GPT-2** d12 | **4274** | 3 | 16 | **0.974** | **0.919** |
| Nemotron d12 | 80 | 1 | 2 | 0.741 | 0.748 |
| Phi d12 | 80 | 1 | 2 | 0.863 | 0.577 |
| **GPT-2** SNIPS d3 | **5000** | 3 | 16 | **0.980** | **0.822** |
| Nemotron SNIPS d3 | 150 | 1 | 2 | 0.825 | 0.450 |

The modern backbones look worse. **We have no evidence they are worse** —
they were starved. Publishing this comparison as-is would be misleading, and
the repo's own ground rules already forbid it:

> "Never compare tiny-budget sanity runs to full-budget GPT-2 headline as if
> equivalent." — `MODERN_BACKBONE_PENDING_CHECKLIST.md` §4

Worse for the high-cardinality sets: 150 training examples across CLINC150's
150 intents is ~1 example per class. The checklist already marks MASSIVE /
CLINC150 / BANKING77 as **"NOT usable as capability measurements."**

## 2. Why it was never fixed

CPU. From `run_modern_multidataset_reduced.sh`:

> "full 5000x3 budget would take months"

4B-class models at batch 16 x 3 epochs on an M4 Pro is not viable. This was a
hardware constraint, not an oversight.

## 3. What already exists in our favour

`kaggle_gpu_pack/` is a complete, offline, GPU-ready runner built for exactly
this job:

- `notebooks/modern_backbone_e2e_kaggle.ipynb` — end-to-end, resumable
- All five datasets bundled locally (no internet needed for data)
- Cardinality-scaled budget already designed:
  - `pruned_max_train = max(400, 20 x num_intents)`
  - `probe_max_train  = min(3000, max(600, 40 x num_intents))`
  - `PRUNED_EPOCHS=3`, `BATCH_SIZE=16` — **matches the GPT-2 protocol**
- Emits `kaggle_run_summary.csv`

So the plan is mostly *execute and integrate*, not *build*.

---

## 4. Plan

### Phase 0 — Decide the comparison axis  *(30 min, needs your call)*

Two defensible designs. **Pick one before running anything.**

**(a) Matched budget — recommended.** Every backbone gets the same
`max_train`, epochs, batch, seeds. Cleanest claim: "under identical training
budget, backbone X wins." Requires re-running GPT-2 at the capped budget too,
which is cheap (124M).

**(b) Best-effort per backbone.** Each model gets its full budget. Answers
"what is each capable of," but the GPT-2 advantage stays confounded with data
volume and a reviewer will say so.

I recommend **(a) matched budget**, with GPT-2's existing full-data rows kept
as a separate clearly-labelled "GPT-2 at full budget" reference row.

Note there is already a `run_matched_budget_3k.sh` and
`atis_*_budget3k.json` artifacts — a matched-budget protocol was started
before. Reuse that convention rather than inventing a new one.

### Phase 1 — Verify the GPU path on one cell  *(1-2 h)*

Do **not** launch the full matrix blind.

1. Upload `kaggle_gpu_pack/` as a Kaggle Dataset; attach Nemotron + Phi
   (or let them download from HF).
2. Run **one** cell only: Nemotron / ATIS / depth-3 at the real budget.
3. Check: does it finish, what is the wall-clock, does the artifact schema
   match the existing `results/*.json` so `aggregate_results.py` can read it?

Gate: if one cell takes more than ~20 min, the 30-cell matrix will not fit in
Kaggle's session limit and must be split across sessions.

### Phase 2 — Run the matrix  *(GPU hours, mostly unattended)*

3 backbones x 5 datasets x {probe sweep, pruned d3, pruned d12}.

GPT-2 rows at matched budget can run locally on CPU in parallel — it is 124M.

Resumable via `run_if_missing`, so a session timeout costs only the cell in
flight.

### Phase 3 — Latency, measured on ONE device  *(2-3 h, LOCAL, do not skip)*

**This is the part the Kaggle pack does not solve, and it is the paper's
headline claim.**

The abstract claims ~5 ms P50 vs ~359 ms autoregressive, a ~70x speedup. A
latency number measured on a Kaggle T4/P100 is not comparable to one measured
on an M4 Pro CPU. Mixing them would break the central result.

Therefore:
- Train on GPU (Phase 2), then **export heads and measure latency locally on
  CPU for every backbone**, identical protocol, same machine, same batch size,
  same warmup, P50 over the same number of trials.
- Report the device explicitly in the caption.
- Expect the speedup ratio to *shrink* for 4B models: a pruned 4B backbone is
  still far heavier than a pruned 124M one. **If the ~70x claim only holds for
  GPT-2, the abstract must say so.** That is a real possible outcome of this
  plan and we should not pre-commit to the current number surviving.

### Phase 4 — Seeds  *(GPU, unattended)*

Checklist item 8 is still open. Three seeds per configuration for the headline
table, so the cross-backbone comparison carries variance rather than single
runs. Without this a reviewer can attribute any gap to seed noise.

### Phase 5 — Integrate  *(3-4 h, local)*

Checklist items currently PENDING:
- `aggregate_results.py` -> modern rows in `RESULTS_FINAL_TABLES.md`
- `make_figures.py` -> probe / pareto / latency plots include all backbones
- `latex/paper.tex` — abstract, results, claim language
- A verification script in the spirit of `verify_paper_numbers.py`, so every
  number in the tex is checked against the artifact JSONs

---

## 5. Cost

| phase | where | time |
|---|---|---|
| 0 decide axis | — | 30 min |
| 1 verify one cell | Kaggle GPU | 1-2 h |
| 2 full matrix | Kaggle GPU | unattended, likely multi-session |
| 3 latency | **local CPU** | 2-3 h |
| 4 seeds | Kaggle GPU | unattended |
| 5 integrate | local | 3-4 h |

Roughly **1-2 days wall-clock**, of which ~8 h is hands-on. Not months —
because the GPU pack already exists.

## 6. Risks, stated honestly

1. **The result may not favour the modern backbones.** At matched budget a 4B
   model may still lose to GPT-2 on ATIS, which is a small closed-set task
   where 124M is plenty. That is a legitimate finding — "scaling the backbone
   does not help closed-set intent detection" — but it is a different paper
   from the one currently written, and you should decide now whether you are
   willing to publish it.
2. **The ~70x speedup likely shrinks** for 4B backbones. See Phase 3.
3. **CLINC150 at 3000 train examples / 150 intents = 20 per class.** Better
   than 1, still thin. Label it accordingly.
4. **Kaggle session limits** may force splitting Phase 2; the runner is
   resumable, so this costs wall-clock rather than work.
5. **Artifact schema drift** between the Kaggle pack and
   `experiments/code/results/` would break aggregation. Phase 1 checks this
   before it can waste a full matrix.

## 7. What I need from you

1. **Comparison axis** — matched budget (recommended) or best-effort?
2. **Kaggle access** — do you have a working Kaggle account with GPU quota?
   If not, the alternative is a cloud GPU rental, or accepting ATIS+SNIPS only
   at reduced scope.
3. **Willingness to publish an unfavourable result** — see risk 1. This
   determines whether we run the experiment or reframe the paper.
