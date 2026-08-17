# SLM Paper — Project Status

_Last updated: 2026-08-16 09:50 (after reduced-budget modern-backbone multi-dataset matrix completed, verified, and checklist synced)_

## TL;DR

**Not submission-ready.** Core GPT-2-scale research (the paper's main thesis) is reported as complete in `PROJECT_REPORT.md`. The **modern-backbone (Nemotron-mini-4B / Phi-3.5-mini) validation work is partially done**: ATIS + SNIPS results are usable, but MASSIVE/CLINC150/BANKING77 results from the just-finished run are **not usable** due to a data-starved training budget. Submission logistics (TeX compile, anonymity scrub, OpenReview readiness) haven't been started.

---

## 1) What's DONE

### Core paper (GPT-2-scale experiments)
- `PROJECT_REPORT.md` states the core research is complete; remaining work there is submission mechanics, not experiments.

### Modern-backbone sanity pack (ATIS) — Nemotron + Phi
- Probe sweeps, pruned d3/d12, and generative baselines all complete for both models on ATIS.
- Key numbers: Nemotron recommended depth=32, Phi recommended depth=16; pruned discriminative >> generative in both accuracy and latency reliability.

### Modern-backbone multi-dataset matrix (reduced budget) — Nemotron + Phi
- **32/32 artifacts** generated and verified as valid JSON: 5 datasets (ATIS, SNIPS, MASSIVE, CLINC150, BANKING77) × 2 models × {probe_sweep, pruned_d3, pruned_d12}, plus 2 ATIS generative baselines.
- Runner: `experiments/code/scripts/run_modern_multidataset_reduced.sh` (resumable, run-if-missing, stop-on-first-failure). Fully finished (`MODERN_MULTIDATASET_REDUCED_DONE` in `/tmp/modern_multidataset_reduced.log`).
- **Usable findings**: on ATIS (17 intents) and SNIPS (7 intents), depth-12 pruning consistently beats depth-3 on intent accuracy and slot F1 for both models. Consistent with the earlier ATIS-only sanity pack.

### Infra / process fixes made along the way
- Recovered from an earlier deadlocked chained-waiter pipeline (killed stale PIDs, rebuilt as a sequential resumable runner).
- Diagnosed and fixed a real system-level issue: the Mac was sleeping mid-run (no `caffeinate`), stalling training for hours; `caffeinate -w <worker-pid>` was attached and resolved it.
- Diagnosed a second real issue: a transient CPU-contention/thrashing incident (load avg spiked to 139 on a 14-core box, confirmed via `pmset`/`ps`/`vm_stat`) inflated two `train_time_s` readings (Nemotron `massive_pruned_depth3` ≈ 9hr, `banking77_pruned_depth12` ≈ 16hr) — root-caused as system noise, not a training bug. Cleaned up a leftover zombie waiter process from the old deadlocked chain in the process.

---

## 2) What's PENDING

### A) Modern-backbone research (blocks paper claims)
| # | Item | Status | Why it matters |
|---|---|---|---|
| 1 | **Rerun MASSIVE/CLINC150/BANKING77 at an adequate per-class training budget** | PENDING (new, highest priority) | Current reduced budget (`max_train=150` total) gives ~1–2.5 examples/class for 60/150/77-intent datasets. Both models collapse to near-floor accuracy (0.10–0.25) regardless of model or depth — this is data starvation, not a real finding. Not usable in the paper as-is. |
| 2 | Strip/footnote vacuous `slot_f1 = 1.0` metric | PENDING | CLINC150 and BANKING77 are intent-only (`n_slots=0`); slot F1 trivially reports 1.0. Must not appear in a results table looking like a real number. |
| 3 | Exclude two contaminated `train_time_s` values from any latency table | PENDING | `massive_pruned_depth3` and `banking77_pruned_depth12` (Nemotron) timings are inflated by the system-contention incident above; not representative. |
| 4 | Multi-seed reruns for modern models | PENDING | Needed for statistical stability before any modern-backbone number is claimed as a finding. |
| 5 | Aggregate modern results into `experiments/RESULTS_FINAL_TABLES.md` | PENDING | Blocked on #1–#4 above. |
| 6 | Regenerate figures with modern backbones | PENDING | Blocked on #1–#4 above. |
| 7 | Manuscript text update (`PROJECT_REPORT.md`, `OUTLINE.md`, `paper1_efficient_slm/latex/paper.tex`) | PENDING | Currently zero mentions of Nemotron/Phi/"modern backbone" in `PROJECT_REPORT.md` — the manuscript hasn't been touched with this evidence yet. |

Full detail lives in `paper1_efficient_slm/MODERN_BACKBONE_PENDING_CHECKLIST.md`.

### B) Submission logistics (from `paper1_efficient_slm/PENDING_AUDIT_PROJECT_REPORT_OUTLINE.md`)
1. Install TeX toolchain, compile submission PDF, verify main text ≤ 9 pages.
2. Swap fallback preamble to official `iclr2027_conference.sty` + bibliography stack.
3. Anonymize release repo (paper + supplement + code + git identities + paths).
4. Finalize author list and OpenReview profile readiness.
5. (Camera-ready, optional) second backbone size + rotation-based INT8/fused kernels; slot-loss lambda sweep (currently fixed at 2.0).
6. Consistency pass: `OUTLINE.md` vs `latex/paper.tex` claim-by-claim, so no claim drift.

---

## 3) Ground rules (don't violate these when writing up results)
- Never compare tiny-budget sanity runs to the full-budget GPT-2 headline as if equivalent.
- Any modern-backbone claim must state: dataset, train budget, seed count, and latency protocol.
- If only ATIS/SNIPS coverage is available for modern backbones, label it **preliminary**, not final.

## 4) Suggested next single action
Size a per-class training budget for MASSIVE/CLINC150/BANKING77 (rule of thumb: `max_train ≥ 20×num_intents`), estimate compute/wall-clock cost, and get a go/no-go before burning more CPU time on numbers that can't be used.
