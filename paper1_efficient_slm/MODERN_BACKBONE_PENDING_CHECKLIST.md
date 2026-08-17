# Modern Backbone Pending Checklist (Nemotron + Phi)

Last updated: 2026-08-15 20:31 (after reduced-budget multi-dataset extension completion)
Plan ID: `modern-multidataset-20260815064755` (superseded `modern-burndown-20260814200900`)

## 1) Model readiness

| Item | Status | Evidence | Done criteria |
|---|---|---|---|
| Nemotron model loads + truncation works | DONE | `experiments/code/results/atis_probe_sweep_usmodern_nemotron.json`, `atis_pruned_depth3_usmodern_nemotron.json`, `atis_pruned_depth12_usmodern_nemotron.json` | Probe + pruned runs produce stable artifacts |
| Phi model loads + truncation works | DONE | `experiments/code/results/atis_probe_sweep_usmodern_phi.json` | At least one probe-sweep artifact exists |

## 2) Experiment matrix (burn down one by one)

| # | Experiment | Status | Artifact(s) | Notes |
|---:|---|---|---|---|
| 1 | **Phi ATIS probe sweep (sanity budget)** | DONE | `experiments/code/results/atis_probe_sweep_usmodern_phi.json` | recommended depth=16 (`8:0.6667,16:0.7917,24:0.7917,32:0.5417`) |
| 2 | Phi ATIS pruned d3 | DONE | `atis_pruned_depth3_usmodern_phi.json` | completed |
| 3 | Phi ATIS pruned d12 | DONE | `atis_pruned_depth12_usmodern_phi.json` | completed via recovery worker |
| 4 | Nemotron ATIS generative baseline | DONE | `atis_generative_usmodern_nemotron.json` | completed via recovery worker |
| 5 | Phi ATIS generative baseline | DONE | `atis_generative_usmodern_phi.json` | completed via recovery worker |
| 6 | Multi-dataset extension (SNIPS/MASSIVE/CLINC150/BANKING77) for Nemotron | DONE (reduced budget) | `snips/massive/clinc150/banking77_{probe_sweep,pruned_depth3,pruned_depth12}_usmodern_nemotron.json` | **NOT publication-grade** — see §6 caveat below |
| 7 | Multi-dataset extension (SNIPS/MASSIVE/CLINC150/BANKING77) for Phi | DONE (reduced budget) | `snips/massive/clinc150/banking77_{probe_sweep,pruned_depth3,pruned_depth12}_usmodern_phi.json` | **NOT publication-grade** — see §6 caveat below |
| 8 | Multi-seed reruns for modern models | PENDING | seed-tagged json files | statistical stability |
| 9 | **NEW:** Rerun MASSIVE/CLINC150/BANKING77 at adequate per-class budget | PENDING | TBD | required before #6/#7 numbers can be used in the paper |

## 3) Evidence integration backlog

| Item | Status | Output path | Done criteria |
|---|---|---|---|
| Aggregate modern results into master table generator | PENDING | `experiments/RESULTS_FINAL_TABLES.md` | modern rows appear with reproducible commands |
| Regenerate figures including modern backbones | PENDING | `experiments/figures/*.pdf` | probe/pareto/latency plots include modern models |
| Manuscript text update for modern-backbone section | PENDING | `PROJECT_REPORT.md`, `paper1_efficient_slm/OUTLINE.md`, `paper1_efficient_slm/latex/paper.tex` | claim language aligned with evidence scope |

## 4) Ground rules for claiming results

- Never compare tiny-budget sanity runs to full-budget GPT-2 headline as if equivalent.
- Any modern-backbone claim must include: dataset, train budget, seeds, and latency protocol.
- If only ATIS sanity is available, label it **preliminary**.

## 6) Reduced-budget multi-dataset results — verified but caveat REQUIRED

32/32 artifacts verified present and valid JSON (5 datasets x 2 models x {probe,d3,d12} + 2 ATIS generative).

**Budget used**: `probe_sweep(max_train=120, epochs=10)`, `pruned(max_train=150, epochs=1, batch=2)`.

**ATIS & SNIPS (7-17 intents): results are directionally usable.**
Depth-12 pruning beats depth-3 on intent accuracy for both models on SNIPS (Nemotron 0.825->0.927, Phi 0.812->0.920) and on slot F1 across the board. Consistent with the original ATIS sanity-pack finding that pruning depth matters.

**MASSIVE (60 intents) / CLINC150 (150 intents) / BANKING77 (77 intents): results are NOT usable as capability measurements.**
`max_train=150` total examples means as few as ~1 example/class (CLINC150) to ~2.5 examples/class (MASSIVE). Both models collapse to near-floor intent accuracy (0.10-0.25) on these three regardless of model or depth -- this reflects data starvation, not backbone/depth differences. Do not report these numbers as findings; do not compare them to ATIS/SNIPS as if on equal footing.

**Additional known-bad metric**: `slot_f1 = 1.0` for every CLINC150/BANKING77 run is vacuous -- both datasets are intent-only (`n_slots=0`), so "tagging zero slots correctly" trivially scores 1.0. Strip or explicitly footnote this before it lands in any table.

**One-time data quality issue (resolved)**: `massive_pruned_depth3` and `banking77_pruned_depth12` (Nemotron) logged wildly inflated `train_time_s` (9hr and 16hr respectively) due to a transient system-wide CPU contention/thrashing incident (load avg spiked to 139 on a 14-core box) that occurred while those two jobs happened to run. Confirmed via `pmset`/`ps`/`vm_stat` root-cause investigation, not a training bug. **Do not use those two `train_time_s` values in any latency/efficiency table.** All other timings in this batch were captured after the contention cleared and look clean.

## 5) Next single step

Current active step: **#9 Rerun MASSIVE/CLINC150/BANKING77 at an adequate per-class training budget** before these datasets can be used in evidence integration (§3 backlog). Multi-seed reruns (#8) and evidence integration remain blocked on this.

