# Paper 1 — Execution Plan

**Title (working):** *Generate Less, Classify More: Probe-Guided Pruning and
Discriminative Heads for Efficient Intent Detection and Slot Filling with Small
Language Models*

**Target:** ICLR 2027 (primary). **Backup:** AISTATS 2027 (2 extra weeks).
Fallback if we slip: EMNLP/NAACL Findings or NeurIPS ENLSP / MLSys workshop.
**Public-benchmark-only** (see `../COMPLIANCE.md`).

### Hard deadlines (AoE)
| Milestone | Date | What must be true |
|---|---|---|
| **ICLR abstract + registration** | **Sep 18, 2026** | Title, authors, abstract registered on OpenReview. Core results (C1, C2, and at least C3 baseline) LOCKED with real public-benchmark numbers. |
| **ICLR full paper** | **Sep 25, 2026** | Complete PDF: all main tables, figures, ablations, reproducibility appendix. |
| AISTATS 2027 (backup) | Oct 8, 2026 | +2 weeks buffer if ICLR slips or we want breadth first. |

Today is ~Aug 7, 2026 → **~6 weeks to abstract, ~7 weeks to full paper.** This is
deadline-driven: we run the FAST PATH first (3-4 datasets, 1 SLM) to lock the
story before Sep 18, then add breadth in the remaining days / camera-ready
(ICLR camera-ready is post-acceptance in ~Jan 2027, so breadth can land later).

---

## 0. The one-sentence claim we must defend
> Converting a generative SLM into a probe-pruned **discriminative single-pass**
> intent+slot model gives a large CPU speedup at **equal-or-better accuracy** and
> **zero structured-output parse failures** — and depth can be chosen *a priori*
> from a cheap linear probe.

Every experiment exists to support or falsify one clause of that sentence.

---

## 1. Scope & contributions (locked)
- **C1 — Probe-guided depth selection.** Linear-probe layer sweep predicts the
  right pruned depth *before* expensive fine-tuning.
- **C2 — Generative→discriminative conversion.** Speed + a *reliability* guarantee
  (parse-failure rate → 0 by construction).
- **C3 — Implicit-slot recovery.** Cheap intent-conditioned rules lift slot recall
  past the span-tagger ceiling (bridge to Paper 2).
- **C4 — Negative-results study.** Why common CPU inference tricks don't pay off
  for shallow closed-set tasks.

Out of scope (state explicitly): open-set/novel-intent detection, free-form
generation, multilingual beyond MASSIVE-en (optional stretch).

---

## 2. Workstreams & milestones

### M0 — Infra & scaffolding (W1)
- Create `experiments/code/` (data loaders, probe, train, eval, latency harness,
  implicit, negresults).
- Pin environment (uv), model revisions (SHAs), dataset versions.
- Define the **named CPU** + thread config for all latency numbers.
- Deliverable: `make repro` runs an end-to-end smoke test on SNIPS.

### M1 — Baselines (W2)  → supports C2
- B1: Generative SLM (small public decoder) fine-tuned to emit `{intent, slots}`.
  Record intent acc, slot F1, exact match, latency P50/P95, **parse-failure rate**.
- B2: Encoder discriminative (JointBERT-style DistilBERT/MiniLM).
- Deliverable: baseline table on all 5 datasets.

### M2 — Probe-guided pruning (W3)  → supports C1
- Freeze base decoder; train linear probes at depths {L/4, L/2, 3L/4, L}.
- Plot probe accuracy vs depth; pick pruned depth within ε of full-depth.
- Fine-tune pruned backbone + linear intent head + CRF BIO slot head.
- Deliverable: probe-sweep figure + "predicted vs actual best depth" validation.

### M3 — Implicit-slot recovery (W4)  → supports C3
- Run implicit-slot detector (value ∉ substring of utterance) per dataset.
- Decompose CRF false negatives: implicit vs genuine miss.
- Add intent-conditioned rules; measure ΔF1, rule precision, latency overhead.
- Deliverable: recovery table + FN decomposition chart.

### M4 — Ablations (W4–W5)  → hardens C1/C2
- CRF vs softmax slot head.
- Slot-loss weight λ ∈ {1, 2, 2.5, 3}.
- Pruned-depth sweep → quality/latency **Pareto curve**.
- Seed variance (≥3 seeds) — report mean ± std everywhere.
- Deliverable: ablation section tables.

### M5 — Negative results (W5)  → supports C4
- On public models: dynamic INT8, rotation-INT8 (QuaRot/Hadamard), ONNX/graph
  export, graph compilation, vendor CPU extensions. Record result + root cause.
- Deliverable: negative-results table + short root-cause writeup.

### M6 — Writing & figures (W6, register abstract Sep 18)
- Fill `paper1_efficient_slm/OUTLINE.md` sections from real numbers.
- Figures: probe sweep, Pareto curve, latency bars, FN decomposition, arch diagram.
- Deliverable: full draft v1.

### M7 — Review & polish (W7, submit Sep 25)
- Self-review vs venue rubric; fix threats-to-validity; reproducibility appendix.
- Optional: hand to code-review / a reviewer agent for a mock rebuttal pass.
- Deliverable: submission-ready PDF + released code repo.

---

## 3. Experiment matrix (what fills the tables)

| Dataset | Intent | Slots | Used for |
|---|---|---|---|
| CLINC150 |  (+OOS) | – | C1, C2 |
| BANKING77 |  (fine-grained) | – | C1, C2 |
| ATIS |  |  | C1, C2, C3 |
| SNIPS |  |  | C1, C2, C3 |
| MASSIVE (en) |  |  | C1, C2, C3 |

Models: 1 small generative SLM (Apache-2.0, e.g. SmolLM2/Qwen-small) as the prune
target; 1 encoder baseline (DistilBERT/MiniLM). Optionally a second SLM size for a
scaling point.

---

## 4. Success / go-no-go gates
- **Gate A (after M1–M2):** does the pruned discriminative model match generative
  intent acc within ~1 pt at ≥5× lower latency on ≥3 datasets? If no → revisit
  depth/heads before writing.
- **Gate B (after M2):** does the probe *predict* the empirically best depth (±1
  layer)? This is C1's whole thesis. If weak → reframe C1 as an empirical study.
- **Gate C (after M3):** does rule recovery beat the measured span-recall ceiling
  with high precision? If no → fold C3 entirely into Paper 2.

---

## 5. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Results too "engineering", not novel enough for main track | Lead with C1 (probe→depth *decision procedure*) + C2 reliability framing, not raw speed |
| Single-model result looks anecdotal | ≥2 model sizes + 5 datasets + seed variance |
| Slot F1 drops too much after pruning | Keep more depth for slots than intent; report the depth/quality Pareto honestly |
| License friction | Prefer Apache-2.0/MIT models; record SHAs and sources |
| Latency numbers not reproducible | Fully specify CPU + threads + warmup; publish harness |
| Reviewer: "constrained decoding already exists" | Position novelty on probe-guided depth + discriminative reliability, cite prior constrained-decoding work |

---

## 6. Deliverables checklist
- [ ] `experiments/code/` runnable end-to-end on public data
- [ ] Baseline table (5 datasets)
- [ ] Probe-sweep figure + depth-prediction validation
- [ ] Main results table (generative vs encoder vs ours)
- [ ] Ablations (CRF, λ, depth Pareto, seeds)
- [ ] Implicit-slot recovery table + FN decomposition
- [ ] Negative-results table
- [ ] Full paper draft (OUTLINE.md filled), **<= 9 pages main text**, official ICLR 2027 LaTeX template
- [ ] Mandatory **AI-Use Statement** section
- [ ] **Fully anonymized** PDF + supplementary + released code (anon repo)
- [ ] Reproducibility appendix + released code (permissive license)
- [ ] All co-authors have active OpenReview profiles (before Sep 18)

## 6b. ICLR 2027 submission compliance
Full checklist in `SUBMISSION_REQUIREMENTS.md`. Non-negotiables that shape how we
work:
- **Double-blind:** no names/affiliations/identifying self-citations anywhere,
  including code and appendix. Release code via an **anonymized repo**, strip PDF
  and Git metadata. The `COMPLIANCE.md` public-only scrub doubles as anonymization.
- **Authorship frozen at Sep 18** — decide the author list NOW; no adds after.
- **OpenReview profiles active before Sep 18** for every co-author.
- **1-submission cap** unless an author has a prior top-venue publication — confirm
  this early; it affects whether Paper 2 can also go to ICLR 2027.
- **9-page main-text limit**; references/appendix/AI-statement excluded. Budget the
  outline to fit 9 pages from the start (push detail to appendix).

---

## 7. ICLR 2027 deadline-driven schedule (backward-planned from Sep 25)

Strategy: **fast path first** to lock the story, breadth second. v1 submission =
4 datasets (SNIPS, ATIS, MASSIVE-en, CLINC150) + 1 SLM prune target + 1 encoder
baseline. BANKING77 and a 2nd model size are stretch / camera-ready.

| Week | Dates | Focus | Milestone(s) | Must-finish gate |
|---|---|---|---|---|
| W1 | Aug 7–13 | Infra + SNIPS smoke test | M0 | `make repro` green on SNIPS |
| W2 | Aug 14–20 | Baselines (generative + encoder) on 4 datasets | M1 | Baseline table done; parse-failure rate captured |
| W3 | Aug 21–27 | Probe sweep + pruned discriminative training | M2 | **Gate A + Gate B** (probe predicts depth) |
| W4 | Aug 28–Sep 3 | Implicit-slot recovery + core ablations | M3, M4 (start) | **Gate C**; CRF-vs-softmax + depth Pareto |
| W5 | Sep 4–10 | Finish ablations + negative-results study | M4, M5 | All main + ablation tables frozen |
| W6 | Sep 11–17 | Write draft + figures; **register abstract** | M6 | Abstract-ready numbers locked |
| **Sep 18** | | **ICLR abstract + registration deadline (AoE)** | | Registered on OpenReview |
| W7 | Sep 18–24 | Polish, validity threats, repro appendix, self-review | M7 | Submission-ready PDF |
| **Sep 25** | | **ICLR full paper deadline (AoE)** | | Submitted |
| Buffer | Sep 26–Oct 8 | Add breadth / fixes | — | **AISTATS 2027 fallback (Oct 8)** if needed |

### Critical path
M0 → M1 → M2 are the bottleneck: the whole story is dead without a working
pruned-discriminative model that clears Gate A/B. Protect W1–W3 ruthlessly; cut
scope (drop a dataset, defer 2nd model) rather than slip these.

### Parallel admin track (do NOT let this slip to the last week)
- **W1 (Aug 7–13):** finalize author list; confirm the 1-submission-cap status
  (does any author have a prior top-venue paper?); every co-author creates/activates
  an OpenReview profile; set up the anonymized code repo.
- **W6 (by Sep 17):** grab the official ICLR 2027 LaTeX template; draft the AI-Use
  Statement; run the first anonymity + public-only scrub on the abstract.
- **W7 (by Sep 24):** second full scrub (PDF + supplementary + git metadata),
  verify <= 9 pages, verify all OpenReview profiles active.

### De-risking levers if we fall behind
- Drop to **3 datasets** (SNIPS, MASSIVE-en, CLINC150); ATIS is smallest-value add.
- **1 seed** for non-headline ablations, 3 seeds only for the main result.
- Defer the full negative-results study (C4) to an appendix / camera-ready; keep a
  1-paragraph summary in the main text.
- If Gate B (probe predicts depth) is weak by Sep 3, reframe C1 as an *empirical
  characterization* rather than a predictive procedure — still publishable, less
  risky, no timeline hit.

### What can safely land in camera-ready (post-acceptance, ~Jan 2027)
BANKING77, a 2nd SLM size (scaling point), multilingual MASSIVE, and the expanded
negative-results appendix. None are needed to defend the core claim at submission.
