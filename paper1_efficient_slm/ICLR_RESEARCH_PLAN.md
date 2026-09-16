# ICLR 2027 Research-First Plan (Paper 1)

## 1) What a strong ICLR paper needs (practical reviewer bar)

A good ICLR paper is usually judged on five axes:

1. **Novelty / conceptual contribution**
   - Not just faster engineering; needs a reusable idea or decision procedure.
2. **Technical quality**
   - Methods are internally consistent; equations/claims match implementation.
3. **Empirical rigor**
   - Fair baselines, proper ablations, variance/seeds, no cherry-picking.
4. **Clarity + positioning**
   - Clear claim, explicit scope, honest limitations, proper related work framing.
5. **Reproducibility + ethics/compliance**
   - Enough details to rerun; anonymous and policy-compliant submission package.

For this project, that means reviewers should walk away with:
- A credible **decision rule** (probe-guided depth choice),
- A credible **reliability result** (parse-failure elimination by construction),
- Honest tradeoff accounting vs encoder baselines.

---

## 2) Non-negotiable ICLR 2027 constraints (gates, not suggestions)

From `SUBMISSION_REQUIREMENTS.md`:

- **Abstract/registration deadline:** Sep 18, 2026 (AoE)
- **Full paper deadline:** Sep 25, 2026 (AoE)
- **Main text <= 9 pages**
- **Official ICLR template only**
- **Mandatory AI-Use statement**
- **Double-blind anonymity everywhere** (PDF + supplement + code)
- **Anonymized code release** (not personal/company public repo)
- **OpenReview profiles active for all authors before registration**

Interpretation: if anonymity/page-limit/admin readiness is weak, strong experiments still get desk-risked.

---

## 3) Current evidence snapshot (what is already strong)

Based on `PROJECT_REPORT.md` + `OUTLINE.md`:

- C1/C2/C3/C4 structure exists and is coherent.
- Real tables/figures exist; claims are mostly numerically grounded.
- Positive signal: team already corrected confounds (seed variance, CRF-loss bug, fairness notes).
- Remaining risk is now less “do we have results?” and more “is the contribution framed as ICLR-level science vs engineering optimization?”

---

## 4) Research-first execution plan (ordered by reviewer impact)

## Phase A — Claim hardening (highest priority)
**Goal:** Make the central claim falsifiable and reviewer-proof.

1. Freeze the single sentence claim and map each clause to exact evidence table/figure.
2. Add explicit null hypotheses for each contribution (C1–C4).
3. Define failure conditions up front (e.g., C1 fails if probe depth does not predict within ±1 layer).

**Exit gate A:** Every headline claim has one primary table and one sanity/robustness check.

## Phase B — Evidence quality upgrades
**Goal:** Remove common rejection hooks around rigor.

1. Ensure matched-budget comparisons are the default in main claims.
2. Keep seed variance (mean ± std) wherever claims are comparative.
3. Ensure all ablations are mechanistic (not laundry-list).
4. Explicitly separate “statistically indistinguishable” from “better”.

**Exit gate B:** No claim depends on single-seed or mismatched-budget evidence.

## Phase C — Novelty positioning against prior art
**Goal:** Make contribution look like method/science, not benchmarking exercise.

1. Related-work rewrite around three contrasts:
   - constrained decoding methods,
   - compression/pruning methods,
   - probe-based representation studies.
2. State exact novelty delta:
   - probe used as **pre-finetune decision mechanism**,
   - discriminative conversion as **reliability guarantee** (not only speed).
3. Add “what this paper does NOT claim” paragraph (anti-overclaim shield).

**Exit gate C:** Novelty statement survives adversarial question: “Why is this not just pruning + heads?”

## Phase D — ICLR package compliance
**Goal:** eliminate procedural rejection risk.

1. Page-budget pass to guarantee <=9 pages main text.
2. AI-use statement final and consistent with workflow.
3. Full anonymity scrub twice (content + metadata + code history).
4. OpenReview author/admin checklist complete before Sep 18.

**Exit gate D:** submission artifact passes dry-run checklist with zero blockers.

---

## 5) Concrete weekly plan to submission

- **Week 1 (now):** Phase A + B claim/evidence hardening
- **Week 2:** Phase C novelty framing + related work surgery
- **Week 3:** Main-paper polish + figure narrative coherence
- **Week 4 (pre-Sep 18):** registration package + abstract lock + anonymity scrub #1
- **Week 5 (Sep 18–25):** final PDF freeze, compliance scrub #2, submission

---

## 6) Go/No-Go criteria before final upload

Proceed only if all are true:

- [ ] Core claim mapped to concrete evidence with variance-aware stats
- [ ] Matched-budget fairness reflected in headline comparisons
- [ ] Novelty statement is explicit and defensible in 3 sentences
- [ ] Main text <=9 pages in official template
- [ ] AI-use statement + reproducibility section present
- [ ] Full double-blind + anonymized code path validated

If any box fails by Sep 22, prioritize compliance + claim integrity over adding new experiments.

---

## 7) Reviewer-facing one-liner (final framing)

This paper's value is not “we found yet another faster model,” but: 
**we provide a reproducible pre-finetune depth-selection procedure and a reliability-preserving decoder-to-discriminative conversion with explicit tradeoff boundaries.**
