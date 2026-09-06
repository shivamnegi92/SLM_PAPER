# Reference Papers — Notes & Implications

Three papers in `references/`. Extracted text in `references/text/`.

---

## 1. `2304.15004v2` — Schaeffer, Miranda & Koyejo, *Are Emergent Abilities of LLMs a Mirage?* (NeurIPS 2023)

**Core claim.** Apparent "emergent abilities" are largely an artifact of the
*researcher's choice of metric*, not a phase change in the model. Nonlinear
(Accuracy) or discontinuous (Multiple Choice Grade) metrics manufacture sharp
jumps; linear (Token Edit Distance) or continuous (Brier Score) metrics on the
*same fixed model outputs* reveal smooth, predictable improvement.

Three factors manufacture emergence:
1. A metric that nonlinearly/discontinuously deforms per-token error rate.
2. Insufficient **resolution** (resolution ≈ 1 / test-set size).
3. Insufficient sampling of the large-model regime.

Key quantitative shape: with per-token error `p`, sequence accuracy over length
`L` falls **geometrically** (`p^L`) while Token Edit Distance falls only
**quasi-linearly** — hence accuracy looks like a cliff, edit distance looks like a ramp.

### Why this matters for *our* paper — directly load-bearing

This paper is the **theoretical justification for our entire Block A result**,
and we should cite it as such rather than as background.

- **Our STEER metric is exactly a discontinuous metric.** `STEER = 1[argmax == clean]`
  is top-1 accuracy — the *most* discontinuous choice available. Our finding that
  linear interventions produce **0% STEER but positive logit lift** is a textbook
  instance of Schaeffer's mechanism: the underlying quantity (logit margin) is
  moving smoothly and predictably, but our chosen metric thresholds it to zero.
  Reframe: *"the softmax bottleneck we document is the intervention-side dual of
  the metric-choice effect Schaeffer et al. identify on the evaluation side."*
- **We already did the right thing by reporting `delta_logit_clean` alongside STEER.**
  That is our continuous metric. We should make this explicit and elevate it:
  report a **continuous decision metric** (margin, or `p(clean) − p(corrupt)`)
  next to the discontinuous STEER in every table. This preempts the reviewer
  objection "your method just fails" — no, the linear methods move the continuous
  quantity, they just cannot cross the discontinuity.
- **Resolution warning applies to us hard.** Schaeffer: resolution = 1/test-set size.
  Our n_test = 9 per seed gives resolution 11.1%. Any effect smaller than that is
  invisible *by construction*. This is a principled, citable justification for the
  seed-pooling we already did (n=27, resolution 3.7%) and for pushing to n≥120.
- **Actionable metric to add:** a **Brier-style continuous score** on the
  clean-vs-corrupt binary, i.e. `(p(clean)/(p(clean)+p(corrupt)) − 1)^2`. This
  gives a strictly-proper, continuous companion to STEER and would let us show a
  smooth frontier where the discontinuous metric shows a cliff.

---

## 2. `2501.16496v1` — Sharkey, Chughtai et al., *Open Problems in Mechanistic Interpretability* (Apollo / Anthropic / DeepMind / MIT, 2025)

82-page community agenda. The relevant parts:

### Our research question is listed verbatim as an open problem
Section A.2.2, "Using mechanistic interpretability for better control of AI system behavior":

> **1. Can we improve steering methods through interpretability?**
> **a. How can we make activation steering more precise and reduce its side effects?**
> **b. Can we develop methods to steer entire mechanisms rather than just single features?**

That is *exactly* the STEER↔BREAK frontier we are measuring, posed as an open
problem by the field's leading labs. **This is the single highest-value citation
in the whole reference set** — it converts our contribution from "we tried some
steering variants" into "we make measurable progress on a named open problem."
It belongs in the abstract and intro.

Also relevant, A.2.2 item 2c:
> "Can mechanistic interpretability help us determine which classes of model edit
> are even possible, without damaging generalization in undesirable ways?"

Our BREAK / control-drop axis is a direct empirical answer to this.

### Warnings we must respect

- **Subspace / DAS interpretability illusion.** Makelov et al. (2023) argue
  subspace activation patching via distributed alignment search *can produce
  illusory mechanisms*; Wu et al. (2024c) contest this. Implication for our
  Method 3 (subspace-projected edits): we **cannot** claim the subspace is "the"
  tracking mechanism on the strength of intervention success alone. Safe framing:
  the subspace is a **regularizer with causal provenance** — it constrains edits
  to directions that demonstrably carry the clean/corrupt distinction, and we
  justify it by the *frontier improvement it buys*, not by ontological claims.
  We should cite both sides of that dispute.
- **The "Hydra effect."** Networks compensate for ablated components, so an
  intervention's measured effect conflates the true causal path with downstream
  compensation. Listed as an open measurement problem. Relevant to our BREAK
  metric: some breakage may be compensation, not direct damage. Worth one
  sentence in limitations.
- **Circuit faithfulness is low / metric-dependent** (Miller et al. 2024):
  faithfulness scores depend on the patching implementation used. Reinforces that
  we should report our intervention implementation in full detail (we do).
- **Attribution patching is only first-order.** Our `heads.py` uses AtP, then
  verifies top-k with real patching against a random-k baseline — which is exactly
  the recommended mitigation. Good; say so explicitly.

---

## 3. `2602.11180v1` — Naseem, *Mechanistic Interpretability for LLM Alignment* (Macquarie)

Survey connecting circuit discovery, feature visualization, activation steering
and causal intervention to alignment practice (RLHF, constitutional AI, scalable
oversight). Identifies superposition, polysemanticity, and interpreting emergent
behavior at scale as key blockers.

**Use for us:** framing and related-work scaffolding. It supplies the
"interpretability-driven control" narrative and the argument that black-box
behavioral alignment gives "limited guarantees about generalization to novel
situations." Our controllable-override result is a small, concrete instance of
the interpretability-driven control it calls for. Lower citation priority than
the other two, but useful for intro/related-work breadth.

---

## Consolidated actions for the paper

| # | Action | Source |
|---|---|---|
| 1 | Cite Open Problems A.2.2.1a in the **abstract/intro** — we address a named open problem | 2501.16496 |
| 2 | Reframe Block A's 0% STEER as the **intervention-side dual of the metric-choice effect** | 2304.15004 |
| 3 | Report a **continuous decision metric** (margin / Brier-style) beside STEER everywhere | 2304.15004 |
| 4 | Justify seed-pooling + larger n via **resolution = 1/test-set size** | 2304.15004 |
| 5 | Frame the subspace as a **causally-provenanced regularizer**, not "the mechanism"; cite the Makelov/Wu dispute | 2501.16496 |
| 6 | Add **Hydra-effect** caveat to the BREAK limitation paragraph | 2501.16496 |
| 7 | Note that our AtP→real-patching→random-baseline pipeline is the recommended mitigation for first-order attribution error | 2501.16496 |
| 8 | Use the alignment survey for related-work breadth on interpretability-driven control | 2602.11180 |
