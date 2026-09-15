# Abstract (draft 1) — written before the paper, as a framing test

**Title (working):** Effective, Specific, and Non-Selective: Activation
Interventions on Sequentially Computed State

---

## Abstract (187 words)

Interpretability research increasingly validates activation interventions by
showing that they produce an intended counterfactual answer. RAVEL established
that this is insufficient for static entity attributes: a faithful
intervention must also leave unrelated attributes intact, and MIB reports that
full-vector interventions fail exactly this test. We ask whether the same
distinction holds when the target is not a stored attribute but a state
computed over several reasoning steps. Using a transfer task with a known
causal program (`holder_{t+1} = recipient_t`), we construct an intervention
that passes a progressively strengthened battery of controls: it changes the
answer, it is specific to the donor whose state was transplanted
(DS = +0.718, with non-injected donors at 0.000), it is inert under a
norm-matched random direction at every magnitude, and it is graded in
magnitude. We then query the same intervened forward pass about quantities the
causal program says must not change. Selectivity is 0/240 in two model
families (95% CI [0.0, 1.6%]). Phi-3.5-mini answers "which object moved?" with
100% accuracy unintervened, and with a person's name afterwards. Behavioral
control of a reasoning answer does not imply control of the reasoning state.

---

## Self-assessment against the framing test

**Does it acknowledge RAVEL explicitly?** Yes — second sentence, by name,
before any contribution claim. MIB in the third.

**Is the contribution precise?** The extension is stated as a question
(static attribute → sequentially computed state), not as a new method or a
new diagnostic.

**Is anything inflated?** Checked against the frozen constraints:
- no "competence amplifies the failure"
- no novelty claim for the cross-question panel
- no implication that DAS/MIB patch arbitrary answer states
- no claim that selective reasoning interventions are impossible
- the object-identity example uses Phi's real numbers (100% → person's name)

**What it deliberately omits:** the non-uniqueness falsification, the
geometry results, and the position sweep. None are needed for the claim.

**Known soft spots, to state plainly in the paper:**
1. Llama's completeness is weak (44.2%) and on a *different probe set* than
   Phi's, because Llama's `current_holder` was excluded at 46.7% competence.
   The abstract therefore claims two-family replication only for
   **selectivity**, which is accurate — both models are 0/240.
2. One task family. The limitations section must say the result is a
   counterexample in a controlled setting, not a general law.
3. `transfer_count` was excluded in both models (18–21% competence), so
   selectivity rests on two probes per model.

**Verdict:** the framing holds without inflation. The one-sentence contribution
is *"RAVEL's effectiveness/selectivity distinction survives the move from
static attributes to computed state, and a heavily-controlled intervention
still fails it."* That is honest, citable, and small.

---

## Alternative framing, if the above reads too negative

Lead with the diagnostic value rather than the failure:

> We report a controlled setting in which every standard credential for an
> activation intervention — effectiveness, donor specificity, norm control,
> and dose-response — is satisfied by an intervention with zero selectivity,
> and show that a two-probe consequence check detects this at negligible cost.

Same evidence, framed as "cheap check catches a case the usual battery
misses." Decide after Section 5 is drafted; do not decide now.
