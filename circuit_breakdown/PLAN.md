# PLAN — Circuit Breakdown (hardened)

> **Historical roadmap, retained for provenance.** The active execution plan is
> [PENDING_PLAN.md](PENDING_PLAN.md), with the frozen comparison in
> [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md). Actual models are Llama-3.2-3B,
> Phi-3.5-mini and Nemotron-Mini-4B; Gemma references below describe an unexecuted
> proposal. ICLR is the working venue target, not a confirmed submission date.
> Original "repair", no-degradation and universal-depth claims are not results.

Hardened rewrite of the "Circuit Breakdown" blueprint. Every change below exists
to survive a hostile ICLR/ICML reviewer. Read `../COMPLIANCE.md` first: this is
public-model / public-benchmark only.

---

## 0. Contribution & positioning (decide this before running anything)

**Claim (falsifiable):** In ≤3B SLMs, multi-step state-tracking is implemented by
a *small, localizable* set of attention components; patching that subset recovers
the majority of the clean-vs-corrupt logit difference; and a steering vector
derived from those sites can causally **break** (and partially **repair**)
tracking behavior with degradation on general benchmarks that stays within noise.

**What makes it novel (the differentiators):**
1. **Cross-architecture comparison** — Llama-3.2-3B vs Gemma-2-2B. Which parts of
   the tracking circuit are universal vs. architecture-specific? (cheapest real
   novelty; two models ≈ 1.3× the work, not 2×.)
2. **Consumer-hardware auditability, honestly measured** — full causal audit in
   **fp32** on an Apple-silicon laptop, with attribution patching as the enabling
   trick. Wall-clock + peak-RAM reported as first-class results.
3. **Break *and* repair** with the *same* localized direction, plus a
   capability-regression control proving we didn't just lobotomize the model.

**Prior work we MUST cite & differentiate from** (localization for tracking is a
crowded field — do not pretend otherwise):
- IOI circuit — Wang et al. 2022 (the patching template we follow).
- Causal tracing / ROME — Meng et al. 2022.
- Entity Tracking in Language Models — Kim & Schuster, ACL 2024 (closest task).
- How do LMs Bind Entities in Context? — Feng & Steinhardt 2023 (variable binding).
- Function Vectors — Todd et al. 2024 (steering, Phase 3).
- Attribution Patching / AtP* — Nanda; Syed et al. (our speed story).
- Self-repair / backup heads / Hydra effect — McGrath et al.; Wang et al.
  (our main threat to validity — see §4).

**Venue:** pick ONE. Parent project targets ICLR/EMNLP. Given scope (1 task
family, 2 small models) and a ~15-week clock landing in **late Dec 2026**:
- **Floor:** interp workshop (BlackboxNLP, ATTRIB) — realistic accept.
- **Stretch:** ICLR 2027 main track — needs the cross-arch story to land cleanly.
- Kill the "ICML vs ICLR" ambiguity in the blueprint. This doc assumes ICLR.

---

## 1. Metrics — replace the gameable "0.05%" (see METRICS.md for math)

The "modify <0.05% of activation vectors" line is a **vanity metric** (you pick
the denominator, so it's meaningless). It may appear ONLY as a secondary framing
line. Primary metrics are:

- **Normalized logit-difference recovered** (faithfulness) — % of clean-vs-corrupt
  logit gap restored by patching a component subset. Headline number.
- **Completeness & minimality** (Wang et al. definitions) — the circuit is
  sufficient AND every member matters.
- **KL divergence to clean** over full vocab (not just target token).
- **Circuit size** — # attention heads / components, with error bars across the
  prompt distribution. THIS is the honest "small number" ("4 heads, 2 layers").
- **Random-patching baseline** — same # of *random* components; must recover ~0.

---

## 2. Phase 1 — Environment & baseline (Days 1–7)

Goal: a scriptable interp pipeline that is numerically trustworthy on this Mac.

- `uv venv` (do **not** use `~/.code-puppy-venv`); install from Walmart index.
- TransformerLens + torch. **Device policy:**
  - Localization/patching math runs in **float32** (fp16 on MPS is a NaN factory
    for logit-diff attribution; precision > speed here).
  - **CPU-float32 fallback path from day one.** TransformerLens hooks on MPS
    have CPU-fallback ops and flakiness; a 2–3B model doing a few-token forward
    pass on CPU is tolerable. (We got ambushed by MPS before — see kennel.)
  - Provide a `--device {auto,mps,cpu}` flag; default `auto` → try mps, verify no
    NaNs on a canary, else fall back to cpu.
- Models: **Llama-3.2-3B first** (already downloaded at `../llama-3.2-3b`,
  fp16 safetensors; load as fp32 for math). Gemma-2-2B second.
  -  **Gemma-2 gotcha:** logit *and* attention **soft-capping** + GQA. Verify
    the installed TransformerLens version applies soft-caps, or direct logit
    attribution numbers are wrong. Get Llama green end-to-end BEFORE porting.
- Baseline sanity script: load model, run one prompt, print residual-stream
  tensor shapes per layer, print peak RAM, assert **no swap** (log RSS).

**Gate G1 (end of Wk 1):** shapes correct, no NaNs in fp32 residual stream on the
chosen device, peak RAM < unified RAM (no swap). If MPS NaNs → lock to CPU fp32.

---

## 3. Phase 2 — Dataset + clean activation patching (Weeks 2–5)

### 3.1 Dataset (FIXED — this was the fatal bug)
Original pair had the **same answer both sides** → zero patching signal. Two
corrected task families (`src/dataset.py`, self-validating):

- **`intermediate`** — query a non-final state so corruption changes the target:
  - clean:  `...London to Paris to Berlin. The location before Berlin was ___` → **Paris**
  - corrupt: `...London to Tokyo to Berlin. The location before Berlin was ___` → **Tokyo**
- **`transfer`** — object-passing chains requiring true step composition (not
  recency): `Alice has the key. Alice gives it to Bob. Bob gives it to Carol.
  Who has the key? ___` → **Carol**; corrupt one hop → different final holder.

**Hard requirements enforced by the generator:**
- `clean_target != corrupt_target` (else no signal). Self-check asserts this.
- **Token-position alignment:** clean/corrupt differ ideally at ONE token
  position and have **equal token length** (patching needs aligned positions).
  Generator uses short single-token entity/city names and offers a
  `--tokenizer` alignment check.
- A **distribution**, not one example: vary chain length, entity/city set,
  surface template. ≥150 pairs/condition. Single-prompt heatmaps are anecdote.

### 3.2 Patching protocol
- Run clean, cache all activations. Run corrupt. Patch clean→corrupt at each
  (component × position); measure logit-diff recovered. Average over the pair
  distribution; report **error bars**.
- Use **attribution patching / AtP\*** to get the full map in ~2 passes (huge on
  a laptop and *reinforces* the consumer-hardware narrative), THEN verify the
  top-k candidates with real activation patching.
- **Random-patching baseline** on every plot.

**Deliverable:** clean-vs-corrupt heatmaps (attention heads × layers) with the
minimal faithful circuit highlighted, per model.

**Gate G2 (end of Wk 5):** a small subset (target: single-digit # heads)
recovers **>70%** of the logit diff on ≥1 model, beating random baseline by a
wide margin. If not → simplify the task (shorter chains); do NOT proceed to
steering on a circuit you can't localize.

---

## 4. Phase 3 — Intervention: break, then (try to) repair (Weeks 6–10)

Goal: steer behavior **without changing weights**, and prove it doesn't wreck
general capability.

- **Derive the direction explicitly** — patching finds *where*, not *which
  direction*. Use **diff-of-means** (clean − corrupt residuals at the circuit
  sites); optionally **DAS** as a stronger localization. Name the method; don't
  hand-wave "a steering vector from the nodes."
- **Break (easy, reliable):** ablate / add negative direction → tracking accuracy
  drops. This is the guaranteed result.
- **Repair (hard, bonus):** on prompts the model gets *wrong*, add the direction
  and measure accuracy lift. Report honestly with CIs — repair often fails to
  generalize; the paper does NOT hinge on it.
- **Threat to validity — self-repair / backup heads:** ablating one head can be
  silently compensated by a backup head, corrupting importance estimates. Use
  path patching for localization and explicitly test for backup-head
  compensation. Discuss the Hydra effect in limitations.

### 4.1 Capability regression (the anti-"so what" control)
- Evaluate intervened model on **ARC-Easy / MMLU subset** locally.
- **Do NOT claim "zero degradation."** Claim **"degradation within noise"** with
  enough items + multiple seeds + **confidence intervals**. 2–3B models score low
  on MMLU (~40–50%), so "no change" is noisy — power it or reviewers call it
  underpowered.

**Gate G3 (end of Wk 8):** break works reliably AND capability delta is within
CI. Repair is a stretch goal, not a gate.

---

## 5. Phase 4 — Writing (Weeks 11–15)

Standard 9-page ICLR format.
- Frame consumer-HW constraint as a **feature** (auditable on a laptop), backed by
  real wall-clock + peak-RAM tables.
- Document exact head-isolation criteria via the **established** faithfulness /
  completeness / minimality definitions (METRICS.md) — do not invent bespoke ones.
- Figures: clean-vs-corrupt heatmaps (both models, side by side to show
  universal vs. arch-specific) + benchmark-regression curves with CIs.
- Limitations: single task family, small models, self-repair confound, fp32
  choice, MPS numerics.

---

## 6. Timeline with buffer & kill criteria (the blueprint had neither)

| Phase | Weeks | Gate |
|---|---|---|
| 1 Setup + baseline | Wk 1 | G1: fp32 clean, no swap |
| 2 Dataset + patching | Wk 2–5 | G2: >70% logit-diff by small subset, beats random |
| 3 Break/repair + regression | Wk 6–10 | G3: break reliable, capability within CI |
| 4 Writing | Wk 11–14 | draft complete |
| **Buffer** | **Wk 15** | overflow / rebuttal figures |

Kill rules: fail G2 → downscope to a single-model, single-task workshop note.
Fail G3-break → the paper becomes "localization + auditability," drop steering.

---

## 7. Risk register (updated)
- **Never quantize** (blueprint was right) — *and* prefer **fp32** for patching
  math, not fp16. Quantization + fp16 both distort activation geometry.
- **MPS flakiness / NaNs** → CPU-fp32 fallback wired from day one.
- **Gemma-2 soft-capping** → verify handling before trusting its numbers.
- **Self-repair** → path patching + explicit backup-head test.
- **Over-claiming** ("fix errors", "zero degradation") → hedged to
  break-reliable / repair-bonus / degradation-within-noise.
- **Novelty** → cross-architecture comparison is the moat; without it this is
  "another IOI-style circuit."
