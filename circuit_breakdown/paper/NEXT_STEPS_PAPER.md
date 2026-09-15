# What's next

State as of commit `0721508`. Experiments on the main claim are **frozen**;
`results/frozen_e91985c/` is the snapshot the paper is written from.

(Not to be confused with the repo-root `NEXT_STEPS.md`, a Sep-7 handoff doc
from earlier work. This file covers the paper only.)

---

## Done

| | |
|---|---|
| Experiments | Frozen at `e91985c`, both models, 120 independent pairs each |
| Main result | Selectivity 0/240 in two families, 95% CI [0.0, 1.6%] |
| Metric validated | Counter fires 24/24 unpatched, 0/24 patched |
| Paper | Full draft, 9 sections, ~3k words |
| Figures | 4, all reviewed for overlap and legibility |
| Numbers guarded | `verify_paper_numbers.py`, 44/44 against frozen logs |
| Rendered | `paper/paper.html`, self-contained, figures inlined |

---

## Next, in the order I would do them

### 1. Read the draft and decide the venue  *(you, ~1 hour)*

`paper/paper.html` is open in Chrome. Nothing else should happen until you
have read it end to end, because venue choice changes the next three items.

| venue | fit | note |
|---|---|---|
| **ICML/NeurIPS interpretability workshop** | **strong** | Scope and scale match a 4-8 page workshop paper. Most honest fit. |
| **BlackboxNLP** | strong | Explicitly hosts this kind of probing/intervention negative result. |
| ACL/EMNLP Findings | plausible | Would want a third model family and a second task. |
| ICML/NeurIPS main track | **weak** | Needs the layer x position study below. Not supported by current evidence. |

The original plan targeted ICML main track. The evidence does not support
that. A workshop paper with a clean negative result is a real contribution; a
rejected main-track submission is not.

### 2. Tighten the draft  *(~2 hours)*

- Sections 1 and 7 are the loosest; both can lose roughly a third.
- Section 6 (supporting geometry) can move to an appendix if pages are tight.
  It supports the interpretation but is not load-bearing.
- Add a short related-work paragraph on activation steering, since reviewers
  from that community will ask how this relates to their methods.

### 3. Convert to the venue template  *(~1 hour once chosen)*

`pandoc` is installed; `pdflatex` is **not**. Either
`brew install --cask mactex-no-gui` (large) or `brew install basictex`
(smaller, then `tlmgr install` what the template needs). Do this only after
the venue is picked, so the template is right the first time.

### 4. Write the appendix  *(~1 hour)*

The material exists and needs assembling:
- Full probe templates and the item generator.
- The competence gate, and why `transfer_count` was excluded in both models.
- The non-uniqueness falsification, if a reviewer would otherwise raise
  non-uniqueness as an alternative explanation.
- The interchange audit, including the specificity measurement error, which is
  better shown than hidden.

---

## Deliberately NOT next

**The layer x position map.** Highest-ceiling idea available: the effect
appears at position -1 and is absent at -2 through -7, suggesting a transition
from state-level to answer-level control along the trajectory. Mapping it
means layers x positions x probes x models with a selectivity panel per cell.
That is a second paper, roughly a month. It belongs in section 9 as future
work.

**A third model family.** Only worth it for a Findings-level submission, and
it needs the full protocol re-run.

**More reasoning tasks.** Same reasoning. The single-task limitation is
already stated plainly in section 8.

---

## Open risks a reviewer will probe

1. **"You patched the final token, so you measured an answer."** Conceded
   outright in section 7. The defence is that the intervention passes four
   credentials a practitioner would accept, so the credential stack is the
   object of study. This argument will most likely decide the review.
2. **"RAVEL already showed this."** Addressed in sections 1 and 7: RAVEL
   covers static entity attributes, this covers computed state. If a reviewer
   rejects that distinction, the paper has no contribution.
3. **"Llama is weak."** Stated in section 8. Selectivity is claimed for both
   models, completeness only for Phi.
4. **Two probes per model.** `transfer_count` excluded for low competence.
   Honest, but thin; a reviewer may push for more probes.

---

## Commands

```bash
cd /Users/s0n0611/Documents/GitHub/SLM_PAPER/circuit_breakdown
source .venv/bin/activate

python src/verify_paper_numbers.py          # 44 checks against the frozen snapshot
./scripts/build_paper.sh                    # verify, then render paper/paper.html
python src/make_paper_figures.py            # rebuild figures 1-3
python src/make_selectivity_explainer.py    # rebuild figure 4
```
