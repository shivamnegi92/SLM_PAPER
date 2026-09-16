# GitHub Learnings -> Innovation Plan (for this project)

Date: 2026-09-05
Scope: Latest practical mechanistic-interpretability codebases and patterns,
then a concrete plan for `circuit_breakdown`.

## 1) What GitHub says is actually working

Most active/high-signal repos:
- TransformerLens
- SAELens
- NNSight
- Pyvene
- ACDC (automated circuit discovery)
- steering-vectors
- sparsify / SAE repos
- ARENA implementations

Practical trend:
1. Use attribution patching as cheap first pass.
2. Verify with exact activation/path/head patching.
3. Report train/dev/test + random controls + CIs.
4. Distinguish logit lift vs behavioral flip (top-1).

## 2) Main takeaways for our current project

We already did a lot right:
- Cross-arch localization matrix (3x2)
- Head-level mover circuit
- Relative-depth invariance finding
- Capability sweep with CIs

But our current blocker is exactly what many repos/issues report:
- Additive vectors often increase target logits but do not flip top-1 reliably.
- Therefore "constructive steering" needs path- or objective-aware intervention,
  not just mean-difference or simple low-rank linear maps.

## 3) Innovation direction (recommended)

### Core innovation thesis
"Interventions that are causally valid under patching are often not behaviorally
valid under decision-level control unless they optimize sequence-level objectives
and respect path constraints."

This gives us a publishable angle beyond "we found a circuit".

## 4) Concrete execution plan

### Block A — Intervention Failure Atlas (highest ROI, immediate)
Goal: turn our current negative results into a structured scientific contribution.

Methods to compare (same split/harness):
1. Single-layer additive vector
2. Multi-layer additive vector
3. DAS-style low-rank map
4. Prompt-specific optimization (sequence objective, no weight updates)

Metrics (all required):
- top-1 STEER rate (corrupt->clean)
- BREAK side effect (clean->wrong)
- clean-target logit lift
- margin lift (clean target vs current top-1)
- transfer to held-out templates/entities

Deliverable:
- `intervention_failure_atlas.json`
- figure: behavior flip vs logit lift across methods

### Block B — Attribution Reliability Benchmark
Goal: quantify when attribution patching can be trusted.

Compare on same tasks:
- exact patching ranking
- attribution patching ranking
- random baseline ranking

Metrics:
- rank correlation
- top-k overlap
- behavioral agreement of chosen top-k

Deliverable:
- reliability table + confidence intervals
- clear guidance: "use attribution for X, not Y"

### Block C — Path-Constrained Constructive Intervention
Goal: improve top-1 STEER with objective-aware inference intervention.

Approach:
- optimize intervention vectors at selected layers to maximize
  clean-target margin on corrupt prompts under a norm budget.
- add penalties for clean breakage and KL drift.
- keep inference-time only (no model weight update).

Deliverable:
- if success: first robust constructive steering result in this project
- if not: strong evidence of structural nonlinearity/overwriting

### Block D — Parent Thesis Bridge (highest impact for acceptance)
Goal: connect to SLM_PAPER "interface vs capacity" claim mechanistically.

Experiment:
- complete schema prompts vs underspecified schema prompts.
- test whether identified tracking circuit engagement differs causally.

Deliverable:
- mechanistic evidence that interface completeness changes circuit use,
  not just output quality.

## 5) Prioritized order (do this)
1. Block A (Failure Atlas)
2. Block C (Path-constrained constructive intervention)
3. Block B (Attribution reliability)
4. Block D (Parent-thesis bridge)

## 6) Why this beats generic mech-interp papers
- Not just localization: includes decision-level intervention realism.
- Honest negative + positive structure with unified evaluation.
- Cross-architecture + causal + capability + deployment-relevant constraints.

## 7) Acceptance impact estimate (internal)
Current: ~6.5/10
After A+C with meaningful flip gains or strong impossibility evidence: 7.0+
After D (parent-thesis bridge): 7.5-8.0 potential

## 8) Immediate next command set
- implement `src/intervention_atlas.py`
- add margin objective intervention variant to `src/intervene_das.py`
- run n>=60 per task, bootstrap CIs
- append section to RESULTS.md: "Decision-level intervention benchmark"
