# Design: First-Principles ICLR 2027 Topic Scouting

**Status:** Approved for written-spec review  
**Objective:** Select the strongest overall ICLR paper direction under local, CPU-only research constraints without anchoring the search to intent detection.

## 1. Optimization objective

Rank topics by expected research value:

- Novelty: 30%
- Acceptance probability: 25%
- Potential field impact: 20%
- CPU feasibility: 15%
- Execution speed: 10%

This weighting intentionally rejects both extremes: spectacular but infeasible ideas and easy but forgettable benchmark exercises.

## 2. Non-negotiable constraints

Every candidate must:

1. Make a central AI/ML contribution.
2. Require no commercial or hosted LLM API.
3. Be executable on commodity CPU hardware without a required GPU stage.
4. Use public models, datasets, and reproducible evaluation.
5. Focus on technical GenAI, SLM, VLM, inference-time learning, or local SLM tool use.
6. State a falsifiable hypothesis and a credible negative outcome.
7. Explain what new knowledge the field gains if the hypothesis is supported.

The existing intent-classification work may be reused as one experimental asset, but it cannot define the topic space or determine the winner.

## 3. Search strategy

Use a first-principles, adversarially filtered search rather than a trend-only or compute-convenience-only search.

Generate 30 independent candidates across seven families:

1. Adaptive SLM inference (4 candidates)
2. Inference-time learning and test-time adaptation (4 candidates)
3. Representation, memory, and KV-cache compression (4 candidates)
4. Uncertainty, hallucination detection, and selective generation (4 candidates)
5. Tiny VLM inference and cross-modal reasoning (4 candidates)
6. Mechanistic understanding of small generative models (5 candidates)
7. Local SLM tool use, routing, and executable feedback (5 candidates)

The allocation totals 30 candidates. Candidates must represent distinct research questions, not cosmetic variants of one method.

## 4. Candidate record

Each of the 30 candidates will include:

- Research question
- Proposed central claim
- Technical mechanism
- Why the idea matters
- Nearest crowded research area
- Strongest support argument
- Strongest refutation argument
- CPU-only experiment path
- Required public models and datasets
- Main evaluation metrics
- Minimum publishable result
- Kill criterion

## 5. Selection stages

### Stage A: Hard gates (30 to at most 20)

Reject any candidate that fails one or more of these:

- Hidden GPU dependency
- Dependence on proprietary or unavailable data
- Pure systems optimization without ML insight
- No reliable baseline implementation
- No measurable quality-cost trade-off
- Evidence limited to one model or one task family
- Contribution collapses into an ordinary ablation
- Research claim cannot survive a negative result

### Stage B: Twenty adversarial review loops

Run the survivors through these independent stress tests:

1. Novelty
2. Importance
3. Falsifiability
4. CPU feasibility
5. Dataset availability
6. Baseline strength
7. Statistical power
8. Cross-model generalization
9. Cross-task generalization
10. Saturated-area risk
11. "Just engineering" objection
12. "Just an ablation" objection
13. "Use a larger model" objection
14. Hidden compute dependency
15. Evaluation contamination
16. Reproducibility
17. Mechanistic or theoretical depth
18. Eight-page narrative coherence
19. Value of negative findings
20. Weighted expected-value score

The record must show how rankings change after every loop rather than presenting only the final score.

### Stage C: Finalists

For the top three candidates, produce:

- One-sentence thesis
- Method sketch
- Exact hypotheses
- Baseline matrix
- Dataset/model matrix
- Ablation plan
- Reliability and failure analysis
- Estimated CPU runtime
- Reviewer attacks and rebuttals
- Three-day feasibility pilot
- Kill-or-continue decision rule

## 6. Final deliverable

The final report will contain:

1. Thirty independent topics with support and refutation.
2. Explicit hard-gate results and the top-20 shortlist.
3. Twenty-loop ranking history.
4. Top-ten ranking with weighted scores.
5. Detailed blueprints for the top three.
6. One final winner and one lower-risk fallback.
7. A direct comparison against the previous intent-centered proposal.
8. A concrete next experiment that can falsify the winner quickly.

## 7. Winner standard

The winning contribution must be broader than "method X improves latency or accuracy." It should establish a reusable phenomenon or principle and introduce a method that exploits, predicts, or corrects it.

Target paper shape:

> We identify a previously under-measured inference phenomenon, explain when it appears, and introduce a lightweight method that exploits or mitigates it across multiple models and task families.

## 8. Scope controls

- VLM work is included only when the full evaluation is genuinely CPU-feasible.
- Tool-use research must use local deterministic tools (for example calculator, unit conversion, retrieval over a fixed corpus, date/time, SQL over a sandbox database, and constrained Python), never a paid or hosted model API.
- Tool-use candidates must separate at least four failure stages: invocation decision, tool selection, argument binding, and post-execution answer synthesis.
- Tool execution must be sandboxed, deterministic where possible, and scored separately from final-answer quality.
- Candidate US-origin model families include Microsoft Phi, NVIDIA Nemotron, Meta Llama, IBM Granite, Google Gemma, OpenAI GPT-2 as a non-tool-trained control, and Apple OpenELM as an additional control. Exact checkpoint provenance, license, chat template, and native function-call support must be verified from official model cards before inclusion.
- No topic receives bonus points merely because code already exists in this repository.
- Benchmark creation alone cannot win unless it exposes a new scientific phenomenon.
- Negative results are valuable only when they identify a mechanism or boundary condition.
- The final winner must have a credible three-day pilot before a full experiment campaign begins.
