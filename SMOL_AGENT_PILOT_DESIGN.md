# Smol-Agent Falsification Pilot: Experimental Design

**Target venue:** ICLR 2027  
**Status:** Design for review; no empirical claims  
**Primary topic:** Adaptive lifecycle control for CPU-scale small language models  
**Fallback:** The planning tax in small agents

## 1. Decision summary

Before implementing a learned six-action controller, run a small experiment that asks whether adaptive control has enough **oracle headroom** to justify a full paper.

The pilot compares fixed and simple routing policies on two already-available local models. It deliberately excludes learned routing, large benchmark integrations, web APIs, GPU execution, and broad model sweeps. If fixed policies behave similarly, or a keyword heuristic captures nearly all oracle improvement, the primary topic is killed early.

### Primary research question

> Under a fixed local CPU budget, does choosing among direct answering, planning, and tool use have enough input-dependent value to justify an adaptive Smol-Agent controller?

### Pilot decision

The pilot is not intended to prove the final paper. It decides among four outcomes:

1. **Continue:** sufficient cross-model routing headroom exists.
2. **Pivot to planning tax:** direct-versus-plan crossover is strong, but broad routing is not.
3. **Pivot to executable feedback:** execution/repair dominates matched extra reasoning.
4. **Kill this research family:** fixed policies or trivial heuristics are sufficient.

## 2. Hypotheses

### H1 — Policy heterogeneity

No fixed policy—direct, always-plan, or always-tool—wins across all task strata.

**Refutation:** one fixed policy is statistically tied for best in every stratum and both models.

### H2 — Oracle routing headroom

An item-level oracle policy materially outperforms the best fixed policy on both models.

**Refutation:** oracle gain is small, model-specific, or due only to parse-valid output.

### H3 — Non-trivial routing

A cue-balanced heuristic router recovers less than 70% of oracle headroom.

**Refutation:** keywords, numbers, dates, or explicit tool names explain nearly all routing value.

### H4 — Planning-tax crossover

Explicit planning hurts simple tasks but helps sufficiently compositional tasks, with the direction replicated across both models and prompt-template controls.

**Refutation:** planning is uniformly helpful, uniformly harmful, or template-dependent.

### H5 — Grounded execution value

On tool-essential tasks, execute-observe synthesis improves answer correctness beyond matched additional generation, not merely JSON validity.

**Refutation:** matched resampling or extra reasoning yields the same improvement.

## 3. Scope

### Included

- Two local models: Phi-3.5 Mini Instruct and Nemotron Mini 4B Instruct.
- Five deterministic local tools.
- Exactly 500 evaluation items.
- Greedy decoding for the primary pilot.
- Five policies: direct, always-tool, always-plan, heuristic, and oracle.
- Stage-wise correctness, risk, token, latency, and memory measurements.
- Cue-balanced and paraphrased evaluation sets.

### Excluded

- Learned controller training.
- Clarification, repair, and abstention as independently learned actions.
- Third model family.
- Hosted LLM APIs or web-dependent tools.
- GPU execution.
- Fine-tuning model weights.
- Full BFCL, tau-bench, or ToolSandbox integration.
- Multi-agent frameworks.

These exclusions are intentional. The pilot tests whether the research premise exists before creating a framework around it.

## 4. Models and inference contract

| Model | Pilot role | Tool protocol |
|---|---|---|
| `phi-3.5-mini` | General-instruction comparison | Explicit experiment-owned JSON schema; do not claim native tool support |
| `nemotron-mini-4b` | Tool-specialized positive control | Official model-card tool template where locally supported |

For every run, record:

- Exact checkpoint path and file hashes.
- Model configuration and dtype/quantization.
- Chat template and rendered prompt hash.
- Decoding parameters.
- Input/output token counts.
- CPU model, OS, Python, PyTorch, and Transformers versions.
- Thread count.
- Warm/cold wall time and peak resident memory.

### Primary decoding configuration

- Greedy decoding.
- Temperature `0` or framework-equivalent deterministic setting.
- Fixed maximum output tokens per policy stage.
- Maximum one plan and one tool call in the pilot.
- No hidden retries.
- Identical final-answer budget after tool execution.

A small stochastic-decoding sensitivity check is allowed only after the deterministic matrix is complete.

## 5. Deterministic tool suite

Each tool must have a typed schema, deterministic execution, explicit error classes, and no network access.

### 5.1 Calculator

Safe arithmetic expression evaluation using an allow-listed parser. Operations: addition, subtraction, multiplication, division, exponentiation, parentheses, percentages, and rounding.

### 5.2 Unit conversion

A frozen conversion registry for length, mass, temperature, volume, and time. Unsupported dimensions return a typed `unsupported_conversion` error.

### 5.3 Date arithmetic

ISO-date parsing, weekday lookup, date differences, and bounded date addition/subtraction. The timezone is fixed to UTC where relevant.

### 5.4 SQLite

Read-only queries over a frozen synthetic database with documented schema. The executor rejects mutation statements and multiple statements.

### 5.5 Frozen-corpus retrieval

Lexical retrieval over a versioned local corpus. Results include document ID, passage, and deterministic score. There is no live search or remote embedding service.

## 6. Evaluation dataset

Create exactly 500 versioned JSONL items. Strata are mutually exclusive at the top level, although metadata may describe secondary properties.

| Stratum | Count | Purpose |
|---|---:|---|
| Direct-answer | 100 | Tool use and planning should usually be unnecessary |
| Single-tool essential | 120 | Correct answer requires one deterministic tool result |
| Tool helpful but optional | 60 | Tests cost-sensitive choice rather than forced invocation |
| Ambiguous / clarification-worthy | 60 | Request lacks a required argument or has multiple valid interpretations |
| Impossible / abstention-worthy | 50 | Neither model knowledge nor available tools can support an answer |
| Malformed-call repair | 60 | A supplied candidate call contains schema, type, or semantic errors |
| Multi-step / multi-tool | 50 | Requires decomposition across two to four operations |
| **Total** | **500** | |

### Item schema

Each item must include:

- Stable item ID and generator version.
- Top-level stratum and difficulty metadata.
- User request.
- Allowed tool schemas.
- Gold lifecycle action.
- Gold tool name and arguments where applicable.
- Deterministic tool output or error.
- Gold final answer and acceptable normalization rules.
- Whether clarification or abstention is valid.
- Complexity: operation count, branch count, distractor count, and tool count.
- Cue-control group and paraphrase family.

### Cue balancing

The dataset must prevent obvious lexical shortcuts:

- Include numbers and dates in direct-answer items.
- Express some tool-essential requests without explicit tool words.
- Include explicit tool names where tool use is unnecessary.
- Balance request length across strata.
- Hold out paraphrase templates and tool descriptions for evaluation.
- Keep paired items whose semantics are similar but gold actions differ.

### Leakage controls

- Programmatically generate private-template evaluation items.
- Freeze generator code before model evaluation.
- Do not tune prompts on the held-out evaluation split.
- Record item/template provenance.
- Deduplicate normalized requests across train, development, and evaluation splits.

## 7. Policies

### P1 — Direct

The model answers without planning or tool execution.

### P2 — Always tool

The model must select one available tool and provide arguments, then synthesize a final answer from the result. Invalid calls receive no repair during the primary pilot.

### P3 — Always plan

The model emits a short bounded plan before answering or selecting a tool. Plan length is capped to prevent unbounded reasoning cost.

### P4 — Heuristic router

A transparent rule set chooses direct, plan, or tool using lexical and structural features. Rules are frozen on the development split.

### P5 — Oracle router

For each item, select the best observed result among P1–P3 using gold correctness, with deterministic cost-aware tie-breaking. This is an upper-bound diagnostic, not a deployable method.

### Matched-generation control

For H5, direct generation receives an additional output budget matched separately by generated tokens and observed CPU time. This control is analyzed on tool-essential items and is not treated as a sixth routing policy.

## 8. Execution matrix

Primary matrix:

- 2 models
- 500 items
- 5 policies
- 1 deterministic decoding configuration

Total: **5,000 model–item–policy runs**, plus warmups and the matched-generation H5 subset.

Run order must be randomized within model while preserving item-policy pairing. Checkpoint loading should be amortized, but cold-start load time must be reported separately.

## 9. Metrics

### Primary endpoints

1. Final-answer correctness.
2. End-task success.
3. Oracle gain over the best fixed policy.
4. Fraction of oracle headroom recovered by the heuristic router.
5. Correctness per CPU-second.

### Stage-wise diagnostics

- Gold-action accuracy.
- Tool necessity decision.
- Tool-name accuracy.
- Argument exact and normalized match.
- Schema validity.
- Execution success.
- Tool-result faithfulness.
- Final synthesis correctness.
- Unnecessary-call and missed-needed-call rates.
- Valid clarification and abstention rates.

### Efficiency

- Prompt and generated tokens.
- P50/P95 end-to-end latency.
- Tool execution latency.
- Peak resident memory.
- Correct answers per CPU-hour.

### Composite utility

Composite utility is secondary because arbitrary weights can manufacture a winner. Report raw metrics first. If utility is used, predeclare weights and provide a sensitivity plot over plausible call, latency, error, and abstention costs.

## 10. Statistical analysis

- Use paired item-level bootstrap confidence intervals for policy differences.
- Use McNemar tests for paired binary correctness comparisons.
- Report absolute percentage-point changes, not only relative gains.
- Correct the small set of predeclared primary comparisons using Holm’s method.
- Stratify by model, task stratum, and complexity.
- Estimate the planning crossover with an interaction between planning policy and controlled task complexity.
- Do not use significance as the sole continuation criterion; effect size and replication matter.

The pilot is exploratory for effect-size estimation. A full paper run must use frozen hypotheses and an independently generated evaluation set.

## 11. Go/no-go rules

Proceed to the full adaptive-controller experiment only when all core gates pass:

| Gate | Continue condition |
|---|---|
| G1: Heterogeneity | Different fixed policies win meaningful task strata, with at least a 5-point correctness difference in two or more strata |
| G2: Oracle headroom | Oracle exceeds the best fixed policy by at least 8 points overall and at least 5 points on each model |
| G3: Non-triviality | Heuristic recovers less than 70% of oracle headroom on held-out cue-balanced items |
| G4: Non-format gain | Headroom remains after restricting to parse-valid outputs |
| G5: Cross-model replication | Direction of G1–G4 agrees for Phi and Nemotron |
| G6: CPU relevance | The oracle allocation indicates a plausible quality–latency Pareto improvement rather than accuracy at unlimited cost |

### Pivot rules

- **Planning crossover passes, broad routing fails:** pivot to *The Planning Tax*.
- **Execution feedback beats matched reasoning, broad routing fails:** pivot to *Execute or Think Longer?*
- **Most failures are invalid serialization:** do not claim adaptive agency; treat constrained decoding as a separate engineering/method question.
- **Oracle headroom fails:** stop. Do not build a learned controller.

## 12. Failure taxonomy

Every failed run receives one primary label:

1. Wrong lifecycle action.
2. Wrong tool.
3. Missing required argument.
4. Incorrect argument value or type.
5. Invalid schema/serialization.
6. Tool runtime error.
7. Ignored or contradicted tool result.
8. Correct evidence but wrong synthesis.
9. Unnecessary planning overhead.
10. Unsupported answer instead of clarification/abstention.

A small manually audited subset must validate automatic labeling precision before failure-distribution claims are made.

## 13. Proposed repository architecture

Implementation should extend the existing `experiments/code` package without coupling agentic experiments to the current intent/slot pipeline.

```text
experiments/code/
├── data/agentic/
│   ├── corpus/
│   ├── database/
│   └── generated/
├── src/slmpaper/agentic/
│   ├── schemas.py
│   ├── tools.py
│   ├── task_generator.py
│   ├── policies.py
│   ├── prompts.py
│   ├── runner.py
│   ├── evaluation.py
│   └── statistics.py
├── scripts/
│   ├── generate_agentic_pilot.py
│   ├── run_agentic_pilot.py
│   └── analyze_agentic_pilot.py
└── tests/agentic/
```

Design constraints:

- Pure deterministic logic remains testable without model dependencies.
- Tool execution is separated from model prompting.
- Policies share one typed interface.
- Run manifests make every result traceable to model, prompt, item, and code version.
- Raw generations are immutable; analysis produces derived artifacts.
- No module should become an agent-framework junk drawer.

## 14. Artifact contract

Each run writes:

- Immutable JSONL records for every stage.
- One manifest containing git commit, environment, model hashes, prompts, and decoding settings.
- Aggregate JSON and CSV metrics.
- Failure-taxonomy counts.
- Latency and memory summaries.
- A Markdown decision report stating which gates passed or failed.

Secrets, personal data, and hosted-service credentials are prohibited. Synthetic database records must not resemble real people.

## 15. Three-day execution schedule

### Day 1 — Harness and dataset contract

- Implement and test typed schemas and five deterministic tools.
- Implement item validation and task generators.
- Generate a 100-item smoke set and manually audit 20 items.
- Verify official prompt templates and deterministic inference for both models.

**Exit:** tools and items are deterministic; both models complete one example under every fixed policy.

### Day 2 — Pilot matrix

- Freeze the 500-item dataset.
- Run P1–P4 for Phi and Nemotron.
- Compute P5 oracle results from fixed-policy outputs.
- Run the H5 matched-generation subset.
- Capture raw generations, latency, tokens, and memory.

**Exit:** complete paired result matrix with no silent retries or missing cells.

### Day 3 — Analysis and decision

- Validate automatic grading on a manual audit subset.
- Produce primary, stage-wise, and efficiency tables.
- Bootstrap policy differences and test the planning interaction.
- Evaluate G1–G6 and apply pivot rules.
- Write the pilot decision report.

**Exit:** explicit continue, pivot, or kill decision supported by measured effects.

The schedule is a sequencing target, not a promise of wall-clock completion. CPU inference throughput may require reducing the pilot only through a documented design deviation—not silently dropping difficult strata.

## 16. Full experiment only after pilot success

If the pilot passes, the next design adds:

- A learned value/risk controller.
- Clarification, bounded repair, and abstention as explicit actions.
- Granite or Llama as a third independent family.
- Independently generated evaluation items.
- Cross-tool and held-out-tool transfer.
- Calibration and selective-risk curves.
- Matched CPU/token/time ablations.
- Multiple seeds where stochastic components exist.

This full phase is intentionally not designed in implementation detail until pilot evidence establishes that adaptive routing is worth learning.

## 17. Review checklist

- [x] Research question is falsifiable.
- [x] Primary and fallback outcomes are explicit.
- [x] No LLM API or GPU is required.
- [x] Pilot uses available model families.
- [x] Dataset strata and counts sum to 500.
- [x] Baselines include fixed, heuristic, and oracle policies.
- [x] Formatting-only gains cannot satisfy continuation gates.
- [x] CPU efficiency is measured rather than inferred.
- [x] Kill criteria are defined before implementation.
- [x] Full-controller scope is deferred until evidence exists.
