# Smol-Agent and CPU-Efficient GenAI Research Scouting for ICLR 2027

**Decision:** pursue **Adaptive Lifecycle Control for Smol Agents** as the leading paper direction, subject to a three-day falsification pilot.  
**Fallback:** **The Planning Tax in Small Agents**.  
**Constraints:** local public models; no hosted LLM API; no required GPU; CPU-reproducible; technical AI/ML contribution; no dependence on the existing intent task.

## Executive conclusion

The strongest paper is not “small models can call tools.” Toolformer, ReAct, TinyAgent, BFCL-related work, and tool-specialized checkpoints already occupy that neighborhood. The stronger research question is:

> **Under a fixed CPU budget, when should a small local model answer, clarify, plan, act, repair, or abstain—and can a calibrated bounded policy choose better than fixed agent loops?**

This direction wins because it studies a reusable inference principle: heterogeneous agentic computation should be allocated according to predicted value and risk. Tool use becomes one action inside a closed control loop rather than the entire contribution.

The recommended paper shape is:

> We identify action-specific capability boundaries in 1–4B local models, show that fixed direct/plan/tool policies waste compute or amplify errors on different inputs, and introduce a bounded risk-calibrated policy that improves utility per CPU-second by selecting among answer, clarification, planning, tool execution, repair, and abstention.

This is a research recommendation, not an empirical claim. All performance claims remain to be tested.

## Research method

Thirty independent candidates were generated across seven families, hard-filtered to twenty, and subjected to twenty adversarial review loops. Planning scores used the approved expected-value weighting:

- Novelty: 30%
- Acceptance probability: 25%
- Potential impact: 20%
- CPU feasibility: 15%
- Execution speed: 10%

No candidate received credit because code already existed in this repository. Existing Phi and Nemotron artifacts reduce pilot setup cost but did not determine the ranking.

## Sources consulted

### Official model sources

| Source | URL | Use in this report |
|---|---|---|
| Microsoft Phi-3.5 Mini Instruct model card | https://huggingface.co/microsoft/Phi-3.5-mini-instruct | Size, intended constrained deployment, license, prompt format |
| Phi-3 Technical Report | https://arxiv.org/abs/2404.14219 | Phi family background |
| Microsoft Phi-4 Mini Instruct model card | https://huggingface.co/microsoft/Phi-4-mini-instruct | Candidate checkpoint; exact tool protocol still unverified here |
| NVIDIA Nemotron Mini 4B Instruct model card | https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct | Verified function-calling specialization and template |
| NVIDIA Minitron compression paper | https://arxiv.org/abs/2407.14679 | Compression lineage |
| Meta Llama 3.2 3B Instruct model card | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct | Candidate checkpoint; exact tool support must be rechecked |
| The Llama 3 Herd of Models | https://arxiv.org/abs/2407.21783 | Model-family reference |
| IBM Granite 3.3 2B Instruct model card | https://huggingface.co/ibm-granite/granite-3.3-2b-instruct | Candidate checkpoint; exact template/license must be rechecked |
| Granite 3.0 Language Models | https://arxiv.org/abs/2409.16013 | Model-family reference |
| Google Gemma 3 model card | https://ai.google.dev/gemma/docs/core/model_card_3 | Optional model-family reference |
| Gemma 3 Technical Report | https://arxiv.org/abs/2503.19786 | Optional family reference |
| OpenAI GPT-2 report | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | Historical non-instruction control |
| Apple OpenELM | https://arxiv.org/abs/2404.14619 | Modern non-tool-specialized control candidate |

### Agent, tool-use, memory, and test-time-compute sources

| Work | URL | Novelty boundary established |
|---|---|---|
| ReAct | https://arxiv.org/abs/2210.03629 | Interleaved reasoning and action is established |
| Toolformer | https://arxiv.org/abs/2302.04761 | Whether/when/which tool and arguments are established questions |
| Gorilla | https://arxiv.org/abs/2305.15334 | API retrieval/call generation |
| API-Bank | https://arxiv.org/abs/2304.08244 | Tool-augmented benchmark coverage |
| ToolLLM / ToolBench | https://arxiv.org/abs/2307.16789 | Multi-tool planning and execution trajectories |
| AgentBench | https://arxiv.org/abs/2308.03688 | Broad interactive-agent evaluation |
| MetaTool | https://openreview.net/forum?id=0yx6yOyfO2 | Tool awareness/selection benchmarking |
| Berkeley Function Calling Leaderboard | https://arxiv.org/abs/2402.04253 | Function selection and argument correctness |
| tau-bench | https://arxiv.org/abs/2406.12045 | Stateful tool-agent-user interaction |
| ToolSandbox | https://arxiv.org/abs/2408.04682 | Stateful conversational tool-use evaluation |
| TinyAgent | https://arxiv.org/abs/2409.00608 | Small-model function calling at the edge |
| ToolACE | https://arxiv.org/abs/2409.00920 | Function-calling data/evaluation |
| Reflexion | https://arxiv.org/abs/2303.11366 | Feedback, reflection, and episodic memory |
| CRITIC | https://arxiv.org/abs/2305.11738 | Tool-interactive correction |
| Self-Refine | https://arxiv.org/abs/2303.17651 | Iterative self-feedback and refinement |
| MemGPT | https://arxiv.org/abs/2310.08560 | Hierarchical/virtual context management |
| Scaling LLM Test-Time Compute Optimally | https://arxiv.org/abs/2408.03314 | Adaptive inference-compute allocation |
| Small Language Models are the Future of Agentic AI | https://arxiv.org/abs/2506.02153 | Directly relevant 2025 SLM-agent position/reference |

## Key evidence extracted before analysis

From the locally inspected official Phi-3.5 Mini model card:

> “The model provides uses for general purpose AI systems and applications which require: 1) Memory/compute constrained environments 2) Latency bound scenarios 3) Strong reasoning…” — Microsoft Phi-3.5 Mini model card

> “Architecture: Phi-3.5-mini has 3.8B parameters and is a dense decoder-only Transformer model…” — Microsoft Phi-3.5 Mini model card

The inspected Phi-3.5 card documents ordinary system/user/assistant chat formatting but did not document a tool schema or function-calling benchmark. Therefore, this report treats Phi-3.5 as a general-instruction comparison model, not as verified natively tool-trained.

From the locally inspected official Nemotron Mini model card:

> “Nemotron-Mini-4B-Instruct is a model for generating responses for roleplaying, retrieval augmented generation, and function calling.” — NVIDIA Nemotron Mini model card

> “It is a small language model (SLM) optimized through distillation, pruning and quantization for speed and on-device deployment.” — NVIDIA Nemotron Mini model card

The card also documents `<tool>`, `<toolcall>`, and tool-response serialization. Nemotron Mini is therefore the strongest verified tool-specialized positive control in the current model set.

From Toolformer:

> “We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction.” — Toolformer abstract, arXiv:2302.04761

This blocks any novelty claim based solely on deciding whether/when/which tool to call or generating its arguments.

From ReAct:

> “We explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner…” — ReAct abstract, arXiv:2210.03629

This blocks claiming interleaved planning and acting itself as new.

## Model audit

“US-origin” here means released by a US-headquartered organization, not that every contributor, training datum, or compute location was US-based.

| Model | Organization | Approx. size | Tool-support status used here | Intended experimental role | Caveat |
|---|---|---:|---|---|---|
| Nemotron-Mini-4B-Instruct | NVIDIA | 4B class | **Verified official function-calling specialization/template** | Positive control | Custom NVIDIA license; measure actual CPU latency |
| Phi-3.5-mini-instruct | Microsoft | 3.8B | **No native support verified in inspected card** | General-instruct contrast | 128K maximum context is not a practical CPU experiment target |
| Phi-4-mini-instruct | Microsoft | ~3.8B | **Unverified here** | Possible newer replacement | Recheck exact official template and license before use |
| Llama-3.2-3B-Instruct | Meta | 3B | **Checkpoint-specific status unverified here** | Independent model family | Custom Llama license and template controls |
| Granite-3.3-2B-Instruct | IBM | 2B class | **Likely documented, but unverified here** | Smallest primary family | Verify exact parameters, license, and template |
| Gemma-3-1B/4B-IT | Google | 1B/4B | **Family guidance is not proof of native training** | Optional scaling endpoint | Gemma terms and text-only loading path |
| OpenELM-3B-Instruct | Apple | 3B | **No verified native tool support** | Optional modern control | Smaller tool ecosystem |
| GPT-2 | OpenAI | 124M–1.5B | None in original release | Historical negative control only | Not a fair modern agent baseline |

### Recommended model set

For the three-day pilot:

1. **Nemotron-Mini-4B-Instruct** — verified tool-specialized endpoint.
2. **Phi-3.5-mini-instruct** — existing general-instruction comparison.
3. **Granite-3.3-2B-Instruct** or **Llama-3.2-3B-Instruct** — independent family after official checkpoint verification.

Do not begin with seven models. Four families are enough for a full paper; three are enough to falsify it.

### CPU reporting contract

“Runs on CPU” is not sufficient. Record exact quantization, resident memory, KV-cache configuration, prompt/output tokens, warm/cold latency, threads, CPU model, RAM, tokens/second, P50/P95 end-to-end latency, and utility per CPU-second. Checkpoint size is not runtime memory.

## Prior-work boundary

| Existing area | Already established | Defensible remaining question |
|---|---|---|
| Toolformer/tool calling | Calling decision, selection, arguments, use of results | Joint risk-calibrated allocation among heterogeneous actions under a CPU budget |
| ReAct/planning | Interleaved reasoning and action | When planning helps or harms 1–4B agents and when to skip it |
| Reflexion/CRITIC/Self-Refine | Feedback and iterative correction | Bounded diagnosis-conditioned executable repair at matched compute |
| BFCL/MetaTool/TinyAgent | Function-call correctness and small edge agents | Complete lifecycle decomposition plus selective risk and efficiency |
| MemGPT/agent memory | Hierarchical and episodic memory | Whether typed state beats transcript memory per byte/token for small agents |
| Test-time scaling | Allocating reasoning/search compute | Allocating heterogeneous agent actions: clarify, execute, repair, or abstain |

The report therefore does **not** claim that tools, planning, feedback, memory, or abstention are individually novel.
## Thirty candidate research directions

### Family A — Adaptive SLM inference

1. **Budget-conditioned token generation.** Hypothesis: a learned stopping policy can predict when further tokens will not improve answer correctness. Support: generation cost is output-length dependent. Refutation: confidence may merely track answer length. CPU path: 1–4B local models on short-form QA/reasoning. Metrics: risk–latency frontier and accuracy/token. Kill: no gain over fixed max-token thresholds.
2. **Layer-and-token joint early exit.** Hypothesis: allocating both transformer depth and output length per example dominates either control alone. Support: easy inputs waste compute along two axes. Refutation: routing overhead and cache incompatibility may erase savings. CPU path: prunable GPT-2/Phi-like backbones. Metrics: quality, P50/P95 latency, FLOPs. Kill: joint policy is not Pareto-superior.
3. **Confidence-triggered self-verification.** Hypothesis: SLMs benefit more when verification is selectively invoked than when every answer is verified. Support: fixed verification wastes compute on easy items. Refutation: model confidence may be poorly calibrated. CPU path: local verifier prompts and deterministic checks. Metrics: selective risk and accuracy/CPU-second. Kill: simple entropy threshold matches the method.
4. **Difficulty-aware speculative decoding with a tiny drafter.** Hypothesis: draft acceptance can be predicted from prompt representations, enabling adaptive draft length. Support: fixed draft lengths waste rejected tokens. Refutation: implementation complexity may dominate at 1–4B. CPU path: GPT-2 drafter plus Phi/Nemotron target. Metrics: accepted tokens, latency, exact output parity. Kill: no real wall-clock gain.

### Family B — Inference-time learning and adaptation

5. **Unlabeled test-time calibration under shift.** Hypothesis: entropy/consistency statistics can recalibrate small generative models without weight updates. Support: SLM confidence drifts under domain shift. Refutation: pseudo-label feedback can reinforce errors. CPU path: temperature/vector scaling from augmentations. Metrics: ECE, Brier score, selective risk. Kill: static calibration is equally good.
6. **Retrieval-conditioned test-time prompt adaptation.** Hypothesis: choosing demonstrations by uncertainty and diversity outperforms nearest-neighbor retrieval. Support: small models are sensitive to example choice. Refutation: gains may be retrieval leakage. CPU path: frozen embeddings and local inference. Metrics: accuracy, latency, prompt tokens. Kill: random or nearest-neighbor matches.
7. **Error-memory online adaptation without gradients.** Hypothesis: compact summaries of prior failures improve later decisions. Support: episodic feedback may compensate for small capacity. Refutation: memory contamination and order effects. CPU path: structured local memory. Metrics: forward transfer, forgetting, tokens. Kill: shuffled memory performs equally.
8. **Self-generated micro-curricula at inference.** Hypothesis: solving easier decomposed variants first improves hard-query performance. Support: curricula may scaffold SLM reasoning. Refutation: extra tokens may simply create more errors. CPU path: bounded local generation. Metrics: hard-task accuracy per CPU-second. Kill: matched self-consistency wins.

### Family C — Representation, memory, and KV-cache compression

9. **Attention-salience KV eviction with correctness alarms.** Hypothesis: cache entries can be evicted using salience while an uncertainty alarm prevents catastrophic loss. Support: long contexts contain redundant tokens. Refutation: attention is not reliable importance. CPU path: short 1–4B contexts with controlled distractors. Metrics: memory, latency, answer accuracy. Kill: recency eviction matches it.
10. **Task-aware KV quantization.** Hypothesis: cache precision should depend on token role and predicted reuse. Support: prompt/tool-result tokens differ in importance. Refutation: kernel overhead may exceed memory benefit. CPU path: software quantization on local models. Metrics: RAM, latency, perplexity/task accuracy. Kill: uniform quantization dominates.
11. **Structured state versus transcript memory.** Hypothesis: compact typed state beats full natural-language trajectories for small agents. Support: irrelevant history interferes with limited capacity. Refutation: state extraction may lose nuance. CPU path: deterministic local environments. Metrics: success, memory tokens, stale-state errors. Kill: sliding window is equal or better.
12. **Memory-value prediction.** Hypothesis: an SLM can predict whether reading/writing a memory item is worth its context cost. Support: indiscriminate memory bloats prompts. Refutation: the predictor may cost as much as retrieval. CPU path: lightweight classifier/probe. Metrics: value-regret, accuracy/byte. Kill: simple recency heuristic matches.

### Family D — Uncertainty, hallucination detection, and selective generation

13. **Semantic-consistency calibration without multiple full samples.** Hypothesis: hidden-state perturbations approximate expensive self-consistency. Support: sampling is costly on CPU. Refutation: perturbation scores may not reflect semantic uncertainty. CPU path: one forward pass plus shallow probes. Metrics: AUROC, ECE, latency. Kill: token entropy wins.
14. **Evidence-sufficiency routing.** Hypothesis: an SLM can decide whether evidence supports answer, retrieval, clarification, or abstention. Support: hallucination often begins with unsupported action choice. Refutation: task cues may make routing trivial. CPU path: frozen corpus and local tools. Metrics: selective risk and unsupported-claim rate. Kill: lexical heuristic matches.
15. **Counterfactual prompt stability as hallucination signal.** Hypothesis: answer instability under meaning-preserving rewrites predicts factual error. Support: brittle responses indicate weak support. Refutation: correct answers may also vary stylistically. CPU path: deterministic paraphrase templates. Metrics: error detection and compute. Kill: no correlation after normalization.
16. **Risk-controlled structured generation.** Hypothesis: conformal/selective thresholds can bound invalid or unsupported structured outputs. Support: agent actions have asymmetric risk. Refutation: exchangeability fails under shift. CPU path: schema tasks and local models. Metrics: coverage-risk curves. Kill: bounds do not hold on shifted sets.

### Family E — Tiny VLM inference and cross-modal reasoning

17. **Question-conditioned visual token pruning.** Hypothesis: text queries identify dispensable patches before deep fusion. Support: many image tokens are irrelevant. Refutation: small-object evidence may be pruned. CPU path: small CLIP/vision-language encoders on VQA subsets. Metrics: accuracy, tokens, latency. Kill: uniform downsampling is equal.
18. **Visual-evidence sufficiency and refusal.** Hypothesis: tiny VLMs can abstain when the image lacks required evidence. Support: hallucination can be framed as missing evidence. Refutation: datasets may reward priors. CPU path: synthetic occlusion/counterfactuals. Metrics: selective VQA accuracy. Kill: confidence is unrelated to evidence removal.
19. **Cross-modal early exit.** Hypothesis: easy image-text pairs can exit before all fusion layers. Support: alignment difficulty varies. Refutation: suitable CPU VLMs and hooks may be brittle. CPU path: compact dual/fusion encoders. Metrics: quality-latency frontier. Kill: fixed shallow model matches.
20. **Modality-drop causal diagnostics.** Hypothesis: controlled modality removal exposes when tiny VLMs answer from language priors rather than vision. Support: causal diagnostic is broadly useful. Refutation: diagnostic-only contribution may be insufficient. CPU path: VQA/captioning subsets. Metrics: causal reliance scores. Kill: scores do not predict hallucination.

### Family F — Mechanistic understanding of small generative models

21. **Where tool affordances emerge by layer.** Hypothesis: tool-necessity, selection, and argument representations emerge at different depths. Support: stage-specific representations may explain failures. Refutation: probes can read information unused causally. CPU path: layer probes plus activation interventions. Metrics: probe accuracy and causal effect. Kill: interventions do not follow probe results.
22. **Representation competition between knowledge and tool use.** Hypothesis: small models exhibit a capacity trade-off between internal recall and external-action policies. Support: limited capacity may force competition. Refutation: differences may reflect training data. CPU path: matched prompts across models. Metrics: paired behavioral/representation shifts. Kill: no reproducible trade-off.
23. **Mechanism of planning-induced degradation.** Hypothesis: explicit planning harms SLMs when early plan errors become high-confidence conditioning context. Support: autoregressive error propagation. Refutation: prompt formatting may explain it. CPU path: controlled plan corruption and activation tracing. Metrics: error amplification. Kill: effect disappears with template control.
24. **Tool-result faithfulness circuits.** Hypothesis: specific heads/layers mediate copying and transforming tool outputs. Support: agents often ignore correct tool results. Refutation: distributed representations may resist localization. CPU path: causal patching on small open models. Metrics: intervention effect and localization. Kill: no stable circuit across seeds/models.
25. **Compression-induced agentic capability phase transitions.** Hypothesis: quantization/pruning damages action selection before language fluency. Support: structured control may be fragile. Refutation: backend artifacts may dominate. CPU path: 4/8-bit and depth variants. Metrics: stage-wise failure curves. Kill: degradation is uniform.

### Family G — Agentic Small Language Models (Smol Agents)

26. **Adaptive answer–clarify–plan–act–repair–abstain policy.** Hypothesis: calibrated lifecycle routing beats fixed direct, plan-first, and tool-first policies at matched CPU cost. Support: task needs and action risks vary. Refutation: superficial heuristics may solve routing. CPU path: Phi/Nemotron/Granite or Llama with deterministic local tools. Metrics: utility, selective risk, CPU-seconds. Kill: heuristic router or fixed policy matches.
27. **The planning tax in small agents.** Hypothesis: planning helps only above a measurable decomposition threshold and otherwise amplifies errors. Support: small models condition on flawed plans. Refutation: result may be prompt-template specific. CPU path: matched direct/plan/ReAct policies. Metrics: accuracy delta by complexity and latency. Kill: no stable crossover across models.
28. **Executable feedback versus extra reasoning.** Hypothesis: one execute-observe-repair cycle yields more gain than matched extra generation. Support: tools provide grounded state. Refutation: execution supplies privileged information, making comparison trivial. CPU path: deterministic tools and matched token/time controls. Metrics: gain/CPU-second and repair localization. Kill: textual feedback or resampling matches.
29. **Risk-calibrated tool routing and clarification.** Hypothesis: selective policies can reduce unnecessary, invalid, and harmful calls with modest coverage loss. Support: agent actions have asymmetric costs. Refutation: confidence may fail under shift. CPU path: ambiguous/irrelevant/impossible tasks. Metrics: coverage-risk and harmful-call rate. Kill: thresholds fail OOD.
30. **Diagnosis-conditioned bounded repair.** Hypothesis: one or two typed repairs capture most recoverable failures and dominate open-ended retry. Support: errors have distinct repair classes. Refutation: gains may be mere resampling. CPU path: injected schema/runtime/semantic errors. Metrics: repair accuracy and marginal CPU cost. Kill: raw-error retry is equally effective.
## Hard-gate filtering (30 → 20)

Hard gates: central ML contribution; no API/GPU requirement; public reproducibility; multiple-model/task potential; measurable quality–cost or risk trade-off; defensible novelty beyond formatting/benchmarking; a useful negative outcome.

### Passed (20)

| Candidate | Pass rationale |
|---:|---|
| 1 | General adaptive-compute question with measurable stopping regret. |
| 2 | Joint two-axis compute allocation is stronger than ordinary early exit. |
| 3 | Selective verification supports matched-budget causal tests. |
| 4 | Adaptive speculative inference is measurable, though engineering-heavy. |
| 5 | Calibration under shift is fundamental and CPU-feasible. |
| 6 | Demonstration selection remains relevant and locally testable. |
| 9 | KV eviction plus correctness alarm has clear quality-memory trade-off. |
| 10 | Token-role-aware cache precision is an ML allocation problem. |
| 11 | Structured state versus transcript memory directly tests agent design. |
| 13 | Single-pass semantic uncertainty could replace expensive sampling. |
| 14 | Evidence-sufficiency routing connects hallucination to agent control. |
| 16 | Risk-controlled outputs provide formal selective-evaluation framing. |
| 17 | Query-conditioned VLM token pruning is technical and falsifiable. |
| 19 | Cross-modal early exit tests input-dependent multimodal compute. |
| 21 | Causal layer study could explain stage-wise tool-use failures. |
| 24 | Tool-result faithfulness is important and mechanistically testable. |
| 26 | Full adaptive agent lifecycle is broad, novel enough, and CPU-feasible. |
| 27 | Planning-tax crossover is a crisp scientific phenomenon. |
| 28 | Matched-compute execution versus reasoning is timely and decisive. |
| 30 | Bounded diagnosis-conditioned repair is testable and practical. |

### Rejected (10)

| Candidate | Failed gate and reason |
|---:|---|
| 7 | Broad memory idea is crowded by Reflexion/MemGPT and vulnerable to contamination; weaker novelty than #11. |
| 8 | Micro-curriculum generation is token-expensive and too close to decomposition/test-time reasoning literature. |
| 12 | Memory-value predictor risks becoming a small router ablation; #11 offers a clearer scientific comparison. |
| 15 | Paraphrase stability is heavily studied and may conflate correctness with harmless wording variation. |
| 18 | VLM refusal evaluation is useful but likely underpowered without expensive multimodal breadth. |
| 20 | Diagnostic-only contribution lacks a method and risks benchmark-paper positioning. |
| 22 | Knowledge/tool representation competition is difficult to identify causally across differently trained families. |
| 23 | Planning-degradation mechanism is retained as the mechanistic component of stronger candidate #27, not standalone. |
| 25 | Compression/quantization phase transitions are backend-sensitive and expensive to control convincingly. |
| 29 | Risk-calibrated routing is absorbed into broader candidate #26; standalone novelty is weaker. |

No pass decision relied on existing intent-classification code or results.
## Twenty adversarial ranking loops

| Loop | Stress test | Top three after loop | Main change / refutation |
|---:|---|---|---|
| 1 | Constraint compliance | 26, 28, 27 | All three are local, API-free, CPU-testable; VLM candidates lose feasibility points. |
| 2 | Novelty against prior work | 26, 27, 28 | #28 is penalized because CRITIC/Reflexion already show feedback helps; matched compute must carry novelty. |
| 3 | Field importance | 26, 28, 14 | Reliable local agents affect deployment broadly; narrow kernel/cache topics fall. |
| 4 | Falsifiability | 27, 28, 26 | #27 has the crispest crossover hypothesis; #26 needs predeclared utility and routing baselines. |
| 5 | Three-day CPU feasibility | 30, 27, 26 | Typed error injection makes #30 fastest to pilot; #26 remains feasible with a small tool suite. |
| 6 | Prior-art crowding | 26, 27, 14 | Toolformer blocks broad call/selection novelty; #26 survives by covering six-way lifecycle control. |
| 7 | Baseline strength | 28, 26, 27 | #28 supports clean matched-token/time baselines; all three must include heuristic routers. |
| 8 | Statistical power | 27, 26, 30 | Per-item direct-versus-plan pairing makes #27 efficient; rare repair classes hurt #30 power. |
| 9 | Cross-model generality | 26, 28, 27 | Lifecycle actions are architecture-independent; mechanistic #21/#24 risk family-specificity. |
| 10 | Cross-task generality | 26, 14, 28 | #26 spans QA, calculation, retrieval, SQL, and ambiguous requests. |
| 11 | “Just engineering” attack | 27, 26, 21 | #27 gains via a scientific phenomenon; #26 must model value/risk, not implement a workflow. |
| 12 | “Just an ablation” attack | 26, 27, 28 | #26 offers a policy-learning/control contribution; #30 is vulnerable as a retry ablation. |
| 13 | “Use a larger model” attack | 26, 27, 11 | The question concerns bounded local agents; #11 offers capacity-sensitive memory evidence. |
| 14 | Hidden compute dependency | 14, 26, 27 | #14 is cheapest; #17/#19 lose points for multimodal preprocessing and model breadth. |
| 15 | Benchmark contamination | 30, 26, 27 | Synthetic typed failures favor #30; #26 requires cue-balanced, paraphrased, private-template test splits. |
| 16 | Reproducibility | 30, 27, 26 | Deterministic tools and frozen policies are reproducible; external web APIs are excluded. |
| 17 | Mechanistic depth | 21, 27, 26 | #21 leads mechanistically, but probe-causality risk prevents overall leadership. |
| 18 | Eight-page narrative | 26, 27, 28 | #26 unifies routing, planning tax, feedback, risk, and stopping around one lifecycle policy. |
| 19 | Value if headline fails | 27, 26, 28 | A null result still maps when planning harms; #26 still yields a stage-wise capability boundary. |
| 20 | Weighted expected value | **26**, **28**, **27** | #26 wins on breadth and impact; #28 is cleanest method comparison; #27 is strongest phenomenon-first fallback. |

### Final top ten

Weights: novelty 30%, acceptance probability 25%, impact 20%, CPU feasibility 15%, execution speed 10%. Scores are disciplined planning estimates, not empirical results.

| Rank | Candidate | Topic | Score / 10 | Primary residual risk |
|---:|---:|---|---:|---|
| 1 | **26** | Adaptive answer–clarify–plan–act–repair–abstain policy | **8.95** | Router gains may collapse to superficial heuristics. |
| 2 | **28** | Executable feedback versus extra reasoning | **8.65** | Execution provides privileged information. |
| 3 | **27** | Planning tax in small agents | **8.35** | Crossover may depend on prompt template. |
| 4 | 14 | Evidence-sufficiency routing | 8.05 | Lexical cues may trivialize decisions. |
| 5 | 30 | Diagnosis-conditioned bounded repair | 7.95 | Could reduce to resampling/error-message exposure. |
| 6 | 11 | Structured state versus transcript memory | 7.75 | State extraction may discard essential nuance. |
| 7 | 21 | Layer emergence of tool affordances | 7.65 | Probe information may not be used causally. |
| 8 | 3 | Confidence-triggered self-verification | 7.55 | Confidence calibration may fail under shift. |
| 9 | 17 | Question-conditioned VLM token pruning | 7.35 | CPU VLM breadth may be insufficient. |
| 10 | 24 | Tool-result faithfulness circuits | 7.30 | Stable localized circuits may not exist. |
## Top-three experiment blueprints

### Finalist 1 — Candidate 26: adaptive lifecycle control for Smol Agents

**Working title:** *Small Models, Bounded Agency: Risk-Calibrated Allocation of Agentic Compute on CPUs*

**Thesis:** Under a fixed CPU budget, a calibrated policy choosing among **answer, clarify, plan, act, repair, and abstain** outperforms fixed direct, plan-first, and tool-first policies because small-model failures occur at different control boundaries.

**Method:** Fit a lightweight value/risk controller on frozen-model signals (token probabilities where available, answer/tool-call entropy, prompt embeddings, schema validity, and execution status). The controller chooses an action and enforces a bounded transition graph; it cannot loop indefinitely. Compare a simple logistic/GBDT controller with rule-based and oracle controllers so the paper is about allocation, not neural-router decoration.

**Models:** Nemotron-Mini-4B-Instruct (verified tool-specialized positive control); Phi-3.5-mini-instruct (general-instruct contrast); Granite 3.3 2B Instruct or Llama 3.2 3B Instruct after official checkpoint/template verification. GPT-2 is optional historical control only.

**Tasks/tools:** Local deterministic calculator, unit conversion, date arithmetic, SQLite over a frozen database, and retrieval over a frozen public corpus. Mix direct-answer, tool-needed, ambiguous, irrelevant, malformed, impossible, and multi-step examples. Use selected BFCL-style single-call categories where local reproducibility permits; report benchmark version/commit.

**Baselines:** direct answer; always tool; always plan then act; ReAct-style bounded loop; heuristic keyword router; confidence-threshold router; random policy; oracle action policy; proposed controller.

**Metrics:** stage-wise accuracy; end-task exact/F1; unnecessary/missed/invalid/harmful call rates; clarification utility; abstention coverage-risk; repair success; tool-result faithfulness; P50/P95 wall time; generated tokens; peak RAM; accuracy and utility per CPU-second. Predeclare utility coefficients and include coefficient sensitivity.

**Ablations:** remove each action; remove risk features; remove execution state; rules versus learned controller; one versus two repairs; constrained versus unconstrained decoding; official versus normalized tool templates; in-domain versus cue-balanced OOD prompts; quantized versus higher-precision subset.

**CPU budget:** pilot: 3 models × 500 items × 5 policies, greedy decoding, roughly 7,500 runs. Full: 3 models × 2,000–3,000 items × focused policies; cache deterministic tool outputs; cap plan/repair tokens. Report actual runtime rather than promising an estimate.

**Three-day pilot:** Day 1 verify three checkpoints/templates and implement five deterministic tools plus 150 seed tasks. Day 2 run direct, always-tool, always-plan, heuristic, and oracle policies. Day 3 fit/evaluate the lightweight controller on 500 cue-balanced examples and produce stage-wise/utility curves.

**Continue criterion:** adaptive policy beats best fixed and heuristic policy across at least two model families; improvement survives cue-balanced OOD prompts; gains include final-answer quality or risk, not merely JSON validity; controller overhead preserves utility per CPU-second.

**Reviewer attack:** “This is orchestration engineering.” **Defense:** formulate a constrained risk-sensitive policy-allocation problem, compare learned/rule/oracle policies, identify action-specific capability boundaries, and show cross-family generalization under matched budgets.

**Strongest failure mode:** surface cues make routing trivial. Include adversarial paraphrases, numerically worded no-tool tasks, implicit tool-needed tasks, and held-out tool descriptions.

---

### Finalist 2 — Candidate 28: executable feedback versus extra reasoning

**Working title:** *Execute or Think Longer? Test-Time Compute Allocation for Small Tool-Using Models*

**Thesis:** For small local agents, one bounded execute–observe–repair cycle yields more task improvement per CPU-second than spending the same budget on additional unconstrained reasoning—but only when feedback is valid, relevant, and actionable.

**Method:** Equalize three budgets separately: generated tokens, wall-clock time, and measured CPU time. Compare direct response, longer chain-of-thought/plan, self-consistency, textual verifier feedback, real tool execution feedback, noisy execution feedback, and execute-plus-one-repair. Decompose gains by invocation, argument, runtime, interpretation, and synthesis failures.

**Models/tools:** Same three primary model families and five deterministic tools as finalist 1. Use tasks with answerable-by-reasoning, tool-essential, tool-helpful, and tool-useless strata.

**Baselines:** matched extra tokens; matched additional samples; raw error retry; structured diagnosis; textual ground-truth-equivalent feedback; oracle execution result; no feedback.

**Metrics:** accuracy gain per token/second; repair success; calibration; failure-stage transitions; robustness to noisy/misleading tool output; result faithfulness.

**Ablations:** information-equivalent text versus execution; valid/noisy/stale feedback; one/two repairs; schema versus semantic errors; deterministic versus stochastic decoding.

**Three-day pilot:** 3 models × 300 examples × 6 conditions. Continue if execution feedback beats both token- and CPU-matched reasoning in at least two families and the advantage persists against information-equivalent textual feedback.

**Reviewer attack:** “Of course external information beats thinking.” **Defense:** include information-equivalent text, tasks where tools add no information, and failure-stage localization. If those controls erase the result, kill the topic.

**Strongest failure mode:** the result is a tautology about privileged information rather than test-time allocation.

---

### Finalist 3 — Candidate 27: the planning tax

**Working title:** *The Planning Tax: When Explicit Plans Harm Small Language-Model Agents*

**Thesis:** Small agents exhibit a reproducible complexity threshold: explicit planning helps above the threshold but harms below it or when early plan errors become high-confidence context. Adaptive planning avoids this tax.

**Method:** Construct tasks with controlled step depth, branching, distractors, tool count, and recoverability. Compare direct action, plan-then-act, bounded ReAct, oracle plan, corrupted plan, and adaptive plan/no-plan routing. Estimate the crossover point and test whether early-plan correctness mediates final success.

**Models/tasks:** Same three families; arithmetic, date, database, retrieval, and composed multi-tool tasks with programmatically controlled complexity.

**Metrics:** direct-minus-plan accuracy; success by complexity; plan correctness; error amplification; latency; tokens; calibration of the plan/no-plan decision.

**Ablations:** hidden versus visible plan; short versus verbose plan; correct versus controlled-corrupted plan; official templates; tool count; context length; router features; resampling at matched compute.

**Three-day pilot:** Generate 600 deterministic tasks across 1–4 steps; run direct and plan-first on three models; test for a replicated interaction between task complexity and policy. Continue only if the crossover direction holds in at least two models and survives prompt-template changes.

**Reviewer attack:** “Planning quality, not planning itself, explains everything.” **Defense:** include oracle and controlled-corrupted plans, mediation analysis, and matched-length non-plan controls.

**Strongest failure mode:** no stable crossover—planning is uniformly good/bad or entirely template-dependent.

---

## Final selection

**Winner:** Candidate 26, adaptive lifecycle control. It contains the broadest reusable principle and can incorporate candidates 27 and 28 as two mechanism-focused analyses without turning the paper into a kitchen sink: planning and repair are actions governed by the same bounded policy.

**Fallback:** Candidate 27, the planning tax. It has the cleanest phenomenon-first narrative and remains publishable if a full six-action controller proves too broad.

**Scope guard:** Candidate 26 must not claim tool invocation, planning, feedback, or memory individually as novel. The novelty claim is risk-calibrated allocation among heterogeneous agentic actions under fixed CPU budgets, supported by stage-wise cross-model evidence.
## Final recommendation

### Winner: Candidate 26

# **Small Models, Bounded Agency: Risk-Calibrated Allocation of Agentic Compute on CPUs**

The paper should test whether a bounded controller can choose among answer, clarification, planning, tool execution, repair, and abstention more effectively than fixed agent loops under matched CPU budgets.

### Why it wins

- **Novelty:** the complete six-way lifecycle policy is less occupied than tool calling, planning, or feedback individually.
- **Impact:** the result applies to local assistants, edge agents, private enterprise agents, and constrained deployments.
- **Scientific value:** even failure yields a stage-wise map of where 1–4B agents break.
- **Feasibility:** deterministic tools and three local model families avoid API/GPU dependence.
- **Narrative:** candidates 27 and 28 become mechanism studies—planning is one optional action and execution repair is another.

### Lower-risk fallback

# **The Planning Tax: When Explicit Plans Harm Small Language-Model Agents**

This fallback has a narrower, cleaner claim and a cheap paired experiment. Use it if the six-action controller is too broad or if routing gains collapse to heuristics.

## Immediate experiment

Run the three-day falsification pilot before building a full framework:

1. Verify official templates/licenses for Nemotron, Phi, and Granite or Llama.
2. Build five deterministic local tools: calculator, unit conversion, date arithmetic, SQLite, and frozen-corpus retrieval.
3. Create 500 cue-balanced cases: direct-answer, explicit tool-needed, implicit tool-needed, ambiguous, impossible, malformed, and multi-step.
4. Compare direct, always-tool, always-plan, heuristic routing, and oracle routing.
5. Fit the smallest sensible controller only after fixed-policy results exist.
6. Stop if heuristics match the controller, gains are formatting-only, or the effect fails across two families.

## Confidence assessment

| Finding | Confidence | Reason |
|---|---|---|
| Nemotron Mini 4B is explicitly function-calling optimized | High | Direct quotation from locally inspected official model card and documented serialization |
| Phi-3.5 Mini is 3.8B and intended for constrained/latency-sensitive settings | High | Direct official-card quotations |
| Phi-3.5 card does not establish native tool support | Medium-high | No tool schema found in inspected card; absence claims remain version-sensitive |
| Tool-call decision/selection/arguments are not individually novel | High | Toolformer abstract states these capabilities directly |
| Interleaved reasoning and action are established | High | ReAct abstract states the formulation directly |
| Adaptive six-way lifecycle control is a defensible gap | Medium | Strong synthesis of audited prior work, but a 2026–2027 literature refresh is still required |
| Candidate 26 will empirically outperform fixed policies | Unknown | This is the hypothesis to test, not a result |
| Granite/Llama/Phi-4 native tool-template details | Unknown | Current official pages were not re-fetched in the child research environment |
| CPU runtime for the proposed matrix | Unknown | Must be measured on the specified machine and quantization stack |

## Uncertainties and required re-verification

1. Re-fetch the current Phi-4 Mini card and identify exact official function-call syntax.
2. Inspect each Llama 3.2 checkpoint’s `tokenizer_config.json` and `chat_template`; do not inherit claims from another Llama generation.
3. Verify Granite 3.3 2B exact parameter count, license, and tool template.
4. Distinguish Gemma SDK-side formatting from checkpoint-native tool training.
5. Freeze benchmark commits and parsers; leaderboards change independently of papers.
6. Test for benchmark contamination using paraphrased and programmatically generated private-template splits.
7. Treat quantized community files as separate artifacts from official checkpoints.

## Retractions and corrections

- **Retracted:** any earlier implication that Phi-3.5 Mini has verified native tool use. The inspected official card did not establish it.
- **Retracted:** any blanket statement that Llama 3.2, Granite 3.3, or Gemma 3 natively supports tool calling at every small checkpoint. Exact checkpoint evidence remains unverified here.
- **Corrected:** GPT-2 is not a primary Smol-Agent baseline. It is a historical negative control because the original release is neither instruction-tuned nor tool-aware.
- **Corrected:** “US-built” is used only as an organization-origin filter; it does not imply geographic purity of contributors, data, or compute.
- **Not claimed:** that candidate 26 works. The report selects the hypothesis with the best expected research value; only the falsification pilot can justify proceeding.

## Citation lookup keys

Use official BibTeX before submission; do not reconstruct full author lists from memory.

- Phi-3: `arXiv:2404.14219`
- Llama 3: `arXiv:2407.21783`
- Granite 3.0: `arXiv:2409.16013`
- Gemma 3: `arXiv:2503.19786`
- OpenELM: `arXiv:2404.14619`
- ReAct: `arXiv:2210.03629`, ICLR 2023
- Toolformer: `arXiv:2302.04761`, NeurIPS 2023
- AgentBench: `arXiv:2308.03688`, ICLR 2024
- BFCL: `arXiv:2402.04253`
- tau-bench: `arXiv:2406.12045`
- ToolSandbox: `arXiv:2408.04682`
- TinyAgent: `arXiv:2409.00608`
- Reflexion: `arXiv:2303.11366`
- CRITIC: `arXiv:2305.11738`
- Test-time compute scaling: `arXiv:2408.03314`
- Small Language Models are the Future of Agentic AI: `arXiv:2506.02153`

---

**Bottom line:** Smol Agents is a valid family and currently contains the winner, but the contribution must be **bounded, risk-calibrated allocation of agentic actions**, not another function-calling demo with suspiciously attractive JSON.
