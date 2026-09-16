# Localizing Task Information Is Not the Same as Controlling an Answer

Research manuscript draft, updated 2026-09-07. Working target: ICLR 2027.
The declared experiment matrix and post-run consistency audit are complete.
This is not a submission or an acceptance forecast; author review and
submission formatting remain separate from completing the measurements.

## Abstract

Activation patching can identify internal representations that influence an
answer without establishing that a low-rank approximation of those
representations supports efficient control. We study this distinction in three
locally evaluated small language-model checkpoints. A methodological audit of
the initial experiments exposes three separate problems: answer-extraction
shortcuts in nominal tracking tasks, an unscored-token capability intervention,
and a disruption metric that conflates pre-existing errors with damage. We
introduce a shared-tokenizer manifest, baseline-conditioned intervention
metrics, scored-prefix capability exposure, and a simulator-backed swap task.
Full-space and rank-constrained edits are compared under train-normalized
per-layer budgets and development-selected optimization settings. Head
localization uses separate discovery and verification examples, with a bounded
receiver-clamp check for downstream compensation. On two answer-extraction
tasks with 150 test pairs per task, full-space override reaches 100% in all
six checkpoint/task comparisons, versus 8.7%-56.0% for the selected rank8
subspace and 0% for its matched random control. Each edit is optimized separately
with the target supplied. All checkpoints fail the harder swap task's competence
gate, and the declared capability-preservation bounds are not all established.
The contribution is this controlled evaluation and its measurement audit, not
a new general principle about localization, a reusable editor, or a safety claim.

## 1. Introduction

A representation can carry information about an answer without providing an
effective low-dimensional control direction. Conversely, an edit that changes
the answer need not identify the original computation. Distinguishing these
statements requires more than showing either a successful patch or a failed
steering vector. The edit budget, optimization objective, task construction,
and measurement protocol all affect the observed result.

Our initial experiments localized answer-sensitive residual sites and late
attention-head outputs in three small-model checkpoints. Larger follow-ups
suggested different full-space versus tracking-subspace gaps across models.
Those observations motivate, rather than settle, the study. The legacy tables
used small seed-specific samples and did not isolate checkpoint architecture
from all tokenizer, layer, training-data, and optimization differences.

We therefore ask a narrower question: under a specified task distribution and
validated editing procedure, how does restricting an edit to a learned
activation-difference subspace affect target override and collateral damage?
We retain negative and mixed outcomes, including failure of a model to solve
the harder task well enough for a causal-control interpretation.

## 2. Related Work

Wang et al. (2022) analyze indirect object identification in GPT-2 small using
causal interventions and evaluate the explanation through faithfulness,
completeness, and minimality. Their work motivates component-level controls;
our head subset and receiver diagnostic do not establish the same degree of
end-to-end circuit explanation.

Schaeffer, Miranda, and Koyejo (2023) demonstrate that nonlinear or discontinuous
metrics can create apparent discontinuities absent under alternative metrics.
We use both top-1 success and continuous target/margin measurements. Applying
this measurement principle to intervention experiments is not by itself a new
theoretical result, and a positive target-logit change is not synonymous with
a successful or harmless edit.

Sharkey et al. (2025) survey open problems in mechanistic interpretability,
including limitations of existing methods and their use toward scientific and
engineering goals. This study addresses measurement validity for one local
intervention setting. It does not claim to solve the broader problem of
interpretability-guided control or to establish novelty from an open-problem
listing alone.

Makelov, Lange, and Nanda (2023) show that subspace interventions can change
behavior without faithfully identifying the feature's original mechanism.
They also describe a success case and the additional evidence needed for
faithfulness; this is not a blanket rejection of subspace patching. The broad
distinction between manipulating behavior and identifying a mechanism is thus
prior work. Our contribution is the specific small-model comparison under a
frozen evaluation protocol, with explicit measurement failures and controls.

## 3. Data and Experimental Separation

The study uses existing local Llama-3.2-3B, Phi-3.5-mini, and Nemotron-Mini-4B
checkpoints. Configuration and tokenizer fingerprints are recorded. Directory
names and file-size hashes are not evidence of a particular remote model
revision; exact historical checkpoint provenance remains a reproducibility
limitation. In particular, the original size fingerprint enumerates root
`.safetensors` files and does not cover Nemotron's current `pytorch_model.bin`.
Post-run content checksums identify the locally available files for future
reproduction; they cannot retrospectively authenticate earlier model runs.
All data is public or generated from generic templates.

The legacy `intermediate` task queries the city preceding the final city in an
explicit route. The `transfer` task queries the last recipient of a named
object, followed by a distracting transfer. Both defeat a naive last-name rule,
but a query-aware extraction rule solves them. The changed input word is also
the requested answer. Accordingly, these tasks test controlled answer-sensitive
computation, not proof of multistep state reasoning.

The new `container_swap` task changes an initial object location and composes
at least three swaps. A deterministic simulator supplies both counterfactual
answers. Each queried object changes location at least twice; its final
location differs from its initial edited token. A final irrelevant swap
prevents a simple last-mentioned-box answer. Test wording is held out from
training, development, and demonstrations. This remains a synthetic task.

The frozen manifest contains three disjoint seed blocks per task. Each has 64
training pairs, 24 development pairs, 50 test pairs, and three separate
few-shot demonstrations. All three tokenizers must accept the same semantic
pairs and a single aligned token edit. No prompt is reused across splits or
seed blocks. Test examples are not used to select learning rates or simplify
the task. Intervention claims require at least 80% clean and counterfactual
accuracy on the seed-0 development block; otherwise the model receives
baseline-only evaluation. The four-example learning-rate calibration also
uses seed-0 development data, not three independent calibration replicates.

## 4. Interventions

Let $x_c$ and $x_k$ denote the clean and counterfactual prompts, with respective
targets $t_c$ and $t_k$. The counterfactual target is a valid answer to its own
prompt. Steering toward $t_c$ on $x_k$ is therefore target-informed override,
not automatically repair of a naturally wrong model response.

For each edited layer, a basis is fitted from training residual differences at
the aligned edited position. SVD numerical rank is checked explicitly. The
tracking condition uses eight directions. A matched random control uses eight
orthonormal directions projected outside that selected basis. This control is
a small random slice of the complement, not the entire complement of all task
information.

Layer-output depth fractions match Llama layers [18,20,22,24] to [21,23,25,28]
in the 32-layer models. Each layer's edit budget is 0.30 times its mean training
residual norm at the target position. Full-space, tracking, and control methods
use the same target cross-entropy objective and 32 Adam steps. Each receives
the same development learning-rate grid, scored on four seed-0 development
examples. The rank correction to learning rate
is only an approximate step-size adjustment; optimization traces and sensitivity
to the finite step budget limit any interpretation of failure as impossibility.

## 5. Measurements

Target override is the fraction of counterfactual prompts whose final
full-vocabulary argmax equals $t_c$. Newly achieved override conditions on the
unedited counterfactual prediction not already being $t_c$. We separately
record full target probability, target versus best-other-token margin, two-way
target probability, and third-token outputs.

Same-sign damage applies the learned positive edit to $x_c$ and conditions on
the unedited clean prediction being correct. Negative-edit disruption applies
the opposite edit and uses the same baseline-correct population. The older
unconditional negative-edit error rate is retained only for legacy comparisons.
An empty eligible population produces an undefined rate, not zero damage.

For activation patching, faithfulness is the recovered clean/counterfactual
logit difference divided by the original difference. Nonpositive or nearly
zero denominators are recorded as invalid rather than clipped into a favorable
score. A 100% logit-gap recovery is not 100% general question-answering accuracy.

Binary rates use Wilson intervals; binary method differences use paired
gain/loss bounds with a Bonferroni adjustment and exact McNemar tests.
Continuous contrasts use seed-stratified paired bootstrap intervals and paired
sign permutations with the Monte Carlo plus-one correction. Method samples
are joined on identities and token-input hashes. Inference is conditional on
the observed seed blocks. Equal rates and $p=1$ do not establish equivalence.

## 6. Capability and Mechanism Checks

The original capability evaluator edited the final token of a teacher-forced
sequence, while scoring only earlier token predictions. A causal decoder cannot
propagate that edit backward to scored positions. The legacy zero-change result
is thus an evaluation blind spot, not evidence of safe deployment.

The corrected evaluator edits the last context token before HellaSwag
(Zellers et al., 2019) or ARC-Easy (Clark et al., 2018) answer continuations.
Text evaluation edits a fixed prefix and scores
subsequent tokens in five preselected, non-overlapping windows. Baseline,
active-prefix, equal-norm random, zero, and global exposure are distinguished.
These are exposure tests on unrelated inputs, not an entity-trigger deployment
policy. Global exposure is not assumed to bound all single-position damage.

Head attribution ranks components on training examples. Top-head patching is
verified on disjoint test examples against five random sets with matched depth
and size. A source-head ablation is combined with restoring selected downstream
head outputs to their unablated baseline. The difference relative to free
downstream computation, compared with random receiver clamps, is a bounded
diagnostic for compensation. It is not an exhaustive backup-head search or a
complete causal graph.

## 7. Results and Evidence Status

<!-- validated-results:start -->
The declared local experiment matrix is complete. This records
execution, not universal control, equivalence, or publication readiness.
The summary is in [the validated results supplement](VALIDATED_RESULTS.md);
per-seed tables, controls, intervals and audit limitations are in
[the detailed results](RESULTS_DETAILS.md).

**llama-3.2-3b / intermediate:** full-space override is 100.0% and tracking-subspace override is 10.0%, on 150 unique test pairs across 3 seeds. The paired difference is 90.0 percentage points with 95% interval [79.9, 94.3] percentage points. The interval supports a full-space advantage in this tested regime. Same-sign damage is reported separately from negative-edit disruption.

**llama-3.2-3b / transfer:** full-space override is 100.0% and tracking-subspace override is 25.3%, on 150 unique test pairs across 3 seeds. The paired difference is 74.7 percentage points with 95% interval [62.8, 81.7] percentage points. The interval supports a full-space advantage in this tested regime. Same-sign damage is reported separately from negative-edit disruption.

**llama-3.2-3b / container_swap:** the development competence gate was not met (clean 12.5%, counterfactual 8.3%). Baseline-only test outcomes are retained; no causal-control conclusion is drawn.

**phi-3.5-mini / intermediate:** full-space override is 100.0% and tracking-subspace override is 8.7%, on 150 unique test pairs across 3 seeds. The paired difference is 91.3 percentage points with 95% interval [81.5, 95.2] percentage points. The interval supports a full-space advantage in this tested regime. Same-sign damage is reported separately from negative-edit disruption.

**phi-3.5-mini / transfer:** full-space override is 100.0% and tracking-subspace override is 20.0%, on 150 unique test pairs across 3 seeds. The paired difference is 80.0 percentage points with 95% interval [68.5, 86.3] percentage points. The interval supports a full-space advantage in this tested regime. Same-sign damage is reported separately from negative-edit disruption.

**phi-3.5-mini / container_swap:** the development competence gate was not met (clean 12.5%, counterfactual 41.7%). Baseline-only test outcomes are retained; no causal-control conclusion is drawn.

**nemotron-mini-4b / intermediate:** full-space override is 100.0% and tracking-subspace override is 12.7%, on 150 unique test pairs across 3 seeds. The paired difference is 87.3 percentage points with 95% interval [76.8, 92.2] percentage points. The interval supports a full-space advantage in this tested regime. Same-sign damage is reported separately from negative-edit disruption.

**nemotron-mini-4b / transfer:** full-space override is 100.0% and tracking-subspace override is 56.0%, on 150 unique test pairs across 3 seeds. The paired difference is 44.0 percentage points with 95% interval [32.0, 53.1] percentage points. The interval supports a full-space advantage in this tested regime. Same-sign damage is reported separately from negative-edit disruption.

**nemotron-mini-4b / container_swap:** the development competence gate was not met (clean 12.5%, counterfactual 20.8%). Baseline-only test outcomes are retained; no causal-control conclusion is drawn.

The declared capability-preservation criteria were not all established. Report the observed costs and uncertainty rather than claiming negligible degradation. A failed bound can reflect imprecision or measured damage; inspect the paired intervals.

The head results are held-out component and selected-receiver diagnostics, not an exhaustive self-repair analysis or a complete causal graph. Three checkpoints do not isolate architecture from training and tokenization differences.
<!-- validated-results:end -->

### Post-Run Consistency Audit

The CPU-only audit rebuilt the six paired summaries from 2,700 saved
method/example evaluations, checked tokenized identities against the manifest,
and verified case checkpoints, recorded budgets and prediction-derived flags.
It also rebuilt nine capability outputs and three head comparisons and matched
the final snapshot against its source files. The 2,700 evaluations reuse
semantic examples across methods and models; they are not independent examples.

For the nine active-prefix edits, the declared accuracy bound was met on
HellaSwag for three edits and on ARC-Easy for none. The five-window text-ratio
bound was met for all nine edits. These are descriptive counts over reused
benchmark inputs, not nine independent datasets. Full control tables and
head intervals are provided in [the detailed results](RESULTS_DETAILS.md).

Consistency checks cannot reconstruct unsaved full logits or establish
historical weight provenance. Baseline-only test files lack per-item identities,
and head runs lack run-time model/source content hashes. The audit therefore
does not certify complete reproducibility, optimizer convergence, or safety.

## 8. Limitations and Conclusion

Three checkpoints cannot isolate architecture from pretraining data, tuning,
tokenization, or other implementation differences. Eight basis directions do
not exhaust the causal representation. Finite-budget optimization failure
does not prove that steering is impossible. Synthetic single-token tasks and
fixed few-shot demonstrations limit generalization, even with held-out wording.
Three seeds and a small number of capability edits also limit uncertainty over
training distributions, edit selection, and real deployment inputs.

The central methodological conclusion is that localization, successful target
override, and preserved capability are distinct claims. Each needs a matching
intervention protocol, valid scored outputs, appropriate controls, and explicit
sampling assumptions. The empirical conclusion must remain no broader than the
validated tables support.

## Reproducibility Statement

The frozen protocol describes the manifest, training/development/test separation,
layer budgets and optimization procedure. The supplement provides per-seed
results, paired uncertainty and control conditions. A separate read-only auditor
checks recorded identities and statistics without loading model weights, and a
deterministic reporter regenerates the detailed tables. The
[reproduction guide](../docs/REPRODUCIBILITY.md) documents commands, post-run
file checksums, environment capture, and the historical provenance limitations.

## References

1. Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. (2022).
   [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593).
2. Schaeffer, R., Miranda, B., and Koyejo, S. (2023).
   [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004).
3. Sharkey, L., Chughtai, B., et al. (2025).
   [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496).
4. Clark, P., et al. (2018).
   [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457).
5. Makelov, A., Lange, G., and Nanda, N. (2023).
   [Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](https://arxiv.org/abs/2311.17030).
6. Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. (2019).
   [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830).