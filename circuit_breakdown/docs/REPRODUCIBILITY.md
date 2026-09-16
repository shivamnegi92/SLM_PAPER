# Post-Run Audit and Reproduction

Updated 2026-09-07. This guide covers the completed local study, not a new
experiment. See [the frozen protocol](../STUDY_PROTOCOL.md),
[manuscript](../paper/MANUSCRIPT.md), and [detailed tables](../paper/RESULTS_DETAILS.md).

## What Is Reproducible Here

The [consistency audit](../results/postrun_audit_v1/consistency.json) checks:

- Six paired summaries rebuilt from 2,700 method/example records, with rate
  denominators, intervals and comparisons. The same semantic examples recur
  across methods/models; they are not 2,700 independent examples.
- Local tokenizer reconstruction of sample IDs, target IDs and token-input
  hashes, plus per-case checkpoint equality and frozen source/configuration checks.
- Recorded per-layer budgets, basis diagnostics, calibration selection,
  prediction-derived outcome flags and terminal optimization trace agreement.
- Nine capability runs against the 240-item HellaSwag and 200-item ARC caches:
  choice-score argmax, paired accuracy statistics, five-window text statistics,
  zero controls, equal-norm controls and saved edit/checkpoint metadata.
- Three head-result paired comparisons and unablated receiver-clamp no-ops.
- Nine baseline-only runs' flag counts, rates and intervals.
- Equality of the final JSON snapshot with its source artifacts. Its byte
  checksum binds the detailed report to the audited snapshot.

No full model is loaded by the audit. It loads local tokenizers and small saved
edit tensors on CPU. It does not change the frozen experiment source or results.

## Recheck Without Model Runs

Run from `circuit_breakdown`, using the existing Python environment and caches.
Use output names that do not already exist; the tools refuse overwrites.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python src/audit_study.py \
  --snapshot results/final_validated_evidence_v1 \
  --output results/my_recheck/consistency.json
.venv/bin/python src/report_study.py \
  --snapshot results/final_validated_evidence_v1 \
  --audit results/my_recheck/consistency.json \
  --output paper/my_recheck_details.md
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Detailed Markdown uses links relative to the paper directory. Keep regenerated
reports in that directory or adjust links in a separate distribution copy.
The original queue/finalizer is not an idempotent post-run command: its snapshot
and supplement destinations use exclusive creation. Do not restart the queue
merely to regenerate tables.

## Checksums and Versions

The [reproduction manifest](../results/postrun_audit_v1/reproducibility/manifest.json)
records content hashes and sizes for code, tests, public input caches, saved
results, paper documents, and active root model assets. The
[version pins](../results/postrun_audit_v1/reproducibility/requirements.lock.txt)
record the installed environment at capture time, without credentials, index
URLs or machine-specific executable paths.

```bash
.venv/bin/python src/capture_reproducibility.py \
  --verify results/postrun_audit_v1/reproducibility/manifest.json

# For a later capture, choose a new directory.
.venv/bin/python src/capture_reproducibility.py \
  --output-dir results/my_later_capture
```

Weight hashing is streamed and includes `.bin` as well as `.safetensors`.
It performs disk reads, not inference. Model assets are expected in the sibling
`llama-3.2-3b`, `phi-3.5-mini`, and `nemotron-mini-4b` directories. Redundant
conversion archives and download caches are excluded from the active model
inventory. No models or dependencies are downloaded or installed by these tools.

The version list is a capture, not a tested clean-room installation recipe.
Dependency availability and behavior on another platform are not guaranteed.
Only limited dependency versions were recorded in the original experiment;
capturing the environment later cannot prove the entire historical environment.

## Irrecoverable Evidence Limits

1. Original model fingerprints recorded configuration and sizes of root
   `.safetensors` files, not weight content. Nemotron's current `.bin` file was
   outside that size inventory. The new hashes identify today's local files,
   not a historically authenticated remote revision or execution.
2. Head artifacts have manifest/input identities but no run-time model/source
   content hashes. Newly captured source hashes cannot repair that omission.
3. Baseline-only harder-task files contain flags, not item identities. Their
   rates can be checked, but original ordering cannot independently be proven.
4. Full original logits and activations are not stored. The audit cannot replay
   their forward computation or independently reconstruct every continuous delta
   and faithfulness numerator from the saved summaries.
5. Recorded finite-budget traces and basis diagnostics are not convergence
   proofs. The scalar continuous statistics are checked against saved records.
6. A successful audit does not establish capability preservation: active bounds
   were met on HellaSwag for 3/9 edits, ARC for 0/9, and fixed text windows for
   9/9. These are reused-input, selected-edit results, not population guarantees.

## Sharing and New Experiments

The original local artifacts may contain identifying absolute paths. Make an
anonymized release copy, verify its numerical equivalence, and record new
checksums before submission. Do not modify the original snapshot to anonymize
it, redistribute model weights without checking licenses, or publish automatically.

A reusable-edit experiment or harder-task investigation needs a new protocol,
fresh development/test separation and distinct output names. Do not modify
historical comparison hashes to force compatibility. See
[the submission checklist](SUBMISSION_CHECKLIST.md) for remaining author-facing work.