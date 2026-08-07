# COMPLIANCE — Public-Only Hard Rule

This paper workspace is **strictly public-benchmark-only**. The following are
**forbidden** anywhere in `SLM_PAPER/`:

- Any employer / company / product / team name or internal codename.
- Any internal problem framing, use-case narrative, or domain wording.
- Any internal dataset names, table names, file paths, or schema.
- Any results, latencies, or accuracies measured on internal/proprietary data.
- Any internal document, report, slide, or Confluence/Jira reference.

**Allowed:**
- Generic ML methodology (pruning, CRF BIO tagging, linear probing, rule-based
  post-processing) — these are standard techniques, not proprietary.
- Public datasets only: ATIS, SNIPS, MASSIVE, CLINC150, BANKING77 (see
  `experiments/datasets.md`).
- Public model checkpoints with clear licenses (see `experiments/datasets.md`).
- Publicly citable papers only.

**Rule of thumb:** if a sentence would only make sense to someone inside a
specific company, it does not belong here. Rewrite it as a general research claim
and prove it on a public benchmark.

All results tables in the papers start as **placeholders (TBD)** and are filled
in *only* from runs on the public datasets in `experiments/`.
