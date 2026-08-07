# ICLR 2027 — Submission Requirements Checklist

Authoritative checklist for Paper 1. Verify against the official ICLR 2027 CFP
before submitting (dates/rules can be updated by the PCs).

## Deadlines (AoE)
- [ ] **Abstract + registration:** Sep 18, 2026. Title, abstract, and full author
      list registered on OpenReview.
- [ ] **Full paper:** Sep 25, 2026.
- [ ] Backup: AISTATS 2027 full paper Oct 8, 2026.

## Formatting & structure
- [ ] **Main text <= 9 pages** at initial submission (references + appendix do NOT
      count).
- [ ] Use the **official ICLR 2027 LaTeX style template** (get it from the CFP /
      OpenReview; do not reuse an old year's `.sty`).
- [ ] **Mandatory AI-Use Statement** section disclosing significant use of AI tools
      in research ideation or writing. Does NOT count toward the page limit.
      (We will disclose AI-assisted tooling honestly here.)
- [ ] Reproducibility statement + appendix (encouraged; strengthens review).

## Double-blind anonymity (critical)
- [ ] No author names or affiliations anywhere in the PDF.
- [ ] No identifying self-citations (cite own prior work in third person; do not
      write "our previous work [X]").
- [ ] **Anonymize supplementary materials too** (code, data, appendices).
- [ ] Released code goes to an **anonymized repo** (e.g., anonymous.4open.science),
      not a personal/company GitHub, until after decisions.
- [ ] Strip metadata: PDF author fields, Git author/committer info, file paths,
      dataset/model provenance that reveals identity or organization.
- [ ] Re-run the `COMPLIANCE.md` scrub: no employer/product/team/domain wording
      (this also serves anonymity).

## OpenReview / authorship
- [ ] Every co-author has an **active OpenReview profile** BEFORE Sep 18.
- [ ] Final author list locked at abstract deadline \u2014 **no new authors after
      Sep 18**. Decide authorship now.
- [x] **1-submission limit CONFIRMED APPLICABLE.** Decision (2026-08-07): no
      intended co-author has a prior top-ML-venue publication, so the team is
      capped at **ONE ICLR 2027 submission**. Consequence: **Paper 1 gets the
      ICLR slot; Paper 2 must go to AISTATS 2027 (Oct 8) or EMNLP/Findings/a
      workshop — NOT ICLR 2027.** Do not split this work across two ICLR papers.

## Submission mechanics
- [ ] Submit via OpenReview (PDF + optional supplementary zip).
- [x] **Anonymized code release CONFIRMED (2026-08-07):** release review code via
      an anonymized repo (e.g., anonymous.4open.science), not a personal or company
      GitHub, until after decisions.
- [ ] Confirm license for released code (permissive) and that all datasets/models
      used are public with compatible licenses (see `../experiments/datasets.md`).

## Pre-submission scrub (run twice: at abstract lock and before final upload)
- [ ] `grep` the PDF/tex/appendix/code for author names, org names, and internal
      terms \u2014 zero hits.
- [ ] `git log` on any released code shows anonymized identities.
- [ ] Figures/screenshots contain no identifying paths, usernames, or watermarks.
- [ ] AI-Use Statement present and accurate.
- [ ] Page count of main text <= 9.
