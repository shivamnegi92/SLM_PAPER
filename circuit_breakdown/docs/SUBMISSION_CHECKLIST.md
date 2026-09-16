# ICLR 2027 Submission Checklist

Checked against public official pages on 2026-09-07. This is a local preparation
checklist, not an external submission, a venue endorsement or an acceptance forecast.

## Sources Consulted

- [ICLR 2027 call for papers](https://iclr.cc/Conferences/2027/CallForPapers).
- [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
- [ICLR 2027 policy on LLM use](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

## Key Extractions

From the author guidelines:

> "At the time of submission, the main text should be 9 pages or fewer."

> "Required: AI use statement"

From the AI policy:

> "Ultimately the paper's authors are responsible for the contents of their submissions."

The author guidelines and call for papers agree on the deadlines below. Sources
may change; recheck them before the actual submission.

## Verified Requirements

| Item | Requirement | Source |
|---|---|---|
| Abstract | September 18, 2026, 23:59 AoE (UTC-12); genuine abstract, not a placeholder | Call for papers; author guidelines |
| Full paper and supplements | September 25, 2026, 23:59 AoE | Call for papers; author guidelines |
| Initial main text | At most nine pages | Author guidelines, Paper formatting |
| Template | Use the [official 2027 LaTeX style files](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip) | Author guidelines |
| References and appendices | References do not count toward the main-text limit; appendices may follow references; reviewers need not read them | Author guidelines |
| Anonymity | Double blind applies to both paper and supplementary material | Author guidelines |
| AI use | Disclosure in a mandatory paper section and in the submission form; section does not count toward the page limit | AI policy; author guidelines |
| Reproducibility statement | Recommended, not mandatory; excluded from main-text limit | Author guidelines |
| Ethics statement | Recommended where appropriate, not universally mandatory; at most one page and excluded from main-text limit | Author guidelines |
| Authors | Finalize author membership before abstract deadline; no additions afterward | Author guidelines |
| Reviewing eligibility | Check the reciprocal-reviewer rules or the newcomer exemption against the actual author list | Author guidelines |

The initial nine-page rule is explicit. The guidelines contain later-stage
wording about ten pages; do not use that as permission for a ten-page initial
submission. Recheck rebuttal/camera-ready rules when relevant.

## Remaining Author Tasks

- [ ] Review the scientific claim, novelty relative to related work, all figures,
  and the [audit limitations](REPRODUCIBILITY.md). A passed consistency check is
  not an independent human scientific review.
- [ ] Convert the draft into the official initial-submission template and compile
  a PDF; verify the actual page count and readability. No submission PDF has
  been produced by this post-run audit.
- [ ] Finalize authorship and OpenReview profiles before the deadline. Profile
  moderation can take time; author status has not been inspected here.
- [ ] Resolve reciprocal-reviewer eligibility or the newcomer exemption. Do not
  infer eligibility from the existence of this local project.
- [ ] Create a separate anonymous code/artifact package. Remove identifying
  absolute paths, personal metadata and operational logs from the release copy;
  retain the original data and recompute checksums for the copy.
- [ ] Check model/dataset licenses and avoid distributing restricted weights.
- [ ] Finalize and attest the AI-use statement based on the complete work history.
- [ ] Check submission-form requirements, then submit only with explicit author
  approval. No login, upload, publishing or account changes were performed here.

## AI-Use Draft for Author Review

The following describes assistance visible in this project workflow. It is not
an author attestation and must be checked against the complete research history:

> Generative AI assistance was used for experiment planning, implementation of
> research methods and synthetic task generators, result interpretation, evidence
> auditing, literature discovery, and manuscript/table drafting. Reported
> numerical results were obtained from local model experiments and saved
> artifacts, with executable consistency checks used during analysis.

Before inserting a final statement in the paper, authors must describe the
review they actually performed and accept responsibility for the final content.
Do not copy a claim such as "we reviewed all AI-assisted work" until it is true.
Method implementation, synthetic-data generation, experiment design and result
interpretation are among the policy's required disclosure categories; this
workflow is not merely grammar correction.

## Confidence and Limits

- High confidence: dates, initial page limit, anonymity and disclosure rules
  above are directly supported by official pages retrieved on the stated date.
- Unknown: author eligibility, completed human review, license clearance for a
  release, final PDF compliance, and submission acceptance. No such claim is made.
- Verified references added to the draft: [Makelov et al. (2023)](https://arxiv.org/abs/2311.17030)
  and [Zellers et al. (2019)](https://arxiv.org/abs/1905.07830). The former discusses
  both interpretability counterexamples and a success case; it does not justify
  rejecting all subspace patching or claiming our general distinction as novel.