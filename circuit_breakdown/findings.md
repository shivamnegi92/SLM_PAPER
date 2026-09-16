# Post-Run Findings

- The declared matrix is complete; the original evidence snapshot and paper
  supplement already exist. Missing Nemotron runs are no longer pending work.
- The collector checks expected file presence and calibration completion, then
  trusts saved summaries. This is not an independent raw-record audit.
- The paired analyzer can rebuild rate summaries, denominators, intervals, and
  comparisons without loading model weights. Reuse it for consistency checks.
- The current plan mixes executed headings with stale queued/running prose.
- The human-readable supplement lacks the detailed per-seed intervention,
  control/text exposure, and head-uncertainty tables promised by the manuscript.
- Full-space override is 100% in the six eligible comparisons, with track8
  between 8.7% and 56%. This is target-informed, per-example optimization.
- All three harder-task competence gates failed. Capability preservation was
  not established across all declared checks. These remain reported outcomes.
- All six real paired summaries were independently rebuilt from 2,700 saved
  method/example records; all stored statistics matched. These are repeated
  method/model evaluations, not 2,700 independent semantic examples.
- Head artifacts contain manifest and input identities but no run-time model
  weight/source content hashes. Later checksums cannot retroactively prove
  the exact weights used by those runs.
- All nine capability outputs pass recomputation from saved choices/losses;
  active-prefix bounds: HellaSwag 3/9, ARC-Easy 0/9, text-window ratio 9/9.
  These are descriptive gate counts across reused items, not independent trials.
- Calibration competence and learning-rate selection use seed-0 development
  data; the equal learning-rate grid uses four development examples per method.
- Baseline-only hard-task files contain counts/flags, not per-item identities.
  The audit can recompute their rates but cannot retrospectively prove ordering.
- Official ICLR 2027 author guidelines (web, checked 2026-09-07):
  "At the time of submission, the main text should be 9 pages or fewer."
  "Required: AI use statement". Abstract/paper deadlines are September 18/25,
  2026 at 23:59 AoE. Source: https://iclr.cc/Conferences/2027/AuthorGuidelines
- Makelov, Lange and Nanda (2023) primary abstract explicitly says subspace
  intervention/control and mechanism attribution can diverge, while also
  describing a success case. Do not claim the general distinction as new.
  Source: https://arxiv.org/abs/2311.17030
- The original model_fingerprint only globs root *.safetensors. The active
  Nemotron directory instead contains pytorch_model.bin, so even that weight's
  size was absent from the old weight inventory. Capture .bin content now but
  do not claim retrospective authentication.
- Official AI policy requires disclosure of method implementation, synthetic
  data, experiment design and result interpretation, not just prose editing.
  Human review is pending; do not attest that it already happened.