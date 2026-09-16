# Pending Audit — `PROJECT_REPORT.md` and `OUTLINE.md`

## A) Pending from `PROJECT_REPORT.md`

### Submission/logistics pending (explicit)
1. Install TeX toolchain and compile submission PDF; verify main text <= 9 pages.
2. Swap fallback preamble to official `iclr2027_conference.sty` and bibliography stack.
3. Anonymize release repo + strip metadata (PDF + git identities + paths).
4. Finalize author list and OpenReview profile readiness.
5. Optional camera-ready additions: second backbone size + rotation-based INT8/fused kernels.

### Research-status note
- Document states research is complete and remaining items are submission mechanics.

## B) Pending from `OUTLINE.md`

### Still-open technical/research tasks
1. Slot-loss weight lambda sweep is marked stretch (currently fixed at 2.0).
2. Not-run negative-results extensions: QuaRot/Hadamard rotation-INT8 + fused kernels + vendor CPU extensions + ONNX-INT8.

### Potential narrative debt (not necessarily missing experiments)
1. Header language near top still says skeleton/TBD context historically; should remain consistent with now-filled-real-results status.
2. Ensure all section claims in outline are exactly mirrored in `latex/paper.tex` wording to avoid drift.

## C) Priority ordering

### Must-do before ICLR submission
- TeX compile + page budget + official template migration
- Full anonymity scrub (paper + supplement + code)
- OpenReview/admin completeness

### Nice-to-have / camera-ready
- Additional optimization variants and second-backbone breadth
- Lambda sweep if time allows after freeze

## D) Recommended immediate next actions (this week)
- Freeze thesis + claim map in manuscript (Phase A done via `PHASE_A_CLAIM_HARDENING.md`).
- Run one consistency pass: `OUTLINE.md` vs `latex/paper.tex` claim-by-claim.
- Open a submission readiness checklist issue with 4 hard blockers and owners.
