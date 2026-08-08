# LaTeX draft — Paper 1

Full draft of *Generate Less, Classify More* with all real numbers from
`../../experiments/RESULTS_FINAL_TABLES.md`, `RESULTS_M1_*.md`,
`RESULTS_M2_probe_pruning.md`, and `RESULTS_M5_negresults.md`.

## Files
- `paper.tex` — main manuscript (double-blind: no author names).
- `references.bib` — bibliography.

## Compile (once a LaTeX toolchain is installed)
```bash
# macOS: brew install --cask mactex-no-gui   (uses Walmart sysproxy if needed)
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```
This machine currently has no `pdflatex`; the source is written to compile
cleanly with a standard TeX Live install.

## Before ICLR 2027 submission (see ../SUBMISSION_REQUIREMENTS.md)
1. **Swap the preamble** for the official `iclr2027_conference.sty` /
   `.bst` when released (currently an `article` fallback so the draft builds).
   Switch `\bibliographystyle{plainnat}` to the ICLR style.
2. **Verify <= 9 pages** main text (references + appendix + AI-Use excluded).
3. **Double-blind:** keep authors anonymous; strip PDF metadata; release code
   via an anonymized repo (anonymous.4open.science), not a personal/company repo.
4. Keep the **AI-Use Statement** (already included; excluded from page count).
5. Add figures: probe-sweep curve, depth Pareto, latency bars (data already in
   the results JSON; generate with the aggregation script).

## Status of numbers
All tables use REAL measured values (gpt2-124M backbone, DistilBERT baseline,
CPU P50 latency), 3-seed variance where available. Regenerate the source tables
with `python ../../experiments/code/scripts/aggregate_results.py`.
