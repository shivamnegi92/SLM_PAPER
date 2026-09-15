#!/usr/bin/env bash
# Render the paper to a single self-contained HTML file.
#
# pandoc is available on this machine but pdflatex is not, so PDF via LaTeX is
# not an option here. --embed-resources inlines the figures as base64 so the
# output is one file that can be emailed or opened anywhere with no sidecar
# image directory.
#
# Numbers are verified against the frozen snapshot BEFORE rendering, so a stale
# or edited draft cannot be published by accident.
set -euo pipefail

cd "$(dirname "$0")/.."
PAPER="paper/PAPER_DRAFT.md"
OUT="paper/paper.html"

echo "Verifying paper against results/frozen_e91985c/ ..."
python src/verify_paper_numbers.py > /dev/null
echo "  numbers OK"

echo "Rendering ${OUT} ..."
pandoc "${PAPER}" \
    --from=gfm \
    --to=html5 \
    --standalone \
    --embed-resources \
    --toc \
    --toc-depth=2 \
    --metadata title="Effective, Specific, and Non-Selective" \
    --resource-path=paper \
    --css=paper/paper.css \
    --output="${OUT}"

echo "  wrote ${OUT} ($(du -h "${OUT}" | cut -f1))"
