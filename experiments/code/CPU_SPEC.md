# Reproducibility: Named CPU & Thread Config

All latency numbers (P50/P95, batch=1, warmed) in this paper are measured on:

- **Hardware:** Apple Silicon (arm64), development machine
- **Threads:** `torch.get_num_threads()` default = 10 (not pinned further;
  document actual value alongside every latency table -- if reported numbers
  move to a different machine before submission, re-capture this file)
- **torch:** 2.13.0
- **transformers:** 5.14.1
- **Python:** 3.11.13 (uv-managed)
- **OS:** macOS (Darwin)

Captured 2026-08-07 via `python3 -c "import torch; print(torch.get_num_threads())"`.

Any latency claim in the paper MUST cite this file's git commit hash for exact
reproducibility, and should be re-verified on the final target CPU before
submission if that differs from a laptop (e.g. a cloud CPU instance for the
official run).
