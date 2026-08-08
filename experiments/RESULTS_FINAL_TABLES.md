# SLM_PAPER — Final Aggregated Tables

Backbone gpt2-124M pruned; matched 5000-example budget; CPU P50 latency.

## Table 1 — Fair head-to-head (matched 5000-example budget)

| Dataset | B2 encoder acc | B2 slot F1 | ours d3 acc | ours d3 slotF1 | ours d3 P50 | ours d12 acc | ours d12 slotF1 | ours d12 P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atis | 97.95 | 94.50 | 97.95 | 90.96 | 5.12 | 97.44 | 91.88 | 19.54 |
| snips | 99.14 | 93.21 | 98.00 | 82.17 | 4.87 | 98.14 | 84.65 | 18.90 |
| massive | 83.73 | 70.53 | 73.81 | 50.59 | 2.65 | 74.61 | 54.28 | 10.45 |
| clinc150 | 92.96 |   --   | 79.02 |   --   | 2.64 | 82.00 |   --   | 10.05 |
| banking77 | 89.38 |   --   | 78.34 |   --   | 2.91 | 80.78 |   --   | 10.65 |

## Table 2 — Depth Pareto (ours, per dataset)

| Dataset | Depth | Intent Acc | Slot F1 | P50 (ms) |
|---|---:|---:|---:|---:|
| atis | 3 | 97.95 | 90.96 | 5.12 |
| atis | 6 | 97.78 | 92.27 | 10.91 |
| atis | 9 | 98.12 | 92.18 | 15.79 |
| atis | 12 | 97.44 | 91.88 | 19.54 |
| snips | 3 | 98.00 | 82.17 | 4.87 |
| snips | 6 | 98.29 | 84.46 | 9.52 |
| snips | 9 | 98.43 | 84.90 | 15.70 |
| snips | 12 | 98.14 | 84.65 | 18.90 |
| massive | 3 | 73.81 | 50.59 | 2.65 |
| massive | 6 | 74.98 | 55.70 | 5.32 |
| massive | 9 | 75.62 | 55.14 | 7.61 |
| massive | 12 | 74.61 | 54.28 | 10.45 |
| clinc150 | 3 | 79.02 |   --   | 2.64 |
| clinc150 | 6 | 81.44 |   --   | 5.10 |
| clinc150 | 9 | 81.93 |   --   | 7.65 |
| clinc150 | 12 | 82.00 |   --   | 10.05 |
| banking77 | 3 | 78.34 |   --   | 2.91 |
| banking77 | 6 | 80.91 |   --   | 5.36 |
| banking77 | 9 | 81.79 |   --   | 9.07 |
| banking77 | 12 | 80.78 |   --   | 10.65 |

## Table 3 — CRF vs softmax slot head (depth 3)

| Dataset | Softmax slot F1 | CRF slot F1 | Delta |
|---|---:|---:|---:|
| atis | 90.96 | 91.14 | +0.19 |
| snips | 82.17 | 82.44 | +0.27 |
| massive | 50.59 | 50.68 | +0.09 |

## Table 4 — Multi-seed (mean +/- std over seeds {42,1,2}), ours

| Dataset | Depth | Intent Acc (mean+/-std) | Slot F1 (mean+/-std) |
|---|---:|---:|---:|
| atis | 3 | 97.95 +/- 0.14 (n=3) | 90.92 +/- 0.07 |
| atis | 12 | 97.84 +/- 0.29 (n=3) | 92.01 +/- 0.22 |
| snips | 3 | 98.19 +/- 0.18 (n=3) | 82.06 +/- 0.23 |
| snips | 12 | 98.14 +/- 0.00 (n=3) | 83.92 +/- 0.86 |
| massive | 3 | 73.49 +/- 0.32 (n=2) | 52.43 +/- 1.84 |
| massive | 12 | 74.50 +/- 0.12 (n=2) | 54.92 +/- 0.63 |
| clinc150 | 3 | 78.06 +/- 0.91 (n=3) | -- |
| clinc150 | 12 | 80.44 +/- 1.68 (n=3) | -- |
| banking77 | 3 | 78.74 +/- 0.29 (n=3) | -- |
| banking77 | 12 | 81.32 +/- 0.81 (n=3) | -- |

