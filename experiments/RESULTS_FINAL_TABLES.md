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
| atis | 1 | 94.71 | 86.46 | 1.94 |
| atis | 2 | 97.78 | 90.74 | 3.26 |
| atis | 3 | 97.95 | 90.96 | 5.12 |
| atis | 6 | 97.78 | 92.27 | 10.91 |
| atis | 9 | 98.12 | 92.18 | 15.79 |
| atis | 12 | 97.44 | 91.88 | 19.54 |
| snips | 1 | 97.71 | 71.27 | 1.72 |
| snips | 2 | 98.14 | 78.26 | 3.19 |
| snips | 3 | 98.00 | 82.17 | 4.87 |
| snips | 6 | 98.29 | 84.46 | 9.52 |
| snips | 9 | 98.43 | 84.90 | 15.70 |
| snips | 12 | 98.14 | 84.65 | 18.90 |
| massive | 1 | 69.70 | 47.66 | 1.15 |
| massive | 2 | 72.46 | 51.32 | 1.84 |
| massive | 3 | 73.81 | 50.59 | 2.65 |
| massive | 6 | 74.98 | 55.70 | 5.32 |
| massive | 9 | 75.62 | 55.14 | 7.61 |
| massive | 12 | 74.61 | 54.28 | 10.45 |
| clinc150 | 1 | 75.87 |   --   | 1.19 |
| clinc150 | 2 | 77.29 |   --   | 1.79 |
| clinc150 | 3 | 79.02 |   --   | 2.64 |
| clinc150 | 6 | 81.44 |   --   | 5.10 |
| clinc150 | 9 | 81.93 |   --   | 7.65 |
| clinc150 | 12 | 82.00 |   --   | 10.05 |
| banking77 | 1 | 75.68 |   --   | 1.20 |
| banking77 | 2 | 77.69 |   --   | 1.80 |
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

## Table 5 — Slot-loss weight lambda sweep (ATIS depth-3, softmax slot head)

| lambda | Intent Acc | Slot F1 | P50 (ms) |
|---:|---:|---:|---:|
| 1.0 | 98.12 | 89.56 | 4.89 |
| 1.5 | 97.95 | 90.68 | 5.07 |
| 2.0 | 97.95 | 90.96 | 4.98 |
| 2.5 | 97.78 | 91.23 | 5.18 |
| 3.0 | 97.61 | 91.53 | 6.72 |

## Table 6 — Dense per-layer frozen probe (intent accuracy vs. depth)

| Dataset | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| atis | 95.2 | 95.2 | 95.1 | 95.6 | 95.6 | 94.9 | 95.9 | 95.1 | 95.1 | 96.2 | 95.2 | 95.9 |
| snips | 96.7 | 96.9 | 96.1 | 96.5 | 97.0 | 97.0 | 96.7 | 97.0 | 96.7 | 96.7 | 96.6 | 94.9 |
| massive | 74.4 | 75.0 | 75.0 | 74.1 | 75.5 | 73.5 | 74.8 | 74.3 | 72.3 | 72.3 | 72.1 | 72.9 |
| clinc150 | 87.6 | 85.6 | 86.8 | 87.6 | 86.9 | 87.1 | 85.8 | 86.9 | 84.6 | 85.0 | 85.5 | 86.1 |
| banking77 | 81.9 | 79.9 | 80.5 | 79.8 | 76.9 | 78.5 | 79.3 | 79.1 | 77.5 | 76.2 | 73.5 | 75.7 |

## Table 7 — Spearman correlation: frozen probe vs fine-tuned intent accuracy

| Dataset | n depths | rho | p |
|---|---:|---:|---:|
| atis | 6 | -0.612 | 0.197 |
| snips | 6 | +0.353 | 0.493 |
| massive | 6 | -0.754 | 0.084 |
| clinc150 | 6 | -0.486 | 0.329 |
| banking77 | 6 | -0.771 | 0.072 |
| **within-dataset mean** | -- | **-0.454** | -- |
| **within-dataset median** | -- | **-0.612** | -- |
| **pooled** (see caveat) | 30 | **+0.871** | 0.000 |

_Caveat: pooled rho conflates within-dataset ranking with between-dataset difficulty. Within-dataset rho is the metric aligned with C1 (depth selection within a dataset). Per-dataset n is small so individual values are noisy._

## Table 7b — Probe-selected depth vs empirically shallowest-acceptable fine-tuned depth

| Dataset | dense probe pick | fine-tune best depth | fine-tune best acc | shallowest within 1pt |
|---|---:|---:|---:|---:|
| atis | 1 | 9 | 98.12 | 2 |
| snips | 1 | 9 | 98.43 | 1 |
| massive | 1 | 9 | 75.62 | 6 |
| clinc150 | 1 | 12 | 82.00 | 6 |
| banking77 | 1 | 9 | 81.79 | 6 |

## Table 8 — Dense per-layer token-level slot probe (BIO-tag accuracy vs. depth)

| Dataset | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| atis | 95.4 | 95.6 | 96.1 | 96.4 | 96.4 | 96.4 | 96.3 | 96.2 | 95.6 | 95.7 | 95.6 | 95.5 |
| snips | 77.4 | 84.5 | 84.9 | 84.7 | 85.6 | 84.9 | 81.4 | 84.8 | 85.0 | 84.6 | 83.5 | 84.6 |
| massive | 82.9 | 83.2 | 82.2 | 83.2 | 82.9 | 82.5 | 82.9 | 82.1 | 82.6 | 82.5 | 81.5 | 83.8 |

