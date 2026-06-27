# M12 Stage 4 — Confidence Calibration Fit

**Date:** 2026-05-30

Temperature scaling (no retrain) + physics-honest ceiling 0.94.

| Metric | Raw | Calibrated | Cal+Ceiling |
|---|---|---|---|
| Mean top-confidence | 0.9408 | 0.9319 | 0.8841 |
| ECE | 0.0160 | 0.0098 | — |
| NLL | 0.1689 | 0.1650 | — |

**Fitted T = 1.260** | Label unchanged: True

## Gates

| Gate | Pass |
|---|---|
| G_CF_1_matrix_rebuilt | PASS |
| G_CF_2_ece_improves | PASS |
| G_CF_3_softened | PASS |
| G_CF_4_label_invariant | PASS |
