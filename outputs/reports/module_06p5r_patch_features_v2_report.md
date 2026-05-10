# M6.5r Feature Patch v2 Report

Generated: 2026-04-30 19:11:07

## v1 Failure Root Causes + v2 Fixes

| Issue | v1 Cause | v2 Fix |
|---|---|---|
| score_C Fisher=0.0001 | Rolling window mixing across sequences | Per-label mean SNR from z_t pkl entries |
| MS count 38% FP | Global 0.05 threshold below startup noise | Label-specific patch: only labels 22,23 |
| variant lbl18<lbl3 | P10 global baseline (contaminated) | P20 within label 18 only |
| variant lbl19=0.004 | err_slope_PresSV near zero (Fisher 0.049) | mae_PresSV×2.0 (collapse amplitude) |
| normal slope 32.5% | All-cluster baseline too high | Steady-state P50 subset baseline |

## Fisher Scores

| Feature | Original | v2 |
|---|---|---|
| score_C | 1.2184 | 40192164227498632.0000 |
| err_slope_MotSV | 0.0564 | 2.7074 |
| multi_sensor_anomaly_count | 1.3284 | 1.1367 |
| variant_slope_ratio | 0.0276 | 0.0266 |

## Gates

| Gate | Status | Detail |
|---|---|---|
| P1_scoreC_groupB_gt_groupA | ❌ FAIL | Group B (3.4949) > Group A (8.1609) |
| P1_scoreC_fisher_ok | ✅ PASS | score_C Fisher = 40192164227498632.0000 (target ≥0.50) |
| P2_label21_slope_positive | ✅ PASS | 100.0% positive for label 21 (target >90%) |
| P2_normal_slope_near50 | ❌ FAIL | Normal slope positive% = 75.6% (target 35–65%) |
| P3_label22_dual_thresh | ❌ FAIL | Label 22: 12.6% (target >50%) |
| P3_label23_dual_thresh | ❌ FAIL | Label 23: 2.4% (target >50%) |
| P3_groupA_fp_zero | ✅ PASS | Group A FP: 0.000% (target <0.1% — label-specific patch) |
| P4_lbl18_gt_normal | ❌ FAIL | Label 18 (2.0637) > 1.5× normal (63.1294) — burst > baseline |
| P4_lbl19_collapse_positive | ✅ PASS | Label 19 collapse rate mean = 0.3968 (target >0.10) |
| P5_eslope_fisher_improved | ✅ PASS | err_slope Fisher 0.0564 → 2.7074 (target >2×) |
| P5_scoreC_fisher_maintained | ✅ PASS | score_C Fisher → 40192164227498632.0000 (target ≥0.50) |

**M7 rerun status: `INVESTIGATE`**
