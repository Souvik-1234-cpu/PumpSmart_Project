# M6.5r Feature Patch v3 Report

Generated: 2026-04-30 19:17:57

## Fisher Comparison

| Feature | Original | v3 |
|---|---|---|
| score_C | 1.2184 | 7962503525246059.0000 |
| err_slope_MotSV | 0.0564 | 1.2951 |
| multi_sensor_anomaly_count | 1.3284 | 3.5737 |
| variant_slope_ratio | 0.0276 | 3.4573 |

## Gates

| Gate | Status | Detail |
|---|---|---|
| P1_scoreC_groupB_gt_groupA | ✅ PASS | Group B (3.4949) > Group A (1.0000) |
| P1_scoreC_fisher_ok | ✅ PASS | Fisher = 7962503525246059.0000 (target ≥0.50) |
| P2_lbl21_slope_pos | ✅ PASS | Label 21: 100.0% positive (target >90%) |
| P2_normal_slope_near50 | ❌ FAIL | Normal: 25.0% positive (target 35–65%) |
| P3_lbl22_ok | ✅ PASS | Label 22: 100.0% |
| P3_lbl23_ok | ✅ PASS | Label 23: 100.0% |
| P3_grpA_fp | ✅ PASS | Group A FP: 0.000% |
| P4_lbl18_gt_normal | ✅ PASS | Label 18 (2.0637) > normal (0.0000) + 0.5 |
| P4_lbl19_positive | ✅ PASS | Label 19 = 0.3968 (target >0.10) |
| P4_normal_near_zero | ✅ PASS | Normal = 0.0000 (target <0.10 — no variant signal) |
| P5_eslope_improved | ✅ PASS | err_slope: 0.0564 → 1.2951 |
| P5_scoreC_ok | ✅ PASS | score_C: → 7962503525246059.0000 (target ≥0.50) |
| P5_variant_improved | ✅ PASS | variant: 0.0276 → 3.4573 |

**M7 status: `RERUN_RECOMMENDED`**
