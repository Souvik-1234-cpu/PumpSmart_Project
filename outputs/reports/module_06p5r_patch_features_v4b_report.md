# M6.5r Feature Patch v4b Report

Generated: 2026-05-01 00:04:17

## Key Fix

score_C assigned per-sequence using linspace boundaries — guarantees 100% row coverage with no remaining rows.

## Fisher Scores

| Feature | Original | v4b |
|---|---|---|
| score_C | 1.2184 | 1.1383 |
| err_slope_MotSV | 0.0564 | 1.2951 |
| multi_sensor_anomaly_count | 1.3284 | 3.5737 |
| variant_slope_ratio | 0.0276 | 3.4573 |

## Gates

| Gate | Status | Detail |
|---|---|---|
| P1_groupB_gt_groupA | ✅ PASS | Group B (3.4950) > Group A (1.0000) |
| P1_within_label_variance | ✅ PASS | Group B std=0.5395 (target >0.01) |
| P1_lbl21_variance | ✅ PASS | Label 21 std=0.3975 (target >0.01) |
| P1_fisher_moderate | ✅ PASS | Fisher=1.1383 (moderate, not inflated) |
| P1_full_coverage | ✅ PASS | All labels with pkl data have >1 unique score_C value |
| P2_lbl21_slope | ✅ PASS | Label 21: 100.0% |
| P2_normal_neg | ✅ PASS | Normal: 25.0% (target <65%) |
| P3_lbl22 | ✅ PASS | Label 22: 100.0% |
| P3_lbl23 | ✅ PASS | Label 23: 100.0% |
| P3_grpA_fp | ✅ PASS | Group A FP: 0.000% |
| P4_normal_zero | ✅ PASS | Normal=0.0000 |
| P4_lbl18_positive | ✅ PASS | lbl18=2.064 |
| P4_lbl19_positive | ✅ PASS | lbl19=0.3968 |
| P5_eslope_improved | ✅ PASS | err_slope Fisher → 1.2951 |
| P5_scoreC_moderate | ✅ PASS | score_C Fisher → 1.1383 |
| P5_variant_improved | ✅ PASS | variant Fisher → 3.4573 |

**M7 status: `RERUN_READY`**
