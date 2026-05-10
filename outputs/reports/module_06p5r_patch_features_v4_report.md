# M6.5r Feature Patch v4 Report

Generated: 2026-04-30 23:07:54

## Key Fix

score_C changed from **per-label mean** (constant within label) to **per-sequence SNR** (varies between sequences of same label).

This restores within-label variance → prevents artificial Fisher inflation → allows other features (err_slope_MotSV, masked_channel_flag, multi_sensor_anomaly_count) to compete correctly in SHAP ranking.

## Fisher Comparison

| Feature | Original | v4 |
|---|---|---|
| score_C | 1.2184 | 1.918440 |
| err_slope_MotSV | 0.0564 | 1.295148 |
| multi_sensor_anomaly_count | 1.3284 | 3.573664 |
| variant_slope_ratio | 0.0276 | 3.457310 |

## Gates

| Gate | Status | Detail |
|---|---|---|
| P1_scoreC_groupB_gt_groupA | ✅ PASS | Group B (3.4944) > Group A (1.0000) |
| P1_scoreC_withinlabel_variance | ✅ PASS | Group B score_C std=0.5325 (target >0.01 — per-sequence variation) |
| P1_scoreC_lbl21_variance | ✅ PASS | Label 21 score_C std=0.2846 (target >0.01) |
| P1_scoreC_fisher_reasonable | ✅ PASS | Fisher=1.9184 (target: moderate, not astronomical constant-value inflated) |
| P2_lbl21_slope_pos | ✅ PASS | Label 21: 100.0% positive |
| P2_normal_slope_neg | ✅ PASS | Normal mostly negative/zero: 25.0% positive (target <65%) |
| P3_lbl22_ok | ✅ PASS | Label 22: 100.0% |
| P3_lbl23_ok | ✅ PASS | Label 23: 100.0% |
| P3_grpA_zero | ✅ PASS | Group A FP: 0.000% |
| P4_normal_zero | ✅ PASS | Normal=0.0000 (target <0.01) |
| P4_lbl18_positive | ✅ PASS | Label 18=2.064 (target >0.5) |
| P4_lbl19_positive | ✅ PASS | Label 19=0.3968 (target >0.10) |
| P5_eslope_improved | ✅ PASS | err_slope: 0.0564 → 1.2951 |
| P5_scoreC_moderate | ✅ PASS | score_C Fisher moderate: 1.9184 |
| P5_variant_improved | ✅ PASS | variant: 0.0276 → 3.4573 |

**M7 status: `RERUN_READY`**
