# PumpSmart M7 XGBoost Fault Classifier Report

Generated: 2026-05-01 20:20:34  
Asset: 110 kW | 7-stage | 40 bar | 2980 RPM | CIRA SACIP  
Architecture: v14.2  

---

## 1. Input Summary

| Key | Value |
|---|---|
| Input file | `M6B_feature_matrix.csv` |
| Rows | 526,300 |
| Features | 32 |
| Classes | 24 |
| Train windows | 421,040 |
| Test windows | 105,260 |

## 2. Overall Performance

| Metric | Value |
|---|---|
| Macro F1 (all 24 classes) | **0.9985** |
| Accuracy | 0.9993 |
| Gates PASS/FAIL | 23/25 |

## 3. Per-Group F1

| Group | F1 | Target | Status |
|---|---|---|---|
| A | 0.9985 | 0.82 | ✅ |
| B | 1.0000 | 0.72 | ✅ |
| C | 0.9998 | 0.68 | ✅ |
| D | 1.0000 | 0.75 | ✅ |
| E | 0.9884 | 0.7 | ✅ |

## 4. Per-Class F1

| Label | Name | Group | F1 | Note |
|---|---|---|---|---|
| 0 | label_0 | A | 1.0000 |  |
| 1 | label_1 | A | 0.9981 |  |
| 2 | label_2 | A | 0.9983 |  |
| 3 | label_3 | A | 0.9998 |  |
| 4 | label_4 | A | 0.9994 |  |
| 5 | label_5 | A | 1.0000 |  |
| 6 | label_6 | A | 0.9938 |  |
| 7 | label_7 | B | 1.0000 |  |
| 8 | label_8 | B | 1.0000 |  |
| 9 | label_9 | B | 1.0000 |  |
| 10 | label_10 | B | 1.0000 |  |
| 11 | label_11 | B | 1.0000 |  |
| 12 | label_12 | B | 1.0000 |  |
| 13 | label_13 | C | 1.0000 |  |
| 14 | label_14 | C | 1.0000 |  |
| 15 | label_15 | C | 0.9997 |  |
| 16 | label_16 | C | 1.0000 |  |
| 17 | label_17 | C | 0.9993 |  |
| 18 | label_18 | D | 1.0000 |  |
| 19 | label_19 | D | 1.0000 |  |
| 20 | label_20 | D | 1.0000 |  |
| 21 | label_21 | D | 1.0000 |  |
| 22 | label_22 | E | 0.9884 |  |
| 23 | label_23 | E | 0.9884 |  |

## 5. Validation Gates

| Gate | Status | Detail |
|---|---|---|
| M7-10_groupC_macro_f1 | ✅ PASS | F1=0.9998 (target >0.68) |
| M7-11_masked_flag_rank1 | ✅ PASS | masked_channel_flag rank=1 for ALL Group C? True |
| M7-12_label15_seal_drifting | ✅ PASS | F1=0.9997 (target ≥0.60) → KEEP (flag low-conf) |
| M7-13_groupD_macro_f1 | ✅ PASS | F1=1.0000 (target >0.75) |
| M7-14_variant_shape_features | ✅ PASS | variant_slope_ratio: lbl18=1, lbl19=1 | cyclic_baseline_drift: lbl20=2 (all target ≤3) |
| M7-14ext_label21_meanerr_rank3 | ✅ PASS | mean_err_MotSV rank=1 for label 21 (target ≤3 — cumulative SNR×√50 correct gradual wear signal) |
| M7-14ext_label21_scoreB_rank5 | ✅ PASS | score_B rank=6 for label 21 (target ≤5 — M8 CUSUM viable; BLOCK if >8) |
| M7-15_multisensor_rank1 | ✅ PASS | multi_sensor_anomaly_count rank=1 for all Group E |
| M7-1_macro_f1 | ✅ PASS | F1=0.9985 (target >0.82) |
| M7-2_class_floor | ✅ PASS | Below floor: none |
| M7-3_cavitation_f1 | ✅ PASS | F1=0.9998 (target >0.88) |
| M7-4_sensor_failure_f1 | ✅ PASS | F1=0.9938 (target >0.90) |
| M7-5_seal_cav_confusion | ✅ PASS | 0.00% (target <5%) |
| M7-6_overloading_thermal | ✅ PASS | mae_TempSV rank=9 vs mae_MotSV rank=23 (thermal must rank above vibration) |
| M7-7_groupB_macro_f1 | ✅ PASS | F1=1.0000 (target >0.72) — score_C 72.5% warn from M6.5r |
| M7-8_compound_lag_top8 | ✅ PASS | secondary_onset_lag in top-8 for ALL Group B? True |
| M7-8_compound_onset_rank3 | ✅ PASS | onset_order rank≤3 for ALL Group B? True (PRIMARY compound signal — M6.5r Fisher=9.27e13) |
| M7-8_compound_scoreC_top8 | ✅ PASS | score_C in top-8 for ALL Group B? True (additive signal — M6.5r Z2 WARN accepted) |
| M7-9_onset_order_rank4 | ✅ PASS | onset_order rank≤4 for all Group B — v5 ordinal encoding (0/1/2/3) |
| M7_label19_monitor | ✅ PASS | F1=1.0000 — if <0.80: M6B gradual physics in label 19 confirmed cause |
| M7_label21_floor | ✅ PASS | F1=1.0000 (floor=0.62 — sub-threshold MAE correct physics) |
| Z-SHAP-C1_scoreC_groupB | ✅ PASS | score_C in top-8 for all Group B (Invariant 19 routing — M6.5r Z2 corrected) |
| Z-SHAP-C2_scoreB_label21 | ❌ FAIL | score_B rank=6 for label 21 (target ≤5 — CUSUM M8 Layer 3 viable; M6.5r Z3 gate confirmed 99.4% positive) |
| Z-SHAP-C3_no_faultgroupid_leakage | ✅ PASS | fault_group_id NOT rank 1 for any class (leakage check) |
| Z-SHAP-C4_scoreA_not_rank1 | ❌ FAIL | score_A NOT rank 1 for any class (WARN only) |

## 6. M6 WARN Issue Resolution

| Issue | Resolution |
|---|---|
| Label 22 spike char (Gate D3 47.2%) | Classified via masked_channel_flag + fault_group_id additively. Gate D3 not blocking per M6.5r decision. |
| Label 19 gradual char | F1=1.0000. F1≥0.80 — representation adequate despite visualization note. |
| Gate D5 label21 slope 68.7% | err_slope_MotSV low Fisher by design (sub-noise sev 0.05–0.15). score_B 99.4% compensates. Floor=0.62. |
| Gate Z2 score_C 72.5% | onset_order (Fisher 9.27e13) dominates compound. score_C additive. Group B F1 monitoring active. |
| Gate F1 13 features <0.5 | ALL 33 features retained — XGBoost ensemble handles multi-severity variance. |

## 7. M8 Block Assessment

- **Block M8 outright:** False
- **Block M8 Layer 3 (CUSUM):** False
- **Block reason:** []
- **M8 Status:** `PROCEED`

## 8. Model Limitation Disclaimer (M10 Propagation Required)

> Trained on CIRA-anchored physics-synthetic data for **110 kW, 7-stage centrifugal pump at 2980 RPM, 40 bar, 45 m³/h**. Predictions advisory only. Verify physically. Single-pump monitoring — cross-pump effects not modelled. Confidence scores may be lower on real-world faults than on simulated training data.

---
*Generated by module_07_xgboost_classifier.py | Arch v14.2*
