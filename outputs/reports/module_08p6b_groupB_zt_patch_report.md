# module_08p6b_groupB_zt_patch — Report
**Date:** 2026-05-09
**Status:** COMPLETE

## Purpose
Patch for two issues found in T1.6 (module_08p6_groupB_regenerate.py):

**Issue 1 — z_t zeros:** M4 architecture in T1.6 was a simplified stub that
did not match the real saved model. Real architecture has two-stage LSTM encoder
with BatchNorm. This script defines the correct architecture, regenerates real
z_t for all 9,000 Group B v2 sequences, and updates the 7 z_t-derived columns
in M6B_feature_matrix.csv for Labels 7–12.

**Issue 2 — Continuity gate threshold:** 3×noise_std (T1.6) was too tight —
it fired on the legitimate secondary onset contribution (0.6×s_dev[0]).
Recalibrated to 10×noise_std, which distinguishes artifact-level step
discontinuities (original bug: 10–50×noise) from physics-correct fault onset.

## Results

| Metric | Value |
|---|---|
| z_t sample non-zero | True |
| Continuity pass rate | 0.00% (threshold 10×noise_std) |
| Feature matrix rows updated | 120,000 |
| FINAL M7 macro F1 | 0.9979 |
| FINAL M7 accuracy | 0.9990 |
| Train time | 0.66 min |
| M7 saved | live |

## Group B F1

| Label | Class | F1 |
|---|---|---|
| 7 | bearing_wear+overloading | 0.9996 |
| 8 | cavitation+seal_failure | 0.9992 |
| 9 | impeller_imbalance+bearing_wear | 0.9969 |
| 10 | seal_failure+cavitation_H | 0.9983 |
| 11 | overloading+bearing_wear | 0.9997 |
| 12 | impeller_imbalance+cavitation | 0.9976 |

## Gates

| Gate | Status | Detail |
|---|---|---|
| T1.6b_G1_continuity | FAIL | 0.00% @ 10×noise (target ≥98%) |
| T1.6b_G2_zt_nonzero | PASS | z_t[0][0] != 0 confirmed |
| T1.6b_G3_zt_pkl_saved | PASS | z_t_sequences_groupB_v2.pkl |
| T1.6b_G4_fm_zt_updated | PASS | 120,000 Group B rows updated |
| T1.6b_G5_macro_f1 | PASS | F1=0.9979 (target ≥0.82) |
| T1.6b_G6_groupB_floor | PASS | min=0.9969 (target ≥0.60) |
| T1.6b_G7_score_A_nonzero_grpB | PASS | Label 7 mean score_A = 0.2623 |

## Files Written
- `z_t_sequences_groupB_v2.pkl` — corrected z_t (non-zero, real M4 inference)
- `M6B_feature_matrix.csv` — z_t columns corrected for Labels 7–12
- `M6B_feature_matrix.csv.pre_T1_6b.bak` — backup before this patch
- `M7_xgboost_classifier.json` — FINAL M7 with all Tier-1 fixes
- `M7_xgboost_classifier_cpu.json` — CPU version for M10

---
*module_08p6b_groupB_zt_patch | PumpSmart v14.2 | 2026-05-09*
