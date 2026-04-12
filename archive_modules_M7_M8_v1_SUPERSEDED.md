> ⚠️ ARCHIVED — SUPERSEDED
> This file is the original combined M7+M8 spec (v1.0, pre-bias-audit).
> It has been replaced by:
>   - `module_M7_xgboost_classifier.md`  (multi-label arch, Stage 1/2/3, compound faults)
>   - `module_M8_lstm_ae_v2.md`           (4-state alert, all bias-audit fixes)
> Do NOT use this file for coding. Kept for design history only.
> Archived: 2026-04-12

---

# PumpSmart — M7 + M8: Critical ML Modules
# XGBoost Fault Classifier + LSTM-AE v2 Production Anomaly Detector
# Status: M7 NEXT ACTIVE | M8 follows immediately after M7
# All design decisions derived from M6.5 audit findings (completed_modules_M1_to_M6p5.md)
# Updated: 2026-04-12 | Split from: module_pathway_M1_to_M12_v10.md
# Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset

---

## SECTION 1 — M6.5 AUDIT QUICK-REFERENCE TABLE
*(All M7 + M8 design decisions trace back to these findings — do not skip)*

| Finding | Root Cause | M7 Impact | M8 Impact |
|---|---|---|---|
| Overloading Gate 3 = 0.00% (MAE=0.093) | Temp.SV/Mot.TV low weight → sub-threshold | `mean_err_TempSV` must be SHAP rank 1 for overloading | Mech C Temp.SV PRIMARY — not single-window MAE |
| Seal failure Gate 3 = 29.17% (MAE=0.196) | Pres.SV decline too gradual per window | `err_slope_PresSV` must be SHAP top-3 for seal | Mech C Pres.SV PRIMARY — Gate M8-9/10 |
| Bearing seam discontinuity = 5.75% | Spike seed t=49→t=50 step change | `err_onset_lag` discriminates from imbalance | Attention must NOT peak at t=49–50 — Gate M8-8 |
| Fisher rank 1 = Pmp_SV_mean | Pmp.SV dominant fault channel | Expected SHAP rank 1 for bearing/cavitation/imbalance | Weight increase 2.0→2.5 Fisher-validated |
| Cavitation MAE = 0.675 (6.1× threshold) | Hydraulic shock — always acute | Cavitation F1 > 0.88 — easiest to classify | Bypass WATCH/WARN → directly DANGER |
| Normal probe 86.67% = NOT FPR problem | Edge-case probe sampling artifact | No action needed in M7 | Gate M8-2 on full 9711-window pool ONLY |

---

## SECTION 2 — M7: XGBOOST FAULT CLASSIFIER

### Why M7 Runs Before M8

```
M7 runs FIRST — it validates that the M6.5 feature matrix is
physically meaningful before M8 uses it to calibrate fuzzy boundaries.

If M7 SHAP is physically wrong → M6.5 features are corrupt →
M8 fuzzy calibration will be wrong → fix M6.5 first, do not proceed to M8.

This sequencing is non-negotiable.
```

### Input

```
File    : data/synthetic/M6_feature_matrix.csv
Shape   : 8400 rows × 25 columns (24 features + label)
Labels  : 0=normal, 1=cavitation, 2=bearing_wear, 3=seal_failure,
          4=overloading, 5=impeller_imbalance, 6=sensor_failure
Source  : M6.5 LSTM-AE feature extractor (Gate 3 :50 fix applied)
```

### Architecture

```
Model     : XGBoost (xgboost>=2.0)
Training  : device='cuda'  (RTX 4060 Laptop)
Deployment: device='cpu'   (M10 Flask inference)
Split     : 80% train / 20% test — stratified by label
Tuning    : Optuna 50 trials, 5-fold stratified CV
Objective : multi:softprob
Weights   : inverse class frequency (balanced)
Explainer : SHAP TreeExplainer — top-3 per prediction + global importance plot
```

*(remaining original content preserved — see git history)*
