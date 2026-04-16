# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
# PART 2B OF 3 — Gate M8-14-ext, Outputs, Paste Keys, Dependency Chain

**Document version:** v1.0 — 22-class M6B alignment (v14.0)
**Date:** 2026-04-16
**Part 1A (LSTM-AE + Fuzzy):** `module_M8_lstm_ae_v2_architecture.md`
**Part 1B (Mechanisms):** `module_M8_lstm_ae_v2_mechanisms.md`
**Part 2A (Gates + Alert Machine):** `module_M8_lstm_ae_v2_gates_and_outputs.md`
**Prerequisite:** M7 all 16 gates passed | `M7_all_16_gates_pass = True`
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)

> ⚠️ READ Part 1A, Part 1B, and Part 2A BEFORE this file.
> This file = Gate M8-14-ext detail + outputs + paste keys + dependency chain.

---

## Gate M8-14-ext — Full Specification (Label 21: bearing_wear_gradual)

```
GATE-M8-14-ext : Layer 3 CUSUM + Layer 4 Rolling Baseline detection of label 21

PHYSICS JUSTIFICATION:
  bearing_wear_gradual (label 21) operates in Paris–Erdogan low-ΔK regime.
  Severity 0.05–0.25 → per-window MAE NEVER crosses threshold (0.110058).
  MAE-based gates (M8-1 through M8-14) DO NOT apply to label 21.
  Gate M8-14-ext is the SOLE gate governing label 21 detection quality.
  This gate must PASS before M8 status = READY for M9.

GATE TARGETS:

  Target A — CUSUM WATCH rate:
    ≥75% of label 21 validation sequences (sev 0.05–0.25)
    → cusum_bearing_gradual_flag fires → WATCH state
    within 500 windows (~8 min at 1Hz) of simulated onset
    Measure: count(sequences WATCH within 500w) / count(all label 21 sequences)

  Target B — CUSUM + Layer 4 WARN rate:
    ≥60% of label 21 validation sequences (sev 0.05–0.25)
    → WARN state reached
    within 800 windows (~13 min) of simulated onset
    Requires BOTH cusum_bearing_gradual_flag AND rolling_baseline_drift_flag
    Measure: count(sequences WARN within 800w) / count(all label 21 sequences)
    Note: Layer 4 requires 5000-window burn-in — measure on sequences
    after burn-in window only. Document burn-in-excluded count separately.

  Target C — Normal pool false-positive rate for CUSUM:
    CUSUM does NOT fire (S_pos stays below H) for ≥95% of normal pool windows
    Measured on full 9711-window CIRA normal pool
    If CUSUM fires on > 5% normal windows → H is too low → increase H

  Target D — Severity stratification:
    Report WATCH rate separately for sev 0.05–0.10 and sev 0.10–0.25
    Expected: sev 0.10–0.25 WATCH rate ≥85%
             sev 0.05–0.10 WATCH rate ≥50% (very mild — hardest detection case)
    Do NOT aggregate — sub-threshold severity breakdown is critical
    for deployment decision (sev < 0.05 = not yet actionable)

FAIL ACTIONS (in order):
  1. Target A fail (≥75% WATCH): Lower k toward 0.3×sigma. Retune H.
     Re-run on label 21 mild subset. Re-check Target C (FPR).
  2. Target B fail (≥60% WARN): Shorten baseline_long 5000→3000 windows.
     Lower drift_ratio threshold 1.10→1.07.
     Re-validate normal pool drift_ratio stays in [0.95, 1.05].
  3. Target C fail (> 5% FPR on normal): Raise H toward 6×sigma.
     Re-run Target A. Trade-off: higher H = lower WATCH rate = re-check Target A.
  4. All targets fail: Increase label 21 sequences in M6B
     from 1,000 → 1,500. Re-run M6.5r Gate D5 first.

DO NOT:
  × Raise MAE threshold (0.110058) to detect label 21
  × Apply CUSUM to other fault classes to compensate
  × Merge label 21 into label 1 (bearing_wear) for gate convenience
  × Report only aggregate WATCH rate without severity stratification
```

---

## M8 Outputs (22-class aligned)

```
models/lstm_ae_v2_best.pth                      ← production model weights
models/M8_threshold_config.json                 ← cluster-conditional thresholds (4 values)
models/M8_fuzzy_config.json                     ← lower_bound, upper_bound, calibration log
models/M8_cusum_config.json                     ← target, k, H, reset_policy (label 21 only)
models/M8_rolling_baseline_config.json          ← window_short, window_long, drift thresholds
outputs/M8_roc_curve.png
outputs/M8_tpr_group_A.png                      ← per-class TPR bar (Group A, 7 classes)
outputs/M8_tpr_group_B.png                      ← compound DANGER rate (6 classes, labels 7–12)
outputs/M8_tpr_group_C.png                      ← masked secondary TPR (5 classes, labels 13–17)
outputs/M8_tpr_group_D.png                      ← variant alert path correctness (labels 18–20)
outputs/M8_tpr_label21_cusum.png                ← CUSUM WATCH rate by severity (label 21)
outputs/M8_tpr_label21_layer4.png               ← Rolling Baseline WARN rate by severity (label 21)
outputs/M8_tpr_group_E.png                      ← multi-sensor detection rate (2 classes)
outputs/M8_attention_heatmap.png                ← seam check [Finding F3]
outputs/M8_mech_c_drift_plots.png               ← Spearman drift per channel (all 8)
outputs/M8_fuzzy_calibration.png                ← two-population MAE with exclusions marked
outputs/M8_channel_error_dist.png               ← per-channel reconstruction error by fault type
outputs/M8_detection_coverage.png               ← which mechanisms detect which faults (22 rows)
outputs/M8_cusum_trajectory_label21.png         ← S_pos over time for label 21 mild sequences
outputs/M8_drift_ratio_label21.png              ← short/long baseline ratio over time (label 21)
outputs/reports/module_08_lstm_ae_v2_report.md
```

---

## M8 Paste Text Keys (52 Keys)

```
───────────────────────────────────────────────────────────────────
TRAINING + MODEL
───────────────────────────────────────────────────────────────────
M8_val_loss                          : [value]
M8_best_epoch                        : [value]
M8_training_time_min                 : [value]
M8_n_parameters                      : [~505,096 expected]
M8_vram_peak_gb                      : [value — RTX 4060 8GB]

───────────────────────────────────────────────────────────────────
GROUP A — SINGLE-SOURCE TPR
───────────────────────────────────────────────────────────────────
M8_tpr_group_A_overall               : [% — gate > 90%, excl. overloading+seal mild]
M8_tpr_cavitation                    : [% — expected ~100%, Finding F5]
M8_tpr_bearing_wear                  : [%]
M8_tpr_impeller_imbalance            : [%]
M8_tpr_sensor_failure_single         : [%]
M8_tpr_overloading                   : [% — gate M8-7 ≥80% via Mech C Temp.SV ONLY]
M8_tpr_seal_failure                  : [% — gate M8-9 WATCH ≤20 min via Pres.SV drift]

───────────────────────────────────────────────────────────────────
GROUP B — COMPOUND TPR (6 classes, labels 7–12)
───────────────────────────────────────────────────────────────────
M8_tpr_group_B_danger_rate           : [% — gate ≥85% reaching DANGER, all 6 classes]
M8_tpr_label07_bearing_overload      : [%]
M8_tpr_label08_cav_seal              : [%]
M8_tpr_label09_imbal_bearing         : [%]
M8_tpr_label10_seal_cav              : [%]
M8_tpr_label11_imbal_cav             : [%]
M8_tpr_label12_bearing_seal          : [% — Mot.SV then Pres.SV drift expected]

───────────────────────────────────────────────────────────────────
GROUP C — MASKED FAULT TPR (5 classes, labels 13–17)
───────────────────────────────────────────────────────────────────
M8_tpr_group_C_overall               : [% — gate ≥65% via secondary path]
M8_tpr_label13_bearing_MotSV_masked  : [% — target ≥65%]
M8_tpr_label14_cav_PresSV_masked     : [% — target ≥65%]
M8_tpr_label15_overload_TempSV_masked: [% — target ≥65%]
M8_tpr_label16_imbal_PmpSV_masked    : [% — target ≥65%]
M8_tpr_label17_seal_MotPV_masked     : [% — target ≥50%, weakest path]

───────────────────────────────────────────────────────────────────
GROUP D — SEVERITY VARIANT ALERT PATH (labels 18–20)
───────────────────────────────────────────────────────────────────
M8_tpr_label18_cav_intermittent      : [% — burst tracking WATCH+, gate ≥78%]
M8_tpr_label19_seal_fast             : [% — DANGER within 3 windows]
M8_tpr_label20_overload_cyclic       : [% — WATCH via baseline drift]

───────────────────────────────────────────────────────────────────
LABEL 21 — CUSUM + LAYER 4 DETECTION (Gate M8-14-ext)
───────────────────────────────────────────────────────────────────
M8_label21_cusum_watch_rate_overall  : [% — gate ≥75%, all severities]
M8_label21_cusum_watch_rate_low      : [% — sev 0.05–0.10, expected ≥50%]
M8_label21_cusum_watch_rate_high     : [% — sev 0.10–0.25, expected ≥85%]
M8_label21_warn_rate_cusum_layer4    : [% — gate ≥60%, within 800 windows]
M8_label21_cusum_fpr_normal_pool     : [% — gate < 5% on 9711-window pool]
M8_label21_cusum_H                   : [calibrated value]
M8_label21_cusum_k                   : [calibrated value]
M8_label21_cusum_watch_windows       : [median windows to WATCH fire]
M8_label21_layer4_drift_ratio_P95    : [P95 of drift_ratio on normal pool — expected < 1.05]
M8_label21_layer4_warn_windows       : [median windows to WARN fire]
M8_gate_M8_14ext_target_A            : PASS/FAIL
M8_gate_M8_14ext_target_B            : PASS/FAIL
M8_gate_M8_14ext_target_C            : PASS/FAIL
M8_gate_M8_14ext_target_D_low        : [% — sev 0.05–0.10 WATCH rate]
M8_gate_M8_14ext_target_D_high       : [% — sev 0.10–0.25 WATCH rate]

───────────────────────────────────────────────────────────────────
GROUP E — MULTI-SENSOR DETECTION
───────────────────────────────────────────────────────────────────
M8_tpr_group_E_2ch_thermal           : [% — gate ≥88%]
M8_tpr_group_E_2ch_pumpside          : [% — gate ≥88%]

───────────────────────────────────────────────────────────────────
GLOBAL METRICS + THRESHOLDS
───────────────────────────────────────────────────────────────────
M8_fpr_full_pool                     : [% — gate < 5%, full 9711-window pool]
M8_fpr_startup_cluster               : [% — report separately]
M8_youden_j                          : [value — gate > 0.85]
M8_separation_ratio                  : [value — gate > 5.0×]
M8_fuzzy_lower_bound                 : [value — expected 0.07–0.09]
M8_fuzzy_upper_bound                 : [value — expected 0.15–0.50]
M8_fuzzy_transition_width            : [upper - lower — gate ≥0.05]
M8_rolling_watch_threshold           : [calibrated — target ~0.085]
M8_rolling_warn_threshold            : [calibrated — target ~0.095]
M8_slope_threshold                   : [calibrated — target ~0.0003/window]
M8_slow_drift_overload_watch_min     : [minutes — gate ≤15, via Temp.SV drift]
M8_slow_drift_seal_watch_min         : [minutes — gate ≤20, via Pres.SV drift]
M8_slow_drift_bearing_watch_min      : [minutes — gate ≤15, via Mot.SV drift]
M8_attention_seam_ratio              : [value — gate < 1.0 = PASS]
M8_threshold_startup                 : [value]
M8_threshold_steady_state            : [value — baseline 0.110058]
M8_threshold_high_load               : [value]
M8_threshold_cooldown                : [value]

───────────────────────────────────────────────────────────────────
GATE SUMMARY
───────────────────────────────────────────────────────────────────
M8_gate_thermal_lag                  : PASS/FAIL
M8_gate_cavitation_exclusivity       : PASS/FAIL
M8_gate_group_C_secondary_path       : PASS/FAIL
M8_gate_group_B_danger_rate          : PASS/FAIL
M8_gate_group_D_alert_path           : PASS/FAIL
M8_gate_group_E_multi_sensor         : PASS/FAIL
M8_gate_M8_14ext_all_targets         : PASS/FAIL
M8_all_15_gates_pass                 : True/False
Status_for_M9                        : READY/BLOCKED
```

---

## Module Dependency Chain

```
UPSTREAM (required before M8):
  M7_all_16_gates_pass = True              ← NON-NEGOTIABLE prerequisite
    Includes Gate M7-14-ext (label 21 slope SHAP) — must PASS
  M6B_feature_matrix.csv (~196,000×26)    ← for fuzzy boundary calibration reference
  M6B_combined_sequences.pkl              ← all Groups A–E (22 classes) for validation
  M3 normalized data (117,970 rows)        ← real CIRA normal training pool
  M4 threshold (0.110058) LOCKED           ← starting reference for steady_state
  fault_rules_v3.json (LOCKED)             ← label 0–21 map + Group E label integers
  M6.5r Gate D5 PASS                       ← label 21 err_slope_MotSV validated
  M6B Step 3 Gate G11 PASS                 ← label 21 sequence generation verified

DOWNSTREAM (M8 outputs feed into):
  M10 Flask  ← lstm_ae_v2_best.pth
             + M8_threshold_config.json (4 cluster thresholds)
             + M8_fuzzy_config.json (lower_bound, upper_bound)
             + M8_cusum_config.json (label 21 CUSUM params)
             + M8_rolling_baseline_config.json (label 21 Layer 4 params)
  M10 Flask  ← alert_state output dict is API response for /api/anomaly_detect
             + cusum_bearing_gradual_flag + drift_ratio in output dict
  M10 UI     ← four-state display: NORMAL / WATCH / WARN / DANGER
             + label 21 specific UI note (Paris–Erdogan regime message)
  M12        ← M8 model + all config files + M5 physics engine (adversarial test)
             + Gate M8-14-ext targets used as M12 adversarial baseline

FILE MANIFEST (M8 outputs to GitHub):
  models/lstm_ae_v2_best.pth
  models/M8_threshold_config.json
  models/M8_fuzzy_config.json
  models/M8_cusum_config.json              ← NEW in v14.0
  models/M8_rolling_baseline_config.json   ← NEW in v14.0
  outputs/reports/module_08_lstm_ae_v2_report.md
  All outputs/*.png files listed above
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-16 | **NEW FILE (v14.0 split)**. Extracted from `module_M8_lstm_ae_v2_gates_and_outputs.md` Part 2. Contains: Gate M8-14-ext full spec (4 targets: CUSUM WATCH rate, CUSUM+Layer4 WARN rate, CUSUM FPR on normal pool, severity stratification; fail actions in order; DO NOT list); M8 outputs extended for 22-class (Group B 6 classes, Group C 5 classes, Group D labels 18–20, label 21 CUSUM + Layer 4 plots, 2 new model config files); paste keys expanded 35→52 (Group B per-label 6 keys, Group C per-label 5 keys, Group D labels 18–20, label 21 CUSUM/Layer4 14 keys, Group E 2 keys, global 18 keys, gate summary 9 keys); dependency chain: M7_all_16_gates_pass, ~196,000×26, label 0–21, M6.5r Gate D5, M6B Gate G11, 5 M8 config files to M10, label 21 UI note, M12 adversarial baseline from M8-14-ext targets. |

---

*GitHub is the ONLY source of truth for this spec.*
*Part 1A: `module_M8_lstm_ae_v2_architecture.md`*
*Part 1B: `module_M8_lstm_ae_v2_mechanisms.md`*
*Part 2A: `module_M8_lstm_ae_v2_gates_and_outputs.md`*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
