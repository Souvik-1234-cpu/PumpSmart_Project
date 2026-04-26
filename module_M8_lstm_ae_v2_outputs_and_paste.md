# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
## Part 2B of 3 — Gate M8-14-ext, Gate M8-15, Outputs, Paste Keys, Dependency Chain

| Field | Value |
|-------|-------|
| **Document version** | v2.0 — v14.2 TCN-AE score routing + Gate M8-15 + 57 paste keys |
| **Date** | 2026-04-19 |
| **Part 1A (Architecture)** | `module_M8_lstm_ae_v2_architecture.md` |
| **Part 1B (Mechanisms)** | `module_M8_lstm_ae_v2_mechanisms.md` |
| **Part 2A (Gates + Alert Machine)** | `module_M8_lstm_ae_v2_gates_and_outputs.md` |
| **Prerequisite** | M7 all 16 gates passed — `M7_all_16_gates_pass = True` |
| **Asset** | 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP) |

> **NOTE:** READ Part 1A, Part 1B, and Part 2A BEFORE this file.
> This file = Gate M8-14-ext detail + Gate M8-15 detail + outputs + paste keys + dependency chain.

> **⚠️ SCORE ROUTING REMINDER — Invariant 19 — enforced throughout this file:**
>
> | Score | Routes To |
> |-------|-----------|
> | `score_B` | CUSUM only |
> | `score_A` | Rolling Baseline only |
> | `score_C` | XGBoost only |

---

## Gate M8-14-ext — Full Specification (Label 21: `bearing_wear_gradual`)

**Gate:** Layer 3 CUSUM on `score_B` + Layer 4 Rolling Baseline on `score_A`

### Physics Justification

`bearing_wear_gradual` (label 21) operates in Paris-Erdogan low-dK regime. Severity 0.05–0.25 → per-window MAE **NEVER** crosses threshold (0.110058). MAE-based gates (M8-1 through M8-14) **DO NOT** apply to label 21. Gate M8-14-ext is the **SOLE** gate governing label 21 detection quality. This gate must PASS before M8 status = READY for M9.

> CUSUM operates on `score_B` (drift slope from TCN-AE) — NOT raw MAE.
> Layer 4 operates on `score_A` (severity from TCN-AE) — NOT raw MAE.
> (Invariant 19 — NEVER cross-route)

### Gate Targets

**Target A — CUSUM WATCH rate:**
- ≥75% of label 21 validation sequences (sev 0.05–0.25) → `cusum_bearing_gradual_flag` fires → WATCH state within **500 windows** (~8 min at 1 Hz) of simulated onset
- Measure: `count(sequences WATCH within 500w) / count(all label 21 sequences)`
- Note: CUSUM fires on `score_B` (drift slope), not raw Mot.SV MAE. Ensure `score_B` is non-zero for label 21 sequences before tuning H.

**Target B — CUSUM + Layer 4 WARN rate:**
- ≥60% of label 21 validation sequences (sev 0.05–0.25) → WARN state reached within **800 windows** (~13 min)
- Requires BOTH `cusum_bearing_gradual_flag` AND `rolling_baseline_drift_flag`
- Layer 4 operates on `score_A` (severity from TCN-AE) via `drift_ratio`
- Note: Layer 4 requires 5,000-window burn-in — measure on sequences after burn-in window only. Document burn-in-excluded count separately.

**Target C — Normal pool false-positive rate for CUSUM:**
- CUSUM does NOT fire (`S_pos` stays below H) for ≥95% of normal pool windows
- Measured on full 9,711-window CIRA normal pool
- If CUSUM fires on >5% normal windows → H is too low → increase H

**Target D — Severity stratification:**
- Report WATCH rate separately for sev 0.05–0.10 and sev 0.10–0.25

| Severity Range | Expected WATCH Rate |
|---------------|-------------------|
| sev 0.10–0.25 | ≥85% |
| sev 0.05–0.10 | ≥50% (hardest detection case) |

> Do NOT aggregate — sub-threshold severity breakdown is critical for deployment decision (sev <0.05 = not yet actionable).

### Fail Actions (in order)

| Failure | Action |
|---------|--------|
| **Target A fail** (<75% WATCH) | Lower k toward 0.3×sigma(score_B). Retune H. Re-run on label 21 mild subset. Re-check Target C (FPR). If `score_B` is near-zero for label 21: verify TCN-AE score_B head training. |
| **Target B fail** (<60% WARN) | Shorten `baseline_long` 24hr → 12hr. Lower `drift_ratio` threshold 1.10 → 1.07. Re-validate normal pool `drift_ratio` stays in [0.95, 1.05]. |
| **Target C fail** (>5% FPR on normal) | Raise H toward 6×sigma(score_B). Re-run Target A. Trade-off: higher H = lower WATCH rate = re-check Target A. |
| **All targets fail** | Increase label 21 sequences in M6B from 2,000 → 2,500. Re-run M6.5r Gate D5 first. Verify `score_B` non-zero in `z_t_sequences_groupD.pkl`. |

### ⛔ DO NOT

- Raise MAE threshold (0.110058) to detect label 21
- Apply CUSUM to other fault classes to compensate
- Merge label 21 into label 1 (`bearing_wear`) for gate convenience
- Report only aggregate WATCH rate without severity stratification
- Route `score_A` to CUSUM or `score_B` to Rolling Baseline (Invariant 19 violation)

---

## Gate M8-15 — Full Specification (score_C: TCN-AE Compound Chain Gate)

**Gate:** TCN-AE `score_C` calibration and compound chain detection quality

### Physics Justification

`score_C` (chain transition score from TCN-AE) captures the discontinuity in z_t trajectory at the compound fault transition boundary.

```
score_C = max(||z_t_reconstructed[n] - z_t_reconstructed[n-1]||_2) over N_windows
```

This signal is **ONLY** meaningful for Group B compound sequences (labels 7–12). For Group A single-fault sequences, `score_C` should remain low (no chain transition). Gate M8-15 validates that `score_C` is well-calibrated before deployment. `score_C` feeds XGBoost M7 (onset_order feature) — mis-calibration corrupts M7.

### Gate Targets

**Target A — Normal pool score_C baseline:**
- P95 of `score_C` on full 9,711-window normal pool = `score_C_normal_p95`
- `score_C_normal_p95` is used as WATCH trigger threshold in state machine (Part 2A)
- Expected: `score_C_normal_p95` should be well below Group B median `score_C`

**Target B — Group B compound detection rate:**
- ≥80% of Group B validation sequences (labels 7–12, 6 classes) show `score_C > score_C_warn_threshold` at or after `secondary_onset_lag` window
- `score_C_warn_threshold` = P5 of `score_C` on confirmed Group B phase-2 windows
- Report per compound class separately:

| Label | Compound Chain | Expected score_C Behaviour |
|-------|---------------|---------------------------|
| 7 | bearing→overloading | spike at thermal runaway onset |
| 8 | cavitation→seal | spike at Joukowsky shock transmission |
| 9 | imbalance→bearing | spike at Paris fatigue onset |
| 10 | seal→cavitation | slow accumulation then NPSHa spike |
| 11 | overloading→bearing | spike at thermal creep → bearing load |
| 12 | imbalance→cavitation | spike at BPF → bubble nucleation |

**Target C — Group A single-fault false signal rate:**
- ≤10% of Group A single-fault sequences (labels 1–6) show `score_C > score_C_warn_threshold` (spurious chain signal)
- Group A sequences have NO compound transition — `score_C` should stay low
- If >10%: `score_C` head is detecting noise as chain transitions

**Target D — Timing accuracy:**
- For ≥70% of Group B sequences that pass Target B: `score_C` first crosses `score_C_warn_threshold` within ±20 windows of actual `secondary_onset_lag` stored in `M6B_sequence_meta.csv`
- Tests whether TCN-AE correctly localises the chain transition event

### Fail Actions

| Failure | Action |
|---------|--------|
| **Target B fail** (<80% Group B detection) | Increase Group B sequences in M6B Step 1 (1,500→2,000 per class). Verify `secondary_onset_lag` features in M6.5r (`onset_order` column). Retrain TCN-AE score_C head with higher weight on compound sequences. |
| **Target C fail** (>10% Group A false signal) | Add Group A sequences to TCN-AE training as negative examples for score_C head. Increase TCN-AE score_C head dropout. Lower `score_C_warn_threshold`. |
| **Target D fail** (timing wrong) | Verify M6B `secondary_onset_lag` physics correctness (Part 2A of File 2). Check that `M6B_sequence_meta.csv` lag column matches actual signal onset in sequences. |
| **All targets fail** | Retrain TCN-AE from scratch with explicit compound chain supervision signal. Consider adding a binary `chain_transition` label to z_t training pool. |

---

## M8 Outputs (22-class aligned — v14.2)

### Model Files

```
models/lstm_ae_v2_best.pth                      ← production Level 1 model weights
models/tcn_ae_best.pth                          ← production Level 2 TCN-AE weights (NEW v14.2)
models/M8_threshold_config.json                 ← cluster-conditional thresholds (4 values)
                                                   + score_C thresholds (3 values: normal_p95, warn, danger)
models/M8_fuzzy_config.json                     ← lower_bound, upper_bound, calibration log
models/M8_cusum_config.json                     ← target, k, H, reset_policy (label 21, score_B based)
models/M8_rolling_baseline_config.json          ← window_short (6hr), window_long (24hr), drift thresholds
                                                   (score_A based — NOT raw MAE)
```

### Output Plots & Reports

```
outputs/M8_roc_curve.png
outputs/M8_tpr_group_A.png                      ← per-class TPR bar (Group A, 7 classes)
outputs/M8_tpr_group_B.png                      ← compound DANGER rate (6 classes, labels 7-12)
outputs/M8_tpr_group_C.png                      ← masked secondary TPR (5 classes, labels 13-17)
outputs/M8_tpr_group_D.png                      ← variant alert path correctness (labels 18-20)
outputs/M8_tpr_label21_cusum.png                ← CUSUM WATCH rate by severity (score_B based)
outputs/M8_tpr_label21_layer4.png               ← Rolling Baseline WARN rate (score_A based)
outputs/M8_tpr_group_E.png                      ← multi-sensor detection rate (2 classes)
outputs/M8_attention_heatmap.png                ← seam check [Finding F3]
outputs/M8_mech_c_drift_plots.png               ← Spearman drift per channel (all 8)
outputs/M8_fuzzy_calibration.png                ← two-population MAE with exclusions marked
outputs/M8_channel_error_dist.png               ← per-channel reconstruction error by fault type
outputs/M8_detection_coverage.png               ← which mechanisms detect which faults (22 rows)
outputs/M8_cusum_trajectory_label21.png         ← S_pos over time (score_B based, label 21 mild)
outputs/M8_drift_ratio_label21.png              ← short/long baseline ratio (score_A based, label 21)
outputs/M8_score_C_group_B.png                  ← score_C trajectory per compound class (NEW v14.2)
outputs/M8_score_C_calibration.png              ← score_C distribution: normal vs Group A vs Group B
outputs/M8_tcn_receptive_field.png              ← Glass diagram: 5 dilations, RF=63 windows
outputs/reports/module_08_lstm_ae_v2_report.md
```

---

## M8 Paste Text Keys (57 Keys — v14.2)

> **NOTE:** Fill AFTER M8 script runs. Do not fill in advance. Bracketed values = targets for reference while running.

### Training + Model

| Key | Target / Value |
|-----|---------------|
| `M8_val_loss` | [fill after run] |
| `M8_best_epoch` | [fill after run] |
| `M8_training_time_min` | [fill after run] |
| `M8_n_parameters_level1` | ~505,096 expected (LSTM-AE) |
| `M8_n_parameters_level2` | [fill after run — TCN-AE] |
| `M8_vram_peak_gb` | [fill after run — RTX 4060 8GB] |

### Group A — Single-Source TPR

| Key | Target |
|-----|--------|
| `M8_tpr_group_A_overall` | gate >90%, excl. overloading+seal mild |
| `M8_tpr_cavitation` | expected ~100% (Finding F5) |
| `M8_tpr_bearing_wear` | [fill] |
| `M8_tpr_impeller_imbalance` | [fill] |
| `M8_tpr_sensor_failure_single` | [fill] |
| `M8_tpr_overloading` | gate M8-7 ≥80% via Mech C Temp.SV ONLY |
| `M8_tpr_seal_failure` | gate M8-9 WATCH ≤20 min via Pres.SV drift |

### Group B — Compound TPR (6 classes, labels 7–12)

| Key | Target |
|-----|--------|
| `M8_tpr_group_B_danger_rate` | gate ≥85% reaching DANGER, all 6 classes |
| `M8_tpr_label07_bearing_overload` | [fill] |
| `M8_tpr_label08_cav_seal` | [fill] |
| `M8_tpr_label09_imbal_bearing` | [fill] |
| `M8_tpr_label10_seal_cav` | [fill — score_C NPSHa spike expected] |
| `M8_tpr_label11_overload_bearing` | [fill] |
| `M8_tpr_label12_imbal_cav` | [fill — score_C BPF spike expected] |

### Group C — Masked Fault TPR (5 classes, labels 13–17)

| Key | Target |
|-----|--------|
| `M8_tpr_group_C_overall` | gate ≥65% via secondary path |
| `M8_tpr_label13_bearing_MotSV_masked` | target ≥65% |
| `M8_tpr_label14_cav_PresSV_masked` | target ≥65% |
| `M8_tpr_label15_seal_PresSV_drifting` | target ≥65% |
| `M8_tpr_label16_overload_TempSV_stuck` | target ≥65% |
| `M8_tpr_label17_impal_PmpSV_flatline` | target ≥50% (weakest path — Pmp.PV only) |

### Group D — Severity Variant Alert Path (labels 18–20)

| Key | Target |
|-----|--------|
| `M8_tpr_label18_cav_intermittent` | burst tracking WATCH+, gate ≥78% |
| `M8_tpr_label19_seal_fast` | DANGER within 3 windows |
| `M8_tpr_label20_overload_cyclic` | WATCH via baseline drift |

### Label 21 — CUSUM + Layer 4 Detection (Gate M8-14-ext)

| Key | Target |
|-----|--------|
| `M8_label21_cusum_watch_rate_overall` | gate ≥75%, all severities — score_B based |
| `M8_label21_cusum_watch_rate_low` | sev 0.05–0.10, expected ≥50% |
| `M8_label21_cusum_watch_rate_high` | sev 0.10–0.25, expected ≥85% |
| `M8_label21_warn_rate_cusum_layer4` | gate ≥60%, within 800 windows |
| `M8_label21_cusum_fpr_normal_pool` | gate <5% on 9,711-window pool |
| `M8_label21_cusum_H` | calibrated value — starts at 5×sigma(score_B) |
| `M8_label21_cusum_k` | calibrated value — starts at 0.5×sigma(score_B) |
| `M8_label21_cusum_watch_windows` | median windows to WATCH fire |
| `M8_label21_layer4_drift_ratio_P95` | P95 of drift_ratio on normal pool, expected <1.05 |
| `M8_label21_layer4_warn_windows` | median windows to WARN fire |
| `M8_gate_M8_14ext_target_A` | PASS/FAIL |
| `M8_gate_M8_14ext_target_B` | PASS/FAIL |
| `M8_gate_M8_14ext_target_C` | PASS/FAIL |
| `M8_gate_M8_14ext_target_D_low` | [% — sev 0.05–0.10 WATCH rate] |
| `M8_gate_M8_14ext_target_D_high` | [% — sev 0.10–0.25 WATCH rate] |

### score_C — TCN-AE Compound Gate (Gate M8-15, NEW v14.2)

| Key | Target |
|-----|--------|
| `M8_score_C_normal_p95` | calibrated `score_C_normal_p95` value |
| `M8_score_C_warn_threshold` | P5 of Group B phase-2 score_C |
| `M8_score_C_group_B_detection_rate` | gate ≥80% |
| `M8_score_C_group_A_false_rate` | gate ≤10% |
| `M8_score_C_timing_accuracy` | % within ±20 windows — gate ≥70% |
| `M8_gate_M8_15_target_A` | PASS/FAIL |
| `M8_gate_M8_15_target_B` | PASS/FAIL |
| `M8_gate_M8_15_target_C` | PASS/FAIL |
| `M8_gate_M8_15_target_D` | PASS/FAIL |

### Group E — Multi-Sensor Detection

| Key | Target |
|-----|--------|
| `M8_tpr_group_E_2ch_thermal` | gate ≥88% |
| `M8_tpr_group_E_2ch_pumpside` | gate ≥88% |

### Global Metrics + Thresholds

| Key | Target |
|-----|--------|
| `M8_fpr_full_pool` | gate <5%, full 9,711-window pool |
| `M8_fpr_startup_cluster` | report separately |
| `M8_youden_j` | gate >0.85 |
| `M8_separation_ratio` | gate >5.0× |
| `M8_fuzzy_lower_bound` | expected 0.07–0.09 |
| `M8_fuzzy_upper_bound` | expected 0.15–0.50 |
| `M8_fuzzy_transition_width` | gate ≥0.05 |
| `M8_rolling_watch_threshold` | target ~0.085 |
| `M8_rolling_warn_threshold` | target ~0.095 |
| `M8_slope_threshold` | target ~0.0003/window |
| `M8_slow_drift_overload_watch_min` | gate ≤15, via Temp.SV drift Mech C |
| `M8_slow_drift_seal_watch_min` | gate ≤20, via Pres.SV drift Mech C |
| `M8_slow_drift_bearing_watch_min` | gate ≤15, via Mot.SV drift Mech C |
| `M8_attention_seam_ratio` | gate <1.0 = PASS |
| `M8_threshold_startup` | value >0.110058 |
| `M8_threshold_steady_state` | baseline 0.110058 |
| `M8_threshold_high_load` | value ≤0.110058 |
| `M8_threshold_cooldown` | value ~0.110058 |

### Gate Summary

| Key | Value |
|-----|-------|
| `M8_gate_thermal_lag` | PASS/FAIL |
| `M8_gate_cavitation_exclusivity` | PASS/FAIL |
| `M8_gate_group_C_secondary_path` | PASS/FAIL |
| `M8_gate_group_B_danger_rate` | PASS/FAIL |
| `M8_gate_group_D_alert_path` | PASS/FAIL |
| `M8_gate_group_E_multi_sensor` | PASS/FAIL |
| `M8_gate_M8_14ext_all_targets` | PASS/FAIL |
| `M8_gate_M8_15_score_C` | PASS/FAIL |
| `M8_all_16_gates_pass` | True/False |
| `Status_for_M9` | READY/BLOCKED |

> **NOTE:** Gate count is now **16** (15 original + Gate M8-15 added in v14.2). `M8_all_16_gates_pass` must be `True` before M9 starts.

---

## Module Dependency Chain

### Upstream (required before M8)

| Dependency | Details |
|-----------|---------|
| `M7_all_16_gates_pass = True` | **NON-NEGOTIABLE prerequisite.** Includes Gate M7-14-ext (label 21 slope SHAP) — must PASS. |
| `M6B_combined_sequences.pkl` (~31,800 sequences) | All Groups A–E (22 classes) for validation |
| `z_t_sequences_group[A-E].pkl` (6 files) | Level 2 TCN-AE input (NEW v14.2) |
| `M6B_sequence_meta.csv` | `secondary_onset_lag` for Gate M8-15 Target D |
| M3 normalized data (117,970 rows) | Real CIRA normal training pool |
| `lstm_ae_baseline.pth` (FROZEN) | M4 weights — Level 1 starting point (fine-tune only) |
| M4 threshold 0.110058 (LOCKED) | Level 1 steady_state reference |
| `fault_rules_v3.json` (LOCKED) | Label 0–21 map + Group E label integers + `physics_context` dicts (Invariant 18) |
| M6.5r Gate D5 PASS | Label 21 `err_slope_MotSV` validated |
| M6B Step 3 Gate G11 PASS | Label 21 sequence generation verified |
| M6B Step 3 Gate G11-ext PASS | `err_slope_MotSV > 0` in ≥95% label 21 seqs |

### Downstream (M8 outputs feed into)

| Consumer | Files / Data |
|---------|-------------|
| **M10 Flask** | `lstm_ae_v2_best.pth` (Level 1) + `tcn_ae_best.pth` (Level 2) + `M8_threshold_config.json` (4 cluster thresholds + 3 `score_C` thresholds) + `M8_fuzzy_config.json` + `M8_cusum_config.json` (label 21 CUSUM params — `score_B` based) + `M8_rolling_baseline_config.json` (label 21 Layer 4 params — `score_A` based) |
| **M10 Flask API** `/api/anomaly_detect` | `alert_state` output dict + `score_A`, `score_B`, `score_C` + `cusum_bearing_gradual_flag` + `drift_ratio` + `physics_context` dict (Invariant 18) + real-world conditions disclaimer (mandatory) |
| **M10 UI** | Four-state display: NORMAL / WATCH / WARN / DANGER + label 21 specific UI note (Paris-Erdogan regime message) |
| **M7/M10** | `score_C` fed to XGBoost inference (`onset_order` feature — Invariant 19) |
| **M12** | M8 model + all config files + M5 physics engine (adversarial test) + Gate M8-14-ext targets used as M12 adversarial baseline + Gate M8-15 `score_C` thresholds used in M12 compound chain tests |

### File Manifest (M8 outputs to GitHub — v14.2)

```
models/lstm_ae_v2_best.pth                      ← GitHub push + Spaces upload
models/tcn_ae_best.pth                          ← GitHub push (NEW v14.2)
models/M8_threshold_config.json                 ← GitHub push
models/M8_fuzzy_config.json                     ← GitHub push
models/M8_cusum_config.json                     ← GitHub push
models/M8_rolling_baseline_config.json          ← GitHub push
outputs/reports/module_08_lstm_ae_v2_report.md  ← Spaces upload
outputs/M8_*.png (19 plots)                     ← GitHub push
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-16 | NEW FILE (v14.0 split). Gate M8-14-ext full spec (4 targets). M8 outputs extended for 22-class. Paste keys expanded 35→52. Dependency chain with `M7_all_16_gates_pass`, label 0–21, M6.5r Gate D5, M6B Gate G11, 5 M8 config files to M10. |
| v2.0 | 2026-04-19 | v14.2 TCN-AE UPDATES: Gate M8-15 added (score_C calibration — 4 targets, fail actions). Gate M8-14-ext: CUSUM now operates on `score_B` (not raw MAE); Layer 4 now operates on `score_A` (not raw MAE); fail actions updated. Paste keys expanded 52→57: added `M8_n_parameters_level2`, 5 score_C gate keys, `M8_gate_M8_15_score_C`, `M8_all_16_gates_pass`. Outputs extended: `models/tcn_ae_best.pth`, `M8_score_C_group_B.png`, `M8_score_C_calibration.png`, `M8_tcn_receptive_field.png`, score_C thresholds in `M8_threshold_config.json`. Dependency chain: z_t pkl files (6), `M6B_sequence_meta.csv`, M6B Gate G11-ext, M7/M10 score_C → XGBoost, M12 Gate M8-15 thresholds, `tcn_ae_best.pth` in M10. Group C label names corrected to v14.2 canonical map (labels 13–17). |

---

> **GitHub is the ONLY source of truth for this spec.**
>
> - Part 1A: `module_M8_lstm_ae_v2_architecture.md`
> - Part 1B: `module_M8_lstm_ae_v2_mechanisms.md`
> - Part 2A: `module_M8_lstm_ae_v2_gates_and_outputs.md`
>
> **Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset
> **Standard:** ISO 10816-3 vibration, ISO 13373-3 condition monitoring
> **Canonical source of truth:** `pasted-text.txt` (v14.2, 2026-04-19)
