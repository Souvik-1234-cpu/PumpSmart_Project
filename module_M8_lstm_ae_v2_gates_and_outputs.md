# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
# PART 2 OF 2 — Gates, Inference Protocol, Outputs, Paste Keys

**Document version:** v2.0 — 21-class M6B alignment
**Date:** 2026-04-15
**Companion file:** `module_M8_lstm_ae_v2_architecture.md` (Part 1 — Architecture, Mechanisms, Detection Coverage)
**Prerequisite:** M7 all 15 gates passed | `M7_all_15_gates_pass = True`
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Status:** NOT STARTED — begins only after M7 gates confirmed

> ⚠️ READ PART 1 FIRST before reading this file.
> Part 1 = architecture + mechanisms. Part 2 = validation + deployment.

---

## STAGE 4 — Four-State Alert Machine

```
STATE DEFINITIONS:
  NORMAL  : rolling_score < 2.0  AND no channel_drift_flag AND no slope trigger
  WATCH   : rolling_mean_200 > 0.085  OR slope trigger  OR ANY channel_drift_flag
  WARN    : rolling_mean_100 > 0.095  OR rolling_score in [2.0, 3.5]
  DANGER  : single_window_MAE > cluster_threshold  OR rolling_score > 3.5

STATE ESCALATION:
  NORMAL  → WATCH  : sustained low-level anomaly / drift beginning
  WATCH   → WARN   : trend confirmed over 100+ windows
  WARN    → DANGER : threshold crossed — immediate maintenance required
  DANGER  → WARN   : MAE below threshold for 50+ consecutive windows
  WARN    → WATCH  : rolling_mean_200 below 0.085 for 200+ windows
  WATCH   → NORMAL : ALL mechanisms clear for 300+ consecutive windows

FOUR-STATE M10 UI MESSAGES:
  NORMAL  : "System operating within normal parameters"
  WATCH   : "Early anomaly trend — monitor closely"
  WARN    : "Sustained anomaly — schedule maintenance"
  DANGER  : "Fault confirmed — immediate action required"
```

### Fault-Specific Alert Exceptions

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CAVITATION EXCEPTION (Finding F5 — MAE = 0.675, 6.1× threshold):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if cluster == 'startup' AND single_window_MAE > 3 × cluster_threshold:
      alert_state = DANGER  # bypass WATCH and WARN entirely
  Physics: cavitation is acute hydraulic shock.
  Impeller pitting begins within 60–180s of onset.
  No time for rolling mean accumulation.
  DO NOT route cavitation through rolling mean accumulator.
  Gate M8-12: ZERO cavitation DANGER alerts outside startup cluster.

  cavitation_intermittent (Group D label 16):
  MAE spikes during burst windows only — drops between bursts.
  Alert escalation: WATCH on first burst, WARN after 3 bursts in 100 windows,
  DANGER if burst frequency increases (slope of burst_count > 0).
  Do NOT de-escalate to NORMAL between bursts — hold WATCH minimum.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OVERLOADING EXCEPTION (Finding F1 — Gate 3 = 0.00%):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Primary detection = Mech C Temp.SV drift (POSITIVE Spearman > 0.70)
  Gate M8-7 denominator = overloading validation sequences ONLY
  Gate M8-7 numerator = sequences where Temp.SV drift fires ≤ 15 min
  Single-window MAE crossing excluded from overloading TPR measurement
  Overloading mild sequences will NOT cross single-window threshold — by design.

  overloading_cyclic (Group D label 18):
  Temp.SV shows sawtooth with RISING BASELINE — not monotonic.
  Detection: Mech B slope of baseline_drift > 0.0002/window
  PLUS Temp.SV Spearman > 0.70 on baseline-detrended signal.
  Alert: WATCH on first cycle, WARN after baseline drift confirmed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SEAL FAILURE EXCEPTION (Finding F2 — Gate 3 = 29.17%):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Primary detection   = Mech C Pres.SV drift flag (NEGATIVE Spearman)
  Secondary confirm   = Mech A rolling mean
  Mild seal (0.2–0.4): Mech C fires first → Mech A confirms
  Severe seal (0.5+) : single-window MAE also fires
  Combined: Pres.SV drift (negative) + thermal_decoupling = HIGH CONFIDENCE seal
  Gate M8-9: Pres.SV drift WATCH ≤ 20 min for sev 0.2 sequences
  Gate M8-10: Pres.SV drift flag fires BEFORE total MAE reaches WARN state

  seal_failure_fast (Group D label 17):
  Pres.SV drops in ≤20 steps — slope extremely steep.
  Single-window MAE fires immediately — no need for rolling accumulation.
  Alert: DANGER within 1–3 windows of onset.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GROUP C — MASKED FAULT ALERT BEHAVIOUR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Masked faults: primary detection channel is flatline (sensor dead).
  Alert MUST route through secondary Mech C path.
  Max reachable alert state = WARN (not DANGER) if secondary signal only.
  DANGER requires either: (a) secondary channel MAE crosses threshold,
  OR (b) 3+ Mech C flags simultaneously active.
  M10 UI note: "Primary sensor unavailable — detection via secondary signal"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GROUP B — COMPOUND FAULT ALERT BEHAVIOUR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 1 (primary fault active): alert follows primary fault exception rules.
  Phase 2 (secondary fault onset at secondary_onset_lag):
    Additional Mech C flag fires on secondary channel → escalate alert by 1 level.
    If already at DANGER: hold DANGER, add secondary_fault_type to output dict.
  Expected: compound sequences always reach DANGER within 200 windows.
  Gate M8-14: Group B TPR ≥ 85% reaching DANGER.
```

---

## STAGE 5 — Cluster-Conditional Thresholds

```
Separate anomaly threshold per operating mode:
  startup      : threshold_startup       > 0.110058  (wider — BPF harmonics elevate MAE)
  steady_state : threshold_steady_state  = 0.110058  (M4 baseline — reference anchor)
  high_load    : threshold_high_load    ≤ 0.110058  (tighter — faults caught immediately)
  cooldown     : threshold_cooldown     ≈ 0.110058  (similar to steady_state)

Calibration:
  P99 of normal MAE distribution PER CLUSTER on full 9711-window normal pool.
  Threshold set on REAL CIRA validation set — NOT synthetic [Bias 2 fix].
  Startup threshold wider: accommodates BPF harmonic MAE elevation at pump start.
  Store all four values in: models/M8_threshold_config.json

WHY REAL CIRA ANCHORING MATTERS (Bias 2 fix):
  Synthetic sequences generated from physics equations and CIRA seeds.
  A threshold calibrated purely on synthetic data is physics-biased.
  Real CIRA validation set represents actual pump behaviour:
  manufacturing tolerances, fluid impurities, ambient conditions.
  Anchoring threshold to real data prevents systematic false-alarm drift
  when model is deployed on a pump with slightly different characteristics.
```

---

## M8 Production Inference Protocol (8-Step)

```
Step 1: Load cluster label → cluster-conditional threshold from M8_threshold_config.json
Step 2: Run N=20 MC Dropout forward passes → mean_MAE + uncertainty_std
Step 3: Compute fuzzy fault membership: μ_fault(mean_MAE)
        from M8_fuzzy_config.json [lower_bound, upper_bound]
Step 4: Update rolling accumulator (5-window): sum of last 5 μ_fault scores
Step 5: Update Mech A: rolling mean MAE (200-window + 100-window)
        Update Mech B: slope detector (500-window linear regression)
Step 6: Update Mech C: per-channel Spearman drift (300-window)
        ← computed on RAW channel errors, BYPASSING weight matrix
        Check flatline std < 0.001 for sensor_failure detection
        Check multi_sensor_anomaly_count for Group E (count ≥ 2)
Step 7: Apply fault-specific exceptions:
        — Cavitation:          startup + MAE > 3×threshold → DANGER immediately
        — Cavitation_interm:   burst tracking → WATCH→WARN on burst count
        — Overloading:         Temp.SV Spearman > 0.70 (positive) → overloading_early
        — Overloading_cyclic:  baseline drift slope → WATCH→WARN
        — Seal failure:        Pres.SV Spearman > 0.70 (negative) + thermal_decoupling
        — Seal_fast:           steep Pres.SV slope → DANGER within 3 windows
        — Bearing wear:        Mot.SV Spearman > 0.70 (positive) + coupling preserved
        — Sensor fail single:  channel std < 0.001 → sensor_failure
        — Sensor fail 2ch (E): multi_sensor_anomaly_count ≥ 2 → sensor_failure_2ch
        — Group B compound:    2nd Mech C flag at secondary_onset_lag → escalate
        — Group C masked:      max alert = WARN unless secondary MAE crosses threshold
Step 8: Determine alert state → output dict

OUTPUT DICT (complete):
{
  alert_state            : "NORMAL" / "WATCH" / "WARN" / "DANGER"
  anomaly_flag           : bool              (hard threshold — legacy compatibility)
  fuzzy_membership       : float [0, 1]
  rolling_mean_mae       : float             (200-window Mech A)
  mae_slope              : float             (500-window Mech B slope)
  channel_drift          : {
    "Mot.SV"  : bool,  "Pmp.SV"  : bool,  "Pres.SV" : bool,
    "Temp.SV" : bool,  "Mot.TV"  : bool,  "Pmp.TV"  : bool,
    "Mot.PV"  : bool,  "Pmp.PV"  : bool
  }
  early_fault_type       : None / "overloading_early" / "seal_failure_early" /
                           "bearing_wear_early" / "sensor_failure" /
                           "sensor_failure_2ch" / "compound_secondary_onset"
  secondary_fault_type   : None / string     (Group B second fault, post-lag)
  masked_detection       : bool              (True if Group C secondary path active)
  multi_sensor_count     : int               (0, 1, or 2 for Group E)
  severity               : "LOW" / "MEDIUM" / "HIGH"  (MC Dropout std zones)
  uncertainty_std        : float             (MC Dropout spread — confidence proxy)
  confidence             : float [0, 1]
  attention_heatmap      : array(50,)        (timesteps driving reconstruction error)
  cluster                : "startup" / "steady_state" / "high_load" / "cooldown"
}
```

---

## M8 All 14 Validation Gates

```
GATE-M8-1 : TPR fault detection — Group A single-source
             > 90% on Group A fault validation sequences
             Report SEPARATELY per fault class — do not aggregate
             Cavitation ~100% expected — do not let it mask other classes
             Denominator EXCLUDES overloading mild (Gate M8-7) and seal mild (Gate M8-9)

GATE-M8-2 : FPR false alarm
             < 5% on FULL 9711-window normal pool [Finding F6]
             NOT on 30-window probe subset — that result is INVALID
             Measured cluster-by-cluster: report startup FPR separately
             Startup naturally higher MAE — cluster threshold prevents false alarms

GATE-M8-3 : Youden's J
             > 0.85  (J = TPR − FPR)
             Computed on Group A fault pool vs full normal pool

GATE-M8-4 : Separation ratio
             > 5.0×  (M4 baseline was 4.11×)
             = mean_fault_MAE / mean_normal_MAE
             Computed on Group A included fault population (cavitation dominated)

GATE-M8-5 : False alarms absolute count
             ≤ 8 windows on normal validation pool
             (same standard as M4 — 0.55% of 1457 val windows)

GATE-M8-6 : Fuzzy boundaries valid
             lower_bound < upper_bound
             lower_bound in [0.07, 0.09]   (P95 normal — expected range)
             upper_bound in [0.15, 0.50]   (P5 fault selective)
             Transition zone width ≥ 0.05  (meaningful fuzzy region exists)
             If transition width < 0.05 → selective exclusion not working → audit

GATE-M8-7 : Overloading detection via Mech C ONLY
             ≥ 80% TPR on mild overloading sequences (sev 0.2–0.5)
             via Temp.SV Spearman drift flag within ≤ 15 min
             [Finding F1 — PRIMARY and ONLY reliable path]
             Gate FAILS if WATCH fires via Mech A before Temp.SV drift flag
             Applies to both overloading (label 5) and overloading_cyclic (label 18)
             Document if < 80% — do NOT raise global threshold to compensate

GATE-M8-8 : Attention seam check
             For bearing_wear validation sequences:
             seam_ratio = mean_attention(t=49,50) / mean_attention(t=10,40)
             Gate: seam_ratio < 1.0  (fault onset dominates over seam artifact)
             [Finding F3 — spike seed seam must not be learned as fault signal]
             FAIL action: add gradient penalty at t=49–50, retrain M8

GATE-M8-9 : Slow drift seal detection
             WATCH fires ≤ 20 min for seal_failure sev 0.2 sequences
             Via Pres.SV Spearman drift (NEGATIVE) [Finding F2 — PRIMARY]
             thermal_decoupling_flag must ALSO be True simultaneously
             Combined flag = high-confidence seal_failure_early
             Applies to seal_failure (label 4) only — NOT seal_failure_fast (label 17)

GATE-M8-10 : Pres.SV drift fires first
             For seal_failure mild sequences (label 4):
             timestep(Pres.SV drift flag) < timestep(WARN state)
             [Finding F2 — Mech C catches it BEFORE rolling mean responds]

GATE-M8-11 : Thermal lag validation
             For bearing_wear validation sequences:
             peak Mot.SV reconstruction error must PRECEDE peak Mot.TV error
             by 20–40 timesteps
             [Physics: heat conduction lag — M2 r=0.9793 + M5 Euler integration]
             FAIL = model detecting thermal consequence, not mechanical cause
             → will misclassify bearing wear on high-ambient installations

GATE-M8-12 : Cavitation cluster exclusivity
             ZERO cavitation DANGER alerts on steady_state or high_load windows
             in the normal validation pool
             [Physics: cavitation requires low NPSH margin — startup only]
             FAIL → audit M6B cluster assignment, check for startup seed mis-labeling
             Consequence of failure: operators panic-stop healthy pump under full load

GATE-M8-13 : Group C masked fault TPR
             ≥ 65% TPR on ALL Group C sequences (4 classes, labels 12–15)
             via secondary Mech C path ONLY (primary channel absent)
             Report per masked-class F1 individually
             FAIL on any single class < 55% → BLOCK → verify M6B Gate G10
             secondary signal strength before M8 retraining

GATE-M8-14 : Group B, D, E TPR
             Group B (compound, labels 7–11)    : ≥ 85% reaching DANGER
             Group D (variants, labels 16–18)    : ≥ 78% correct alert path
                 cavitation_intermittent: burst tracking → WATCH+
                 seal_failure_fast:      DANGER within 3 windows
                 overloading_cyclic:     WATCH via baseline drift Mech B+C
             Group E (multi-sensor, labels 19–20) : ≥ 88% multi_sensor_count=2 detected
             Report each group separately — do NOT aggregate into single TPR
             FAIL on any group → document in paste text, flag for M12 adversarial
```

---

## Adaptive Actions After M8

| M8 Result | Gate | Adaptive Action |
|-----------|------|-----------------|
| Overloading TPR < 80% | M8-7 | Lower Spearman threshold 0.70→0.65 for Temp.SV ONLY. Re-validate FPR impact |
| Seal WATCH > 20 min | M8-9 | Shorten Mech C window 300→200 for Pres.SV ONLY. Re-run Gate M8-9 |
| FPR > 5% at startup | M8-2 | Raise startup cluster threshold ONLY — never global threshold |
| Separation ratio < 5.0× | M8-4 | Audit normal pool via M4 AE error, remove near-fault windows, retrain |
| Attention seam ratio > 1.0 | M8-8 | Add gradient penalty at t=49–50 specifically. Retrain M8 |
| Gate M8-11 fails (thermal lag) | M8-11 | Reduce Mot.TV weight 0.3→0.1. Force vibration-first detection. Retrain |
| Gate M8-12 fails (cavitation in high_load) | M8-12 | Audit M6B cluster assignment — startup seed mis-labeling |
| Gate M8-13 fails (masked TPR < 55% any class) | M8-13 | Verify M6B Gate G10 secondary signal strength. Increase masked sequences 1200→2000 |
| Group B TPR < 85% | M8-14 | Increase compound sequences in M6B Step 1. Verify secondary_onset_lag in M6.5r |
| Group D cavitation_intermittent not WATCH | M8-14 | Implement burst_count tracker in Step 7 inference. Verify D16 sequences in M6B |
| Group D overloading_cyclic WATCH misses | M8-14 | Implement baseline_detrend in Mech B for cyclic signal. Verify D18 sequences |
| Group E multi-sensor not detected | M8-14 | Verify multi_sensor_anomaly_count feature in M6.5r Gate D3 |
| Energy conservation fail | M8-13 | Add L2 regularization on 64-dim bottleneck layer, retrain |
| Compound sequences sub-threshold | — | Investigate M6B causal lag — secondary fault onset may be too mild |

---

## M8 Outputs

```
models/lstm_ae_v2_best.pth                    ← production model weights
models/M8_threshold_config.json               ← cluster-conditional thresholds (4 values)
models/M8_fuzzy_config.json                   ← lower_bound, upper_bound, calibration log
outputs/M8_roc_curve.png
outputs/M8_tpr_group_A.png                    ← per-class TPR bar (Group A, 7 classes)
outputs/M8_tpr_group_B.png                    ← compound pair DANGER rate (5 classes)
outputs/M8_tpr_group_C.png                    ← masked fault secondary TPR (4 classes)
outputs/M8_tpr_group_D.png                    ← variant alert path correctness (3 classes)
outputs/M8_tpr_group_E.png                    ← multi-sensor detection rate (2 classes)
outputs/M8_attention_heatmap.png              ← seam check visualization [Finding F3]
outputs/M8_mech_c_drift_plots.png             ← Spearman drift per channel (all 8)
outputs/M8_fuzzy_calibration.png              ← two-population MAE with exclusions marked
outputs/M8_channel_error_dist.png             ← per-channel reconstruction error by fault type
outputs/M8_detection_coverage.png             ← which mechanisms detect which faults
outputs/reports/module_08_lstm_ae_v2_report.md
```

---

## M8 Paste Text Keys (35 Keys)

```
M8_val_loss                          : [value]
M8_best_epoch                        : [value]
M8_tpr_group_A_overall               : [% — gate > 90%, excl. overloading+seal mild]
M8_tpr_cavitation                    : [% — expected ~100%, Finding F5]
M8_tpr_bearing_wear                  : [%]
M8_tpr_impeller_imbalance            : [%]
M8_tpr_sensor_failure_single         : [%]
M8_tpr_overloading                   : [% — gate M8-7 ≥80% via Mech C Temp.SV ONLY]
M8_tpr_seal_failure                  : [% — gate M8-9 WATCH ≤20 min via Pres.SV drift]
M8_tpr_group_B_danger_rate           : [% — gate ≥85% reaching DANGER]
M8_tpr_group_B_bearing_overload      : [%]
M8_tpr_group_B_cav_seal              : [%]
M8_tpr_group_B_imbal_bearing         : [%]
M8_tpr_group_B_seal_cav              : [%]
M8_tpr_group_B_imbal_cav             : [%]
M8_tpr_group_C_overall               : [% — gate ≥65% via secondary path]
M8_tpr_group_C_bearing_masked        : [%]
M8_tpr_group_C_cav_masked            : [%]
M8_tpr_group_C_overload_masked       : [%]
M8_tpr_group_C_imbal_masked          : [%]
M8_tpr_group_D_cav_intermittent      : [% — burst tracking correct]
M8_tpr_group_D_seal_fast             : [% — DANGER within 3 windows]
M8_tpr_group_D_overload_cyclic       : [% — baseline drift detected]
M8_tpr_group_E_2ch_thermal           : [% — gate ≥88%]
M8_tpr_group_E_2ch_pumpside          : [% — gate ≥88%]
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
M8_gate_thermal_lag                  : PASS/FAIL
M8_gate_cavitation_exclusivity       : PASS/FAIL
M8_gate_group_C_secondary_path       : PASS/FAIL
M8_gate_group_B_danger_rate          : PASS/FAIL
M8_gate_group_D_alert_path           : PASS/FAIL
M8_gate_group_E_multi_sensor         : PASS/FAIL
M8_threshold_startup                 : [value]
M8_threshold_steady_state            : [value — baseline 0.110058]
M8_threshold_high_load               : [value]
M8_threshold_cooldown                : [value]
M8_all_14_gates_pass                 : True/False
Status_for_M9                        : READY/BLOCKED
```

---

## Module Dependency Chain

```
UPSTREAM (required before M8):
  M7_all_15_gates_pass = True              ← NON-NEGOTIABLE prerequisite
  M6B_feature_matrix.csv (~189,000×26)    ← for fuzzy boundary calibration reference
  M6B_combined_sequences.pkl              ← all Groups A–E for fault validation pool
  M3 normalized data (117,970 rows)        ← real CIRA normal training pool
  M4 threshold (0.110058) LOCKED           ← starting reference for steady_state
  fault_rules_v3.json (LOCKED)             ← label 0–20 map, group assignments

DOWNSTREAM (M8 outputs feed into):
  M10 Flask  ← lstm_ae_v2_best.pth + M8_threshold_config.json + M8_fuzzy_config.json
  M10 Flask  ← alert_state output dict is API response for /api/anomaly_detect route
  M10 Flask  ← secondary_fault_type + masked_detection + multi_sensor_count in UI
  M10 UI     ← four-state display: NORMAL / WATCH / WARN / DANGER
  M12        ← M8 model + M4 config + M3 config + M5 physics engine (adversarial test)
```

---

## Cross-Module Invariants Relevant to M8

1. Models saved: `torch.save(state_dict)` | Loaded: `map_location='cpu'` for M10
2. Normalization baselines LOCKED at `M3_normalization_config.json`
3. M4 threshold `0.110058` is starting reference — M8 produces its own cluster-conditional thresholds
4. Channel weights INCREASED vs M4 — Fisher-validated from M6.5
5. Faults NEVER in training pool — LSTM-AE is anomaly detector only
6. Mech C operates on RAW channel errors — bypasses weight matrix by design
7. Threshold calibrated on REAL CIRA validation set — not synthetic
8. All M6B Groups (A–E) in fault validation pool only — never in training
9. Cavitation gate: startup cluster only — any cavitation DANGER outside startup = FAIL
10. `if pump_type == 'household': return physics_advisory_only()` — NO EXCEPTIONS
11. Label strings always resolved via `fault_rules_v3.json` — NEVER hardcoded
12. Group C masked fault max alert = WARN unless secondary MAE crosses threshold
13. Group B compound: second Mech C flag fires at secondary_onset_lag — not before
14. M8 outputs raw alert_state dict — M10 handles all UI display formatting

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic file — 13 gates, bias-audit updates |
| v2.0 | 2026-04-15 | **SPLIT into Part 1 + Part 2.** Part 2 = Gates + Outputs (this file). Updated: 13→14 gates (Gate M8-14 added for Group B/D/E TPR), Group C gate M8-13 explicit, compound + masked + variant alert exceptions added to Stage 4, output dict extended with secondary_fault_type + masked_detection + multi_sensor_count, 31→35 paste keys, dependency chain updated to fault_rules_v3.json + M6B_combined_sequences. v1.0 monolithic file converted to redirect stub. |

---

*GitHub is the ONLY source of truth for this spec.*
*Do NOT reference any Spaces .md pathway files — all outdated.*
*Companion: `module_M8_lstm_ae_v2_architecture.md` (Part 1)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
