# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
# PART 1B OF 3 — Stage 3 Slow Drift Mechanisms, Detection Map, Training Data

**Document version:** v1.0 — 22-class M6B alignment (v14.0)
**Date:** 2026-04-16
**Part 1A (LSTM-AE + Fuzzy):** `module_M8_lstm_ae_v2_architecture.md`
**Part 2 (Gates + Outputs):** `module_M8_lstm_ae_v2_gates_and_outputs.md`
**Prerequisite:** M7 all 16 gates passed | `M7_all_16_gates_pass = True`
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)

> ⚠️ This file covers STAGE 3 only.
> READ Part 1A first for LSTM-AE architecture, channel weights, and fuzzy layer.
> READ Part 2 for gates, alert state machine, paste keys, and adaptive actions.

---

## Detection Layer Architecture (M8 Overview)

```
Layer 1 — Single-window MAE + Fuzzy Membership   ← Stage 1 + Stage 2 (Part 1A)
Layer 2 — Mech A: Rolling Mean Gate              ← Stage 3A (this file)
Layer 2 — Mech B: Slope Detector                 ← Stage 3B (this file)
Layer 2 — Mech C: Per-Channel Drift Monitor      ← Stage 3C (this file)
Layer 3 — CUSUM Accumulator                      ← Stage 3D (this file) — label 21
Layer 4 — Rolling Baseline Comparator            ← Stage 3E (this file) — label 21
Layer 5 — Alert State Machine                    ← Stage 4 (Part 2)
Layer 6 — Cluster-Conditional Thresholds         ← Stage 5 (Part 2)

Layers 3 and 4 are NEW in v14.0.
They exist SOLELY for label 21 (bearing_wear_gradual, sev 0.05–0.25).
No other fault class requires them — do NOT apply them broadly.
```

---

## STAGE 3 — Slow Drift Detection

```
LIABILITY BASIS (ISO 13374 Level 3 / LIABILITY FRAMEWORK):
  Category 3 fault = progressive degradation = MODEL’S RESPONSIBILITY.
  Fault developing over days/weeks → per-window MAE too small for threshold.
  Without these mechanisms → fault missed entirely → liability exposure.
  M6B severity 0.2–0.3 sequences generated SPECIFICALLY to calibrate these.

For label 21 (bearing_wear_gradual): severity 0.05–0.25 means MAE NEVER
crosses threshold even in severe windows. Layers 3 and 4 are mandatory.
```

---

### STAGE 3A — MECHANISM A: Rolling Mean Gate (~3 minute horizon)

```
Computation:
  rolling_mean_MAE_200 = mean(MAE, last 200 windows)
  rolling_mean_MAE_100 = mean(MAE, last 100 windows)

Thresholds (calibrated on mild-severity M6B sequences):
  rolling_mean_MAE_200 > 0.085 → WATCH state
  rolling_mean_MAE_100 > 0.095 → WARN state

Physics basis:
  200 windows = ~3 min at 1Hz sampling.
  Seal wear at severity 0.2 raises mean MAE by ~0.008/100 windows.
  Detectable in WATCH within ~10 minutes of onset.
  For a fault developing over weeks: 10 minutes = operationally instantaneous.

Calibration targets:
  Mild bearing sev 0.2–0.3 : WATCH fires ≤10 min of simulated onset
  Mild seal sev 0.2–0.3   : WATCH fires ≤15 min of simulated onset
  Normal pool: rolling_mean_MAE_200 stays below 0.085 in steady_state/high_load
  Adjust thresholds per cluster if cross-cluster contamination observed.

Label 21 (bearing_wear_gradual) — NOT detected by Mech A:
  MAE never accumulates above 0.085 rolling mean at sev 0.05–0.25.
  Mech A will NOT fire for label 21. This is CORRECT.
  Layer 3 CUSUM (Stage 3D) is the mandatory detection path.
  Do NOT lower Mech A threshold to detect label 21 — FPR will rise.
```

---

### STAGE 3B — MECHANISM B: Slope Detector (~8 minute horizon)

```
Computation:
  slope = linear_regression_slope(MAE_values, last 500 windows)

Threshold:
  slope > 0.0003/window → escalate alert state by 1 level

Physics basis:
  Bearing degradation over 8h produces slope ~0.0001–0.0005/window.
  At 0.0003/window threshold: 500 windows = ~8 min to confirm trend.
  Slope detector is the SECOND confirmation layer after rolling mean.
  Never used in isolation — always combined with Mech A or Mech C.

Group D — Severity Variant implications:
  cavitation_intermittent (label 18) → slope NOT monotonic (on-off bursts)
  overloading_cyclic (label 20)      → slope of BASELINE drift, not instantaneous MAE
    Use cyclic_baseline_drift feature from M6.5r to distinguish sawtooth from trend.
  seal_failure_fast (label 19)       → slope strongly negative, fires rapidly

Label 21 (bearing_wear_gradual) — Mech B PARTIAL signal only:
  err_slope_MotSV (the M6.5r feature) is positive and monotonic.
  However, MAE-based slope at 0.0003/window threshold will NOT fire reliably
  for sev 0.05–0.15 windows — MAE amplitude too low.
  Mech B may fire at sev 0.20–0.25 (later stage gradual wear).
  Layer 3 CUSUM (Stage 3D) is the PRIMARY detection mechanism for label 21.
  Mech B is SECONDARY confirmation — not relied upon for gate.
```

---

### STAGE 3C — MECHANISM C: Per-Channel Drift Monitor

```
Computation (for each of 8 channels independently):
  channel_error[ch] = |reconstructed[ch] − input[ch]|   ← RAW, bypasses weight matrix
  spearman_r[ch] = spearman_correlation(
      channel_error[ch], time_index, last 300 consecutive windows
  )
  if spearman_r[ch] > 0.70 → channel_drift_flag[ch] = True

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION MANDATE:
  channel_error = abs(model_output[ch] - model_input[ch])  # BEFORE weight matrix
  NOT: weighted_channel_error
  Reason: Temp.SV weight=0.5 suppresses in MAE but Mech C needs raw signal.
  Pres.SV weight=2.5 would exaggerate — raw error gives honest channel signal.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANNEL → FAULT TYPE MAPPING:
```

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Temp.SV drift  (Spearman_r > 0.70, POSITIVE trend)                     │
│ → overloading_early flag                                                │
│ [Finding F1 — PRIMARY and ONLY reliable detection path for overloading]  │
│ Physics: overloading = motor overheating → Temp.SV* rises monotonically │
│ Gate M8-7: overloading TPR ≥80% measured via THIS flag ONLY             │
│ Also fires for: overloading_cyclic (label 20) — Mech B slope            │
│ of baseline drift used to distinguish cyclic from sustained overload     │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Pres.SV drift  (Spearman_r > 0.70, NEGATIVE slope)                     │
│ → seal_failure_early flag                                               │
│ [Finding F2 — PRIMARY detection path for mild seal failure]              │
│ Physics: seal failure = progressive pressure loss → Pres.SV* ↓          │
│ Cross-check: thermal_decoupling must ALSO be True simultaneously         │
│ Gate M8-9:  WATCH fires ≤20 min via this flag                           │
│ Gate M8-10: This flag fires BEFORE total MAE reaches WARN level          │
│ Also fires for: cavitation→seal_failure (label 8) —                    │
│ Pres.SV drift begins AFTER secondary_onset_lag timesteps                 │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Mot.SV drift   (Spearman_r > 0.70, POSITIVE trend)                     │
│ → bearing_wear_early flag                                               │
│ Physics: bearing degradation → Mot.SV* rises before Mot.TV (20–40s lag) │
│ Thermal coupling must ALSO be preserved (r > 0.85) simultaneously       │
│ Both: Mot.SV drift + thermal coupling preserved = bearing confirmed      │
│ Also fires for: impeller_imbalance→bearing_wear (label 9) —             │
│ Mot.SV drift appears at secondary_onset_lag after PmpSV initial spike    │
│ Also fires for: bearing_wear→seal_failure (label 12) —                  │
│ Mot.SV drift (primary) precedes Pres.SV drift (secondary) by lag steps   │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Mot.SV VERY SLOW drift  (Spearman_r > 0.65, POSITIVE, window = 500)    │
│ → bearing_wear_gradual_early flag (label 21 SPECIFIC)                   │
│ [Finding F7 — PRIMARY Mech C path for label 21]                         │
│ Physics: Paris–Erdogan low-ΔK regime — crack growth rate sub-critical    │
│ Mot.SV* rises at ~0.002–0.005 per 100 windows (vs 0.01–0.03 for std BW) │
│ LOWER Spearman threshold (0.65 not 0.70) because signal is very weak    │
│ LONGER window (500 not 300) to accumulate enough trend signal            │
│ Layer 3 CUSUM (Stage 3D) fires BEFORE this flag at typical severities    │
│ This flag is CONFIRMATION, not primary — both together = high confidence │
│ DO NOT apply 500-window Spearman to other fault classes                  │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Single channel flatline                                                 │
│ std(channel_error[ch], last 100 windows) < 0.001                        │
│ → sensor_failure flag                                                   │
│ Physics: dead sensor → value locked → reconstruction error constant     │
│ → std collapses to near-zero                                            │
│ Group E (multi-sensor): TWO channels flatline simultaneously             │
│ multi_sensor_anomaly_count = 2 → sensor_failure_2ch variant             │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Group C — Masked Fault Detection via Secondary Path (5 classes)         │
│ When primary fault channel is flatline (masked_channel_flag = True):     │
│ bearing_wear_MotSV_masked (13)  → Mech C: Mot.TV + Temp.SV drift        │
│ cavitation_PresSV_masked (14)   → Mech C: Pmp.SV kurtosis bursts        │
│ overloading_TempSV_masked (15)  → Mech C: Mot.TV (r=0.997 coupling)     │
│ impeller_PmpSV_masked (16)      → Mech C: Pmp.PV + cross-channel        │
│ seal_failure_MotPV_masked (17)  → Mech C: Pres.SV drift (WEAKEST path)  │
│   Label 17 physics: MotPV stuck-high; only Pres.SV slow decline remains  │
│   Spearman_r threshold lowered to 0.60 for label 17 (weaker signal)      │
│   Gate M8-13 covers all 5 Group C classes. Label 17 expected weakest.    │
└───────────────────────────────────────────────────────────────────────────┘
```

---

### STAGE 3D — LAYER 3: CUSUM Accumulator (label 21 primary detector)

```
SCOPE: Label 21 (bearing_wear_gradual) ONLY.
Do NOT apply CUSUM to other fault classes — Mech A/B/C are sufficient.

PHYSICS BASIS:
  Paris–Erdogan law: da/dN = C(ΔK)^m
  At low ΔK (sub-critical regime): crack growth is sub-threshold per cycle.
  Per-window MAE never crosses threshold at sev 0.05–0.15.
  But the DIRECTION of error is consistently positive (Mot.SV* rising).
  CUSUM detects cumulative directional deviation — not magnitude.
  This is EXACTLY what sub-threshold progressive degradation looks like.

COMPUTATION:
  target  = mean(normal_pool_MAE_Mot.SV)   # computed from CIRA normal pool
  k       = 0.5 × sigma(normal_pool_MAE_Mot.SV)  # allowance = half std dev
  S_pos   = 0   # accumulator, reset to 0 at each NORMAL verdict

  for each new window w:
    x_w = channel_error_MotSV[w]           # RAW Mot.SV channel error
    S_pos = max(0, S_pos + (x_w - target - k))
    if S_pos > H:
      fire cusum_bearing_gradual_flag = True
      emit WATCH alert
      reset S_pos = 0   # or hold, see adaptive action below

  Threshold H calibration:
    Target: fires within 300–500 windows (~5–8 min) for sev 0.10 sequences
    Does NOT fire for 500 consecutive normal pool windows
    Start with H = 5 × sigma(normal_pool_MAE_MotSV)
    Tune on M6B label 21 mild sequences — adjust until gate target met

RESET POLICY:
  Reset S_pos = 0 only if 100 consecutive windows give x_w < target.
  This prevents false reset on brief normal interludes within gradual wear.
  If reset fires during a confirmed gradual wear sequence → lower k slightly.

INTEGRATION WITH ALERT STATE MACHINE (Part 2):
  cusum_bearing_gradual_flag = True → WATCH state
  cusum_bearing_gradual_flag + Mech C Mot.SV slow drift flag → WARN state
  cusum_bearing_gradual_flag + Layer 4 rolling baseline flag → WARN state
  All three simultaneously → DANGER (escalate, do not wait for MAE threshold)
```

---

### STAGE 3E — LAYER 4: Rolling Baseline Comparator (label 21 secondary detector)

```
SCOPE: Label 21 (bearing_wear_gradual) ONLY.
Secondary to Layer 3 CUSUM — confirms multi-hour trend.

PHYSICS BASIS:
  Bearing degradation over days → the BASELINE of Mot.SV channel error drifts up.
  Even if each individual window looks near-normal, the 24h mean is higher than
  the 7-day mean. This is the hallmark of Paris–Erdogan slow accumulation.

COMPUTATION:
  baseline_short = mean(channel_error_MotSV, last 1000 windows)  # ~17 min
  baseline_long  = mean(channel_error_MotSV, last 5000 windows)  # ~83 min
  drift_ratio    = baseline_short / baseline_long

  if drift_ratio > 1.10:
    fire rolling_baseline_drift_flag = True
    emit WATCH (if not already in WATCH from CUSUM)

  if drift_ratio > 1.25:
    escalate to WARN

CALIBRATION TARGET:
  Normal pool: drift_ratio stays in [0.95, 1.05] for 95% of windows
  Label 21 sev 0.10: drift_ratio crosses 1.10 within 800–1200 windows
  Label 21 sev 0.20: drift_ratio crosses 1.25 within 600–900 windows
  Adjust window lengths if targets not met.

INTEGRATION:
  rolling_baseline_drift_flag alone → WATCH (soft alert, low confidence)
  rolling_baseline_drift_flag + cusum_bearing_gradual_flag → WARN (high confidence)
  All three (CUSUM + rolling baseline + Mech C slow drift) → DANGER

NOTE: Layer 4 requires minimum 5000 windows of operational history.
  Do NOT activate at machine startup — enable after burn-in = 5000 windows.
  Use CUSUM only during burn-in period.
```

---

## M8 Detection Coverage Map (22 fault classes)

| Fault | Label | Layer 1 (MAE+Fuzzy) | Mech A (Rolling) | Mech B (Slope) | Mech C (Per-Ch) | Layer 3 (CUSUM) | Layer 4 (Baseline) |
|-------|-------|---------------------|-----------------|----------------|-----------------|-----------------|--------------------|
| normal | 0 | ✔️ below threshold | ✔️ stable | ✔️ flat | ✔️ flat | ✔️ S=0 | ✔️ ratio~1.0 |
| bearing_wear sev 0.8 | 1 | ✔️ DANGER | ✔️ | ✔️ | ✔️ Mot.SV drift | — | — |
| bearing_wear sev 0.2 | 1 | MAE≈0.098 sub-threshold | ✔️ WATCH ~10min | ✔️ ~8min | ✔️ Mot.SV drift | — | — |
| impeller_imbalance | 2 | ✔️ | ✔️ | ✔️ | ✔️ Pmp.SV | — | — |
| cavitation severe | 3 | ✔️ DANGER (bypass) | ❌ bypassed | ❌ bypassed | ✔️ Pres.SV | — | — |
| seal_failure slow | 4 | ❌ 29% windows | ✔️ WATCH ~15min | ✔️ | ✔️ Pres.SV PRIMARY | — | — |
| overloading mild | 5 | ❌ MAE≈0.093 sub-threshold | ❌ slow | ✔️ | ✔️ Temp.SV PRIMARY | — | — |
| sensor_failure | 6 | ✔️ DANGER MAE≈0.170 | ✔️ | ✔️ | ✔️ flatline std<0.001 | — | — |
| bearing_wear→overloading | 7 | ✔️ both channels | ✔️ | ✔️ | ✔️ Mot.SV+Temp.SV | — | — |
| cavitation→seal_failure | 8 | ✔️ Pmp.SV dominant | ✔️ | ✔️ | ✔️ Pmp.SV+Pres.SV | — | — |
| impeller→bearing_wear | 9 | ✔️ | ✔️ | ✔️ | ✔️ Pmp.SV then Mot.SV | — | — |
| seal_failure→cavitation | 10 | ✔️ | ✔️ | ✔️ | ✔️ Pres.SV then Pmp.SV | — | — |
| impeller→cavitation | 11 | ✔️ | ✔️ | ✔️ | ✔️ Pmp.SV kurtosis | — | — |
| bearing_wear→seal_failure | 12 | ✔️ | ✔️ | ✔️ | ✔️ Mot.SV then Pres.SV | — | — |
| bearing_wear_MotSV_masked | 13 | ❌ MotSV absent | ✔️ secondary | ✔️ | ✔️ Mot.TV+Temp.SV | — | — |
| cavitation_PresSV_masked | 14 | ❌ PresSV absent | ✔️ Pmp.SV | ✔️ | ✔️ Pmp.SV kurtosis | — | — |
| overloading_TempSV_masked | 15 | ❌ TempSV absent | ❌ | ✔️ | ✔️ Mot.TV r=0.997 | — | — |
| impeller_PmpSV_masked | 16 | ❌ PmpSV absent | ✔️ PmpPV | ✔️ | ✔️ Pmp.PV+cross-ch | — | — |
| seal_failure_MotPV_masked | 17 | ❌ MotPV stuck | ✔️ weak | ✔️ weak | ✔️ Pres.SV drift (weak) | — | — |
| cavitation_intermittent | 18 | ✔️ burst windows | ❌ not monotonic | ❌ | ✔️ Pmp.SV bursts | — | — |
| seal_failure_fast | 19 | ✔️ MAE high quickly | ✔️ rapid | ✔️ | ✔️ Pres.SV sharp drop | — | — |
| overloading_cyclic | 20 | ❌ sawtooth | ❌ ambiguous | ✔️ baseline drift | ✔️ Temp.SV cyclic | — | — |
| **bearing_wear_gradual** | **21** | **❌ sub-threshold** | **❌** | **✔️ partial (sev≥0.20)** | **✔️ Mot.SV slow (0.65, 500w)** | **✔️ PRIMARY** | **✔️ CONFIRM** |
| sensor_failure_2ch_thermal | E-a | ✔️ additive MAE | ✔️ | ✔️ | ✔️ 2× flatline | — | — |
| sensor_failure_2ch_pumpside | E-b | ✔️ additive MAE | ✔️ | ✔️ | ✔️ 2× flatline | — | — |

> ⚠️ Label 21 is the ONLY class using Layer 3 + Layer 4. Both are mandatory for Gate M8-14-ext.

---

## M8 Training Data Composition (Bias 2 Partial Fix)

```
NORMAL TRAINING POOL (model learns ONLY normal — faults never appear in training):
  Real CIRA normal (M3 normalized)   : 9,711 windows  ← 30% of effective pool
  Synthetic normal (M6B Type-A)      : from ~9,000 normal sequences windowed
  Total normal training pool         : ~33,000 windows (approx)
  Real : Synthetic ratio             : ~30% : 70%
  Cluster distribution maintained    : startup 42.3%, cooldown 22.8% etc.

BIAS 2 RATIONALE — WHY 30:70 REAL:SYNTHETIC:
  Pure synthetic training → model learns only physics-idealized normal patterns
  → too sensitive to real-world deviations from theory → elevated FPR in field.
  30% real CIRA anchors the model to actual pump behaviour:
  manufacturing tolerances, sensor placement variation, ambient noise.
  70% synthetic provides coverage of all 4 operating modes with
  controlled severity distribution (Weibull-skewed from M6B).

VALIDATION ONLY — FAULT SEQUENCES (never in training):
  All M6B fault sequences (Groups A–E, 22 classes) → windowed → validation pool
  Purpose: calibrate thresholds + fuzzy bounds + measure TPR/FPR + gate checks
  Severity 0.2–0.3 subset → calibrate Mech A/B/C rolling thresholds
  Group B compound → verify both fault channels produce high MAE
  Group C masked (5 classes, labels 13–17) → verify secondary detection path fires
  Group D variants (labels 18–20) → verify detection character matches variant type
  Label 21 mild (sev 0.05–0.15) → calibrate CUSUM threshold H and rolling baseline
  Label 21 moderate (sev 0.15–0.25) → verify Layer 4 drift_ratio crosses 1.10
  Group E multi-sensor → verify multi-channel flatline detection

WHY FAULTS NEVER IN TRAINING:
  LSTM-AE is anomaly detector, NOT classifier.
  Training on faults = model learns to reconstruct faults as normal
  = complete failure of the anomaly detection purpose.
  Faults only appear in validation to calibrate the boundary.

MECH C CALIBRATION SUBSET:
  Use ONLY mild severity (sev 0.2–0.4) sequences for Mech C tuning.
  Spearman threshold 0.70 target:
    ≥80% mild overloading sequences → Temp.SV drift fires ≤15 min
    ≥80% mild seal sequences → Pres.SV drift fires ≤20 min
  Spearman threshold 0.65 for label 17 (seal_failure_MotPV_masked):
    Target: ≥60% label 17 sequences → Pres.SV drift fires ≤25 min
    Lower target reflects weakest secondary signal path.
  Spearman threshold 0.65 / window=500 for label 21:
    Target: ≥70% label 21 sev≥0.10 sequences → slow drift fires ≤20 min
  Adjust per-channel threshold if targets not met (see Part 2 adaptive actions).

LABEL 21 CUSUM CALIBRATION SUBSET:
  Use ONLY label 21 mild (sev 0.05–0.15) for CUSUM H tuning.
  Target: CUSUM fires within 300–500 windows for sev 0.10
  Normal pool: S_pos stays below H for 500 consecutive normal windows
  Adjust k and H until both conditions simultaneously satisfied.
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-16 | **NEW FILE (v14.0 split)**. Extracted from `module_M8_lstm_ae_v2_architecture.md` Part 1. Contains: Stage 3 Mech A/B/C + detection map + training data. v14.0 additions: Mech A label 21 exclusion note; Mech B label 21 partial signal note; Mech C label 21 slow drift block (Spearman 0.65, 500w); Mech C label 12 Mot.SV→Pres.SV sequential note; Group C 5 classes (label 17 `seal_failure_MotPV_masked` added, Spearman 0.60, weakest path); Stage 3D CUSUM Layer 3 (label 21 primary); Stage 3E Rolling Baseline Layer 4 (label 21 confirm); detection map expanded to 24 rows (22 fault classes + Group E variants); training data: label 21 mild + moderate calibration subsets. |

---

*GitHub is the ONLY source of truth for this spec.*
*Part 1A (LSTM-AE + Fuzzy): `module_M8_lstm_ae_v2_architecture.md`*
*Part 2 (Gates + Outputs): `module_M8_lstm_ae_v2_gates_and_outputs.md`*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
