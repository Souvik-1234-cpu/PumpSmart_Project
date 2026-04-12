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

### M6.5 Findings Applied to M7

#### Finding 1 — Overloading (thermal-dominant, Gate 3 = 0.00%)

```
mean_err_TempSV   → expected SHAP rank 1 for overloading class
err_slope_TempSV  → expected SHAP rank 2
fuzzy_fault_membership → expected SHAP rank 3

Physics: only Temp.SV and Mot.TV channels are elevated in overloading.
XGBoost must exploit thermal features — not vibration features.

SHAP PHYSICS FAIL if:
  mean_err_MotSV or mean_err_PmpSV rank ABOVE mean_err_TempSV for overloading
  → model confusing overloading with vibration fault
  → will misclassify overloading in high-ambient-temperature installations
```

#### Finding 2 — Seal Failure (slow hydraulic, Gate 3 = 29.17%)

```
err_slope_PresSV (NEGATIVE slope) → expected SHAP top-3 for seal_failure
thermal_decoupling_flag           → expected SHAP top-3 (r=-0.013 confirmed M5)
pres_monotonic_flag               → expected top-3

Physics: seal failure = monotonic pressure DECLINE + thermal decoupling.
Not a spike event — a progressive loss of hydraulic containment.

SHAP PHYSICS FAIL if:
  max_err_PresSV (spike feature) ranks ABOVE err_slope_PresSV for seal_failure
  → model confusing seal failure with cavitation (both are pressure faults)
  → cavitation is chaotic pressure, seal is monotonic pressure DECLINE
```

#### Finding 3 — Bearing Seam Discontinuity (5.75% flagged)

```
err_onset_lag → expected to discriminate bearing_wear from impeller_imbalance

Bearing wear    : gradual onset → err_onset_lag HIGH (fault develops slowly)
Impeller imbal  : immediate onset → err_onset_lag LOW (imbalance is instantaneous)

If err_onset_lag NOT in top-5 SHAP for bearing class → investigate M6.5
The seam discontinuity at t=49-50 in bearing sequences must not corrupt
onset lag computation — onset lag should reference t=0 of sequence, not seam.
```

#### Finding 4 — Fisher Rank 1 = Pmp_SV_mean

```
Pmp_SV_mean → expected SHAP rank 1 for: bearing_wear, cavitation, impeller_imbalance
Mot_SV features → expected top-3 for bearing_wear (vibration propagation motor side)

Fisher rank 1 confirms Pmp.SV is the dominant fault discriminator across all
vibration fault classes. If XGBoost SHAP contradicts Fisher rank → data quality
issue in M6.5 feature extraction → investigate before proceeding to M8.
```

#### Finding 5 — Cavitation (6.1× threshold, always acute)

```
mean_err_PmpSV      → expected SHAP rank 1 for cavitation (hydraulic shock)
thermal_decoupling_flag → expected top-3 (r=0.376 — weak thermal coupling confirmed)
kurtosis_err_PresSV → expected top-3 (chaotic pressure — NOT monotonic like seal)

Physics: cavitation = pressure chaos + pump vibration spike + no thermal coupling.
This is the key differentiator from seal failure (both are pressure faults).
  Cavitation  → kurtosis high (chaotic)  + thermal decoupling moderate
  Seal failure → kurtosis low (monotonic) + thermal decoupling strong
```

### M7 Validation Gates (10 gates)

```
GATE-M7-1 : Overall accuracy           > 85%

GATE-M7-2 : Per-class F1               > 0.80 for ALL 7 classes
             Document any class below 0.80 — do not hide

GATE-M7-3 : Cavitation F1              > 0.88 (safety-critical — hydraulic shock)
             Cavitation missed = impeller pitting within 60-180s of onset

GATE-M7-4 : Sensor failure F1          > 0.92 (single-channel flatline — easiest class)

GATE-M7-5 : SHAP physically causal     top-3 features must match physics per fault type
             (see Finding 1-5 mappings above)

GATE-M7-6 : TV dominance check         Mot.TV or Pmp.TV must NOT be top-3 SHAP
             for bearing_wear or impeller_imbalance
             (TV = placement-dependent — must not dominate vibration fault classification)

GATE-M7-7 : Overloading SHAP order     mean_err_TempSV SHAP value > mean_err_MotSV
             SHAP value for overloading class
             (thermal cause must rank above vibration — Finding 1)

GATE-M7-8 : Seal failure SHAP type     slope/monotonic feature SHAP value >
             max/spike feature SHAP value for seal_failure class
             (monotonic pressure decline, not spike — distinguishes seal from cavitation)

GATE-M7-9 : Bearing thermal lag order  err_slope_MotSV SHAP rank ABOVE mean_err_MotTV
             for bearing_wear class
             Physics: vibration rises 20-40 steps BEFORE thermal effect
             If Mot.TV ranks above Mot.SV slope → model detecting thermal consequence,
             not mechanical cause → will fail on high-ambient field installations

GATE-M7-10: Seal-cavitation confusion  < 5% of seal_failure test samples
             predicted as cavitation, and vice versa
             (both pressure faults — most likely confusion pair in deployment)
```

### Expected SHAP Top-3 Per Fault (Full Reference Table)

| Fault Class | Expected SHAP Rank 1 | Expected SHAP Rank 2 | Expected SHAP Rank 3 | Physics Basis |
|---|---|---|---|---|
| cavitation | mean_err_PmpSV | kurtosis_err_PresSV | thermal_decoupling_flag | Hydraulic shock + pressure chaos + no thermal |
| bearing_wear | Pmp_SV_mean | err_slope_MotSV | corr_delta_PmpSV_PresSV | Vibration propagation + gradual rise + coupling shift |
| seal_failure | err_slope_PresSV | thermal_decoupling_flag | pres_monotonic_flag | Monotonic pressure decline + hydraulic fault |
| overloading | mean_err_TempSV | err_slope_TempSV | fuzzy_fault_membership | Thermal dominant — only temperature channels elevated |
| impeller_imbalance | Pmp_SV_mean | err_auc_primary | err_onset_lag (LOW) | Immediate high-energy vibration onset |
| sensor_failure | max_err (single channel) | all others near 0 | err_onset_lag | Single channel isolated anomaly |
| normal | fuzzy_fault_membership (low) | all mean_err near 0 | — | Baseline reconstruction — no fault signal |

### Adaptive Actions After M7 (Before Starting M8)

| M7 Result | Trigger | Action Before Proceeding to M8 |
|---|---|---|
| Overloading F1 < 0.80 | Expected possible — thermal features only | Add `err_slope_TempSV` explicitly if missing, re-run M6.5 |
| Seal-cavitation confusion > 5% | Both pressure faults | Add `onset_speed` feature: fast onset (t<10)=cavitation, slow (t>50)=seal |
| Bearing-imbalance confusion > 5% | Both vibration faults | Verify `err_onset_lag` computed correctly in M6.5 (reference t=0 not seam) |
| SHAP TV dominance detected | Low-weight channels leaking | Flag for M8: reduce Mot.TV weight further (0.3 → 0.1) |
| Gate M7-9 fails (thermal lag wrong) | Thermal over-reliance | Flag for M8: hard constraint — vibration channels must dominate |
| Gate M7-8 fails (seal SHAP type) | Spike vs slope confusion | Verify `pres_monotonic_flag` computed over full 200-step sequence |

### M7 Outputs

```
models/xgboost_fault_classifier.json        ← cuda-trained model
models/xgboost_fault_classifier_cpu.json    ← cpu-converted for M10 deployment
outputs/M7_shap_global.png                  ← global feature importance (all classes)
outputs/M7_shap_per_class/                  ← one SHAP beeswarm plot per fault type
outputs/M7_confusion_matrix.png
outputs/M7_per_class_f1.png
outputs/reports/module_07_xgboost_report.md
```

### M7 Paste Text Keys

```
M7_accuracy                       : [%]
M7_f1_normal                      : [value]
M7_f1_cavitation                  : [value — gate > 0.88]
M7_f1_bearing_wear                : [value]
M7_f1_seal_failure                : [value]
M7_f1_overloading                 : [value — document if < 0.80]
M7_f1_impeller_imbalance          : [value]
M7_f1_sensor_failure              : [value — gate > 0.92]
M7_shap_rank1_cavitation          : [feature — expected mean_err_PmpSV]
M7_shap_rank1_bearing             : [feature — expected Pmp_SV_mean]
M7_shap_rank1_overloading         : [feature — expected mean_err_TempSV]
M7_shap_rank1_seal_failure        : [feature — expected err_slope_PresSV]
M7_gate_tv_dominance              : PASS/FAIL
M7_gate_bearing_thermal_lag       : PASS/FAIL
M7_gate_seal_cavitation_confusion : [% — gate < 5%]
M7_all_10_gates_pass              : True/False
Status_for_M8                     : READY/BLOCKED
```

---

## SECTION 3 — M8: LSTM-AE v2 PRODUCTION MODEL + FUZZY LOGIC

### Safety Mandate (NON-NEGOTIABLE)

```
Asset              : 110 kW, 7-stage, 40 bar, 450m head multistage centrifugal pump
Replacement cost   : >₹50 lakh (industrial capital asset)
Failure consequence: Process shutdown + secondary damage + personnel injury risk
Standard           : ISO 10816-3 (vibration), ISO 13373-3 (condition monitoring)

M8 is NOT an experiment. M4 was the baseline — M8 is the PRODUCTION model.
Every architectural decision must be physically justified.
Every gate must pass before M12 adversarial validation begins.
False negative on this asset = catastrophic failure.

PREREQUISITE: M7 all_10_gates_pass = True before M8 starts.
```

### M8 Architecture — All 5 Stages

---

#### STAGE 1 — LSTM-AE Reconstruction

```
Input    : (batch, 50, 8) — 50-timestep windows, 8 normalized channels
Encoder  : LSTM(8→128, layers=2, dropout=0.3) → multi-head temporal attention
           → bottleneck(128→64)
Decoder  : LSTM(64→128) → LayerNorm → output(128→8)
           Hidden state seeded from encoder bottleneck

Loss function:
  total_loss = 0.5×MAE + 0.3×MSE + 0.2×grad_penalty
  grad_penalty = mean(|dX_reconstructed/dt − dX_input/dt|)
  Physics basis: penalizes unphysical rate-of-change in reconstruction.
  Prevents model from producing smooth output for genuinely erratic input
  (critical for cavitation — highly erratic pressure signal).

Optimizer : AdamW | LR: CosineAnnealingWarmRestarts (T0=20)
AMP       : GradScaler + autocast (CUDA — RTX 4060)
MC Dropout: N=20 forward passes at inference → mean_MAE + uncertainty_std
Parameters: ~505,096 (same order as M4)

CHANNEL WEIGHTS — M6.5 Fisher rank 1 = Pmp_SV_mean VALIDATED:
  Mot.SV  = 2.5  (was 2.0 in M4) ← vibration dominant, Fisher validated
  Pmp.SV  = 2.5  (was 2.0 in M4) ← Fisher rank 1 — highest fault discriminability
  Pres.SV = 2.5  (was 2.0 in M4) ← primary seal failure + cavitation channel
  Mot.PV  = 2.0  (was 1.5 in M4)
  Pmp.PV  = 2.0  (was 1.5 in M4)
  Temp.SV = 0.5  (was 1.0 in M4) ← LOW WEIGHT — but Mech C monitors UNWEIGHTED
  Mot.TV  = 0.3  (was 0.8 in M4) ← placement-dependent — low weight
  Pmp.TV  = 0.3  (was 0.8 in M4) ← placement-dependent — low weight

CRITICAL DESIGN NOTE:
  Mech C (per-channel drift monitor) operates on RAW channel reconstruction
  error — bypasses channel weight matrix entirely.
  Temp.SV at weight 0.5 retains FULL Mech C monitoring sensitivity.
  This is the architectural design that makes overloading detectable.
  Without this: weighted MAE suppresses Temp.SV → overloading invisible.
  With this: Mech C sees unweighted Temp.SV drift → overloading_early fires.
  [Finding 1 — the solution to Gate 3 = 0.00% in M6.5]
```

---

#### STAGE 2 — Fuzzy Membership Layer

```
PURPOSE: Convert continuous MAE value to fault probability [0,1].
         Handles the transition zone between clearly normal and clearly fault.
         Used as: (a) detection component, (b) feature for M7 (fuzzy_fault_membership)

CALIBRATION PROTOCOL — MODIFIED from M6.5 Findings 1 + 2 + 6:

NORMAL population:
  Full 9711-window real CIRA normal pool
  NOT any probe subset [Finding 6 — probe 86.67% is sampling artifact]
  P95 of normal MAE distribution → lower_bound (fuzzy onset)
  Expected range: [0.07, 0.09]

FAULT population — SELECTIVE EXCLUSION (physics-derived):
  EXCLUDE: overloading mild (sev 0.2–0.5)
    Reason: MAE=0.093 sub-threshold [Finding 1] — including drags upper_bound
    down toward normal, narrows fuzzy zone, raises FPR for all other faults
    These sequences are handled by Mech C, not fuzzy layer
  EXCLUDE: seal_failure mild (sev 0.2–0.4)
    Reason: MAE near 0.12, too close to normal boundary [Finding 2]
    Same reasoning — Mech C handles these via Pres.SV drift
  INCLUDE: cavitation ALL severities         (MAE=0.675 — 6.1× threshold)
  INCLUDE: bearing_wear ALL severities
  INCLUDE: sensor_failure ALL severities
  INCLUDE: impeller_imbalance ALL severities
  INCLUDE: overloading severe (sev 0.5–1.0) only
  INCLUDE: seal_failure severe (sev 0.5–1.0) only
  P5 of included fault MAE distribution → upper_bound (fuzzy saturation)
  Expected range: [0.15, 0.50] (dominated by cavitation MAE=0.675)

FUZZY FUNCTION:
  μ_fault(e) = 0.0                                      if e < lower_bound
             = (e − lower_bound) / (upper − lower)      if lower ≤ e ≤ upper
             = 1.0                                      if e > upper_bound

WHY EXCLUSION IS CRITICAL:
  Including overloading/seal mild sequences in fault population drags
  upper_bound DOWN toward normal territory.
  Narrower fuzzy zone → more windows in transition → higher FPR.
  On a 110 kW asset, elevated FPR = operators ignore alerts = missed real faults.
  Overloading and seal mild sequences are NOT lost — Mech C catches them.
```

---

#### STAGE 3 — Slow Drift Detection: Three Mechanisms

```
LIABILITY BASIS (from v10 LIABILITY FRAMEWORK):
  Category 3 fault = slow drift = MODEL'S RESPONSIBILITY.
  A fault developing over days/weeks produces per-window MAE too small
  for single-window detection. Without these mechanisms → fault missed
  entirely → liability exposure on 110 kW asset.
  M6A severity 0.2–0.3 sequences were generated SPECIFICALLY to calibrate
  these mechanisms. This is why they exist.
```

##### MECHANISM A — Rolling Mean Gate (medium horizon: ~3 minutes)

```
  Computation:
    rolling_mean_MAE_200 = mean(MAE, last 200 windows)
    rolling_mean_MAE_100 = mean(MAE, last 100 windows)

  Thresholds (calibrated on mild severity M6 sequences):
    rolling_mean_MAE_200 > 0.085 → WATCH state
    rolling_mean_MAE_100 > 0.095 → WARN state

  Physics basis:
    200 windows = ~3 min at 1Hz sampling.
    Seal wear at severity 0.2 raises mean MAE by ~0.008/100 windows.
    Detectable in WATCH within ~10 minutes of onset.
    For a fault developing over weeks: 10 minutes = operationally instantaneous.
```

##### MECHANISM B — Slope Detector (long horizon: ~8 minutes to confirm)

```
  Computation:
    slope = linear_regression_slope(MAE_values, last 500 windows)

  Threshold:
    slope > 0.0003/window → escalate alert state by 1 level

  Physics basis:
    Bearing degradation over 8h produces slope ~0.0001–0.0005.
    At 0.0003/window threshold: 500 windows = ~8 min to confirm trend.
    For a fault developing over weeks → 8 minutes = instantaneous detection.
    Slope detector is the SECOND confirmation layer after rolling mean.
```

##### MECHANISM C — Per-Channel Drift Monitor (single-channel slow faults)

```
  Computation (for each of 8 channels independently):
    channel_error[ch] = |reconstructed[ch] − input[ch]|  (RAW — no weight matrix)
    spearman_r[ch] = spearman_correlation(channel_error[ch], time_index,
                                          last 300 consecutive windows)
    if spearman_r[ch] > 0.70 → channel_drift_flag[ch] = True

  CHANNEL → FAULT TYPE MAPPING (physics-causal):

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Temp.SV drift (spearman_r > 0.70, POSITIVE)                        │
  │ → overloading_early flag                                            │
  │ [Finding 1 — PRIMARY and ONLY reliable detection path]              │
  │ Physics: overloading = motor overheating → Temp.SV* rises monoton. │
  │ Gate M8-7: overloading TPR ≥ 80% measured via THIS flag ONLY       │
  │ NOT via single-window MAE crossing threshold                        │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Pres.SV drift (spearman_r > 0.70, NEGATIVE slope)                  │
  │ → seal_failure_early flag                                           │
  │ [Finding 2 — PRIMARY detection path for mild seal failure]          │
  │ Physics: seal failure = progressive pressure loss → Pres.SV* ↓     │
  │ Cross-check: thermal_decoupling confirmed (r=−0.013 from M5)       │
  │ Combined: Pres.SV drift + thermal_decoupling = HIGH CONFIDENCE seal │
  │ Gate M8-9: WATCH must fire ≤ 20 min via this flag                  │
  │ Gate M8-10: This flag fires BEFORE total MAE reaches WARN level     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Mot.SV drift (spearman_r > 0.70, POSITIVE)                         │
  │ → bearing_wear_early flag                                           │
  │ Physics: bearing degradation → Mot.SV* rises before Mot.TV (20-40s lag) │
  │ Thermal coupling must be preserved simultaneously (r > 0.85)       │
  │ Both: Mot.SV drift + thermal coupling preserved = bearing confirmed │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Single channel flatline                                             │
  │ std(channel_error[ch], last 100 windows) < 0.001                   │
  │ → sensor_failure flag                                               │
  │ Physics: dead sensor → channel value locked → reconstruction error  │
  │ constant → std collapses to near-zero                               │
  └─────────────────────────────────────────────────────────────────────┘

  IMPLEMENTATION MANDATE:
    Mech C MUST bypass channel weight matrix.
    Use: channel_error = abs(output[ch] - input[ch])  # before weighting
    NOT: weighted_channel_error
    Reason: Temp.SV weight=0.5 suppresses it in MAE — but Mech C needs
    raw Temp.SV error to detect overloading. This is the core fix for
    Finding 1 (Gate 3 = 0.00% in M6.5).
```

---

#### STAGE 4 — Four-State Alert Machine

```
STATE DEFINITIONS:
  NORMAL  : rolling_score < 2.0 AND no channel_drift_flag AND no slope trigger
  WATCH   : rolling_mean_200 > 0.085 OR slope trigger OR ANY channel_drift_flag
  WARN    : rolling_mean_100 > 0.095 OR rolling_score in [2.0, 3.5]
  DANGER  : single_window_MAE > cluster_threshold OR rolling_score > 3.5

STATE ESCALATION RULES:
  NORMAL → WATCH  : sustained low-level anomaly / drift beginning
  WATCH  → WARN   : trend confirmed over 100+ windows
  WARN   → DANGER : threshold crossed — immediate maintenance action required
  DANGER → WARN   : MAE below threshold for 50+ consecutive windows
  WARN   → WATCH  : rolling_mean_200 below 0.085 for 200+ windows
  WATCH  → NORMAL : ALL mechanisms clear for 300+ consecutive windows

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAVITATION EXCEPTION (Finding 5 — MAE=0.675, 6.1× threshold):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if cluster == 'startup' AND single_window_MAE > 3 × cluster_threshold:
      alert_state = DANGER  # bypass WATCH and WARN entirely
  Physics: cavitation is acute hydraulic shock — no slow escalation possible.
  Impeller pitting begins within 60–180s of onset. No time for trend detection.
  DO NOT route cavitation through rolling mean accumulator.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERLOADING EXCEPTION (Finding 1 — Gate 3 = 0.00%):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Overloading TPR measured EXCLUSIVELY via Mech C Temp.SV drift flag.
  Gate M8-7 denominator = overloading validation sequences only.
  Gate M8-7 numerator = sequences where Temp.SV drift flag fires ≤ 15 min.
  Single-window MAE threshold crossing is excluded from overloading TPR.
  Overloading mild sequences will NOT cross single-window threshold.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEAL FAILURE EXCEPTION (Finding 2 — Gate 3 = 29.17%):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Primary detection = Mech C Pres.SV drift flag (NEGATIVE Spearman)
  Secondary confirmation = rolling mean (Mech A)
  For mild seal failure (sev 0.2–0.4): Mech C fires first, then Mech A confirms.
  For severe seal failure (sev 0.5–1.0): single-window MAE also fires.
  Combined: Pres.SV drift + thermal_decoupling_flag = HIGH CONFIDENCE seal.
```

---

#### STAGE 5 — Cluster-Conditional Thresholds

```
Separate anomaly threshold per operating mode:
  startup      : threshold_startup      > 0.110058 (wider — BPF harmonics elevate MAE)
  steady_state : threshold_steady_state = 0.110058 (M4 baseline — reference)
  high_load    : threshold_high_load    ≤ 0.110058 (tighter — faults caught faster)
  cooldown     : threshold_cooldown     ≈ 0.110058 (similar to steady_state)

Calibration:
  P99 of normal MAE distribution PER CLUSTER on full 9711-window normal pool.
  Startup threshold must accommodate BPF harmonic MAE elevation without
  compromising sensitivity at steady_state or high_load.
  Store in: models/M8_threshold_config.json
```

---

### M8 Training Data Composition

```
NORMAL TRAINING POOL (model learns to reconstruct ONLY normal — never faults):
  Real CIRA normal (M3 normalized)  : 9,711 clean windows → 80% train / 20% val
  Synthetic normal (M6 Type-A)      : 1,200 sequences → windowed → ~24,000 windows
  Total normal training pool        : ~33,000 windows
  Cluster distribution maintained   : startup 42.3%, cooldown 22.8%, etc.

VALIDATION ONLY — FAULT SEQUENCES (never appear in training):
  Synthetic fault (M6)              : 7,200 fault sequences → windowed
  Purpose: calibrate thresholds + fuzzy boundaries + measure TPR/FPR + gate checks
  Mild severity (sev 0.2–0.3)       : calibrate Mech A/B thresholds
                                      (0.085, 0.095 rolling mean targets)
                                      (0.0003/window slope target)
  WHY faults never in training      : LSTM-AE is anomaly detector, not classifier.
                                      Training on faults = model learns to reconstruct
                                      faults as normal = complete failure of purpose.

MECH C CALIBRATION SUBSET:
  Use ONLY mild severity (sev 0.2–0.4) sequences for Mech C threshold tuning.
  Spearman threshold 0.70 target:
    Verify ≥ 80% of mild overloading sequences trigger Temp.SV drift ≤ 15 min
    Verify ≥ 80% of mild seal sequences trigger Pres.SV drift ≤ 20 min
    Adjust thresholds per fault type if targets not met (see adaptive actions).
```

---

### M8 All 13 Validation Gates

```
GATE-M8-1 : TPR fault detection        > 90% on fault validation set
             Report SEPARATELY per fault class (see paste keys)
             Denominator EXCLUDES overloading (Gate M8-7) and seal mild (Gate M8-9)
             Do NOT let cavitation 100% TPR mask other classes in headline number

GATE-M8-2 : FPR false alarm            < 5% on FULL 9711-window normal pool
             NOT on probe subset [Finding 6 — probe result is sampling artifact]
             Measured cluster-by-cluster: report startup FPR separately

GATE-M8-3 : Youden's J                 > 0.85 (J = TPR − FPR)

GATE-M8-4 : Separation ratio           > 5.0x (M4 baseline was 4.11x)
             = mean_fault_MAE / mean_normal_MAE
             Computed on included fault population (cavitation dominated)

GATE-M8-5 : False alarms absolute      ≤ 8 windows on normal validation pool
             (same standard as M4 — 0.55% of 1457 val windows)

GATE-M8-6 : Fuzzy boundaries valid
             lower_bound < upper_bound
             lower_bound in [0.07, 0.09]   (P95 normal — expected range)
             upper_bound in [0.15, 0.50]   (P5 fault selective — cavitation dom.)
             Transition zone width ≥ 0.05  (meaningful fuzzy region)

GATE-M8-7 : Overloading detection      ≥ 80% via Mech C Temp.SV drift flag ONLY
             Measured on mild overloading sequences (sev 0.2–0.5)
             Gate FAILS if WATCH fires via rolling mean BEFORE Temp.SV drift flag
             [Finding 1 — Mech C is PRIMARY and ONLY reliable path]
             Document if < 80% — do NOT raise global threshold to compensate

GATE-M8-8 : Attention alignment        Attention peaks at fault onset, NOT at seam
             For bearing_wear validation sequences:
             Compute: ratio = mean_attention(t=49,50) / mean_attention(t=10,40)
             Gate: ratio < 1.0  (onset region dominates over seam region)
             [Finding 3 — 5.75% seam discontinuity must not be learned as fault signal]
             If FAIL: add gradient penalty at seam timesteps, retrain M8

GATE-M8-9 : Slow drift seal detection  WATCH fires ≤ 20 min for seal sev 0.2
             Via Pres.SV Spearman drift flag (NEGATIVE slope) [Finding 2 — PRIMARY]
             thermal_decoupling_flag must ALSO be True simultaneously
             Combined flag = high-confidence seal_failure_early

GATE-M8-10: Pres.SV drift fires first  For seal_failure mild sequences:
             Pres.SV drift flag must fire BEFORE total MAE reaches WARN level
             [Finding 2 — Mech C catches it before rolling mean responds]
             Verify: timestep of Pres.SV flag < timestep of WARN state

GATE-M8-11: Thermal lag validation     For bearing_wear validation sequences:
             peak Mot.SV* reconstruction error must PRECEDE peak Mot.TV* error
             by 20–40 timesteps
             [Physics: heat conduction lag — confirmed M2 correlation + M5 Euler]
             FAIL if Mot.TV peaks BEFORE Mot.SV → model detecting thermal artifact,
             not mechanical cause → will fail on high-ambient field installations

GATE-M8-12: Cavitation cluster exclusivity
             ZERO cavitation DANGER alerts must fire on steady_state or
             high_load windows in the normal validation pool
             [Physics: cavitation requires low NPSH margin — only possible at startup]
             Any cavitation flag outside startup cluster → cluster label leak → FAIL
             Consequence of failure: operators panic-stop healthy pump under full load

GATE-M8-13: Energy conservation check on reconstruction
             Mean sum(all 8 channel reconstruction errors) for normal validation pool
             must be < 0.25
             [8 channels × 0.026765 mean_MAE × 1.1 headroom = ~0.235 ceiling]
             If exceeded → encoder bottleneck leaking fault-like signal into normal
             reconstruction → threshold will drift upward in deployment → missed faults
```

---

### M8 Inference Protocol (Production — 8-Step Output)

```
Step 1: Load cluster label → select cluster-conditional threshold from M8_threshold_config.json
Step 2: Run N=20 MC Dropout forward passes → mean_MAE + uncertainty_std
Step 3: Compute fuzzy fault membership: μ_fault(mean_MAE) from M8_fuzzy_config.json
Step 4: Update rolling accumulator (5-window): sum of last 5 μ_fault scores
Step 5: Update rolling mean MAE (200-window, Mech A) + slope detector (500-window, Mech B)
Step 6: Update per-channel Spearman drift monitor (300-window, Mech C)
        — computed on RAW channel errors, bypassing weight matrix
Step 7: Apply state machine exceptions:
        — Cavitation: startup + MAE > 3×threshold → DANGER immediately
        — Overloading: Temp.SV Spearman > 0.70 → overloading_early flag
        — Seal failure: Pres.SV Spearman > 0.70 (negative) + thermal_decoupling → seal_failure_early
        — Bearing wear: Mot.SV Spearman > 0.70 + thermal coupling preserved → bearing_wear_early
        — Sensor failure: channel std < 0.001 over 100 windows → sensor_failure flag
Step 8: Determine alert state → output dict

OUTPUT DICT (complete):
{
  alert_state          : "NORMAL" / "WATCH" / "WARN" / "DANGER"
  anomaly_flag         : True/False          (hard threshold — legacy compatibility)
  fuzzy_membership     : float [0, 1]
  rolling_mean_mae     : float               (200-window Mech A value)
  mae_slope            : float               (500-window Mech B slope)
  channel_drift        : {                   (Mech C per-channel flags)
    "Mot.SV"           : bool,
    "Pmp.SV"           : bool,
    "Pres.SV"          : bool,
    "Temp.SV"          : bool,
    "Mot.TV"           : bool,
    "Pmp.TV"           : bool,
    "Mot.PV"           : bool,
    "Pmp.PV"           : bool
  }
  early_fault_type     : None / "overloading_early" / "seal_failure_early" /
                         "bearing_wear_early" / "sensor_failure"
  severity             : "LOW" / "MEDIUM" / "HIGH"   (MC Dropout std zones)
  uncertainty_std      : float               (MC Dropout spread — confidence proxy)
  confidence           : float [0, 1]
  attention_heatmap    : array(50,)          (which timesteps drove reconstruction error)
  cluster              : "startup" / "steady_state" / "high_load" / "cooldown"
}
```

---

### Adaptive Actions After M8 (Result-Driven Recalibration)

| M8 Result | Trigger | Adaptive Action |
|---|---|---|
| Overloading TPR < 80% (Gate M8-7) | Mech C not sensitive enough | Lower Spearman threshold 0.70 → 0.65 for Temp.SV ONLY. Re-validate FPR impact |
| Seal WATCH > 20 min (Gate M8-9) | Pres.SV drift too slow to detect | Shorten Mech C window 300 → 200 for Pres.SV ONLY. Re-run Gate M8-9 |
| FPR > 5% at startup (Gate M8-2) | Startup MAE naturally elevated | Raise startup cluster threshold ONLY — never global threshold |
| Separation ratio < 5.0x (Gate M8-4) | Normal pool contaminated | Audit normal pool for near-fault windows via M4 AE error, remove, retrain |
| Attention seam ratio > 1.0 (Gate M8-8) | Seam artifact learned | Add gradient penalty at t=49–50 specifically. Retrain M8 |
| Gate M8-11 fails (thermal lag wrong) | Thermal over-reliance | Reduce Mot.TV weight 0.3 → 0.1. Force vibration-first detection. Retrain |
| Gate M8-12 fails (cavitation in high_load) | Cluster label leaking | Audit M6 cluster assignment — check startup seed mis-labeling |
| Gate M8-13 fails (energy conservation) | Encoder bottleneck unstable | Add L2 regularization on bottleneck layer (64-dim). Retrain M8 |
| Bearing-imbalance TPR similar | onset_lag not discriminating | Verify `err_onset_lag` in M6.5 uses sequence t=0, not seam as reference |
| Seal-cavitation TPR confusion | Both pressure faults leaking | Verify `thermal_decoupling_flag` computed correctly — use M5 coupling threshold |

---

### M8 Outputs

```
models/lstm_ae_v2_best.pth                  ← production model weights
models/M8_threshold_config.json             ← cluster-conditional thresholds (4 values)
models/M8_fuzzy_config.json                 ← lower_bound, upper_bound, calibration log
outputs/M8_roc_curve.png
outputs/M8_per_class_tpr.png                ← SEPARATE TPR bar per fault type [Finding 5]
outputs/M8_attention_heatmap.png            ← seam check visualization [Finding 3]
outputs/M8_mech_c_drift_plots.png           ← Spearman drift per channel (all 8)
outputs/M8_fuzzy_calibration.png            ← two-population MAE distributions with exclusions marked
outputs/M8_channel_error_dist.png           ← per-channel reconstruction error by fault type
outputs/reports/module_08_lstm_ae_v2_report.md
```

---

### M8 Paste Text Keys (28 keys)

```
M8_val_loss                       : [value]
M8_best_epoch                     : [value]
M8_tpr_overall                    : [% — gate > 90%, excludes overloading + seal mild]
M8_tpr_cavitation                 : [% — expected ~100%, Finding 5]
M8_tpr_bearing_wear               : [%]
M8_tpr_impeller_imbalance         : [%]
M8_tpr_sensor_failure             : [%]
M8_tpr_overloading                : [% — gate M8-7 ≥ 80% via Mech C Temp.SV only]
M8_tpr_seal_failure               : [% — gate M8-9 WATCH ≤ 20 min via Pres.SV drift]
M8_fpr_full_pool                  : [% — gate < 5%, measured on full 9711 windows]
M8_fpr_startup_cluster            : [% — report separately]
M8_youden_j                       : [value — gate > 0.85]
M8_separation_ratio               : [value — gate > 5.0x]
M8_fuzzy_lower_bound              : [value — P95 normal, expected 0.07–0.09]
M8_fuzzy_upper_bound              : [value — P5 fault selective, expected 0.15–0.50]
M8_fuzzy_transition_width         : [upper - lower — must be ≥ 0.05]
M8_rolling_watch_threshold        : [calibrated — target ~0.085]
M8_rolling_warn_threshold         : [calibrated — target ~0.095]
M8_slope_threshold                : [calibrated — target ~0.0003/window]
M8_slow_drift_seal_watch_min      : [minutes — gate ≤ 20, via Pres.SV drift]
M8_slow_drift_bearing_watch_min   : [minutes — gate ≤ 15, via Mot.SV drift]
M8_slow_drift_overload_watch_min  : [minutes — gate ≤ 15, via Temp.SV drift]
M8_attention_seam_ratio           : [ratio — gate < 1.0 = PASS, ≥ 1.0 = FAIL]
M8_gate_thermal_lag               : PASS/FAIL
M8_gate_cavitation_exclusivity    : PASS/FAIL
M8_gate_energy_conservation       : PASS/FAIL
M8_all_13_gates_pass              : True/False
M8_threshold_startup              : [value]
M8_threshold_steady_state         : [value — baseline 0.110058]
M8_threshold_high_load            : [value]
M8_threshold_cooldown             : [value]
Status_for_M9                     : READY/BLOCKED
```

---

## SECTION 4 — MODULE DEPENDENCY REMINDER

```
M6_feature_matrix.csv  →  M7 (XGBoost training input)
M7 SHAP validation     →  PREREQUISITE before M8 begins (gate check)
M7 XGBoost model       →  M10 Flask (fault classification route)
M8 models/configs      →  M10 Flask (anomaly detection route)
M8 alert_state output  →  M10 Flask UI (4-state condition indicator)
M8 + M5 physics engine →  M12 adversarial validation
M8 threshold_config    →  M12 cluster-conditional gate checks
M6.5 audit findings    →  embedded in EVERY gate in M7 and M8 above

SEQUENCING LAW:
  M7 all_10_gates_pass = True  →  M8 may begin
  M8 all_13_gates_pass = True  →  M12 may begin
  M12 PRODUCTION_VALIDATED     →  M10/M11 deployment approved
```

---

*File: modules_M7_M8_critical_ML.md*
*Version: 1.0 | Created: 2026-04-12*
*Derived from: module_pathway_M1_to_M12_v10.md + completed_modules_M1_to_M6p5.md*
*Next files: modules_M9_M10_M11_deployment.md | module_M12_validation_suite.md*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
