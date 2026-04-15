# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
# PART 1 OF 2 — Architecture, Mechanisms, Detection Coverage

**Document version:** v2.0 — 21-class M6B alignment
**Date:** 2026-04-15
**Companion file:** `module_M8_lstm_ae_v2_gates_and_outputs.md` (Part 2 — Gates, Outputs, Paste Keys)
**Prerequisite:** M7 all 15 gates passed | `M7_all_15_gates_pass = True`
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Status:** NOT STARTED — begins only after M7 gates confirmed

> ⚠️ READ BOTH PARTS BEFORE WRITING ANY M8 CODE.
> Part 1 = what the model does. Part 2 = how it is validated and deployed.

---

## Safety Mandate (NON-NEGOTIABLE)

```
Asset              : 110 kW, 7-stage, 40 bar, 450m head multistage centrifugal pump
Replacement cost   : >₹50 lakh (industrial capital asset)
Failure consequence: Process shutdown + secondary damage + personnel injury risk
Standard           : ISO 10816-3 (vibration), ISO 13373-3 (condition monitoring)
Pipeline level     : ISO 13374 Level 3 condition monitoring

M8 is NOT an experiment. M4 was the baseline — M8 is the PRODUCTION model.
Every architectural decision must be physically justified.
Every gate must pass before M12 adversarial validation begins.
False negative on this asset = catastrophic failure.

PREREQUISITE: M7_all_15_gates_pass = True before M8 starts.
If M7 gates fail → fix M6.5r → rerun M7 → only then start M8.
```

---

## v1.0 → v2.0 Change Summary

| Item | v1.0 (OLD — INVALID) | v2.0 (CURRENT — USE THIS) | Reason |
|------|----------------------|--------------------------|--------|
| M7 prerequisite gate count | 10 gates | **15 gates (per-group A–E)** | M7 updated to 21-class |
| Fault validation pool | M6A 7200 + M6B 1600 compound | **M6B_combined_sequences: ~27,000 sequences windowed** | M6B expanded dataset |
| Compound TPR tracking | 2 compound pairs | **Per-group: B (5 classes), C (4), D (3), E (2)** | 21-class alignment |
| Upstream reference | `M6_feature_matrix.csv` 10000×29 | **`M6B_feature_matrix.csv` ~189,000×26** | M6.5r output |
| Training data ratio | 30% real + 70% M6A synthetic | **30% real CIRA + 70% M6B synthetic normal** | M6B replaces M6A |
| Gate count | 13 gates | **14 gates** (Gate M8-14 added: Group D/E TPR) | New fault groups |
| Alert state machine | 4 states | **4 states UNCHANGED** — NORMAL/WATCH/WARN/DANGER | No change needed |

> v1.0 is INVALID. Do not reference v1.0 gate numbers or upstream file names.

---

## M6.5 Audit Findings — Critical Inputs for M8 Design

**Read before writing a single line of M8 code. Every finding directly constrains
threshold calibration, channel weighting, and detection strategy.**

| Finding | Root Cause | M8 Action |
|---------|-----------|----------|
| **F1** Overloading Gate 3 = 0.00% (MAE=0.093) | Temp.SV weight=0.5 → sub-threshold weighted MAE | Mech C Temp.SV drift = PRIMARY detection path. Gate M8-7: ≥80% TPR via Mech C ONLY |
| **F2** Seal failure Gate 3 = 29.17% (MAE=0.196) | Pres.SV gradual decline — per-window MAE low | Mech C Pres.SV drift (negative Spearman) = PRIMARY. Gate M8-9: WATCH ≤20 min |
| **F3** Bearing seam discontinuity 5.75% | Spike seed t=49→50 step change | Attention must NOT peak at seam. Gate M8-8: seam_ratio < 1.0 |
| **F4** Fisher rank 1 = PmpSV_mean | Pmp.SV dominant fault channel | Weight increase 2.0→2.5 Fisher-validated |
| **F5** Cavitation MAE = 0.675 (6.1×) | Hydraulic shock — always acute | Bypass WATCH/WARN → DANGER immediately at startup |
| **F6** Normal probe 86.67% NOT FPR problem | Edge-case probe sampling artifact | Gate M8-2 on full 9711-window pool ONLY — never on 30-window probe |

---

## M8 Architecture — Stages 1, 2, 3

*(Stage 4 — Alert Machine and Stage 5 — Cluster Thresholds are in Part 2)*

---

### STAGE 1 — LSTM-AE Reconstruction

```
Input    : (batch, 50, 8) — 50-timestep windows, 8 normalized channels
           Window size = 50 (M2 optimal, M6.5r fixed — NEVER change)
Encoder  : LSTM(8→128, layers=2, dropout=0.3)
           → Multi-head temporal attention over encoder outputs
           → Bottleneck(128→64)
Decoder  : LSTM(64→128) → LayerNorm → Output(128→8)
           Hidden state seeded from encoder bottleneck

Loss function (3-component — physics-weighted):
  total_loss = 0.5×MAE + 0.3×MSE + 0.2×grad_penalty

  grad_penalty = mean(|dX_reconstructed/dt − dX_input/dt|)
  Physics basis: penalizes unphysical rate-of-change in reconstruction.
  Critical for cavitation — highly erratic pressure signal.
  Without grad_penalty: model produces smooth output for erratic cavitation input
  → reconstruction error underestimated → cavitation MAE artificially lowered.

Optimizer : AdamW
Scheduler : CosineAnnealingWarmRestarts (T0=20)
AMP       : GradScaler + autocast (CUDA — RTX 4060 Laptop)
MC Dropout: N=20 forward passes at inference → mean_MAE + uncertainty_std
Parameters: ~505,096 (same order as M4 — no architecture bloat)
```

#### Channel Weights — Fisher Validated from M6.5

```
Channel     M4 Weight   M8 Weight   Reason
─────────────────────────────────────────────────────────────────────
Mot.SV      2.0         2.5         Fisher rank 2 confirmed — vibration dominant
Pmp.SV      2.0         2.5         Fisher rank 1 confirmed — HIGHEST discriminability
Pres.SV     2.0         2.5         Primary seal + cavitation channel
Mot.PV      1.5         2.0         Displacement — secondary vibration
Pmp.PV      1.5         2.0         Displacement — BPF harmonics
Temp.SV     1.0         0.5         LOW WEIGHT — but Mech C monitors UNWEIGHTED
Mot.TV      0.8         0.3         Placement-dependent — low weight
Pmp.TV      0.8         0.3         Placement-dependent — low weight

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DESIGN NOTE — WHY Temp.SV WEIGHT IS LOW BUT STILL DETECTABLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temp.SV weight = 0.5 → weighted MAE contribution suppressed.
Overloading raises ONLY thermal channels → weighted MAE stays sub-threshold.
This is EXPECTED and CORRECT behaviour (Finding 1 from M6.5).

Mech C (Stage 3C) operates on RAW channel reconstruction error,
BYPASSING the weight matrix entirely.
Temp.SV at weight 0.5 retains FULL Mech C monitoring sensitivity.

Without this design: overloading invisible to model.
With this design:    Mech C sees unweighted Temp.SV drift → overloading_early fires.
This is the architectural solution to Gate 3 = 0.00% in M6.5.
```

---

### STAGE 2 — Fuzzy Membership Layer

```
PURPOSE:
  Convert continuous MAE value to fault probability [0, 1].
  Handles the transition zone between clearly normal and clearly fault.
  Used as: (a) primary detection component, (b) feature for M7 (fuzzy_fault_membership)
  Captures early-stage faults where MAE hovers near threshold — hard threshold misses.

CALIBRATION PROTOCOL:

  NORMAL POPULATION:
    Full 9711-window real CIRA normal pool [Finding 6 — never use probe subset]
    P95 of normal MAE distribution → lower_bound (fuzzy onset)
    Expected range: [0.07, 0.09]

  FAULT POPULATION — SELECTIVE EXCLUSION (physics-derived):
    EXCLUDE: overloading mild (sev 0.2–0.5)
      Reason: MAE=0.093 sub-threshold [Finding 1]
      Including drags upper_bound toward normal → narrows fuzzy zone → raises FPR
      These sequences handled by Mech C, NOT fuzzy layer
    EXCLUDE: seal_failure mild (sev 0.2–0.4)
      Reason: MAE near 0.12, too close to normal boundary [Finding 2]
      Same reasoning — Mech C handles via Pres.SV drift
    INCLUDE: cavitation ALL severities         (MAE = 0.675)
    INCLUDE: bearing_wear ALL severities
    INCLUDE: sensor_failure ALL severities
    INCLUDE: impeller_imbalance ALL severities
    INCLUDE: overloading severe (sev 0.5–1.0) only
    INCLUDE: seal_failure severe (sev 0.5–1.0) only
    INCLUDE: ALL Group B compound sequences (M6B)
    INCLUDE: ALL Group C masked sequences (M6B)
    INCLUDE: ALL Group D severity variant sequences (M6B)
    INCLUDE: ALL Group E multi-sensor sequences (M6B)
    P5 of included fault MAE distribution → upper_bound
    Expected range: [0.15, 0.50] (dominated by cavitation MAE=0.675)

  FUZZY FUNCTION:
    μ_fault(e) = 0.0                                      if e < lower_bound
               = (e − lower_bound) / (upper − lower)      if lower ≤ e ≤ upper
               = 1.0                                      if e > upper_bound

  WHY EXCLUSION IS CRITICAL:
    Including overloading/seal mild sequences in fault population drags
    upper_bound DOWN toward normal territory.
    Narrower fuzzy zone → more windows in transition → higher FPR.
    On a 110 kW asset: elevated FPR = operators ignore alerts = missed real faults.
    Overloading and seal mild sequences are NOT lost — Mech C catches them.
    This exclusion is physics-driven, not ad-hoc.

  GROUP B–E SEQUENCES IN FUZZY CALIBRATION:
    Compound (Group B): both fault channels active → MAE well above threshold
    Masked (Group C): secondary path only → MAE moderate but above lower_bound
    Variants (Group D): follow base fault MAE character
    Multi-sensor (Group E): 2-channel anomaly → MAE additive → clearly above threshold
    All included in fault population for upper_bound → P5 pull toward realistic minimum
```

---

### STAGE 3 — Slow Drift Detection: Three Mechanisms

```
LIABILITY BASIS (ISO 13374 Level 3 / LIABILITY FRAMEWORK):
  Category 3 fault = progressive degradation = MODEL'S RESPONSIBILITY.
  Fault developing over days/weeks → per-window MAE too small for threshold.
  Without these mechanisms → fault missed entirely → liability exposure.
  M6B severity 0.2–0.3 sequences generated SPECIFICALLY to calibrate these.
```

#### MECHANISM A — Rolling Mean Gate (medium horizon: ~3 minutes)

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

Calibration target:
  Mild bearing sev 0.2–0.3: WATCH fires ≤ 10 min of simulated onset
  Mild seal sev 0.2–0.3:    WATCH fires ≤ 15 min of simulated onset
  Normal pool: rolling_mean_MAE_200 stays below 0.085 in steady_state/high_load
  Adjust thresholds per cluster if cross-cluster contamination observed.
```

#### MECHANISM B — Slope Detector (long horizon: ~8 minutes to confirm)

```
Computation:
  slope = linear_regression_slope(MAE_values, last 500 windows)

Threshold:
  slope > 0.0003/window → escalate alert state by 1 level

Physics basis:
  Bearing degradation over 8h produces slope ~0.0001–0.0005/window.
  At 0.0003/window threshold: 500 windows = ~8 min to confirm trend.
  For a fault developing over weeks → 8 minutes = instantaneous detection.
  Slope detector is the SECOND confirmation layer after rolling mean.
  Never used in isolation — always combined with Mech A or Mech C.

  GROUP D — Severity Variant implication:
  cavitation_intermittent → slope NOT monotonic (on-off bursts)
  overloading_cyclic      → slope of BASELINE drift (not instantaneous MAE)
  Use cyclic_baseline_drift feature from M6.5r to distinguish sawtooth from trend.
```

#### MECHANISM C — Per-Channel Drift Monitor (single-channel slow faults)

```
Computation (for each of 8 channels independently):
  channel_error[ch] = |reconstructed[ch] − input[ch]|   ← RAW, bypasses weight matrix
  spearman_r[ch] = spearman_correlation(
      channel_error[ch], time_index, last 300 consecutive windows
  )
  if spearman_r[ch] > 0.70 → channel_drift_flag[ch] = True

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION MANDATE:
  channel_error = abs(model_output[ch] - model_input[ch])  # BEFORE weight matrix
  NOT: weighted_channel_error
  Reason: Temp.SV weight=0.5 suppresses in MAE but Mech C needs raw signal.
  Pres.SV weight=2.5 would exaggerate — raw error gives honest channel signal.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANNEL → FAULT TYPE MAPPING:

┌─────────────────────────────────────────────────────────────────────────┐
│ Temp.SV drift  (Spearman_r > 0.70, POSITIVE trend)                     │
│ → overloading_early flag                                                │
│ [Finding 1 — PRIMARY and ONLY reliable detection path]                  │
│ Physics: overloading = motor overheating → Temp.SV* rises monotonically │
│ Gate M8-7: overloading TPR ≥ 80% measured via THIS flag ONLY           │
│ NOT via single-window MAE crossing threshold                            │
│ Also fires for: overloading_cyclic (Group D) — but Mech B slope        │
│ of baseline drift used to distinguish cyclic from sustained overload    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Pres.SV drift  (Spearman_r > 0.70, NEGATIVE slope)                     │
│ → seal_failure_early flag                                               │
│ [Finding 2 — PRIMARY detection path for mild seal failure]              │
│ Physics: seal failure = progressive pressure loss → Pres.SV* ↓         │
│ Cross-check: thermal_decoupling must ALSO be True simultaneously        │
│ Combined: Pres.SV drift (negative) + thermal_decoupling = HIGH CONF.   │
│ Gate M8-9:  WATCH fires ≤ 20 min via this flag                         │
│ Gate M8-10: This flag fires BEFORE total MAE reaches WARN level         │
│ Also fires for: cavitation→seal_failure (Group B label 8) —            │
│ Pres.SV drift begins AFTER secondary_onset_lag timesteps                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Mot.SV drift   (Spearman_r > 0.70, POSITIVE trend)                     │
│ → bearing_wear_early flag                                               │
│ Physics: bearing degradation → Mot.SV* rises before Mot.TV (20–40s lag)│
│ Thermal coupling must ALSO be preserved (r > 0.85) simultaneously      │
│ Both: Mot.SV drift + thermal coupling preserved = bearing confirmed     │
│ Also fires for: impeller_imbalance→bearing_wear (Group B label 9) —    │
│ Mot.SV drift appears at secondary_onset_lag after PmpSV initial spike   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Single channel flatline                                                 │
│ std(channel_error[ch], last 100 windows) < 0.001                       │
│ → sensor_failure flag                                                   │
│ Physics: dead sensor → value locked → reconstruction error constant     │
│ → std collapses to near-zero                                            │
│ Group E (multi-sensor): TWO channels flatline simultaneously            │
│ multi_sensor_anomaly_count = 2 → sensor_failure_2ch variant            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Group C — Masked Fault Detection via Secondary Path                     │
│ When primary fault channel is flatline (masked_channel_flag = True):    │
│ bearing_wear_MotSV_masked   → Mech C monitors Mot.TV + Temp.SV drift   │
│ cavitation_PresSV_masked    → Mech C monitors Pmp.SV kurtosis          │
│ overloading_TempSV_masked   → Mech C monitors Mot.TV (r=0.997 coupling)│
│ impeller_PmpSV_masked       → Mech C monitors Pmp.PV + cross-channel   │
│ These are the HARDEST detection cases. Gate M8-13 covers Group C TPR.  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## M8 Detection Coverage Map

| Fault | Single Window | Mech A Rolling | Mech B Slope | Mech C Per-Channel |
|-------|--------------|----------------|--------------|--------------------|
| Bearing wear sev 0.8 | ✅ DANGER | ✅ | ✅ | ✅ Mot.SV drift |
| Bearing wear sev 0.2 | MAE≈0.098 sub-threshold | ✅ WATCH ~10min | ✅ ~8min | ✅ Mot.SV drift |
| Cavitation severe | ✅ DANGER immediately | ❌ bypassed | ❌ bypassed | ✅ Pres.SV |
| Cavitation intermittent (D) | ✅ burst windows | ❌ slope not monotonic | ❌ | ✅ Pmp.SV kurtosis bursts |
| Seal failure slow | ❌ 29% windows | ✅ WATCH ~15min | ✅ | ✅ **Pres.SV PRIMARY** |
| Seal failure fast (D) | ✅ MAE high quickly | ✅ rapid | ✅ | ✅ Pres.SV sharp drop |
| Overloading mild | ❌ MAE≈0.093 sub-threshold | ❌ slow | ✅ | ✅ **Temp.SV PRIMARY** |
| Overloading cyclic (D) | ❌ sawtooth | ❌ slope ambiguous | ✅ baseline drift | ✅ Temp.SV cyclic pattern |
| Sensor failure single | ✅ DANGER MAE≈0.170 | ✅ | ✅ | ✅ flatline std<0.001 |
| Sensor failure 2ch (E) | ✅ additive MAE | ✅ | ✅ | ✅ 2× flatline count |
| Compound bearing+seal (B) | ✅ both channels elevate | ✅ | ✅ | ✅ Mot.SV + Pres.SV drift |
| Compound cavitation+imbal (B) | ✅ Pmp.SV dominant | ✅ | ✅ | ✅ Pmp.SV spikes |
| Bearing masked (C) | ❌ MotSV absent | ✅ secondary | ✅ | ✅ Mot.TV secondary path |
| Cavitation masked (C) | ❌ PresSV absent | ✅ Pmp.SV | ✅ | ✅ Pmp.SV kurtosis |
| Overloading masked (C) | ❌ TempSV absent | ❌ | ✅ | ✅ Mot.TV (r=0.997) |
| Impeller masked (C) | ❌ PmpSV absent | ✅ PmpPV | ✅ | ✅ Pmp.PV + cross-ch |

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
  All M6B fault sequences (Groups A–E) → windowed → validation pool
  Purpose: calibrate thresholds + fuzzy bounds + measure TPR/FPR + gate checks
  Severity 0.2–0.3 subset → calibrate Mech A/B/C rolling thresholds
  Group B compound → verify both fault channels produce high MAE
  Group C masked   → verify secondary detection path fires
  Group D variants → verify detection character matches variant type
  Group E multi-sensor → verify multi-channel flatline detection

WHY FAULTS NEVER IN TRAINING:
  LSTM-AE is anomaly detector, NOT classifier.
  Training on faults = model learns to reconstruct faults as normal
  = complete failure of the anomaly detection purpose.
  Faults only appear in validation to calibrate the boundary.

MECH C CALIBRATION SUBSET:
  Use ONLY mild severity (sev 0.2–0.4) sequences for Mech C tuning.
  Spearman threshold 0.70 target:
    ≥ 80% mild overloading sequences → Temp.SV drift fires ≤ 15 min
    ≥ 80% mild seal sequences        → Pres.SV drift fires ≤ 20 min
  Adjust per-channel threshold if targets not met (see Part 2 adaptive actions).
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic file — bias-audit updates incorporated |
| v2.0 | 2026-04-15 | **SPLIT into Part 1 + Part 2**. Part 1 = Architecture (this file). Updated: M7 prerequisite 10→15 gates, M6B fault validation pool replaces M6A, Group B–E coverage in Mech C + detection map, training pool updated to M6B synthetic normal. v1.0 monolithic file converted to redirect stub. |

---

*GitHub is the ONLY source of truth for this spec.*
*Do NOT reference any Spaces .md pathway files — all outdated.*
*Companion: `module_M8_lstm_ae_v2_gates_and_outputs.md` (Part 2)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
