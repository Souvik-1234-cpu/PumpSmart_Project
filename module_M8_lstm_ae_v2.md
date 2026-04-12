# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
## Fuzzy Logic + Three-Mechanism Slow Drift Detection + Four-State Alert Machine

**Document version:** v1.0 — Post Bias-Audit  
**Date:** 2026-04-12  
**Prerequisite:** M7 all 10 gates passed | `M7_all_10_gates_pass = True`  
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)  
**Status:** NOT STARTED — begins only after M7 gates confirmed

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

PREREQUISITE: M7_all_10_gates_pass = True before M8 starts.
If M7 gates fail → fix M6.5 → rerun M7 → only then start M8.
```

---

## What Changed Since Original M8 Spec (Bias-Audit Updates)

| Item | Original Spec | Revised Spec | Reason |
|------|--------------|-------------|--------|
| Training data ratio | Real CIRA normal only | **30% real CIRA + 70% M6A+M6B synthetic normal** | Bias 2 partial fix — prevent pure-physics dominance |
| Fault validation pool | M6A 7200 single-fault sequences | **M6A 7200 + M6B 1600 compound sequences** | Bias 4 — compound faults must be reconstructable-or-not |
| Threshold calibration | On synthetic validation | **Threshold set on REAL CIRA validation set** | Bias 2 — anchor to real pump behaviour |
| Alert states | 3 states (Normal/Watch/Fault) | **4 states: NORMAL / WATCH / WARN / DANGER** | Progressive confidence (Bias 1) |
| Slow drift mandate | Informally noted | **Explicitly tied to liability framework — Category 3** | v10 LIABILITY FRAMEWORK |
| Fuzzy exclusion rule | Not specified | **Exclude overloading mild + seal mild from upper_bound calibration** | M6.5 Finding 1+2 |
| Compound fault in validation | None | **M6B compound sequences in fault validation pool** | Bias 4 |

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

## M8 Architecture — Five Stages

---

### STAGE 1 — LSTM-AE Reconstruction

```
Input    : (batch, 50, 8) — 50-timestep windows, 8 normalized channels
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
    INCLUDE: compound sequences ALL (M6B)
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
```

---

### STAGE 3 — Slow Drift Detection: Three Mechanisms

```
LIABILITY BASIS (ISO 13374 Level 3 / v10 LIABILITY FRAMEWORK):
  Category 3 fault = progressive degradation = MODEL'S RESPONSIBILITY.
  Fault developing over days/weeks → per-window MAE too small for threshold.
  Without these mechanisms → fault missed entirely → liability exposure.
  M6A severity 0.2–0.3 sequences generated SPECIFICALLY to calibrate these.
```

#### MECHANISM A — Rolling Mean Gate (medium horizon: ~3 minutes)

```
Computation:
  rolling_mean_MAE_200 = mean(MAE, last 200 windows)
  rolling_mean_MAE_100 = mean(MAE, last 100 windows)

Thresholds (calibrated on mild-severity M6 sequences):
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
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Mot.SV drift   (Spearman_r > 0.70, POSITIVE trend)                     │
│ → bearing_wear_early flag                                               │
│ Physics: bearing degradation → Mot.SV* rises before Mot.TV (20–40s lag)│
│ Thermal coupling must ALSO be preserved (r > 0.85) simultaneously      │
│ Both: Mot.SV drift + thermal coupling preserved = bearing confirmed     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Single channel flatline                                                 │
│ std(channel_error[ch], last 100 windows) < 0.001                       │
│ → sensor_failure flag                                                   │
│ Physics: dead sensor → value locked → reconstruction error constant     │
│ → std collapses to near-zero                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### STAGE 4 — Four-State Alert Machine

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

#### Fault-Specific Exceptions

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAVITATION EXCEPTION (Finding 5 — MAE = 0.675, 6.1× threshold):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if cluster == 'startup' AND single_window_MAE > 3 × cluster_threshold:
      alert_state = DANGER  # bypass WATCH and WARN entirely
  Physics: cavitation is acute hydraulic shock.
  Impeller pitting begins within 60–180s of onset.
  No time for rolling mean accumulation.
  DO NOT route cavitation through rolling mean accumulator.
  Gate M8-12: ZERO cavitation DANGER alerts outside startup cluster.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERLOADING EXCEPTION (Finding 1 — Gate 3 = 0.00%):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Primary detection = Mech C Temp.SV drift (POSITIVE Spearman > 0.70)
  Gate M8-7 denominator = overloading validation sequences ONLY
  Gate M8-7 numerator = sequences where Temp.SV drift fires ≤ 15 min
  Single-window MAE crossing excluded from overloading TPR measurement
  Overloading mild sequences will NOT cross single-window threshold — by design.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEAL FAILURE EXCEPTION (Finding 2 — Gate 3 = 29.17%):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Primary detection   = Mech C Pres.SV drift flag (NEGATIVE Spearman)
  Secondary confirm   = Mech A rolling mean
  Mild seal (0.2–0.4): Mech C fires first → Mech A confirms
  Severe seal (0.5+) : single-window MAE also fires
  Combined: Pres.SV drift (negative) + thermal_decoupling = HIGH CONFIDENCE seal
  Gate M8-9: Pres.SV drift WATCH ≤ 20 min for sev 0.2 sequences
  Gate M8-10: Pres.SV drift flag fires BEFORE total MAE reaches WARN state
```

---

### STAGE 5 — Cluster-Conditional Thresholds

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

## M8 Training Data Composition (Bias 2 Partial Fix)

```
NORMAL TRAINING POOL (model learns ONLY normal — faults never appear in training):
  Real CIRA normal (M3 normalized)   : 9,711 windows  ← 30% of effective pool
  Synthetic normal (M6A Type-A)      : 1,200 sequences → windowed → ~24,000 windows
  Total normal training pool         : ~33,000 windows
  Real : Synthetic ratio             : ~30% : 70%
  Cluster distribution maintained    : startup 42.3%, cooldown 22.8% etc.

BIAS 2 RATIONALE — WHY 30:70 REAL:SYNTHETIC:
  Pure synthetic training → model learns only physics-idealized normal patterns
  → too sensitive to real-world deviations from theory → elevated FPR in field.
  30% real CIRA anchors the model to actual pump behaviour:
  manufacturing tolerances, sensor placement variation, ambient noise.
  70% synthetic provides coverage of all 4 operating modes with
  controlled severity distribution (Weibull-skewed from M6A).
  This is the maximum real fraction available given 9711 windows total.

VALIDATION ONLY — FAULT SEQUENCES (never in training):
  Single-fault sequences (M6A)  : 7,200 fault sequences → windowed
  Compound-fault sequences (M6B): 1,600 compound sequences → windowed  ← NEW
  Purpose: calibrate thresholds + fuzzy bounds + measure TPR/FPR + gate checks
  Severity 0.2–0.3 subset       : calibrate Mech A/B rolling thresholds
  Compound sequences            : verify reconstruction is appropriately poor
                                  (compound faults should exceed threshold)

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
  Adjust per-channel threshold if targets not met (see adaptive actions).
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
Step 7: Apply fault-specific exceptions:
        — Cavitation:   startup + MAE > 3×threshold → DANGER immediately
        — Overloading:  Temp.SV Spearman > 0.70 (positive) → overloading_early
        — Seal failure: Pres.SV Spearman > 0.70 (negative) + thermal_decoupling → seal_failure_early
        — Bearing wear: Mot.SV Spearman > 0.70 (positive) + coupling preserved → bearing_wear_early
        — Sensor fail:  channel_error std < 0.001 over 100 windows → sensor_failure
Step 8: Determine alert state → output dict

OUTPUT DICT (complete):
{
  alert_state          : "NORMAL" / "WATCH" / "WARN" / "DANGER"
  anomaly_flag         : bool              (hard threshold — legacy compatibility)
  fuzzy_membership     : float [0, 1]
  rolling_mean_mae     : float             (200-window Mech A)
  mae_slope            : float             (500-window Mech B slope)
  channel_drift        : {
    "Mot.SV"  : bool,   "Pmp.SV"  : bool,   "Pres.SV" : bool,
    "Temp.SV" : bool,   "Mot.TV"  : bool,   "Pmp.TV"  : bool,
    "Mot.PV"  : bool,   "Pmp.PV"  : bool
  }
  early_fault_type     : None / "overloading_early" / "seal_failure_early" /
                         "bearing_wear_early" / "sensor_failure"
  severity             : "LOW" / "MEDIUM" / "HIGH"  (MC Dropout std zones)
  uncertainty_std      : float             (MC Dropout spread — confidence proxy)
  confidence           : float [0, 1]
  attention_heatmap    : array(50,)        (timesteps driving reconstruction error)
  cluster              : "startup" / "steady_state" / "high_load" / "cooldown"
}
```

---

## M8 Detection Coverage Map

| Fault | Single Window | Mech A Rolling | Mech B Slope | Mech C Per-Channel |
|-------|--------------|----------------|--------------|--------------------|
| Bearing wear sev 0.8 | ✅ DANGER | ✅ | ✅ | ✅ Mot.SV drift |
| Bearing wear sev 0.2 | MAE≈0.098 sub-threshold | ✅ WATCH ~10min | ✅ ~8min | ✅ Mot.SV drift |
| Cavitation severe | ✅ DANGER immediately | ❌ bypassed | ❌ bypassed | ✅ Pres.SV |
| Seal failure slow | ❌ 29% windows | ✅ WATCH ~15min | ✅ | ✅ **Pres.SV PRIMARY** |
| Overloading mild | ❌ MAE≈0.093 sub-threshold | ❌ slow | ✅ | ✅ **Temp.SV PRIMARY** |
| Sensor failure | ✅ DANGER MAE≈0.170 | ✅ | ✅ | ✅ flatline |
| Compound bearing+seal | ✅ both channels elevate | ✅ | ✅ | ✅ both drift |

---

## M8 All 13 Validation Gates

```
GATE-M8-1 : TPR fault detection
             > 90% on fault validation set
             Report SEPARATELY per fault class — do not aggregate
             Cavitation ~100% expected — do not let it mask other classes
             Denominator EXCLUDES overloading (Gate M8-7) and seal mild (Gate M8-9)

GATE-M8-2 : FPR false alarm
             < 5% on FULL 9711-window normal pool [Finding 6]
             NOT on 30-window probe subset — that result is INVALID
             Measured cluster-by-cluster: report startup FPR separately
             Startup naturally higher MAE — cluster threshold prevents false alarms

GATE-M8-3 : Youden's J
             > 0.85  (J = TPR − FPR)

GATE-M8-4 : Separation ratio
             > 5.0×  (M4 baseline was 4.11×)
             = mean_fault_MAE / mean_normal_MAE
             Computed on included fault population (cavitation dominated)

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
             [Finding 1 — PRIMARY and ONLY reliable path]
             Gate FAILS if WATCH fires via Mech A before Temp.SV drift flag
             Document if < 80% — do NOT raise global threshold to compensate

GATE-M8-8 : Attention seam check
             For bearing_wear validation sequences:
             seam_ratio = mean_attention(t=49,50) / mean_attention(t=10,40)
             Gate: seam_ratio < 1.0  (fault onset dominates over seam artifact)
             [Finding 3 — spike seed seam must not be learned as fault signal]
             FAIL action: add gradient penalty at t=49–50, retrain M8

GATE-M8-9 : Slow drift seal detection
             WATCH fires ≤ 20 min for seal_failure sev 0.2 sequences
             Via Pres.SV Spearman drift (NEGATIVE) [Finding 2 — PRIMARY]
             thermal_decoupling_flag must ALSO be True simultaneously
             Combined flag = high-confidence seal_failure_early

GATE-M8-10: Pres.SV drift fires first
             For seal_failure mild sequences:
             timestep(Pres.SV drift flag) < timestep(WARN state)
             [Finding 2 — Mech C catches it BEFORE rolling mean responds]

GATE-M8-11: Thermal lag validation
             For bearing_wear validation sequences:
             peak Mot.SV reconstruction error must PRECEDE peak Mot.TV error
             by 20–40 timesteps
             [Physics: heat conduction lag — M2 r=0.9793 + M5 Euler integration]
             FAIL = model detecting thermal consequence, not mechanical cause
             → will misclassify bearing wear on high-ambient installations

GATE-M8-12: Cavitation cluster exclusivity
             ZERO cavitation DANGER alerts on steady_state or high_load windows
             in the normal validation pool
             [Physics: cavitation requires low NPSH margin — startup only]
             FAIL → audit M6 cluster assignment, check for startup seed mis-labeling
             Consequence of failure: operators panic-stop healthy pump under full load

GATE-M8-13: Energy conservation check
             mean(sum_8_channels(reconstruction_error)) for normal validation < 0.25
             [8 channels × 0.026765 mean_MAE × 1.1 headroom = ~0.235 ceiling]
             FAIL = encoder bottleneck leaking fault-like signal into normal reconstruction
             → threshold drifts upward in deployment → missed faults
             FAIL action: add L2 regularization on 64-dim bottleneck layer, retrain
```

---

## Adaptive Actions After M8

| M8 Result | Trigger | Adaptive Action |
|-----------|---------|----------------|
| Overloading TPR < 80% (M8-7) | Mech C not sensitive enough | Lower Spearman threshold 0.70 → 0.65 for Temp.SV ONLY. Re-validate FPR impact |
| Seal WATCH > 20 min (M8-9) | Pres.SV drift too slow | Shorten Mech C window 300 → 200 for Pres.SV ONLY. Re-run Gate M8-9 |
| FPR > 5% at startup (M8-2) | Startup MAE elevated naturally | Raise startup cluster threshold ONLY — never global threshold |
| Separation ratio < 5.0× (M8-4) | Normal pool contaminated | Audit normal pool via M4 AE error, remove near-fault windows, retrain |
| Attention seam ratio > 1.0 (M8-8) | Seam artifact learned | Add gradient penalty at t=49–50 specifically. Retrain M8 |
| Gate M8-11 fails (thermal lag) | Thermal over-reliance | Reduce Mot.TV weight 0.3 → 0.1. Force vibration-first detection. Retrain |
| Gate M8-12 fails (cavitation in high_load) | Cluster label leak | Audit M6 cluster assignment — startup seed mis-labeling |
| Gate M8-13 fails (energy conservation) | Bottleneck unstable | Add L2 on 64-dim bottleneck layer. Retrain M8 |
| Bearing-imbalance TPR similar | onset_lag not discriminating | Verify M6.5 `err_onset_lag` uses t=0, not seam as reference |
| Seal-cavitation confusion | Both pressure faults | Verify `thermal_decoupling_flag` uses M5 coupling threshold correctly |
| Compound sequences all DANGER | All compound seqs exceed threshold | Expected and correct — both fault channels active → high MAE |
| Compound sequences sub-threshold | Compound signals cancel | Investigate M6B causal lag — secondary fault onset may be too mild |

---

## M8 Outputs

```
models/lstm_ae_v2_best.pth                  ← production model weights
models/M8_threshold_config.json             ← cluster-conditional thresholds (4 values)
models/M8_fuzzy_config.json                 ← lower_bound, upper_bound, calibration log
outputs/M8_roc_curve.png
outputs/M8_per_class_tpr.png                ← SEPARATE TPR bar per fault class [Finding 5]
outputs/M8_attention_heatmap.png            ← seam check visualization [Finding 3]
outputs/M8_mech_c_drift_plots.png           ← Spearman drift per channel (all 8 channels)
outputs/M8_fuzzy_calibration.png            ← two-population MAE with exclusions marked
outputs/M8_channel_error_dist.png           ← per-channel reconstruction error by fault type
outputs/M8_detection_coverage.png           ← which mechanisms detect which faults
outputs/reports/module_08_lstm_ae_v2_report.md
```

---

## M8 Paste Text Keys (31 Keys)

```
M8_val_loss                       : [value]
M8_best_epoch                     : [value]
M8_tpr_overall                    : [% — gate > 90%, excludes overloading + seal mild]
M8_tpr_cavitation                 : [% — expected ~100%, Finding 5]
M8_tpr_bearing_wear               : [%]
M8_tpr_impeller_imbalance         : [%]
M8_tpr_sensor_failure             : [%]
M8_tpr_overloading                : [% — gate M8-7 ≥ 80% via Mech C Temp.SV ONLY]
M8_tpr_seal_failure               : [% — gate M8-9 WATCH ≤ 20 min via Pres.SV drift]
M8_tpr_compound_bearing_seal      : [%]  ← NEW — compound pair TPR
M8_tpr_compound_cavitation_imbal  : [%]  ← NEW — compound pair TPR
M8_fpr_full_pool                  : [% — gate < 5%, full 9711-window pool]
M8_fpr_startup_cluster            : [% — report separately]
M8_youden_j                       : [value — gate > 0.85]
M8_separation_ratio               : [value — gate > 5.0×]
M8_fuzzy_lower_bound              : [value — P95 normal, expected 0.07–0.09]
M8_fuzzy_upper_bound              : [value — P5 fault selective, expected 0.15–0.50]
M8_fuzzy_transition_width         : [upper - lower — gate ≥ 0.05]
M8_rolling_watch_threshold        : [calibrated — target ~0.085]
M8_rolling_warn_threshold         : [calibrated — target ~0.095]
M8_slope_threshold                : [calibrated — target ~0.0003/window]
M8_slow_drift_overload_watch_min  : [minutes — gate ≤ 15, via Temp.SV drift]
M8_slow_drift_seal_watch_min      : [minutes — gate ≤ 20, via Pres.SV drift]
M8_slow_drift_bearing_watch_min   : [minutes — gate ≤ 15, via Mot.SV drift]
M8_attention_seam_ratio           : [value — gate < 1.0 = PASS]
M8_gate_thermal_lag               : PASS/FAIL
M8_gate_cavitation_exclusivity    : PASS/FAIL
M8_gate_energy_conservation       : PASS/FAIL
M8_threshold_startup              : [value]
M8_threshold_steady_state         : [value — baseline 0.110058]
M8_threshold_high_load            : [value]
M8_threshold_cooldown             : [value]
M8_all_13_gates_pass              : True/False
Status_for_M9                     : READY/BLOCKED
```

---

## Module Dependency Chain

```
UPSTREAM (required before M8):
  M7_all_10_gates_pass = True          ← NON-NEGOTIABLE prerequisite
  M6_feature_matrix.csv (10000 × 29)   ← for fuzzy boundary calibration reference
  M6A sequences (8400, 200, 8)         ← fault validation pool
  M6B sequences (1600, 200, 8)         ← compound fault validation pool ← NEW
  M3 normalized data (117970 rows)     ← real CIRA normal training pool
  M4 threshold (0.110058) LOCKED       ← starting reference for steady_state

DOWNSTREAM (M8 outputs feed into):
  M10 Flask  ← lstm_ae_v2_best.pth + M8_threshold_config.json + M8_fuzzy_config.json
  M10 Flask  ← alert_state output dict is API response for /api/anomaly_detect route
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
8. Compound sequences (M6B) in fault validation pool only — never in training
9. Cavitation gate: startup cluster only — any cavitation DANGER outside startup = FAIL
10. `if pump_type == 'household': return physics_advisory_only()` — NO EXCEPTIONS

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Standalone file created — split from modules_M7_M8_critical_ML.md. Bias-audit updates: 30:70 ratio, real CIRA threshold anchor, M6B compound in validation, 4-state alert, 2 new paste text keys for compound TPR |

---

**Derived from:** `modules_M7_M8_critical_ML.md` v1.0 + bias-audit discussion 2026-04-12  
**Previous file:** `module_M7_xgboost_classifier.md`  
**Next file:** `modules_M9_M10_M11_deployment.md` (update pending)  
**Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset  
**Standard:** ISO 10816-3 vibration, ISO 13373-3 condition monitoring
