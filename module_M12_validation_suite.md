# PumpSmart — M12: Physics-Governed Adversarial Validation Suite
# Status: NOT STARTED — begins ONLY after M11 deployment confirmed
# Updated: 2026-04-12 | Derived from: module_pathway_M1_to_M12_v10.md
# Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset

---

## PURPOSE AND RELATIONSHIP TO M6

```
M6 trained M7 and M8 on synthetic sequences.
M8 validated against M6 synthetic sequences (model has SEEN these).

M12 generates COMPLETELY FRESH sequences using M5 physics engine —
parametrically different from M6 training data.
Model has NEVER been exposed to M12 sequences.

M12 = adversarial stress-test on a never-seen distribution.
If M8 passes M12 → PRODUCTION_VALIDATED.
If M8 fails M12 → RECALIBRATION_REQUIRED → fix M6/M8 → re-run M12.
```

| Dimension | M6 (Training) | M12 (Adversarial) |
|---|---|---|
| Purpose | Train M7 + M8 | Stress-test M8 on deployment-like conditions |
| Model exposure | Yes — seen in M8 validation | Never — completely held out |
| Generation | Physics engine + spike seeds | M5 physics engine ONLY (no spike seeds) |
| Trigger | One-time during training | On-demand via /api/validate_model (M10) |
| Primary metric | Label distribution balance | Detection latency (timesteps) |
| Slow drift configs | Present but seen by M8 | Present and UNSEEN — true adversarial |

---

## PREREQUISITE

```
M11 deployment confirmed (HF Space health check = healthy)
  AND
M8 all_13_gates_pass = True
  AND
M12 sequences generated fresh using M5 physics engine
  (NOT reusing any sequence from data/synthetic/M6_sequences.pkl)
```

---

## PRIMARY METRIC: DETECTION LATENCY

```
Physics basis for latency budget:
  110 kW, 2980 RPM, 40 bar multistage pump:
  Bearing wear to catastrophic failure    : ~600–1800 seconds
  Cavitation impeller pitting onset       : 60–180 seconds
  Maintenance crew response time minimum  : ~300 seconds
  → Detection budget: 300 timesteps (M8 must detect within 30% of failure window)
  → Standard gate : detection_lag ≤ 60 timesteps (2× safety margin on maintenance budget)
  → Cavitation gate: detection_lag ≤ 30 timesteps (impeller pitting risk)

SLOW DRIFT LATENCY GATES (v10.0 — liability-critical):
  Seal failure sev 0.2 (weeks-scale)  : WATCH state must fire ≤ 20 min (1200 steps)
  Bearing wear sev 0.25 (days-scale)  : WATCH state must fire ≤ 15 min (900 steps)
  Overloading sev 0.2 (days-scale)    : WATCH state must fire ≤ 15 min (900 steps)
  These gates validate M8 trend accumulator on NEVER-SEEN slow sequences.

DETECTION LAG DEFINITION:
  detection_lag = timestep when alert_state first reaches target level
                  MINUS timestep of fault onset (t=0 of M12 sequence)
  For slow drift: target level = WATCH
  For acute fault: target level = DANGER
```

---

## ALL 16 TEST CONFIGURATIONS

---

### CONFIG 1 — Bearing Wear (Standard, sev 0.6)

```
Fault type  : bearing_wear
Cluster     : steady_state
Severity    : 0.6
Steps       : 500
Generation  : M5 physics engine (smooth — NO spike seed)

Physics invariants:
  Mot.SV* rises monotonically from 1.0 → ~1.6 over 500 steps
  Mot.TV* rises with lag 20–40 steps behind Mot.SV* (thermal lag)
  Thermal coupling r(Mot.TV, Temp.SV) > 0.85 (bearing = thermal fault)
  All other channels: remain within 1.0 ± 0.1 (no leak to non-fault channels)

Gates:
  GATE-C1-1: detection_lag ≤ 60 timesteps (DANGER)
  GATE-C1-2: thermal_lag confirmed: peak Mot.SV error BEFORE peak Mot.TV error
  GATE-C1-3: XGBoost classifies as bearing_wear (confidence > 0.7)
```

---

### CONFIG 2 — Bearing Wear (High Noise, sev 0.6)

```
Fault type  : bearing_wear
Cluster     : high_load
Severity    : 0.6
Steps       : 500
Noise level : 3× normal (sigma multiplier = 3.0 on all channels)
Generation  : M5 physics engine + noise augmentation

Physics invariants:
  Same as Config 1 but with elevated sensor noise
  Fault signal must still be detectable above noise floor
  Tests M8 robustness to poor sensor installation

Gates:
  GATE-C2-1: detection_lag ≤ 90 timesteps (DANGER) — relaxed for high noise
  GATE-C2-2: alert_state reaches at least WARN within 120 timesteps
  GATE-C2-3: MC Dropout uncertainty_std elevated (expected > 0.015 in high noise)
```

---

### CONFIG 3 — Bearing Wear (Subtle Early, sev 0.25)

```
Fault type  : bearing_wear
Cluster     : steady_state
Severity    : 0.25
Steps       : 1500
Generation  : M5 physics engine (smooth, no spike seed, no seam artifact)

Physics invariants:
  Mot.SV* rises very slowly: ~0.0003/step
  Per-step MAE well below 0.110058 for first 800+ steps
  Total MAE never crosses threshold in first 500 steps
  Thermal coupling preserved throughout

Critical note:
  This config specifically tests M8 SLOW DRIFT detection.
  Single-window MAE will NOT cross threshold early.
  Mech C (Mot.SV Spearman) is the PRIMARY detection path.
  Mech A (rolling mean) provides secondary confirmation.
  This is an UNSEEN slow drift sequence — true adversarial test.

Gates:
  GATE-C3-1: WATCH state fires ≤ 15 min (900 steps) via Mot.SV drift flag
  GATE-C3-2: total MAE remains below threshold for first 300 steps
             (confirms fault is genuinely subtle — not trivially detectable)
  GATE-C3-3: thermal_lag maintained: Mot.SV error Spearman fires BEFORE
             Mot.TV error Spearman (vibration precedes thermal effect)
```

---

### CONFIG 4 — Cavitation (Standard, sev 0.7)

```
Fault type  : cavitation
Cluster     : startup ONLY
Severity    : 0.7
Steps       : 300
Generation  : M5 physics engine

Physics invariants:
  Pres.SV* drops erratically (chaotic, high kurtosis)
  Pmp.SV* spikes sharply (hydraulic shock → pump vibration)
  Thermal coupling WEAK (r ~ 0.3–0.4) — hydraulic fault, not thermal
  Mot.TV*, Temp.SV*: remain near 1.0 (NOT elevated for cavitation)
  Cluster: startup ONLY — never in steady_state or high_load

Gates:
  GATE-C4-1: detection_lag ≤ 30 timesteps (DANGER) — cavitation is acute
  GATE-C4-2: alert_state = DANGER (bypasses WATCH/WARN — cavitation exception)
  GATE-C4-3: XGBoost classifies as cavitation (confidence > 0.85)
  GATE-C4-4: thermal_decoupling_flag = True in M8 output
```

---

### CONFIG 5 — Cavitation (High Noise, sev 0.7)

```
Fault type  : cavitation
Cluster     : startup
Severity    : 0.7
Steps       : 300
Noise level : 3× normal
Generation  : M5 physics engine + noise augmentation

Physics invariants: same as Config 4
Cavitation MAE is 6.1× threshold — high noise should NOT mask detection.
Tests whether Pmp.SV spike is distinguishable from noise at startup.

Gates:
  GATE-C5-1: detection_lag ≤ 50 timesteps (DANGER) — relaxed for high noise
  GATE-C5-2: alert_state = DANGER
  GATE-C5-3: uncertainty_std elevated (MC Dropout spread under noise)
```

---

### CONFIG 6 — Seal Failure (Slow Leak, sev 0.3)

```
Fault type  : seal_failure
Cluster     : steady_state
Severity    : 0.3
Steps       : 1000
Generation  : M5 physics engine (smooth monotonic pressure decline)

Physics invariants:
  Pres.SV* declines monotonically: ~0.0004/step
  Thermal coupling BROKEN: r(Mot.TV, Temp.SV) < 0.1 (hydraulic fault)
  thermal_decoupling_flag must be True
  Mot.SV*, Pmp.SV*: remain near 1.0 (not a vibration fault)
  Per-window MAE well below threshold for many early windows

Critical: Mech C Pres.SV drift (NEGATIVE Spearman) is PRIMARY path.
combined with thermal_decoupling_flag = HIGH CONFIDENCE seal_failure_early.

Gates:
  GATE-C6-1: WATCH state fires ≤ 20 min (1200 steps) via Pres.SV drift
  GATE-C6-2: thermal_decoupling_flag = True in M8 output
  GATE-C6-3: Pres.SV drift flag fires BEFORE total MAE reaches WARN level
  GATE-C6-4: XGBoost classifies as seal_failure (at WARN/DANGER state)
```

---

### CONFIG 7 — Seal Failure (Fast Degradation, sev 0.8)

```
Fault type  : seal_failure
Cluster     : high_load
Severity    : 0.8
Steps       : 400
Generation  : M5 physics engine

Physics invariants:
  Pres.SV* drops rapidly: ~0.002/step
  Thermal decoupling confirmed
  Severe seal failure → single-window MAE should cross threshold

Gates:
  GATE-C7-1: detection_lag ≤ 60 timesteps (DANGER)
  GATE-C7-2: alert_state reaches DANGER
  GATE-C7-3: XGBoost classifies as seal_failure (confidence > 0.75)
```

---

### CONFIG 8 — Overloading (Standard, sev 0.6)

```
Fault type  : overloading
Cluster     : steady_state ONLY
Severity    : 0.6
Steps       : 800
Generation  : M5 physics engine

Physics invariants:
  Temp.SV* rises monotonically: thermal overload
  Mot.TV* rises with slight lag behind Temp.SV*
  Pres.SV*, Pmp.SV*, Mot.SV*: remain near 1.0
  Thermal coupling STRONGLY PRESERVED (r=0.997 from M5) — both thermal channels rise
  NOT a vibration fault — vibration channels must stay flat

Critical: Mech C Temp.SV drift (POSITIVE Spearman) is PRIMARY detection path.
Single-window MAE will not cross threshold reliably (Finding 1).

Gates:
  GATE-C8-1: WATCH state fires ≤ 15 min (900 steps) via Temp.SV drift flag
  GATE-C8-2: XGBoost classifies as overloading (at WATCH/WARN state)
  GATE-C8-3: Vibration channels Mot.SV*, Pmp.SV* remain below 1.2
             (confirms overloading is thermal — no vibration contamination)
  GATE-C8-4: Thermal coupling r(Mot.TV, Temp.SV) > 0.90 in M12 sequence
```

---

### CONFIG 9 — Impeller Imbalance (Standard, sev 0.7)

```
Fault type  : impeller_imbalance
Cluster     : high_load
Severity    : 0.7
Steps       : 400
Generation  : M5 physics engine

Physics invariants:
  Pmp.PV* and Pmp.SV* rise immediately (imbalance = immediate onset)
  Pres.SV* oscillates (impeller imbalance → pressure ripple)
  Mot.PV* rises (vibration propagates to motor side)
  Thermal channels: rise slowly (secondary effect from vibration energy)
  err_onset_lag SMALL (fast onset — distinguishes from bearing_wear)

Gates:
  GATE-C9-1: detection_lag ≤ 60 timesteps (DANGER)
  GATE-C9-2: XGBoost classifies as impeller_imbalance (confidence > 0.7)
  GATE-C9-3: err_onset_lag < 10 timesteps (immediate onset confirmed)
```

---

### CONFIG 10 — Sensor Failure (Sanity Check)

```
Fault type  : sensor_failure
Cluster     : any
Severity    : 0.8
Steps       : 300
Generation  : M5 physics engine (flatline on ONE channel)

Physics invariants:
  One channel (e.g., Pres.SV) goes flatline at t=50
  All other 7 channels: remain normal
  Single-channel reconstruction error collapses to near-zero std
  Total MAE: moderate elevation (1 of 8 channels flat)

Gates:
  GATE-C10-1: sensor_failure flag fires within 100 timesteps of flatline
  GATE-C10-2: channel_drift[flatline_channel] = True, all others False
  GATE-C10-3: XGBoost classifies as sensor_failure (confidence > 0.90)
  GATE-C10-4: M8 output clearly identifies WHICH channel failed
```

---

### CONFIG 11 — Multi-Fault: Bearing Wear + Cavitation Simultaneously

```
Fault types : bearing_wear (sev 0.5) + cavitation (sev 0.7) simultaneously
Cluster     : startup (cavitation requires startup cluster)
Steps       : 300
Generation  : M5 physics engine (compound fault)

Physics invariants:
  Pmp.SV* elevated (BOTH bearing AND cavitation raise this channel)
  Mot.SV* elevated (bearing component)
  Pres.SV* erratic (cavitation component)
  Thermal coupling: partial (bearing preserves, cavitation disrupts)
  This is the hardest config — two fault channels active simultaneously

Gates:
  GATE-C11-1: DANGER fires within 30 timesteps (cavitation dominates speed)
  GATE-C11-2: alert_state = DANGER (at least one fault at threshold)
  GATE-C11-3: XGBoost flags cavitation OR bearing_wear as primary
              (either is acceptable — document which one fires)
              If cavitation flagged: correct (higher MAE channel)
              If bearing flagged: acceptable (both present) — note in report
  GATE-C11-4: channel_drift shows BOTH Pmp.SV and Mot.SV elevated
```

---

### CONFIG 12 — Bearing Wear with Thermal Coupling Broken

```
Fault type  : bearing_wear
Cluster     : steady_state
Severity    : 0.7
Steps       : 500
Forced      : r(Mot.TV, Temp.SV) = 0.15 (thermal decoupling forced)
Generation  : M5 physics engine (coupling artificially suppressed)

Physics context:
  Normally bearing wear PRESERVES thermal coupling (r > 0.85).
  This config tests what happens if ambient conditions or installation
  factors break the thermal path (e.g., poor thermal contact).
  M8 should still detect via vibration channels (Mot.SV*, Pmp.SV*).
  Tests whether M8 over-relies on thermal coupling for bearing detection.

Gates:
  GATE-C12-1: detection_lag ≤ 90 timesteps (relaxed — harder config)
  GATE-C12-2: DANGER or WARN state reached via vibration channels
  GATE-C12-3: Mot.SV drift flag fires (Mech C vibration path works independently)
  GATE-C12-4: If detection_lag > 90 → document over-reliance on thermal coupling
              Flag for M8 retraining: reduce Mot.TV weight further
```

---

### CONFIG 13 — Seal Failure Slow Drift (Weeks-Scale, sev 0.2)
[NEW v10.0 — SLOW DRIFT LIABILITY GATE]

```
Fault type  : seal_failure
Cluster     : steady_state
Severity    : 0.2
Steps       : 2000+
Generation  : M5 physics engine ONLY (NO spike seed — pure smooth physics)

Physics invariants:
  Pres.SV* declines at rate ~0.0004/step (extremely gradual)
  Total MAE must remain BELOW threshold for first 500+ steps
  (confirms this is genuinely a slow drift scenario)
  Thermal coupling BROKEN throughout (r < 0.1) — hydraulic fault
  thermal_decoupling_flag = True from onset
  All other channels: flat, near 1.0

This config simulates a seal that has been leaking for weeks.
In any single 50-step window, the signal is sub-threshold.
Only the accumulation over 2000+ steps makes it detectable.
This is the PRIMARY liability test — the exact scenario PumpSmart
must catch to justify its role on a 110 kW asset.

Detection path:
  Mech C: Pres.SV Spearman (NEGATIVE slope) over 300 windows → WATCH
  Mech A: rolling mean MAE (200-window) → secondary WATCH confirmation
  Combined: Pres.SV drift + thermal_decoupling_flag = HIGH CONFIDENCE seal

Gates:
  GATE-C13-1: WATCH state fires ≤ 20 min (1200 steps) via Pres.SV drift flag
              [PRIMARY liability gate — non-negotiable]
  GATE-C13-2: total MAE remains BELOW threshold (0.110058) for first 500 steps
              (validates this is a true slow drift scenario — not trivially detectable)
  GATE-C13-3: thermal_decoupling_flag = True in M8 output from onset
  GATE-C13-4: Pres.SV drift flag fires BEFORE rolling mean WATCH trigger
              (Mech C is faster than Mech A for single-channel slow fault)
  GATE-C13-5: No false DANGER state fired in first 500 steps
              (WATCH is the correct state — DANGER would be over-alert)
```

---

### CONFIG 14 — Bearing Wear Slow Drift (Days-Scale, sev 0.25)
[NEW v10.0 — SLOW DRIFT LIABILITY GATE]

```
Fault type  : bearing_wear
Cluster     : steady_state
Severity    : 0.25
Steps       : 1500+
Generation  : M5 physics engine ONLY (smooth, NO spike seed, NO seam artifact)

Physics invariants:
  Mot.SV* rises gradually at ~0.0003/step
  Thermal coupling PRESERVED (r > 0.85) — bearing = thermal fault
  Mot.TV* rises with 20–40 step lag behind Mot.SV*
  Total MAE remains below threshold for first 400+ steps
  All non-fault channels: remain near 1.0

This config simulates a bearing in early degradation (days before failure).
The Spearman drift on Mot.SV* is detectable long before the bearing fails.
This is the operationally critical early warning window.

Detection path:
  Mech C: Mot.SV Spearman (POSITIVE slope) over 300 windows → WATCH
  Mech A: rolling mean → secondary confirmation
  Thermal coupling check: r(Mot.TV, Temp.SV) > 0.85 → confirms bearing (not seal)

Gates:
  GATE-C14-1: WATCH state fires ≤ 15 min (900 steps) via Mot.SV drift flag
              [PRIMARY liability gate — non-negotiable]
  GATE-C14-2: total MAE remains BELOW threshold for first 300 steps
              (confirms genuine slow drift scenario)
  GATE-C14-3: thermal coupling r > 0.85 confirmed in M12 sequence
              (validates bearing vs seal distinction — coupling preserved)
  GATE-C14-4: Mot.SV Spearman fires BEFORE Mot.TV Spearman
              (vibration precedes thermal — thermal lag validated in slow drift)
  GATE-C14-5: Attention heatmap: does NOT cluster at any specific timestep
              (smooth drift — no seam artifact — no pure spike seed character)
```

---

### CONFIG 15 — Cluster Transition Fault (Startup → Steady-State Boundary)
[NEW — Physics-justified addition]

```
Fault type  : bearing_wear
Cluster     : startup → transitions to steady_state at step 50
Severity    : 0.6
Steps       : 400
Fault onset : t=40 (10 steps before cluster transition at t=50)
Generation  : M5 physics engine (two-phase: startup then steady_state)

Physics context:
  Most dangerous operational moment for multistage pump is startup → ss transition.
  Pressure ramps from ~0.8 bar to ~40 bar over ~60–120 seconds.
  Normal MAE is naturally elevated during this transition.
  A fault injected at t=40 overlaps with the transition MAE elevation.
  Tests whether M8 cluster-conditional thresholds correctly separate
  fault-elevated MAE from transition-elevated MAE.

Physics invariants:
  t=0 to t=49   : startup cluster — MAE elevated (normal startup behaviour)
  t=40 to t=49  : bearing fault also begins (10 steps of overlap)
  t=50 onwards  : steady_state cluster — threshold changes to ss value
  Mot.SV* rises from t=40 regardless of cluster transition
  Cluster-conditional threshold must NOT mask the fault as startup noise

Gates:
  GATE-C15-1: fault detected within 60 timesteps of cluster transition (t=110)
  GATE-C15-2: M8 correctly applies startup threshold for t<50, ss threshold for t≥50
  GATE-C15-3: DANGER or WARN fires in steady_state cluster region
  GATE-C15-4: No missed detection by attributing fault signal to startup noise
```

---

### CONFIG 16 — Compound: Sensor Drift + Bearing Fault Simultaneously
[NEW — Physics-justified addition]

```
Fault types : sensor_drift (Pres.SV calibration drift) +
              bearing_wear (sev 0.3) simultaneously
Cluster     : steady_state
Steps       : 800
Generation  : M5 physics engine (compound scenario)

Physics context:
  In industrial plants, sensor calibration drift accumulates over months.
  A bearing fault developing on top of a slowly drifting pressure sensor
  is the most dangerous compound scenario:
  — Pres.SV drifts +0.05 systematic bias over 500 steps (calibration drift)
  — Bearing wear simultaneously raises Mot.SV* from t=0
  Risk: Pres.SV drift flag might fire for WRONG reason (calibration, not seal)
        while bearing fault is the real issue.
  M8 must disambiguate via cross-channel analysis.

Physics invariants:
  Pres.SV*: systematic upward bias (not seal failure — wrong direction for seal)
  Mot.SV*: gradual rise (bearing degradation)
  Thermal coupling: preserved (bearing = thermal fault, r > 0.85)
  Mot.TV*: rises with 20–40 step lag (thermal consequence of bearing)

Correct M8 response:
  Pres.SV drift flag fires (Mech C) → sensor_failure flag (correct — calibration issue)
  Mot.SV drift flag also fires (Mech C) → bearing_wear_early flag (correct)
  Thermal coupling preserved → confirms bearing, not seal (Pres.SV is sensor drift)
  Dual flags: both fire simultaneously → compound event detected

Gates:
  GATE-C16-1: Mot.SV drift flag fires within 600 steps (bearing_wear_early)
  GATE-C16-2: Pres.SV drift flag fires AND is tagged as sensor_failure
              (upward drift − wrong direction for seal failure which is downward)
  GATE-C16-3: thermal_coupling preserved (r > 0.80) — confirms bearing not seal
  GATE-C16-4: M8 output contains BOTH early_fault_type signals simultaneously
  GATE-C16-5: XGBoost classifies bearing_wear as primary fault
              (not sensor_failure — sensor_failure is secondary compound element)
```

---

## DETECTION COVERAGE MATRIX

| Config | Fault Type | Severity | Steps | Primary Detection | Key Gate |
|---|---|---|---|---|---|
| C1 | bearing_wear | 0.6 | 500 | Single-window DANGER | lag ≤ 60 |
| C2 | bearing_wear | 0.6 | 500 | Single-window (high noise) | lag ≤ 90 |
| C3 | bearing_wear | 0.25 | 1500 | Mech C Mot.SV drift | WATCH ≤ 15 min |
| C4 | cavitation | 0.7 | 300 | Direct DANGER bypass | lag ≤ 30 |
| C5 | cavitation | 0.7 | 300 | Direct DANGER (high noise) | lag ≤ 50 |
| C6 | seal_failure | 0.3 | 1000 | Mech C Pres.SV drift | WATCH ≤ 20 min |
| C7 | seal_failure | 0.8 | 400 | Single-window DANGER | lag ≤ 60 |
| C8 | overloading | 0.6 | 800 | Mech C Temp.SV drift | WATCH ≤ 15 min |
| C9 | impeller_imbalance | 0.7 | 400 | Single-window DANGER | lag ≤ 60 |
| C10 | sensor_failure | 0.8 | 300 | Flatline std → flag | flag ≤ 100 steps |
| C11 | bearing + cavitation | 0.5+0.7 | 300 | Cavitation dominates | DANGER ≤ 30 |
| C12 | bearing (thermal broken) | 0.7 | 500 | Mech C Mot.SV only | WARN ≤ 90 |
| C13 | seal_failure slow drift | 0.2 | 2000 | Mech C Pres.SV drift | WATCH ≤ 20 min |
| C14 | bearing slow drift | 0.25 | 1500 | Mech C Mot.SV drift | WATCH ≤ 15 min |
| C15 | bearing (transition) | 0.6 | 400 | Cluster-conditional | detect ≤ t=110 |
| C16 | bearing + sensor drift | 0.3+drift | 800 | Dual Mech C flags | both flags fire |

---

## RECALIBRATION LOOP

```
IF any gate FAILS → status: RECALIBRATION_REQUIRED

Recalibration protocol per config:

Config 3 FAILS (bearing slow drift WATCH > 15 min):
  → Reduce Mech C window for Mot.SV: 300 → 250 windows
  → Re-run M8 Gate M8-11 (thermal lag) to ensure still valid
  → Re-run M12 Config 3 and Config 14

Config 6 FAILS (seal slow drift WATCH > 20 min):
  → Reduce Mech C window for Pres.SV: 300 → 200 windows
  → Lower Spearman threshold for Pres.SV: 0.70 → 0.60
  → Re-run M8 Gate M8-9, Gate M8-10
  → Re-run M12 Config 6 and Config 13

Config 8 FAILS (overloading WATCH > 15 min):
  → Lower Spearman threshold for Temp.SV: 0.70 → 0.65
  → Re-validate M8 Gate M8-7 (FPR impact check)
  → Re-run M12 Config 8

Config 12 FAILS (bearing detection > 90 steps, thermal coupling broken):
  → Reduce Mot.TV weight: 0.3 → 0.1 in M8
  → Verify Gate M8-11 still passes (thermal lag not worsened)
  → Retrain M8 → Re-run full M12 suite

Config 13 FAILS (seal liability gate — most critical):
  → Reduce Mech C window for Pres.SV: 300 → 200
  → Reduce Spearman threshold: 0.70 → 0.60 for Pres.SV
  → Add more mild seal sequences (sev 0.2) to M6 → re-run M8 Mech C calibration
  → Re-run M12 Config 13 (must pass before any deployment)

Config 15 FAILS (cluster transition — fault masked as startup noise):
  → Widen cluster-conditional threshold at startup → ss boundary
  → Add transition_flag feature: if cluster changes within last 20 windows
    → apply tighter threshold for fault channels specifically
  → Re-run M12 Config 15

Config 16 FAILS (compound scenario — bearing not detected under sensor drift):
  → Add inter-channel consistency check:
    if Pres.SV drift = POSITIVE (upward) AND Mot.SV drift = POSITIVE
    → tag Pres.SV as sensor_drift (not seal) automatically
    (seal failure is always NEGATIVE Pres.SV drift)
  → Re-run M12 Config 16

AFTER RECALIBRATION:
  Re-run FULL M8 gate suite (all 13 gates) before re-running M12.
  Do not re-run M12 on partial M8 fixes — always full M8 gate check first.
```

---

## SAFETY GATE

```
IF all 16 configs pass all gates:
  status: PRODUCTION_VALIDATED
  Output: outputs/M12_validation_certificate.md
  Contents of certificate:
    — Date of validation
    — M8 model SHA (from torch.save state dict hash)
    — All 16 config results (PASS/FAIL + detection_lag)
    — Statement: "M8 LSTM-AE v2 validated on 16 adversarial configurations
                  generated fresh from M5 physics engine.
                  System cleared for deployment on 110 kW, 40 bar,
                  7-stage multistage centrifugal pump — CIRA SACIP configuration."
    — Disclaimer: "Validation is specific to CIRA SACIP sensor installation.
                   Deployment on different pump or sensor layout requires
                   re-validation with installation-specific data."

IF any config FAILS:
  status: RECALIBRATION_REQUIRED
  Output: outputs/M12_recalibration_log.csv
  Contents:
    — Config ID, gate ID, failure value, target value, recalibration action
    — Re-run instructions (see recalibration loop above)
  Deployment BLOCKED until PRODUCTION_VALIDATED.
```

---

## M12 OUTPUTS

```
src/module_12_validation_suite.py
outputs/M12_validation_log.csv          ← per-config, per-gate results
outputs/M12_detection_latency.png       ← latency bar chart (all 16 configs)
outputs/M12_alert_state_traces.png      ← alert state over time per config
outputs/M12_channel_drift_traces.png    ← Mech C Spearman per config
outputs/M12_validation_certificate.md   ← if PRODUCTION_VALIDATED
outputs/M12_recalibration_log.csv       ← if RECALIBRATION_REQUIRED
outputs/reports/module_12_validation_report.md
```

---

## M12 PASTE TEXT KEYS

```
M12_configs_total          : 16
M12_configs_passed         : [X/16]
M12_configs_failed         : [list of failed config IDs]
M12_C3_watch_latency_steps : [value — gate ≤ 900]
M12_C6_watch_latency_steps : [value — gate ≤ 1200]
M12_C8_watch_latency_steps : [value — gate ≤ 900]
M12_C13_watch_latency_steps: [value — gate ≤ 1200 — LIABILITY GATE]
M12_C14_watch_latency_steps: [value — gate ≤ 900 — LIABILITY GATE]
M12_C4_danger_latency_steps: [value — gate ≤ 30 — cavitation]
M12_C11_compound_result    : cavitation_primary/bearing_primary/both_flagged
M12_C12_thermal_broken_lag : [steps — gate ≤ 90]
M12_C15_transition_result  : PASS/FAIL
M12_C16_compound_result    : both_flags_fired/partial/FAIL
M12_validation_status      : PRODUCTION_VALIDATED / RECALIBRATION_REQUIRED
M12_certificate_generated  : True/False
Status_for_deployment      : CLEARED / BLOCKED
```

---

*File: module_M12_validation_suite.md*
*Version: 1.0 | Created: 2026-04-12*
*Derived from: module_pathway_M1_to_M12_v10.md + modules_M7_M8_critical_ML.md*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*All 16 configs generated fresh from M5 physics engine — never seen by M8 during training*
