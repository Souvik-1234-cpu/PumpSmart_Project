# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
# PART 2A OF 3 — Alert Machine, Gates, Adaptive Actions

**Document version:** v3.0 — 22-class M6B alignment (v14.0)
**Date:** 2026-04-16
**Part 1A (LSTM-AE + Fuzzy):** `module_M8_lstm_ae_v2_architecture.md`
**Part 1B (Mechanisms):** `module_M8_lstm_ae_v2_mechanisms.md`
**Part 2B (Outputs + Paste Keys):** `module_M8_lstm_ae_v2_outputs_and_paste.md`
**Prerequisite:** M7 all 16 gates passed | `M7_all_16_gates_pass = True`
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Status:** NOT STARTED — begins only after M7 gates confirmed

> ⚠️ READ Part 1A + Part 1B FIRST before reading this file.
> Part 2A = alert machine + gates + adaptive actions.
> Part 2B = outputs, paste keys, dependency chain, revision history.

---

## STAGE 4 — Four-State Alert Machine

```
STATE DEFINITIONS:
  NORMAL  : rolling_score < 2.0  AND no channel_drift_flag AND no slope trigger
             AND cusum_bearing_gradual_flag = False
             AND rolling_baseline_drift_flag = False
  WATCH   : rolling_mean_200 > 0.085  OR slope trigger  OR ANY channel_drift_flag
             OR cusum_bearing_gradual_flag = True
             OR rolling_baseline_drift_flag = True
  WARN    : rolling_mean_100 > 0.095  OR rolling_score in [2.0, 3.5]
             OR (cusum_bearing_gradual_flag AND Mech_C_MotSV_slow_drift_flag)
             OR (cusum_bearing_gradual_flag AND rolling_baseline_drift_flag)
  DANGER  : single_window_MAE > cluster_threshold  OR rolling_score > 3.5
             OR (cusum + rolling_baseline + Mech_C_MotSV_slow_drift all True)

STATE ESCALATION:
  NORMAL  → WATCH  : sustained low-level anomaly / drift beginning
  WATCH   → WARN   : trend confirmed over 100+ windows
  WARN    → DANGER : threshold crossed — immediate maintenance required
  DANGER  → WARN   : MAE below threshold for 50+ consecutive windows
  WARN    → WATCH  : rolling_mean_200 below 0.085 for 200+ windows
  WATCH   → NORMAL : ALL mechanisms clear for 300+ consecutive windows
                     AND cusum S_pos < 0.5×H AND drift_ratio < 1.05

FOUR-STATE M10 UI MESSAGES:
  NORMAL  : "System operating within normal parameters"
  WATCH   : "Early anomaly trend — monitor closely"
  WARN    : "Sustained anomaly — schedule maintenance"
  DANGER  : "Fault confirmed — immediate action required"
```

### Fault-Specific Alert Exceptions

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CAVITATION EXCEPTION (Finding F5 — MAE = 0.675, 6.1× threshold):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if cluster == 'startup' AND single_window_MAE > 3 × cluster_threshold:
      alert_state = DANGER  # bypass WATCH and WARN entirely
  Physics: cavitation is acute hydraulic shock.
  Impeller pitting begins within 60–180s of onset.
  DO NOT route cavitation through rolling mean accumulator.
  Gate M8-12: ZERO cavitation DANGER alerts outside startup cluster.

  cavitation_intermittent (Group D label 18):
  MAE spikes during burst windows only — drops between bursts.
  Alert: WATCH on first burst, WARN after 3 bursts in 100 windows,
  DANGER if burst frequency increases (slope of burst_count > 0).
  Do NOT de-escalate to NORMAL between bursts — hold WATCH minimum.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OVERLOADING EXCEPTION (Finding F1 — Gate 3 = 0.00%):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Primary detection = Mech C Temp.SV drift (POSITIVE Spearman > 0.70)
  Gate M8-7 denominator = overloading validation sequences ONLY
  Gate M8-7 numerator = sequences where Temp.SV drift fires ≤15 min
  Single-window MAE crossing excluded from overloading TPR measurement

  overloading_cyclic (Group D label 20):
  Temp.SV shows sawtooth with RISING BASELINE — not monotonic.
  Detection: Mech B slope of baseline_drift > 0.0002/window
  PLUS Temp.SV Spearman > 0.70 on baseline-detrended signal.
  Alert: WATCH on first cycle, WARN after baseline drift confirmed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SEAL FAILURE EXCEPTION (Finding F2 — Gate 3 = 29.17%):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Primary detection   = Mech C Pres.SV drift flag (NEGATIVE Spearman)
  Secondary confirm   = Mech A rolling mean
  Mild seal (0.2–0.4): Mech C fires first → Mech A confirms
  Severe seal (0.5+) : single-window MAE also fires
  Gate M8-9: Pres.SV drift WATCH ≤20 min for sev 0.2 sequences
  Gate M8-10: Pres.SV drift flag fires BEFORE total MAE reaches WARN state

  seal_failure_fast (Group D label 19):
  Pres.SV drops in ≤20 steps — slope extremely steep.
  Single-window MAE fires immediately — no need for rolling accumulation.
  Alert: DANGER within 1–3 windows of onset.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BEARING WEAR GRADUAL EXCEPTION (label 21 — Paris–Erdogan sub-threshold):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Single-window MAE NEVER crosses threshold at sev 0.05–0.25 — by design.
  Standard rolling mean (Mech A) also WILL NOT fire — by design.
  DO NOT treat absence of MAE crossing as model failure.
  DO NOT raise global threshold to make label 21 invisible.

  Detection path (mandatory, in order):
    Layer 3 CUSUM (Stage 3D, Part 1B)    → WATCH  [primary]
    Layer 4 Rolling Baseline (Stage 3E)  → WATCH→WARN [confirm]
    Mech C Mot.SV slow drift (0.65, 500w) → WARN [secondary confirm]
    All three simultaneously              → DANGER

  Alert state escalation for label 21:
    cusum_bearing_gradual_flag only           → WATCH
    cusum + rolling_baseline                  → WARN
    cusum + rolling_baseline + Mech_C_slow    → DANGER

  M10 UI note for label 21:
    "Gradual bearing degradation detected — Paris–Erdogan regime.
     MAE sub-threshold by design. Alert via cumulative drift analysis."

  Gate M8-14-ext (detail in Part 2B):
    ≥75% label 21 sequences → WATCH via CUSUM within 500 windows
    ≥60% label 21 sequences → WARN via CUSUM + Layer 4 within 800 windows
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GROUP C — MASKED FAULT ALERT BEHAVIOUR (5 classes, labels 13–17):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Masked faults: primary detection channel is flatline (sensor dead).
  Alert MUST route through secondary Mech C path.
  Max reachable alert state = WARN if secondary signal only.
  DANGER requires either: (a) secondary channel MAE crosses threshold,
  OR (b) 3+ Mech C flags simultaneously active.
  Label 17 (seal_failure_MotPV_masked): weakest secondary path (Pres.SV only).
    Max reachable state = WARN unless Pres.SV MAE independently crosses threshold.
    Gate M8-13 documents label 17 expected to be lowest TPR in Group C.
  M10 UI note: "Primary sensor unavailable — detection via secondary signal"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GROUP B — COMPOUND FAULT ALERT BEHAVIOUR (6 classes, labels 7–12):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 1 (primary fault active): alert follows primary fault exception rules.
  Phase 2 (secondary fault onset at secondary_onset_lag):
    Additional Mech C flag fires on secondary channel → escalate alert by 1 level.
    If already at DANGER: hold DANGER, add secondary_fault_type to output dict.
  Expected: all 6 compound sequences reach DANGER within 200 windows.
  Gate M8-14: Group B TPR ≥85% reaching DANGER (6 classes, labels 7–12).

  Label 12 (bearing_wear→seal_failure) specific:
  Phase 1: Mot.SV drift fires (bearing primary) → WATCH
  Phase 2: Pres.SV drift fires at secondary_onset_lag → WARN→DANGER
  Both Mech C flags active simultaneously = highest confidence compound.
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
  Synthetic sequences from physics equations → a purely synthetic threshold
  is physics-biased. Real CIRA validation anchors to actual pump behaviour:
  manufacturing tolerances, fluid impurities, ambient conditions.
  Prevents systematic false-alarm drift in deployment.
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
Step 6: Update Mech C: per-channel Spearman drift (300-window standard)
        Label 17: Pres.SV Spearman window=300, threshold=0.60
        Label 21: Mot.SV Spearman window=500, threshold=0.65
        Check flatline std < 0.001 for sensor_failure detection
        Check multi_sensor_anomaly_count for Group E (count ≥ 2)
Step 6B: Update Layer 3 CUSUM (Mot.SV channel error, label 21 only)
         Update Layer 4 Rolling Baseline (short/long ratio, label 21 only)
         [Layer 4 disabled until 5000-window burn-in complete]
Step 7: Apply fault-specific exceptions:
        — Cavitation (label 3):          startup + MAE > 3×threshold → DANGER immediately
        — Cavitation_intermittent (18):  burst tracking → WATCH→WARN on burst count
        — Overloading (label 5):         Temp.SV Spearman > 0.70 (positive) → overloading_early
        — Overloading_cyclic (20):       baseline drift slope → WATCH→WARN
        — Seal failure (label 4):        Pres.SV Spearman > 0.70 (negative) + thermal_decoupling
        — Seal_fast (19):                steep Pres.SV slope → DANGER within 3 windows
        — Bearing wear (label 1):        Mot.SV Spearman > 0.70 (positive) + coupling preserved
        — Bearing_gradual (21):          CUSUM → WATCH; CUSUM+baseline → WARN; all 3 → DANGER
        — Bearing→seal compound (12):    Mot.SV drift then Pres.SV drift at lag → escalate
        — Sensor fail single (6):        channel std < 0.001 → sensor_failure
        — Sensor fail 2ch (E-a, E-b):    multi_sensor_anomaly_count ≥ 2 → sensor_failure_2ch
        — Group B compound (7–12):       2nd Mech C flag at secondary_onset_lag → escalate
        — Group C masked (13–17):        max alert = WARN unless secondary MAE crosses threshold
        — Label 17 masked:               max alert = WARN (weakest secondary path)
Step 8: Determine alert state → output dict

OUTPUT DICT (complete):
{
  alert_state                  : "NORMAL" / "WATCH" / "WARN" / "DANGER"
  anomaly_flag                 : bool
  fuzzy_membership             : float [0, 1]
  rolling_mean_mae             : float
  mae_slope                    : float
  channel_drift                : {
    "Mot.SV": bool, "Pmp.SV": bool, "Pres.SV": bool, "Temp.SV": bool,
    "Mot.TV": bool, "Pmp.TV": bool, "Mot.PV": bool, "Pmp.PV": bool
  }
  cusum_bearing_gradual_flag   : bool         (label 21 Layer 3)
  cusum_S_pos                  : float        (current accumulator value)
  rolling_baseline_drift_flag  : bool         (label 21 Layer 4)
  drift_ratio                  : float        (short/long baseline ratio)
  early_fault_type             : None / "overloading_early" / "seal_failure_early" /
                                 "bearing_wear_early" / "bearing_wear_gradual_early" /
                                 "sensor_failure" / "sensor_failure_2ch" /
                                 "compound_secondary_onset"
  secondary_fault_type         : None / string
  masked_detection             : bool
  multi_sensor_count           : int
  severity                     : "LOW" / "MEDIUM" / "HIGH"
  uncertainty_std              : float
  confidence                   : float [0, 1]
  attention_heatmap            : array(50,)
  cluster                      : "startup" / "steady_state" / "high_load" / "cooldown"
}
```

---

## M8 All 15 Validation Gates

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

GATE-M8-3 : Youden’s J
             > 0.85  (J = TPR − FPR)
             Computed on Group A fault pool vs full normal pool

GATE-M8-4 : Separation ratio
             > 5.0×  (M4 baseline was 4.11×)
             = mean_fault_MAE / mean_normal_MAE
             Computed on Group A included fault population (cavitation dominated)

GATE-M8-5 : False alarms absolute count
             ≤8 windows on normal validation pool
             (same standard as M4 — 0.55% of 1457 val windows)

GATE-M8-6 : Fuzzy boundaries valid
             lower_bound < upper_bound
             lower_bound in [0.07, 0.09]
             upper_bound in [0.15, 0.50]
             Transition zone width ≥0.05
             If width < 0.05 → selective exclusion not working → audit

GATE-M8-7 : Overloading detection via Mech C ONLY
             ≥80% TPR on mild overloading sequences (sev 0.2–0.5)
             via Temp.SV Spearman drift flag within ≤15 min [Finding F1]
             Applies to overloading (label 5) and overloading_cyclic (label 20)
             Document if < 80% — do NOT raise global threshold to compensate

GATE-M8-8 : Attention seam check
             seam_ratio = mean_attention(t=49,50) / mean_attention(t=10,40)
             Gate: seam_ratio < 1.0 for bearing_wear sequences [Finding F3]
             FAIL action: add gradient penalty at t=49–50, retrain M8

GATE-M8-9 : Slow drift seal detection
             WATCH fires ≤20 min for seal_failure (label 4) sev 0.2 sequences
             Via Pres.SV Spearman drift (NEGATIVE) [Finding F2]
             thermal_decoupling_flag must ALSO be True simultaneously

GATE-M8-10 : Pres.SV drift fires first
             For seal_failure mild sequences (label 4):
             timestep(Pres.SV drift flag) < timestep(WARN state) [Finding F2]

GATE-M8-11 : Thermal lag validation
             Peak Mot.SV reconstruction error precedes peak Mot.TV error
             by 20–40 timesteps for bearing_wear sequences
             [Physics: heat conduction lag — M2 r=0.9793 + M5 Euler integration]
             FAIL = model detecting thermal consequence, not mechanical cause

GATE-M8-12 : Cavitation cluster exclusivity
             ZERO cavitation DANGER alerts on steady_state or high_load
             in normal validation pool [Finding F5]
             FAIL → audit M6B cluster assignment

GATE-M8-13 : Group C masked fault TPR (5 classes, labels 13–17)
             ≥65% TPR on ALL Group C sequences via secondary Mech C path
             Report per masked-class F1 individually:
               bearing_wear_MotSV_masked (13)  : target ≥65%
               cavitation_PresSV_masked (14)   : target ≥65%
               overloading_TempSV_masked (15)  : target ≥65%
               impeller_PmpSV_masked (16)      : target ≥65%
               seal_failure_MotPV_masked (17)  : target ≥50% (weakest — Pres.SV only)
             FAIL on any class < 50% → BLOCK → verify M6B Gate G10 secondary signal

GATE-M8-14 : Group B, D, E TPR (22-class aligned)
             Group B (labels 7–12, 6 compound classes) : ≥85% reaching DANGER
               Report each compound class separately
               Label 12 (bearing→seal): Mot.SV then Pres.SV drift → DANGER expected
             Group D (labels 18–20, 3 variant classes)  : ≥78% correct alert path
               Label 18 cavitation_intermittent : burst tracking → WATCH+
               Label 19 seal_failure_fast        : DANGER within 3 windows
               Label 20 overloading_cyclic       : WATCH via baseline drift Mech B+C
             Group E (E-a, E-b, 2 multi-sensor)  : ≥88% multi_sensor_count=2 detected
             Report each group separately — do NOT aggregate
             FAIL on any group → document in paste text, flag for M12 adversarial

GATE-M8-14-ext : Label 21 (bearing_wear_gradual) CUSUM + Layer 4 detection
             Full spec in Part 2B (`module_M8_lstm_ae_v2_outputs_and_paste.md`)
             Gate targets:
               ≥75% label 21 sequences → WATCH via CUSUM within 500 windows
               ≥60% label 21 sequences → WARN via CUSUM+Layer4 within 800 windows
             FAIL → retune CUSUM H and k on label 21 mild calibration subset
             DO NOT raise MAE threshold to fix this gate — CUSUM params only
```

---

## Adaptive Actions After M8

| M8 Result | Gate | Adaptive Action |
|-----------|------|-----------------|
| Overloading TPR < 80% | M8-7 | Lower Spearman threshold 0.70→0.65 for Temp.SV ONLY. Re-validate FPR impact |
| Seal WATCH > 20 min | M8-9 | Shorten Mech C window 300→200 for Pres.SV ONLY |
| FPR > 5% at startup | M8-2 | Raise startup cluster threshold ONLY — never global threshold |
| Separation ratio < 5.0× | M8-4 | Audit normal pool, remove near-fault windows, retrain |
| Attention seam ratio > 1.0 | M8-8 | Add gradient penalty at t=49–50. Retrain M8 |
| Gate M8-11 fails (thermal lag) | M8-11 | Reduce Mot.TV weight 0.3→0.1. Force vibration-first detection. Retrain |
| Gate M8-12 fails (cavitation in high_load) | M8-12 | Audit M6B cluster assignment — startup seed mis-labeling |
| Gate M8-13 fails any class < 50% | M8-13 | Verify M6B Gate G10 secondary signal. Increase masked sequences 1200→2000 |
| Label 17 TPR < 40% | M8-13 | Lower Spearman threshold to 0.55 for Pres.SV (label 17 only). Document in paste text |
| Group B TPR < 85% | M8-14 | Increase compound sequences in M6B Step 1. Verify secondary_onset_lag in M6.5r |
| Label 12 not reaching DANGER | M8-14 | Verify bearing→seal causal lag in M6B. Check Mot.SV + Pres.SV both active |
| Group D label 18 burst miss | M8-14 | Implement burst_count tracker in Step 7 inference |
| Group D label 20 cyclic miss | M8-14 | Implement baseline_detrend in Mech B for cyclic signal |
| Group E multi-sensor miss | M8-14 | Verify multi_sensor_anomaly_count in M6.5r Gate D3 |
| Label 21 CUSUM < 75% WATCH | M8-14-ext | Retune CUSUM H: lower toward 4×sigma. Re-run on label 21 mild subset |
| Label 21 Layer 4 < 60% WARN | M8-14-ext | Shorten baseline_long window 5000→3000. Retune drift_ratio threshold |
| Energy conservation fail | M8-13 | Add L2 regularization on 64-dim bottleneck, retrain |
| All 15 gates pass | — | Proceed to Part 2B for outputs + paste text, then to M9 |

---

## Cross-Module Invariants Relevant to M8

1. Models saved: `torch.save(state_dict)` | Loaded: `map_location='cpu'` for M10
2. Normalization baselines LOCKED at `M3_normalization_config.json`
3. M4 threshold `0.110058` is starting reference — M8 produces its own cluster-conditional thresholds
4. Channel weights INCREASED vs M4 — Fisher-validated from M6.5
5. Faults NEVER in training pool — LSTM-AE is anomaly detector only
6. Mech C operates on RAW channel errors — bypasses weight matrix by design
7. Threshold calibrated on REAL CIRA validation set — not synthetic
8. All M6B Groups (A–E, 22 classes) in fault validation pool only — never in training
9. Cavitation gate: startup cluster only — any cavitation DANGER outside startup = FAIL
10. `if pump_type == 'household': return physics_advisory_only()` — NO EXCEPTIONS
11. Label strings always resolved via `fault_rules_v3.json` — NEVER hardcoded
12. Group C masked fault max alert = WARN unless secondary MAE crosses threshold
13. Label 17 max alert = WARN (weakest secondary path — Pres.SV only)
14. Group B compound: 2nd Mech C flag fires at secondary_onset_lag — not before
15. M8 outputs raw alert_state dict — M10 handles all UI display formatting
16. Layer 3 CUSUM + Layer 4 Rolling Baseline = label 21 ONLY — do NOT apply broadly
17. Layer 4 disabled during burn-in (≤5000 windows) — CUSUM only during burn-in
18. Label 21 sub-threshold MAE = design behaviour — DO NOT raise threshold to fix

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic — 13 gates |
| v2.0 | 2026-04-15 | Split Part 1+2. 13→14 gates. Group B/D/E exceptions added to Stage 4. Output dict extended. 35 paste keys. |
| v3.0 | 2026-04-16 | **v14.0 UPGRADE + FURTHER SPLIT**: This file = Part 2A (alert machine + gates + adaptive actions). Paste keys + outputs + dependency chain → `module_M8_lstm_ae_v2_outputs_and_paste.md` (Part 2B). M7 prerequisite 15→16 gates. 22-class. Stage 4 state machine: CUSUM + rolling baseline flags added to NORMAL/WATCH/WARN/DANGER conditions; WATCH→NORMAL requires cusum + drift_ratio clear. Alert exceptions: label 21 block added (Paris–Erdogan, 3-layer escalation, M10 UI note); Group C updated to 5 classes labels 13–17, label 17 weakest path noted; Group B updated to 6 classes labels 7–12, label 12 specific behaviour added; overloading cyclic label 18→20; seal fast label 17→19; cavitation intermittent label 16→18. Inference protocol Step 6B added (CUSUM + Layer 4 updates); Step 7 label refs corrected; output dict: cusum_bearing_gradual_flag, cusum_S_pos, rolling_baseline_drift_flag, drift_ratio, bearing_wear_gradual_early added. 14→15 gates: Gate M8-13 updated to 5 classes labels 13–17 with label 17 floor=50%; Gate M8-14 updated to 6+3+2 classes with label 12 specific note; Gate M8-14-ext added (stub + targets, detail in Part 2B). Adaptive actions: label 17 action, label 12 action, label 21 CUSUM/Layer4 retuning actions. Invariants: 14→18, label 21 + Layer 3/4 invariants added. |

---

*GitHub is the ONLY source of truth for this spec.*
*Part 1A: `module_M8_lstm_ae_v2_architecture.md`*
*Part 1B: `module_M8_lstm_ae_v2_mechanisms.md`*
*Part 2B: `module_M8_lstm_ae_v2_outputs_and_paste.md`*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
