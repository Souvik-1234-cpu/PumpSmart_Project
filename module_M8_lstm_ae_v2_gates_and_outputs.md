# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
## Part 2A of 3 — Alert State Machine, Gates, Adaptive Actions

| Field | Value |
|-------|-------|
| **Document version** | v4.0 — v14.2 TCN-AE score routing + gate alignment |
| **Date** | 2026-04-19 |
| **Part 1A (Architecture)** | `module_M8_lstm_ae_v2_architecture.md` |
| **Part 1B (Mechanisms)** | `module_M8_lstm_ae_v2_mechanisms.md` |
| **Part 2B (Outputs + Paste Keys)** | `module_M8_lstm_ae_v2_outputs_and_paste.md` |
| **Prerequisite** | M7 all 16 gates passed — `M7_all_16_gates_pass = True` |
| **Asset** | 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP) |
| **Status** | NOT STARTED — begins only after M7 gates confirmed |

> **NOTE:** READ Part 1A + Part 1B FIRST before reading this file.
> - Part 1A = Level 1 LSTM-AE + Level 2 TCN-AE architecture + score routing rules
> - Part 1B = TCN Glass analogy + Mech A/B/C + Layer 3 CUSUM + Layer 4 Rolling Baseline
> - Part 2A = Alert state machine + 15 gates + adaptive actions *(this file)*
> - Part 2B = Outputs + paste keys + dependency chain

> **⚠️ SCORE ROUTING RULES — Invariant 19 — NEVER CROSS — enforced throughout this file:**
>
> | Score | Routes To |
> |-------|-----------|
> | `score_B` | CUSUM only |
> | `score_A` | Rolling Baseline only |
> | `score_C` | XGBoost only |

---

## Stage 4 — Four-State Alert Machine

### State Definitions

**NORMAL:**
```
rolling_score < 2.0
AND no channel_drift_flag
AND no slope trigger
AND cusum_bearing_gradual_flag = False
AND rolling_baseline_drift_flag = False
AND score_C < score_C_normal_p95
```

**WATCH:**
```
rolling_mean_200 > 0.085
OR slope trigger
OR ANY channel_drift_flag
OR cusum_bearing_gradual_flag = True
OR rolling_baseline_drift_flag = True
OR score_C > score_C_normal_p95
```

**WARN:**
```
rolling_mean_100 > 0.095
OR rolling_score in [2.0, 3.5]
OR (cusum_bearing_gradual_flag AND Mech_C_MotSV_slow_drift_flag)
OR (cusum_bearing_gradual_flag AND rolling_baseline_drift_flag)
OR score_C > score_C_warn_threshold
```

**DANGER:**
```
single_window_MAE > cluster_threshold
OR rolling_score > 3.5
OR (cusum + rolling_baseline + Mech_C_MotSV_slow_drift all True)
OR score_C > score_C_danger_threshold
```

**score_C threshold calibration:**

| Threshold | Derivation |
|-----------|-----------|
| `score_C_normal_p95` | P95 of score_C on full 9,711-window normal pool |
| `score_C_warn_threshold` | P5 of score_C on Group B sequences at phase transition |
| `score_C_danger_threshold` | P1 of score_C on confirmed compound sequences |

### State Escalation Rules

| Transition | Condition |
|-----------|-----------|
| NORMAL → WATCH | Sustained low-level anomaly / drift beginning |
| WATCH → WARN | Trend confirmed over 100+ windows |
| WARN → DANGER | Threshold crossed — immediate maintenance required |
| DANGER → WARN | MAE below threshold for 50+ consecutive windows |
| WARN → WATCH | `rolling_mean_200` below 0.085 for 200+ windows |
| WATCH → NORMAL | ALL mechanisms clear for 300+ consecutive windows **AND** `cusum S_pos < 0.5*H` **AND** `drift_ratio < 1.05` **AND** `score_C < score_C_normal_p95` |

### M10 UI Messages

| State | Message |
|-------|---------|
| NORMAL | "System operating within normal parameters" |
| WATCH | "Early anomaly trend — monitor closely" |
| WARN | "Sustained anomaly — schedule maintenance" |
| DANGER | "Fault confirmed — immediate action required" |

---

## Fault-Specific Alert Exceptions

### Cavitation Exception (Finding F5 — MAE = 0.675, 6.1×)

```python
if cluster == 'startup' AND single_window_MAE > 3 * cluster_threshold:
    alert_state = DANGER   # bypass WATCH and WARN entirely
```

> **Physics:** Cavitation is acute hydraulic shock. Impeller pitting begins within 60–180s of onset. **DO NOT route cavitation through rolling mean accumulator.**
> **Gate M8-12:** ZERO cavitation DANGER alerts outside startup cluster.

**`cavitation_intermittent` (Group D label 18):**
- MAE spikes during burst windows only — drops between bursts
- Alert: WATCH on first burst, WARN after 3 bursts in 100 windows, DANGER if burst frequency increases (slope of `burst_count > 0`)
- Do NOT de-escalate to NORMAL between bursts — hold WATCH minimum
- `score_C` may also elevate if burst pattern resembles compound transition

### Overloading Exception (Finding F1 — Gate 3 = 0.00%)

> **Primary detection** = Mech C Temp.SV drift (POSITIVE Spearman > 0.70)

- Gate M8-7 denominator = overloading validation sequences ONLY
- Gate M8-7 numerator = sequences where Temp.SV drift fires ≤15 min
- Single-window MAE crossing excluded from overloading TPR measurement

**`overloading_cyclic` (Group D label 20):**
- Temp.SV shows sawtooth with RISING BASELINE — not monotonic
- Detection: Mech B slope of `baseline_drift > 0.0002/window` PLUS Temp.SV Spearman > 0.70 on baseline-detrended signal
- Alert: WATCH on first cycle, WARN after baseline drift confirmed

### Seal Failure Exception (Finding F2 — Gate 3 = 29.17%)

| Path | Condition |
|------|-----------|
| Primary detection | Mech C Pres.SV drift flag (NEGATIVE Spearman) |
| Secondary confirm | Mech A rolling mean |
| Mild seal (sev 0.2–0.4) | Mech C fires first → Mech A confirms |
| Severe seal (sev 0.5+) | Single-window MAE also fires |

- Gate M8-9: Pres.SV drift WATCH ≤20 min for sev 0.2 sequences
- Gate M8-10: Pres.SV drift flag fires BEFORE total MAE reaches WARN state

**`seal_failure_fast` (Group D label 19):**
- Pres.SV drops in ≤20 steps — slope extremely steep
- Single-window MAE fires immediately — no need for rolling accumulation
- Alert: DANGER within 1–3 windows of onset
- Governing equation: `Q_leak = Cd * A_orifice * sqrt(2 * dP / rho)`
- Seal blowout is NOT laminar — **do NOT use Hagen-Poiseuille model**

### Bearing Wear Gradual Exception (Label 21 — Paris-Erdogan sub-threshold)

> Single-window MAE **NEVER** crosses threshold at sev 0.05–0.25 — by design.
> Standard rolling mean (Mech A) also **WILL NOT** fire — by design.
> **DO NOT** treat absence of MAE crossing as model failure.
> **DO NOT** raise global threshold to make label 21 detectable.

**Detection path (mandatory, in order):**

| Layer | Path | Alert State |
|-------|------|-------------|
| Layer 3 CUSUM on `score_B` (Stage 3E, Part 1B) | Primary | WATCH |
| Layer 4 Rolling Baseline on `score_A` (Stage 3F) | Confirm | WATCH → WARN |
| Mech C Mot.SV slow drift (0.65, 500w) | Secondary confirm | WARN |
| All three simultaneously | — | **DANGER** |

**Alert state escalation for label 21:**

| Condition | Alert State |
|-----------|-------------|
| `cusum_bearing_gradual_flag` only | WATCH |
| CUSUM + rolling baseline | WARN |
| CUSUM + rolling baseline + Mech C slow drift | **DANGER** |

> **M10 UI note for label 21:**
> *"Gradual bearing degradation detected — Paris-Erdogan regime. MAE sub-threshold by design. Alert via cumulative drift analysis."*

**Gate M8-14-ext targets:**
- ≥75% label 21 sequences → WATCH via CUSUM within 500 windows
- ≥60% label 21 sequences → WARN via CUSUM + Layer 4 within 800 windows

### Group C — Masked Fault Alert Behaviour (5 classes, labels 13–17)

> Masked faults: primary detection channel is flatline (sensor dead). Alert MUST route through secondary Mech C path. **Max reachable alert state = WARN** if secondary signal only.

DANGER requires either:
- (a) Secondary channel MAE independently crosses cluster threshold, OR
- (b) 3+ Mech C flags simultaneously active

| Label | Class | Note |
|-------|-------|------|
| 17 | `impeller_PmpSV_flatline` | Weakest secondary path (Pmp.PV only). Max reachable = WARN unless Pmp.PV MAE independently crosses threshold. Gate M8-13 documents label 17 as lowest expected TPR in Group C. |

> **M10 UI note:** *"Primary sensor unavailable — detection via secondary signal"*

### Group B — Compound Fault Alert Behaviour (6 classes, labels 7–12)

- **Phase 1** (primary fault active): alert follows primary fault exception rules
- **Phase 2** (secondary fault onset at `secondary_onset_lag`): additional Mech C flag fires on secondary channel → escalate alert by 1 level. If already at DANGER: hold DANGER, add `secondary_fault_type` to output dict
- `score_C` (chain transition from TCN-AE) elevates at phase boundary
- Expected: all 6 compound sequences reach DANGER within 200 windows
- Gate M8-14: Group B TPR ≥85% reaching DANGER (6 classes, labels 7–12)

**Label 12 (`impeller_imbalance->cavitation`) specific:**
- Phase 1: Pmp.SV kurtosis bursts (impeller BPF) → WATCH
- Phase 2: Pres.SV erratic + spikes at `secondary_onset_lag` → WARN → DANGER
- `score_C` spikes sharply at BPF-to-hydraulic-shock transition

**Label 10 (`seal_failure->cavitation`) specific:**
- Longest lag 400–800 steps = 8–16 windows; requires Glass 4–5 (d=8,16) in TCN
- `score_C` accumulates slowly then spikes at NPSHa crossing NPSHr boundary
- Phase 2 DANGER should arrive within 200 windows of secondary onset

---

## Stage 5 — Cluster-Conditional Thresholds

| Cluster | Threshold | Rationale |
|---------|-----------|-----------|
| `startup` | `threshold_startup > 0.110058` | Wider — BPF harmonics elevate MAE |
| `steady_state` | `threshold_steady_state = 0.110058` | M4 baseline — reference anchor |
| `high_load` | `threshold_high_load <= 0.110058` | Tighter — faults caught immediately |
| `cooldown` | `threshold_cooldown ~= 0.110058` | Similar to steady_state |

**Calibration:**
- P99 of normal MAE distribution **per cluster** on full 9,711-window real CIRA normal pool
- Threshold set on REAL CIRA validation set — NOT synthetic (Bias 2 fix)
- Startup threshold wider: accommodates BPF harmonic MAE elevation at pump start
- Store all four values in: `models/M8_threshold_config.json`
- `score_C` thresholds (`normal_p95`, `warn`, `danger`) also stored in `M8_threshold_config.json`

> **Why real CIRA anchoring matters (Bias 2 fix):** Synthetic sequences from physics equations → purely synthetic threshold is physics-biased. Real CIRA validation anchors to actual pump behaviour: manufacturing tolerances, fluid impurities, ambient conditions. Prevents systematic false-alarm drift in deployment.

---

## M8 Production Inference Protocol (8-Step)

```
Step 1: Load cluster label → cluster-conditional threshold from M8_threshold_config.json

Step 2: Run N=20 MC Dropout forward passes (Level 1 LSTM-AE)
        → mean_MAE per channel (R^8) + z_t (R^64) + uncertainty_std

Step 3: Append z_t to circular buffer (63 × 64). Run Level 2 TCN-AE forward pass.
        → score_A (severity), score_B (drift slope), score_C (chain transition)
        Score routing:
          score_B → Step 6B CUSUM only
          score_A → Step 6B Rolling Baseline only
          score_C → output dict (fed to XGBoost in M7/M10)

Step 4: Compute fuzzy fault membership: mu_fault(mean_MAE)
        from M8_fuzzy_config.json [lower_bound, upper_bound]

Step 5: Update rolling accumulator (5-window): sum of last 5 mu_fault scores
        Update Mech A: rolling mean MAE (200-window + 100-window)
        Update Mech B: slope detector (500-window linear regression)

Step 6: Update Mech C: per-channel Spearman drift (300-window standard)
        Label 17: Pmp.PV Spearman window=300, threshold=0.60
        Label 21: Mot.SV Spearman window=500, threshold=0.65
        Check flatline std < 0.001 for sensor_failure detection
        Check multi_sensor_anomaly_count for Group E (count >= 2)

Step 6B: Update Layer 3 CUSUM on score_B (label 21 only):
           S_pos = max(0, S_pos + (score_B[w] - target - k))
           Fire cusum_bearing_gradual_flag if S_pos > H
         Update Layer 4 Rolling Baseline on score_A (label 21 only):
           drift_ratio = mean(score_A, last 6hr) / mean(score_A, last 24hr)
           [Layer 4 disabled until 5,000-window burn-in complete]

Step 7: Apply fault-specific exceptions:
        cavitation (label 3):              startup + MAE > 3*threshold → DANGER immediately
        cavitation_intermittent (label 18):burst tracking → WATCH→WARN on burst count
        overloading (label 5):             Temp.SV Spearman > 0.70 (positive) → overloading_early
        overloading_cyclic (label 20):     baseline drift slope → WATCH→WARN
        seal_failure (label 4):            Pres.SV Spearman > 0.70 (negative) + thermal_decoupling
        seal_failure_fast (label 19):      steep Pres.SV slope → DANGER within 3 windows
        bearing_wear (label 1):            Mot.SV Spearman > 0.70 (positive) + coupling preserved
        bearing_wear_gradual (label 21):   score_B→CUSUM→WATCH; +score_A→baseline→WARN; all 3→DANGER
        impeller→cavitation (label 12):    Pmp.SV kurtosis then Pres.SV erratic → score_C spike
        seal→cavitation (label 10):        score_C slow accumulation → spike at NPSHa/NPSHr crossing
        sensor_failure single (label 6):   channel std < 0.001 → sensor_failure
        sensor_failure_2ch (E-a, E-b):     multi_sensor_anomaly_count >= 2 → sensor_failure_2ch
        Group B compound (labels 7-12):    2nd Mech C flag at secondary_onset_lag → escalate
        Group C masked (labels 13-17):     max alert = WARN unless secondary MAE > threshold
        Label 17 masked:                   max alert = WARN (weakest secondary path)

Step 8: Determine alert state → output dict
```

### Output Dict (Complete)

```json
{
  "alert_state"                : "NORMAL" / "WATCH" / "WARN" / "DANGER",
  "anomaly_flag"               : bool,
  "fuzzy_membership"           : float [0, 1],
  "rolling_mean_mae"           : float,
  "mae_slope"                  : float,
  "score_A"                    : float,
  "score_B"                    : float,
  "score_C"                    : float,
  "channel_drift": {
    "Mot.SV": bool, "Pmp.SV": bool, "Pres.SV": bool, "Temp.SV": bool,
    "Mot.TV": bool, "Pmp.TV": bool, "Mot.PV": bool,  "Pmp.PV": bool
  },
  "cusum_bearing_gradual_flag" : bool,
  "cusum_S_pos"                : float,
  "rolling_baseline_drift_flag": bool,
  "drift_ratio"                : float,
  "early_fault_type"           : null / "overloading_early" / "seal_failure_early" /
                                 "bearing_wear_early" / "bearing_wear_gradual_early" /
                                 "sensor_failure" / "sensor_failure_2ch" /
                                 "compound_secondary_onset",
  "secondary_fault_type"       : null / string,
  "masked_detection"           : bool,
  "multi_sensor_count"         : int,
  "severity"                   : "LOW" / "MEDIUM" / "HIGH",
  "uncertainty_std"            : float,
  "confidence"                 : float [0, 1],
  "attention_heatmap"          : array(50,),
  "cluster"                    : "startup" / "steady_state" / "high_load" / "cooldown",
  "physics_context"            : dict,
  "disclaimer"                 : str
}
```

> **`score_A`** → feeds Rolling Baseline (Layer 4)
> **`score_B`** → feeds CUSUM (Layer 3)
> **`score_C`** → feeds XGBoost M7/M10 (onset_order feature)
> **`physics_context`** → from `fault_rules_v3.json` — Invariant 18 (mandatory in every alert)
> **`disclaimer`** → real-world conditions note — mandatory (Part 1A)

---

## M8 All 15 Validation Gates

| Gate | Description | Target | Notes |
|------|-------------|--------|-------|
| **M8-1** | TPR fault detection — Group A single-source | >90% on Group A fault validation sequences | Report SEPARATELY per fault class. Cavitation ~100% expected — do not let it mask others. Denominator EXCLUDES overloading mild (Gate M8-7) and seal mild (Gate M8-9). |
| **M8-2** | FPR false alarm | <5% on FULL 9,711-window real CIRA normal pool | NOT on 30-window probe subset — that result is INVALID [Finding F6]. Measured cluster-by-cluster: report startup FPR separately. |
| **M8-3** | Youden's J | >0.85 (J = TPR − FPR) | Computed on Group A fault pool vs full normal pool. |
| **M8-4** | Separation ratio | >5.0× (M4 baseline was 4.11×) | = mean_fault_MAE / mean_normal_MAE. Computed on Group A included fault population (cavitation dominated). |
| **M8-5** | False alarms absolute count | ≤8 windows on normal validation pool | Same standard as M4 — 0.55% of 1,457 val windows. |
| **M8-6** | Fuzzy boundaries valid | `lower_bound < upper_bound`; lower in [0.07, 0.09]; upper in [0.15, 0.50]; transition width ≥0.05 | If width <0.05 → selective exclusion not working → audit. |
| **M8-7** | Overloading detection via Mech C ONLY | ≥80% TPR on mild overloading sequences (sev 0.2–0.5) via Temp.SV Spearman drift flag ≤15 min | Applies to label 5 and label 20. Document if <80% — do NOT raise global threshold. [Finding F1] |
| **M8-8** | Attention seam check | `seam_ratio = mean_attention(t=49,50) / mean_attention(t=10,40)` < 1.0 for bearing_wear sequences | FAIL action: add gradient penalty at t=49–50, retrain M8. [Finding F3] |
| **M8-9** | Slow drift seal detection | WATCH fires ≤20 min for label 4 sev 0.2 sequences via Pres.SV Spearman drift (NEGATIVE) | `thermal_decoupling_flag` must ALSO be True simultaneously. [Finding F2] |
| **M8-10** | Pres.SV drift fires first | For seal_failure mild sequences: `timestep(Pres.SV drift flag) < timestep(WARN state)` | [Finding F2] |
| **M8-11** | Thermal lag validation | Peak Mot.SV reconstruction error precedes peak Mot.TV error by 20–40 timesteps for bearing_wear sequences | Physics: heat conduction lag — M2 r=0.9793 + M5 Euler integration. FAIL = model detecting thermal consequence, not mechanical cause. |
| **M8-12** | Cavitation cluster exclusivity | ZERO cavitation DANGER alerts on steady_state or high_load in normal validation pool | FAIL → audit M6B cluster assignment. [Finding F5] |
| **M8-13** | Group C masked fault TPR (5 classes, labels 13–17) | ≥65% TPR ALL Group C via secondary Mech C path | Report per class: label 13 ≥65%, label 14 ≥65%, label 15 ≥65%, label 16 ≥65%, label 17 ≥50% (weakest). FAIL any class <50% → BLOCK → verify M6B Gate G10 secondary signal. |
| **M8-14** | Group B, D, E TPR (22-class aligned) | Group B ≥85% DANGER; Group D ≥78% correct alert path; Group E ≥88% multi_sensor_count=2 | Report each group separately. Label 10: score_C slow then spike at NPSHa crossing. Label 12: score_C spike at BPF-to-shock. FAIL → document in paste text, flag for M12 adversarial. |
| **M8-14-ext** | Label 21 CUSUM + Layer 4 detection | ≥75% WATCH via CUSUM on `score_B` within 500w; ≥60% WARN via CUSUM + Layer 4 on `score_A` within 800w | Full spec in Part 2B. FAIL → retune CUSUM H and k. DO NOT raise MAE threshold. DO NOT cross-route scores (Invariant 19 violation). |
| **M8-15** | score_C calibration — TCN-AE compound gate *(NEW v4.0)* | Normal P95 < `score_C_normal_p95`; ≥80% Group B show `score_C > score_C_warn_threshold` at lag; ≤10% Group A false signal | FAIL → retune TCN-AE score_C head. Verify Group B `secondary_onset_lag` in M6.5r. |

---

## Adaptive Actions After M8

| M8 Result | Gate | Adaptive Action |
|-----------|------|----------------|
| Overloading TPR < 80% | M8-7 | Lower Spearman 0.70→0.65 for Temp.SV ONLY. Re-validate FPR. |
| Seal WATCH > 20 min | M8-9 | Shorten Mech C window 300→200 for Pres.SV ONLY. |
| FPR > 5% at startup | M8-2 | Raise startup cluster threshold ONLY — never global. |
| Separation ratio < 5.0× | M8-4 | Audit normal pool, remove near-fault windows, retrain. |
| Attention seam ratio > 1.0 | M8-8 | Add gradient penalty at t=49–50. Retrain M8. |
| Gate M8-11 fails (thermal lag) | M8-11 | Reduce Mot.TV weight 0.3→0.1. Force vibration-first detection. |
| Gate M8-12 fails (cav in high_load) | M8-12 | Audit M6B cluster assignment — startup seed mis-labeling. |
| Gate M8-13 any class < 50% | M8-13 | Verify M6B Gate G10 secondary signal. Increase masked seqs 1,200→2,000. |
| Label 17 TPR < 40% | M8-13 | Lower Spearman to 0.55 for Pmp.PV (label 17 only). Document. |
| Group B TPR < 85% | M8-14 | Increase compound seqs in M6B Step 1. Verify `secondary_onset_lag`. |
| Label 10 DANGER not reached | M8-14 | Verify seal→cav lag 400–800 in M6B. Check score_C at NPSHa crossing. |
| Label 12 DANGER not reached | M8-14 | Verify impeller→cav lag 100–300 in M6B. Check score_C BPF spike. |
| Group D label 18 burst miss | M8-14 | Implement `burst_count` tracker in Step 7 inference. |
| Group D label 20 cyclic miss | M8-14 | Implement `baseline_detrend` in Mech B for cyclic signal. |
| Group E multi-sensor miss | M8-14 | Verify `multi_sensor_anomaly_count` in M6.5r Gate D3. |
| score_C FPR > 10% on Group A | M8-15 | Retune score_C threshold. Increase TCN-AE score_C head dropout. |
| score_C < 80% on Group B | M8-15 | Verify `secondary_onset_lag` features in M6.5r. Increase Group B seqs. |
| Label 21 CUSUM < 75% WATCH | M8-14-ext | Retune CUSUM H: lower toward 4×sigma(score_B). Re-run label 21 mild. |
| Label 21 Layer 4 < 60% WARN | M8-14-ext | Shorten `baseline_long` window. Retune `drift_ratio` 1.10→1.08. |
| **score_B cross-routed to baseline** | **INVARIANT 19** | **⛔ ARCHITECTURE VIOLATION — revert immediately** |
| **score_A cross-routed to CUSUM** | **INVARIANT 19** | **⛔ ARCHITECTURE VIOLATION — revert immediately** |
| All 15 gates pass | — | Proceed to Part 2B for outputs + paste text, then to M9. |

---

## Cross-Module Invariants Relevant to M8

| # | Invariant |
|---|-----------|
| 1 | Models saved: `torch.save(state_dict)` \| Loaded: `map_location='cpu'` for M10 |
| 2 | Normalization baselines LOCKED at `M3_normalization_config.json` |
| 3 | M4 threshold 0.110058 = Level 1 starting reference ONLY — never applied to TCN-AE output |
| 4 | Channel weights INCREASED vs M4 — Fisher-validated from M6.5 |
| 5 | Faults NEVER in training pool — LSTM-AE is anomaly detector only |
| 6 | Mech C operates on RAW channel errors — bypasses weight matrix by design |
| 7 | Level 1 threshold calibrated on REAL CIRA validation set — not synthetic |
| 8 | All M6B Groups (A–E, 22 classes) in fault validation pool only — never in training |
| 9 | Cavitation gate: startup cluster only — any cavitation DANGER outside startup = FAIL |
| 10 | `if pump_type == 'household': return physics_advisory_only()` — NO EXCEPTIONS |
| 11 | Label strings always resolved via `fault_rules_v3.json` — NEVER hardcoded |
| 12 | Group C masked fault max alert = WARN unless secondary MAE crosses threshold |
| 13 | Label 17 max alert = WARN (weakest secondary path — Pmp.PV only) |
| 14 | Group B compound: 2nd Mech C flag fires at `secondary_onset_lag` — not before |
| 15 | M8 outputs raw `alert_state` dict — M10 handles all UI display formatting |
| 16 | Layer 3 CUSUM + Layer 4 Rolling Baseline = label 21 ONLY — do NOT apply broadly |
| 17 | Layer 4 disabled during burn-in (≤5,000 windows) — CUSUM only during burn-in |
| 18 | Label 21 sub-threshold MAE = design behaviour — DO NOT raise threshold to fix |
| **19** | **`score_B` → CUSUM only \| `score_A` → Rolling Baseline only \| `score_C` → XGBoost only — NEVER cross-route** |
| 20 | Level 2 TCN-AE input = z_t sequences only — raw sensor data NEVER enters Level 2 |
| 21 | `physics_context` dict required in every M10 alert output (Invariant 18) |
| 22 | Real-world conditions disclaimer mandatory in every M10 alert (Part 1A) |
| 23 | Score routing rules enforced in Step 3 and Step 6B of inference protocol |

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic — 13 gates. |
| v2.0 | 2026-04-15 | Split Part 1+2. 13→14 gates. Group B/D/E exceptions. Output dict extended. |
| v3.0 | 2026-04-16 | v14.0: Part 2A + Part 2B split. 15 gates. CUSUM + Rolling Baseline flags in state machine. Stage 4 NORMAL/WATCH/WARN/DANGER conditions extended. Label 21 block added. Group C updated to 5 classes labels 13–17. Group B updated to 6 classes labels 7–12. Label 12 specific note. Inference protocol Step 6B added. Output dict extended (cusum, drift_ratio). Adaptive actions extended. Invariants 14→18. |
| v4.0 | 2026-04-19 | v14.2 TCN-AE SCORE ROUTING: score_C added to state machine (all 4 state conditions). score_C threshold calibration protocol added (Stage 5). score_A and score_B added to output dict with routing labels. Step 3 of inference protocol: TCN-AE forward pass + score routing. Step 6B updated: CUSUM on score_B (not raw MAE), Rolling on score_A. Invariant 19 added (score routing never cross) and Invariants 20–23. Gate M8-15 added (score_C calibration — new Group B compound gate). Label 10 DANGER action: score_C at NPSHa/NPSHr crossing. Label 12 DANGER action: score_C BPF-to-shock spike. Label 17 Group C corrected: `impeller_PmpSV_flatline`, secondary = Pmp.PV. `physics_context` and `disclaimer` added to output dict. Cavitation intermittent: score_C note added. WATCH→NORMAL condition: `score_C < score_C_normal_p95` added. Real-world conditions note linked from Part 1A. |

---

> **GitHub is the ONLY source of truth for this spec.**
>
> - Part 1A: `module_M8_lstm_ae_v2_architecture.md`
> - Part 1B: `module_M8_lstm_ae_v2_mechanisms.md`
> - Part 2B: `module_M8_lstm_ae_v2_outputs_and_paste.md`
>
> **Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset
> **Standard:** ISO 10816-3 vibration, ISO 13373-3 condition monitoring
> **Canonical source of truth:** `pasted-text.txt` (v14.2, 2026-04-19)
