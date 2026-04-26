# PumpSmart — M12: Physics-Governed Adversarial Validation Suite

| Field | Value |
|-------|-------|
| **Document version** | v3.0 — Architecture v14.2 (TCN-AE Level 2 + score_A/B/C + CUSUM score_B + adaptive threshold + physics_context gates) |
| **Date** | 2026-04-21 |
| **Status** | NOT STARTED — begins ONLY after M11 deployment confirmed |
| **Pump** | 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP |

---

## Purpose and Relationship to M6

- M6 trained M7 and M8 on synthetic sequences.
- M8 validated against M6 synthetic sequences (model has **SEEN** these).
- M12 generates **COMPLETELY FRESH** sequences using M5 physics engine — parametrically different from M6 training data.
- Model has **NEVER** been exposed to M12 sequences.

**M12 = adversarial stress-test on a never-seen distribution.**

- If M8 passes M12 → `PRODUCTION_VALIDATED`
- If M8 fails M12 → `RECALIBRATION_REQUIRED` → fix M6/M8 → re-run M12

| Dimension | M6 (Training) | M12 (Adversarial) |
|-----------|---------------|-------------------|
| Purpose | Train M7 + M8 | Stress-test on deployment-like conditions |
| Model exposure | Yes — seen in M8 validation | Never — completely held out |
| Generation | Physics engine + spike seeds | M5 physics engine ONLY (no spike seeds) |
| Trigger | One-time during training | On-demand via `/api/validate_model` (M10) |
| Primary metric | Label distribution balance | Detection latency (timesteps) |
| Slow drift configs | Present but seen by M8 | Present and UNSEEN — true adversarial |
| Compound configs | M6B training data (seen) | Fresh from M5 only (unseen) |

---

## What Changed in v3.0 (Architecture v14.2)

| Item | v2.0 (arch v14.0) | v3.0 (arch v14.2) |
|------|-------------------|-------------------|
| Model under test | M8 LSTM-AE v2 + M7 XGBoost | M4 LSTM-AE Level 1 + TCN-AE Level 2 (M8) + M7 XGBoost 22-class |
| M8 gate reference | `all_13_gates_pass` | `all_gates_pass` (count per File 3 final value) |
| Config 3 gates | 6 gates | 8 gates (+CUSUM score_B rising, +zt_buffer fills) |
| Config 14 gates | 6 gates | 8 gates (+CUSUM score_B rising, +adaptive_threshold warmup not complete at WATCH — expected) |
| Slow drift XGBoost gates (C3, C6, C8, C13, C14) | score_C not referenced | score_C = None expected when zt_buffer < 6 — not a failure condition |
| Config 17 gates | 6 gates | 7 gates (+score_C non-null by step 200, top-3 SHAP) |
| Config 18 gates | 6 gates | 7 gates (+score_C non-null by step 300, top-3 SHAP) |
| DANGER-state gates (all acute configs) | XGBoost Stage 3 fields only | + physics_context field non-null confirmed in M10 response |
| WATCH/WARN/DANGER gates | No limitation_flags check | + limitation_flags non-empty confirmed per severity tier |
| Safety gate certificate | M8 LSTM-AE v2 only reference | M4 LSTM-AE L1 + TCN-AE L2 + M7 XGBoost 22-class + M9 Physics Tools + TCN-AE model hash added |
| Paste keys | 18 | 25 (+7 new keys) |

---

## Prerequisite

- M11 deployment confirmed (HF Space `/health` = healthy, version = 4.0)
- **AND** M8 `all_gates_pass` = True (full 4-layer suite: Level 1 + Level 2 + Layer 3 + Layer 4)
- **AND** M12 sequences generated fresh using M5 physics engine
  - NOT reusing any sequence from `data/synthetic/M6_sequences.pkl`
  - NOT reusing any sequence from `data/synthetic/M6B_compound_sequences.pkl`

### Model Stack Under Test in M12

| Layer | Component | Detail |
|-------|-----------|--------|
| Level 1 | M4 LSTM-AE | `lstm_ae_baseline_best.pth` — static threshold 0.110058 |
| Level 2 | TCN-AE | `tcn_ae_level2_best.pth` — score_A, score_B, score_C |
| Layer 3 | CUSUM on score_B | control limit 5.0 |
| Layer 4 | Adaptive threshold on score_A | θ_t = μ(6hr) + 3σ(6hr) |
| Classifier | M7 XGBoost 22-class | ~35 features including score_A/B/C |

All models loaded `map_location='cpu'` via M10 Flask app `/api/validate_model` route.

---

## Primary Metric: Detection Latency

### Physics Basis for Latency Budget

110 kW, 2980 RPM, 40 bar multistage pump:

- Bearing wear to catastrophic failure: ~600–1800 seconds
- Cavitation impeller pitting onset: 60–180 seconds
- Maintenance crew response time minimum: ~300 seconds

> **Detection budget:** 300 timesteps (M8 must detect within 30% of failure window)
> **Standard gate:** detection_lag ≤ 60 timesteps (2× safety margin)
> **Cavitation gate:** detection_lag ≤ 30 timesteps (impeller pitting risk)

### Slow Drift Latency Gates (Liability-Critical)

| Fault | Scale | WATCH must fire |
|-------|-------|----------------|
| Seal failure sev 0.2 | weeks-scale | ≤ 20 min (1200 steps) |
| Bearing wear sev 0.25 | days-scale | ≤ 15 min (900 steps) |
| Overloading sev 0.2 | days-scale | ≤ 15 min (900 steps) |

These gates validate the full 4-layer detection stack on **NEVER-SEEN** slow sequences.

### Detection Lag Definition

```
detection_lag = timestep(first target alert_state) - timestep(fault onset t=0)
  For slow drift  : target alert_state = WATCH
  For acute fault : target alert_state = DANGER
```

### XGBoost Output at Each State (M7 22-class schema)

**WATCH state** — Stage 1 or Stage 2 output expected
- Stage 1: top-3 candidates listed (primary_conf < 0.50)
- Stage 2: probable fault + secondary_faults dict
- NOTE: score_C = None is acceptable at WATCH for slow drift configs when `zt_buffer_len < ZT_BUFFER_MIN` (6). Not a failure. XGBoost operates on reduced feature set without score_C.

**WARN state** — Stage 2 output expected (sustained anomaly, conf rising)
- Probable primary fault + secondary fault candidates shown
- score_C expected non-None by WARN (buffer fills ~6 windows in)

**DANGER state** — Stage 3 output expected (confirmed classification)
- Primary fault confirmed + secondary_faults for compound cases
- physics_context field non-null in M10 response (v4.0 check)
- limitation_flags non-empty in M10 response (v4.0 check)

---

## All 18 Test Configurations

---

### Config 1 — Bearing Wear (Standard, sev 0.6)

| Parameter | Value |
|-----------|-------|
| Fault type | bearing_wear |
| Cluster | steady_state |
| Severity | 0.6 |
| Steps | 500 |
| Generation | M5 physics engine (smooth — NO spike seed) |

**Physics Invariants**
- Mot.SV\* rises monotonically 1.0 → ~1.6 over 500 steps
- Mot.TV\* rises with lag 20–40 steps behind Mot.SV\* (thermal lag)
- Thermal coupling r(Mot.TV, Temp.SV) > 0.85
- All other channels within 1.0 ± 0.1

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C1-1 | detection_lag ≤ 60 timesteps (DANGER) |
| GATE-C1-2 | thermal_lag confirmed — peak Mot.SV error BEFORE peak Mot.TV error |
| GATE-C1-3 | XGBoost Stage 3 output: bearing_wear as primary (confidence > 0.70) |
| GATE-C1-4 | physics_context field non-null in M10 Route 1 response at DANGER |
| GATE-C1-5 | limitation_flags non-empty in M10 response at DANGER |

---

### Config 2 — Bearing Wear (High Noise, sev 0.6)

| Parameter | Value |
|-----------|-------|
| Fault type | bearing_wear |
| Cluster | high_load |
| Severity | 0.6 |
| Steps | 500 |
| Noise level | 3× normal (sigma multiplier = 3.0 on all channels) |

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C2-1 | detection_lag ≤ 90 timesteps (DANGER) — relaxed for high noise |
| GATE-C2-2 | alert_state reaches at least WARN within 120 timesteps |
| GATE-C2-3 | MC Dropout uncertainty_std elevated (expected > 0.015 under noise) |
| GATE-C2-4 | XGBoost Stage 2 or Stage 3: bearing_wear as primary or top candidate |
| GATE-C2-5 | physics_context field non-null in M10 response at DANGER |
| GATE-C2-6 | limitation_flags includes L6_confidence_proxy at WARN/DANGER |

---

### Config 3 — Bearing Wear (Subtle Early, sev 0.25)

| Parameter | Value |
|-----------|-------|
| Fault type | bearing_wear |
| Cluster | steady_state |
| Severity | 0.25 |
| Steps | 1500 |
| Generation | M5 physics engine ONLY (NO spike seed, NO seam artifact) |

**Physics Invariants**
- Mot.SV\* rises very slowly at ~0.0003/step
- Total MAE below 0.110058 for first 800+ steps
- Thermal coupling preserved throughout

**Primary detection path:** CUSUM S_n(score_B) accumulating + Mech C Mot.SV Spearman

> This is an **UNSEEN** slow drift sequence — true adversarial test. Same physics as `bearing_wear_gradual` in training. CUSUM + rolling baseline (score_A) are the primary path. Level 1 MAE staying below threshold is EXPECTED.

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C3-1 | WATCH state fires ≤ 15 min (900 steps) via Mot.SV drift flag |
| GATE-C3-2 | total MAE remains BELOW threshold for first 300 steps |
| GATE-C3-3 | thermal_lag validated — Mot.SV Spearman fires BEFORE Mot.TV Spearman |
| GATE-C3-4 | XGBoost Stage 1 or 2 at WATCH (low confidence expected for sev 0.25). score_C = None acceptable if zt_buffer_len < 6 at WATCH trigger |
| GATE-C3-5 *(NEW v3.0)* | CUSUM S_n(score_B) rises above 1.0 within 900 steps. Confirms TCN-AE Level 2 drift detection path is active. S_n must be strictly increasing — not flat or oscillating. |
| GATE-C3-6 *(NEW v3.0)* | zt_buffer_len reaches ZT_BUFFER_MIN (6) before GATE-C3-5 checked. TCN-AE score_B is only trusted after buffer fills. Log zt_buffer_len at the timestep WATCH fires. |

---

### Config 4 — Cavitation (Standard, sev 0.7)

| Parameter | Value |
|-----------|-------|
| Fault type | cavitation |
| Cluster | startup ONLY |
| Severity | 0.7 |
| Steps | 300 |

**Physics Invariants**
- Pres.SV\* erratic (chaotic, high kurtosis)
- Pmp.SV\* spikes sharply (hydraulic shock)
- thermal_decoupling_flag = True (r ~ 0.3–0.4)
- Mot.TV\*, Temp.SV\*: near 1.0 (NOT elevated)
- Cluster: startup ONLY

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C4-1 | detection_lag ≤ 30 timesteps (DANGER — cavitation is acute) |
| GATE-C4-2 | alert_state = DANGER (bypasses WATCH/WARN — cavitation exception) |
| GATE-C4-3 | XGBoost Stage 3 output: cavitation as primary (confidence > 0.85) |
| GATE-C4-4 | thermal_decoupling_flag = True in M8 output |
| GATE-C4-5 | physics_context field non-null in M10 response at DANGER; if_ignored field references NPSHa/NPSHr crossing and 60–180s timeline |
| GATE-C4-6 | limitation_flags non-empty in M10 response at DANGER |

---

### Config 5 — Cavitation (High Noise, sev 0.7)

| Parameter | Value |
|-----------|-------|
| Fault type | cavitation |
| Cluster | startup |
| Severity | 0.7 |
| Noise level | 3× normal |

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C5-1 | detection_lag ≤ 50 timesteps (DANGER) — relaxed for noise |
| GATE-C5-2 | alert_state = DANGER |
| GATE-C5-3 | uncertainty_std elevated (MC Dropout spread under noise) |
| GATE-C5-4 | XGBoost Stage 2 or 3: cavitation as primary or top candidate |
| GATE-C5-5 | physics_context field non-null in M10 response at DANGER |
| GATE-C5-6 | limitation_flags includes L6_confidence_proxy at DANGER |

---

### Config 6 — Seal Failure (Slow Leak, sev 0.3)

| Parameter | Value |
|-----------|-------|
| Fault type | seal_failure |
| Cluster | steady_state |
| Severity | 0.3 |
| Steps | 1000 |

**Physics Invariants**
- Pres.SV\* declines monotonically at ~0.0004/step
- thermal_decoupling_flag = True (r < 0.1)
- Mot.SV\*, Pmp.SV\*: near 1.0
- Total MAE below threshold for many early windows

**Primary detection path:** Mech C Pres.SV drift (NEGATIVE Spearman) + thermal_decoupling

> NOTE: score_C = None is acceptable at WATCH for this config. zt_buffer may not yet be full when Spearman-based WATCH fires. XGBoost operates on reduced feature set at early WATCH — not a failure.

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C6-1 | WATCH state fires ≤ 20 min (1200 steps) via Pres.SV drift |
| GATE-C6-2 | thermal_decoupling_flag = True in M8 output |
| GATE-C6-3 | Pres.SV drift flag fires BEFORE total MAE reaches WARN level |
| GATE-C6-4 | XGBoost Stage 2 output at WARN: seal_failure as probable primary; score_C present in feature set if zt_buffer_len ≥ 6 |
| GATE-C6-5 | physics_context field non-null in M10 response at WARN |

---

### Config 7 — Seal Failure (Fast Degradation, sev 0.8)

| Parameter | Value |
|-----------|-------|
| Fault type | seal_failure |
| Cluster | high_load |
| Severity | 0.8 |
| Steps | 400 |

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C7-1 | detection_lag ≤ 60 timesteps (DANGER) |
| GATE-C7-2 | alert_state reaches DANGER |
| GATE-C7-3 | XGBoost Stage 3 output: seal_failure as primary (confidence > 0.75) |
| GATE-C7-4 | physics_context field non-null in M10 response at DANGER |
| GATE-C7-5 | limitation_flags non-empty in M10 response at DANGER |

---

### Config 8 — Overloading (Standard, sev 0.6)

| Parameter | Value |
|-----------|-------|
| Fault type | overloading |
| Cluster | steady_state ONLY |
| Severity | 0.6 |
| Steps | 800 |

**Physics Invariants**
- Temp.SV\* rises monotonically (thermal overload)
- Mot.TV\* rises with slight lag behind Temp.SV\*
- Pres.SV\*, Pmp.SV\*, Mot.SV\*: near 1.0 (NOT vibration)
- Thermal coupling STRONGLY PRESERVED (r = 0.997)

**Primary detection path:** Mech C Temp.SV drift (POSITIVE Spearman)

> Single-window MAE will not cross threshold reliably (M6.5 Finding 1). score_C = None is acceptable at early WATCH for this config. WATCH fires via Spearman before z_t buffer fills in many cases.

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C8-1 | WATCH state fires ≤ 15 min (900 steps) via Temp.SV drift flag |
| GATE-C8-2 | XGBoost Stage 2 output: overloading as probable primary at WATCH/WARN; score_C acceptable as None at first WATCH trigger |
| GATE-C8-3 | Vibration channels Mot.SV\*, Pmp.SV\* remain below 1.2 |
| GATE-C8-4 | Thermal coupling r(Mot.TV, Temp.SV) > 0.90 in M12 sequence |
| GATE-C8-5 | physics_context field non-null in M10 response at WARN |

---

### Config 9 — Impeller Imbalance (Standard, sev 0.7)

| Parameter | Value |
|-----------|-------|
| Fault type | impeller_imbalance |
| Cluster | high_load |
| Severity | 0.7 |
| Steps | 400 |

**Physics Invariants**
- Pmp.PV\* and Pmp.SV\* rise immediately (immediate onset)
- Pres.SV\* oscillates (pressure ripple from imbalance)
- Mot.PV\* rises (vibration propagation)
- err_onset_lag SMALL (fast onset — distinguishes from bearing_wear)

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C9-1 | detection_lag ≤ 60 timesteps (DANGER) |
| GATE-C9-2 | XGBoost Stage 3 output: impeller_imbalance as primary (conf > 0.70) |
| GATE-C9-3 | err_onset_lag < 10 timesteps (immediate onset confirmed) |
| GATE-C9-4 | physics_context field non-null in M10 response at DANGER |
| GATE-C9-5 | limitation_flags non-empty in M10 response at DANGER |

---

### Config 10 — Sensor Failure (Sanity Check)

| Parameter | Value |
|-----------|-------|
| Fault type | sensor_failure |
| Cluster | any |
| Severity | 0.8 |
| Steps | 300 |

**Physics Invariants**
- One channel (e.g. Pres.SV) goes flatline at t=50
- All other 7 channels remain normal
- channel_error std collapses to near-zero for flatline channel

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C10-1 | sensor_failure flag fires within 100 timesteps of flatline onset |
| GATE-C10-2 | channel_drift[flatline_channel] = True, all others False |
| GATE-C10-3 | XGBoost Stage 3 output: sensor_failure as primary (conf > 0.90) |
| GATE-C10-4 | M8 output identifies WHICH specific channel failed |
| GATE-C10-5 | physics_context field non-null in M10 response at DANGER |
| GATE-C10-6 | limitation_flags includes L1_single_installation at DANGER |

---

### Config 11 — Compound: Bearing Wear + Cavitation Simultaneously

| Parameter | Value |
|-----------|-------|
| Fault types | bearing_wear (sev 0.5) + cavitation (sev 0.7) simultaneously |
| Cluster | startup (cavitation requires startup) |
| Steps | 300 |

**Physics Invariants**
- Pmp.SV\* elevated (BOTH bearing AND cavitation)
- Mot.SV\* elevated (bearing component)
- Pres.SV\* erratic (cavitation component)
- Thermal coupling: partial

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C11-1 | DANGER fires within 30 timesteps (cavitation dominates speed) |
| GATE-C11-2 | alert_state = DANGER |
| GATE-C11-3 | XGBoost Stage 3 output: primary_fault = cavitation OR bearing_wear (either acceptable); secondary_faults dict MUST contain the other fault with prob > 0.30 |
| GATE-C11-4 | channel_drift: BOTH Pmp.SV AND Mot.SV elevated |
| GATE-C11-5 | physics_context field non-null in M10 response at DANGER |
| GATE-C11-6 | limitation_flags includes L4_synthetic_domain at DANGER |

---

### Config 12 — Bearing Wear with Thermal Coupling Broken

| Parameter | Value |
|-----------|-------|
| Fault type | bearing_wear |
| Cluster | steady_state |
| Severity | 0.7 |
| Steps | 500 |
| Forced | r(Mot.TV, Temp.SV) = 0.15 (thermal decoupling forced) |

**Physics Context**

Tests what happens when ambient conditions break the thermal path. M8 must still detect via vibration channels (Mot.SV\*, Pmp.SV\*) alone. Tests whether M8 over-relies on thermal coupling for bearing detection.

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C12-1 | detection_lag ≤ 90 timesteps (relaxed — harder config) |
| GATE-C12-2 | DANGER or WARN state reached via vibration channels alone |
| GATE-C12-3 | Mot.SV drift flag fires (Mech C vibration path works independently) |
| GATE-C12-4 | XGBoost Stage 2 or 3: bearing_wear as primary (not seal_failure). Pres.SV must remain near 1.0 to prevent seal false positive |
| GATE-C12-5 | If lag > 90 → document thermal over-reliance → reduce Mot.TV weight in M8 and re-run |
| GATE-C12-6 | physics_context field non-null in M10 response at WARN/DANGER |

---

### Config 13 — Seal Failure Slow Drift (Weeks-Scale, sev 0.2)

> ⚠️ **PRIMARY LIABILITY GATE — non-negotiable**

| Parameter | Value |
|-----------|-------|
| Fault type | seal_failure |
| Cluster | steady_state |
| Severity | 0.2 |
| Steps | 2000+ |
| Generation | M5 physics engine ONLY (pure smooth physics — NO spike seed) |

**Physics Invariants**
- Pres.SV\* declines at ~0.0004/step (weeks-scale degradation)
- Total MAE BELOW threshold for first 500+ steps
- Thermal coupling BROKEN throughout (r < 0.1)
- All other channels flat near 1.0

This config simulates a seal leaking for weeks before detection. Single-window MAE will never cross threshold in first 500 steps. Only Spearman drift over 300+ windows makes it detectable. **This is the PRIMARY liability test for the 110 kW asset.**

**Primary detection path:** Mech C Pres.SV drift (NEGATIVE Spearman)
**Secondary confirm:** Mech A rolling mean + thermal_decoupling_flag

> NOTE: score_C = None is acceptable at WATCH trigger. CUSUM S_n(score_B) may be low at WATCH trigger for sev 0.2 — this is expected. Spearman path is the primary gate here, not CUSUM.

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C13-1 | **PRIMARY LIABILITY GATE** — WATCH state fires ≤ 20 min (1200 steps) via Pres.SV drift. Must pass before deployment. |
| GATE-C13-2 | total MAE remains BELOW threshold for first 500 steps — validates genuine slow drift, not trivially detectable |
| GATE-C13-3 | thermal_decoupling_flag = True in M8 output from onset |
| GATE-C13-4 | Pres.SV drift flag fires BEFORE rolling mean WATCH trigger |
| GATE-C13-5 | No false DANGER state fired in first 500 steps |
| GATE-C13-6 | XGBoost Stage 1 output at early WATCH (low conf expected for sev 0.2); Stage 2 expected at WARN (probable seal_failure); score_C = None acceptable at WATCH — not a failure |
| GATE-C13-7 | physics_context field non-null in M10 response at WARN; `why` field references seal leakage hydraulics |

---

### Config 14 — Bearing Wear Slow Drift (Days-Scale, sev 0.25)

> ⚠️ **LIABILITY GATE — non-negotiable**

| Parameter | Value |
|-----------|-------|
| Fault type | bearing_wear |
| Cluster | steady_state |
| Severity | 0.25 |
| Steps | 1500+ |
| Generation | M5 physics engine ONLY (smooth, NO spike seed, NO seam) |

**Physics Invariants**
- Mot.SV\* rises gradually at ~0.0003/step
- Thermal coupling PRESERVED (r > 0.85)
- Mot.TV\* rises with 20–40 step lag
- Total MAE below threshold for first 400+ steps

**Primary detection path:** CUSUM S_n(score_B) accumulating + Mech C Mot.SV Spearman

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C14-1 | **LIABILITY GATE** — WATCH state fires ≤ 15 min (900 steps) via Mot.SV drift flag. Must pass before deployment. |
| GATE-C14-2 | total MAE remains BELOW threshold for first 300 steps |
| GATE-C14-3 | thermal coupling r > 0.85 confirmed |
| GATE-C14-4 | Mot.SV Spearman fires BEFORE Mot.TV Spearman (thermal lag validated) |
| GATE-C14-5 | Attention heatmap does NOT cluster at any specific timestep — smooth drift, no seam artifact |
| GATE-C14-6 | XGBoost Stage 1 or 2 at WATCH (low conf expected for sev 0.25); score_C = None acceptable if buffer < 6 at WATCH trigger |
| GATE-C14-7 *(NEW v3.0)* | CUSUM S_n(score_B) rising above 1.0 within 900 steps. Strictly increasing — confirms TCN-AE Level 2 drift path active. If CUSUM flat while Spearman fires → investigate score_B routing. |
| GATE-C14-8 *(NEW v3.0)* | adaptive_threshold_active = False at WATCH trigger expected. 6-hour warmup not complete at this stage. WATCH fires via CUSUM path (Layer 3) + Spearman (Mech C). Level 1 static threshold (0.110058) governs Level 1 gate. Adaptive threshold takes over at Layer 4 only after 216-call warmup. |

---

### Config 15 — Cluster Transition Fault (Startup to Steady-State Boundary)

| Parameter | Value |
|-----------|-------|
| Fault type | bearing_wear |
| Cluster | startup → transitions to steady_state at step 50 |
| Severity | 0.6 |
| Steps | 400 |
| Fault onset | t=40 (10 steps before cluster transition at t=50) |

**Physics Context**

Most dangerous operational moment: startup → ss transition. Fault injected at t=40 overlaps with natural MAE elevation at transition. Tests whether cluster-conditional thresholds correctly separate fault-elevated MAE from transition-elevated MAE.

**Physics Invariants**
- t=0–49: startup cluster — MAE elevated (normal startup behaviour)
- t=40–49: bearing fault also begins (10 steps overlap)
- t=50+: steady_state cluster — threshold changes
- Mot.SV\*: rises from t=40 regardless of cluster change

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C15-1 | fault detected within 60 timesteps of cluster transition (t=110) |
| GATE-C15-2 | M8 applies startup threshold for t<50, ss threshold for t≥50 |
| GATE-C15-3 | DANGER or WARN fires in steady_state cluster region |
| GATE-C15-4 | No missed detection by attributing fault to startup noise |
| GATE-C15-5 | XGBoost Stage 2 or 3: bearing_wear as primary in ss cluster region |
| GATE-C15-6 | physics_context field non-null in M10 response at WARN/DANGER |

---

### Config 16 — Compound: Sensor Drift + Bearing Fault Simultaneously

| Parameter | Value |
|-----------|-------|
| Fault types | sensor_drift (Pres.SV calibration drift, UPWARD) + bearing_wear (sev 0.3) simultaneously |
| Cluster | steady_state |
| Steps | 800 |

**Physics Context**

Sensor calibration drift accumulates over months. Bearing fault developing on top of a drifting sensor is the most dangerous field scenario — risk of misclassification is highest here.

> Key distinction:
> - `seal_failure` = NEGATIVE Pres.SV\* drift (pressure loss from leakage)
> - `sensor_drift` = POSITIVE Pres.SV\* drift (calibration offset bias)
>
> M8 must disambiguate via sign direction + cross-channel analysis.

**Physics Invariants**
- Pres.SV\*: systematic UPWARD bias (NOT seal failure — wrong direction)
- Mot.SV\*: gradual rise (bearing degradation)
- Thermal coupling PRESERVED (r > 0.85) — confirms bearing, not seal
- Mot.TV\*: rises with 20–40 step lag

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C16-1 | Mot.SV drift flag fires within 600 steps (bearing_wear_early) |
| GATE-C16-2 | Pres.SV drift fires AND tagged as sensor_failure; UPWARD direction ≠ seal failure (always DOWNWARD) |
| GATE-C16-3 | thermal_coupling r > 0.80 (confirms bearing not seal) |
| GATE-C16-4 | M8 output contains BOTH early_fault_type flags simultaneously |
| GATE-C16-5 | XGBoost Stage 3 output: primary_fault = bearing_wear; secondary_faults dict contains sensor_failure with prob > 0.30 |
| GATE-C16-6 | physics_context field non-null in M10 response at DANGER |
| GATE-C16-7 | limitation_flags includes L4_synthetic_domain at DANGER |

---

### Config 17 — Compound: Bearing Wear + Seal Failure (Pure Physics)

> *NEW v2.0 — M6B Compound Pair Adversarial Test*

| Parameter | Value |
|-----------|-------|
| Fault types | bearing_wear (sev 0.5) + seal_failure (sev 0.4) simultaneously |
| Cluster | steady_state |
| Steps | 600 |
| Generation | M5 physics engine ONLY — NO M6B sequences reused |
| Causal lag | seal_failure onset at t=80 after bearing_wear (causal cascade) |

**Physics Basis** *(same as M6B causal pair 1)*

Bearing wear degrades shaft alignment → seal compression uneven → progressive seal failure begins ~80 steps after bearing onset.

- Mot.SV\* rises from t=0 (bearing component)
- Pres.SV\* declines monotonically from t=80 (seal component)
- Mot.TV\* rises with 20–40 step lag (bearing thermal)
- Thermal coupling PARTIALLY BROKEN by t=150 (seal decoupling begins)
- compound_interaction_flag expected HIGH in M6.5 features

> This is an **UNSEEN** compound sequence. M6B training data has bearing+seal at different severity and lag values. Tests M7 multi-label architecture on genuinely novel compound pattern.

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C17-1 | M8 WATCH fires within 60 steps (Mot.SV drift begins immediately) |
| GATE-C17-2 | Second WATCH channel (Pres.SV drift) fires by t=380 (80+300 window) |
| GATE-C17-3 | DANGER fires within 200 steps total |
| GATE-C17-4 | XGBoost Stage 3 output: primary_fault = bearing_wear (onset first); secondary_faults dict: seal_failure with prob > 0.30 |
| GATE-C17-5 | compound_interaction_flag feature value > 0.5 in extracted features; Spearman lag shift between Mot.SV and Pres.SV errors detectable |
| GATE-C17-6 | thermal_coupling r > 0.70 at t=0–100 (bearing phase); thermal_coupling r < 0.30 at t=200+ (seal phase dominating) |
| GATE-C17-7 *(NEW v3.0)* | score_C non-None by step 200 (zt_buffer fills within ~6 windows). score_C must rank in top-3 SHAP features for Stage 3 output. Confirms TCN-AE chain transition signal captured for compound event. If score_C missing from top-3: investigate TCN-AE Level 2 training. |

---

### Config 18 — Compound: Overloading + Seal Failure (Thermal + Hydraulic Cascade)

> *NEW v2.0 — M6B Compound Pair Adversarial Test*

| Parameter | Value |
|-----------|-------|
| Fault types | overloading (sev 0.4) + seal_failure (sev 0.3) simultaneously |
| Cluster | high_load (overloading occurs at high load only) |
| Steps | 800 |
| Generation | M5 physics engine ONLY |
| Causal lag | seal_failure onset at t=100 after overloading (thermal → seal) |

**Physics Basis** *(same as M6B causal pair 3)*

Motor overloading → thermal expansion of shaft → seal compression changes → seal begins to leak at elevated temperature.

- Temp.SV\* rises from t=0 (overloading thermal)
- Mot.TV\* rises in tandem (thermal coupling preserved at first)
- Pres.SV\* begins declining from t=100 (seal starts leaking)
- thermal_decoupling begins at t=100+ as seal failure introduces hydraulic signal

> This tests the most operationally dangerous cascade: model trained on overloading ALONE (single-channel thermal) must recognise that an emerging pressure decline at t=100 is a SECOND fault starting, not noise.

**Gates**

| Gate | Description |
|------|-------------|
| GATE-C18-1 | Temp.SV drift WATCH fires within 900 steps (overloading detection) |
| GATE-C18-2 | Pres.SV drift WATCH fires by t=1100 (t=100 onset + 300 window + margin) |
| GATE-C18-3 | M8 alert_state reaches WARN or DANGER within 400 steps |
| GATE-C18-4 | XGBoost Stage 3 output: primary_fault = overloading (onset first); secondary_faults dict: seal_failure with prob > 0.30 |
| GATE-C18-5 | thermal_decoupling_flag transitions from False to True at t ~100; coupling breaks when seal failure starts — validates M5 simulation |
| GATE-C18-6 | Two separate early_fault_type flags in M8 output: overloading_early fires first, seal_failure_early fires second |
| GATE-C18-7 *(NEW v3.0)* | score_C non-None by step 300 (zt_buffer fills within ~6 windows). score_C must rank in top-3 SHAP features for Stage 3 output. Confirms chain transition signal detected across thermal→hydraulic cascade. If score_C missing: thermal→hydraulic onset_lag too gradual for TCN-AE. Recalibration action: lower score_C sensitivity threshold in TCN-AE or increase severity of M6B cascade training examples. |

---

## Detection Coverage Matrix (All 18 Configs)

| Config | Fault Type | Severity | Steps | Primary Detection Path | Key Gate | New Gates (v3.0) |
|--------|-----------|---------|-------|------------------------|----------|-----------------|
| C1 | bearing_wear | 0.6 | 500 | Single-window DANGER, Mech C Mot.SV | lag ≤ 60 | +physics_context, +limit_flags |
| C2 | bearing_wear (hi noise) | 0.6 | 500 | Single-window DANGER, Mech C + MC Dropout | lag ≤ 90 | +physics_context, +L6 flag |
| C3 | bearing_wear (subtle) | 0.25 | 1500 | CUSUM score_B + Mech C Mot.SV | WATCH ≤ 15min | +CUSUM S_n gate, +zt_buffer gate |
| C4 | cavitation | 0.7 | 300 | Direct DANGER bypass, Pres.SV erratic | lag ≤ 30 | +physics_context if_ignored |
| C5 | cavitation (hi noise) | 0.7 | 300 | Direct DANGER, Pres.SV + MC Dropout | lag ≤ 50 | +physics_context, +L6 flag |
| C6 | seal_failure (slow) | 0.3 | 1000 | Mech C Pres.SV drift, Spearman primary | WATCH ≤ 20min | +physics_context at WARN |
| C7 | seal_failure (fast) | 0.8 | 400 | Single-window DANGER, Pres.SV + thermal | lag ≤ 60 | +physics_context, +limit_flags |
| C8 | overloading | 0.6 | 800 | Mech C Temp.SV drift, Spearman primary | WATCH ≤ 15min | +physics_context at WARN |
| C9 | impeller_imbalance | 0.7 | 400 | Single-window DANGER, Pmp.SV immediate | lag ≤ 60 | +physics_context, +limit_flags |
| C10 | sensor_failure | 0.8 | 300 | Flatline std flag, channel_drift flag | flag ≤ 100 steps | +physics_context, +L1 flag |
| C11 | bearing + cavitation | 0.5+0.7 | 300 | Cavitation dominates, dual channel drift | DANGER ≤ 30 + secondary_faults | +physics_context, +L4 flag |
| C12 | bearing (thermal broken) | 0.7 | 500 | Mech C Mot.SV only, vibration path only | WARN ≤ 90 | +physics_context at WARN |
| C13 | seal slow drift | 0.2 | 2000 | Mech C Pres.SV drift, Spearman (LIABILITY) | **WATCH ≤ 20min — PRIMARY LIABILITY** | +physics_context why field |
| C14 | bearing slow drift | 0.25 | 1500 | CUSUM score_B + Mech C Mot.SV (LIABILITY) | **WATCH ≤ 15min — LIABILITY** | +CUSUM S_n gate, +adaptive_threshold warmup gate |
| C15 | bearing (transition) | 0.6 | 400 | Cluster-conditional, startup→ss boundary | detect ≤ t=110 | +physics_context at WARN |
| C16 | bearing + sensor drift | 0.3+drift | 800 | Dual Mech C flags, sign disambiguation | both flags + secondary_faults | +physics_context, +L4 flag |
| C17 | bearing + seal | 0.5+0.4 | 600 | Dual Mech C + Stage 3, causal cascade | secondary_faults ≥ 0.30 | +score_C top-3 SHAP gate |
| C18 | overloading + seal | 0.4+0.3 | 800 | Dual Mech C + Stage 3, thermal→hydraulic cascade | thermal→hydraulic cascade gate | +score_C top-3 SHAP gate |

---

## Recalibration Loop

If any gate FAILS → `status: RECALIBRATION_REQUIRED`

### Per-Config Actions

**C3 FAILS — bearing slow drift WATCH > 15 min**
- Reduce Mech C window for Mot.SV: 300 → 250
- Re-run M8 Gate M8-11 (thermal lag) to verify still valid
- Re-run M12 C3 and C14

**C3 FAILS — CUSUM S_n flat (GATE-C3-5)**
- Verify score_B routing in `anomaly.py` (Invariant 19)
- Check TCN-AE Level 2 output — score_B may be near zero for sev 0.25
- If score_B near zero at sev 0.25: lower cusum_k by 10% and re-test
- Re-run M12 C3 and C14

**C6 FAILS — seal slow drift WATCH > 20 min**
- Reduce Mech C window for Pres.SV: 300 → 200
- Lower Spearman threshold for Pres.SV: 0.70 → 0.60
- Re-run M8 Gate M8-9, Gate M8-10
- Re-run M12 C6 and C13

**C8 FAILS — overloading WATCH > 15 min**
- Lower Spearman threshold for Temp.SV: 0.70 → 0.65
- Re-validate M8 Gate M8-7 (FPR impact check)
- Re-run M12 C8

**C11 FAILS — secondary_faults not populated**
- Verify M7 predict_proba threshold for secondary: 0.30 correct
- Check compound_interaction_flag computed correctly in M6.5 feature extract
- Re-run M7 compound confusion gates, then re-run M12 C11

**C12 FAILS — bearing detection > 90 steps, thermal broken**
- Reduce Mot.TV weight: 0.3 → 0.1 in M8
- Verify M8 Gate M8-11 still passes
- Retrain M8 → Re-run full M12 suite

**C13 FAILS — seal liability gate (most critical)**
- Reduce Mech C window for Pres.SV: 300 → 200
- Lower Spearman threshold: 0.70 → 0.60 for Pres.SV
- Add more mild seal sequences (sev 0.2) to M6 → recalibrate M8 Mech C
- Re-run M12 C13 (**must pass before ANY deployment**)

**C14 FAILS — CUSUM S_n flat (GATE-C14-7)**
- Verify score_B is routed to CUSUM and NOT to rolling baseline (Invariant 19)
- Check TCN-AE Level 2 output — score_B near zero for sev 0.25 bearing
- If score_B too low: reduce cusum_k by 10% and re-test
- Verify zt_buffer fills to at least 6 windows before CUSUM is evaluated
- Re-run M12 C3 and C14 together

**C15 FAILS — cluster transition masked**
- Widen cluster-conditional boundary detection window
- Add transition_flag: if cluster changes within last 20 windows → apply tighter fault-channel threshold specifically
- Re-run M12 C15

**C16 FAILS — compound misclassified**
- Add inter-channel consistency check: if Pres.SV drift POSITIVE AND Mot.SV drift POSITIVE → tag Pres.SV as sensor_drift (not seal) automatically (seal failure is ALWAYS negative Pres.SV drift)
- Re-run M12 C16

**C17 FAILS — bearing+seal compound, secondary_faults missing**
- Verify compound_interaction_flag feature computed correctly for Spearman lag shift between Mot.SV and Pres.SV channel errors
- Verify M6B bearing+seal sequences trained M7 correctly
- Check secondary fault threshold (0.30 may need lowering to 0.25)
- Re-run M7 compound gates then M12 C17

**C17 FAILS — score_C missing from top-3 SHAP (GATE-C17-7)**
- Verify score_C is included in M7 XGBoost feature set
- Check TCN-AE Level 2 output — score_C value near zero may rank low
- If score_C = 0.0 for compound bearing+seal: investigate TCN-AE C training
- Re-run M12 C17

**C18 FAILS — overloading+seal cascade, second fault flag missing**
- Verify M5 physics correctly simulates thermal→hydraulic cascade; Pres.SV decline from t=100 must be detectable above noise
- Verify Mech C Pres.SV window 300 sufficient for 300-step onset
- If thermal_decoupling_flag transition not detected: check M5 coupling equation
- Re-run M12 C18

**C18 FAILS — score_C missing from top-3 SHAP (GATE-C18-7)**
- Same investigation as C17 score_C failure
- Check whether thermal→hydraulic transition is too gradual for TCN-AE
- Recalibration: lower score_C sensitivity threshold in TCN-AE or increase severity of M6B cascade training examples
- Re-run M12 C18

> **After any recalibration:** Re-run FULL M8 gate suite before re-running M12. Do not re-run M12 on partial fixes — full M8 gate check first.

---

## Safety Gate

### If All 18 Configs Pass All Gates

```
status: PRODUCTION_VALIDATED
Output: outputs/M12_validation_certificate.md
```

**Certificate Contents**
- Date of validation
- M4 LSTM-AE Level 1 model SHA (from torch.save state dict hash)
- TCN-AE Level 2 model SHA (from torch.save state dict hash) *(NEW v3.0)*
- M7 XGBoost model hash (pickle SHA256)
- All 18 config results (PASS/FAIL + detection_lag per config)
- Statement:

  > "M4 LSTM-AE Level 1 + TCN-AE Level 2 (M8) + M7 XGBoost (22-class, labels 0–21) + M9 Physics Tools validated on 18 adversarial configurations generated fresh from M5 physics engine. Model has NEVER been exposed to M12 sequences during training. System cleared for deployment on 110 kW, 40 bar, 7-stage multistage centrifugal pump — CIRA SACIP configuration."

- Disclaimer:

  > "Validation is specific to CIRA SACIP sensor installation. Deployment on different pump or sensor layout requires re-validation with installation-specific data. PumpSmart is a condition monitoring tool per ISO 13374 Level 3. It is NOT a Safety Instrumented System per IEC 61511."

### If Any Config Fails

```
status: RECALIBRATION_REQUIRED
Output: outputs/M12_recalibration_log.csv
Contents: Config ID, Gate ID, failure value, target value, recalibration action
Deployment BLOCKED until PRODUCTION_VALIDATED.
```

---

## M12 Outputs

| File | Description |
|------|-------------|
| `src/module_12_validation_suite.py` | Main validation script |
| `outputs/M12_validation_log.csv` | Per-config, per-gate results |
| `outputs/M12_detection_latency.png` | Latency bar chart (all 18 configs) |
| `outputs/M12_alert_state_traces.png` | Alert state over time per config |
| `outputs/M12_channel_drift_traces.png` | Mech C Spearman per config |
| `outputs/M12_compound_fault_traces.png` | C17, C18 dual-channel drift |
| `outputs/M12_xgboost_stage_outputs.png` | Stage 1/2/3 confidence over time |
| `outputs/M12_cusum_accumulator_traces.png` | *(NEW v3.0)* score_B S_n over time for C3 + C14; confirms CUSUM accumulation path |
| `outputs/M12_zt_buffer_depth_log.csv` | *(NEW v3.0)* zt_buffer_len at each WATCH trigger for all slow drift configs |
| `outputs/M12_validation_certificate.md` | If PRODUCTION_VALIDATED |
| `outputs/M12_recalibration_log.csv` | If RECALIBRATION_REQUIRED |
| `outputs/reports/module_12_validation_report.md` | Full validation report |

---

## M12 Paste Text Keys

| Key | Value / Gate |
|-----|-------------|
| `M12_configs_total` | 18 |
| `M12_configs_passed` | [X/18] |
| `M12_configs_failed` | [list of failed config IDs] |
| `M12_C3_watch_latency_steps` | [value — gate ≤ 900] |
| `M12_C6_watch_latency_steps` | [value — gate ≤ 1200] |
| `M12_C8_watch_latency_steps` | [value — gate ≤ 900] |
| `M12_C13_watch_latency_steps` | [value — gate ≤ 1200 — LIABILITY] |
| `M12_C14_watch_latency_steps` | [value — gate ≤ 900 — LIABILITY] |
| `M12_C4_danger_latency_steps` | [value — gate ≤ 30 — cavitation] |
| `M12_C11_compound_result` | cavitation_primary/bearing_primary + secondary_faults_populated |
| `M12_C16_compound_result` | both_flags_fired + secondary_faults_populated |
| `M12_C17_compound_result` | bearing_primary + seal_secondary_prob (value) |
| `M12_C18_compound_result` | overloading_primary + seal_secondary_fires_at_step |
| `M12_C12_thermal_broken_lag` | [steps — gate ≤ 90] |
| `M12_C15_transition_result` | PASS/FAIL |
| `M12_cusum_rising_C3` | True/False *(NEW v3.0)* |
| `M12_cusum_rising_C14` | True/False *(NEW v3.0)* |
| `M12_zt_buffer_fills_before_tcn_gates` | True/False *(NEW v3.0)* — confirms zt_buffer_len ≥ 6 before TCN score_B is trusted in all configs |
| `M12_score_C_non_null_C17_step` | [step number — gate ≤ 200] *(NEW v3.0)* |
| `M12_score_C_non_null_C18_step` | [step number — gate ≤ 300] *(NEW v3.0)* |
| `M12_physics_context_in_danger_response` | True/False *(NEW v3.0)* — spot-checked on C1, C4, C7 DANGER responses |
| `M12_limitation_flags_in_watch_response` | True/False *(NEW v3.0)* — spot-checked on C3, C6, C14 WATCH responses |
| `M12_validation_status` | `PRODUCTION_VALIDATED` / `RECALIBRATION_REQUIRED` |
| `M12_certificate_generated` | True/False |
| `Status_for_deployment` | `CLEARED` / `BLOCKED` |

---

## Document Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-04-12 | Initial creation — 16 configs derived from `module_pathway_M1_to_M12_v10.md` |
| v2.0 | 2026-04-12 | Bias-audit cascade: +Config 17 (bearing+seal compound), +Config 18 (overloading+seal compound), Stage 1/2/3 XGBoost output refs in all gates, multi-label secondary_faults gates in C11/C16/C17/C18, updated detection matrix, recalibration loop additions for C11/C17/C18, 2 new output plots, 4 new paste keys |
| v3.0 | 2026-04-21 | Architecture v14.2: Model stack under test updated to M4 LSTM-AE Level 1 + TCN-AE Level 2 + M7 XGBoost 22-class. score_C gates added to C17 and C18 (GATE-C17-7, GATE-C18-7). CUSUM S_n rising gates added to C3 (GATE-C3-5/6) and C14 (GATE-C14-7/8). score_C = None acceptable clause added to all slow drift configs (C3, C6, C8, C13, C14) at early WATCH. physics_context non-null gate added to all DANGER configs. limitation_flags non-empty gate added per severity tier. Certificate updated with TCN-AE model hash field. 2 new output files (CUSUM accumulator traces, zt_buffer log). 7 new paste keys (total 25). Recalibration loop extended for CUSUM flat failure and score_C missing failure cases. |

---

*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head | Dataset: CIRA SACIP | Configs total: 18 | Standards: ISO 10816-3 | ISO 13373-3 | ISO 13374 Level 3 | IEC 61511 boundary | Architecture: v14.2 | Classes: 22 (labels 0–21) | Detection layers: 4 | Model stack: M4 LSTM-AE L1 + TCN-AE L2 + M7 XGBoost + M9 Physics Tools*
