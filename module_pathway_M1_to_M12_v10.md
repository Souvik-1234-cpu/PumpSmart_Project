# PumpSmart — Module Pathway v10.0
# Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring
# Full Pipeline: M1 → M12 (Unified Reference Document)
# Updated: 2026-04-11 | Author: Souvik | Version: 10.0
# Supersedes: module_pathway_M1_to_M12_v9.md
#
# KEY CHANGES IN v10.0 (vs v9.0):
#   - SLOW DRIFT DETECTION ARCHITECTURE added (liability-critical)
#   - M6A: Severity 0.2–0.3 mandate explicitly tied to trend accumulator training
#   - M8: Three-layer temporal detection (single-window + rolling mean + slope detector)
#   - M8: Per-channel drift monitor added for single-channel slow fault detection
#   - M8: State machine expanded to NORMAL → WATCH → WARN → DANGER (4 states)
#   - M10: UI updated to display 4-state alert system
#   - LIABILITY FRAMEWORK documented — defines scope boundary for model responsibility
#   - Cross-module invariant #15 added: slow drift must be catchable by M8 trend layer

---

## ASSET CONTEXT — READ BEFORE EVERY MODULE

```
PUMP NAMEPLATE (110 kW INDUSTRIAL MULTISTAGE CENTRIFUGAL)
─────────────────────────────────────────────────────────────
Motor shaft power  : 110 kW  ← IEC Frame 315mm, 400V, 2-pole
Motor speed        : 2980 RPM
Pump stages        : 7 impellers (multistage centrifugal)
Pump flow rate     : 45 m³/h
Pump total head    : 450 m
Pump max pressure  : 40 bar
Pump hydraulic kW  : ~55 kW  (P_hyd = ρgQH/η = 1000×9.81×(45/3600)×450/0.65)

NOTE: "10 kW" in original Zenodo source refers to a sub-duty point.
Nameplate motor is definitively 110 kW.
USE 110 kW FOR ALL PHYSICS CALCULATIONS IN M5+.

Replacement cost   : >₹50 lakh (industrial capital asset)
Failure consequence: Process shutdown, secondary damage, safety risk
Standard           : ISO 10816-3 (vibration), ISO 13373-3 (condition monitoring)
```

This is not a household pump. Every ML decision has a physical safety consequence.
False negatives on a 110 kW, 40 bar, 450m head multistage pump = catastrophic failure.

---

## LIABILITY FRAMEWORK — MODEL RESPONSIBILITY BOUNDARY
## (Added v10.0 — NON-NEGOTIABLE, governs M6A through M12)

```
THREE CATEGORIES OF FAILURE — ONLY ONE IS MODEL'S RESPONSIBILITY:

CATEGORY 1 — SENSOR HARDWARE FAILURE
  Scenario : A physical sensor malfunctions and feeds wrong values to model
  Liability : NOT the model's responsibility
  Defense   : sensor_failure fault class (M6A/M7) flags dead/drifting sensors
              Model reports what sensors feed it — cannot fabricate correct readings
              Disclaimer in M10 UI: "Inference quality depends on sensor integrity"

CATEGORY 2 — NOVEL OUT-OF-DISTRIBUTION EVENT
  Scenario : Physically random, unpredictable external event
             (e.g., solid gravel ingested from water source → immediate impeller failure)
  Liability : NOT the model's responsibility
  Defense   : LSTM-AE will flag ANY signal deviating from normal as anomalous
              XGBoost will misclassify cause — but anomaly flag will fire
              These events cannot be trained on — future cannot be predicted
              M10 disclaimer: "Model trained on CIRA SACIP installation patterns only"

CATEGORY 3 — SLOW DRIFT FAULT (weeks-scale degradation)
  Scenario : Progressive fault (seal wear, bearing degradation) developing over days/weeks
             Single 50-step window never crosses threshold → fault missed entirely
  Liability : THIS IS THE MODEL'S RESPONSIBILITY
  Why       : Within training distribution, physically predictable, detectable
              with correct architecture — this is the exact purpose of condition monitoring
  Solution  : Three-layer temporal detection in M8 (see M8 section)
              Severity 0.2–0.3 sequences in M6A train the trend accumulator

PumpSmart is a CONDITION MONITORING system (ISO 13374 Level 3):
  Level 1 — Basic Process Control (DCS)        ← hardwired trips
  Level 2 — Safety Instrumented System (SIS)   ← IEC 61511 certified
  Level 3 — Condition Monitoring (PumpSmart)   ← THIS SYSTEM
  Level 4 — Planned Maintenance                ← informed by PumpSmart alerts

PumpSmart value: catch Category 3 faults early enough that maintenance is
scheduled before SIS trip (Level 2) or emergency shutdown (Level 1).
System is advisory — NOT a replacement for hardwired protection.
```

---

## DATASET

```
Source  : CIRA (Italian Aerospace Research Centre), SACIP project
          Zenodo record 15301820 → https://zenodo.org/records/15301820
Pumps   : A, B, C (multistage centrifugal, industrial)
Files   : 9 CSVs — 3 pumps × 3 operational days
Size    : 26.3 MB total
Sampling: 1 second uniform (all 9 files confirmed)

Columns (11 total):
  Timestamp       → YYYY-MM-DD hh:mm:ss
  X_ACR_Mot.PV   → Motor casing vibrational velocity, mm/s
  X_ACR_Mot.SV   → Motor casing broadband peak acceleration envelope, mm/s²
  X_ACR_Mot.TV   → Motor casing accelerometer contact temperature, °C
  X_ACR_Pmp.PV   → Pump casing vibrational velocity, mm/s
  X_ACR_Pmp.SV   → Pump casing broadband peak acceleration envelope, mm/s²
  X_ACR_Pmp.TV   → Pump casing accelerometer contact temperature, °C
  X_Temp.SV      → Motor casing surface temperature, °C
  X_Pres.SV      → Pump discharge pressure, bar
  Barometer      → Atmospheric pressure, mbar (dropped before M3)
  Temperature    → Ambient temperature, °C (used as T_ambient in M3)
```

---

## INVIOLABLE RULES (Apply M1 → M12)

```
TIME SERIES INTEGRITY:
  NEVER concatenate raw CSVs before M1 cleaning + segmentation
  NEVER create windows crossing segment boundaries
  segment_id preserved in ALL downstream dataframes
  Windows generated per segment only
  Combining normalized window pools: ALLOWED only after M3

NaN POLICY:
  Drop ALL rows with any null value. No interpolation. No fill. Hard drop.
  After dropping → re-segment via timestamp gap detection
  Gap threshold = gap > 2× median sampling interval = new segment

NORMALIZATION (MANDATORY — RAW SENSOR VALUES NEVER ENTER ML):
  P*  = P_actual / P_cluster_mean
  a*  = a_actual / a_cluster_mean
  ΔT* = (T - T_cluster_min) / (T_cluster_max - T_cluster_min)
  Normal operation → 0 to 1.0
  Fault → drift above 1.0 or anomalous temporal pattern
  Baselines: M3_normalization_config.json (per-cluster, per-sensor)

SCOPE BOUNDARY (NEVER VIOLATE):
  if pump_type == "household": return physics_advisory_only()
  else: return ml_prediction()
```

---

## CONFIRMED PHYSICAL COUPLINGS (from M2 — must hold in ALL synthetic data)

```
X_ACR_Mot.TV ↔ X_Temp.SV     r = 0.9793  (motor heat → casing)
X_ACR_Pmp.PV ↔ X_ACR_Pmp.SV  r = 0.8882  (displacement → peak vibration)
X_ACR_Pmp.PV ↔ X_Pres.SV     r = 0.8779  (impeller displacement → pressure)

FAULT PROPAGATION (physics-causal, must be reproduced in M6 and M12):
  Bearing wear       : Mot.SV↑ → Mot.TV↑, Temp.SV↑, Pmp.SV↑  [lag 20-40 steps]
  Impeller imbalance : Pmp.PV↑ + Pmp.SV↑ → Pres.SV oscillates, Mot.PV↑
  Cavitation         : Pres.SV drops+erratic → Pmp.SV↑↑, Pmp.TV↑ [startup ONLY]
  Seal failure       : Pres.SV↓ progressive → Pmp.TV↑ [ss or high_load]
  Overloading        : Temp.SV↑ drift → Mot.TV↑, Mot.SV↑ [steady_state ONLY]
  Sensor failure     : Target channel only → flatline/spike; all 7 others stay normal

THERMAL DECOUPLING BY FAULT TYPE (from M5 validation plots):
  Bearing wear (steady_state) : r=0.972  ← coupling PRESERVED (heat source = bearing)
  Overloading (steady_state)  : r=0.997  ← coupling STRONGLY PRESERVED (thermal load)
  Seal failure (steady_state) : r=-0.013 ← coupling BROKEN (hydraulic fault, not thermal)
  Cavitation (startup)        : r=0.376  ← coupling WEAK (hydraulic, not thermal)
  Bearing wear (high_load)    : r=0.949  ← coupling preserved
  Normal (steady_state)       : r=-0.062 ← baseline near-zero (expected at steady-state)
```

---

## CONFIRMED OPERATIONAL BOUNDS (M2/M3 real data)

```
Cluster distribution (117,970 rows):
  Startup      : 49,884 rows (42.3%)
  Cooldown     : 26,851 rows (22.8%)
  High-load    : 26,600 rows (22.5%)
  Steady-state : 14,635 rows (12.4%)

Outlet Pressure (bar):
  Startup      : 0.43 – 0.85    Cooldown     : 0.45 – 44.4
  Steady-state : 0.69 – 43.3    High-load    : 37.4 – 44.8
  Transient max observed: 46.7 bar (Joukowsky water hammer)

Motor Vibration SV (mm/s²):
  Startup      : 0.39 – 0.58    Cooldown     : 0.42 – 12.1
  Steady-state : 12.0 – 21.5    High-load    : 22.2 – 77.7
  Absolute max transient: 456.6 mm/s² (anomalous spike)

Pump Vibration SV (mm/s²):
  Startup      : 0.38 – 0.57    Cooldown     : 0.41 – 0.61
  Steady-state : 24.5 – 55.3    High-load    : 19.6 – 34.7
```

---

## M4 v8 WINSORIZATION BOUNDS (cluster-conditional — LOCKED for M6 + M12)

```
Channel           | Cluster      | Upper Multiplier | Physics Basis
─────────────────────────────────────────────────────────────────────
X_Pres.SV_norm   | startup      | 3.0x             | Joukowsky transient headroom
                 | steady_state | 5.6x             | Wide valid range (std=13 bar)
                 | high_load    | 2.0x             | Tight — faults caught immediately
                 | cooldown     | 3.0x             | Depressurization transients
X_ACR_Pmp.PV    | startup      | 3.2x             | ISO 13373-3: BPF harmonics
                 | all others   | 2.6x             | v8 behaviour preserved
X_ACR_Mot.SV    | all clusters | 6.7x             | Uniform — no cluster physics
X_ACR_Pmp.SV    | all clusters | 8.8x             | Uniform — broadband RMS spike
X_ACR_Mot.PV    | all clusters | 2.2x             | Uniform — displacement bounded
Source: M4_spike_config.json — DO NOT OVERRIDE in M6 or M12
```

---

## ══════════════════════════════════════════════════
## M1 — DATA INGESTION & HARD CLEANING
## Status: ✅ COMPLETED (2026-03-25)
## ══════════════════════════════════════════════════

### Key Results

| Metric | Value |
|---|---|
| Raw files processed | 9 |
| Total raw rows | 173,730 |
| Total clean rows | 147,217 |
| Total dropped rows | 26,513 (15.26%) |
| Total segments created | 66 |
| Usable segments | 27 → 25 (after C_Day3 exclusion) |
| Worst null column | Barometer (Pump_B_Day3: 65.46%) |
| Sampling interval | 1s uniform, all files |

### Outputs
```
data/clean/Pump_*_clean.csv
data/clean/segment_registry.csv
outputs/M1_file_summary.csv
outputs/plots/M1_null_heatmap.png
outputs/plots/M1_segment_timeline.png
```

---

## ══════════════════════════════════════════════════
## M2 — EDA + OPERATING MODE CLUSTERING
## Status: ✅ COMPLETED (2026-03-26)
## ══════════════════════════════════════════════════

### Key Results

| Metric | Value |
|---|---|
| Usable rows (post-M1) | 117,970 |
| Optimal K | 4 |
| Silhouette score | 0.5458 |
| Optimal window size | 50s |
| Stationary sensors (ADF) | 8/8 |
| Top correlation | Mot.TV ↔ Temp.SV: r=0.9793 |
| PCA variance captured | PC1=47.37%, PC2=32.97% |

### Operating Mode Map

| Cluster | Mode | Rows | % |
|---|---|---|---|
| C0 | cooldown | 26,851 | 22.8% |
| C2 | startup | 49,884 | 42.3% |
| C1 | steady_state | 14,635 | 12.4% |
| C3 | high_load | 26,600 | 22.5% |

### Outputs
```
outputs/M2_cluster_bounds.csv
outputs/M2_labelled_data.csv
outputs/M2_cluster_bounds_units.json
outputs/plots/M2_*.png (5 plots)
```

---

## ══════════════════════════════════════════════════
## M3 — DIMENSIONLESS NORMALIZATION
## Status: ✅ COMPLETED (2026-03-28)
## ══════════════════════════════════════════════════

### Key Results

| Metric | Value |
|---|---|
| Normalised rows | 117,970 |
| Clusters used | 4 |
| Channels normalised | 8 |
| Range issues | None |

### Normalization Formulas (LOCKED)
```
Pressure   : P*  = P_actual / P_cluster_mean
Vibration  : a*  = a_actual / a_cluster_mean
Temperature: ΔT* = (T - T_cluster_min) / (T_cluster_max - T_cluster_min)

T_ambient source: T_cluster_min (cluster-relative, NOT hardcoded ambient)
Reason: climate-agnostic — valid across field deployments
Small negatives (flash evaporative cooling in cooldown): PRESERVED, not clipped
```

### Outputs
```
data/normalized/normalised_data.csv
models/M3_normalization_config.json   ← LOCKED baselines
outputs/reports/module_03_normalization_report.md
```

---

## ══════════════════════════════════════════════════
## M4 — LSTM-AE PATTERN EXTRACTION (NORMAL BASELINE)
## Status: ✅ COMPLETED v8 (2026-03-28)
## ══════════════════════════════════════════════════

### Architecture
```
Input    : (batch, 50, 8) — 50-timestep windows, 8 normalized channels
Encoder  : LSTM(8→128, layers=2, dropout=0.3) → bottleneck(128→64)
Decoder  : LSTM(64→128) — hidden state seeded from encoder + LayerNorm
Loss     : 0.6×MAE + 0.4×MSE (physics-weighted)
Channel weights: Mot.SV=2.0, Pmp.SV=2.0, Pres.SV=2.0, Mot.PV=1.5,
                 Pmp.PV=1.5, Temp.SV=1.0, Mot.TV=0.8, Pmp.TV=0.8
Optimizer: AdamW | LR: CosineAnnealingWarmRestarts (T0=20)
AMP      : GradScaler + autocast (CUDA)
Parameters: 505,096
```

### Key Results

| Metric | Value |
|---|---|
| Clean windows (normal) | 9,711 |
| Train / Val split | 8,254 / 1,457 |
| Best val loss | 0.026862 |
| Best epoch | 141 / 150 |
| Mean MAE (val) | 0.026765 |
| Anomaly threshold | 0.110058 (mean + 3σ ∪ P99) |
| Separation ratio | 4.11x |
| False alarms (val) | 8 (0.55%) |
| Spike seeds extracted | 1,044 → M4_spike_seeds.npy |
| Spike rows excluded | 12,620 |

### Spike Seed Fault Hints (LOCKED for M6)

| Fault Hint | Count | M6 Fault Mapping |
|---|---|---|
| mechanical_transient | 472 | bearing_wear, impeller_imbalance |
| pressure_transient | 408 | cavitation, seal_failure |
| impeller_cavitation | 113 | cavitation (direct) |
| bearing_impact | 44 | bearing_wear (direct) |
| pressure_spike_highload | 7 | overloading |

### Outputs
```
models/lstm_ae_baseline_best.pth
data/synthetic/M4_spike_seeds.npy        ← shape=(1044, 50, 8) → M6 + M12 input
data/synthetic/M4_spike_seeds_meta.csv
data/synthetic/M4_spike_config.json      ← cluster-conditional winsor bounds (LOCKED)
outputs/M4_threshold_config.json         ← threshold=0.110058, separation=4.11x
```

---

## ══════════════════════════════════════════════════
## M5 — PHYSICS ENGINE
## Status: ✅ COMPLETED (2026-03-29)
## ══════════════════════════════════════════════════

### Key Results

| Metric | Value |
|---|---|
| Equations implemented | 20 |
| Nameplate pass count | 20/20 ✅ |
| S5 validation cases | 19/19 PASS ✅ |
| S7 fault sequence tests | 24/26 PASS |
| Known failures | bearing_wear_steady_state_s100, overloading_steady_state_s140 |
| Failure reason | Edge-case severity values outside declared production range |
| Bearing heat model | Euler integration (dt steps) — PATCH 3 applied |
| Sensor failure spike guard | ±5σ clamp — PATCH 5 applied |
| Overloading severity range | [0.5, 1.0] — mild overloading via sub-cluster (M6) |

### Thermal Coupling Validation (M5 Plots Confirmed)

| Fault / Cluster | r(Mot.TV*, Temp.SV*) | Physics Interpretation |
|---|---|---|
| Bearing wear (steady_state) | 0.972 | Coupling preserved — heat from bearing |
| Overloading (steady_state) | 0.997 | Strongly preserved — thermal overload |
| Seal failure (steady_state) | -0.013 | Coupling BROKEN — hydraulic fault |
| Cavitation (startup) | 0.376 | Weak — hydraulic not thermal |
| Bearing wear (high_load) | 0.949 | Preserved |
| Normal (steady_state) | -0.062 | Baseline near-zero (correct) |

### Outputs
```
src/module_05_physics_engine.py
models/fault_rules.json              ← 6 fault types, physics-causal rules
models/M5_physics_config.json
models/unit_registry.json
outputs/plots/M5_fault_signatures.png
outputs/plots/M5_thermal_coupling.png
outputs/reports/module_05_physics_engine_report.md
```

---

## ══════════════════════════════════════════════════
## M6A — SYNTHETIC DATASET GENERATOR (HYBRID)
## Status: 🔲 NOT STARTED — NEXT ACTIVE MODULE
## ══════════════════════════════════════════════════

## ARCHITECTURE DECISION: HYBRID PATH C (LOCKED 2026-04-08)

### Three-Source Strategy

```
SOURCE 1: Real CIRA Normal Windows (M3 normalized pool)
  → 1200 Type-A (normal) sequences across all 4 clusters

SOURCE 2: M4 Spike Seeds (Real CIRA fault windows)
  → Physics pseudo-labeling via M5 (cosine similarity > 0.85)
  → Used as fault ONSET SEEDS (t=0 to t=49)
  → M5 physics engine continues causal progression t=50 to t=199
  → Provides real anomaly character to synthetic fault sequences

SOURCE 3: Physics Synthetic (M5 engine — pure)
  → Fills gaps where spike seeds are sparse
  → Covers full severity spectrum [0.2 → 1.0]
  → Sub-cluster augmentation for mild faults (severity < 0.5)
  → Pure physics for severe faults (severity ≥ 0.5)
```

### WHY SEVERITY 0.2–0.3 SEQUENCES ARE MANDATORY (v10.0 — LIABILITY CRITICAL)

```
Severity 0.2–0.3 sequences are NOT generated for class balance.
They are the SPECIFIC TRAINING DATA for M8's trend accumulator.

Without severity 0.2–0.3 sequences:
  M8 learns only acute fault patterns (severity ≥ 0.5)
  Slow drift faults (seal wear over weeks, bearing degradation over days)
  develop at effective severity 0.2–0.3 in any single 50-step window
  → M8 never fires → fault missed entirely → LIABILITY EXPOSURE

With severity 0.2–0.3 sequences:
  M8 learns what very-early-stage fault signatures look like
  Trend accumulator (rolling mean + slope detector) is calibrated
  on these mild sequences → correctly escalates WATCH state over time

Physically: at severity 0.2, seal wear shifts Pres.SV* by ~0.08 above baseline
per window. No single window crosses 0.110058. But over 200+ windows
(~3 minutes of operation), rolling mean crosses WATCH threshold.
Over 500+ windows (~8 minutes), slope detector fires.
For a fault developing over weeks, 8 minutes = operationally instantaneous.

MANDATORY SEVERITY DISTRIBUTION PER FAULT CLASS:
  bearing_wear   : 30% at [0.2,0.4], 40% at [0.4,0.7], 30% at [0.7,1.0]
  seal_failure   : 35% at [0.2,0.4], 35% at [0.4,0.7], 30% at [0.7,1.0]
  overloading    : mild sub-cluster [0.2,0.5] + physics [0.5,1.0]
  cavitation     : 20% at [0.2,0.5], 80% at [0.5,1.0] (startup physics)
  impeller_imbal : 25% at [0.2,0.5], 75% at [0.5,1.0]
  sensor_failure : gradual drift [0.2,0.5] + flatline/spike [0.5,1.0]
```

### Severity × Source Matrix (LOCKED)

| Fault Type | Mild (sev 0.2–0.5) Source | Severe (sev 0.5–1.0) Source | Allowed Clusters |
|---|---|---|---|
| bearing_wear | Sub-cluster augmentation | Spike seeds (44) + Physics | startup, steady_state, high_load |
| impeller_imbalance | Sub-cluster augmentation | Physics synthetic | steady_state, high_load |
| cavitation | Sub-cluster (startup only) | Spike seeds (113) + Physics | startup ONLY |
| seal_failure | Sub-cluster (ss/hl) | Physics synthetic | steady_state, high_load |
| overloading | Sub-cluster (steady_state) | Physics synthetic | steady_state ONLY |
| sensor_failure | Physics (gradual drift) | Spike seeds (mech_trans) + Physics | ANY cluster |

### Sub-Cluster Augmentation Logic (Mild Faults)

```
1. From M3 normalized pool, select windows near cluster boundary
   (within 90th–97.5th percentile of primary fault channel)
2. Apply fault-specific perturbation at reduced severity (0.2–0.5)
3. Enforce physical coupling: r(Mot.TV, Temp.SV) must remain > 0.85
   EXCEPT for cavitation (r allowed to drop to 0.3)
4. Validate: perturbed window must produce MAE > 0.110058 (M4 threshold)
   If not → increase perturbation until threshold is crossed
5. Enforce cluster-conditional winsor ceilings from M4_spike_config.json
```

### Sequence Count (LOCKED — based on RTX 4060 capacity analysis)

| Class | Count | Rationale |
|---|---|---|
| normal | 1200 | Real CIRA windows (sampled from 9711 clean windows) |
| bearing_wear | 1200 | 44 spike seeds + sub-cluster + physics |
| impeller_imbalance | 1200 | Sub-cluster + physics (no direct spike seeds) |
| cavitation | 1200 | 113 spike seeds + sub-cluster (startup only) + physics |
| seal_failure | 1200 | Sub-cluster + physics |
| overloading | 1200 | Mild: sub-cluster [0.2,0.5]; Severe: physics [0.5,1.0] |
| sensor_failure | 1200 | Physics (gradual drift + flatline + spike variants) |
| **TOTAL** | **8400** | ~13 MB in memory — well within 8GB VRAM |

### Spike Seed Integration Protocol

```
Each spike seed window (shape 50×8) used as fault onset:
  t=0  to t=49 : REAL CIRA anomaly (from M4_spike_seeds.npy)
  t=50 to t=199: M5 physics engine continues causal progression
                 Initial condition = last state of spike seed window
                 Slope matching at seam (dX/dt continuity, no discontinuity)

Pseudo-labeling:
  Each spike seed matched against M5 fault signature templates
  Cosine similarity > 0.85 → assigned fault label + used as seed
  Cosine similarity ≤ 0.85 → discarded (ambiguous signal)

Validation:
  Compare synthetic cavitation stats vs 113 impeller_cavitation seeds
  If distribution mismatch (KS test p < 0.05) → flag physics rule mismatch
```

### M6 Physics Invariants (ALL must hold per sequence)

```
1. Cluster-conditional winsor ceilings respected (M4_spike_config.json)
2. Physical couplings: r(Mot.TV, Temp.SV) consistent with fault type
3. Fault channels drift ABOVE 1.0; non-fault channels stay near 1.0
4. No negative pressure values; no temperature below cluster minimum
5. Conservation of energy: thermal rise proportional to mechanical dissipation
6. Rate-of-change (dX/dt) used as primary fault language — NOT absolute thresholds
7. All fault sequences: MAE > 0.110058 when passed through M4 LSTM-AE
8. All normal sequences: MAE < 0.110058 when passed through M4 LSTM-AE
9. Mild sequences (sev 0.2–0.3): MAE in range [0.110058, 0.140] — near-threshold
   (ensures trend accumulator training data is correctly calibrated)
```

### M6 Outputs
```
data/synthetic/M6_sequences.pkl          ← 8400 sequences, shape each (200, 8)
data/synthetic/M6_sequence_meta.csv      ← seq_id, label, severity, source, cluster, seed_idx
data/synthetic/M6_validation_report.json ← spike seed match scores, coupling fidelity
outputs/plots/M6_label_distribution.png
outputs/plots/M6_fault_signatures_grid.png
outputs/plots/M6_coupling_fidelity.png
outputs/plots/M6_mae_distribution.png
outputs/reports/module_06a_synthetic_report.md
```

### Paste Text Keys (fill after M6A run)
```
M6_total_sequences        : 8400
M6_sequences_per_class    : 1200
M6_spike_seeds_used       : [count of seeds with cosine_sim > 0.85]
M6_spike_seeds_discarded  : [count rejected due to low similarity]
M6_subcluster_mild_count  : [sequences generated from sub-cluster augmentation]
M6_physics_severe_count   : [sequences from pure physics]
M6_mild_severity_count    : [sequences with severity 0.2–0.4 — trend accumulator data]
M6_coupling_fidelity_pass : [% sequences passing coupling r check]
M6_mae_gate_pass_rate     : [% fault sequences with MAE > threshold]
M6_mild_mae_range         : [min–max MAE for severity 0.2–0.3 sequences]
M6_physics_violations     : [list or NONE]
Status for M6.5           : READY/BLOCKED
```

---

## ══════════════════════════════════════════════════
## M6.5 — LSTM-AE FEATURE EXTRACTOR (XGBoost Bridge)
## Status: 🔲 NOT STARTED (after M6A)
## ══════════════════════════════════════════════════

### Why This Module Exists

```
PROBLEM: XGBoost cannot consume raw time-series sequences (shape 200×8).
         Flattening destroys temporal ordering → 1600-dim sparse feature space.

SOLUTION: Run all M6 sequences through the M4 LSTM-AE (inference only).
          The reconstruction error array (shape 200×8) encodes temporal
          anomaly information. Extract statistical features from this error
          array → produce one static row per sequence → XGBoost trains on rows.

RESULT: XGBoost sees static tabular data (like BioSmart) but every feature
        carries temporal meaning from the LSTM-AE's reconstruction behavior.
```

### Feature Extraction — 24 Features per Sequence

```
Per-channel mean reconstruction error (8 features):
  mean_err_MotPV, mean_err_MotSV, mean_err_MotTV,
  mean_err_PmpPV, mean_err_PmpSV, mean_err_PmpTV,
  mean_err_TempSV, mean_err_PresSV

Per-channel max reconstruction error (8 features):
  max_err_MotSV, max_err_PmpSV, max_err_PresSV, ...

Temporal evolution features (5 features):
  error_onset_lag     : timestep where error first crosses 2σ
  err_slope_primary   : rate of error growth on highest-error channel
  err_auc_primary     : area under error curve (fault energy proxy)
  kurtosis_err_PmpSV  : spike character of pump vibration error
  kurtosis_err_PresSV : spike character of pressure error

Cross-channel features (2 features):
  corr_delta_PmpSV_PresSV : change in correlation vs normal baseline
  thermal_decoupling_flag : 1 if MotTV error < 0.05 (hydraulic fault indicator)

Fuzzy fault membership (1 feature — NEW in v9.0):
  fuzzy_fault_membership  : μ_fault(e) from M8 fuzzy membership function
                            0.0 = clearly normal, 1.0 = clearly fault
                            Gives XGBoost a soft confidence signal

Label (carried from M6 metadata):
  label : 0=normal, 1=cavitation, 2=bearing_wear, 3=seal_failure,
          4=overloading, 5=impeller_imbalance, 6=sensor_failure
```

### Fuzzy Membership Function (Preview for M8 — used here as feature)

```
Boundaries derived from M4 results:
  μ_normal → μ_fault transition zone = [P95_normal, P5_fault]
  = [0.074673, ~0.15]  (estimated — calibrated precisely in M8)

  μ_fault(e) = 0.0                        if e < 0.074673
             = (e - 0.074673) / 0.075327  if 0.074673 ≤ e ≤ 0.15
             = 1.0                        if e > 0.15

Note: Exact boundaries will be recalibrated in M8 using two-population method.
M6.5 uses M4-derived estimate. M7 trained on this estimate is still valid
because M8 retrains LSTM-AE and M7 features will be re-extracted in M8.
```

### M6.5 Outputs
```
data/synthetic/M6_feature_matrix.csv    ← 8400 rows × 25 columns (24 features + label)
outputs/plots/M6B_feature_distributions.png
outputs/plots/M6B_fuzzy_membership_dist.png
outputs/reports/module_06b_feature_extractor_report.md
```

### Paste Text Keys
```
M6B_feature_matrix_rows   : 8400
M6B_features_per_row      : 24 + label
M6B_fuzzy_mean_normal     : [value] (should be near 0)
M6B_fuzzy_mean_fault      : [value] (should be near 1)
M6B_top_discriminating_feature: [feature with highest class separation]
Status for M7             : READY/BLOCKED
```

---

## ══════════════════════════════════════════════════
## M7 — STATIC FAULT CLASSIFIER (XGBoost)
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

### Input Clarification (v9.0 — CRITICAL)

```
XGBoost trains on: data/synthetic/M6_feature_matrix.csv
                   ← 8400 rows × 25 columns
                   ← Each row = one fault sequence, distilled to 24 features
                   ← Label column = fault type (0–6)

XGBoost does NOT see: raw sequences (200×8)
XGBoost does NOT see: raw sensor readings

This is the same paradigm as BioSmart — static tabular rows —
but features are temporally meaningful because they summarize
LSTM-AE reconstruction error evolution over 200 timesteps.
```

### Architecture
```
Input     : 8400 × 24 feature matrix (from M6.5)
Split     : 80% train, 20% test (stratified by label)
Model     : XGBoost (device='cuda' training, device='cpu' deployment)
Tuning    : Optuna — 50 trials, 5-fold CV
Output    : Fault class (7 types: normal + 6 faults)
Explainer : SHAP TreeExplainer (top-3 features per prediction)
Weights   : Inverse class frequency
```

### Expected SHAP Top Features Per Fault

| Fault | Expected Top-3 SHAP Features | Physics Basis |
|---|---|---|
| cavitation | mean_err_PmpSV, pres_chaotic_flag, thermal_decoupling_flag | Hydraulic fault — no thermal coupling |
| bearing_wear | err_slope_MotSV, mean_err_MotTV, corr_delta | Gradual vibration + thermal rise |
| overloading | mean_err_TempSV, err_slope_TempSV, fuzzy_fault_membership | Temperature-dominant fault |
| seal_failure | pres_monotonic_flag, mean_err_PresSV, mean_err_PmpTV | Progressive pressure drop |
| sensor_failure | max_err on single channel, all others near 0 | Single-channel anomaly |

### Requirements
```
Overall accuracy     : > 85%
Per-class F1         : > 0.80 for all fault types
Cavitation F1        : > 0.88 (safety-critical)
Sensor failure F1    : > 0.92 (easiest class — single channel)
SHAP top-3 features  : must be physically causal per fault type
SHAP TV dominance    : TV channels must NOT be top-3 for vibration faults
```

### Paste Text Keys
```
M7_accuracy             : [%]
M7_f1_per_class         : {normal, cavitation, bearing, seal, overload, imbalance, sensor}
M7_top3_shap_cavitation : [features]
M7_top3_shap_bearing    : [features]
M7_top3_shap_overload   : [features]
M7_shap_tv_dominance    : False (correct physics behaviour)
Status for M8           : READY/BLOCKED
```

---

## ══════════════════════════════════════════════════
## M8 — LSTM-AE v2 PRODUCTION MODEL + FUZZY LOGIC
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

## 🚨 M8 SAFETY MANDATE — HIGH-VALUE ASSET PROTECTION 🚨

```
Asset: 110 kW, 7-stage, 40 bar, 450m head multistage pump
Failure consequence: Process shutdown + secondary damage + injury
THIS MANDATE IS NON-NEGOTIABLE.
```

### M8 Architecture (Updated v10.0 — Four-Layer Detection)

```
STAGE 1: LSTM-AE Reconstruction (same as M4 + improvements)
  - Temporal attention (multi-head) over encoder outputs
  - Gradient penalty in loss: total_loss = 0.5×MAE + 0.3×MSE + 0.2×grad_penalty
  - MC Dropout (N=20 forward passes) for uncertainty quantification
  - Channel weights INCREASED for primary channels vs M4:
      M8: Mot.SV=2.5, Pmp.SV=2.5, Pres.SV=2.5 (was 2.0)
          Mot.PV=2.0, Pmp.PV=2.0 (was 1.5)
          Temp.SV=0.5, Mot.TV=0.3, Pmp.TV=0.3 (was 1.0/0.8/0.8)
  Reason: Further reduce reliance on placement-dependent temperature channels

STAGE 2: Fuzzy Membership Layer (core detection component)
  Boundaries calibrated by two-population method on M6 synthetic data:
    P95 of normal MAE distribution → lower fuzzy bound
    P5  of fault MAE distribution  → upper fuzzy bound
    Transition zone = [lower_bound, upper_bound]

  μ_fault(e) = 0.0                                    if e < lower_bound
             = (e - lower_bound) / (upper - lower)    if in transition
             = 1.0                                    if e > upper_bound

STAGE 3: SLOW DRIFT DETECTION — THREE MECHANISMS (v10.0 — LIABILITY CRITICAL)
  ─────────────────────────────────────────────────────────────────────────────
  This stage directly addresses Category 3 liability (see LIABILITY FRAMEWORK).
  A fault developing over days/weeks produces MAE elevation too small for
  single-window detection. These three mechanisms catch it reliably.

  MECHANISM A — Rolling Mean Gate (medium horizon: minutes)
    rolling_mean_MAE(last 200 windows) > 0.085  → WATCH state
    rolling_mean_MAE(last 100 windows) > 0.095  → WARN state
    Physics basis: 200 windows = ~3 min at 1Hz. Seal wear at severity 0.2
    raises mean MAE by ~0.008/100 windows. Detectable in WATCH within 10 min.

  MECHANISM B — Slope Detector (long horizon: hours)
    slope(MAE, last 500 windows) > 0.0003/window  → escalate alert state by 1
    Computed via linear regression on rolling MAE time series
    Physics basis: bearing degradation over 8h produces slope ~0.0001–0.0005
    At 0.0003/window, 500 windows = ~8 minutes to confirm trend
    For a fault developing over weeks, 8 minutes = operationally instantaneous

  MECHANISM C — Per-Channel Drift Monitor (single-channel slow faults)
    For each of 8 channels independently:
      If channel reconstruction error shows MONOTONIC INCREASE
      over 300+ consecutive windows → channel_drift_flag[channel] = True
      → early fault flag even if total MAE < all thresholds
    Threshold: Spearman rank correlation of channel_error vs time > 0.70
    Physics basis: seal failure shows in Pres.SV* alone initially.
    Total MAE averages across 8 channels — dilutes single-channel signal.
    Per-channel monitor catches this before total MAE responds.
    Fault type mapping:
      Pres.SV drift  → seal_failure_early flag
      Mot.SV drift   → bearing_wear_early flag
      Temp.SV drift  → overloading_early flag
      single channel flatline → sensor_failure flag

STAGE 4: Four-State Alert Machine (v10.0 — replaces binary flag)
  NORMAL  : rolling_score < 2.0 AND no channel_drift_flag AND no slope trigger
  WATCH   : rolling_mean > 0.085 OR slope trigger OR any channel_drift_flag
  WARN    : rolling_mean > 0.095 OR rolling_score 2.0–3.5
  DANGER  : single_window_MAE > 0.110058 OR rolling_score > 3.5

  State escalation rules:
    NORMAL → WATCH : sustained low-level anomaly detected (trend beginning)
    WATCH  → WARN  : trend confirmed over 100+ windows (maintenance schedule)
    WARN   → DANGER: threshold crossed (immediate maintenance action)
    DANGER → WARN  : MAE drops below threshold for 50+ consecutive windows
    WARN   → WATCH : rolling mean drops below 0.085 for 200+ windows
    WATCH  → NORMAL: all mechanisms clear for 300+ consecutive windows

STAGE 5: Cluster-Conditional Thresholds
  Separate threshold per operating mode (startup/steady_state/high_load/cooldown)
  startup threshold > steady_state threshold (wider normal range at startup)
  high_load threshold < startup threshold (tighter — faults caught immediately)
```

### Detection Coverage by Fault Type and Detection Horizon

| Fault | Single Window | Rolling Mean (WATCH) | Slope Detector | Per-Channel Monitor |
|---|---|---|---|---|
| Bearing wear severe (sev 0.8) | ✅ DANGER | ✅ | ✅ | ✅ |
| Bearing wear mild (sev 0.2) | ❌ misses | ✅ WATCH ~10min | ✅ ~8min | ✅ Mot.SV drift |
| Cavitation severe | ✅ DANGER | ✅ | N/A (fast) | ✅ Pres.SV |
| Seal failure slow (weeks) | ❌ misses | ✅ WATCH ~3min | ✅ ~8min | ✅ Pres.SV drift |
| Overloading mild | ❌ misses | ✅ WATCH ~15min | ✅ | ✅ Temp.SV drift |
| Sensor failure | ✅ DANGER | ✅ | ✅ | ✅ flatline |

### M8 Training Data Composition
```
TRAINING (model learns to reconstruct NORMAL — never sees faults):
  Real CIRA normal (M3): 9,711 clean windows → 80% train / 20% val
  Synthetic normal (M6 Type-A): 1200 sequences → windowed → ~24,000 windows
  Total normal training pool: ~33,000 windows

VALIDATION ONLY (faults appear ONLY here — never in training):
  Synthetic fault sequences (M6): 7200 fault sequences → windowed
  Purpose: calibrate threshold + fuzzy boundaries + measure TPR/FPR
  CRITICAL: mild severity (0.2–0.3) sequences used to calibrate
  rolling mean thresholds (0.085, 0.095) and slope detector (0.0003/window)
  WHY faults only in validation: LSTM-AE is anomaly detector, not classifier.
  Training on faults would teach model to reconstruct faults as normal.
```

### M8 Validation Gates (minimum 10)
```
GATE-M8-1:  TPR (fault detection)    > 90% on synthetic fault validation set
GATE-M8-2:  FPR (false alarm)        < 5%  on real normal validation set
GATE-M8-3:  Youden's J               > 0.85 (J = TPR - FPR)
GATE-M8-4:  Separation ratio         > 5.0x (M4 was 4.11x)
GATE-M8-5:  False alarms val         ≤ 8 windows
GATE-M8-6:  Fuzzy boundaries valid   lower_bound < upper_bound, both physical
GATE-M8-7:  Overloading detection    ≥ 80% (document if below — expected limitation)
GATE-M8-8:  Attention alignment      attention peaks at fault onset timesteps
GATE-M8-9:  Slow drift detection     seal_failure sev 0.2 detected in WATCH ≤ 20 min
GATE-M8-10: Per-channel monitor      Pres.SV drift flag fires before total MAE WARN
```

### Inference Protocol (Production)
```
1. Load cluster label → select cluster-conditional threshold
2. Run N=20 MC Dropout forward passes → mean MAE + uncertainty (std)
3. Compute fuzzy fault membership: μ_fault(mean_MAE)
4. Update rolling accumulator (5-window): sum of last 5 μ_fault scores
5. Update rolling mean MAE (200-window) and slope detector (500-window)
6. Update per-channel drift monitor (300-window Spearman)
7. Determine alert state: NORMAL / WATCH / WARN / DANGER
8. Output: {
     alert_state       : NORMAL/WATCH/WARN/DANGER
     anomaly_flag      : True/False (hard threshold — legacy compatibility)
     fuzzy_membership  : float [0,1]
     rolling_mean_mae  : float (200-window)
     mae_slope         : float (500-window trend)
     channel_drift     : dict {channel: bool} (per-channel flags)
     severity          : LOW/MEDIUM/HIGH (MC Dropout std zones)
     confidence        : float [0,1]
     attention_heatmap : array (50,) — which timesteps drove detection
   }
```

### Paste Text Keys
```
M8_val_loss               : [value]
M8_best_epoch             : [value]
M8_tpr_fault_detection    : [%, gate > 90%]
M8_fpr_false_alarm        : [%, gate < 5%]
M8_youden_j               : [value, gate > 0.85]
M8_separation_ratio       : [value, gate > 5.0x]
M8_fuzzy_lower_bound      : [value — P95 of normal MAE]
M8_fuzzy_upper_bound      : [value — P5 of fault MAE]
M8_rolling_watch_threshold: [calibrated value, target ~0.085]
M8_rolling_warn_threshold : [calibrated value, target ~0.095]
M8_slope_threshold        : [calibrated value, target ~0.0003/window]
M8_slow_drift_gate_pass   : [True/False — Gate M8-9]
M8_perchannel_gate_pass   : [True/False — Gate M8-10]
M8_overloading_tpr        : [%, document if < 80%]
M8_all_gates_pass         : [True/False]
M8_threshold_startup      : [value]
M8_threshold_steady_state : [value]
M8_threshold_high_load    : [value]
M8_threshold_cooldown     : [value]
Status for M9             : READY/BLOCKED
```

---

## ══════════════════════════════════════════════════
## M9 — PUMP SELECTOR + HOUSEHOLD ADVISOR
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

Physics-only calculation tools. No ML inference in this module.

```
INDUSTRIAL PUMP SELECTOR:
  Input  : Flow rate (m³/h), head (m), fluid properties
  Output : Required motor power, NPSH check, cavitation risk flag
  Physics: Bernoulli, affinity laws, NPSH margin calculation

HOUSEHOLD ADVISOR (advisory only):
  Input  : Usage scenario (domestic/agricultural)
  Output : Recommended flow rate, pipe sizing, motor sizing
  Rule   : if pump_type == "household": return physics_advisory_only()
  Label  : "Advisory guidance only — not a monitoring tool"
```

---

## ══════════════════════════════════════════════════
## M10 — FLASK WEB APPLICATION
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

### Routes
```
POST /api/anomaly_detect    ← CSV upload → LSTM-AE (M8) inference
POST /api/classify_fault    ← snapshot → XGBoost (M7) classification
POST /api/select_pump       ← form → Industrial pump selector (M9)
GET  /api/household         ← form → Household advisor (physics only)
POST /api/validate_model    ← M12 entry point
GET  /health                ← Docker/Hugging Face health check
```

### Alert State Display (v10.0 — Four-State System)
```
M8 outputs alert_state → M10 renders 4-zone condition indicator:
  🟢 NORMAL  (rolling_score < 2.0, no drift flags)
              "System operating within normal parameters"
  🟡 WATCH   (rolling mean rising OR slope trend OR channel drift flag)
              "Early anomaly trend detected — monitor closely"
              Shows: which channel drifting, trend duration, slope value
  🟠 WARN    (rolling mean > 0.095 OR rolling_score 2.0–3.5)
              "Sustained anomaly — schedule maintenance inspection"
              Shows: estimated time to DANGER at current trend rate
  🔴 DANGER  (single window MAE > threshold OR rolling_score > 3.5)
              "Fault confirmed — immediate maintenance action required"
              Shows: XGBoost fault classification + SHAP explanation

This is UX rendering of M8 four-state output — not independent logic.
Aligns with ISO 13374 condition monitoring alert levels.

WATCH state is the key addition vs v9.0 — this is the early warning
that catches slow drift faults days/weeks before DANGER fires.
```

### Mandatory Disclaimers (displayed before any industrial inference)
```
DISCLAIMER 1 (Model scope):
"This model is trained on CIRA SACIP dataset (1 specific installation).
 Sensor placement must follow ISO 13373 guidelines.
 r=0.9793 coupling between Mot.TV and Temp.SV is installation-specific.
 Model outputs are advisory — consult qualified engineer before action."

DISCLAIMER 2 (Sensor dependency):
"Inference quality depends entirely on sensor hardware integrity.
 Sensor malfunction or miscalibration will affect model output.
 Verify sensor health independently before acting on alerts."

DISCLAIMER 3 (Safety system boundary):
"PumpSmart is a condition monitoring tool (ISO 13374 Level 3).
 It is NOT a Safety Instrumented System (SIS) per IEC 61511.
 Hardwired process trips remain the primary safety barrier."
```

---

## ══════════════════════════════════════════════════
## M11 — DOCKER + HUGGING FACE DEPLOYMENT
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

```
Dockerfile:
  Base: python:3.11-slim
  Expose: port 7860 (Hugging Face default)
  CMD: gunicorn app:app --bind 0.0.0.0:7860

Model loading: ALL models map_location="cpu" at startup
XGBoost      : device="cpu" at inference
LSTM-AE      : MC Dropout N=20 on CPU — acceptable latency

GitHub Actions: push to main → build Docker → deploy to HF Spaces
```

---

## ══════════════════════════════════════════════════
## M12 — PHYSICS-GOVERNED MODEL VALIDATION SUITE
## Status: 🔲 NOT STARTED (Post-M11)
## ══════════════════════════════════════════════════

M12 closes the validation gap: M8 validates against M6 synthetic data
(which the model has already seen). M12 generates COMPLETELY FRESH,
parametrically different fault sequences using M5 physics engine —
configurations the model has NEVER been exposed to.

### Relationship to M6

| Dimension | M6 (Training) | M12 (Adversarial) |
|---|---|---|
| Purpose | Train M7/M8 | Stress-test M8 |
| Model exposure | Yes — seen in M8 val | Never — held out |
| Trigger | One-time | On-demand via Flask |
| Primary metric | Label distribution | Detection latency |

### Primary Metric: Detection Latency

```
For 110 kW, 2980 RPM, 40 bar pump:
  Bearing wear to failure: ~600–1800 seconds
  Maintenance response time: ~300 seconds minimum
  → Detection budget: 300 timesteps (30-window budget)
  → M12 gate: detection lag ≤ 60 timesteps (2× safety margin)
  → Cavitation gate: ≤ 30 timesteps (impeller pitting within 60–180s of onset)

SLOW DRIFT SPECIFIC GATE (v10.0):
  Seal failure at severity 0.2 (weeks-scale): WATCH state must fire ≤ 20 min
  Bearing wear at severity 0.25 (days-scale): WATCH state must fire ≤ 15 min
  These gates validate M8's trend accumulator on never-seen slow sequences
```

### Mandatory Test Configurations (14 minimum — 2 added in v10.0)
```
CONFIG 1-3  : Bearing wear at 3 severity levels (standard/high noise/subtle early)
CONFIG 4-5  : Cavitation startup (standard/high noise) — CRITICAL
CONFIG 6-7  : Seal failure (slow leak/fast degradation)
CONFIG 8    : Overloading (steady_state, standard)
CONFIG 9    : Impeller imbalance
CONFIG 10   : Sensor failure (sanity check)
CONFIG 11   : Bearing wear + cavitation simultaneously (multi-fault)
CONFIG 12   : Bearing wear with coupling_intact=False (broken r=0.9793)
CONFIG 13   : Seal failure severity 0.2 over 2000+ steps (slow drift WATCH gate) [NEW v10.0]
CONFIG 14   : Bearing wear severity 0.25 over 1500+ steps (slow drift WATCH gate) [NEW v10.0]
```

### Safety Gate
```
IF any gate fails → status: RECALIBRATION_REQUIRED
  Action: augment M6 with failing config → re-run M8 → re-run M12
IF all gates pass → status: PRODUCTION_VALIDATED
  Output: outputs/M12_validation_certificate.md
```

---

## MODULE PROGRESS TRACKER

```
M1   Data Ingestion & Cleaning          : ✅ COMPLETED (2026-03-25)
M2   EDA + Operating Mode Clustering    : ✅ COMPLETED (2026-03-26)
M3   Dimensionless Normalization        : ✅ COMPLETED (2026-03-28)
M4   LSTM-AE Baseline (v8)              : ✅ COMPLETED (2026-03-28)
M5   Physics Engine                     : ✅ COMPLETED (2026-03-29)
M6A  Synthetic Dataset Generator        : 🔲 ACTIVE — NEXT MODULE
M6.5 LSTM-AE Feature Extractor          : 🔲 NOT STARTED (after M6A)
M7   XGBoost Fault Classifier           : 🔲 NOT STARTED
M8   LSTM-AE v2 + Fuzzy Logic           : 🔲 NOT STARTED (safety mandate + slow drift arch locked above)
M9   Pump Selector + Household Advisor  : 🔲 NOT STARTED
M10  Flask Web Application              : 🔲 NOT STARTED
M11  Docker + Hugging Face Deployment   : 🔲 NOT STARTED
M12  Physics-Governed Validation Suite  : 🔲 NOT STARTED (post-M11)
```

---

## COMPLETE MODULE DEPENDENCY CHAIN

```
M1 → M2 → M3 → M4 → M5 → M6A → M6.5 → M7 → M8 → M9 → M10 → M11 → M12
                ↑              ↑         ↑              ↑
           spike seeds     physics    feature        production
           for M6+M12      bounds     matrix         thresholds
                           fault      (24-col        + fuzzy
                           rules      XGBoost        boundaries
                                      input)         for M12

M6A mild sequences (sev 0.2–0.3) → calibrate M8 trend accumulator thresholds
M6.5 outputs: feature_matrix.csv → M7 input
M8  outputs:  alert_state (4-level) → M10 UI + M12 validation
M12 reads:    M5 physics engine + M8 model + M4 config + M3 config
              Config 13+14 specifically test slow drift (M8 Gates 9+10)
```

---

## CROSS-MODULE INVARIANTS (enforced M1 → M12)

```
1.  segment_id preserved in ALL dataframes through M6
2.  Windows NEVER cross segment boundaries
3.  Normalization baselines LOCKED at M3_normalization_config.json
4.  Winsor ceilings LOCKED at M4_spike_config.json (M6 + M12 read, do not override)
5.  M4 threshold=0.110058 is the fault/normal boundary for M6 validation gate
6.  M8 cluster-conditional thresholds are the production boundary for M12
7.  Physical couplings (r>0.87) must hold in ALL synthetic sequences (M6 + M12)
8.  Conservation of energy + mass in all synthetic sequences
9.  Household pump → physics_advisory_only() always — no ML inference
10. XGBoost: device="cuda" train | device="cpu" deploy
11. All models: torch.save(state_dict) | torch.load(map_location="cpu")
12. M7 trains on M6.5 feature_matrix.csv — NOT raw sequences
13. Fuzzy logic is a core M8 detection component — not just M10 display
14. M12 MUST pass PRODUCTION_VALIDATED before deployment on 110 kW asset
15. M8 MUST detect slow drift faults (sev 0.2–0.3) via trend accumulator
    within 20 minutes of fault onset — this is the Category 3 liability gate
    M6A severity 0.2–0.3 sequences are the TRAINING DATA for this requirement
```

---

## FILE STRUCTURE

```
PumpSmart_Project/
├── config.py
├── data/
│   ├── raw/                           ← 9 original CSVs (never modified)
│   ├── clean/                         ← M1 output
│   ├── normalized/                    ← M3 output
│   └── synthetic/                     ← M4 spike seeds + M6 sequences + M6.5 features
├── models/                            ← .pth, .json model files
├── outputs/
│   ├── reports/
│   ├── plots/
│   └── M12_validation_log.csv
├── src/                               ← module_01 through module_12 scripts
└── app/                               ← Flask web app (M10)
```

---

*Document version: v10.0 | Supersedes: module_pathway_M1_to_M12_v9.md*
*Last updated: 2026-04-11 | Active module: M6A*
*Key additions in v10.0:*
*  - LIABILITY FRAMEWORK: three-category model responsibility boundary*
*  - M6A: Severity 0.2–0.3 mandatory — tied explicitly to trend accumulator training*
*  - M8: Three-mechanism slow drift detection (rolling mean + slope + per-channel)*
*  - M8: Four-state alert machine NORMAL/WATCH/WARN/DANGER (replaces binary flag)*
*  - M8: Two new validation gates (M8-9 slow drift, M8-10 per-channel monitor)*
*  - M10: WATCH state UI + three mandatory disclaimers*
*  - M12: Two new slow drift test configs (Config 13+14)*
*  - Cross-module invariant #15: slow drift detection mandate*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
