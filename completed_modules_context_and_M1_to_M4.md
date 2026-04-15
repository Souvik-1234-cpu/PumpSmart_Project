# PumpSmart — Completed Modules Reference: Context + M1 to M4
# Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring
# PART 1 OF 2 — Asset Context + LOCKED Foundations + M1–M4 Results
#
# Status: ALL SECTIONS IN THIS FILE ARE PERMANENTLY LOCKED
# Updated: 2026-04-15 | Author: Souvik
# Split from: completed_modules_M1_to_M6p5.md (original monolithic file)
#
# THIS FILE CONTAINS:
#   - Asset context + nameplate (LOCKED)
#   - Liability framework (NON-NEGOTIABLE)
#   - Dataset description + inviolable rules (LOCKED)
#   - Confirmed physical couplings + operational bounds (LOCKED)
#   - M4 winsorization bounds (LOCKED)
#   - M1 through M4 full results, outputs, paste keys
#
# COMPANION FILE: completed_modules_M5_to_M6p5r.md
#   → Contains M5, M6A, M6B, M6.5, M6.5r results + 6 audit findings
#   → Cross-module invariants 1–15
#   → File structure + module progress tracker

---

## ASSET CONTEXT — READ BEFORE EVERY MODULE

```
PUMP NAMEPLATE (110 kW INDUSTRIAL MULTISTAGE CENTRIFUGAL)
─────────────────────────────────────────────────────────────────
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
## (NON-NEGOTIABLE — governs M6A through M12)

```
THREE CATEGORIES OF FAILURE — ONLY ONE IS MODEL'S RESPONSIBILITY:

CATEGORY 1 — SENSOR HARDWARE FAILURE
  Scenario : A physical sensor malfunctions and feeds wrong values to model
  Liability : NOT the model's responsibility
  Defense   : sensor_failure fault class (M6B/M7) flags dead/drifting sensors
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
  Solution  : Three-layer temporal detection in M8 (see module_M8_lstm_ae_v2_architecture.md)
              Severity 0.2–0.3 sequences in M6B train the trend accumulator

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

## DATASET — CIRA SACIP

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
  Barometer      → Atmospheric pressure, mbar (DROPPED before M3)
  Temperature    → Ambient temperature, °C (used as T_ambient in M3)
```

---

## INVIOLABLE RULES (Apply M1 → M12 — NEVER VIOLATE)

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
  Baselines: M3_normalization_config.json (per-cluster, per-sensor) — LOCKED

SCOPE BOUNDARY (NEVER VIOLATE):
  if pump_type == "household": return physics_advisory_only()
  else: return ml_prediction()
```

---

## CONFIRMED PHYSICAL COUPLINGS (from M2 — LOCKED — must hold in ALL synthetic data)

```
X_ACR_Mot.TV ↔ X_Temp.SV     r = 0.9793  (motor heat → casing)
X_ACR_Pmp.PV ↔ X_ACR_Pmp.SV  r = 0.8882  (displacement → peak vibration)
X_ACR_Pmp.PV ↔ X_Pres.SV     r = 0.8779  (impeller displacement → pressure)

FAULT PROPAGATION (physics-causal — must be reproduced in M6B and M12):
  Bearing wear       : Mot.SV↑ → Mot.TV↑, Temp.SV↑, Pmp.SV↑  [lag 20-40 steps]
  Impeller imbalance : Pmp.PV↑ + Pmp.SV↑ → Pres.SV oscillates, Mot.PV↑
  Cavitation         : Pres.SV drops+erratic → Pmp.SV↑↑, Pmp.TV↑ [startup ONLY]
  Seal failure       : Pres.SV↓ progressive → Pmp.TV↑ [steady_state or high_load]
  Overloading        : Temp.SV↑ drift → Mot.TV↑, Mot.SV↑ [steady_state ONLY]
  Sensor failure     : Target channel only → flatline/spike; all 7 others stay normal

THERMAL DECOUPLING BY FAULT TYPE (from M5 validation — LOCKED):
  Bearing wear (steady_state) : r=0.972  ← coupling PRESERVED (heat source = bearing)
  Overloading (steady_state)  : r=0.997  ← coupling STRONGLY PRESERVED (thermal load)
  Seal failure (steady_state) : r=-0.013 ← coupling BROKEN (hydraulic fault, not thermal)
  Cavitation (startup)        : r=0.376  ← coupling WEAK (hydraulic, not thermal)
  Bearing wear (high_load)    : r=0.949  ← coupling preserved
  Normal (steady_state)       : r=-0.062 ← baseline near-zero (expected at steady-state)
```

---

## CONFIRMED OPERATIONAL BOUNDS (M2/M3 real data — LOCKED)

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

## M4 v8 WINSORIZATION BOUNDS (cluster-conditional — LOCKED for M6B + M12)

```
Channel           | Cluster      | Upper Multiplier | Physics Basis
───────────────────────────────────────────────────────────────────────────────
X_Pres.SV_norm   | startup      | 3.0x             | Joukowsky transient headroom
                 | steady_state | 5.6x             | Wide valid range (std=13 bar)
                 | high_load    | 2.0x             | Tight — faults caught immediately
                 | cooldown     | 3.0x             | Depressurization transients
X_ACR_Pmp.PV    | startup      | 3.2x             | ISO 13373-3: BPF harmonics
                 | all others   | 2.6x             | v8 behaviour preserved
X_ACR_Mot.SV    | all clusters | 6.7x             | Uniform — no cluster physics
X_ACR_Pmp.SV    | all clusters | 8.8x             | Uniform — broadband RMS spike
X_ACR_Mot.PV    | all clusters | 2.2x             | Uniform — displacement bounded
Source: M4_spike_config.json — DO NOT OVERRIDE in M6B or M12
```

---

## ╔══════════════════════════════════════════════════╗
## M1 — DATA INGESTION & HARD CLEANING
## Status: ✅ COMPLETED (2026-03-25)
## ╚══════════════════════════════════════════════════╝

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

### Engineering Notes
```
Pump_C_Day3: excluded entirely — >50% null across multiple columns
Barometer column: dropped globally before M3 (not used in ML)
Temperature column: kept — used as T_ambient in M3 normalization
Gap detection threshold: gap > 2s = new segment boundary
```

### Outputs
```
data/clean/Pump_*_clean.csv
data/clean/segment_registry.csv
outputs/M1_file_summary.csv
outputs/plots/M1_null_heatmap.png
outputs/plots/M1_segment_timeline.png
outputs/reports/module_01_cleaning_report.md
```

### Paste Text Keys (LOCKED)
```
M1_raw_rows        : 173,730
M1_clean_rows      : 147,217
M1_drop_pct        : 15.26%
M1_segments        : 66 total → 25 usable
M1_worst_null_col  : Barometer (Pump_B_Day3: 65.46%)
M1_sampling        : 1s uniform, all files
```

---

## ╔══════════════════════════════════════════════════╗
## M2 — EDA + OPERATING MODE CLUSTERING
## Status: ✅ COMPLETED (2026-03-26)
## ╚══════════════════════════════════════════════════╝

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

### Top Correlations Confirmed
```
Mot.TV ↔ Temp.SV    : r=0.9793  (thermal coupling — dominant pair)
Pmp.PV ↔ Pmp.SV    : r=0.8882  (mechanical vibration coupling)
Pmp.PV ↔ Pres.SV   : r=0.8779  (impeller → pressure coupling)
All ADF tests       : p<0.05 — all 8 channels stationary
Window size optimal : 50s — captures one full operational event
```

### Outputs
```
outputs/M2_cluster_bounds.csv
outputs/M2_labelled_data.csv
outputs/M2_cluster_bounds_units.json
outputs/plots/M2_pca_clusters.png
outputs/plots/M2_silhouette.png
outputs/plots/M2_correlation_heatmap.png
outputs/plots/M2_cluster_profiles.png
outputs/plots/M2_window_size_selection.png
outputs/reports/module_02_eda_clustering_report.md
```

### Paste Text Keys (LOCKED)
```
M2_rows            : 117,970
M2_optimal_k       : 4
M2_silhouette      : 0.5458
M2_window_size     : 50
M2_stationarity    : 8/8 ADF pass
M2_top_correlation : Mot.TV↔Temp.SV r=0.9793
M2_pca_pc1         : 47.37%
M2_pca_pc2         : 32.97%
```

---

## ╔══════════════════════════════════════════════════╗
## M3 — DIMENSIONLESS NORMALIZATION
## Status: ✅ COMPLETED (2026-03-28)
## ╚══════════════════════════════════════════════════╝

### Key Results

| Metric | Value |
|---|---|
| Normalised rows | 117,970 |
| Clusters used | 4 |
| Channels normalised | 8 |
| Range issues | None |
| Config file | models/M3_normalization_config.json — LOCKED |

### Normalization Formulas (LOCKED — never change)
```
Pressure   : P*  = P_actual / P_cluster_mean
Vibration  : a*  = a_actual / a_cluster_mean
Temperature: ΔT* = (T - T_cluster_min) / (T_cluster_max - T_cluster_min)

T_ambient source: T_cluster_min (cluster-relative, NOT hardcoded ambient)
Reason: climate-agnostic — valid across field deployments in different regions

Small negatives in temperature (flash evaporative cooling in cooldown):
  PRESERVED, not clipped — physically real phenomenon
  Clipping would introduce false floor → distort thermal fault signatures

Normal operation → normalized values in [0, 1.0]
Fault → drift above 1.0 or anomalous temporal pattern
```

### Outputs
```
data/normalized/normalised_data.csv       ← 117,970 rows, 8 normalized channels + cluster_id
models/M3_normalization_config.json       ← LOCKED baselines (per-cluster, per-sensor)
outputs/reports/module_03_normalization_report.md
outputs/plots/M3_normalized_distributions.png
```

### Paste Text Keys (LOCKED)
```
M3_rows_normalized : 117,970
M3_clusters        : 4
M3_channels        : 8
M3_range_issues    : None
M3_config_file     : models/M3_normalization_config.json (LOCKED)
```

---

## ╔══════════════════════════════════════════════════╗
## M4 — LSTM-AE PATTERN EXTRACTION (NORMAL BASELINE)
## Status: ✅ COMPLETED v8 (2026-03-28)
## ╚══════════════════════════════════════════════════╝

### Architecture (LOCKED)
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

### Key Results (LOCKED)

| Metric | Value |
|---|---|
| Clean windows (normal) | 9,711 |
| Train / Val split | 8,254 / 1,457 |
| Best val loss | 0.026862 |
| Best epoch | 141 / 150 |
| Mean MAE (val) | 0.026765 |
| Anomaly threshold | **0.110058** (mean + 3σ ∪ P99) — LOCKED |
| Separation ratio | 4.11x |
| False alarms (val) | 8 (0.55%) |
| Spike seeds extracted | 1,044 → M4_spike_seeds.npy |
| Spike rows excluded | 12,620 |

### ⚠️ THRESHOLD 0.110058 IS LOCKED — DO NOT MODIFY
```
This threshold is the fault/normal boundary for:
  - M6B physics invariant #7 (fault sequences must exceed it)
  - M6B physics invariant #8 (normal sequences must not exceed it)
  - M6B physics invariant #9 (mild sev 0.2–0.3 must be in [0.110058, 0.140])
  - M6.5r Gate W3 validation
  - M8 cluster-conditional threshold calibration baseline
Any change to this value invalidates the entire M6B dataset.
```

### Spike Seed Fault Hints (LOCKED for M6B + M12)

| Fault Hint | Count | M6B Fault Mapping |
|---|---|---|
| mechanical_transient | 472 | bearing_wear, impeller_imbalance |
| pressure_transient | 408 | cavitation, seal_failure |
| impeller_cavitation | 113 | cavitation (direct) |
| bearing_impact | 44 | bearing_wear (direct) |
| pressure_spike_highload | 7 | overloading |

### Channel Weight Rationale (M4 — informs M8 design)
```
Mot.SV=2.0, Pmp.SV=2.0, Pres.SV=2.0 : primary mechanical fault channels
Mot.PV=1.5, Pmp.PV=1.5               : displacement — correlated with SV
Temp.SV=1.0                           : thermal — secondary signal
Mot.TV=0.8, Pmp.TV=0.8               : accelerometer temp — placement-dependent

CONSEQUENCE FOR M8:
  Low weight on Temp.SV (1.0) and TV channels (0.8) means overloading
  (temperature-dominant fault) has LOW weighted MAE even at moderate severity.
  M4 threshold 0.110058 was calibrated on NORMAL data only.
  Overloading at sev <0.5 likely stays sub-threshold → M8 must compensate
  with per-channel drift monitor on Temp.SV (Mechanism C).
  [This was confirmed by M6.5 audit: overloading Gate 3 pass = 0.00%]
```

### Outputs
```
models/lstm_ae_baseline_best.pth
data/synthetic/M4_spike_seeds.npy        ← shape=(1044, 50, 8) → M6B + M12 input
data/synthetic/M4_spike_seeds_meta.csv
data/synthetic/M4_spike_config.json      ← cluster-conditional winsor bounds (LOCKED)
outputs/M4_threshold_config.json         ← threshold=0.110058, separation=4.11x
outputs/reports/module_04_lstm_ae_baseline_report.md
```

### Paste Text Keys (LOCKED)
```
M4_clean_windows   : 9,711
M4_train_val       : 8254 / 1457
M4_val_loss        : 0.026862
M4_best_epoch      : 141
M4_mean_mae        : 0.026765
M4_threshold       : 0.110058  ← LOCKED — do not change
M4_separation      : 4.11x
M4_false_alarms    : 8 (0.55%)
M4_spike_seeds     : 1,044
M4_spike_excluded  : 12,620
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic file `completed_modules_M1_to_M6p5.md` created |
| v2.0 | 2026-04-15 | **SPLIT into Part 1 (this file) + Part 2**. Part 1 = LOCKED context + M1–M4. Original monolithic file converted to redirect stub. Liability framework updated: M6A references → M6B. Spike seed mapping updated: M6A → M6B. |

---

*GitHub is the ONLY source of truth for this spec.*
*Companion: `completed_modules_M5_to_M6p5r.md` (Part 2 — M5 through M6.5r + invariants + tracker)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
