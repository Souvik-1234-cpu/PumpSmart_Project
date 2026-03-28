# PumpSmart — Module Pathway v7.0
# Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring
# Full Pipeline: M1 → M8 (Unified Reference Document)
# Updated: 2026-03-28 | Author: Souvik | Version: 7.0

---

## ⚠️ ASSET CONTEXT — READ BEFORE EVERY MODULE

```
PUMP NAMEPLATE (110 kW INDUSTRIAL MULTISTAGE CENTRIFUGAL)
─────────────────────────────────────────────────────────
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

## DATASET

```
Source  : CIRA (Italian Aerospace Research Centre), SACIP project
          Zenodo record 15301820 → https://zenodo.org/records/15301820
Pumps   : A, B, C (multistage centrifugal, industrial)
Files   : 9 CSVs — 3 pumps × 3 operational days
Size    : 26.3 MB total (985 KB to 6.1 MB per file)
Sampling: 1 second uniform (all 9 files confirmed)

Columns (11 total):
  Timestamp       → YYYY-MM-DD hh:mm:ss
  X_ACR_Mot.PV   → Motor casing displacement, mm (ISO 10816-3)
  X_ACR_Mot.SV   → Motor casing vibration velocity, mm/s (ISO 10816-3)
  X_ACR_Mot.TV   → Motor casing surface temperature, °C (IEC 60034-1)
  X_ACR_Pmp.PV   → Pump casing displacement, mm (ISO 10816-3)
  X_ACR_Pmp.SV   → Pump casing vibration velocity, mm/s (ISO 10816-3)
  X_ACR_Pmp.TV   → Pump casing surface temperature, °C (IEC 60034-1)
  X_Temp.SV      → Process fluid / bearing temperature, °C (ISO 13373-2)
  X_Pres.SV      → Pump discharge pressure, bar (ISO 5167)
  Barometer      → Atmospheric pressure, mbar (dropped before M3)
  Temperature    → Ambient temperature, °C (dropped before M3)
```

---

## INVIOLABLE RULES (Apply M1 → M11)

```
TIME SERIES INTEGRITY:
  NEVER concatenate raw CSVs before M1 cleaning + segmentation
  NEVER create windows crossing segment boundaries
  segment_id preserved in ALL downstream dataframes
  Windows generated per segment only:
    for seg_id, seg_df in df.groupby('segment_id'): [window here]
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
  if pump_type == 'household': return physics_advisory_only()
  else: return ml_prediction()
  Reason: household monoblock ≠ industrial multistage = OOD inference = safety risk
```

---

## CONFIRMED PHYSICAL COUPLINGS (from M2 — must hold in ALL synthetic data)

```
X_ACR_Mot.TV ↔ X_Temp.SV     r = 0.9793  (motor heat → casing)
X_ACR_Pmp.PV ↔ X_ACR_Pmp.SV  r = 0.8882  (displacement → peak vibration)
X_ACR_Pmp.PV ↔ X_Pres.SV     r = 0.8779  (impeller displacement → pressure)

FAULT PROPAGATION (physics-causal, must be reproduced in M6 synthetic data):
  Bearing wear       : Mot.SV↑ → Mot.TV↑, Temp.SV↑, Pmp.SV↑
  Impeller imbalance : Pmp.PV↑ + Pmp.SV↑ → Pres.SV oscillates, Mot.PV↑
  Cavitation         : Pres.SV drops+erratic → Pmp.SV↑↑, Pmp.TV↑
  Seal failure       : Pres.SV↓ progressive → Pmp.TV↑
  Overloading        : Temp.SV↑ drift → Mot.TV↑, Mot.SV↑
  Sensor failure     : Target channel only → flatline/spike; others normal
```

---

## CONFIRMED OPERATIONAL BOUNDS (M2/M3 real data — use for M5 physics validation)

```
Cluster distribution (117,970 rows):
  Startup      : 49,884 rows (42.3%) ← DOMINANT — highest fault risk zone
  Cooldown     : 26,851 rows (22.8%)
  High-load    : 26,600 rows (22.5%)
  Steady-state : 14,635 rows (12.4%)

Outlet Pressure (bar):
  Startup      : 0.43 – 0.85    Cooldown     : 0.45 – 44.4
  Steady-state : 0.69 – 43.3    High-load    : 37.4 – 44.8
  → Cavitation risk: Startup (P ≈ 0.43–0.85 bar) — low NPSH zone
  → Transient max observed: 46.7 bar (Joukowsky water hammer)

Motor Vibration SV (mm/s):
  Startup      : 0.39 – 0.58    Cooldown     : 0.42 – 12.1
  Steady-state : 12.0 – 21.5    High-load    : 22.2 – 77.7
  → ISO 10816-3 alarm: High-load P97.5 = 77.7 mm/s
  → Absolute max transient spike: 456.6 mm/s (anomalous)

Pump Vibration SV (mm/s):
  Startup      : 0.38 – 0.57    Cooldown     : 0.41 – 0.61
  Steady-state : 24.5 – 55.3    High-load    : 19.6 – 34.7
  → Steady-state outlier max: 291.6 mm/s

Motor Temperature (°C):
  Startup      : 30.6 – 53.8    Cooldown     : 18.8 – 30.7
  Steady-state : 19.7 – 47.8    High-load    : 23.3 – 40.6
```

---

## M4 v8 WINSORIZATION BOUNDS (cluster-conditional — physics-correct)

```
These bounds are LOCKED for M6 synthetic data generation.
M6 must read from M4_spike_config.json — DO NOT override.

Channel           | Cluster      | Upper Multiplier | Physics Basis
─────────────────────────────────────────────────────────────────────
X_Pres.SV_norm   | startup      | 3.0x             | Joukowsky transient headroom
                 | steady_state | 5.6x             | std=13 bar — wide valid range
                 | high_load    | 2.0x             | Tight — faults caught immediately
                 | cooldown     | 3.0x             | Depressurization transients
X_ACR_Pmp.PV    | startup      | 3.2x             | ISO 13373-3: BPF harmonics 2-4x
                 | all others   | 2.6x             | v7 behaviour preserved
X_ACR_Mot.SV    | all clusters | 6.7x             | Uniform — no cluster physics
X_ACR_Pmp.SV    | all clusters | 8.8x             | Uniform — broadband RMS spike
X_ACR_Mot.PV    | all clusters | 2.2x             | Uniform — displacement bounded
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
| Usable segments (≥70 rows) | 27 → 25 (after C_Day3 exclusion) |
| Worst null column | Barometer (Pump_B_Day3: 65.46%) |
| Sampling interval | 1s uniform, all files |

### Per-File Breakdown

| File | Raw | Clean | Drop% | Segments | Usable | Worst Null |
|---|---|---|---|---|---|---|
| Pump_A_Day1 | 4,981 | 4,859 | 2.45% | 11 | 3 | X_Temp.SV 1.53% |
| Pump_A_Day2 | 21,429 | 19,164 | 10.57% | 7 | 4 | Temperature 10.32% |
| Pump_A_Day3 | 21,600 | 21,592 | 0.04% | 3 | 2 | X_ACR_Pmp.PV 0.03% |
| Pump_B_Day1 | 4,981 | 4,636 | 6.93% | 19 | 4 | X_Temp.SV 3.45% |
| Pump_B_Day2 | 35,829 | 33,956 | 5.23% | 13 | 5 | X_ACR_Pmp.TV 5.02% |
| Pump_B_Day3 | 29,700 | 10,187 | 65.70% | 3 | 2 | Barometer 65.46% |
| Pump_C_Day1 | 4,981 | 4,871 | 2.21% | 5 | 3 | X_Temp.SV 1.55% |
| Pump_C_Day2 | 21,429 | 19,218 | 10.32% | 2 | 2 | Temperature 10.32% |
| Pump_C_Day3 | 28,800 | 28,734 | 0.23% | 3 | 2 | X_ACR_Pmp.PV 0.19% |

### Engineering Decisions

- **Pump_B_Day3**: Barometer + Temperature sensor failure — 65.5% rows dropped.
  Pump ran continuously; sensor logged NaN, not timestamp gaps.
  Clean segments before/after fault block preserved as valid training data.
- **C_Day3**: Excluded entirely post-M1 (100% Barometer corruption in linked files).
- **Day1 files**: Gap threshold = 8s (5× median, above natural 1s jitter)
- **Day2/Day3 files**: Gap threshold = 2s (continuous 1s data)
- **Warmup rows**: 300 standard; 600 for post-sensor-dropout segments

### Outputs
```
data/clean/Pump_*_clean.csv          ← 9 files, segment_id preserved
data/clean/segment_registry.csv      ← 66 segments, usability flag
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
| Usable segments | 25 |
| Optimal K (elbow + silhouette) | 4 |
| Silhouette score | 0.5458 (strong separation) |
| Optimal window size | 50s |
| Stationary sensors (ADF test) | 8/8 |
| Top correlation | Mot.TV ↔ Temp.SV: r=0.9793 |
| PCA variance captured | PC1=47.37%, PC2=32.97% (80.3% total) |

### Operating Mode Map

| Cluster ID | Mode | Rows | % | Centroid Characteristics |
|---|---|---|---|---|
| C0 | cooldown | 26,851 | 22.8% | Low vib (0.88 mm/s), low pressure (8.3 bar), low temp |
| C2 | startup | 49,884 | 42.3% | Very low pressure (0.62 bar), HIGH temp (41.9°C thermal lag) |
| C1 | steady_state | 14,635 | 12.4% | Moderate vib (16.1 mm/s), high stable pressure (35.8 bar) |
| C3 | high_load | 26,600 | 22.5% | High vib (36.3 mm/s), highest pressure (42.0 bar) |

> **Physics note**: Startup has higher mean TV (41.9°C) than high_load (39.5°C) despite lower hydraulic load.
> Correct physics: 7-stage pump has motor thermal run-in before hydraulics are fully loaded.
> Affinity law → low flow at startup = low shaft power, but motor already at thermal steady-state from prior cycle.

### ADF Stationarity Results

All 8 channels stationary: Mot.PV (p=0.0), Mot.SV (p=0.0), Mot.TV (p=0.00044),
Pmp.PV (p=0.0), Pmp.SV (p=0.0), Pmp.TV (p=1e-06), Temp.SV (p=0.000857), Pres.SV (p=2e-06)

### Audit Fixes (2026-03-28)

| Issue | Fix |
|---|---|
| No unit documentation for cluster bounds | Created M2_cluster_bounds_units.json |
| Time-series Y-axes raw column names, no units | Regenerated with unit labels |
| STEP 8 comment incorrectly linked high temp → high_load | Patched in source script |

### Outputs
```
outputs/M2_cluster_bounds.csv         ← normalization baselines for M3
outputs/M2_labelled_data.csv          ← full dataset with cluster labels
outputs/M2_cluster_bounds_units.json  ← unit registry
outputs/plots/M2_kmeans_selection.png
outputs/plots/M2_cluster_pca.png
outputs/plots/M2_cluster_centroids.png
outputs/plots/M2_timeseries_clusters.png
outputs/plots/M2_correlation_matrix.png
```

---

## ══════════════════════════════════════════════════
## M3 — DIMENSIONLESS FEATURE ENGINEERING (NORMALIZATION)
## Status: ✅ COMPLETED (2026-03-28)
## ══════════════════════════════════════════════════

### Key Results

| Metric | Value |
|---|---|
| Normalised rows | 117,970 |
| Clusters used | 4 |
| Channels normalised | 8 |
| Range issues | None |
| Small negatives (Pmp.TV, Temp.SV) | Preserved — flash evaporative cooling in cooldown |

### Normalization Formulas (LOCKED — do not change downstream)

```
Pressure  : P*  = P_actual / P_cluster_mean
Vibration : a*  = a_actual / a_cluster_mean
Temperature: ΔT* = (T - T_cluster_min) / (T_cluster_max - T_cluster_min)

Key decision: Temperature uses cluster min/max (NOT ambient-relative).
Reason: Climate-agnostic — ambient temperature varies across field deployments.
Cluster min is the coldest observed state = physical zero reference.
```

### Architecture Decision: Why Cluster-Relative Normalization

Raw sensor values span wildly different scales across operating modes:
- Startup: Pres.SV ≈ 0.62 bar | High-load: Pres.SV ≈ 42 bar
- A single global σ treats these identically → destroys fault signal in startup
- Cluster-relative normalization → normal operation always lands at ≈1.0
- Fault → drift above 1.0 regardless of operating mode
- This makes the LSTM-AE threshold mode-independent at inference time

### Outputs
```
data/normalized/normalised_data.csv   ← 117,970 rows, 8 norm channels + segment_id + operating_mode
M3_normalization_config.json          ← per-cluster per-sensor baselines (locked)
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
Decoder  : LSTM(64→128, layers=2) — hidden state seeded from encoder
Output   : Linear(128→8) → reconstruction
Parameters: 505,096
Loss     : 0.6×MAE + 0.4×MSE (physics-weighted)
Channel weights: Mot.SV=2.0, Pmp.SV=2.0, Pres.SV=2.0, Mot.PV=1.5,
                 Pmp.PV=1.5, Temp.SV=1.0, Mot.TV=0.8, Pmp.TV=0.8
Optimizer: AdamW | LR: CosineAnnealingWarmRestarts (T0=20)
AMP      : GradScaler + autocast (CUDA)
```

### v8 Physics Fixes (vs v7)

| Fix | Change | Physics Basis |
|---|---|---|
| FIX-1 | Pmp.PV startup ceiling: 2.6x → **3.2x** | ISO 13373-3: BPF harmonics = 2–4x steady-state during ramp-up |
| FIX-2 | Pres.SV cluster-conditional: startup=**3.0x**, hl=**2.0x** | Joukowsky: ΔP=ρcΔv — startup denominator (0.621 bar) makes global σ meaningless |
| FIX-3 | pressure_transient spike ratio reference → **high_load mean** | Water hammer energy governed by operating pressure (~42 bar), not startup (~0.62 bar) |
| FIX-4 | Per-cluster winsor bounds saved to **M4_spike_config.json** | M6 reads exact physics-correct ceilings — no downstream compensation |

### Key Results

| Metric | Value |
|---|---|
| Clean windows (normal) | 9,711 |
| Train / Val split | 8,254 / 1,457 |
| Best val loss | **0.026862** (physics-weighted) |
| Best epoch | 141 / 150 |
| Mean MAE (val) | 0.026765 |
| Anomaly threshold | **0.110058** (mean + 3σ ∪ P99) |
| Separation ratio | **4.11x** |
| False alarms (val) | 8 (0.55% — well under 1% gate) |
| Spike rows excluded | 12,620 |
| Spike seeds extracted | **1,044** → M4_spike_seeds.npy |
| Peak VRAM | 0.20 GB / 8 GB |
| Training time | 51.1s |
| Overfit triggered | False |

### Spike Seed Bank (→ M6 input)

| Fault Hint | Windows | Primary Channel Signal |
|---|---|---|
| mechanical_transient | 472 | Multi-channel simultaneous spike |
| pressure_transient | 408 | Pres.SV during startup/cooldown |
| impeller_cavitation | 113 | Pmp.SV dominant |
| bearing_impact | 44 | Mot.SV dominant |
| pressure_spike_high_load | 7 | Tight 2.0x ceiling correctly isolated |

### Validation Gates (10/10 PASS)

```
GATE1  no_overfit              : PASS (gap = -0.0009)
GATE2  mae_lt_006              : PASS (0.0268 << 0.06)
GATE3  threshold_range         : PASS (0.05 < 0.110 < 0.3)
GATE4  separation_gt3          : PASS (4.11x)
GATE5  false_alarms_lt1pct     : PASS (0.55%)
GATE6  tv_channels_ok          : PASS (Mot.TV=0.0171, Pmp.TV=0.0149)
GATE7  spike_seeds_saved       : PASS (1,044 windows)
GATE8  val_loss_lt_005         : PASS (0.0269)
GATE9  pmpPV_startup_3.2x      : PASS (FIX-1 verified)
GATE10 pres_cluster_ordered    : PASS (startup<hl confirmed)
```

### Known Architectural Flags (Deferred to M8 — see SAFETY MANDATE below)

```
FLAG-1: Phase-lag on mode transitions (Pmp.PV MAE=0.079, Mot.PV MAE=0.079)
        Root cause: LSTM hidden state temporal smoothing
        Impact on M4: NONE — conservative reconstruction reduces false alarms
                      on startup→steady-state transitions (highest fault risk zone)
        Impact on production: ADDRESSED IN M8 (see mandate below)

FLAG-2: Single threshold across all clusters
        Root cause: M4 is a baseline model, not production
        Impact on M4: 8 false alarms on 1,457 val windows — acceptable
        Impact on production: ADDRESSED IN M8
```

### Per-Channel MAE (Val)

| Channel | MAE | Note |
|---|---|---|
| Mot.PV | 0.0584 | Step-transition lag (see FLAG-1) |
| Pmp.PV | 0.0466 | Step-transition lag (see FLAG-1) |
| Mot.SV | 0.0308 | Good tracking |
| Pmp.SV | 0.0205 | Good tracking |
| Mot.TV | 0.0171 | Excellent — slow thermal dynamics |
| Temp.SV | 0.0160 | Excellent |
| Pmp.TV | 0.0149 | Excellent |
| Pres.SV | 0.0093 | Best — smooth after cluster-conditional winsor |

### Outputs
```
models/lstm_ae_baseline_best.pth      ← best checkpoint (map_location='cpu' for deploy)
models/lstm_ae_baseline_final.pth     ← final epoch checkpoint
models/lstm_ae_baseline_meta.json     ← architecture + training config (v8)
data/synthetic/M4_spike_seeds.npy     ← shape=(1044, 50, 8) → M6 input
data/synthetic/M4_spike_seeds_meta.csv ← fault hints + operating modes
data/synthetic/M4_spike_config.json   ← cluster-conditional winsor bounds (locked for M6)
outputs/M4_threshold_config.json      ← threshold=0.110058, separation=4.11x
outputs/plots/M4_training_curve.png
outputs/plots/M4_error_distribution.png
outputs/plots/M4_per_channel_mae.png
outputs/plots/M4_spike_seeds_distribution.png
outputs/plots/M4_reconstruction_sample.png
outputs/plots/M4_v8_cluster_winsor_bounds.png
outputs/reports/module_04_lstm_ae_baseline_report.md
```

---

## ══════════════════════════════════════════════════
## M5 — PHYSICS ENGINE
## Status: 🔲 NOT STARTED (Next active module)
## ══════════════════════════════════════════════════

### Objective
Implement all physics equations for the 110 kW, 7-stage, 40 bar centrifugal pump.
Validate every equation against nameplate + M2/M3 confirmed operational bounds.
Output: physics validation pass/fail table, fault envelope boundaries for M6.

### Equations to Implement and Validate

```
1. Hydraulic Power       : P_hyd = ρgQH/η  → expect ~55 kW at design point
2. Affinity Laws         : Q∝N, H∝N², P∝N³ (verify with cluster speed proxy)
3. NPSH Available        : NPSHa = (P_inlet - P_vapour) / (ρg)
4. NPSH Required         : NPSHr = nameplate (validate cavitation zone = startup cluster)
5. Joukowsky Water Hammer: ΔP = ρcΔv (validate 15–25 bar surge in 40 bar system)
6. BEP Efficiency        : η = P_hyd / P_shaft (validate against 110 kW shaft input)
7. Vibration ISO 10816-3 : alarm thresholds per cluster (validate vs M2 bounds)
8. Thermal Rise          : ΔT = P_loss / (ṁ × Cp) (validate vs M3 Temp.SV range)
9. Per-Stage Head        : H_stage = H_total / n_stages = 450/7 = 64.3 m per stage
10. Slip Factor          : σ = 1 - (π/Z) (7-blade impeller → σ ≈ 0.55)
```

### Validation Criteria

- ALL equations must validate against nameplate (110 kW, 45 m³/h, 450 m, 40 bar, 7 impellers, 2980 RPM)
- NPSHa < NPSHr must be confirmed for startup cluster (P=0.43–0.85 bar)
- Joukowsky ΔP must confirm water hammer can reach 46.7 bar (observed transient max)
- Conservation of energy + mass must hold in all sequences
- No unphysical values in any output (negative pressure, T below ambient, etc.)

### Paste Text Keys (fill after M5 run)
```
M5_equations_implemented  : [count]
M5_nameplate_pass_count   : [N]/10
M5_npsh_cavitation_zone   : CONFIRMED/FAILED
M5_joukowsky_max_bar       : [value]
M5_bep_efficiency_pct      : [value]
M5_physics_violations      : [list or NONE]
Status for M6              : READY/BLOCKED
```

---

## ══════════════════════════════════════════════════
## M6 — SYNTHETIC DATASET GENERATOR
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

### Objective
Generate physics-causal synthetic fault sequences in normalized space.
Seed from M4 spike windows. Validate against M5 physics bounds.

### Sequence Targets

| Type | Count | Description |
|---|---|---|
| Type A (Normal) | ~200 | Clean normal operation across all 4 clusters |
| Type B (Progressive Fault) | ~300 | Physics-governed onset + causal cascade |
| Type C (Sensor Dropout) | ~160 | Single channel; pump continues normally |
| **Total** | **~660** | Labeled sequences for M7 XGBoost training |

### Fault Types (M6 must implement all 6)

```
bearing_wear       : Mot.SV↑ → Mot.TV↑, Temp.SV↑, Pmp.SV↑
impeller_imbalance : Pmp.PV↑ + Pmp.SV↑ → Pres.SV oscillates, Mot.PV↑
cavitation         : Pres.SV drops+erratic → Pmp.SV↑↑, Pmp.TV↑
seal_failure       : Pres.SV↓ progressive → Pmp.TV↑
overloading        : Temp.SV↑ drift → Mot.TV↑, Mot.SV↑
sensor_failure     : Target channel only → flatline/spike; all others normal
```

### M6 MUST READ FROM M4 (Locked Invariants)

```python
# M6 must load and respect these files — NO overriding:
M4_spike_seeds.npy          # 1,044 real spike windows as generation seeds
M4_spike_seeds_meta.csv     # fault_hint per window
M4_spike_config.json        # cluster-conditional winsor ceilings (FIX-4)
M4_threshold_config.json    # threshold=0.110058 — fault sequences must exceed this

# Fault injection rule:
# Synthetic fault windows must produce MAE > 0.110058 when passed through M4 model
# Normal sequences must produce MAE < 0.110058
# This is the physics-ML consistency gate for M6
```

### Paste Text Keys
```
M6_total_sequences        : [count]
M6_type_a_count           : [count]
M6_type_b_count           : [count]
M6_type_c_count           : [count]
M6_sanity_check_pass_rate : [%]
M6_label_distribution     : {fault_type: count}
M6_physics_violations     : [list or NONE]
Status for M7             : READY/BLOCKED
```

---

## ══════════════════════════════════════════════════
## M7 — STATIC FAULT CLASSIFIER (XGBoost)
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

### Objective
Train XGBoost on M6 synthetic + real spike windows to classify fault type.
SHAP values for explainability. device='cuda' train, device='cpu' deploy.

### Architecture

```
Input     : Single snapshot (8 normalized features) or windowed statistics
            (mean, std, max, skew per channel = 32 features)
Model     : XGBoost (device='cuda' training, device='cpu' deployment)
Output    : Fault class (6 types + normal)
Explainer : SHAP TreeExplainer (top-3 features per prediction)
```

### Requirements

```
Accuracy             : >85% overall
Per-class F1 cavitation: >0.80 (safety-critical — 110 kW pump at startup risk)
Per-class F1 bearing   : >0.75
SHAP top-3 features    : must be physically causal per fault type
False negative on bearing/cavitation: < 5% (high-consequence faults)
```

### Paste Text Keys
```
M7_accuracy           : [%]
M7_f1_cavitation      : [value]
M7_f1_bearing         : [value]
M7_f1_per_class       : {fault: f1}
M7_top3_shap_cavitation: [features]
M7_top3_shap_bearing  : [features]
Status for M8         : READY/BLOCKED
```

---

## ══════════════════════════════════════════════════
## M8 — LSTM-AE v2 PRODUCTION MODEL
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

---

## 🚨 M8 SAFETY MANDATE — HIGH-VALUE ASSET PROTECTION 🚨

```
╔══════════════════════════════════════════════════════════════════════╗
║  ASSET: 110 kW, 7-stage, 40 bar, 450m head multistage pump          ║
║  FAILURE CONSEQUENCE: Process shutdown + secondary damage + injury   ║
║  THIS MANDATE IS NON-NEGOTIABLE. EVERY ITEM BELOW IS REQUIRED.      ║
╚══════════════════════════════════════════════════════════════════════╝

WHY M4 IS INSUFFICIENT FOR PRODUCTION ON THIS ASSET:

  Issue 1 — Phase-lag on mode transitions (FLAG-1 from M4 audit):
    The vanilla LSTM-AE smooths over abrupt step transitions in Pmp.PV
    and Mot.PV (MAE ~0.079 on normal transition windows). While this is
    conservative for M4 baseline purposes, it means the model adapts
    slowly when the pump enters HIGH-RISK transition zones (startup →
    steady-state). On a 110 kW, 40 bar pump, a bearing impact or
    cavitation onset during this transition zone is the MOST DANGEROUS
    scenario. The production model MUST track these transitions precisely.

  Issue 2 — Single global threshold:
    M4 uses threshold=0.110058 for ALL operating modes. Startup cluster
    has Pres.SV std=1.03 (normalized), high_load has std=0.046.
    A single threshold means the model is simultaneously too sensitive
    (false alarms in startup) and not sensitive enough (misses early
    faults in tight high_load regime). For a 110 kW asset, this is
    not acceptable in production.

  Issue 3 — No reconstruction confidence:
    M4 outputs a binary flag (normal/anomaly). Maintenance engineers on
    a 110 kW pump need severity grading to prioritize response.
    A false-negative with no confidence score = undetected catastrophic
    failure. Production model must output uncertainty.

  Issue 4 — No temporal attention on fault timesteps:
    A 3-timestep bearing impact event is diluted across 50 timesteps
    equally. On a 2980 RPM, 7-impeller pump with BPF = 349 Hz,
    a bearing race defect produces high-energy impulses at precise
    intervals. The model MUST weight these anomalous timesteps higher
    than surrounding normal operation.

MANDATORY M8 ARCHITECTURE REQUIREMENTS:

  ✅ REQUIRED-1: Temporal Attention
     Implement multi-head attention over LSTM encoder outputs.
     Attention scores must be logged per inference for explainability.
     This directly resolves FLAG-1 (phase-lag) and Issue 4 (impulse dilution).

  ✅ REQUIRED-2: Gradient Penalty in Loss Function
     Add temporal gradient penalty term to loss:
       grad_penalty = mean(|diff(recon) - diff(target)|)
       total_loss   = 0.5×MAE + 0.3×MSE + 0.2×grad_penalty
     This forces the model to track RATE OF CHANGE, not just absolute value.
     Critical for catching progressive fault onset on bearing wear + seal failure.

  ✅ REQUIRED-3: Cluster-Conditional Thresholds
     Train separate threshold calibration per operating mode:
       startup threshold      > steady_state threshold
       high_load threshold   < startup threshold (tighter — pressure stable)
     Use M4_spike_config.json cluster distributions as threshold priors.

  ✅ REQUIRED-4: Uncertainty Quantification
     Implement MC Dropout (dropout active at inference, N=20 forward passes).
     Output: mean reconstruction error + std across passes → confidence interval.
     Severity grading: LOW (<μ+1σ), MEDIUM (μ+1σ to μ+2σ), HIGH (>μ+2σ).

  ✅ REQUIRED-5: Validation Against M4 Baseline
     M8 must improve on M4 in ALL of:
       - Phase-transition MAE (Pmp.PV, Mot.PV) must reduce by >20%
       - False alarm count on val set must reduce or hold at ≤8
       - Separation ratio must improve from 4.11x to >5.0x
       - Gate count: minimum 12 gates (M4 had 10)

  ✅ REQUIRED-6: Production Inference Protocol
     All inference must follow:
       1. Load cluster label from M3 config → select cluster threshold
       2. Run N=20 MC Dropout forward passes
       3. Compute mean MAE + uncertainty
       4. Apply cluster-conditional threshold
       5. Output: {anomaly_flag, severity, confidence, attention_heatmap}

VALIDATION GATES (minimum — M8 must define more):
  GATE-M8-1  : phase_transition_MAE_reduction > 20% vs M4
  GATE-M8-2  : separation_ratio > 5.0x
  GATE-M8-3  : false_alarms_val ≤ 8 (M4 baseline)
  GATE-M8-4  : uncertainty_coverage_80pct (80% of normal windows within μ±σ)
  GATE-M8-5  : attention_heads_physically_causal
               (attention peaks must align with known fault timesteps)
  GATE-M8-6  : cluster_threshold_ordered
               (startup_thr > high_load_thr confirmed)
  GATE-M8-7  : grad_penalty_reduces_phase_lag (verified on val set)
  GATE-M8-8  : mc_dropout_calibrated (reliability diagram R² > 0.85)

This mandate was written because M4 FLAGS 1 and 2 were consciously deferred
here. They are NOT optional improvements — they are REQUIRED for safe
deployment on a 110 kW industrial asset where model underperformance
= equipment destruction and safety risk.
```

### Paste Text Keys
```
M8_architecture           : LSTM-AE + Attention + MC Dropout
M8_val_loss               : [value]
M8_best_epoch             : [value]
M8_separation_ratio       : [value, must be >5.0x]
M8_phase_lag_improvement  : [% vs M4, must be >20%]
M8_false_alarms_val       : [count, must be ≤8]
M8_uncertainty_coverage   : [%, must be >80%]
M8_all_gates_pass         : [True/False]
M8_threshold_startup      : [value]
M8_threshold_steady_state : [value]
M8_threshold_high_load    : [value]
M8_threshold_cooldown     : [value]
Status for M9             : READY/BLOCKED
```

---

## MODULE PROGRESS TRACKER

```
M1  Data Ingestion & Cleaning          : ✅ COMPLETED (2026-03-25)
M2  EDA + Operating Mode Clustering    : ✅ COMPLETED (2026-03-26)
M3  Dimensionless Normalization        : ✅ COMPLETED (2026-03-28)
M4  LSTM-AE Baseline (v8)              : ✅ COMPLETED (2026-03-28)
M5  Physics Engine                     : 🔲 ACTIVE — next module
M6  Synthetic Dataset Generator        : 🔲 NOT STARTED
M7  XGBoost Fault Classifier           : 🔲 NOT STARTED
M8  LSTM-AE v2 Production              : 🔲 NOT STARTED (safety mandate locked above)
M9  Pump Selector + Household Advisor  : 🔲 NOT STARTED
M10 Flask Web Application              : 🔲 NOT STARTED
M11 Docker + Hugging Face Deployment   : 🔲 NOT STARTED
```

---

## CROSS-MODULE INVARIANTS (enforced M1 → M11)

```
1. segment_id preserved in ALL dataframes through M6
2. Windows NEVER cross segment boundaries
3. Normalization baselines LOCKED at M3_normalization_config.json
4. Winsor ceilings LOCKED at M4_spike_config.json (M6 reads, does not override)
5. M4 threshold=0.110058 is the fault/normal boundary for M6 validation
6. Physical couplings (r>0.87) must hold in ALL synthetic sequences
7. Conservation of energy + mass in all synthetic sequences
8. Household pump → physics_advisory_only() always — no ML inference
9. XGBoost: device='cuda' train | device='cpu' deploy
10. All models: torch.save(state_dict) | torch.load(map_location='cpu')
```

---

## FILE STRUCTURE

```
PumpSmart_Project/
├── config.py                          ← DEVICE, all DIR paths
├── data/
│   ├── raw/                           ← 9 original CSVs (never modified)
│   ├── clean/                         ← M1 output (9 CSVs + segment_registry)
│   ├── normalized/                    ← M3 output (normalised_data.csv)
│   └── synthetic/                     ← M6 output + M4 spike seeds
├── models/                            ← .pth, .pkl, .json model files
├── outputs/
│   ├── reports/                       ← markdown report per module
│   └── plots/                         ← all PNG charts
├── src/                               ← module_01 through module_11 scripts
└── app/                               ← Flask web app (M10)
```

---

*Document version: v7.0 | Last updated: 2026-03-28 | Next active module: M5*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
