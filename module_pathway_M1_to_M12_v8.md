# PumpSmart — Module Pathway v8.0
# Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring
# Full Pipeline: M1 → M12 (Unified Reference Document)
# Updated: 2026-03-29 | Author: Souvik | Version: 8.0
# Supersedes: module_pathway_M1_to_M8_v7.md

---

## ⚠️ ASSET CONTEXT — READ BEFORE EVERY MODULE

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

## INVIOLABLE RULES (Apply M1 → M12)

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

FAULT PROPAGATION (physics-causal, must be reproduced in M6 and M12 synthetic data):
  Bearing wear       : Mot.SV↑ → Mot.TV↑, Temp.SV↑, Pmp.SV↑
  Impeller imbalance : Pmp.PV↑ + Pmp.SV↑ → Pres.SV oscillates, Mot.PV↑
  Cavitation         : Pres.SV drops+erratic → Pmp.SV↑↑, Pmp.TV↑
  Seal failure       : Pres.SV↓ progressive → Pmp.TV↑
  Overloading        : Temp.SV↑ drift → Mot.TV↑, Mot.SV↑
  Sensor failure     : Target channel only → flatline/spike; others normal
```

---

## CONFIRMED OPERATIONAL BOUNDS (M2/M3 real data — use for M5 and M12 physics validation)

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
These bounds are LOCKED for M6 and M12 synthetic data generation.
Both M6 and M12 must read from M4_spike_config.json — DO NOT override.

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

### Normalization Formulas (LOCKED)

```
Pressure  : P*  = P_actual / P_cluster_mean
Vibration : a*  = a_actual / a_cluster_mean
Temperature: ΔT* = (T - T_cluster_min) / (T_cluster_max - T_cluster_min)

Key decision: Temperature uses cluster min/max (NOT ambient-relative).
Reason: Climate-agnostic — ambient temperature varies across field deployments.
Cluster min is the coldest observed state = physical zero reference.
```

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
| False alarms (val) | 8 (0.55%) |
| Spike rows excluded | 12,620 |
| Spike seeds extracted | **1,044** → M4_spike_seeds.npy |

### Outputs
```
models/lstm_ae_baseline_best.pth
data/synthetic/M4_spike_seeds.npy        ← shape=(1044, 50, 8) → M6 + M12 input
data/synthetic/M4_spike_seeds_meta.csv
data/synthetic/M4_spike_config.json      ← cluster-conditional winsor bounds (locked)
outputs/M4_threshold_config.json         ← threshold=0.110058, separation=4.11x
```

---

## ══════════════════════════════════════════════════
## M5 — PHYSICS ENGINE
## Status: 🔲 NOT STARTED (Next active module)
## ══════════════════════════════════════════════════

### Objective
Implement all physics equations for the 110 kW, 7-stage, 40 bar centrifugal pump.
Validate every equation against nameplate + M2/M3 confirmed operational bounds.
Output: physics validation pass/fail table, fault envelope boundaries for M6 AND M12.

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

### Note: M5 is also OWNER of unit_registry.json
```
models/unit_registry.json — generated here (deferred from M3).
Reason: fault physics equations (m/s, m/s², bar, °C) require
        ISO alarm metadata not available until M5 is written.
M12 ALSO reads this file for physical plausibility enforcement
during adversarial sequence generation.
```

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
These sequences are TRAINING DATA for M7 and VALIDATION DATA for M8.
They are NOT the adversarial test suite — that role belongs to M12.

### Sequence Targets

| Type | Count | Description |
|---|---|---|
| Type A (Normal) | ~200 | Clean normal operation across all 4 clusters |
| Type B (Progressive Fault) | ~300 | Physics-governed onset + causal cascade |
| Type C (Sensor Dropout) | ~160 | Single channel; pump continues normally |
| **Total** | **~660** | Labeled sequences for M7 XGBoost training |

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

MANDATORY M8 ARCHITECTURE REQUIREMENTS:

  ✅ REQUIRED-1: Temporal Attention
     Multi-head attention over LSTM encoder outputs.
     Attention scores logged per inference for explainability.
     Resolves FLAG-1 (phase-lag) and impulse-dilution at BPF=349 Hz.

  ✅ REQUIRED-2: Gradient Penalty in Loss Function
     Add temporal gradient penalty:
       grad_penalty = mean(|diff(recon) - diff(target)|)
       total_loss   = 0.5×MAE + 0.3×MSE + 0.2×grad_penalty
     Forces model to track rate-of-change, not just absolute value.

  ✅ REQUIRED-3: Cluster-Conditional Thresholds
     Separate threshold per operating mode:
       startup threshold > steady_state threshold
       high_load threshold < startup threshold (tighter)
     Use M4_spike_config.json cluster distributions as priors.

  ✅ REQUIRED-4: Uncertainty Quantification (MC Dropout)
     MC Dropout active at inference (N=20 forward passes).
     Output: mean error + std → confidence interval.
     Severity: LOW (<μ+1σ), MEDIUM (μ+1σ to μ+2σ), HIGH (>μ+2σ).

  ✅ REQUIRED-5: Validation Against M4 Baseline
     M8 must improve on M4 in ALL of:
       - Phase-transition MAE (Pmp.PV, Mot.PV): reduce by >20%
       - False alarms: ≤8 on val set
       - Separation ratio: >5.0x (M4 was 4.11x)
       - Gate count: minimum 12 gates

  ✅ REQUIRED-6: Production Inference Protocol
     1. Load cluster label → select cluster threshold
     2. Run N=20 MC Dropout forward passes
     3. Compute mean MAE + uncertainty
     4. Apply cluster-conditional threshold
     5. Output: {anomaly_flag, severity, confidence, attention_heatmap}
```

### M8 Validation Gates (minimum)

```
GATE-M8-1  : phase_transition_MAE_reduction > 20% vs M4
GATE-M8-2  : separation_ratio > 5.0x
GATE-M8-3  : false_alarms_val ≤ 8
GATE-M8-4  : uncertainty_coverage_80pct
GATE-M8-5  : attention_heads_physically_causal
GATE-M8-6  : cluster_threshold_ordered (startup_thr > high_load_thr)
GATE-M8-7  : grad_penalty_reduces_phase_lag
GATE-M8-8  : mc_dropout_calibrated (reliability diagram R² > 0.85)
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
Status for M12            : READY after M11 deployment
```

### Outputs
```
models/lstm_ae_v2_best.pth
models/lstm_ae_v2_meta.json
models/M8_threshold_config.json        ← cluster-conditional thresholds
outputs/plots/M8_loss_curves.png
outputs/plots/M8_tpr_fpr_curve.png
outputs/plots/M8_per_fault_mae.png
outputs/plots/M8_normal_vs_fault_mae.png
outputs/reports/module_08_lstm_ae_v2_report.md
```

---

## ══════════════════════════════════════════════════
## M9 — PUMP SELECTOR + HOUSEHOLD ADVISOR
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

### Objective
Physics-only calculation tools. No ML inference in this module.

```
INDUSTRIAL PUMP SELECTOR:
  Input  : Flow rate (m³/h), head (m), fluid properties
  Output : Required motor power, NPSH check, cavitation risk flag,
           recommended pump type
  Physics: Bernoulli equation, affinity laws, NPSH margin calculation

HOUSEHOLD ADVISOR (advisory only — labelled clearly in UI):
  Input  : Usage scenario (domestic/agricultural)
  Output : Recommended flow rate, pipe sizing, motor sizing estimate
  Basis  : Physics engine ONLY — NO ML
  Label  : "Advisory guidance only — not a monitoring tool"
  Rule   : if pump_type == 'household': return physics_advisory_only()
```

### Outputs
```
src/pump_selector.py
src/household_advisor.py
outputs/reports/module_09_selector_report.md
```

---

## ══════════════════════════════════════════════════
## M10 — FLASK WEB APPLICATION
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

### Objective
Full-stack Flask app integrating all models and tools.

### Routes

```
POST /api/anomaly_detect    ← CSV upload → LSTM-AE (M8) inference
POST /api/classify_fault    ← snapshot → XGBoost (M7) classification
POST /api/select_pump       ← form → Industrial pump selector (M9)
GET  /api/household         ← form → Household advisor (M9, physics only)
POST /api/validate_model    ← M12 entry point: fault config → physics test
GET  /health                ← health check (for Docker/Hugging Face)
```

### Mandatory UI Elements

```
DISCLAIMER (must appear before ANY industrial inference):
  "This model is trained on CIRA SACIP dataset (1 specific installation).
   Sensor placement must follow ISO 13373 guidelines.
   r=0.9793 coupling between Mot.TV and Temp.SV is installation-specific.
   Model outputs are advisory — consult qualified engineer before action."

HOUSEHOLD ADVISOR label:
  "Advisory guidance only — no fault prediction or condition monitoring"
```

### Outputs
```
app/app.py
app/templates/
app/static/
outputs/reports/module_10_flask_report.md
```

---

## ══════════════════════════════════════════════════
## M11 — DOCKER + HUGGING FACE DEPLOYMENT
## Status: 🔲 NOT STARTED
## ══════════════════════════════════════════════════

### Objective
Containerize Flask app and deploy to Hugging Face Spaces.

```
Dockerfile:
  Base: python:3.11-slim
  Expose: port 7860 (Hugging Face default)
  CMD: gunicorn app:app --bind 0.0.0.0:7860

Model loading rule:
  ALL models loaded with map_location='cpu' at startup
  XGBoost: device='cpu' at inference (GPU not available on HF Spaces)
  LSTM-AE: MC Dropout N=20 passes on CPU — acceptable latency for advisory tool

GitHub Actions CI/CD:
  On push to main → build Docker → push to Hugging Face Spaces
```

### Paste Text Keys
```
M11_docker_build        : PASS/FAIL
M11_deployment_url      : [Hugging Face Spaces URL]
M11_health_check        : PASS/FAIL
M11_model_load_time_s   : [value]
M11_inference_latency_s : [value, LSTM-AE CPU N=20]
Status for M12          : READY
```

---

## ══════════════════════════════════════════════════
## M12 — PHYSICS-GOVERNED MODEL VALIDATION SUITE
## Status: 🔲 NOT STARTED (Post-M11)
## ══════════════════════════════════════════════════

---

## 🔬 M12 DESIGN RATIONALE — WHY THIS MODULE IS NON-NEGOTIABLE

```
╔══════════════════════════════════════════════════════════════════════╗
║  VALIDATION GAP THAT M12 CLOSES:                                     ║
║                                                                      ║
║  M8 validates against M6 synthetic data — but the model has already  ║
║  seen the M6 fault library during training (M8 val set includes M6). ║
║  M6 uses fixed severity bands and parameter ranges.                  ║
║                                                                      ║
║  Critical risk: The model may have learned M6's specific parameter   ║
║  choices, not the underlying physics. It could pass M8 validation    ║
║  but fail on novel fault scenarios the model has never encountered.  ║
║                                                                      ║
║  For a 110 kW, 40 bar industrial pump:                               ║
║  An undetected fault = catastrophic failure, secondary damage,       ║
║  process shutdown. There must be ZERO GAP between model testing      ║
║  and real-world deployment edge cases.                               ║
║                                                                      ║
║  M12 closes this gap by generating COMPLETELY FRESH, PARAMETRICALLY  ║
║  DIFFERENT fault sequences using the SAME physics engine (M5)        ║
║  but configurations the model has NEVER been exposed to.             ║
║                                                                      ║
║  Ground truth is PERFECT: you control every physical parameter.      ║
║  No labeling ambiguity. No real-world uncontrolled conditions.       ║
║  This is a physics-governed digital test bench.                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Relationship to M6 — Key Distinctions

| Dimension | M6 (Training Synthetic) | M12 (Adversarial Validation) |
|---|---|---|
| **Purpose** | Give M7/M8 fault examples to learn from | Stress-test M8 against unseen scenarios |
| **Model exposure** | Yes — seen in M8 val split | **Never** — completely held out |
| **Fault parameter control** | Fixed severity bands | Fully parametric per run |
| **Trigger** | One-time during M6 module | **On-demand via Flask app** |
| **Sequence count** | ~660 fixed | Unlimited — generated live |
| **Ground truth** | Known at generation time | Known by construction — user sets every param |
| **Primary metric** | Label distribution, coupling fidelity | **Detection latency** (timesteps from onset to alarm) |
| **Safety gate** | M6 physics sanity checks | Model recalibration flag if detection lag > limit |

### M12 Architecture

```
INPUTS (user-configurable via Flask /api/validate_model route):
  fault_type       : [bearing_wear | impeller_imbalance | cavitation |
                      seal_failure | overloading | sensor_failure]
  starting_cluster : [startup | steady_state | high_load | cooldown]
  fault_onset_t    : timestep index where fault begins (e.g., t=35 of 200)
  severity_lambda  : exponential growth rate λ for bearing wear (0.005–0.020)
  severity_linear  : linear slope K for seal/overloading (0.002–0.015)
  thermal_lag_tau  : timestep lag between primary and thermal channels (10–60)
  noise_sigma      : Gaussian noise level on all channels (0.01–0.08)
  coupling_intact  : [True | False] — test model when physical coupling breaks
  multi_fault      : optional second fault type (advanced testing)

PHYSICS ENGINE CALL (M5 physics_engine.py):
  Generates sequence of shape (200, 8) in normalized space
  Using exact parameters above
  Enforces: conservation laws, no negative pressure, cluster-relative bounds
  Reads: M3_normalization_config.json, M4_spike_config.json, models/unit_registry.json

MODEL INFERENCE (M8 lstm_ae_v2_best.pth):
  Rolling 50-step windows across the 200-step sequence
  At each step: {anomaly_flag, severity, confidence, attention_heatmap}
  Alarm timestep = first window where anomaly_flag = True

PRIMARY METRICS:
  Detection Latency = alarm_timestep - fault_onset_t
    → Ideal: < 10 timesteps (10 seconds at 1Hz sampling)
    → Acceptable: 10–30 timesteps
    → Concerning: 31–60 timesteps — flag for investigation
    → FAIL: > 60 timesteps — safety gate breach → recalibration mandated
  
  Detection Rate = fraction of M12 runs where fault was detected at all
    → Gate: > 95% detection across all fault types at active severity
  
  False Alarm Rate = fraction of M12 normal sequences triggering alarm
    → Gate: < 5% (consistent with M8 FPR gate)

  Attention Alignment Score:
    Pearson r between attention heatmap and ground-truth fault onset mask
    → Gate: r > 0.6 (model attention must peak near injected fault timesteps)
    → Confirms model is detecting the physics, not artefacts
```

### Why Detection Latency Is the Primary Metric

```
For a 110 kW, 2980 RPM, 7-stage centrifugal pump at 40 bar:

Bearing wear progression rate (Paris law analogy):
  a(t) = a_0 × exp(λt)   λ ≈ 0.01/s at active severity
  From first detectable vibration to bearing failure: ~600–1800 seconds
  From M8 alarm to maintenance response: minimum 300 seconds (operator reaction)

Detection latency budget:
  Total time to failure: 600–1800s
  Maintenance response time: 300s
  → Maximum acceptable detection latency: 600 - 300 = 300 seconds = 300 timesteps
  → At 10 timesteps per window (1Hz): budget = 30 windows
  → M12 safety gate: detection lag ≤ 60 timesteps (2× safety margin on 30-window budget)

Cavitation (most time-critical fault):
  Implosion shocks cause impeller pitting within 60–180 seconds of onset
  Maintenance response: 300s minimum
  → Cavitation MUST be detected within 30 timesteps from onset
  → M12 applies tighter 30-timestep gate for cavitation specifically

This engineering basis makes M12 the FINAL SAFETY BARRIER between
a validated model and a deployed model on a high-value industrial asset.
It is not optional validation hygiene — it is a physical safety requirement.
```

### M12 Test Protocol

```
MANDATORY TEST SUITE (minimum 12 configurations per run):
  CONFIG 1  : Bearing wear, high_load, λ=0.008, τ=30, σ=0.02 — standard
  CONFIG 2  : Bearing wear, startup, λ=0.015, τ=20, σ=0.05 — high noise
  CONFIG 3  : Bearing wear, steady_state, λ=0.005, τ=45 — subtle early fault
  CONFIG 4  : Cavitation, startup, standard parameters — CRITICAL
  CONFIG 5  : Cavitation, startup, high noise — CRITICAL stress test
  CONFIG 6  : Seal failure, high_load, K=0.004 — progressive slow leak
  CONFIG 7  : Seal failure, steady_state, K=0.008 — fast seal degradation
  CONFIG 8  : Overloading, steady_state, K=0.006 — standard
  CONFIG 9  : Impeller imbalance, steady_state, λ_imb=0.003 — standard
  CONFIG 10 : Sensor failure (flatline), any cluster — easy sanity check
  CONFIG 11 : Bearing wear + cavitation simultaneously — multi-fault test
  CONFIG 12 : Bearing wear, coupling_intact=False — model robustness to
              broken r=0.9793 coupling (sensor relocation scenario)

ADDITIONAL USER-DEFINED CONFIGS:
  Engineer can configure any parametric combination via the Flask UI.
  Results logged to outputs/M12_validation_log.csv with timestamp.
  Each run: config params + detection_latency + detection_rate + attention_score.
```

### Safety Recalibration Gate

```
IF ANY of the following are triggered across the mandatory 12 configs:
  - Detection lag > 60 timesteps for bearing_wear at active severity
  - Detection lag > 30 timesteps for cavitation at any severity
  - Detection rate < 95% for any fault type at active severity
  - Attention alignment score < 0.6 on >3 configs
  - False alarm rate > 5% on normal sequences

THEN:
  M12 outputs status: RECALIBRATION_REQUIRED
  Action: Generate additional M6-style synthetic sequences with
          the failing configuration parameters → augment M8 training set
          → re-run M8 training → re-run M11 deployment → re-run M12
  This loop is closed BEFORE the model is considered production-safe.

IF ALL gates pass:
  M12 outputs status: PRODUCTION_VALIDATED
  Generates: outputs/M12_validation_certificate.md
  Contents: all 12 config results, detection latency per fault,
            attention scores, safety gate status
  This certificate is the final deliverable confirming model safety
  for deployment on the 110 kW asset.
```

### Paste Text Keys
```
M12_configs_run             : [count]
M12_bearing_detection_lag   : [timesteps, gate ≤60]
M12_cavitation_detection_lag: [timesteps, gate ≤30]
M12_overall_detection_rate  : [%, gate >95%]
M12_false_alarm_rate        : [%, gate <5%]
M12_attention_alignment     : [r value, gate >0.6]
M12_multi_fault_detected    : [True/False]
M12_broken_coupling_robust  : [True/False]
M12_safety_gate_status      : PRODUCTION_VALIDATED / RECALIBRATION_REQUIRED
M12_certificate_file        : outputs/M12_validation_certificate.md
```

### Outputs
```
src/module_12_validation_suite.py     ← physics engine caller + M8 inference loop
outputs/M12_validation_log.csv        ← all run results (timestamped)
outputs/plots/M12_detection_latency_per_fault.png
outputs/plots/M12_attention_heatmaps.png
outputs/plots/M12_severity_vs_latency.png
outputs/M12_validation_certificate.md ← PRODUCTION_VALIDATED or RECALIBRATION_REQUIRED
outputs/reports/module_12_validation_report.md
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
M12 Physics-Governed Validation Suite  : 🔲 NOT STARTED (post-M11)
```

---

## COMPLETE MODULE DEPENDENCY CHAIN

```
M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10 → M11 → M12
                 ↑              ↑              ↑              ↑
            spike seeds     physics bounds  production    adversarial
            for M6+M12      for M6+M12      thresholds    test bench
                            unit_registry   for M12       (closes all
                                                           validation gaps)

M12 reads from: M5 (physics_engine.py + unit_registry.json)
                M8 (lstm_ae_v2_best.pth + M8_threshold_config.json)
                M4 (M4_spike_config.json + M4_threshold_config.json)
                M3 (M3_normalization_config.json)
M12 writes to : outputs/M12_validation_log.csv
                outputs/M12_validation_certificate.md
```

---

## CROSS-MODULE INVARIANTS (enforced M1 → M12)

```
1.  segment_id preserved in ALL dataframes through M6
2.  Windows NEVER cross segment boundaries
3.  Normalization baselines LOCKED at M3_normalization_config.json
4.  Winsor ceilings LOCKED at M4_spike_config.json (M6 + M12 read, do not override)
5.  M4 threshold=0.110058 is the fault/normal boundary for M6 validation
6.  M8 cluster-conditional thresholds are the production boundary for M12
7.  Physical couplings (r>0.87) must hold in ALL synthetic sequences (M6 + M12)
8.  Conservation of energy + mass in all synthetic sequences
9.  Household pump → physics_advisory_only() always — no ML inference
10. XGBoost: device='cuda' train | device='cpu' deploy
11. All models: torch.save(state_dict) | torch.load(map_location='cpu')
12. M12 MUST pass PRODUCTION_VALIDATED before model is considered deployment-safe
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
│   ├── plots/                         ← all PNG charts
│   └── M12_validation_log.csv         ← M12 adversarial test results
├── src/                               ← module_01 through module_12 scripts
└── app/                               ← Flask web app (M10)
```

---

*Document version: v8.0 | Supersedes: module_pathway_M1_to_M8_v7.md*
*Last updated: 2026-03-29 | Active module: M5*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Key addition in v8: M12 Physics-Governed Model Validation Suite*
