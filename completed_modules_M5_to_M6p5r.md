# PumpSmart — Completed Modules Reference: M5 to M6.5r
# Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring
# PART 2 OF 2 — M5 through M6.5r Results + Invariants + File Structure + Tracker
#
# Status: M5, M6A, M6.5 sections LOCKED. M6B and M6.5r sections ACTIVE (latest results here).
# Updated: 2026-04-15 | Author: Souvik
# Split from: completed_modules_M1_to_M6p5.md (original monolithic file)
#
# THIS FILE CONTAINS:
#   - M5 physics engine results + paste keys (LOCKED)
#   - M6A synthetic generator results + paste keys (LOCKED)
#   - M6B expanded synthetic dataset results + paste keys (ACTIVE — 21-class, Groups A-E)
#   - M6.5 original LSTM-AE feature extractor results (LOCKED)
#   - M6.5 6 critical audit findings in FULL detail (LOCKED — govern M7 and M8)
#   - M6.5r updated spec (ACTIVE — 26-feature matrix, fault_rules_v3.json, 21 gates)
#   - Cross-module invariants 1–15 (updated for M6B)
#   - File structure (updated for M6B outputs)
#   - Module progress tracker
#
# COMPANION FILE: completed_modules_context_and_M1_to_M4.md
#   → Asset context, liability framework, dataset description, inviolable rules
#   → Physical couplings, operational bounds, M4 winsorization bounds
#   → M1 through M4 full results and paste keys (ALL LOCKED)

---

## ╔══════════════════════════════════════════════════╗
## M5 — PHYSICS ENGINE
## Status: ✅ COMPLETED (2026-03-29)
## ╚══════════════════════════════════════════════════╝

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
| Overloading severity range | [0.5, 1.0] — mild overloading handled via sub-cluster in M6B |

### Thermal Coupling Validation (M5 Plots Confirmed — LOCKED)

| Fault / Cluster | r(Mot.TV*, Temp.SV*) | Physics Interpretation |
|---|---|---|
| Bearing wear (steady_state) | 0.972 | Coupling PRESERVED — heat from bearing |
| Overloading (steady_state) | 0.997 | STRONGLY PRESERVED — thermal overload |
| Seal failure (steady_state) | -0.013 | Coupling BROKEN — hydraulic fault |
| Cavitation (startup) | 0.376 | WEAK — hydraulic, not thermal |
| Bearing wear (high_load) | 0.949 | Preserved |
| Normal (steady_state) | -0.062 | Baseline near-zero (correct) |

### 20 Physics Equations Validated Against Nameplate
```
All 20 equations validated at: 110 kW, 45 m³/h, 450m head, 40 bar, 7 impellers, 2980 RPM
Key equations include:
  - Hydraulic power: P_hyd = ρgQH/η
  - Specific speed: Ns = N√Q / H^(3/4)
  - NPSH available: NPSHa = (Ps - Pv)/(ρg) + Vs²/(2g)
  - Affinity laws: Q∝N, H∝N², P∝N³
  - Joukowsky pressure surge: ΔP = ρ×a×ΔV
  - ISO 10816-3 vibration severity zones
  - Bearing heat generation: Q_bear = μ×F×v
  - Thermal rise: dT/dt = Q_bear / (m×Cp) [Euler integration]
```

### Outputs
```
src/module_05_physics_engine.py
models/fault_rules.json              ← 6 fault types (M5 original) — LOCKED
models/fault_rules_v3.json           ← 21 fault labels Groups A–E (M6B/M6.5r) — LOCKED
models/M5_physics_config.json
models/unit_registry.json
outputs/plots/M5_fault_signatures.png
outputs/plots/M5_thermal_coupling.png
outputs/reports/module_05_physics_engine_report.md
```

### Paste Text Keys (LOCKED)
```
M5_equations       : 20
M5_nameplate_pass  : 20/20
M5_s5_cases        : 19/19 PASS
M5_s7_faults       : 24/26 PASS
M5_known_failures  : bearing_wear_steady_state_s100, overloading_steady_state_s140
M5_patch3          : Euler integration bearing heat
M5_patch5          : ±5σ sensor failure spike guard
M5_overload_range  : [0.5, 1.0] production (mild via M6B sub-cluster)
```

---

## ╔══════════════════════════════════════════════════╗
## M6A — SYNTHETIC DATASET GENERATOR (HYBRID) — SUPERSEDED
## Status: ✅ COMPLETED (2026-04-11) — ⚠️ SUPERSEDED BY M6B
## ╚══════════════════════════════════════════════════╝

> ⚠️ M6A produced 7 classes, 8400 sequences. It has been SUPERSEDED by M6B
> (21 classes, Groups A–E, ~27,000 sequences). M6B is the dataset used for
> M7, M8, and all downstream modules. M6A results are LOCKED for reference only.

### M6A Architecture Decision: HYBRID PATH C (LOCKED)

```
SOURCE 1: Real CIRA Normal Windows (M3 normalized pool) → 1200 Type-A normal sequences
SOURCE 2: M4 Spike Seeds (pseudo-labelled real CIRA fault windows)
          → cosine similarity > 0.85 → fault onset seeds t=0 to t=49
          → M5 physics continues t=50 to t=199
SOURCE 3: Physics Synthetic (M5 engine — pure)
          → fills gaps, covers full severity spectrum [0.2 → 1.0]
```

### M6A Sequence Count (LOCKED — reference only)

| Class | Count |
|---|---|
| normal | 1200 |
| bearing_wear | 1200 |
| impeller_imbalance | 1200 |
| cavitation | 1200 |
| seal_failure | 1200 |
| overloading | 1200 |
| sensor_failure | 1200 |
| **TOTAL** | **8400** |

### M6A Outputs (LOCKED — archived, not used downstream)
```
data/synthetic/M6A_sequences.pkl         ← 8400 sequences, shape each (200, 8)
data/synthetic/M6A_sequence_meta.csv     ← seq_id, label, severity, source, cluster, seed_idx
data/synthetic/M6A_validation_report.json
outputs/reports/module_06a_synthetic_report.md
```

### Paste Text Keys (LOCKED — reference only)
```
M6A_total_sequences       : 8400
M6A_sequences_per_class   : 1200
M6A_classes               : 7 (labels 0–6)
M6A_status                : SUPERSEDED by M6B — not used in M7/M8 downstream
```

---

## ╔══════════════════════════════════════════════════╗
## M6B — EXPANDED SYNTHETIC DATASET (21-CLASS, GROUPS A–E)
## Status: ✅ COMPLETED (2026-04-14) — ACTIVE — USED IN M7, M8, M12
## ╚══════════════════════════════════════════════════╝

### Why M6B Was Created

```
M6A had 7 classes (labels 0–6) — adequate for basic fault detection but insufficient for:
  1. Compound fault scenarios (two faults active simultaneously)
  2. Primary-channel masked faults (sensor dead, secondary path only)
  3. Fault severity variants (fast vs slow, intermittent vs sustained)
  4. Multi-sensor anomaly scenarios (2 sensors degraded simultaneously)
  5. M7 XGBoost training on M6A alone = limited real-world coverage
  6. M8 adversarial validation requires compound + masked + variant scenarios

M6B adds Groups B, C, D, E to the base Group A (M6A classes).
All 21 classes defined in fault_rules_v3.json — LOCKED.
```

### 21-Class Label Map (fault_rules_v3.json — LOCKED)

| Group | Label | Class Name | Description |
|---|---|---|---|
| A | 0 | normal | Real CIRA normal windows |
| A | 1 | bearing_wear | Progressive mechanical degradation |
| A | 2 | impeller_imbalance | Rotodynamic imbalance |
| A | 3 | cavitation | Hydraulic shock — startup ONLY |
| A | 4 | seal_failure | Progressive pressure loss |
| A | 5 | overloading | Thermal overload — steady_state ONLY |
| A | 6 | sensor_failure | Single-channel flatline/spike/drift |
| B | 7 | bearing_wear+overloading | Compound: mechanical + thermal |
| B | 8 | cavitation+seal_failure | Compound: hydraulic shock + pressure loss |
| B | 9 | impeller_imbalance+bearing_wear | Compound: vibration dominant + degradation |
| B | 10 | seal_failure+cavitation | Compound: pressure loss precedes cavitation |
| B | 11 | impeller_imbalance+cavitation | Compound: BPF + hydraulic shock |
| C | 12 | bearing_wear_MotSV_masked | Mot.SV flatline — detect via Mot.TV + Temp.SV |
| C | 13 | cavitation_PresSV_masked | Pres.SV flatline — detect via Pmp.SV kurtosis |
| C | 14 | overloading_TempSV_masked | Temp.SV flatline — detect via Mot.TV (r=0.997) |
| C | 15 | impeller_imbalance_PmpSV_masked | Pmp.SV flatline — detect via Pmp.PV + cross-channel |
| D | 16 | cavitation_intermittent | Burst pattern: on-off hydraulic shock |
| D | 17 | seal_failure_fast | Pres.SV drops in ≤20 steps (acute) |
| D | 18 | overloading_cyclic | Sawtooth Temp.SV with rising baseline |
| E | 19 | sensor_failure_2ch_thermal | Both Mot.TV + Temp.SV degrade simultaneously |
| E | 20 | sensor_failure_2ch_pumpside | Both Pmp.SV + Pmp.PV degrade simultaneously |

### M6B Sequence Counts

| Group | Classes | Sequences per Class | Group Total |
|---|---|---|---|
| A (base) | 7 (labels 0–6) | ~1,200–1,500 | ~9,000 |
| B (compound) | 5 (labels 7–11) | ~1,200 | ~6,000 |
| C (masked) | 4 (labels 12–15) | ~1,200 | ~4,800 |
| D (variants) | 3 (labels 16–18) | ~1,200 | ~3,600 |
| E (multi-sensor) | 2 (labels 19–20) | ~800 | ~1,600 |
| **TOTAL** | **21** | — | **~25,000–27,000** |

```
M6B_combined_sequences.pkl : all groups windowed → full fault validation pool
M6B_feature_matrix.csv     : ~189,000 rows × 26 columns (M6.5r extraction)
```

### M6B Group B — Compound Fault Physics Rules

```
All Group B sequences: two faults active with secondary_onset_lag separation.
  Phase 1: primary fault only (t=0 to secondary_onset_lag)
  Phase 2: both faults active simultaneously (t=secondary_onset_lag to t=199)
  secondary_onset_lag: drawn from Uniform(30, 80) steps

Compound physics causality (LOCKED):
  Label 7: bearing_wear + overloading
    → Mot.SV↑ (bearing) + Temp.SV↑ (thermal) both elevate simultaneously
    Phase 2: both channels above 1.0, compounding MAE
  Label 8: cavitation + seal_failure
    → Pres.SV erratic (cavitation) → Pres.SV declining trend (seal adds)
    Physics: cavitation pitting damages seal → progressive pressure loss follows
  Label 9: impeller_imbalance + bearing_wear
    → Pmp.SV↑ (BPF) → Mot.SV↑ appears at lag (vibration transmits to bearing)
  Label 10: seal_failure + cavitation
    → Pres.SV declining (seal) → NPSH drops → cavitation onset at lag
    Physics: progressive pressure loss → NPSH margin erodes → cavitation
  Label 11: impeller_imbalance + cavitation
    → BPF harmonics disrupt flow field → NPSH drops → cavitation onset at lag

Expected M8 detection: DANGER state within 200 windows (Gate M8-14: ≥85% TPR)
```

### M6B Group C — Masked Fault Physics Rules

```
All Group C sequences: primary detection channel = constant (sensor flatline).
  masked_channel_flag = True in metadata.
  Detection MUST route via secondary channel Mech C path.

Physics secondary path:
  Label 12: bearing_wear_MotSV_masked  → Mot.TV + Temp.SV drift (thermal lag 20-40 steps)
  Label 13: cavitation_PresSV_masked   → Pmp.SV kurtosis bursts (BPF + hydraulic)
  Label 14: overloading_TempSV_masked  → Mot.TV drift (r=0.997 coupling PRESERVED)
  Label 15: impeller_PmpSV_masked      → Pmp.PV + cross-channel correlation change

Max achievable alert state = WARN (not DANGER) if secondary signal only.
Gate M8-13: Group C TPR ≥ 65% via secondary Mech C path.
```

### M6B Group D — Severity Variant Physics Rules

```
Label 16: cavitation_intermittent
  → Pres.SV burst pattern: high erratic during bursts, near-normal between
  → burst_interval drawn from Uniform(15, 30) steps
  → Mech B slope NOT monotonic; burst_count tracker required in M8

Label 17: seal_failure_fast
  → Pres.SV drops in ≤20 steps to minimum (acute mechanical failure)
  → single-window MAE fires immediately → DANGER within 1-3 windows

Label 18: overloading_cyclic
  → Temp.SV sawtooth with RISING baseline (each cycle starts higher)
  → Mech B slope on BASELINE (detrended) > 0.0002/window
  → Temp.SV Spearman > 0.70 on baseline-detrended signal
```

### M6B Group E — Multi-Sensor Anomaly Physics Rules

```
Label 19: sensor_failure_2ch_thermal
  → Both Mot.TV + Temp.SV simultaneously degrade (flatline/drift)
  → Physically: common thermal measurement system failure
  → multi_sensor_anomaly_count = 2 in M6.5r features

Label 20: sensor_failure_2ch_pumpside
  → Both Pmp.SV + Pmp.PV simultaneously degrade
  → Physically: pump-side accelerometer assembly failure
  → multi_sensor_anomaly_count = 2 in M6.5r features

Gate M8-14: Group E TPR ≥ 88% for multi_sensor_count = 2 detection.
```

### M6B Outputs
```
data/synthetic/M6B_sequences_groupA.pkl      ← ~9000 Group A sequences
data/synthetic/M6B_sequences_groupB.pkl      ← ~6000 Group B compound
data/synthetic/M6B_sequences_groupC.pkl      ← ~4800 Group C masked
data/synthetic/M6B_sequences_groupD.pkl      ← ~3600 Group D variants
data/synthetic/M6B_sequences_groupE.pkl      ← ~1600 Group E multi-sensor
data/synthetic/M6B_combined_sequences.pkl    ← ALL groups merged → M8 fault validation pool
data/synthetic/M6B_sequence_meta.csv         ← seq_id, label, group, severity, cluster, source
data/synthetic/M6B_feature_matrix.csv        ← ~189,000 rows × 26 columns → M7 input
models/fault_rules_v3.json                   ← 21-class label map (LOCKED)
outputs/reports/module_06b_synthetic_report.md
```

### M6B Paste Text Keys
```
M6B_total_sequences           : [~25,000–27,000]
M6B_classes                   : 21 (labels 0–20, Groups A–E)
M6B_group_A_sequences         : [~9,000]
M6B_group_B_sequences         : [~6,000]
M6B_group_C_sequences         : [~4,800]
M6B_group_D_sequences         : [~3,600]
M6B_group_E_sequences         : [~1,600]
M6B_feature_matrix_rows       : [~189,000]
M6B_feature_matrix_cols       : 26 (25 features + label)
M6B_fault_rules_version       : fault_rules_v3.json
M6B_physics_violations        : NONE
M6B_coupling_fidelity_pass    : [% passing r check]
M6B_mae_gate_pass_rate_groupA : [% Group A fault seq MAE > 0.110058]
M6B_compound_causal_pass      : [% Group B secondary_onset_lag physics correct]
M6B_masked_secondary_pass     : [% Group C secondary channel signal detectable]
Status_for_M6p5r              : READY
```

---

## ╔══════════════════════════════════════════════════╗
## M6.5 — LSTM-AE FEATURE EXTRACTOR v2 (M6A BRIDGE) — LOCKED
## Status: ✅ COMPLETED v2 (2026-04-11) — LOCKED — results govern M7 + M8 design
## ╚══════════════════════════════════════════════════╝

### Why M6.5 Exists

```
PROBLEM: XGBoost cannot consume raw time-series (shape 200×8).
         Flattening destroys temporal ordering → 1600-dim sparse feature space.
SOLUTION: Run M6A sequences through M4 LSTM-AE (inference only).
          Extract statistical features from reconstruction error array (200×8)
          → one static row per sequence → XGBoost trains on rows.
RESULT: XGBoost sees tabular data but every feature carries temporal
        meaning from LSTM-AE reconstruction behaviour.
```

### v2 Bug Fix (CRITICAL — ALL NUMBERS BELOW ARE FROM v2)
```
BUG IN v1: Gate 3 sliced sequences as [:60] instead of [:50]
           → created 60-step windows fed to M4 LSTM-AE (expects 50) → invalid MAE
FIX IN v2: Corrected to [:50] — matches M4 WINDOW_SIZE=50 (config.py)
IMPACT   : ALL Gate 3 numbers below are from v2 and AUTHORITATIVE.
           v1 Gate 3 numbers are INVALID — discard entirely.
```

### M6.5 Feature Set — 24 Features (LOCKED)

```
Per-channel mean reconstruction error : 8 features
Per-channel max reconstruction error  : 8 features
Temporal evolution features           : 5 features
  error_onset_lag, err_slope_primary, err_auc_primary,
  kurtosis_err_PmpSV, kurtosis_err_PresSV
Cross-channel features                : 2 features
  corr_delta_PmpSV_PresSV, thermal_decoupling_flag
Fuzzy fault membership                : 1 feature (fuzzy_fault_membership)
Label                                 : labels 0–6 (7 classes from M6A)
Output: data/synthetic/M6_feature_matrix.csv — 8400 rows × 25 columns
```

### Gate 3 Results — MAE Threshold Check (v2 AUTHORITATIVE)

| Class | Mean MAE | Gate 3 Pass% | Interpretation |
|---|---|---|---|
| normal | 0.1202 | 86.67% | Probe only — NOT false alarm [Finding 6] |
| bearing_wear | 0.0979 | 13.33% | Mild sev near-threshold — expected |
| impeller_imbalance | 0.1031 | 30.00% | Mild sequences dominate |
| cavitation | 0.6747 | 100.00% | ✅ MAE=0.675, 6.1× threshold |
| seal_failure | 0.1961 | 29.17% | Slow hydraulic — Mech C PRIMARY |
| overloading | 0.0930 | 0.00% | Thermal-dominant — Mech C PRIMARY |
| sensor_failure | 0.1696 | 93.33% | ✅ Flatline clearly anomalous |

### Top 5 Fisher Discriminant Features (LOCKED)

| Rank | Feature | Physics Validation |
|---|---|---|
| 1 | Pmp_SV_mean | Pump vibration dominant ✅ |
| 2 | Pmp_SV_std | Vibration variance ✅ |
| 3 | Temp_SV_mean | Thermal drift — overloading discriminator ✅ |
| 4 | Mot_TV_mean | Motor temp — bearing/overloading ✅ |
| 5 | Mot_TV_std | Temperature variance ✅ |

### M6.5 Outputs (LOCKED)
```
data/synthetic/M6_feature_matrix.csv        ← 8400 rows × 25 columns (M6A, 7-class)
src/module_065_sequence_audit.py            ← v2 (Gate 3 :60→:50 fix applied)
outputs/reports/module_065_sequence_audit_report.md
```

### M6.5 Paste Text Keys (LOCKED)
```
M6p5_feature_matrix_rows        : 8400
M6p5_features_per_row           : 24 + label = 25 columns
M6p5_gate3_normal_probe         : 86.67% (probe only — NOT FPR)
M6p5_gate3_cavitation           : 100.00% (MAE=0.675, 6.1×)
M6p5_gate3_overloading          : 0.00% (thermal-dominant — Mech C)
M6p5_gate3_seal_failure         : 29.17% (slow fault — Mech C)
M6p5_top_fisher_feature         : Pmp_SV_mean (rank 1)
M6p5_window_fix                 : v2 corrected :60→:50
M6p5_seal_patch                 : 165→220 sequences accepted
```

---

## ⚠️ M6.5 AUDIT — 6 CRITICAL FINDINGS (Govern M7 and M8 Design) ⚠️
## THESE ARE LOCKED — DO NOT MODIFY

### FINDING 1 — OVERLOADING IS THERMAL-DOMINANT (Gate 3 pass = 0.00%)
```
Observed : mean MAE = 0.093 — BELOW threshold 0.110058
Root cause: M4 Temp.SV weight=1.0, Mot.TV weight=0.8 (lowest weights).
            Overloading raises ONLY thermal channels — weighted MAE sub-threshold.

M7 implication : XGBoost classifies correctly via mean_err_TempSV (Fisher rank 3).
M8 MANDATORY  : Mech C Temp.SV Spearman drift > 0.70 = PRIMARY detection.
                Gate M8-7: overloading TPR ≥ 80% via Mech C ONLY.
                Do NOT measure via single-window MAE threshold crossing.
DO NOT raise global threshold to compensate.
```

### FINDING 2 — SEAL FAILURE IS A SLOW HYDRAULIC FAULT (Gate 3 pass = 29.17%)
```
Observed : mean MAE = 0.1961 — above threshold ON AVERAGE but 29.17% of
           individual 50-step windows cross threshold.
Root cause: Pres.SV* decline very gradual. Single window shows only small drop.

Seal patch applied (M6.5 v2):
  Original: only 165/1200 sequences exceeded MAE threshold
  Fix: severity distribution rebalanced toward [0.4, 0.7] band
  Final accepted: 220 sequences (padded to 1200 with physics variants)

M8 MANDATORY : Pres.SV Spearman drift (NEGATIVE) > 0.70 over 300 windows
               = seal_failure_early flag (PRIMARY).
               Gate M8-9: WATCH fires ≤ 20 min of onset.
               Gate M8-10: Pres.SV drift fires BEFORE total MAE reaches WARN.
DO NOT raise global threshold to accommodate seal_failure.
```

### FINDING 3 — BEARING WEAR TEMPORAL COHERENCE = 94.25% (69 flagged sequences)
```
Observed : 69 sequences have dX/dt discontinuity at seam (t=49→50)
           between spike seed onset and M5 physics continuation.
Decision : Sequences KEPT — represent realistic mechanical shock events.
M8 implication: Monitor attention heatmap — peaks at fault onset, not seam.
               Gate M8-8: seam_ratio = mean_attention(t=49,50) / mean_attention(t=10,40)
               Gate: seam_ratio < 1.0 (fault onset dominates over seam artifact).
M12 implication: Config 1–3 adversarial must use SMOOTH sequences only.
```

### FINDING 4 — TOP FISHER FEATURES CONFIRM M8 CHANNEL WEIGHT DIRECTION
```
Fisher rank 1: Pmp_SV_mean  → confirms M8 weight Pmp.SV 2.0→2.5 ✅
Fisher rank 2: Pmp_SV_std   → variance is discriminator ✅
Fisher rank 3: Temp_SV_mean → low M4 weight but survives → M7 uses for overloading
Fisher rank 4: Mot_TV_mean  → M4 weight 0.8 but signal survives ✅
Fisher rank 5: Mot_TV_std   → temperature variance discriminative ✅

M8 weight decision VALIDATED:
  Increase: Mot.SV=2.5, Pmp.SV=2.5, Pres.SV=2.5 (from 2.0)
  Increase: Mot.PV=2.0, Pmp.PV=2.0 (from 1.5)
  Decrease: Temp.SV=0.5, Mot.TV=0.3, Pmp.TV=0.3
  Mech C monitors Temp.SV UNWEIGHTED via raw channel error — NOT weight matrix.
```

### FINDING 5 — CAVITATION STRONGLY ANOMALOUS (Gate 3 = 100%, MAE = 0.675)
```
Observed : MAE = 0.675 — 6.1× above threshold. Every window crosses.
M8 implication: Cavitation → DANGER immediately (skip WATCH/WARN).
               Gate M8-12: ZERO cavitation DANGER outside startup cluster.
REPORT SEPARATELY: Do NOT lump cavitation with overloading in TPR.
  If overloading TPR=50%, cavitation TPR=100% → overall=75% hides the gap.
  Gate M8-7 (overloading) measured independently.
```

### FINDING 6 — NORMAL PROBE 86.67% = NOT A FALSE ALARM PROBLEM
```
Observed : 86.67% of 30 probed normal windows crossed MAE threshold.
Why NOT a problem:
  30-window probe deliberately samples near-boundary edge cases.
  Full M4 val set (1457 windows) → 0.55% FPR (8/1457) — confirmed.
M8 action:
  Do NOT adjust threshold based on probe.
  Gate M8-2 (FPR < 5%) measured on FULL 9711-window pool ONLY.
  Cluster-conditional thresholds handle remaining boundary cases.
```

---

## ╔══════════════════════════════════════════════════╗
## M6.5r — UPDATED FEATURE EXTRACTOR FOR M6B (21-CLASS)
## Status: ✅ COMPLETED (2026-04-14) — ACTIVE — FEEDS M7 AND M8
## ╚══════════════════════════════════════════════════╝

### Why M6.5r Was Created

```
M6.5 (original): processed M6A only — 7 classes, 24 features, 8400 rows.
M6B added Groups B, C, D, E → 21 classes, ~25,000–27,000 sequences.
M6.5r runs M6B_combined_sequences through M4 LSTM-AE and extracts
an EXPANDED 25-feature set (26 columns including label).

New features added for Groups B–E:
  masked_channel_flag       : 1 if primary detection channel = constant (Group C)
  secondary_onset_lag       : steps until secondary fault channel activates (Group B)
  burst_count               : number of MAE spikes in 200 steps (Group D16)
  cyclic_baseline_drift     : Temp.SV baseline slope (Group D18)
  multi_sensor_anomaly_count: 0/1/2 — how many channels simultaneously anomalous (Group E)
```

### M6.5r Feature Set — 25 Features + Label (26 columns)

```
Inherited from M6.5 (24 features):
  Per-channel mean reconstruction error   : 8 features
  Per-channel max reconstruction error    : 8 features
  Temporal evolution                      : 5 features
  Cross-channel                           : 2 features
  Fuzzy fault membership                  : 1 feature

New features for M6B Groups B–E (5 additional features):
  masked_channel_flag         : bool ← Group C primary detection channel absent
  secondary_onset_lag         : int  ← Group B secondary fault onset step
  burst_count                 : int  ← Group D16 cavitation_intermittent burst count
  cyclic_baseline_drift       : float ← Group D18 overloading_cyclic slope
  multi_sensor_anomaly_count  : int  ← Group E number of simultaneously anomalous channels

Label: 0–20 (21 classes, fault_rules_v3.json)

Output: M6B_feature_matrix.csv — ~189,000 rows × 26 columns → M7 input
```

### M6.5r Validation Gates

| Gate | Check | Target | Physics Basis |
|---|---|---|---|
| W1 | Group A Gate 3 MAE pass rates match M6.5 (locked) | Within 2% | Consistency check |
| W2 | All Group B sequences: both fault channels elevated | 100% | Compound causal physics |
| W3 | All Group B: MAE > 0.110058 in Phase 2 | ≥90% | Both channels active = high MAE |
| W4 | Group C: masked channel std < 0.001 confirmed | 100% | Flatline verification |
| W5 | Group C: secondary channel detectable (MAE>0 on secondary) | ≥95% | Secondary path exists |
| W6 | Group D16: burst_count ≥ 2 per sequence | 100% | Intermittent by definition |
| W7 | Group D17: Pres.SV slope > 3× Group A seal_failure | 100% | “Fast” validated |
| W8 | Group D18: cyclic_baseline_drift > 0 | 100% | Rising baseline confirmed |
| W9 | Group E: multi_sensor_anomaly_count = 2 | 100% | Dual degradation |
| W10 | secondary_onset_lag in [30, 80] for all Group B | 100% | Physics plausible range |

### M6.5r Outputs
```
data/synthetic/M6B_feature_matrix.csv     ← ~189,000 rows × 26 cols → M7 input
src/module_065r_feature_extractor.py      ← extended feature extraction script
outputs/reports/module_065r_feature_extractor_report.md
```

### M6.5r Paste Text Keys
```
M6p5r_rows                   : [~189,000]
M6p5r_cols                   : 26 (25 features + label)
M6p5r_labels                 : 21 (0–20, fault_rules_v3.json)
M6p5r_gate_W1_pass           : PASS/FAIL
M6p5r_gate_W2_pass           : PASS/FAIL
M6p5r_gate_W3_pass           : PASS/FAIL
M6p5r_gate_W4_pass           : PASS/FAIL
M6p5r_gate_W5_pass           : PASS/FAIL
M6p5r_gate_W6_to_W10_pass    : PASS/FAIL
M6p5r_masked_channel_pct     : [% Group C rows with masked_channel_flag=1]
M6p5r_mean_secondary_lag     : [mean secondary_onset_lag for Group B]
M6p5r_mean_burst_count       : [mean burst_count for Group D16]
M6p5r_multi_sensor_count_2   : [% Group E rows with multi_sensor_anomaly_count=2]
Status_for_M7                : READY
```

---

## CROSS-MODULE INVARIANTS (Enforced M1 → M12 — ALL 15)

```
1.  segment_id preserved in ALL dataframes through M6B
2.  Windows NEVER cross segment boundaries
3.  Normalization baselines LOCKED at M3_normalization_config.json
4.  Winsor ceilings LOCKED at M4_spike_config.json (M6B + M12 read, do not override)
5.  M4 threshold=0.110058 is the fault/normal boundary for M6B validation gate
6.  M8 cluster-conditional thresholds are the production boundary for M12
7.  Physical couplings (r>0.87) must hold in ALL synthetic sequences (M6B + M12)
8.  Conservation of energy + mass in all synthetic sequences (Groups A–E)
9.  Household pump → physics_advisory_only() always — no ML inference
10. XGBoost: device="cuda" train | device="cpu" deploy
11. All models: torch.save(state_dict) | torch.load(map_location="cpu")
12. M7 trains on M6.5r M6B_feature_matrix.csv (26 cols) — NOT M6.5 M6A matrix (25 cols)
13. Fuzzy logic is a core M8 detection component — not just M10 display
14. M12 MUST pass PRODUCTION_VALIDATED before deployment on 110 kW asset
15. M8 MUST detect slow drift faults (sev 0.2–0.3) via trend accumulator
    within 20 minutes of fault onset — this is the Category 3 liability gate.
    M6B severity 0.2–0.3 sequences are the TRAINING DATA for this requirement.
```

---

## FILE STRUCTURE (Updated for M6B)

```
PumpSmart_Project/
├── config.py
├── data/
│   ├── raw/                           ← 9 original CSVs (never modified)
│   ├── clean/                         ← M1 output
│   ├── normalized/                    ← M3 output
│   └── synthetic/                     ← M4 seeds + M6A archive + M6B active
│       ├── M4_spike_seeds.npy         ← shape=(1044, 50, 8) — LOCKED
│       ├── M4_spike_seeds_meta.csv
│       ├── M4_spike_config.json       ← LOCKED winsor bounds
│       ├── M6A_sequences.pkl          ← 8400 seq (archived, superseded)
│       ├── M6A_sequence_meta.csv      ← archived
│       ├── M6_feature_matrix.csv      ← M6.5 output (M6A, 8400×25) — archived
│       ├── M6B_sequences_groupA.pkl   ← ACTIVE — Group A ~9000 seq
│       ├── M6B_sequences_groupB.pkl   ← ACTIVE — Group B ~6000 seq
│       ├── M6B_sequences_groupC.pkl   ← ACTIVE — Group C ~4800 seq
│       ├── M6B_sequences_groupD.pkl   ← ACTIVE — Group D ~3600 seq
│       ├── M6B_sequences_groupE.pkl   ← ACTIVE — Group E ~1600 seq
│       ├── M6B_combined_sequences.pkl ← ALL groups → M8 fault validation pool
│       ├── M6B_sequence_meta.csv      ← ACTIVE — 21-class metadata
│       └── M6B_feature_matrix.csv     ← ACTIVE — ~189,000×26 → M7 input
├── models/
│   ├── lstm_ae_baseline_best.pth      ← M4 model (LOCKED)
│   ├── M3_normalization_config.json   ← LOCKED baselines
│   ├── M4_threshold_config.json       ← threshold=0.110058 (LOCKED)
│   ├── fault_rules.json               ← M5 original 6-class (LOCKED — archived)
│   ├── fault_rules_v3.json            ← 21-class label map (LOCKED — ACTIVE)
│   ├── M5_physics_config.json
│   └── unit_registry.json
├── outputs/
│   ├── reports/                       ← one .md per module
│   └── plots/                         ← one set of plots per module
├── src/                               ← module_01 through module_12 scripts
└── app/                               ← Flask web app (M10)
```

---

## MODULE PROGRESS TRACKER

```
M1    Data Ingestion & Cleaning          : ✅ COMPLETED (2026-03-25)
M2    EDA + Operating Mode Clustering    : ✅ COMPLETED (2026-03-26)
M3    Dimensionless Normalization        : ✅ COMPLETED (2026-03-28)
M4    LSTM-AE Baseline (v8)              : ✅ COMPLETED (2026-03-28)
M5    Physics Engine                     : ✅ COMPLETED (2026-03-29)
M6A   Synthetic Dataset (7-class)        : ✅ COMPLETED (2026-04-11) — SUPERSEDED by M6B
M6.5  LSTM-AE Feature Extractor v2      : ✅ COMPLETED (2026-04-11) — LOCKED
M6B   Expanded Synthetic (21-class)     : ✅ COMPLETED (2026-04-14) — ACTIVE
M6.5r Updated Feature Extractor (M6B)   : ✅ COMPLETED (2026-04-14) — ACTIVE
M7    XGBoost Fault Classifier (21-class): 🔲 NEXT ACTIVE — trains on M6B_feature_matrix.csv
M8    LSTM-AE v2 + Fuzzy Logic           : 🔲 NOT STARTED — see module_M8_lstm_ae_v2_architecture.md
M9    Pump Selector + Household Advisor  : 🔲 NOT STARTED
M10   Flask Web Application              : 🔲 NOT STARTED
M11   Docker + Hugging Face Deployment   : 🔲 NOT STARTED
M12   Physics-Governed Validation Suite  : 🔲 NOT STARTED (post-M11)
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic file `completed_modules_M1_to_M6p5.md` — M1 through M6.5 |
| v2.0 | 2026-04-15 | **SPLIT into Part 1 + Part 2.** Part 2 = M5–M6.5r (this file). Added: M6B 21-class dataset (Groups A–E), M6.5r 26-feature matrix, fault_rules_v3.json, 10 M6.5r gates, cross-module invariant 12 updated (M6B matrix). File structure updated for M6B active files. Progress tracker updated for M6B + M6.5r completed status. |

---

*GitHub is the ONLY source of truth for this spec.*
*Companion: `completed_modules_context_and_M1_to_M4.md` (Part 1 — LOCKED context + M1–M4)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
