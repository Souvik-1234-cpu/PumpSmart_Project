# PumpSmart — Completed Modules Reference: M5 to M6B
# Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring
# PART 2A OF 3 — M5 results + M6A results + M6B 22-class spec (v14.1)
#
# Status: M5, M6A sections LOCKED. M6B section = SPEC ONLY — NOT YET EXECUTED.
# Updated: 2026-04-18 | Author: Souvik
# Split from: completed_modules_M5_to_M6p5r.md (v3.0, too large)
#
# THIS FILE CONTAINS:
#   - M5 physics engine results + paste keys (LOCKED)
#   - M6A synthetic generator results + paste keys (LOCKED — superseded by M6B)
#   - M6B 22-class expanded synthetic dataset SPEC + paste keys (🔴 NEXT ACTIVE — script not yet run)
#   - v14.0 additions: label 21 bearing_wear_gradual, 4-layer detection architecture,
#                      CUSUM runtime state, rolling baseline comparator
#   - v14.1 corrections: pump-side multi-sensor pair confirmed as PmpSV+PmpPV;
#                        Group B/C canonical map reaffirmed; seal_failure_fast
#                        treated as rapid hydraulic discharge, not laminar pipe flow
#
# COMPANION FILES:
#   completed_modules_context_and_M1_to_M4.md    → Part 1 LOCKED: context, M1–M4 results
#   completed_modules_M6p5_to_invariants.md      → Part 2B: M6.5, audit findings, M6.5r, invariants
#
# GitHub is the ONLY source of truth. Spaces .md files are OUTDATED — do not use.

---

## ╔══════════════════════════════════════════════════╗
## M5 — PHYSICS ENGINE
## Status: ✅ COMPLETED (2026-03-29) — LOCKED
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
models/fault_rules.json              ← 6 fault types (M5 original, Group A only) — LOCKED
                                        NOTE: fault_rules_v3.json (22-class) is an M6B output,
                                        NOT an M5 output. Written by M6B Step 3.
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
> (22 classes, Groups A–E, ~26,000–28,000 sequences). M6B is the dataset used for
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
## M6B — EXPANDED SYNTHETIC DATASET (22-CLASS, GROUPS A–E)  ← v14.1
## Status: 🔴 NEXT ACTIVE — SPEC LOCKED, SCRIPT NOT YET RUN
## ╚══════════════════════════════════════════════════╝

> ⚠️ IMPORTANT: M6B has NOT been executed. No output files exist yet.
> fault_rules_v3.json, M6B_*.pkl, M6B_sequence_meta.csv, M6B_feature_matrix.csv
> are ALL pending — they will be created when the M6B script runs.
> This section is the LOCKED SPEC that governs the M6B script.

### Why M6B Was Created

```
M6A had 7 classes (labels 0–6) — adequate for basic fault detection but insufficient for:
  1. Compound fault scenarios (two faults active with causal lag)
  2. Primary-channel masked faults (sensor dead, secondary path only)
  3. Fault severity variants (fast vs slow, intermittent vs sustained)
  4. Slowly-progressing faults below detection threshold (pre-WATCH zone)
  5. Multi-sensor anomaly scenarios (2 sensors degraded simultaneously)
  6. M7 XGBoost training on M6A alone = limited real-world coverage
  7. M8 adversarial validation requires compound + masked + variant scenarios

M6B adds Groups B, C, D, E to the base Group A (M6A classes).
All 22 classes will be defined in fault_rules_v3.json — written by M6B Step 3.
```

### 22-Class Label Map — v14.1 (LOCKED — will be written to fault_rules_v3.json by M6B Step 3)

| Group | Label | Class Name | Description |
|---|---|---|---|
| A | 0 | normal | Real CIRA normal windows |
| A | 1 | bearing_wear | Progressive mechanical degradation |
| A | 2 | impeller_imbalance | Rotodynamic imbalance |
| A | 3 | cavitation | Hydraulic shock — startup ONLY |
| A | 4 | seal_failure | Progressive pressure loss |
| A | 5 | overloading | Thermal overload — steady_state ONLY |
| A | 6 | sensor_failure | Single-channel flatline/spike/drift |
| B | 7 | bearing_wear→overloading | Compound: MotSV rise first, TempSV/MotTV runaway after. Lag 200–400s |
| B | 8 | cavitation→seal_failure | Compound: PmpSV spikes first, PresSV progressive drop after. Lag 50–150s |
| B | 9 | impeller_imbalance→bearing_wear | Compound: PmpSV BPF first, MotSV Paris-law drift after. Lag 300–600s |
| B | 10 | seal_failure→cavitation | Compound: PresSV drops first, PmpSV spikes appear after. Lag 100–200s |
| B | 11 | overloading→bearing_wear | Compound: TempSV rises first, MotSV begins drift after. Lag 400–600s |
| B | 12 | impeller_imbalance→cavitation | Compound: PmpSV BPF first, PresSV erratic + PmpSV spikes. Lag 100–300s |
| C | 13 | bearing_wear_MotSV_masked | MotSV flatlined — detect via MotTV + TempSV thermal lag |
| C | 14 | cavitation_PresSV_masked | PresSV dropout — detect via PmpSV kurtosis bursts |
| C | 15 | seal_failure_PresSV_drifting | PresSV drifting — detect via secondary hydraulic channels |
| C | 16 | overloading_TempSV_stuck | TempSV stuck — detect via MotTV (r=0.997 coupling) |
| C | 17 | impeller_imbalance_PmpSV_flatline | PmpSV flatline — detect via PmpPV + cross-channel |
| D | 18 | cavitation_intermittent | NPSHa oscillates around NPSHr boundary — burst pattern |
| D | 19 | seal_failure_fast | Rapid hydraulic discharge causes large-Δ PresSV drop in ≤20 steps |
| D | 20 | overloading_cyclic | Thermal sawtooth: load ON/OFF + rising baseline per cycle |
| D | 21 | bearing_wear_gradual | ← NEW v14.0. Paris law small ΔK. MotSV rises barely above baseline over 150+ steps. Primary discriminator: err_slope_MotSV. Requires CUSUM+rolling accumulator. |
| E | [fault_rules_v3.json] | sensor_failure_2ch_thermal | MotTV + TempSV simultaneously degrade |
| E | [fault_rules_v3.json] | sensor_failure_2ch_pump | PmpSV + PmpPV simultaneously degrade |

> ⚠️ Group E exact labels confirmed in fault_rules_v3.json (written by M6B Step 3).
>    Total 22 classes = labels 0–21. Do not hardcode Group E label numbers from this file.

### M6B Sequence Counts (Target — populated after script runs)

| Group | Classes | Sequences per Class | Group Total |
|---|---|---|---|
| A (base) | 7 (labels 0–6) | 1,200 | ~8,400 (LOCKED from M6A) |
| B (compound) | 6 (labels 7–12) | 1,200 | ~7,200 |
| C (masked) | 5 (labels 13–17) | 800 | ~4,000 |
| D (variants) | 4 (labels 18–21) | 600 (labels 18–20) + 1,000 (label 21) | ~2,800 |
| E (multi-sensor) | 2 (labels per fault_rules_v3.json) | 400 | ~800 |
| **TOTAL** | **22** | — | **~26,000–28,000** |

```
⚠️ NONE OF THESE FILES EXIST YET — created by M6B script:
data/synthetic/M6B_combined_sequences.pkl : all groups → full fault validation pool
data/synthetic/M6B_feature_matrix.csv     : ~196,000 rows × 26 columns (from M6.5r)
models/fault_rules_v3.json                : written by M6B Step 3 (22-class, labels 0–21)
```

### M6B Group B — Compound Chain Fault Physics Rules

```
All Group B sequences: two faults active with secondary_onset_lag separation.
  Phase 1: primary fault only (t=0 to secondary_onset_lag)
  Phase 2: both faults active simultaneously (t=secondary_onset_lag to t=199)
  secondary_onset_lag: drawn from fault-specific range (see label map above)
  Sequences per class: 1,200

Each compound chain = UNIQUE INTEGER LABEL (single-label XGBoost).
M10 API maps label → "Primary: X → Secondary: Y" in UI display.

Physics discriminator (causal propagation vs independent — governs M10 BIAS logic):
  MotSV rises → MotTV rises 30 steps LATER    = SINGLE (thermal lag)
  MotSV rises + PresSV drops SIMULTANEOUSLY   = COMPOUND (uncorrelated)
  PresSV drops → PmpSV spikes AFTER           = SINGLE (seal→cavitation chain)
  PresSV drops + MotTV rises SIMULTANEOUSLY   = COMPOUND (no physical link)

Expected M8 detection: DANGER state within 200 windows (Gate M8-14: ≥85% TPR)
```

### M6B Group C — Masked Fault Physics Rules

```
All Group C sequences: primary detection channel = constant (sensor failed).
  masked_channel_flag = True in metadata.
  Detection MUST route via secondary channel Mech C path.

Physics secondary path:
  Label 13: bearing_wear_MotSV_masked   → MotTV + TempSV drift (thermal lag 20–40 steps)
  Label 14: cavitation_PresSV_masked    → PmpSV kurtosis bursts (BPF + hydraulic)
  Label 15: seal_failure_PresSV_drift   → secondary hydraulic channels (PmpSV, PmpPV cross-channel)
  Label 16: overloading_TempSV_stuck    → MotTV drift (r=0.997 coupling PRESERVED)
  Label 17: impeller_PmpSV_flatline     → PmpPV + cross-channel correlation change

Max achievable alert state = WARN (not DANGER) if secondary signal only.
Gate M8-13: Group C TPR ≥ 65% via secondary Mech C path.
```

### M6B Group D — Severity Variant Physics Rules

```
Label 18: cavitation_intermittent
  → NPSHa oscillates around NPSHr boundary
  → PmpSV burst pattern: high erratic during bursts, near-normal between
  → burst_interval drawn from Uniform(15, 30) steps
  → MAE must stay ABOVE threshold even in low-NPSHa oscillation phase (Finding 5 constraint)
  → Mech B slope NOT monotonic; burst_count tracker required in M8

Label 19: seal_failure_fast
  → Rapid hydraulic discharge through enlarged effective seal leak area
  → PresSV drops in ≤20 steps to minimum
  → single-window MAE fires immediately → DANGER within 1–3 windows
  → Must show faster PresSV drop than standard seal_failure (Finding 2 constraint)
  → Governing equation: Q_leak = Cd × A_orifice × sqrt(2 × ΔP / ρ)
  → Do NOT model with Hagen–Poiseuille; seal blowout is not laminar pipe flow

Label 20: overloading_cyclic
  → Thermal sawtooth: Temp.SV load ON/OFF with RISING baseline across cycles
  → Each cycle starts higher than previous — cyclic_baseline_drift > 0.0002/window
  → Temp.SV Spearman > 0.70 on baseline-detrended signal
  → TempSV sawtooth steeper than standard overloading — accumulator fires within 15 min

Label 21: bearing_wear_gradual  ← NEW v14.0
  → Paris–Erdogan crack growth with SMALL ΔK (low stress intensity range):
     da/dN = C × ΔK^m   [same equation as label 1, smaller ΔK input]
  → MotSV rises BARELY above baseline over 150+ steps
  → Weibull β=1.5, severity=0.05–0.25 (low end of crack growth spectrum)
  → CIRA anchor: same 44 bearing-impact spike seeds as label 1 (fully anchored)
  → Primary discriminator: err_slope_MotSV (small, consistent positive slope)
  → Sequences at severity < 0.15 will have MAE < 0.110058 — PHYSICALLY CORRECT
     (fault genuinely below alarm level; CUSUM/rolling accumulator catches it)
  → Physics validity: AMBER-GREEN (conditionally valid — requires Layer 3+4 to be useful)
  → Target: 1,000 sequences (higher count — harder to learn)
  → G11 extension: err_slope_MotSV > 0 in ≥95% of label 21 sequences
  → XGBoost output message: "bearing_wear_gradual — plan inspection within 7–14 days"
  → Detection path:
       Layer 4 (Rolling Baseline): shift detected at ~Week 5 (pre-threshold)
       Layer 3 (CUSUM): alert at ~Week 5.5 (~30% bearing degraded)
       Layer 2 (Accumulator): WATCH triggered at ~Week 6
       Layer 1 (LSTM-AE): threshold crossing at ~Week 7 (too late alone)
  → LSTM-AE 50-step window alone is INSUFFICIENT for label 21. Requires all 4 layers.
```

### M6B Group E — Multi-Sensor Anomaly Physics Rules

```
Labels assigned in fault_rules_v3.json (written by M6B Step 3). Do not hardcode here.

sensor_failure_2ch_thermal:
  → Both MotTV + TempSV simultaneously degrade (flatline/drift)
  → Physically: common thermal measurement system failure (shared excitation rail)
  → multi_sensor_anomaly_count = 2 in M6.5r features

sensor_failure_2ch_pump:
  → Both PmpSV + PmpPV simultaneously degrade
  → Physically: moisture ingress to pump-side junction box
  → Motor-side sensors (MotPV, MotSV, MotTV) remain normal
  → multi_sensor_anomaly_count = 2 in M6.5r features

Gate M8-14: Group E TPR ≥ 88% for multi_sensor_count = 2 detection.
Target: ~400 sequences per variant = ~800 sequences total
```

### M6B 3-Step Script Structure (SPEC LOCKED — not yet run)

```
Step 1 (⬜ PENDING):
  Group B — 6 compound chains (labels 7–12)
  Validation gates G8 (temporal lag correct), G9 (causal channel order)
  Output: data/synthetic/M6B_sequences_groupB.pkl

Step 2 (⬜ PENDING):
  Group C — 5 masked faults (labels 13–17)
  Group D — 4 severity variants (labels 18–21, incl. bearing_wear_gradual)
  Group E — 2 multi-sensor failures
  Validation gates G10 (masked secondary signal), G11 (severity MAE)
  G11 extension for label 21: err_slope_MotSV > 0 in ≥95% of sequences
  Output: data/synthetic/M6B_sequences_groupC.pkl
          data/synthetic/M6B_sequences_groupD.pkl
          data/synthetic/M6B_sequences_groupE.pkl

Step 3 (⬜ PENDING):
  Full merge: M6A (Group A, 8,400 locked) + M6B Groups B+C+D+E
  Writes: models/fault_rules_v3.json (22-class, labels 0–21) — LOCKED after run
  Full validation suite (physics, coupling, MAE, temporal gates)
  Writes: outputs/reports/module_06B_synthetic_report.md
  Output: data/synthetic/M6B_combined_sequences.pkl
          data/synthetic/M6B_sequence_meta.csv

LOCKED FILES — DO NOT OVERWRITE:
  models/fault_rules.json      (v1 — M5/M6A reference, frozen)
  data/synthetic/M6A_* outputs (frozen after M6A completion)
```

### M6B Planned Outputs (⚠️ NONE EXIST YET — written when M6B script runs)
```
data/synthetic/M6B_sequences_groupA.pkl      ← ~8,400 Group A sequences (from M6A, locked)
data/synthetic/M6B_sequences_groupB.pkl      ← ~7,200 Group B compound (1,200 × 6)
data/synthetic/M6B_sequences_groupC.pkl      ← ~4,000 Group C masked (800 × 5)
data/synthetic/M6B_sequences_groupD.pkl      ← ~2,800 Group D variants (600×3 + 1,000×1)
data/synthetic/M6B_sequences_groupE.pkl      ← ~800 Group E multi-sensor (400 × 2)
data/synthetic/M6B_combined_sequences.pkl    ← ALL groups merged → M8 fault validation pool
data/synthetic/M6B_sequence_meta.csv         ← seq_id, label, group, severity, cluster, source
models/fault_rules_v3.json                   ← 22-class label map (written by M6B Step 3)
outputs/reports/module_06b_synthetic_report.md
```

### M6B Paste Text Keys (⚠️ Populate AFTER script runs — do not fill in advance)
```
M6B_total_sequences           : [fill after run — target ~26,000–28,000]
M6B_classes                   : 22 (labels 0–21, Groups A–E)
M6B_group_A_sequences         : [fill — ~8,400 locked from M6A]
M6B_group_B_sequences         : [fill — target ~7,200]
M6B_group_C_sequences         : [fill — target ~4,000]
M6B_group_D_sequences         : [fill — target ~2,800 (incl. 1,000 label 21)]
M6B_group_E_sequences         : [fill — target ~800]
M6B_label21_sequences         : [fill — target 1,000]
M6B_label21_slope_gate        : [fill — err_slope_MotSV > 0 in ≥95% seqs]
M6B_fault_rules_version       : [fill — fault_rules_v3.json written in Step 3]
M6B_physics_violations        : [fill — expect NONE]
M6B_coupling_fidelity_pass    : [fill from M6B run log]
M6B_mae_gate_B_pass           : [fill from M6B run log]
M6B_mae_gate_C_pass           : [fill from M6B run log]
M6B_combined_output           : data/synthetic/M6B_combined_sequences.pkl
M6B_meta_output               : data/synthetic/M6B_sequence_meta.csv
Status_for_M6p5r              : READY / BLOCKED
```

---

## ╔══════════════════════════════════════════════════╗
## FOUR-LAYER DETECTION ARCHITECTURE — v14.1
## Governs M8 + M10 design. LOCKED.
## ╚══════════════════════════════════════════════════╝

```
Four detection layers run in cascade across M8 (training) and M10 (runtime):

Layer 1 — LSTM-AE 50-step window (M8):
  Detects  : "Is THIS window anomalous?"
  Memory   : 50 steps only (encoder hidden state resets each window)
  Blind to : cross-window trends
  Threshold: 0.110058 (LOCKED)

Layer 2 — Fuzzy Logic + Rolling Accumulator (M8):
  Detects  : "Has anomaly score been elevated across recent windows?"
  Memory   : last 20–40 windows
  Catches  : faults where MAE = 0.07–0.10 consistently across windows
  States   : WATCH (score 2.0–3.5) → WARN (3.5–5.0) → FAULT (5.0+)

Layer 3 — CUSUM Runtime State (M10):  ← NEW v14.0
  Formula  : S_n = max(0, S_{n-1} + (mae_channel_n − μ0) − k)
  Reference: μ0 = M3 cluster baseline MAE per channel (READ-ONLY from M3_normalization_config.json)
  Allowance: k = 0.5 × (threshold − μ0) per channel
  Channels : mae_MotSV, mae_PresSV, mae_TempSV
  Fires when S_n exceeds control limit (configurable, default=5.0)
  Catches  : label 21 (bearing_wear_gradual) at ~Week 5.5 (~30% bearing degraded)
  State    : PERSISTENT across API calls in M10 runtime memory
  Resets   : only on explicit operator acknowledge or pump restart
  IMPORTANT: CUSUM is NOT in M6.5r feature matrix — train-serve skew risk if added
             CUSUM lives in M10 deployment layer ONLY

Layer 4 — Rolling Baseline Comparator (M10):  ← NEW v14.0
  Monitors : 30-window rolling mean of err_slope_MotSV, err_slope_PresSV, err_slope_TempSV
  Reference: M3_normalization_config.json normal baselines (mean, σ per cluster)
  Limit    : rolling mean > μ_normal + 2σ_normal → TREND ALERT
  Catches  : pre-threshold drift weeks before Layer 1 fires
  State    : PERSISTENT across API calls in M10 runtime memory
  Output   : feeds bearing_wear_gradual advisory: "Plan bearing inspection within 7–14 days"
  Physics  : rising rolling slope = da/dN trend (Paris law) — ISO 7870 Shewhart control chart

Why CUSUM was NOT added to M6.5r feature matrix (REJECTED — Option A):
  M6.5r generates features per-window independently.
  CUSUM at window [100:150] needs MAE history from [50:100], [0:50], etc.
  In real deployment, CUSUM accumulates across streaming windows indefinitely.
  Train (sequence-internal CUSUM) ≠ Deploy (cross-session CUSUM) → train-serve skew.
  Therefore: CUSUM belongs in M10 runtime only. Feature matrix unchanged at 26 cols.

Three M10 API output states:
  State 1: fault_stage=early, fault_type=unknown → action: MONITOR
  State 2: fault_stage=developing, compound=False → action: ALERT (single label)
  State 3: fault_stage=developing, compound=True  → action: CRITICAL
           causal_chain=[e.g. bearing_wear→overloading]
           (multi-label UI display, single integer label internally)
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic file `completed_modules_M1_to_M6p5.md` |
| v2.0 | 2026-04-15 | SPLIT into Part 1 + Part 2. Added M6B 21-class spec, M6.5r 26-feature spec, 15 invariants |
| v3.0 | 2026-04-15 | CORRECTION: M6B/M6.5r status, Group E labels, seq counts, PENDING markers |
| v4.0 | 2026-04-16 | **v14.0 UPGRADE**: Split into Part 2A (this file) + Part 2B (M6.5/invariants). Added label 21 bearing_wear_gradual to Group D. Classes 21→22. Sequences ~26,000–28,000. Group B corrected to 6 chains (labels 7–12). Group C corrected to 5 masked scenarios (labels 13–17). 4-layer detection architecture added (CUSUM Layer 3 + Rolling Baseline Layer 4). CUSUM rejected from feature matrix (train-serve skew). All seq counts updated to v14.0 targets. |
| v5.0 | 2026-04-18 | **v14.1 CORRECTION**: Header/version metadata updated to v14.1. Confirmed canonical Group B and Group C label map remains unchanged from v14.0. Confirmed Group E pump-side multi-sensor pair as `PmpSV + PmpPV`. Clarified `seal_failure_fast` as rapid hydraulic discharge / orifice-flow behavior, not laminar pipe-flow wording. No completed module results changed. |

---

*GitHub is the ONLY source of truth for this spec.*
*Companion Part 1: `completed_modules_context_and_M1_to_M4.md` (LOCKED — context + M1–M4)*
*Companion Part 2B: `completed_modules_M6p5_to_invariants.md` (M6.5 + audit + M6.5r + invariants)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
