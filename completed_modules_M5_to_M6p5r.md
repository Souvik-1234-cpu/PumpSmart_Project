# PumpSmart — Completed Modules Reference: M5 to M6B
 
**Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring**
**PART 2A OF 3 — M5 results + M6A results + M6B 22-class spec (v14.2)**
 
| Field | Value |
|-------|-------|
| Status | M5, M6A sections LOCKED. M6B section = SPEC ONLY — NOT YET EXECUTED. |
| Updated | 2026-04-19 |
| Author | Souvik |
 
**This file contains:**
- M5 physics engine results + paste keys (LOCKED)
- M6A synthetic generator results + paste keys (LOCKED — superseded by M6B)
- M6B 22-class expanded synthetic dataset SPEC + paste keys (NEXT ACTIVE — script not yet run)
- v14.2 additions: TCN-AE Level 2 replaces LSTM v2; corrected sequence lengths and counts per physics audit; M6A rerun status per label; M6.5 superseded by M6.5r; architecture decision record added
**Companion files:**
- `completed_modules_context_and_M1_to_M4.md` → Part 1 LOCKED: context, M1–M4 results
- `completed_modules_M6p5_to_invariants.md` → Part 2B: M6.5, audit findings, M6.5r, invariants
> **Canonical source of truth:** `pasted-text.txt` (v14.2, 2026-04-19)
> GitHub is the ONLY source of truth. Spaces `.md` files are OUTDATED — do not use.
 
---
 
## M5 — Physics Engine
 
**Status: COMPLETED (2026-03-29) — LOCKED**
 
### Key Results
 
| Metric | Value |
|--------|-------|
| Equations implemented | 20 |
| Nameplate pass count | 20/20 |
| S5 validation cases | 19/19 PASS |
| S7 fault sequence tests | 24/26 PASS |
| Known failures | `bearing_wear_steady_state_s100`, `overloading_steady_state_s140` |
| Failure reason | Edge-case severity values outside declared production range |
| Bearing heat model | Euler integration (dt steps) — PATCH 3 applied |
| Sensor failure spike guard | +/−5 sigma clamp — PATCH 5 applied |
| Overloading severity range | [0.5, 1.0] — mild overloading handled via sub-cluster in M6B |
 
### Thermal Coupling Validation (M5 Plots Confirmed — LOCKED)
 
| Fault / Cluster | r(Mot.TV\*, Temp.SV\*) | Physics Interpretation |
|----------------|----------------------|----------------------|
| Bearing wear (steady_state) | 0.972 | Coupling PRESERVED — heat from bearing |
| Overloading (steady_state) | 0.997 | STRONGLY PRESERVED — thermal overload |
| Seal failure (steady_state) | −0.013 | Coupling BROKEN — hydraulic fault |
| Cavitation (startup) | 0.376 | WEAK — hydraulic, not thermal |
| Bearing wear (high_load) | 0.949 | Preserved |
| Normal (steady_state) | −0.062 | Baseline near-zero (correct) |
 
### 20 Physics Equations Validated Against Nameplate
 
All 20 equations validated at: **110 kW, 45 m³/h, 450 m head, 40 bar, 7 impellers, 2980 RPM**
 
Key equations include:
 
| Equation | Formula |
|----------|---------|
| Hydraulic power | `P_hyd = rho*g*Q*H / eta` |
| Specific speed | `Ns = N*sqrt(Q) / H^(3/4)` |
| NPSH available | `NPSHa = (Ps - Pv)/(rho*g) + Vs^2/(2g)` |
| Affinity laws | Q ∝ N, H ∝ N², P ∝ N³ |
| Joukowsky pressure surge | `dP = rho × a × dV` |
| ISO 10816-3 vibration severity zones | — |
| Bearing heat generation | `Q_bear = mu × F × v` |
| Thermal rise | `dT/dt = Q_bear / (m × Cp)` [Euler integration] |
 
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
 
| Key | Value |
|-----|-------|
| `M5_equations` | 20 |
| `M5_nameplate_pass` | 20/20 |
| `M5_s5_cases` | 19/19 PASS |
| `M5_s7_faults` | 24/26 PASS |
| `M5_known_failures` | `bearing_wear_steady_state_s100`, `overloading_steady_state_s140` |
| `M5_patch3` | Euler integration bearing heat |
| `M5_patch5` | +/−5 sigma sensor failure spike guard |
| `M5_overload_range` | [0.5, 1.0] production (mild via M6B sub-cluster) |
 
---
 
## M6A — Synthetic Dataset Generator (Hybrid) — SUPERSEDED
 
**Status: COMPLETED (2026-04-11) — SUPERSEDED BY M6B**
 
> M6A produced 7 classes, 8,400 sequences. It has been **SUPERSEDED by M6B** (22 classes, Groups A–E, ~31,800 sequences). M6B is the dataset used for M7, M8, and all downstream modules. M6A results are LOCKED for reference only.
 
> **M6.5 SUPERSEDED STATUS:** M6.5 produced a feature matrix from M6A 8,400 sequences (26 features). M6.5 is FULLY SUPERSEDED by M6.5r which operates on M6B ~31,800 sequences with ~35 features (adds z_t features + score_A/B/C from TCN-AE). Do not reference M6.5 outputs for any downstream module.
 
### M6A Architecture Decision: HYBRID PATH C (LOCKED)
 
| Source | Description |
|--------|-------------|
| **SOURCE 1** | Real CIRA Normal Windows (M3 normalized pool) → 1,200 Type-A normal sequences |
| **SOURCE 2** | M4 Spike Seeds (pseudo-labelled real CIRA fault windows) → cosine similarity > 0.85 → fault onset seeds t=0 to t=49 → M5 physics continues t=50 to t=199 |
| **SOURCE 3** | Physics Synthetic (M5 engine — pure) → fills gaps, covers full severity spectrum [0.2 → 1.0] |
 
### M6A Sequence Count (LOCKED — reference only)
 
| Class | Count |
|-------|-------|
| normal | 1,200 |
| bearing_wear | 1,200 |
| impeller_imbalance | 1,200 |
| cavitation | 1,200 |
| seal_failure | 1,200 |
| overloading | 1,200 |
| sensor_failure | 1,200 |
| **TOTAL** | **8,400** |
 
### M6A Outputs (LOCKED — archived, not used downstream)
 
```
data/synthetic/M6A_sequences.pkl         ← 8,400 sequences, shape each (200, 8)
data/synthetic/M6A_sequence_meta.csv     ← seq_id, label, severity, source, cluster, seed_idx
data/synthetic/M6A_validation_report.json
outputs/reports/module_06a_synthetic_report.md
```
 
### Paste Text Keys (LOCKED — reference only)
 
| Key | Value |
|-----|-------|
| `M6A_total_sequences` | 8,400 |
| `M6A_sequences_per_class` | 1,200 |
| `M6A_classes` | 7 (labels 0–6) |
| `M6A_status` | SUPERSEDED by M6B — not used in M7/M8 downstream |
 
---
 
## M6B — Expanded Synthetic Dataset (22-Class, Groups A–E) — v14.2
 
**Status: NEXT ACTIVE — SPEC LOCKED (v14.2), SCRIPT NOT YET RUN**
 
> **NOTE:** M6B has NOT been executed. No output files exist yet. `fault_rules_v3.json`, `M6B_*.pkl`, `M6B_sequence_meta.csv`, `M6B_feature_matrix.csv` are ALL pending — they will be created when the M6B script runs. This section is the LOCKED SPEC that governs the M6B script.
 
### Why M6B Was Created
 
M6A had 7 classes (labels 0–6) — adequate for basic fault detection but insufficient for:
 
1. Compound fault scenarios (two faults active with causal lag)
2. Primary-channel masked faults (sensor dead, secondary path only)
3. Fault severity variants (fast vs slow, intermittent vs sustained)
4. Slowly-progressing faults below detection threshold (pre-WATCH zone)
5. Multi-sensor anomaly scenarios (2 sensors degraded simultaneously)
6. M7 XGBoost training on M6A alone = limited real-world coverage
7. M8 adversarial validation requires compound + masked + variant scenarios
M6B adds Groups B, C, D, E to the base Group A (M6A classes). All 22 classes will be defined in `fault_rules_v3.json` — written by M6B Step 3.
 
### Architecture Decision Record: TCN-AE Selected for Level 2
 
| Field | Value |
|-------|-------|
| Decision date | 2026-04-19 |
| Previous design | Hierarchical LSTM v2 (now SUPERSEDED) |
| Selected design | 5-layer dilated causal TCN-AE |
 
**Reasoning:**
 
1. **Inter-window amnesia:** LSTM v2 hidden state resets at each z_t window boundary. For compound faults (lag 400–800 steps = 8–16 windows), the chain transition signal is completely lost across window boundaries. TCN has no hidden state — receptive field spans all 63 windows simultaneously.
2. **Vanishing gradient:** For Label 21 (`bearing_wear_gradual`), N_windows = 20 windows minimum. LSTM gradient path = 0.9²⁰ ≈ 0.12 (12% signal remaining at step 1). TCN gradient path = constant 5 layers regardless of sequence length.
3. **Transformer rejected:** N_windows = 6–20 is too short for attention weights to be meaningful. Attention requires long sequences to distinguish relevant vs irrelevant positions. At N=6, attention degenerates to near-uniform weights.
4. **Receptive field:** 5 layers, dilation=[1,2,4,8,16], kernel=3
   - `RF = 1 + (3-1) × (1+2+4+8+16) = 63 windows = 3,150 raw seconds at 1 Hz`
   - Covers the longest compound chain (Label 10: 400–800 step lag) with margin.
**TCN-AE outputs:**
 
| Score | Description |
|-------|-------------|
| `score_A` | Severity (reconstruction error magnitude) |
| `score_B` | Drift slope (temporal trend in z_t space) |
| `score_C` | Chain transition score (compound fault signal) |
 
**Score routing rules (INVARIANT 19 — NEVER CROSS):**
 
| Score | Routed To |
|-------|-----------|
| `score_B` | CUSUM only |
| `score_A` | Rolling baseline only |
| `score_C` | XGBoost only |
 
### 22-Class Label Map — v14.2 (LOCKED)
 
> Canonical map. Corrected sequence lengths and counts per 2026-04-19 physics audit.
 
| Group | Label | Class Name | Seq Length | Count | M6A Status | Description |
|-------|-------|-----------|-----------|-------|-----------|-------------|
| A | 0 | `normal` | 200 | 2,000 | VALID | Real CIRA normal windows |
| A | 1 | `bearing_wear` | 250 | 1,500 | RERUN M6B Step 0 | Progressive mechanical degradation (Paris law) |
| A | 2 | `impeller_imbalance` | 200 | 1,500 | VALID | Rotodynamic imbalance |
| A | 3 | `cavitation` | 150 | 1,500 | VALID | Hydraulic shock — startup ONLY |
| A | 4 | `seal_failure` | 400 | 1,500 | RERUN M6B Step 0 | Progressive pressure loss |
| A | 5 | `overloading` | 300 | 1,500 | RERUN M6B Step 0 | Thermal overload — steady_state ONLY |
| A | 6 | `sensor_failure` | 150 | 1,200 | VALID | Single-channel flatline/spike/drift |
| B | 7 | `bearing_wear->overloading` | 600 | 1,500 | — | Compound: MotSV first, TempSV+MotTV after. Lag 200–400s |
| B | 8 | `cavitation->seal_failure` | 550 | 1,500 | — | Compound: PmpSV spikes first, PresSV drop after. Lag 50–150s |
| B | 9 | `impeller_imbalance->bearing_wear` | 700 | 1,500 | — | Compound: PmpSV BPF first, MotSV Paris-law drift after. Lag 300–600s |
| B | 10 | `seal_failure->cavitation` | 900 | 1,500 | — | Compound: PresSV drops first, PmpSV spikes after. Lag 400–800s |
| B | 11 | `overloading->bearing_wear` | 800 | 1,500 | — | Compound: TempSV rises first, MotSV drift begins. Lag 400–600s |
| B | 12 | `impeller_imbalance->cavitation` | 450 | 1,500 | — | Compound: PmpSV BPF first, PresSV erratic + spikes. Lag 100–300s |
| C | 13 | `bearing_wear_MotSV_masked` | 300 | 1,200 | — | MotSV flatlined — detect via MotTV + TempSV thermal lag |
| C | 14 | `cavitation_PresSV_masked` | 210 | 1,200 | — | PresSV dropout — detect via PmpSV kurtosis bursts |
| C | 15 | `seal_failure_PresSV_drifting` | 500 | 1,200 | — | PresSV drifting — detect via secondary hydraulic channels |
| C | 16 | `overloading_TempSV_stuck` | 350 | 1,200 | — | TempSV stuck — detect via MotTV (r=0.997 coupling) |
| C | 17 | `impeller_imbalance_PmpSV_flatline` | 250 | 1,200 | — | PmpSV flatline — detect via PmpPV + cross-channel |
| D | 18 | `cavitation_intermittent` | 300 | 1,200 | — | NPSHa oscillates around NPSHr boundary — burst pattern |
| D | 19 | `seal_failure_fast` | 150 | 800 | — | Rapid hydraulic discharge; PresSV drops in ≤20 steps → DANGER |
| D | 20 | `overloading_cyclic` | 600 | 1,200 | — | Thermal sawtooth with rising baseline; 3 cycles × ~150s rise |
| D | 21 | `bearing_wear_gradual` | 1,000 | 2,000 | — | Paris-Erdogan low dK. MotSV barely above baseline. CUSUM+Layer4 required. |
| E | [fault_rules_v3.json] | `sensor_failure_2ch_thermal` | 250 | 800 | — | MotTV + TempSV simultaneously degrade (common rail failure) |
| E | [fault_rules_v3.json] | `sensor_failure_2ch_pump` | 250 | 800 | — | PmpSV + PmpPV simultaneously degrade (junction box moisture ingress) |
 
> **NOTE:** Group E exact label integers assigned in `fault_rules_v3.json` by M6B Step 3. Total 22 classes = labels 0–21. Do not hardcode Group E label numbers from this file.
 
**M6A rerun status detail:**
 
Labels 1, 4, 5 require rerun — old sequences were 200 steps (physically wrong):
- **Label 1:** 200 → 250 steps (Paris law rise takes 150s to reach 2× baseline)
- **Label 4:** 200 → 400 steps (orifice leak at 40 bar takes 300s for 15% Pres drop)
- **Label 5:** 200 → 300 steps (thermal mass thermal runaway takes 200s to Temp\* > 1.3)
Labels 0, 2, 3, 6 + normal windows: VALID — carry forward from M6A unchanged.
 
### M6B Sequence Counts — v14.2 Targets
 
| Group | Classes | Sequences per Class | Group Total | Labels | Status |
|-------|---------|-------------------|-------------|--------|--------|
| A — Normal | 1 | 2,000 | 2,000 | 0 | VALID from M6A |
| A — Single faults | 6 | 1500/1500/1500/1500/1500/1200 | 9,200 | 1–6 | 1,4,5: RERUN; 2,3,6: VALID |
| B — Compound chains | 6 | 1,500 each | 9,000 | 7–12 | PENDING |
| C — Masked faults | 5 | 1,200 each | 6,000 | 13–17 | PENDING |
| D — Severity variants | 4 | 1200/800/1200/2000 | 5,200 | 18–21 | PENDING |
| E — Multi-sensor | 2 | 800 each | 1,600 | 0–21 | PENDING |
| **TOTAL** | **22** | — | **~31,800** | **0–21** | **PENDING** |
 
**RAM check:**
- Worst-case: ~31,800 × max_steps(900) × 8 × float32 ≈ **825 MB**
- Average steps ~400 → **~367 MB** average — well within 16 GB RAM
- M6.5r feature matrix target: **~196,000 rows × ~35 columns** → M7 input (22-class)
- VRAM: M4 sliding window `batch_size=32` → ~2 GB VRAM within 8 GB RTX 4060
> **NOTE:** NONE OF THESE FILES EXIST YET — created by M6B script:
> - `data/synthetic/M6B_combined_sequences.pkl` — all groups → full fault validation pool
> - `data/synthetic/M6B_feature_matrix.csv` — ~196,000 rows × ~35 columns (from M6.5r)
> - `models/fault_rules_v3.json` — written by M6B Step 3 (22-class, labels 0–21)
 
### M6B Group B — Compound Chain Fault Physics Rules
 
All Group B sequences: two faults active with `secondary_onset_lag` separation.
- **Phase 1:** primary fault only (t=0 to secondary_onset_lag)
- **Phase 2:** both faults active simultaneously (t=secondary_onset_lag to end)
- `secondary_onset_lag`: drawn from physics-verified fault-specific range (see table above)
- Sequences per class: 1,500
Each compound chain = UNIQUE INTEGER LABEL (single-label XGBoost). M10 API maps label → "Primary: X → Secondary: Y" in UI display.
 
**Physics-verified lag derivations (v14.2):**
 
| Label | Lag | Physics Derivation |
|-------|-----|-------------------|
| **7** (lag 200–400s) | Bearing heat → oil viscosity drop → friction torque → thermal runaway | Thermal runaway from bearing heat takes 200–400s at 1 Hz. Old blanket 30–80 steps = physically impossible. |
| **8** (lag 50–150s) | Joukowsky shock (`dP = rho × a_wave × dV = 19.1 bar`) → axial thrust → seal face damage | Acute mechanical event: 50–150s physically correct. |
| **9** (lag 300–600s) | BPF radial load → Paris law fatigue crack (`da/dN = C*dK^m, m=3 steel`) | Paris fatigue crack propagation to detectable level: 300–600s. Old blanket 30–80 = completely wrong for fatigue mechanism. |
| **10** (lag 400–800s) ⚠️ | `Q_leak = Cd × A_gap × sqrt(2*dP/rho)` → Q_leak ~0.0019 m³/s at 40 bar → operating point shifts: 45 → ~40 m³/h → internal recirculation → hf_suction rises 1.5–2.5 m → NPSHa crosses NPSHr ~4.5 m (7-stage, 2980 RPM) | NPSHa margin loss over 400–800s at 1 Hz. Old 100–200 step spec = physically impossible. Cavitation CANNOT onset in <100s via this hydraulic pathway. |
| **11** (lag 400–600s) | Thermal creep → bearing metal fatigue → load increase | 400–600s thermal creep to reach bearing load threshold. Old 40–80 steps = physically impossible. |
| **12** (lag 100–300s) | BPF pressure oscillation → local low-pressure zone at blade tips → bubble nucleation | 100–300s for BPF to establish persistent low-P zone. |
 
> **Expected detection:** DANGER state within 200 windows (Gate M8-14: ≥85% TPR)
 
### M6B Group C — Masked Fault Physics Rules
 
All Group C sequences: primary detection channel = constant (sensor failed). `masked_channel_flag = True` in metadata. Detection MUST route via secondary channel path.
 
| Label | Class | Secondary Detection Path |
|-------|-------|--------------------------|
| 13 | `bearing_wear_MotSV_masked` | MotTV + TempSV drift (thermal lag 20–40 steps) |
| 14 | `cavitation_PresSV_masked` | PmpSV kurtosis bursts (BPF + hydraulic) |
| 15 | `seal_failure_PresSV_drift` | Secondary hydraulic channels (PmpSV, PmpPV cross-channel) |
| 16 | `overloading_TempSV_stuck` | MotTV drift (r=0.997 coupling PRESERVED) |
| 17 | `impeller_PmpSV_flatline` | PmpPV + cross-channel correlation change |
 
> Max achievable alert state = **WARN** (not DANGER) if secondary signal only.
> Gate M8-13: Group C TPR ≥ 65% via secondary path.
 
### M6B Group D — Severity Variant Physics Rules
 
**Label 18: `cavitation_intermittent`**
- NPSHa oscillates around NPSHr boundary
- PmpSV burst pattern: high erratic during bursts, near-normal between
- `burst_interval` drawn from Uniform(15, 30) steps
- MAE must stay ABOVE threshold even in low-NPSHa oscillation phase (Finding 5)
- Mech B slope NOT monotonic; `burst_count` tracker required in M8
**Label 19: `seal_failure_fast`**
- Rapid hydraulic discharge through enlarged effective seal leak area
- PresSV drops in ≤20 steps to minimum
- Single-window MAE fires immediately → DANGER within 1–3 windows
- Must show faster PresSV drop than standard `seal_failure` (Finding 2 constraint)
- Governing equation: `Q_leak = Cd × A_orifice × sqrt(2 × dP / rho)`
- Do NOT model with Hagen-Poiseuille; seal blowout is not laminar pipe flow
**Label 20: `overloading_cyclic`**
- Thermal sawtooth: Temp.SV load ON/OFF with RISING baseline across cycles
- Each cycle starts higher than previous — `cyclic_baseline_drift > 0.0002/window`
- Temp.SV Spearman > 0.70 on baseline-detrended signal
- TempSV sawtooth steeper than standard overloading — accumulator fires within 15 min
**Label 21: `bearing_wear_gradual`**
- Paris-Erdogan crack growth with SMALL dK (low stress intensity range): `da/dN = C × dK^m` [same equation as label 1, smaller dK input]
- MotSV rises BARELY above baseline over 150+ steps
- Weibull beta=1.5, severity=0.05–0.25 (low end of crack growth spectrum)
- CIRA anchor: same 44 bearing-impact spike seeds as label 1 (fully anchored)
- Primary discriminator: `err_slope_MotSV` (small, consistent positive slope)
- Sequences at severity < 0.15 will have MAE < 0.110058 — PHYSICALLY CORRECT (fault genuinely below alarm level; CUSUM/rolling accumulator catches it)
- Target: 2,000 sequences (highest count — hardest class)
- G11-ext: `err_slope_MotSV > 0` in ≥95% of label 21 sequences
- XGBoost output: *"bearing_wear_gradual — plan inspection within 7–14 days"*
**Detection path for Label 21:**
 
| Layer | Detection |
|-------|-----------|
| Layer 4 (Rolling Baseline) | Slope shift ~Week 5 (pre-threshold) |
| Layer 3 (CUSUM on score_B) | Alert ~Week 5.5 (~30% bearing degraded) |
| Layer 2 (TCN-AE score_B) | Drift accumulation across Glass 5 (d=16) |
| Layer 1 (LSTM-AE) | Threshold crossing ~Week 7 (too late alone) |
 
> ⚠️ **LSTM-AE 50-step window ALONE is INSUFFICIENT for label 21.** Requires all 4 layers.
> **DO NOT raise global MAE threshold 0.110058 to compensate for label 21.**
 
### M6B Group E — Multi-Sensor Anomaly Physics Rules
 
> Labels assigned in `fault_rules_v3.json` (written by M6B Step 3). Do not hardcode.
 
**`sensor_failure_2ch_thermal`:**
- Both MotTV + TempSV simultaneously degrade (flatline/drift)
- Physically: common thermal measurement system failure (shared excitation rail)
- `multi_sensor_anomaly_count = 2` in M6.5r features
**`sensor_failure_2ch_pump`:**
- Both PmpSV + PmpPV simultaneously degrade
- Physically: moisture ingress to pump-side junction box
- Motor-side sensors (MotPV, MotSV, MotTV) remain normal
- `multi_sensor_anomaly_count = 2` in M6.5r features
> Gate G11: exactly 2 channels anomalous; remaining 6 within ±0.20 normalized baseline.
> Gate M8-14: Group E TPR ≥ 88% for `multi_sensor_count = 2` detection.
> Target: 800 sequences per variant = 1,600 sequences total.
 
### z_t Export Requirements — All Groups
 
For every sequence in every group, after synthetic generation:
 
1. Run M4 LSTM-AE (`lstm_ae_baseline.pth`, FROZEN) in sliding window mode
2. Per 50-step window: export z_t in ℝ⁶⁴ + MAE in ℝ⁸
3. Stack per sequence: shape (N_windows × 64) where N_windows = steps / 50
4. Save per group:
```
z_t_sequences_groupA_normal.pkl
z_t_sequences_groupA_faults.pkl
z_t_sequences_groupB.pkl
z_t_sequences_groupC.pkl
z_t_sequences_groupD.pkl
z_t_sequences_groupE.pkl
```
 
> These z_t sequences feed Level 2 TCN-AE (M8). **Raw sensor data NEVER enters Level 2 directly.** (Invariant 16)
 
### M6B 4-Step Script Structure (SPEC LOCKED — not yet run)
 
**Step 0 (PENDING — NEW v14.2):**
- Re-generate Group A Labels 1, 4, 5 at corrected lengths (250, 400, 300 steps)
- Carry forward Labels 0, 2, 3, 6 + normal windows from M6A unchanged
- Run M4 LSTM-AE sliding window → export `z_t_sequences_groupA_*.pkl`
- Output: `data/synthetic/M6B_sequences_groupA.pkl`
**Step 1 (PENDING):**
- Group B — 6 compound chains (labels 7–12)
- Validation gates G8 (temporal lag correct), G9 (causal channel order)
- Run M4 sliding window → export `z_t_sequences_groupB.pkl`
- Output: `data/synthetic/M6B_sequences_groupB.pkl`
**Step 2 (PENDING):**
- Group C — 5 masked faults (labels 13–17)
- Group D — 4 severity variants (labels 18–21, incl. `bearing_wear_gradual`)
- Group E — 2 multi-sensor failures
- Validation gates G10 (masked secondary signal), G11 (severity MAE)
- G11-ext for label 21: `err_slope_MotSV > 0` in ≥95% of sequences
- Run M4 sliding window → export `z_t_sequences_groupC/D/E.pkl`
- Outputs: `data/synthetic/M6B_sequences_groupC.pkl`, `...groupD.pkl`, `...groupE.pkl`
**Step 3 (PENDING):**
- Full merge: M6B Step 0 Group A + M6B Groups B+C+D+E
- Writes: `models/fault_rules_v3.json` (22-class, labels 0–21) — LOCKED after run
- Full validation suite (physics, coupling, MAE, temporal gates)
- Writes: `outputs/reports/module_06B_synthetic_report.md`
- Outputs: `data/synthetic/M6B_combined_sequences.pkl`, `data/synthetic/M6B_sequence_meta.csv`
**LOCKED FILES — DO NOT OVERWRITE:**
```
models/fault_rules.json              (v1 — M5/M6A reference, frozen)
data/synthetic/M6A_*                 (frozen after M6A completion)
models/lstm_ae_baseline.pth          (frozen Level 1 weights)
models/M4_threshold_config.json      (threshold=0.110058 — Level 1 only, NEVER change)
```
 
### M6B Planned Outputs (NONE EXIST YET — written when M6B script runs)
 
```
data/synthetic/M6B_sequences_groupA.pkl      ← ~11,200 Group A seqs (Step 0 output)
data/synthetic/M6B_sequences_groupB.pkl      ← ~9,000 Group B compound (1,500 × 6)
data/synthetic/M6B_sequences_groupC.pkl      ← ~6,000 Group C masked (1,200 × 5)
data/synthetic/M6B_sequences_groupD.pkl      ← ~5,200 Group D variants
data/synthetic/M6B_sequences_groupE.pkl      ← ~1,600 Group E multi-sensor (800 × 2)
data/synthetic/M6B_combined_sequences.pkl    ← ALL groups merged → M8 fault validation pool
data/synthetic/M6B_sequence_meta.csv         ← seq_id, label, group, severity, cluster, source
z_t_sequences_groupA_normal.pkl              ← z_t exports per group
z_t_sequences_groupA_faults.pkl
z_t_sequences_groupB.pkl
z_t_sequences_groupC.pkl
z_t_sequences_groupD.pkl
z_t_sequences_groupE.pkl
models/fault_rules_v3.json                   ← 22-class label map (written by Step 3)
outputs/reports/module_06b_synthetic_report.md
```
 
### M6B Paste Text Keys (Populate AFTER script runs — do not fill in advance)
 
| Key | Value |
|-----|-------|
| `M6B_total_sequences` | [fill after run — target ~31,800] |
| `M6B_classes` | 22 (labels 0–21, Groups A–E) |
| `M6B_group_A_sequences` | [fill — target ~11,200] |
| `M6B_group_B_sequences` | [fill — target ~9,000] |
| `M6B_group_C_sequences` | [fill — target ~6,000] |
| `M6B_group_D_sequences` | [fill — target ~5,200] |
| `M6B_group_E_sequences` | [fill — target ~1,600] |
| `M6B_label21_sequences` | [fill — target 2,000] |
| `M6B_label21_slope_gate` | [fill — `err_slope_MotSV > 0` in ≥95% seqs] |
| `M6B_fault_rules_version` | [fill — `fault_rules_v3.json` written in Step 3] |
| `M6B_physics_violations` | [fill — expect NONE] |
| `M6B_coupling_fidelity_pass` | [fill from M6B run log] |
| `M6B_mae_gate_B_pass` | [fill from M6B run log] |
| `M6B_mae_gate_C_pass` | [fill from M6B run log] |
| `M6B_combined_output` | `data/synthetic/M6B_combined_sequences.pkl` |
| `M6B_meta_output` | `data/synthetic/M6B_sequence_meta.csv` |
| `M6B_zt_exports` | `z_t_sequences_group[A-E].pkl` (6 files) |
| `Status_for_M6p5r` | READY / BLOCKED |
 
---
 
## Four-Layer Detection Architecture — v14.2
 
**Governs M8 + M10 design. LOCKED.**
 
Four detection layers run in cascade across M8 (training) and M10 (runtime):
 
### Layer 1 — LSTM-AE 50-step Window (M8)
 
| Property | Value |
|----------|-------|
| Detects | "Is THIS window anomalous?" |
| Memory | 50 steps only (encoder hidden state resets each window) |
| Blind to | Cross-window trends |
| Threshold | 0.110058 (LOCKED — static, Level 1 only) |
| Output | Per-channel MAE + z_t in ℝ⁶⁴ |
 
### Layer 2 — TCN-AE (M8 Level 2) ⭐ *REPLACES LSTM v2 — v14.2*
 
| Property | Value |
|----------|-------|
| Input | z_t sequences (N_windows × 64) — NEVER raw sensor data |
| Architecture | 5-layer dilated causal TCN-AE; dilation=[1,2,4,8,16], kernel=3 |
| Receptive field | RF = 1 + (3-1)×(1+2+4+8+16) = 63 windows = 3,150 raw seconds |
| Detects | Cross-window patterns — compound chains, drift trends, chain transitions |
| Output | score_A (severity), score_B (drift slope), score_C (chain transition) |
| Catches | Compound faults (Glass 3–5), gradual bearing (Glass 5, d=16) |
 
### Layer 3 — CUSUM Runtime State (M10)
 
| Property | Value |
|----------|-------|
| Formula | `S_n = max(0, S_{n-1} + (mae_channel_n - mu0) - k)` |
| Operates on | `score_B` (drift slope from TCN-AE) — NOT raw MAE |
| Reference | `mu0` = M3 cluster baseline MAE per channel (READ-ONLY) |
| Fires when | S_n exceeds control limit (default=5.0) |
| Catches | Label 21 at ~Week 5.5 (~30% bearing degraded) |
| State | PERSISTENT across API calls in M10 runtime memory |
| Resets | Only on explicit operator acknowledge or pump restart |
 
### Layer 4 — Rolling Baseline Comparator (M10)
 
| Property | Value |
|----------|-------|
| Operates on | `score_A` (severity from TCN-AE) |
| Adaptive threshold | `theta_t = mu_rolling(6hr) + 3*sigma_rolling(6hr)` |
| Updates | Every 50 seconds in M10 runtime |
| Catches | Pre-threshold drift weeks before Layer 1 fires |
| Physics | Rising rolling slope = da/dN trend (Paris law) — ISO 7870 Shewhart chart |
| State | PERSISTENT across API calls in M10 runtime memory |
 
### Score Routing Rules (INVARIANT 19 — NEVER CROSS)
 
| Score | Routed To |
|-------|-----------|
| `score_B` | CUSUM only |
| `score_A` | Rolling baseline only |
| `score_C` | XGBoost only |
 
### Two-Speed Adaptation Design
 
| Speed | Mechanism | Purpose |
|-------|-----------|---------|
| Fast (6hr rolling) | Controls false alarm rate | Adapts to operating point shifts |
| Slow (CUSUM weeks) | Detects secular drift | Immune to baseline creep |
 
Both are needed simultaneously — **adaptive threshold paradox for Label 21:**
- If threshold adapts too fast → CUSUM accumulation zeroed → Label 21 missed
- If threshold is fully static → false alarms on load changes
- **Solution:** `score_A` (fast rolling) handles load changes; `score_B` fed to CUSUM (slow) handles Paris-law drift independently
> **Why CUSUM was NOT added to M6.5r feature matrix:** M6.5r generates features per-window independently. CUSUM at window [100:150] needs MAE history from windows [0:100]. In deployment, CUSUM accumulates across streaming windows indefinitely. Train (sequence-internal CUSUM) ≠ Deploy (cross-session CUSUM) → train-serve skew. Therefore: CUSUM belongs in M10 runtime only. Feature matrix unchanged at ~35 cols.
 
### Three M10 API Output States
 
| State | Fields | Action |
|-------|--------|--------|
| 1 | `fault_stage=early`, `fault_type=unknown` | MONITOR |
| 2 | `fault_stage=developing`, `compound=False` | ALERT (single label) |
| 3 | `fault_stage=developing`, `compound=True` | CRITICAL — `causal_chain=[e.g. bearing_wear->overloading]` |
 
---
 
## Document Revision History
 
| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic file `completed_modules_M1_to_M6p5.md` |
| v2.0 | 2026-04-15 | SPLIT into Part 1 + Part 2. Added M6B 21-class spec, M6.5r 26-feature spec |
| v3.0 | 2026-04-15 | CORRECTION: M6B/M6.5r status, Group E labels, seq counts, PENDING markers |
| v4.0 | 2026-04-16 | v14.0 UPGRADE: Split into Part 2A + Part 2B. Label 21 added. 22 classes. 4-layer detection architecture added. CUSUM Layer 3 + Rolling Baseline Layer 4. |
| v5.0 | 2026-04-18 | v14.1 CORRECTION: Group B/C canonical map reaffirmed. Group E pump-side pair confirmed as PmpSV+PmpPV. `seal_failure_fast` clarified. |
| v6.0 | 2026-04-19 | v14.2 PHYSICS AUDIT: All 22 sequence lengths corrected per first-principles derivation. All 22 sequence counts updated → ~31,800 total. Group B lags corrected from blanket 30–80 to physics-verified per-label. Label 10 lag: 100–200 → 400–800 (CRITICAL correction). M6A rerun status added per label (1,4,5: RERUN; 0,2,3,6: VALID). Step 0 added to M6B script structure for Group A reruns. TCN-AE Level 2 replaces LSTM v2 (architecture decision record added). M6.5 marked SUPERSEDED by M6.5r. z_t export requirements added to M6B spec. Score routing rules (score_A/B/C) added. Two-speed adaptation explanation added. Paste keys updated with new targets and z_t file list. |
 
---
 
> **GitHub is the ONLY source of truth for this spec.**
>
> Companion Part 1: `completed_modules_context_and_M1_to_M4.md` (LOCKED — context + M1–M4)
> Companion Part 2B: `completed_modules_M6p5_to_invariants.md` (M6.5 + audit + M6.5r + invariants)
>
> **Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset
> **Standards:** ISO 10816-3 vibration, ISO 13373-3 condition monitoring
> **Canonical source of truth:** `pasted-text.txt` (v14.2, 2026-04-19)
