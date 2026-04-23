# PumpSmart — M6: Synthetic Dataset Generation
## Physics-Informed Synthetic Fault Sequences

| Field | Value |
|-------|-------|
| **Document version** | v3.0 — Architecture v14.2 (TCN-AE Level 2 + 22-class fault universe) |
| **Date** | 2026-04-23 |
| **Supersedes** | v2.0 (arch v14.0, 7-class, 8,400 sequences, 200-step uniform) |

---

> ⚠️ **STATUS: PARTIALLY SUPERSEDED BY M6B**
>
> - M6A Group A Labels **0, 2, 3, 6** (normal, impeller_imbalance, sensor_failure) → **VALID** — no rerun required
> - M6A Group A Labels **1, 4, 5** (bearing_wear, seal_failure, overloading) → **RERUN REQUIRED** in M6B Step 0
>   - Reason: sequence lengths corrected (250/400/300 steps)
>   - Original 200-step versions must **NOT** be used downstream
> - M6.5 feature matrix (7-class, 25 features) → **SUPERSEDED**
>   - Replaced by M6.5r (22-class, ~35 features, zt + score_A/B/C)

---

### Pump Nameplate (CIRA SACIP)

| Parameter | Value |
|-----------|-------|
| Motor shaft power | 110 kW, IEC Frame 315mm, 400V, 2-pole |
| Speed | 2980 RPM |
| Stages | 7 impellers (multistage centrifugal) |
| Flow rate | 45 m³/h |
| Total head | 450 m |
| Max pressure | 40 bar |
| Hydraulic kW | ~55 kW (η × ρgQH) |

---

## Purpose and Relationship to Other Modules

- **M5 Physics Engine** encodes the causal physics of each fault type as Python functions that generate physically valid normalised time-series.
- **M6** calls M5 to build the full labelled synthetic dataset that trains M7 (XGBoost classifier) and provides the validation set for M8 (TCN-AE Level 2).
- **M6.5r** extracts features from all M6 sequences via M4 LSTM-AE inference, producing the ~35-feature matrix fed to M7 XGBoost. score_A, score_B, score_C from TCN-AE Level 2 are included as direct features once M8 is complete.
- **M8 TCN-AE Level 2** is trained on zt sequences exported by M6 per group.

**Relationship to M12:**
- M6 sequences are **SEEN** by M8 during training.
- M12 generates **COMPLETELY FRESH** sequences (parametrically different).
- M12 is the adversarial held-out test — model must **NEVER** see M12 data before M12 validation runs.

---

## Inviolable Rules — Apply to Every Sequence in M6

1. **Normalisation LOCKED** at `M3_normalization_config.json`. Raw sensor values NEVER enter any sequence as-is. All values are dimensionless ratios (P\*, a\*, ΔT\*).

2. **Windows NEVER cross segment boundaries.** Each sequence starts from a cluster centroid — not a real segment boundary.

3. **Cavitation sequences ONLY in startup cluster context.** NPSHa is marginal only at startup (P_suction = 0.43–0.85 bar).

4. **Overloading sequences ONLY in steady-state cluster context.** Thermal run-in makes high-load cluster temperatures paradoxically low.

5. **Seal failure:** Pres.SV\* decline is ALWAYS negative (pressure loss). Sensor drift (Label 13): Pres.SV\* drift direction context-dependent. Upward Pres.SV\* drift → sensor_drift, not seal_failure. **NEVER confused.**

6. **Conservation of energy and mass** must hold in ALL sequences.
   - No negative pressure (Pres.SV\* ≥ 0 at all timesteps)
   - No temperature below ambient (ΔT\* ≥ −0.1 acceptable; < −0.5 = error)
   - No SV\* > 5.0 (physical ceiling from M4 spike reference: 12.6 normalised)

7. **Thermal coupling** r(Mot.TV, Temp.SV) > 0.85 in ALL bearing_wear and overloading sequences. This coupling is PRESERVED by physics (same thermal mass), not enforced artificially.

8. **segment_id preserved** in all downstream dataframes through M6.

---

## M6A — Original 7-Class Dataset

*Completed 2026-04-11. Architecture: v14.0 (pre-TCN-AE). Hybrid Path C locked 2026-04-08.*

### Source Architecture — Hybrid Path C (LOCKED)

- **Source 1:** Real CIRA normal windows from M3 pool (1,200 Type-A sequences)
- **Source 2:** M4 Spike Seeds (cosine sim ≥ 0.85) — fault label onset t=0–49, M5 physics continues t=50–199
- **Source 3:** Pure M5 Physics Synthetic — fills gaps, covers severity 0.2–1.0

> **WHY SEVERITY 0.2–0.3 IS MANDATORY:** These sequences are the SPECIFIC TRAINING DATA for M8's trend accumulator. Without them, M8 learns only acute faults (sev ≥ 0.5) and misses slow drift entirely → **LIABILITY EXPOSURE**. Category 3 liability gate.

---

### M6A Group A — Label Table (v3.0 Corrected Lengths)

| Label | Fault Type | Seq Length | Count | Source | Status |
|-------|-----------|-----------|-------|--------|--------|
| 0 | normal | 200 steps | 1,200 | Real CIRA M3 | ✅ VALID |
| 1 | bearing_wear | 250 steps | 1,200 | Spike + M5 | ⚠️ RERUN in M6B Step 0 *(was 200 steps — length insufficient for thermal lag physics)* |
| 2 | impeller_imbalance | 200 steps | 1,200 | Sub-cluster M5 | ✅ VALID |
| 3 | cavitation | 150 steps | 1,200 | Spike + M5 | ✅ VALID *(startup only — 150 steps correct; acute onset)* |
| 4 | seal_failure | 400 steps | 1,200 | Sub-cluster M5 | ⚠️ RERUN in M6B Step 0 *(was 200 steps — pressure decline too fast at 200; 400 required)* |
| 5 | overloading | 300 steps | 1,200 | Sub-cluster M5 | ⚠️ RERUN in M6B Step 0 *(was 200 steps — thermal rise physically requires 300 minimum)* |
| 6 | sensor_failure | 150 steps | 1,200 | Physics variants | ✅ VALID *(flatline/spike/drift — 150 steps correct; hardware failure is fast)* |

**TOTAL M6A: 8,400 sequences (7 classes × 1,200)**

> **NOTE:** Labels 1, 4, 5 generated at 200-step in v1.0 are **INVALIDATED**. Their 200-step pkl files must NOT be used in M6.5r or M8 training. Replacement sequences are generated in M6B Step 0 at correct lengths.

---

### M6A Severity Distribution

*Per fault class, from v2.0 audit — still valid.*

| Tier | SV\* Range | Sequences | Description |
|------|-----------|-----------|-------------|
| Early | 1.0–1.3 | 20 | Subtle, hardest for model |
| Active | 1.3–2.0 | 20 | Clear fault onset |
| Severe | > 2.0 | 10 | Obvious, calibration anchor |

*(×40 per cluster × variable cluster mix = 1,200 total per class)*

**Severity 0.2–0.3 mandate (M8 trend accumulator training):**
- bearing_wear: min 200 sequences at sev 0.2–0.3
- seal_failure: min 200 sequences at sev 0.2–0.3
- overloading: min 200 sequences at sev 0.2–0.3

*(These are the sequences that WATCH state detection is trained on)*

---

### M6A Generation Patch — seal_failure

*Applied in v2.0, still valid.*

- **Root cause:** seal_failure physics produces very gradual Pres.SV\* decline. Only 165/1,200 sequences exceeded MAE threshold at v1.0 first audit.
- Severity distribution rebalanced toward 0.4–0.7 band.
- Final accepted: 220 sequences + padded to 1,200 with physics variants.
- **PRIMARY detection path in M8:** Pres.SV\* per-channel drift monitor (Mech C).

---

## M6.5 — LSTM-AE Feature Extractor → XGBoost Bridge (v2.0 — SUPERSEDED)

> **STATUS: SUPERSEDED** by M6.5r (22-class, ~35 features). Do NOT use M6.5 v2.0 feature matrix for M7 training.

### M6.5 v2.0 Summary (Historical Record Only)

- **Input:** 8,400 M6A sequences → M4 LSTM-AE inference
- **Output:** `data/synthetic/M6_feature_matrix.csv` — Shape: 8,400 rows × 25 columns (24 features + label)
- **Gate 3 fix:** window slice corrected to 50 (was 60 in v1.0 — INVALID)

### M6.5 v2.0 Audit Results (Authoritative — Archived)

| Class | Mean MAE | Gate 3 Pass | Interpretation |
|-------|----------|-------------|----------------|
| normal | 0.120 | 86.67% | Probe only (full FPR = 0.55%) |
| bearing_wear | 0.098 | 13.33% | Mild sev near-threshold — correct |
| impeller_imbalance | 0.103 | 30.00% | Mild sequences dominate — correct |
| cavitation | 0.675 | 100.00% | Strongly anomalous — hydraulic shock |
| seal_failure | 0.196 | 29.17% | Slow hydraulic — Mech C PRIMARY path |
| overloading | 0.093 | 0.00% | Thermal-dominant — Mech C PRIMARY path |
| sensor_failure | 0.170 | 93.33% | Single-channel flatline — clearly anom. |

### Top 5 Fisher Features (v2.0)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | PmpSV_mean | Pump vibration dominant fault channel |
| 2 | PmpSV_std | Variance of pump vibration error |
| 3 | TempSV_mean | Thermal drift overloading discriminator |
| 4 | MotTV_mean | Motor temperature bearing/overloading |
| 5 | MotTV_std | Temperature variance |

> These Fisher rankings **VALIDATE** M8 channel weight direction (Pmp.SV rank 1).

### M6.5 v2.0 Archived Files (Do NOT Use for Retraining)

- `data/synthetic/M6_feature_matrix.csv` — 8,400 × 25
- `outputs/reports/module_065_sequence_audit_report.md`
- `src/module_065_sequence_audit.py` — v2 (Gate 3 fix applied)

---

## M6B — Expanded 22-Class Dataset (v14.2 Canonical Spec)

M6B extends M6A from 7 classes to the full 22-class fault universe. M6B generates Groups B, C, D, E plus reruns M6A labels 1, 4, 5 (Step 0).

### Total Dataset Post-M6B

| Label Range | Count |
|-------------|-------|
| Labels 0, 21 | 2,000 sequences each |
| Labels 1–12 | 1,500 sequences each |
| Labels 13–18, 20 | 1,200 sequences each |
| Label 19 (seal_fast) | 800 sequences |
| Group E (2 sub-types) | 800 sequences each |
| **TOTAL** | **~31,800 sequences** |
| Feature matrix (M6.5r) | ~31,800 rows × ~35 features |

---

### Step 0 — Re-Generate M6A Labels 1, 4, 5

*Required before M6B Steps 1–5.*

**Purpose:** Replace invalid 200-step M6A sequences for labels 1, 4, 5 with physics-correct lengths. These 3 classes are used in compound fault chains in Groups B–D and MUST have correct lengths first.

**Label 1 — bearing_wear: 250 steps**

Physics basis: Thermal lag (Mot.TV lags Mot.SV by 20–40 steps) requires at least 200 steps to manifest. Severity 0.2–0.3 detection needs 250 steps for Spearman window (300 windows) to be meaningful. 200 was too short.

**Label 4 — seal_failure: 400 steps**

Physics basis: Pres.SV\* decline at severity 0.3 (K_seal ≈ 0.0004/step) requires 400 steps to produce a detectable Spearman correlation. At 200 steps the drift was undetectable — not even Mech C could fire.

**Label 5 — overloading: 300 steps**

Physics basis: Thermal overloading (K_ol ≈ 0.003–0.010/step) at mild severity (0.2–0.3) requires 300 steps for Temp.SV\* to rise detectably above noise level. 200 steps produced sub-noise rise at mild severities.

**Output of Step 0:**

| File | Shape |
|------|-------|
| `data/synthetic/M6B_groupA_rerun_label1.pkl` | 1,200 seqs × 250 × 8 |
| `data/synthetic/M6B_groupA_rerun_label4.pkl` | 1,200 seqs × 400 × 8 |
| `data/synthetic/M6B_groupA_rerun_label5.pkl` | 1,200 seqs × 300 × 8 |

M6A labels 0, 2, 3, 6 remain at original files — no rerun.

---

### Group A — Single Faults (22-class Label Map, Full)

| Label | Fault Name | Steps | Count | Group | TCN L2 Target |
|-------|-----------|-------|-------|-------|---------------|
| 0 | normal | 200 | 2,000 | A | N/A (normal) |
| 1 | bearing_wear | 250 | 1,500 | A | dilation 1,2 |
| 2 | impeller_imbalance | 200 | 1,500 | A | dilation 1 |
| 3 | cavitation | 150 | 1,500 | A | dilation 1 |
| 4 | seal_failure | 400 | 1,500 | A | dilation 2,4 |
| 5 | overloading | 300 | 1,500 | A | dilation 2,4 |
| 6 | sensor_failure | 150 | 1,500 | A | dilation 1 |
| 21 | bearing_wear_gradual | 1,000 | 2,000 | A | dilation 4,8,16 *(weeks-scale liability class — Spearman + CUSUM S_n primary path)* |

---

### Group B — Compound Faults (Causal Pairs, Physics-Verified Lags)

M6B generates compound fault sequences where Fault B starts at a physically determined lag after Fault A onset (causal cascade, not simultaneous).

| Label | Fault Pair (A→B) | Steps | Count | Lag A→B | Physics Basis |
|-------|-----------------|-------|-------|---------|---------------|
| 7 | bearing_wear + overloading | 600 | 1,500 | ~200s | Bearing heat → thermal load |
| 8 | cavitation + seal_failure | 550 | 1,500 | ~150s | Pressure drop → seal load |
| 9 | impeller_imbalance + bearing_wear | 700 | 1,500 | 300–600s | Imbalance → shaft load → bearing fatigue |
| 10 | seal_failure + cavitation | 900 | 1,500 | 400–800s | Pressure loss → NPSHa margin |
| 11 | overloading + bearing_wear | 800 | 1,500 | 400–600s | Thermal → lubricant thin |
| 12 | impeller_imbalance + cavitation | 450 | 1,500 | ~100s | Imbalance → pressure ripple → NPSHa |

> **NOTE:** All lag values physics-verified against 110 kW, 7-stage nameplate. Lags in steps (1 step = 1 second at 1 Hz sampling). Labels 9, 10, 11 lags are ranges — not single values — because propagation timescale depends on severity and cluster context.

`compound_interaction_flag`: computed in M6.5r as Spearman lag shift between the two primary fault channels. Expected HIGH in Groups B sequences. This flag is **Feature 33** in the 35-feature M6.5r matrix.

---

### Group C — Masked Fault Sequences (Fault Hidden by Sensor Anomaly)

| Label | Description | Steps | Count | Masking Channel |
|-------|-------------|-------|-------|----------------|
| 13 | bearing_wear (Mot.SV masked) | 300 | 1,200 | Mot.SV → calibration drift |
| 14 | cavitation (Pres.SV masked) | 210 | 1,200 | Pres.SV → stuck/flatline |
| 15 | seal_failure (Pres.SV drifting) | 500 | 1,200 | Pres.SV → upward sensor drift |
| 16 | overloading (Temp.SV stuck) | 350 | 1,200 | Temp.SV → frozen at last value |
| 17 | impeller_imbalance (Pmp.SV flat) | 250 | 1,200 | Pmp.SV → flatline hardware |
| 18 | cavitation + intermittent | 300 | 1,200 | Cavitation intermittent onset |

> **KEY DISTINCTION for Label 15:**
> - `seal_failure` = **NEGATIVE** Pres.SV\* drift (hydraulic loss)
> - `sensor_drift` = **POSITIVE** Pres.SV\* drift (calibration bias)
>
> M8 must disambiguate via sign + cross-channel analysis.

---

### Group D — Cyclic / Transient Fault Sequences

| Label | Description | Steps | Count | Notes |
|-------|-------------|-------|-------|-------|
| 19 | seal_failure_fast (acute) | 150 | 800 | Fast degradation, sev 0.8 |
| 20 | overloading_cyclic | 600 | 1,200 | Cyclic thermal load pattern |
| 21 | bearing_wear_gradual | 1,000 | 2,000 | Weeks-scale **(LIABILITY CLASS)** |

---

### Group E — Multi-Channel Sensor Fault (2-channel Thermal + Pump)

| Sub-type | Steps | Count | Target Channels |
|---------|-------|-------|----------------|
| sensor_2ch_thermal_pump | 250 | 800 | Mot.TV + Pmp.TV both anomalous (differentiates from single-channel) |

---

### Output Files Per Group

```
data/synthetic/M6B_groupA_rerun_label1.pkl
data/synthetic/M6B_groupA_rerun_label4.pkl
data/synthetic/M6B_groupA_rerun_label5.pkl
data/synthetic/M6B_groupB_compound.pkl
data/synthetic/M6B_groupC_masked.pkl
data/synthetic/M6B_groupD_cyclic.pkl
data/synthetic/M6B_groupE_sensor2ch.pkl
data/synthetic/zt_sequences_groupA.pkl      ← M4 sliding window zt per sequence
data/synthetic/zt_sequences_groupB.pkl
data/synthetic/zt_sequences_groupC.pkl
data/synthetic/zt_sequences_groupD.pkl
data/synthetic/zt_sequences_groupE.pkl
data/synthetic/physics_context_strings.json ← Per-sequence physics context for M10
```

---

## M6.5r — Updated Feature Extractor (22-Class, ~35 Features)

> **STATUS:** Supersedes M6.5 v2.0. Runs on ALL M6B sequences (31,800 total).

- **Input:** All M6B pkl files (Groups A–E) + M6A valid files (labels 0, 2, 3, 6)
- **Output:** `data/synthetic/M6B_feature_matrix.csv` — Shape: ~31,800 rows × ~35 columns (34 features + label)

### Feature Set Breakdown (~35 Features)

**Per-Channel Statistics (8 channels × 3 stats = 24 features)**

For each of 8 channels: `mean_err`, `std_err`, `slope_err`

**Cross-Channel Features (7 features)**

| Feature | Description |
|---------|-------------|
| `corr_MotTV_TempSV` | r(Mot.TV, Temp.SV) thermal coupling |
| `corr_PmpPV_PmpSV` | r(Pmp.PV, Pmp.SV) imbalance coupling |
| `corr_PmpPV_PresSV` | r(Pmp.PV, Pres.SV) cavitation coupling |
| `pres_monotonic_flag` | seal_failure key feature (monotone Pres.SV decline) |
| `pres_chaotic_flag` | cavitation key feature (chaotic Pres.SV) |
| `thermal_decoupling_flag` | r(Mot.TV, Temp.SV) < 0.3 → hydraulic fault |
| `compound_interaction_flag` | Spearman lag shift between primary channels |

**Zt-Derived Features (3 features)**

| Feature | Description |
|---------|-------------|
| `mean_zt_magnitude` | Mean of Level 1 zt error vector magnitude |
| `std_zt_magnitude` | Std of zt magnitude |
| `zt_drift_slope` | Slope of zt over N_windows |

**TCN-AE Output Features (3 features — added AFTER M8 is complete)**

| Feature | Description |
|---------|-------------|
| `score_A` | TCN-AE severity score |
| `score_B` | TCN-AE drift slope score (CUSUM input) |
| `score_C` | TCN-AE chain transition score (compound detection) |

> **NOTE:** score_A/B/C = None until TCN-AE (M8) is trained. XGBoost M7 trains with reduced 32-feature set initially. Full 35-feature retraining occurs after M8 completion.

**Causal Ordering Feature (1 feature)**

| Feature | Description |
|---------|-------------|
| `onset_order` | Which channel's MAE first exceeded 0.5× baseline, at which window index (encodes causal sequence) |

### Output Files

```
data/synthetic/M6B_feature_matrix.csv     — full 35-feature, 31,800 rows
outputs/reports/module_065r_audit_report.md
src/module_065r_feature_retrain.py
```

---

## Validation Gates — M6B

*Must ALL pass before M7 training.*

| Gate | Description |
|------|-------------|
| GATE-1 | Label distribution matches targets: Labels 0,21 = 2,000 each; Labels 1–12 = 1,500 each; Labels 13–18,20 = 1,200 each; Label 19 = 800; Group E = 800 each |
| GATE-2 | Group A — all channels in [−0.1, 1.1] normalised range throughout |
| GATE-3 | Group B — MAE vs cluster centroid progressively increases: t=0 to lag: MAE < 0.10 / t=lag to lag+T: MAE ∈ [0.10, 0.40] / t=lag+T+: MAE > 0.40 |
| GATE-4 | Group C — seal_failure (Label 15) Pres.SV\* drift direction **NEGATIVE**; sensor_drift — Pres.SV\* drift direction **POSITIVE**. THESE MUST NOT BE CONFUSED in generation |
| GATE-5 | Thermal coupling preservation: bearing_wear r(Mot.TV, Mot.SV) > 0.85 per seq; overloading r(Mot.TV, Temp.SV) > 0.90 per seq; cavitation r(Mot.TV, Temp.SV) < 0.5 (decoupled — expected) |
| GATE-6 | Temporal coherence (dX/dt continuity): each class pass rate target > 90%. Cavitation exception: 91% acceptable (hydraulic shock = non-smooth) |
| GATE-7 | No negative pressure, no T\* < −0.5, no SV\* > 5.0 |
| GATE-8 | `zt_sequences_groupX.pkl` files all written and loadable. Shape check: (N_sequences, N_windows, 64+8) per group |
| GATE-9 | `onset_order` feature non-null for all Group B compound sequences. Confirms causal ordering captured |
| GATE-10 | `physics_context_strings.json` written with entry for all 22 labels. Each entry contains: `what`, `why`, `sensor_signature`, `timeline`, `recommended_action`, `if_ignored`, `model_limitation` |

---

## Detection Path Summary (Per Class)

*Informs M8 channel weight decisions.*

| Label | Class | Primary M8 Path | Notes |
|-------|-------|----------------|-------|
| 0 | normal | N/A | Baseline |
| 1 | bearing_wear | Mech C Mot.SV\* Spearman | + thermal lag |
| 2 | impeller_imbalance | Single-window Pmp.SV\* | Fast onset |
| 3 | cavitation | Single-window DANGER bypass | Startup only |
| 4 | seal_failure | Mech C Pres.SV\* Spearman | NEGATIVE drift |
| 5 | overloading | Mech C Temp.SV\* Spearman | Steady-state only |
| 6 | sensor_failure | Channel std collapse flag | 1 channel only |
| 7 | bearing+overloading | Dual Mech C | Mot.SV + Temp.SV |
| 8 | cavitation+seal | DANGER → Pres.SV decline | Cascade |
| 9 | imbalance+bearing | Pmp.SV → Mot.SV lag | Shaft coupling |
| 10 | seal+cavitation | Pres.SV → NPSHa margin | 900-step sequence |
| 11 | overloading+bearing | Temp.SV → lubricant | Thermal cascade |
| 12 | imbalance+cavitation | Pmp.SV ripple → pressure | Fast compound |
| 13–18 | Masked variants | Cross-channel disambiguation | Sign direction key |
| 19 | seal_failure_fast | Single-window WARN/DANGER | sev 0.8 |
| 20 | overloading_cyclic | Mech C Temp.SV + CUSUM | Cyclic pattern |
| 21 | bearing_wear_gradual | CUSUM S_n(score_B) PRIMARY | **LIABILITY class** |

---

## Paste Text — M6 Module Status (v3.0 — Architecture v14.2)

| Key | Value |
|-----|-------|
| `M6A_status` | PARTIALLY SUPERSEDED BY M6B |
| `M6A_valid_labels` | 0, 2, 3, 6 (normal, imbalance, cavitation, sensor) |
| `M6A_rerun_labels` | 1, 4, 5 (bearing, seal, overloading) — IN M6B Step 0 |
| `M6A_sequence_count` | 8,400 (7 classes × 1,200) — historical |
| `M6B_total_sequences` | ~31,800 |
| `M6B_label_count` | 22 classes |
| `M6B_feature_matrix_shape` | ~31,800 rows × ~35 features (M6.5r) |
| `M6B_groups` | A (single) + B (compound) + C (masked) + D (cyclic) + E (sensor2ch) |
| `M6A_labels_1_4_5_status` | RERUN REQUIRED at 250/400/300 steps respectively |
| `M6_label21_count` | 2,000 sequences at 1,000 steps (weeks-scale LIABILITY) |
| `M6_zt_export` | zt_sequences_groupA/B/C/D/E.pkl (one per group) |
| `M6_physics_context_export` | physics_context_strings.json (22 entries) |
| `M6_onset_order_feature` | added in M6.5r — causal ordering per compound sequence |
| `M6_score_ABC_status` | score_A/B/C = None until M8 TCN-AE complete; Full 35-feature matrix available post-M8 |
| `M6_seal_direction_rule` | seal_failure = NEGATIVE Pres.SV\* drift **(LOCKED)**; sensor_drift = POSITIVE Pres.SV\* drift **(LOCKED)** |
| `M6_GroupB_lag_Label9` | 300–600 steps (physics-verified range) |
| `M6_GroupB_lag_Label10` | 400–800 steps (physics-verified range) |
| `M6_GroupB_lag_Label11` | 400–600 steps (physics-verified range) |
| `M6_gate_status` | GATES 1–10 must pass before M7 training |
| `Status for M7` | READY after M6B Step 0 + Gates 1–7 pass (score_A/B/C = None for initial M7 training) |

---

## File Manifest

### GitHub Push (This File)

```
docs/modules_M6_synthetic_generation.md    ← THIS FILE (v3.0)
```

### Data Files (Generated by Scripts — Not Pushed to GitHub)

```
data/synthetic/M6B_groupA_rerun_label1.pkl
data/synthetic/M6B_groupA_rerun_label4.pkl
data/synthetic/M6B_groupA_rerun_label5.pkl
data/synthetic/M6B_groupB_compound.pkl
data/synthetic/M6B_groupC_masked.pkl
data/synthetic/M6B_groupD_cyclic.pkl
data/synthetic/M6B_groupE_sensor2ch.pkl
data/synthetic/zt_sequences_groupA/B/C/D/E.pkl
data/synthetic/M6B_feature_matrix.csv
data/synthetic/physics_context_strings.json
```

### Reports (GitHub Push After Run)

```
outputs/reports/module_065r_audit_report.md
outputs/reports/module_06B_generation_report.md
```

### Locked Files (Do NOT Modify)

```
data/synthetic/M6_feature_matrix.csv      ← M6.5 v2.0 (archived, superseded)
models/M3_normalization_config.json       ← Normalisation baselines LOCKED
models/M4_spike_seeds_meta.csv            ← Spike seeds LOCKED
models/M4_threshold_config.json          ← Level 1 threshold 0.110058 LOCKED
```

---

## Next Prompt — Ready for M6B Step 0

📦 File 13 done. Starting M6B Step 0 (Label 1/4/5 rerun).

- **Finding:** M6A labels 1, 4, 5 require rerun at 250/400/300 steps.
- **Locked:** M6A labels 0, 2, 3, 6 valid. M6.5r supersedes M6.5 v2.
- **Uploading:** `modules_M6_synthetic_generation.md` (v3.0).
- **Action:** Provide M6B Step 0 complete generation script.
