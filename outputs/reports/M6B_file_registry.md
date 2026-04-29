# PumpSmart M6B File Registry

**Arch version:** v14.2 | **Script:** module_06B_steps1to3_combined v1.0 | **Date:** 2026-04-28

> Machine-readable index: `outputs/reports/M6B_file_registry.json`  
> Total files tracked: 17

---

## Legend

| Column | Meaning |
|--------|---------|
| **File** | Filename (path relative to project root) |
| **Type** | File format |
| **Step** | Which script step created it |
| **Size** | Disk size at time of registry generation |
| **Shape** | Logical shape / row×col / entry count |
| **Locked** | If ✓: do NOT overwrite without arch version bump |
| **Consumers** | Downstream modules that read this file |

---

## M6B Step0 v2 (prereq)

| File | Type | Size | Shape | Locked | Consumers |
|------|------|------|-------|--------|-----------|
| `M6B_sequences_groupA_rerun.pkl` | pkl | 46.30 MB | dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | ✓ | Step1 compound seeding, Step3 merge |
| `z_t_sequences_groupA_faults_rerun.pkl` | pkl | 8.62 MB | list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | ✓ | M8 TCN-AE Level 2 training |

### `M6B_sequences_groupA_rerun.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequences_groupA_rerun.pkl`  
**Shape:** dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 4500 seqs  
**Downstream:** Step1 compound seeding, Step3 merge  

Group A labels 1,4,5 at physics-correct lengths (250,400,300 steps). CIRA-seeded bearing/seal/overloading sequences with F1/F4 fixes applied.

> **Note:** LOCKED — do not regenerate. Created by prerequisite scripts.

---

### `z_t_sequences_groupA_faults_rerun.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\z_t_sequences_groupA_faults_rerun.pkl`  
**Shape:** list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 4500 entries  
**Downstream:** M8 TCN-AE Level 2 training  

z_t latent vectors from frozen M4 LSTM-AE for Group A rerun labels (1,4,5). Each entry: {z_t: (N_windows,64), mae: (N_windows,8)}. N_windows=T//50.

> **Note:** LOCKED — do not regenerate. Created by prerequisite scripts.

---

## M6B Step0b v2 (prereq)

| File | Type | Size | Shape | Locked | Consumers |
|------|------|------|-------|--------|-----------|
| `M6B_sequences_groupA_carried.pkl` | pkl | 36.39 MB | dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | ✓ | Step1 compound seeding, Step3 merge |
| `z_t_sequences_groupA_normal.pkl` | pkl | 2.48 MB | list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | ✓ | M8 TCN-AE normal baseline |
| `z_t_sequences_groupA_faults.pkl` | pkl | 4.44 MB | list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | ✓ | M8 TCN-AE fault training |

### `M6B_sequences_groupA_carried.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequences_groupA_carried.pkl`  
**Shape:** dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 6200 seqs  
**Downstream:** Step1 compound seeding, Step3 merge  

Group A labels 0,2,3,6 carried from M6A v5 with F2/F3/F5 fixes. Normal (2000), imbalance (1500), cavitation (1500), sensor_failure (1200).

> **Note:** LOCKED — do not regenerate. Created by prerequisite scripts.

---

### `z_t_sequences_groupA_normal.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\z_t_sequences_groupA_normal.pkl`  
**Shape:** list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 2000 entries  
**Downstream:** M8 TCN-AE normal baseline  

z_t latent vectors for Group A normal sequences (2000 seqs, label 0). Normal operation baseline for L2 TCN-AE reconstruction training.

> **Note:** LOCKED — do not regenerate. Created by prerequisite scripts.

---

### `z_t_sequences_groupA_faults.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\z_t_sequences_groupA_faults.pkl`  
**Shape:** list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 4200 entries  
**Downstream:** M8 TCN-AE fault training  

z_t latent vectors for Group A carried fault labels (2,3,6). Cavitation dual-sig confirmed: Pres.SV* shift=-0.2304, Pmp.SV* shift=+0.2003.

> **Note:** LOCKED — do not regenerate. Created by prerequisite scripts.

---

## Step 1

| File | Type | Size | Shape | Locked | Consumers |
|------|------|------|-------|--------|-----------|
| `M6B_sequences_groupB.pkl` | pkl | 193.36 MB | dict{sequences: list[ndarray(T,8)], metadata: list[dict]} |  | Step3 merge, M8 TCN-AE score_C training |
| `z_t_sequences_groupB.pkl` | pkl | 35.27 MB | list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] |  | M8 TCN-AE Level 2 training, M6.5r score_C feature extraction |

### `M6B_sequences_groupB.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequences_groupB.pkl`  
**Shape:** dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 9000 seqs | T per label: {7:600,8:550,9:700,10:900,11:800,12:450}  
**Downstream:** Step3 merge, M8 TCN-AE score_C training, M7 compound classification  

Group B compound chain sequences. 6 labels (7-12), 1500 seqs each = 9000 total. Each sequence: two faults active with physics-verified lag (50–600s). Phase 1: primary fault only. Phase 2: primary+secondary superimposed. Compound chain: bearing+OL, cav+seal, imbal+bearing, seal+cav, OL+bearing, imbal+cav.

---

### `z_t_sequences_groupB.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\z_t_sequences_groupB.pkl`  
**Shape:** list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 9000 entries | N_w up to 18 (900 steps / 50)  
**Downstream:** M8 TCN-AE Level 2 training, M6.5r score_C feature extraction  

z_t latent vectors from frozen M4 LSTM-AE for all 9000 Group B sequences. Captures score_A per window + z_t bottleneck representation. Compound chains show characteristic z_t transition at secondary onset (score_C source). G9 gate: weighted MAE > 0.110058 in ≥90% sequences.

---

## Step 2A

| File | Type | Size | Shape | Locked | Consumers |
|------|------|------|-------|--------|-----------|
| `M6B_sequences_groupC.pkl` | pkl | 62.85 MB | dict{sequences: list[ndarray(T,8)], metadata: list[dict]} |  | Step3 merge, M8 masked fault discrimination training |
| `z_t_sequences_groupC.pkl` | pkl | 11.51 MB | list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] |  | M8 TCN-AE masked fault training, M6.5r masked_channel_flag feature |

### `M6B_sequences_groupC.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequences_groupC.pkl`  
**Shape:** dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 6000 seqs | T: {13:300,14:210,15:500,16:350,17:250}  
**Downstream:** Step3 merge, M8 masked fault discrimination training, M7 Group C classification  

Group C masked fault sequences. 5 labels (13-17), 1200 seqs each = 6000 total. Each: base fault (bearing/cav/seal/OL/imbal) with one primary channel degraded (flatline, positive drift, or stuck). Sensor failure precedes full fault onset. Key: label 15 Pres.SV drifts UP (sensor bias) while seal failure causes NEGATIVE Pres.SV drift — M8 must disambiguate by sign + cross-channel. G10: non-masked channels ≥50% of MAE.

---

### `z_t_sequences_groupC.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\z_t_sequences_groupC.pkl`  
**Shape:** list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 6000 entries  
**Downstream:** M8 TCN-AE masked fault training, M6.5r masked_channel_flag feature  

z_t latent vectors for Group C masked fault sequences. Characteristic: reduced z_t signal on masked channel path, secondary channels carry fault signal. Used to train M8 secondary-path detection. G10 validation: non-masked channel MAE contributions measurable in z_t space.

---

## Step 2B

| File | Type | Size | Shape | Locked | Consumers |
|------|------|------|-------|--------|-----------|
| `M6B_sequences_groupD.pkl` | pkl | 103.06 MB | dict{sequences: list[ndarray(T,8)], metadata: list[dict]} |  | Step3 merge, M8 CUSUM label-21 training |
| `z_t_sequences_groupD.pkl` | pkl | 18.83 MB | list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] |  | M8 CUSUM score_B source, M8 L3 CUSUM training |

### `M6B_sequences_groupD.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequences_groupD.pkl`  
**Shape:** dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 5200 seqs | T: {18:300,19:150,20:600,21:1000}  
**Downstream:** Step3 merge, M8 CUSUM label-21 training, M7 severity variant classification  

Group D severity variant sequences. 4 labels (18-21), counts: {18:1200, 19:800, 20:1200, 21:2000}. Label 18 (cav_intermittent): 3-7 burst pattern, NPSHa oscillation. Label 19 (seal_fast): turbulent orifice Q=Cd·A·√(2dP/ρ), Pres.SV collapses ≤20 steps. Label 20 (OL_cyclic): thermal sawtooth with rising baseline drift. Label 21 (bearing_gradual): Paris law low-dK, sev 0.05–0.25, ≥60% seqs below MAE threshold — CUSUM-only detection, PRIMARY LIABILITY CLASS. G11-ext: slope_MotSV > 0 in ≥95% seqs.

> **Note:** Label 21 (2000 seqs, 1000 steps) is largest single class. Sub-threshold % should be ≥60% — check results dict key step2_label21_subthreshold_pct.

---

### `z_t_sequences_groupD.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\z_t_sequences_groupD.pkl`  
**Shape:** list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 5200 entries | label 21: up to 20 windows each  
**Downstream:** M8 CUSUM score_B source, M8 L3 CUSUM training, M6.5r score_B feature  

z_t latent vectors for Group D sequences. Critical for label 21: z_t drift slope (score_B) is the L3 CUSUM input. Label 21 z_t shows slow monotonic drift in z_t space detectable by TCN-AE with high dilation (d=16), invisible to fixed-threshold L1. Also includes label 21 z_t for sub-threshold % validation.

---

## Step 3A

| File | Type | Size | Shape | Locked | Consumers |
|------|------|------|-------|--------|-----------|
| `M6B_sequences_groupE.pkl` | pkl | 10.64 MB | dict{sequences: list[ndarray(T,8)], metadata: list[dict]} |  | Step3 merge, M8 multi-sensor detection |
| `z_t_sequences_groupE.pkl` | pkl | 1.96 MB | list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] |  | M8 multi-sensor path training, M6.5r multi_sensor_anomaly_count feature |

### `M6B_sequences_groupE.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequences_groupE.pkl`  
**Shape:** dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 1600 seqs | T=200 both variants  
**Downstream:** Step3 merge, M8 multi-sensor detection, M7 Group E classification  

Group E multi-sensor failure sequences. 2 variants × 800 = 1600 total. E_thermal (label 22): Mot.TV + Temp.SV both fail — shared thermal excitation rail. E_pump (label 23): Pmp.SV + Pmp.PV both fail — moisture ingress to pump-side junction box. G11: exactly 2 channels anomalous; remaining 6 within ±0.20 baseline (≥90% seqs). multi_sensor_anomaly_count=2 in all metadata entries.

---

### `z_t_sequences_groupE.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\z_t_sequences_groupE.pkl`  
**Shape:** list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 1600 entries  
**Downstream:** M8 multi-sensor path training, M6.5r multi_sensor_anomaly_count feature  

z_t latent vectors for Group E sequences. Characteristic: MAE spike on exactly 2 channel dimensions simultaneously. M8 uses multi_sensor_count=2 flag derived from this z_t pattern. Gate M8-14: Group E TPR ≥ 88% for multi_sensor_count=2 detection.

---

## Step 3B

| File | Type | Size | Shape | Locked | Consumers |
|------|------|------|-------|--------|-----------|
| `M6B_combined_sequences.pkl` | pkl | 452.73 MB | dict{sequences: list[ndarray(T,8)], metadata: list[dict]} |  | M8 full fault validation pool, M8 adversarial testing |
| `M6B_sequence_meta.csv` | csv | 4.18 MB | CSV |  | M7 stratified splits, M7 SHAP grouping |

### `M6B_combined_sequences.pkl`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_combined_sequences.pkl`  
**Shape:** dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | ~31800 seqs | T varies per label  
**Downstream:** M8 full fault validation pool, M8 adversarial testing, archive  

FULL MERGED DATASET — all groups A through E. ~32500 total sequences, 22+2 classes, labels 0–23. Groups: A(10700) + B(9000) + C(6000) + D(5200) + E(1600). Primary input for M8 TCN-AE fault validation pool and adversarial testing. Each sequence is ndarray(T,8) in normalized space (P*, a*, ΔT*). Normalization: cluster-relative (M3 config). Sequences generated in M6B LOCKED channel order: Mot.SV=0, Pmp.SV=1, Mot.TV=2, Pmp.PV=3, Temp.SV=4, Pres.SV=5, Pmp.TV=6, Mot.PV=7.

> **Note:** This is the single authoritative dataset file for PumpSmart v14.2. M7 does NOT read this directly — M7 reads M6B_feature_matrix.csv (M6.5r output). M8 reads z_t pkl files, not this file directly for training.

---

### `M6B_sequence_meta.csv`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequence_meta.csv`  
**Shape:** CSV | ~32500 rows × ~15 cols  
**Downstream:** M7 stratified splits, M7 SHAP grouping, M8 adversarial test selection, debugging  

Metadata table for all ~31800 sequences. One row per sequence. Columns: seq_id, label, fault_name, group, severity, cluster_id, cluster_name, steps, source, arch_version, and group-specific fields (secondary_onset_step/lag for B, masked_channel for C, variant for D/E). Used for stratified train/val/test splits in M7. Used for SHAP analysis grouping in M7. Used for gate validation reporting.

---

## Step 3C

| File | Type | Size | Shape | Locked | Consumers |
|------|------|------|-------|--------|-----------|
| `M6B_physics_context_strings.json` | json | 0.02 MB | JSON | ✓ | M10 Flask API 7-field output, M10 advisory text lookup |

### `M6B_physics_context_strings.json`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_physics_context_strings.json`  
**Shape:** JSON | 24 label entries | ~180 lines  
**Downstream:** M10 Flask API 7-field output, M10 advisory text lookup, M12 output validation  

Static physics knowledge lookup per fault label (0–23). Encodes: probable_condition, expected_sensor_behaviour, risk_if_ignored, recommended_action for each class. Seeds M10 Flask API 7-field output (fields 3, 4, 5, 6). NOT per-sequence — one canonical entry per label. Climate-agnostic (normalized space). Contains accurate physics references: Paris law, Joukowsky, Q_leak orifice, Cp·m thermal mass, ISO 1940, IEC 60034, NPSHa/NPSHr physics.

> **Note:** LOCKED after generation. Edit requires arch_version bump and M10 re-test.

---

## Step 3D

| File | Type | Size | Shape | Locked | Consumers |
|------|------|------|-------|--------|-----------|
| `fault_rules_v3.json` | json | 0.00 MB | JSON | ✓ | M7 label decoding, M8 sequence config |

### `fault_rules_v3.json`

**Path:** `C:\Users\user\Desktop\PumpSmart_Project\models\fault_rules_v3.json`  
**Shape:** JSON | ~120 lines | 24 classes (labels 0–23)  
**Downstream:** M7 label decoding, M8 sequence config, M10 API label→display map, M12 validation  

22+2 class canonical fault universe definition for PumpSmart v14.2. Supersedes fault_rules.json (v1, M5/M6A reference — do not overwrite that). Contains: label_map (int→name), sequence_steps, sequence_counts, compound_lag_ranges, group definitions, validation gates, physics_notes. Used by M7 for label decoding, M8 for sequence configuration, M10 for API response label→display mapping. fault_rules_v3.json = LOCKED after Step 3D — any change requires new arch version.

> **Note:** DO NOT overwrite models/fault_rules.json (v1 — M5/M6A reference, archived). This is fault_rules_v3.json — separate file.

---

## Gate Summary (at time of generation)

| Gate | Rate | Pass |
|------|------|------|
| G1_groupB | 1.0000 | ✓ |
| G2_groupB | 1.0000 | ✓ |
| G8_temporal_ordering | 1.000 | ✓ |
| G9_compound_mae | 1.000 | ✓ |
| G10_masked_secondary | 1.000 | ✓ |
| G1_groupC | 1.0000 | ✓ |
| G2_groupC | 1.0000 | ✓ |
| G11ext_gradual_slope | 1.000 | ✓ |
| G1_groupD | 1.0000 | ✓ |
| G2_groupD | 1.0000 | ✓ |
| G11_multisensor | 1.000 | ✓ |
| G1_final | 1.0000 | ✓ |
| G2_final | 1.0000 | ✓ |
| thermal_coupling_fidelity | 0.4266 | ~ |
| physics_violations_final | 0.0000 | ~ |

---

*Generated automatically by `module_06B_steps1to3_combined` — do not edit manually.*