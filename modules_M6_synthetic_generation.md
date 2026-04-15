# ⛔ SUPERSEDED — DO NOT USE FOR SCRIPTING

> **This file is the original pre-audit M6 combined specification (v1.0, 2026-04-12).**
> It describes M6B as 4 compound pairs × 400 sequences = 1,600 sequences (multi-hot labels)
> and M6.5 as a 10,000 × 29 column matrix. **Both are architecturally obsolete.**
>
> **Current architecture (v12.0):**
> - M6B = 21 classes (labels 0–20), ~25,000–27,000 sequences, Groups A–E, single-label
> - M6.5r = ~189,000 rows × 26 columns (windowed), output = `M6B_feature_matrix.csv`
> - M7 input = `M6B_feature_matrix.csv` (~189,000 × 26), 21 classes
>
> **Canonical source of truth:** [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md) (Part 2)
>
> This file is retained for historical audit trail only. **Do not reference for any script generation.**

---

# PumpSmart — Module M6 Complete Specification
## Synthetic Data Generation Pipeline: M6A → M6B → M6.5

**Document version:** v1.0 — Post Bias-Audit  
**Date:** 2026-04-12  
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)  
**Status:** ~~M6A REGENERATION REQUIRED | M6B NEW | M6.5 SIGNIFICANT CHANGES~~ → **ALL SUPERSEDED BY v12.0**

---

## Why This Document Exists

After M6A was completed successfully (8400 sequences, 7.29× separation, 0 physics violations),
a critical architecture review identified five model biases. This document captures the full
resolution framework and revised specifications for all three M6 sub-modules.

---

## Bias Audit Summary

| Bias | Description | Resolution | Resolvable? |
|------|-------------|------------|-------------|
| **Bias 1** | Causal order hardcoded — real world has order uncertainty | Progressive confidence output (M7 + M10) | YES — M7/M10 |
| **Bias 2** | Single pump dataset — normalization is pump-specific | Accept; normalization makes patterns relative not absolute | PARTIAL |
| **Bias 3** | Uniform severity `[0.1, 1.0]` — over-represents catastrophic faults | Weibull-skewed severity in M6A; sample weights in M7 | YES — M6A + M7 |
| **Bias 4** | No cross-fault contamination — real failures cascade as chain reactions | New M6B compound generator + multi-label M7 | YES — M6B + M7 |
| **Bias 5** | Sensor failure treated independently of process faults | Pre-inference sensor validation middleware in M10 | YES — M10 |

---

## Fault Progression Model (Bias 1 Architecture)

```
STAGE 1 — Anomaly Detection (LSTM-AE, M8)
──────────────────────────────────────────
MAE crosses threshold → anomaly detected
Model does NOT classify yet
Output: "Minor anomaly in [channel] — multiple causes possible"
Confidence: LOW

        ↓ (few timesteps — fault propagates)

STAGE 2 — Multi-hypothesis Warning
────────────────────────────────────
Multiple channels show correlated deviation
Output: "Probable causes: [cavitation 42%] [seal_failure 31%] [sensor 27%]"
Confidence: MEDIUM

        ↓ (more timesteps — secondary channels confirm)

STAGE 3 — Hard Classification (XGBoost, M7)
────────────────────────────────────────────
Secondary channels confirm causal chain
Output: "PRIMARY: seal_failure 87% | ALSO: overloading 34%"
Confidence: HIGH (>75% threshold)
```

This architecture is implemented via:
- `predict_proba()` in M7 XGBoost
- Confidence threshold (75%) in M10 Flask API
- 4-state alert machine: NORMAL → WATCH → WARN → DANGER

---

## Real Fault Cascade Chain (Bias 4 Physics Basis)

```
t=0    Impeller imbalance develops (BPF vibration onset)
         ↓ radial bearing overload
t+50   Bearing wear begins (Mot.SV rising)
         ↓ shaft wobble → seal face misalignment
t+120  Seal leakage begins (Pres.SV progressive drop)
         ↓ pressure loss → motor compensates
t+200  Overloading begins (Temp.SV drift, Mot.TV rise)
         ↓ operator unaware
t=CATASTROPHIC FAILURE
```

This chain reaction is the physical justification for M6B compound fault generation.

---

## M6A — Revised Specification (REGENERATION REQUIRED)

### Why Regenerate

M6A metadata is missing two critical columns (`severity`, `fault_stage`) that are required
for M7 sample weighting and M10 progressive confidence gating. Regeneration takes ~2 minutes.

### Changes from Original M6A

| Parameter | Original | Revised | Reason |
|-----------|----------|---------|--------|
| Severity distribution | `uniform(0.1, 1.0)` | Weibull `k=0.8` clipped `[0.05, 1.0]` | Bias 3 — early-stage focus |
| Metadata columns | `seq_id, label, source, cluster` | + `severity, fault_stage` | Required for M7 + M10 |
| Causal lag ranges | Fixed per fault type | ±20% wider | Bias 1 — order uncertainty |
| Output shape | `(8400, 200, 8)` | `(8400, 200, 8)` — unchanged | — |

### Weibull Severity Distribution

```python
# Replaces: sev = np.random.uniform(0.1, 1.0)
sev = np.random.weibull(0.8) * 0.7
sev = np.clip(sev, 0.05, 1.0)

# Resulting distribution:
# sev 0.05–0.30 → ~55% of sequences  (early fault — HARDEST to detect)
# sev 0.30–0.70 → ~30% of sequences  (developing fault)
# sev 0.70–1.00 → ~15% of sequences  (advanced fault — easiest to detect)
```

**Why early-stage focus matters:** The model's most critical job is early warning.
Late-stage faults are obvious. Early-stage faults are when the operator needs the
system most. Weibull distribution trains the model on proportionally more of the hard cases.

### fault_stage Column Definition

```python
if   sev <= 0.30: fault_stage = "early"
elif sev <= 0.65: fault_stage = "developing"
else:             fault_stage = "advanced"
```

### Revised Causal Lag Ranges (±20% widened)

| Fault | Original Lag | Revised Lag Range | Physics Basis |
|-------|-------------|-------------------|---------------|
| bearing_wear | 20–40 steps | 16–48 steps | Radial load → bearing temperature |
| impeller_imbalance | 10–25 steps | 8–30 steps | BPF → lateral vibration |
| seal_failure | 5–15 steps | 4–18 steps | Pres.SV onset variability |
| cavitation | onset t=0–10 | t=0–12 steps | Startup transient timing |
| overloading | 8–20 steps | 6–24 steps | Thermal rise lag |
| sensor_failure | immediate | immediate ±2 steps | Laplace spike character |

### M6A Output Files (Revised)

```
data/synthetic/M6_synthetic_sequences.npy    → shape (8400, 200, 8)
data/synthetic/M6_synthetic_metadata.csv     → columns:
    seq_id | label | severity | fault_stage | source | cluster | seed_idx
outputs/plots/module_06a_*_sanity_plot.png
outputs/plots/module_06a_*_mae_distribution.png
outputs/reports/module_06a_*_report.md
```

### M6A Validation Gates (Unchanged)

```
✅ Total sequences    : 8400 / 8400
✅ Per-class count    : 1200 each
✅ Separation ratio   : > 5.0× (M6A v3 achieved 7.29×)
✅ MAE gate pass      : 100% fault sequences above M4 threshold
✅ Physics violations : 0 in 1400 audited
```

### M6A Locked Results (from v3 run — preserved as reference)

```
M6_total_sequences         : 8400
M6_sequences_per_class     : 1200
M6_mae_normal_mean         : 0.029345
M6_mae_fault_mean          : 0.213966
M6_separation_ratio        : 7.29×
M6_mae_gate_pass_pct       : 100.0%
M6_physics_violations      : 0 in 1400 audited
M6_bugs_fixed              : BUG1 BUG2 BUG3 BUG4 BUG5 BUG6 BUG7

Per-class MAE:
  normal             : 0.029345
  cavitation         : 0.432068  ← most anomalous
  bearing_wear       : 0.228169
  seal_failure       : 0.178987
  overloading        : 0.148646
  impeller_imbalance : 0.137061  ← closest to threshold
  sensor_failure     : 0.158862
```

---

## M6B — New Module: Compound Fault Generator

### Purpose

Real industrial pump failures are rarely isolated single-fault events. One fault causally
generates another. M6B generates physically causal compound fault sequences with multi-hot
labels, enabling M7 to become a true multi-label classifier.

### Module Name

```
src/module_06b_compound_generator.py
```

### Compound Fault Pairs — Physically Causal

**Pair 1: impeller_imbalance + bearing_wear**
```
Physics:   BPF vibration → radial bearing overload → wear accumulation
Channels:  Pmp.PV↑ + Pmp.SV↑ (imbalance onset) → Mot.SV↑ + Mot.TV↑ (bearing response)
Causal lag: bearing signal appears 15–30 steps AFTER imbalance onset
Multi-hot: [0, 1, 1, 0, 0, 0, 0]
           [normal, bearing_wear, imbalance, cavitation, seal, overload, sensor]
```

**Pair 2: bearing_wear + seal_failure**
```
Physics:   Shaft wobble from worn bearing → mechanical seal face misalignment → leakage
Channels:  Mot.SV↑ + Mot.TV↑ (bearing) → Pres.SV↓ progressive (seal leakage)
Causal lag: seal signal appears 20–40 steps AFTER bearing onset
Multi-hot: [0, 1, 0, 1, 0, 0, 0]
```

**Pair 3: seal_failure + overloading**
```
Physics:   Pressure loss from seal leak → motor works harder to maintain flow → thermal rise
Channels:  Pres.SV↓ (seal) → Temp.SV↑ + Mot.TV↑ (overload response)
Causal lag: overload signal appears 10–25 steps AFTER seal onset
Multi-hot: [0, 0, 0, 1, 1, 0, 0]
```

**Pair 4: cavitation + impeller_imbalance**
```
Physics:   Vapour bubble collapse erosion → blade mass asymmetry → mechanical imbalance
Channels:  Pres.SV erratic + Pmp.SV↑ (cavitation) → Pmp.PV oscillation grows (imbalance)
Causal lag: imbalance signal appears 25–50 steps AFTER cavitation onset
Multi-hot: [1, 0, 1, 0, 0, 0, 0]
           [cavitation, bearing_wear, imbalance, seal, overload, ...]
```

### Sequence Counts

```
Pair 1: impeller_imbalance + bearing_wear  → 400 sequences
Pair 2: bearing_wear + seal_failure        → 400 sequences
Pair 3: seal_failure + overloading         → 400 sequences
Pair 4: cavitation + impeller_imbalance    → 400 sequences
─────────────────────────────────────────────────────────
TOTAL                                      → 1600 sequences
```

### M6B Output Files

```
data/synthetic/M6B_compound_sequences.npy    → shape (1600, 200, 8)
data/synthetic/M6B_compound_metadata.csv     → columns:
    seq_id | primary_fault | secondary_fault | causal_lag |
    severity | fault_stage | label_vector (multi-hot string) |
    source | cluster
outputs/plots/module_06b_*_compound_sanity_plot.png
outputs/reports/module_06b_*_report.md
```

### M6B Physics Invariants

All compound sequences must satisfy:
1. Primary fault channel deviates first (causal_lag enforced)
2. Secondary fault channel onset occurs AT causal_lag ± 5 steps
3. Physical couplings from M2 preserved per fault type
4. No negative pressure, no temperature below cluster minimum
5. Conservation of energy: thermal rise proportional to mechanical dissipation
6. All compound sequences MUST exceed M4 threshold (0.110058) when passed through LSTM-AE
7. No cross-cluster contamination (cavitation ONLY in startup cluster)

### M6B Validation Gates

```
✅ Compound pairs     : 4 × 400 = 1600 sequences
✅ Causal lag check   : secondary onset at primary_onset + causal_lag ± 5 steps
✅ MAE gate           : 100% compound sequences above M4 threshold
✅ Physics violations : 0 in all audited
✅ Multi-hot encoding : no all-zero rows, no single-class compound rows
```

### M6B Paste Text Keys

```
M6B_total_compound_sequences : 1600
M6B_pair1_count              : 400
M6B_pair2_count              : 400
M6B_pair3_count              : 400
M6B_pair4_count              : 400
M6B_mae_gate_pass_pct        : target 100%
M6B_physics_violations       : target 0
M6B_causal_lag_verified      : True/False
Status_for_M6.5              : READY / NEEDS REVIEW
```

---

## M6.5 — Revised Specification: LSTM-AE Feature Extractor + XGBoost Bridge

### Why M6.5 Exists

XGBoost cannot consume raw time-series sequences of shape `(200, 8)`. Flattening
destroys temporal ordering and creates a 1600-dim sparse feature space. M6.5 solves
this by running all sequences through the M4 LSTM-AE (inference only) and extracting
24 statistical features from the reconstruction error array — one static row per sequence.
XGBoost then trains on static tabular rows where every feature carries temporal meaning.

### What Changes in M6.5 (vs v2 completed)

| Item | v2 (completed) | v1.1 (revised) |
|------|----------------|----------------|
| Input sequences | 8400 from M6A | 10000 from M6A + M6B |
| Output rows | 8400 | 10000 |
| Label column | single integer (0–6) | single integer + `label_vector` (multi-hot) |
| New columns | — | `is_compound`, `fault_stage`, `severity` |
| New feature | — | `compound_interaction_flag` (feature 25) |
| Output CSV | 8400 × 25 | 10000 × 29 |

### Complete Feature List (25 Features)

**Per-channel mean reconstruction error — 8 features**
```
mean_err_MotPV, mean_err_MotSV, mean_err_MotTV,
mean_err_PmpPV, mean_err_PmpSV, mean_err_PmpTV,
mean_err_TempSV, mean_err_PresSV
```

**Per-channel max reconstruction error — 8 features**
```
max_err_MotSV, max_err_PmpSV, max_err_PresSV,
max_err_MotTV, max_err_PmpTV, max_err_TempSV,
max_err_MotPV, max_err_PmpPV
```

**Temporal evolution features — 5 features**
```
error_onset_lag      : timestep where error first crosses 2× normal baseline
err_slope_primary    : rate of error growth on highest-error channel
err_auc_primary      : area under error curve (fault energy proxy)
kurtosis_err_PmpSV   : spike character of pump vibration error
kurtosis_err_PresSV  : spike character of pressure error
```

**Cross-channel features — 2 features**
```
corr_delta_PmpSV_PresSV    : change in Pmp.SV↔Pres.SV correlation vs normal baseline
thermal_decoupling_flag    : 1 if Mot.TV error < 0.05 (hydraulic fault indicator)
```

**Fuzzy fault membership — 1 feature**
```
fuzzy_fault_membership : soft score [0.0, 1.0]
  0.0 = clearly normal
  1.0 = clearly fault
  Transition zone calibrated from M4: P95(normal MAE) to P5(fault MAE)
```

**NEW: Compound interaction feature — 1 feature**
```
compound_interaction_flag :
  For single-fault sequences → 0.0
  For compound sequences → Spearman r of (err_primary_channel, err_secondary_channel)
                           shifted by causal_lag
  Physics: if bearing_wear + seal_failure compound,
           Mot.SV error and Pres.SV error should be
           temporally offset by causal_lag steps
           High positive r → confirmed compound causal relationship
```

### M6.5 Output Schema

```
M6_feature_matrix.csv — 10000 rows × 29 columns

Columns:
  [0–7]   mean_err_* per channel          (8 features)
  [8–15]  max_err_* per channel           (8 features)
  [16–20] temporal evolution features     (5 features)
  [21–22] cross-channel features          (2 features)
  [23]    fuzzy_fault_membership          (1 feature)
  [24]    compound_interaction_flag       (1 feature)  ← NEW
  [25]    label                           (int 0–6, primary fault)
  [26]    label_vector                    (str "[0,1,1,0,0,0,0]", multi-hot)
  [27]    is_compound                     (bool)
  [28]    fault_stage                     (str: early/developing/advanced)
  [28]    severity                        (float)
```

### M6.5 Locked Audit Results (v2 — authoritative, still valid for M6A portion)

```
Gate 3 MAE Threshold Check (window=50, threshold=0.110058):

Class              Mean MAE   Gate 3 Pass   Interpretation
─────────────────────────────────────────────────────────
normal             0.1202     86.67%        Probe only — full val FPR 0.55% ✅
bearing_wear       0.0979     13.33%        Mild-sev near-threshold — trend calibration ✅
impeller_imbalance 0.1031     30.00%        Mild sequences dominate — correct ✅
cavitation         0.6747     100.00%       Strongly anomalous — hydraulic shock ✅
seal_failure       0.1961     29.17%        Slow hydraulic fault — Mech C primary path ✅
overloading        0.0930     0.00%         Thermal-dominant — Mech C primary path ✅
sensor_failure     0.1696     93.33%        Single-channel flatline clearly anomalous ✅

Top 5 Fisher Discriminant Features:
  Rank 1: mean_err_PmpSV     (pump vibration — dominant fault channel)
  Rank 2: std_err_PmpSV      (variance of pump vibration error)
  Rank 3: mean_err_TempSV    (thermal drift — overloading discriminator)
  Rank 4: mean_err_MotTV     (motor temperature — bearing/overloading)
  Rank 5: std_err_MotTV      (temperature variance)
```

**Key finding:** Overloading (Gate 3 pass = 0%) and seal_failure (29.17%) are
**thermal-dominant** and **hydraulic-slow** respectively. Their PRIMARY detection
path in M8 is Mechanism C (per-channel drift monitor), NOT single-window threshold.
M7 XGBoost still classifies them correctly via `mean_err_TempSV` (rank 3).

### M6.5 Validation Gates (Revised)

```
✅ Dataset shape    : (10000, 29)          [was (8400, 25)]
✅ NaN count        : 0
✅ Inf count        : 0
✅ Class balance    : 1200 per single-fault class (×7) + 400 per compound pair (×4)
✅ compound_flag    : is_compound=True for all 1600 M6B rows
✅ fuzzy_mean_normal: < 0.15 (near 0)
✅ fuzzy_mean_fault : > 0.70 (near 1)
✅ compound_interaction_flag: > 0.5 for confirmed compound pairs
✅ Fisher rank 1    : must be a PmpSV or vibration channel feature
```

### M6.5 Output Files

```
data/synthetic/M6_feature_matrix.csv         → 10000 rows × 29 columns
outputs/plots/module_065_feature_distributions.png
outputs/plots/module_065_fuzzy_membership_dist.png
outputs/plots/module_065_compound_interaction_dist.png
outputs/reports/module_065_sequence_audit_report.md
```

### M6.5 Paste Text Keys

```
M6p5_feature_matrix_rows         : 10000
M6p5_features_per_row            : 25 (+ 4 metadata cols)
M6p5_compound_rows               : 1600
M6p5_single_fault_rows           : 8400
M6p5_fuzzy_mean_normal           : target < 0.15
M6p5_fuzzy_mean_fault            : target > 0.70
M6p5_compound_interaction_mean   : target > 0.50 for compound seqs
M6p5_fisher_rank1_feature        : must be PmpSV channel
M6p5_top_discriminating_feature  : feature with highest class separation
Status_for_M7                    : READY / NEEDS REVIEW
```

---

## Total Dataset After M6A + M6B + M6.5

```
Source          | Type          | Sequences | Shape per seq
─────────────────────────────────────────────────────────
M6A (revised)   | Single-fault  | 8,400     | (200, 8)
M6B (new)       | Compound      | 1,600     | (200, 8)
─────────────────────────────────────────────────────────
TOTAL RAW       |               | 10,000    | —

M6.5 Feature Matrix:
  Rows    : 10,000
  Columns : 29 (25 features + 4 metadata)
  File    : M6_feature_matrix.csv
  Size    : ~7 MB (well within RAM)
```

**Label distribution in feature matrix:**

```
Single-fault classes (from M6A):
  normal             : 1200
  cavitation         : 1200
  bearing_wear       : 1200
  seal_failure       : 1200
  overloading        : 1200
  impeller_imbalance : 1200
  sensor_failure     : 1200
  Subtotal           : 8400

Compound classes (from M6B, multi-hot):
  impeller_imbalance + bearing_wear  : 400
  bearing_wear + seal_failure        : 400
  seal_failure + overloading         : 400
  cavitation + impeller_imbalance    : 400
  Subtotal                           : 1600

GRAND TOTAL                          : 10000
```

---

## Cross-Module Invariants for M6 Sub-modules

These rules must hold across M6A, M6B, and M6.5 without exception:

1. `segment_id` preserved — windows NEVER cross segment boundaries
2. Normalization baselines LOCKED at `M3_normalization_config.json`
3. Winsor ceilings LOCKED at `M4_spike_config.json` — M6A/M6B DO NOT override
4. M4 threshold `0.110058` is the fault/normal boundary for all validation gates
5. Physical couplings `r > 0.87` must hold in ALL synthetic sequences (per fault type)
6. Conservation of energy and mass in all sequences
7. Cavitation sequences ONLY in startup cluster
8. Overloading sequences ONLY in steady-state cluster
9. Compound sequences: secondary fault channel onset must occur AFTER causal_lag steps
10. `severity` column present in all metadata CSVs (M6A and M6B)
11. `fault_stage` column present in all metadata CSVs (M6A and M6B)
12. M6.5 feature matrix must include all 10000 rows before M7 training begins

---

## Execution Order

```
Step 1: Regenerate M6A
        → module_06a_synthetic_generator_v4.py
        → Outputs: M6_synthetic_sequences.npy + revised metadata

Step 2: Run M6B (new)
        → module_06b_compound_generator.py
        → Outputs: M6B_compound_sequences.npy + M6B metadata

Step 3: Run M6.5 (revised)
        → module_065_sequence_audit.py  (v3)
        → Input: M6A sequences + M6B sequences (combined)
        → Output: M6_feature_matrix.csv (10000 × 29)

Step 4: Verify M6.5 gates pass → Status: READY for M7
```

---

## What This Enables Downstream

| Module | M6 Output Used | Capability Unlocked |
|--------|---------------|---------------------|
| M7 | `M6_feature_matrix.csv` | Multi-label XGBoost + sample weighting by severity |
| M8 | M6A + M6B raw sequences | Compound fault reconstruction — LSTM-AE trained on real mixed signals |
| M10 | `fault_stage` from M6.5 | Progressive confidence API (Stage 1 → Stage 3) |
| M12 | M5 physics engine (fresh) | Adversarial validation — model never saw M12 sequences |

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Initial document — post bias-audit architecture discussion |
| SUPERSEDED | 2026-04-15 | v12.0 architecture: 21 classes, ~27k seqs, M6.5r=~189k×26. Canonical: completed_modules_M5_to_M6p5r.md |

---

**Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head  
**Dataset:** CIRA SACIP — Zenodo 15301820  
**Standard:** ISO 10816-3 vibration, ISO 13373-3 condition monitoring
