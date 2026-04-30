# PumpSmart — Module M7: XGBoost Fault Classifier
## 22-Class Single-Label Fault Classification via M6.5r Feature Bridge

| Field | Value |
|-------|-------|
| **Document version** | v4.0 — v14.2 Domain 4 z_t features + score_A/B/C + label map correction |
| **Date** | 2026-04-21 |
| **Prerequisite** | M6.5r all gates passed — `data/synthetic/M6B_feature_matrix.csv` (~196,000 × ~36) available |
| **Asset** | 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP) |
| **Status** | ACTIVE — M6.5r COMPLETE 2026-04-29. Input file confirmed. Script pending. |
| **Script filename** | `module_07_xgboost_classifier.py` |

> **NOTE:** Read `modules_M6B_script_plan.md` (File 7, v3.0) and `modules_M6p5r_feature_retrain.md` (File 8, v4.0) BEFORE this file.
> - File 7 = M6B step plan, z_t exports, `fault_rules_v3.json` spec
> - File 8 = feature matrix spec (~35 features, Domain 1–4)
> - This file = M7 XGBoost classifier spec only

---

## Why M7 Runs Before M8

> M7 runs **FIRST** — it validates that the M6.5r feature matrix is physically meaningful before M8 uses it to calibrate fuzzy boundaries.
>
> If M7 SHAP is physically wrong → M6.5r features are corrupt → M8 fuzzy calibration will be wrong → fix M6.5r first, do not proceed to M8.
>
> **This sequencing is NON-NEGOTIABLE.**

---

## v3.0 → v4.0 Architecture Change Summary (v14.2)

| Item | v3.0 (OLD) | v4.0 (CURRENT — USE THIS) |
|------|-----------|--------------------------|
| Input columns | 26 (~25 features + label) | ~36 (~35 features + label) |
| Feature count | 25 | ~35 (Domain 4 z_t features added) |
| Missing features in v3.0 | None listed | `score_A`, `score_B`, `score_C`, `onset_order`, `z_t_pca_1`, `z_t_pca_2`, `z_t_norm`, `z_t_recon_err` (Domain 4 — all NEW) |
| Group B SHAP rank 1 | `secondary_onset_lag` | `score_C` (new primary compound discriminator) |
| Group B SHAP rank 2 | mae channel | `secondary_onset_lag` |
| Label 21 SHAP rank 2 | `mean_err_MotSV` | `score_B` (CUSUM drift target — v14.2) |
| Gate M7-8 | `secondary_onset_lag` rank 1–2 | `score_C` rank 1 + `secondary_onset_lag` rank 2 |
| Gate M7-14-ext | `err_slope_MotSV` top-3 only | `err_slope_MotSV` rank 1 + `score_B` rank ≤2 |
| New gate added | None | Gate Z-SHAP (`score_C` + `score_B` routing) |
| `onset_order` feature | Absent | Added — key Group B phase boundary marker |
| Label 21 seq count note | "1,000 training sequences" | **2,000 sequences** (v14.2 correction) |
| Label map Group B label 11 | `impeller_imbalance->cavitation` | **`overloading->bearing_wear`** ✅ CORRECTED |
| Label map Group B label 12 | `bearing_wear->seal_failure` | **`impeller_imbalance->cavitation`** ✅ CORRECTED |
| Adaptive action Group B | "increase to 1,500/class" | Already at 1,500 — removed; replaced with `score_C` calibration action |
| Dependency gates listed | W1–W3, F1, D1–D5 | W1–W3, F1, D1–D5, Z1, Z2, Z3 added |
| Paste key `n_features` | 25 | ~35 |
| Date | 2026-04-16 | 2026-04-21 |

> **⚠️ NOTE ON LABEL MAP CORRECTION:** v3.0 had wrong Group B label 11 and label 12 class names. This was corrected in File 7 v3.0 and File 8 v4.0. v4.0 of this file now matches the canonical v14.2 label map. **All v3.0 references to label 12 = `bearing_wear->seal_failure` are INVALID.**

## v2.0 → v3.0 Architecture Change Summary (retained for audit trail)

| Item | v2.0 | v3.0 |
|------|------|------|
| Input rows | ~189,000 windows | ~196,000 windows |
| Classes | 21 | 22 |
| Label range | label_int (0–20) | label_int (0–21) |
| Group B | 5 compound classes | 6 compound classes (labels 7–12) |
| Group C | 4 masked classes | 5 masked classes (labels 13–17) |
| Group D | 3 variant classes | 4 variant classes (labels 18–21) |
| `num_class` | 21 | 22 |
| Gate count | 15 | 16 (Gate M7-14-ext added) |

> **v2.0 label map is INVALID. v1.0 is INVALID.**

---

## Input Specification

```
File    : data/synthetic/M6B_feature_matrix.csv       ← PENDING (M6.5r output)
Rows    : 526,300 windows (variable seq lengths produce more windows than ~196k spec estimate — correct)
Columns : 34 total (33 features + label_int)
```

### Exact Column Order (matches M6.5r output spec — File 8 v4.0)

```
[0]   label_int               — int 0-21, target column

DOMAIN 1 — Per-channel MAE (8 features):
[1]   mae_MotPV
[2]   mae_MotSV
[3]   mae_MotTV
[4]   mae_PmpPV
[5]   mae_PmpSV
[6]   mae_PmpTV
[7]   mae_TempSV
[8]   mae_PresSV

DOMAIN 2 — Statistical features (9 features):
[9]   mean_err_MotSV
[10]  std_err_MotSV
[11]  kurtosis_PmpSV
[12]  err_slope_MotSV
[13]  err_slope_TempSV
[14]  err_slope_PresSV
[15]  thermal_coupling_ratio
[16]  cross_channel_MotSV_PmpSV
[17]  max_err_all

DOMAIN 3 — Compound/masked/variant/multi-sensor (8 features):
[18]  masked_channel_flag
[19]  secondary_onset_lag
[20]  burst_count
[21]  cyclic_baseline_drift
[22]  multi_sensor_anomaly_count
[23]  fault_group_id
[24]  variant_slope_ratio
[25]  thermal_decoupling_flag

DOMAIN 4 — z_t latent features + TCN-AE scores (NEW v14.2, ~10 features):
[26]  z_t_pca_1
[27]  z_t_pca_2
[28]  z_t_norm
[29]  z_t_recon_err
[30]  score_A
[31]  score_B
[32]  score_C
[33]  onset_order
```

> **NOTE:** Exact column indices [26–33] assume 8 Domain 4 features. If M6.5r Gate Z1 fails and `z_t_pca_3` is added, indices shift by 1. Always read column names from CSV header — **do NOT rely on fixed indices.**

| Column Set | Value |
|-----------|-------|
| Feature columns for XGBoost | All columns except `label_int` (~35 features) |
| Target column | `label_int` (0–21) |

### fault_group_id Rule

`fault_group_id` IS included as a feature column. It is CONFIRMED non-leaking by M6.5r Gate F1 (6 groups vs 22 labels). **HOWEVER:** Gate Z-SHAP in this module validates that `fault_group_id` is NOT SHAP rank 1 for any class. If it is rank 1 for ANY class → BLOCK → label leakage investigation before M8.

### Score Routing — Invariant 19 (NEVER VIOLATE)

| Score | Routes To | M7 Role |
|-------|-----------|---------|
| `score_B` | CUSUM only (M8 Layer 3) | M7 READS as feature only |
| `score_A` | Rolling Baseline only (M8 Layer 4) | M7 READS as feature only |
| `score_C` | XGBoost only (M7 + M10) | M7 READS as feature only |

> M7 reads ALL three as input features. M7 does **NOT** route them — routing is enforced at M8 inference time. Do NOT remove `score_A` or `score_B` from M7 input — they are valid features.

---

### ⚠️ Class Imbalance — MANDATORY XGBoost Configuration (Added v5.0)

Actual class distribution confirmed from M6.5r run (526,300 total windows):

| Largest classes | Windows | % | Risk |
|----------------|---------|---|------|
| label 21 — bearing_wear_gradual | 78,000 | 14.8% | Dominates loss if unweighted |
| label 4 — seal_failure | 51,686 | 9.8% | |
| label 2 — impeller_imbalance | 40,459 | 7.7% | |

| Smallest classes | Windows | % | Risk |
|-----------------|---------|---|------|
| label 19 — seal_failure_fast | 4,000 | 0.8% | Smallest — monitor F1 specifically |
| label 6 — sensor_failure | 6,000 | 1.1% | |
| label 22 — sensor_failure_2ch_thermal | 5,600 | 1.1% | |
| label 23 — sensor_failure_2ch_pump | 5,600 | 1.1% | |

**REQUIRED:** Compute per-class `sample_weight` proportional to inverse frequency
before calling `xgb.train()`. Do NOT use uniform weights — label 19 will be
undertrained and its F1 will be artificially suppressed.

**Label 19 special watch:** Physics visualization (M6.5r Section 11A) shows a
gradual character in the representative sequence rather than the expected ≤20-step
Pres.SV* collapse (turbulent orifice blowout). If F1(label_19) < 0.80 after M7
training, this is the likely cause. Flag for review — NOT a blocking issue for
M7 delivery. Do NOT re-run M6B over this.

---

## 22-Class Label Map (from `fault_rules_v3.json` — v14.2 CORRECTED)

> **ALWAYS** load label map from `models/fault_rules_v3.json` — **NEVER hardcode label strings.** Group E label integers are assigned in `fault_rules_v3.json` (written by M6B Step 3). DO NOT hardcode Group E integers anywhere in this script.

| Label | Class | Group |
|-------|-------|-------|
| 0 | `normal` | A (single-source) |
| 1 | `bearing_wear` | A |
| 2 | `impeller_imbalance` | A |
| 3 | `cavitation` | A |
| 4 | `seal_failure` | A |
| 5 | `overloading` | A |
| 6 | `sensor_failure` | A |
| 7 | `bearing_wear->overloading` | B (compound) |
| 8 | `cavitation->seal_failure` | B (compound) |
| 9 | `impeller_imbalance->bearing_wear` | B (compound) |
| 10 | `seal_failure->cavitation` | B (compound) |
| 11 | `overloading->bearing_wear` | B (compound) ✅ CORRECTED in v4.0 |
| 12 | `impeller_imbalance->cavitation` | B (compound) ✅ CORRECTED in v4.0 |
| 13 | `bearing_wear_MotSV_masked` | C (masked) |
| 14 | `cavitation_PresSV_masked` | C (masked) |
| 15 | `seal_failure_PresSV_drifting` | C (masked) |
| 16 | `overloading_TempSV_masked` | C (masked) |
| 17 | `impeller_imbalance_PmpSV_masked` | C (masked) |
| 18 | `cavitation_intermittent` | D (severity variant) |
| 19 | `seal_failure_fast` | D (severity variant) |
| 20 | `overloading_cyclic` | D (severity variant) |
| 21 | `bearing_wear_gradual` | D (severity variant) |
| E-a | `sensor_failure_2ch_thermal` | E (multi-sensor) |
| E-b | `sensor_failure_2ch_pumpside` | E (multi-sensor) |

---

## Architecture

### Single-Label Design Decision (Final)

**Why single-label (not multi-label):** Compound chain sequences (labels 7–12) are assigned UNIQUE INTEGER LABELS. `bearing_wear->overloading` = label 7 (not [label 1, label 5] simultaneously).

Reasoning:
1. The compound chain is a DISTINCT pattern — temporal ordering of two signals is fundamentally different from two random co-occurring faults.
2. A single integer label keeps M7 as a standard 22-way XGBoost classifier. No MultiOutputClassifier, no label binarization, no calibration mismatch.
3. The compound interpretation (Primary → Secondary) lives in M10 API label-to-display mapping, NOT in the classifier architecture.
4. SHAP over 22 scalar outputs is interpretable per class.

> **M10 API maps:** `label_int 7` → `{"primary": "bearing_wear", "secondary": "overloading", "compound": true, "causal_chain": true}`

### Model Instantiation

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    objective          = 'multi:softprob',
    num_class          = 22,
    device             = 'cuda',        # RTX 4060 training
    eval_metric        = 'mlogloss',
    tree_method        = 'hist',
    use_label_encoder  = False
)

# Deploy (M10 Flask): load with device='cpu'
# model_cpu = xgb.XGBClassifier()
# model_cpu.load_model('models/M7_xgboost_classifier_cpu.json')
```

### Training Split

```python
from sklearn.model_selection import train_test_split

X = df[feature_cols]      # shape (~196,000, ~35)
y = df['label_int']       # shape (~196,000,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    stratify     = y,     # stratify on 22-class label
    random_state = 42
)
```

### Sample Weighting

```python
import numpy as np

# Inverse class frequency — severity info already in feature slopes
class_counts = np.bincount(y_train, minlength=22)
class_weight = 1.0 / (class_counts + 1e-6)
class_weight /= class_weight.mean()     # normalize to unit mean

sample_weight = class_weight[y_train.values]
```

> **Label 21 (`bearing_wear_gradual`) weighting note:** Label 21 has 2,000 training sequences (v14.2 correction — v3.0 incorrectly stated 1,000). Inverse weighting naturally downweights label 21 slightly. This is CORRECT — more sequences = more confident signal for hard-to-detect class. Do NOT manually override label 21 weight upward. Sub-threshold detection is handled by M8 Layer 3 (CUSUM on `score_B`) and Layer 4 (Rolling Baseline on `score_A`) — NOT by inflating M7 weight.

### Hyperparameter Tuning

```python
import optuna

# 50 trials, 5-fold stratified CV on 22-class label
params = {
    'n_estimators'     : optuna.suggest_int(200, 1500),
    'max_depth'        : optuna.suggest_int(3, 9),
    'learning_rate'    : optuna.suggest_float(0.01, 0.3, log=True),
    'subsample'        : optuna.suggest_float(0.6, 1.0),
    'colsample_bytree' : optuna.suggest_float(0.6, 1.0),
    'min_child_weight' : optuna.suggest_int(1, 10),
    'gamma'            : optuna.suggest_float(0.0, 1.0),
    'reg_alpha'        : optuna.suggest_float(1e-4, 10.0, log=True),
    'reg_lambda'       : optuna.suggest_float(1e-4, 10.0, log=True),
}
# Objective: macro F1 across all 22 classes
```

---

## M7 Validation Gates — Per-Group Architecture (17 Gates)

> Gate count increases from 16 (v3.0) to 17 (v4.0): Gate Z-SHAP added.

### Overall Gates

| Gate | Description | Target | Notes |
|------|-------------|--------|-------|
| **M7-1** | Overall macro F1 | >0.82 across all 22 classes | Report single-fault (Group A) AND per-group separately. Group B compound and Group D label 21 expected lower — document, do NOT hide in macro average. |
| **M7-2** | No single class F1 below floor | All 22 classes F1 >0.70 | Exception: label 21 floor = **0.62** (sub-threshold MAE in low-severity windows is physically correct). If any OTHER class <0.70 → BLOCK M8. |

### Group A — Single-Source Gates

| Gate | Description | Target | Notes |
|------|-------------|--------|-------|
| **M7-3** | Cavitation F1 | >0.88 | Safety-critical — hydraulic shock. MAE 6.1× threshold. Missed cavitation = impeller pitting within 60–180s. |
| **M7-4** | Sensor failure F1 | >0.90 | Single-channel isolated anomaly. Easiest class. Fail = `masked_channel_flag` or `multi_sensor_anomaly_count` corrupted. |
| **M7-5** | Seal-cavitation confusion rate | <5% | Both are pressure faults — highest-risk confusion pair in deployment. If >5%: verify `kurtosis_PmpSV` and `err_slope_PresSV` computed correctly. |
| **M7-6** | Overloading SHAP thermal dominance | `mae_TempSV` SHAP > `mae_MotSV` SHAP for overloading class | Thermal cause MUST rank above vibration. Fail = model confusing overloading with mechanical fault. |

### Group B — Compound Chain Gates

| Gate | Description | Target | Notes |
|------|-------------|--------|-------|
| **M7-7** | Group B macro F1 | >0.72 | Lower target than Group A. Report per-compound-class F1 individually (6 classes: labels 7–12). |
| **M7-8** | `score_C` rank 1 AND `secondary_onset_lag` rank 2 | Both in top-2 for ALL Group B classes (labels 7–12) | `score_C` = character change at secondary onset (TCN-AE level). `secondary_onset_lag` = when the change happened (time-domain). If `score_C` NOT rank 1 → verify `z_t_sequences_groupB.pkl` in M6.5r. If `secondary_onset_lag` NOT rank 2 → verify M6.5r Domain 3 computation. EITHER failure → **BLOCK M8 → fix M6.5r → re-run.** |
| **M7-9** | `onset_order` SHAP rank ≤4 | For ALL Group B classes (labels 7–12) | Phase boundary marker — must be visible to the model. Fail = compound phase transition not learnable. If `onset_order` absent from CSV → M6.5r Section 9 incomplete → BLOCK. |

### Group C — Masked Fault Gates

| Gate | Description | Target | Notes |
|------|-------------|--------|-------|
| **M7-10** | Group C macro F1 | >0.68 | Report per-masked-class F1 (5 classes: labels 13–17). Label 17 expected weakest (Pmp.PV only, max alert = WARN). |
| **M7-11** | `masked_channel_flag` SHAP rank 1 | For ALL Group C classes (labels 13–17) | Primary discriminator for masked faults. Fail → secondary-signal path cannot be relied on in deployment. |
| **M7-12** | Hard case check — `seal_failure_PresSV_drifting` (label 15) | F1 <0.60 → remove from pool; F1 ≥0.60 → keep, flag low-confidence | Report label 15 F1 separately in paste text. |

### Group D — Severity Variant Gates

| Gate | Description | Target | Notes |
|------|-------------|--------|-------|
| **M7-13** | Group D macro F1 | >0.75 | Report per-variant F1 (4 classes: labels 18–21). |
| **M7-14** | Variant shape features in top-3 | `variant_slope_ratio` in top-3 for labels 18 & 19; `cyclic_baseline_drift` in top-3 for label 20 | Fail = model not exploiting progression-shape features. |
| **M7-14-ext** | `err_slope_MotSV` rank 1 AND `score_B` rank ≤2 for label 21 | Both in top-2 | `err_slope_MotSV` rank 1 = model learns slope-based early detection. `score_B` rank ≤2 = confirms M8 Layer 3 CUSUM target is correct. EITHER failure → **BLOCK M8 Layer 3 → fix M6.5r → re-run.** |

> **⚠️ NOTE on `err_slope_MotSV` vs `score_B` — DO NOT conflate:**
> - `err_slope_MotSV` = per-window linear regression slope (50 steps) — PRIMARY M7 signal
> - `score_B` = OLS slope of `z_t_recon_err` over N_windows — PRIMARY M8 CUSUM input
>
> Both appearing in label 21 SHAP top-2 = validation that both timescales are learnable.

### Group E — Multi-Sensor Failure Gates

| Gate | Description | Target |
|------|-------------|--------|
| **M7-15** | `multi_sensor_anomaly_count` SHAP rank 1 | For BOTH Group E classes. Fail = model conflating multi-sensor failure with compound faults. |

### Gate Z-SHAP — Score Routing Validation (NEW in v4.0)

Validates that Domain 4 scores behave as Invariant 19 requires.

| Check | Test | Action on Fail |
|-------|------|----------------|
| **C1** | `score_C` SHAP rank 1 for ALL Group B classes (labels 7–12) AND `score_C > Group A score_C_P50` in ≥80% Group B windows | **BLOCK** → verify `z_t_sequences_groupB.pkl` in M6.5r Section 3; verify `score_C = max delta z_t_recon` (NOT mean delta) |
| **C2** | `score_B` SHAP rank ≤2 for label 21 AND `score_B > 0` in ≥90% of label 21 fault-active windows | **BLOCK M8 Layer 3** → verify label 21 z_t export from M6B Step 2 + M6.5r `score_B` OLS slope computation |
| **C3** | `fault_group_id` NOT SHAP rank 1 for ANY of the 22 classes | **BLOCK** → label leakage investigation in M6.5r `fault_group_id` derivation |
| **C4** | `score_A` NOT SHAP rank 1 for any single class | WARN → investigate; do NOT block M8 |

---

## Expected SHAP Top-3 Per Group (Physics Reference Table — v14.2)

### Group A — Single-Source (labels 0–6)

| Class | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|------------|------------|------------|---------------|
| normal | all features near zero | `fault_group_id = 0` | — | No fault signal |
| bearing_wear | `mae_MotSV` | `err_slope_MotSV` | `thermal_coupling_ratio` | Paris law fatigue + motor vibration rise |
| impeller_imbalance | `mae_PmpSV` | `cross_channel_MotSV_PmpSV` | `mae_PmpPV` | BPF broadband + coupled vibration |
| cavitation | `kurtosis_PmpSV` | `mae_PmpSV` | `err_slope_PresSV` (neg) | Hydraulic shock impulses + chaotic pressure |
| seal_failure | `err_slope_PresSV` | `mae_PresSV` | `thermal_coupling_ratio` (low) | Monotonic pressure decline + thermal decoupling |
| overloading | `mae_TempSV` | `err_slope_TempSV` | `mae_MotTV` | Thermal dominant — ONLY temperature channels |
| sensor_failure | `masked_channel_flag` | `max_err_all` | `multi_sensor_anomaly_count` (=1) | Single isolated channel anomaly |

### Group B — Compound Chains (labels 7–12) — v14.2 Corrected Label Map

| Class | Label | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|-------|------------|------------|------------|---------------|
| `bearing_wear->overloading` | 7 | `score_C` | `secondary_onset_lag` | `err_slope_TempSV` | score_C spikes at thermal runaway onset |
| `cavitation->seal_failure` | 8 | `score_C` | `secondary_onset_lag` | `mae_PmpSV` | score_C spikes at PresSV decline onset |
| `impeller_imbalance->bearing_wear` | 9 | `score_C` | `secondary_onset_lag` | `mae_MotSV` | score_C spikes at MotSV exponential rise |
| `seal_failure->cavitation` | 10 | `score_C` | `secondary_onset_lag` | `kurtosis_PmpSV` | score_C spikes at NPSHa/NPSHr crossing |
| `overloading->bearing_wear` | 11 | `score_C` | `secondary_onset_lag` | `mae_MotSV` | score_C spikes at bearing onset after thermal phase |
| `impeller_imbalance->cavitation` | 12 | `score_C` | `secondary_onset_lag` | `kurtosis_PmpSV` | score_C spikes at BPF-to-bubble-nucleation transition |

> `onset_order`: expected in top-5 for ALL Group B classes (phase boundary binary).
> **SHAP FAIL condition:** `fault_group_id` rank 1 for ANY Group B class → label leakage.

### Group C — Masked Faults (labels 13–17)

| Class | Label | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|-------|------------|------------|------------|---------------|
| `bearing_wear_MotSV_masked` | 13 | `masked_channel_flag` | `mae_MotTV` | `mae_TempSV` | MotSV absent; thermal secondary path carries signal |
| `cavitation_PresSV_masked` | 14 | `masked_channel_flag` | `mae_PmpSV` | `kurtosis_PmpSV` | PresSV absent; PmpSV spikes remain |
| `seal_failure_PresSV_drifting` | 15 | `masked_channel_flag` | `mae_MotTV` | `err_slope_MotSV` (weak) | TempSV absent; MotTV coupling carries signal |
| `overloading_TempSV_masked` | 16 | `masked_channel_flag` | `mae_PmpPV` | `cross_channel_MotSV_PmpSV` | PmpSV absent; PmpPV + coupling path |
| `impeller_imbalance_PmpSV_masked` | 17 | `masked_channel_flag` | `err_slope_PresSV` | `mae_PresSV` | MotPV stuck-high; Pres.SV slow drift only — weakest secondary path |

### Group D — Severity Variants (labels 18–21)

| Class | Label | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|-------|------------|------------|------------|---------------|
| `cavitation_intermittent` | 18 | `burst_count` | `kurtosis_PmpSV` | `variant_slope_ratio` | On-off NPSHa crossing — spike bursts appear/vanish |
| `seal_failure_fast` | 19 | `err_slope_PresSV` | `variant_slope_ratio` | `mae_PresSV` | PresSV drops in ≤20 steps vs slow seal failure |
| `overloading_cyclic` | 20 | `cyclic_baseline_drift` | `err_slope_TempSV` | `mae_TempSV` | Sawtooth with rising baseline — not monotonic |
| `bearing_wear_gradual` | 21 | `err_slope_MotSV` | **`score_B`** | `mean_err_MotSV` | Paris-Erdogan low-dK regime; slope positive even at MAE <threshold; `score_B` confirms CUSUM target |

> **Note label 21:** `score_B` replaces `mean_err_MotSV` at rank 2 vs v3.0. This is physically correct: `score_B` = OLS slope of `z_t_recon_err` over N_windows = multi-window confirmation of the Paris law drift that `err_slope_MotSV` measures per single window. Both present in top-3 = M7 and M8 Layer 3 aligned.

### Group E — Multi-Sensor Failure

| Class | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|------------|------------|------------|---------------|
| `sensor_failure_2ch_thermal` | `multi_sensor_anomaly_count` | `mae_MotTV` | `mae_TempSV` | Both thermal channels anomalous; vibration/pressure normal |
| `sensor_failure_2ch_pumpside` | `multi_sensor_anomaly_count` | `mae_PmpSV` | `mae_PmpPV` | Both pump-side sensors fail; motor-side normal |

---

## M6.5 Audit Findings Applied to M7 v4.0

**Finding 1 — Overloading Thermal-Dominant (Gate 3 = 0.00%):**
Overloading MAE = 0.093 — sub-threshold in M4. M7 WILL classify overloading correctly via `mae_TempSV` + `err_slope_TempSV`. Classification works even when anomaly detection misses it. Gate M7-6 enforces `mae_TempSV` SHAP > `mae_MotSV` SHAP for overloading. Fail → M6.5r thermal feature extraction is wrong.

**Finding 2 — Seal Failure Slow Hydraulic (Gate 3 = 29.17%):**
`err_slope_PresSV` (negative slope over 50-step window) is the primary M7 discriminator for `seal_failure`. Gate M7-5 enforces <5% confusion with cavitation. Key distinction: cavitation → kurtosis HIGH (chaotic spikes) + slope erratic; seal_failure → kurtosis LOW (smooth decline) + slope consistently negative.

**Finding 3 — Bearing Seam Discontinuity (94.25% coherence):**
`err_slope_MotSV` computed over 50-step windows is robust to the t=49→50 seam. Monitor confusion between: `bearing_wear` (label 1), `bearing_wear->overloading` compound (label 7), `overloading->bearing_wear` compound (label 11) ✅, `impeller_imbalance->cavitation` compound (label 12) ✅, `bearing_wear_gradual` (label 21) on seam-windows.

**Finding 4 — Label 21 Sub-Threshold Detection:**
`bearing_wear_gradual` (label 21) severity 0.05–0.15 → per-window MAE < M4 threshold (0.110058) is PHYSICALLY CORRECT. XGBoost must classify label 21 via `err_slope_MotSV` (positive, monotonic) alone — not via MAE amplitude. `score_B` in top-3 confirms CUSUM target learnable. Gate M7-14-ext validates `err_slope_MotSV` rank 1 + `score_B` rank ≤2. **Do NOT raise M4 threshold.**

> **M6.5r Gate D5 confirmation (2026-04-29):** `err_slope_MotSV > 0` confirmed in
> 68.7% of label 21 fault-active windows (gate target ≥95% → WARN accepted).
> This is expected physics: Paris law at severity 0.05–0.15 produces slope below
> noise floor at the 50-step window scale.
> `score_B` (z_t drift slope, sequence-level) = positive in **99.4%** of label 21
> windows → M8 Layer 3 CUSUM is fully viable. Do NOT raise M4 threshold.
> Do NOT filter sub-threshold label 21 windows as normal.

**Finding 5 — Cavitation Always Acute:**
Cavitation MAE = 0.675 (6.1× threshold). M7 F1 gate >0.88 is conservative — expect closer to 0.95. Risk: cavitation dominating macro F1 and masking weaker classes. **REPORT per-group F1 separately.**

**Finding 6 — Domain 4 z_t Feature Alignment (NEW in v4.0):**
`score_C` is computed from z_t sequences in M6.5r Domain 4 as the max delta `z_t_recon_err` between consecutive windows over N_windows. For compound chains: `score_C` spikes at `secondary_onset_lag` step. For single-source: `score_C` ≈ low and stable. If `score_C` not rank 1 for Group B → either z_t export from M6B is wrong OR M6.5r `score_C` formula needs verification. **Use max delta — NOT mean delta — for `score_C`.**

> **M6.5r Gate Z2 result (2026-04-29):** `score_C > Group A P50` confirmed in
> 72.5% of Group B windows (gate target ≥80% → WARN accepted).
> Root cause: z_t pkl files use T//50 non-overlapping windowing = 4–18 delta
> points per sequence. Max-delta has lower statistical power than stride-25 would give.
> `onset_order` (Fisher = 9.27×10¹³, rank 1 overall) dominates compound
> classification — `score_C` contributes additively.
> **If Group B macro F1 < 0.72 after M7 training:** revisit score_C formula —
> try mean-delta instead of max-delta as a first diagnostic step.

---

## Adaptive Actions After M7

| M7 Result | Gate | Action Before M8 |
|-----------|------|-----------------|
| Overall macro F1 <0.82 | M7-1 | Report per-group breakdown; identify weakest group |
| Overloading F1 <0.70 | M7-2 | Verify `err_slope_TempSV` computed over full 50-step window in M6.5r |
| Seal-cavitation confusion >5% | M7-5 | Verify `kurtosis_PmpSV` + `err_slope_PresSV` both in features |
| `score_C` NOT rank 1 for any Group B class | M7-8 | **BLOCK** → verify `z_t_sequences_groupB.pkl` in M6.5r Section 3; verify `score_C = max delta z_t_recon` (NOT mean delta) |
| `secondary_onset_lag` NOT rank 2 for any Group B | M7-8 | **BLOCK** → fix M6.5r Domain 3 `secondary_onset_lag` computation |
| `onset_order` NOT rank ≤4 for any Group B class | M7-9 | **BLOCK** → verify `onset_order` column in `M6B_sequence_meta.csv` + M6.5r Section 9 |
| `masked_channel_flag` NOT rank 1 for any Group C | M7-11 | **BLOCK** → fix M6.5r `masked_channel_flag` logic; re-run |
| Label 15 F1 <0.60 | M7-12 | Remove label 15 from training pool; retain in M12 adversarial |
| `multi_sensor_anomaly_count` NOT rank 1 for Group E | M7-15 | Check Gate G11 pass rate in M6B Step 3 |
| `fault_group_id` rank 1 for ANY class | Z-SHAP C3 | **BLOCK** → label leakage in M6.5r `fault_group_id` derivation |
| `score_C` NOT rank 1 for Group B (Z-SHAP C1) | Z-SHAP C1 | **BLOCK** → same as M7-8 `score_C` action above |
| `score_B` NOT rank ≤2 for label 21 (Z-SHAP C2) | Z-SHAP C2 | **BLOCK M8 Layer 3** → verify label 21 z_t export from M6B Step 2 + M6.5r `score_B` OLS slope |
| `score_A` rank 1 for any class | Z-SHAP C4 | WARN → investigate `score_A` formula in M6.5r; do NOT block M8 |
| `err_slope_MotSV` NOT rank 1 for label 21 | M7-14-ext | Verify `err_slope_MotSV` 50-step linregress in M6.5r Section 7; verify label 21 windows not mislabelled normal in M6.5r Section 10 |
| Group B F1 <0.72 | M7-7 | Verify `score_C` head calibration in M6B Step 1 z_t export (do NOT increase Group B sequences — already at 1,500/class) |
| Group C F1 <0.68 | M7-10 | Verify Gate G10 secondary signal strength in M6B Step 2 |
| Label 21 F1 <0.62 | M7-2 (exc) | Increase label 21 sequences (2,000→2,500); check Gate D5 in M6.5r |
| **All 17 gates pass** | — | **Proceed to M8; write READY in paste text** |

---

## M7 Outputs

```
models/M7_xgboost_classifier.json              ← cuda-trained (22-class)
models/M7_xgboost_classifier_cpu.json          ← cpu-converted for M10 Flask deployment
outputs/M7_shap_group_A.png                    ← Group A beeswarm (7 classes)
outputs/M7_shap_group_B.png                    ← Group B beeswarm (6 compound classes)
outputs/M7_shap_group_C.png                    ← Group C beeswarm (5 masked classes)
outputs/M7_shap_group_D.png                    ← Group D beeswarm (4 variant classes incl. label 21)
outputs/M7_shap_group_E.png                    ← Group E beeswarm (2 multi-sensor classes)
outputs/M7_confusion_matrix_22class.png        ← full 22×22 confusion matrix
outputs/M7_confusion_matrix_group.png          ← 5×5 group-level confusion
outputs/M7_per_class_f1.png                    ← bar chart, 22 classes
outputs/M7_per_group_f1.png                    ← bar chart, 5 groups
outputs/M7_domain4_shap_scores.png             ← score_A/B/C + onset_order SHAP values (NEW v4.0)
outputs/reports/module_07_xgboost_report.md
```

---

## Module Dependency Chain

### Upstream (required before M7)

| Dependency | Details |
|-----------|---------|
| M6B Steps 0–3 complete | `M6B_combined_sequences.pkl`, `fault_rules_v3.json`, `z_t_sequences_group[A-E].pkl` (7 pkl files), `M6B_sequence_meta.csv` all written |
| M6.5r gates evaluated 2026-04-29 | 8 PASS, 4 WARN (D3/D5/Z2/F1 — all physically accepted, non-blocking) |
| `M6B_feature_matrix.csv` written and confirmed | **526,300 × 34** (33 features + label_int) — 282.6 MB |

### Downstream (M7 outputs feed into)

| Consumer | What It Uses |
|---------|-------------|
| **M8** | M7 SHAP validation is PREREQUISITE before M8 starts. Gate M7-14-ext PASS → confirms `score_B` learnable → M8 Layer 3 CUSUM safe. Gate Z-SHAP PASS → confirms `score_A/B/C` routing correct. |
| **M10** | `M7_xgboost_classifier_cpu.json` loaded in Flask `/classify` route |
| **M10** | `label_int` 0–21 mapped to display strings via `fault_rules_v3.json` |
| **M12** | M7 per-class F1 used as baseline for adversarial degradation test |

---

## Cross-Module Invariants Relevant to M7

| # | Invariant |
|---|-----------|
| 1 | `device='cuda'` for XGBoost training; `device='cpu'` for M10 deployment |
| 2 | `model.save_model('M7_xgboost_classifier.json')` — JSON format only |
| 3 | M7 trains on `M6B_feature_matrix.csv` — NEVER on raw sequences (seq_len, 8) |
| 4 | `predict_proba()` output used — NOT `predict()` — raw probabilities feed M8 |
| 5 | SHAP computed on X_test not X_train — test-set SHAP only |
| 6 | Label strings always resolved via `fault_rules_v3.json` — NEVER hardcoded |
| 7 | `fault_group_id` is metadata — confirm non-leaking via Gate Z-SHAP C3 |
| 8 | M8 governs WATCH/WARN/DANGER states — M7 outputs raw `predict_proba` only |
| 9 | Label 21 sub-threshold MAE windows are NOT mislabelled — carry label 21, detected via slope not amplitude. Do NOT filter them out. |
| 10 | `score_B` → CUSUM (M8 Layer 3); `score_A` → Rolling Baseline (M8 Layer 4); `score_C` → XGBoost (M7) only. (Invariant 19 — NEVER cross-route) |
| 11 | M7 reads `score_A`, `score_B`, `score_C` as input features — it does NOT route them. Routing is enforced at M8 inference time. |
| 12 | All Domain 4 features are read from `M6B_feature_matrix.csv` — M7 does NOT recompute them. |

---

## M7 Paste Text Keys

> **══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT AFTER M7 COMPLETES ══**

### Input / Training

| Key | Target / Value |
|-----|---------------|
| `M7_input_file` | `data/synthetic/M6B_feature_matrix.csv` |
| `M7_n_classes` | 22 |
| `M7_n_features` | ~35 (confirm exact count at runtime from CSV header) |
| `M7_n_windows_train` | [fill] |
| `M7_n_windows_test` | [fill] |

### Per-Group F1

| Key | Target |
|-----|--------|
| `M7_macro_f1_all22` | [fill — gate >0.82] |
| `M7_macro_f1_group_A` | [fill] |
| `M7_macro_f1_group_B` | [fill — gate >0.72] |
| `M7_macro_f1_group_C` | [fill — gate >0.68] |
| `M7_macro_f1_group_D` | [fill — gate >0.75] |
| `M7_macro_f1_group_E` | [fill] |

### Per-Class F1

| Key | Target |
|-----|--------|
| `M7_f1_cavitation` | [fill — gate >0.88] |
| `M7_f1_sensor_failure` | [fill — gate >0.90] |
| `M7_f1_overloading` | [fill — document if <0.80] |
| `M7_f1_seal_failure` | [fill] |
| `M7_f1_bearing_wear` | [fill] |
| `M7_f1_impeller_imbalance` | [fill] |
| `M7_f1_bearing_wear_gradual_label21` | [fill — gate floor = 0.62] |
| `M7_f1_label15_seal_drifting` | [fill — gate check: <0.60 = remove from pool] |

### SHAP Ranks

| Key | Expected |
|-----|---------|
| `M7_shap_rank1_bearing_wear` | `mae_MotSV` |
| `M7_shap_rank1_cavitation` | `kurtosis_PmpSV` |
| `M7_shap_rank1_seal_failure` | `err_slope_PresSV` |
| `M7_shap_rank1_overloading` | `mae_TempSV` |
| `M7_shap_rank1_compound_all6` | `score_C` for all labels 7–12 |
| `M7_shap_rank2_compound_all6` | `secondary_onset_lag` for all labels 7–12 |
| `M7_shap_rank1_masked_all5` | `masked_channel_flag` for labels 13–17 |
| `M7_shap_rank1_multisensor_both` | `multi_sensor_anomaly_count` |
| `M7_shap_top3_label21` | `err_slope_MotSV`, `score_B`, `mean_err_MotSV` |
| `M7_onset_order_rank_group_B` | [fill — expected ≤4 for all Group B classes] |

### Gate Summary

| Key | Value |
|-----|-------|
| `M7_gate_M7_1_macro_f1` | PASS/FAIL |
| `M7_gate_M7_2_class_floor` | PASS/FAIL [list any class below floor] |
| `M7_gate_M7_3_cavitation_f1` | PASS/FAIL |
| `M7_gate_M7_4_sensor_failure_f1` | PASS/FAIL |
| `M7_gate_M7_5_seal_cav_confusion` | [% — gate <5%] PASS/FAIL |
| `M7_gate_M7_6_overloading_shap` | PASS/FAIL |
| `M7_gate_M7_7_group_B_f1` | PASS/FAIL |
| `M7_gate_M7_8_score_C_rank1_group_B` | PASS/FAIL [list any failing label] |
| `M7_gate_M7_8_lag_rank2_group_B` | PASS/FAIL [list any failing label] |
| `M7_gate_M7_9_onset_order_rank` | PASS/FAIL |
| `M7_gate_M7_10_group_C_f1` | PASS/FAIL |
| `M7_gate_M7_11_masked_flag_rank1` | PASS/FAIL |
| `M7_gate_M7_12_label15_hard_case` | [F1 value] KEEP/REMOVE |
| `M7_gate_M7_13_group_D_f1` | PASS/FAIL |
| `M7_gate_M7_14_variant_shap` | PASS/FAIL |
| `M7_gate_M7_14ext_label21_slope_rank1` | PASS/FAIL |
| `M7_gate_M7_14ext_score_B_rank2` | PASS/FAIL |
| `M7_gate_M7_15_multisensor_rank1` | PASS/FAIL |
| `M7_gate_Z_SHAP_C1_score_C` | PASS/FAIL |
| `M7_gate_Z_SHAP_C2_score_B` | PASS/FAIL |
| `M7_gate_Z_SHAP_C3_fault_group_id` | PASS/FAIL **(BLOCK if FAIL)** |
| `M7_gate_Z_SHAP_C4_score_A` | PASS/WARN |
| `M7_all_17_gates_pass` | True/False |
| `Status_for_M8` | READY/BLOCKED |

> **══ END PASTE UPDATE ══**

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v5.0 | 2026-04-29 | M6.5r COMPLETE update. Input spec corrected to 526,300 × 34 (was ~196,000 × ~36). Class imbalance section added: label 21 = 14.8%, label 19 = 0.8% smallest. Gate D5/Z2 WARN context appended to Findings 4 and 6. Label 19 watch flag added. Status set to ACTIVE. |
| v1.0 | 2026-04-12 | Original: 7-class multi-label, 10,000 × 29, MultiOutputClassifier. **INVALID.** |
| v2.0 | 2026-04-15 | Full rewrite: 21-class single-label, ~189,000 × 26, XGBClassifier, per-group F1 gates (A–E), SHAP per-group. v1.0 INVALID. |
| v3.0 | 2026-04-16 | v14.0 UPGRADE: 22-class throughout. ~196,000 rows. Label map: label 12 `bearing_wear->seal_failure` (Group B 6 classes); labels 13–17 Group C (5 classes); labels 18–21 Group D (4 classes incl. label 21). Input fixed to 26 cols. Gate M7-14-ext added. Group D SHAP labels 18–21. Finding 4 label 21 added. |
| v4.0 | 2026-04-21 | v14.2 MAJOR UPDATE: LABEL MAP CORRECTION — label 11 corrected to `overloading->bearing_wear`; label 12 corrected to `impeller_imbalance->cavitation`. v3.0 label map for labels 11–12 was WRONG. DOMAIN 4 FEATURES ADDED: ~36 columns (~35 features). 8 new Domain 4 features: `z_t_pca_1/2`, `z_t_norm`, `z_t_recon_err`, `score_A/B/C`, `onset_order`. GATE UPDATES: M7-8 `score_C` rank 1 (new primary); M7-9 `onset_order` rank ≤4; M7-14-ext `score_B` rank ≤2 added. NEW GATE Z-SHAP (17 gates total). SHAP TABLES: Group B rank 1 = `score_C`; label 21 rank 2 = `score_B`. Finding 6 added. Label 21 count corrected 1,000→2,000. New output `M7_domain4_shap_scores.png`. Invariants 10–12 added. Paste keys expanded 37→47. |

---

> **GitHub is the ONLY source of truth for this spec.** Do NOT reference any Spaces `.md` pathway files — all outdated.
>
> **Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset
> **Standard:** ISO 10816-3 vibration, ISO 13373-3 condition monitoring
