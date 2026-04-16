# PumpSmart — Module M7: XGBoost Fault Classifier
## 22-Class Single-Label Fault Classification via M6.5r Feature Bridge

**Document version:** v3.0 — Architecture Update: 22-class, M6B_feature_matrix input (v14.0)
**Date:** 2026-04-16
**Prerequisite:** M6.5r all gates passed | `data/synthetic/M6B_feature_matrix.csv` (~196,000 × 26) available
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Status:** NOT STARTED — ACTIVE after M6B + M6.5r complete

---

## Why M7 Runs Before M8

```
M7 runs FIRST — it validates that the M6.5r feature matrix is
physically meaningful before M8 uses it to calibrate fuzzy boundaries.

If M7 SHAP is physically wrong → M6.5r features are corrupt →
M8 fuzzy calibration will be wrong → fix M6.5r first, do not proceed to M8.

This sequencing is NON-NEGOTIABLE.
```

---

## v2.0 → v3.0 Architecture Change Summary (v14.0)

| Item | v2.0 | v3.0 (CURRENT — USE THIS) | Reason |
|------|------|--------------------------|--------|
| Input rows | ~189,000 windows | **~196,000 windows** | Label 21 adds ~7,000 windows |
| Classes | 21 | **22** | Label 21 `bearing_wear_gradual` added |
| Label range | `label_int (0–20)` | **`label_int (0–21)`** | Label 21 = bearing_wear_gradual |
| Group B | 5 compound classes (labels 7–11) | **6 compound classes (labels 7–12)** | Label 12 `bearing_wear+seal_failure` added |
| Group C | 4 masked classes (labels 12–15) | **5 masked classes (labels 13–17)** | Label 17 `seal_failure_MotPV_masked` added; all C labels shifted +1 |
| Group D | 3 variant classes (labels 16–18) | **4 variant classes (labels 18–21)** | Labels renumbered 18–21; label 21 = bearing_wear_gradual |
| num_class | 21 | **22** | |
| Gate count | 15 | **16** (Gate M7-14-ext added) | Label 21 slope SHAP gate |

> **v2.0 label map is INVALID.** All references to labels 12–20 must use v3.0 map below.

---

## v1.0 → v2.0 Architecture Change Summary (retained for audit)

| Item | v1.0 (OLD — INVALID) | v2.0 | Reason |
|------|----------------------|------|--------|
| Input file | `M6_feature_matrix.csv` | `M6B_feature_matrix.csv` | M6B expanded dataset |
| Input rows | 10,000 | ~189,000 windows | Full windowed pool |
| Classes | 7 | 21 | Groups A+B+C+D+E |
| Classifier type | Multi-label | Single-label (XGBClassifier) | Compound labels are unique integers |
| Label column | multi-hot | label_int (single integer) | Compound = own label |

> **v1.0 is INVALID.** Do not use any v1.0 numbers, gates, or script fragments.

---

## Input Specification

```
File    : data/synthetic/M6B_feature_matrix.csv
Rows    : ~196,000 windows (from ~28,000 sequences × 7 windows each)
Columns : 26 total
  [0]      label_int    — int 0–21, target column
  [1–8]   mae_MotPV, mae_MotSV, mae_MotTV, mae_PmpPV,
            mae_PmpSV, mae_PmpTV, mae_TempSV, mae_PresSV
  [9–17]  mean_err_MotSV, std_err_MotSV, kurtosis_PmpSV,
            err_slope_MotSV, err_slope_TempSV, err_slope_PresSV,
            thermal_coupling_ratio, cross_channel_MotSV_PmpSV, max_err_all
  [18–25] masked_channel_flag, secondary_onset_lag,
            burst_count, cyclic_baseline_drift,
            multi_sensor_anomaly_count, fault_group_id,
            variant_slope_ratio, thermal_decoupling_flag

Feature columns for XGBoost : [1–25] — 25 features
Target column               : label_int (0–21)
Do NOT include fault_group_id as a feature — verified non-leaking by M6.5r Gate F1,
but confirm SHAP rank ≠ 1 for any class before proceeding to M8.
```

### 22-Class Label Map (from fault_rules_v3.json)

| Label | Class | Group |
|-------|-------|-------|
| 0 | normal | A |
| 1 | bearing_wear | A |
| 2 | impeller_imbalance | A |
| 3 | cavitation | A |
| 4 | seal_failure | A |
| 5 | overloading | A |
| 6 | sensor_failure | A |
| 7 | bearing_wear→overloading | B (compound) |
| 8 | cavitation→seal_failure | B (compound) |
| 9 | impeller_imbalance→bearing_wear | B (compound) |
| 10 | seal_failure→cavitation | B (compound) |
| 11 | impeller_imbalance→cavitation | B (compound) |
| 12 | bearing_wear→seal_failure | B (compound) |
| 13 | bearing_wear_MotSV_masked | C (masked) |
| 14 | cavitation_PresSV_masked | C (masked) |
| 15 | overloading_TempSV_masked | C (masked) |
| 16 | impeller_imbalance_PmpSV_masked | C (masked) |
| 17 | seal_failure_MotPV_masked | C (masked) |
| 18 | cavitation_intermittent | D (severity variant) |
| 19 | seal_failure_fast | D (severity variant) |
| 20 | overloading_cyclic | D (severity variant) |
| 21 | bearing_wear_gradual | D (severity variant) |
| E-a | sensor_failure_2ch_thermal | E (multi-sensor) |
| E-b | sensor_failure_2ch_pumpside | E (multi-sensor) |

> ⚠️ Group E label integers are assigned in `fault_rules_v3.json` (written by M6B Step 3).
> Do NOT hardcode Group E label integers — resolve from fault_rules_v3.json at runtime.

**Always load label map from `fault_rules_v3.json` — NEVER hardcode label strings.**

---

## Architecture

### Single-Label Design Decision (Final)

```
WHY SINGLE-LABEL (not multi-label):

Compound chain sequences (labels 7–12) are assigned UNIQUE INTEGER LABELS.
bearing_wear→overloading = label 7 (not [label 1, label 5] simultaneously).

Reasoning:
  1. The compound chain is a DISTINCT pattern — temporal ordering of two signals
     is fundamentally different from two random co-occurring faults.
  2. A single integer label keeps M7 as a standard 22-way XGBoost classifier.
     No MultiOutputClassifier, no label binarization, no calibration mismatch.
  3. The compound interpretation (Primary → Secondary) lives in M10 API
     label-to-display mapping, NOT in the classifier architecture.
  4. SHAP over 22 scalar outputs is interpretable per class.

M10 API maps: label_int 7 → {"primary": "bearing_wear", "secondary": "overloading",
                              "compound": true, "causal_chain": true}
```

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    objective          = 'multi:softprob',
    num_class          = 22,
    device             = 'cuda',      # RTX 4060 training
    eval_metric        = 'mlogloss',
    tree_method        = 'hist',
    use_label_encoder  = False
)
```

### Training Split

```python
from sklearn.model_selection import train_test_split

X = df[feature_cols]      # shape (~196,000, 25)
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
# Inverse class frequency only — severity info is already in feature slopes
import numpy as np

class_counts = np.bincount(y_train, minlength=22)
class_weight = 1.0 / (class_counts + 1e-6)
class_weight /= class_weight.mean()     # normalize to unit mean

sample_weight = class_weight[y_train.values]

# Note: label 21 (bearing_wear_gradual) has 1,000 training sequences —
# higher than other Group D classes (600 each). Inverse weighting
# will naturally downweight label 21 slightly. This is CORRECT —
# more sequences = more confident signal for a hard-to-detect class.
```

### Hyperparameter Tuning

```python
import optuna

# 50 trials, 5-fold stratified CV on 22-class label
# Search space:
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

## M7 Validation Gates — Per-Group Architecture (16 Gates)

### Overall Gates

```
GATE-M7-1 : Overall macro F1
             > 0.82 across all 22 classes
             Report single-fault (Group A) and per-group separately
             Group B compound and Group D label 21 expected lower — document, do NOT hide

GATE-M7-2 : No single class F1 below floor
             All 22 classes F1 > 0.70
             Exception: label 21 (bearing_wear_gradual) floor = 0.62
               (sub-threshold MAE in low-severity windows is physically correct —
                F1 reflects detection via slope only, not amplitude)
             If any other class < 0.70 → BLOCK M8 until investigated
```

### Group A — Single-Source Gates

```
GATE-M7-3 : Cavitation F1 > 0.88
             Safety-critical — hydraulic shock. MAE 6.1× threshold.
             Missed cavitation = impeller pitting within 60–180s of onset.

GATE-M7-4 : Sensor failure F1 > 0.90
             Single-channel isolated anomaly. Easiest class.
             Fail = masked_channel_flag or multi_sensor_anomaly_count corrupted.

GATE-M7-5 : Seal–cavitation confusion rate < 5%
             Both are pressure faults — highest-risk confusion pair in deployment.
             If > 5%: verify kurtosis_PmpSV and err_slope_PresSV computed correctly.

GATE-M7-6 : Overloading SHAP thermal dominance
             mean_err_TempSV SHAP value > mean_err_MotSV SHAP value
             for overloading class specifically.
             Thermal cause MUST rank above vibration.
             Fail = model confusing overloading with mechanical fault.
```

### Group B — Compound Chain Gates

```
GATE-M7-7 : Group B macro F1 > 0.72
             Lower target than Group A — compound patterns are harder.
             Report per-compound-class F1 individually (6 classes: labels 7–12).

GATE-M7-8 : secondary_onset_lag SHAP rank 1 or 2 for ALL Group B classes (labels 7–12)
             This is the primary discriminator between compound and single-source.
             If secondary_onset_lag not in top-2 → M6.5r secondary_onset_lag
             computation is wrong → BLOCK → fix M6.5r, re-run.

GATE-M7-9 : secondary channel MAE SHAP rank ≤4 for ALL Group B classes
             Secondary channel contribution must be visible to the model.
             Fail = compound sequences have inadequate secondary signal strength.
             Note: feature is now `max_err_all` as secondary channel proxy —
             confirm in SHAP output; `secondary_channel_mae_max` does not exist.
```

### Group C — Masked Fault Gates

```
GATE-M7-10 : Group C macro F1 > 0.68
              Hardest group — primary detector absent.
              Report per-masked-class F1 (5 classes: labels 13–17).
              Label 17 (seal_failure_MotPV_masked) expected weakest — weakest secondary path.

GATE-M7-11 : masked_channel_flag SHAP rank 1 for ALL Group C classes (labels 13–17)
              This binary flag is the primary discriminator for masked faults.
              If not rank 1 → masked_channel_flag logic in M6.5r is broken.
              Fail = secondary-signal path cannot be relied on in deployment.

GATE-M7-12 : Hard case check — seal_failure_PresSV_drifting
              If included: F1 for this class reported separately.
              F1 < 0.60 → remove from training pool, add to M12 adversarial only.
              F1 >= 0.60 → keep in pool, flag as low-confidence in M10 output.
```

### Group D — Severity Variant Gates

```
GATE-M7-13 : Group D macro F1 > 0.75
              Variants share base fault physics but differ in progression shape.
              Report per-variant F1 (4 classes: labels 18–21).

GATE-M7-14 : variant_slope_ratio SHAP in top-3 for label 18 (cavitation_intermittent)
              and label 19 (seal_failure_fast).
              cyclic_baseline_drift SHAP in top-3 for label 20 (overloading_cyclic).
              Fail = model not exploiting progression-shape features.

GATE-M7-14-ext : err_slope_MotSV SHAP in top-3 for label 21 (bearing_wear_gradual)
              Primary discriminator for gradual Paris–Erdogan regime.
              Sub-threshold windows rely entirely on positive slope accumulation.
              Fail = model cannot distinguish gradual bearing wear from normal
              via slope feature → label 21 will be missed in deployment.
              Action on fail: verify err_slope_MotSV computed over full 50-step
              window (not truncated) in M6.5r Section 5.
```

### Group E — Multi-Sensor Failure Gates

```
GATE-M7-15 : multi_sensor_anomaly_count SHAP rank 1 for BOTH Group E classes
              Count of simultaneously anomalous channels = primary discriminator.
              Fail = model conflating multi-sensor failure with compound faults.
```

---

## Expected SHAP Top-3 Per Group (Physics Reference Table)

### Group A — Single-Source

| Class | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|------------|------------|------------|---------------|
| cavitation | `kurtosis_PmpSV` | `mae_PmpSV` | `err_slope_PresSV` (negative) | Hydraulic shock impulses + chaotic pressure |
| bearing_wear | `mae_MotSV` | `err_slope_MotSV` | `thermal_coupling_ratio` | Paris law fatigue + motor vibration rise |
| seal_failure | `err_slope_PresSV` | `mae_PresSV` | `thermal_coupling_ratio` (low) | Monotonic pressure decline + thermal decoupling |
| overloading | `mae_TempSV` | `err_slope_TempSV` | `mae_MotTV` | Thermal dominant — ONLY temperature channels |
| impeller_imbalance | `mae_PmpSV` | `cross_channel_MotSV_PmpSV` | `mae_PmpPV` | BPF broadband + coupled vibration |
| sensor_failure | `masked_channel_flag` | `max_err_all` | `multi_sensor_anomaly_count` (=1) | Single isolated channel anomaly |
| normal | all features near zero | `fault_group_id` = 0 | — | No fault signal |

### Group B — Compound Chains (labels 7–12)

| Class | Label | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|-------|------------|------------|------------|---------------|
| bearing_wear→overloading | 7 | `secondary_onset_lag` | `err_slope_TempSV` | `max_err_all` | Temporal lag then thermal runaway |
| cavitation→seal_failure | 8 | `secondary_onset_lag` | `mae_PmpSV` | `err_slope_PresSV` | PmpSV spikes first → PresSV decline |
| impeller_imbalance→bearing_wear | 9 | `secondary_onset_lag` | `mae_PmpSV` | `mae_MotSV` | BPF first → MotSV exponential after lag |
| seal_failure→cavitation | 10 | `secondary_onset_lag` | `err_slope_PresSV` | `kurtosis_PmpSV` | PresSV decline → PmpSV spikes after lag |
| impeller_imbalance→cavitation | 11 | `secondary_onset_lag` | `mae_PmpSV` | `kurtosis_PmpSV` | BPF → NPSHa drop → bubble collapse |
| bearing_wear→seal_failure | 12 | `secondary_onset_lag` | `mae_MotSV` | `err_slope_PresSV` | Axial thrust → seal face wear → PresSV drop |

**SHAP FAIL condition:** If `fault_group_id` is rank 1 for ANY Group B class → label leakage.
Investigate `fault_group_id` derivation in M6.5r before proceeding to M8.

### Group C — Masked Faults (labels 13–17)

| Class | Label | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|-------|------------|------------|------------|---------------|
| bearing_wear_MotSV_masked | 13 | `masked_channel_flag` | `mae_MotTV` | `mae_TempSV` | MotSV absent; thermal secondary path carries signal |
| cavitation_PresSV_masked | 14 | `masked_channel_flag` | `mae_PmpSV` | `kurtosis_PmpSV` | PresSV absent; PmpSV spikes remain |
| overloading_TempSV_masked | 15 | `masked_channel_flag` | `mae_MotTV` | `err_slope_MotSV` (weak) | TempSV absent; MotTV r=0.997 coupling carries signal |
| impeller_imbalance_PmpSV_masked | 16 | `masked_channel_flag` | `mae_PmpPV` | `cross_channel_MotSV_PmpSV` | PmpSV absent; PmpPV + coupling path |
| seal_failure_MotPV_masked | 17 | `masked_channel_flag` | `err_slope_PresSV` | `mae_PresSV` | MotPV stuck-high; Pres.SV slow drift only — weakest secondary path |

### Group D — Severity Variants (labels 18–21)

| Class | Label | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|-------|------------|------------|------------|---------------|
| cavitation_intermittent | 18 | `burst_count` | `kurtosis_PmpSV` | `variant_slope_ratio` | On-off NPSHa crossing — spike bursts appear/vanish |
| seal_failure_fast | 19 | `err_slope_PresSV` | `variant_slope_ratio` | `mae_PresSV` | PresSV drops in ≤20 steps vs slow seal failure |
| overloading_cyclic | 20 | `cyclic_baseline_drift` | `err_slope_TempSV` | `mae_TempSV` | Sawtooth with rising baseline — not monotonic |
| bearing_wear_gradual | 21 | `err_slope_MotSV` | `mean_err_MotSV` | `std_err_MotSV` | Paris–Erdogan low-ΔK regime; slope positive even at MAE < threshold |

### Group E — Multi-Sensor Failure

| Class | SHAP Rank 1 | SHAP Rank 2 | SHAP Rank 3 | Physics Basis |
|-------|------------|------------|------------|---------------|
| sensor_failure_2ch_thermal | `multi_sensor_anomaly_count` | `mae_MotTV` | `mae_TempSV` | Both thermal channels anomalous; vibration/pressure normal |
| sensor_failure_2ch_pumpside | `multi_sensor_anomaly_count` | `mae_PresSV` | `mae_PmpTV` | Both pump-side sensors fail; motor-side normal |

---

## M6.5 Audit Findings Applied to M7 v3.0

### Finding 1 — Overloading Thermal-Dominant (Gate 3 = 0.00%)

```
Observation: overloading MAE = 0.093 — sub-threshold in M4.
M7 implication: XGBoost WILL classify overloading correctly via
  mae_TempSV + err_slope_TempSV (Fisher rank 3+5 from M6.5 original).
  Classification works even when anomaly detection misses it.

Gate-M7-6 enforces this: mae_TempSV SHAP > mae_MotSV SHAP for overloading.
If this fails → M6.5r thermal feature extraction is wrong.
```

### Finding 2 — Seal Failure Slow Hydraulic (Gate 3 = 29.17%)

```
M7 implication: err_slope_PresSV (negative slope over 50-step window)
is the primary M7 discriminator for seal_failure.
Gate-M7-5 enforces < 5% confusion with cavitation.
Key distinction:
  cavitation  → kurtosis HIGH (chaotic spikes) + slope erratic
  seal_failure → kurtosis LOW (smooth decline) + slope consistently negative
```

### Finding 3 — Bearing Seam Discontinuity (94.25% coherence)

```
M7 implication: err_slope_MotSV computed over 50-step windows is robust
to the t=49→50 seam because windows starting after step 50 see clean M5
physics continuation. Windows crossing the seam are a minority.
Monitor: confusion between bearing_wear (label 1),
  bearing_wear→overloading compound (label 7),
  bearing_wear→seal_failure compound (label 12), and
  bearing_wear_gradual (label 21) on seam-windows.
```

### Finding 4 — Label 21 Sub-Threshold Detection

```
Observation: bearing_wear_gradual (label 21) severity 0.05–0.15 →
  per-window MAE < M4 threshold (0.110058) is PHYSICALLY CORRECT.
M7 implication: XGBoost must classify label 21 via err_slope_MotSV
  (positive, monotonic) alone — not via MAE amplitude.
Gate-M7-14-ext validates this.
Do NOT raise M4 threshold to make label 21 detectable by Layer 1.
Layer 3 (CUSUM) and Layer 4 (Rolling Baseline) are the correct
detection mechanisms for sub-threshold gradual degradation.
```

### Finding 5 — Cavitation Always Acute

```
Cavitation MAE = 0.675 (6.1× threshold).
M7 F1 gate > 0.88 is conservative — expect closer to 0.95.
Risk: cavitation dominating macro F1 and masking weaker classes.
REPORT per-group F1 separately. Do NOT report only macro F1.
```

---

## Adaptive Actions After M7

| M7 Result | Gate | Action Before M8 |
|-----------|------|------------------|
| Overloading F1 < 0.70 | G-M7-2 | Verify `err_slope_TempSV` computed over full 50-step window in M6.5r |
| Seal–cavitation confusion > 5% | G-M7-5 | Verify `kurtosis_PmpSV` and `err_slope_PresSV` both in features |
| `secondary_onset_lag` not rank 1–2 for Group B (labels 7–12) | G-M7-8 | BLOCK — fix M6.5r secondary_onset_lag computation; re-run |
| `masked_channel_flag` not rank 1 for Group C (labels 13–17) | G-M7-11 | BLOCK — fix M6.5r masked_channel_flag logic; re-run |
| `multi_sensor_anomaly_count` not rank 1 for Group E | G-M7-15 | Check Gate G11 pass rate in M6B Step 3 — may be generation issue |
| `fault_group_id` rank 1 for ANY class | SHAP FAIL | BLOCK — label leakage in M6.5r fault_group_id derivation |
| Group B F1 < 0.72 | G-M7-7 | Increase compound sequences in M6B (target 1,200 → 1,500 per class) |
| Group C F1 < 0.68 | G-M7-10 | Increase masked sequences in M6B; verify G10 secondary signal strength |
| Gate G-M7-12 seal hard case F1 < 0.60 | Hard case | Remove from training pool; retain in M12 adversarial config only |
| `err_slope_MotSV` not in top-3 for label 21 | G-M7-14-ext | Verify err_slope_MotSV 50-step linregress in M6.5r Section 5; verify label 21 windows not mislabelled normal |
| Label 21 F1 < 0.62 | G-M7-2 (exception) | Increase label 21 sequences (1,000 → 1,400); check Gate D5 in M6.5r |
| All gates pass | — | Proceed to M8 — write M8 ready status in paste text |

---

## M7 Outputs

```
models/M7_xgboost_classifier.json          ← cuda-trained (22-class)
models/M7_xgboost_classifier_cpu.json      ← cpu-converted for M10 Flask deployment
outputs/M7_shap_group_A.png                ← Group A beeswarm (7 classes)
outputs/M7_shap_group_B.png                ← Group B beeswarm (6 compound classes)
outputs/M7_shap_group_C.png                ← Group C beeswarm (5 masked classes)
outputs/M7_shap_group_D.png                ← Group D beeswarm (4 variant classes incl. label 21)
outputs/M7_shap_group_E.png                ← Group E beeswarm (2 multi-sensor classes)
outputs/M7_confusion_matrix_22class.png    ← full 22×22 confusion matrix
outputs/M7_confusion_matrix_group.png      ← 5×5 group-level confusion
outputs/M7_per_class_f1.png                ← bar chart, 22 classes
outputs/M7_per_group_f1.png                ← bar chart, 5 groups
outputs/reports/module_07_xgboost_report.md
```

---

## M7 Paste Text Keys

```
M7_input_file                        : data/synthetic/M6B_feature_matrix.csv
M7_n_classes                         : 22
M7_n_features                        : 25
M7_n_windows_train                   : [fill]
M7_n_windows_test                    : [fill]
M7_macro_f1_all22                    : [fill — gate > 0.82]
M7_macro_f1_group_A                  : [fill]
M7_macro_f1_group_B                  : [fill — gate > 0.72]
M7_macro_f1_group_C                  : [fill — gate > 0.68]
M7_macro_f1_group_D                  : [fill — gate > 0.75]
M7_macro_f1_group_E                  : [fill]
M7_f1_cavitation                     : [fill — gate > 0.88]
M7_f1_sensor_failure                 : [fill — gate > 0.90]
M7_f1_overloading                    : [fill — document if < 0.80]
M7_f1_seal_failure                   : [fill]
M7_f1_bearing_wear                   : [fill]
M7_f1_impeller_imbalance             : [fill]
M7_f1_bearing_wear_gradual_label21   : [fill — gate floor = 0.62]
M7_shap_rank1_bearing_wear           : [fill — expected mae_MotSV]
M7_shap_rank1_cavitation             : [fill — expected kurtosis_PmpSV]
M7_shap_rank1_seal_failure           : [fill — expected err_slope_PresSV]
M7_shap_rank1_overloading            : [fill — expected mae_TempSV]
M7_shap_rank1_compound_classes       : [fill — expected secondary_onset_lag all 6]
M7_shap_rank1_masked_classes         : [fill — expected masked_channel_flag all 5]
M7_shap_rank1_multisensor_classes    : [fill — expected multi_sensor_anomaly_count]
M7_shap_top3_label21                 : [fill — expected err_slope_MotSV, mean_err_MotSV, std_err_MotSV]
M7_gate_fault_group_id_leakage       : PASS/FAIL
M7_gate_overloading_thermal_shap     : PASS/FAIL
M7_gate_seal_cav_confusion           : [% — gate < 5%]
M7_gate_secondary_onset_lag_rank     : PASS/FAIL
M7_gate_masked_channel_flag_rank     : PASS/FAIL
M7_gate_multisensor_count_rank       : PASS/FAIL
M7_gate_label21_slope_shap           : PASS/FAIL (Gate M7-14-ext)
M7_all_16_gates_pass                 : True/False
Status_for_M8                        : READY/BLOCKED
```

---

## Module Dependency Chain

```
UPSTREAM (required before M7):
  M6B Steps 1–3 complete — M6B_combined_sequences.pkl written (22-class)
  M6.5r all gates W1–W3, F1, D1–D5 PASS (D5 = label 21 slope gate)
  M6B_feature_matrix.csv written (~196,000 × 26)
  fault_rules_v3.json written by M6B Step 3 (LOCKED — 22 classes)

DOWNSTREAM (M7 outputs feed into):
  M8  → M7 SHAP validation is PREREQUISITE gate before M8 starts
        M7 Gate M7-14-ext must PASS before M8 Layer 3/4 design is finalised
  M10 → M7_xgboost_classifier_cpu.json loaded in Flask /classify route
  M10 → label_int 0–21 mapped to display strings via fault_rules_v3.json
  M12 → M7 per-class F1 used as baseline for adversarial degradation test
```

---

## Cross-Module Invariants Relevant to M7

1. `device='cuda'` for XGBoost training; `device='cpu'` for M10 deployment
2. `model.save_model('M7_xgboost_classifier.json')` — JSON format only
3. M7 trains on `M6B_feature_matrix.csv` — NEVER on raw sequences `(200, 8)`
4. `predict_proba()` output used — NOT `predict()` — raw probabilities to M8
5. SHAP computed on `X_test` not `X_train` — test-set SHAP only
6. Label strings always resolved via `fault_rules_v3.json` — NEVER hardcoded
7. `fault_group_id` is metadata — confirm non-leaking via SHAP before M8
8. M8 governs WATCH/WARN/FAULT states — M7 outputs raw `predict_proba` only
9. Label 21 sub-threshold MAE windows are NOT mislabelled — they carry label 21
   and are detected via slope, not amplitude. Do NOT filter them out.

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original: 7-class multi-label, 10,000 × 29, MultiOutputClassifier |
| v2.0 | 2026-04-15 | Full rewrite: 21-class single-label, ~189,000 × 26, XGBClassifier, per-group F1 gates (A–E), SHAP per-group. v1.0 INVALID. |
| v3.0 | 2026-04-16 | **v14.0 UPGRADE**: 22-class throughout. ~196,000 rows. Label map fully corrected: label 12 `bearing_wear→seal_failure` (Group B, 6 classes); labels 13–17 Group C (5 classes); labels 18–21 Group D (4 classes incl. label 21 `bearing_wear_gradual`). Input column list fixed to canonical M6.5r order (removed non-existent `secondary_channel_mae_max`; added `burst_count`, `thermal_decoupling_flag`). `num_class=22`. `minlength=22`. Label 21 weighting note added. Gate M7-14-ext added (err_slope_MotSV SHAP top-3 for label 21). Gate M7-2 exception for label 21 floor = 0.62. Group B SHAP table: label 12 row added. Group C SHAP table: label 17 row added. Group D SHAP table: labels renumbered 18–21; label 21 row added. Finding 4 added for label 21. Bearing seam confusion updated to include labels 12 and 21. Adaptive actions: Group B/C updated for 6/5 classes; label 21 action added. Outputs: 22×22 matrix, Group B 6 classes, C 5 classes, D 4 classes. Paste keys: n_classes=22, label 21 F1 key, label 21 SHAP key, Gate M7-14-ext key, all_16_gates. Dependency chain: D5 gate, 22-class, M7-14-ext note. Invariant 9 added for label 21. |

---

*GitHub is the ONLY source of truth for this spec.*
*Do NOT reference any Spaces .md pathway files — all outdated.*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
