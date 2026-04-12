# PumpSmart — Module M7: XGBoost Fault Classifier
## Static Multi-Label Fault Classification via LSTM-AE Feature Bridge

**Document version:** v1.0 — Post Bias-Audit  
**Date:** 2026-04-12  
**Prerequisite:** M6.5 all gates passed | `M6_feature_matrix.csv` (10000 × 29) available  
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)  
**Status:** NOT STARTED — NEXT ACTIVE after M6A regeneration + M6B + M6.5 v3

---

## Why M7 Runs Before M8

```
M7 runs FIRST — it validates that the M6.5 feature matrix is
physically meaningful before M8 uses it to calibrate fuzzy boundaries.

If M7 SHAP is physically wrong → M6.5 features are corrupt →
M8 fuzzy calibration will be wrong → fix M6.5 first, do not proceed to M8.

This sequencing is NON-NEGOTIABLE.
```

---

## What Changed Since Original M7 Spec (Bias-Audit Updates)

| Item | Original Spec | Revised Spec | Reason |
|------|--------------|-------------|--------|
| Input rows | 8400 (M6A only) | **10000** (M6A + M6B compound) | Bias 4 fix — compound faults |
| Input columns | 25 (24 feat + label) | **29** (25 feat + label_vector + is_compound + fault_stage + severity) | M6B metadata |
| Classifier type | Multi-class (single label) | **Multi-label** (MultiOutputClassifier) | Bias 4 fix |
| Label column | single int 0–6 | primary `label` + `label_vector` (multi-hot) | Compound faults |
| Sample weights | inverse class frequency | **inverse class freq × severity weight** | Bias 3 fix |
| Severity weighting | None | `sample_weight = 1 / (severity + 0.1)` | Early-stage focus |
| Output | single fault class | **primary fault + secondary faults + conf%** | Bias 1 fix |
| Confidence gating | None | **75% threshold → Stage 1 / Stage 2 / Stage 3** | Bias 1 fix |
| New feature used | — | `compound_interaction_flag` (feature 25 from M6.5) | Bias 4 |
| SHAP | Global only | **Per-class SHAP beeswarm** (7 single + 4 compound) | Multi-label |

---

## Input Specification

```
File    : data/synthetic/M6_feature_matrix.csv
Shape   : 10000 rows × 29 columns

Column layout:
  [0–7]   mean_err_* per channel          (8 features)
  [8–15]  max_err_* per channel           (8 features)
  [16–20] temporal evolution features     (5 features)
  [21–22] cross-channel features          (2 features)
  [23]    fuzzy_fault_membership          (1 feature)
  [24]    compound_interaction_flag       (1 feature)  ← NEW in M6.5 v3
  [25]    label                           (int 0–6, primary fault)
  [26]    label_vector                    (str → list, multi-hot [0,1,1,0,0,0,0])
  [27]    is_compound                     (bool)
  [28]    fault_stage                     (str: early/developing/advanced)
  [29]    severity                        (float, from M6A/M6B metadata)

Label mapping (primary fault):
  0 = normal
  1 = cavitation
  2 = bearing_wear
  3 = seal_failure
  4 = overloading
  5 = impeller_imbalance
  6 = sensor_failure

Compound label_vector layout:
  [cavitation, bearing_wear, impeller_imbalance, seal_failure,
   overloading, sensor_failure, normal]
  Example: bearing_wear + seal_failure → [0, 1, 0, 1, 0, 0, 0]
```

---

## Architecture

### Multi-Label Design (Bias 4 Fix)

```python
# WHY MULTI-LABEL:
# Real pump faults cascade. One fault generates another.
# A model that outputs ONE label is wrong when TWO are simultaneously active.
# bearing_wear + seal_failure compound sequence:
#   → must output [bearing_wear: 0.87, seal_failure: 0.61]
#   NOT just → [bearing_wear: 0.87]

from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

# One binary XGBoost per fault class (7 classifiers)
base_xgb = xgb.XGBClassifier(
    objective     = 'binary:logistic',
    device        = 'cuda',   # train on RTX 4060
    eval_metric   = 'logloss',
    use_label_encoder = False
)
model = MultiOutputClassifier(base_xgb, n_jobs=1)

# Each classifier outputs P(fault_k=1) independently
# Compound sequences train BOTH classifiers simultaneously
# predict_proba returns list of 7 probability arrays
```

### Training Split

```python
from sklearn.model_selection import train_test_split

X = feature_matrix[feature_cols]           # shape (10000, 25)
Y = label_matrix                           # shape (10000, 7)  multi-hot

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size    = 0.20,
    stratify     = label_primary,          # stratify on primary label
    random_state = 42
)
```

### Sample Weighting (Bias 3 Fix)

```python
# WHY BOTH WEIGHTS:
# 1. Inverse class frequency → balances underrepresented classes
# 2. Severity weight       → early-stage sequences (hard to detect) get MORE weight
#    Early fault (sev=0.1) → weight = 1/(0.1 + 0.1) = 5.0
#    Severe fault (sev=1.0) → weight = 1/(1.0 + 0.1) = 0.91
# Result: model trained HARDER on early-stage sequences
#         which are the operationally critical cases

class_freq   = Y_train.sum(axis=0) / len(Y_train)
class_weight = 1.0 / (class_freq + 1e-6)

# Per-sample severity weight
sev_weight   = 1.0 / (X_train['severity'] + 0.1)

# Combined per-sample weight
sample_weight = sev_weight * class_weight[Y_train.values.argmax(axis=1)]
sample_weight /= sample_weight.mean()     # normalize to unit mean
```

### Hyperparameter Tuning

```python
import optuna

# 50 trials, 5-fold stratified CV on primary label
# Search space:
params = {
    'n_estimators'     : optuna.suggest_int(100, 1000),
    'max_depth'        : optuna.suggest_int(3, 9),
    'learning_rate'    : optuna.suggest_float(0.01, 0.3, log=True),
    'subsample'        : optuna.suggest_float(0.6, 1.0),
    'colsample_bytree' : optuna.suggest_float(0.6, 1.0),
    'min_child_weight' : optuna.suggest_int(1, 10),
    'gamma'            : optuna.suggest_float(0.0, 1.0),
    'reg_alpha'        : optuna.suggest_float(1e-4, 10.0, log=True),
    'reg_lambda'       : optuna.suggest_float(1e-4, 10.0, log=True),
}
# Objective: mean macro F1 across all 7 binary classifiers
```

---

## Progressive Confidence Output (Bias 1 Fix)

This is the output architecture that resolves the causal order uncertainty problem.
A pressure drop at t=0 could be cavitation, seal failure, or sensor failure.
The model must NOT hard-classify until confidence is sufficient.

```python
def classify_with_confidence(X_window, model, threshold=0.75):
    """
    Returns progressive output depending on confidence level.
    """
    proba = model.predict_proba(X_window)  # list of 7 arrays, each shape (n, 2)
    fault_probs = {fault_names[i]: proba[i][0][1] for i in range(7)}

    primary_fault = max(fault_probs, key=fault_probs.get)
    primary_conf  = fault_probs[primary_fault]

    # Secondary faults (compound) = any fault with prob > 0.30
    secondary = {
        k: v for k, v in fault_probs.items()
        if v > 0.30 and k != primary_fault and k != 'normal'
    }

    if primary_conf < 0.50:
        # Stage 1 — Low confidence: list top-3 possibilities only
        top3 = sorted(fault_probs.items(), key=lambda x: -x[1])[:3]
        return {
            'stage'             : 1,
            'message'           : 'Minor anomaly detected — multiple causes possible',
            'top3_candidates'   : top3,
            'action'            : 'Monitor all channels closely'
        }

    elif primary_conf < threshold:  # 0.50 – 0.75
        # Stage 2 — Medium confidence: probable cause with secondary candidates
        return {
            'stage'             : 2,
            'message'           : f'Probable fault: {primary_fault} ({primary_conf:.0%})',
            'secondary_faults'  : secondary,
            'action'            : 'Schedule inspection within 48h'
        }

    else:  # conf >= 0.75
        # Stage 3 — High confidence: confirmed classification
        return {
            'stage'             : 3,
            'message'           : f'CONFIRMED: {primary_fault} ({primary_conf:.0%})',
            'secondary_faults'  : secondary,  # compound faults shown here
            'fault_stage'       : X_window['fault_stage'],
            'action'            : get_action(primary_fault, X_window['fault_stage'])
        }

def get_action(fault, stage):
    urgency = {
        'early'      : 'Schedule inspection within 72h',
        'developing' : 'Schedule inspection within 24h',
        'advanced'   : 'SHUTDOWN RECOMMENDED — immediate maintenance'
    }
    return urgency.get(stage, 'Inspect system')
```

---

## M6.5 Audit Findings Applied to M7

### Finding 1 — Overloading (Gate 3 = 0.00%, MAE = 0.093)

```
Expected SHAP:
  Rank 1 : mean_err_TempSV        (thermal dominant fault)
  Rank 2 : err_slope_TempSV       (rising temperature trend)
  Rank 3 : fuzzy_fault_membership (low MAE but fuzzy captures drift)

Physics: ONLY Temp.SV and Mot.TV channels are elevated in overloading.
XGBoost must exploit thermal features — NOT vibration features.

SHAP PHYSICS FAIL if:
  mean_err_MotSV or mean_err_PmpSV ranks ABOVE mean_err_TempSV for overloading
  → model confusing overloading with vibration fault
  → will misclassify overloading in high-ambient-temperature installations
```

### Finding 2 — Seal Failure (Gate 3 = 29.17%, MAE = 0.196)

```
Expected SHAP:
  Rank 1 : err_slope_PresSV (NEGATIVE slope)  (monotonic pressure decline)
  Rank 2 : thermal_decoupling_flag             (r = -0.013 confirmed M5)
  Rank 3 : pres_monotonic_flag

Key distinction from cavitation:
  Cavitation  → kurtosis HIGH (chaotic pressure) + thermal decoupling MODERATE
  Seal failure → kurtosis LOW  (monotonic decline) + thermal decoupling STRONG

SHAP PHYSICS FAIL if:
  max_err_PresSV (spike feature) ranks ABOVE err_slope_PresSV for seal_failure
  → model confusing seal failure with cavitation
  → seal failure missed until catastrophic pressure loss
```

### Finding 3 — Bearing Seam Discontinuity (5.75% flagged sequences)

```
err_onset_lag must discriminate bearing_wear from impeller_imbalance:
  Bearing wear    → gradual onset → err_onset_lag HIGH (fault develops slowly)
  Impeller imbal  → immediate onset → err_onset_lag LOW (instantaneous imbalance)

Verification:
  err_onset_lag must reference t=0 of the 200-step sequence
  NOT t=49–50 (the seam between spike seed and M5 physics continuation)
  If computed relative to seam → bearing onset_lag = 0 always → useless feature
```

### Finding 4 — Fisher Rank 1 = PmpSV_mean

```
mean_err_PmpSV → expected SHAP rank 1 for: bearing_wear, cavitation, impeller_imbalance
Mot_SV features → expected top-3 for bearing_wear (vibration propagation motor-side)

Fisher rank 1 confirms Pmp.SV is the dominant fault discriminator for vibration classes.
If XGBoost SHAP contradicts Fisher rank → M6.5 feature extraction has a bug.
→ Investigate before proceeding to M8.
```

### Finding 5 — Cavitation (MAE = 0.675, always acute)

```
Expected SHAP:
  Rank 1 : mean_err_PmpSV          (hydraulic shock → pump vibration dominant)
  Rank 2 : kurtosis_err_PresSV     (chaotic pressure — NOT monotonic)
  Rank 3 : thermal_decoupling_flag (r = 0.376 weak — hydraulic not thermal)

Cavitation is easiest to classify — MAE 6.1× above threshold.
Expect F1 > 0.88 without difficulty.
Risk: cavitation dominating overall accuracy metrics — report per-class separately.
```

---

## Expected SHAP Top-3 Per Fault (Full Reference Table)

| Fault Class | Expected SHAP Rank 1 | Expected SHAP Rank 2 | Expected SHAP Rank 3 | Physics Basis |
|-------------|---------------------|---------------------|---------------------|---------------|
| cavitation | mean_err_PmpSV | kurtosis_err_PresSV | thermal_decoupling_flag | Hydraulic shock + chaotic pressure + no thermal |
| bearing_wear | mean_err_PmpSV | err_slope_MotSV | corr_delta_PmpSV_PresSV | Vibration propagation + gradual rise + coupling shift |
| seal_failure | err_slope_PresSV | thermal_decoupling_flag | pres_monotonic_flag | Monotonic pressure decline + hydraulic fault |
| overloading | mean_err_TempSV | err_slope_TempSV | fuzzy_fault_membership | Thermal dominant — only temperature channels elevated |
| impeller_imbalance | mean_err_PmpSV | err_auc_primary | err_onset_lag (LOW) | Immediate high-energy BPF vibration |
| sensor_failure | max_err (one channel) | all others ≈00 | compound_interaction_flag ≈00 | Single channel isolated flatline/spike |
| normal | fuzzy_fault_membership (near 0) | all mean_err near 0 | — | No fault signal |
| **COMPOUND: bearing+seal** | compound_interaction_flag | err_slope_PresSV | err_slope_MotSV | Spearman offset between Mot.SV and Pres.SV errors at causal_lag |
| **COMPOUND: cavitation+imbalance** | compound_interaction_flag | mean_err_PmpSV | kurtosis_err_PresSV | Hydraulic onset then BPF growth |

---

## M7 Validation Gates (10 Gates)

```
GATE-M7-1 : Overall accuracy
             > 85% on test set (stratified split)
             Report single-fault and compound-fault accuracy SEPARATELY
             Compound accuracy expected lower — document, do not hide

GATE-M7-2 : Per-class F1 (all 7 primary classes)
             > 0.80 for ALL classes
             Document any class below 0.80 — investigate before M8

GATE-M7-3 : Cavitation F1
             > 0.88 (safety-critical — hydraulic shock)
             Missed cavitation = impeller pitting within 60–180s of onset

GATE-M7-4 : Sensor failure F1
             > 0.92 (single-channel flatline — easiest class)
             Fail = M6.5 compound_interaction_flag corrupted single-fault sequences

GATE-M7-5 : SHAP physically causal
             Top-3 features per fault class MUST match physics mapping table above
             SHAP computed via SHAP TreeExplainer on X_test
             Report: actual rank 1–3 features per class vs expected

GATE-M7-6 : TV dominance check
             Mot.TV or Pmp.TV must NOT appear in top-3 SHAP for:
               bearing_wear or impeller_imbalance
             (TV channels = placement-dependent contact thermometer)
             (Must not dominate vibration fault classification)
             (Will fail on installations with different casing geometry)

GATE-M7-7 : Overloading SHAP thermal dominance
             mean_err_TempSV SHAP value > mean_err_MotSV SHAP value
             for the overloading class specifically
             (thermal cause must rank above vibration — Finding 1)

GATE-M7-8 : Seal failure SHAP type check
             slope/monotonic feature SHAP value > max/spike feature SHAP value
             for the seal_failure class specifically
             (monotonic pressure decline — NOT spike character like cavitation)
             Distinguishes seal from cavitation (both are pressure faults)

GATE-M7-9 : Bearing thermal lag SHAP order
             err_slope_MotSV SHAP rank ABOVE mean_err_MotTV for bearing_wear
             Physics: vibration rises 20–40 steps BEFORE thermal effect
             FAIL = model detecting thermal consequence, not mechanical cause
             → will misclassify on high-ambient field installations

GATE-M7-10: Seal–cavitation confusion rate
             < 5% of seal_failure test samples predicted as cavitation
             < 5% of cavitation test samples predicted as seal_failure
             (Both are pressure faults — highest risk confusion pair in deployment)
             If > 5%: verify kurtosis_err_PresSV computed correctly in M6.5
```

---

## Adaptive Actions After M7

| M7 Result | Trigger | Action Before M8 |
|-----------|---------|------------------|
| Overloading F1 < 0.80 | Thermal features insufficient | Verify `err_slope_TempSV` computed over full 200 steps in M6.5 |
| Seal–cavitation confusion > 5% | Both pressure faults | Add `onset_speed` feature: fast onset (t<10) = cavitation, slow (t>50) = seal |
| Bearing–imbalance confusion > 5% | Both vibration faults | Verify `err_onset_lag` references t=0, not seam t=49–50 in M6.5 |
| SHAP TV dominance detected | Low-weight channels leaking | Flag for M8: reduce Mot.TV weight 0.3 → 0.1 |
| Gate M7-9 fails (thermal lag wrong) | Thermal over-reliance | Flag for M8: vibration channels must dominate hard constraint |
| Gate M7-8 fails (seal SHAP type) | Slope vs spike confusion | Verify `pres_monotonic_flag` computed over full 200-step sequence in M6.5 |
| Compound F1 < 0.70 | compound_interaction_flag not discriminating | Verify Spearman lag shift implemented correctly in M6.5 |
| Gate M7-5 SHAP wrong for all classes | Feature matrix corrupted | Re-run M6.5 v3 entirely, check M4 inference runs on correct window size (50) |

---

## M7 Outputs

```
models/xgboost_fault_classifier.json        ← cuda-trained model (train/eval only)
models/xgboost_fault_classifier_cpu.json    ← cpu-converted for M10 Flask deployment
outputs/M7_shap_global.png                  ← global feature importance all classes
outputs/M7_shap_per_class/
    M7_shap_cavitation.png
    M7_shap_bearing_wear.png
    M7_shap_seal_failure.png
    M7_shap_overloading.png
    M7_shap_impeller_imbalance.png
    M7_shap_sensor_failure.png
    M7_shap_compound_bearing_seal.png
    M7_shap_compound_cavitation_imbalance.png
outputs/M7_confusion_matrix.png             ← primary label only
outputs/M7_confusion_matrix_compound.png    ← compound pairs only
outputs/M7_per_class_f1.png
outputs/M7_confidence_distribution.png      ← histogram of primary_conf per class
outputs/reports/module_07_xgboost_report.md
```

---

## M7 Paste Text Keys

```
M7_accuracy                       : [% — gate > 85%]
M7_accuracy_single_fault          : [% — on 8400 single-fault test rows]
M7_accuracy_compound              : [% — on 1600 compound test rows, expected lower]
M7_f1_normal                      : [value]
M7_f1_cavitation                  : [value — gate > 0.88]
M7_f1_bearing_wear                : [value]
M7_f1_seal_failure                : [value]
M7_f1_overloading                 : [value — document if < 0.80]
M7_f1_impeller_imbalance          : [value]
M7_f1_sensor_failure              : [value — gate > 0.92]
M7_shap_rank1_cavitation          : [feature — expected mean_err_PmpSV]
M7_shap_rank1_bearing             : [feature — expected mean_err_PmpSV]
M7_shap_rank1_overloading         : [feature — expected mean_err_TempSV]
M7_shap_rank1_seal_failure        : [feature — expected err_slope_PresSV]
M7_shap_rank1_imbalance           : [feature — expected mean_err_PmpSV]
M7_shap_rank1_sensor_failure      : [feature — expected max_err single channel]
M7_gate_tv_dominance              : PASS/FAIL
M7_gate_bearing_thermal_lag       : PASS/FAIL
M7_gate_seal_cavitation_confusion : [% — gate < 5%]
M7_gate_compound_interaction_shap : PASS/FAIL (top-3 for compound classes)
M7_confidence_threshold           : 0.75 (locked)
M7_stage1_pct                     : [% of test samples returning Stage 1 output]
M7_stage3_pct                     : [% of test samples returning Stage 3 output]
M7_all_10_gates_pass              : True/False
Status_for_M8                     : READY/BLOCKED
```

---

## Module Dependency Chain

```
UPSTREAM (required before M7):
  M6A regenerated (Weibull severity + fault_stage + severity columns)
  M6B complete    (1600 compound sequences + multi-hot metadata)
  M6.5 v3 run    (10000 × 29 feature matrix with compound_interaction_flag)
  M6.5 all gates PASS

DOWNSTREAM (M7 outputs feed into):
  M8  → M7 SHAP validation is PREREQUISITE gate before M8 starts
  M10 → xgboost_fault_classifier_cpu.json loaded in Flask classify route
  M10 → M7 progressive confidence output dict is Flask API response schema
```

---

## Cross-Module Invariants Relevant to M7

1. XGBoost: `device='cuda'` for training, `device='cpu'` for M10 deployment
2. Save model: `model.save_model('xgboost_fault_classifier.json')`
3. M7 trains on `M6_feature_matrix.csv` — NEVER on raw sequences `(200, 8)`
4. `predict_proba()` output used — NOT `predict()` — for confidence gating
5. SHAP computed on `X_test` not `X_train` — test-set SHAP only
6. Compound sequences included in BOTH train and test sets
7. `fault_stage` and `severity` columns are metadata only — NOT features for XGBoost
8. `label_vector` column is training target — parse from string to list before fitting

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Standalone file created — split from modules_M7_M8_critical_ML.md. All bias-audit updates incorporated: multi-label, severity weighting, progressive confidence, compound SHAP gates |

---

**Derived from:** `modules_M7_M8_critical_ML.md` v1.0 + bias-audit discussion 2026-04-12  
**Next file:** `module_M8_lstm_ae_v2.md`  
**Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset  
**Standard:** ISO 10816-3 vibration, ISO 13373-3 condition monitoring
