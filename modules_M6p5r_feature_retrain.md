# PumpSmart — Module M6.5r: Feature Matrix Re-Extraction (Retrain)
## 21 Classes · ~27,000 Sequences · 25-Column Output → M7 Input

**Document version:** v1.0 — Final Architecture Lock  
**Date:** 2026-04-15  
**Prerequisite:** M6B complete — `data/synthetic/M6B_combined_sequences.pkl` exists (21 classes, ~27,000 sequences)  
**Script filename:** `module_06p5r_feature_retrain.py`  
**Output file:** `data/synthetic/M6B_feature_matrix.csv` (~27,000 rows × 25 columns + label)

---

## Why M6.5r Exists — Engineering Rationale

M6.5 (original) was run on M6A data only — 7 classes, 8,400 sequences, using a 60-step
window that caused boundary violations across segment edges (6 audit findings logged
in `completed_modules_M1_to_M6p5.md`).

M6.5r re-runs the **entire feature extraction pipeline** on the full M6B combined pool:

| What changed | M6.5 (original) | M6.5r (this module) |
|---|---|---|
| Input sequences | 8,400 (M6A only, 7 classes) | ~27,000 (M6A + M6B, 21 classes) |
| Window size | 60 steps (caused boundary violations) | **50 steps** (fixed — no boundary violations) |
| Classes | 7 | **21** |
| Feature columns | 24 | **25** (+ `secondary_onset_lag` for compound chains) |
| Output file | `M6_feature_matrix.csv` | `M6B_feature_matrix.csv` |
| M7 input | 7-class XGBoost | **21-class XGBoost** |

The window size correction from 60→50 is not cosmetic — it eliminates the
boundary bleed identified in the M6.5 audit where features from one fault segment
contaminated adjacent normal segments. At 200 steps per sequence with onset at step 50,
a 50-step window gives exactly 3 clean windows per sequence: `[0,50)`, `[50,100)`, `[100,150)`.

---

## Input Specification

```
File:    data/synthetic/M6B_combined_sequences.pkl
Format:  dict {label_int: np.ndarray shape (N_sequences, 200, 8)}
Labels:  0–20 (integer) — see fault_rules_v3.json for string mapping
Channels (order fixed): MotPV, MotSV, MotTV, PmpPV, PmpSV, PmpTV, TempSV, PresSV
Dtype:   float32, normalized (all values relative to M3 cluster baselines)
Size:    ~27,000 sequences × 200 timesteps × 8 channels ≈ 173 MB
```

**CRITICAL:** Load `fault_rules_v3.json` (written by M6B Step 3) for label→class mapping.
Do NOT hardcode label strings. All label resolution must go through `fault_rules_v3.json`.

---

## Windowing Logic — Fixed at 50 Steps

```python
WINDOW_SIZE = 50      # Fixed — NOT 60 (boundary violation fix from M6.5 audit)
STRIDE      = 25      # 50% overlap — same as M6A/M6.5
ONSET_STEP  = 50      # All M6B sequences: fault onset at step 50 (enforced in M6B)

# Per sequence: generate windows STRICTLY within sequence boundaries
for start in range(0, SEQ_LEN - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    window = sequence[start:end, :]   # shape (50, 8)
    # → extract 25 features from this window
    # → label = sequence label (inherited, not inferred)

# Windows per 200-step sequence: (200 - 50) / 25 + 1 = 7 windows
# Total rows in matrix: ~27,000 × 7 ≈ 189,000 rows (before deduplication)
# After onset-split (pre-fault windows labelled 'normal'):
#   windows [0,50): pre-onset → label = 0 (normal)
#   windows [50,200): fault-active → label = sequence label
```

**Onset-split rule (mandatory):**  
For compound chain sequences (labels 7–11), the secondary fault onset step is stored in  
`fault_rules_v3.json` as `secondary_onset_step`. Windows before `secondary_onset_step`  
carry the PRIMARY fault label only. Windows after carry the COMPOUND label.  
This ensures the model learns both the early single-fault signal AND the compound transition.

---

## Feature Set — 25 Columns

All features computed **per window** (50 timesteps × 8 channels).
Features are grouped by physical domain.

### Domain 1 — Per-Channel Reconstruction Error (8 features)

Computed using the **frozen M4 LSTM-AE model** (loaded in inference mode).

| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| `mae_MotPV` | mean(|x - x̂|) over 50 steps | Motor velocity reconstruction error |
| `mae_MotSV` | mean(|x - x̂|) over 50 steps | Motor acceleration error — primary bearing fault indicator |
| `mae_MotTV` | mean(|x - x̂|) over 50 steps | Motor thermal error |
| `mae_PmpPV` | mean(|x - x̂|) over 50 steps | Pump velocity error |
| `mae_PmpSV` | mean(|x - x̂|) over 50 steps | Pump acceleration error — BPF / cavitation indicator |
| `mae_PmpTV` | mean(|x - x̂|) over 50 steps | Pump thermal error |
| `mae_TempSV` | mean(|x - x̂|) over 50 steps | Surface thermal error |
| `mae_PresSV` | mean(|x - x̂|) over 50 steps | Pressure reconstruction error |

**M4 model load rule:**
```python
model = LSTMAEModel(config)
model.load_state_dict(torch.load('models/M4_lstm_ae.pt', map_location='cpu'))
model.eval()  # NEVER retrain the M4 model here — inference only
model.to(config.DEVICE)
```
Threshold reference: `M4_threshold = 0.110058` (from M4 output — do NOT recompute).

---

### Domain 2 — Statistical Features (9 features)

| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| `mean_err_MotSV` | mean of windowed MotSV reconstructed error | Trend direction of bearing fault signal |
| `std_err_MotSV` | std of windowed MotSV error | Variability — spike vs gradual fault |
| `kurtosis_PmpSV` | scipy.stats.kurtosis over raw PmpSV window | Impulsiveness — cavitation bubble collapse signature |
| `err_slope_MotSV` | linregress slope of MotSV error over 50 steps | Rate-of-rise — Paris law crack growth velocity |
| `err_slope_TempSV` | linregress slope of TempSV error over 50 steps | Thermal drift rate — overloading severity indicator |
| `err_slope_PresSV` | linregress slope of PresSV error over 50 steps | Pressure drop rate — seal failure vs cavitation |
| `thermal_coupling_ratio` | corr(MotTV_err, TempSV_err) over window | r = 0.9793 baseline; deviations → hydraulic fault |
| `cross_channel_MotSV_PmpSV` | corr(MotSV_err, PmpSV_err) over window | Vibration coupling — imbalance propagation to bearings |
| `max_err_all` | max(all 8 channel MAEs) | Peak anomaly — dominated by primary fault channel |

---

### Domain 3 — Compound / Masked / Variant Discriminators (7 features)

These features are NEW vs M6.5 original. They exist to give M7 the signal
it needs to separate compound chains, masked faults, and severity variants
from their superficially similar single-fault counterparts.

| Feature | Formula | Physical Meaning | Fault Groups Served |
|---------|---------|------------------|---------------------|
| `secondary_channel_mae_max` | max MAE of channels NOT in primary fault's sensor set | Is a second, physically unrelated channel also anomalous? | Group B (compound) |
| `secondary_onset_lag` | step index of secondary channel first exceeding 0.5× threshold, minus primary onset step | Temporal separation between primary and secondary signal | Group B (compound) |
| `masked_channel_flag` | 1 if any channel MAE < 0.02 AND sequence-level MAE > threshold | Primary detector flatlined while overall anomaly present | Group C (masked) |
| `variant_slope_ratio` | err_slope_PmpSV / err_slope_PresSV | Intermittent vs fast vs cyclic progression ratio | Group D (variants) |
| `cyclic_baseline_drift` | mean of TempSV in last 25 steps minus first 25 steps of window | Rising thermal baseline across cycles | Group D (overloading_cyclic) |
| `multi_sensor_anomaly_count` | count of channels with MAE > 0.15 simultaneously | 2 = multi-sensor failure; 1 = single fault | Group E (multi-sensor) |
| `fault_group_id` | {0: normal, 1: single, 2: compound, 3: masked, 4: variant, 5: multi_sensor} | Structural grouping feature — not physics, but M7 regularizer | All groups |

**Note:** `fault_group_id` is a derived metadata feature, not a physics measurement.
It is included as an M7 regularizer to enforce group-level boundaries during XGBoost
tree splitting. It does NOT leak label information — it is set from sequence metadata
(fault_rules_v3.json group field), not from the label integer itself.

---

## Fisher Score Validation Gate (Gate F1)

Before writing the matrix to CSV, compute Fisher discriminant score for every feature:

```
Fisher Score = (between-class variance) / (within-class variance)

Gate F1 Pass: Fisher score > 0.5 for ALL 25 features
Action on fail: flag feature in report — do NOT drop automatically.
                Souvik to review flagged features before M7.
```

Expected top Fisher features (from M6.5 original M6A run):
- `mae_MotSV` — highest for bearing_wear class
- `kurtosis_PmpSV` — highest for cavitation / cavitation_intermittent
- `err_slope_PresSV` — highest for seal_failure, seal_failure_fast
- `thermal_coupling_ratio` — highest for compound thermal chains (C1, C6)
- `secondary_onset_lag` — highest discriminator for Group B vs Group A
- `masked_channel_flag` — binary discriminator for Group C

---

## Output Specification

```
File:    data/synthetic/M6B_feature_matrix.csv
Rows:    ~189,000 (windows) — may reduce after onset-split deduplication
Columns: 26 total:
           [label_int, mae_MotPV, mae_MotSV, mae_MotTV, mae_PmpPV,
            mae_PmpSV, mae_PmpTV, mae_TempSV, mae_PresSV,
            mean_err_MotSV, std_err_MotSV, kurtosis_PmpSV,
            err_slope_MotSV, err_slope_TempSV, err_slope_PresSV,
            thermal_coupling_ratio, cross_channel_MotSV_PmpSV, max_err_all,
            secondary_channel_mae_max, secondary_onset_lag,
            masked_channel_flag, variant_slope_ratio,
            cyclic_baseline_drift, multi_sensor_anomaly_count,
            fault_group_id, label_str]

Size:    ~189,000 × 26 ≈ 38 MB CSV — trivially fits RAM
Dtype:   float32 for all numeric; int for label_int; str for label_str
```

Also write:
```
data/synthetic/M6B_feature_matrix_metadata.json
  ├── window_size: 50
  ├── stride: 25
  ├── onset_step: 50
  ├── n_sequences: [fill at runtime]
  ├── n_windows: [fill at runtime]
  ├── n_classes: 21
  ├── class_distribution: {label_str: count, ...}
  ├── fisher_scores: {feature_name: score, ...}
  ├── gate_F1_status: PASS / FAIL
  ├── m4_threshold_used: 0.110058
  └── generated_by: "module_06p5r_feature_retrain.py"
```

---

## Validation Gates — M6.5r Specific

| Gate | Test | Pass Criterion | Action on Fail |
|------|------|----------------|----------------|
| **W1** | Window boundary integrity | Zero windows cross sequence boundaries | BLOCK — fix windowing loop |
| **W2** | Onset-split correctness | Pre-onset windows labelled 0 (normal); fault-active windows carry correct label | BLOCK |
| **W3** | Compound onset lag | `secondary_onset_lag` > 0 for all Group B windows in fault-active zone | WARN — log cases |
| **F1** | Fisher score gate | All 25 features score > 0.5 | FLAG — review only, do not drop |
| **D1** | Class balance check | No single class > 20% of total windows | WARN — log imbalance |
| **D2** | Masked class secondary signal | `masked_channel_flag = 1` in ≥ 90% of Group C fault-active windows | WARN |
| **D3** | Multi-sensor anomaly count | `multi_sensor_anomaly_count = 2` in ≥ 90% of Group E windows | WARN |

---

## Script Architecture

```
module_06p5r_feature_retrain.py
├── HEADER (config, dirs, logging, results dict — mandatory)
├── SECTION 1: Load M6B sequences + fault_rules_v3.json
├── SECTION 2: Load frozen M4 LSTM-AE (inference only — map_location='cpu')
├── SECTION 3: Window generator (50-step, 25-stride, boundary-safe)
├── SECTION 4: Per-channel MAE extractor (M4 inference pass)
├── SECTION 5: Statistical feature extractor (Domains 1+2)
├── SECTION 6: Compound/masked/variant discriminator features (Domain 3)
├── SECTION 7: Onset-split labelling (compound secondary onset)
├── SECTION 8: Fisher score computation + Gate F1
├── SECTION 9: Validation gates W1–W3, D1–D3
├── SECTION 10: Write M6B_feature_matrix.csv + metadata JSON
├── SECTION 11: Diagnostic plots (Fisher bar chart, class distribution,
│              feature correlation heatmap, per-group MAE boxplot)
├── SECTION 12: PASTE TEXT UPDATE (print between banner lines)
├── SECTION 13: Write outputs/reports/module_06p5r_report.md
└── SECTION 14: FILE MANIFEST + NEXT PROMPT
```

**GPU usage:** M4 inference pass (Section 4) uses `torch.cuda.amp.autocast()`
and `pin_memory=True` DataLoader for batch inference. Batch size = 512 windows.
All other sections run on CPU (pandas, numpy, scipy).

```python
# M4 inference DataLoader
dataset = WindowDataset(windows_tensor)  # shape (N, 50, 8)
loader  = DataLoader(dataset, batch_size=512, pin_memory=True, num_workers=4)
scaler  = torch.cuda.amp.GradScaler()    # NOT used in inference — autocast only
with torch.no_grad():
    with torch.cuda.amp.autocast():
        for batch in loader:
            batch = batch.to(config.DEVICE)
            recon = model(batch)
            mae   = torch.mean(torch.abs(batch - recon), dim=1)  # (batch, 8)
```

---

## M7 Handoff — What M7 Receives

```
Input file:       data/synthetic/M6B_feature_matrix.csv
Rows:             ~189,000 windows
Feature columns:  25 (columns 1–25, excluding label_int and label_str)
Target:           label_int (0–20, integer, single label — XGBoost single-class)
Classes:          21
Class names:      from fault_rules_v3.json — label_str column

M7 must NOT re-extract features.
M7 must NOT re-run M4 inference.
M7 reads the CSV directly and trains on it.
All feature engineering is FROZEN at M6.5r output.
```

**Why single-label (not multi-label) for compound chains:**  
Compound chain sequences (labels 7–11) are assigned a **unique compound label**  
(e.g., `bearing_wear→overloading` = label 7). M7 treats this as a 21-way  
classification, not a multi-output problem. The compound interpretation  
("Primary A → Secondary B") is handled in M10 API label→display mapping,  
not in the classifier architecture. This keeps M7 simple, interpretable,  
and avoids the multi-label probability calibration problem.

---

## SHAP Expectations Post-M7 (Reference for M7 Gate Design)

After M7 trains on M6B_feature_matrix.csv, the following SHAP top-3 features
are physically expected per fault group:

| Fault Group | Expected SHAP Top Feature | Expected #2 | Expected #3 |
|---|---|---|---|
| Group A — single source | `mae_MotSV` (bearing) / `kurtosis_PmpSV` (cavitation) | `err_slope_*` | `max_err_all` |
| Group B — compound | `secondary_onset_lag` | `secondary_channel_mae_max` | `thermal_coupling_ratio` |
| Group C — masked | `masked_channel_flag` | `mae_*` (non-masked channels) | `max_err_all` |
| Group D — variants | `variant_slope_ratio` | `cyclic_baseline_drift` | `err_slope_PresSV` |
| Group E — multi-sensor | `multi_sensor_anomaly_count` | `mae_MotTV` + `mae_TempSV` | `fault_group_id` |

If SHAP top-1 is `fault_group_id` for ANY class → **FAIL** — model is
using metadata, not physics. Investigate label leakage.

---

## Locked Files — DO NOT Overwrite in M6.5r

```
models/M4_lstm_ae.pt              — frozen M4 model weights (inference only)
data/synthetic/M6_sequences.pkl  — M6A frozen sequences
models/M3_normalization_config.json
models/M4_spike_config.json
models/fault_rules.json           — v1, M6A reference (frozen per Invariant 16)
data/synthetic/M6B_combined_sequences.pkl — M6B output (read-only input to M6.5r)
```

---

## Paste Keys — Populate After M6.5r Runs

```
M6p5r_window_size:              50
M6p5r_n_sequences_in:          [fill — target ~27,000]
M6p5r_n_windows_out:           [fill — target ~189,000]
M6p5r_n_classes:               21
M6p5r_gate_W1_boundary:        PASS / FAIL
M6p5r_gate_W2_onset_split:     PASS / FAIL
M6p5r_gate_W3_compound_lag:    PASS / WARN
M6p5r_gate_F1_fisher:          PASS / FAIL — list any flagged features
M6p5r_gate_D1_class_balance:   PASS / WARN — list any class > 20%
M6p5r_gate_D2_masked_flag:     PASS / WARN
M6p5r_gate_D3_multisensor:     PASS / WARN
M6p5r_feature_matrix_rows:     [fill]
M6p5r_feature_matrix_cols:     25
M6p5r_top_fisher_feature:      [fill — expected mae_MotSV or kurtosis_PmpSV]
M6p5r_output_file:             data/synthetic/M6B_feature_matrix.csv
Status_for_M7:                 READY / BLOCKED
```

---

## Module Pathway Context

```
M6B ✅ COMPLETE (prerequisite)
  ↓
M6.5r ← ACTIVE (this module)
  ↓
M7  — 21-class XGBoost
        Input:  data/synthetic/M6B_feature_matrix.csv
        Target: label_int (0–20)
        Output: models/M7_xgboost_classifier.json
  ↓
M8  — LSTM-AE v2 + Fuzzy Logic + 3-state alert
        Threshold unchanged: 0.110058
        TPR now measured across all 20 fault classes separately
  ↓
M9 → M10 → M11 → M12
```

---

*GitHub is the ONLY source of truth for this spec.  
Do NOT reference any Spaces .md pathway files — all outdated.  
Next file to update: `completed_modules_M1_to_M6p5.md` (M6.5 audit section update)*
