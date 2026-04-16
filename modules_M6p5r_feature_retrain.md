# PumpSmart — Module M6.5r: Feature Matrix Re-Extraction (Retrain)
## 22 Classes · ~26,000–28,000 Sequences · 26-Column Output (25 features + label) → M7 Input

**Document version:** v3.0
**Date:** 2026-04-16
**Status:** ⬜ NOT STARTED — BLOCKED until `data/synthetic/M6B_combined_sequences.pkl`
and `models/fault_rules_v3.json` exist (both written by M6B Step 3)
**Script filename:** `module_06p5r_feature_retrain.py`
**Output file:** `data/synthetic/M6B_feature_matrix.csv` (~196,000 rows × **26 columns**)

> ⚠️ This spec is LOCKED (v14.0). No output files from this module exist yet.
> All files listed under "Output Specification" are PENDING.

---

## Why M6.5r Exists — Engineering Rationale

M6.5 (original) was run on M6A data only — 7 classes, 8,400 sequences, using a
50-step window correctly (v2 fix applied). M6.5r re-runs the **entire feature
extraction pipeline** on the full M6B combined pool with an expanded feature set
that covers compound, masked, variant, and multi-sensor scenarios.

| What changed | M6.5 (original) | M6.5r (this module) |
|---|---|---|
| Input sequences | 8,400 (M6A only, 7 classes) | ~26,000–28,000 (M6A + M6B, **22 classes**) |
| Window size | 50 steps (v2 fix applied) | **50 steps** (unchanged — correct) |
| Classes | 7 | **22** |
| Feature columns | 24 | **25** (6 new features; +1 vs M6.5’s 24) |
| Total columns in CSV | 25 (24 features + label) | **26 (25 features + label)** |
| Output file | `M6_feature_matrix.csv` | `M6B_feature_matrix.csv` |
| M7 input | 7-class XGBoost | **22-class XGBoost** |

---

## Input Specification

```
File:    data/synthetic/M6B_combined_sequences.pkl      ← ⏳ PENDING (M6B output)
Format:  dict {label_int: np.ndarray shape (N_sequences, 200, 8)}
Labels:  0–21 (integer) — see fault_rules_v3.json for string mapping
Channels (order fixed): MotPV, MotSV, MotTV, PmpPV, PmpSV, PmpTV, TempSV, PresSV
Dtype:   float32, normalized (all values relative to M3 cluster baselines)
Size:    ~26,000–28,000 sequences × 200 timesteps × 8 channels ≈ 175–190 MB

Also required:
models/fault_rules_v3.json   ← ⏳ PENDING (written by M6B Step 3)
  — DO NOT attempt to load this file before M6B Step 3 completes.
  — All label resolution must go through fault_rules_v3.json.
  — Do NOT hardcode label strings anywhere in this script.
```

---

## Windowing Logic — Fixed at 50 Steps

```python
WINDOW_SIZE = 50      # Fixed — matches M4 WINDOW_SIZE=50 in config.py
STRIDE      = 25      # 50% overlap — same as M6A/M6.5
ONSET_STEP  = 50      # All M6B sequences: fault onset at step 50 (enforced in M6B)

# Per sequence: generate windows STRICTLY within sequence boundaries
for start in range(0, SEQ_LEN - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    window = sequence[start:end, :]   # shape (50, 8)
    # → extract 25 features from this window
    # → label = sequence label (inherited, not inferred)

# Windows per 200-step sequence: (200 - 50) / 25 + 1 = 7 windows
# Total rows in matrix: ~28,000 × 7 ≈ 196,000 rows (target ~196,000)
# After onset-split (pre-fault windows labelled 'normal'):
#   windows [0,50): pre-onset → label = 0 (normal)
#   windows [50,200): fault-active → label = sequence label
```

**Onset-split rule (mandatory):**
For compound chain sequences (labels 7–12), the secondary fault onset step is stored
in `M6B_sequence_meta.csv` as `secondary_onset_lag`. Windows before
`secondary_onset_step` carry the PRIMARY fault label only. Windows after carry
the COMPOUND label. This ensures the model learns both the early single-fault
signal AND the compound transition.

**Label 21 windowing note:**
Label 21 (`bearing_wear_gradual`) sequences have MAE < threshold in many early windows
(severity 0.05–0.15). This is PHYSICALLY CORRECT — do NOT re-label these windows as normal.
All label 21 fault-active windows (t ≥ 50) carry label 21 regardless of per-window MAE.
The gradual slope (`err_slope_MotSV > 0`) accumulates across windows — this is the signal.

---

## Feature Set — 25 Features + Label = 26 Columns Total

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
model.load_state_dict(torch.load('models/lstm_ae_baseline_best.pth', map_location='cpu'))
model.eval()  # NEVER retrain the M4 model here — inference only
model.to(config.DEVICE)
```
Threshold reference: `M4_threshold = 0.110058` (from `M4_threshold_config.json` — do NOT recompute).

---

### Domain 2 — Statistical Features (9 features)

| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| `mean_err_MotSV` | mean of windowed MotSV reconstructed error | Trend direction of bearing fault signal |
| `std_err_MotSV` | std of windowed MotSV error | Variability — spike vs gradual fault |
| `kurtosis_PmpSV` | scipy.stats.kurtosis over raw PmpSV window | Impulsiveness — cavitation bubble collapse signature |
| `err_slope_MotSV` | linregress slope of MotSV error over 50 steps | Rate-of-rise — Paris law crack growth velocity; **primary discriminator for label 21** |
| `err_slope_TempSV` | linregress slope of TempSV error over 50 steps | Thermal drift rate — overloading severity indicator |
| `err_slope_PresSV` | linregress slope of PresSV error over 50 steps | Pressure drop rate — seal failure vs cavitation |
| `thermal_coupling_ratio` | corr(MotTV_err, TempSV_err) over window | r = 0.9793 baseline; deviations → hydraulic fault |
| `cross_channel_MotSV_PmpSV` | corr(MotSV_err, PmpSV_err) over window | Vibration coupling — imbalance propagation to bearings |
| `max_err_all` | max(all 8 channel MAEs) | Peak anomaly — dominated by primary fault channel |

**Note on `err_slope_MotSV` for label 21:**
For `bearing_wear_gradual`, `err_slope_MotSV` is expected to be **positive and consistent
across windows** even when per-window MAE < threshold. This is the primary XGBoost
discriminator for label 21. Gate D5 validates this explicitly.

---

### Domain 3 — Compound / Masked / Variant / Multi-Sensor Discriminators (8 features)

Domain 1 (8) + Domain 2 (9) + Domain 3 (8) = **25 features total**.
CSV has 26 columns: 25 features + `label_int`.

| Feature | Formula | Physical Meaning | Fault Groups Served |
|---------|---------|------------------|---------------------|
| `masked_channel_flag` | 1 if any channel MAE < 0.02 AND sequence-level MAE > threshold | Primary detector flatlined while overall anomaly present | Group C (masked, labels 13–17) |
| `secondary_onset_lag` | step index of secondary channel first exceeding 0.5× threshold minus primary onset step | Temporal separation between primary and secondary signal | Group B (compound, labels 7–12) |
| `burst_count` | count of MAE spikes > threshold separated by ≥10 normal steps in 200-step sequence | Number of discrete anomaly bursts — intermittent vs sustained | Group D label 18 (cavitation_intermittent) |
| `cyclic_baseline_drift` | mean of TempSV in last 25 steps minus first 25 steps of window | Rising thermal baseline across cycles | Group D label 20 (overloading_cyclic) |
| `multi_sensor_anomaly_count` | count of channels with MAE > 0.15 simultaneously | 2 = multi-sensor failure; 1 = single fault | Group E (multi-sensor) |
| `fault_group_id` | {0: normal, 1: single_source, 2: compound, 3: masked, 4: variant, 5: multi_sensor} | Structural grouping — M7 tree-splitting regularizer | All groups |
| `variant_slope_ratio` | err_slope_PmpSV / err_slope_PresSV | Intermittent vs fast vs cyclic progression ratio | Group D labels 19/20 (fast/cyclic) |
| `thermal_decoupling_flag` | 1 if thermal_coupling_ratio < 0.5 (baseline r=0.9793) | Coupling broken — hydraulic fault present | Groups B, C hydraulic chain |

**Notes on Domain 3:**
- `fault_group_id` is derived from `M6B_sequence_meta.csv` group field — NOT from label integer.
  It does NOT leak label information (group is broader than label: 6 groups vs 22 labels).
  If M7 SHAP top-1 = `fault_group_id` for ANY class → **FAIL** (label leakage investigation required).
- `burst_count` is computed over the full 200-step sequence (not per 50-step window);
  the per-window value = 1 if the window contains a burst, 0 otherwise.
- `secondary_onset_lag` = 0 for all non-compound sequences (Groups A, C, D, E).
- For label 21: `variant_slope_ratio` is near-zero (Mot.SV slope >> Pres.SV slope);
  `burst_count` = 0 (no spikes); `cyclic_baseline_drift` ≈ 0. These ZERO values ARE
  discriminative for label 21 vs other Group D classes.

---

## Fisher Score Validation Gate (Gate F1)

Before writing the matrix to CSV, compute Fisher discriminant score for every feature:

```
Fisher Score = (between-class variance) / (within-class variance)

Gate F1 Pass: Fisher score > 0.5 for ALL 25 features
Action on fail: flag feature in report — do NOT drop automatically.
                Souvik to review flagged features before M7.
```

Expected top Fisher features (from M6.5 original + physics reasoning):
- `mae_MotSV` — highest for bearing_wear class
- `kurtosis_PmpSV` — highest for cavitation / cavitation_intermittent (label 18)
- `err_slope_PresSV` — highest for seal_failure, seal_failure_fast (label 19)
- `err_slope_MotSV` — highest discriminator for label 21 (bearing_wear_gradual)
- `thermal_coupling_ratio` — highest for compound thermal chains (labels 7, 9)
- `secondary_onset_lag` — highest discriminator for Group B (labels 7–12) vs Group A
- `masked_channel_flag` — binary discriminator for Group C (labels 13–17)
- `burst_count` — highest discriminator for cavitation_intermittent (label 18)
- `multi_sensor_anomaly_count` — highest discriminator for Group E

---

## Output Specification (⚠️ ALL PENDING — written when M6.5r script runs)

```
File:    data/synthetic/M6B_feature_matrix.csv         ← ⏳ PENDING
Rows:    ~196,000 (windows) — may reduce after onset-split deduplication
Columns: 26 total:
           [label_int,
            mae_MotPV, mae_MotSV, mae_MotTV, mae_PmpPV,
            mae_PmpSV, mae_PmpTV, mae_TempSV, mae_PresSV,
            mean_err_MotSV, std_err_MotSV, kurtosis_PmpSV,
            err_slope_MotSV, err_slope_TempSV, err_slope_PresSV,
            thermal_coupling_ratio, cross_channel_MotSV_PmpSV, max_err_all,
            masked_channel_flag, secondary_onset_lag,
            burst_count, cyclic_baseline_drift,
            multi_sensor_anomaly_count, fault_group_id,
            variant_slope_ratio, thermal_decoupling_flag]

Size:    ~196,000 × 26 ≈ 39 MB CSV — trivially fits RAM
Dtype:   float32 for all numeric; int for label_int
```

Also write:
```
data/synthetic/M6B_feature_matrix_metadata.json        ← ⏳ PENDING
  ├── window_size: 50
  ├── stride: 25
  ├── onset_step: 50
  ├── n_sequences: [fill at runtime]
  ├── n_windows: [fill at runtime]
  ├── n_classes: 22
  ├── class_distribution: {label_str: count, ...}
  ├── fisher_scores: {feature_name: score, ...}
  ├── gate_F1_status: PASS / FAIL
  ├── m4_threshold_used: 0.110058
  └── generated_by: "module_06p5r_feature_retrain.py"

outputs/reports/module_06p5r_report.md                 ← ⏳ PENDING
```

---

## Validation Gates — M6.5r Specific

| Gate | Test | Pass Criterion | Action on Fail |
|------|------|----------------|----------------|
| **W1** | Window boundary integrity | Zero windows cross sequence boundaries | BLOCK — fix windowing loop |
| **W2** | Onset-split correctness | Pre-onset windows labelled 0 (normal); fault-active windows carry correct label | BLOCK |
| **W3** | Compound onset lag | `secondary_onset_lag` > 0 for all Group B (labels 7–12) windows in fault-active zone | WARN — log cases |
| **F1** | Fisher score gate | All 25 features score > 0.5 | FLAG — review only, do not drop |
| **D1** | Class balance check | No single class > 20% of total windows | WARN — log imbalance |
| **D2** | Masked class secondary signal | `masked_channel_flag = 1` in ≥ 90% of Group C (labels 13–17) fault-active windows | WARN |
| **D3** | Multi-sensor anomaly count | `multi_sensor_anomaly_count = 2` in ≥ 90% of Group E windows | WARN |
| **D4** | burst_count gate | `burst_count` ≥ 2 for ≥ 95% of label 18 (cavitation_intermittent) sequences | WARN |
| **D5** | Label 21 slope gate | `err_slope_MotSV` > 0 in ≥ 95% of label 21 (bearing_wear_gradual) fault-active windows | WARN — critical for Layer 3/4 detection |

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
├── SECTION 6: Compound/masked/variant/multi-sensor discriminator features (Domain 3)
├── SECTION 7: Onset-split labelling (compound secondary onset, label 21 sub-threshold note)
├── SECTION 8: Fisher score computation + Gate F1
├── SECTION 9: Validation gates W1–W3, D1–D5
├── SECTION 10: Write M6B_feature_matrix.csv + metadata JSON
├── SECTION 11: Diagnostic plots (Fisher bar chart, class distribution,
│              feature correlation heatmap, per-group MAE boxplot,
│              label 21 err_slope_MotSV histogram)
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
Rows:             ~196,000 windows
Feature columns:  25 (columns 1–25, i.e. all columns except label_int)
Target:           label_int (0–21, integer, single label — XGBoost single-class)
Classes:          22
Class names:      from fault_rules_v3.json — label_str column

M7 must NOT re-extract features.
M7 must NOT re-run M4 inference.
M7 reads the CSV directly and trains on it.
All feature engineering is FROZEN at M6.5r output.
```

**Why single-label (not multi-label) for compound chains:**
Compound chain sequences (labels 7–12) are assigned a **unique compound label**
(e.g., `bearing_wear+overloading` = label 7). M7 treats this as a 22-way
classification, not a multi-output problem. The compound interpretation
(“Primary A → Secondary B”) is handled in M10 API label→display mapping,
not in the classifier architecture.

---

## SHAP Expectations Post-M7 (Reference for M7 Gate Design)

| Fault Group | Expected SHAP Top Feature | Expected #2 | Expected #3 |
|---|---|---|---|
| Group A — single source | `mae_MotSV` (bearing) / `kurtosis_PmpSV` (cavitation) | `err_slope_*` | `max_err_all` |
| Group B — compound (labels 7–12) | `secondary_onset_lag` | `thermal_coupling_ratio` | `max_err_all` |
| Group C — masked (labels 13–17) | `masked_channel_flag` | `mae_*` (non-masked channels) | `max_err_all` |
| Group D label 18 — intermittent | `burst_count` | `kurtosis_PmpSV` | `variant_slope_ratio` |
| Group D labels 19/20 — fast/cyclic | `variant_slope_ratio` | `cyclic_baseline_drift` | `err_slope_PresSV` |
| Group D label 21 — gradual | `err_slope_MotSV` | `mean_err_MotSV` | `std_err_MotSV` |
| Group E — multi-sensor | `multi_sensor_anomaly_count` | `mae_MotTV` + `mae_TempSV` | `thermal_decoupling_flag` |

**SHAP gate:** If `fault_group_id` is SHAP top-1 for ANY class → FAIL — label leakage investigation required.

---

## Locked Files — DO NOT Overwrite in M6.5r

```
models/lstm_ae_baseline_best.pth          — frozen M4 model weights (inference only)
data/synthetic/M6_sequences.pkl           — M6A frozen sequences
models/M3_normalization_config.json       — LOCKED baselines
models/M4_spike_config.json               — LOCKED winsor bounds
models/M4_threshold_config.json           — threshold=0.110058 (LOCKED)
models/fault_rules.json                   — v1 M6A reference (frozen per Invariant 16)
data/synthetic/M6B_combined_sequences.pkl — M6B output (read-only input to M6.5r)
```

---

## Paste Keys (⚠️ ALL PENDING — populate after M6.5r script runs)

```
M6p5r_window_size                  : 50
M6p5r_n_sequences_in               : [fill after run — target ~26,000–28,000]
M6p5r_n_windows_out                : [fill after run — target ~196,000]
M6p5r_n_classes                    : 22
M6p5r_feature_matrix_rows          : [fill after run — target ~196,000]
M6p5r_feature_matrix_cols          : 26 (25 features + label_int)
M6p5r_gate_W1_boundary             : [PASS/FAIL]
M6p5r_gate_W2_onset_split          : [PASS/FAIL]
M6p5r_gate_W3_compound_lag         : [PASS/WARN]
M6p5r_gate_F1_fisher               : [PASS/FAIL — list any flagged features]
M6p5r_gate_D1_class_balance        : [PASS/WARN — list any class > 20%]
M6p5r_gate_D2_masked_flag          : [PASS/WARN]
M6p5r_gate_D3_multisensor          : [PASS/WARN]
M6p5r_gate_D4_burst_count          : [PASS/WARN]
M6p5r_gate_D5_label21_slope        : [PASS/WARN — err_slope_MotSV > 0 in ≥95% label 21 windows]
M6p5r_top_fisher_feature           : [fill after run — expected mae_MotSV or kurtosis_PmpSV]
M6p5r_label21_slope_pct_positive   : [% label 21 fault-active windows with err_slope_MotSV > 0]
M6p5r_output_file                  : data/synthetic/M6B_feature_matrix.csv
Status_for_M7                      : PENDING — set to READY after all BLOCK gates pass
```

---

## Module Pathway — Corrected Status (v14.0)

```
M6A ✅ COMPLETE (8,400 seq, 7 classes) — LOCKED
  ↓
M6B 🔴 NEXT ACTIVE — spec locked (v14.0), script not yet run
  Outputs needed: M6B_combined_sequences.pkl, M6B_sequence_meta.csv,
                  fault_rules_v3.json (22-class), all M6B_sequences_group*.pkl
  ↓
M6.5r ⬜ NOT STARTED — blocked until M6B Step 3 completes
  This module — extracts 26-column feature matrix from M6B sequences
  Output: data/synthetic/M6B_feature_matrix.csv (~196,000 × 26)
  ↓
M7 ⬜ NOT STARTED — blocked until M6B_feature_matrix.csv exists
  Input:  data/synthetic/M6B_feature_matrix.csv (~196,000 × 26)
  Target: label_int (0–21), 22-class XGBoost
  Output: models/M7_xgboost_classifier.json
  ↓
M8 ⬜ NOT STARTED — LSTM-AE v2 + 4-Layer Detection (threshold unchanged: 0.110058)
  Layer 3 (CUSUM) + Layer 4 (Rolling Baseline) mandatory for label 21 detection
  ↓
M9 → M10 → M11 → M12
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-15 | Original M6.5r spec — feature set, windowing, gates, script architecture, M7 handoff |
| v2.0 | 2026-04-15 | Corrections: status ⬜ NOT STARTED. 26-column output confirmed. Domain 3 feature list corrected. Gate D4 added. Module pathway corrected. Paste keys set to PENDING. |
| v3.0 | 2026-04-16 | **v14.0 UPGRADE**: 22 classes throughout. ~196,000 rows target (from ~189,000). Label 21 `bearing_wear_gradual` windowing note added. `err_slope_MotSV` note added as primary label 21 discriminator. Domain 3 label refs updated: burst_count → label 18, cyclic_baseline_drift → label 20, variant_slope_ratio → labels 19/20, masked_channel_flag → labels 13–17, secondary_onset_lag → labels 7–12. Label 21 zero-value discriminator note added. Gate D5 added for label 21 slope. SHAP table: label 21 row added, D16→D18, D17/18→D19/20. Metadata JSON n_classes corrected to 22. M7 handoff: 22-way classification, label_int 0–21. Module pathway updated: 22-class, ~196k, 4-layer M8. Paste keys: n_classes=22, ~196k, D5 gate + label21_slope_pct added. |

---

*GitHub is the ONLY source of truth for this spec.*
*Canonical reference: [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md)*
*Fault universe + physics rules: [`modules_M6B_synthetic_expanded.md`](./modules_M6B_synthetic_expanded.md)*
*Script plan + API design: [`modules_M6B_script_plan.md`](./modules_M6B_script_plan.md)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
