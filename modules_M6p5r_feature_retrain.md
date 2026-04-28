# PumpSmart — Module M6.5r: Feature Matrix Re-Extraction (Retrain)
## 22 Classes · ~31,800 Sequences · ~35-Feature Output (~36 Columns) → M7 + M8 Input

| Field | Value |
|-------|-------|
| **Document version** | v4.0 — v14.2 z_t features + score_A/B/C + corrected seq counts |
| **Date** | 2026-04-19 |
| **Status** | NEXT ACTIVE — M6B COMPLETE (2026-04-28). All blocking files now exist. Script not yet run. |
| **Script filename** | `module_06p5r_feature_retrain.py` |
| **Output file** | `data/synthetic/M6B_feature_matrix.csv` (~196,000 rows × ~36 columns) |

> **NOTE:** This spec is LOCKED (v14.2). No output files from this module exist yet. All files listed under "Output Specification" are PENDING.

---

## Why M6.5r Exists — Engineering Rationale

M6.5 (original) was run on M6A data only — 7 classes, 8,400 sequences, using a 50-step window. M6.5r re-runs the entire feature extraction pipeline on the full M6B combined pool with an expanded feature set that covers compound, masked, variant, multi-sensor, AND TCN-AE Level 2 inputs.

| What Changed | M6.5 (original) | M6.5r (this module) |
|-------------|----------------|---------------------|
| Input sequences | 8,400 (M6A only, 7 classes) | ~31,800 (M6B all groups, 22 classes) |
| Window size | 50 steps (v2 fix applied) | 50 steps (unchanged — correct) |
| Classes | 7 | 22 |
| Feature columns | 24 | ~35 (adds z_t features + score_A/B/C) |
| Total columns in CSV | 25 (24 features + label) | ~36 (~35 features + label) |
| Sequence lengths | Blanket 200 steps | Variable per label (150–1,000 steps) |
| Output file | `M6_feature_matrix.csv` | `M6B_feature_matrix.csv` |
| M7 input | 7-class XGBoost | 22-class XGBoost |
| M8 Level 2 input | Not applicable | z_t sequences + score_A/B/C feed TCN-AE |

> **Note on ~35 feature count:** The exact count depends on z_t dimensionality reduction choice (see Domain 4). Minimum = 28 (no z_t reduction), maximum = 36 (z_t reduced to 8 PCA components + scores). The ~35 figure is the expected working count. Exact count locked after M6B runs.

---

## Input Specification

**Primary input:**

```
File:    data/synthetic/M6B_combined_sequences.pkl      ← PENDING (M6B output)
Format:  dict {label_int: list of np.ndarray, each shape (seq_len, 8)}
         NOTE: seq_len is VARIABLE per label — NOT blanket 200 steps.
Labels:  0-21 (integer) — see fault_rules_v3.json for string mapping
Channels (order fixed — M6B LOCKED via m6b_physics_lib.py):
  Index 0: Mot.SV  | Index 1: Pmp.SV  | Index 2: Mot.TV  | Index 3: Pmp.PV
  Index 4: Temp.SV | Index 5: Pres.SV | Index 6: Pmp.TV  | Index 7: Mot.PV
NOTE: M6A v1-v4 used wrong order (Mot.PV=0). All M6B v2 sequences use
M6B LOCKED order. M6.5r must read columns in this exact index order.
Dtype:   float32, normalized (all values relative to M3 cluster baselines)
Size:    ~31,800 sequences x avg 400 steps x 8 channels ~= ~367 MB average
```

**Also required:**

| File | Status | Note |
|------|--------|------|
| `models/fault_rules_v3.json` | PENDING (M6B Step 3) | **DO NOT** attempt to load before M6B Step 3 completes. All label resolution must go through this file. Do NOT hardcode label strings anywhere in this script. |
| `z_t_sequences_groupA_normal.pkl` | PENDING (M6B Step 0) | List of arrays shape (N_windows, 64) per sequence |
| `z_t_sequences_groupA_faults.pkl` | PENDING (M6B Step 0) | — |
| `z_t_sequences_groupB.pkl` | PENDING (M6B Step 1) | — |
| `z_t_sequences_groupC.pkl` | PENDING (M6B Step 2) | — |
| `z_t_sequences_groupD.pkl` | PENDING (M6B Step 2) | — |
| `z_t_sequences_groupE.pkl` | PENDING (M6B Step 2) | — |

> N_windows = seq_len / 50 (variable per label). These are Level 1 LSTM-AE bottleneck exports from M6B. **Raw sensor data NEVER enters Level 2 TCN-AE directly.** (Invariant 16)

---

## Windowing Logic — Fixed at 50 Steps, Variable Sequence Length

```python
WINDOW_SIZE = 50      # Fixed — matches M4 WINDOW_SIZE=50 in config.py
STRIDE      = 25      # 50% overlap — same as M6A/M6.5
ONSET_STEP  = 50      # All M6B sequences: fault onset at step 50 (enforced in M6B)

# Per sequence: generate windows STRICTLY within sequence boundaries
# seq_len is VARIABLE — use actual sequence length, NOT 200
for start in range(0, seq_len - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    window = sequence[start:end, :]   # shape (50, 8)
    # -> extract ~35 features from this window
    # -> label = sequence label (inherited, not inferred)
```

**Windows per sequence (variable — examples):**

| Label | Class | Steps | Windows |
|-------|-------|-------|---------|
| 3 | cavitation | 150 | 5 |
| 0 | normal | 200 | 7 |
| 6 | sensor_failure | 150 | 5 |
| 1 | bearing_wear | 250 | 9 |
| 10 | seal→cavitation | 900 | 35 |
| 21 | bearing_wear_gradual | 1,000 | 39 |

> Target total rows in matrix: **~196,000** (longer sequences partially offset fewer short ones)

### Onset-Split Rule (Mandatory)

For compound chain sequences (labels 7–12), the secondary fault onset step is stored in `M6B_sequence_meta.csv` as `secondary_onset_lag`. Windows before `secondary_onset_step` carry the **PRIMARY** fault label only. Windows after carry the **COMPOUND** label. This ensures the model learns both the early single-fault signal AND the compound transition.

### Label 21 Windowing Note

Label 21 (`bearing_wear_gradual`) sequences have MAE <threshold in many early windows (severity 0.05–0.15). This is **PHYSICALLY CORRECT** — do NOT re-label these windows as normal. All label 21 fault-active windows (t ≥ 50) carry label 21 regardless of per-window MAE. The gradual slope (`err_slope_MotSV > 0`) accumulates across windows — this is the M7 signal. `score_B` (drift slope from TCN-AE) is the CUSUM input for M8 Layer 3 — computed from z_t sequences, NOT from per-window `err_slope_MotSV` directly.

---

## Feature Set — ~35 Features + Label = ~36 Columns Total

All features computed per window (50 timesteps × 8 channels). Features grouped by physical domain.

---

### Domain 1 — Per-Channel Reconstruction Error (8 features)

Computed using the frozen M4 LSTM-AE model (loaded in inference mode).

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| `mae_MotPV` | mean(\|x − x̂\|) over 50 steps | Motor velocity reconstruction error |
| `mae_MotSV` | mean(\|x − x̂\|) over 50 steps | Motor acceleration — primary bearing indicator |
| `mae_MotTV` | mean(\|x − x̂\|) over 50 steps | Motor thermal error |
| `mae_PmpPV` | mean(\|x − x̂\|) over 50 steps | Pump velocity error |
| `mae_PmpSV` | mean(\|x − x̂\|) over 50 steps | Pump acceleration — BPF/cavitation indicator |
| `mae_PmpTV` | mean(\|x − x̂\|) over 50 steps | Pump thermal error |
| `mae_TempSV` | mean(\|x − x̂\|) over 50 steps | Surface thermal error |
| `mae_PresSV` | mean(\|x − x̂\|) over 50 steps | Pressure reconstruction error |

**M4 model load rule:**

```python
model = LSTMAEModel(config)
model.load_state_dict(torch.load('models/lstm_ae_baseline_best.pth', map_location='cpu'))
model.eval()  # NEVER retrain the M4 model here — inference only
model.to(config.DEVICE)
```

> Threshold reference: `M4_threshold = 0.110058` (from `M4_threshold_config.json` — do NOT recompute).

---

### Domain 2 — Statistical Features (9 features)

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| `mean_err_MotSV` | mean of windowed MotSV reconstructed error | Trend direction of bearing fault signal |
| `std_err_MotSV` | std of windowed MotSV error | Variability — spike vs gradual fault |
| `kurtosis_PmpSV` | `scipy.stats.kurtosis` over raw PmpSV window | Impulsiveness — cavitation bubble collapse |
| `err_slope_MotSV` | `linregress` slope of MotSV error over 50 steps | Rate-of-rise — Paris law; **primary M7 label 21 discriminator** |
| `err_slope_TempSV` | `linregress` slope of TempSV error over 50 steps | Thermal drift rate — overloading severity |
| `err_slope_PresSV` | `linregress` slope of PresSV error over 50 steps | Pressure drop rate — seal failure vs cavitation |
| `thermal_coupling_ratio` | corr(MotTV_err, TempSV_err) over window | r=0.9793 baseline; deviations = hydraulic fault |
| `cross_channel_MotSV_PmpSV` | corr(MotSV_err, PmpSV_err) over window | Vibration coupling — imbalance to bearings |
| `max_err_all` | max(all 8 channel MAEs) | Peak anomaly — dominated by primary fault channel |

> **⚠️ NOTE on `err_slope_MotSV` vs `score_B`:**
>
> - `err_slope_MotSV` = per-window linear regression slope (50 steps) — PRIMARY M7 XGBoost discriminator for label 21. Gate D5 validates this.
> - `score_B` = drift slope computed by TCN-AE Level 2 from z_t sequences over N_windows — PRIMARY M8 Layer 3 CUSUM input (Invariant 19).
>
> They measure the same physical phenomenon (Paris law crack growth) at different timescales and model levels. **Do NOT conflate them. Do NOT route `score_B` to M7 as a replacement for `err_slope_MotSV` or vice versa.**

---

### Domain 3 — Compound / Masked / Variant / Multi-Sensor Discriminators (8 features)

| Feature | Formula | Fault Groups Served |
|---------|---------|---------------------|
| `masked_channel_flag` | 1 if any channel MAE <0.02 AND seq MAE > threshold | Group C (labels 13–17) |
| `secondary_onset_lag` | Step of secondary channel first >0.5× threshold minus primary onset step | Group B (labels 7–12) |
| `burst_count` | Count MAE spikes >threshold separated by ≥10 normal steps in full variable-length sequence | Group D label 18 |
| `cyclic_baseline_drift` | mean TempSV last 25 steps minus first 25 steps | Group D label 20 |
| `multi_sensor_anomaly_count` | Count channels with MAE >0.15 simultaneously | Group E |
| `fault_group_id` | `{0:normal, 1:single, 2:compound, 3:masked, 4:variant, 5:multi_sensor}` | All groups |
| `variant_slope_ratio` | `err_slope_PmpSV / err_slope_PresSV` | Group D labels 19/20 |
| `thermal_decoupling_flag` | 1 if `thermal_coupling_ratio < 0.5` | Groups B, C hydraulic chain |

**Domain 3 notes:**

- `fault_group_id` derived from `M6B_sequence_meta.csv` group field — NOT from label integer. Does NOT leak label info (6 groups vs 22 labels). If M7 SHAP top-1 = `fault_group_id` for ANY class → FAIL (leakage investigation).
- `burst_count` computed over full variable-length sequence (not per 50-step window); per-window value = 1 if window contains a burst, 0 otherwise.
- `secondary_onset_lag` = 0 for all non-compound sequences (Groups A, C, D, E).
- Label 21: `variant_slope_ratio` ≈ 0; `burst_count` = 0; `cyclic_baseline_drift` ≈ 0. These ZERO values ARE discriminative for label 21 vs other Group D classes.

---

### Domain 4 — z_t Latent Features + TCN-AE Scores (~10 features) ⭐ *NEW v14.2*

> **⭐ THIS DOMAIN IS NEW IN v4.0 — Added for TCN-AE Level 2 (M8) alignment**
>
> These features are extracted from the z_t sequences exported by M6B (via M4 LSTM-AE sliding window inference). They represent the Level 1 bottleneck representation of each window and the TCN-AE scores derived from z_t sequences.
>
> **Why Domain 4 exists:** M8 Level 2 TCN-AE operates on z_t sequences (N_windows × 64) — NOT raw sensor windows. M6.5r must compute z_t-derived features so that M7 XGBoost has access to the same latent space signals that M8 uses. Specifically, `score_C` (chain transition — fed to XGBoost per Invariant 19) must appear as a feature in the M6.5r matrix so M7 can learn compound chain onset patterns.

| Feature | Source | Physical Meaning | Consumer |
|---------|--------|-----------------|----------|
| `z_t_pca_1` | PCA component 1 of z_t (64→1) | Dominant latent direction | M7 XGBoost |
| `z_t_pca_2` | PCA component 2 of z_t (64→1) | Secondary latent direction | M7 XGBoost |
| `z_t_norm` | L2 norm of z_t (64-dim) | Distance from normal manifold | M7 XGBoost |
| `z_t_recon_err` | `‖z_t_actual − z_t_reconstructed‖₂` per window | Level 1 reconstruction quality | M7 + M8 Level 2 |
| `score_A` | Severity: `mean(z_t_recon_err over N_windows)` | How far z_t is from normal | M8 Layer 4 Rolling Baseline |
| `score_B` | Drift slope: OLS slope of `z_t_recon_err` over N_windows | Is anomaly GROWING? Paris law drift = positive slope | **M8 Layer 3 CUSUM only** (Invariant 19) |
| `score_C` | Chain transition: `max(‖z_t_recon[n] − z_t_recon[n-1]‖)` | Did fault CHARACTER change? Compound fault transition | **M7 XGBoost + M8** (Invariant 19) |
| `onset_order` | Binary: 0 = pre-secondary-onset window; 1 = post-secondary-onset window | Temporal position re: compound fault transition | M7 XGBoost — key feature for Group B (labels 7–12) |

> **⚠️ SCORE ROUTING INVARIANT — INVARIANT 19 — NEVER CROSS:**
>
> | Score | Routes To |
> |-------|-----------|
> | `score_B` | CUSUM only (M8 Layer 3) |
> | `score_A` | Rolling Baseline only (M8 Layer 4) |
> | `score_C` | XGBoost only (M7 + M10) |
>
> These scores are WRITTEN here in M6.5r as feature columns. Their routing is ENFORCED at M8 inference time. **Do NOT apply `score_B` to rolling baseline or `score_A` to CUSUM in any script.**

**z_t source files (per group):**

| File | Domain 4 Role |
|------|--------------|
| `z_t_sequences_groupA_normal.pkl` | Domain 4 features for Group A normal sequences |
| `z_t_sequences_groupA_faults.pkl` | Domain 4 features for Group A fault sequences |
| `z_t_sequences_groupB.pkl` | Domain 4 features for Group B (`score_C` key) |
| `z_t_sequences_groupC.pkl` | Domain 4 features for Group C |
| `z_t_sequences_groupD.pkl` | Domain 4 features for Group D (`score_B` key for label 21) |
| `z_t_sequences_groupE.pkl` | Domain 4 features for Group E |

> **PCA fit rule:** PCA fit ONLY on Group A normal z_t sequences. Apply (transform only) to all groups. **Never refit on fault sequences — leakage.**

**Feature count summary:** Domain 1 (8) + Domain 2 (9) + Domain 3 (8) + Domain 4 (~10) = **~35 features total**. CSV has ~36 columns: ~35 features + `label_int`.

---

## Fisher Score Validation Gate (Gate F1)

Before writing matrix to CSV, compute Fisher discriminant score for every feature:

```
Fisher Score = (between-class variance) / (within-class variance)

Gate F1 Pass: Fisher score > 0.5 for ALL ~35 features
Action on fail: flag feature in report — do NOT drop automatically.
               Souvik to review flagged features before M7.
```

**Expected top Fisher features (updated for v14.2):**

| Feature | Expected Top Class |
|---------|-------------------|
| `mae_MotSV` | Highest for `bearing_wear` class |
| `kurtosis_PmpSV` | Highest for `cavitation` / label 18 (`cavitation_intermittent`) |
| `err_slope_PresSV` | Highest for `seal_failure`, label 19 (`seal_failure_fast`) |
| `err_slope_MotSV` | **Highest M7 discriminator for label 21** (`bearing_wear_gradual`) |
| `score_C` | Highest for Group B compound chains (labels 7–12) at transition |
| `score_B` | Consistent positive slope for label 21 — confirms CUSUM target |
| `thermal_coupling_ratio` | Highest for compound thermal chains (labels 7, 9) |
| `secondary_onset_lag` | Highest discriminator for Group B vs Group A |
| `masked_channel_flag` | Binary discriminator for Group C (labels 13–17) |
| `burst_count` | Highest discriminator for `cavitation_intermittent` (label 18) |
| `multi_sensor_anomaly_count` | Highest discriminator for Group E |
| `onset_order` | Expected high for Group B (phase boundary marker) |

---

## Output Specification (ALL PENDING)

**Primary output:**

```
File:    data/synthetic/M6B_feature_matrix.csv         ← PENDING
Rows:    ~196,000 (variable per sequence length — confirm at runtime)
Dtype:   float32 for all numeric; int for label_int
```

**Columns (~36 total):**

```
label_int,
mae_MotPV, mae_MotSV, mae_MotTV, mae_PmpPV,
mae_PmpSV, mae_PmpTV, mae_TempSV, mae_PresSV,
mean_err_MotSV, std_err_MotSV, kurtosis_PmpSV,
err_slope_MotSV, err_slope_TempSV, err_slope_PresSV,
thermal_coupling_ratio, cross_channel_MotSV_PmpSV, max_err_all,
masked_channel_flag, secondary_onset_lag,
burst_count, cyclic_baseline_drift,
multi_sensor_anomaly_count, fault_group_id,
variant_slope_ratio, thermal_decoupling_flag,
z_t_pca_1, z_t_pca_2, z_t_norm, z_t_recon_err,
score_A, score_B, score_C, onset_order
```

> Size: ~196,000 × ~36 ≈ ~54 MB CSV — trivially fits RAM

**Also write:**

```
data/synthetic/M6B_feature_matrix_metadata.json        ← PENDING
  window_size              : 50
  stride                   : 25
  onset_step               : 50
  n_sequences              : [fill at runtime]
  n_windows                : [fill at runtime]
  n_classes                : 22
  class_distribution       : {label_str: count, ...}
  fisher_scores            : {feature_name: score, ...}
  gate_F1_status           : PASS / FAIL
  m4_threshold_used        : 0.110058
  z_t_pca_variance_explained: [fill at runtime]
  generated_by             : "module_06p5r_feature_retrain.py"

outputs/reports/module_06p5r_report.md                 ← PENDING
```

---

## Validation Gates — M6.5r Specific

| Gate | Test | Pass Criterion | Action on Fail |
|------|------|---------------|----------------|
| **W1** | Window boundary integrity | Zero windows cross sequence boundaries | **BLOCK** — fix windowing loop |
| **W2** | Onset-split correctness | Pre-onset windows labelled 0; fault-active correct | **BLOCK** |
| **W3** | Compound onset lag | `secondary_onset_lag > 0` for all Group B fault-active windows | WARN — log cases |
| **F1** | Fisher score gate | All ~35 features score >0.5 | FLAG — review, do not drop |
| **D1** | Class balance check | No single class >20% of total windows | WARN — log imbalance |
| **D2** | Masked class secondary signal | `masked_channel_flag = 1` in ≥90% Group C fault-active windows | WARN |
| **D3** | Multi-sensor anomaly count | `multi_sensor_anomaly_count = 2` in ≥90% Group E windows | WARN |
| **D4** | `burst_count` gate | `burst_count >= 2` in ≥95% label 18 sequences | WARN |
| **D5** | Label 21 slope gate | `err_slope_MotSV > 0` in ≥95% label 21 fault-active windows | WARN — critical for M7 + M8 Layer 3 |
| **Z1** | z_t PCA variance | 2 PCA components explain ≥50% variance of normal z_t pool | WARN — add `z_t_pca_3` if fails |
| **Z2** | `score_C` Group B signal | `score_C > P50` of Group A `score_C` in ≥80% Group B compound windows at phase transition | WARN — verify TCN `score_C` head calibration |
| **Z3** | `score_B` label 21 | `score_B > 0` in ≥90% of label 21 fault-active windows | WARN — critical: CUSUM cannot fire if `score_B ≈ 0` |

---

## Script Architecture

```
module_06p5r_feature_retrain.py

HEADER      Config, dirs, logging, results dict (mandatory)
SECTION 1   Load M6B sequences + fault_rules_v3.json + M6B_sequence_meta.csv
SECTION 2   Load frozen M4 LSTM-AE (inference only — map_location='cpu')
SECTION 3   Load z_t pkl files (6 files — all groups)                [NEW v14.2]
SECTION 4   Fit PCA on Group A normal z_t (2 components) — apply all [NEW v14.2]
SECTION 5   Window generator (50-step, 25-stride, boundary-safe, variable seq_len)
SECTION 6   Per-channel MAE extractor (M4 inference pass — Domain 1)
SECTION 7   Statistical feature extractor (Domain 2)
SECTION 8   Compound/masked/variant/multi-sensor features (Domain 3)
SECTION 9   z_t latent features + score_A/B/C extraction (Domain 4)  [NEW v14.2]
SECTION 10  Onset-split labelling (compound secondary onset, label 21 note)
SECTION 11  Fisher score computation + Gate F1
SECTION 12  Validation gates W1-W3, D1-D5, Z1-Z3
SECTION 13  Write M6B_feature_matrix.csv + metadata JSON
SECTION 14  Diagnostic plots:
              Fisher bar chart
              Class distribution
              Feature correlation heatmap
              Per-group MAE boxplot
              Label 21 err_slope_MotSV histogram
              score_C distribution: Group A vs Group B    [NEW v14.2]
              score_B trajectory: label 21                [NEW v14.2]
SECTION 15  PASTE TEXT UPDATE (print between banner lines)
SECTION 16  Write outputs/reports/module_06p5r_report.md
SECTION 17  FILE MANIFEST + NEXT PROMPT
```

**GPU usage:** M4 inference pass (Section 6) uses `torch.cuda.amp.autocast()` and `pin_memory=True` DataLoader for batch inference. Batch size = 512 windows. All other sections run on CPU (pandas, numpy, scipy).

```python
# M4 inference DataLoader
dataset = WindowDataset(windows_tensor)   # shape (N, 50, 8)
loader  = DataLoader(dataset, batch_size=512, pin_memory=True, num_workers=4)

with torch.no_grad():
    with torch.cuda.amp.autocast():
        for batch in loader:
            batch = batch.to(config.DEVICE)
            recon = model(batch)
            mae   = torch.mean(torch.abs(batch - recon), dim=1)  # (batch, 8)
            z_t   = model.encode(batch)                           # (batch, 64)
```

---

## M7 Handoff — What M7 Receives

| Property | Value |
|----------|-------|
| **Input file** | `data/synthetic/M6B_feature_matrix.csv` |
| **Rows** | ~196,000 windows |
| **Feature columns** | ~35 (all columns except `label_int`) |
| **Target** | `label_int` (0–21, integer, single label — XGBoost 22-class) |
| **Classes** | 22 |
| **Class names** | From `fault_rules_v3.json` — `label_str` column |

> M7 must NOT re-extract features. M7 must NOT re-run M4 inference. M7 must NOT re-run z_t PCA. M7 reads the CSV directly and trains on it. **All feature engineering is FROZEN at M6.5r output.**

**Key M7 feature expectations from Domain 4:**
- `score_C` → top SHAP feature for Group B compound classes (labels 7–12)
- `score_B` → top SHAP feature for label 21 (`bearing_wear_gradual`)
- `onset_order` → key compound timing feature for Group B

> **Why single-label (NOT multi-label) for compound chains:** Compound chain sequences (labels 7–12) are assigned a unique compound label (e.g., `bearing_wear->overloading` = label 7). M7 treats this as a 22-way classification, not a multi-output problem. The compound interpretation (Primary A → Secondary B) is handled in M10 API label→display mapping, not in the classifier architecture.

---

## M8 Handoff — What M8 Level 2 Receives ⭐ *NEW v14.2*

M8 Level 2 (TCN-AE) does **NOT** read the feature matrix CSV directly. M8 Level 2 reads the z_t pkl files exported by M6B:

```
z_t_sequences_group[A-E].pkl → shape per sequence: (N_windows × 64)
```

The feature matrix (M6.5r output) contains `score_A`, `score_B`, `score_C` as computed features **for M7**. At M8 training time, `score_A/B/C` are RECOMPUTED from z_t sequences by the TCN-AE itself — the CSV values are reference/validation only.

**At M8 inference time (M10 runtime):**

| Score | Routes To | Invariant |
|-------|-----------|-----------|
| `score_B` | CUSUM (Layer 3) | Invariant 19 |
| `score_A` | Rolling Baseline (Layer 4) | Invariant 19 |
| `score_C` | XGBoost M7 (`onset_order` feature) | Invariant 19 |

---

## SHAP Expectations Post-M7 (Reference for M7 Gate Design)

| Fault Group | SHAP Top Feature | #2 | #3 |
|-------------|-----------------|----|----|
| Group A — single source | `mae_MotSV` (bearing) / `kurtosis_PmpSV` (cavitation) | `err_slope_*` | `max_err_all` |
| Group B — compound (labels 7–12) | `score_C` / `secondary_onset_lag` | `onset_order` | `thermal_coupling_ratio` |
| Group C — masked (labels 13–17) | `masked_channel_flag` | `mae_*` (non-masked) | `max_err_all` |
| Group D label 18 — intermittent | `burst_count` | `kurtosis_PmpSV` | `variant_slope_ratio` |
| Group D labels 19/20 — fast/cyclic | `variant_slope_ratio` | `cyclic_baseline_drift` | `err_slope_PresSV` |
| Group D label 21 — gradual | `err_slope_MotSV` | `score_B` | `mean_err_MotSV` |
| Group E — multi-sensor | `multi_sensor_anomaly_count` | `mae_MotTV + mae_TempSV` | `thermal_decoupling_flag` |

> **SHAP gate:** If `fault_group_id` is SHAP top-1 for ANY class → **FAIL** — label leakage investigation.

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
z_t_sequences_group[A-E].pkl             — M6B output (read-only input to M6.5r)
```

---

## Paste Keys (ALL PENDING — populate after M6.5r script runs)

| Key | Target / Value |
|-----|---------------|
| `M6p5r_window_size` | 50 |
| `M6p5r_n_sequences_in` | [fill after run — target ~31,800] |
| `M6p5r_n_windows_out` | [fill after run — target ~196,000] |
| `M6p5r_n_classes` | 22 |
| `M6p5r_feature_matrix_rows` | [fill after run — target ~196,000] |
| `M6p5r_feature_matrix_cols` | ~36 (~35 features + label_int — confirm at runtime) |
| `M6p5r_domain4_features` | [fill — `z_t_pca_1`, `z_t_pca_2`, `z_t_norm`, `z_t_recon_err`, `score_A`, `score_B`, `score_C`, `onset_order`] |
| `M6p5r_z_t_pca_variance_explained` | [fill — target ≥50% for 2 components] |
| `M6p5r_gate_W1_boundary` | PASS/FAIL |
| `M6p5r_gate_W2_onset_split` | PASS/FAIL |
| `M6p5r_gate_W3_compound_lag` | PASS/WARN |
| `M6p5r_gate_F1_fisher` | PASS/FAIL — list any flagged features |
| `M6p5r_gate_D1_class_balance` | PASS/WARN — list any class >20% |
| `M6p5r_gate_D2_masked_flag` | PASS/WARN |
| `M6p5r_gate_D3_multisensor` | PASS/WARN |
| `M6p5r_gate_D4_burst_count` | PASS/WARN |
| `M6p5r_gate_D5_label21_slope` | PASS/WARN — `err_slope_MotSV > 0` in ≥95% label 21 |
| `M6p5r_gate_Z1_pca_variance` | PASS/WARN |
| `M6p5r_gate_Z2_score_C_group_B` | PASS/WARN |
| `M6p5r_gate_Z3_score_B_label21` | PASS/WARN — `score_B > 0` in ≥90% label 21 windows |
| `M6p5r_top_fisher_feature` | [fill after run — expected `mae_MotSV` or `kurtosis_PmpSV`] |
| `M6p5r_label21_slope_pct_positive` | [% label 21 fault-active windows with `err_slope_MotSV > 0`] |
| `M6p5r_score_C_group_B_pct` | [% Group B windows with `score_C > Group A P50`] |
| `M6p5r_score_B_label21_pct_positive` | [% label 21 windows with `score_B > 0`] |
| `M6p5r_output_file` | `data/synthetic/M6B_feature_matrix.csv` |
| `Status_for_M7` | PENDING — set to READY after all BLOCK gates pass |

---

## Module Pathway — Corrected Status (v14.2)

```
M6A [COMPLETE] (8,400 seq, 7 classes) — LOCKED
  │
  ▼
M6B [COMPLETE — LOCKED 2026-04-28]
  All outputs present:
    M6B_combined_sequences.pkl (452.7 MB) — 32,500 seqs, 24 classes
    M6B_sequence_meta.csv (4.2 MB) — 32,500 rows
    fault_rules_v3.json (4.3 KB) — 24-class canonical map LOCKED
    z_t_sequences_groupA_faults_rerun.pkl, z_t_sequences_groupA_normal.pkl,
    z_t_sequences_groupA_faults.pkl, z_t_sequences_groupB.pkl,
    z_t_sequences_groupC.pkl, z_t_sequences_groupD.pkl,
    z_t_sequences_groupE.pkl — all present
  │
  ▼
M6.5r [NEXT ACTIVE] — M6B complete, all inputs confirmed present (2026-04-28)
  This module: extracts ~36-column feature matrix from M6B sequences + z_t exports
  Output: data/synthetic/M6B_feature_matrix.csv (~196,000 × ~36)
  │
  ▼
M7 [NOT STARTED] — blocked until M6B_feature_matrix.csv exists
  Input:  data/synthetic/M6B_feature_matrix.csv (~196,000 × ~36)
  Target: label_int (0-21), 22-class XGBoost
  Output: models/M7_xgboost_classifier.json
  │
  ▼
M8 [NOT STARTED] — Level 1 LSTM-AE + Level 2 TCN-AE (replaces LSTM v2 — v14.2)
  Level 1 threshold: 0.110058 (LOCKED — static, Level 1 only)
  Level 2 TCN-AE: 5-layer dilated causal, dilation=[1,2,4,8,16], RF=63 windows
  Layer 3 (CUSUM on score_B) + Layer 4 (Rolling Baseline on score_A): label 21 only
  │
  ▼
M9 → M10 → M11 → M12
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-15 | Original M6.5r spec — feature set, windowing, gates, script architecture, M7 handoff. |
| v2.0 | 2026-04-15 | Corrections: status NOT STARTED. 26-column output confirmed. Domain 3 feature list corrected. Gate D4 added. Module pathway corrected. Paste keys PENDING. |
| v3.0 | 2026-04-16 | v14.0 UPGRADE: 22 classes throughout. ~196,000 rows target. Label 21 windowing note added. `err_slope_MotSV` as primary label 21 discriminator. Domain 3 label refs updated. Gate D5 added. SHAP table label 21 row added. Module pathway updated. |
| v4.0 | 2026-04-19 | v14.2 UPGRADE: Input seq counts updated to ~31,800. Variable sequence lengths documented (150–1,000 steps). Windowing logic updated: windows per sequence variable with examples per label. Domain 4 added: z_t latent features (`z_t_pca_1/2`, `z_t_norm`, `z_t_recon_err`) + `score_A/B/C` + `onset_order` (~10 features). Score routing Invariant 19 documented in Domain 4. Feature total updated 25→~35. CSV columns updated 26→~36. z_t pkl input files listed (6 files). PCA fit rule added (fit on normal only). SECTIONS 3+4+9 added to script architecture. Gates Z1/Z2/Z3 added. Paste keys expanded. M8 Handoff section added. SHAP table: Group B top = `score_C`; label 21 #2 = `score_B`; `onset_order` added. Fisher top features: `score_C` + `score_B` added. Module pathway: TCN-AE Level 2 replaces LSTM v2; z_t exports added to M6B output list. |

---

> **GitHub is the ONLY source of truth for this spec.**
>
> Canonical reference: `completed_modules_M5_to_M6p5r.md`
> Fault universe + physics rules: `modules_M6B_synthetic_expanded.md`
> Script plan + API design: `modules_M6B_script_plan.md`
>
> **Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset
> **Canonical source of truth:** `pasted-text.txt` (v14.2, 2026-04-19)
