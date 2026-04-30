# PumpSmart v14.2 — Cowork Update Master File
## Post M6.5r Completion | Date: 2026-04-29
## Prepared by: Claude (Souvik's PumpSmart Assistant)

---

## OVERVIEW

M6.5r (Feature Matrix Re-Extraction) completed successfully on 2026-04-29.
Status: READY for M7. Active module shifts from M6.5r → M7.

This file tells Cowork EXACTLY which files to update, where each file lives,
what text to find, and what to replace it with. No interpretation required —
every change is fully specified as find-and-replace operations.

**DO NOT** modify any file not listed here.
**DO NOT** modify any section of a listed file not described here.
**DO NOT** modify any LOCKED artifact (M1–M4 outputs, pkl files, model weights).

---

## EXPLICIT BLOCKLIST — NEVER TOUCH THESE

```
models/lstm_ae_baseline_best.pth
models/M3_normalization_config.json
models/M4_threshold_config.json
models/fault_rules_v3.json
data/synthetic/M6B_combined_sequences.pkl
data/synthetic/z_t_sequences_groupA_normal.pkl
data/synthetic/z_t_sequences_groupA_faults.pkl
data/synthetic/z_t_sequences_groupA_faults_rerun.pkl
data/synthetic/z_t_sequences_groupB.pkl
data/synthetic/z_t_sequences_groupC.pkl
data/synthetic/z_t_sequences_groupD.pkl
data/synthetic/z_t_sequences_groupE.pkl
completed_modules_context_and_M1_to_M4.md
completed_modules_M5_to_M6p5r.md
module_M8_*.md  (any file matching this pattern)
modules_M9_*.md
modules_M10_*.md
modules_M11_*.md
modules_M12_*.md
```

---

## FILE 1 — pasted-text.txt

**Full path:** `C:\Users\user\Desktop\PumpSmart_Project\pasted-text.txt`
**Action:** 4 targeted find-and-replace operations. Do NOT rewrite the file.

---

### Change 1.1 — M6.5r section status block

**FIND** (exact text):
```
STATUS             : PLANNED — runs after M6B complete
PURPOSE            : Build ~35-feature matrix for XGBoost M7 training
INPUT              : All z_t_sequences_group*.pkl + synthetic_group*.pkl
OUTPUT             : M6p5r_feature_matrix.csv (~196,000 × ~35)
```

**REPLACE WITH:**
```
STATUS             : COMPLETE — 2026-04-29
PURPOSE            : Build 33-feature matrix for XGBoost M7 training
INPUT              : All z_t_sequences_group*.pkl + M6B_combined_sequences.pkl
OUTPUT             : M6B_feature_matrix.csv (526,300 × 34) — 282.6 MB
SCRIPT             : module_06p5r_feature_retrain.py v2.0
REPORT             : outputs/reports/M6.5r_Feature_Matrix_Report.md
```

---

### Change 1.2 — M6.5r paste keys block (all PENDING values)

**FIND** (anchor at first line, replace through `Status_for_M7` line):
```
M6p5r_window_size            : 50
M6p5r_n_sequences_in         : [fill after run — target ~31,800]
M6p5r_n_windows_out          : [fill after run — target ~196,000]
M6p5r_n_classes              : 22
M6p5r_feature_matrix_rows    : [fill after run — target ~196,000]
M6p5r_feature_matrix_cols    : ~36 (~35 features + label_int — confirm at runtime)
```
Continue selecting through the line ending with:
```
Status_for_M7                : PENDING — set to READY after all BLOCK gates pass
```

**REPLACE ENTIRE BLOCK WITH:**
```
M6p5r_window_size                   : 50
M6p5r_stride                        : 25
M6p5r_n_sequences_in                : 32500
M6p5r_n_windows_out                 : 526300
M6p5r_n_classes                     : 24
M6p5r_feature_matrix_rows           : 526300
M6p5r_feature_matrix_cols           : 34 (33 features + label_int)
M6p5r_z_t_pca_variance_explained    : 0.6923 (PASS — target >=0.50)
M6p5r_top_fisher_feature            : onset_order (Fisher = 9.27e13)
M6p5r_label21_slope_pct_positive    : 68.7% (WARN — score_B 99.4% compensates)
M6p5r_score_C_group_B_pct           : 72.5% (WARN — onset_order dominates)
M6p5r_score_B_label21_pct_positive  : 99.4% (PASS — CUSUM L3 viable)
M6p5r_gate_W1_boundary              : PASS
M6p5r_gate_W2_onset_split           : PASS
M6p5r_gate_W3_compound_lag          : PASS
M6p5r_gate_D1_class_balance         : PASS
M6p5r_gate_D2_masked_flag           : PASS (100.0%)
M6p5r_gate_D3_multisensor           : WARN (47.2% — spike char label 22, non-blocking)
M6p5r_gate_D4_burst_count           : PASS (100.0%)
M6p5r_gate_D5_label21_slope         : WARN (68.7% — score_B compensates, non-blocking)
M6p5r_gate_Z1_pca_variance          : PASS (69.2%)
M6p5r_gate_Z2_score_C_group_B       : WARN (72.5% — T//50 pkl windowing limits deltas)
M6p5r_gate_Z3_score_B_label21       : PASS (99.4%)
M6p5r_gate_F1_fisher                : WARN (13/33 features Fisher<0.5 — accepted)
M6p5r_output_file                   : data/synthetic/M6B_feature_matrix.csv
M6p5r_boundary_violations           : 0
M6p5r_domain4_features              : z_t_pca_1, z_t_pca_2, z_t_norm, z_t_recon_err,
                                       score_A, score_B, score_C, onset_order
M6p5r_window_count_note             : 526,300 vs ~196,000 target — correct. Variable
                                       seq lengths: label 21 (1000s x 2000 seqs = 78k
                                       windows alone). Not an error.
Status_for_M7                       : READY
```

---

### Change 1.3 — Module pathway: M6.5r status line

**FIND** (any of these variants — use whichever appears in the file):
```
M6.5r NEXT ACTIVE — runs after M6B complete
```
OR
```
M6.5r [NEXT ACTIVE] — M6B complete, all inputs confirmed present (2026-04-28)
```

**REPLACE WITH:**
```
M6.5r [COMPLETE — LOCKED 2026-04-29]
  Script    : module_06p5r_feature_retrain.py v2.0
  Output    : data/synthetic/M6B_feature_matrix.csv (526,300 x 34, 282.6 MB)
  Gates     : 8 PASS, 4 WARN (D3/D5/Z2/F1 — all physically justified, non-blocking)
  Report    : outputs/reports/M6.5r_Feature_Matrix_Report.md
```

---

### Change 1.4 — Module pathway: M7 status line

**FIND** (any of these variants):
```
M7 [NOT STARTED] — blocked until M6B_feature_matrix.csv exists
```
OR
```
M7 NOT STARTED — blocked until M6B_feature_matrix.csv (~196,000 x ~35)
```

**REPLACE WITH:**
```
M7 [ACTIVE — UNBLOCKED 2026-04-29]  <-- CURRENT ACTIVE MODULE
  Input  : data/synthetic/M6B_feature_matrix.csv (526,300 x 34) confirmed exists
  Target : label_int 0-23, 24-class XGBoost classifier
  Output : models/M7_xgboost_classifier.json
  Script : module_07_xgboost_classifier.py
```

---

### Change 1.5 — Active module declaration (if present)

**FIND:**
```
Active module: M6.5r.
```
OR
```
Active module: M6.5r. Confirm before every response.
```

**REPLACE WITH:**
```
Active module: M7. Confirm before every response. Never skip ahead.
```

---

## FILE 2 — modules_M6_synthetic_generation.md

**Full path:** Search project root and docs/ for this filename.
**Action:** Module pathway updates only — 2 changes.

---

### Change 2.1 — M6.5r NEXT ACTIVE block

**FIND:**
```
M6.5r NEXT ACTIVE — M6B complete (2026-04-28), all input files exist
  Input: M6B_combined_sequences.pkl (452.7 MB) + all z_t group pkl files
  Target: M6B_feature_matrix.csv (~196,000 rows x ~36 cols)
```

**REPLACE WITH:**
```
M6.5r COMPLETE — LOCKED (2026-04-29)
  Input  : M6B_combined_sequences.pkl (452.7 MB) + 7 z_t group pkl files
  Output : M6B_feature_matrix.csv — 526,300 rows x 34 cols (282.6 MB)
  Gates  : 8 PASS / 4 WARN (D3/D5/Z2/F1 — all physically justified)
  Report : outputs/reports/M6.5r_Feature_Matrix_Report.md
```

---

### Change 2.2 — Feature matrix output row in any table

**FIND** (any approximate row like this):
```
M6p5r_feature_matrix.csv | ~196,000 rows x ~35 features
```

**REPLACE WITH:**
```
M6B_feature_matrix.csv | 526,300 rows x 34 cols (33 features + label_int) | 282.6 MB
```

---

## FILE 3 — modules_M6p5r_feature_retrain.md

**Full path:** Search project root and docs/ for this filename.
**Action:** 4 changes — header status, output row, paste keys table, module pathway.

---

### Change 3.1 — Document header status field

**FIND:**
```
| **Status** | NEXT ACTIVE — M6B COMPLETE (2026-04-28). All blocking files now exist. Script not yet run. |
```

**REPLACE WITH:**
```
| **Status** | COMPLETE — LOCKED 2026-04-29. All gates evaluated. Status_for_M7 = READY. |
```

---

### Change 3.2 — Output file row in header table

**FIND:**
```
| **Output file** | `data/synthetic/M6B_feature_matrix.csv` (~196,000 rows × ~36 columns) |
```

**REPLACE WITH:**
```
| **Output file** | `data/synthetic/M6B_feature_matrix.csv` (526,300 rows × 34 columns — 282.6 MB) |
```

---

### Change 3.3 — Entire paste keys table

**FIND** the table block that starts with:
```
| `M6p5r_window_size` | 50 |
| `M6p5r_n_sequences_in` | [fill after run — target ~31,800] |
```
Select through the last row ending with:
```
| `Status_for_M7` | PENDING — set to READY after all BLOCK gates pass |
```

**REPLACE ENTIRE TABLE WITH:**
```
| Key | Actual Value |
|-----|-------------|
| `M6p5r_window_size` | 50 |
| `M6p5r_stride` | 25 |
| `M6p5r_n_sequences_in` | 32,500 |
| `M6p5r_n_windows_out` | 526,300 |
| `M6p5r_n_classes` | 24 |
| `M6p5r_feature_matrix_rows` | 526,300 |
| `M6p5r_feature_matrix_cols` | 34 (33 features + label_int) |
| `M6p5r_domain4_features` | `z_t_pca_1`, `z_t_pca_2`, `z_t_norm`, `z_t_recon_err`, `score_A`, `score_B`, `score_C`, `onset_order` |
| `M6p5r_z_t_pca_variance_explained` | 0.6923 (69.2%) |
| `M6p5r_gate_W1_boundary` | **PASS** |
| `M6p5r_gate_W2_onset_split` | **PASS** |
| `M6p5r_gate_W3_compound_lag` | **PASS** |
| `M6p5r_gate_F1_fisher` | **WARN** — 13/33 features Fisher < 0.5 (accepted — multi-severity variance expected) |
| `M6p5r_gate_D1_class_balance` | **PASS** — max class 14.8% (label 21) |
| `M6p5r_gate_D2_masked_flag` | **PASS** — 100.0% |
| `M6p5r_gate_D3_multisensor` | **WARN** — 47.2% (label 22 spike char, not blocking) |
| `M6p5r_gate_D4_burst_count` | **PASS** — 100.0% |
| `M6p5r_gate_D5_label21_slope` | **WARN** — 68.7% (Paris law sub-noise; score_B=99.4% compensates) |
| `M6p5r_gate_Z1_pca_variance` | **PASS** — 69.2% |
| `M6p5r_gate_Z2_score_C_group_B` | **WARN** — 72.5% (pkl T//50 windowing; onset_order dominates) |
| `M6p5r_gate_Z3_score_B_label21` | **PASS** — 99.4% |
| `M6p5r_top_fisher_feature` | `onset_order` (Fisher = 9.27×10¹³) |
| `M6p5r_label21_slope_pct_positive` | 68.7% |
| `M6p5r_score_C_group_B_pct` | 72.5% |
| `M6p5r_score_B_label21_pct_positive` | 99.4% |
| `M6p5r_output_file` | `data/synthetic/M6B_feature_matrix.csv` |
| `M6p5r_boundary_violations` | 0 |
| `Status_for_M7` | **READY** |
```

---

### Change 3.4 — Module pathway block

**FIND:**
```
M6.5r [NEXT ACTIVE] — M6B complete, all inputs confirmed present (2026-04-28)
  This module: extracts ~36-column feature matrix from M6B sequences + z_t exports
  Output: data/synthetic/M6B_feature_matrix.csv (~196,000 × ~36)
```

**REPLACE WITH:**
```
M6.5r [COMPLETE — LOCKED 2026-04-29]
  Output  : data/synthetic/M6B_feature_matrix.csv (526,300 x 34, 282.6 MB)
  Gates   : 8 PASS / 4 WARN — Status_for_M7 = READY
  Report  : outputs/reports/M6.5r_Feature_Matrix_Report.md
```

Then **FIND** the M7 line immediately below in the same pathway block:
```
M7 [NOT STARTED] — blocked until M6B_feature_matrix.csv (~196,000 × ~36)
  Input:  data/synthetic/M6B_feature_matrix.csv (~196,000 × ~36)
  Target: label_int (0-21), 22-class XGBoost
```

**REPLACE WITH:**
```
M7 [ACTIVE — CURRENT MODULE — unblocked 2026-04-29]
  Input  : data/synthetic/M6B_feature_matrix.csv (526,300 x 34) confirmed exists
  Target : label_int 0-23, 24-class XGBoost classifier
  Output : models/M7_xgboost_classifier.json
  Script : module_07_xgboost_classifier.py
```

---

## FILE 4 — module_M7_xgboost_classifier.md

**Full path:** Search project root and docs/ for this filename.
**Action:** 7 changes — status, upstream dependency table, input spec, class
            imbalance block (INSERT new), Finding 4 addendum, Finding 6 addendum,
            revision history entry. DO NOT change gate targets, SHAP expected ranks,
            or XGBoost architecture parameters.

---

### Change 4.1 — Document header status field

**FIND:**
```
| **Status** | NOT STARTED — ACTIVE after M6B + M6.5r complete |
```

**REPLACE WITH:**
```
| **Status** | ACTIVE — M6.5r COMPLETE 2026-04-29. Input file confirmed. Script pending. |
```

---

### Change 4.2 — Upstream dependency table rows

**FIND:**
```
| M6.5r all gates passed | Gates W1–W3, F1, D1–D5, **Z1, Z2, Z3** (Z-gates NEW v4.0) |
| `M6B_feature_matrix.csv` written | ~196,000 × ~36 |
```

**REPLACE WITH:**
```
| M6.5r gates evaluated 2026-04-29 | 8 PASS, 4 WARN (D3/D5/Z2/F1 — all physically accepted, non-blocking) |
| `M6B_feature_matrix.csv` written and confirmed | **526,300 × 34** (33 features + label_int) — 282.6 MB |
```

---

### Change 4.3 — Input specification rows

**FIND:**
```
Rows    : ~196,000 windows
Columns : ~36 total (~35 features + label_int)
```

**REPLACE WITH:**
```
Rows    : 526,300 windows (variable seq lengths produce more windows than ~196k spec estimate — correct)
Columns : 34 total (33 features + label_int)
```

---

### Change 4.4 — INSERT class imbalance block

**FIND** the heading or closing line of the Input Specification section, e.g.:
```
> M7 must NOT re-extract features. M7 must NOT re-run M4 inference.
```

**INSERT the following block AFTER that line** (do not delete the found line):

```markdown

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
```

---

### Change 4.5 — Addendum to Finding 4 (label 21 sub-threshold)

**FIND** the paragraph in Finding 4 that contains:
```
`bearing_wear_gradual` (label 21) severity 0.05–0.15 → per-window MAE < M4 threshold (0.110058) is PHYSICALLY CORRECT.
```

**APPEND the following immediately after that paragraph** (do not delete or change the found text):

```markdown

> **M6.5r Gate D5 confirmation (2026-04-29):** `err_slope_MotSV > 0` confirmed in
> 68.7% of label 21 fault-active windows (gate target ≥95% → WARN accepted).
> This is expected physics: Paris law at severity 0.05–0.15 produces slope below
> noise floor at the 50-step window scale.
> `score_B` (z_t drift slope, sequence-level) = positive in **99.4%** of label 21
> windows → M8 Layer 3 CUSUM is fully viable. Do NOT raise M4 threshold.
> Do NOT filter sub-threshold label 21 windows as normal.
```

---

### Change 4.6 — Addendum to Finding 6 (score_C)

**FIND** the paragraph in Finding 6 that contains:
```
`score_C` is computed from z_t sequences in M6.5r Domain 4 as the max delta
```

**APPEND the following immediately after the Finding 6 paragraph block** (do not delete):

```markdown

> **M6.5r Gate Z2 result (2026-04-29):** `score_C > Group A P50` confirmed in
> 72.5% of Group B windows (gate target ≥80% → WARN accepted).
> Root cause: z_t pkl files use T//50 non-overlapping windowing = 4–18 delta
> points per sequence. Max-delta has lower statistical power than stride-25 would give.
> `onset_order` (Fisher = 9.27×10¹³, rank 1 overall) dominates compound
> classification — `score_C` contributes additively.
> **If Group B macro F1 < 0.72 after M7 training:** revisit score_C formula —
> try mean-delta instead of max-delta as a first diagnostic step.
```

---

### Change 4.7 — Revision history — add new entry at top of table

**FIND** the revision history table header:
```
| Version | Date | Change |
|---------|------|--------|
| v4.0 |
```

**INSERT this new row immediately after the header row** (before v4.0 row):

```
| v5.0 | 2026-04-29 | M6.5r COMPLETE update. Input spec corrected to 526,300 × 34 (was ~196,000 × ~36). Class imbalance section added: label 21 = 14.8%, label 19 = 0.8% smallest. Gate D5/Z2 WARN context appended to Findings 4 and 6. Label 19 watch flag added. Status set to ACTIVE. |
```

---

## FILE 5 — modules_M6B_script_plan.md (if exists as separate file)

**Full path:** Search project root and docs/ for this filename.
**Action:** Module pathway update only — same as File 2 Changes 2.1.

Apply **Change 2.1** identically (find M6.5r NEXT ACTIVE block, replace with COMPLETE block).

Also find any M7 NOT STARTED line and apply the same replacement as **Change 1.4**.

---

## VERIFICATION CHECKLIST

After all changes are applied, verify each item. Report PASS or FAIL for each.

```
[ ] pasted-text.txt : Status_for_M7 = READY  present
[ ] pasted-text.txt : M6p5r_n_windows_out = 526300  present
[ ] pasted-text.txt : M6p5r_gate_Z3_score_B_label21 = PASS (99.4%)  present
[ ] pasted-text.txt : Active module = M7  present
[ ] pasted-text.txt : M6.5r shows COMPLETE not NEXT ACTIVE
[ ] modules_M6p5r_feature_retrain.md : Status = COMPLETE LOCKED 2026-04-29
[ ] modules_M6p5r_feature_retrain.md : Output file row = 526,300 x 34
[ ] modules_M6p5r_feature_retrain.md : Paste keys table has no [fill] placeholders
[ ] module_M7_xgboost_classifier.md : Status = ACTIVE
[ ] module_M7_xgboost_classifier.md : Input rows = 526,300 (not ~196,000)
[ ] module_M7_xgboost_classifier.md : Class imbalance section present
[ ] module_M7_xgboost_classifier.md : Finding 4 has Gate D5 addendum
[ ] module_M7_xgboost_classifier.md : Finding 6 has Gate Z2 addendum
[ ] module_M7_xgboost_classifier.md : v5.0 entry in revision history
[ ] NO changes made to any file in the BLOCKLIST above
[ ] NO changes made to any module_M8/M9/M10/M11/M12 files
```

If any FIND string is not found in a file, log it as:
`NOT FOUND: [File number] Change [number] — [brief description]`
Do NOT guess or approximate. Skip that change and move on.

---

## GITHUB PUSH — After All File Updates Are Verified

**Repository:** PumpSmart_Project (Souvik's GitHub)
**Branch:** main (or current working branch — check before pushing)
**Working directory:** `C:\Users\user\Desktop\PumpSmart_Project`

Execute the push ONLY after the verification checklist is fully PASS.
Do NOT push if any BLOCK gate change failed or if the blocklist files were accidentally modified.

### Step G1 — Open terminal at project root

```
cd C:\Users\user\Desktop\PumpSmart_Project
```

### Step G2 — Confirm git status (check what changed)

```
git status
```

Expected modified files (and ONLY these):
```
modified: pasted-text.txt
modified: modules_M6_synthetic_generation.md   (or equivalent path)
modified: modules_M6p5r_feature_retrain.md
modified: module_M7_xgboost_classifier.md
modified: modules_M6B_script_plan.md           (only if it existed and was changed)
```

Also stage the new report file:
```
outputs/reports/M6.5r_Feature_Matrix_Report.md   ← new file, untracked
outputs/reports/Cowork_Update_Master_M6.5r_Complete.md  ← this file, untracked
```

If `git status` shows ANY file from the BLOCKLIST above as modified — **STOP.
Do NOT push. Report the accidental modification to Souvik immediately.**

### Step G3 — Stage the files

```
git add pasted-text.txt
git add modules_M6_synthetic_generation.md
git add modules_M6p5r_feature_retrain.md
git add module_M7_xgboost_classifier.md
git add outputs/reports/M6.5r_Feature_Matrix_Report.md
git add outputs/reports/Cowork_Update_Master_M6.5r_Complete.md
```

If `modules_M6B_script_plan.md` was modified:
```
git add modules_M6B_script_plan.md
```

Do NOT use `git add .` — this risks staging unintended files including large pkl/pth files.

### Step G4 — Commit

```
git commit -m "M6.5r COMPLETE — update status, paste keys, M7 unblocked

- pasted-text.txt: M6.5r paste keys filled (526300 windows, 4 WARNs accepted)
- pasted-text.txt: Active module set to M7
- modules_M6p5r_feature_retrain.md: Status COMPLETE LOCKED 2026-04-29
- module_M7_xgboost_classifier.md: Status ACTIVE, input spec corrected 526300x34
- module_M7_xgboost_classifier.md: Class imbalance section added, label 19 watch flag
- Add M6.5r_Feature_Matrix_Report.md (physics validation + gate analysis)
- Add Cowork_Update_Master_M6.5r_Complete.md (this update instruction file)

Gates: 8 PASS / 4 WARN (D3/D5/Z2/F1 — all physically justified)
Status_for_M7: READY | score_B label21: 99.4% PASS (CUSUM viable)"
```

### Step G5 — Push

```
git push origin main
```

(Replace `main` with the actual branch name if different.)

### Step G6 — Verify on GitHub

Open the repository on GitHub and confirm:
- `pasted-text.txt` shows `Status_for_M7 : READY` in the diff
- `module_M7_xgboost_classifier.md` shows Status = ACTIVE in the diff
- `outputs/reports/M6.5r_Feature_Matrix_Report.md` appears as a new file
- No .pth, .pkl, or .json model files appear in the commit diff

---

### Files NOT pushed to GitHub (large binary / data files)

These were produced by M6.5r but are too large for GitHub and stay local only:

```
data/synthetic/M6B_feature_matrix.csv          282.6 MB  — local only
data/synthetic/M6B_feature_matrix_metadata.json — local only
outputs/plots/module_06p5r_physics_viz_*.png    — local only (5 group plots)
outputs/plots/module_06p5r_physics_label*.png   — local only (24 per-label plots)
outputs/plots/module_06p5r_mae_all_classes_summary.png — local only
outputs/plots/module_06p5r_*.png (all diagnostic plots) — local only
```

If Hugging Face Spaces is being used as the large-file host, upload
`M6B_feature_matrix.csv` and `M6B_feature_matrix_metadata.json` there separately.

---

*Master file prepared: 2026-04-29*
*Source: Claude analysis of M6.5r script output + physics graph validation*
*Next active module after these updates: M7 XGBoost Fault Classifier*
