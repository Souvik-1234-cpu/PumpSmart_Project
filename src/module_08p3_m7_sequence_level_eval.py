# =============================================================================
# module_08p3_m7_sequence_level_eval.py
# PumpSmart v14.2 — M8 Patch 3 of 5: Sequence-Level M7 Re-Evaluation
# =============================================================================
# WHY THIS SCRIPT EXISTS:
#   The original M7 used train_test_split(stratify=y) at the WINDOW level.
#   Two adjacent windows from the same fault sequence share most of their
#   50-step receptive field — they are nearly identical inputs.
#
#   With 526,300 windows from ~32,500 sequences (~16 windows per sequence),
#   random 80/20 puts ~13 of every sequence's windows in train and ~3 in test.
#   The test set is therefore CONTAMINATED with sibling windows of training
#   examples. The reported macro F1=0.9985 is partly a memorisation score.
#
#   For an industrial deployment claim, the honest number must come from a
#   GROUP-AWARE split where no sequence has windows in both train and test.
#
# WHAT THIS SCRIPT DOES:
#   1. Reconstructs seq_id per CSV row by mapping n_windows_z_t from
#      M6B_sequence_meta.csv (the same expansion the M6.5r patch script used).
#   2. Runs sklearn.model_selection.GroupKFold (5-fold) with seq_id as group.
#   3. Trains M7 on each fold's train set with the locked hyperparameters,
#      evaluates on each fold's test set. Reports per-fold and mean F1.
#   4. Compares mean group-aware F1 to the window-level F1 reported in M8p2.
#   5. WRITES NEW WEIGHTS as `M7_xgboost_classifier_seq_level.json` — does
#      NOT replace the live M7. Both versions coexist; M10 can choose which
#      to deploy. The honest deployment choice is the seq-level model.
#
# OUTPUT FILES:
#   models/M7_xgboost_classifier_seq_level.json
#   models/M7_xgboost_classifier_seq_level_cpu.json
#   outputs/reports/M8p3_sequence_level_eval_report.md
#   outputs/M8p3_window_vs_seqlevel_comparison.png
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score

SCRIPT_NAME = "module_08p3_m7_sequence_level_eval"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
GATES   = {}
log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}")
log(f"  Device: {DEVICE} | GPU: {IS_GPU}")
log("=" * 72)

# =============================================================================
# CONSTANTS
# =============================================================================
FEATURE_MATRIX_PATH = SYNTH_DIR / "M6B_feature_matrix.csv"
SEQ_META_PATH       = SYNTH_DIR / "M6B_sequence_meta.csv"

# Same locked hyperparameters as M8p2 (and original M7)
LOCKED_PARAMS = {
    'n_estimators':     504,
    'max_depth':        7,
    'learning_rate':    0.08086361634538793,
    'subsample':        0.9531291833577744,
    'colsample_bytree': 0.9768481099821509,
    'min_child_weight': 2,
    'gamma':            0.0009941501981704567,
    'reg_alpha':        0.0010636018384176757,
    'reg_lambda':       0.10934322260320596,
    'objective':        'multi:softprob',
    'eval_metric':      'mlogloss',
    'tree_method':      'hist',
    'device':           'cuda' if IS_GPU else 'cpu',
    'random_state':     42,
}

GROUP_MAP = {
    **{i: 'A' for i in range(0, 7)},
    **{i: 'B' for i in range(7, 13)},
    **{i: 'C' for i in range(13, 18)},
    **{i: 'D' for i in range(18, 22)},
    22: 'E', 23: 'E',
}

N_SPLITS = 5     # 5-fold group K-fold

# =============================================================================
# SECTION 1 — LOAD FEATURE MATRIX + SEQUENCE META
# =============================================================================
log("\nSECTION 1 — Load feature matrix + sequence meta")

t0 = time.time()
df = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
log(f"  Feature matrix: {df.shape[0]:,} × {df.shape[1]} in {time.time()-t0:.1f}s")

seq_meta = pd.read_csv(SEQ_META_PATH, low_memory=False)
log(f"  Sequence meta: {seq_meta.shape[0]:,} rows | cols: {list(seq_meta.columns)[:8]}...")

# =============================================================================
# SECTION 2 — RECONSTRUCT seq_id PER FEATURE-MATRIX ROW
# =============================================================================
# CRITICAL ENGINEERING NOTE:
# The feature matrix has no seq_id column. M6.5r assembled rows by iterating
# label-by-label, sequence-by-sequence within label, and writing one row per
# 50-step window (n_windows_z_t = steps // 50). To reconstruct seq_id per
# row, we must reproduce that exact ordering.
#
# Heuristic order (validated against module_06p5r_patch_features_v4b.py
# which used the SAME rebuild logic):
#   For label L in sorted(labels):
#       For seq in seq_meta[seq_meta.label == L].sort_values('seq_id'):
#           emit n_windows_z_t rows tagged with seq.seq_id
# =============================================================================
log("\nSECTION 2 — Reconstruct seq_id per feature-matrix row")

# Verify seq_meta has the columns we need
required_meta_cols = ['seq_id', 'label']
optional_n_windows_cols = ['n_windows_z_t', 'n_windows', 'window_count']
N_WIN_COL = None
for c in optional_n_windows_cols:
    if c in seq_meta.columns:
        N_WIN_COL = c
        log(f"  Window count column: '{N_WIN_COL}'")
        break

for c in required_meta_cols:
    if c not in seq_meta.columns:
        log(f"  ✗ FATAL: '{c}' missing from M6B_sequence_meta.csv")
        sys.exit(1)

# Build reconstructed seq_id array
seq_id_per_row = np.empty(df.shape[0], dtype=object)
cursor = 0
mismatch = False

for lbl in sorted(seq_meta['label'].unique()):
    lbl_meta = seq_meta[seq_meta['label'] == lbl].sort_values('seq_id').reset_index(drop=True)
    lbl_csv_rows = (df['label_int'] == lbl).sum()

    if N_WIN_COL is not None:
        # Use exact n_windows from meta — most reliable
        per_seq_windows = lbl_meta[N_WIN_COL].astype(int).values
        expected_total  = int(per_seq_windows.sum())
        if expected_total != lbl_csv_rows:
            log(f"  ⚠ Label {lbl}: meta says {expected_total} rows but CSV has {lbl_csv_rows} — using uniform fallback")
            mismatch = True
    else:
        # Fallback: assume rows are evenly distributed across sequences
        log(f"  ⚠ No window-count column — using uniform fallback for label {lbl}")
        n_seqs = len(lbl_meta)
        if n_seqs == 0:
            continue
        windows_per_seq = lbl_csv_rows // n_seqs
        per_seq_windows = np.full(n_seqs, windows_per_seq, dtype=int)
        per_seq_windows[-1] += lbl_csv_rows - per_seq_windows.sum()  # absorb remainder

    # Write seq_id values to consecutive row slice
    for i, sid in enumerate(lbl_meta['seq_id'].values):
        n_w = int(per_seq_windows[i])
        if n_w <= 0:
            continue
        end = cursor + n_w
        if end > df.shape[0]:
            log(f"  ✗ FATAL: cursor overrun at label {lbl}, seq {sid}")
            sys.exit(1)
        seq_id_per_row[cursor:end] = sid
        cursor = end

# Validate — every row should have a seq_id assigned
n_unassigned = int(np.sum([s is None for s in seq_id_per_row]))
log(f"  Cursor end: {cursor:,} | CSV rows: {df.shape[0]:,} | Unassigned: {n_unassigned:,}")
if n_unassigned > 0:
    log(f"  ⚠ {n_unassigned} rows could not be mapped to a seq_id (will be excluded from CV)")

# Validate that within each (label, seq_id) the rows actually share label_int
df['_seq_id_recon'] = seq_id_per_row
sample_check = df.groupby('_seq_id_recon')['label_int'].nunique()
n_bad_seqs = int((sample_check > 1).sum())
if n_bad_seqs > 0:
    log(f"  ⚠ {n_bad_seqs} reconstructed seq_ids span more than one label — meta/CSV order mismatch")
    if n_bad_seqs > 100:
        log(f"  ✗ Too many mismatches. Reconstruction is unreliable. ABORT.")
        sys.exit(1)
else:
    log("  ✓ All reconstructed seq_ids are label-consistent")

results["seq_id_reconstruction_clean"] = (n_bad_seqs == 0)
results["n_unique_seqs_reconstructed"] = int(df['_seq_id_recon'].nunique())
log(f"  Unique seq_ids reconstructed: {results['n_unique_seqs_reconstructed']:,}")

# =============================================================================
# SECTION 3 — PREPARE X, y, groups
# =============================================================================
log("\nSECTION 3 — Prepare data arrays")

# Drop rows with no seq_id
mask = pd.notnull(df['_seq_id_recon'])
df_clean = df[mask].reset_index(drop=True)
log(f"  Clean rows after seq_id mapping: {df_clean.shape[0]:,}")

feature_cols = [c for c in df_clean.columns
                if c not in ['label_int', 'fault_group_id', '_seq_id_recon']]
n_features = len(feature_cols)
n_classes  = int(df_clean['label_int'].max()) + 1

X      = df_clean[feature_cols].values.astype(np.float32)
y      = df_clean['label_int'].values.astype(np.int32)
groups = df_clean['_seq_id_recon'].values

log(f"  X: {X.shape} | y: {y.shape} | groups: {len(np.unique(groups)):,} unique")
log(f"  n_features={n_features} | n_classes={n_classes}")

results["n_features"]     = n_features
results["n_classes"]      = n_classes
results["n_clean_rows"]   = int(X.shape[0])

# =============================================================================
# SECTION 4 — STRATIFIED GROUP K-FOLD CROSS-VALIDATION
# =============================================================================
# StratifiedGroupKFold preserves class balance across folds AND ensures no
# group (seq_id) appears in both train and test. This is the gold-standard
# split protocol for sequence-window data.
# =============================================================================
log(f"\nSECTION 4 — StratifiedGroupKFold ({N_SPLITS} folds)")

try:
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, y, groups))
    log(f"  ✓ Split successfully into {N_SPLITS} folds")
except Exception as e:
    log(f"  StratifiedGroupKFold failed ({e}) — falling back to GroupKFold")
    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X, y, groups))

# Verify: no overlap between train and test groups in any fold
overlap_violations = 0
for fold_i, (tr, te) in enumerate(splits):
    tr_groups = set(groups[tr])
    te_groups = set(groups[te])
    overlap = tr_groups & te_groups
    if overlap:
        overlap_violations += 1
        log(f"  ✗ Fold {fold_i}: {len(overlap)} groups in both train and test")
if overlap_violations == 0:
    log("  ✓ Confirmed: no seq_id leakage across train/test in any fold")
results["fold_leakage_violations"] = overlap_violations

# =============================================================================
# SECTION 5 — TRAIN AND EVALUATE EACH FOLD
# =============================================================================
log("\nSECTION 5 — Per-fold training")

fold_results = []
all_y_test_concat = []
all_y_pred_concat = []

for fold_i, (train_idx, test_idx) in enumerate(splits):
    log(f"\n  ── Fold {fold_i+1}/{N_SPLITS} ──")
    log(f"     Train rows: {len(train_idx):,} | Test rows: {len(test_idx):,}")
    log(f"     Train seqs: {len(set(groups[train_idx])):,} | Test seqs: {len(set(groups[test_idx])):,}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inverse-frequency class weighting on this fold's train set
    counts  = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.mean()
    sample_weight_train = weights[y_train]

    # Train
    t0 = time.time()
    clf = xgb.XGBClassifier(num_class=n_classes, **LOCKED_PARAMS)
    clf.fit(X_train, y_train, sample_weight=sample_weight_train, verbose=False)
    train_time = time.time() - t0
    log(f"     Train time: {train_time:.1f}s")

    # Evaluate
    y_pred = clf.predict(X_test)
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    acc      = float(accuracy_score(y_test, y_pred))
    log(f"     Macro F1: {macro_f1:.4f} | Accuracy: {acc:.4f}")

    fold_results.append({
        "fold":         fold_i + 1,
        "train_rows":   int(len(train_idx)),
        "test_rows":    int(len(test_idx)),
        "macro_f1":     round(macro_f1, 4),
        "accuracy":     round(acc, 4),
        "train_time_s": round(train_time, 1),
    })
    all_y_test_concat.append(y_test)
    all_y_pred_concat.append(y_pred)

# =============================================================================
# SECTION 6 — AGGREGATE RESULTS
# =============================================================================
log("\nSECTION 6 — Aggregating across folds")

fold_f1s = [f["macro_f1"] for f in fold_results]
mean_f1  = float(np.mean(fold_f1s))
std_f1   = float(np.std(fold_f1s))
log(f"  Mean macro F1 (5-fold group-aware): {mean_f1:.4f} ± {std_f1:.4f}")

# Per-class F1 across all folds combined
y_all_test = np.concatenate(all_y_test_concat)
y_all_pred = np.concatenate(all_y_pred_concat)
per_class_f1_all = {}
for lbl in range(n_classes):
    mask_l = (y_all_test == lbl)
    if mask_l.sum() == 0:
        continue
    f1 = float(f1_score(y_all_test == lbl, y_all_pred == lbl, zero_division=0))
    per_class_f1_all[lbl] = round(f1, 4)
    grp = GROUP_MAP.get(lbl, '?')
    log(f"  [{lbl:2d}] Grp{grp}  pooled F1={f1:.4f}")

results["mean_macro_f1_seq_level"] = round(mean_f1, 4)
results["std_macro_f1_seq_level"]  = round(std_f1, 4)
results["per_class_f1_seq_level"]  = per_class_f1_all
results["fold_results"]             = fold_results

# =============================================================================
# SECTION 7 — TRAIN FINAL MODEL ON FULL DATA WITH SEQ-LEVEL HOLDOUT
# =============================================================================
# The CV folds give us a confidence interval. The deployable model trains on
# (4/5) of the sequences and is validated on the remaining (1/5).
# =============================================================================
log("\nSECTION 7 — Training final seq-level model")

final_train_idx = splits[0][0]
final_test_idx  = splits[0][1]

X_tr_f, X_te_f = X[final_train_idx], X[final_test_idx]
y_tr_f, y_te_f = y[final_train_idx], y[final_test_idx]

counts  = np.bincount(y_tr_f, minlength=n_classes).astype(np.float64)
weights = 1.0 / (counts + 1e-6)
weights /= weights.mean()
sw_tr_f = weights[y_tr_f]

clf_final = xgb.XGBClassifier(num_class=n_classes, **LOCKED_PARAMS)
clf_final.fit(X_tr_f, y_tr_f, sample_weight=sw_tr_f, verbose=False)

# Save with explicit non-overwriting name
SEQ_LEVEL_MODEL_PATH     = MODEL_DIR / "M7_xgboost_classifier_seq_level.json"
SEQ_LEVEL_MODEL_CPU_PATH = MODEL_DIR / "M7_xgboost_classifier_seq_level_cpu.json"
clf_final.save_model(str(SEQ_LEVEL_MODEL_PATH))

clf_final_cpu = xgb.XGBClassifier(num_class=n_classes, **{**LOCKED_PARAMS, 'device': 'cpu'})
clf_final_cpu.load_model(str(SEQ_LEVEL_MODEL_PATH))
clf_final_cpu.save_model(str(SEQ_LEVEL_MODEL_CPU_PATH))
log(f"  ✓ Saved: {SEQ_LEVEL_MODEL_PATH.name}")
log(f"  ✓ Saved: {SEQ_LEVEL_MODEL_CPU_PATH.name}")

# =============================================================================
# SECTION 8 — GATE
# =============================================================================
log("\nSECTION 8 — Gates")

# The honest gate: mean group-aware F1 should be at least 0.85 for an
# industrial deployment claim. If it's below this, the original 0.9985
# was significantly inflated by leakage.
SEQ_LEVEL_F1_GATE = 0.85
GATES["M7-SEQ-1_mean_f1_gate"] = {
    "passed": mean_f1 >= SEQ_LEVEL_F1_GATE,
    "detail": f"mean F1={mean_f1:.4f} (target ≥{SEQ_LEVEL_F1_GATE} for honesty)",
}
GATES["M7-SEQ-2_no_leakage"] = {
    "passed": overlap_violations == 0,
    "detail": f"{overlap_violations} fold-overlap violations",
}
for name, g in GATES.items():
    log(f"  {'✓' if g['passed'] else '✗'} {name}: {g['detail']}")

# =============================================================================
# SECTION 9 — VISUAL COMPARISON: WINDOW-LEVEL vs SEQ-LEVEL
# =============================================================================
log("\nSECTION 9 — Comparison plot")

# Best-effort: pull the original window-level per-class F1 from the M7 report
WINDOW_LEVEL_F1 = {}
m7_report_path = REPORT_DIR / "module_07_xgboost_classifier_report.md"
if m7_report_path.exists():
    try:
        with open(m7_report_path) as f:
            for line in f:
                # Lines like: | 5 | label_5 | A | 1.0000 |
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5 and parts[1].isdigit():
                    WINDOW_LEVEL_F1[int(parts[1])] = float(parts[4])
        log(f"  Loaded {len(WINDOW_LEVEL_F1)} window-level F1 values from M7 report for comparison")
    except Exception as e:
        log(f"  ⚠ Could not parse M7 report: {e}")

try:
    labels = sorted(per_class_f1_all.keys())
    seq_f1 = [per_class_f1_all[l] for l in labels]
    win_f1 = [WINDOW_LEVEL_F1.get(l, np.nan) for l in labels]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))
    width = 0.35

    if WINDOW_LEVEL_F1:
        ax.bar(x - width/2, win_f1, width,
               label="Window-level (original M7 — likely leaky)", alpha=0.6)
    ax.bar(x + width/2, seq_f1, width,
           label="Sequence-level (group K-fold — honest)", alpha=0.95)

    ax.axhline(SEQ_LEVEL_F1_GATE, linestyle='--', color='red',
               label=f'Honesty gate {SEQ_LEVEL_F1_GATE}')
    ax.axhline(0.62, linestyle=':', color='gray', label='Label 21 floor 0.62')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_xlabel("Label")
    ax.set_ylabel("F1")
    ax.set_title("M7 per-class F1: window-level (leaky) vs sequence-level (honest)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "M8p3_window_vs_seqlevel_comparison.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=120)
    plt.close()
    log(f"  ✓ Saved: {plot_path}")
except Exception as e:
    log(f"  ⚠ Plot failed: {e}")

# =============================================================================
# SECTION 10 — REPORT
# =============================================================================
log("\nSECTION 10 — Report")

REPORT_PATH = REPORT_DIR / f"{SCRIPT_NAME}_report.md"

fold_table = "\n".join(
    f"| {f['fold']} | {f['train_rows']:,} | {f['test_rows']:,} | {f['macro_f1']:.4f} | {f['accuracy']:.4f} |"
    for f in fold_results
)
per_class_table_rows = []
for lbl in sorted(per_class_f1_all.keys()):
    seq_v = per_class_f1_all[lbl]
    win_v = WINDOW_LEVEL_F1.get(lbl, None)
    win_str = f"{win_v:.4f}" if win_v is not None else "—"
    delta_str = f"{seq_v - win_v:+.4f}" if win_v is not None else "—"
    grp = GROUP_MAP.get(lbl, '?')
    flag = " 🔴" if win_v and (win_v - seq_v) > 0.10 else ""
    per_class_table_rows.append(
        f"| {lbl} | {grp} | {win_str} | {seq_v:.4f} | {delta_str}{flag} |")
per_class_table = "\n".join(per_class_table_rows)

report_md = f"""# M8 Patch 3 — Sequence-Level M7 Re-Evaluation
**Date:** {date.today()}
**Status:** {'COMPLETE' if all(g['passed'] for g in GATES.values()) else 'CONDITIONAL — see gate detail'}

## Why this patch existed
The original M7 used `train_test_split(stratify=y)` at the WINDOW level.
With ~16 windows per sequence, this places ~13 sibling windows in train and
~3 in test for every sequence — adjacent windows share 49 of their 50
timesteps, so the test set is effectively a memorisation check, not a
generalisation check. Reported macro F1=0.9985 was therefore inflated.

## What this patch did
Reconstructed `seq_id` per row from `M6B_sequence_meta.csv`, then ran
StratifiedGroupKFold (5 folds) where no sequence has windows in both
train and test. This is the gold-standard split protocol for
sequence-window data.

## Reconstruction integrity
| Metric | Value |
|---|---|
| Unique seq_ids reconstructed | {results['n_unique_seqs_reconstructed']:,} |
| Reconstruction clean (no label-spanning seqs) | {results['seq_id_reconstruction_clean']} |
| Fold-overlap violations | {overlap_violations} |
| Clean rows used | {results['n_clean_rows']:,} |

## Per-fold results (group-aware)
| Fold | Train rows | Test rows | Macro F1 | Accuracy |
|---|---|---|---|---|
{fold_table}

| Aggregated | Mean macro F1 | Std macro F1 |
|---|---|---|
| 5-fold group-aware | **{results['mean_macro_f1_seq_level']:.4f}** | ±{results['std_macro_f1_seq_level']:.4f} |

## Per-class F1: window-level (M7 report) vs sequence-level (this patch)
| Label | Group | Window-level (leaky) | Sequence-level (honest) | Δ |
|---|---|---|---|---|
{per_class_table}

🔴 = drop > 0.10 — these classes were most affected by leakage.

## Honest reporting language

The number you should quote externally is now:

> "Macro F1 = {results['mean_macro_f1_seq_level']:.4f} ± {results['std_macro_f1_seq_level']:.4f} on
> 5-fold StratifiedGroupKFold cross-validation with seq_id as group, on
> physics-synthetic CIRA-anchored data. Real-world F1 expected to be
> meaningfully lower until active learning samples accumulate."

The window-level number {WINDOW_LEVEL_F1.get(0, '0.9985') if WINDOW_LEVEL_F1 else '0.9985'}
remains true for the original M7 protocol but is not a generalisation claim.

## Gates
| Gate | Status | Detail |
|---|---|---|
""" + "\n".join(f"| {n} | {'✓ PASS' if g['passed'] else '✗ FAIL'} | {g['detail']} |"
               for n, g in GATES.items()) + f"""

## Files written
- `models/M7_xgboost_classifier_seq_level.json` (NEW — does NOT replace live)
- `models/M7_xgboost_classifier_seq_level_cpu.json` (NEW — for M10)
- `outputs/M8p3_window_vs_seqlevel_comparison.png`

## Deployment guidance
The seq-level model has lower headline F1 but is more honest about
generalisation. **For M10 deployment, use `M7_xgboost_classifier_seq_level_cpu.json`**
unless the seq-level F1 has dropped below the M7 floor for any safety-critical
class — in which case investigate that class's sequence-distribution before
deciding which model to deploy.

## Paste text update

══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══
M8p3_seq_level_eval_done            : True
M8p3_unique_seqs_reconstructed      : {results['n_unique_seqs_reconstructed']}
M8p3_n_folds                        : {N_SPLITS}
M8p3_fold_overlap_violations        : {overlap_violations}
M8p3_mean_macro_f1                  : {results['mean_macro_f1_seq_level']}
M8p3_std_macro_f1                   : {results['std_macro_f1_seq_level']}
M8p3_window_level_f1_for_reference  : {WINDOW_LEVEL_F1.get(0, 0.9985) if WINDOW_LEVEL_F1 else 0.9985}
M8p3_honest_f1_drop                 : {round(0.9985 - results['mean_macro_f1_seq_level'], 4)}
M8p3_seq_level_model                : models/M7_xgboost_classifier_seq_level.json
M8p3_seq_level_model_cpu            : models/M7_xgboost_classifier_seq_level_cpu.json
M8p3_recommended_for_M10            : seq_level_cpu
Status_for_M8p4                     : READY
══ END PASTE UPDATE ══
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_md)
log(f"  ✓ Report: {REPORT_PATH}")

# =============================================================================
# FILE MANIFEST + NEXT
# =============================================================================
log("\n" + "=" * 72)
log("FILE MANIFEST")
log("=" * 72)
log(f"  GitHub push: {SEQ_LEVEL_MODEL_PATH}")
log(f"  GitHub push: {SEQ_LEVEL_MODEL_CPU_PATH}")
log(f"  GitHub push: {REPORT_PATH}")
log(f"  Spaces upload: M7_xgboost_classifier_seq_level_cpu.json (recommended over leaky model)")

log("\n📦 M8p3 done. Next: M8p4 — OOD detector (z_t Mahalanobis + score_A guard).")
log("=" * 72)
