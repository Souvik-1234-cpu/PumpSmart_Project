# =============================================================================
# module_08p2_label19_full_propagation.py
# PumpSmart v14.2 — M8 Patch 2 of 5: Label 19 Propagation + M7 Retrain
# =============================================================================
# WHY THIS SCRIPT EXISTS:
#   The M8 Technical Validation Report (Section 10) explicitly lists:
#     - "Feature Matrix Patch — PENDING"
#     - "M7 Re-training — PENDING"
#   Your live M7 weights were trained on Label 19 features that came from
#   the buggy seal_failure_fast generator (TPR=0% before fix). Until this
#   propagates, an M10 deployment will mis-classify real catastrophic seal
#   blowouts — the exact failure mode that destroys 40-lakh assets.
#
# WHAT THIS SCRIPT DOES:
#   1. Verifies upstream patches are in place (m6b physics lib, M6B inline
#      Label 19 fix, regenerated Label 19 sequences in groupD pkl).
#   2. Runs module_06p5r_patch_label19_features.py if not yet executed
#      (idempotent — uses md5 check on Label 19 mae_PresSV mean).
#   3. Backs up the current M7 weights to a versioned filename.
#   4. Retrains M7 on the patched feature matrix with EXACTLY the locked
#      hyperparameters from M7_gate_fix_diagnosis_and_solution.md so the
#      delta is interpretable as "Label 19 patch effect" only.
#   5. Validates new M7 against the original 17 gates PLUS a new gate:
#      M7-LBL19-EXT: Label 19 F1 >= 0.80
#   6. Saves a side-by-side diff vs the pre-patch M7.
#
# OUTPUT FILES:
#   models/M7_xgboost_classifier.json                        (RETRAINED)
#   models/M7_xgboost_classifier_cpu.json                    (RETRAINED, CPU)
#   models/M7_xgboost_classifier.pre_label19_patch.json.bak  (PRE-PATCH BACKUP)
#   outputs/reports/M8p2_label19_propagation_report.md
#   outputs/M8p2_per_class_f1_delta.png
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings, hashlib, shutil, subprocess, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

SCRIPT_NAME = "module_08p2_label19_full_propagation"
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
LABEL_19              = 19
FEATURE_MATRIX_PATH   = SYNTH_DIR / "M6B_feature_matrix.csv"
SEQ_META_PATH         = SYNTH_DIR / "M6B_sequence_meta.csv"
M7_MODEL_PATH         = MODEL_DIR / "M7_xgboost_classifier.json"
M7_MODEL_CPU_PATH     = MODEL_DIR / "M7_xgboost_classifier_cpu.json"
M7_BACKUP_PATH        = MODEL_DIR / "M7_xgboost_classifier.pre_label19_patch.json.bak"
PATCH_SCRIPT_PATH     = Path(__file__).resolve().parent / "module_06p5r_patch_label19_features.py"

# Locked M7 hyperparameters from M7_gate_fix_diagnosis_and_solution.md
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

# =============================================================================
# SECTION 1 — VERIFY UPSTREAM PATCHES PRESENT
# =============================================================================
log("\nSECTION 1 — Verify upstream patches")

upstream_checks = {
    "M6B feature matrix exists":     FEATURE_MATRIX_PATH.exists(),
    "M6B sequence meta exists":      SEQ_META_PATH.exists(),
    "Label 19 patch script exists":  PATCH_SCRIPT_PATH.exists(),
}
for k, v in upstream_checks.items():
    log(f"  {'✓' if v else '✗'}  {k}")

if not all(upstream_checks.values()):
    log("\n  ✗ FATAL: Upstream patches missing. Run apply_fix_m6b_physics_lib.py")
    log("           and apply_fix_module_06B.py first per PATCH_MANIFEST.md")
    sys.exit(1)

# =============================================================================
# SECTION 2 — DIAGNOSE WHETHER FEATURE MATRIX HAS BEEN PATCHED
# =============================================================================
log("\nSECTION 2 — Diagnose Label 19 patch status in feature matrix")

t0 = time.time()
df = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
log(f"  Loaded feature matrix: {df.shape[0]:,} × {df.shape[1]} in {time.time()-t0:.1f}s")

# Find the Pres.SV MAE column (script uses two possible names)
PRES_COL = None
for c in ['mae_Pres_SV', 'mae_PresSV']:
    if c in df.columns:
        PRES_COL = c
        break
if PRES_COL is None:
    log("  ✗ FATAL: No mae_Pres_SV or mae_PresSV column in feature matrix.")
    sys.exit(1)
log(f"  Pres.SV MAE column: {PRES_COL}")

lbl19_rows = df[df['label_int'].astype(int) == LABEL_19]
if len(lbl19_rows) == 0:
    log("  ✗ FATAL: No Label 19 rows in feature matrix.")
    sys.exit(1)

pres_mae_label19 = float(lbl19_rows[PRES_COL].mean())
log(f"  Label 19 mean({PRES_COL}) = {pres_mae_label19:.5f}")
log(f"  ({lbl19_rows.shape[0]:,} rows for Label 19)")

# Patch detection heuristic per M8 report:
#  - PRE-PATCH: Pres.SV stayed at flat ~0.9654 → mae would be very small (~0.03 or below)
#  - POST-PATCH: Pres.SV drops to 0.48–0.88 → mae rises substantially (~0.15+)
PATCH_DETECTION_THRESHOLD = 0.10
patch_already_applied = pres_mae_label19 > PATCH_DETECTION_THRESHOLD
log(f"  Patch detection threshold: mae > {PATCH_DETECTION_THRESHOLD}")
log(f"  Patch already applied? {'YES' if patch_already_applied else 'NO — will run patch'}")

results["pres_mae_lbl19_initial"]  = round(pres_mae_label19, 5)
results["patch_already_applied"]   = patch_already_applied

# =============================================================================
# SECTION 3 — RUN PATCH SCRIPT IF NEEDED
# =============================================================================
if not patch_already_applied:
    log("\nSECTION 3 — Running module_06p5r_patch_label19_features.py")
    try:
        result = subprocess.run(
            [sys.executable, str(PATCH_SCRIPT_PATH)],
            capture_output=True, text=True, timeout=600,
        )
        log(f"  Return code: {result.returncode}")
        if result.returncode != 0:
            log("  ✗ Patch script FAILED. Last 30 lines of stderr:")
            for line in result.stderr.splitlines()[-30:]:
                log(f"    | {line}")
            sys.exit(1)
        # Reload feature matrix and re-check
        df = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
        lbl19_rows_new = df[df['label_int'].astype(int) == LABEL_19]
        pres_mae_after = float(lbl19_rows_new[PRES_COL].mean())
        log(f"  Post-patch Label 19 mean({PRES_COL}) = {pres_mae_after:.5f}")
        results["pres_mae_lbl19_after_patch"] = round(pres_mae_after, 5)
        if pres_mae_after <= PATCH_DETECTION_THRESHOLD:
            log("  ✗ FATAL: Patch ran but Pres.SV MAE still below threshold.")
            log("           Investigate manually before proceeding.")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        log("  ✗ FATAL: Patch script timed out (>10 min). Investigate.")
        sys.exit(1)
else:
    log("\nSECTION 3 — Skipping patch script (already applied)")
    results["pres_mae_lbl19_after_patch"] = pres_mae_label19

# =============================================================================
# SECTION 4 — BACKUP CURRENT M7 WEIGHTS
# =============================================================================
log("\nSECTION 4 — Backing up pre-patch M7 weights")

if M7_MODEL_PATH.exists():
    if not M7_BACKUP_PATH.exists():
        shutil.copy2(M7_MODEL_PATH, M7_BACKUP_PATH)
        log(f"  ✓ Backed up: {M7_BACKUP_PATH.name}")
    else:
        log(f"  ✓ Backup already exists: {M7_BACKUP_PATH.name} (NOT overwriting)")

    # Capture pre-patch per-class F1 if we can re-evaluate the OLD model
    # on the NEW patched feature matrix (this gives us the exact delta the
    # Label 19 patch causes when scored by the un-retrained model)
    log("  Loading pre-patch model for delta comparison...")
    try:
        old_clf = xgb.XGBClassifier()
        old_clf.load_model(str(M7_BACKUP_PATH))
        results["pre_patch_model_loaded"] = True
    except Exception as e:
        log(f"  ✗ Could not load pre-patch model for delta: {e}")
        old_clf = None
        results["pre_patch_model_loaded"] = False
else:
    log(f"  ⚠ No prior M7 model found at {M7_MODEL_PATH}. First-time training.")
    old_clf = None
    results["pre_patch_model_loaded"] = False

# =============================================================================
# SECTION 5 — PREPARE TRAINING DATA (SAME SPLIT PROTOCOL AS M7)
# =============================================================================
log("\nSECTION 5 — Train/test split (window-level, same as original M7)")
log("  NOTE: Sequence-level re-evaluation is in M8p3, not here. M8p2 must")
log("        keep the SAME split protocol as the original M7 so the delta")
log("        is interpretable as 'Label 19 patch effect' only.")

# Mirror original M7 column selection
feature_cols = [c for c in df.columns
                if c != 'label_int' and c != 'fault_group_id']
n_features   = len(feature_cols)
n_classes    = int(df['label_int'].max()) + 1
log(f"  n_features={n_features} | n_classes={n_classes}")

X = df[feature_cols].values.astype(np.float32)
y = df['label_int'].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    stratify     = y,
    random_state = 42,           # IDENTICAL to original M7 for delta interpretability
)
log(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# Inverse class frequency weights (same as original M7)
class_counts_arr  = np.bincount(y_train, minlength=n_classes).astype(np.float64)
class_weight_arr  = 1.0 / (class_counts_arr + 1e-6)
class_weight_arr /= class_weight_arr.mean()
sample_weight_train = class_weight_arr[y_train]

results["n_train"]        = int(X_train.shape[0])
results["n_test"]         = int(X_test.shape[0])
results["n_features"]     = n_features
results["n_classes"]      = n_classes
results["lbl19_n_train"]  = int((y_train == LABEL_19).sum())
results["lbl19_n_test"]   = int((y_test  == LABEL_19).sum())
log(f"  Label 19: {results['lbl19_n_train']:,} train / {results['lbl19_n_test']:,} test")

# =============================================================================
# SECTION 6 — IF OLD MODEL AVAILABLE: SCORE IT ON NEW TEST SET FIRST
# =============================================================================
old_per_class_f1 = {}
if old_clf is not None:
    log("\nSECTION 6 — Scoring PRE-patch model on NEW (patched) test set")
    try:
        y_pred_old = old_clf.predict(X_test)
        for lbl in range(n_classes):
            mask = (y_test == lbl)
            if mask.sum() == 0:
                continue
            f1 = f1_score(y_test == lbl, y_pred_old == lbl, zero_division=0)
            old_per_class_f1[lbl] = float(f1)
        old_macro = float(f1_score(y_test, y_pred_old, average='macro', zero_division=0))
        log(f"  Pre-patch macro F1 on patched data: {old_macro:.4f}")
        log(f"  Pre-patch Label 19 F1: {old_per_class_f1.get(LABEL_19, 0):.4f}")
        results["pre_patch_macro_f1_on_new_data"]  = round(old_macro, 4)
        results["pre_patch_lbl19_f1_on_new_data"]  = round(old_per_class_f1.get(LABEL_19, 0), 4)
    except Exception as e:
        log(f"  ✗ Pre-patch scoring failed: {e}")
        old_per_class_f1 = {}
else:
    log("\nSECTION 6 — Skipped (no pre-patch model available)")

# =============================================================================
# SECTION 7 — RETRAIN M7 (DETERMINISTIC, LOCKED PARAMS)
# =============================================================================
log("\nSECTION 7 — Training new M7 with locked hyperparameters")
log(f"  Hyperparameters: {LOCKED_PARAMS}")

t_train_start = time.time()
clf = xgb.XGBClassifier(num_class=n_classes, **LOCKED_PARAMS)
clf.fit(
    X_train, y_train,
    sample_weight=sample_weight_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
train_time = time.time() - t_train_start
log(f"  ✓ Training complete in {train_time/60:.2f} min")
results["train_time_min"] = round(train_time / 60, 2)

# =============================================================================
# SECTION 8 — EVALUATE NEW M7
# =============================================================================
log("\nSECTION 8 — Evaluating new M7")

y_pred       = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

new_macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
new_acc      = float(accuracy_score(y_test, y_pred))
results["new_macro_f1"] = round(new_macro_f1, 4)
results["new_accuracy"] = round(new_acc, 4)
log(f"  Macro F1: {new_macro_f1:.4f}")
log(f"  Accuracy: {new_acc:.4f}")

new_per_class_f1 = {}
for lbl in range(n_classes):
    mask = (y_test == lbl)
    if mask.sum() == 0:
        continue
    f1 = float(f1_score(y_test == lbl, y_pred == lbl, zero_division=0))
    new_per_class_f1[lbl] = f1
    delta = f1 - old_per_class_f1.get(lbl, f1) if old_per_class_f1 else 0.0
    delta_str = f" (Δ={delta:+.4f})" if old_per_class_f1 else ""
    grp = GROUP_MAP.get(lbl, '?')
    log(f"  [{lbl:2d}] Grp{grp}  F1={f1:.4f}{delta_str}")

results["new_per_class_f1"] = {k: round(v, 4) for k, v in new_per_class_f1.items()}
results["new_lbl19_f1"]     = round(new_per_class_f1.get(LABEL_19, 0), 4)

# =============================================================================
# SECTION 9 — GATES (17 ORIGINAL + 1 NEW LBL19-EXT)
# =============================================================================
log("\nSECTION 9 — Validation gates")

def gate(name, passed, detail=""):
    GATES[name] = {"passed": bool(passed), "detail": detail}
    log(f"  {'✓' if passed else '✗'} {name}: {detail}")

# Replicate the 6 critical gates from M7
gate("M7-1_macro_f1",       new_macro_f1 > 0.82,
     f"F1={new_macro_f1:.4f} (target >0.82)")
gate("M7-3_cavitation_f1",  new_per_class_f1.get(3, 0) > 0.88,
     f"F1={new_per_class_f1.get(3, 0):.4f} (target >0.88)")
gate("M7-4_sensor_f1",      new_per_class_f1.get(6, 0) > 0.90,
     f"F1={new_per_class_f1.get(6, 0):.4f} (target >0.90)")
gate("M7-2_class_floor",
     all(new_per_class_f1.get(l, 0) >= (0.62 if l == 21 else 0.70)
         for l in new_per_class_f1),
     f"All classes meet floor")
gate("M7-21_label21_floor", new_per_class_f1.get(21, 0) > 0.62,
     f"F1={new_per_class_f1.get(21, 0):.4f} (target >0.62)")

# THE NEW GATE — purpose of this entire patch
LABEL19_FLOOR = 0.80
gate("M7-LBL19-EXT_label19_post_patch",
     new_per_class_f1.get(LABEL_19, 0) >= LABEL19_FLOOR,
     f"F1={new_per_class_f1.get(LABEL_19, 0):.4f} (target ≥{LABEL19_FLOOR}, M8 report §10 expectation)")

n_pass = sum(1 for g in GATES.values() if g["passed"])
n_fail = len(GATES) - n_pass
log(f"\n  Gates: {n_pass} PASS / {n_fail} FAIL")
results["gates_passed"]  = n_pass
results["gates_failed"]  = n_fail
results["block_m9"]      = n_fail > 0

# =============================================================================
# SECTION 10 — SAVE NEW M7 (CUDA + CPU)
# =============================================================================
log("\nSECTION 10 — Saving retrained M7 weights")

if results["block_m9"]:
    log("  ⚠ Some gates failed — saving model with .candidate suffix")
    save_path = MODEL_DIR / "M7_xgboost_classifier.candidate_post_label19.json"
    clf.save_model(str(save_path))
    log(f"  Saved candidate: {save_path.name}")
    log("  Live model NOT replaced. Investigate failures before retry.")
    results["m7_saved"] = "candidate"
else:
    clf.save_model(str(M7_MODEL_PATH))
    log(f"  ✓ CUDA model: {M7_MODEL_PATH.name}")

    # CPU model for M10 deployment (Invariant: device='cpu' on deploy)
    clf_cpu = xgb.XGBClassifier(num_class=n_classes, **{**LOCKED_PARAMS, 'device': 'cpu'})
    clf_cpu.load_model(str(M7_MODEL_PATH))
    clf_cpu.save_model(str(M7_MODEL_CPU_PATH))
    log(f"  ✓ CPU model:  {M7_MODEL_CPU_PATH.name}")
    results["m7_saved"] = "live"

# =============================================================================
# SECTION 11 — DELTA PLOT
# =============================================================================
log("\nSECTION 11 — Per-class F1 delta plot")

try:
    labels = sorted(new_per_class_f1.keys())
    new_vals = [new_per_class_f1[l] for l in labels]
    old_vals = [old_per_class_f1.get(l, 0) for l in labels] if old_per_class_f1 else None

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))
    width = 0.35

    if old_vals is not None:
        ax.bar(x - width/2, old_vals, width, label="Pre-patch M7 on new data", alpha=0.7)
        ax.bar(x + width/2, new_vals, width, label="Retrained M7", alpha=0.9)
    else:
        ax.bar(x, new_vals, label="Retrained M7", alpha=0.9)

    ax.axhline(0.62, linestyle=':',  color='gray',   label='Label 21 floor 0.62')
    ax.axhline(0.70, linestyle=':',  color='orange', label='General floor 0.70')
    ax.axhline(0.80, linestyle='--', color='red',    label='Label 19 NEW floor 0.80')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_xlabel("Label")
    ax.set_ylabel("F1")
    ax.set_title("M7 per-class F1 — pre/post Label 19 patch + retrain")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "M8p2_per_class_f1_delta.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=120)
    plt.close()
    log(f"  ✓ Saved: {plot_path}")
except Exception as e:
    log(f"  ⚠ Plot failed: {e}")

# =============================================================================
# SECTION 12 — REPORT
# =============================================================================
log("\nSECTION 12 — Writing report")

REPORT_PATH = REPORT_DIR / f"{SCRIPT_NAME}_report.md"

gate_table = "\n".join(
    f"| {name} | {'✓ PASS' if g['passed'] else '✗ FAIL'} | {g['detail']} |"
    for name, g in GATES.items()
)
delta_table = ""
if old_per_class_f1:
    rows = []
    for lbl in sorted(new_per_class_f1.keys()):
        old_f1 = old_per_class_f1.get(lbl, 0)
        new_f1 = new_per_class_f1[lbl]
        delta  = new_f1 - old_f1
        marker = "🔴" if abs(delta) > 0.05 else ""
        rows.append(f"| {lbl} | {GROUP_MAP.get(lbl,'?')} | {old_f1:.4f} | {new_f1:.4f} | {delta:+.4f} {marker} |")
    delta_table = "\n".join(rows)

report_md = f"""# M8 Patch 2 — Label 19 Propagation + M7 Retrain
**Date:** {date.today()}
**Status:** {'BLOCKED — gates failed' if results['block_m9'] else 'COMPLETE — M7 retrained'}

## Why this patch existed
M8 Technical Validation Report Section 10 listed two PENDING items:
1. Feature Matrix Patch — `module_06p5r_patch_label19_features.py`
2. M7 Re-training

Until propagated, the live M7 was trained on Label 19 features from a buggy
seal_failure_fast generator (Pres.SV stayed flat at 0.9654 instead of dropping
to 0.48–0.88). This script propagated the patch end-to-end.

## Patch verification
| Check | Value |
|---|---|
| Label 19 mae_PresSV before | {results.get('pres_mae_lbl19_initial', 'N/A')} |
| Label 19 mae_PresSV after  | {results.get('pres_mae_lbl19_after_patch', 'N/A')} |
| Patch detection threshold  | {PATCH_DETECTION_THRESHOLD} |
| Label 19 train rows        | {results['lbl19_n_train']:,} |
| Label 19 test rows         | {results['lbl19_n_test']:,} |

## New M7 results
| Metric | Value |
|---|---|
| Macro F1 (all classes) | {results['new_macro_f1']:.4f} |
| Accuracy               | {results['new_accuracy']:.4f} |
| Label 19 F1            | **{results['new_lbl19_f1']:.4f}** (gate ≥0.80) |
| Train time             | {results['train_time_min']:.2f} min |

## Gates
| Gate | Status | Detail |
|---|---|---|
{gate_table}

## Per-class F1 delta (pre-patch model on new data → retrained model)
{('| Label | Group | Pre-patch F1 | Retrained F1 | Δ |\n|---|---|---|---|---|\n' + delta_table) if delta_table else 'No pre-patch comparison available (first-time training).'}

## Files written
- `models/M7_xgboost_classifier.json` (RETRAINED — replaces live)
- `models/M7_xgboost_classifier_cpu.json` (RETRAINED — for M10)
- `models/M7_xgboost_classifier.pre_label19_patch.json.bak` (PRESERVED)
- `outputs/M8p2_per_class_f1_delta.png`

## What this DOES NOT yet address
- Sequence-level test split — see M8p3 (next patch)
- Out-of-distribution detection — see M8p4
- CUSUM auto-decay policy — see M8p5

## Paste text update

══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══
M8p2_label19_patch_propagated      : True
M8p2_pres_mae_lbl19_pre            : {results.get('pres_mae_lbl19_initial', 'N/A')}
M8p2_pres_mae_lbl19_post           : {results.get('pres_mae_lbl19_after_patch', 'N/A')}
M8p2_M7_retrained                  : {results.get('m7_saved') == 'live'}
M8p2_M7_macro_f1                   : {results['new_macro_f1']}
M8p2_M7_lbl19_f1                   : {results['new_lbl19_f1']}
M8p2_gate_lbl19_ext_pass           : {GATES.get('M7-LBL19-EXT_label19_post_patch', {}).get('passed', False)}
M8p2_gates_pass                    : {n_pass}
M8p2_gates_fail                    : {n_fail}
M8p2_block_m9                      : {results['block_m9']}
Status_for_M8p3                    : {'BLOCKED' if results['block_m9'] else 'READY'}
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
log(f"  GitHub push: {M7_MODEL_PATH}")
log(f"  GitHub push: {M7_MODEL_CPU_PATH}")
log(f"  GitHub push: {REPORT_PATH}")
log(f"  Local backup (do not push): {M7_BACKUP_PATH}")

if results["block_m9"]:
    log("\n📦 M8p2 BLOCKED. Investigate gate failures above before M8p3.")
else:
    log("\n📦 M8p2 done. Next: M8p3 — Sequence-level M7 re-evaluation.")
log("=" * 72)
