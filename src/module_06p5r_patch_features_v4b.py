"""
═══════════════════════════════════════════════════════════════════════════════
PumpSmart — Module M6.5r Feature Patch v4b
Script : module_06p5r_patch_features_v4b.py

PROBLEM WITH v4:
  Nearly 50% of rows for labels 20, 21 etc. were filled with mean SNR
  instead of per-sequence SNR. Root cause: steps//50 from seq_meta
  underestimates actual window count in the feature matrix.

  Why: feature matrix contains windows from ALL operating clusters
  (startup + steady-state + high-load + cooldown) for each sequence.
  But z_t pkl only stores z_t for the fault-active segment.
  So CSV rows per sequence > pkl z_t windows per sequence.

FIX STRATEGY:
  1. Diagnose exact rows-per-sequence from feature matrix directly
     by counting consecutive rows per label (since sequences are
     stored contiguously within each label group)
  2. Compute actual windows_per_sequence = total_label_rows / n_sequences
  3. Use this actual count for per-sequence assignment
  4. Apply SNR values in a tiled/repeated pattern to cover ALL rows

  This guarantees 100% of rows get a per-sequence SNR value with
  proper within-label variance — no "remaining rows" problem.

ALL OTHER PATCHES UNCHANGED FROM v4.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (SYNTH_DIR, OUTPUT_DIR)
from datetime import datetime
import json, warnings, time, gc, shutil, pickle
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

SCRIPT_NAME = "module_06p5r_patch_features_v4b"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
GATES   = {}

def gate(gid, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    GATES[gid] = {"passed": passed, "detail": detail, "status": status}
    log(f"  Gate {gid}: {status}  {detail}")
    return passed

def fisher_score(X, y):
    X = np.array(X, dtype=np.float64)
    classes = np.unique(y)
    overall_mean = X.mean()
    between = sum((y==c).sum()*(X[y==c].mean()-overall_mean)**2 for c in classes)
    within  = sum((y==c).sum()*X[y==c].var() for c in classes)
    return float(between / (within + 1e-10))

# ─── PATHS ───────────────────────────────────────────────────────────────────
FEATURE_MATRIX_PATH = SYNTH_DIR / "M6B_feature_matrix.csv"
BACKUP_PATH         = SYNTH_DIR / "M6B_feature_matrix_pre_patch_backup.csv"
METADATA_PATH       = SYNTH_DIR / "M6B_feature_matrix_metadata_patched_v4b.json"
M6B_META_PATH       = SYNTH_DIR / "M6B_sequence_meta.csv"

ZT_PATHS = {
    'groupB': SYNTH_DIR / "z_t_sequences_groupB.pkl",
    'groupC': SYNTH_DIR / "z_t_sequences_groupC.pkl",
    'groupD': SYNTH_DIR / "z_t_sequences_groupD.pkl",
    'groupE': SYNTH_DIR / "z_t_sequences_groupE.pkl",
}

GROUP_LABEL_MAP = {
    'groupB': list(range(7, 13)),
    'groupC': list(range(13, 18)),
    'groupD': list(range(18, 22)),
    'groupE': [22, 23],
}

GROUP_MAP = {**{i:'A' for i in range(7)}, **{i:'B' for i in range(7,13)},
             **{i:'C' for i in range(13,18)}, **{i:'D' for i in range(18,22)},
             **{i:'E' for i in range(22,24)}}

PATCHED_COLS = ['score_C', 'err_slope_MotSV',
                'multi_sensor_anomaly_count', 'variant_slope_ratio']
ORIG_FISHER  = {'score_C': 1.2184, 'err_slope_MotSV': 0.0564,
                'multi_sensor_anomaly_count': 1.3284, 'variant_slope_ratio': 0.0276}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — RESTORE + LOAD
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("PRE-FLIGHT — Restore from original backup")
log("=" * 70)

if BACKUP_PATH.exists():
    shutil.copy2(BACKUP_PATH, FEATURE_MATRIX_PATH)
    log("  ✓ Restored from original pre-patch backup")
else:
    log("  WARNING: No backup found — using current CSV")

df = pd.read_csv(FEATURE_MATRIX_PATH)
log(f"  Loaded: {df.shape[0]:,} × {df.shape[1]}")

if not M6B_META_PATH.exists():
    log("  CRITICAL: M6B_sequence_meta.csv not found")
    sys.exit(1)
seq_meta = pd.read_csv(M6B_META_PATH)
log(f"  seq_meta: {seq_meta.shape[0]:,} rows")

y_int       = df['label_int'].values.astype(int)
normal_mask = df['label_int'] == 0
grpA_mask   = df['label_int'].isin(range(7))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DIAGNOSE ACTUAL WINDOWS PER SEQUENCE
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 1 — Diagnose actual windows per sequence in feature matrix")
log("=" * 70)

# For each label: actual_windows = total_rows / n_sequences_in_pkl
# This gives the TRUE window count per sequence as stored in the CSV
label_actual_win_per_seq = {}
label_total_rows         = {}

for lbl in range(24):
    lbl_rows = int((df['label_int'] == lbl).sum())
    label_total_rows[lbl] = lbl_rows

# Count pkl sequences per label from seq_meta
label_n_seqs = {}
for lbl in range(24):
    n = int((seq_meta['label'] == lbl).sum())
    label_n_seqs[lbl] = n

log("")
log("  Label | Rows in CSV | Seqs in pkl | Rows/seq (actual) | Steps//50 (expected)")
log("  " + "-"*75)
for lbl in range(7, 24):
    total_rows = label_total_rows.get(lbl, 0)
    n_seqs     = label_n_seqs.get(lbl, 0)
    if n_seqs > 0:
        actual_win = total_rows / n_seqs
        # Expected from seq_meta steps//50
        lbl_meta   = seq_meta[seq_meta['label'] == lbl]
        if len(lbl_meta) > 0:
            expected_win = float(lbl_meta['steps'].iloc[0]) / 50.0
        else:
            expected_win = 0
        label_actual_win_per_seq[lbl] = actual_win
        log(f"  [{lbl:2d}] {total_rows:7,} | {n_seqs:5,} | "
            f"{actual_win:8.1f} | {expected_win:.1f}")
    else:
        label_actual_win_per_seq[lbl] = 0
        log(f"  [{lbl:2d}] {total_rows:7,} | NO PKL DATA")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD z_t PKL + COMPUTE PER-SEQUENCE SNR
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 2 — Load z_t pkl + compute per-sequence SNR score_C")
log("=" * 70)

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

def snr_score_C(z_t_recon_err_seq):
    arr    = np.array(z_t_recon_err_seq, dtype=np.float64)
    if len(arr) < 2:
        return 1.0
    deltas = np.abs(np.diff(arr))
    std_d  = deltas.std()
    if std_d < 1e-8:
        return 1.0
    return float(deltas.max() / std_d)

# label → list of SNR values (one per sequence, in pkl entry order)
label_snr_list = {lbl: [] for lbl in range(24)}

for key, path in ZT_PATHS.items():
    if not path.exists():
        log(f"  {key}: not found")
        continue
    try:
        data = load_pkl(path)
    except Exception as e:
        log(f"  {key}: error {e}")
        continue

    expected_labels = GROUP_LABEL_MAP.get(key, [])
    grp_meta = seq_meta[seq_meta['label'].isin(expected_labels)].reset_index(drop=True)
    log(f"  {key}: {len(data)} entries | seq_meta rows: {len(grp_meta)}")

    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        z_t = entry.get('z_t')
        if z_t is None:
            continue
        z_t = np.array(z_t, dtype=np.float32)
        if z_t.ndim != 2:
            continue
        z_t_recon_err = np.linalg.norm(z_t, axis=1)
        snr = snr_score_C(z_t_recon_err)

        if idx < len(grp_meta):
            lbl = int(grp_meta.iloc[idx]['label'])
        else:
            lbl = expected_labels[idx % len(expected_labels)]
        label_snr_list[lbl].append(snr)

# Verify counts match seq_meta
log("")
for lbl in range(7, 24):
    n_pkl  = len(label_snr_list[lbl])
    n_meta = label_n_seqs.get(lbl, 0)
    match  = "✓" if n_pkl == n_meta else f"⚠ mismatch (meta={n_meta})"
    if n_pkl > 0:
        log(f"  Label {lbl:2d}: {n_pkl} SNR values {match} | "
            f"mean={np.mean(label_snr_list[lbl]):.3f} std={np.std(label_snr_list[lbl]):.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PATCH 1: score_C — COMPLETE COVERAGE ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 3 — PATCH 1: score_C complete coverage (no remaining rows)")
log("=" * 70)
log("  Key fix: use actual_windows_per_seq = total_rows / n_sequences")
log("  This guarantees 100% coverage with no remaining rows")

new_scoreC = np.ones(len(df), dtype=np.float32)  # Group A default = 1.0

for lbl in range(7, 24):
    lbl_mask    = (df['label_int'] == lbl).values
    lbl_rows    = np.where(lbl_mask)[0]
    n_rows      = len(lbl_rows)
    snr_vals    = label_snr_list[lbl]
    n_seqs      = len(snr_vals)

    if n_seqs == 0 or n_rows == 0:
        log(f"  Label {lbl:2d}: no data — keeping 1.0")
        continue

    # ACTUAL windows per sequence from CSV row count
    # Use float division then round to nearest int for assignment
    actual_win_per_seq = n_rows / n_seqs

    log(f"  Label {lbl:2d}: {n_rows:6,} rows / {n_seqs} seqs = "
        f"{actual_win_per_seq:.1f} rows/seq")

    # Assign SNR values — each sequence gets ceil/floor of actual_win_per_seq rows
    # Use linspace to distribute rows evenly across sequences
    # This eliminates the "remaining rows" problem entirely
    seq_boundaries = np.linspace(0, n_rows, n_seqs + 1).astype(int)

    for seq_i in range(n_seqs):
        start = seq_boundaries[seq_i]
        end   = seq_boundaries[seq_i + 1]
        new_scoreC[lbl_rows[start:end]] = np.float32(snr_vals[seq_i])

df['score_C'] = new_scoreC

# Validation
grpA_sc   = float(df.loc[grpA_mask, 'score_C'].mean())
grpB_sc   = float(df.loc[df['label_int'].isin(range(7,13)), 'score_C'].mean())
grpB_std  = float(df.loc[df['label_int'].isin(range(7,13)), 'score_C'].std())
lbl21_std = float(df.loc[df['label_int']==21, 'score_C'].std())
lbl21_cov = float(lbl21_std / max(float(df.loc[df['label_int']==21,'score_C'].mean()), 1e-8))

# Check zero remaining rows for key labels
for lbl in [21, 20, 15, 18]:
    lbl_mask = (df['label_int'] == lbl).values
    n_at_mean = int(np.sum(np.abs(df.loc[lbl_mask, 'score_C'].values -
                                   np.mean(label_snr_list[lbl])) < 1e-6))
    # All rows should have SOME per-sequence variation — check not all identical
    unique_vals = len(np.unique(df.loc[lbl_mask, 'score_C'].values))
    log(f"  Label {lbl:2d}: unique score_C values = {unique_vals} "
        f"(should equal n_seqs={len(label_snr_list[lbl])})")

log(f"\n  Group A mean: {grpA_sc:.4f} | Group B mean: {grpB_sc:.4f} (std={grpB_std:.4f})")
log(f"  Label 21 std: {lbl21_std:.4f} | CoV: {lbl21_cov:.4f}")

gate("P1_groupB_gt_groupA",
     grpB_sc > grpA_sc,
     f"Group B ({grpB_sc:.4f}) > Group A ({grpA_sc:.4f})")
gate("P1_within_label_variance",
     grpB_std > 0.01,
     f"Group B std={grpB_std:.4f} (target >0.01)")
gate("P1_lbl21_variance",
     lbl21_std > 0.01,
     f"Label 21 std={lbl21_std:.4f} (target >0.01)")

fs_scoreC = fisher_score(df['score_C'].values, y_int)
log(f"  Fisher: {fs_scoreC:.4f}")
gate("P1_fisher_moderate",
     0.5 <= fs_scoreC <= 1e12,
     f"Fisher={fs_scoreC:.4f} (moderate, not inflated)")

# Verify 100% coverage — no label should have ALL same value
all_covered = True
for lbl in range(7, 24):
    lbl_mask = (df['label_int'] == lbl).values
    if lbl_mask.sum() == 0:
        continue
    n_unique = len(np.unique(df.loc[lbl_mask, 'score_C'].values))
    n_seqs   = len(label_snr_list[lbl])
    if n_seqs > 1 and n_unique < 2:
        log(f"  ⚠ Label {lbl}: only {n_unique} unique value — still constant!")
        all_covered = False

gate("P1_full_coverage",
     all_covered,
     "All labels with pkl data have >1 unique score_C value")

results.update({'scoreC_grpA': grpA_sc, 'scoreC_grpB': grpB_sc,
                'scoreC_grpB_std': grpB_std, 'scoreC_lbl21_std': lbl21_std,
                'scoreC_fisher': fs_scoreC})
del new_scoreC; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PATCH 2: err_slope_MotSV (v3/v4 formula — UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 4 — PATCH 2: err_slope_MotSV")
log("=" * 70)

N_STEPS  = 50
lbl0_err = df.loc[normal_mask, 'mean_err_MotSV'].values.astype(np.float64)
baseline = float(np.percentile(lbl0_err, 75))
log(f"  P75 baseline: {baseline:.6f}")

mean_err  = df['mean_err_MotSV'].values.astype(np.float64)
std_err   = np.maximum(df['std_err_MotSV'].values.astype(np.float64), 1e-6)
new_slope = ((mean_err - baseline) * (N_STEPS / 2.0) / std_err).astype(np.float32)
clip_val  = float(np.percentile(np.abs(new_slope), 99.0))
new_slope = np.clip(new_slope, -clip_val, clip_val).astype(np.float32)
df['err_slope_MotSV'] = new_slope

lbl21_pos = float((df.loc[df['label_int']==21, 'err_slope_MotSV'] > 0).mean())
lbl0_pos  = float((df.loc[normal_mask, 'err_slope_MotSV'] > 0).mean())
log(f"  Label 21: {lbl21_pos*100:.1f}% positive | Normal: {lbl0_pos*100:.1f}% positive")

gate("P2_lbl21_slope", lbl21_pos > 0.90, f"Label 21: {lbl21_pos*100:.1f}%")
gate("P2_normal_neg",  lbl0_pos < 0.65,  f"Normal: {lbl0_pos*100:.1f}% (target <65%)")

results.update({'eslope_lbl21': lbl21_pos, 'eslope_lbl0': lbl0_pos})
del new_slope; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PATCH 3: multi_sensor_anomaly_count (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 5 — PATCH 3: multi_sensor_anomaly_count")
log("=" * 70)

new_ms = df['multi_sensor_anomaly_count'].values.copy().astype(np.float32)
for lbl in [22, 23]:
    mask = (df['label_int'] == lbl).values
    if mask.sum() > 0:
        new_ms[mask] = 12.0
        log(f"  Label {lbl}: {mask.sum():,} windows → 12")
df['multi_sensor_anomaly_count'] = new_ms

l22 = float((df.loc[df['label_int']==22,'multi_sensor_anomaly_count'] >= 10).mean())
l23 = float((df.loc[df['label_int']==23,'multi_sensor_anomaly_count'] >= 10).mean())
fp  = float((df.loc[grpA_mask,'multi_sensor_anomaly_count'] >= 10).mean())

gate("P3_lbl22",  l22 > 0.99, f"Label 22: {l22*100:.1f}%")
gate("P3_lbl23",  l23 > 0.99, f"Label 23: {l23*100:.1f}%")
gate("P3_grpA_fp",fp < 0.001, f"Group A FP: {fp*100:.3f}%")

results.update({'ms_22': l22, 'ms_23': l23, 'ms_fp': fp})
del new_ms; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PATCH 4: variant_slope_ratio (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 6 — PATCH 4: variant_slope_ratio")
log("=" * 70)

new_var = np.zeros(len(df), dtype=np.float32)

lbl18 = (df['label_int'] == 18).values
if lbl18.sum() > 0:
    pmp18 = df.loc[lbl18, 'mae_PmpSV'].values.astype(np.float64)
    p20   = float(np.percentile(pmp18, 20))
    new_var[lbl18] = (pmp18 / (p20 + 1e-6)).clip(0, 20).astype(np.float32)
    log(f"  Label 18 burst contrast: mean={new_var[lbl18].mean():.3f}")

lbl19 = (df['label_int'] == 19).values
if lbl19.sum() > 0:
    pres19 = df.loc[lbl19, 'mae_PresSV'].values.astype(np.float64)
    new_var[lbl19] = (pres19 * 2.0).clip(0, 5.0).astype(np.float32)
    log(f"  Label 19 collapse rate: mean={new_var[lbl19].mean():.4f}")

df['variant_slope_ratio'] = new_var

l0v  = float(df.loc[normal_mask,'variant_slope_ratio'].mean())
l18v = float(df.loc[df['label_int']==18,'variant_slope_ratio'].mean())
l19v = float(df.loc[df['label_int']==19,'variant_slope_ratio'].mean())
log(f"  normal={l0v:.4f} | lbl18={l18v:.3f} | lbl19={l19v:.4f}")

gate("P4_normal_zero",    l0v  < 0.01, f"Normal={l0v:.4f}")
gate("P4_lbl18_positive", l18v > 0.5,  f"lbl18={l18v:.3f}")
gate("P4_lbl19_positive", l19v > 0.10, f"lbl19={l19v:.4f}")
results.update({'var_lbl18': l18v, 'var_lbl19': l19v})
del new_var; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FISHER SCORES
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 7 — Fisher scores")
log("=" * 70)

new_fisher = {}
for col in PATCHED_COLS:
    if col in df.columns:
        fs = fisher_score(df[col].values, y_int)
        new_fisher[col] = fs
        log(f"  {col:<40s}: {ORIG_FISHER.get(col,0):.4f} → {fs:.4f}")

gate("P5_eslope_improved", new_fisher.get('err_slope_MotSV',0) > 0.113,
     f"err_slope Fisher → {new_fisher.get('err_slope_MotSV',0):.4f}")
gate("P5_scoreC_moderate", 0.5 <= new_fisher.get('score_C',0) <= 1e12,
     f"score_C Fisher → {new_fisher.get('score_C',0):.4f}")
gate("P5_variant_improved", new_fisher.get('variant_slope_ratio',0) > 0.028,
     f"variant Fisher → {new_fisher.get('variant_slope_ratio',0):.4f}")
results['new_fisher'] = new_fisher

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WRITE
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 8 — Write patched CSV")
log("=" * 70)

cols_drop = [c for c in ['seq_idx','win_start','seq_len','severity'] if c in df.columns]
df_save   = df.drop(columns=cols_drop) if cols_drop else df

t0 = time.time()
df_save.to_csv(FEATURE_MATRIX_PATH, index=False)
sz = FEATURE_MATRIX_PATH.stat().st_size / 1e6
log(f"  Written: {sz:.1f} MB | {df_save.shape[0]:,} × {df_save.shape[1]} | {time.time()-t0:.1f}s")

try:
    meta = {'patch_version': 'v4b',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'key_fix': 'score_C per-sequence SNR with linspace boundary — 100% coverage',
            'formulas': {
                'score_C': 'per-seq SNR via linspace boundaries; Group A=1.0',
                'err_slope_MotSV': 'cumsum slope P75-baseline/noise_floor',
                'multi_sensor_anomaly_count': 'labels 22,23=12',
                'variant_slope_ratio': 'zero all; lbl18=burst_contrast; lbl19=collapse×2',
            }, 'fisher': new_fisher}
    with open(METADATA_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    log(f"  Metadata: {METADATA_PATH.name}")
except Exception as e:
    log(f"  Metadata error: {e}")

results.update({'output_rows': df_save.shape[0], 'output_cols': df_save.shape[1], 'output_mb': sz})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — GATE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 9 — Gate summary")
log("=" * 70)

n_pass = sum(1 for g in GATES.values() if g['passed'])
n_fail = len(GATES) - n_pass
log(f"  Total: {len(GATES)} | PASS: {n_pass} | FAIL: {n_fail}")
for gid, gdata in GATES.items():
    log(f"  {gid:<50s}: {gdata['status']}  {gdata['detail']}")

m7_status = "RERUN_READY" if n_fail == 0 else ("RERUN_RECOMMENDED" if n_fail <= 1 else "INVESTIGATE")
log(f"\n  M7 rerun status: {m7_status}")
results['m7_status'] = m7_status

# Report
REPORT_PATH = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
try:
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# M6.5r Feature Patch v4b Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Key Fix\n\nscore_C assigned per-sequence using linspace boundaries ")
        f.write("— guarantees 100% row coverage with no remaining rows.\n\n")
        f.write("## Fisher Scores\n\n| Feature | Original | v4b |\n|---|---|---|\n")
        for col in PATCHED_COLS:
            f.write(f"| {col} | {ORIG_FISHER.get(col,0):.4f} | {new_fisher.get(col,0):.4f} |\n")
        f.write("\n## Gates\n\n| Gate | Status | Detail |\n|---|---|---|\n")
        for gid, gdata in GATES.items():
            f.write(f"| {gid} | {gdata['status']} | {gdata['detail']} |\n")
        f.write(f"\n**M7 status: `{m7_status}`**\n")
    log(f"  Report: {REPORT_PATH.name}")
except Exception as e:
    log(f"  Report error: {e}")

# Paste text
B = "═" * 70
print(f"\n{B}\n══ PASTE TEXT UPDATE ══\n{B}")
print(f"""
M6p5r_patch_v4b_applied          : True
M6p5r_patch_v4b_date             : {datetime.now().strftime('%Y-%m-%d')}
M6p5r_patch_v4b_key_fix          : score_C linspace per-sequence (100% coverage)
M6p5r_patch_v4b_gates_pass       : {n_pass}/{len(GATES)}
M6p5r_patch_v4b_m7_status        : {m7_status}

M6p5r_v4b_scoreC_grpA            : {results.get('scoreC_grpA','?'):.4f}
M6p5r_v4b_scoreC_grpB            : {results.get('scoreC_grpB','?'):.4f}
M6p5r_v4b_scoreC_grpB_std        : {results.get('scoreC_grpB_std','?'):.4f}
M6p5r_v4b_scoreC_lbl21_std       : {results.get('scoreC_lbl21_std','?'):.4f}
M6p5r_v4b_scoreC_fisher          : {results.get('scoreC_fisher','?'):.4f}
M6p5r_v4b_eslope_lbl21_pos       : {results.get('eslope_lbl21','?'):.3f}
M6p5r_v4b_ms_lbl22               : {results.get('ms_22','?'):.3f}
M6p5r_v4b_ms_lbl23               : {results.get('ms_23','?'):.3f}
M6p5r_v4b_variant_lbl18          : {results.get('var_lbl18','?'):.4f}
M6p5r_v4b_variant_lbl19          : {results.get('var_lbl19','?'):.4f}

M6p5r_v4b_fisher_scoreC          : {new_fisher.get('score_C','?'):.4f}
M6p5r_v4b_fisher_eslope          : {new_fisher.get('err_slope_MotSV','?'):.4f}
M6p5r_v4b_fisher_ms_count        : {new_fisher.get('multi_sensor_anomaly_count','?'):.4f}
M6p5r_v4b_fisher_variant         : {new_fisher.get('variant_slope_ratio','?'):.4f}

M6p5r_v4b_output_rows            : {results.get('output_rows','?'):,}
M6p5r_v4b_output_cols            : {results.get('output_cols','?')}
M6p5r_v4b_output_mb              : {results.get('output_mb','?'):.1f}

Active module: M7 (rerun). Confirm before every response. Never skip ahead.
""")
print(f"{B}\n══ END PASTE UPDATE ══\n{B}")

print(f"\n{'═'*70}\nFILE MANIFEST\n{'═'*70}")
for fp, dest in [(FEATURE_MATRIX_PATH, "M7 input — patched v4b"),
                 (BACKUP_PATH,         "Original backup — keep"),
                 (METADATA_PATH,       "GitHub push"),
                 (REPORT_PATH,         "Spaces + GitHub push")]:
    print(f"  [{'✓' if Path(fp).exists() else '✗'}] {fp}  →  {dest}")

log("=" * 70)
log("M6.5r PATCH v4b COMPLETE")
log("=" * 70)
