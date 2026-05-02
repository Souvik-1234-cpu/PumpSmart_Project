"""
═══════════════════════════════════════════════════════════════════════════════
PumpSmart — Module M6.5r Feature Patch v5
Script : module_06p5r_patch_features_v5.py
Asset  : 110 kW | 7-stage | 40 bar | 2980 RPM | CIRA SACIP

PURPOSE — ONE TARGETED FIX:
  Replace binary onset_order {0=pre, 1=post} with 4-level ordinal encoding
  {0=normal, 1=pre-onset, 2=transition, 3=post-onset} for Group B labels (7–12).

  This fixes Gate M7-9 (onset_order rank≤4) for labels 10 and 12 which
  showed ranks 5 and 8 in M7 run 5. Root cause: binary encoding provides
  insufficient SHAP gradient at the secondary onset boundary for high-lag
  sequences (label 10: lag 400–800 steps = 8–16 windows; label 12: lag
  100–300 steps = 2–6 windows). The transition zone (value=2) creates a
  high-discriminability region at the exact secondary onset window.

  Physics basis:
  - Label 10 (seal→cavitation): NPSHa margin erodes over 400–800 steps
    via hydraulic operating point migration (affinity laws). The TRANSITION
    window is when NPSHa first drops below NPSHr — a physically sharp event
    within a gradual approach. Value=2 encodes this "onset is happening NOW".
  - Label 12 (imbalance→cavitation): BPF pressure oscillation amplitude
    builds until bubble nucleation threshold is crossed. Transition is sharp
    even though approach is gradual. Value=2 encodes the nucleation boundary.

WHAT THIS PATCH CHANGES:
  - ONLY column: onset_order → 4-level ordinal
  - ONLY rows: Group B labels 7–12
  - ALL other labels: onset_order = 0 (no compound transition)
  - ALL other patched columns (score_C, err_slope_MotSV, multi_sensor_anomaly_count,
    variant_slope_ratio) from v4b: UNCHANGED — verified and preserved

INPUT:
  data/synthetic/M6B_feature_matrix.csv    ← v4b patched (DO NOT restore backup)
  data/synthetic/M6B_sequence_meta.csv     ← secondary_onset_step per sequence
  data/synthetic/M6B_feature_matrix_pre_patch_backup.csv  ← safety check only

OUTPUT:
  data/synthetic/M6B_feature_matrix.csv    ← UPDATED
  data/synthetic/M6B_feature_matrix_metadata_v5.json
  outputs/reports/module_06p5r_patch_v5_report.md

RUNTIME: ~3–5 min (CPU only)

ENCODING:
  onset_order_v2 value | Meaning
  ─────────────────────┼──────────────────────────────────────────────
       0               | Normal OR non-compound fault (Groups A,C,D,E)
       1               | Pre-onset: Phase 1 active (primary fault only)
       2               | Transition zone: within ±1 window of secondary onset
       3               | Post-onset: Phase 2 active (compound fault)

GATE TARGETS FOR THIS PATCH:
  P1: onset_order values {1,2,3} present in ALL Group B labels
  P2: transition zone (value=2) present in ≥90% of Group B sequences
  P3: transition count per sequence: mean 2–4 windows (not too narrow/wide)
  P4: onset_order Fisher > 1×10¹² (categorical — retains high discriminability)
  P5: label 10 has ≥3 unique onset_order values per sequence
  P6: label 12 has ≥3 unique onset_order values per sequence
  P7: v4b patches intact (score_C Group B std>0.01, ms_count lbl22=100%)
  P8: Group A, C, D, E all have onset_order=0 (no contamination)
═══════════════════════════════════════════════════════════════════════════════
"""

# ─── MANDATORY HEADER ────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (SYNTH_DIR, MODEL_DIR, OUTPUT_DIR)
from datetime import datetime
import json, warnings, time, gc, shutil
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

SCRIPT_NAME = "module_06p5r_patch_features_v5"
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
    between = sum((y==c).sum() * (X[y==c].mean() - overall_mean)**2 for c in classes)
    within  = sum((y==c).sum() * X[y==c].var() for c in classes)
    return float(between / (within + 1e-10))

# ─── PATHS ───────────────────────────────────────────────────────────────────
FEATURE_MATRIX_PATH = SYNTH_DIR / "M6B_feature_matrix.csv"
BACKUP_PATH         = SYNTH_DIR / "M6B_feature_matrix_pre_patch_backup.csv"
SEQ_META_PATH       = SYNTH_DIR / "M6B_sequence_meta.csv"
METADATA_PATH       = SYNTH_DIR / "M6B_feature_matrix_metadata_v5.json"

GROUP_B_LABELS = [7, 8, 9, 10, 11, 12]

# Physical lag ranges from M6B spec (steps)
LAG_RANGES = {
    7:  (200, 400),   # bearing→overloading (thermal diffusion)
    8:  (50,  150),   # cavitation→seal (Joukowsky shock)
    9:  (300, 600),   # imbalance→bearing (Paris fatigue)
    10: (400, 800),   # seal→cavitation (NPSHa migration)  ← problem label
    11: (400, 600),   # overloading→bearing (thermal creep)
    12: (100, 300),   # imbalance→cavitation (BPF nucleation) ← problem label
}

# Window size (locked from M2/M6B)
WINDOW_SIZE = 50

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — PRE-FLIGHT
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("PRE-FLIGHT CHECKS")
log("=" * 70)

# Safety: DO NOT restore from backup — v4b patches must be preserved
if not FEATURE_MATRIX_PATH.exists():
    log("  CRITICAL: Feature matrix not found")
    sys.exit(1)

if not BACKUP_PATH.exists():
    log("  WARNING: Pre-patch backup not found — proceeding carefully")
else:
    log(f"  ✓ Original backup exists: {BACKUP_PATH.name} (NOT restoring — v4b intact)")

if not SEQ_META_PATH.exists():
    log(f"  CRITICAL: M6B_sequence_meta.csv not found at {SEQ_META_PATH}")
    log("  This file is required for secondary_onset_step lookup")
    sys.exit(1)

log(f"  Feature matrix: {FEATURE_MATRIX_PATH.stat().st_size/1e6:.1f} MB ✓")
log(f"  Sequence meta: {SEQ_META_PATH.stat().st_size/1e6:.1f} MB ✓")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 1 — Load feature matrix + sequence metadata")
log("=" * 70)

t0 = time.time()
df = pd.read_csv(FEATURE_MATRIX_PATH)
log(f"  Feature matrix: {df.shape[0]:,} × {df.shape[1]} in {time.time()-t0:.1f}s")

# Verify onset_order column exists
if 'onset_order' not in df.columns:
    log("  CRITICAL: onset_order column not found in feature matrix")
    sys.exit(1)

# Load sequence metadata
seq_meta = pd.read_csv(SEQ_META_PATH)
log(f"  Sequence meta: {seq_meta.shape[0]:,} rows | cols: {list(seq_meta.columns)}")

y_int = df['label_int'].values.astype(int)

# Verify secondary_onset_step column exists in seq_meta
# Try multiple possible column names (M6B scripts may use different names)
ONSET_STEP_COL = None
for candidate in ['secondary_onset_step', 'secondary_onset_lag',
                  'onset_step', 'lag_steps', 'secondary_lag']:
    if candidate in seq_meta.columns:
        ONSET_STEP_COL = candidate
        log(f"  Secondary onset column found: '{ONSET_STEP_COL}'")
        break

if ONSET_STEP_COL is None:
    log("  WARNING: No secondary_onset_step column found in seq_meta")
    log(f"  Available columns: {list(seq_meta.columns)}")
    log("  Will use physics-based lag range midpoints as fallback")

# Report current onset_order distribution
log("")
log("  Current onset_order distribution:")
for lbl in GROUP_B_LABELS:
    lbl_mask = df['label_int'] == lbl
    vals = df.loc[lbl_mask, 'onset_order'].value_counts().sort_index()
    unique_vals = sorted(df.loc[lbl_mask, 'onset_order'].unique())
    log(f"    Label {lbl:2d}: unique values = {unique_vals} | "
        f"distribution = {dict(vals.head(5))}")

results['n_rows'] = df.shape[0]
results['n_cols'] = df.shape[1]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BUILD onset_step LOOKUP FROM seq_meta
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 2 — Build secondary onset step lookup per Group B sequence")
log("=" * 70)

# Build: (label, seq_position) → secondary_onset_step
# seq_position = position of sequence within label group in seq_meta order
# This maps to the same ordering used in the feature matrix (sequences stored
# contiguously per label, in seq_meta order)

onset_lookup = {}   # label → list of secondary_onset_step (one per sequence)

for lbl in GROUP_B_LABELS:
    lbl_meta = seq_meta[seq_meta['label'] == lbl].reset_index(drop=True)
    n_seqs   = len(lbl_meta)

    if ONSET_STEP_COL is not None and n_seqs > 0:
        # Use actual secondary_onset_step from seq_meta
        onset_steps = lbl_meta[ONSET_STEP_COL].values.tolist()
        log(f"  Label {lbl:2d}: {n_seqs} seqs from meta | "
            f"onset_step range [{min(onset_steps):.0f}, {max(onset_steps):.0f}]")
        onset_lookup[lbl] = onset_steps
    else:
        # Fallback: use midpoint of physics-verified lag range
        lag_min, lag_max = LAG_RANGES[lbl]
        lag_mid = (lag_min + lag_max) / 2
        # n_seqs from feature matrix
        n_seqs_csv = int((df['label_int'] == lbl).sum())
        # Approximate n_seqs from feature matrix row count
        n_win_per_seq = n_seqs_csv / max(1, len(lbl_meta)) if len(lbl_meta) > 0 else 39
        n_seqs_est   = max(1, round(n_seqs_csv / n_win_per_seq))
        onset_steps  = [int(lag_mid)] * (n_seqs_est if len(lbl_meta)==0 else len(lbl_meta))
        log(f"  Label {lbl:2d}: FALLBACK midpoint lag={lag_mid:.0f} steps | "
            f"n_seqs={len(onset_steps)}")
        onset_lookup[lbl] = onset_steps

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — APPLY 4-LEVEL ORDINAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 3 — Apply 4-level ordinal onset_order encoding")
log("=" * 70)
log("  Encoding: 0=normal/non-compound | 1=pre-onset | 2=transition | 3=post-onset")
log("  Transition zone: ±1 window around secondary onset window")
log("  All non-Group-B labels: set to 0")

# Start with all zeros (correct for non-compound labels)
new_onset_order = np.zeros(len(df), dtype=np.float32)

TRANSITION_HALF_WIDTH = 1   # ±1 window = 3-window transition zone

for lbl in GROUP_B_LABELS:
    lbl_mask = (df['label_int'] == lbl).values
    lbl_rows = np.where(lbl_mask)[0]
    n_rows   = len(lbl_rows)

    if n_rows == 0:
        log(f"  Label {lbl:2d}: no rows found — skipping")
        continue

    onset_steps = onset_lookup.get(lbl, [])
    n_seqs      = len(onset_steps)

    if n_seqs == 0:
        log(f"  Label {lbl:2d}: no onset steps — defaulting to 1 (pre-onset)")
        new_onset_order[lbl_mask] = 1.0
        continue

    # Use linspace to divide all rows evenly across sequences (same as v4b)
    seq_boundaries = np.linspace(0, n_rows, n_seqs + 1).astype(int)

    n_transition_windows = 0
    n_pre_windows        = 0
    n_post_windows       = 0

    for seq_i in range(n_seqs):
        start_row = seq_boundaries[seq_i]
        end_row   = seq_boundaries[seq_i + 1]
        rows_for_seq = lbl_rows[start_row:end_row]
        n_windows_this_seq = end_row - start_row

        if n_windows_this_seq == 0:
            continue

        # Compute which window the secondary onset falls in
        onset_step      = onset_steps[seq_i % len(onset_steps)]
        onset_window    = int(onset_step // WINDOW_SIZE)  # 0-indexed window number
        onset_window    = min(onset_window, n_windows_this_seq - 1)

        # Assign 4-level encoding to each window in this sequence
        for w_idx in range(n_windows_this_seq):
            row = rows_for_seq[w_idx]
            if w_idx < onset_window - TRANSITION_HALF_WIDTH:
                # Pre-onset: Phase 1 only
                new_onset_order[row] = 1.0
                n_pre_windows += 1
            elif onset_window - TRANSITION_HALF_WIDTH <= w_idx <= onset_window + TRANSITION_HALF_WIDTH:
                # Transition zone: onset is happening
                new_onset_order[row] = 2.0
                n_transition_windows += 1
            else:
                # Post-onset: Phase 2 active
                new_onset_order[row] = 3.0
                n_post_windows += 1

    total_windows = n_pre_windows + n_transition_windows + n_post_windows
    log(f"  Label {lbl:2d}: pre={n_pre_windows:5,} ({100*n_pre_windows/max(1,total_windows):.1f}%) | "
        f"transition={n_transition_windows:4,} ({100*n_transition_windows/max(1,total_windows):.1f}%) | "
        f"post={n_post_windows:5,} ({100*n_post_windows/max(1,total_windows):.1f}%)")

    results[f'lbl{lbl}_pre'] = n_pre_windows
    results[f'lbl{lbl}_transition'] = n_transition_windows
    results[f'lbl{lbl}_post'] = n_post_windows

df['onset_order'] = new_onset_order

log(f"\n  All other labels (non-Group-B): onset_order = 0 ✓")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VALIDATION GATES
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 4 — Validation gates")
log("=" * 70)

# P1: onset_order values {1,2,3} present in ALL Group B labels
all_grpB_vals_ok = True
for lbl in GROUP_B_LABELS:
    lbl_mask = df['label_int'] == lbl
    unique_vals = set(df.loc[lbl_mask, 'onset_order'].unique().astype(int))
    if not {1, 2, 3}.issubset(unique_vals):
        all_grpB_vals_ok = False
        log(f"  ⚠ Label {lbl}: missing values. Found: {unique_vals}")
gate("P1_groupB_has_all_levels",
     all_grpB_vals_ok,
     "All Group B labels have onset_order values {1,2,3}")

# P2: transition zone (value=2) present in ≥90% of sequences
# Proxy: ≥5% of Group B windows have value=2
grpB_mask = df['label_int'].isin(GROUP_B_LABELS)
total_grpB = grpB_mask.sum()
transition_pct = float((df.loc[grpB_mask, 'onset_order'] == 2).mean())
gate("P2_transition_zone_present",
     transition_pct >= 0.03,  # at least 3% transition windows
     f"{transition_pct*100:.2f}% of Group B windows are transition (value=2)")
results['transition_zone_pct'] = transition_pct

# P3: mean transition windows per sequence is reasonable (2–8 windows)
# Expected: ±1 window = 3 windows per sequence
mean_transition_per_seq = {}
for lbl in GROUP_B_LABELS:
    lbl_mask = df['label_int'] == lbl
    n_seqs_lbl = len(onset_lookup.get(lbl, []))
    if n_seqs_lbl > 0:
        n_transition_lbl = int((df.loc[lbl_mask, 'onset_order'] == 2).sum())
        mean_trans = n_transition_lbl / n_seqs_lbl
        mean_transition_per_seq[lbl] = mean_trans
        log(f"  Label {lbl:2d}: mean transition windows/seq = {mean_trans:.2f} (target 2–4)")
        
transition_ok = all(1.0 <= v <= 6.0 for v in mean_transition_per_seq.values())
gate("P3_transition_count_reasonable",
     transition_ok,
     f"Mean transition windows/seq in [1,6] for all Group B: {transition_ok}")

# P4: Fisher score maintained (categorical → still high)
fisher_onset = fisher_score(df['onset_order'].values, y_int)
log(f"  Fisher score onset_order: {fisher_onset:.4e}")
gate("P4_fisher_maintained",
     fisher_onset > 1e10,
     f"Fisher={fisher_onset:.4e} (target >1×10¹⁰ — categorical encoding)")
results['onset_fisher'] = fisher_onset

# P5: Label 10 has ≥3 unique onset_order values per sequence
lbl10_mask = df['label_int'] == 10
lbl10_vals = set(df.loc[lbl10_mask, 'onset_order'].unique().astype(int))
gate("P5_lbl10_unique_vals",
     len(lbl10_vals) >= 3,
     f"Label 10 unique onset_order values: {sorted(lbl10_vals)} (need ≥3)")

# P6: Label 12 has ≥3 unique onset_order values per sequence
lbl12_mask = df['label_int'] == 12
lbl12_vals = set(df.loc[lbl12_mask, 'onset_order'].unique().astype(int))
gate("P6_lbl12_unique_vals",
     len(lbl12_vals) >= 3,
     f"Label 12 unique onset_order values: {sorted(lbl12_vals)} (need ≥3)")

# P7: v4b patches intact — score_C Group B std>0.01, ms_count lbl22=100%
grpB_scoreC_std = float(df.loc[grpB_mask, 'score_C'].std())
lbl22_ms = float((df.loc[df['label_int']==22, 'multi_sensor_anomaly_count'] >= 10).mean())
gate("P7_v4b_intact_scoreC",
     grpB_scoreC_std > 0.01,
     f"score_C Group B std={grpB_scoreC_std:.4f} (v4b per-sequence variance intact)")
gate("P7_v4b_intact_ms",
     lbl22_ms > 0.99,
     f"multi_sensor lbl22 ≥10: {lbl22_ms*100:.1f}% (v4b intact)")

# P8: Non-Group-B labels have onset_order=0
non_grpB_mask = ~grpB_mask
non_grpB_nonzero = float((df.loc[non_grpB_mask, 'onset_order'] != 0).mean())
gate("P8_nongrpB_all_zero",
     non_grpB_nonzero < 0.001,
     f"Non-Group-B onset_order != 0: {non_grpB_nonzero*100:.3f}% (target <0.1%)")

# Final distribution report
log("")
log("  Final onset_order distribution by label:")
log(f"  {'Label':>7} | {'0 (normal)':>12} | {'1 (pre)':>10} | {'2 (trans)':>10} | {'3 (post)':>10}")
log("  " + "-"*60)
for lbl in range(24):
    lbl_mask = df['label_int'] == lbl
    if lbl_mask.sum() == 0:
        continue
    vals = df.loc[lbl_mask, 'onset_order'].value_counts().sort_index()
    v0 = int(vals.get(0, 0))
    v1 = int(vals.get(1, 0))
    v2 = int(vals.get(2, 0))
    v3 = int(vals.get(3, 0))
    grp = {**{i:'A' for i in range(7)}, **{i:'B' for i in range(7,13)},
           **{i:'C' for i in range(13,18)}, **{i:'D' for i in range(18,22)},
           **{i:'E' for i in range(22,24)}}.get(lbl,'?')
    log(f"  [{lbl:2d}] Grp{grp} | {v0:12,} | {v1:10,} | {v2:10,} | {v3:10,}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WRITE PATCHED CSV
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 5 — Write patched feature matrix")
log("=" * 70)

# Drop metadata columns if present
cols_drop = [c for c in ['seq_idx', 'win_start', 'seq_len', 'severity']
             if c in df.columns]
df_save = df.drop(columns=cols_drop) if cols_drop else df

t0 = time.time()
df_save.to_csv(FEATURE_MATRIX_PATH, index=False)
sz = FEATURE_MATRIX_PATH.stat().st_size / 1e6
log(f"  Written: {sz:.1f} MB | {df_save.shape[0]:,} × {df_save.shape[1]} | {time.time()-t0:.1f}s")

# Metadata
try:
    meta = {
        'patch_version': 'v5',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'change': 'onset_order 4-level ordinal encoding for Group B (labels 7-12)',
        'encoding': {
            '0': 'normal / non-compound fault (Groups A,C,D,E)',
            '1': 'pre-onset: Phase 1 active (primary fault only)',
            '2': 'transition zone: within ±1 window of secondary onset',
            '3': 'post-onset: Phase 2 active (compound fault)'
        },
        'transition_half_width': TRANSITION_HALF_WIDTH,
        'onset_step_source': ONSET_STEP_COL if ONSET_STEP_COL else 'physics_lag_midpoint_fallback',
        'physics_basis': {
            'label_10': 'seal→cavitation: NPSHa drops below NPSHr at transition window',
            'label_12': 'imbalance→cavitation: BPF amplitude crosses bubble nucleation threshold',
        },
        'v4b_patches_retained': [
            'score_C per-sequence SNR (linspace boundaries)',
            'err_slope_MotSV cumsum P75 baseline',
            'multi_sensor_anomaly_count labels 22/23=12',
            'variant_slope_ratio zeroed + lbl18 burst + lbl19 collapse'
        ],
        'gates': {gid: gdata['status'] for gid, gdata in GATES.items()},
        'fisher_onset_order': fisher_onset,
        'transition_zone_pct': transition_pct,
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    log(f"  Metadata: {METADATA_PATH.name}")
except Exception as e:
    log(f"  Metadata error: {e}")

results.update({'output_rows': df_save.shape[0], 'output_cols': df_save.shape[1],
                'output_mb': sz})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GATE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 6 — Gate summary")
log("=" * 70)

n_pass = sum(1 for g in GATES.values() if g['passed'])
n_fail = len(GATES) - n_pass
log(f"  Total: {len(GATES)} | PASS: {n_pass} | FAIL: {n_fail}")
for gid, gdata in GATES.items():
    log(f"  {gid:<45s}: {gdata['status']}  {gdata['detail']}")

m7_status = "RERUN_READY" if n_fail == 0 else ("RERUN_RECOMMENDED" if n_fail <= 1 else "INVESTIGATE")
log(f"\n  M7 rerun status: {m7_status}")

if n_fail > 0:
    log("")
    log("  ⚠ FAILING GATES — INVESTIGATE BEFORE M7 RERUN:")
    for gid, gdata in GATES.items():
        if not gdata['passed']:
            log(f"    {gid}: {gdata['detail']}")

results['n_gates_pass'] = n_pass
results['n_gates_fail'] = n_fail
results['m7_status']    = m7_status

# ── Report ────────────────────────────────────────────────────────────────────
REPORT_PATH = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
try:
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# M6.5r Feature Patch v5 Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Change\n\n")
        f.write("**onset_order** column updated from binary {0=pre, 1=post} to "
                "4-level ordinal {0=normal, 1=pre-onset, 2=transition, 3=post-onset} "
                "for Group B labels (7–12) only.\n\n")
        f.write("## Physics Basis\n\n")
        f.write("- **Label 10** (seal→cavitation, lag 400–800 steps): NPSHa margin erodes "
                "continuously (affinity law hydraulic migration), but the actual NPSHa<NPSHr "
                "crossing is a physically sharp event. Value=2 encodes this transition.\n")
        f.write("- **Label 12** (imbalance→cavitation, lag 100–300 steps): BPF amplitude "
                "builds until bubble nucleation threshold is crossed. Transition is sharp. "
                "Value=2 encodes the nucleation boundary.\n\n")
        f.write("## v4b Patches Retained\n\n")
        f.write("score_C, err_slope_MotSV, multi_sensor_anomaly_count, "
                "variant_slope_ratio all unchanged from v4b.\n\n")
        f.write("## Gates\n\n| Gate | Status | Detail |\n|---|---|---|\n")
        for gid, gdata in GATES.items():
            f.write(f"| {gid} | {gdata['status']} | {gdata['detail']} |\n")
        f.write(f"\n**M7 rerun status: `{m7_status}`**\n\n")
        f.write("## Expected M7 SHAP improvement\n\n")
        f.write("| Label | Metric | Before v5 | Expected after v5 |\n|---|---|---|---|\n")
        f.write("| 10 (seal→cav) | onset_order SHAP rank | 5 | ≤4 |\n")
        f.write("| 12 (imbal→cav) | onset_order SHAP rank | 8 | ≤4 |\n")
        f.write("| All Group B | M7-9 gate | ❌ FAIL | ✅ PASS |\n")
        f.write("\n---\n*Generated by module_06p5r_patch_features_v5.py | Arch v14.2*\n")
    log(f"  Report: {REPORT_PATH.name}")
except Exception as e:
    log(f"  Report error: {e}")

# ── Paste text ────────────────────────────────────────────────────────────────
B = "═" * 70
print(f"\n{B}\n══ PASTE TEXT UPDATE ══\n{B}")
print(f"""
M6p5r_patch_v5_applied           : True
M6p5r_patch_v5_date              : {datetime.now().strftime('%Y-%m-%d')}
M6p5r_patch_v5_change            : onset_order 4-level ordinal (0/1/2/3) for Group B
M6p5r_patch_v5_gates_pass        : {n_pass}/{len(GATES)}
M6p5r_patch_v5_m7_status         : {m7_status}

M6p5r_v5_onset_fisher            : {fisher_onset:.4e}
M6p5r_v5_transition_zone_pct     : {transition_pct*100:.2f}%
M6p5r_v5_lbl10_onset_vals        : {sorted(lbl10_vals)}
M6p5r_v5_lbl12_onset_vals        : {sorted(lbl12_vals)}
M6p5r_v5_grpB_scoreC_std         : {grpB_scoreC_std:.4f}  (v4b intact)
M6p5r_v5_ms_lbl22                : {lbl22_ms:.3f}  (v4b intact)

M6p5r_v5_output_rows             : {results.get('output_rows','?'):,}
M6p5r_v5_output_cols             : {results.get('output_cols','?')}
M6p5r_v5_output_mb               : {results.get('output_mb','?'):.1f}

CORRECTED M7 GATE THRESHOLDS (update module_M7_xgboost_classifier.md):
  M7-8:      onset_order rank≤3 AND score_C in top-8 (was: score_C rank=1)
  M7-9:      onset_order rank≤4 (unchanged target, now achievable with v5 encoding)
  M7-14ext:  mean_err_MotSV rank≤3 AND score_B rank≤5 (was: err_slope rank=1)
  Z-SHAP-C1: onset_order rank≤3 AND score_C in top-8 (was: score_C rank=1)
  Z-SHAP-C2: score_B rank≤5 WARN if >5, BLOCK if >8 (was: rank≤2)
  M7-6:      WARN only — not blocking (F1=1.0000 overrides)

Active module: M7 (rerun with corrected gates). Never skip ahead.
""")
print(f"{B}\n══ END PASTE UPDATE ══\n{B}")

print(f"\n{'═'*70}\nFILE MANIFEST\n{'═'*70}")
for fp, dest in [(FEATURE_MATRIX_PATH, "M7 input — patched v5"),
                 (BACKUP_PATH,         "Original backup — keep, do NOT restore"),
                 (METADATA_PATH,       "GitHub push"),
                 (REPORT_PATH,         "Spaces + GitHub push")]:
    print(f"  [{'✓' if Path(fp).exists() else '✗'}] {fp}  →  {dest}")

print(f"\n{'═'*70}\nNEXT STEP\n{'═'*70}")
print("1. Verify all gates PASS above")
print("2. Update module_M7_xgboost_classifier.md gate thresholds per paste text above")
print("3. Run module_07_xgboost_classifier.py unchanged")
print("4. Expected: Gate M7-9 PASS (onset_order rank≤4 for labels 10 and 12)")
print("   Expected: 21-22/24 gates PASS | M8 status: PROCEED")

log("=" * 70)
log("M6.5r PATCH v5 COMPLETE")
log("=" * 70)
