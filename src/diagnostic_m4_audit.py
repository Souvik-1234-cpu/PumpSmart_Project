# =============================================================================
# diagnostic_m4_audit.py — PumpSmart M4 Pre-Fix Data Quality Audit
# Run BEFORE making any changes to module_04_lstm_ae_baseline.py
# Purpose: Identify root cause of v3→v5 regression
# =============================================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, CLEAN_DIR, NORM_DIR, MODEL_DIR, OUTPUT_DIR)
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from collections import Counter

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
log("=" * 65)
log("TEST 1 — M3 Data Distribution Audit")
log("=" * 65)

df = pd.read_csv(NORM_DIR / "normalised_data.csv", parse_dates=['Timestamp'])
norm_cols = [c for c in df.columns if '_norm' in c]

print("\n── Describe (key percentiles) ──")
print(df[norm_cols].describe(percentiles=[.01, .05, .25, .75, .95, .99, .999]).T.to_string())

print("\n── % of rows ABOVE 1.0 per channel ──")
for c in norm_cols:
    pct = (df[c] > 1.0).mean() * 100
    flag = " ⚠️ " if pct > 55 or pct < 40 else "  ✅"
    print(f"  {flag} {c}: {pct:.2f}% above 1.0")

print("\n── EXTREME OUTLIERS — values above 5× normalised ──")
any_extreme = False
for c in norm_cols:
    n_5  = (df[c] > 5.0).sum()
    n_10 = (df[c] > 10.0).sum()
    n_20 = (df[c] > 20.0).sum()
    if n_5 > 0:
        any_extreme = True
        print(f"  ⚠️  {c}: >5.0 → {n_5} rows | >10.0 → {n_10} rows | "
              f">20.0 → {n_20} rows | MAX = {df[c].max():.4f}")
if not any_extreme:
    print("  ✅ No values above 5.0 — distribution clean")

print("\n── Negative values (flash evaporative cooling check) ──")
for c in norm_cols:
    n_neg = (df[c] < 0).sum()
    if n_neg > 0:
        print(f"  ℹ️  {c}: {n_neg} negative rows | MIN = {df[c].min():.6f}")

print("\n── Per-cluster distribution check ──")
if 'cluster_label' in df.columns:
    for cluster in sorted(df['cluster_label'].unique()):
        sub = df[df['cluster_label'] == cluster]
        print(f"\n  Cluster {cluster} ({len(sub):,} rows):")
        for c in norm_cols:
            print(f"    {c}: mean={sub[c].mean():.4f} "
                  f"std={sub[c].std():.4f} "
                  f"max={sub[c].max():.4f}")
else:
    log("  cluster_label column not found — skipping cluster breakdown")

# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 65)
log("TEST 2 — Random Seed Reproducibility Audit")
log("=" * 65)

SEED = 42
rng1 = np.random.default_rng(SEED)
rng2 = np.random.default_rng(SEED)
idx1 = rng1.permutation(10973)
idx2 = rng2.permutation(10973)
seed_ok = np.array_equal(idx1, idx2)
print(f"\n  np.random.default_rng({SEED}) reproducible: "
      f"{'✅ YES' if seed_ok else '❌ NO — NumPy version issue'}")
print(f"  NumPy version: {np.__version__}")
print(f"  PyTorch version: {torch.__version__}")

# Simulate both runs to check if val compositions differ
N_TOTAL = 10973
VAL_SPLIT = 0.15
split = int(N_TOTAL * (1 - VAL_SPLIT))

rng_run1 = np.random.default_rng(SEED)
idx_run1 = rng_run1.permutation(N_TOTAL)
val_run1 = set(idx_run1[split:].tolist())

rng_run2 = np.random.default_rng(SEED)
idx_run2 = rng_run2.permutation(N_TOTAL)
val_run2 = set(idx_run2[split:].tolist())

overlap = len(val_run1 & val_run2)
print(f"\n  Val set overlap between 2 identical-seed runs: "
      f"{overlap}/{len(val_run1)} windows "
      f"({'✅ 100% identical' if overlap == len(val_run1) else '❌ DIFFERENT — seed not controlling split'})")

# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 65)
log("TEST 3 — Window-Level Outlier Source Identification")
log("=" * 65)

# Rebuild windows exactly as M4 does
CHANNELS = [
    'X_ACR_Mot.PV_norm', 'X_ACR_Mot.SV_norm', 'X_ACR_Mot.TV_norm',
    'X_ACR_Pmp.PV_norm', 'X_ACR_Pmp.SV_norm', 'X_ACR_Pmp.TV_norm',
    'X_Temp.SV_norm',    'X_Pres.SV_norm'
]
WINDOW_SIZE = 50
STEP_SIZE   = 10

try:
    reg = pd.read_csv(CLEAN_DIR / "segment_registry.csv")
    warmup_map  = reg.set_index('segment_id')['warmup_rows'].to_dict()
    usable_segs = reg[reg['usable_for_windowing'] == True]['segment_id'].tolist()
except Exception as e:
    log(f"ERROR loading registry: {e}"); raise

windows, seg_ids, window_maxvals = [], [], []

for seg_id in usable_segs:
    seg_df  = df[df['segment_id'] == seg_id].copy()
    warmup  = int(warmup_map.get(seg_id, 300))
    seg_df  = seg_df.iloc[warmup:].reset_index(drop=True)
    if len(seg_df) < WINDOW_SIZE:
        continue
    sensor_data = seg_df[CHANNELS].values.astype(np.float32)
    if np.isnan(sensor_data).any():
        continue
    for i in range(0, len(seg_df) - WINDOW_SIZE + 1, STEP_SIZE):
        w = sensor_data[i : i + WINDOW_SIZE]
        if w.shape[0] == WINDOW_SIZE:
            windows.append(w)
            seg_ids.append(seg_id)
            window_maxvals.append(float(w.max()))

windows_arr    = np.array(windows, dtype=np.float32)
window_maxvals = np.array(window_maxvals)
log(f"  Rebuilt {len(windows_arr):,} windows for audit")

print("\n── Window max-value distribution ──")
for thresh in [2.0, 5.0, 10.0, 20.0, 50.0]:
    n = (window_maxvals > thresh).sum()
    pct = n / len(window_maxvals) * 100
    if n > 0:
        print(f"  Windows with any value >{thresh}: {n} ({pct:.3f}%)")

# Compute per-window mean MAE from raw values
# (proxy: std across time within window — high std = high reconstruction cost)
window_temporal_std = windows_arr.std(axis=1).mean(axis=1)  # shape (N,)
p95_std = np.percentile(window_temporal_std, 95)
p99_std = np.percentile(window_temporal_std, 99)

print(f"\n── Per-window temporal std distribution ──")
print(f"  Mean: {window_temporal_std.mean():.6f}")
print(f"  Std:  {window_temporal_std.std():.6f}")
print(f"  P95:  {p95_std:.6f}")
print(f"  P99:  {p99_std:.6f}")

# Find which segments own the high-variance windows
high_var_mask = window_temporal_std > p95_std
high_var_segs = [seg_ids[i] for i in np.where(high_var_mask)[0]]
seg_counts    = Counter(high_var_segs)

print(f"\n── Segments owning P95+ temporal-variance windows ──")
print(f"  Total high-variance windows: {high_var_mask.sum()}")
for seg, count in seg_counts.most_common(10):
    total_in_seg = seg_ids.count(seg)
    print(f"  {seg}: {count} high-var / {total_in_seg} total "
          f"({count/total_in_seg*100:.1f}%)")

# Find the actual worst windows
top10_idx = np.argsort(window_temporal_std)[-10:][::-1]
print(f"\n── Top 10 highest-variance windows ──")
print(f"  {'Rank':<5} {'Segment':<20} {'MaxVal':>8} {'TempStd':>10} {'Channel with max':>20}")
for rank, idx in enumerate(top10_idx, 1):
    w       = windows_arr[idx]
    max_val = float(w.max())
    max_ch  = CHANNELS[int(np.unravel_index(w.argmax(), w.shape)[1])]
    t_std   = float(window_temporal_std[idx])
    print(f"  {rank:<5} {seg_ids[idx]:<20} {max_val:>8.4f} {t_std:>10.6f} {max_ch:>20}")

# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 65)
log("TEST 4 — v3 vs v5 Distribution Change Forensics")
log("=" * 65)

# Compare key stats to v3 reference values from paste text
V3_REF = {
    'mean_mae':   0.049839,
    'std_mae':    0.198503,
    'p99':        0.518297,
    'threshold':  0.645347,
}
V5_OBS = {
    'mean_mae':   0.057390,
    'std_mae':    0.241876,
    'p99':        1.545694,
    'threshold':  1.545694,
}

print("\n  Metric          v3 Reference    v5 Observed     Delta")
print("  " + "-" * 60)
for k in V3_REF:
    v3  = V3_REF[k]
    v5  = V5_OBS[k]
    delta_pct = (v5 - v3) / v3 * 100
    flag = "⚠️ " if abs(delta_pct) > 20 else "✅"
    print(f"  {flag} {k:<16} {v3:>12.6f}  {v5:>12.6f}  {delta_pct:>+8.1f}%")

print("\n── Std/Mean ratio (healthy < 2.0) ──")
print(f"  v3: {V3_REF['std_mae'] / V3_REF['mean_mae']:.2f}×")
print(f"  v5: {V5_OBS['std_mae'] / V5_OBS['mean_mae']:.2f}×  "
      f"{'⚠️  Fat tail — outlier windows dominating P99' if V5_OBS['std_mae']/V5_OBS['mean_mae'] > 3 else '✅ Acceptable'}")

# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 65)
log("DIAGNOSTIC SUMMARY")
log("=" * 65)
print("""
  After reviewing output above, answer these 4 questions:

  Q1 [Test 1] Are there values >5× normalised in any channel?
     YES → Raw spikes entering windows. Fix = winsorize at 5σ before windowing.
     NO  → Distribution is clean, problem is elsewhere.

  Q2 [Test 2] Is val split 100% identical across runs?
     NO  → NumPy seed not stable. Fix = segment-level split mandatory.
     YES → Random split is reproducible, problem is data distribution.

  Q3 [Test 3] Do P95+ windows cluster in 1-2 segments?
     YES → Segment-level split removes outlier contamination from val set.
     NO  → Outliers are spread everywhere — need winsorization.

  Q4 [Test 4] Is Std/Mean ratio > 4×?
     YES → Confirmed fat-tail distribution from extreme spike windows.
     NO  → Std/Mean normal, threshold instability has different cause.
""")

log("Diagnostic complete. Paste full output back for targeted fix decision.")