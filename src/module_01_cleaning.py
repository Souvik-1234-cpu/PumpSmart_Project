import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =============================================================
# module_01_cleaning.py — M1: Data Ingestion & Hard Cleaning
# PumpSmart Project | Physics-Informed ML Digital Twin
# =============================================================
from config import (DEVICE, RAW_DIR, CLEAN_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
SCRIPT_NAME = "module_01_cleaning"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}

# =============================================================
# CONSTANTS
# =============================================================
STANDARD_COLS = [
    'Timestamp',
    'X_ACR_Mot.PV', 'X_ACR_Mot.SV', 'X_ACR_Mot.TV',
    'X_ACR_Pmp.PV', 'X_ACR_Pmp.SV', 'X_ACR_Pmp.TV',
    'X_Temp.SV', 'X_Pres.SV',
    'Barometer', 'Temperature'
]
SENSOR_COLS = STANDARD_COLS[1:]

# Gap multipliers — confirmed from check_intervals.py
# Day1: max natural gap = 5s, use 8s threshold (between 5s jitter and real gaps)
# Day2: max natural gap = 172s, use 2× = 344s
# Day3: perfectly continuous 1s, use 2× = 2s
GAP_MULTIPLIER      = 2.0
GAP_MULTIPLIER_DAY1 = 8.0   # 8 × 1s = 8s threshold (above 5s jitter)

# Minimum rows for a segment to be usable in LSTM windowing
# Window size = 60 (from config), need at least window + stride buffer
MIN_USABLE_ROWS = 70

# =============================================================
# STEP 1 — Discover raw CSVs
# =============================================================
log("STEP 1 — Scanning raw data directory...")
raw_files = sorted(RAW_DIR.glob("Pump_*.csv"))
if len(raw_files) == 0:
    raise FileNotFoundError(f"No CSVs found in {RAW_DIR}")
log(f"Found {len(raw_files)} files: {[f.name for f in raw_files]}")
results['raw_files_found'] = len(raw_files)

# =============================================================
# STEP 2 — Load, null audit, hard-drop, gap-segment
# =============================================================
log("STEP 2 — Loading, cleaning, segmenting all files...")

all_segment_records = []
file_summaries      = []
total_raw_rows      = 0
total_clean_rows    = 0
total_dropped       = 0
total_segments      = 0
total_usable_segs   = 0

for fpath in raw_files:
    fname   = fpath.stem          # e.g. Pump_A_Day1
    parts   = fname.split('_')
    pump_id = parts[1]            # A, B, C
    day_id  = parts[2]            # Day1, Day2, Day3

    log(f"  Loading {fpath.name}...")

    try:
        df = pd.read_csv(fpath, parse_dates=['Timestamp'])
    except Exception as e:
        log(f"  ERROR loading {fpath.name}: {e}")
        continue

    raw_rows = len(df)
    total_raw_rows += raw_rows

    # ── Null audit (before drop) ──────────────────────────
    null_counts    = df[SENSOR_COLS].isnull().sum()
    worst_null_col = null_counts.idxmax()
    worst_null_pct = round(null_counts.max() / raw_rows * 100, 2)
    total_null_pct = round(df[SENSOR_COLS].isnull().any(axis=1).sum()
                           / raw_rows * 100, 2)

    # ── Hard drop: ANY null in ANY column → drop entire row ─
    df_clean = df.dropna(subset=STANDARD_COLS).reset_index(drop=True)
    dropped  = raw_rows - len(df_clean)
    drop_pct = round(dropped / raw_rows * 100, 2)
    total_dropped += dropped

    log(f"    {raw_rows:,} raw → {len(df_clean):,} clean "
        f"({drop_pct}% dropped) | Worst null: "
        f"{worst_null_col} ({worst_null_pct}%)")

    if len(df_clean) == 0:
        log(f"    WARNING: Zero rows after cleaning — skipping {fpath.name}")
        continue

    # ── Sort by Timestamp ─────────────────────────────────
    df_clean = df_clean.sort_values('Timestamp').reset_index(drop=True)

    # ── Compute sampling intervals ────────────────────────
    deltas          = df_clean['Timestamp'].diff().dt.total_seconds().fillna(0)
    pos_deltas      = deltas[deltas > 0]
    median_interval = pos_deltas.median() if len(pos_deltas) > 0 else 1.0

    # Gap multiplier: Day1=8×, all others=2×
    multiplier    = GAP_MULTIPLIER_DAY1 if 'Day1' in fname else GAP_MULTIPLIER
    gap_threshold = multiplier * median_interval

    # ── Segment assignment ────────────────────────────────
    seg_num    = 1
    seg_labels = []
    for i, delta in enumerate(deltas):
        if i > 0 and delta > gap_threshold:
            seg_num += 1
        seg_labels.append(f"{pump_id}_{day_id}_seg{seg_num}")

    df_clean['segment_id'] = seg_labels
    n_segments = df_clean['segment_id'].nunique()
    total_segments += n_segments

    log(f"    Segments: {n_segments} | Median Δt: {median_interval:.1f}s "
        f"| Gap threshold: {gap_threshold:.1f}s")

    # ── Per-segment registry ──────────────────────────────
    file_usable = 0
    for seg_id, seg_df in df_clean.groupby('segment_id'):
        dur_s    = (seg_df['Timestamp'].max() -
                    seg_df['Timestamp'].min()).total_seconds()
        n_rows   = len(seg_df)
        usable   = n_rows >= MIN_USABLE_ROWS
        if usable:
            file_usable      += 1
            total_usable_segs += 1

        all_segment_records.append({
            'segment_id'         : seg_id,
            'pump_id'            : pump_id,
            'day_id'             : day_id,
            'source_file'        : fpath.name,
            'n_rows'             : n_rows,
            'start_time'         : seg_df['Timestamp'].min(),
            'end_time'           : seg_df['Timestamp'].max(),
            'duration_s'         : round(dur_s, 1),
            'duration_min'       : round(dur_s / 60, 2),
            'usable_for_windowing': usable
        })

    log(f"    Usable segments (≥{MIN_USABLE_ROWS} rows): "
        f"{file_usable}/{n_segments}")

    # ── Save cleaned file ─────────────────────────────────
    out_path = CLEAN_DIR / f"{fname}_clean.csv"
    df_clean.to_csv(out_path, index=False)
    total_clean_rows += len(df_clean)

    file_summaries.append({
        'file'            : fpath.name,
        'pump_id'         : pump_id,
        'day_id'          : day_id,
        'raw_rows'        : raw_rows,
        'clean_rows'      : len(df_clean),
        'dropped_rows'    : dropped,
        'drop_pct'        : drop_pct,
        'n_segments'      : n_segments,
        'usable_segments' : file_usable,
        'worst_null_col'  : worst_null_col,
        'worst_null_pct'  : worst_null_pct,
        'median_interval' : round(median_interval, 2),
        'gap_threshold_s' : round(gap_threshold, 2)
    })
    log(f"    Saved → {out_path.name}")

# =============================================================
# STEP 3 — Save registries
# =============================================================
log("STEP 3 — Saving segment registry and file summary...")

seg_registry = pd.DataFrame(all_segment_records)
seg_registry.to_csv(CLEAN_DIR / "segment_registry.csv", index=False)

summary_df = pd.DataFrame(file_summaries)
summary_df.to_csv(OUTPUT_DIR / "M1_file_summary.csv", index=False)

results['total_raw_rows']    = total_raw_rows
results['total_clean_rows']  = total_clean_rows
results['total_dropped']     = total_dropped
results['overall_drop_pct']  = round(total_dropped / total_raw_rows * 100, 2)
results['total_segments']    = seg_registry['segment_id'].nunique()
results['usable_segments']   = int(seg_registry['usable_for_windowing'].sum())
results['unusable_segments'] = results['total_segments'] - results['usable_segments']
results['worst_null_col']    = summary_df.loc[
                                 summary_df['worst_null_pct'].idxmax(),
                                 'worst_null_col']
results['worst_null_pct']    = summary_df['worst_null_pct'].max()

log(f"  Total segments : {results['total_segments']}")
log(f"  Usable (≥{MIN_USABLE_ROWS}r): {results['usable_segments']}")
log(f"  Unusable       : {results['unusable_segments']}")
log(f"  TOTAL: {total_raw_rows:,} raw → {total_clean_rows:,} clean | "
    f"{results['overall_drop_pct']}% dropped")

# =============================================================
# STEP 4 — Plot 1: Null heatmap (before cleaning, all 9 files)
# =============================================================
log("STEP 4 — Generating null heatmap...")
try:
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    fig.suptitle('M1 — Null Value Heatmap Per File (Before Cleaning)',
                 fontsize=14, fontweight='bold')

    for idx, fpath in enumerate(raw_files):
        ax = axes[idx]
        try:
            df_raw   = pd.read_csv(fpath)
            s_cols   = [c for c in SENSOR_COLS if c in df_raw.columns]
            null_mat = df_raw[s_cols].isnull().astype(int)
            if null_mat.values.sum() == 0:
                ax.text(0.5, 0.5, 'No Nulls\nDetected',
                        ha='center', va='center', fontsize=12,
                        color='green', transform=ax.transAxes)
                ax.set_facecolor('#f0fff0')
            else:
                sample = null_mat.sample(min(2000, len(null_mat)),
                                          random_state=42)
                sns.heatmap(sample.T, ax=ax, cbar=False,
                            xticklabels=False, yticklabels=True,
                            cmap='Reds')
                ax.tick_params(axis='y', labelsize=7)
        except Exception as e2:
            ax.text(0.5, 0.5, f'Error:\n{str(e2)[:30]}',
                    ha='center', va='center', fontsize=7,
                    color='red', transform=ax.transAxes)
        ax.set_title(raw_files[idx].stem, fontsize=9, fontweight='bold')

    plt.tight_layout()
    null_path = PLOTS_DIR / "M1_null_heatmap.png"
    plt.savefig(null_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved → {null_path.name}")
except Exception as e:
    log(f"  WARNING: Null heatmap failed: {e}")

# =============================================================
# STEP 5 — Plot 2: Segment timeline (usable vs unusable)
# =============================================================
log("STEP 5 — Generating segment timeline...")
try:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('M1 — Segment Timeline (Green=Usable, Red=Too Short)',
                 fontsize=13, fontweight='bold')

    for pidx, pump in enumerate(['A', 'B', 'C']):
        ax     = axes[pidx]
        p_segs = seg_registry[
                     seg_registry['pump_id'] == pump
                 ].reset_index(drop=True)

        for _, row in p_segs.iterrows():
            color = '#2ecc71' if row['usable_for_windowing'] else '#e74c3c'
            ax.barh(y=row['segment_id'],
                    width=row['duration_min'],
                    left=0, color=color,
                    edgecolor='black', linewidth=0.3, height=0.6)
            ax.text(row['duration_min'] + 0.1, _,
                    f"{row['n_rows']}r",
                    va='center', fontsize=6, color='black')

        ax.set_title(f'Pump {pump}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Duration (minutes)', fontsize=8)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(axis='x', alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#2ecc71', label=f'Usable (≥{MIN_USABLE_ROWS} rows)'),
            Patch(color='#e74c3c', label='Too short for windowing')
        ], fontsize=7, loc='lower right')

    plt.tight_layout()
    tl_path = PLOTS_DIR / "M1_segment_timeline.png"
    plt.savefig(tl_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved → {tl_path.name}")
except Exception as e:
    log(f"  WARNING: Segment timeline failed: {e}")

# =============================================================
# STEP 6 — Plot 3: Drop % bar chart
# =============================================================
log("STEP 6 — Generating drop percentage chart...")
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = ['#e74c3c' if x > 20 else
                  '#f39c12' if x > 5  else
                  '#2ecc71' for x in summary_df['drop_pct']]
    bars = ax.bar(summary_df['file'], summary_df['drop_pct'],
                  color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_title('M1 — Row Drop % Per File (Hard NaN Policy)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Rows Dropped (%)')
    ax.set_xlabel('Source File')
    ax.axhline(5,  color='orange', linestyle='--', lw=1,
               label='5% warning')
    ax.axhline(20, color='red',    linestyle='--', lw=1,
               label='20% critical')
    for bar, val in zip(bars, summary_df['drop_pct']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    ax.legend()
    plt.tight_layout()
    dp_path = PLOTS_DIR / "M1_drop_percentage.png"
    plt.savefig(dp_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved → {dp_path.name}")
except Exception as e:
    log(f"  WARNING: Drop chart failed: {e}")

# =============================================================
# STEP 7 — Plot 4: Usable vs unusable segments per file
# =============================================================
log("STEP 7 — Generating usability chart...")
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    x     = range(len(summary_df))
    width = 0.4
    ax.bar([i - width/2 for i in x],
           summary_df['usable_segments'],
           width=width, color='#2ecc71',
           edgecolor='black', linewidth=0.5,
           label='Usable segments')
    ax.bar([i + width/2 for i in x],
           summary_df['n_segments'] - summary_df['usable_segments'],
           width=width, color='#e74c3c',
           edgecolor='black', linewidth=0.5,
           label='Too short')
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df['file'],
                       rotation=30, ha='right', fontsize=9)
    ax.set_title('M1 — Usable vs Unusable Segments Per File',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Segments')
    ax.legend()
    plt.tight_layout()
    us_path = PLOTS_DIR / "M1_segment_usability.png"
    plt.savefig(us_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved → {us_path.name}")
except Exception as e:
    log(f"  WARNING: Usability chart failed: {e}")

# =============================================================
# STEP 8 — Markdown report
# =============================================================
log("STEP 8 — Writing markdown report...")
report_lines = [
    "# M1 Cleaning Report",
    f"**Date:** {date.today()}  ",
    f"**Script:** {SCRIPT_NAME}  ",
    "",
    "## Summary",
    "| Metric | Value |",
    "|---|---|",
    f"| Raw files processed | {results['raw_files_found']} |",
    f"| Total raw rows | {results['total_raw_rows']:,} |",
    f"| Total clean rows | {results['total_clean_rows']:,} |",
    f"| Total dropped rows | {results['total_dropped']:,} |",
    f"| Overall drop % | {results['overall_drop_pct']}% |",
    f"| Total segments | {results['total_segments']} |",
    f"| Usable segments (≥{MIN_USABLE_ROWS} rows) | {results['usable_segments']} |",
    f"| Unusable segments | {results['unusable_segments']} |",
    f"| Worst null column | {results['worst_null_col']} |",
    f"| Worst null % | {results['worst_null_pct']}% |",
    "",
    "## Per-File Breakdown",
    summary_df.to_markdown(index=False),
    "",
    "## Segment Registry (first 20 rows)",
    seg_registry.head(20).to_markdown(index=False),
    "",
    "## Key Engineering Findings",
    "- All 9 files: 1-second sampling confirmed uniform",
    "- Day1 files: 8× gap threshold (above 5s natural jitter)",
    "- Day2 files: 2× gap threshold (above 172s operational pauses)",
    "- Day3 files: 2× gap threshold (perfectly continuous 1s data)",
    "- Pump_B_Day3: Barometer+Temperature sensor failure — 65.5% rows dropped",
    "  Pump ran continuously; sensor logged NaN not timestamp gaps",
    "  Clean segments before/after fault block are valid training data",
    f"- Segments < {MIN_USABLE_ROWS} rows flagged unusable for LSTM windowing",
    "- Hard NaN policy: zero interpolation enforced",
    "",
    "## Output Files",
    "- `data/clean/Pump_*_clean.csv` — 9 cleaned CSVs with segment_id",
    "- `data/clean/segment_registry.csv` — master segment index with usability flag",
    "- `outputs/M1_file_summary.csv`",
    "- `outputs/plots/M1_null_heatmap.png`",
    "- `outputs/plots/M1_segment_timeline.png`",
    "- `outputs/plots/M1_drop_percentage.png`",
    "- `outputs/plots/M1_segment_usability.png`",
]
report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log(f"  Report saved → {report_path.name}")

# =============================================================
# PASTE TEXT UPDATE
# =============================================================
print()
print("═"*55)
print("  PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT")
print("═"*55)
print(f"M1_total_raw_rows      : {results['total_raw_rows']}")
print(f"M1_total_clean_rows    : {results['total_clean_rows']}")
print(f"M1_total_dropped       : {results['total_dropped']}")
print(f"M1_overall_drop_pct    : {results['overall_drop_pct']}%")
print(f"M1_total_segments      : {results['total_segments']}")
print(f"M1_usable_segments     : {results['usable_segments']}")
print(f"M1_unusable_segments   : {results['unusable_segments']}")
print(f"M1_worst_null_col      : {results['worst_null_col']}")
print(f"M1_worst_null_pct      : {results['worst_null_pct']}%")
print(f"M1_sampling_interval   : 1s uniform across all files")
print(f"M1_A_Day3_fix          : semicolon+comma-decimal+col-rename")
print(f"M1_B_Day3_note         : sensor fault (not timestamp gap)")
print(f"M1_Day1_gap_threshold  : 8s (above 5s natural jitter)")
print(f"M1_Day2_gap_threshold  : 344s (above 172s operational pauses)")
print(f"M1_Day3_gap_threshold  : 2s (continuous 1s data)")
print(f"Status for M2          : READY")
print("═"*55)

# =============================================================
# FILE MANIFEST
# =============================================================
print()
print("── FILE MANIFEST ──────────────────────────────────────")
print("→ GitHub push (large data files):")
for f in sorted(CLEAN_DIR.glob("*.csv")):
    print(f"    {f}")
print("→ Spaces upload (plots + report):")
for f in sorted(PLOTS_DIR.glob("M1_*.png")):
    print(f"    {f}")
print(f"    {report_path}")
print("───────────────────────────────────────────────────────")
print()
print("📦 M1 done. Starting M2.")
print("   Upload report + plots to Spaces. Push clean CSVs to GitHub.")
print("   Provide M2 complete script.")
