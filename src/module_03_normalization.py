import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, CLEAN_DIR, NORM_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_NAME = "module_03_normalization"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
NORM_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}

# =============================================================
# SENSOR CLASSIFICATION
# Pressure  → P* = P / cluster_mean
# Vibration → a* = a / cluster_mean
# Temperature → ΔT* = (T - T_ambient) / (T_cluster_max - T_ambient)
# =============================================================
PRESSURE_COLS    = ['X_Pres.SV']
VIBRATION_COLS   = ['X_ACR_Mot.PV', 'X_ACR_Mot.SV',
                    'X_ACR_Pmp.PV', 'X_ACR_Pmp.SV']
TEMPERATURE_COLS = ['X_ACR_Mot.TV', 'X_ACR_Pmp.TV', 'X_Temp.SV']
ALL_SENSOR_COLS  = PRESSURE_COLS + VIBRATION_COLS + TEMPERATURE_COLS

# Environmental cols — carried through but NOT normalized
ENV_COLS         = ['Barometer', 'Temperature']
AMBIENT_COL      = 'Temperature'   # T_ambient per row

# =============================================================
# STEP 1 — Load cluster bounds from M2
# =============================================================
log("STEP 1 — Loading M2 cluster bounds...")
try:
    bounds_df = pd.read_csv(OUTPUT_DIR / "M2_cluster_bounds.csv")
    log(f"  Loaded: {len(bounds_df)} clusters")
    log(f"\n  Cluster modes:")
    for _, row in bounds_df.iterrows():
        log(f"    C{int(row['cluster_id'])}: {row['operating_mode']} "
            f"({int(row['n_rows'])} rows)")
except Exception as e:
    raise RuntimeError(f"Cannot load M2_cluster_bounds.csv: {e}")

# Build normalization config dict — one entry per cluster per sensor
norm_config = {}
for _, row in bounds_df.iterrows():
    cid  = int(row['cluster_id'])
    mode = row['operating_mode']
    norm_config[cid] = {'operating_mode': mode}

    for col in ALL_SENSOR_COLS:
        norm_config[cid][col] = {
            'mean'  : float(row[f'{col}_mean']),
            'std'   : float(row[f'{col}_std']),
            'p2_5'  : float(row[f'{col}_p2_5']),
            'p97_5' : float(row[f'{col}_p97_5']),
            'max'   : float(row[f'{col}_max']),
            'min'   : float(row[f'{col}_min']),
        }

    # T_ambient baseline: use cluster mean of ambient Temperature
    if f'{AMBIENT_COL}_mean' in row:
        norm_config[cid]['T_ambient_mean'] = float(
            row[f'{AMBIENT_COL}_mean']
        )
    else:
        norm_config[cid]['T_ambient_mean'] = 20.0  # fallback

log(f"  Normalization config built for {len(norm_config)} clusters")

# =============================================================
# STEP 2 — Load M2 labelled data (has cluster_id per row)
# =============================================================
log("STEP 2 — Loading M2 labelled dataset...")
try:
    labelled_df = pd.read_csv(
        OUTPUT_DIR / "M2_labelled_data.csv",
        parse_dates=['Timestamp']
    )
    log(f"  Loaded: {len(labelled_df):,} rows | "
        f"{labelled_df['segment_id'].nunique()} segments")
except Exception as e:
    raise RuntimeError(f"Cannot load M2_labelled_data.csv: {e}")

# =============================================================
# STEP 3 — Apply dimensionless normalization per row
# =============================================================
log("STEP 3 — Applying dimensionless normalization...")

df = labelled_df.copy()

# Output normalized column names
NORM_PRESSURE_COLS    = [f'{c}_norm' for c in PRESSURE_COLS]
NORM_VIBRATION_COLS   = [f'{c}_norm' for c in VIBRATION_COLS]
NORM_TEMPERATURE_COLS = [f'{c}_norm' for c in TEMPERATURE_COLS]

# Initialise norm columns
for col in ALL_SENSOR_COLS:
    df[f'{col}_norm'] = np.nan

# Process row-by-row via cluster_id lookup
# Vectorised per cluster for speed
for cid, cfg in norm_config.items():
    mask = df['cluster_id'] == cid
    n    = mask.sum()
    if n == 0:
        continue

    # T_ambient for this cluster
    # Use per-row ambient Temperature if available, else cluster mean
    if AMBIENT_COL in df.columns:
        T_amb = df.loc[mask, AMBIENT_COL].values
    else:
        T_amb = np.full(n, cfg['T_ambient_mean'])

    # Pressure normalisation: P* = P / cluster_mean
    for col in PRESSURE_COLS:
        p_mean = cfg[col]['mean']
        if p_mean > 0:
            df.loc[mask, f'{col}_norm'] = (
                df.loc[mask, col].values / p_mean
            )
        else:
            df.loc[mask, f'{col}_norm'] = 0.0

    # Vibration normalisation: a* = a / cluster_mean
    for col in VIBRATION_COLS:
        a_mean = cfg[col]['mean']
        if a_mean > 0:
            df.loc[mask, f'{col}_norm'] = (
                df.loc[mask, col].values / a_mean
            )
        else:
            df.loc[mask, f'{col}_norm'] = 0.0

    # Temperature normalisation: ΔT* = (T - T_amb) / (T_max - T_amb)
    for col in TEMPERATURE_COLS:
        T_max = cfg[col]['max']
        T_vals= df.loc[mask, col].values
        denom = T_max - T_amb
        # Avoid division by zero
        denom = np.where(np.abs(denom) < 0.1, 0.1, denom)
        df.loc[mask, f'{col}_norm'] = (T_vals - T_amb) / denom

    log(f"  C{cid} ({cfg['operating_mode']}): "
        f"{n:,} rows normalised")

# =============================================================
# STEP 4 — Validation: check normalised ranges
# =============================================================
log("STEP 4 — Validating normalised value ranges...")
print()
print("=== NORMALISED VALUE RANGES ===")
print(f"  {'Column':<25} {'Min':>8} {'Max':>8} "
      f"{'Mean':>8} {'% > 1.0':>9} {'Status'}")
print("  " + "-"*72)

norm_cols_all = ([f'{c}_norm' for c in ALL_SENSOR_COLS])
range_issues  = []

for col in norm_cols_all:
    series  = df[col].dropna()
    if len(series) == 0:
        continue
    v_min   = series.min()
    v_max   = series.max()
    v_mean  = series.mean()
    pct_gt1 = (series > 1.0).sum() / len(series) * 100

    # Normal: 0–1, some values slightly above 1 for high-load is OK
    # Flag if mean > 2.0 (systematic calibration error)
    # Flag if any value > 10.0 (unphysical)
    if v_mean > 2.0 or v_max > 10.0:
        status = "⚠️  CHECK"
        range_issues.append(col)
    else:
        status = "✅"

    print(f"  {col:<25} {v_min:>8.4f} {v_max:>8.4f} "
          f"{v_mean:>8.4f} {pct_gt1:>8.2f}%  {status}")

results['norm_range_issues'] = range_issues
results['norm_cols']         = norm_cols_all

# =============================================================
# STEP 5 — Save normalised dataset
# =============================================================
log("\nSTEP 5 — Saving normalised dataset...")

# Columns to keep in normalised CSV
keep_cols = (
    ['Timestamp', 'segment_id', 'source_file',
     'cluster_id', 'operating_mode'] +
    ALL_SENSOR_COLS +           # raw values preserved
    norm_cols_all               # normalised values
)
# Add ambient cols if present
for c in ENV_COLS:
    if c in df.columns and c not in keep_cols:
        keep_cols.append(c)

df_norm = df[[c for c in keep_cols if c in df.columns]]

norm_path = NORM_DIR / "normalised_data.csv"
df_norm.to_csv(norm_path, index=False)
log(f"  Saved → {norm_path.name} ({len(df_norm):,} rows)")

results['normalised_rows']    = len(df_norm)
results['normalised_path']    = str(norm_path)

# =============================================================
# STEP 6 — Save normalisation config JSON (used by M4, M8, M10)
# =============================================================
log("STEP 6 — Saving normalisation config JSON...")

# Add global stats to config
norm_config['meta'] = {
    'created'         : str(date.today()),
    'pressure_cols'   : PRESSURE_COLS,
    'vibration_cols'  : VIBRATION_COLS,
    'temperature_cols': TEMPERATURE_COLS,
    'ambient_col'     : AMBIENT_COL,
    'window_size'     : 50,
    'warmup_rows'     : 300,
    'formula_pressure': 'P* = P_actual / P_cluster_mean',
    'formula_vibration': 'a* = a_actual / a_cluster_mean',
    'formula_temperature':
        'dT* = (T - T_ambient) / (T_cluster_max - T_ambient)',
    'normal_range'    : '0.0 to 1.0',
    'fault_indicator' : 'drift above 1.0 or anomalous temporal pattern'
}

config_path = OUTPUT_DIR / "M3_normalization_config.json"
with open(config_path, 'w') as f:
    json.dump(norm_config, f, indent=2)
log(f"  Saved → {config_path.name}")
results['config_path'] = str(config_path)

# =============================================================
# STEP 7 — PLOT 1: Raw vs Normalised distributions side-by-side
# =============================================================
log("STEP 7 — Generating raw vs normalised distribution plots...")
try:
    fig, axes = plt.subplots(3, len(ALL_SENSOR_COLS) // 3 + 1,
                             figsize=(20, 12))
    axes = axes.flatten()
    fig.suptitle('M3 — Raw vs Normalised Sensor Distributions',
                 fontsize=13, fontweight='bold')

    for idx, col in enumerate(ALL_SENSOR_COLS):
        ax   = axes[idx]
        raw  = df[col].dropna()
        norm = df[f'{col}_norm'].dropna()

        ax.hist(raw, bins=60, alpha=0.5, color='steelblue',
                label='Raw', density=True)
        ax2 = ax.twinx()
        ax2.hist(norm, bins=60, alpha=0.5, color='darkorange',
                 label='Norm', density=True)
        ax.set_title(col, fontsize=8, fontweight='bold')
        ax.set_xlabel('Value', fontsize=7)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)

        # Add vertical line at 1.0 for normalised
        ax2.axvline(1.0, color='red', linestyle='--',
                    linewidth=0.8, alpha=0.7)

    # Hide unused subplots
    for idx in range(len(ALL_SENSOR_COLS), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M3_raw_vs_norm_distributions.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M3_raw_vs_norm_distributions.png")
except Exception as e:
    log(f"  WARNING: Distribution plot failed: {e}")

# =============================================================
# STEP 8 — PLOT 2: Normalised value heatmap by cluster
# =============================================================
log("STEP 8 — Generating normalised value range heatmap by cluster...")
try:
    heat_data = []
    for cid in sorted(norm_config.keys()):
        if cid == 'meta':
            continue
        mask = df['cluster_id'] == cid
        mode = norm_config[cid]['operating_mode']
        row  = {'cluster': f"C{cid}:{mode}"}
        for col in ALL_SENSOR_COLS:
            vals = df.loc[mask, f'{col}_norm'].dropna()
            row[col] = round(vals.mean(), 3) if len(vals) > 0 else 0
        heat_data.append(row)

    heat_df = pd.DataFrame(heat_data).set_index('cluster')

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(heat_df, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, linewidths=0.5,
                vmin=0, vmax=2.0,
                cbar_kws={'label': 'Normalised value (1.0 = cluster mean)'})
    ax.set_title('M3 — Mean Normalised Values per Cluster\n'
                 '(green=normal ≤1.0, red=elevated >1.0)',
                 fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M3_norm_heatmap_by_cluster.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M3_norm_heatmap_by_cluster.png")
except Exception as e:
    log(f"  WARNING: Heatmap failed: {e}")

# =============================================================
# STEP 9 — PLOT 3: Time-series of normalised signals
# =============================================================
log("STEP 9 — Generating normalised time-series plot...")
try:
    best_seg = (df.groupby('segment_id').size().idxmax())
    seg_plot = df[df['segment_id'] == best_seg].reset_index(drop=True)

    plot_norm_cols = [f'{c}_norm' for c in
                      ['X_ACR_Mot.SV', 'X_ACR_Pmp.SV',
                       'X_Temp.SV', 'X_Pres.SV']]

    fig, axes = plt.subplots(len(plot_norm_cols), 1,
                             figsize=(16, 10), sharex=True)
    fig.suptitle(f'M3 — Normalised Sensor Signals\n'
                 f'Segment: {best_seg} | '
                 f'Red dashed = 1.0 (cluster mean)',
                 fontsize=11, fontweight='bold')

    colors = ['steelblue', 'darkorange', 'green', 'purple']
    for idx, col in enumerate(plot_norm_cols):
        ax = axes[idx]
        ax.plot(seg_plot.index, seg_plot[col],
                color=colors[idx], linewidth=0.6, alpha=0.8)
        ax.axhline(1.0, color='red', linestyle='--',
                   linewidth=1.0, alpha=0.7)
        ax.axhline(0.0, color='gray', linestyle=':',
                   linewidth=0.5, alpha=0.5)
        ax.set_ylabel(col.replace('_norm', '*'), fontsize=8)
        ax.grid(alpha=0.2)
        raw_col = col.replace('_norm', '')
        ax.set_title(f'{raw_col} → {col}', fontsize=8)

    axes[-1].set_xlabel('Sample index (1s intervals)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M3_normalised_timeseries.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M3_normalised_timeseries.png")
except Exception as e:
    log(f"  WARNING: Time-series plot failed: {e}")

# =============================================================
# STEP 10 — Markdown report
# =============================================================
log("STEP 10 — Writing markdown report...")

issue_str = (', '.join(results['norm_range_issues'])
             if results['norm_range_issues'] else 'None')

report_lines = [
    f"# M3 Normalisation Report",
    f"**Date:** {date.today()}  ",
    f"**Script:** {SCRIPT_NAME}  ",
    "",
    "## Summary",
    "| Metric | Value |",
    "|---|---|",
    f"| Normalised rows | {results['normalised_rows']:,} |",
    f"| Clusters used | {len(norm_config)-1} |",
    f"| Pressure cols | {', '.join(PRESSURE_COLS)} |",
    f"| Vibration cols | {', '.join(VIBRATION_COLS)} |",
    f"| Temperature cols | {', '.join(TEMPERATURE_COLS)} |",
    f"| Range issues | {issue_str} |",
    f"| Config saved | M3_normalization_config.json |",
    "",
    "## Normalisation Formulas",
    "- **Pressure:** P\\* = P_actual / P_cluster_mean",
    "- **Vibration:** a\\* = a_actual / a_cluster_mean",
    "- **Temperature:** ΔT\\* = (T − T_ambient) / "
    "(T_cluster_max − T_ambient)",
    "",
    "## Normal Operating Range",
    "- All normalised values expected in **0.0 – 1.0**",
    "- Values > 1.0 indicate elevated condition",
    "- Values > 2.0 indicate potential fault",
    "",
    "## Output Files",
    "- `data/normalized/normalised_data.csv`",
    "- `outputs/M3_normalization_config.json` → used by M4, M8, M10",
    "- `outputs/plots/M3_raw_vs_norm_distributions.png`",
    "- `outputs/plots/M3_norm_heatmap_by_cluster.png`",
    "- `outputs/plots/M3_normalised_timeseries.png`",
]

report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log(f"  Report saved → {report_path.name}")

# =============================================================
# PASTE TEXT UPDATE
# =============================================================
print()
print("═"*60)
print("  PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT")
print("═"*60)
print(f"M3_normalised_rows       : {results['normalised_rows']}")
print(f"M3_clusters_used         : {len(norm_config)-1}")
print(f"M3_pressure_formula      : P* = P / P_cluster_mean")
print(f"M3_vibration_formula     : a* = a / a_cluster_mean")
print(f"M3_temperature_formula   : dT* = (T-T_amb)/(T_max-T_amb)")
print(f"M3_normal_range          : 0.0 to 1.0")
print(f"M3_fault_indicator       : drift above 1.0")
print(f"M3_range_issues          : {issue_str}")
print(f"M3_config_file           : M3_normalization_config.json")
print(f"M3_normalised_data_file  : data/normalized/normalised_data.csv")
print(f"Status for M4            : READY")
print("═"*60)

# =============================================================
# FILE MANIFEST
# =============================================================
print()
print("── FILE MANIFEST ──────────────────────────────────────────")
print("→ Spaces upload (md + json + small csv):")
print(f"    {report_path}")
print(f"    {config_path}")
print("→ GitHub push (data + plots):")
print(f"    {norm_path}")
print(f"    {config_path}")
for f in sorted(PLOTS_DIR.glob("M3_*.png")):
    print(f"    {f}")
print("───────────────────────────────────────────────────────────")
print()
print("📦 M3 done. Starting M4.")
print("   Upload M3_normalization_config.json + report to Spaces.")
print("   Push normalised_data.csv + plots to GitHub.")
print("   Provide M4 complete script.")
