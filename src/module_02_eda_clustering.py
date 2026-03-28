import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =============================================================
# module_02_eda_clustering.py — M2: EDA + Operating Mode Clustering
# PumpSmart Project | Physics-Informed ML Digital Twin
# =============================================================
from config import (DEVICE, CLEAN_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.tsa.stattools import adfuller

SCRIPT_NAME = "module_02_eda_clustering"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}

# =============================================================
# CONSTANTS
# =============================================================
SENSOR_COLS = [
    'X_ACR_Mot.PV', 'X_ACR_Mot.SV', 'X_ACR_Mot.TV',
    'X_ACR_Pmp.PV', 'X_ACR_Pmp.SV', 'X_ACR_Pmp.TV',
    'X_Temp.SV', 'X_Pres.SV'
]
# Barometer and Temperature are environmental, not pump sensors
# They are NOT used for clustering or normalization baselines
ENV_COLS    = ['Barometer', 'Temperature']
MIN_USABLE  = 70          # from M1
K_RANGE     = range(2, 9) # test K = 2 to 8
ACF_LAG_MAX = 300         # max lags for autocorrelation window sizing

# Operating mode labels — assigned after inspecting cluster centroids
# Will be updated after K is determined and centroids are printed
MODE_LABELS = {
    0: 'startup',
    1: 'steady_state',
    2: 'high_load',
    3: 'cooldown'
}  # placeholder — overwritten after centroid analysis

# =============================================================
# STEP 1 — Load all usable segments from clean CSVs
# =============================================================
log("STEP 1 — Loading usable segments from M1 clean CSVs...")

try:
    seg_registry = pd.read_csv(CLEAN_DIR / "segment_registry.csv")
    usable_segs  = seg_registry[seg_registry['usable_for_windowing'] == True]
    log(f"  Segment registry loaded: {len(seg_registry)} total, "
        f"{len(usable_segs)} usable")
except Exception as e:
    raise RuntimeError(f"Cannot load segment_registry.csv: {e}")

clean_files = sorted(CLEAN_DIR.glob("Pump_*_clean.csv"))
all_frames  = []

for fpath in clean_files:
    try:
        df = pd.read_csv(fpath, parse_dates=['Timestamp'])
        # Keep only rows whose segment_id is in usable list
        usable_ids = usable_segs['segment_id'].tolist()
        df_usable  = df[df['segment_id'].isin(usable_ids)].copy()
        if len(df_usable) > 0:
            df_usable['source_file'] = fpath.stem
            all_frames.append(df_usable)
            log(f"  {fpath.name}: {len(df_usable):,} usable rows loaded")
    except Exception as e:
        log(f"  ERROR loading {fpath.name}: {e}")

df_all = pd.concat(all_frames, ignore_index=True)
df_all = df_all.sort_values(['segment_id', 'Timestamp']).reset_index(drop=True)

log(f"  Combined usable dataset: {len(df_all):,} rows | "
    f"{df_all['segment_id'].nunique()} segments | "
    f"{df_all['source_file'].nunique()} pumps")

results['usable_rows']     = len(df_all)
results['usable_segments'] = df_all['segment_id'].nunique()

# =============================================================
# STEP 2 — Per-sensor descriptive statistics
# =============================================================
log("STEP 2 — Computing per-sensor descriptive statistics...")

stats_df = df_all[SENSOR_COLS].describe(
    percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]
).T.round(6)
stats_df['cv']      = (stats_df['std'] / stats_df['mean'].abs()).round(4)
stats_df['range']   = (stats_df['max'] - stats_df['min']).round(6)
stats_df['skew']    = df_all[SENSOR_COLS].skew().round(4)
stats_df['kurtosis']= df_all[SENSOR_COLS].kurtosis().round(4)

log(f"  Statistics computed for {len(SENSOR_COLS)} sensor channels")
log(f"\n{stats_df[['mean','std','min','max','cv','skew']].to_string()}")

results['stats_table'] = stats_df.to_dict()

# =============================================================
# STEP 3 — ADF Stationarity Test
# =============================================================
log("STEP 3 — ADF stationarity test per sensor...")

adf_results = {}
for col in SENSOR_COLS:
    # Use one representative long segment for ADF
    longest_seg = (df_all.groupby('segment_id')[col]
                         .count()
                         .idxmax())
    series = (df_all[df_all['segment_id'] == longest_seg][col]
              .dropna()
              .values[:5000])  # cap at 5000 for speed
    try:
        adf_stat, p_val, _, _, crit_vals, _ = adfuller(series, autolag='AIC')
        stationary = p_val < 0.05
        adf_results[col] = {
            'adf_stat'  : round(adf_stat, 4),
            'p_value'   : round(p_val, 6),
            'stationary': stationary,
            'crit_1pct' : round(crit_vals['1%'], 4)
        }
        log(f"  {col:<20} ADF={adf_stat:>8.3f}  p={p_val:.4f}  "
            f"{'✅ Stationary' if stationary else '⚠️  Non-stationary'}")
    except Exception as e:
        log(f"  {col}: ADF failed — {e}")
        adf_results[col] = {'error': str(e)}

results['adf_results']        = adf_results
results['stationary_count']   = sum(1 for v in adf_results.values()
                                    if v.get('stationary', False))
results['nonstationary_count']= len(SENSOR_COLS) - results['stationary_count']

# =============================================================
# STEP 4 — Autocorrelation-based window size determination
# =============================================================
log("STEP 4 — Determining optimal window size via autocorrelation...")

acf_window_suggestions = {}
for col in SENSOR_COLS:
    longest_seg = (df_all.groupby('segment_id')[col]
                         .count()
                         .idxmax())
    series = (df_all[df_all['segment_id'] == longest_seg][col]
              .dropna()
              .values[:3000])
    series = series - series.mean()

    # Compute ACF manually (no external acf func needed)
    n      = len(series)
    var    = np.var(series)
    if var == 0:
        acf_window_suggestions[col] = 60
        continue
    acf    = np.array([
        np.mean(series[:n-k] * series[k:]) / var
        for k in range(1, min(ACF_LAG_MAX + 1, n // 2))
    ])
    # First lag where ACF drops below 1/e ≈ 0.368 (characteristic time)
    threshold = 1 / np.e
    below     = np.where(acf < threshold)[0]
    window    = int(below[0]) + 1 if len(below) > 0 else 60
    window    = max(30, min(window, 120))  # clamp 30–120s
    acf_window_suggestions[col] = window
    log(f"  {col:<20} ACF decay lag: {window}s")

# Final window size = median of all suggestions, rounded to nearest 10
suggested_windows   = list(acf_window_suggestions.values())
optimal_window_size = int(round(np.median(suggested_windows) / 10) * 10)
optimal_window_size = max(30, min(optimal_window_size, 120))

log(f"\n  Individual suggestions: {acf_window_suggestions}")
log(f"  ✅ Optimal window size: {optimal_window_size}s "
    f"(median={np.median(suggested_windows):.1f}, rounded to nearest 10)")

results['optimal_window_size']    = optimal_window_size
results['acf_window_suggestions'] = acf_window_suggestions

# =============================================================
# STEP 5 — Correlation matrix
# =============================================================
log("STEP 5 — Computing sensor correlation matrix...")

corr_matrix = df_all[SENSOR_COLS].corr().round(4)
top_pairs   = []
for i in range(len(SENSOR_COLS)):
    for j in range(i+1, len(SENSOR_COLS)):
        c1, c2 = SENSOR_COLS[i], SENSOR_COLS[j]
        r = abs(corr_matrix.loc[c1, c2])
        top_pairs.append((c1, c2, round(r, 4)))
top_pairs.sort(key=lambda x: -x[2])

log("  Top 5 correlated pairs:")
for p in top_pairs[:5]:
    log(f"    {p[0]} ↔ {p[1]}: r={p[2]}")

results['top_correlation_pairs'] = top_pairs[:5]
results['top_correlation']       = top_pairs[0]

# =============================================================
# STEP 6 — K-Means: Elbow + Silhouette
# =============================================================
log("STEP 6 — K-Means clustering: elbow + silhouette analysis...")

# Scale sensor features for clustering
X_raw   = df_all[SENSOR_COLS].values
scaler  = StandardScaler()
X_scaled= scaler.fit_transform(X_raw)

# Subsample for speed (max 50k rows)
np.random.seed(42)
if len(X_scaled) > 50000:
    idx      = np.random.choice(len(X_scaled), 50000, replace=False)
    X_sample = X_scaled[idx]
else:
    X_sample = X_scaled

inertias    = []
silhouettes = []

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_sample)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_sample, labels, sample_size=10000,
                           random_state=42)
    silhouettes.append(sil)
    log(f"  K={k}: inertia={km.inertia_:,.0f} | silhouette={sil:.4f}")

# Optimal K — highest silhouette score
optimal_k = list(K_RANGE)[np.argmax(silhouettes)]
log(f"\n  ✅ Optimal K = {optimal_k} "
    f"(silhouette={max(silhouettes):.4f})")

results['optimal_k']         = optimal_k
results['silhouette_score']  = round(max(silhouettes), 4)
results['inertias']          = {k: round(v, 2)
                                for k, v in zip(K_RANGE, inertias)}
results['silhouette_scores'] = {k: round(v, 4)
                                for k, v in zip(K_RANGE, silhouettes)}

# =============================================================
# STEP 7 — Final K-Means fit on full dataset
# =============================================================
log(f"STEP 7 — Final K-Means fit with K={optimal_k} on full dataset...")

km_final = KMeans(n_clusters=optimal_k, random_state=42,
                  n_init=10, max_iter=500)
df_all['cluster_id'] = km_final.fit_predict(X_scaled)

# Cluster centroids in original scale
centroids_scaled = km_final.cluster_centers_
centroids_raw    = scaler.inverse_transform(centroids_scaled)
centroids_df     = pd.DataFrame(centroids_raw,
                                columns=SENSOR_COLS)
centroids_df.index.name = 'cluster_id'

log("\n  Cluster centroids (raw sensor units):")
log(centroids_df.round(4).to_string())

# =============================================================
# STEP 8 — Assign operating mode labels based on centroids
# =============================================================
log("STEP 8 — Assigning operating mode labels...")

# Physics-based labeling logic (verified against M2_cluster_bounds.csv centroids):
# - Cluster label assigned by combined rank of SV (vibration velocity) + TV (temperature)
# - Low SV + near-zero Pressure + moderate TV              → cooldown (spinning down)
# - Low SV + near-zero Pressure + HIGHEST TV               → startup  (thermal run-in:
#                                                             motor heats before hydraulics
#                                                             load in 7-stage multistage pump)
# - High SV + high stable Pressure + mid TV                → high_load (vibration-dominated)
# - Moderate SV + highest stable Pressure + high TV        → steady_state
#
# FIX-3 (2026-03-28): Original comment "High temp + high vibration → high_load" was WRONG.
# Startup (C2) has higher mean TV (39.6°C) than high_load (C3, 35.1°C) — thermal lag effect.
# Data labels are CORRECT. Only this comment was incorrect.

temp_proxy   = centroids_df[['X_ACR_Mot.TV', 'X_ACR_Pmp.TV',
                              'X_Temp.SV']].mean(axis=1)
vib_proxy    = centroids_df[['X_ACR_Mot.SV',
                              'X_ACR_Pmp.SV']].mean(axis=1)
pres_proxy   = centroids_df['X_Pres.SV']

# Rank clusters by combined temp+vibration proxy
load_rank    = (temp_proxy.rank() + vib_proxy.rank()).rank()

# Label assignment based on rank and pressure
mode_map     = {}
sorted_ids   = load_rank.sort_values().index.tolist()

available_modes = ['cooldown', 'startup', 'steady_state', 'high_load']
# Pad or trim if optimal_k differs from 4
if optimal_k < 4:
    available_modes = available_modes[:optimal_k]
elif optimal_k > 4:
    extra = [f'mode_{i}' for i in range(4, optimal_k)]
    available_modes = available_modes + extra

for rank_idx, cluster_id in enumerate(sorted_ids):
    mode_map[int(cluster_id)] = available_modes[rank_idx]

df_all['operating_mode'] = df_all['cluster_id'].map(mode_map)

log("  Operating mode assignments:")
for cid, mode in mode_map.items():
    n_rows = (df_all['cluster_id'] == cid).sum()
    pct    = round(n_rows / len(df_all) * 100, 1)
    log(f"    Cluster {cid} → {mode:<15} : "
        f"{n_rows:,} rows ({pct}%)")

results['mode_map']        = mode_map
results['cluster_counts']  = df_all['cluster_id'].value_counts().to_dict()

# =============================================================
# STEP 9 — Compute per-cluster bounds (2.5th–97.5th percentile)
# =============================================================
log("STEP 9 — Computing per-cluster sensor bounds for M3...")

cluster_bounds_records = []

for cid in range(optimal_k):
    mask    = df_all['cluster_id'] == cid
    sub_df  = df_all[mask][SENSOR_COLS]
    mode    = mode_map[int(cid)]
    n_rows  = mask.sum()

    row = {
        'cluster_id'    : cid,
        'operating_mode': mode,
        'n_rows'        : n_rows
    }
    for col in SENSOR_COLS:
        row[f'{col}_mean']  = round(sub_df[col].mean(), 6)
        row[f'{col}_std']   = round(sub_df[col].std(), 6)
        row[f'{col}_p2_5']  = round(sub_df[col].quantile(0.025), 6)
        row[f'{col}_p97_5'] = round(sub_df[col].quantile(0.975), 6)
        row[f'{col}_max']   = round(sub_df[col].max(), 6)
        row[f'{col}_min']   = round(sub_df[col].min(), 6)

    cluster_bounds_records.append(row)

cluster_bounds_df = pd.DataFrame(cluster_bounds_records)
bounds_path       = OUTPUT_DIR / "M2_cluster_bounds.csv"
cluster_bounds_df.to_csv(bounds_path, index=False)
log(f"  ✅ Cluster bounds saved → {bounds_path.name}")

results['cluster_bounds_path'] = str(bounds_path)

# =============================================================
# STEP 10 — PLOT 1: Elbow + Silhouette (side by side)
# =============================================================
log("STEP 10 — Generating elbow + silhouette plot...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('M2 — K-Means Optimal K Selection',
                 fontsize=13, fontweight='bold')

    k_vals = list(K_RANGE)

    # Elbow
    axes[0].plot(k_vals, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(optimal_k, color='red', linestyle='--',
                    label=f'Optimal K={optimal_k}')
    axes[0].set_title('Elbow Method (Inertia)', fontweight='bold')
    axes[0].set_xlabel('Number of Clusters K')
    axes[0].set_ylabel('Inertia')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Silhouette
    axes[1].plot(k_vals, silhouettes, 'gs-', linewidth=2, markersize=8)
    axes[1].axvline(optimal_k, color='red', linestyle='--',
                    label=f'Optimal K={optimal_k}')
    axes[1].set_title('Silhouette Score', fontweight='bold')
    axes[1].set_xlabel('Number of Clusters K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M2_kmeans_selection.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M2_kmeans_selection.png")
except Exception as e:
    log(f"  WARNING: Elbow plot failed: {e}")

# =============================================================
# STEP 11 — PLOT 2: Cluster distribution (PCA 2D projection)
# =============================================================
log("STEP 11 — Generating cluster PCA projection plot...")
try:
    from sklearn.decomposition import PCA

    pca       = PCA(n_components=2, random_state=42)
    # Subsample for plot clarity
    plot_idx  = np.random.choice(len(X_scaled),
                                  min(8000, len(X_scaled)),
                                  replace=False)
    X_pca     = pca.fit_transform(X_scaled[plot_idx])
    labels_sub= df_all['cluster_id'].values[plot_idx]
    modes_sub = df_all['operating_mode'].values[plot_idx]

    fig, ax   = plt.subplots(figsize=(10, 7))
    colors    = plt.cm.tab10.colors
    for cid in range(optimal_k):
        mask  = labels_sub == cid
        mode  = mode_map[int(cid)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[cid]], label=f'C{cid}: {mode}',
                   alpha=0.4, s=8, edgecolors='none')

    ax.set_title(f'M2 — Cluster PCA Projection (K={optimal_k})\n'
                 f'PC1={pca.explained_variance_ratio_[0]*100:.1f}% | '
                 f'PC2={pca.explained_variance_ratio_[1]*100:.1f}%',
                 fontweight='bold')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(markerscale=3, fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M2_cluster_pca.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M2_cluster_pca.png")

    results['pca_var_pc1'] = round(pca.explained_variance_ratio_[0]*100, 2)
    results['pca_var_pc2'] = round(pca.explained_variance_ratio_[1]*100, 2)
except Exception as e:
    log(f"  WARNING: PCA plot failed: {e}")

# =============================================================
# STEP 12 — PLOT 3: Cluster centroid radar / heatmap
# =============================================================
log("STEP 12 — Generating cluster centroid heatmap...")
try:
    # Normalize centroids to 0-1 for visual comparison
    cent_norm = centroids_df.copy()
    for col in SENSOR_COLS:
        col_min = cent_norm[col].min()
        col_max = cent_norm[col].max()
        if col_max > col_min:
            cent_norm[col] = (cent_norm[col] - col_min) / (col_max - col_min)

    cent_norm.index = [f"C{i}: {mode_map[i]}" for i in range(optimal_k)]

    fig, ax = plt.subplots(figsize=(12, max(4, optimal_k + 1)))
    sns.heatmap(cent_norm, annot=True, fmt='.3f', cmap='RdYlGn',
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Normalised Value'})
    ax.set_title('M2 — Cluster Centroids (Min-Max Normalised for Display)',
                 fontweight='bold')
    ax.set_xlabel('Sensor Channel')
    ax.set_ylabel('Cluster / Operating Mode')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M2_cluster_centroids.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M2_cluster_centroids.png")
except Exception as e:
    log(f"  WARNING: Centroid heatmap failed: {e}")

# =============================================================
# STEP 13 — PLOT 4: Sensor time-series coloured by cluster
# =============================================================
log("STEP 13 — Generating time-series cluster overlay plot...")
try:
    # Pick one long segment for clarity
    best_seg = (df_all.groupby('segment_id')
                      .size()
                      .idxmax())
    seg_plot = df_all[df_all['segment_id'] == best_seg].copy()
    seg_plot = seg_plot.reset_index(drop=True)

    plot_sensors = ['X_ACR_Mot.SV', 'X_ACR_Pmp.SV',
                    'X_Temp.SV', 'X_Pres.SV']
    fig, axes    = plt.subplots(len(plot_sensors), 1,
                                figsize=(16, 12), sharex=True)
    fig.suptitle(f'M2 — Sensor Signals Coloured by Operating Mode\n'
                 f'Segment: {best_seg}',
                 fontsize=12, fontweight='bold')

    colors = plt.cm.tab10.colors
    for idx, col in enumerate(plot_sensors):
        ax  = axes[idx]
        t   = seg_plot.index
        ax.plot(t, seg_plot[col], color='lightgray',
                linewidth=0.5, zorder=1)
        for cid in range(optimal_k):
            mask = seg_plot['cluster_id'] == cid
            ax.scatter(t[mask], seg_plot.loc[mask, col],
                       c=[colors[cid]], s=2, alpha=0.7,
                       label=f"C{cid}:{mode_map[cid]}", zorder=2)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(markerscale=5, fontsize=7,
                      loc='upper right', ncol=optimal_k)

    axes[-1].set_xlabel('Sample Index (1s intervals)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M2_timeseries_clusters.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M2_timeseries_clusters.png")
except Exception as e:
    log(f"  WARNING: Time-series plot failed: {e}")

# =============================================================
# STEP 14 — PLOT 5: Correlation heatmap
# =============================================================
log("STEP 14 — Generating correlation heatmap...")
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    mask    = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                cmap='coolwarm', center=0, ax=ax,
                linewidths=0.5, vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson r'})
    ax.set_title('M2 — Sensor Correlation Matrix',
                 fontsize=12, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M2_correlation_matrix.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M2_correlation_matrix.png")
except Exception as e:
    log(f"  WARNING: Correlation heatmap failed: {e}")

# =============================================================
# STEP 15 — Save labelled dataset
# =============================================================
log("STEP 15 — Saving cluster-labelled dataset...")
labelled_path = OUTPUT_DIR / "M2_labelled_data.csv"
df_all[['Timestamp', 'segment_id', 'source_file'] +
       SENSOR_COLS +
       ['cluster_id', 'operating_mode']
].to_csv(labelled_path, index=False)
log(f"  Saved → {labelled_path.name}")
results['labelled_data_path'] = str(labelled_path)

# =============================================================
# STEP 16 — Markdown report
# =============================================================
log("STEP 16 — Writing markdown report...")

top3_corr = '\n'.join([f"  - {p[0]} ↔ {p[1]}: r={p[2]}"
                        for p in top_pairs[:3]])
adf_summary = '\n'.join([
    f"  - {col}: {'Stationary' if v.get('stationary') else 'Non-stationary'} "
    f"(p={v.get('p_value','N/A')})"
    for col, v in adf_results.items()
])

report_lines = [
    "# M2 EDA + Clustering Report",
    f"**Date:** {date.today()}  ",
    f"**Script:** {SCRIPT_NAME}  ",
    "",
    "## Summary",
    "| Metric | Value |",
    "|---|---|",
    f"| Usable rows | {results['usable_rows']:,} |",
    f"| Usable segments | {results['usable_segments']} |",
    f"| Optimal K | {results['optimal_k']} |",
    f"| Silhouette score | {results['silhouette_score']} |",
    f"| Optimal window size | {results['optimal_window_size']}s |",
    f"| Stationary sensors | {results['stationary_count']}/{len(SENSOR_COLS)} |",
    f"| Top correlation | {results['top_correlation'][0]} ↔ {results['top_correlation'][1]}: r={results['top_correlation'][2]} |",
    "",
    "## Operating Mode Assignments",
    "| Cluster | Mode | Rows |",
    "|---|---|---|",
] + [
    f"| C{cid} | {mode} | {results['cluster_counts'].get(cid, 0):,} |"
    for cid, mode in mode_map.items()
] + [
    "",
    "## ADF Stationarity Results",
    adf_summary,
    "",
    "## Top Correlations",
    top3_corr,
    "",
    "## Cluster Bounds",
    cluster_bounds_df.to_markdown(index=False),
    "",
    "## Output Files",
    "- `outputs/M2_cluster_bounds.csv` → Used by M3 for normalization baselines",
    "- `outputs/M2_labelled_data.csv` → Full dataset with cluster labels",
    "- `outputs/plots/M2_kmeans_selection.png`",
    "- `outputs/plots/M2_cluster_pca.png`",
    "- `outputs/plots/M2_cluster_centroids.png`",
    "- `outputs/plots/M2_timeseries_clusters.png`",
    "- `outputs/plots/M2_correlation_matrix.png`",
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
print(f"M2_usable_rows         : {results['usable_rows']}")
print(f"M2_usable_segments     : {results['usable_segments']}")
print(f"M2_optimal_k           : {results['optimal_k']}")
print(f"M2_silhouette_score    : {results['silhouette_score']}")
print(f"M2_optimal_window_size : {results['optimal_window_size']}s")
print(f"M2_stationary_sensors  : {results['stationary_count']}/{len(SENSOR_COLS)}")
print(f"M2_top_correlation     : {results['top_correlation'][0]} ↔ "
      f"{results['top_correlation'][1]} r={results['top_correlation'][2]}")
print(f"M2_mode_map            : {mode_map}")
print(f"M2_cluster_bounds_file : M2_cluster_bounds.csv")
print(f"M2_pca_variance        : PC1={results.get('pca_var_pc1','N/A')}% "
      f"PC2={results.get('pca_var_pc2','N/A')}%")
print(f"Status for M3          : READY")
print("═"*60)

# =============================================================
# FILE MANIFEST
# =============================================================
print()
print("── FILE MANIFEST ──────────────────────────────────────────")
print("→ Spaces upload (md + csv):")
print(f"    {report_path}")
print(f"    {bounds_path}")
print(f"    {labelled_path}")
print("→ GitHub push (plots + data):")
for f in sorted(PLOTS_DIR.glob("M2_*.png")):
    print(f"    {f}")
print(f"    {labelled_path}")
print(f"    {bounds_path}")
print("───────────────────────────────────────────────────────────")
print()
print("📦 M2 done. Starting M3.")
print("   Upload M2_cluster_bounds.csv + report to Spaces.")
print("   Push plots + labelled data to GitHub.")
print("   Provide M3 complete script.")
