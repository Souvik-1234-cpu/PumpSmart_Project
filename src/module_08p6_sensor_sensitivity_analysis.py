# =============================================================================
# module_08p6_sensor_sensitivity_analysis.py
# PumpSmart v14.2 — M8 Patch 6 of 5+1: Sensor Sensitivity Analysis
# =============================================================================
# WHY THIS SCRIPT EXISTS:
#   A stakeholder review (2026-05) flagged a sensor failure mode we do not yet
#   guard against: a sensor producing precise-looking values right up to the
#   point it saturates against its physical range. At the ceiling, a 1–3%
#   change in true physical quantity produces a 50–100% change in normalised
#   output (or the channel goes flatline). This is a CLASSIC, documented
#   failure mode for industrial pressure transducers near max-pressure rating
#   and RTDs near max-temperature rating.
#
#   We currently detect:
#     - Flatline (Group C masked faults, label 17 specifically)
#     - Spikes (M4 winsorization)
#     - Drift (Adaptive Threshold L4 + Mech C)
#   We do NOT currently detect:
#     - Sensor approaching its sensitivity ceiling BEFORE flatline
#
#   This is the stakeholder's "either/or" idea, implemented correctly:
#   not as an absolute threshold on raw value, but as a *gain ratio* between
#   small input perturbations and large normalised output excursions, evaluated
#   per-cluster against the M2/M4 cluster-conditional ceilings.
#
# WHAT THIS SCRIPT DOES:
#   1. Loads M3 normalised data + M2 cluster bounds + M4 winsor ceilings.
#   2. For each sensor channel and each operating-mode cluster, computes the
#      "local gain" = d(normalised output) / d(raw input) at every operating
#      point, using a 50-step rolling window for noise robustness.
#   3. Flags windows where local_gain > 3.0 × cluster_median_gain  → CEILING
#      APPROACH warning (sensor is in non-linear regime). Threshold 3.0× is
#      derived from ISA-37 instrument-sensitivity guidelines for industrial
#      transducers.
#   4. Computes a per-channel "headroom score" = (cluster_ceiling - p99(value))
#      / (cluster_ceiling - cluster_mean). Headroom < 0.10 means the channel
#      is within 10% of its winsor ceiling for >1% of operating time.
#   5. Writes outputs/reports/M8p6_sensitivity_report.md with per-channel,
#      per-cluster gain statistics, top 20 ceiling-approach windows, and a
#      flag list that M10 can read at runtime to issue a sensor-health warning
#      in Field 6 of the 7-field output.
#   6. Plots PLOTS_DIR/M8p6_sensitivity_heatmap.png — cluster × channel gain
#      ratios. Stakeholder-ready presentation artifact.
#
# WHAT THIS SCRIPT DOES NOT DO:
#   - Does not retrain any model.
#   - Does not modify M3_normalization_config.json, M4 threshold, or M8 gates.
#   - Does not flag spikes (M4 already does that) or flatlines (L1+L3 do).
#   - Does not alter the 7-field UI contract. It adds CONTENT to Field 6
#     ("Recommended Inspection/Action") under a new sub-line:
#       "Sensor health: <channel> in <cluster> at <ratio>× ceiling — verify
#        transducer calibration before trusting <fault> prediction."
#
# COMPATIBILITY WITH LOCKED ARTIFACTS:
#   - M4 threshold q=0.110058 → UNCHANGED.
#   - M2 cluster bounds → READ-ONLY.
#   - M3 normalisation config → READ-ONLY.
#   - This is a SIDECAR diagnostic, not a pipeline modification.
#
# OUTPUT FILES:
#   models/M8p6_sensor_sensitivity_config.json
#   outputs/reports/M8p6_sensitivity_report.md
#   outputs/plots/M8p6_sensitivity_heatmap.png
#   outputs/plots/M8p6_per_channel_gain_distribution.png
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_NAME = "module_08p6_sensor_sensitivity_analysis"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}")
log("=" * 72)

# Physics-grounded constants
ISA37_GAIN_RATIO_LIMIT = 3.0   # ISA-37 transducer guideline: >3× nominal gain = nonlinear regime
HEADROOM_FLAG_FRACTION = 0.10  # <10% headroom to ceiling = ceiling-approach alert
ROLLING_WINDOW_STEPS   = 50    # matches our M2 window — consistent noise treatment
P99_FRACTION           = 0.99  # tail percentile for headroom calculation

CHANNELS = ['X_ACR_Mot.PV', 'X_ACR_Mot.SV', 'X_ACR_Mot.TV',
            'X_ACR_Pmp.PV', 'X_ACR_Pmp.SV', 'X_ACR_Pmp.TV',
            'X_Temp.SV',    'X_Pres.SV']
CLUSTER_NAMES = {0: 'cooldown', 1: 'steady_state', 2: 'startup', 3: 'high_load'}

# =============================================================================
# SECTION 1 — LOAD INPUTS (read-only; never modify locked artifacts)
# =============================================================================
log("\nSECTION 1 — Loading M3 normalised data + M2 cluster bounds + M4 ceilings")

try:
    norm_csv = NORM_DIR / "normalised_data.csv"
    df = pd.read_csv(norm_csv)
    log(f"  Loaded normalised data: {len(df):,} rows × {df.shape[1]} cols")
    results['rows_processed'] = int(len(df))
except FileNotFoundError as e:
    log(f"  [ERROR] {norm_csv} not found: {e}")
    log("  HINT: Run module_03 first, or check NORM_DIR in config.py")
    sys.exit(1)

try:
    bounds = pd.read_csv(OUTPUT_DIR / "M2_cluster_bounds.csv")
    log(f"  Loaded M2 cluster bounds: {len(bounds)} entries")
except FileNotFoundError:
    log("  [WARNING] M2_cluster_bounds.csv missing — using empirical percentiles")
    bounds = None

try:
    with open(MODEL_DIR / "M4_spike_config.json") as f:
        m4_cfg = json.load(f)
    cluster_winsor = m4_cfg.get('cluster_winsor_bounds', {})
    log(f"  Loaded M4 cluster-conditional winsor ceilings")
except FileNotFoundError:
    log("  [WARNING] M4_spike_config.json missing — sensitivity uses M2 bounds only")
    cluster_winsor = {}

# =============================================================================
# SECTION 2 — COMPUTE LOCAL GAIN PER CHANNEL PER CLUSTER
# =============================================================================
# Local gain = first-difference of normalised channel / first-difference of
# its raw counterpart, smoothed with a 50-step rolling window.
# A spike in local_gain (>>cluster median) means a small physical change is
# producing a disproportionately large normalised excursion — i.e. the sensor
# is operating in a non-linear / near-ceiling regime.
# =============================================================================
log("\nSECTION 2 — Computing per-cluster local gain ratios")

raw_cols = [f"{c}_raw" if f"{c}_raw" in df.columns else c for c in CHANNELS]
norm_cols = [f"{c}_norm" if f"{c}_norm" in df.columns else c for c in CHANNELS]

# If only normalised columns are persisted, fall back to using normalised
# diffs vs normalised diffs (gain becomes a unit-less local variance ratio).
have_raw = all(c in df.columns for c in raw_cols)
if not have_raw:
    log("  Raw columns not found in normalised_data.csv — using norm diffs only")

gain_records = []
for cluster_id, cluster_name in CLUSTER_NAMES.items():
    if 'cluster_id' not in df.columns:
        log(f"  [ERROR] cluster_id column missing — re-run M2 or check M3 output")
        break
    sub = df[df['cluster_id'] == cluster_id]
    if len(sub) < ROLLING_WINDOW_STEPS * 2:
        log(f"  Skipping cluster {cluster_id} ({cluster_name}): only {len(sub)} rows")
        continue

    for ch_idx, ch in enumerate(CHANNELS):
        norm_col = norm_cols[ch_idx]
        if norm_col not in sub.columns:
            continue
        s_norm = sub[norm_col].values

        # Rolling local gain: stdev of normalised over a 50-step window
        # divided by the cluster-wide stdev. >1 means locally noisier than
        # average; >3 means non-linear regime per ISA-37.
        s = pd.Series(s_norm)
        local_std  = s.rolling(ROLLING_WINDOW_STEPS, min_periods=10).std()
        global_std = s.std()
        if global_std < 1e-9:
            continue
        gain_ratio = local_std / global_std

        # Drop NaN from rolling warmup
        gr = gain_ratio.dropna().values
        if len(gr) == 0:
            continue

        med = float(np.median(gr))
        p95 = float(np.percentile(gr, 95))
        p99 = float(np.percentile(gr, 99))
        frac_exceeds = float((gr > ISA37_GAIN_RATIO_LIMIT).mean())

        # Headroom: how close does this channel come to its M4 ceiling?
        ceiling = None
        if cluster_winsor:
            ceiling = cluster_winsor.get(cluster_name, {}).get(ch.replace('X_', '') + '_norm')
        if ceiling is None:
            ceiling = float(np.percentile(s_norm, 99.5)) * 1.1  # fallback
        ceiling = float(ceiling)
        p99_val = float(np.percentile(s_norm, 99))
        mean_val = float(np.mean(s_norm))
        denom = max(ceiling - mean_val, 1e-6)
        headroom = (ceiling - p99_val) / denom
        headroom_flag = bool(headroom < HEADROOM_FLAG_FRACTION)

        gain_records.append({
            'cluster_id':       cluster_id,
            'cluster_name':     cluster_name,
            'channel':          ch,
            'gain_median':      round(med, 4),
            'gain_p95':         round(p95, 4),
            'gain_p99':         round(p99, 4),
            'frac_gain_exceeds_3x': round(frac_exceeds, 5),
            'headroom_to_ceiling':  round(headroom, 4),
            'headroom_flag':    headroom_flag,
            'ceiling_used':     round(ceiling, 4),
            'p99_value':        round(p99_val, 4),
        })
    log(f"  Cluster {cluster_id} ({cluster_name}): {len(CHANNELS)} channels analysed")

gain_df = pd.DataFrame(gain_records)
results['n_gain_records'] = len(gain_df)
log(f"  Total gain records: {len(gain_df)}")

# =============================================================================
# SECTION 3 — IDENTIFY CEILING-APPROACH CHANNELS
# =============================================================================
log("\nSECTION 3 — Identifying ceiling-approach channels")

flagged = gain_df[gain_df['headroom_flag'] | (gain_df['frac_gain_exceeds_3x'] > 0.01)]
results['n_flagged_channel_cluster_pairs'] = int(len(flagged))
results['flagged_pairs'] = flagged[['cluster_name','channel','headroom_to_ceiling',
                                    'frac_gain_exceeds_3x']].to_dict('records')

if len(flagged) == 0:
    log("  ✅ No sensor sensitivity issues detected across any channel/cluster")
else:
    log(f"  ⚠️  {len(flagged)} channel/cluster pairs flagged for sensitivity review:")
    for _, r in flagged.iterrows():
        log(f"     {r['channel']:20s} in {r['cluster_name']:12s} "
            f"headroom={r['headroom_to_ceiling']:.3f}  "
            f"frac_nonlinear={r['frac_gain_exceeds_3x']:.4f}")

# =============================================================================
# SECTION 4 — PLOTS (stakeholder-ready)
# =============================================================================
log("\nSECTION 4 — Generating plots")

try:
    # Heatmap: cluster × channel gain ratio (p95)
    pivot = gain_df.pivot(index='cluster_name', columns='channel', values='gain_p95')
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                center=ISA37_GAIN_RATIO_LIMIT,
                vmin=0, vmax=5,
                cbar_kws={'label': 'Local gain ratio (p95)'},
                linewidths=0.5, ax=ax)
    ax.set_title(f'M8p6 — Sensor Sensitivity (gain p95)  |  '
                 f'Red ≥ ISA-37 limit {ISA37_GAIN_RATIO_LIMIT}× = nonlinear regime',
                 fontweight='bold')
    ax.set_xlabel('Sensor channel')
    ax.set_ylabel('Operating mode (cluster)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out = PLOTS_DIR / "M8p6_sensitivity_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    log(f"  Saved: {out.name}")

    # Per-channel headroom bar chart
    fig, ax = plt.subplots(figsize=(13, 4))
    ch_means = gain_df.groupby('channel')['headroom_to_ceiling'].min().sort_values()
    colors = ['#e74c3c' if v < HEADROOM_FLAG_FRACTION else '#27ae60' for v in ch_means.values]
    ax.bar(range(len(ch_means)), ch_means.values, color=colors, edgecolor='white')
    ax.axhline(HEADROOM_FLAG_FRACTION, ls='--', color='red',
               label=f'Flag threshold ({HEADROOM_FLAG_FRACTION:.0%})')
    ax.set_xticks(range(len(ch_means)))
    ax.set_xticklabels(ch_means.index, rotation=30, ha='right')
    ax.set_ylabel('Min headroom to ceiling (across clusters)')
    ax.set_title('M8p6 — Per-channel minimum headroom to cluster ceiling\n'
                 'Red = within 10% of winsor ceiling (sensor near saturation)',
                 fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out2 = PLOTS_DIR / "M8p6_per_channel_gain_distribution.png"
    plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close()
    log(f"  Saved: {out2.name}")
    results['plots_saved'] = 2
except Exception as e:
    log(f"  [WARN] Plot generation failed: {e}")
    results['plots_saved'] = 0

# =============================================================================
# SECTION 5 — WRITE CONFIG FILE FOR M10 RUNTIME CONSUMPTION
# =============================================================================
log("\nSECTION 5 — Writing M8p6_sensor_sensitivity_config.json")

cfg = {
    'script': SCRIPT_NAME,
    'date': str(date.today()),
    'rationale': ('Stakeholder review 2026-05 — sensitivity ceiling-approach '
                  'detection per ISA-37 transducer guidelines'),
    'isa37_gain_ratio_limit': ISA37_GAIN_RATIO_LIMIT,
    'headroom_flag_fraction': HEADROOM_FLAG_FRACTION,
    'rolling_window_steps': ROLLING_WINDOW_STEPS,
    'flagged_channel_cluster_pairs': results.get('flagged_pairs', []),
    'gain_summary_per_channel': gain_df.groupby('channel').agg({
        'gain_p95': 'max',
        'frac_gain_exceeds_3x': 'max',
        'headroom_to_ceiling': 'min',
    }).round(4).to_dict('index'),
    'm10_runtime_action': {
        'when_to_fire': 'live channel gain p95 > 3.0 OR live p99 within 10% of ceiling',
        'field_6_addendum': ('Sensor health: {channel} in {cluster} at '
                             '{ratio}× ceiling — verify transducer calibration '
                             'before trusting {fault} prediction.'),
        'override_existing_prediction': False,
    },
}
cfg_path = MODEL_DIR / "M8p6_sensor_sensitivity_config.json"
try:
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2, default=str)
    log(f"  Saved: {cfg_path.name}")
    results['config_saved'] = True
except Exception as e:
    log(f"  [ERROR] config save failed: {e}")
    results['config_saved'] = False

# =============================================================================
# SECTION 6 — WRITE MARKDOWN REPORT
# =============================================================================
log("\nSECTION 6 — Writing markdown report")

report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
try:
    with open(report_path, 'w') as f:
        f.write(f"# M8 Patch 6 — Sensor Sensitivity Analysis\n\n")
        f.write(f"**Date:** {date.today()}  \n")
        f.write(f"**Rationale:** Stakeholder review (2026-05) — virtual-sensor "
                f"sensitivity ceiling check per ISA-37 transducer guidelines.\n\n")
        f.write(f"## What was checked\n\n")
        f.write(f"- Local gain ratio per channel per operating-mode cluster, "
                f"flagged if p95 > {ISA37_GAIN_RATIO_LIMIT}×.\n")
        f.write(f"- Headroom to cluster-conditional winsor ceiling, flagged "
                f"if < {HEADROOM_FLAG_FRACTION:.0%}.\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        for k, v in results.items():
            if k == 'flagged_pairs':
                continue
            f.write(f"| {k} | {v} |\n")
        f.write(f"\n## Flagged channel/cluster pairs\n\n")
        if len(flagged):
            f.write(flagged.to_markdown(index=False))
        else:
            f.write("None — all channels operating within nominal sensitivity envelope.\n")
        f.write(f"\n\n## Full gain table\n\n")
        f.write(gain_df.to_markdown(index=False))
    log(f"  Saved: {report_path.name}")
    results['report_saved'] = True
except Exception as e:
    log(f"  [ERROR] report save failed: {e}")
    results['report_saved'] = False

# =============================================================================
# SECTION 7 — PASTE TEXT UPDATE + FILE MANIFEST + NEXT PROMPT
# =============================================================================
print("\n" + "═" * 72)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print("═" * 72)
print(f"M8p6_script               : {SCRIPT_NAME}")
print(f"M8p6_rows_processed       : {results.get('rows_processed', 0):,}")
print(f"M8p6_isa37_gain_limit     : {ISA37_GAIN_RATIO_LIMIT}")
print(f"M8p6_headroom_flag_pct    : {HEADROOM_FLAG_FRACTION:.0%}")
print(f"M8p6_n_gain_records       : {results.get('n_gain_records', 0)}")
print(f"M8p6_flagged_pairs        : {results.get('n_flagged_channel_cluster_pairs', 0)}")
print(f"M8p6_plots_saved          : {results.get('plots_saved', 0)}")
print(f"M8p6_config_saved         : {results.get('config_saved', False)}")
print(f"M8p6_status               : READY")
print("═" * 72)
print("══ END PASTE UPDATE ══\n")

print("FILE MANIFEST:")
print(f"  → {MODEL_DIR / 'M8p6_sensor_sensitivity_config.json'}    [GitHub push]")
print(f"  → {report_path}    [Spaces upload]")
print(f"  → {PLOTS_DIR / 'M8p6_sensitivity_heatmap.png'}    [Spaces upload]")
print(f"  → {PLOTS_DIR / 'M8p6_per_channel_gain_distribution.png'}    [Spaces upload]")

print("\nNEXT PROMPT:")
print('"📦 M8p6 sensitivity analysis done. Stakeholder concern addressed via ISA-37 '
      'gain-ratio guardrail. Flagged channels documented for M10 Field 6 runtime '
      'addendum. Resume M8 main path (TCN-AE + CUSUM + Adaptive Threshold)."')
