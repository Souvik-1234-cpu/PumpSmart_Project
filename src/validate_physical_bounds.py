import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, CLEAN_DIR, OUTPUT_DIR, PLOTS_DIR
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

SCRIPT_NAME = "validate_physical_bounds"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# =============================================================
# PHYSICS-DERIVED HARD BOUNDS
# Nameplate: 10kW, 45m³/h, 450m head, 40bar, 7 impellers, 2980RPM
# Cross-validated against Pump_C_Day2 (cleanest file)
# =============================================================
PHYSICS_BOUNDS = {
    # (hard_min, hard_max, warn_min, warn_max, unit, physical_basis)
    'X_ACR_Mot.PV': (
        0.0,      0.020,    # hard: displacement physically 0–20mm/s
        0.0001,   0.010,    # warn: normal operating range
        'm/s',    'Motor accelerometer displacement — ISO 10816 limit ~4.5mm/s RMS'
    ),
    'X_ACR_Mot.SV': (
        0.0,      600.0,    # hard: peak g-force — accelerometer max range
        0.1,      100.0,    # warn: normal operation
        'm/s²',   'Motor vibration peak — severe fault ~100m/s²'
    ),
    'X_ACR_Mot.TV': (
        0.0,      350.0,    # hard: above 350°C → sensor burnout
        1.0,      200.0,    # warn: motor winding limit ~180°C
        '°C',     'Motor contact temp — IEC motor class F limit 155°C'
    ),
    'X_ACR_Pmp.PV': (
        0.0,      0.020,
        0.0001,   0.010,
        'm/s',    'Pump accelerometer displacement'
    ),
    'X_ACR_Pmp.SV': (
        0.0,      600.0,
        0.1,      100.0,
        'm/s²',   'Pump vibration peak'
    ),
    'X_ACR_Pmp.TV': (
        0.0,      350.0,
        1.0,      200.0,
        '°C',     'Pump contact temp — fluid boiling + bearing limit'
    ),
    'X_Temp.SV': (
        0.0,      400.0,    # hard: above 400°C → physically impossible
        1.0,      250.0,    # warn: motor casing — severe overload
        '°C',     'Motor casing temp — severe overload ~200°C'
    ),
    'X_Pres.SV': (
        0.0,      50.0,     # hard: nameplate 40bar max + 25% safety margin
        0.1,      42.0,     # warn: above nameplate = overpressure fault
        'bar',    'Outlet pressure — nameplate max 40 bar'
    ),
    'Barometer': (
        800.0,    1100.0,   # hard: sea level ±200 mbar covers all altitudes
        950.0,    1050.0,   # warn: normal weather range
        'mbar',   'Atmospheric pressure — physical range 870–1084 mbar globally'
    ),
    'Temperature': (
        -10.0,    60.0,     # hard: Italy ambient — below -10 or above 60 impossible
        5.0,      45.0,     # warn: normal lab/industrial ambient
        '°C',     'Ambient temperature — CIRA facility in Italy'
    ),
}

SENSOR_COLS = list(PHYSICS_BOUNDS.keys())

# =============================================================
# STEP 1 — Load all clean CSVs
# =============================================================
log("STEP 1 — Loading all clean CSVs...")
clean_files = sorted(CLEAN_DIR.glob("Pump_*_clean.csv"))
all_results = {}
grand_violations = []

for fpath in clean_files:
    try:
        df = pd.read_csv(fpath)
        file_label = fpath.stem.replace('_clean', '')
        log(f"  {fpath.name}: {len(df):,} rows loaded")
        all_results[file_label] = {'df': df, 'violations': {}}
    except Exception as e:
        log(f"  ERROR loading {fpath.name}: {e}")

# =============================================================
# STEP 2 — Per-column string corruption check (multi-dot)
# =============================================================
log("\nSTEP 2 — String corruption check on clean CSVs...")
print()
print("=== STRING CORRUPTION CHECK ===")

any_string_corruption = False
for label, data in all_results.items():
    df = data['df']
    for col in SENSOR_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            n_corrupt = df[col].fillna('').apply(
                lambda x: str(x).count('.') > 1
            ).sum()
            if n_corrupt > 0:
                pct = round(n_corrupt / len(df) * 100, 2)
                print(f"  🔴 {label} | {col}: "
                      f"{n_corrupt} string-corrupt rows ({pct}%)")
                any_string_corruption = True
                grand_violations.append({
                    'file': label, 'col': col,
                    'type': 'string_corrupt',
                    'n_rows': n_corrupt, 'pct': pct,
                    'example': 'multi-dot string'
                })

if not any_string_corruption:
    print("  ✅ No string corruption found in any clean CSV")

# =============================================================
# STEP 3 — Physical bounds violation check
# =============================================================
log("\nSTEP 3 — Physical bounds validation...")
print()
print("=== PHYSICAL BOUNDS VIOLATIONS ===")
print(f"{'File':<20} {'Column':<20} {'Type':<10} "
      f"{'N_rows':>8} {'Pct':>7} {'Min_val':>12} {'Max_val':>12}")
print("-" * 95)

summary_rows = []

for label, data in all_results.items():
    df   = data['df']
    file_clean = True

    for col in SENSOR_COLS:
        if col not in df.columns:
            continue

        # Force numeric
        series = pd.to_numeric(df[col], errors='coerce')
        n_null = series.isnull().sum()

        h_min, h_max, w_min, w_max, unit, basis = PHYSICS_BOUNDS[col]

        # Hard violations — physically impossible
        hard_mask = (series < h_min) | (series > h_max)
        n_hard    = hard_mask.sum() - n_null  # don't double count nulls
        n_hard    = max(0, n_hard)

        # Warn violations — outside normal but not impossible
        warn_mask = ((series < w_min) | (series > w_max)) & ~hard_mask
        n_warn    = warn_mask.sum()

        pct_hard = round(n_hard / len(df) * 100, 3) if len(df) > 0 else 0
        pct_warn = round(n_warn / len(df) * 100, 3) if len(df) > 0 else 0

        val_min  = round(series.min(), 4) if not series.isnull().all() else None
        val_max  = round(series.max(), 4) if not series.isnull().all() else None

        if n_hard > 0:
            print(f"  🔴 {label:<18} {col:<20} {'HARD':<10} "
                  f"{n_hard:>8} {pct_hard:>6}% "
                  f"{val_min:>12} {val_max:>12}")
            file_clean = False
            grand_violations.append({
                'file': label, 'col': col, 'type': 'HARD',
                'n_rows': n_hard, 'pct': pct_hard,
                'val_min': val_min, 'val_max': val_max,
                'hard_min': h_min, 'hard_max': h_max,
                'basis': basis
            })

        if n_warn > 0:
            print(f"  ⚠️  {label:<18} {col:<20} {'WARN':<10} "
                  f"{n_warn:>8} {pct_warn:>6}% "
                  f"{val_min:>12} {val_max:>12}")
            grand_violations.append({
                'file': label, 'col': col, 'type': 'WARN',
                'n_rows': n_warn, 'pct': pct_warn,
                'val_min': val_min, 'val_max': val_max,
                'warn_min': w_min, 'warn_max': w_max,
                'basis': basis
            })

        summary_rows.append({
            'file'    : label,
            'col'     : col,
            'dtype'   : str(df[col].dtype),
            'n_rows'  : len(df),
            'n_null'  : n_null,
            'val_min' : val_min,
            'val_max' : val_max,
            'hard_min': h_min,
            'hard_max': h_max,
            'n_hard'  : n_hard,
            'pct_hard': pct_hard,
            'n_warn'  : n_warn,
            'pct_warn': pct_warn,
            'status'  : 'HARD' if n_hard > 0 else
                        'WARN' if n_warn > 0 else 'OK'
        })

    if file_clean:
        log(f"  ✅ {label}: all columns within physical bounds")

# =============================================================
# STEP 4 — Cross-file consistency check
# =============================================================
log("\nSTEP 4 — Cross-file sensor range consistency check...")
print()
print("=== CROSS-FILE SENSOR RANGES ===")
print(f"{'Column':<20}", end='')
for label in all_results.keys():
    short = label.replace('Pump_', '').replace('_clean', '')
    print(f" {short:>14}", end='')
print()
print("-" * (20 + 15 * len(all_results)))

for col in SENSOR_COLS:
    print(f"  {col:<18}", end='')
    for label, data in all_results.items():
        df = data['df']
        if col not in df.columns:
            print(f" {'N/A':>14}", end='')
            continue
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) == 0:
            print(f" {'EMPTY':>14}", end='')
        else:
            print(f" {series.mean():>6.3f}±{series.std():>5.3f}", end='')
    print()

# =============================================================
# STEP 5 — Decide: which files need reprocessing
# =============================================================
print()
print("=== REPROCESSING DECISION ===")
hard_by_file = {}
for v in grand_violations:
    if v['type'] == 'HARD':
        f = v['file']
        hard_by_file[f] = hard_by_file.get(f, 0) + v['n_rows']

if not hard_by_file:
    print("  ✅ No HARD violations in any file.")
    print("  ✅ All clean CSVs are physically valid.")
    print("  ✅ SAFE TO PROCEED TO M3.")
else:
    for f, n in hard_by_file.items():
        total = len(all_results[f]['df'])
        pct   = round(n / total * 100, 2)
        print(f"  🔴 {f}: {n} HARD-violation rows ({pct}%)")
        if pct > 5:
            print(f"     → ACTION REQUIRED: Drop these rows and "
                  f"re-segment before M3")
        else:
            print(f"     → MINOR (<5%): Drop rows in-place, "
                  f"proceed to M3")

# =============================================================
# STEP 6 — Auto-fix: drop all HARD-violation rows from clean CSVs
# =============================================================
print()
log("STEP 6 — Auto-fixing: dropping HARD-violation rows from clean CSVs...")

fixed_files = []
for label, data in all_results.items():
    df        = data['df'].copy()
    pre_rows  = len(df)
    drop_mask = pd.Series([False] * len(df), index=df.index)

    for col in SENSOR_COLS:
        if col not in df.columns:
            continue
        series    = pd.to_numeric(df[col], errors='coerce')
        h_min, h_max = PHYSICS_BOUNDS[col][0], PHYSICS_BOUNDS[col][1]
        hard_mask = (series < h_min) | (series > h_max) | series.isnull()
        drop_mask = drop_mask | hard_mask

    df_fixed   = df[~drop_mask].reset_index(drop=True)
    post_rows  = len(df_fixed)
    n_dropped  = pre_rows - post_rows
    pct_dropped= round(n_dropped / pre_rows * 100, 3)

    fpath = CLEAN_DIR / f"{label}_clean.csv"
    df_fixed.to_csv(fpath, index=False)

    if n_dropped > 0:
        log(f"  🔧 {label}: dropped {n_dropped} HARD rows "
            f"({pct_dropped}%) → {post_rows:,} rows remain")
        fixed_files.append(label)
    else:
        log(f"  ✅ {label}: no rows dropped")

# =============================================================
# STEP 7 — Plot: violation heatmap across all files
# =============================================================
log("\nSTEP 7 — Generating validation heatmap...")
try:
    summary_df = pd.DataFrame(summary_rows)

    # Pivot: files × sensors, value = pct_hard
    pivot_hard = summary_df.pivot_table(
        index='file', columns='col',
        values='pct_hard', aggfunc='sum'
    ).fillna(0)

    pivot_warn = summary_df.pivot_table(
        index='file', columns='col',
        values='pct_warn', aggfunc='sum'
    ).fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Physical Bounds Validation — All Clean CSVs',
                 fontsize=13, fontweight='bold')

    import seaborn as sns

    sns.heatmap(pivot_hard, ax=axes[0], annot=True, fmt='.2f',
                cmap='Reds', linewidths=0.5,
                cbar_kws={'label': '% rows HARD violation'})
    axes[0].set_title('🔴 HARD Violations (physically impossible values)',
                      fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=30)

    sns.heatmap(pivot_warn, ax=axes[1], annot=True, fmt='.2f',
                cmap='Oranges', linewidths=0.5,
                cbar_kws={'label': '% rows WARN violation'})
    axes[1].set_title('⚠️  WARN Violations (outside normal range)',
                      fontweight='bold')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    out_path = PLOTS_DIR / "M1_physical_validation.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved → {out_path.name}")
except Exception as e:
    log(f"  WARNING: Plot failed: {e}")

# =============================================================
# STEP 8 — Save violation report CSV
# =============================================================
violations_df = pd.DataFrame(grand_violations)
viol_path     = OUTPUT_DIR / "physical_violations_report.csv"
if len(violations_df) > 0:
    violations_df.to_csv(viol_path, index=False)
    log(f"  Violation report saved → {viol_path.name}")
else:
    log("  No violations found — no report needed")

# =============================================================
# PASTE TEXT
# =============================================================
n_hard_total = sum(1 for v in grand_violations if v['type'] == 'HARD')
n_warn_total = sum(1 for v in grand_violations if v['type'] == 'WARN')
n_fixed      = len(fixed_files)

print()
print("═"*60)
print("  VALIDATION SUMMARY")
print("═"*60)
print(f"  Total HARD violation entries : {n_hard_total}")
print(f"  Total WARN violation entries : {n_warn_total}")
print(f"  Files auto-fixed             : {n_fixed}")
if fixed_files:
    for f in fixed_files:
        print(f"    - {f}")
if n_hard_total == 0:
    print("  STATUS: ✅ ALL FILES PHYSICALLY VALID — SAFE FOR M3")
else:
    print("  STATUS: 🔧 HARD violations dropped — re-check before M3")
print("═"*60)

# FILE MANIFEST
print()
print("── FILE MANIFEST ──────────────────────────────────────────")
print("→ GitHub push:")
for f in sorted(CLEAN_DIR.glob("Pump_*_clean.csv")):
    print(f"    {f}")
print(f"    {viol_path}")
print(f"    {PLOTS_DIR / 'M1_physical_validation.png'}")
print("───────────────────────────────────────────────────────────")
