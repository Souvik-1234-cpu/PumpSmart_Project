import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, CLEAN_DIR
import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

SENSOR_COLS = [
    'X_ACR_Mot.PV', 'X_ACR_Mot.SV', 'X_ACR_Mot.TV',
    'X_ACR_Pmp.PV', 'X_ACR_Pmp.SV', 'X_ACR_Pmp.TV',
    'X_Temp.SV', 'X_Pres.SV', 'Barometer', 'Temperature'
]

log("Loading Pump_C_Day3 raw as pure strings...")
raw = pd.read_csv(RAW_DIR / "Pump_C_Day3.csv",
                  dtype=str, sep=None, engine='python')

log(f"Raw shape: {raw.shape}")
print()

# For each sensor col, count how many rows have multi-dot corruption
print("=== Per-column corruption count ===")
corruption_counts = {}
for col in SENSOR_COLS:
    if col not in raw.columns:
        log(f"  {col}: NOT FOUND in raw file")
        continue
    total     = len(raw[col].dropna())
    n_corrupt = raw[col].dropna().apply(
        lambda x: str(x).count('.') > 1
    ).sum()
    pct = round(n_corrupt / total * 100, 2) if total > 0 else 0
    corruption_counts[col] = n_corrupt
    status = "🔴 CORRUPTED" if n_corrupt > 0 else "✅ Clean"
    print(f"  {col:<20}: {n_corrupt:>6} / {total} rows corrupt "
          f"({pct}%) {status}")

print()
total_rows    = len(raw)
# A row is corrupt if ANY sensor column has multi-dot value
any_corrupt_mask = pd.Series([False] * total_rows)
for col in SENSOR_COLS:
    if col not in raw.columns:
        continue
    col_corrupt = raw[col].fillna('').apply(
        lambda x: str(x).count('.') > 1
    )
    any_corrupt_mask = any_corrupt_mask | col_corrupt

n_any_corrupt  = any_corrupt_mask.sum()
n_fully_clean  = total_rows - n_any_corrupt
pct_corrupt    = round(n_any_corrupt / total_rows * 100, 2)
pct_clean      = round(n_fully_clean / total_rows * 100, 2)

print("=== Row-level summary ===")
print(f"  Total raw rows        : {total_rows:,}")
print(f"  Rows with ANY corrupt : {n_any_corrupt:,} ({pct_corrupt}%)")
print(f"  Fully clean rows      : {n_fully_clean:,} ({pct_clean}%)")
print()

# Show a few examples of corrupt values
print("=== Sample corrupt values ===")
for col in SENSOR_COLS:
    if col not in raw.columns:
        continue
    examples = raw[col][raw[col].fillna('').apply(
        lambda x: str(x).count('.') > 1
    )].head(3).tolist()
    if examples:
        print(f"  {col}: {examples}")

print()
# Check what the clean CSV has now
log("Checking current clean CSV...")
clean = pd.read_csv(CLEAN_DIR / "Pump_C_Day3_clean.csv")
log(f"Clean shape: {clean.shape}")
print()
print("=== Clean CSV dtype check ===")
for col in SENSOR_COLS:
    if col not in clean.columns:
        continue
    print(f"  {col:<20}: dtype={clean[col].dtype} | "
          f"min={clean[col].min():.4f} | "
          f"max={clean[col].max():.4f} | "
          f"nulls={clean[col].isnull().sum()}")

print()
print("=== VERDICT ===")
if pct_corrupt > 5:
    print(f"  🔴 {pct_corrupt}% of Pump_C_Day3 rows had corruption.")
    print("  These were 'recovered' by fix_clean_dtypes.py using last-dot rule.")
    print("  Recommendation: DROP all recovered rows, keep only originally")
    print("  clean rows. Re-run M1 segmentation on C_Day3 only.")
else:
    print(f"  ✅ Only {pct_corrupt}% corrupt — minimal impact on training.")
