import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR
import pandas as pd
import re
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# =============================================================
# UNIVERSAL COLUMN RENAME — maps any pump-prefix to X_ standard
# Pattern: {A|B|C}_ACR_Mot.PV → X_ACR_Mot.PV etc.
# =============================================================
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        new = col
        # Strip pump prefix A_, B_, C_
        new = re.sub(r'^[ABC]_ACR_', 'X_ACR_', new)
        # Pres.PV → X_Pres.SV
        new = re.sub(r'^[ABC]_Pres\.PV$', 'X_Pres.SV', new)
        # Temp.PV → X_Temp.SV
        new = re.sub(r'^[ABC]_Temp\.PV$', 'X_Temp.SV', new)
        if new != col:
            rename_map[col] = new
    return df.rename(columns=rename_map)

# =============================================================
# FIX CORRUPTED NUMERIC STRINGS
# Handles both:
#   space-decimal  : '0 001333' → '0.001333'  (A_Day3)
#   multi-dot      : '19.015.625' → '19.015625' (C_Day3)
# =============================================================
def fix_numeric_col(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series  # already numeric, skip

    def fix_val(v):
        s = str(v).strip()
        # Space decimal: '0 001333' → '0.001333'
        s = re.sub(r'(?<=\d) (?=\d)', '.', s)
        # Multi-dot: keep only first dot as decimal
        # '19.015.625' → '19015.625' is wrong; correct is '19.015625'
        # Rule: if more than one dot, remove all dots then insert
        # one before last 3 digits (standard sensor precision)
        dots = s.count('.')
        if dots > 1:
            # Remove all dots, treat last 3+ chars as decimals
            digits = s.replace('.', '')
            # Re-insert single decimal: split at len-3 from right
            if len(digits) > 3:
                s = digits[:-3] + '.' + digits[-3:]
            else:
                s = '0.' + digits
        return s

    fixed = series.apply(fix_val)
    return pd.to_numeric(fixed, errors='coerce')

# =============================================================
# MAIN — process all 9 files
# =============================================================
STANDARD_COLS = [
    'Timestamp',
    'X_ACR_Mot.PV', 'X_ACR_Mot.SV', 'X_ACR_Mot.TV',
    'X_ACR_Pmp.PV', 'X_ACR_Pmp.SV', 'X_ACR_Pmp.TV',
    'X_Temp.SV', 'X_Pres.SV',
    'Barometer', 'Temperature'
]

raw_files = sorted(RAW_DIR.glob("Pump_*.csv"))
log(f"Found {len(raw_files)} files to fix")

for fpath in raw_files:
    log(f"\nProcessing {fpath.name}...")

    try:
        # ── Detect delimiter ──────────────────────────────
        with open(fpath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        sep = ';' if first_line.count(';') > first_line.count(',') else ','
        log(f"  Delimiter: '{sep}'")

        # ── Load ──────────────────────────────────────────
        df = pd.read_csv(fpath, sep=sep)
        log(f"  Raw columns : {df.columns.tolist()}")
        log(f"  Raw shape   : {df.shape}")

        # ── Standardise column names ──────────────────────
        df = standardise_columns(df)
        log(f"  Fixed columns: {df.columns.tolist()}")

        # ── Fix corrupted numeric strings ─────────────────
        for col in df.columns:
            if col == 'Timestamp':
                continue
            df[col] = fix_numeric_col(df[col])

        # ── Parse Timestamp ───────────────────────────────
        df['Timestamp'] = pd.to_datetime(df['Timestamp'],
                                          errors='coerce',
                                          utc=True)
        df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

        # ── Verify all standard cols present ─────────────
        missing = [c for c in STANDARD_COLS if c not in df.columns]
        if missing:
            log(f"  WARNING — missing after fix: {missing}")
        else:
            log(f"  All standard columns present ✅")

        # ── Keep only standard cols ───────────────────────
        df = df[[c for c in STANDARD_COLS if c in df.columns]]

        # ── Null check after fix ──────────────────────────
        null_counts = df.isnull().sum()
        log(f"  Null counts:\n{null_counts.to_string()}")

        # ── Overwrite with clean standard CSV ────────────
        df.to_csv(fpath, index=False)
        log(f"  ✅ Overwritten → {fpath.name} | Shape: {df.shape}")

    except Exception as e:
        log(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

log("\n" + "="*50)
log("ALL FILES PROCESSED")
log("Re-run inspect_columns.py to verify all 9 files now have identical schema")
log("="*50)
