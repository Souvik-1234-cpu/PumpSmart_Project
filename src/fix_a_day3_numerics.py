import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR
import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

fpath = RAW_DIR / "Pump_A_Day3.csv"
log(f"Loading {fpath.name}...")

# Load with dtype=str to force everything as string first
df = pd.read_csv(fpath, dtype=str)

log(f"Shape  : {df.shape}")
log(f"Sample raw value X_ACR_Mot.PV[0]: {repr(df['X_ACR_Mot.PV'].iloc[0])}")

# =============================================================
# FIX: comma-decimal strings → float
# '0,001333523' → 0.001333523
# Works regardless of pandas dtype reporting (str vs object)
# =============================================================
SENSOR_COLS = [c for c in df.columns if c != 'Timestamp']

log("Converting comma-decimal strings to float64...")
for col in SENSOR_COLS:
    df[col] = (df[col].astype(str)
                       .str.strip()
                       .str.strip('"')
                       .str.strip("'")
                       .str.replace(',', '.', regex=False))
    df[col] = pd.to_numeric(df[col], errors='coerce')
    log(f"  {col}: {df[col].dtype} | "
        f"nulls={df[col].isnull().sum()} | "
        f"sample={df[col].iloc[0]}")

# Parse Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Verify
all_numeric = all(pd.api.types.is_numeric_dtype(df[c]) for c in SENSOR_COLS)
log(f"\nAll numeric: {'✅ YES' if all_numeric else '❌ NO'}")
log(f"Dtypes:\n{df.dtypes.to_string()}")

# Overwrite
df.to_csv(fpath, index=False)
log(f"\n✅ Pump_A_Day3.csv fixed and saved → {fpath}")
