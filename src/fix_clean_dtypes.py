import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLEAN_DIR
import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

SENSOR_COLS = [
    'X_ACR_Mot.PV', 'X_ACR_Mot.SV', 'X_ACR_Mot.TV',
    'X_ACR_Pmp.PV', 'X_ACR_Pmp.SV', 'X_ACR_Pmp.TV',
    'X_Temp.SV', 'X_Pres.SV', 'Barometer', 'Temperature'
]

clean_files = sorted(CLEAN_DIR.glob("Pump_*_clean.csv"))
log(f"Found {len(clean_files)} clean CSVs to verify/fix")

for fpath in clean_files:
    df = pd.read_csv(fpath, dtype=str)
    needs_fix = False
    fixed_cols = []

    for col in SENSOR_COLS:
        if col not in df.columns:
            continue
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ''
        try:
            float(str(sample).replace(',', '.'))
            # Try direct conversion
            test = pd.to_numeric(df[col], errors='coerce')
            if test.isnull().sum() > df[col].isnull().sum():
                # Conversion introduced new nulls — string corruption present
                needs_fix = True
                fixed_cols.append(col)
        except:
            needs_fix = True
            fixed_cols.append(col)

    if needs_fix:
        log(f"  {fpath.name} — fixing cols: {fixed_cols}")
        for col in SENSOR_COLS:
            if col not in df.columns:
                continue
            df[col] = (df[col].astype(str)
                               .str.strip()
                               .str.replace(',', '.', regex=False)
                               # Remove all but the first dot
                               .apply(lambda x: (
                                   x.split('.')[0] + '.' +
                                   ''.join(x.split('.')[1:])
                                   if x.count('.') > 1 else x
                               )))
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        pre  = len(df)
        df   = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)
        post = len(df)
        log(f"    Dropped {pre-post} unfixable rows | "
            f"Final shape: {df.shape}")
        df.to_csv(fpath, index=False)
        log(f"    ✅ Fixed and saved → {fpath.name}")
    else:
        log(f"  {fpath.name} — ✅ All numeric, no fix needed")

log("\nAll clean CSVs verified. Re-run module_02_eda_clustering.py")
