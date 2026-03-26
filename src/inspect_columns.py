import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR
import pandas as pd

raw_files = sorted(RAW_DIR.glob("Pump_*.csv"))
for fpath in raw_files:
    try:
        df = pd.read_csv(fpath, nrows=2)
        print(f"\n{'='*50}")
        print(f"FILE : {fpath.name}")
        print(f"COLS : {df.columns.tolist()}")
        print(f"ROW1 : {df.iloc[0].tolist()}")
    except Exception as e:
        print(f"\n{fpath.name} → ERROR: {e}")
