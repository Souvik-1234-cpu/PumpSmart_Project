import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLEAN_DIR
import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Mark all C_Day3 segments as unusable in registry
reg_path = CLEAN_DIR / "segment_registry.csv"
reg = pd.read_csv(reg_path)

before = reg['usable_for_windowing'].sum()
c_day3_mask = reg['source_file'] == 'Pump_C_Day3.csv'
reg.loc[c_day3_mask, 'usable_for_windowing'] = False
after = reg['usable_for_windowing'].sum()

reg.to_csv(reg_path, index=False)
log(f"Segment registry updated:")
log(f"  C_Day3 segments flagged unusable: {c_day3_mask.sum()}")
log(f"  Usable segments: {before} → {after}")
log(f"  Saved → segment_registry.csv")

# Also write an empty but valid C_Day3 clean CSV 
# (with headers only so downstream code doesn't crash on missing file)
clean_path = CLEAN_DIR / "Pump_C_Day3_clean.csv"
cols = ['Timestamp','X_ACR_Mot.PV','X_ACR_Mot.SV','X_ACR_Mot.TV',
        'X_ACR_Pmp.PV','X_ACR_Pmp.SV','X_ACR_Pmp.TV',
        'X_Temp.SV','X_Pres.SV','Barometer','Temperature','segment_id']
pd.DataFrame(columns=cols).to_csv(clean_path, index=False)
log(f"  Pump_C_Day3_clean.csv → replaced with empty header-only file")
log(f"\n✅ C_Day3 excluded. Rerun module_02_eda_clustering.py now.")
