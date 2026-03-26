import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR
import pandas as pd

raw_files = sorted(RAW_DIR.glob("Pump_*.csv"))
print(f"{'File':<22} {'Rows':>7} {'Median Δt':>10} {'Min Δt':>8} {'Max Δt':>10} {'Mode Δt':>8}")
print("-"*70)
for fpath in raw_files:
    df = pd.read_csv(fpath, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp')
    deltas = df['Timestamp'].diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    print(f"{fpath.name:<22} {len(df):>7} "
          f"{deltas.median():>10.1f}s "
          f"{deltas.min():>8.1f}s "
          f"{deltas.max():>10.1f}s "
          f"{deltas.mode()[0]:>8.1f}s")
