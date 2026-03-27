import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLEAN_DIR
import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

log("Flagging post-dropout segments in registry...")
reg = pd.read_csv(CLEAN_DIR / "segment_registry.csv",
                  parse_dates=['start_time', 'end_time'])

# Sort by pump + start time
reg = reg.sort_values(['pump_id', 'day_id', 'start_time']).reset_index(drop=True)

# For each consecutive segment pair within same pump+day,
# compute the gap between end of prev segment and start of next
reg['prev_end']     = reg.groupby(['pump_id','day_id'])['end_time'].shift(1)
reg['gap_to_prev_s']= (reg['start_time'] - reg['prev_end']
                       ).dt.total_seconds().fillna(0)

# A post-dropout segment is one where gap > 600s (10 min)
# meaning the pump ran for a long time without env sensor data
LONG_GAP_THRESHOLD  = 600  # seconds

reg['post_dropout'] = reg['gap_to_prev_s'] > LONG_GAP_THRESHOLD

# Warmup rows: standard=300, post_dropout=600
reg['warmup_rows']  = reg['post_dropout'].apply(
    lambda x: 600 if x else 300
)

# Drop helper column
reg = reg.drop(columns=['prev_end'])

reg.to_csv(CLEAN_DIR / "segment_registry.csv", index=False)

log("Updated segment registry:")
log(f"  Standard segments  (warmup=300): "
    f"{(reg['warmup_rows']==300).sum()}")
log(f"  Post-dropout segs  (warmup=600): "
    f"{(reg['warmup_rows']==600).sum()}")
log("\nPost-dropout segments flagged:")
flagged = reg[reg['post_dropout'] == True][
    ['segment_id','pump_id','day_id','gap_to_prev_s','warmup_rows']
]
print(flagged.to_string(index=False))
log("\n✅ segment_registry.csv updated with warmup_rows column")
log("M4 will use warmup_rows per segment during window generation")
