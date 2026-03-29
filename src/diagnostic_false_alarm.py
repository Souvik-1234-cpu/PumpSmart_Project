import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NORM_DIR, CLEAN_DIR, MODEL_DIR, DEVICE, IS_GPU
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from collections import Counter
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ── Constants (must match M4 exactly) ────────────────────────────────────────
CHANNELS = [
    'X_ACR_Mot.PV_norm', 'X_ACR_Mot.SV_norm', 'X_ACR_Mot.TV_norm',
    'X_ACR_Pmp.PV_norm', 'X_ACR_Pmp.SV_norm', 'X_ACR_Pmp.TV_norm',
    'X_Temp.SV_norm',    'X_Pres.SV_norm'
]
WINDOW_SIZE   = 50
STEP_SIZE     = 10
THRESHOLD     = 0.530801
SEED          = 42
VAL_SPLIT     = 0.15
WINSOR_BOUNDS = {
    'X_Pres.SV_norm':    (0.0, 5.6376),
    'X_ACR_Mot.SV_norm': (0.0, 6.6538),
    'X_ACR_Pmp.SV_norm': (0.0, 8.8348),
    'X_ACR_Mot.PV_norm': (0.0, 2.1645),
    'X_ACR_Pmp.PV_norm': (0.0, 2.5933),
}

# ── Model definition (copy from M4 — must be identical) ──────────────────────
class LSTMEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim, bottleneck, n_layers, dropout):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden_dim, num_layers=n_layers,
                             batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim, bottleneck, num_layers=1,
                             batch_first=True)
        self.bn    = nn.LayerNorm(bottleneck)
    def forward(self, x):
        out, _      = self.lstm1(x)
        out, (h, _) = self.lstm2(out)
        return self.bn(h.squeeze(0))

class LSTMDecoder(nn.Module):
    def __init__(self, bottleneck, hidden_dim, n_features,
                 n_layers, dropout, seq_len):
        super().__init__()
        self.seq_len  = seq_len
        self.n_layers = n_layers
        self.fc_h     = nn.Linear(bottleneck, hidden_dim)
        self.fc_c     = nn.Linear(bottleneck, hidden_dim)
        self.lstm1    = nn.LSTM(bottleneck, hidden_dim, num_layers=n_layers,
                                batch_first=True, dropout=dropout)
        self.lstm2    = nn.LSTM(hidden_dim, n_features, num_layers=1,
                                batch_first=True)
        self.out      = nn.Linear(n_features, n_features)
    def forward(self, z):
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        h0    = self.fc_h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)
        c0    = self.fc_c(z).unsqueeze(0).repeat(self.n_layers, 1, 1)
        out, _ = self.lstm1(z_rep, (h0, c0))
        out, _ = self.lstm2(out)
        return self.out(out)

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim, bottleneck,
                 n_layers, dropout, seq_len):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim, bottleneck,
                                   n_layers, dropout)
        self.decoder = LSTMDecoder(bottleneck, hidden_dim, n_features,
                                   n_layers, dropout, seq_len)
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Step 1: Load + winsorize data ────────────────────────────────────────────
log("Loading M3 normalised data...")
df = pd.read_csv(NORM_DIR / "normalised_data.csv", parse_dates=['Timestamp'])
for c, (lo, hi) in WINSOR_BOUNDS.items():
    df[c] = df[c].clip(lower=lo, upper=hi)
log(f"Loaded and winsorized: {len(df):,} rows")

reg         = pd.read_csv(CLEAN_DIR / "segment_registry.csv")
warmup_map  = reg.set_index('segment_id')['warmup_rows'].to_dict()
usable_segs = reg[reg['usable_for_windowing'] == True]['segment_id'].tolist()

# ── Step 2: Rebuild windows ───────────────────────────────────────────────────
log("Rebuilding windows...")
windows, seg_ids = [], []
for seg_id in usable_segs:
    seg_df = df[df['segment_id'] == seg_id].copy()
    warmup = int(warmup_map.get(seg_id, 300))
    seg_df = seg_df.iloc[warmup:].reset_index(drop=True)
    if len(seg_df) < WINDOW_SIZE:
        continue
    data = seg_df[CHANNELS].values.astype(np.float32)
    if np.isnan(data).any():
        continue
    for i in range(0, len(seg_df) - WINDOW_SIZE + 1, STEP_SIZE):
        w = data[i : i + WINDOW_SIZE]
        if w.shape[0] == WINDOW_SIZE:
            windows.append(w)
            seg_ids.append(seg_id)

windows_arr = np.array(windows, dtype=np.float32)
log(f"Total windows: {len(windows_arr):,}")

# ── Step 3: Reproduce val split (identical seed as M4) ───────────────────────
rng     = np.random.default_rng(SEED)
idx     = rng.permutation(len(windows_arr))
split   = int(len(idx) * (1 - VAL_SPLIT))
val_idx = idx[split:]
log(f"Val windows: {len(val_idx):,}")

# ── Step 4: Load model ────────────────────────────────────────────────────────
log("Loading best model weights...")
model = LSTMAutoencoder(
    n_features=len(CHANNELS), hidden_dim=128, bottleneck=64,
    n_layers=2, dropout=0.35, seq_len=WINDOW_SIZE
).to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_DIR / "lstm_ae_baseline_best.pth", map_location='cpu')
)
model.to(DEVICE)
model.eval()
log(f"Model loaded on {DEVICE}")

# ── Step 5: Score every val window ───────────────────────────────────────────
log("Scoring val windows...")
all_maes, all_maxvals, all_segs, all_ch_maes = [], [], [], []

with torch.no_grad():
    for i_batch in range(0, len(val_idx), 256):
        batch_idx = val_idx[i_batch : i_batch + 256]
        xb        = torch.tensor(windows_arr[batch_idx]).to(DEVICE)
        with autocast(enabled=IS_GPU):
            pred = model(xb)
        err    = torch.abs(pred - xb).mean(dim=(1, 2)).cpu().numpy()
        ch_err = torch.abs(pred - xb).mean(dim=1).cpu().numpy()  # (B, 8)
        for j, (mae, orig_idx) in enumerate(zip(err, batch_idx)):
            all_maes.append(float(mae))
            all_maxvals.append(float(windows_arr[orig_idx].max()))
            all_segs.append(seg_ids[orig_idx])
            all_ch_maes.append(ch_err[j].tolist())

all_maes    = np.array(all_maes)
all_maxvals = np.array(all_maxvals)
all_ch_maes = np.array(all_ch_maes)
fa_mask     = all_maes > THRESHOLD
fa_idx      = np.where(fa_mask)[0]
log(f"False alarms: {fa_mask.sum()} / {len(all_maes)}")

# ── Step 6: Forensics report ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"FALSE ALARM FORENSICS  —  {fa_mask.sum()} windows above {THRESHOLD}")
print(f"{'='*65}")

SHORT = ['Mot.PV','Mot.SV','Mot.TV','Pmp.PV','Pmp.SV','Pmp.TV','Temp.SV','Pres.SV']

sorted_fa = fa_idx[np.argsort(all_maes[fa_idx])[::-1]]
print(f"\n{'Rank':<5} {'Segment':<22} {'MAE':>7} {'WinMax':>7}  "
      f"{'Dominant Channel':<14}  Verdict")
print("-" * 80)
for rank, i in enumerate(sorted_fa, 1):
    mae     = all_maes[i]
    wmax    = all_maxvals[i]
    dom_ch  = SHORT[int(np.argmax(all_ch_maes[i]))]
    if wmax > 5.0:
        verdict = "SPIKE ESCAPED WINSOR"
    elif wmax > 2.5:
        verdict = "HIGH-LOAD BOUNDARY"
    elif mae > THRESHOLD * 1.5:
        verdict = "GENUINE ANOMALY PATTERN"
    else:
        verdict = "MARGINAL — near threshold"
    print(f"{rank:<5} {all_segs[i]:<22} {mae:>7.4f} {wmax:>7.4f}  "
          f"{dom_ch:<14}  {verdict}")

print(f"\n── Segments contributing false alarms ──")
seg_counts = Counter([all_segs[i] for i in fa_idx])
for seg, count in seg_counts.most_common():
    print(f"  {seg}: {count} window(s)")

print(f"\n── False alarm window-max distribution ──")
fa_maxvals = all_maxvals[fa_idx]
print(f"  Min:  {fa_maxvals.min():.4f}")
print(f"  Max:  {fa_maxvals.max():.4f}")
print(f"  Mean: {fa_maxvals.mean():.4f}")
print(f"  Windows with max > 5.0 (spike escaped): {(fa_maxvals > 5.0).sum()}")
print(f"  Windows with max 2.5–5.0 (high-load):   {((fa_maxvals > 2.5) & (fa_maxvals <= 5.0)).sum()}")
print(f"  Windows with max < 2.5  (true anomaly):  {(fa_maxvals < 2.5).sum()}")

print(f"\n── Dominant error channel in false alarms ──")
dom_channels = [SHORT[int(np.argmax(all_ch_maes[i]))] for i in fa_idx]
for ch, count in Counter(dom_channels).most_common():
    print(f"  {ch}: {count} windows")

print(f"\n── DECISION ──────────────────────────────────────────────")
n_spike   = int((fa_maxvals > 5.0).sum())
n_hiload  = int(((fa_maxvals > 2.5) & (fa_maxvals <= 5.0)).sum())
n_genuine = int((fa_maxvals < 2.5).sum())
if n_spike > 0:
    print(f"  {n_spike} spike-escaped windows → tighten winsor OR raise threshold")
if n_hiload > 0:
    print(f"  {n_hiload} high-load boundary windows → raise threshold to P99.5")
if n_genuine > 0:
    print(f"  {n_genuine} genuine anomaly patterns → DO NOT fix, model is correctly sensitive")
print(f"{'='*65}")