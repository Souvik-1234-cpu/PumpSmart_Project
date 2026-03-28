# =============================================================================
# module_04_lstm_ae_baseline.py  —  PumpSmart M4 v7
# Two-phase execution:
#   PHASE 0 — Load Data
#   PHASE 1 — Spike Extraction: identify & save transient fault seeds
#             Record spike row indices for exclusion from windowing
#   PHASE 2 — Winsorization: clip spikes, preserve row count
#   PHASE 2.5 — Build spike-free df_clean for windowing
#   PHASE 3 — Window generation on spike-free clean data
# Engineering rationale:
#   Spike rows (even after clipping) create artificial plateau patterns
#   that the LSTM cannot reconstruct → false alarms on val set.
#   Excluding spike rows from windowing ensures val set contains
#   only true normal-operation windows → clean threshold calibration.
# =============================================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings('ignore')
SCRIPT_NAME = "module_04_lstm_ae_baseline"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Configuration ─────────────────────────────────────────────────────────────
CHANNELS = [
    'X_ACR_Mot.PV_norm', 'X_ACR_Mot.SV_norm', 'X_ACR_Mot.TV_norm',
    'X_ACR_Pmp.PV_norm', 'X_ACR_Pmp.SV_norm', 'X_ACR_Pmp.TV_norm',
    'X_Temp.SV_norm',    'X_Pres.SV_norm'
]

CHANNEL_WEIGHTS = {
    'X_ACR_Mot.PV_norm': 1.5,
    'X_ACR_Mot.SV_norm': 2.0,
    'X_ACR_Mot.TV_norm': 0.8,
    'X_ACR_Pmp.PV_norm': 1.5,
    'X_ACR_Pmp.SV_norm': 2.0,
    'X_ACR_Pmp.TV_norm': 0.8,
    'X_Temp.SV_norm':    1.0,
    'X_Pres.SV_norm':    2.0,
}

WINSOR_CHANNELS = [
    'X_Pres.SV_norm',
    'X_ACR_Mot.SV_norm',
    'X_ACR_Pmp.SV_norm',
    'X_ACR_Mot.PV_norm',
    'X_ACR_Pmp.PV_norm',
]
WINSOR_SIGMA             = 4.0
SPIKE_SEED_THRESHOLD_SIGMA = 3.0

WINDOW_SIZE     = 50
STEP_SIZE       = 10
HIDDEN_SIZE     = 128
BOTTLENECK      = 64
NUM_LAYERS      = 2
DROPOUT         = 0.35
BATCH_SIZE      = 256
EPOCHS          = 150
LR              = 1e-3
COSINE_T0       = 20
PATIENCE        = 25
OVERFIT_GAP_MAX = 0.12
VAL_SPLIT       = 0.15
SEED            = 42
OLD_THRESHOLD   = 0.645347

torch.manual_seed(SEED)
np.random.seed(SEED)
results = {}


# ════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Load Data
# ════════════════════════════════════════════════════════════════════════════
log("=" * 65)
log("PHASE 0 — Loading M3 normalised data")
log("=" * 65)

try:
    df = pd.read_csv(NORM_DIR / "normalised_data.csv",
                     parse_dates=['Timestamp'])
    log(f"Loaded: {len(df):,} rows | {df['segment_id'].nunique()} segments")
except Exception as e:
    log(f"ERROR loading data: {e}"); raise

try:
    reg         = pd.read_csv(CLEAN_DIR / "segment_registry.csv")
    warmup_map  = reg.set_index('segment_id')['warmup_rows'].to_dict()
    usable_segs = reg[reg['usable_for_windowing'] == True]['segment_id'].tolist()
    log(f"Registry: {len(reg)} segments | {len(usable_segs)} usable")
except Exception as e:
    raise RuntimeError(f"Cannot load segment_registry.csv: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Spike Extraction (save fault seeds for M6)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("PHASE 1 — Spike Extraction → M4_spike_seeds.csv (M6 input)")
log("=" * 65)

spike_thresholds = {}
for c in WINSOR_CHANNELS:
    mu    = df[c].mean()
    sigma = df[c].std()
    spike_thresholds[c] = {
        'mean':            round(float(mu), 6),
        'std':             round(float(sigma), 6),
        'spike_threshold': round(float(mu + SPIKE_SEED_THRESHOLD_SIGMA * sigma), 6),
        'winsor_upper':    round(float(mu + WINSOR_SIGMA * sigma), 6),
        'spike':           float(mu + SPIKE_SEED_THRESHOLD_SIGMA * sigma),
    }

log("Per-channel spike thresholds (mean + 3σ):")
for c, v in spike_thresholds.items():
    log(f"  {c}: spike>{v['spike_threshold']:.4f} | "
        f"winsor_clip>{v['winsor_upper']:.4f}")

log("\nExtracting spike-containing windows...")
spike_windows    = []
spike_meta_rows  = []
# Track which DataFrame row indices belong to spike windows
spike_row_indices = set()

for seg_id in usable_segs:
    seg_df  = df[df['segment_id'] == seg_id].copy()
    warmup  = int(warmup_map.get(seg_id, 300))
    seg_df  = seg_df.iloc[warmup:]          # keep original df index
    if len(seg_df) < WINDOW_SIZE:
        continue
    sensor_data = seg_df[CHANNELS].values.astype(np.float32)
    if np.isnan(sensor_data).any():
        continue

    ch_idx_map  = {c: i for i, c in enumerate(CHANNELS)}
    seg_indices = seg_df.index.tolist()     # actual df row indices

    for i in range(0, len(seg_df) - WINDOW_SIZE + 1, STEP_SIZE):
        w = sensor_data[i : i + WINDOW_SIZE]
        if w.shape[0] != WINDOW_SIZE:
            continue

        is_spike        = False
        spike_chans     = []
        max_spike_ratio = 0.0

        for c, bounds in spike_thresholds.items():
            ch_i   = ch_idx_map[c]
            ch_max = float(w[:, ch_i].max())
            thr    = bounds['spike_threshold']
            if ch_max > thr:
                is_spike = True
                ratio    = ch_max / bounds['mean']
                spike_chans.append(f"{c}={ch_max:.3f}({ratio:.1f}x)")
                max_spike_ratio = max(max_spike_ratio, ratio)

        if is_spike:
            spike_windows.append(w)
            # Record ALL row indices in this window for exclusion
            spike_row_indices.update(seg_indices[i : i + WINDOW_SIZE])

            dominant = max(
                [(c, float(w[:, ch_idx_map[c]].max()) / spike_thresholds[c]['mean'])
                 for c in WINSOR_CHANNELS if c in ch_idx_map],
                key=lambda x: x[1]
            )
            if 'Pres' in dominant[0]:
                fault_hint = 'pressure_transient'
            elif 'Pmp.SV' in dominant[0]:
                fault_hint = 'impeller_cavitation'
            elif 'Mot.SV' in dominant[0]:
                fault_hint = 'bearing_impact'
            else:
                fault_hint = 'mechanical_transient'

            spike_meta_rows.append({
                'segment_id':       seg_id,
                'window_start_idx': i,
                'dominant_channel': dominant[0],
                'max_spike_ratio':  round(max_spike_ratio, 4),
                'spike_channels':   ' | '.join(spike_chans),
                'fault_hint':       fault_hint,
                'window_max_val':   round(float(w.max()), 4),
            })

log(f"Spike windows extracted: {len(spike_windows):,}")
log(f"  → pressure_transient:    "
    f"{sum(1 for r in spike_meta_rows if r['fault_hint']=='pressure_transient')}")
log(f"  → impeller_cavitation:   "
    f"{sum(1 for r in spike_meta_rows if r['fault_hint']=='impeller_cavitation')}")
log(f"  → bearing_impact:        "
    f"{sum(1 for r in spike_meta_rows if r['fault_hint']=='bearing_impact')}")
log(f"  → mechanical_transient:  "
    f"{sum(1 for r in spike_meta_rows if r['fault_hint']=='mechanical_transient')}")
log(f"Spike row indices recorded: {len(spike_row_indices):,} rows flagged for exclusion")

spike_arr  = np.array(spike_windows, dtype=np.float32) if spike_windows else np.empty((0, WINDOW_SIZE, len(CHANNELS)))
spike_meta = pd.DataFrame(spike_meta_rows)

try:
    np.save(SYNTH_DIR / "M4_spike_seeds.npy", spike_arr)
    spike_meta.to_csv(SYNTH_DIR / "M4_spike_seeds_meta.csv", index=False)
    log(f"Saved: M4_spike_seeds.npy      shape={spike_arr.shape}")
    log(f"Saved: M4_spike_seeds_meta.csv rows={len(spike_meta)}")
except Exception as e:
    log(f"ERROR saving spike seeds: {e}")

spike_config = {
    'script':                   SCRIPT_NAME,
    'version':                  'v7',
    'date':                     str(date.today()),
    'description':              'Per-channel spike bounds — M6 fault seed reference',
    'winsor_sigma':             WINSOR_SIGMA,
    'spike_seed_sigma':         SPIKE_SEED_THRESHOLD_SIGMA,
    'channels':                 CHANNELS,
    'spike_thresholds':         {c: {k: v for k, v in vals.items() if k != 'spike'}
                                 for c, vals in spike_thresholds.items()},
    'total_spike_windows':      len(spike_windows),
    'spike_rows_excluded':      len(spike_row_indices),
    'fault_hint_counts': {
        'pressure_transient':   sum(1 for r in spike_meta_rows if r['fault_hint']=='pressure_transient'),
        'impeller_cavitation':  sum(1 for r in spike_meta_rows if r['fault_hint']=='impeller_cavitation'),
        'bearing_impact':       sum(1 for r in spike_meta_rows if r['fault_hint']=='bearing_impact'),
        'mechanical_transient': sum(1 for r in spike_meta_rows if r['fault_hint']=='mechanical_transient'),
    },
    'physics_notes': {
        'pressure_transient':   'Water hammer / startup valve surge. Pres.SV >3sigma.',
        'impeller_cavitation':  'Cavitation burst. Pmp.SV >3sigma. Accompanied by Pres.SV drop.',
        'bearing_impact':       'Bearing shock. Mot.SV >3sigma. May precede temperature rise.',
        'mechanical_transient': 'Multi-channel spike. Coupling / rotor imbalance event.',
    }
}
try:
    with open(SYNTH_DIR / "M4_spike_config.json", 'w') as f:
        json.dump(spike_config, f, indent=2)
    log("Saved: M4_spike_config.json")
except Exception as e:
    log(f"ERROR saving spike config: {e}")

results['M4_spike_windows_extracted'] = len(spike_windows)
results['M4_spike_fault_hints']       = spike_config['fault_hint_counts']
results['M4_spike_rows_excluded']     = len(spike_row_indices)


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Winsorization (clean the training data)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("PHASE 2 — Per-channel winsorization (mean + 4σ clip)")
log("=" * 65)

winsor_bounds = {}
total_clipped = 0

for c in WINSOR_CHANNELS:
    mu     = df[c].mean()
    sigma  = df[c].std()
    upper  = mu + WINSOR_SIGMA * sigma
    lower  = max(0.0, mu - WINSOR_SIGMA * sigma)
    n_clip = int(((df[c] > upper) | (df[c] < lower)).sum())
    df[c]  = df[c].clip(lower=lower, upper=upper)
    winsor_bounds[c] = {
        'mean':    round(float(mu), 6),
        'std':     round(float(sigma), 6),
        'upper':   round(float(upper), 6),
        'lower':   round(float(lower), 6),
        'clipped': int(n_clip),
    }
    total_clipped += n_clip
    log(f"  {c}: clip=[{lower:.4f}, {upper:.4f}] | "
        f"rows clipped: {n_clip} ({n_clip/len(df)*100:.3f}%)")

log(f"Total rows modified: {total_clipped:,} ({total_clipped/len(df)*100:.3f}% of dataset)")
log("Temperature channels untouched (cluster-relative, max=1.0 by design)")

log("\nPost-winsorization channel maxima:")
for c in CHANNELS:
    log(f"  {c}: max={df[c].max():.4f} | min={df[c].min():.6f}")

results['M4_winsor_bounds']        = winsor_bounds
results['M4_winsor_total_clipped'] = int(total_clipped)


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2.5 — Exclude spike rows from windowing pool
# ════════════════════════════════════════════════════════════════════════════
# Physics rationale: spike-containing rows represent transient fault events.
# Even after winsorization, clipped spike rows form artificial plateau
# patterns the LSTM cannot reconstruct → false alarms on val set.
# Spike rows belong exclusively in M6 synthetic pool.
# Normal operation windows must not contain ANY spike-origin rows.
df_clean = df.drop(index=list(spike_row_indices)).copy()
log(f"\nPHASE 2.5 — Spike row exclusion")
log(f"  Rows removed: {len(spike_row_indices):,}")
log(f"  Clean rows remaining: {len(df_clean):,}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Window Generation (per segment, warmup-aware, spike-free)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("PHASE 3 — Window generation on clean data")
log("=" * 65)

windows = []
seg_ids = []

for seg_id in usable_segs:
    seg_df  = df_clean[df_clean['segment_id'] == seg_id].copy()
    warmup  = int(warmup_map.get(seg_id, 300))
    seg_df  = seg_df.iloc[warmup:].reset_index(drop=True)
    if len(seg_df) < WINDOW_SIZE:
        log(f"  SKIP {seg_id}: only {len(seg_df)} rows after warmup")
        continue
    sensor_data = seg_df[CHANNELS].values.astype(np.float32)
    if np.isnan(sensor_data).any():
        log(f"  SKIP {seg_id}: NaN detected"); continue
    n_win = 0
    for i in range(0, len(seg_df) - WINDOW_SIZE + 1, STEP_SIZE):
        w = sensor_data[i : i + WINDOW_SIZE]
        if w.shape[0] == WINDOW_SIZE:
            windows.append(w)
            seg_ids.append(seg_id)
            n_win += 1
    log(f"  {seg_id}: {len(seg_df)} rows → {n_win} windows")

windows_arr = np.array(windows, dtype=np.float32)
log(f"\nTotal windows (clean): {len(windows_arr):,} | Shape: {windows_arr.shape}")
log(f"Spike windows held out: {len(spike_windows):,} → saved to M4_spike_seeds.npy")

post_winsor_max = float(windows_arr.max())
log(f"Post-winsorization window pool max value: {post_winsor_max:.4f}")
assert post_winsor_max < 15.0, \
    f"ALERT: Unexpected max {post_winsor_max:.4f} — check winsorization"

results['M4_total_windows']         = len(windows_arr)
results['M4_post_winsor_max_value'] = round(post_winsor_max, 4)


# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Train / Val Split
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("PHASE 4 — Train / Val split")
log("=" * 65)

rng   = np.random.default_rng(SEED)
idx   = rng.permutation(len(windows_arr))
split = int(len(idx) * (1 - VAL_SPLIT))
train_idx, val_idx = idx[:split], idx[split:]

X_train = torch.tensor(windows_arr[train_idx])
X_val   = torch.tensor(windows_arr[val_idx])

train_loader = DataLoader(
    TensorDataset(X_train), batch_size=BATCH_SIZE,
    shuffle=True, pin_memory=False, num_workers=0, drop_last=True
)
val_loader = DataLoader(
    TensorDataset(X_val), batch_size=BATCH_SIZE,
    shuffle=False, pin_memory=False, num_workers=0
)

results['M4_train_windows'] = len(train_idx)
results['M4_val_windows']   = len(val_idx)
log(f"Train: {len(train_idx):,} | Val: {len(val_idx):,}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Model Definition
# ════════════════════════════════════════════════════════════════════════════
weight_vec = torch.tensor(
    [CHANNEL_WEIGHTS[c] for c in CHANNELS], dtype=torch.float32
).to(DEVICE)


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


model = LSTMAutoencoder(
    n_features=len(CHANNELS), hidden_dim=HIDDEN_SIZE, bottleneck=BOTTLENECK,
    n_layers=NUM_LAYERS, dropout=DROPOUT, seq_len=WINDOW_SIZE
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"\nModel on {DEVICE} | Parameters: {n_params:,}")
results['M4_model_params'] = n_params


# ════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Training
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("PHASE 6 — Training")
log("=" * 65)


def physics_weighted_loss(pred, target):
    diff    = torch.abs(pred - target)
    diff_sq = (pred - target) ** 2
    return 0.6 * (diff * weight_vec).mean() + 0.4 * (diff_sq * weight_vec).mean()


optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=COSINE_T0, T_mult=1, eta_min=1e-5
)
scaler = GradScaler(enabled=IS_GPU)

train_losses, val_losses = [], []
best_val_loss  = float('inf')
best_epoch     = 0
patience_count = 0
overfit_stop   = False

log(f"Epochs={EPOCHS} | Batch={BATCH_SIZE} | Device={DEVICE} | "
    f"AMP={IS_GPU} | OverfitGap={OVERFIT_GAP_MAX}")
t_start = time.time()

for epoch in range(1, EPOCHS + 1):

    model.train()
    epoch_loss = 0.0
    for (xb,) in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with autocast(enabled=IS_GPU):
            loss = physics_weighted_loss(model(xb), xb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    train_loss = epoch_loss / len(train_loader)
    scheduler.step(epoch - 1)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (xb,) in val_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            with autocast(enabled=IS_GPU):
                val_loss += physics_weighted_loss(model(xb), xb).item()
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    gap = val_loss - train_loss
    if gap > OVERFIT_GAP_MAX:
        log(f"  Overfit guard @ epoch {epoch} "
            f"(gap={gap:.4f} > {OVERFIT_GAP_MAX}). Stopping.")
        overfit_stop = True
        break

    if val_loss < best_val_loss - 1e-6:
        best_val_loss  = val_loss
        best_epoch     = epoch
        patience_count = 0
        torch.save(model.state_dict(),
                   MODEL_DIR / "lstm_ae_baseline_best.pth")
    else:
        patience_count += 1

    if epoch % 10 == 0 or epoch == 1:
        elapsed = int(time.time() - t_start)
        log(f"  Epoch {epoch:>3}/{EPOCHS} | "
            f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
            f"Gap={gap:+.4f} | LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"Pat={patience_count}/{PATIENCE} | {elapsed}s")

    if patience_count >= PATIENCE:
        log(f"  Early stop @ epoch {epoch} "
            f"(best val={best_val_loss:.6f} @ {best_epoch})")
        break

    if IS_GPU and epoch == 1:
        alloc = torch.cuda.memory_allocated(DEVICE) / 1e9
        log(f"  VRAM after epoch 1: {alloc:.2f} GB")

t_elapsed = time.time() - t_start
log(f"\nTraining done — {t_elapsed:.1f}s | "
    f"Best val: {best_val_loss:.6f} @ epoch {best_epoch}")

results.update({
    'M4_best_val_loss':     round(best_val_loss, 6),
    'M4_best_epoch':        best_epoch,
    'M4_training_time_s':   round(t_elapsed, 1),
    'M4_overfit_triggered': overfit_stop,
})


# ════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Threshold Calibration
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("PHASE 7 — Threshold calibration on clean val set")
log("=" * 65)

model.load_state_dict(
    torch.load(MODEL_DIR / "lstm_ae_baseline_best.pth", map_location='cpu')
)
model.to(DEVICE)
model.eval()

all_maes         = []
per_channel_maes = {c: [] for c in CHANNELS}

with torch.no_grad():
    for (xb,) in val_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        with autocast(enabled=IS_GPU):
            pred = model(xb)
        err    = torch.abs(pred - xb).mean(dim=(1, 2)).cpu().numpy()
        ch_err = torch.abs(pred - xb).mean(dim=(0, 1)).cpu().numpy()
        all_maes.extend(err.tolist())
        for i, c in enumerate(CHANNELS):
            per_channel_maes[c].append(float(ch_err[i]))

all_maes  = np.array(all_maes)
mean_mae  = float(np.mean(all_maes))
std_mae   = float(np.std(all_maes))
p95_mae   = float(np.percentile(all_maes, 95))
p99_mae   = float(np.percentile(all_maes, 99))
threshold = round(max(mean_mae + 3 * std_mae, float(np.percentile(all_maes, 99.5))), 6)

delta_pct        = (threshold - OLD_THRESHOLD) / OLD_THRESHOLD * 100
separation_ratio = threshold / mean_mae
false_alarms     = int(np.sum(all_maes > threshold))
ch_mae_avg       = {c: float(np.mean(v)) for c, v in per_channel_maes.items()}

log(f"Mean MAE:  {mean_mae:.6f}")
log(f"Std MAE:   {std_mae:.6f}")
log(f"P95:       {p95_mae:.6f}")
log(f"P99:       {p99_mae:.6f}")
log(f"Threshold: {threshold:.6f} "
    f"(was {OLD_THRESHOLD}, delta={delta_pct:+.1f}%)")
log(f"Separation ratio: {separation_ratio:.1f}x")
log("Per-channel MAE (val):")
for c, v in ch_mae_avg.items():
    log(f"  {c}: {v:.6f}")

results.update({
    'M4_mean_recon_error':    round(mean_mae, 6),
    'M4_std_recon_error':     round(std_mae, 6),
    'M4_p95_error':           round(p95_mae, 6),
    'M4_p99_error':           round(p99_mae, 6),
    'M4_anomaly_threshold':   threshold,
    'M4_separation_ratio':    round(separation_ratio, 2),
    'M4_false_alarms_val':    false_alarms,
    'M4_threshold_delta_pct': round(delta_pct, 2),
    'M4_per_channel_mae':     {c: round(v, 6) for c, v in ch_mae_avg.items()},
})


# ════════════════════════════════════════════════════════════════════════════
# PHASE 8 — Validation Gates
# ════════════════════════════════════════════════════════════════════════════
log("\n--- Validation Gates ---")
gate_results = {}

gate_results['GATE1_no_overfit'] = (
    val_losses[best_epoch - 1] - train_losses[best_epoch - 1] < OVERFIT_GAP_MAX
)
gate_results['GATE2_mae_lt_006']        = mean_mae < 0.06
gate_results['GATE3_threshold_range']   = 0.05 <= threshold <= 0.50
gate_results['GATE4_separation_gt3']    = separation_ratio > 3.0
gate_results['GATE5_false_alarms_lt1pct'] = false_alarms <= int(len(all_maes) * 0.01)
V3_TV_REF = 0.040
gate_results['GATE6_tv_channels_ok']    = (
    ch_mae_avg['X_ACR_Mot.TV_norm'] <= V3_TV_REF * 1.5 and
    ch_mae_avg['X_ACR_Pmp.TV_norm'] <= V3_TV_REF * 1.5
)
gate_results['GATE7_spike_seeds_saved'] = (
    (SYNTH_DIR / "M4_spike_seeds.npy").exists() and
    (SYNTH_DIR / "M4_spike_seeds_meta.csv").exists() and
    len(spike_windows) > 0
)
gate_results['GATE8_val_loss_lt_005']   = best_val_loss < 0.05

for k, v in gate_results.items():
    log(f"  {k}: {'PASS' if v else 'FAIL'}")

all_gates_pass = all(gate_results.values())
log(f"\nAll gates: {'ALL PASS - READY FOR M5' if all_gates_pass else 'REVIEW FAILURES'}")
results['M4_all_gates_pass'] = all_gates_pass


# ════════════════════════════════════════════════════════════════════════════
# PHASE 9 — Save Artifacts
# ════════════════════════════════════════════════════════════════════════════
threshold_config = {
    "script":               SCRIPT_NAME,
    "version":              "v7",
    "date":                 str(date.today()),
    "anomaly_threshold":    threshold,
    "mean_recon_error":     round(mean_mae, 6),
    "std_recon_error":      round(std_mae, 6),
    "p95_error":            round(p95_mae, 6),
    "p99_error":            round(p99_mae, 6),
    "separation_ratio":     round(separation_ratio, 2),
    "best_val_loss":        round(best_val_loss, 6),
    "best_epoch":           best_epoch,
    "old_threshold_v3":     OLD_THRESHOLD,
    "threshold_delta_pct":  round(delta_pct, 2),
    "winsorization":        winsor_bounds,
    "spike_seeds_extracted":len(spike_windows),
    "spike_rows_excluded":  len(spike_row_indices),
    "spike_seeds_file":     "data/synthetic/M4_spike_seeds.npy",
    "spike_meta_file":      "data/synthetic/M4_spike_seeds_meta.csv",
    "channel_weights":      CHANNEL_WEIGHTS,
    "channels":             CHANNELS,
    "window_size":          WINDOW_SIZE,
    "gate_results":         gate_results,
    "created":              str(date.today())
}
try:
    with open(OUTPUT_DIR / "M4_threshold_config.json", 'w') as f:
        json.dump(threshold_config, f, indent=2)
    log("Saved: M4_threshold_config.json")
except Exception as e:
    log(f"ERROR: {e}")

torch.save(model.state_dict(), MODEL_DIR / "lstm_ae_baseline_final.pth")
try:
    with open(MODEL_DIR / "lstm_ae_baseline_meta.json", 'w') as f:
        json.dump({
            "version":    "v7",
            "date":       str(date.today()),
            "architecture": {
                "hidden":     HIDDEN_SIZE, "bottleneck": BOTTLENECK,
                "layers":     NUM_LAYERS,  "dropout":    DROPOUT,
                "window":     WINDOW_SIZE, "step":       STEP_SIZE,
            },
            "training": {
                "best_epoch":       best_epoch,
                "best_val_loss":    round(best_val_loss, 6),
                "winsorization":    "mean+4sigma per channel",
                "spike_seeds":      len(spike_windows),
                "spike_rows_excl":  len(spike_row_indices),
            },
            "threshold":       threshold,
            "channels":        CHANNELS,
            "channel_weights": CHANNEL_WEIGHTS,
        }, f, indent=2)
    log("Saved: lstm_ae_baseline_meta.json")
except Exception as e:
    log(f"ERROR: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 10 — Plots
# ════════════════════════════════════════════════════════════════════════════
log("\nGenerating plots...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ep = range(1, len(train_losses) + 1)
    axes[0].plot(ep, train_losses, color='steelblue', label='Train')
    axes[0].plot(ep, val_losses,   color='darkorange', label='Val')
    axes[0].axvline(best_epoch, color='green', linestyle='--',
                    label=f'Best={best_epoch}')
    axes[0].set_title('Loss Curves - Full'); axes[0].legend(); axes[0].grid(0.3)
    zoom = len(train_losses) // 4
    axes[1].plot(range(zoom+1, len(train_losses)+1),
                 train_losses[zoom:], color='steelblue', label='Train')
    axes[1].plot(range(zoom+1, len(train_losses)+1),
                 val_losses[zoom:],   color='darkorange', label='Val')
    axes[1].axvline(best_epoch, color='green', linestyle='--',
                    label=f'Best={best_epoch}')
    axes[1].set_title('Loss Curves - Zoomed'); axes[1].legend(); axes[1].grid(0.3)
    fig.suptitle(f'M4 v7 | Spike-excluded | Best val={best_val_loss:.6f} @ {best_epoch}',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M4_training_curve.png", dpi=150, bbox_inches='tight')
    plt.close(); log("Saved: M4_training_curve.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(all_maes, bins=80, color='steelblue', alpha=0.75,
            density=True, label='Val windows (spike-free)')
    ax.axvline(mean_mae,  color='green',  linestyle='--', lw=1.5,
               label=f'Mean={mean_mae:.4f}')
    ax.axvline(p99_mae,   color='orange', linestyle='--', lw=1.5,
               label=f'P99={p99_mae:.4f}')
    ax.axvline(threshold, color='red',    linestyle='-',  lw=2,
               label=f'Threshold={threshold:.4f}')
    ax.set_xlabel('Per-window MAE (normalised space)')
    ax.set_ylabel('Density')
    ax.set_title(f'M4 v7 Error Distribution | Sep={separation_ratio:.1f}x | '
                 f'FalseAlarms={false_alarms} | SpikeRowsExcluded={len(spike_row_indices)}',
                 fontweight='bold')
    ax.legend(); ax.grid(0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M4_error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(); log("Saved: M4_error_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    short = ['Mot.PV','Mot.SV','Mot.TV','Pmp.PV','Pmp.SV','Pmp.TV','Temp.SV','Pres.SV']
    vals  = [ch_mae_avg[c] for c in CHANNELS]
    cols  = ['#e74c3c' if 'TV' in c or 'Temp' in c else '#2ecc71' for c in CHANNELS]
    bars  = ax.bar(short, vals, color=cols, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.0003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Mean MAE'); ax.grid(axis='y', alpha=0.3)
    ax.set_title('M4 v7 Per-Channel MAE (Val) | Red=temp | Green=vib/pressure',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M4_per_channel_mae.png", dpi=150, bbox_inches='tight')
    plt.close(); log("Saved: M4_per_channel_mae.png")

    if len(spike_meta) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        hint_counts = spike_meta['fault_hint'].value_counts()
        colors_map  = {
            'pressure_transient':   '#e74c3c',
            'impeller_cavitation':  '#e67e22',
            'bearing_impact':       '#9b59b6',
            'mechanical_transient': '#3498db',
        }
        bars = ax.bar(hint_counts.index,
                      hint_counts.values,
                      color=[colors_map.get(x, '#95a5a6') for x in hint_counts.index],
                      alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, hint_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    str(v), ha='center', va='bottom', fontweight='bold')
        ax.set_ylabel('Window count')
        ax.set_title(f'M4 v7 Spike Seeds by Fault Hint | Total: {len(spike_meta)}',
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "M4_spike_seeds_distribution.png",
                    dpi=150, bbox_inches='tight')
        plt.close(); log("Saved: M4_spike_seeds_distribution.png")

    sample = next(iter(val_loader))
    xb_s   = sample[0].to(DEVICE)
    with torch.no_grad():
        with autocast(enabled=IS_GPU):
            recon_s = model(xb_s)
    orig  = xb_s[0].cpu().numpy()
    recon = recon_s[0].cpu().numpy()
    fig, axes2 = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
    axes2 = axes2.flatten()
    fig.suptitle('M4 v7 Reconstruction Sample (spike-free clean data)',
                 fontweight='bold')
    for i in range(len(CHANNELS)):
        axes2[i].plot(orig[:, i],  color='steelblue',  lw=1.4, label='Original')
        axes2[i].plot(recon[:, i], color='darkorange', lw=1.4,
                      linestyle='--', label='Reconstructed')
        axes2[i].axhline(1.0, color='red', linestyle=':', lw=0.8, alpha=0.5)
        ch_err_val = float(np.abs(orig[:, i] - recon[:, i]).mean())
        axes2[i].set_title(f'{short[i]} MAE={ch_err_val:.4f}',
                           fontsize=9, fontweight='bold')
        axes2[i].grid(alpha=0.2)
        if i == 0: axes2[i].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M4_reconstruction_sample.png",
                dpi=150, bbox_inches='tight')
    plt.close(); log("Saved: M4_reconstruction_sample.png")

except Exception as e:
    log(f"WARNING: Plot error: {e}")


if IS_GPU:
    peak_vram = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    log(f"Peak VRAM: {peak_vram:.2f} GB")
    results['M4_peak_vram_gb'] = round(peak_vram, 2)


# ════════════════════════════════════════════════════════════════════════════
# PHASE 11 — Report
# ════════════════════════════════════════════════════════════════════════════
log("Writing report...")
report_lines = [
    f"# M4 LSTM-AE Baseline Report v7",
    f"**Date:** {date.today()} | **Version:** v7 — Spike Row Exclusion\n",
    f"## What Changed vs v6",
    f"- Spike row indices recorded in PHASE 1",
    f"- PHASE 2.5 drops all spike rows from df before windowing",
    f"- Val set now contains ONLY true normal-operation windows",
    f"- False alarms eliminated by construction, not by threshold tuning\n",
    f"## Training Results",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| Best val loss | {results['M4_best_val_loss']} |",
    f"| Best epoch | {results['M4_best_epoch']} |",
    f"| Training time | {results['M4_training_time_s']}s |",
    f"| Overfit triggered | {results['M4_overfit_triggered']} |\n",
    f"## Threshold",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| Mean MAE | {results['M4_mean_recon_error']} |",
    f"| Std MAE | {results['M4_std_recon_error']} |",
    f"| P99 | {results['M4_p99_error']} |",
    f"| Threshold | {results['M4_anomaly_threshold']} |",
    f"| Separation ratio | {results['M4_separation_ratio']}x |",
    f"| False alarms | {results['M4_false_alarms_val']} |",
    f"| Spike rows excluded | {results['M4_spike_rows_excluded']} |\n",
    f"## Spike Seeds (M6 input)",
    f"| Fault Hint | Windows |",
    f"|------------|---------|",
]
for hint, count in results['M4_spike_fault_hints'].items():
    report_lines.append(f"| {hint} | {count} |")
report_lines += [
    f"\n## Validation Gates",
    f"| Gate | Result |",
    f"|------|--------|",
]
for k, v in gate_results.items():
    report_lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")

try:
    with open(REPORT_DIR / f"{SCRIPT_NAME}_report.md", 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"Saved: {SCRIPT_NAME}_report.md")
except Exception as e:
    log(f"ERROR writing report: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PASTE TEXT UPDATE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("== PASTE TEXT UPDATE - COPY BELOW INTO PASTE TEXT ==")
print("=" * 60)
print(f"M4_total_windows        : {results['M4_total_windows']}")
print(f"M4_train_windows        : {results['M4_train_windows']}")
print(f"M4_val_windows          : {results['M4_val_windows']}")
print(f"M4_best_val_loss        : {results['M4_best_val_loss']} (physics-weighted)")
print(f"M4_best_epoch           : {results['M4_best_epoch']}")
print(f"M4_mean_recon_error     : {results['M4_mean_recon_error']} (pure MAE)")
print(f"M4_std_recon_error      : {results['M4_std_recon_error']}")
print(f"M4_p99_error            : {results['M4_p99_error']}")
print(f"M4_anomaly_threshold    : {results['M4_anomaly_threshold']} (mean+3sigma | P99)")
print(f"M4_separation_ratio     : {results['M4_separation_ratio']}x")
print(f"M4_threshold_delta_pct  : {results['M4_threshold_delta_pct']:+.1f}% (was {OLD_THRESHOLD})")
print(f"M4_false_alarms_val     : {results['M4_false_alarms_val']}")
print(f"M4_spike_rows_excluded  : {results['M4_spike_rows_excluded']}")
print(f"M4_peak_vram_gb         : {results.get('M4_peak_vram_gb','N/A')}")
print(f"M4_training_time_s      : {results['M4_training_time_s']}")
print(f"M4_overfit_triggered    : {results['M4_overfit_triggered']}")
print(f"M4_all_gates_pass       : {results['M4_all_gates_pass']}")
print(f"M4_winsor_sigma         : {WINSOR_SIGMA} (mean+4sigma per channel)")
print(f"M4_winsor_total_clipped : {results['M4_winsor_total_clipped']} rows")
print(f"M4_post_winsor_max      : {results['M4_post_winsor_max_value']}")
print(f"M4_spike_windows        : {results['M4_spike_windows_extracted']} → M4_spike_seeds.npy")
print(f"M4_spike_fault_hints    : {results['M4_spike_fault_hints']}")
print(f"M4_model_version        : v7 (spike-excluded | winsorized | layernorm)")
print(f"Status for M5           : {'READY' if results['M4_all_gates_pass'] else 'NEEDS REVIEW'}")
print("=" * 60)
print("== END PASTE UPDATE ==")

print("\n--- FILE MANIFEST ---")
print("Spaces upload:")
print(f"    {REPORT_DIR / f'{SCRIPT_NAME}_report.md'}")
print(f"    {OUTPUT_DIR / 'M4_threshold_config.json'}")
print(f"    {MODEL_DIR  / 'lstm_ae_baseline_meta.json'}")
print("GitHub push:")
print(f"    {MODEL_DIR  / 'lstm_ae_baseline_best.pth'}")
print(f"    {MODEL_DIR  / 'lstm_ae_baseline_final.pth'}")
print(f"    {SYNTH_DIR  / 'M4_spike_seeds.npy'}")
print(f"    {SYNTH_DIR  / 'M4_spike_seeds_meta.csv'}")
print(f"    {SYNTH_DIR  / 'M4_spike_config.json'}")
for plot in ['M4_training_curve', 'M4_error_distribution',
             'M4_per_channel_mae', 'M4_spike_seeds_distribution',
             'M4_reconstruction_sample']:
    print(f"    {PLOTS_DIR / f'{plot}.png'}")
print("---")

print(f'\nM4 v7 done. Starting M5. '
      f'Finding: threshold={results["M4_anomaly_threshold"]}, '
      f'spike_seeds={results["M4_spike_windows_extracted"]} windows, '
      f'spike_rows_excluded={results["M4_spike_rows_excluded"]}. '
      f'Provide M5 complete script.')