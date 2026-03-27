import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, CLEAN_DIR, NORM_DIR,
                    OUTPUT_DIR, PLOTS_DIR, MODEL_DIR,
                    WARMUP_ROWS)
from datetime import date, datetime
import json, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

SCRIPT_NAME = "module_04_lstm_ae_baseline"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}

# =============================================================
# HYPERPARAMETERS
# =============================================================
WINDOW_SIZE   = 50
STEP_SIZE     = 10
BATCH_SIZE    = 256
EPOCHS        = 60
LR            = 1e-3
LR_PATIENCE   = 8
EARLY_STOP    = 15
HIDDEN_DIM    = 64
BOTTLENECK    = 32
NUM_LAYERS    = 2
DROPOUT       = 0.2
VAL_SPLIT     = 0.15
N_FEATURES    = 8

NORM_COLS = [
    'X_ACR_Mot.PV_norm', 'X_ACR_Mot.SV_norm', 'X_ACR_Mot.TV_norm',
    'X_ACR_Pmp.PV_norm', 'X_ACR_Pmp.SV_norm', 'X_ACR_Pmp.TV_norm',
    'X_Temp.SV_norm',    'X_Pres.SV_norm'
]

log(f"Device: {DEVICE} | GPU: {IS_GPU}")
log(f"Window={WINDOW_SIZE}s | Step={STEP_SIZE}s | "
    f"Batch={BATCH_SIZE} | Epochs={EPOCHS}")

# =============================================================
# STEP 1 — Load normalised data + segment registry
# =============================================================
log("STEP 1 — Loading normalised data and segment registry...")
try:
    df = pd.read_csv(
        NORM_DIR / "normalised_data.csv",
        parse_dates=['Timestamp']
    )
    log(f"  Normalised data: {len(df):,} rows | "
        f"{df['segment_id'].nunique()} segments")
except Exception as e:
    raise RuntimeError(f"Cannot load normalised_data.csv: {e}")

try:
    reg = pd.read_csv(CLEAN_DIR / "segment_registry.csv")
    log(f"  Registry: {len(reg)} segments | "
        f"{reg['usable_for_windowing'].sum()} usable")
except Exception as e:
    raise RuntimeError(f"Cannot load segment_registry.csv: {e}")

warmup_map = reg.set_index('segment_id')['warmup_rows'].to_dict()
df['warmup_rows'] = df['segment_id'].map(warmup_map).fillna(300).astype(int)

results['total_rows'] = len(df)

# =============================================================
# STEP 2 — Window generation (per segment, warmup-aware)
# =============================================================
log("STEP 2 — Generating windows per segment (warmup-aware)...")

windows     = []
seg_ids     = []
cluster_ids = []

usable_segs = reg[reg['usable_for_windowing'] == True]['segment_id'].tolist()

for seg_id in usable_segs:
    seg_df  = df[df['segment_id'] == seg_id].copy()
    warmup  = int(warmup_map.get(seg_id, WARMUP_ROWS))
    seg_df  = seg_df.iloc[warmup:].reset_index(drop=True)

    if len(seg_df) < WINDOW_SIZE:
        log(f"  SKIP {seg_id}: only {len(seg_df)} rows after warmup")
        continue

    sensor_data = seg_df[NORM_COLS].values.astype(np.float32)
    if np.isnan(sensor_data).any():
        log(f"  SKIP {seg_id}: NaN in normalised cols")
        continue

    n_windows = 0
    for i in range(0, len(seg_df) - WINDOW_SIZE + 1, STEP_SIZE):
        w = sensor_data[i : i + WINDOW_SIZE]
        if w.shape[0] == WINDOW_SIZE:
            windows.append(w)
            seg_ids.append(seg_id)
            cid = seg_df['cluster_id'].iloc[i + WINDOW_SIZE // 2]
            cluster_ids.append(int(cid))
            n_windows += 1

    log(f"  {seg_id}: {len(seg_df)} rows → {n_windows} windows "
        f"(warmup={warmup})")

windows_arr = np.array(windows, dtype=np.float32)
log(f"\n  Total windows: {len(windows_arr):,} | "
    f"Shape: {windows_arr.shape}")
results['total_windows'] = len(windows_arr)

# =============================================================
# STEP 3 — Dataset + DataLoader
# =============================================================
log("STEP 3 — Creating Dataset and DataLoader...")

class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = torch.tensor(windows, dtype=torch.float32)
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return self.windows[idx]

full_dataset = WindowDataset(windows_arr)
n_val        = int(len(full_dataset) * VAL_SPLIT)
n_train      = len(full_dataset) - n_val

train_ds, val_ds = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

# ✅ FIX 1: num_workers=0, pin_memory=False — Windows spawn fix
# GPU is NOT affected: .to(DEVICE) inside training loop handles CUDA transfer
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=False, num_workers=0, drop_last=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    pin_memory=False, num_workers=0
)

log(f"  Train: {n_train:,} windows | Val: {n_val:,} windows")
results['n_train_windows'] = n_train
results['n_val_windows']   = n_val

# =============================================================
# STEP 4 — LSTM-AE Architecture
# =============================================================
log("STEP 4 — Building LSTM-AE model...")

class LSTMEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim, bottleneck, n_layers, dropout):
        super().__init__()
        self.lstm1 = nn.LSTM(
            n_features, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout
        )
        self.lstm2 = nn.LSTM(
            hidden_dim, bottleneck, num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        out, _      = self.lstm1(x)
        out, (h, _) = self.lstm2(out)
        return h.squeeze(0)


class LSTMDecoder(nn.Module):
    def __init__(self, bottleneck, hidden_dim, n_features,
                 n_layers, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.lstm1   = nn.LSTM(
            bottleneck, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout
        )
        self.lstm2   = nn.LSTM(
            hidden_dim, n_features, num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(n_features, n_features)

    def forward(self, z):
        z_rep  = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm1(z_rep)
        out, _ = self.lstm2(out)
        return self.output_layer(out)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_features=8, hidden_dim=64, bottleneck=32,
                 n_layers=2, dropout=0.2, seq_len=50):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim, bottleneck, n_layers, dropout)
        self.decoder = LSTMDecoder(bottleneck, hidden_dim, n_features, n_layers, dropout, seq_len)

    def forward(self, x):
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon


model = LSTMAutoEncoder(
    n_features=N_FEATURES,
    hidden_dim=HIDDEN_DIM,
    bottleneck=BOTTLENECK,
    n_layers=NUM_LAYERS,
    dropout=DROPOUT,
    seq_len=WINDOW_SIZE
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"  Model parameters: {n_params:,}")
log(f"  Encoder bottleneck: {BOTTLENECK} dims")
results['model_params'] = n_params

# =============================================================
# STEP 5 — Training
# =============================================================
log("STEP 5 — Training LSTM-AE...")
log(f"  Mixed precision (AMP): {IS_GPU}")

criterion  = nn.L1Loss()
optimizer  = torch.optim.Adam(model.parameters(), lr=LR)

# ✅ FIX 2: verbose= removed — hard-deleted in PyTorch 2.6
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5,
    patience=LR_PATIENCE
)
scaler     = GradScaler(enabled=IS_GPU)

train_losses = []
val_losses   = []
best_val     = float('inf')
patience_cnt = 0
best_epoch   = 0

t_start = datetime.now()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with autocast(enabled=IS_GPU):
            recon = model(batch)
            loss  = criterion(recon, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            with autocast(enabled=IS_GPU):
                recon = model(batch)
                loss  = criterion(recon, batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val - 1e-6:
        best_val     = val_loss
        best_epoch   = epoch
        patience_cnt = 0
        torch.save(model.state_dict(),
                   MODEL_DIR / "lstm_ae_baseline_best.pth")
    else:
        patience_cnt += 1

    if epoch % 5 == 0 or epoch == 1:
        elapsed = (datetime.now() - t_start).seconds
        lr_now  = optimizer.param_groups[0]['lr']
        log(f"  Epoch {epoch:>3}/{EPOCHS} | "
            f"Train={train_loss:.6f} | "
            f"Val={val_loss:.6f} | "
            f"LR={lr_now:.2e} | "
            f"Patience={patience_cnt}/{EARLY_STOP} | "
            f"Elapsed={elapsed}s")

    if patience_cnt >= EARLY_STOP:
        log(f"  Early stopping at epoch {epoch} "
            f"(best val={best_val:.6f} at epoch {best_epoch})")
        break

t_total = (datetime.now() - t_start).seconds
log(f"\n  Training complete: {t_total}s | "
    f"Best val loss: {best_val:.6f} at epoch {best_epoch}")

results['best_val_loss']   = round(best_val, 6)
results['best_epoch']      = best_epoch
results['training_time_s'] = t_total

# =============================================================
# STEP 6 — Reconstruction error → threshold
# =============================================================
log("STEP 6 — Computing reconstruction error distribution...")

model.load_state_dict(
    torch.load(MODEL_DIR / "lstm_ae_baseline_best.pth",
               map_location='cpu')
)
model.to(DEVICE)
model.eval()

all_errors = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(DEVICE, non_blocking=True)
        with autocast(enabled=IS_GPU):
            recon = model(batch)
        err = (batch - recon).abs().mean(dim=(1, 2))
        all_errors.extend(err.cpu().numpy().tolist())

errors_arr = np.array(all_errors)
mean_err   = float(np.mean(errors_arr))
std_err    = float(np.std(errors_arr))
p95_err    = float(np.percentile(errors_arr, 95))
p99_err    = float(np.percentile(errors_arr, 99))
threshold  = max(mean_err + 3 * std_err, p99_err)
threshold  = round(threshold, 6)

log(f"  Mean recon error  : {mean_err:.6f}")
log(f"  Std recon error   : {std_err:.6f}")
log(f"  P95 error         : {p95_err:.6f}")
log(f"  P99 error         : {p99_err:.6f}")
log(f"  Anomaly threshold : {threshold:.6f} (mean + 3σ ∪ P99)")

results['mean_recon_error']  = round(mean_err, 6)
results['std_recon_error']   = round(std_err, 6)
results['p95_error']         = round(p95_err, 6)
results['p99_error']         = round(p99_err, 6)
results['anomaly_threshold'] = threshold

threshold_dict = {
    'anomaly_threshold' : threshold,
    'mean_error'        : round(mean_err, 6),
    'std_error'         : round(std_err, 6),
    'p95_error'         : round(p95_err, 6),
    'p99_error'         : round(p99_err, 6),
    'method'            : 'mean + 3sigma union p99',
    'window_size'       : WINDOW_SIZE,
    'n_features'        : N_FEATURES,
    'norm_cols'         : NORM_COLS,
    'created'           : str(date.today())
}
thresh_path = OUTPUT_DIR / "M4_threshold_config.json"
with open(thresh_path, 'w') as f:
    json.dump(threshold_dict, f, indent=2)
log(f"  Threshold config saved → M4_threshold_config.json")

# =============================================================
# STEP 7 — VRAM usage
# =============================================================
if IS_GPU:
    vram_used = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    log(f"STEP 7 — Peak VRAM used: {vram_used:.2f} GB")
    results['vram_peak_gb'] = round(vram_used, 2)
    if vram_used > 7.0:
        log("  ⚠️  VRAM > 7GB — reduce BATCH_SIZE in config if OOM")
else:
    results['vram_peak_gb'] = 0.0

# =============================================================
# STEP 8 — Training curve plot
# =============================================================
log("STEP 8 — Generating training loss curve...")
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses,
            label='Train Loss (MAE)', color='steelblue', linewidth=1.5)
    ax.plot(epochs_range, val_losses,
            label='Val Loss (MAE)', color='darkorange', linewidth=1.5)
    ax.axvline(best_epoch, color='green', linestyle='--',
               linewidth=1.2, label=f'Best epoch={best_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE Loss')
    ax.set_title('M4 — LSTM-AE Baseline Training Curve', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M4_training_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M4_training_curve.png")
except Exception as e:
    log(f"  WARNING: Training curve failed: {e}")

# =============================================================
# STEP 9 — Error distribution plot
# =============================================================
log("STEP 9 — Generating error distribution plot...")
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors_arr, bins=80, color='steelblue',
            alpha=0.7, edgecolor='none', density=True,
            label='Val reconstruction error')
    ax.axvline(mean_err,   color='green',  linestyle='--', linewidth=1.5, label=f'Mean={mean_err:.4f}')
    ax.axvline(p95_err,    color='orange', linestyle='--', linewidth=1.5, label=f'P95={p95_err:.4f}')
    ax.axvline(threshold,  color='red',    linestyle='-',  linewidth=2.0, label=f'Threshold={threshold:.4f}')
    ax.set_xlabel('Per-window MAE (normalised space)')
    ax.set_ylabel('Density')
    ax.set_title('M4 — Reconstruction Error Distribution\n'
                 'Anomaly threshold = mean + 3σ ∪ P99', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M4_error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M4_error_distribution.png")
except Exception as e:
    log(f"  WARNING: Error distribution plot failed: {e}")

# =============================================================
# STEP 10 — Reconstruction overlay plot
# =============================================================
log("STEP 10 — Generating reconstruction overlay plot...")
try:
    sample_batch = next(iter(val_loader)).to(DEVICE)
    with torch.no_grad():
        with autocast(enabled=IS_GPU):
            sample_recon = model(sample_batch)

    orig  = sample_batch[0].cpu().numpy()
    recon = sample_recon[0].cpu().numpy()

    short_names = ['Mot.PV', 'Mot.SV', 'Mot.TV',
                   'Pmp.PV', 'Pmp.SV', 'Pmp.TV',
                   'Temp.SV', 'Pres.SV']

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes      = axes.flatten()
    fig.suptitle('M4 — Sample Window Reconstruction\n'
                 '(blue=original, orange=reconstructed)', fontweight='bold')

    for idx in range(N_FEATURES):
        ax = axes[idx]
        ax.plot(orig[:, idx],  color='steelblue',  linewidth=1.2, label='Original')
        ax.plot(recon[:, idx], color='darkorange',  linewidth=1.2, linestyle='--', label='Reconstructed')
        ax.axhline(1.0, color='red', linestyle=':', linewidth=0.8)
        ax.set_title(short_names[idx], fontsize=9, fontweight='bold')
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M4_reconstruction_sample.png", dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved → M4_reconstruction_sample.png")
except Exception as e:
    log(f"  WARNING: Reconstruction overlay failed: {e}")

# =============================================================
# STEP 11 — Save final model + metadata
# =============================================================
log("STEP 11 — Saving final model state dict...")
final_model_path = MODEL_DIR / "lstm_ae_baseline_final.pth"
torch.save(model.state_dict(), final_model_path)

model_meta = {
    'architecture'    : 'LSTM-Autoencoder',
    'n_features'      : N_FEATURES,
    'hidden_dim'      : HIDDEN_DIM,
    'bottleneck_dim'  : BOTTLENECK,
    'n_layers'        : NUM_LAYERS,
    'dropout'         : DROPOUT,
    'window_size'     : WINDOW_SIZE,
    'step_size'       : STEP_SIZE,
    'norm_cols'       : NORM_COLS,
    'best_val_loss'   : results['best_val_loss'],
    'best_epoch'      : best_epoch,
    'anomaly_threshold': threshold,
    'training_rows'   : n_train,
    'val_rows'        : n_val,
    'total_windows'   : results['total_windows'],
    'created'         : str(date.today())
}
meta_path = MODEL_DIR / "lstm_ae_baseline_meta.json"
with open(meta_path, 'w') as f:
    json.dump(model_meta, f, indent=2)

log(f"  Model saved → lstm_ae_baseline_final.pth")
log(f"  Metadata    → lstm_ae_baseline_meta.json")

# =============================================================
# STEP 12 — Markdown report
# =============================================================
log("STEP 12 — Writing markdown report...")
report_lines = [
    f"# M4 LSTM-AE Baseline Report",
    f"**Date:** {date.today()}  ",
    f"**Script:** {SCRIPT_NAME}  ",
    "",
    "## Model Architecture",
    "| Component | Value |",
    "|---|---|",
    "| Type | LSTM Autoencoder |",
    "| Input shape | (batch, 50, 8) |",
    f"| Hidden dim | {HIDDEN_DIM} |",
    f"| Bottleneck | {BOTTLENECK} |",
    f"| LSTM layers | {NUM_LAYERS} |",
    f"| Parameters | {results['model_params']:,} |",
    "",
    "## Training Results",
    "| Metric | Value |",
    "|---|---|",
    f"| Best val loss (MAE) | {results['best_val_loss']} |",
    f"| Best epoch | {results['best_epoch']} |",
    f"| Training time | {results['training_time_s']}s |",
    f"| Peak VRAM | {results['vram_peak_gb']} GB |",
    "",
    "## Anomaly Threshold",
    "| Metric | Value |",
    "|---|---|",
    f"| Mean error | {results['mean_recon_error']} |",
    f"| Std error | {results['std_recon_error']} |",
    f"| P95 error | {results['p95_error']} |",
    f"| P99 error | {results['p99_error']} |",
    f"| **Threshold** | **{results['anomaly_threshold']}** |",
    f"| Method | mean + 3σ ∪ P99 |",
    "",
    "## Output Files",
    "- `models/lstm_ae_baseline_best.pth`",
    "- `models/lstm_ae_baseline_final.pth`",
    "- `models/lstm_ae_baseline_meta.json`",
    "- `outputs/M4_threshold_config.json`",
    "- `outputs/plots/M4_training_curve.png`",
    "- `outputs/plots/M4_error_distribution.png`",
    "- `outputs/plots/M4_reconstruction_sample.png`",
]
report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log(f"  Report saved → {report_path.name}")

# =============================================================
# PASTE TEXT UPDATE
# =============================================================
print()
print("═"*60)
print("  PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT")
print("═"*60)
print(f"M4_total_windows       : {results['total_windows']}")
print(f"M4_train_windows       : {results['n_train_windows']}")
print(f"M4_val_windows         : {results['n_val_windows']}")
print(f"M4_best_val_loss       : {results['best_val_loss']}")
print(f"M4_best_epoch          : {results['best_epoch']}")
print(f"M4_mean_recon_error    : {results['mean_recon_error']}")
print(f"M4_anomaly_threshold   : {results['anomaly_threshold']}")
print(f"M4_vram_peak_gb        : {results['vram_peak_gb']}")
print(f"M4_training_time_s     : {results['training_time_s']}")
print(f"M4_model_file          : lstm_ae_baseline_best.pth")
print(f"M4_threshold_config    : M4_threshold_config.json")
print(f"Status for M5          : READY")
print("═"*60)

# =============================================================
# FILE MANIFEST
# =============================================================
print()
print("── FILE MANIFEST ──────────────────────────────────────────")
print("→ Spaces upload:")
print(f"    {report_path}")
print(f"    {thresh_path}")
print(f"    {meta_path}")
print("→ GitHub push:")
print(f"    {MODEL_DIR / 'lstm_ae_baseline_best.pth'}")
print(f"    {MODEL_DIR / 'lstm_ae_baseline_final.pth'}")
print(f"    {MODEL_DIR / 'lstm_ae_baseline_meta.json'}")
print(f"    {thresh_path}")
for f in sorted(PLOTS_DIR.glob("M4_*.png")):
    print(f"    {f}")
print("───────────────────────────────────────────────────────────")
print()
print("📦 M4 done. Starting M5.")
print("   Upload report + threshold config + meta JSON to Spaces.")
print("   Push model .pth files + plots to GitHub.")
print("   Provide M5 complete script.")
