# =============================================================================
# module_06p5r_patch_label19_features.py
# PumpSmart v14.2 — Surgical patch: Label 19 feature matrix rows
# =============================================================================
#
# PURPOSE:
#   After M6B Label 19 sequence patch (module_06B_patch_label19_seal_fast.py),
#   the M6B_feature_matrix.csv still contains features derived from the OLD
#   broken sequences (Pres.SV flat at 0.9654, no real drop signal).
#   M7 XGBoost trained on those rows will have learned the wrong Label 19
#   signature. This script replaces ONLY the 800 Label 19 rows.
#
# SCOPE: Label 19 rows ONLY — all 31,700 other rows untouched.
#
# RUN ORDER:
#   1. module_06B_patch_label19_seal_fast.py  ← already done
#   2. THIS SCRIPT                            ← run now
#   3. module_07_xgboost_classifier.py        ← re-run after this
#
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, SYNTH_DIR, MODEL_DIR, OUTPUT_DIR)
from datetime import date, datetime
import json, pickle, shutil, warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress, kurtosis
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

SCRIPT_NAME = "module_06p5r_patch_label19_features"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

log("=" * 70)
log(f"  PumpSmart — Label 19 Feature Matrix Patch | v14.2 | {date.today()}")
log("=" * 70)

# =============================================================================
# SECTION 0 — CONSTANTS (must match module_06p5r_feature_retrain.py exactly)
# =============================================================================
WINDOW_SIZE  = 50
STRIDE       = 25
LABEL_19     = 19
CH_NAMES     = ["Mot.SV","Pmp.SV","Mot.TV","Pmp.PV",
                "Temp.SV","Pres.SV","Pmp.TV","Mot.PV"]
CH           = {c: i for i, c in enumerate(CH_NAMES)}

# Paths
FEATURE_MATRIX_PATH = SYNTH_DIR / "M6B_feature_matrix.csv"
BACKUP_PATH         = SYNTH_DIR / "M6B_feature_matrix_pre_label19_feature_patch.csv"
GRPD_PKL_PATH       = SYNTH_DIR / "M6B_sequences_groupD.pkl"
ZT_GRPD_PATH        = SYNTH_DIR / "z_t_sequences_groupD.pkl"
META_PATH           = SYNTH_DIR / "M6B_sequence_meta.csv"

# =============================================================================
# SECTION 1 — LOAD M4 LSTM-AE
# =============================================================================
log("\nSECTION 1 — Loading frozen M4 LSTM-AE")

class LSTMAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128, 64, num_layers=1, batch_first=True)
        self.bn    = nn.LayerNorm(64)
    def forward(self, x):
        out1, _          = self.lstm1(x)
        out2, (h_n, c_n) = self.lstm2(out1)
        return self.bn(h_n[-1]), h_n, c_n

class LSTMAEDecoder(nn.Module):
    def __init__(self, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.fc_h    = nn.Linear(64, 128)
        self.fc_c    = nn.Linear(64, 128)
        self.lstm1   = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm2   = nn.LSTM(128, 8, num_layers=1, batch_first=True)
        self.out     = nn.Linear(8, 8)
    def forward(self, z, h_n, c_n):
        h0    = torch.tanh(self.fc_h(h_n[-1])).unsqueeze(0).repeat(2,1,1)
        c0    = torch.tanh(self.fc_c(c_n[-1])).unsqueeze(0).repeat(2,1,1)
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm1(z_rep, (h0, c0))
        out, _ = self.lstm2(out)
        return self.out(out)

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=50):
        super().__init__()
        self.encoder = LSTMAEEncoder()
        self.decoder = LSTMAEDecoder(seq_len=seq_len)
    def forward(self, x):
        z, h_n, c_n = self.encoder(x)
        return self.decoder(z, h_n, c_n)
    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

m4_model = LSTMAutoencoder(seq_len=WINDOW_SIZE)
m4_model.load_state_dict(
    torch.load(MODEL_DIR/"lstm_ae_baseline_final.pth", map_location='cpu'))
m4_model.eval()
for p in m4_model.parameters():
    p.requires_grad_(False)
m4_model.to(DEVICE)
log(f"  M4 loaded → {DEVICE} | Params: {sum(p.numel() for p in m4_model.parameters()):,}")

# =============================================================================
# SECTION 2 — LOAD PATCHED groupD SEQUENCES
# =============================================================================
log("\nSECTION 2 — Loading patched M6B_sequences_groupD.pkl")

with open(GRPD_PKL_PATH, "rb") as f:
    grpD = pickle.load(f)

seqs_D = grpD['sequences']
meta_D = grpD['metadata']

lbl19_positions = [i for i,m in enumerate(meta_D) if m['label'] == LABEL_19]
log(f"  Total sequences: {len(seqs_D)} | Label 19 positions: {len(lbl19_positions)}")
assert len(lbl19_positions) == 800, f"Expected 800, got {len(lbl19_positions)}"

# Verify patch was applied
pres_mins = [seqs_D[p][:, CH["Pres.SV"]].min() for p in lbl19_positions[:10]]
pres_mean_min = float(np.mean(pres_mins))
log(f"  Label 19 Pres.SV min sample (first 10): mean={pres_mean_min:.4f}")
if pres_mean_min > 0.92:
    log("  [FATAL] Label 19 sequences appear unpatched (Pres.SV min > 0.92).")
    log("  Run module_06B_patch_label19_seal_fast.py first.")
    raise RuntimeError("Sequences not patched")
log(f"  Patch confirmed: Pres.SV min={pres_mean_min:.4f} (expected < 0.85) ✓")

# =============================================================================
# SECTION 3 — LOAD EXISTING FEATURE MATRIX
# =============================================================================
log("\nSECTION 3 — Loading existing feature matrix")

if not FEATURE_MATRIX_PATH.exists():
    log(f"  [FATAL] Feature matrix not found: {FEATURE_MATRIX_PATH}")
    raise FileNotFoundError(str(FEATURE_MATRIX_PATH))

# Backup
if not BACKUP_PATH.exists():
    shutil.copy2(FEATURE_MATRIX_PATH, BACKUP_PATH)
    log(f"  Backup: {BACKUP_PATH.name}")
else:
    log(f"  Backup already exists — skipping")

df = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
log(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

# Identify Label 19 rows
lbl19_mask   = df['label_int'].astype(int) == LABEL_19
n_lbl19_rows = lbl19_mask.sum()
log(f"  Label 19 rows: {n_lbl19_rows:,}")

if n_lbl19_rows == 0:
    log("  [FATAL] No Label 19 rows found in feature matrix")
    raise ValueError("Label 19 rows not found")

feature_cols = [c for c in df.columns if c != 'label_int']
log(f"  Feature columns: {len(feature_cols)}")
log(f"  Columns: {feature_cols[:10]}...")

# =============================================================================
# SECTION 4 — FEATURE EXTRACTION FUNCTIONS
# =============================================================================
log("\nSECTION 4 — Feature extraction functions")

def extract_window_features(seq_np, window_idx, window_size=50, stride=25):
    """
    Extract features for one sliding window from a raw sequence.
    Matches module_06p5r_feature_retrain.py feature set exactly.
    """
    T = seq_np.shape[0]
    start = window_idx * stride
    end   = start + window_size
    if end > T:
        return None
    window = seq_np[start:end]   # (50, 8)

    # Run M4 on this window
    x_t   = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z     = m4_model.encode(x_t).squeeze(0).cpu().numpy()     # (64,)
        recon = m4_model(x_t).squeeze(0).cpu().numpy()            # (50, 8)
    mae   = np.mean(np.abs(recon - window), axis=0)               # (8,)

    feats = {}

    # Domain 1: per-channel MAE
    for i, ch in enumerate(CH_NAMES):
        feats[f'mae_{ch.replace(".","_")}'] = float(mae[i])

    # Domain 2: error statistics
    feats['mean_err_MotSV']  = float(np.mean(np.abs(recon[:,CH["Mot.SV"]] - window[:,CH["Mot.SV"]])))
    feats['std_err_MotSV']   = float(np.std( np.abs(recon[:,CH["Mot.SV"]] - window[:,CH["Mot.SV"]])))
    feats['kurtosis_PmpSV']  = float(kurtosis(window[:, CH["Pmp.SV"]]))
    feats['err_slope_MotSV'] = float(linregress(np.arange(window_size),
        np.abs(recon[:,CH["Mot.SV"]] - window[:,CH["Mot.SV"]]))[0])
    feats['err_slope_TempSV']= float(linregress(np.arange(window_size),
        np.abs(recon[:,CH["Temp.SV"]] - window[:,CH["Temp.SV"]]))[0])
    feats['err_slope_PresSV']= float(linregress(np.arange(window_size),
        np.abs(recon[:,CH["Pres.SV"]] - window[:,CH["Pres.SV"]]))[0])
    feats['max_err_all']     = float(np.max(mae))

    # Domain 3: physics cross-channel features
    mot_sv = window[:, CH["Mot.SV"]]
    mot_tv = window[:, CH["Mot.TV"]]
    if mot_sv.std() > 1e-8 and mot_tv.std() > 1e-8:
        feats['thermal_coupling_ratio'] = float(np.corrcoef(mot_sv, mot_tv)[0,1])
    else:
        feats['thermal_coupling_ratio'] = 0.0
    pmp_sv = window[:, CH["Pmp.SV"]]
    if mot_sv.std() > 1e-8 and pmp_sv.std() > 1e-8:
        feats['cross_channel_MotSV_PmpSV'] = float(np.corrcoef(mot_sv, pmp_sv)[0,1])
    else:
        feats['cross_channel_MotSV_PmpSV'] = 0.0

    # Label 19 specific: masked_channel_flag = 0 (no mask), secondary_onset_lag = 0
    feats['masked_channel_flag']   = 0
    feats['secondary_onset_lag']   = 0
    feats['burst_count']           = 0
    feats['cyclic_baseline_drift'] = 0.0
    feats['multi_sensor_anomaly_count'] = 0
    feats['fault_group_id']        = 1     # Group A (single fault)
    feats['variant_slope_ratio']   = 0.0
    feats['thermal_decoupling_flag'] = 0

    # Domain 4: z_t features
    feats['z_t_norm']      = float(np.linalg.norm(z))
    feats['z_t_recon_err'] = float(np.mean(mae))
    feats['mean_err_MotSV_w'] = feats['mean_err_MotSV']  # alias

    # score_A, score_B, score_C: placeholder (computed per-sequence in M8)
    # For M7, these are set as the per-window reconstruction error
    feats['score_A'] = float(np.linalg.norm(z) / 8.0)   # normalized
    feats['score_B'] = 0.0   # slope not computable per-window
    feats['score_C'] = 0.0   # transition not computable per-window

    # onset_order: for Label 19, Pres.SV should have highest MAE
    # Encode: which channel has max MAE (0-indexed in CH order)
    feats['onset_order'] = int(np.argmax(mae))

    return feats


def extract_all_features_for_sequence(seq_np, window_size=50, stride=25):
    """
    Extract features for ALL sliding windows in a sequence.
    Returns list of feature dicts, one per valid window.
    """
    T = seq_np.shape[0]
    rows = []
    w_idx = 0
    while True:
        start = w_idx * stride
        if start + window_size > T:
            break
        feats = extract_window_features(seq_np, w_idx, window_size, stride)
        if feats is not None:
            feats['label_int'] = LABEL_19
            rows.append(feats)
        w_idx += 1
    return rows


# =============================================================================
# SECTION 5 — EXTRACT NEW FEATURES FOR ALL 800 LABEL 19 SEQUENCES
# =============================================================================
log("\nSECTION 5 — Extracting features for 800 patched Label 19 sequences")

new_rows = []
for i, pos in enumerate(lbl19_positions):
    seq = seqs_D[pos]   # (150, 8)
    rows = extract_all_features_for_sequence(seq, WINDOW_SIZE, STRIDE)
    new_rows.extend(rows)
    if (i + 1) % 100 == 0:
        log(f"  Processed {i+1}/800 sequences | rows so far: {len(new_rows)}")

log(f"\n  New Label 19 feature rows: {len(new_rows):,}")

new_df = pd.DataFrame(new_rows)
log(f"  New feature df shape: {new_df.shape}")

# Ensure column alignment with existing feature matrix
all_cols = ['label_int'] + feature_cols
for col in all_cols:
    if col not in new_df.columns:
        log(f"  [WARN] Column {col} missing from new_df — filling with 0.0")
        new_df[col] = 0.0

new_df = new_df[all_cols].copy()

# Verify Label 19 features look physically correct
pres_mae_mean = new_df['mae_Pres_SV'].mean() if 'mae_Pres_SV' in new_df.columns else (
    new_df['mae_PresSV'].mean() if 'mae_PresSV' in new_df.columns else 0.0)
log(f"  Pres.SV MAE mean (should be >> old ~0.006): {pres_mae_mean:.4f}")
if pres_mae_mean < 0.05:
    log("  [WARN] Pres.SV MAE still low — check sequence patch")
else:
    log("  Pres.SV MAE elevated ✓ — feature patch physics valid")

# =============================================================================
# SECTION 6 — REPLACE LABEL 19 ROWS IN FEATURE MATRIX
# =============================================================================
log("\nSECTION 6 — Replacing Label 19 rows in feature matrix")

# Remove old Label 19 rows
df_other = df[~lbl19_mask].copy()
log(f"  Remaining rows after removing old Label 19: {len(df_other):,}")

# Concatenate
df_patched = pd.concat([df_other, new_df], ignore_index=True)
log(f"  Patched matrix shape: {df_patched.shape}")

# Verify label distribution
label_dist = df_patched['label_int'].astype(int).value_counts().sort_index()
log(f"  Label 19 rows in patched matrix: {label_dist.get(19, 0):,}")

# Sanity: total rows should be similar (may differ slightly due to window count)
log(f"  Original rows: {len(df):,} | Patched rows: {len(df_patched):,}")
delta = abs(len(df_patched) - len(df))
if delta > 500:
    log(f"  [WARN] Row count changed by {delta} — check window extraction")

# =============================================================================
# SECTION 7 — SAVE PATCHED FEATURE MATRIX
# =============================================================================
log("\nSECTION 7 — Saving patched feature matrix")

df_patched.to_csv(FEATURE_MATRIX_PATH, index=False)
size_mb = FEATURE_MATRIX_PATH.stat().st_size / 1e6
log(f"  Saved: {FEATURE_MATRIX_PATH.name} | {len(df_patched):,} rows | {size_mb:.1f} MB")

# Also update metadata json if it exists
meta_json_path = SYNTH_DIR / "M6B_feature_matrix_metadata_v5.json"
if meta_json_path.exists():
    with open(meta_json_path) as f:
        meta_json = json.load(f)
    meta_json['label19_patch_applied'] = True
    meta_json['label19_patch_date'] = str(date.today())
    meta_json['label19_patch_rows'] = len(new_rows)
    meta_json['label19_pres_mae_mean_after_patch'] = round(float(pres_mae_mean), 5)
    with open(meta_json_path, 'w') as f:
        json.dump(meta_json, f, indent=2)
    log(f"  Updated metadata: {meta_json_path.name}")

# =============================================================================
# SECTION 8 — POST-PATCH VERIFICATION
# =============================================================================
log("\nSECTION 8 — Post-patch verification")

df_verify = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
lbl19_verify = df_verify[df_verify['label_int'].astype(int) == LABEL_19]
log(f"  Label 19 rows verified: {len(lbl19_verify):,}")

pres_col = 'mae_Pres_SV' if 'mae_Pres_SV' in lbl19_verify.columns else 'mae_PresSV'
if pres_col in lbl19_verify.columns:
    pres_mae_verify = float(lbl19_verify[pres_col].mean())
    log(f"  Pres.SV MAE (patched Label 19): {pres_mae_verify:.5f}")
    gate_pass = pres_mae_verify > 0.05
    log(f"  Physics gate (Pres.SV MAE > 0.05): {'PASS ✓' if gate_pass else 'FAIL ✗'}")
else:
    log(f"  [WARN] Column {pres_col} not found — check column naming")

# Verify other labels untouched
for lbl in [0, 1, 4, 21]:
    n = (df_verify['label_int'].astype(int) == lbl).sum()
    n_orig = (df['label_int'].astype(int) == lbl).sum()
    match = "✓" if n == n_orig else f"✗ (was {n_orig})"
    log(f"  Label {lbl:2d} rows: {n:,} {match}")

# =============================================================================
# SECTION 9 — PASTE TEXT UPDATE
# =============================================================================
print("\n" + "═"*70)
print("══ PASTE TEXT UPDATE ══")
print("═"*70)
print(f"""
LABEL19_FEATURE_PATCH ({date.today()})
label19_feature_patch_applied      : True
label19_new_feature_rows           : {len(new_rows)}
label19_pres_mae_mean_after        : {round(float(pres_mae_mean),5)}
feature_matrix_total_rows          : {len(df_patched)}
feature_matrix_path                : data/synthetic/M6B_feature_matrix.csv
feature_matrix_backup              : M6B_feature_matrix_pre_label19_feature_patch.csv
NEXT_ACTION                        : Re-run module_07_xgboost_classifier.py
""")
print("═"*70)

log(f"\n{'='*70}")
log(f"  Label 19 Feature Patch COMPLETE")
log(f"  New rows: {len(new_rows):,} | Pres.SV MAE: {pres_mae_mean:.5f}")
log(f"  Next: Run module_07_xgboost_classifier.py")
log(f"{'='*70}")
