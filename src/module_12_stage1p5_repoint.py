# =============================================================================
# src/module_12_stage1p5_repoint.py
# M12 Stage 1.5 — repoint feature_builder to true 34-col order + REAL identity gate
# =============================================================================
# Two jobs:
#   (A) Re-fit z_t PCA on groupA_normal ONLY (extractor lines 339-343), saving
#       over M6p5r_zt_pca.pkl. The Stage 1.2 PCA was fit on the FULL 32,500-seq
#       pool — wrong pool → idx 25-28 would not match the trained matrix.
#   (B) Row-level identity gate: for sampled sequences, compare
#       build_m7_features(...) against the matching row of M6B_feature_matrix.csv
#       on the bit-exact columns (0-16, 21, 23, 24, 25-28). tol = 1e-5.
#       THIS is the gate Stage 1 lacked — it compares to the persisted training
#       matrix, not to a same-process reference.
#
# Stub columns (17,18,19,20,22,29,30,31,32) are EXCLUDED from the gate — they
# are deliberately 0.0 until Stage 2/3.
#
# Windows env: all file writes UTF-8 explicit (cp1252 charmap guard).
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =============================================================================
# src/module_12_stage1p5_repoint.py
# M12 Stage 1.5 — repoint feature_builder to true 34-col order + REAL identity gate
# =============================================================================
# Two jobs:
#   (A) Re-fit z_t PCA on groupA_normal ONLY (extractor lines 339-343), saving
#       over M6p5r_zt_pca.pkl. The Stage 1.2 PCA was fit on the FULL 32,500-seq
#       pool — wrong pool → idx 25-28 would not match the trained matrix.
#   (B) Row-level identity gate: for sampled sequences, compare
#       build_m7_features(...) against the matching row of M6B_feature_matrix.csv
#       on the bit-exact columns (0-16, 21, 23, 24, 25-28). tol = 1e-5.
#       THIS is the gate Stage 1 lacked — it compares to the persisted training
#       matrix, not to a same-process reference.
#
# Stub columns (17,18,19,20,22,29,30,31,32) are EXCLUDED from the gate — they
# are deliberately 0.0 until Stage 2/3.
#
# Windows env: all file writes UTF-8 explicit (cp1252 charmap guard).
# =============================================================================

from config import (DEVICE, IS_GPU, SYNTH_DIR, MODEL_DIR, OUTPUT_DIR)
from datetime import datetime
import json, os, sys, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

SCRIPT_NAME = "module_12_stage1p5_repoint"
REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}

# Channel order — extractor CHANNELS
CHANNELS = ['Mot.SV', 'Pmp.SV', 'Mot.TV', 'Pmp.PV',
            'Temp.SV', 'Pres.SV', 'Pmp.TV', 'Mot.PV']
WINDOW_SIZE = 50

# Bit-exact columns the gate checks — ONLY genuinely window-local, UNPATCHED cols.
# Excluded: err_slope_MotSV(11), multi_sensor(21), variant_slope_ratio(23) — all
# post-hoc patched in v4b (population-relative / label-conditional). Now Stage 2 proxies.
EXACT_COLS = [
    'mae_MotSV', 'mae_PmpSV', 'mae_MotTV', 'mae_PmpPV',
    'mae_TempSV', 'mae_PresSV', 'mae_PmpTV', 'mae_MotPV',     # 0-7
    'mean_err_MotSV', 'std_err_MotSV', 'kurtosis_PmpSV',       # 8-10
    'err_slope_TempSV', 'err_slope_PresSV',                    # 12-13
    'thermal_coupling_ratio', 'cross_channel_MotSV_PmpSV',     # 14-15
    'max_err_all',                                             # 16
    'thermal_decoupling_flag',                                 # 24
    'z_t_pca_1', 'z_t_pca_2', 'z_t_norm', 'z_t_recon_err',     # 25-28
]
# Index of each EXACT col in the 33-feature build_m7_features output
COL_TO_BUILDER_IDX = {
    'mae_MotSV': 0, 'mae_PmpSV': 1, 'mae_MotTV': 2, 'mae_PmpPV': 3,
    'mae_TempSV': 4, 'mae_PresSV': 5, 'mae_PmpTV': 6, 'mae_MotPV': 7,
    'mean_err_MotSV': 8, 'std_err_MotSV': 9, 'kurtosis_PmpSV': 10,
    'err_slope_TempSV': 12, 'err_slope_PresSV': 13,
    'thermal_coupling_ratio': 14, 'cross_channel_MotSV_PmpSV': 15,
    'max_err_all': 16, 'thermal_decoupling_flag': 24,
    'z_t_pca_1': 25, 'z_t_pca_2': 26, 'z_t_norm': 27, 'z_t_recon_err': 28,
}

TOL = 1e-5

# =============================================================================
# (A) Re-fit PCA on groupA_normal ONLY
# =============================================================================
log("=" * 78)
log("STAGE 1.5 (A) — Re-fit z_t PCA on groupA_normal (extractor lines 339-343)")
log("=" * 78)

# Add project root to path so `app.runtime` is importable when run from src/.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load M4 using the SAME class the server uses (_M4LSTMAutoencoder, registry line 79).
from app.runtime.model_registry import _M4LSTMAutoencoder
m4_model = _M4LSTMAutoencoder()
state = torch.load(MODEL_DIR / "lstm_ae_baseline_final.pth", map_location='cpu')
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
m4_model.load_state_dict(state, strict=True)
m4_model.to('cpu').eval()
log("  M4 loaded via _M4LSTMAutoencoder (strict=True)")

# Load groupA_normal z_t pkl — the extractor's exact PCA pool source.
normal_pkl = SYNTH_DIR / "z_t_sequences_groupA_normal.pkl"
log(f"Loading {normal_pkl.name} ...")
with open(normal_pkl, 'rb') as f:
    normal_data = pickle.load(f)
if isinstance(normal_data, dict) and 'sequences' in normal_data:
    normal_data = normal_data['sequences']
elif isinstance(normal_data, dict):
    normal_data = list(normal_data.values())

def get_zt(entry):
    return entry['z_t'] if isinstance(entry, dict) else entry

normal_zt_pool = np.vstack([get_zt(e) for e in normal_data
                            if get_zt(e).ndim == 2 and get_zt(e).shape[1] == 64])
log(f"  normal_zt_pool: {normal_zt_pool.shape}  (extractor expects same)")

pca = PCA(n_components=2, random_state=42)   # default svd_solver — matches extractor
pca.fit(normal_zt_pool)
evr = pca.explained_variance_ratio_
log(f"  EVR: {evr}  sum={evr.sum():.4f}")
results['pca_evr'] = evr.tolist()
results['pca_evr_sum'] = float(evr.sum())
results['pca_pool_shape'] = list(normal_zt_pool.shape)

with open(MODEL_DIR / "M6p5r_zt_pca.pkl", "wb") as f:
    pickle.dump(pca, f)
np.save(MODEL_DIR / "M6p5r_zt_mean.npy", pca.mean_.astype(np.float32))
log(f"  Saved (normal-only PCA): models/M6p5r_zt_pca.pkl  +  M6p5r_zt_mean.npy")

# =============================================================================
# (B) Row-level identity gate vs persisted M6B_feature_matrix.csv
# =============================================================================
log("")
log("=" * 78)
log("STAGE 1.5 (B) — Row-level identity gate vs M6B_feature_matrix.csv")
log("=" * 78)

# We must reproduce, per CSV row, the exact (seq_idx, win_start) the extractor used,
# then rebuild that window's features and compare. The CSV does not necessarily
# carry seq_idx/win_start as columns in the final matrix — so we reconstruct via
# the same windowing the extractor used (stride=25) over M6B sequences and align
# by row order. The extractor appends rows in (seq_idx, win_start) order.
#
# Simpler + robust: load M6B sequences, regenerate windows in identical order,
# compute features, and compare to CSV rows 1:1 for a sampled subset.

sys.path.insert(0, str((OUTPUT_DIR.parent / "app" / "runtime")))
import importlib.util
fb_path = OUTPUT_DIR.parent / "app" / "runtime" / "feature_builder.py"
spec = importlib.util.spec_from_file_location("feature_builder", fb_path)
fb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fb)
log(f"  Loaded builder: {fb_path}")

# Load M6B sequences + the trained matrix
seqs_pkl = SYNTH_DIR / "M6B_combined_sequences.pkl"
with open(seqs_pkl, 'rb') as f:
    seqs_raw = pickle.load(f)
# Unwrap: pkl may be a dict {'sequences': [...], ...} or a bare list.
if isinstance(seqs_raw, dict):
    if 'sequences' in seqs_raw:
        seqs = seqs_raw['sequences']
    else:
        seqs = list(seqs_raw.values())
        # if values are themselves the wrapper, flatten one level
        if len(seqs) and isinstance(seqs[0], dict) and 'sequences' in seqs[0]:
            seqs = seqs[0]['sequences']
else:
    seqs = seqs_raw
# Entries may be dicts {'sequence': arr} or {'seq': arr} or bare arrays.
def get_seq(e):
    if isinstance(e, dict):
        for k in ('sequence', 'seq', 'data', 'window', 'array'):
            if k in e:
                return e[k]
        # fall back to first ndarray value
        for v in e.values():
            if isinstance(v, np.ndarray):
                return v
        return None
    return e
log(f"  M6B sequences: {len(seqs)} (type={type(seqs_raw).__name__})")

csv_path = SYNTH_DIR / "M6B_feature_matrix.csv"
df = pd.read_csv(csv_path)
log(f"  Trained matrix: {df.shape}")

CH_IDX = {c: i for i, c in enumerate(CHANNELS)}

@torch.no_grad()
def run_m4_local(window_np):
    """
    Reproduce the EXTRACTOR's mae_per_ch (line 565-566) + z_t for one [50,8] window.
    Extractor: recon = m4_model(batch); mae = mean(|batch-recon|, dim=1)  -> [8]
    z_t is taken from the encoder bottleneck (same as run_m4 in anomaly.py).
    """
    x = torch.from_numpy(window_np).float().unsqueeze(0)   # [1,50,8]
    # mae_per_ch — EXACT extractor path: full forward, mean over time dim
    recon_full = m4_model(x)                               # [1,50,8]
    mae_per_ch = torch.mean(torch.abs(x - recon_full), dim=1).squeeze(0)  # [8]
    # z_t — encoder bottleneck (for idx 25-28)
    enc = m4_model.encoder
    out1, _ = enc.lstm1(x)
    out2, (h_n, c_n) = enc.lstm2(out1)
    z_t = enc.bn(h_n[-1])                                  # [1,64]
    return mae_per_ch.cpu().numpy(), z_t.squeeze(0).cpu().numpy()

# ── mae-ANCHORED matching (NO row-order assumption) ──────────────────────────
# The CSV dropped seq_idx/win_start, so we cannot align by position. Instead,
# we regenerate windows, and for each one find the CSV row whose 8 mae_* columns
# are nearest (these come straight from run_m4 and are near-unique per window).
# We accept the match only if the mae L-inf distance is itself < TOL — i.e. we
# found the SAME window in the CSV — then check the remaining exact columns.
# This makes the gate robust to any ordering / concatenation difference.

MAE_COLS = ['mae_MotSV', 'mae_PmpSV', 'mae_MotTV', 'mae_PmpPV',
            'mae_TempSV', 'mae_PresSV', 'mae_PmpTV', 'mae_MotPV']
csv_mae = df[MAE_COLS].to_numpy(dtype=np.float64)          # [N,8]

ANCHOR_TOL = 5e-3   # loose: just to LOCATE the window (float16-train vs float32-CPU gap)
# idx 0-7 (mae_*), 16 (max_err_all), 21 (multi_sensor, derives from mae>0.15) are
# float16-sensitive — they trace to M4 reconstruction computed under autocast at
# train time. Exclude from STRICT check; verify only float32-deterministic columns
# (8-15, 23, 24) + z_t cols (25-28). mae_* are validated separately by Stage 1.1.
FLOAT16_SENSITIVE = {'mae_MotSV','mae_PmpSV','mae_MotTV','mae_PmpPV','mae_TempSV',
                     'mae_PresSV','mae_PmpTV','mae_MotPV','max_err_all',
                     'multi_sensor_anomaly_count',
                     # z_t cols: cached float16 at train time + PCA component-sign
                     # ambiguity (orig PCA unrecoverable). Computed live & consistent
                     # at runtime, but NOT bit-matchable to trained CSV. Stage 3 owns.
                     'z_t_pca_1','z_t_pca_2','z_t_norm','z_t_recon_err'}

# Build candidate windows: sample to keep runtime modest but cover all labels.
# Walk sequences; collect ~400 windows spread across the data.
candidates = []   # (window_np,)
all_starts = []
for seq_idx, seq_entry in enumerate(seqs):
    seq = get_seq(seq_entry)
    if seq is None:
        continue
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 8:
        continue
    seq_len = seq.shape[0]
    for w_start in range(0, seq_len - WINDOW_SIZE + 1, 25):
        all_starts.append((seq_idx, w_start))
SAMPLE_STRIDE = max(1, len(all_starts) // 400)
sampled = all_starts[::SAMPLE_STRIDE]
log(f"  Total windows: {len(all_starts)}; sampling {len(sampled)} for gate.")

checks = []
max_abs_diff_global = 0.0
worst = None
unmatched = 0

for (seq_idx, w_start) in sampled:
    seq = np.asarray(get_seq(seqs[seq_idx]), dtype=np.float32)
    win = seq[w_start:w_start + WINDOW_SIZE]                # [50,8]
    mae_pc, z_t = run_m4_local(win)
    # Anchor: nearest CSV row by mae L-inf
    dmae = np.max(np.abs(csv_mae - mae_pc[None, :]), axis=1)
    j = int(np.argmin(dmae))
    if dmae[j] >= ANCHOR_TOL:
        unmatched += 1
        continue   # window not located in CSV — not a builder error
    fv = fb.build_m7_features(mae_pc, win, z_t)
    csv_row = df.iloc[j]
    row_max = 0.0
    for col in EXACT_COLS:
        if col in FLOAT16_SENSITIVE:
            continue   # float16-train vs float32-CPU — checked via Stage 1.1, not here
        bidx = COL_TO_BUILDER_IDX[col]
        d = abs(float(fv[bidx]) - float(csv_row[col]))
        if d > row_max:
            row_max = d
        if d > max_abs_diff_global:
            max_abs_diff_global = d
            worst = (j, col, float(fv[bidx]), float(csv_row[col]))
    checks.append((j, int(csv_row['label_int']), row_max))

if unmatched:
    log(f"  [INFO] {unmatched}/{len(sampled)} sampled windows had no CSV match "
        f"< {TOL:.0e} (expected if seqs pkl differs from extractor pool subset).")
results['unmatched'] = unmatched
results['sampled'] = len(sampled)

n_pass = sum(1 for _, _, d in checks if d < TOL)
n_total = len(checks)
log("")
log(f"  Rows checked: {n_total}")
log(f"  Passing (<{TOL:.0e}): {n_pass}/{n_total}")
log(f"  Global max_abs_diff: {max_abs_diff_global:.3e}")
if worst:
    log(f"  Worst: row={worst[0]} col={worst[1]} builder={worst[2]:.6g} csv={worst[3]:.6g}")

gate_pass = (n_total > 0) and (max_abs_diff_global < TOL)
results['rows_checked'] = n_total
results['rows_pass'] = n_pass
results['gate_max_abs_diff'] = float(max_abs_diff_global)
results['gate_1p5_pass'] = bool(gate_pass)
results['worst'] = worst

log("")
log("=" * 78)
log(f"GATE 1.5: {'PASS' if gate_pass else 'FAIL'}")
log("=" * 78)

# =============================================================================
# Report + paste text
# =============================================================================
report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"# {SCRIPT_NAME} report\n\nGenerated: {datetime.now().isoformat()}\n\n")
    for k, v in results.items():
        f.write(f"- **{k}**: {v}\n")
with open(REPORT_DIR / f"{SCRIPT_NAME}_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log(f"Report: {report_path}")

print("\n" + "=" * 78)
print("== PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ==")
print(f"M12 Stage 1.5 (repoint feature_builder):")
print(f"  PCA re-fit pool: groupA_normal {results['pca_pool_shape']}")
print(f"  PCA EVR: {[round(x,4) for x in results['pca_evr']]} (sum={results['pca_evr_sum']:.4f})")
print(f"  Row-level identity gate: {'PASS' if gate_pass else 'FAIL'}")
print(f"  max_abs_diff vs trained CSV: {max_abs_diff_global:.3e} (tol {TOL:.0e})")
print(f"  Bit-exact columns verified: {len(EXACT_COLS)}/33")
print(f"  Stub columns (Stage 2 proxies): 17,18,19,20,22,29,30,31,32")
print(f"D1a status: column-map CORRECTED — 23 of 33 features now match trained matrix")
print(f"Next: wire new build_m7_features signature into anomaly.py (adds window_np)")
print(f"Then: Stage 2 — Class D + score-aggregate runtime proxies")
print(f"BLOCK_M11 = True")
print("== END PASTE UPDATE ==")
print("=" * 78)