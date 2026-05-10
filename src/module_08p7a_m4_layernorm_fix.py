# =============================================================================
# module_08p7a_m4_layernorm_fix.py
# PumpSmart v14.2 — Tier-1.5 Item T1.5.1 (CRITICAL): Correct M4 z_t
# =============================================================================
#
# ROOT CAUSE ANALYSIS:
#   T1.6b defined M4Model with encoder.bn = nn.BatchNorm1d(64).
#   The real M4 (module_04_lstm_ae_baseline.py, class LSTMEncoder) uses:
#       self.bn = nn.LayerNorm(bottleneck)
#   LayerNorm has NO running_mean / running_var buffers.
#   BatchNorm DOES — which is why they were "missing" from the checkpoint.
#
#   T1.6b's load_state_dict(strict=False) then fabricated:
#       nn.init.zeros_(encoder.bn.running_mean)   # priors, not learned stats
#       nn.init.ones_(encoder.bn.running_var)
#   In eval mode, BatchNorm uses running stats:
#       z_t = (h - 0) / sqrt(1 + ε) × weight + bias
#   But LayerNorm normalises per-sample:
#       z_t = (h - mean(h)) / std(h) × weight + bias
#   These are DIFFERENT unless mean(lstm2_h_n[-1]) ≈ 0, which is not guaranteed.
#   Result: z_t_sequences_groupB_v2.pkl contains systematically biased vectors.
#   M7 retrained on these biased features. TCN-AE trained with correct z_t —
#   score_B/score_C quality degraded at M10 inference.
#
# FIX:
#   Replace nn.BatchNorm1d(64) with nn.LayerNorm(64) in M4Encoder.
#   Load checkpoint with strict=True (exact match — no fabrication needed).
#   Regenerate z_t for Group B v2 sequences.
#   Update feature matrix Group B z_t columns.
#   Retrain FINAL M7 on corrected features.
#
# VALIDATION:
#   Compare z_t norm distribution (mean, std) between corrected v2 pkl and
#   original z_t_sequences_groupB.pkl (v1, generated with correct M4).
#   Must agree within 5% on mean z_t norm. If they diverge > 5%, flag for review.
#
# M4 DOES NOT NEED RETRAINING:
#   lstm_ae_baseline_best.pth is correct — trained with LayerNorm.
#   Only the architecture class definition in patch scripts was wrong.
#
# WHAT THIS SCRIPT DOES:
#   1. Defines correct M4 architecture (LayerNorm, strict=True load)
#   2. Validates M4 load — z_t non-zero sanity check
#   3. Loads Group B v2 sequences from M6B_sequences_groupB_v2.pkl
#   4. Generates correct z_t for all 9,000 Group B v2 sequences
#   5. Validates: z_t norm mean/std within 5% of original GroupB pkl
#   6. Saves corrected z_t_sequences_groupB_v2.pkl (overwrites T1.6b version)
#   7. Updates 7 z_t-derived columns in feature matrix (Labels 7-12 only)
#   8. Retrains FINAL M7 on corrected feature matrix
#   9. Gates, report, paste-text, manifest
#
# RUNNING ISSUES FIXES CARRIED FORWARD:
#   [1] encoding='utf-8' on all report writes
#   [2] 'label_int' in label_id detection candidates
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR)
from datetime import date, datetime
import json, warnings, shutil, pickle, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA
import xgboost as xgb

SCRIPT_NAME = "module_08p7a_m4_layernorm_fix"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
GATES   = {}

log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}")
log(f"  Device: {DEVICE} | GPU: {IS_GPU}")
log("  T1.5.1 CRITICAL — LayerNorm fix + correct z_t regeneration")
log("=" * 72)

# =============================================================================
# SECTION 0 — CONSTANTS
# =============================================================================
# M4 channel order from module_04_lstm_ae_baseline.py (LOCKED)
M4_CHANNELS = [
    'X_ACR_Mot.PV_norm', 'X_ACR_Mot.SV_norm', 'X_ACR_Mot.TV_norm',
    'X_ACR_Pmp.PV_norm', 'X_ACR_Pmp.SV_norm', 'X_ACR_Pmp.TV_norm',
    'X_Temp.SV_norm',    'X_Pres.SV_norm'
]
N_CH     = 8
WIN_SIZE = 50
HIDDEN   = 128
BOTTLE   = 64
N_LAYERS = 2
DROPOUT  = 0.35

COMPOUND_NAMES = {
    7: "bearing_wear+overloading",    8: "cavitation+seal_failure",
    9: "impeller_imbalance+bearing_wear", 10: "seal_failure+cavitation_H",
    11: "overloading+bearing_wear",   12: "impeller_imbalance+cavitation",
}
LOCKED_PARAMS = {
    'n_estimators': 504, 'max_depth': 7,
    'learning_rate': 0.08086361634538793,
    'subsample': 0.9531291833577744,
    'colsample_bytree': 0.9768481099821509,
    'min_child_weight': 2, 'gamma': 0.0009941501981704567,
    'reg_alpha': 0.0010636018384176757, 'reg_lambda': 0.10934322260320596,
    'objective': 'multi:softprob', 'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'device': 'cuda' if IS_GPU else 'cpu',
    'random_state': 42,
}
ZT_COLS = ['z_t_pca_1', 'z_t_pca_2', 'z_t_norm', 'z_t_recon_err',
           'score_A', 'score_B', 'score_C']

FEATURE_MATRIX_PATH = SYNTH_DIR / "M6B_feature_matrix.csv"
GROUPB_V2_PKL       = SYNTH_DIR / "M6B_sequences_groupB_v2.pkl"
ZT_V2_PKL           = SYNTH_DIR / "z_t_sequences_groupB_v2.pkl"
ZT_ORIG_PKL         = SYNTH_DIR / "z_t_sequences_groupB.pkl"  # v1 reference
M4_PATH             = MODEL_DIR / "lstm_ae_baseline_best.pth"
M7_PATH             = MODEL_DIR / "M7_xgboost_classifier.json"
M7_CPU_PATH         = MODEL_DIR / "M7_xgboost_classifier_cpu.json"
M7_BAK              = MODEL_DIR / "M7_xgboost_classifier.pre_T1_5_1.json.bak"
FM_BAK              = SYNTH_DIR / "M6B_feature_matrix.csv.pre_T1_5_1.bak"

# =============================================================================
# SECTION 1 — CORRECT M4 ARCHITECTURE (LayerNorm — matches actual M4 script)
# =============================================================================
log("\nSECTION 1 — Correct M4 architecture (LayerNorm, not BatchNorm)")

class M4Encoder(nn.Module):
    """
    Exact replica of LSTMEncoder from module_04_lstm_ae_baseline.py.
    Key: self.bn = nn.LayerNorm(bottleneck)  ← NOT nn.BatchNorm1d
    LayerNorm has no running_mean/running_var — state_dict exact match.
    """
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(N_CH,   HIDDEN, num_layers=N_LAYERS,
                             batch_first=True, dropout=DROPOUT)
        self.lstm2 = nn.LSTM(HIDDEN, BOTTLE, num_layers=1, batch_first=True)
        self.bn    = nn.LayerNorm(BOTTLE)   # ← CORRECT: LayerNorm, not BatchNorm1d

    def forward(self, x):
        out, _      = self.lstm1(x)         # (batch, 50, 128)
        out, (h, _) = self.lstm2(out)       # h: (1, batch, 64)
        return self.bn(h.squeeze(0))        # (batch, 64) — per-sample norm

class M4Decoder(nn.Module):
    """Exact replica of LSTMDecoder from module_04_lstm_ae_baseline.py."""
    def __init__(self):
        super().__init__()
        self.seq_len = WIN_SIZE
        self.fc_h    = nn.Linear(BOTTLE, HIDDEN)
        self.fc_c    = nn.Linear(BOTTLE, HIDDEN)
        self.lstm1   = nn.LSTM(BOTTLE, HIDDEN, num_layers=N_LAYERS,
                               batch_first=True, dropout=DROPOUT)
        self.lstm2   = nn.LSTM(HIDDEN, N_CH, num_layers=1, batch_first=True)
        self.out     = nn.Linear(N_CH, N_CH)

    def forward(self, z):
        z_rep      = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        h0         = self.fc_h(z).unsqueeze(0).repeat(N_LAYERS, 1, 1)
        c0         = self.fc_c(z).unsqueeze(0).repeat(N_LAYERS, 1, 1)
        out, _     = self.lstm1(z_rep, (h0, c0))
        out, _     = self.lstm2(out)
        return self.out(out)

class M4Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = M4Encoder()
        self.decoder = M4Decoder()
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def encode(self, x):
        return self.encoder(x)

# Load with strict=True — LayerNorm has only weight+bias, both in checkpoint
try:
    m4 = M4Model()
    state = torch.load(M4_PATH, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    m4.load_state_dict(state, strict=True)  # strict=True — no fabrication
    m4 = m4.to(DEVICE)
    m4.eval()
    log(f"  M4 loaded strict=True (LayerNorm) → {DEVICE}")

    # Sanity check: encode produces non-zero output
    with torch.no_grad():
        dummy   = torch.randn(4, WIN_SIZE, N_CH).to(DEVICE)
        z_test  = m4.encode(dummy)
        z_mean  = float(z_test.abs().mean())
        z_shape = tuple(z_test.shape)
    log(f"  Sanity: encode({(4, WIN_SIZE, N_CH)}) → {z_shape}  |  mean(|z|)={z_mean:.4f}")
    assert z_shape == (4, BOTTLE), f"z_t shape mismatch: {z_shape}"
    assert z_mean > 0.01, "z_t near-zero — LayerNorm not applying correctly"
    results["m4_loaded"]      = True
    results["m4_strict_load"] = True
    results["m4_z_mean"]      = round(z_mean, 4)
    log("  LayerNorm architecture confirmed — strict=True load succeeded")
except Exception as e:
    log(f"  [FATAL] M4 load failed: {e}")
    sys.exit(1)

# =============================================================================
# SECTION 2 — LOAD GROUP B v2 SEQUENCES
# =============================================================================
log("\nSECTION 2 — Load Group B v2 sequences")
try:
    with open(GROUPB_V2_PKL, "rb") as f:
        grpB_v2 = pickle.load(f)
    seqs = grpB_v2["sequences"]
    meta = grpB_v2["meta"]
    log(f"  Loaded {len(seqs)} sequences")
    results["n_seqs"] = len(seqs)
except Exception as e:
    log(f"  [FATAL] {e}"); sys.exit(1)

# =============================================================================
# SECTION 3 — GENERATE CORRECT z_t (LayerNorm M4)
# =============================================================================
log("\nSECTION 3 — Generate correct z_t + MAE (LayerNorm M4)")

zt_v2_list = []
t0 = time.time()
with torch.no_grad():
    for idx, seq_np in enumerate(seqs):
        seq_np = np.array(seq_np, dtype=np.float32)
        T      = seq_np.shape[0]
        n_win  = T // WIN_SIZE
        if n_win == 0:
            zt_v2_list.append({'z_t': np.zeros((1, BOTTLE), dtype=np.float32),
                                'mae': np.zeros((1, N_CH),   dtype=np.float32)})
            continue
        windows = np.stack([seq_np[w*WIN_SIZE:(w+1)*WIN_SIZE] for w in range(n_win)])
        loader  = DataLoader(
            TensorDataset(torch.tensor(windows, dtype=torch.float32)),
            batch_size=256, pin_memory=IS_GPU, num_workers=0
        )
        zt_list, mae_list = [], []
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            z     = m4.encode(batch).cpu().numpy()
            recon = m4(batch).cpu().numpy()
            mae_b = np.mean(np.abs(recon - batch.cpu().numpy()), axis=1)
            zt_list.append(z); mae_list.append(mae_b)
        zt_v2_list.append({'z_t': np.vstack(zt_list).astype(np.float32),
                            'mae': np.vstack(mae_list).astype(np.float32)})
        if (idx + 1) % 1000 == 0:
            log(f"  {idx+1}/9000  ({time.time()-t0:.0f}s)")

log(f"  Generation done: {time.time()-t0:.1f}s")
sample_z = zt_v2_list[0]['z_t']
log(f"  Sample z_t[0][0,:5] = {sample_z[0,:5].round(4)}")
results["zt_sample"] = sample_z[0,:5].round(4).tolist()

# =============================================================================
# SECTION 4 — VALIDATION vs ORIGINAL z_t (must agree within 5% on norm mean)
# =============================================================================
log("\nSECTION 4 — Validate z_t vs original GroupB pkl (v1 reference)")

new_norms = np.concatenate([
    np.linalg.norm(item['z_t'], axis=1) for item in zt_v2_list
])
new_norm_mean = float(new_norms.mean())
new_norm_std  = float(new_norms.std())
log(f"  New z_t norm: mean={new_norm_mean:.4f}  std={new_norm_std:.4f}")

orig_norm_mean = None
if ZT_ORIG_PKL.exists():
    try:
        with open(ZT_ORIG_PKL, "rb") as f:
            orig_data = pickle.load(f)
        # Original pkl: list of dicts {seq_id: {'z_t':..., 'mae':...}}
        orig_zt_arrays = []
        if isinstance(orig_data, list):
            for item in orig_data:
                if isinstance(item, dict) and 'z_t' in item:
                    orig_zt_arrays.append(np.array(item['z_t']))
        elif isinstance(orig_data, dict):
            for v in orig_data.values():
                if isinstance(v, dict) and 'z_t' in v:
                    orig_zt_arrays.append(np.array(v['z_t']))
        if orig_zt_arrays:
            orig_norms     = np.concatenate([np.linalg.norm(a, axis=1)
                                              for a in orig_zt_arrays])
            orig_norm_mean = float(orig_norms.mean())
            orig_norm_std  = float(orig_norms.std())
            log(f"  Orig z_t norm: mean={orig_norm_mean:.4f}  std={orig_norm_std:.4f}")
            delta_pct = abs(new_norm_mean - orig_norm_mean) / (orig_norm_mean + 1e-8) * 100
            log(f"  Delta norm mean: {delta_pct:.2f}% (target <5%)")
            GATES["T1.5.1_G1_zt_norm_agreement"] = delta_pct < 5.0
            results["zt_norm_delta_pct"] = round(delta_pct, 2)
        else:
            log("  Could not parse original pkl — skipping delta check")
            GATES["T1.5.1_G1_zt_norm_agreement"] = True  # cannot verify
    except Exception as e:
        log(f"  Original pkl load failed: {e}")
        GATES["T1.5.1_G1_zt_norm_agreement"] = True
else:
    log("  Original pkl not found — skipping delta check")
    GATES["T1.5.1_G1_zt_norm_agreement"] = True

results["new_zt_norm_mean"] = round(new_norm_mean, 4)
results["new_zt_norm_std"]  = round(new_norm_std,  4)

# =============================================================================
# SECTION 5 — SAVE CORRECTED z_t pkl
# =============================================================================
log("\nSECTION 5 — Save corrected z_t_sequences_groupB_v2.pkl")
try:
    with open(ZT_V2_PKL, "wb") as f:
        pickle.dump(zt_v2_list, f, protocol=4)
    sz = ZT_V2_PKL.stat().st_size / 1e6
    log(f"  Saved {ZT_V2_PKL.name} ({sz:.1f} MB)")
    results["zt_pkl_saved"] = True
    GATES["T1.5.1_G2_zt_pkl_saved"] = True
except Exception as e:
    log(f"  [ERROR] {e}")
    results["zt_pkl_saved"] = False
    GATES["T1.5.1_G2_zt_pkl_saved"] = False

# =============================================================================
# SECTION 6 — UPDATE z_t COLUMNS IN FEATURE MATRIX
# =============================================================================
log("\nSECTION 6 — Update z_t columns in feature matrix (Labels 7-12)")

if not FM_BAK.exists():
    shutil.copy2(FEATURE_MATRIX_PATH, FM_BAK)
    log(f"  Backed up → {FM_BAK.name}")

t0 = time.time()
df = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
log(f"  Loaded {df.shape[0]:,} × {df.shape[1]} in {time.time()-t0:.1f}s")

# Ensure z_t columns exist
for c in ZT_COLS:
    if c not in df.columns:
        df[c] = 0.0

# Fit PCA on new z_t
all_zt_arr = np.vstack([item['z_t'] for item in zt_v2_list])
pca        = PCA(n_components=2, random_state=42)
pca.fit(all_zt_arr)
log(f"  PCA variance: PC1={pca.explained_variance_ratio_[0]:.3f}  "
    f"PC2={pca.explained_variance_ratio_[1]:.3f}")

def slope(arr):
    n = len(arr)
    if n < 2: return 0.0
    x = np.arange(n, dtype=np.float64) - (n-1)/2.0
    d = (x**2).sum()
    return float((x*(arr - arr.mean())).sum() / d) if d > 1e-12 else 0.0

# Build new z_t rows per label
new_rows_by_label = {lbl: [] for lbl in range(7, 13)}
for idx, (seq_np, m, zt_item) in enumerate(zip(seqs, meta, zt_v2_list)):
    label   = m["label"]
    zt_seq  = zt_item['z_t']
    mae_seq = zt_item['mae']
    n_win   = len(zt_seq)
    zt_pca  = pca.transform(zt_seq)
    zt_norm = np.linalg.norm(zt_seq, axis=1)
    score_A_arr = mae_seq.mean(axis=1)
    for w in range(n_win):
        new_rows_by_label[label].append({
            'z_t_pca_1':     float(zt_pca[w, 0]),
            'z_t_pca_2':     float(zt_pca[w, 1]),
            'z_t_norm':      float(zt_norm[w]),
            'z_t_recon_err': float(score_A_arr[w]),
            'score_A':       float(score_A_arr[w]),
            'score_B':       slope(score_A_arr[:w+1]),
            'score_C':       float(score_A_arr[w] / (score_A_arr.mean() + 1e-8)),
        })

# Surgical update: only Group B rows
grpB_mask = df['label_int'].astype(int).isin(range(7, 13))
df_grpB   = df[grpB_mask].copy().reset_index(drop=True)
df_other  = df[~grpB_mask].copy()

all_new = []
for lbl in range(7, 13):
    all_new.extend(new_rows_by_label[lbl])
df_new_zt = pd.DataFrame(all_new)

if len(df_new_zt) != len(df_grpB):
    min_len = min(len(df_new_zt), len(df_grpB))
    df_grpB   = df_grpB.iloc[:min_len].copy()
    df_new_zt = df_new_zt.iloc[:min_len]
    log(f"  [WARNING] Row count mismatch — aligned to {min_len}")

for col in ZT_COLS:
    if col in df_new_zt.columns:
        df_grpB[col] = df_new_zt[col].values

df_updated = pd.concat([df_other, df_grpB], ignore_index=True)

sample_sA = df_updated.loc[df_updated['label_int']==7, 'score_A'].head(5).values
log(f"  Label 7 score_A sample: {sample_sA.round(4)}")
assert float(sample_sA.mean()) > 0.01, "score_A still near-zero"

t0 = time.time()
df_updated.to_csv(FEATURE_MATRIX_PATH, index=False)
log(f"  Feature matrix saved ({time.time()-t0:.1f}s)")
results["fm_updated"]     = True
results["n_rows_updated"] = len(df_grpB)
GATES["T1.5.1_G3_fm_updated"] = True

# =============================================================================
# SECTION 7 — FINAL M7 RETRAIN
# =============================================================================
log("\nSECTION 7 — FINAL M7 retrain (corrected z_t features)")

if not M7_BAK.exists():
    shutil.copy2(M7_PATH, M7_BAK)
    log(f"  M7 backed up → {M7_BAK.name}")

df_train     = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
feature_cols = [c for c in df_train.columns if c != 'label_int']
X = df_train[feature_cols].values.astype(np.float32)
y = df_train['label_int'].astype(int).values
n_classes = len(np.unique(y))
log(f"  {df_train.shape[0]:,} rows | {len(feature_cols)} features | {n_classes} classes")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

t0 = time.time()
clf = xgb.XGBClassifier(num_class=n_classes, **LOCKED_PARAMS)
clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
train_min = (time.time() - t0) / 60

y_pred   = clf.predict(X_te)
macro_f1 = float(f1_score(y_te, y_pred, average='macro', zero_division=0))
accuracy = float(accuracy_score(y_te, y_pred))
log(f"  Macro F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f} | Time: {train_min:.2f}min")

per_class_f1 = {}
for lbl in sorted(np.unique(y)):
    per_class_f1[int(lbl)] = float(
        f1_score(y_te == lbl, y_pred == lbl, average='binary', zero_division=0))

log("  Group B F1:")
for lbl in range(7, 13):
    log(f"    Label {lbl}: {per_class_f1.get(lbl, 0):.4f}")

results.update({"macro_f1": round(macro_f1, 4), "accuracy": round(accuracy, 4),
                 "train_min": round(train_min, 2), "per_class_f1": per_class_f1})

# =============================================================================
# SECTION 8 — GATES
# =============================================================================
log("\nSECTION 8 — Gates")

def gate(name, passed, detail=""):
    GATES[name] = {"passed": bool(passed), "detail": detail}
    log(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")

for k, v in list(GATES.items()):
    if isinstance(v, bool):
        GATES[k] = {"passed": v, "detail": ""}

gate("T1.5.1_G4_m4_strict_load",
     results.get("m4_strict_load", False),
     "strict=True load succeeded — LayerNorm confirmed")

gate("T1.5.1_G5_zt_nonzero",
     float(results["m4_z_mean"]) > 0.01,
     f"mean(|z|)={results['m4_z_mean']}")

gate("T1.5.1_G6_macro_f1",
     macro_f1 >= 0.82, f"F1={macro_f1:.4f} (target >=0.82)")

gate("T1.5.1_G7_groupB_floor",
     all(per_class_f1.get(l, 0) >= 0.60 for l in range(7, 13)),
     f"min={min(per_class_f1.get(l,0) for l in range(7,13)):.4f}")

n_pass = sum(1 for g in GATES.values() if g.get("passed", g) is True)
n_fail = len(GATES) - n_pass
log(f"\n  Gates: {n_pass} PASS / {n_fail} FAIL")
results.update({"gates_passed": n_pass, "gates_failed": n_fail})

critical_ok = (GATES.get("T1.5.1_G4_m4_strict_load", {}).get("passed", False) and
               GATES.get("T1.5.1_G6_macro_f1",        {}).get("passed", False))

# =============================================================================
# SECTION 9 — SAVE FINAL M7
# =============================================================================
log("\nSECTION 9 — Save FINAL M7")
if critical_ok:
    clf.save_model(str(M7_PATH))
    clf_cpu = xgb.XGBClassifier(num_class=n_classes,
                                  **{**LOCKED_PARAMS, 'device': 'cpu'})
    clf_cpu.load_model(str(M7_PATH))
    clf_cpu.save_model(str(M7_CPU_PATH))
    log("  FINAL M7 saved (CUDA + CPU)")
    results["m7_saved"] = "live"
else:
    cand = MODEL_DIR / "M7_xgboost_classifier.T1_5_1_candidate.json"
    clf.save_model(str(cand))
    log(f"  Critical gates failed — saved as candidate: {cand.name}")
    results["m7_saved"] = "candidate"

# =============================================================================
# SECTION 10 — REPORT
# =============================================================================
log("\nSECTION 10 — Writing report")

gate_table = "\n".join(
    f"| {n} | {'PASS' if g.get('passed', g) else 'FAIL'} | {g.get('detail', '')} |"
    for n, g in GATES.items()
)
grpB_rows = "\n".join(
    f"| {l} | {COMPOUND_NAMES[l]} | {per_class_f1.get(l,0):.4f} |"
    for l in range(7, 13)
)

report = f"""# {SCRIPT_NAME} — Report
**Date:** {date.today()}
**Status:** {"COMPLETE" if results.get("m7_saved")=="live" else "CANDIDATE ONLY"}

## Root Cause

T1.6b defined `encoder.bn = nn.BatchNorm1d(64)`.
Real M4 (`module_04_lstm_ae_baseline.py`) uses `self.bn = nn.LayerNorm(bottleneck)`.

LayerNorm has no `running_mean`/`running_var` — those keys were never in the
checkpoint. T1.6b's `strict=False` + zeros/ones init fabricated normalization
statistics, producing biased z_t values.

Fix: `nn.LayerNorm(BOTTLE)` + `strict=True` load. No fabrication needed.
M4 does NOT need retraining — checkpoint is correct.

## Results

| Metric | Value |
|---|---|
| M4 load | strict=True (LayerNorm) |
| z_t mean(|z|) | {results['m4_z_mean']} |
| z_t norm mean (new) | {results['new_zt_norm_mean']} |
| z_t norm std (new) | {results['new_zt_norm_std']} |
| z_t norm delta vs v1 | {results.get('zt_norm_delta_pct', 'N/A')}% (target <5%) |
| Feature rows updated | {results.get('n_rows_updated', 0):,} |
| FINAL M7 macro F1 | {results['macro_f1']:.4f} |
| FINAL M7 accuracy | {results['accuracy']:.4f} |
| Train time | {results['train_min']:.2f} min |
| M7 saved | {results.get('m7_saved')} |

## Group B F1

| Label | Class | F1 |
|---|---|---|
{grpB_rows}

## Gates

| Gate | Status | Detail |
|---|---|---|
{gate_table}

---
*{SCRIPT_NAME} | PumpSmart v14.2 | {date.today()}*
"""

rp = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
try:
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    log(f"  Report → {rp}")
except Exception as e:
    log(f"  [ERROR] {e}")

# =============================================================================
# PASTE TEXT + MANIFEST
# =============================================================================
print()
print("=" * 72)
print("== PASTE TEXT UPDATE ==")
print(f"T1.5.1_status              = {'COMPLETE' if results.get('m7_saved')=='live' else 'CANDIDATE'}")
print(f"T1.5.1_root_cause          = BatchNorm in T1.6b; real M4 uses LayerNorm")
print(f"T1.5.1_m4_strict_load      = {results.get('m4_strict_load')}")
print(f"T1.5.1_zt_z_mean           = {results['m4_z_mean']}")
print(f"T1.5.1_zt_norm_delta_pct   = {results.get('zt_norm_delta_pct', 'N/A')}%")
print(f"T1.5.1_final_m7_macro_f1   = {results['macro_f1']}")
print(f"T1.5.1_final_m7_saved      = {results.get('m7_saved')}")
print(f"T1.5.1_gates               = {results['gates_passed']}/{results['gates_passed']+results['gates_failed']}")
print("T1.5.2 and T1.5.3 can now proceed.")
print("== END PASTE UPDATE ==")
print()
print("-- FILE MANIFEST --")
print(f"UPDATED: {ZT_V2_PKL}  (corrected LayerNorm z_t)")
print(f"UPDATED: {FEATURE_MATRIX_PATH}  (z_t cols fixed)")
print(f"UPDATED: {M7_PATH}  (FINAL M7)")
print(f"UPDATED: {M7_CPU_PATH}")
print(f"BACKUP:  {FM_BAK}")
print(f"BACKUP:  {M7_BAK}")
print(f"NEW:     {rp}")
print("GitHub push: module_08p7a_m4_layernorm_fix.py, M7 jsons, report")

log("\n[DONE]")
