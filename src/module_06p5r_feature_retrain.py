# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  module_06p5r_feature_retrain.py  v2.0                                      ║
# ║  PumpSmart v14.2 — M6.5r Feature Matrix Re-Extraction                       ║
# ║  FIX: NaN secondary_onset_step crash + label_map parsing                    ║
# ║  NEW: Physics fault-progression visualization (Section 11A)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings, pickle, gc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress, kurtosis, pearsonr
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

SCRIPT_NAME  = "module_06p5r_feature_retrain"
ARCH_VERSION = "v14.2"
REPORT_DIR   = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
gates   = {}

log("=" * 75)
log(f"  PumpSmart — M6.5r Feature Matrix Re-Extraction  v2.0")
log(f"  Script: {SCRIPT_NAME} | Arch: {ARCH_VERSION} | Date: {date.today()}")
log(f"  Device: {DEVICE} | CUDA: {IS_GPU}")
log("=" * 75)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 50
STRIDE      = 25
ONSET_STEP  = 50
M4_THRESHOLD = 0.110058
BATCH_SIZE   = 512

CHANNELS = ['Mot.SV', 'Pmp.SV', 'Mot.TV', 'Pmp.PV',
            'Temp.SV', 'Pres.SV', 'Pmp.TV', 'Mot.PV']
CH = {c: i for i, c in enumerate(CHANNELS)}

COMPOUND_LABELS      = {7, 8, 9, 10, 11, 12}
COMPOUND_PRIMARY_MAP = {
    7: 1, 8: 3, 9: 2, 10: 4, 11: 5, 12: 2,
}
GROUP_E_LABELS = {22, 23}

FAULT_GROUP_MAP = {
    0: 0,
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 19: 1,
    7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2,
    13: 3, 14: 3, 15: 3, 16: 3, 17: 3,
    18: 4, 20: 4, 21: 4,
    22: 5, 23: 5,
}

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL label names — used when fault_rules_v3.json parse fails / partial
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_LABEL_NAMES = {
    0:  "normal",
    1:  "bearing_wear",
    2:  "impeller_imbalance",
    3:  "cavitation",
    4:  "seal_failure",
    5:  "overloading",
    6:  "sensor_failure",
    7:  "bearing_wear+overloading",
    8:  "cavitation+seal_failure",
    9:  "impeller_imbalance+bearing_wear",
    10: "seal_failure+cavitation_H",
    11: "overloading+bearing_wear",
    12: "impeller_imbalance+cavitation",
    13: "bearing_wear_MotSV_masked",
    14: "cavitation_PresSV_masked",
    15: "seal_failure_PresSV_drifting",
    16: "overloading_TempSV_stuck",
    17: "imbalance_PmpSV_flatline",
    18: "cavitation_intermittent",
    19: "seal_failure_fast",
    20: "overloading_cyclic",
    21: "bearing_wear_gradual",
    22: "sensor_failure_2ch_thermal",
    23: "sensor_failure_2ch_pump",
}

# Physics descriptions for visualization annotations
PHYSICS_DESC = {
    0:  ("Normal Operation",    "All channels stable",              "Mot.SV, Pmp.SV, Temp.SV baseline"),
    1:  ("Bearing Wear",        "Paris law fatigue crack growth",    "Mot.SV* rises monotonically"),
    2:  ("Impeller Imbalance",  "ISO 1940 unbalance force = me·ω²", "Pmp.SV* AM envelope oscillation"),
    3:  ("Cavitation",          "NPSHa < NPSHr → bubble collapse",  "Pres.SV* drops, Pmp.SV* kurtosis↑"),
    4:  ("Seal Failure",        "Orifice leak: Q=Cd·A·√(2ΔP/ρ)",   "Pres.SV* smooth negative drift"),
    5:  ("Overloading",         "Cp·m·dT/dt > rated heat dissip.",  "Temp.SV* rising slope (dT*/dt>0)"),
    6:  ("Sensor Failure",      "I/O dropout or cable cut",         "One channel → 0 or constant"),
    7:  ("Bearing+Overloading", "Bearing heat → thermal runaway",   "Mot.SV* + Temp.SV* both rise"),
    8:  ("Cavitation+Seal",     "Low pressure → seal face erosion", "Pres.SV* + Pmp.SV* kurtosis↑"),
    9:  ("Imbalance+Bearing",   "Cyclic force → bearing fatigue",   "Pmp.SV* AM then Mot.SV* rise"),
    10: ("Seal+Cavitation_H",   "Seal leak → NPSHa drops further",  "Pres.SV* step then oscillation"),
    11: ("Overloading+Bearing", "Heat → lube breakdown → bearing",  "Temp.SV* rise then Mot.SV* rise"),
    12: ("Imbalance+Cavitation","Radial force → recirculation",     "Pmp.SV* AM then Pres.SV* drop"),
    13: ("Bearing [MotSV mask]","Mot.SV flatline hides bearing",    "Mot.SV*≈0, Temp.SV* still rises"),
    14: ("Cavitation [Pres mask]","Pres.SV stuck hides cav",        "Pres.SV* stuck, Pmp.SV* kurtosis"),
    15: ("Seal [Pres drift+]",  "Sensor bias hides seal leak",      "Pres.SV* drifts UP (wrong sign)"),
    16: ("Overload [Temp stuck]","Temp sensor stuck hides OL",      "Temp.SV*=const, Mot.PV* rises"),
    17: ("Imbalance [PmpSV flat]","Pmp.SV flatline hides imbal.",   "Pmp.SV*≈0, Mot.SV* AM remains"),
    18: ("Cavitation Intermit.", "NPSHa oscillation (3–7 bursts)",  "Pres.SV* periodic spikes"),
    19: ("Seal Failure Fast",   "Turbulent orifice Q=Cd·A·√(2ΔP/ρ)","Pres.SV* collapses ≤20 steps"),
    20: ("Overloading Cyclic",  "Load cycles with rising baseline", "Temp.SV* sawtooth + drift↑"),
    21: ("Bearing Gradual",     "Paris law low-dK/dN (weeks)",      "Mot.SV* slow slope, MAE<threshold"),
    22: ("2ch Thermal Fail",    "Shared thermal excitation rail",   "Mot.TV* + Temp.SV* both fail"),
    23: ("2ch Pump Fail",       "Moisture ingress pump junction",   "Pmp.SV* + Pmp.PV* both fail"),
}

# Primary sensor index to highlight per fault class
PRIMARY_CHANNEL = {
    0:  CH['Mot.SV'],
    1:  CH['Mot.SV'],
    2:  CH['Pmp.SV'],
    3:  CH['Pres.SV'],
    4:  CH['Pres.SV'],
    5:  CH['Temp.SV'],
    6:  CH['Mot.SV'],
    7:  CH['Mot.SV'],
    8:  CH['Pres.SV'],
    9:  CH['Pmp.SV'],
    10: CH['Pres.SV'],
    11: CH['Temp.SV'],
    12: CH['Pmp.SV'],
    13: CH['Mot.SV'],
    14: CH['Pres.SV'],
    15: CH['Pres.SV'],
    16: CH['Temp.SV'],
    17: CH['Pmp.SV'],
    18: CH['Pres.SV'],
    19: CH['Pres.SV'],
    20: CH['Temp.SV'],
    21: CH['Mot.SV'],
    22: CH['Mot.TV'],
    23: CH['Pmp.SV'],
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PRE-FLIGHT
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 1 — Pre-flight checks...")

try:
    with open(MODEL_DIR / "M4_threshold_config.json") as f:
        thr_cfg = json.load(f)
    M4_THRESHOLD = float(thr_cfg.get("threshold", 0.110058))
    assert abs(M4_THRESHOLD - 0.110058) < 1e-5
    log(f"  M4 threshold: {M4_THRESHOLD} ✓")
except Exception as e:
    log(f"  [WARNING] threshold fallback 0.110058: {e}")
    M4_THRESHOLD = 0.110058
results['M4_threshold'] = M4_THRESHOLD

# ── FIX: fault_rules_v3.json label map parsing ────────────────────────────────
try:
    with open(MODEL_DIR / "fault_rules_v3.json") as f:
        fault_rules_v3 = json.load(f)
    log(f"  fault_rules_v3.json loaded — {len(fault_rules_v3)} top-level keys")
    # The JSON has "label_map": {"0": "normal", "1": "bearing_wear", ...}
    label_str_map = {}
    if "label_map" in fault_rules_v3:
        for k, v in fault_rules_v3["label_map"].items():
            label_str_map[int(k)] = v
        log(f"  Label map (from label_map key): {len(label_str_map)} entries ✓")
    else:
        # Fallback: iterate top-level looking for {label_int: N}
        for k, v in fault_rules_v3.items():
            if isinstance(v, dict) and 'label_int' in v:
                label_str_map[int(v['label_int'])] = k
        if not label_str_map:
            log("  [WARNING] label_map key absent — using CANONICAL_LABEL_NAMES")
            label_str_map = CANONICAL_LABEL_NAMES.copy()
        else:
            log(f"  Label map (fallback scan): {len(label_str_map)} entries")
except Exception as e:
    log(f"  [WARNING] fault_rules_v3.json parse failed: {e} — using canonical names")
    label_str_map = CANONICAL_LABEL_NAMES.copy()

# Always backfill with canonical for any missing entries (labels 0-23)
for k, v in CANONICAL_LABEL_NAMES.items():
    if k not in label_str_map:
        label_str_map[k] = v
log(f"  Final label_str_map: {len(label_str_map)} entries: {sorted(label_str_map.keys())}")
results['fault_rules_v3_labels'] = len(label_str_map)

try:
    with open(MODEL_DIR / "M3_normalization_config.json") as f:
        norm_config = json.load(f)
    log("  M3_normalization_config.json loaded ✓")
except Exception as e:
    log(f"  [FATAL] M3 norm config missing: {e}"); raise

required_files = {
    "M6B_combined_sequences.pkl":    SYNTH_DIR / "M6B_combined_sequences.pkl",
    "M6B_sequence_meta.csv":         SYNTH_DIR / "M6B_sequence_meta.csv",
    "lstm_ae_baseline_best.pth":     MODEL_DIR / "lstm_ae_baseline_best.pth",
    "z_t_groupA_normal":             SYNTH_DIR / "z_t_sequences_groupA_normal.pkl",
    "z_t_groupA_faults":             SYNTH_DIR / "z_t_sequences_groupA_faults.pkl",
    "z_t_groupA_faults_rerun":       SYNTH_DIR / "z_t_sequences_groupA_faults_rerun.pkl",
    "z_t_groupB":                    SYNTH_DIR / "z_t_sequences_groupB.pkl",
    "z_t_groupC":                    SYNTH_DIR / "z_t_sequences_groupC.pkl",
    "z_t_groupD":                    SYNTH_DIR / "z_t_sequences_groupD.pkl",
    "z_t_groupE":                    SYNTH_DIR / "z_t_sequences_groupE.pkl",
}
missing = [n for n, p in required_files.items() if not p.exists()]
if missing:
    raise FileNotFoundError(f"BLOCK: Missing: {missing}")
for name, path in required_files.items():
    log(f"  ✓ {name} ({path.stat().st_size / 1e6:.1f} MB)")
log("  Pre-flight: all files confirmed ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FROZEN M4 LSTM-AE
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 2 — Loading frozen M4 LSTM-AE (inference only)...")

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
        self.fc_h = nn.Linear(64, 128)
        self.fc_c = nn.Linear(64, 128)
        self.lstm1 = nn.LSTM(64,  128, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128,   8, num_layers=1, batch_first=True)
        self.out   = nn.Linear(8, 8)
    def forward(self, z, h_n, c_n):
        h0    = torch.tanh(self.fc_h(h_n[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0    = torch.tanh(self.fc_c(c_n[-1])).unsqueeze(0).repeat(2, 1, 1)
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

try:
    m4_model = LSTMAutoencoder(seq_len=WINDOW_SIZE)
    state    = torch.load(MODEL_DIR / "lstm_ae_baseline_best.pth", map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    m4_model.load_state_dict(state, strict=True)
    m4_model.eval()
    m4_model.to(DEVICE)
    for p in m4_model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in m4_model.parameters())
    log(f"  M4 loaded: {n_params:,} params | device: {DEVICE} | FROZEN ✓")
except Exception as e:
    log(f"  [FATAL] M4 load failed: {e}"); raise
results['m4_params'] = n_params

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD z_t PKL FILES
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 3 — Loading z_t pkl files...")

zt_files = {
    'groupA_normal':       SYNTH_DIR / "z_t_sequences_groupA_normal.pkl",
    'groupA_faults':       SYNTH_DIR / "z_t_sequences_groupA_faults.pkl",
    'groupA_faults_rerun': SYNTH_DIR / "z_t_sequences_groupA_faults_rerun.pkl",
    'groupB':              SYNTH_DIR / "z_t_sequences_groupB.pkl",
    'groupC':              SYNTH_DIR / "z_t_sequences_groupC.pkl",
    'groupD':              SYNTH_DIR / "z_t_sequences_groupD.pkl",
    'groupE':              SYNTH_DIR / "z_t_sequences_groupE.pkl",
}
zt_data = {}
for key, path in zt_files.items():
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list):
            zt_data[key] = data
        elif isinstance(data, dict) and 'sequences' in data:
            zt_data[key] = data['sequences']
        else:
            zt_data[key] = list(data.values()) if isinstance(data, dict) else data
        sample   = zt_data[key][0]
        zt_shape = sample['z_t'].shape if isinstance(sample, dict) else sample.shape
        log(f"  {key}: {len(zt_data[key])} seqs | z_t shape: {zt_shape}")
    except Exception as e:
        log(f"  [FATAL] Cannot load {key}: {e}"); raise

def get_zt(entry):
    return entry['z_t'] if isinstance(entry, dict) else entry

def get_mae_from_zt(entry):
    return entry.get('mae') if isinstance(entry, dict) else None

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PCA FIT ON NORMAL z_t
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 4 — Fitting PCA on Group A normal z_t...")

normal_zt_pool = np.vstack([get_zt(e) for e in zt_data['groupA_normal']
                             if get_zt(e).ndim == 2 and get_zt(e).shape[1] == 64])
log(f"  Normal z_t pool: {normal_zt_pool.shape}")
pca = PCA(n_components=2, random_state=42)
pca.fit(normal_zt_pool)
var_explained = float(pca.explained_variance_ratio_.sum())
log(f"  PCA variance explained: {var_explained:.4f} ({var_explained*100:.1f}%)")
results['M6p5r_z_t_pca_variance_explained'] = var_explained
gate_z1 = "PASS" if var_explained >= 0.50 else "WARN"
log(f"  Gate Z1: {gate_z1}")
gates['Z1_pca_variance'] = gate_z1
del normal_zt_pool; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LOAD SEQUENCES + META
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 5 — Loading M6B sequences and metadata...")

try:
    with open(SYNTH_DIR / "M6B_combined_sequences.pkl", 'rb') as f:
        combined = pickle.load(f)
    if isinstance(combined, dict) and 'sequences' in combined:
        all_sequences = combined['sequences']
        all_metadata  = combined['metadata']
    elif isinstance(combined, list):
        all_sequences = combined
        all_metadata  = [{}] * len(combined)
    else:
        raise ValueError(f"Unknown format: {type(combined)}")
    log(f"  Sequences: {len(all_sequences)} | Metadata: {len(all_metadata)}")
    results['M6p5r_n_sequences_in'] = len(all_sequences)
except Exception as e:
    log(f"  [FATAL] {e}"); raise

try:
    meta_df = pd.read_csv(SYNTH_DIR / "M6B_sequence_meta.csv")
    log(f"  meta_df: {meta_df.shape} | cols: {list(meta_df.columns)}")
except Exception as e:
    log(f"  [WARNING] meta_df fallback from metadata list: {e}")
    meta_df = pd.DataFrame(all_metadata)

# ── FIX: safe NaN handling for secondary_onset_step ───────────────────────────
def safe_int(val, default=0):
    """Convert val to int, returning default if NaN, None, or not convertible."""
    try:
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return int(f)
    except (ValueError, TypeError):
        return default

# Build per-sequence lookup
meta_lookup = {}
for i, row in meta_df.iterrows():
    idx = row.get('seq_id', i)
    try:
        idx_key = int(idx)
    except (ValueError, TypeError):
        idx_key = i
    meta_lookup[idx_key] = row.to_dict()

log(f"  Meta lookup: {len(meta_lookup)} entries ✓")

# ── Build z_t per-label map ───────────────────────────────────────────────────
log("  Building z_t alignment index...")

groupA_labels_carried = {0, 2, 3, 6}
groupA_labels_rerun   = {1, 4, 5}

def build_zt_label_map(zt_list, meta_subset):
    mapping = {}
    for zt_entry, (_, row) in zip(zt_list, meta_subset.iterrows()):
        lbl = safe_int(row.get('label', row.get('label_int', 0)))
        mapping.setdefault(lbl, []).append(zt_entry)
    return mapping

meta_gA_carried = meta_df[meta_df['label'].isin(groupA_labels_carried)].reset_index(drop=True)
meta_gA_rerun   = meta_df[meta_df['label'].isin(groupA_labels_rerun)].reset_index(drop=True)
meta_gB         = meta_df[meta_df['label'].isin(range(7,  13))].reset_index(drop=True)
meta_gC         = meta_df[meta_df['label'].isin(range(13, 18))].reset_index(drop=True)
meta_gD         = meta_df[meta_df['label'].isin(range(18, 22))].reset_index(drop=True)
meta_gE         = meta_df[meta_df['label'].isin({22, 23})].reset_index(drop=True)

zt_gA_all = zt_data['groupA_normal'] + zt_data['groupA_faults']

def safe_map(zt_list, meta_sub, name):
    if len(zt_list) == len(meta_sub):
        return build_zt_label_map(zt_list, meta_sub)
    log(f"  [WARNING] {name} zt/meta size mismatch: zt={len(zt_list)}, meta={len(meta_sub)}")
    m = {}
    for i, entry in enumerate(zt_list):
        if i < len(meta_sub):
            lbl = safe_int(meta_sub.iloc[i].get('label', 0))
            m.setdefault(lbl, []).append(entry)
    return m

zt_label_map_carried = safe_map(zt_gA_all, meta_gA_carried, "GroupA_carried")
zt_label_map_rerun   = safe_map(zt_data['groupA_faults_rerun'], meta_gA_rerun, "GroupA_rerun")
zt_label_map_B       = safe_map(zt_data['groupB'], meta_gB, "GroupB")
zt_label_map_C       = safe_map(zt_data['groupC'], meta_gC, "GroupC")
zt_label_map_D       = safe_map(zt_data['groupD'], meta_gD, "GroupD")
zt_label_map_E       = safe_map(zt_data['groupE'], meta_gE, "GroupE")

zt_per_label = {}
for d in [zt_label_map_carried, zt_label_map_rerun,
          zt_label_map_B, zt_label_map_C, zt_label_map_D, zt_label_map_E]:
    for lbl, entries in d.items():
        zt_per_label.setdefault(lbl, []).extend(entries)

for lbl, entries in sorted(zt_per_label.items()):
    log(f"    label {lbl:2d} ({CANONICAL_LABEL_NAMES.get(lbl,'?'):35s}): {len(entries)} seqs")

zt_label_counters = {lbl: 0 for lbl in zt_per_label}
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — WINDOWING + BATCHED M4 INFERENCE + FEATURE EXTRACTION
# Strategy: collect ALL windows first → single GPU inference pass → split back
# This saturates RTX 4060 (batch=4096 → ~6GB VRAM) vs 32,500 tiny calls
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 6 — Windowing all sequences + BATCHED M4 inference...")

INFERENCE_BATCH = 4096   # saturates RTX 4060 8GB at float16; reduce to 2048 on OOM

class WindowDataset(Dataset):
    def __init__(self, w): self.w = torch.from_numpy(w).float()
    def __len__(self): return len(self.w)
    def __getitem__(self, i): return self.w[i]

# ── PASS 1: collect all windows + per-window metadata ─────────────────────────
log("  Pass 1: slicing windows from all sequences...")

all_windows     = []   # will become (N_total, 50, 8) array
win_meta        = []   # per-window dict: seq_idx, win_start, label_int, seq_len, etc.
viz_sequences   = {}   # label_int → first representative seq (T, 8)
boundary_violations = 0

for seq_idx, seq in enumerate(all_sequences):
    seq = np.array(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 8:
        continue
    seq_len = seq.shape[0]

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {}
    if seq_idx < len(all_metadata):
        m = all_metadata[seq_idx]
        meta = m if isinstance(m, dict) else {}
    if seq_idx in meta_lookup:
        meta = meta_lookup[seq_idx]

    label_int = safe_int(meta.get('label', meta.get('label_int', 0)))

    raw_sos = meta.get('secondary_onset_step', None)
    raw_lag = meta.get('secondary_onset_lag', meta.get('lag_steps', None))
    if label_int in COMPOUND_LABELS:
        if raw_sos is not None and not (isinstance(raw_sos, float) and np.isnan(raw_sos)):
            secondary_onset_step = safe_int(raw_sos, default=ONSET_STEP + 50)
        elif raw_lag is not None and not (isinstance(raw_lag, float) and np.isnan(raw_lag)):
            secondary_onset_step = ONSET_STEP + safe_int(raw_lag, default=50)
        else:
            secondary_onset_step = ONSET_STEP + 50
    else:
        secondary_onset_step = 0

    masked_channel_flag = safe_int(meta.get('masked_channel_flag',
                                   1 if label_int in range(13, 18) else 0))
    severity    = float(meta.get('severity', 0.0) or 0.0)
    group_id    = safe_int(meta.get('group_id', FAULT_GROUP_MAP.get(label_int, 0)))
    sec_lag     = safe_int(meta.get('secondary_onset_lag', meta.get('lag_steps', 0)))
    multi_count = safe_int(meta.get('multi_sensor_anomaly_count', 0))

    if label_int not in viz_sequences and seq_len >= 50:
        viz_sequences[label_int] = seq.copy()

    for w_start in range(0, seq_len - WINDOW_SIZE + 1, STRIDE):
        w_end = w_start + WINDOW_SIZE
        if w_end > seq_len:
            boundary_violations += 1
            continue
        all_windows.append(seq[w_start:w_end, :])
        win_meta.append({
            'seq_idx':              seq_idx,
            'win_start':            w_start,
            'seq_len':              seq_len,
            'label_int':            label_int,
            'secondary_onset_step': secondary_onset_step,
            'masked_channel_flag':  masked_channel_flag,
            'severity':             severity,
            'group_id':             group_id,
            'sec_lag':              sec_lag,
            'multi_count':          multi_count,
        })

    if (seq_idx + 1) % 5000 == 0:
        log(f"  Sliced {seq_idx+1}/{len(all_sequences)} seqs | "
            f"windows so far: {len(all_windows):,}")

log(f"  Pass 1 done: {len(all_windows):,} windows | "
    f"boundary violations: {boundary_violations}")
results['M6p5r_n_windows_out']       = len(all_windows)
results['M6p5r_boundary_violations'] = boundary_violations

# ── PASS 2: single large GPU inference pass ────────────────────────────────────
log(f"  Pass 2: GPU inference (batch={INFERENCE_BATCH}, "
    f"total windows={len(all_windows):,})...")

all_windows_np = np.stack(all_windows, axis=0).astype(np.float32)  # (N, 50, 8)
del all_windows; gc.collect()
log(f"  Windows array: {all_windows_np.shape} | "
    f"{all_windows_np.nbytes / 1e9:.2f} GB RAM")

dataset = WindowDataset(all_windows_np)
loader  = DataLoader(dataset, batch_size=INFERENCE_BATCH,
                     pin_memory=False, num_workers=0, drop_last=False)

all_mae   = []    # (N, 8)
n_batches = len(loader)
m4_model.eval()
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=IS_GPU):
        for b_i, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            recon = m4_model(batch)
            mae   = torch.mean(torch.abs(batch - recon), dim=1)   # (B, 8)
            all_mae.append(mae.cpu().float().numpy())
            if (b_i + 1) % 20 == 0 or (b_i + 1) == n_batches:
                mem = (f" | VRAM: {torch.cuda.memory_allocated()/1e9:.2f}/"
                       f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
                       if IS_GPU else "")
                log(f"  Batch {b_i+1}/{n_batches}{mem}")

all_mae_np = np.vstack(all_mae).astype(np.float32)   # (N_total, 8)
del all_mae, dataset; gc.collect()
log(f"  GPU inference done: MAE array {all_mae_np.shape}")

# ── PASS 3: feature extraction using MAE + z_t lookup ─────────────────────────
log("  Pass 3: feature extraction per window...")

# We need per-sequence z_t data. Build seq_idx → z_t entry map using
# label-ordered counters (same as before — one zt entry per sequence in order)
zt_label_counters_p3 = {lbl: 0 for lbl in zt_per_label}

# Pre-fetch z_t for every sequence in order of first appearance
# (sequences are processed in seq_idx order in all_sequences)
seq_zt_cache = {}    # seq_idx → (zt_seq, zt_pca_proj, zt_norms, zt_recon_err,
                     #             score_A, score_B, score_C,
                     #             mean_zt_mag, std_zt_mag, zt_drift_slope)

log("  Building z_t cache per sequence...")
pca_origin = pca.transform(np.zeros((1, 64), dtype=np.float32))   # (1, 2)

for seq_idx in range(len(all_sequences)):
    meta = meta_lookup.get(seq_idx, {})
    label_int = safe_int(meta.get('label', meta.get('label_int', 0)))

    zt_entry = None
    if (label_int in zt_per_label and
            zt_label_counters_p3.get(label_int, 0) < len(zt_per_label[label_int])):
        zt_entry = zt_per_label[label_int][zt_label_counters_p3[label_int]]
        zt_label_counters_p3[label_int] += 1

    if zt_entry is None:
        seq_zt_cache[seq_idx] = None
        continue

    zt_seq = get_zt(zt_entry)
    if zt_seq is None or zt_seq.ndim != 2 or zt_seq.shape[1] != 64:
        seq_zt_cache[seq_idx] = None
        continue

    N_w_pkl   = zt_seq.shape[0]
    zt_norms  = np.linalg.norm(zt_seq, axis=1)                 # (N_w_pkl,)
    zt_pca_p  = pca.transform(zt_seq)                          # (N_w_pkl, 2)
    zt_rerr   = np.linalg.norm(zt_pca_p - pca_origin, axis=1) # (N_w_pkl,)

    score_A = float(np.mean(zt_rerr))
    if N_w_pkl >= 2:
        score_B = float(linregress(np.arange(N_w_pkl), zt_rerr).slope)
        score_C = float(np.max(np.abs(np.diff(zt_rerr))))
        zt_ds   = float(linregress(np.arange(N_w_pkl), zt_norms).slope)
    else:
        score_B = 0.0; score_C = 0.0; zt_ds = 0.0

    seq_zt_cache[seq_idx] = (zt_seq, zt_pca_p, zt_norms, zt_rerr,
                              score_A, score_B, score_C,
                              float(np.mean(zt_norms)), float(np.std(zt_norms)), zt_ds)

    if (seq_idx + 1) % 5000 == 0:
        log(f"  z_t cache: {seq_idx+1}/{len(all_sequences)}")

log(f"  z_t cache built: {sum(v is not None for v in seq_zt_cache.values())} "
    f"/ {len(seq_zt_cache)} sequences have z_t")

# Now iterate windows using pre-computed MAE + z_t cache
all_rows = []
t_axis   = np.arange(WINDOW_SIZE)

# We also need cyclic_baseline_drift and burst_count per sequence
# Build seq-level aggregates from the MAE array
log("  Computing sequence-level aggregates (burst, drift)...")
seq_start_map = {}   # seq_idx → first window index in all_mae_np
_cur_seq = -1
for w_i, wm in enumerate(win_meta):
    si = wm['seq_idx']
    if si != _cur_seq:
        seq_start_map[si] = w_i
        _cur_seq = si

log("  Building per-window feature rows...")
# Rebuild raw sequences dict for cyclic drift (only need Temp.SV channel)
# We stored viz_sequences already; for drift we need per-seq access
# Use all_sequences directly (still in memory)
seq_cyclic_drift = {}
for seq_idx, seq in enumerate(all_sequences):
    arr = np.array(seq, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] >= 50:
        seq_cyclic_drift[seq_idx] = float(
            arr[-25:, CH['Temp.SV']].mean() - arr[:25, CH['Temp.SV']].mean())
    else:
        seq_cyclic_drift[seq_idx] = 0.0

del all_sequences; gc.collect()   # free ~500 MB

for w_i, (wm, mae_w) in enumerate(zip(win_meta, all_mae_np)):
    seq_idx    = wm['seq_idx']
    w_start    = wm['win_start']
    seq_len    = wm['seq_len']
    label_int  = wm['label_int']
    sec_ons    = wm['secondary_onset_step']
    mask_flag  = wm['masked_channel_flag']
    severity   = wm['severity']
    sec_lag    = wm['sec_lag']

    # ── Onset-split label ────────────────────────────────────────────────
    w_mid = w_start + WINDOW_SIZE // 2
    if label_int in COMPOUND_LABELS and sec_ons > ONSET_STEP:
        window_label = (COMPOUND_PRIMARY_MAP.get(label_int, label_int)
                        if w_mid < sec_ons else label_int)
        onset_order  = 0 if w_mid < sec_ons else 1
    else:
        window_label = label_int
        onset_order  = 0

    # ── We need the raw 50-step window for Domain 2 stats ─────────────
    # Re-slice from all_windows_np (still in memory)
    win_np = all_windows_np[w_i]   # (50, 8) — already sliced in Pass 1

    # Domain 1
    mae_MotSV  = float(mae_w[CH['Mot.SV']])
    mae_PmpSV  = float(mae_w[CH['Pmp.SV']])
    mae_MotTV  = float(mae_w[CH['Mot.TV']])
    mae_PmpPV  = float(mae_w[CH['Pmp.PV']])
    mae_TempSV = float(mae_w[CH['Temp.SV']])
    mae_PresSV = float(mae_w[CH['Pres.SV']])
    mae_PmpTV  = float(mae_w[CH['Pmp.TV']])
    mae_MotPV  = float(mae_w[CH['Mot.PV']])

    # Domain 2
    motsv_w  = win_np[:, CH['Mot.SV']]
    pmpsv_w  = win_np[:, CH['Pmp.SV']]
    tempsv_w = win_np[:, CH['Temp.SV']]
    pressv_w = win_np[:, CH['Pres.SV']]
    mottv_w  = win_np[:, CH['Mot.TV']]

    mean_err_MotSV   = float(np.mean(motsv_w))
    std_err_MotSV    = float(np.std(motsv_w))
    kurtosis_PmpSV   = float(kurtosis(pmpsv_w, fisher=True))
    err_slope_MotSV  = float(linregress(t_axis, motsv_w).slope)
    err_slope_TempSV = float(linregress(t_axis, tempsv_w).slope)
    err_slope_PresSV = float(linregress(t_axis, pressv_w).slope)

    thermal_coupling_ratio = (float(pearsonr(mottv_w, tempsv_w)[0])
                              if np.std(mottv_w) > 1e-9 and np.std(tempsv_w) > 1e-9
                              else 1.0)
    cross_channel_MotSV_PmpSV = (float(pearsonr(motsv_w, pmpsv_w)[0])
                                  if np.std(motsv_w) > 1e-9 and np.std(pmpsv_w) > 1e-9
                                  else 0.0)
    max_err_all = float(mae_w.max())

    # Domain 3
    win_has_burst    = int(mae_w.max() > M4_THRESHOLD and label_int in {18, 20})
    cyc_drift        = seq_cyclic_drift.get(seq_idx, 0.0)
    multi_sensor_cnt = int(np.sum(mae_w > 0.15))
    fault_group_id   = int(FAULT_GROUP_MAP.get(label_int, 0))
    abs_pmp          = abs(float(linregress(t_axis, pmpsv_w).slope))
    variant_slope_ratio     = abs_pmp / max(abs(err_slope_PresSV), 1e-9)
    thermal_decoupling_flag = int(thermal_coupling_ratio < 0.5)

    # Domain 4
    zt_cache = seq_zt_cache.get(seq_idx)
    if zt_cache is not None:
        _, zt_pca_p, zt_norms, zt_rerr, score_A, score_B, score_C, \
            mean_zt_mag, std_zt_mag, zt_ds = zt_cache
        N_w_pkl = len(zt_norms)
        pkl_w   = min(w_start // WINDOW_SIZE, N_w_pkl - 1)
        z_t_pca_1       = float(zt_pca_p[pkl_w, 0])
        z_t_pca_2       = float(zt_pca_p[pkl_w, 1])
        z_t_norm_val    = float(zt_norms[pkl_w])
        z_t_recon_err_v = float(zt_rerr[pkl_w])
    else:
        z_t_pca_1 = z_t_pca_2 = z_t_norm_val = z_t_recon_err_v = 0.0
        score_A = score_B = score_C = 0.0
        mean_zt_mag = std_zt_mag = zt_ds = 0.0

    all_rows.append({
        'label_int':                 window_label,
        'seq_label_int':             label_int,
        'mae_MotSV':                 mae_MotSV,
        'mae_PmpSV':                 mae_PmpSV,
        'mae_MotTV':                 mae_MotTV,
        'mae_PmpPV':                 mae_PmpPV,
        'mae_TempSV':                mae_TempSV,
        'mae_PresSV':                mae_PresSV,
        'mae_PmpTV':                 mae_PmpTV,
        'mae_MotPV':                 mae_MotPV,
        'mean_err_MotSV':            mean_err_MotSV,
        'std_err_MotSV':             std_err_MotSV,
        'kurtosis_PmpSV':            kurtosis_PmpSV,
        'err_slope_MotSV':           err_slope_MotSV,
        'err_slope_TempSV':          err_slope_TempSV,
        'err_slope_PresSV':          err_slope_PresSV,
        'thermal_coupling_ratio':    thermal_coupling_ratio,
        'cross_channel_MotSV_PmpSV': cross_channel_MotSV_PmpSV,
        'max_err_all':               max_err_all,
        'masked_channel_flag':       int(mask_flag),
        'secondary_onset_lag':       sec_lag,
        'burst_count':               win_has_burst,
        'cyclic_baseline_drift':     cyc_drift,
        'multi_sensor_anomaly_count': multi_sensor_cnt,
        'fault_group_id':            fault_group_id,
        'variant_slope_ratio':       variant_slope_ratio,
        'thermal_decoupling_flag':   thermal_decoupling_flag,
        'z_t_pca_1':                 z_t_pca_1,
        'z_t_pca_2':                 z_t_pca_2,
        'z_t_norm':                  z_t_norm_val,
        'z_t_recon_err':             z_t_recon_err_v,
        'score_A':                   score_A,
        'score_B':                   score_B,
        'score_C':                   score_C,
        'onset_order':               onset_order,
        'seq_idx':                   seq_idx,
        'win_start':                 w_start,
        'seq_len':                   seq_len,
        'severity':                  severity,
    })

    if (w_i + 1) % 50000 == 0:
        log(f"  Feature rows: {w_i+1:,}/{len(win_meta):,}")

del all_mae_np, win_meta; gc.collect()
log(f"\n  Total rows: {len(all_rows):,} | boundary violations: {boundary_violations}")
results['M6p5r_n_windows_out']       = len(all_rows)
results['M6p5r_boundary_violations'] = boundary_violations

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BUILD DATAFRAME + GATES W1/W2/W3
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 7 — Building DataFrame + Gates W1/W2/W3...")

df = pd.DataFrame(all_rows)
del all_rows; gc.collect()
log(f"  DataFrame: {df.shape}")

gates['W1_boundary'] = "PASS" if boundary_violations == 0 else "FAIL"
invalid_lbl  = df[~df['label_int'].isin(range(24))]['label_int'].unique()
gates['W2_onset_split'] = "PASS" if len(invalid_lbl) == 0 else "FAIL"

gb_mask = df['label_int'].isin(COMPOUND_LABELS)
if gb_mask.sum() > 0:
    lag_zeros = (df[gb_mask & (df['onset_order'] == 1)]['secondary_onset_lag'] == 0).sum()
    gates['W3_compound_lag'] = "PASS" if lag_zeros == 0 else "WARN"
else:
    gates['W3_compound_lag'] = "WARN"

label_counts = df['label_int'].value_counts().sort_index()
log(f"\n  Class distribution ({len(label_counts)} classes):")
for lbl, cnt in label_counts.items():
    pct  = cnt / len(df) * 100
    name = label_str_map.get(int(lbl), f"label_{lbl}")
    log(f"    {int(lbl):2d}  {name:40s}: {cnt:6d} ({pct:.1f}%)")
results['M6p5r_n_classes'] = len(label_counts)
results['M6p5r_feature_matrix_rows'] = len(df)

max_pct = label_counts.max() / len(df) * 100
gates['D1_class_balance'] = "PASS" if max_pct < 20.0 else "WARN"

for g, v in gates.items():
    log(f"  Gate {g}: {v}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DOMAIN-SPECIFIC GATES D2–D5, Z2, Z3
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 8 — Domain-specific gates...")

# Gate D2 — Group C masked_channel_flag
gc_mask = df['label_int'].isin(range(13, 18))
d2_pct  = df[gc_mask]['masked_channel_flag'].mean() * 100 if gc_mask.sum() > 0 else 0.0
gates['D2_masked_flag'] = "PASS" if d2_pct >= 90.0 else "WARN"
log(f"  Gate D2 (Group C masked): {gates['D2_masked_flag']} | {d2_pct:.1f}%")
results['M6p5r_gate_D2_pct'] = d2_pct

# Gate D3 — Group E 2-channel anomaly
ge_mask = df['label_int'].isin({22, 23})
d3_pct  = (df[ge_mask]['multi_sensor_anomaly_count'] == 2).mean()*100 if ge_mask.sum() > 0 else 0.0
gates['D3_multisensor'] = "PASS" if d3_pct >= 90.0 else "WARN"
log(f"  Gate D3 (Group E 2ch): {gates['D3_multisensor']} | {d3_pct:.1f}%")
results['M6p5r_gate_D3_pct'] = d3_pct

# Gate D4 — Label 18 burst
l18_mask = df['label_int'] == 18
d4_pct   = df[l18_mask]['burst_count'].mean()*100 if l18_mask.sum() > 0 else 0.0
gates['D4_burst_count'] = "PASS" if d4_pct >= 95.0 else "WARN"
log(f"  Gate D4 (label 18 burst): {gates['D4_burst_count']} | {d4_pct:.1f}%")

# Gate D5 — Label 21 slope
l21_mask       = df['label_int'] == 21
l21_fault_mask = l21_mask & (df['win_start'] >= ONSET_STEP)
if l21_fault_mask.sum() > 0:
    d5_pct = (df[l21_fault_mask]['err_slope_MotSV'] > 0).mean()*100
    gates['D5_label21_slope'] = "PASS" if d5_pct >= 95.0 else "WARN"
    results['M6p5r_label21_slope_pct_positive'] = d5_pct
else:
    gates['D5_label21_slope'] = "WARN"; d5_pct = 0.0
log(f"  Gate D5 (label 21 slope): {gates['D5_label21_slope']} | {d5_pct:.1f}%")

# Gate Z2 — score_C Group B vs Group A
ga_sc = df[df['label_int'].isin(range(7))]['score_C']
gb_sc = df[df['label_int'].isin(COMPOUND_LABELS)]['score_C']
if len(ga_sc) > 0 and len(gb_sc) > 0:
    p50A  = float(ga_sc.quantile(0.50))
    z2_pct = (gb_sc > p50A).mean()*100
    gates['Z2_score_C_group_B'] = "PASS" if z2_pct >= 80.0 else "WARN"
    results['M6p5r_score_C_group_B_pct']  = z2_pct
    results['M6p5r_score_C_groupA_p50']   = p50A
else:
    gates['Z2_score_C_group_B'] = "WARN"; z2_pct = 0.0
log(f"  Gate Z2 (score_C GroupB): {gates['Z2_score_C_group_B']} | {z2_pct:.1f}%")

# Gate Z3 — score_B label 21
if l21_mask.sum() > 0:
    z3_pct = (df[l21_mask]['score_B'] > 0).mean()*100
    gates['Z3_score_B_label21'] = "PASS" if z3_pct >= 90.0 else "WARN"
    results['M6p5r_score_B_label21_pct_positive'] = z3_pct
else:
    gates['Z3_score_B_label21'] = "WARN"; z3_pct = 0.0
log(f"  Gate Z3 (score_B label21): {gates['Z3_score_B_label21']} | {z3_pct:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FISHER SCORE + GATE F1
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 9 — Fisher scores...")

FEATURE_COLS = [c for c in [
    'mae_MotSV','mae_PmpSV','mae_MotTV','mae_PmpPV',
    'mae_TempSV','mae_PresSV','mae_PmpTV','mae_MotPV',
    'mean_err_MotSV','std_err_MotSV','kurtosis_PmpSV',
    'err_slope_MotSV','err_slope_TempSV','err_slope_PresSV',
    'thermal_coupling_ratio','cross_channel_MotSV_PmpSV','max_err_all',
    'masked_channel_flag','secondary_onset_lag','burst_count',
    'cyclic_baseline_drift','multi_sensor_anomaly_count','fault_group_id',
    'variant_slope_ratio','thermal_decoupling_flag',
    'z_t_pca_1','z_t_pca_2','z_t_norm','z_t_recon_err',
    'score_A','score_B','score_C','onset_order',
] if c in df.columns]

results['M6p5r_feature_matrix_cols'] = len(FEATURE_COLS) + 1

def fisher_score(X, y):
    classes = np.unique(y)
    mu_g = X.mean(axis=0)
    num  = np.zeros(X.shape[1])
    den  = np.zeros(X.shape[1])
    for c in classes:
        mask = (y == c)
        n_c  = mask.sum()
        mu_c = X[mask].mean(axis=0)
        s_c  = X[mask].std(axis=0)
        num += n_c * (mu_c - mu_g) ** 2
        den += n_c * s_c ** 2
    den[den < 1e-9] = 1e-9
    return num / den

try:
    X_f = df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    y_f = df['label_int'].values
    f_s = fisher_score(X_f, y_f)
    del X_f; gc.collect()
    fisher_df = pd.DataFrame({'feature': FEATURE_COLS, 'fisher_score': f_s})
    fisher_df = fisher_df.sort_values('fisher_score', ascending=False)
    top_feat  = fisher_df.iloc[0]['feature']
    flagged   = fisher_df[fisher_df['fisher_score'] < 0.5]['feature'].tolist()
    gates['F1_fisher'] = "PASS" if not flagged else f"WARN — {len(flagged)} features < 0.5"
    log(f"\n  Top-5 Fisher features:")
    for _, r in fisher_df.head(5).iterrows():
        log(f"    {r['feature']:35s}: {r['fisher_score']:.4f}")
    results['M6p5r_top_fisher_feature'] = top_feat
    results['M6p5r_fisher_scores']      = {r['feature']: float(r['fisher_score'])
                                            for _, r in fisher_df.iterrows()}
    results['M6p5r_gate_F1_flagged']    = flagged
except Exception as e:
    log(f"  [WARNING] Fisher failed: {e}")
    fisher_df = None
    gates['F1_fisher'] = "WARN — computation failed"

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SAVE CSV + METADATA JSON
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 10 — Saving feature matrix...")

final_cols  = ['label_int'] + FEATURE_COLS
output_df   = df[final_cols].copy()
output_path = SYNTH_DIR / "M6B_feature_matrix.csv"

try:
    output_df.to_csv(output_path, index=False)
    fsz = output_path.stat().st_size / 1e6
    log(f"  Saved: {output_path} | {len(output_df):,} × {len(output_df.columns)} | {fsz:.1f} MB")
    results['M6p5r_output_file']         = str(output_path)
    results['M6p5r_feature_matrix_rows'] = len(output_df)
    results['M6p5r_feature_matrix_cols'] = len(output_df.columns)
except Exception as e:
    log(f"  [FATAL] {e}"); raise

meta_out = {
    "window_size": WINDOW_SIZE, "stride": STRIDE, "onset_step": ONSET_STEP,
    "n_sequences": results.get('M6p5r_n_sequences_in', 0),
    "n_windows":   len(output_df), "n_classes": results.get('M6p5r_n_classes', 0),
    "feature_cols": FEATURE_COLS,
    "class_distribution": {str(k): int(v)
                            for k, v in output_df['label_int'].value_counts().sort_index().items()},
    "fisher_scores":  results.get('M6p5r_fisher_scores', {}),
    "gate_F1_status": gates.get('F1_fisher', 'PENDING'),
    "m4_threshold_used": M4_THRESHOLD,
    "z_t_pca_variance_explained": results.get('M6p5r_z_t_pca_variance_explained', 0.0),
    "generated_by": f"{SCRIPT_NAME}.py v2.0", "arch_version": ARCH_VERSION,
    "date": str(date.today()),
}
meta_path = SYNTH_DIR / "M6B_feature_matrix_metadata.json"
with open(meta_path, 'w') as f:
    json.dump(meta_out, f, indent=2)
log(f"  Metadata JSON: {meta_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11A — PHYSICS FAULT-PROGRESSION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
# For each fault class: show (1) raw sensor time-series of the representative
# sequence, (2) sliding-window MAE progression with threshold line,
# (3) key physics annotation.
# Two separate figure sets:
#   A) Group plots (one figure per group, all labels in grid)
#   B) Individual label plots (one per label) — highest detail, best for QA
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 11A — Physics fault-progression visualization...")

# ── Helper: compute MAE trajectory for a single sequence ─────────────────────
def compute_mae_trajectory(seq_arr):
    """Return (win_starts, mae_per_window) for a (T,8) sequence.
    Self-contained inference — does NOT call run_m4_inference (removed in v3).
    """
    wins, starts = [], []
    for s in range(0, len(seq_arr) - WINDOW_SIZE + 1, STRIDE):
        wins.append(seq_arr[s:s + WINDOW_SIZE, :])
        starts.append(s)
    if not wins:
        return np.array([]), np.zeros((0, 8))
    w_np    = np.stack(wins, axis=0).astype(np.float32)   # (N_w, 50, 8)
    w_ten   = torch.from_numpy(w_np).to(DEVICE)
    maes    = []
    m4_model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=IS_GPU):
            for i in range(0, len(w_ten), 512):
                batch = w_ten[i:i + 512]
                recon = m4_model(batch)
                mae   = torch.mean(torch.abs(batch - recon), dim=1)  # (B, 8)
                maes.append(mae.cpu().float().numpy())
    return np.array(starts), np.vstack(maes)   # (N_w,), (N_w, 8)

# Colour map for channels
CH_COLORS = {
    'Mot.SV':  '#E74C3C',  # red
    'Pmp.SV':  '#E67E22',  # orange
    'Mot.TV':  '#9B59B6',  # purple
    'Pmp.PV':  '#2ECC71',  # green
    'Temp.SV': '#F39C12',  # amber
    'Pres.SV': '#3498DB',  # blue
    'Pmp.TV':  '#1ABC9C',  # teal
    'Mot.PV':  '#95A5A6',  # grey
}
CH_LIST   = list(CH_COLORS.keys())
COLOR_LIST = list(CH_COLORS.values())

# ── GROUP PLOTS (5 groups) ────────────────────────────────────────────────────
group_defs = {
    "A_single":   ([0, 1, 2, 3, 4, 5, 6, 19],   "Group A — Single Faults (+ seal_fast)"),
    "B_compound": ([7, 8, 9, 10, 11, 12],         "Group B — Compound Fault Chains"),
    "C_masked":   ([13, 14, 15, 16, 17],          "Group C — Masked Faults"),
    "D_variant":  ([18, 20, 21],                  "Group D — Cyclic / Gradual"),
    "E_multisens":([22, 23],                      "Group E — Multi-Sensor Failure"),
}

for grp_key, (grp_labels, grp_title) in group_defs.items():
    available = [l for l in grp_labels if l in viz_sequences]
    if not available:
        log(f"  [SKIP] {grp_key} — no viz sequences found")
        continue

    n_plots = len(available)
    ncols   = min(3, n_plots)
    nrows   = int(np.ceil(n_plots / ncols))

    # Each subplot = 3 rows: raw sensor signal, MAE trajectory, annotations
    fig = plt.figure(figsize=(7 * ncols, 9 * nrows), facecolor='#0F1117')
    fig.suptitle(f"PumpSmart v14.2 — {grp_title}\n"
                 f"110 kW · 7-stage · 40 bar · 2980 RPM · M4 threshold = {M4_THRESHOLD}",
                 color='white', fontsize=13, fontweight='bold', y=0.98)

    outer = gridspec.GridSpec(nrows, ncols, figure=fig,
                              hspace=0.55, wspace=0.32)

    for plot_i, label_int in enumerate(available):
        row_i = plot_i // ncols
        col_i = plot_i % ncols
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[row_i, col_i],
            hspace=0.08, height_ratios=[2, 1.5, 0.6])

        seq = viz_sequences[label_int]
        T   = len(seq)
        t   = np.arange(T)
        win_starts, mae_traj = compute_mae_trajectory(seq)

        fault_name = CANONICAL_LABEL_NAMES.get(label_int, f"label_{label_int}")
        ph_title, ph_mechanism, ph_signature = PHYSICS_DESC.get(
            label_int, (fault_name, "", ""))
        prim_ch_idx = PRIMARY_CHANNEL.get(label_int, 0)
        prim_ch_name = CHANNELS[prim_ch_idx]

        # ── Subplot A: Raw sensor signals ─────────────────────────────────
        ax_sig = fig.add_subplot(inner[0])
        ax_sig.set_facecolor('#1A1D27')
        for ci, (ch, col) in enumerate(zip(CH_LIST, COLOR_LIST)):
            lw  = 2.0 if ci == prim_ch_idx else 0.7
            alp = 1.0 if ci == prim_ch_idx else 0.35
            ax_sig.plot(t, seq[:, ci], color=col, linewidth=lw, alpha=alp,
                        label=ch if (ci == prim_ch_idx or alp > 0.3) else "_")

        ax_sig.axvline(ONSET_STEP, color='white', linestyle='--',
                       linewidth=1.2, alpha=0.7, label='Fault onset (t=50)')

        # For compound faults: mark secondary onset if available
        if label_int in COMPOUND_LABELS and label_int in meta_df['label'].values:
            row_sample = meta_df[meta_df['label'] == label_int].iloc[0]
            sos = safe_int(row_sample.get('secondary_onset_step', 0),
                           default=safe_int(row_sample.get('lag_steps', 50), default=50) + ONSET_STEP)
            if sos > ONSET_STEP:
                ax_sig.axvline(sos, color='#F39C12', linestyle='--',
                               linewidth=1.2, alpha=0.8, label=f'Secondary onset (t={sos})')

        ax_sig.set_xlim(0, T)
        ax_sig.set_ylabel('Normalised value\n(P* or ΔT*)', color='white', fontsize=8)
        ax_sig.tick_params(colors='white', labelsize=7)
        ax_sig.set_xticklabels([])
        for sp in ax_sig.spines.values(): sp.set_color('#444')
        ax_sig.set_title(f"[{label_int}] {ph_title}", color='white',
                         fontsize=9, fontweight='bold', pad=3)
        # Primary channel label box
        ax_sig.annotate(f"▶ {prim_ch_name}", xy=(0.01, 0.92),
                        xycoords='axes fraction', color=CH_COLORS[prim_ch_name],
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#0F1117', alpha=0.7))
        ax_sig.legend(loc='upper right', fontsize=6, ncol=2,
                      framealpha=0.3, facecolor='#1A1D27',
                      labelcolor='white', handlelength=1.0)

        # ── Subplot B: Sliding-window MAE trajectory ───────────────────
        ax_mae = fig.add_subplot(inner[1])
        ax_mae.set_facecolor('#1A1D27')

        if len(win_starts) > 0:
            # Plot MAE per channel, primary highlighted
            for ci, (ch, col) in enumerate(zip(CH_LIST, COLOR_LIST)):
                lw  = 2.0 if ci == prim_ch_idx else 0.6
                alp = 1.0 if ci == prim_ch_idx else 0.3
                ax_mae.plot(win_starts + WINDOW_SIZE // 2, mae_traj[:, ci],
                            color=col, linewidth=lw, alpha=alp)

            # Total MAE (mean all channels)
            total_mae = mae_traj.mean(axis=1)
            ax_mae.plot(win_starts + WINDOW_SIZE // 2, total_mae,
                        color='white', linewidth=1.2, alpha=0.8,
                        linestyle=':', label='Mean MAE (all ch.)')

            # Threshold line
            ax_mae.axhline(M4_THRESHOLD, color='#FF4444', linewidth=1.5,
                           linestyle='--', label=f'L1 threshold ({M4_THRESHOLD})')
            ax_mae.axvline(ONSET_STEP, color='white', linestyle='--',
                           linewidth=1.0, alpha=0.6)

            # Shade fault-active region
            ax_mae.axvspan(ONSET_STEP, T, color='#FF4444', alpha=0.04)

            ax_mae.set_ylim(bottom=0)
            ax_mae.set_xlim(0, T)
        ax_mae.set_ylabel('MAE (normalised)', color='white', fontsize=8)
        ax_mae.tick_params(colors='white', labelsize=7)
        ax_mae.set_xticklabels([])
        for sp in ax_mae.spines.values(): sp.set_color('#444')
        ax_mae.legend(loc='upper left', fontsize=6, framealpha=0.3,
                      facecolor='#1A1D27', labelcolor='white', handlelength=1.2)

        # ── Subplot C: Physics annotation bar ─────────────────────────
        ax_ann = fig.add_subplot(inner[2])
        ax_ann.set_facecolor('#0D0F16')
        ax_ann.axis('off')
        ax_ann.text(0.01, 0.72, f"Physics: {ph_mechanism}",
                    transform=ax_ann.transAxes,
                    color='#7FDBFF', fontsize=7.5, va='top',
                    fontfamily='monospace')
        ax_ann.text(0.01, 0.30, f"Signature: {ph_signature}",
                    transform=ax_ann.transAxes,
                    color='#AFFFAF', fontsize=7.5, va='top',
                    fontfamily='monospace')
        ax_mae.set_xlabel('Timestep (1 step = 1 s at 1 Hz)', color='white', fontsize=7)

    save_path = PLOTS_DIR / f"module_06p5r_physics_viz_{grp_key}.png"
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    plt.close()
    log(f"  Saved group viz: {save_path}")

# ── INDIVIDUAL LABEL PLOTS (per-label, full detail) ──────────────────────────
log("  Generating per-label detail plots...")

for label_int in sorted(viz_sequences.keys()):
    seq = viz_sequences[label_int]
    T   = len(seq)
    t   = np.arange(T)
    win_starts, mae_traj = compute_mae_trajectory(seq)

    fault_name   = CANONICAL_LABEL_NAMES.get(label_int, f"label_{label_int}")
    ph_title, ph_mechanism, ph_signature = PHYSICS_DESC.get(
        label_int, (fault_name, "N/A", "N/A"))
    prim_ch_idx  = PRIMARY_CHANNEL.get(label_int, 0)
    prim_ch_name = CHANNELS[prim_ch_idx]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [2.5, 2, 1.2],
                                          'hspace': 0.08},
                             facecolor='#0F1117')

    # ── Panel 1: All 8 raw sensor channels ────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#1A1D27')
    for ci, (ch, col) in enumerate(zip(CH_LIST, COLOR_LIST)):
        lw  = 2.2 if ci == prim_ch_idx else 0.9
        alp = 1.0 if ci == prim_ch_idx else 0.55
        zord = 3 if ci == prim_ch_idx else 1
        ax1.plot(t, seq[:, ci], color=col, linewidth=lw, alpha=alp,
                 label=ch, zorder=zord)

    ax1.axvline(ONSET_STEP, color='white', linestyle='--', linewidth=1.5,
                label=f'Fault onset (t={ONSET_STEP})', zorder=4)
    if label_int in COMPOUND_LABELS:
        row_sample = meta_df[meta_df['label'] == label_int]
        if len(row_sample) > 0:
            sos = safe_int(row_sample.iloc[0].get('secondary_onset_step', 0),
                           default=100)
            if sos > ONSET_STEP:
                ax1.axvline(sos, color='#F39C12', linestyle='-.', linewidth=1.5,
                            label=f'Secondary onset (t={sos})', zorder=4)
                ax1.axvspan(ONSET_STEP, sos,   color='#FF6B35', alpha=0.06,
                            label='Primary fault zone')
                ax1.axvspan(sos,        T,     color='#FF0000', alpha=0.06,
                            label='Compound zone')
    else:
        ax1.axvspan(ONSET_STEP, T, color='#FF4444', alpha=0.05, label='Fault active zone')

    ax1.axhline(1.0, color='#888', linestyle=':', linewidth=0.8, alpha=0.6,
                label='Normal baseline (1.0)')
    ax1.set_xlim(0, T)
    ax1.set_ylabel('Normalised sensor value\n(P* or ΔT*, cluster-relative M3)',
                   color='white', fontsize=9)
    ax1.tick_params(colors='white')
    ax1.set_xticklabels([])
    for sp in ax1.spines.values(): sp.set_color('#555')
    ax1.legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.4,
               facecolor='#1A1D27', labelcolor='white', handlelength=1.2)

    title_str = (f"[Label {label_int}]  {ph_title}\n"
                 f"110 kW · 7-stage · 40 bar · 2980 RPM — PumpSmart v14.2  "
                 f"(Primary channel: {prim_ch_name})")
    ax1.set_title(title_str, color='white', fontsize=10, fontweight='bold', pad=6)

    # Annotate primary channel name on the plot
    ax1.annotate(f"◀ Primary fault channel: {prim_ch_name}",
                 xy=(ONSET_STEP + (T - ONSET_STEP) * 0.02,
                     float(seq[ONSET_STEP:, prim_ch_idx].max()) * 1.02),
                 fontsize=8, color=CH_COLORS[prim_ch_name], fontweight='bold')

    # ── Panel 2: Sliding-window MAE per channel + threshold ───────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1A1D27')

    if len(win_starts) > 0:
        win_mids = win_starts + WINDOW_SIZE // 2

        for ci, (ch, col) in enumerate(zip(CH_LIST, COLOR_LIST)):
            lw  = 2.0 if ci == prim_ch_idx else 0.7
            alp = 1.0 if ci == prim_ch_idx else 0.40
            ax2.plot(win_mids, mae_traj[:, ci], color=col,
                     linewidth=lw, alpha=alp, label=ch)

        total_mae = mae_traj.mean(axis=1)
        ax2.plot(win_mids, total_mae, color='white', linewidth=1.5,
                 linestyle=':', label='Mean MAE (all channels)', zorder=5)

        ax2.axhline(M4_THRESHOLD, color='#FF4444', linewidth=2.0,
                    linestyle='--', label=f'L1 threshold ({M4_THRESHOLD})', zorder=6)
        ax2.axvline(ONSET_STEP, color='white', linestyle='--',
                    linewidth=1.2, alpha=0.7, zorder=4)
        ax2.axvspan(ONSET_STEP, T, color='#FF4444', alpha=0.04)

        # Annotate if any windows exceed threshold
        above = win_mids[total_mae > M4_THRESHOLD]
        if len(above) > 0:
            ax2.annotate(f"  ↑ L1 threshold exceeded\n  first at t={above[0]}",
                         xy=(above[0], M4_THRESHOLD),
                         xytext=(above[0] + max(5, T * 0.05), M4_THRESHOLD * 1.3),
                         color='#FF9999', fontsize=8,
                         arrowprops=dict(arrowstyle='->', color='#FF9999', lw=1.2))
        elif label_int == 21:
            ax2.annotate("Sub-threshold (Paris law low dK/dN)\nCUSUM L3 is primary detector",
                         xy=(T // 2, M4_THRESHOLD * 0.6),
                         color='#FFCC00', fontsize=8, ha='center',
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='#1A1D27', alpha=0.7))

        ax2.set_ylim(bottom=0)
        ax2.set_xlim(0, T)
    ax2.set_ylabel('Reconstruction Error (MAE)\nper 50-step window', color='white', fontsize=9)
    ax2.tick_params(colors='white')
    ax2.set_xticklabels([])
    for sp in ax2.spines.values(): sp.set_color('#555')
    ax2.legend(loc='upper left', fontsize=7.5, ncol=4, framealpha=0.4,
               facecolor='#1A1D27', labelcolor='white', handlelength=1.2)

    # ── Panel 3: Physics context box ──────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor('#0D0F16')
    ax3.axis('off')

    group_names = {0: 'A (single)', 1: 'A (single)', 2: 'A (single)', 3: 'A (single)',
                   4: 'A (single)', 5: 'A (single)', 6: 'A (single)', 7: 'B (compound)',
                   8: 'B (compound)', 9: 'B (compound)', 10: 'B (compound)',
                   11: 'B (compound)', 12: 'B (compound)', 13: 'C (masked)',
                   14: 'C (masked)', 15: 'C (masked)', 16: 'C (masked)', 17: 'C (masked)',
                   18: 'D (cyclic)', 19: 'A (fast)', 20: 'D (cyclic)', 21: 'D (gradual)',
                   22: 'E (multi-sensor)', 23: 'E (multi-sensor)'}

    info_lines = [
        f"  Fault class   : {fault_name}  (label {label_int}, Group {group_names.get(label_int, '?')})",
        f"  Physics mech. : {ph_mechanism}",
        f"  Sensor sig.   : {ph_signature}",
        f"  Primary ch.   : {prim_ch_name}  (highlighted in red/colour above)",
    ]
    if label_int in COMPOUND_LABELS:
        primary_lbl = COMPOUND_PRIMARY_MAP.get(label_int, label_int)
        info_lines.append(
            f"  Compound note : Onset-split applied — pre-secondary windows "
            f"carry label {primary_lbl} ({CANONICAL_LABEL_NAMES.get(primary_lbl, '?')})")
    if label_int == 21:
        info_lines.append(
            "  WARNING       : Sub-threshold MAE is PHYSICALLY CORRECT — "
            "detection via CUSUM L3 slope, NOT amplitude")

    text_block = "\n".join(info_lines)
    ax3.text(0.01, 0.95, text_block, transform=ax3.transAxes,
             color='#DDEEFF', fontsize=8.5, va='top',
             fontfamily='monospace', linespacing=1.6)

    ax3.set_xlabel('Timestep (1 step = 1 s, 1 Hz SCADA, 50-step L1 window)',
                   color='white', fontsize=9)
    axes[2].xaxis.set_visible(True)

    out_label_path = PLOTS_DIR / f"module_06p5r_physics_label{label_int:02d}_{fault_name[:20]}.png"
    plt.savefig(out_label_path, dpi=130, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    plt.close()

log(f"  Per-label physics plots: {len(viz_sequences)} files saved to {PLOTS_DIR}")

# ── SUMMARY: MAE vs threshold, all classes on one canvas ─────────────────────
log("  Generating all-class MAE summary plot...")
try:
    all_labels_sorted = sorted(output_df['label_int'].unique())
    n_labels = len(all_labels_sorted)
    ncols    = 6
    nrows    = int(np.ceil(n_labels / ncols))

    fig_sum, axes_sum = plt.subplots(nrows, ncols,
                                     figsize=(4.5 * ncols, 3.5 * nrows),
                                     facecolor='#0F1117')
    axes_sum = axes_sum.flatten()

    for ax_i, lbl in enumerate(all_labels_sorted):
        ax = axes_sum[ax_i]
        ax.set_facecolor('#1A1D27')
        lbl_df = output_df[output_df['label_int'] == lbl]
        prim_ch  = PRIMARY_CHANNEL.get(int(lbl), 0)
        prim_col = [CH_COLORS[c] for c in CH_LIST][prim_ch]

        # Plot MAE of primary channel as histogram
        prim_col_name = 'mae_' + CHANNELS[prim_ch].replace('.', '')
        if prim_col_name in lbl_df.columns:
            vals = lbl_df[prim_col_name].values
            ax.hist(vals, bins=40, color=prim_col, alpha=0.75, edgecolor='none')
            ax.axvline(M4_THRESHOLD, color='#FF4444', linewidth=1.5,
                       linestyle='--', label=f'thr={M4_THRESHOLD}')
            above_pct = (vals > M4_THRESHOLD).mean() * 100
            ax.set_title(f"[{int(lbl)}] {CANONICAL_LABEL_NAMES.get(int(lbl), '')[:18]}\n"
                         f">{M4_THRESHOLD:.3f}: {above_pct:.0f}%",
                         color='white', fontsize=7, pad=2)
        else:
            ax.set_title(f"[{int(lbl)}]", color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        ax.set_xlabel(f'{CHANNELS[prim_ch]} MAE', color='#aaa', fontsize=6)
        for sp in ax.spines.values(): sp.set_color('#444')

    # Hide unused axes
    for ax_i in range(n_labels, len(axes_sum)):
        axes_sum[ax_i].set_visible(False)

    fig_sum.suptitle("M6.5r — MAE Distribution per Class (primary channel) | "
                     f"L1 threshold = {M4_THRESHOLD}",
                     color='white', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    sum_path = PLOTS_DIR / "module_06p5r_mae_all_classes_summary.png"
    plt.savefig(sum_path, dpi=130, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    plt.close()
    log(f"  Saved: {sum_path}")
except Exception as e:
    log(f"  [WARNING] Summary MAE plot failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11B — STANDARD DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 11B — Standard diagnostic plots...")

try:
    # Plot 1: Class distribution
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='#0F1117')
    ax.set_facecolor('#1A1D27')
    lbl_c = output_df['label_int'].value_counts().sort_index()
    ax.bar(lbl_c.index.astype(str), lbl_c.values, color='steelblue', edgecolor='k')
    ax.axhline(len(output_df) * 0.20, color='red', linestyle='--', label='20% threshold')
    ax.set_xlabel("label_int", color='white'); ax.set_ylabel("Window count", color='white')
    ax.set_title("M6.5r — Class distribution (windows)", color='white')
    ax.tick_params(colors='white'); ax.legend(labelcolor='white', facecolor='#0F1117')
    for sp in ax.spines.values(): sp.set_color('#555')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "module_06p5r_class_distribution.png", dpi=120,
                facecolor='#0F1117'); plt.close()

    # Plot 2: Fisher bar
    if fisher_df is not None:
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0F1117')
        ax.set_facecolor('#1A1D27')
        colors_bar = ['tomato' if v < 0.5 else 'steelblue'
                      for v in fisher_df['fisher_score'][::-1]]
        ax.barh(fisher_df['feature'][::-1], fisher_df['fisher_score'][::-1],
                color=colors_bar)
        ax.axvline(0.5, color='red', linestyle='--', label='Gate F1 = 0.5')
        ax.set_xlabel("Fisher Score", color='white')
        ax.set_title("M6.5r — Feature Discriminability (22-class Fisher Score)",
                     color='white')
        ax.tick_params(colors='white'); ax.legend(labelcolor='white', facecolor='#0F1117')
        for sp in ax.spines.values(): sp.set_color('#555')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "module_06p5r_fisher_scores.png", dpi=120,
                    facecolor='#0F1117'); plt.close()

    # Plot 3: Label 21 slope
    if l21_mask.sum() > 0:
        l21_slopes = output_df[l21_mask]['err_slope_MotSV']
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0F1117')
        ax.set_facecolor('#1A1D27')
        ax.hist(l21_slopes, bins=60, color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', label='Zero slope')
        pct_pos = (l21_slopes > 0).mean() * 100
        ax.set_title(f"Label 21 (bearing_wear_gradual) — err_slope_MotSV\n"
                     f"Paris law: positive slope in {pct_pos:.1f}% windows (Gate D5 ≥95%)",
                     color='white')
        ax.set_xlabel("err_slope_MotSV", color='white')
        ax.set_ylabel("Count", color='white')
        ax.tick_params(colors='white'); ax.legend(labelcolor='white', facecolor='#0F1117')
        for sp in ax.spines.values(): sp.set_color('#555')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "module_06p5r_label21_slope.png", dpi=120,
                    facecolor='#0F1117'); plt.close()

    # Plot 4: score_C GroupA vs GroupB
    if len(ga_sc) > 0 and len(gb_sc) > 0:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0F1117')
        ax.set_facecolor('#1A1D27')
        ax.hist(ga_sc, bins=60, alpha=0.65, label='Group A (single fault)', color='steelblue')
        ax.hist(gb_sc, bins=60, alpha=0.65, label='Group B (compound)', color='tomato')
        ax.axvline(ga_sc.quantile(0.50), color='blue', linestyle='--',
                   label=f'Group A P50 = {ga_sc.quantile(0.50):.4f}')
        ax.set_title("score_C distribution — Gate Z2\nExpected: Group B >> Group A",
                     color='white')
        ax.set_xlabel("score_C (max Δz_t_recon — compound transition detector)",
                      color='white')
        ax.set_ylabel("Count", color='white')
        ax.tick_params(colors='white'); ax.legend(labelcolor='white', facecolor='#0F1117')
        for sp in ax.spines.values(): sp.set_color('#555')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "module_06p5r_score_C_groupAB.png", dpi=120,
                    facecolor='#0F1117'); plt.close()

    # Plot 5: score_B label 21
    if l21_mask.sum() > 0:
        l21_sb = output_df[l21_mask]['score_B']
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0F1117')
        ax.set_facecolor('#1A1D27')
        ax.hist(l21_sb, bins=60, color='purple', edgecolor='none', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', label='Zero slope')
        z3_pct_p = (l21_sb > 0).mean() * 100
        ax.set_title(f"Label 21 — score_B (CUSUM drift input, Gate Z3)\n"
                     f"score_B>0: {z3_pct_p:.1f}% (target ≥90%)", color='white')
        ax.set_xlabel("score_B (OLS slope of z_t_recon_err — CUSUM L3 only)",
                      color='white')
        ax.set_ylabel("Count", color='white')
        ax.tick_params(colors='white'); ax.legend(labelcolor='white', facecolor='#0F1117')
        for sp in ax.spines.values(): sp.set_color('#555')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "module_06p5r_label21_score_B.png", dpi=120,
                    facecolor='#0F1117'); plt.close()

    log("  Standard diagnostic plots done ✓")
except Exception as e:
    log(f"  [WARNING] Standard plots partially failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — RESULTS DICT POPULATE
# ═══════════════════════════════════════════════════════════════════════════════
results.update({
    'M6p5r_window_size':             WINDOW_SIZE,
    'M6p5r_stride':                  STRIDE,
    'M6p5r_n_classes':               int(results.get('M6p5r_n_classes', 0)),
    'M6p5r_domain4_features':        ['z_t_pca_1','z_t_pca_2','z_t_norm','z_t_recon_err',
                                       'score_A','score_B','score_C','onset_order'],
    **{f'M6p5r_gate_{k}': v for k, v in gates.items()},
})
block_gates = {k: v for k, v in gates.items() if v == "FAIL"}
results['Status_for_M7'] = "READY" if not block_gates else "BLOCKED"
log(f"\n  Status for M7: {results['Status_for_M7']}")
log(f"  Block gates: {block_gates}")
log(f"  Warn gates:  {[k for k,v in gates.items() if 'WARN' in str(v)]}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — REPORT
# ═══════════════════════════════════════════════════════════════════════════════
report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
try:
    with open(report_path, 'w' , encoding='utf-8') as f:
        f.write(f"# {SCRIPT_NAME} Report  v2.0\n**Date:** {date.today()}\n\n")
        f.write("## Gate Summary\n| Gate | Result |\n|------|--------|\n")
        for k, v in gates.items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Results\n| Key | Value |\n|-----|-------|\n")
        for k, v in results.items():
            if not isinstance(v, (dict, list)):
                f.write(f"| {k} | {v} |\n")
        f.write(f"\n## Feature Columns ({len(FEATURE_COLS)} features)\n")
        for i, col in enumerate(FEATURE_COLS):
            sc = results.get('M6p5r_fisher_scores', {}).get(col, 0)
            f.write(f"- [{i+1:02d}] `{col}` | Fisher: {sc:.4f}\n"
                    if isinstance(sc, float) else f"- [{i+1:02d}] `{col}`\n")
        f.write(f"\n## Output Files\n- `{output_path}`\n- `{meta_path}`\n")
        f.write(f"\n## Visualizations\n")
        for grp_key in group_defs:
            f.write(f"- `module_06p5r_physics_viz_{grp_key}.png`\n")
        f.write(f"- `module_06p5r_mae_all_classes_summary.png`\n")
        f.write(f"- Per-label: `module_06p5r_physics_label##_*.png` "
                f"({len(viz_sequences)} files)\n")
        f.write(f"\n## Status for M7: **{results['Status_for_M7']}**\n")
    log(f"  Report: {report_path}")
except Exception as e:
    log(f"  [WARNING] Report failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PASTE TEXT UPDATE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print("═"*60)
for k in ['M6p5r_window_size','M6p5r_n_sequences_in','M6p5r_n_windows_out',
          'M6p5r_n_classes','M6p5r_feature_matrix_rows','M6p5r_feature_matrix_cols',
          'M6p5r_z_t_pca_variance_explained','M6p5r_top_fisher_feature',
          'M6p5r_label21_slope_pct_positive','M6p5r_score_C_group_B_pct',
          'M6p5r_score_B_label21_pct_positive','Status_for_M7']:
    print(f"{k:<42}: {results.get(k, 'N/A')}")
for k, v in gates.items():
    print(f"M6p5r_gate_{k:<30}: {v}")
print("M6p5r_output_file                         : data/synthetic/M6B_feature_matrix.csv")
print("═"*60)
print("══ END PASTE UPDATE ══")
print("═"*60)

# FILE MANIFEST
print("\n" + "="*60)
print("FILE MANIFEST")
print("="*60)
print(f"OUTPUTS (→ Spaces + GitHub):")
print(f"  {output_path}")
print(f"  {meta_path}")
print(f"  {report_path}")
print(f"PLOTS ({len(viz_sequences) + len(group_defs) + 7} files): {PLOTS_DIR}")
print("="*60)
print("\n📦 M6.5r done. Starting M7.")
print(f"  Windows: {len(output_df):,} × {len(output_df.columns)} features")
print(f"  Top Fisher: {results.get('M6p5r_top_fisher_feature','?')}")
print(f"  Status: {results['Status_for_M7']}")
print(f"  Uploading: M6B_feature_matrix.csv + module_06p5r_report.md")
print(f"  Provide M7 complete script.")

log("\n" + "="*75)
log(f"  M6.5r COMPLETE v2.0 | {len(output_df):,} rows | Status: {results['Status_for_M7']}")
log("="*75)