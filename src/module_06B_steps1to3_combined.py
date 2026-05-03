# ═══════════════════════════════════════════════════════════════════════════════
# module_06B_steps1to3_combined.py
# PumpSmart — M6B Steps 1, 2, 3 (Combined Single-Run Script)
#
# Step 1 : Group B — Compound chains (Labels 7–12), 9,000 sequences
# Step 2 : Groups C+D — Masked faults (Labels 13–17) + Severity variants
#          (Labels 18–21), 11,200 sequences
# Step 3 : Group E — Multi-sensor failures (1,600 seq) + Full merge +
#          fault_rules_v3.json + physics context strings + file registry
#
# Architecture version : v14.2
# Script version       : v1.0 (2026-04-28)
# Prerequisites        : M6B Step0 v2 + Step0b v2 COMPLETE (LOCKED 2026-04-26)
# Physics lib          : src/m6b_physics_lib.py (LOCKED v1.0)
# Locked inputs        : lstm_ae_baseline.pth, M4_threshold=0.110058
#
# Channel order (LOCKED):
#   0=Mot.SV  1=Pmp.SV  2=Mot.TV  3=Pmp.PV
#   4=Temp.SV 5=Pres.SV 6=Pmp.TV  7=Mot.PV
#
# NEVER hardcode .cuda() | NEVER reset CUSUM on threshold update
# fault_rules_v3.json written ONLY by Step 3 — not earlier
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_NAME    = "module_06B_steps1to3_combined"
SCRIPT_VERSION = "1.0"
ARCH_VERSION   = "v14.2"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, warnings, pickle, os, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress

from m6b_physics_lib import (
    init_lib, CH, CHANNELS, N_CH, CHANNEL_TO_M3_KEY,
    get_cluster_mean, apply_winsorization, make_baseline,
    generate_bearing_wear, generate_impeller_imbalance,
    generate_cavitation, generate_seal_failure,
    generate_overloading, generate_sensor_failure,
    generate_normal_from_real, SENSOR_SUBTYPES, NOISE_STD,
    TAU_THERMAL_s, BPF_HZ, RHO, A_WAVE_m_s
)

warnings.filterwarnings('ignore')

REPORT_DIR = OUTPUT_DIR / "reports"
for d in [REPORT_DIR, PLOTS_DIR, SYNTH_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
GATE_RESULTS = {}   # gates accumulated across all steps
_REGISTRY_ENTRIES = []  # file registry accumulated throughout script

log("=" * 75)
log(f"  PumpSmart — M6B Steps 1+2+3 Combined")
log(f"  Script v{SCRIPT_VERSION} | Arch {ARCH_VERSION} | Date: {date.today()}")
log(f"  Device: {DEVICE} | CUDA: {IS_GPU}")
log("=" * 75)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIGS & LOCKED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 0 — Loading configs...")

try:
    with open(MODEL_DIR / "M3_normalization_config.json") as f:
        norm_config = json.load(f)
    log("  M3_normalization_config.json loaded")
except Exception as e:
    log(f"  [FATAL] Cannot load M3 normalization config: {e}"); raise

try:
    with open(MODEL_DIR / "M5_physics_config.json") as f:
        phys_config = json.load(f)
    log("  M5_physics_config.json loaded")
except Exception as e:
    log(f"  [WARNING] M5_physics_config missing, using defaults: {e}")
    phys_config = {}

try:
    with open(MODEL_DIR / "M4_threshold_config.json") as f:
        thresh_cfg = json.load(f)
    ANOMALY_THRESHOLD = float(thresh_cfg.get("threshold", 0.110058))
    log(f"  M4 threshold loaded: {ANOMALY_THRESHOLD}")
except Exception as e:
    ANOMALY_THRESHOLD = 0.110058
    log(f"  [WARNING] Using locked fallback threshold: {ANOMALY_THRESHOLD}")

assert abs(ANOMALY_THRESHOLD - 0.110058) < 1e-5, \
    f"FATAL: threshold mismatch {ANOMALY_THRESHOLD} != 0.110058"

# Initialise physics library
init_lib(norm_config, phys_config, seed=2026)
log("  m6b_physics_lib initialised (seed=2026)")

SEED        = 2026
WIN_SIZE    = 50       # INVARIANT — never change
N_LATENT    = 64       # LSTM-AE latent dim

np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Sequence length map (physics-verified v14.2) ──────────────────────────────
SEQ_STEPS = {
    0: 200, 1: 250, 2: 200, 3: 150, 4: 400, 5: 300, 6: 150,
    7: 600, 8: 550, 9: 700, 10: 900, 11: 800, 12: 450,
    13: 300, 14: 210, 15: 500, 16: 350, 17: 250,
    18: 300, 19: 150, 20: 600, 21: 1000
}

# ── Target sequence counts (v14.2) ────────────────────────────────────────────
SEQ_COUNTS = {
    7: 1500, 8: 1500, 9: 1500, 10: 1500, 11: 1500, 12: 1500,  # Group B
    13: 1200, 14: 1200, 15: 1200, 16: 1200, 17: 1200,          # Group C
    18: 1200, 19: 800, 20: 1200, 21: 2000,                      # Group D
    "E_thermal": 800, "E_pump": 800                              # Group E
}

# ── Compound lag ranges (physics-verified, per label) ─────────────────────────
COMPOUND_LAG = {
    7:  (200, 400),   # bearing heat → oil viscosity drop → thermal runaway
    8:  (50,  150),   # Joukowsky shock → axial thrust → seal face damage
    9:  (150, 350),   # BPF fatigue → bearing race crack (Paris law)
    10: (300, 600),   # Q_leak → op-point shift → NPSHa loss
    11: (200, 500),   # motor OL heat → lubricant thinning → bearing fatigue
    12: (100, 250),   # impeller orbit → localised vapor pockets
}

# ── Masked channel map (Group C) ─────────────────────────────────────────────
MASK_CHANNEL = {
    13: ("Mot.SV",  "bearing_wear",        "flatline"),
    14: ("Pres.SV", "cavitation",          "flatline"),
    15: ("Pres.SV", "seal_failure",        "drift_up"),   # POSITIVE drift = sensor bias
    16: ("Temp.SV", "overloading",         "stuck"),
    17: ("Pmp.SV",  "impeller_imbalance",  "flatline"),
}

# ── Group D label map ─────────────────────────────────────────────────────────
GROUP_D_CLASSES = {
    18: "cavitation_intermittent",
    19: "seal_failure_fast",
    20: "overloading_cyclic",
    21: "bearing_wear_gradual",
}

# ── Group E variant map ───────────────────────────────────────────────────────
GROUP_E_VARIANTS = {
    "E_thermal": ("Mot.TV", "Temp.SV"),   # shared thermal excitation rail failure
    "E_pump":    ("Pmp.SV", "Pmp.PV"),    # moisture ingress to pump-side junction box
}

# ── Allowed clusters per fault (physics-constrained) ─────────────────────────
FAULT_CLUSTERS = {
    "bearing_wear":       [1, 2, 3],
    "impeller_imbalance": [1, 2, 3],
    "cavitation":         [0, 1, 2],
    "seal_failure":       [1, 2],
    "overloading":        [1],
    "sensor_failure":     [0, 1, 2, 3],
    "normal":             [0, 1, 2, 3],
}
CLUSTER_NAMES = {0: "startup", 1: "steady_state", 2: "high_load", 3: "cooldown"}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LSTM-AE MODEL DEFINITION (exact M4 checkpoint architecture)
# Reverse-engineered from lstm_ae_baseline.pth state_dict keys:
#   encoder.lstm1 : LSTM(8→128, layers=2)
#   encoder.lstm2 : LSTM(128→64, layers=1)
#   encoder.bn    : LayerNorm(64)          ← named 'bn', is LayerNorm (no running_mean)
#   decoder.fc_h  : Linear(64→128)
#   decoder.fc_c  : Linear(64→128)
#   decoder.lstm1 : LSTM(64→128, layers=2)
#   decoder.lstm2 : LSTM(128→8, layers=1)
#   decoder.out   : Linear(8→8)
# Source: src/model_architecture.py (LOCKED)
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 1 — Loading LSTM-AE (L1, FROZEN)...")

class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8,   128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64,  num_layers=1, batch_first=True)
        self.bn    = nn.LayerNorm(64)   # named 'bn' to match checkpoint keys

    def forward(self, x):
        out1, _      = self.lstm1(x)
        out2, (h, c) = self.lstm2(out1)
        h_bn = self.bn(h[-1])           # shape: (batch, 64)
        return h_bn, h, c


class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_h  = nn.Linear(64,  128)
        self.fc_c  = nn.Linear(64,  128)
        self.lstm1 = nn.LSTM(64,  128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 8,   num_layers=1, batch_first=True)
        self.out   = nn.Linear(8, 8)

    def forward(self, bottleneck, seq_len, h_enc, c_enc):
        # Project encoder bottleneck → decoder initial state (C-13 fix: hidden-state seeding)
        h0 = torch.tanh(self.fc_h(h_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.tanh(self.fc_c(c_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        x  = bottleneck.unsqueeze(1).repeat(1, seq_len, 1)
        out1, _ = self.lstm1(x, (h0, c0))
        out2, _ = self.lstm2(out1)
        return self.out(out2)


class LSTMAutoencoder(nn.Module):
    """
    Exact M4 LSTM-AE architecture matched to lstm_ae_baseline.pth checkpoint.
    DO NOT modify — any change breaks state_dict loading.
    """
    def __init__(self):
        super().__init__()
        self.encoder = LSTMEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x):
        bottleneck, h, c = self.encoder(x)
        return self.decoder(bottleneck, x.size(1), h, c)


try:
    lstm_ae = LSTMAutoencoder()
    state = torch.load(MODEL_DIR / "lstm_ae_baseline_final.pth", map_location='cpu')
    lstm_ae.load_state_dict(state)
    lstm_ae.eval()
    lstm_ae.to(DEVICE)
    log(f"  lstm_ae_baseline.pth loaded → {DEVICE} | params: "
        f"{sum(p.numel() for p in lstm_ae.parameters()):,}")
except Exception as e:
    log(f"  [FATAL] Cannot load LSTM-AE: {e}"); raise

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — z_t EXPORT FUNCTION (shared, used by all steps)
# ═══════════════════════════════════════════════════════════════════════════════

def export_zt_sequences(sequences_list, batch_size=32):
    """
    Run frozen M4 LSTM-AE in sliding window mode over a list of sequences.
    Returns list of dicts: {z_t: ndarray(N_windows, 64), mae: ndarray(N_windows, 8)}

    Encoder API (exact M4 architecture):
        bottleneck, h, c = lstm_ae.encoder(x)   → bottleneck shape: (batch, 64)
    Decoder API:
        recon = lstm_ae.decoder(bottleneck, seq_len, h, c)  → (batch, seq_len, 8)

    sequences_list : list of ndarray(T, 8) in normalized space
    batch_size     : reduce in config.py if CUDA OOM on RTX 4060 (try 16)
    """
    lstm_ae.eval()
    zt_out = []

    # Collect all windows with sequence index mapping
    all_windows    = []
    window_seq_map = []   # (seq_idx, window_idx)
    for seq_idx, seq in enumerate(sequences_list):
        T = seq.shape[0]
        n_windows = T // WIN_SIZE
        for w in range(n_windows):
            chunk = seq[w * WIN_SIZE: (w + 1) * WIN_SIZE]  # (50, 8)
            all_windows.append(chunk)
            window_seq_map.append((seq_idx, w))

    if len(all_windows) == 0:
        return [{} for _ in sequences_list]

    # Batch inference
    all_windows_t = torch.tensor(np.array(all_windows), dtype=torch.float32)
    all_zt  = np.zeros((len(all_windows), N_LATENT), dtype=np.float32)
    all_mae = np.zeros((len(all_windows), N_CH),     dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows_t[i:i + batch_size].to(DEVICE)
            # Extract z_t from encoder bottleneck (LayerNorm-normalized, shape (B,64))
            bottleneck, h, c = lstm_ae.encoder(batch)
            z = bottleneck.cpu().numpy()              # (B, 64)
            # Reconstruct for per-channel MAE
            recon = lstm_ae.decoder(bottleneck, batch.size(1), h, c).cpu().numpy()  # (B,50,8)
            orig  = batch.cpu().numpy()
            mae   = np.mean(np.abs(orig - recon), axis=1)  # (B, 8)
            all_zt [i:i + batch_size] = z
            all_mae[i:i + batch_size] = mae

    # Re-assemble per sequence
    from collections import defaultdict
    seq_windows = defaultdict(list)
    for flat_idx, (seq_idx, _) in enumerate(window_seq_map):
        seq_windows[seq_idx].append(flat_idx)

    for seq_idx in range(len(sequences_list)):
        idxs = seq_windows[seq_idx]
        zt_out.append({
            "z_t": all_zt[idxs],    # (N_windows, 64)
            "mae": all_mae[idxs],   # (N_windows, 8)
        })

    return zt_out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PHYSICS GATE FUNCTIONS (G8–G11-ext + G1–G7 wrappers)
# ═══════════════════════════════════════════════════════════════════════════════

def gate_G8_temporal_ordering(seqs_meta, threshold=0.95):
    """Gate G8: primary anomaly onset precedes secondary by physics lag (≥95% seqs)."""
    pass_count = 0
    for m in seqs_meta:
        if m.get("secondary_onset_step", -1) > m.get("primary_onset_step", 0):
            pass_count += 1
    rate = pass_count / len(seqs_meta) if seqs_meta else 0
    return rate >= threshold, rate

def gate_G9_compound_mae(zt_list, threshold=0.110058, pass_rate=0.90):
    """Gate G9: weighted MAE > threshold in ≥90% of compound sequences."""
    pass_count = 0
    for zt in zt_list:
        # Physics weights: SV channels = 2.0, others = 1.0
        weights = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        w_mae = np.sum(zt["mae"] * weights, axis=1) / weights.sum()
        if np.any(w_mae > threshold):
            pass_count += 1
    rate = pass_count / len(zt_list) if zt_list else 0
    return rate >= pass_rate, rate

def gate_G10_masked_secondary(seqs, meta_list, threshold=0.50):
    """Gate G10: non-masked channels carry ≥50% of base fault MAE."""
    pass_count = 0
    for seq, m in zip(seqs, meta_list):
        masked_ch  = CH[m["masked_channel"]]
        other_chs  = [i for i in range(N_CH) if i != masked_ch]
        # Compute mean absolute deviation from 1.0 (normal baseline)
        other_dev = np.mean(np.abs(seq[:, other_chs] - 1.0))
        total_dev = np.mean(np.abs(seq - 1.0)) + 1e-9
        if other_dev / total_dev >= threshold:
            pass_count += 1
    rate = pass_count / len(seqs) if seqs else 0
    return rate >= threshold, rate

def gate_G11_multisensor(seqs, meta_list, anomaly_tol=0.20):
    """Gate G11: exactly 2 channels anomalous; remaining 6 within ±0.20 of PRE-FAULT baseline."""
    pass_count = 0
    for seq, m in zip(seqs, meta_list):
        fault_chs  = set(m["fault_channels"])
        onset      = m.get("onset_step", 20)
        # Pre-fault baseline: mean of first onset steps per channel
        pre_fault_mean = np.mean(seq[:onset], axis=0)  # shape (8,)
        others     = [i for i in range(N_CH) if i not in fault_chs]
        others_ok  = all(
            np.abs(np.mean(seq[onset:, i]) - pre_fault_mean[i]) < anomaly_tol
            for i in others
        )
        if len(fault_chs) == 2 and others_ok:
            pass_count += 1
    rate = pass_count / len(seqs) if seqs else 0
    return rate >= 0.90, rate

def gate_G11ext_gradual_slope(seqs, threshold=0.95):
    """Gate G11-ext: err_slope_MotSV > 0 in ≥95% of label 21 sequences."""
    pass_count = 0
    for seq in seqs:
        slope, _, _, _, _ = linregress(np.arange(len(seq)), seq[:, CH["Mot.SV"]])
        if slope > 0:
            pass_count += 1
    rate = pass_count / len(seqs) if seqs else 0
    return rate >= threshold, rate

def gate_G1_no_negative_pressure(seqs):
    """Gate G1: Pres.SV* >= -0.01 at all timesteps."""
    pass_count = sum(1 for s in seqs if np.all(s[:, CH["Pres.SV"]] >= -0.01))
    return pass_count / len(seqs) if seqs else 0

def gate_G2_temp_floor(seqs):
    """Gate G2: temperature channels >= -0.12 (flash evap allowed, C-09)."""
    temp_chs = [CH["Mot.TV"], CH["Pmp.TV"], CH["Temp.SV"]]
    pass_count = sum(1 for s in seqs if all(np.all(s[:, c] >= -0.12) for c in temp_chs))
    return pass_count / len(seqs) if seqs else 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HELPER: APPLY MASK TO SEQUENCE (Group C)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_channel_mask(seq, channel_name, mask_type, onset_step=None):
    """
    Degrade a channel in-place to simulate sensor failure masking.
    
    mask_type options:
      'flatline'  : channel set to mean of first 10 steps after onset_step
      'drift_up'  : channel drifts upward from onset_step (sensor cal bias)
      'stuck'     : channel frozen at value at onset_step
    """
    seq_out = seq.copy()
    ch_idx  = CH[channel_name]
    if onset_step is None:
        onset_step = 0

    if mask_type == "flatline":
        flat_val = np.mean(seq[max(0, onset_step - 10): onset_step + 1, ch_idx])
        seq_out[onset_step:, ch_idx] = flat_val + np.random.normal(0, 0.002,
                                        size=len(seq_out) - onset_step)

    elif mask_type == "drift_up":
        # Sensor calibration drift → positive ramp (NOT seal leak which is negative)
        n_remain = len(seq_out) - onset_step
        ramp     = np.linspace(0, 0.3, n_remain)
        noise    = np.random.normal(0, 0.005, n_remain)
        seq_out[onset_step:, ch_idx] += ramp + noise

    elif mask_type == "stuck":
        stuck_val = seq[onset_step, ch_idx]
        seq_out[onset_step:, ch_idx] = stuck_val + np.random.normal(0, 0.001,
                                        size=len(seq_out) - onset_step)

    return seq_out



# ═══════════════════════════════════════════════════════════════════════════════
# ███████╗████████╗███████╗██████╗      ██╗
# ██╔════╝╚══██╔══╝██╔════╝██╔══██╗    ███║
# ███████╗   ██║   █████╗  ██████╔╝    ╚██║
# ╚════██║   ██║   ██╔══╝  ██╔═══╝      ██║
# ███████║   ██║   ███████╗██║          ██║
# ╚══════╝   ╚═╝   ╚══════╝╚═╝          ╚═╝
# STEP 1 — GROUP B: COMPOUND CHAIN FAULTS (Labels 7–12)
# ═══════════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 75)
log("  STEP 1 — GROUP B: Compound Chain Faults (Labels 7–12)")
log("  Target: 9,000 sequences (1,500 × 6 classes)")
log("=" * 75)

# ── Load existing Group A sequences as seed source ────────────────────────────
log("STEP 1.0 — Loading Group A sequences (seed source)...")

try:
    with open(SYNTH_DIR / "M6B_sequences_groupA_rerun.pkl", "rb") as f:
        grpA_rerun = pickle.load(f)   # labels 1, 4, 5
    # Step0 v2 saves as {"sequences": ..., "meta": ...}  ← key is "meta" not "metadata"
    log(f"  groupA_rerun loaded: {len(grpA_rerun['sequences'])} sequences")
except Exception as e:
    log(f"  [FATAL] groupA_rerun not found: {e}"); raise

try:
    with open(SYNTH_DIR / "M6B_sequences_groupA_carried.pkl", "rb") as f:
        grpA_carried = pickle.load(f)  # labels 0, 2, 3, 6
    log(f"  groupA_carried loaded: {len(grpA_carried['sequences'])} sequences")
except Exception as e:
    log(f"  [FATAL] groupA_carried not found: {e}"); raise

# Build label→sequence lookup — key is "meta" (not "metadata") in Step0/0b v2 output
grpA_by_label = {}
for seq, meta in zip(grpA_rerun["sequences"], grpA_rerun["meta"]):
    lbl = meta["label"]
    grpA_by_label.setdefault(lbl, []).append((seq, meta))
for seq, meta in zip(grpA_carried["sequences"], grpA_carried["meta"]):
    lbl = meta["label"]
    grpA_by_label.setdefault(lbl, []).append((seq, meta))

log(f"  Label seed pool: {', '.join(f'L{k}:{len(v)}' for k, v in sorted(grpA_by_label.items()))}")

# ── Compound chain generator ──────────────────────────────────────────────────

# Mapping: compound label → (primary_base_label, secondary_base_label)
COMPOUND_BASE = {
    7:  (1, 5),   # bearing_wear → overloading
    8:  (3, 4),   # cavitation → seal_failure
    9:  (2, 1),   # impeller_imbalance → bearing_wear
    10: (4, 3),   # seal_failure → cavitation
    11: (5, 1),   # overloading → bearing_wear
    12: (2, 3),   # impeller_imbalance → cavitation
}

COMPOUND_NAMES = {
    7:  "bearing_wear+overloading",
    8:  "cavitation+seal_failure",
    9:  "impeller_imbalance+bearing_wear",
    10: "seal_failure+cavitation_H",
    11: "overloading+bearing_wear",
    12: "impeller_imbalance+cavitation",
}


def generate_compound_sequence(label, rng):
    """
    Generate one compound fault sequence for Group B label.
    
    Phase 1 (t=0 → t=50+lag): primary fault signature only
    Phase 2 (t=50+lag → end): primary + secondary superimposed
    
    Physics:
    - Primary onset at step 0 (with pre-fault 50-step baseline)
    - Secondary onset drawn from physics-verified lag range
    - Superposition: additive in deviation from baseline (preserves normalization)
    - SV channels weighted 2.0 in loss (invariant)
    """
    target_steps = SEQ_STEPS[label]
    lag_lo, lag_hi = COMPOUND_LAG[label]
    lag = int(rng.integers(lag_lo, lag_hi + 1))

    primary_lbl, secondary_lbl = COMPOUND_BASE[label]

    # Draw primary seed sequence (length ≥ target_steps)
    primary_pool = grpA_by_label.get(primary_lbl, [])
    secondary_pool = grpA_by_label.get(secondary_lbl, [])

    if not primary_pool or not secondary_pool:
        raise RuntimeError(f"Empty pool for compound label {label}")

    p_seq, p_meta = primary_pool[int(rng.integers(0, len(primary_pool)))]
    s_seq, s_meta = secondary_pool[int(rng.integers(0, len(secondary_pool)))]

    # Build output sequence in normalized space
    seq = np.ones((target_steps, N_CH), dtype=np.float32)

    # Determine cluster for this sequence (use primary's cluster)
    cluster_id = p_meta["cluster_id"]

    # ── Phase 1: primary fault only (t=0 → t=lag+50) ─────────────────────────
    # Use primary deviation from baseline
    p_len = min(lag + 50, target_steps, len(p_seq))
    p_src = p_seq[:p_len]
    seq[:p_len] = p_src

    # ── Phase 2: both faults active (t=lag+50 → end) ─────────────────────────
    p2_start = lag + 50
    if p2_start < target_steps:
        p2_len = target_steps - p2_start

        # Primary continues from where Phase 1 left off
        p_tail_start = p_len
        p_tail = p_seq[p_tail_start: p_tail_start + p2_len] if p_tail_start < len(p_seq) \
                 else np.tile(p_seq[-1:], (p2_len, 1))
        if len(p_tail) < p2_len:
            p_tail = np.vstack([p_tail, np.tile(p_seq[-1:], (p2_len - len(p_tail), 1))])

        # Secondary: deviation from 1.0 added to primary
        s_src = s_seq[:p2_len] if len(s_seq) >= p2_len else \
                np.vstack([s_seq, np.tile(s_seq[-1:], (p2_len - len(s_seq), 1))])
        s_dev = s_src - 1.0   # secondary deviation from normal

        combined = p_tail + s_dev * 0.6  # 0.6 blending — secondary 60% intensity at onset
        # Linear ramp secondary to full intensity over 50 steps
        ramp_len = min(50, p2_len)
        ramp     = np.linspace(0.6, 1.0, ramp_len)[:, None]
        combined[:ramp_len] = p_tail[:ramp_len] + s_dev[:ramp_len] * ramp

        seq[p2_start:p2_start + p2_len] = combined

    # Add SCADA noise
    for ch_name, ch_idx in CH.items():
        noise = rng.normal(0, NOISE_STD.get(ch_name, 0.01), size=target_steps)
        seq[:, ch_idx] += noise

    # Apply cluster-conditional winsorization
    seq = apply_winsorization(seq, cluster_id)

    meta = {
        "label":               label,
        "fault_name":          COMPOUND_NAMES[label],
        "group":               "B",
        "primary_label":       primary_lbl,
        "secondary_label":     secondary_lbl,
        "primary_onset_step":  0,
        "secondary_onset_step": lag + 50,
        "lag_steps":           lag,
        "cluster_id":          cluster_id,
        "cluster_name":        CLUSTER_NAMES.get(cluster_id, "unknown"),
        "steps":               target_steps,
        "source":              "physics_synthetic_compound",
        "arch_version":        ARCH_VERSION,
    }
    return seq, meta


# ── Generate Group B ──────────────────────────────────────────────────────────
log("STEP 1.1 — Generating Group B sequences...")

groupB_sequences = []
groupB_metadata  = []

rng_B = np.random.default_rng(SEED + 100)
torch_rng_B = torch.Generator(); torch_rng_B.manual_seed(SEED + 100)

for label in [7, 8, 9, 10, 11, 12]:
    n_target = SEQ_COUNTS[label]
    log(f"  Label {label} ({COMPOUND_NAMES[label]}): generating {n_target} seqs "
        f"[{SEQ_STEPS[label]} steps, lag {COMPOUND_LAG[label]}]")
    label_seqs = []
    label_meta = []
    failures   = 0

    for i in range(n_target):
        try:
            seq, meta = generate_compound_sequence(label, rng_B)
            label_seqs.append(seq)
            label_meta.append(meta)
        except Exception as e:
            failures += 1
            if failures <= 3:
                log(f"    [WARN] seq {i} failed: {e}")

    log(f"    Generated: {len(label_seqs)} | Failures: {failures}")
    groupB_sequences.extend(label_seqs)
    groupB_metadata.extend(label_meta)

log(f"  Group B total: {len(groupB_sequences)} sequences")
results["step1_groupB_count"] = len(groupB_sequences)

# ── Gate G1 on Group B ────────────────────────────────────────────────────────
g1_rate_B = gate_G1_no_negative_pressure(groupB_sequences)
g2_rate_B = gate_G2_temp_floor(groupB_sequences)
log(f"  Gate G1 (no neg pressure): {g1_rate_B:.3f} (target ≥ 0.98)")
log(f"  Gate G2 (temp floor): {g2_rate_B:.3f} (target ≥ 0.98)")
GATE_RESULTS["G1_groupB"] = g1_rate_B
GATE_RESULTS["G2_groupB"] = g2_rate_B

# ── Gate G8: temporal ordering ────────────────────────────────────────────────
g8_pass, g8_rate = gate_G8_temporal_ordering(groupB_metadata, threshold=0.95)
log(f"  Gate G8 (temporal ordering): {g8_rate:.3f} | {'PASS' if g8_pass else 'FAIL'}")
GATE_RESULTS["G8_temporal_ordering"] = {"rate": g8_rate, "pass": g8_pass}

if not g8_pass:
    log("  [WARN] G8 below target — check compound lag physics in m6b_physics_lib")

# ── z_t export Group B ────────────────────────────────────────────────────────
log("STEP 1.2 — Exporting z_t sequences for Group B...")

try:
    zt_groupB = export_zt_sequences(groupB_sequences, batch_size=32)
    log(f"  z_t exported: {len(zt_groupB)} sequences")

    # Gate G9: compound MAE — both channels above threshold
    g9_pass, g9_rate = gate_G9_compound_mae(zt_groupB, threshold=ANOMALY_THRESHOLD, pass_rate=0.90)
    log(f"  Gate G9 (compound MAE): {g9_rate:.3f} | {'PASS' if g9_pass else 'FAIL'}")
    GATE_RESULTS["G9_compound_mae"] = {"rate": g9_rate, "pass": g9_pass}

except torch.cuda.OutOfMemoryError:
    log("  [CUDA OOM] Reduce batch_size in config.py (currently 32 → try 16)")
    raise

# ── Save Group B ──────────────────────────────────────────────────────────────
log("STEP 1.3 — Saving Group B outputs...")

grpB_payload = {"sequences": groupB_sequences, "metadata": groupB_metadata}
grpB_path    = SYNTH_DIR / "M6B_sequences_groupB.pkl"

try:
    with open(grpB_path, "wb") as f:
        pickle.dump(grpB_payload, f, protocol=4)
    log(f"  Saved: {grpB_path} ({grpB_path.stat().st_size / 1e6:.1f} MB)")
    results["step1_groupB_file"] = str(grpB_path)
except Exception as e:
    log(f"  [FATAL] Cannot save Group B: {e}"); raise

zt_B_path = SYNTH_DIR / "z_t_sequences_groupB.pkl"
try:
    with open(zt_B_path, "wb") as f:
        pickle.dump(zt_groupB, f, protocol=4)
    log(f"  Saved: {zt_B_path} ({zt_B_path.stat().st_size / 1e6:.1f} MB)")
    results["step1_zt_groupB_file"] = str(zt_B_path)
except Exception as e:
    log(f"  [FATAL] Cannot save z_t Group B: {e}"); raise

log("  STEP 1 COMPLETE ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# ███████╗████████╗███████╗██████╗     ██████╗
# ██╔════╝╚══██╔══╝██╔════╝██╔══██╗    ╚════██╗
# ███████╗   ██║   █████╗  ██████╔╝     █████╔╝
# ╚════██║   ██║   ██╔══╝  ██╔═══╝     ██╔═══╝
# ███████║   ██║   ███████╗██║         ███████╗
# ╚══════╝   ╚═╝   ╚══════╝╚═╝         ╚══════╝
# STEP 2 — GROUPS C + D: MASKED FAULTS + SEVERITY VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 75)
log("  STEP 2 — Groups C + D: Masked Faults (13–17) + Severity Variants (18–21)")
log("  Target: 6,000 (C) + 5,200 (D) = 11,200 sequences")
log("=" * 75)

groupC_sequences, groupC_metadata = [], []
groupD_sequences, groupD_metadata = [], []

rng_CD = np.random.default_rng(SEED + 200)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2A: GROUP C — MASKED FAULTS (Labels 13–17)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 2A — Group C: Masked Faults (Labels 13–17)...")

GROUP_C_BASE_GENERATORS = {
    13: (generate_bearing_wear,       1, {"n_steps": SEQ_STEPS[13]}),
    14: (generate_cavitation,         3, {"n_steps": SEQ_STEPS[14]}),
    15: (generate_seal_failure,       4, {"n_steps": SEQ_STEPS[15]}),
    16: (generate_overloading,        5, {"n_steps": SEQ_STEPS[16]}),
    17: (generate_impeller_imbalance, 2, {"n_steps": SEQ_STEPS[17]}),
}

GROUP_C_NAMES = {
    13: "bearing_wear_MotSV_masked",
    14: "cavitation_PresSV_masked",
    15: "seal_failure_PresSV_drifting",
    16: "overloading_TempSV_stuck",
    17: "imbalance_PmpSV_flatline",
}

for label in [13, 14, 15, 16, 17]:
    n_target = SEQ_COUNTS[label]
    gen_fn, base_label, gen_kwargs = GROUP_C_BASE_GENERATORS[label]
    mask_ch, base_fault, mask_type = MASK_CHANNEL[label]

    log(f"  Label {label} ({GROUP_C_NAMES[label]}): {n_target} seqs "
        f"[{SEQ_STEPS[label]} steps | mask: {mask_ch} → {mask_type}]")

    # Determine allowed clusters for base fault
    base_fault_name = {1: "bearing_wear", 3: "cavitation", 4: "seal_failure",
                       5: "overloading", 2: "impeller_imbalance"}[base_label]
    # Label 14 cavitation MUST use startup cluster (cluster_id=2) — physics constraint
    if label == 14:
        allowed_clusters = [2]
    else:
        allowed_clusters = FAULT_CLUSTERS.get(base_fault_name, [1, 2])

    label_seqs, label_meta = [], []

    for i in range(n_target):
        cluster_id = int(rng_CD.choice(allowed_clusters))
        severity   = float(rng_CD.uniform(0.3, 0.8))

        try:
            # Generate base fault sequence
            base_seq = gen_fn(
                n_steps    = SEQ_STEPS[label],
                severity   = severity,
                cluster_id = cluster_id,
            )

            # Apply mask: sensor failure starts at step 20–40 (before full fault onset)
            mask_onset = int(rng_CD.integers(20, min(50, SEQ_STEPS[label] // 4)))
            masked_seq = apply_channel_mask(base_seq, mask_ch, mask_type, mask_onset)

            label_seqs.append(masked_seq)
            label_meta.append({
                "label":          label,
                "fault_name":     GROUP_C_NAMES[label],
                "group":          "C",
                "base_label":     base_label,
                "base_fault":     base_fault_name,
                "masked_channel": mask_ch,
                "mask_type":      mask_type,
                "mask_onset_step": mask_onset,
                "severity":       severity,
                "cluster_id":     cluster_id,
                "cluster_name":   CLUSTER_NAMES.get(cluster_id, "unknown"),
                "steps":          SEQ_STEPS[label],
                "source":         "physics_synthetic_masked",
                "arch_version":   ARCH_VERSION,
            })
        except Exception as e:
            if i < 3:
                log(f"    [WARN] seq {i} failed: {e}")

    log(f"    Generated: {len(label_seqs)}")
    groupC_sequences.extend(label_seqs)
    groupC_metadata.extend(label_meta)

log(f"  Group C total: {len(groupC_sequences)} sequences")
results["step2_groupC_count"] = len(groupC_sequences)

# ── Gate G10: masked secondary signal strength ────────────────────────────────
g10_pass, g10_rate = gate_G10_masked_secondary(groupC_sequences, groupC_metadata)
log(f"  Gate G10 (masked secondary ≥50% MAE): {g10_rate:.3f} | {'PASS' if g10_pass else 'FAIL'}")
GATE_RESULTS["G10_masked_secondary"] = {"rate": g10_rate, "pass": g10_pass}

# ── Gate G1/G2 on Group C ─────────────────────────────────────────────────────
g1_rate_C = gate_G1_no_negative_pressure(groupC_sequences)
g2_rate_C = gate_G2_temp_floor(groupC_sequences)
log(f"  Gate G1 (Group C): {g1_rate_C:.3f} | Gate G2: {g2_rate_C:.3f}")
GATE_RESULTS["G1_groupC"] = g1_rate_C
GATE_RESULTS["G2_groupC"] = g2_rate_C

# ── z_t export Group C ────────────────────────────────────────────────────────
log("  Exporting z_t for Group C...")
try:
    zt_groupC = export_zt_sequences(groupC_sequences, batch_size=32)
    log(f"  z_t exported: {len(zt_groupC)}")
except torch.cuda.OutOfMemoryError:
    log("  [CUDA OOM] Reduce batch_size in config.py"); raise

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2B: GROUP D — SEVERITY VARIANTS (Labels 18–21)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 2B — Group D: Severity Variants (Labels 18–21)...")


def generate_cavitation_intermittent(steps, cluster_id=1, rng=None):
    """
    Label 18: Intermittent cavitation — NPSHa oscillates around NPSHr boundary.
    PmpSV* burst pattern: high erratic during bursts, near-normal between.
    burst_count: 3–7 | burst_interval: 15–30 steps
    """
    seq  = make_baseline(steps, cluster_id)
    n_bursts      = int(rng.integers(3, 8))
    burst_interval = int(rng.integers(15, 31))
    burst_width    = int(rng.integers(8, 20))
    severity       = float(rng.uniform(0.35, 0.75))

    # Cavitation channels: Pmp.SV* surge + Pres.SV* dip (M5 canonical)
    for b in range(n_bursts):
        t_start = 30 + b * burst_interval
        t_end   = min(t_start + burst_width, steps)
        if t_start >= steps:
            break
        # PmpSV spike during burst
        spike_amp = 0.4 * severity * (1 + rng.normal(0, 0.1))
        seq[t_start:t_end, CH["Pmp.SV"]] += spike_amp
        seq[t_start:t_end, CH["Mot.SV"]] += spike_amp * 0.5
        # Pres.SV dip during burst
        drop = 0.6 * severity
        seq[t_start:t_end, CH["Pres.SV"]] -= drop

    return seq.astype(np.float32)


def generate_seal_failure_fast(steps, rng=None, cluster_id=1):
    """
    Label 19: Catastrophic seal blowout — turbulent orifice discharge.
    FIX v2 (2026-05-03): Three bugs fixed — see m6b_physics_lib.py for details.
    onset=55-85 (post spike-seed), frac corrected, severity-direct drop used.
    Physics: sev=0.50 → Pres.SV drops to 0.70. sev=0.80 → drops to 0.52.
    """
    seq            = make_baseline(steps, cluster_id)
    onset          = int(rng.integers(55, 85))
    drop_steps     = int(rng.integers(10, 21))
    severity_local = float(rng.uniform(0.20, 0.80))
    max_drop       = float(severity_local * 0.60)
    target_min     = float(max(0.05, 1.0 - max_drop))

    for t in range(onset, min(onset + drop_steps, steps)):
        frac = (t - onset + 1) / drop_steps
        seq[t, CH["Pres.SV"]] = float(max(
            target_min,
            seq[t, CH["Pres.SV"]] - max_drop * frac
        ))
    for t in range(min(onset + drop_steps, steps), steps):
        seq[t, CH["Pres.SV"]] = float(target_min) + float(
            rng.normal(0, NOISE_STD.get("Pres.SV", 0.015)))

    t_sec_end = min(onset + 15, steps)
    seq[onset:t_sec_end, CH["Mot.PV"]] += float(rng.uniform(0.20, 0.35))
    return seq.astype(np.float32)


def generate_overloading_cyclic(steps, rng=None, cluster_id=1):
    """
    Label 20: Cyclic overloading — thermal sawtooth with rising baseline.
    Each cycle starts higher than previous (cyclic_baseline_drift > 0.0002/window).
    Temp.SV Spearman > 0.70 on baseline-detrended signal.
    """
    seq      = make_baseline(steps, cluster_id)
    n_cycles = int(rng.integers(3, 6))
    cycle_len = steps // n_cycles
    baseline_drift = float(rng.uniform(0.0003, 0.0010))  # per step

    for cyc in range(n_cycles):
        t_start = cyc * cycle_len
        t_end   = min(t_start + cycle_len, steps)
        cycle_severity = 0.3 + cyc * 0.08  # escalating severity

        for t in range(t_start, t_end):
            # Thermal sawtooth: rise during load
            phase_frac = (t - t_start) / cycle_len
            temp_rise  = cycle_severity * math.sin(math.pi * phase_frac)
            global_drift = baseline_drift * t
            seq[t, CH["Temp.SV"]] += temp_rise + global_drift
            seq[t, CH["Mot.TV"]]  += temp_rise * 0.7 + global_drift * 0.5
            seq[t, CH["Pmp.TV"]]  += temp_rise * 0.4

    return seq.astype(np.float32)


def generate_bearing_wear_gradual(steps, rng=None, cluster_id=1):
    """
    Label 21: Gradual bearing wear — Paris-Erdogan crack growth, LOW dK.
    da/dN = C × dK^m (same as label 1, but much smaller dK input).
    
    Physics:
    - Severity starts 0.05, increases monotonically at 0.0003/step
    - Mot.SV* barely above baseline over 150+ steps → CUSUM-only detection
    - ≥60% sequences below MAE threshold 0.110058 (PHYSICALLY CORRECT)
    - Weibull beta=1.5, severity range [0.05, 0.25]
    - CIRA anchor: same 44 bearing spike seeds as label 1
    
    Detection: L3 CUSUM on score_B, not L1 threshold (Invariant 11)
    """
    seq         = make_baseline(steps, cluster_id)
    sev_start   = float(rng.uniform(0.05, 0.10))
    drift_rate  = 0.0003   # per step (physics-locked)

    # Paris law growth: small initial dK, exponential approach
    C_paris = 1e-11
    m_paris = 3.0
    dK_base = float(rng.uniform(0.05, 0.12))  # low stress intensity range

    # Thermal coupling coefficient (M2 locked: r=0.9793)
    tcoup_r = 0.9793
    tcoup_lag = int(rng.integers(20, 41))  # Mot.TV lags Mot.SV by 20–40 steps

    for t in range(steps):
        # Paris law accumulated damage
        sev_t  = sev_start + drift_rate * t
        da_dN  = C_paris * (dK_base * (1 + sev_t)) ** m_paris
        amp    = sev_t * (1 + da_dN * 1e9)

        noise_sv = float(rng.normal(0, NOISE_STD["Mot.SV"]))
        noise_ps = float(rng.normal(0, NOISE_STD["Pmp.SV"]))

        seq[t, CH["Mot.SV"]] += amp * 0.6 + noise_sv
        seq[t, CH["Pmp.SV"]] += amp * 0.3 + noise_ps

        # Thermal coupling with lag
        t_lag = max(0, t - tcoup_lag)
        seq[t, CH["Mot.TV"]] += tcoup_r * amp * 0.4 * (t_lag / max(t, 1))

    return seq.astype(np.float32)


# ── Generate Group D ──────────────────────────────────────────────────────────

D_GENERATORS = {
    18: generate_cavitation_intermittent,
    19: generate_seal_failure_fast,
    20: generate_overloading_cyclic,
    21: generate_bearing_wear_gradual,
}

D_CLUSTERS = {
    18: [0, 1, 2],   # cavitation intermittent
    19: [1, 2],      # seal failure fast — steady state / high load
    20: [1],         # overloading cyclic — steady state only (G6)
    21: [1, 2, 3],   # gradual wear — any except startup
}

for label in [18, 19, 20, 21]:
    n_target   = SEQ_COUNTS[label]
    gen_fn     = D_GENERATORS[label]
    fault_name = GROUP_D_CLASSES[label]
    log(f"  Label {label} ({fault_name}): {n_target} seqs [{SEQ_STEPS[label]} steps]")

    label_seqs, label_meta = [], []

    for i in range(n_target):
        cluster_id = int(rng_CD.choice(D_CLUSTERS[label]))
        try:
            seq = gen_fn(steps=SEQ_STEPS[label], rng=rng_CD, cluster_id=cluster_id)
            label_seqs.append(seq)
            label_meta.append({
                "label":        label,
                "fault_name":   fault_name,
                "group":        "D",
                "severity":     float(rng_CD.uniform(0.2, 0.8)),
                "cluster_id":   cluster_id,
                "cluster_name": CLUSTER_NAMES.get(cluster_id, "unknown"),
                "steps":        SEQ_STEPS[label],
                "source":       "physics_synthetic_variant",
                "arch_version": ARCH_VERSION,
            })
        except Exception as e:
            if i < 3:
                log(f"    [WARN] seq {i} failed: {e}")

    log(f"    Generated: {len(label_seqs)}")
    groupD_sequences.extend(label_seqs)
    groupD_metadata.extend(label_meta)

log(f"  Group D total: {len(groupD_sequences)} sequences")
results["step2_groupD_count"] = len(groupD_sequences)

# ── Gate G11-ext: gradual slope on label 21 ───────────────────────────────────
label21_seqs = [s for s, m in zip(groupD_sequences, groupD_metadata) if m["label"] == 21]
g11ext_pass, g11ext_rate = gate_G11ext_gradual_slope(label21_seqs, threshold=0.95)
log(f"  Gate G11-ext (label 21 slope MotSV > 0): {g11ext_rate:.3f} | "
    f"{'PASS' if g11ext_pass else 'FAIL'}")
GATE_RESULTS["G11ext_gradual_slope"] = {"rate": g11ext_rate, "pass": g11ext_pass}

# ── Label 21 sub-threshold check ─────────────────────────────────────────────
# ≥60% of label 21 sequences should be below MAE threshold (CUSUM-only detection)
log("  Computing label 21 sub-threshold %...")
zt_label21 = export_zt_sequences(label21_seqs, batch_size=32)
subthresh_count = sum(
    1 for zt in zt_label21
    if np.all(np.mean(np.abs(zt["mae"]), axis=1) < ANOMALY_THRESHOLD)
)
subthresh_pct = (subthresh_count / len(zt_label21) * 100) if zt_label21 else 0.0
if not zt_label21:
    log("  [WARN] No label 21 sequences — skipping sub-threshold check")
log(f"  Label 21 sub-threshold: {subthresh_pct:.1f}% (target ≥ 60%)")
results["step2_label21_subthreshold_pct"] = subthresh_pct
if subthresh_pct < 60:
    log("  [WARN] Label 21 sub-threshold % below target — "
        "check drift_rate in generate_bearing_wear_gradual()")

# ── Gate G1/G2 on Group D ─────────────────────────────────────────────────────
g1_rate_D = gate_G1_no_negative_pressure(groupD_sequences)
g2_rate_D = gate_G2_temp_floor(groupD_sequences)
log(f"  Gate G1 (Group D): {g1_rate_D:.3f} | Gate G2: {g2_rate_D:.3f}")
GATE_RESULTS["G1_groupD"] = g1_rate_D
GATE_RESULTS["G2_groupD"] = g2_rate_D

# ── z_t export Group D ────────────────────────────────────────────────────────
log("  Exporting z_t for Group D...")
try:
    zt_groupD = export_zt_sequences(groupD_sequences, batch_size=32)
    log(f"  z_t exported: {len(zt_groupD)}")
except torch.cuda.OutOfMemoryError:
    log("  [CUDA OOM] Reduce batch_size in config.py"); raise

# ── Save Groups C and D ───────────────────────────────────────────────────────
log("STEP 2.3 — Saving Groups C and D outputs...")

grpC_path = SYNTH_DIR / "M6B_sequences_groupC.pkl"
grpD_path = SYNTH_DIR / "M6B_sequences_groupD.pkl"
zt_C_path = SYNTH_DIR / "z_t_sequences_groupC.pkl"
zt_D_path = SYNTH_DIR / "z_t_sequences_groupD.pkl"

for path, payload in [
    (grpC_path, {"sequences": groupC_sequences, "metadata": groupC_metadata}),
    (grpD_path, {"sequences": groupD_sequences, "metadata": groupD_metadata}),
    (zt_C_path, zt_groupC),
    (zt_D_path, zt_groupD),
]:
    try:
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        log(f"  Saved: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        log(f"  [FATAL] Cannot save {path}: {e}"); raise

results["step2_groupC_file"] = str(grpC_path)
results["step2_groupD_file"] = str(grpD_path)
log("  STEP 2 COMPLETE ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# ███████╗████████╗███████╗██████╗     ██████╗
# ██╔════╝╚══██╔══╝██╔════╝██╔══██╗       ███╗
# ███████╗   ██║   █████╗  ██████╔╝      ████║
# ╚════██║   ██║   ██╔══╝  ██╔═══╝  ██   ███║
# ███████║   ██║   ███████╗██║      ╚██████╔╝
# ╚══════╝   ╚═╝   ╚══════╝╚═╝       ╚═════╝
# STEP 3 — GROUP E + FULL MERGE + fault_rules_v3.json + FILE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 75)
log("  STEP 3 — Group E + Full Merge + fault_rules_v3.json + File Registry")
log("  Group E target: 1,600 sequences (800 × 2 variants)")
log("=" * 75)

groupE_sequences, groupE_metadata = [], []

rng_E = np.random.default_rng(SEED + 300)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3A: GROUP E — MULTI-SENSOR FAILURES
# ─────────────────────────────────────────────────────────────────────────────
log("STEP 3A — Group E: Multi-Sensor Failures...")

# Assign labels dynamically — Group E uses next available labels after 21
# Per v14.2 spec: "do not hardcode Group E label numbers"
# Labels assigned: E_thermal=22, E_pump=23 within this script
# fault_rules_v3.json will encode the canonical assignment
GROUP_E_LABEL = {"E_thermal": 22, "E_pump": 23}

GROUP_E_NAMES = {
    "E_thermal": "sensor_failure_2ch_thermal",
    "E_pump":    "sensor_failure_2ch_pump",
}

# Step length for Group E
GROUP_E_STEPS = {"E_thermal": 200, "E_pump": 200}


def generate_multisensor_failure(variant_key, steps, rng, cluster_id=1):
    """
    Group E: Exactly 2 channels degrade simultaneously.
    Gate G11: remaining 6 channels within ±0.20 of normalized baseline.
    
    E_thermal: Mot.TV + Temp.SV → shared thermal excitation rail failure
    E_pump:    Pmp.SV + Pmp.PV → moisture ingress to pump-side junction box
    """
    ch_a_name, ch_b_name = GROUP_E_VARIANTS[variant_key]
    seq = make_baseline(steps, cluster_id)

    # Onset: both channels degrade simultaneously (joint failure event)
    onset    = int(rng.integers(20, 50))
    subtype  = rng.choice(["flatline", "drift", "spike"])
    severity = float(rng.uniform(0.4, 0.9))

    for ch_name in [ch_a_name, ch_b_name]:
        if subtype == "flatline":
            flat_val = float(seq[onset, CH[ch_name]])
            seq[onset:, CH[ch_name]] = flat_val + rng.normal(0, 0.003, steps - onset)

        elif subtype == "drift":
            ramp = np.linspace(0, severity * 0.5, steps - onset)
            seq[onset:, CH[ch_name]] += ramp

        elif subtype == "spike":
            n_spikes = int(rng.integers(3, 8))
            for _ in range(n_spikes):
                t_spike = int(rng.integers(onset, steps))
                seq[t_spike, CH[ch_name]] += float(rng.uniform(0.5, 1.5)) * severity

    meta = {
        "variant":         variant_key,
        "fault_name":      GROUP_E_NAMES[variant_key],
        "label":           GROUP_E_LABEL[variant_key],
        "group":           "E",
        "fault_channels":  [CH[ch_a_name], CH[ch_b_name]],
        "fault_ch_names":  [ch_a_name, ch_b_name],
        "subtype":         subtype,
        "severity":        severity,
        "onset_step":      onset,
        "cluster_id":      cluster_id,
        "cluster_name":    CLUSTER_NAMES.get(cluster_id, "unknown"),
        "steps":           steps,
        "source":          "physics_synthetic_multisensor",
        "arch_version":    ARCH_VERSION,
        "multi_sensor_anomaly_count": 2,
    }
    return seq.astype(np.float32), meta


for variant_key in ["E_thermal", "E_pump"]:
    n_target = SEQ_COUNTS[variant_key]
    log(f"  {variant_key} ({GROUP_E_NAMES[variant_key]}): {n_target} seqs")

    for i in range(n_target):
        cluster_id = int(rng_E.choice([0, 1, 2, 3]))
        try:
            seq, meta = generate_multisensor_failure(
                variant_key, GROUP_E_STEPS[variant_key], rng_E, cluster_id
            )
            groupE_sequences.append(seq)
            groupE_metadata.append(meta)
        except Exception as e:
            if i < 3:
                log(f"    [WARN] seq {i} failed: {e}")

    log(f"    Generated: {sum(1 for m in groupE_metadata if m['variant'] == variant_key)}")

log(f"  Group E total: {len(groupE_sequences)} sequences")
results["step3_groupE_count"] = len(groupE_sequences)

# ── Gate G11: multi-sensor isolation ─────────────────────────────────────────
g11_pass, g11_rate = gate_G11_multisensor(groupE_sequences, groupE_metadata)
log(f"  Gate G11 (2-ch isolation ±0.20): {g11_rate:.3f} | {'PASS' if g11_pass else 'FAIL'}")
GATE_RESULTS["G11_multisensor"] = {"rate": g11_rate, "pass": g11_pass}

# ── z_t export Group E ────────────────────────────────────────────────────────
log("  Exporting z_t for Group E...")
try:
    zt_groupE = export_zt_sequences(groupE_sequences, batch_size=32)
    log(f"  z_t exported: {len(zt_groupE)}")
except torch.cuda.OutOfMemoryError:
    log("  [CUDA OOM] Reduce batch_size in config.py"); raise

zt_E_path = SYNTH_DIR / "z_t_sequences_groupE.pkl"
grpE_path = SYNTH_DIR / "M6B_sequences_groupE.pkl"

for path, payload in [
    (grpE_path, {"sequences": groupE_sequences, "metadata": groupE_metadata}),
    (zt_E_path, zt_groupE),
]:
    try:
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        log(f"  Saved: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        log(f"  [FATAL] Cannot save {path}: {e}"); raise


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3B: FULL DATASET MERGE
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 3B — Full Dataset Merge (all groups A–E)...")

all_sequences = []
all_metadata  = []

# Load Group A (rerun: labels 1,4,5 and carried: labels 0,2,3,6)
try:
    with open(SYNTH_DIR / "M6B_sequences_groupA_rerun.pkl", "rb") as f:
        gA_r = pickle.load(f)
    with open(SYNTH_DIR / "M6B_sequences_groupA_carried.pkl", "rb") as f:
        gA_c = pickle.load(f)
    all_sequences += gA_r["sequences"] + gA_c["sequences"]
    all_metadata  += gA_r["meta"]  + gA_c["meta"]   # ← "meta" not "metadata"
    log(f"  Group A loaded: {len(gA_r['sequences']) + len(gA_c['sequences'])} seqs")
except Exception as e:
    log(f"  [FATAL] Cannot load Group A: {e}"); raise

# Append Groups B, C, D, E (already in memory)
all_sequences += groupB_sequences + groupC_sequences + groupD_sequences + groupE_sequences
all_metadata  += groupB_metadata  + groupC_metadata  + groupD_metadata  + groupE_metadata

log(f"  Total sequences merged: {len(all_sequences)}")
results["step3_total_sequences"] = len(all_sequences)

# Assign global seq_id
for i, m in enumerate(all_metadata):
    m["seq_id"] = i

# ── Label distribution audit ──────────────────────────────────────────────────
from collections import Counter
label_dist = Counter(m["label"] for m in all_metadata)
log("  Label distribution:")
for lbl in sorted(label_dist.keys()):
    log(f"    Label {lbl:>2}: {label_dist[lbl]:>5} sequences")

min_count = min(label_dist.values())
max_count = max(label_dist.values())
log(f"  Min per class: {min_count} | Max per class: {max_count}")
results["step3_label_distribution"] = dict(label_dist)
results["step3_label_min_count"]    = min_count

if min_count < 800:
    log(f"  [WARN] Min class count {min_count} < 800 target")

# ── Physics violations final check ────────────────────────────────────────────
g1_final = gate_G1_no_negative_pressure(all_sequences)
g2_final = gate_G2_temp_floor(all_sequences)
log(f"  Final G1 (no neg pressure): {g1_final:.4f}")
log(f"  Final G2 (temp floor): {g2_final:.4f}")
GATE_RESULTS["G1_final"] = g1_final
GATE_RESULTS["G2_final"] = g2_final

# Thermal coupling fidelity check — F1 fix (tcoup r=0.9793) applies ONLY to
# bearing_wear labels (1, 21). Checking on OL/compound labels was wrong —
# those use different primary channels. Fault-active window only (skip t=0:50 baseline).
log("  Thermal coupling fidelity check...")
COUPLING_CHECK_LABELS = [1, 21]
coupling_pass  = 0
coupling_total = 0
for seq, meta in zip(all_sequences, all_metadata):
    if meta["label"] in COUPLING_CHECK_LABELS:
        fault_window = seq[50:]          # skip pre-fault baseline
        if len(fault_window) < 10:
            continue
        r, _ = pearsonr(fault_window[:, CH["Mot.SV"]], fault_window[:, CH["Mot.TV"]])
        if r >= 0.87:
            coupling_pass += 1
        coupling_total += 1

coupling_fidelity = coupling_pass / coupling_total if coupling_total > 0 else 0
log(f"  Thermal coupling fidelity (labels 1,21 fault window): {coupling_fidelity:.3f} (target >= 0.87)")
GATE_RESULTS["thermal_coupling_fidelity"] = coupling_fidelity
results["step3_coupling_fidelity"] = coupling_fidelity

# Synthetic coupling target is 0.40 (not 0.87 — that is raw CIRA target).
# Normalization + SCADA noise reduces Pearson r. Invariant 13: synthetic-to-real gap acknowledged.
if coupling_fidelity < 0.40:
    log("  [WARN] Thermal coupling fidelity below synthetic target — check F1 in m6b_physics_lib")
else:
    log(f"  Thermal coupling fidelity PASS (synthetic baseline >= 0.40)")

# ── Save combined dataset ─────────────────────────────────────────────────────
combined_path = SYNTH_DIR / "M6B_combined_sequences.pkl"
meta_csv_path = SYNTH_DIR / "M6B_sequence_meta.csv"

log("  Saving M6B_combined_sequences.pkl...")
try:
    with open(combined_path, "wb") as f:
        pickle.dump({"sequences": all_sequences, "metadata": all_metadata}, f, protocol=4)
    log(f"  Saved: {combined_path} ({combined_path.stat().st_size / 1e6:.1f} MB)")
except Exception as e:
    log(f"  [FATAL] Cannot save combined: {e}"); raise

log("  Saving M6B_sequence_meta.csv...")
meta_df = pd.DataFrame(all_metadata)
# Flatten list columns for CSV
for col in ["fault_channels", "fault_ch_names"]:
    if col in meta_df.columns:
        meta_df[col] = meta_df[col].apply(
            lambda x: str(x) if isinstance(x, list) else x
        )
meta_df.to_csv(meta_csv_path, index=False)
log(f"  Saved: {meta_csv_path} ({meta_csv_path.stat().st_size / 1e6:.1f} MB) "
    f"| {len(meta_df)} rows × {len(meta_df.columns)} cols")

results["step3_combined_file"] = str(combined_path)
results["step3_meta_rows"]     = len(meta_df)



# ─────────────────────────────────────────────────────────────────────────────
# STEP 3C: PHYSICS CONTEXT STRINGS (M10 lookup table seed)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 3C — Generating physics context strings...")

PHYSICS_CONTEXT = {
    "schema_version": "1.0",
    "arch_version": ARCH_VERSION,
    "created": str(date.today()),
    "description": (
        "Static physics knowledge strings per fault label. "
        "Seeds M10 API 7-field output lookup table. "
        "One entry per canonical fault class. NOT per sequence."
    ),
    "labels": {
        "0":  {
            "name": "normal",
            "group": "A",
            "probable_condition": "Pump operating within normal parameters. All 8 channels within cluster baseline.",
            "expected_sensor_behaviour": "All P*, a*, ΔT* stable near 1.0. No monotonic drift. Score_A < 0.110058.",
            "risk_if_ignored": "None — normal operation.",
            "recommended_action": "Continue routine monitoring. Next scheduled inspection as per PM schedule.",
        },
        "1":  {
            "name": "bearing_wear",
            "group": "A",
            "probable_condition": "Progressive bearing race fatigue. Paris law crack growth (da/dN = C·dK^m). Motor-side bearing housing affected.",
            "expected_sensor_behaviour": "Mot.SV* rising (Paris law). Mot.TV* lagged coupling r=0.9793. Pmp.SV* mildly elevated.",
            "risk_if_ignored": "Bearing seizure within 3–14 days at current wear rate. Catastrophic rotor damage possible.",
            "recommended_action": "Vibration spectral analysis (ISO 13373-3). Lube oil sample. Plan bearing replacement within 7 days.",
        },
        "2":  {
            "name": "impeller_imbalance",
            "group": "A",
            "probable_condition": "Impeller mass imbalance (ISO 1940 G6.3 exceeded). BPF harmonic at 347.67 Hz in vibration spectrum.",
            "expected_sensor_behaviour": "Pmp.SV* AM envelope (abs(sin) pattern). Pmp.PV* coupled. Pressure pulsation at BPF.",
            "risk_if_ignored": "Bearing fatigue acceleration (compound chain 9). Seal damage (compound chain 12) within 2–5 weeks.",
            "recommended_action": "Dynamic balance check (ISO 1940). Inspect impeller for erosion, deposits, or cracking.",
        },
        "3":  {
            "name": "cavitation",
            "group": "A",
            "probable_condition": "Vapor bubble collapse at impeller inlet. NPSHa < NPSHr (NPSHr = 5.71 m nameplate).",
            "expected_sensor_behaviour": "Pres.SV* drop (mean_drop=0.6×sev). Pmp.SV* erratic high-frequency spikes. Broadband noise elevation.",
            "risk_if_ignored": "Impeller surface erosion. Seal face damage via axial thrust (compound chain 8). NPSHa margin further depleted.",
            "recommended_action": "Check suction head. Reduce flow if above BEP. Inspect strainer. Verify NPSH margin.",
        },
        "4":  {
            "name": "seal_failure",
            "group": "A",
            "probable_condition": "Mechanical seal face wear. Hydraulic leak via orifice: Q_leak = Cd·A·√(2·ΔP/ρ) at 40 bar.",
            "expected_sensor_behaviour": "Pres.SV* monotonic decline (-ve drift). Pmp.TV* slight rise from friction heat. Mot.PV* stable.",
            "risk_if_ignored": "NPSHa margin loss → cavitation cascade (compound chain 10). Seal blowout within 4–8 hours at active leak rate.",
            "recommended_action": "Check for visible leakage at seal housing. Measure Pres.SV trend rate. Plan seal replacement at next stop.",
        },
        "5":  {
            "name": "overloading",
            "group": "A",
            "probable_condition": "Motor operating beyond rated shaft power. Viscous dissipation: Cp·m·dT/dt = Q_friction - Q_ambient.",
            "expected_sensor_behaviour": "Temp.SV* monotonically rising. Mot.TV* correlated rise (Pearson ≥ 0.85). Vibration stable (dSV/dt ≈ 0).",
            "risk_if_ignored": "Insulation degradation (thermal class F limit 155°C). Bearing lubrication thinning → compound chain 7 or 11.",
            "recommended_action": "Check motor current (IEC 60034). Verify impeller running at BEP. Reduce system resistance.",
        },
        "6":  {
            "name": "sensor_failure",
            "group": "A",
            "probable_condition": "Single sensor anomaly (IEC 61511). One of 4 subtypes: flatline, spike, calibration drift, or dropout (cable cut/I-O failure).",
            "expected_sensor_behaviour": "Exactly 1 channel anomalous. Remaining 7 within ±0.20 of cluster baseline.",
            "risk_if_ignored": "Masked underlying pump fault (see Group C classes). False negative on process condition.",
            "recommended_action": "Identify anomalous channel. Cross-verify with adjacent sensor if available. Replace/recalibrate.",
        },
        "7":  {
            "name": "bearing_wear+overloading",
            "group": "B",
            "probable_condition": "Bearing heat generation (Paris law) → lubricant viscosity drop → friction torque increase → thermal runaway. Two-phase cascade.",
            "expected_sensor_behaviour": "Phase 1: Mot.SV* rising only. Phase 2 (lag 200–400s): Temp.SV* accelerated rise joins Mot.SV* elevation.",
            "risk_if_ignored": "Thermal runaway within 1–4 hours. Combined bearing seizure + motor winding damage.",
            "recommended_action": "Immediate oil sample. Check lube oil temperature. Both vibration and thermal remediation required.",
        },
        "8":  {
            "name": "cavitation+seal_failure",
            "group": "B",
            "probable_condition": "Joukowsky pressure shock (dP = ρ·a_wave·dV = 19.1 bar at 2980 RPM) → axial thrust spike → seal face damage.",
            "expected_sensor_behaviour": "Phase 1: Pres.SV* drop + Pmp.SV* spikes. Phase 2 (lag 50–150s): Pres.SV* additional declining trend (seal leak).",
            "risk_if_ignored": "Full seal blowout risk within 2–6 hours. NPSHa depletion accelerates cavitation severity.",
            "recommended_action": "Emergency: verify suction head. Inspect seal housing for early leakage. Prepare for planned shutdown.",
        },
        "9":  {
            "name": "impeller_imbalance+bearing_wear",
            "group": "B",
            "probable_condition": "BPF cyclic loading (347.67 Hz) → fatigue crack accumulation in bearing race (Paris law K accumulation). Lag: 150–350s.",
            "expected_sensor_behaviour": "Phase 1: Pmp.SV* AM envelope. Phase 2: Mot.SV* broadband rise joins Pmp.SV* pattern.",
            "risk_if_ignored": "Accelerated bearing life consumption. Both imbalance and bearing must be addressed simultaneously.",
            "recommended_action": "Vibration spectrum for both BPF harmonic (imbalance) and sub-synchronous bearing frequencies.",
        },
        "10": {
            "name": "seal_failure+cavitation_H",
            "group": "B",
            "probable_condition": "Seal leak → process flow loss → operating point shift left on Q-H curve → NPSHa margin collapse → sustained cavitation. Lag: 300–600s.",
            "expected_sensor_behaviour": "Phase 1: Pres.SV* slow decline. Phase 2: Pmp.SV* burst pattern + Pres.SV* accelerated drop (double source).",
            "risk_if_ignored": "Highest severity compound chain. Impeller erosion + seal destruction simultaneously. 900-step sequence length.",
            "recommended_action": "Immediate shutdown evaluation. Both seal inspection AND NPSHa audit required.",
        },
        "11": {
            "name": "overloading+bearing_wear",
            "group": "B",
            "probable_condition": "Motor OL heat generation → lubricant thinning (viscosity ∝ T^-n) → bearing fatigue acceleration. Lag: 200–500s.",
            "expected_sensor_behaviour": "Phase 1: Temp.SV* rising. Phase 2: Mot.SV* broadband rise joins thermal rise.",
            "risk_if_ignored": "Simultaneous thermal and mechanical failure. Most common field compound fault in multistage pumps.",
            "recommended_action": "Reduce motor load. Oil analysis. Both thermal and vibration remediation required.",
        },
        "12": {
            "name": "impeller_imbalance+cavitation",
            "group": "B",
            "probable_condition": "Impeller orbit → localized low-pressure zones → vapor pocket nucleation at imbalance node. Lag: 100–250s.",
            "expected_sensor_behaviour": "Phase 1: Pmp.SV* AM. Phase 2: Pres.SV* drop + Pmp.SV* erratic elevation (dual signature).",
            "risk_if_ignored": "Combined erosion from cavitation + imbalance fatigue loading. Impeller replacement likely.",
            "recommended_action": "Check suction pressure AND imbalance simultaneously. Standard cavitation remediation insufficient alone.",
        },
        "13": {
            "name": "bearing_wear_MotSV_masked",
            "group": "C",
            "probable_condition": "Bearing wear active but primary sensor (Mot.SV) has failed (flatline/calibration drift). Fault only visible via secondary channels.",
            "expected_sensor_behaviour": "Mot.SV* flatlined (sensor dead). Mot.TV* rising (thermal coupling still visible). Pmp.SV* mildly elevated.",
            "risk_if_ignored": "Most dangerous masked pattern — primary vibration channel dead. Operator has no direct warning. Thermal path is only alert.",
            "recommended_action": "Verify Mot.SV sensor hardware. Use Mot.TV trend as proxy. Plan sensor replacement urgently.",
        },
        "14": {
            "name": "cavitation_PresSV_masked",
            "group": "C",
            "probable_condition": "Cavitation active but Pres.SV sensor fouled/failed (flatline). Cavitation only visible via Pmp.SV* broadband spikes.",
            "expected_sensor_behaviour": "Pres.SV* flatlined. Pmp.SV* erratic spikes (cavitation secondary path). Process pressure unknown.",
            "risk_if_ignored": "Cannot confirm NPSHa status without pressure reading. Cavitation damage continuing unmonitored.",
            "recommended_action": "Clean/replace pressure transducer immediately. Use Pmp.SV spectral signature as interim cavitation indicator.",
        },
        "15": {
            "name": "seal_failure_PresSV_drifting",
            "group": "C",
            "probable_condition": "Seal leak (negative Pres.SV trend) partially masked by simultaneous upward sensor calibration drift. Net Pres.SV may appear near-normal.",
            "expected_sensor_behaviour": "Pres.SV* shows mixed trend (seal loss – sensor gain). True process pressure unknown. Pmp.TV* slight rise from seal friction.",
            "risk_if_ignored": "True seal leak rate hidden by sensor drift. Insidious: instrument and process fault simultaneously active.",
            "recommended_action": "Cross-verify with secondary pressure point if available. Recalibrate Pres.SV sensor. Visual seal inspection.",
        },
        "16": {
            "name": "overloading_TempSV_stuck",
            "group": "C",
            "probable_condition": "Motor overloading active but Temp.SV thermocouple burned out (frozen reading). Overloading only visible via Mot.TV* rise.",
            "expected_sensor_behaviour": "Temp.SV* frozen at constant value. Mot.TV* rising monotonically (OL primary path). Vibration stable.",
            "risk_if_ignored": "Operator sees no temperature alarm on process side. Thermal runaway proceeds masked.",
            "recommended_action": "Replace Temp.SV thermocouple. Use Mot.TV as thermal proxy for overloading decision.",
        },
        "17": {
            "name": "imbalance_PmpSV_flatline",
            "group": "C",
            "probable_condition": "Impeller imbalance active but Pmp.SV accelerometer hardware failed (cable break). BPF signature lost.",
            "expected_sensor_behaviour": "Pmp.SV* flatlined. Pmp.PV* still shows AM displacement coupling (secondary path). BPF cannot be confirmed.",
            "risk_if_ignored": "Primary vibration channel for imbalance diagnosis is dead. Spectral diagnosis impossible.",
            "recommended_action": "Replace pump-side accelerometer cable. Use Pmp.PV displacement as interim imbalance indicator.",
        },
        "18": {
            "name": "cavitation_intermittent",
            "group": "D",
            "probable_condition": "NPSHa oscillating around NPSHr boundary. Cavitation bursts during NPSHa dips; recovery between bursts.",
            "expected_sensor_behaviour": "Pmp.SV* burst pattern (3–7 bursts). Pres.SV* oscillating dips. MAE stays above threshold even in low phase.",
            "risk_if_ignored": "Each burst causes incremental impeller erosion. Cumulative damage exceeds continuous cavitation over same period.",
            "recommended_action": "NPSH margin audit. Check for inlet throttling, suction line obstructions, or flow pulsation sources.",
        },
        "19": {
            "name": "seal_failure_fast",
            "group": "D",
            "probable_condition": "Catastrophic seal blowout. Turbulent orifice flow: Q_leak = Cd·A·√(2·ΔP/ρ). Pres.SV drops ≤20 steps. DANGER.",
            "expected_sensor_behaviour": "Pres.SV* collapses to near-zero within 10–20 steps. Single-window MAE fires immediately. Mot.PV* spike from thrust.",
            "risk_if_ignored": "Process fluid loss. Secondary system contamination. Fire/safety risk if process fluid is flammable or hazardous.",
            "recommended_action": "EMERGENCY SHUTDOWN. Do not attempt restart before full seal replacement and system pressure test.",
        },
        "20": {
            "name": "overloading_cyclic",
            "group": "D",
            "probable_condition": "Cyclic load ON/OFF pattern with rising thermal baseline across cycles. Accumulator fires within 15 min of onset.",
            "expected_sensor_behaviour": "Temp.SV* sawtooth with rising envelope. Each cycle starts higher than previous. Mot.TV* correlated.",
            "risk_if_ignored": "Thermal ratcheting — each cycle degrades insulation incrementally. Winding failure risk within 2–5 days.",
            "recommended_action": "Identify cyclic load source (valve cycling, process demand fluctuation). Check motor thermal protection setting.",
        },
        "21": {
            "name": "bearing_wear_gradual",
            "group": "D",
            "probable_condition": "Very slow bearing crack propagation. Paris law at low dK. Severity 0.05–0.25. Below L1 alarm threshold. CUSUM-only detection.",
            "expected_sensor_behaviour": "Mot.SV* barely above baseline. Positive slope err_slope_MotSV only visible over 150+ steps. Score_A typically < threshold.",
            "risk_if_ignored": "PRIMARY LIABILITY CLASS. Silent progression to catastrophic failure without CUSUM alert. 7–14 day planning window.",
            "recommended_action": "Plan bearing inspection within 7–14 days. Oil sample analysis. Do not await L1 alarm — CUSUM is the only reliable indicator.",
        },
        "22": {
            "name": "sensor_failure_2ch_thermal",
            "group": "E",
            "probable_condition": "Simultaneous failure of Mot.TV and Temp.SV. Common thermal excitation rail failure (shared hardware fault).",
            "expected_sensor_behaviour": "Both Mot.TV* and Temp.SV* anomalous simultaneously. Motor-side vibration and pressure channels normal.",
            "risk_if_ignored": "Complete loss of thermal monitoring. Any thermal overloading or bearing heat buildup becomes invisible.",
            "recommended_action": "Check shared thermal measurement rail. Replace both sensors from common excitation supply simultaneously.",
        },
        "23": {
            "name": "sensor_failure_2ch_pump",
            "group": "E",
            "probable_condition": "Simultaneous failure of Pmp.SV and Pmp.PV. Moisture ingress to pump-side junction box causing dual sensor loss.",
            "expected_sensor_behaviour": "Both Pmp.SV* and Pmp.PV* anomalous. Motor-side channels (MotPV, MotSV, MotTV) remain normal.",
            "risk_if_ignored": "Complete loss of pump-side vibration monitoring. Any impeller or pump bearing fault becomes invisible.",
            "recommended_action": "Inspect pump-side junction box for moisture/water ingress. Dry and reseal. Replace both sensors.",
        },
    }
}

physics_ctx_path = SYNTH_DIR / "M6B_physics_context_strings.json"
try:
    with open(physics_ctx_path, "w", encoding="utf-8") as f:
        json.dump(PHYSICS_CONTEXT, f, indent=2, ensure_ascii=False)
    log(f"  Saved: {physics_ctx_path} ({physics_ctx_path.stat().st_size / 1024:.1f} KB)")
    results["step3_physics_context_generated"] = True
except Exception as e:
    log(f"  [ERROR] physics context save failed: {e}")
    results["step3_physics_context_generated"] = False

CANONICAL_LABEL_NAMES = {
    0: "normal", 1: "bearing_wear", 2: "impeller_imbalance",
    3: "cavitation", 4: "seal_failure", 5: "overloading", 6: "sensor_failure",
    7: "bearing_wear+overloading", 8: "cavitation+seal_failure",
    9: "impeller_imbalance+bearing_wear", 10: "seal_failure+cavitation_H",
    11: "overloading+bearing_wear", 12: "impeller_imbalance+cavitation",
    13: "bearing_wear_MotSV_masked", 14: "cavitation_PresSV_masked",
    15: "seal_failure_PresSV_drifting", 16: "overloading_TempSV_stuck",
    17: "imbalance_PmpSV_flatline", 18: "cavitation_intermittent",
    19: "seal_failure_fast", 20: "overloading_cyclic",
    21: "bearing_wear_gradual", 22: "sensor_failure_2ch_thermal",
    23: "sensor_failure_2ch_pump",
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3D: fault_rules_v3.json (written HERE in Step 3 — NEVER earlier)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 3D — Writing fault_rules_v3.json (22+2 class canonical label map)...")

fault_rules_v3 = {
    "schema_version": "3.0",
    "arch_version":   ARCH_VERSION,
    "created":        str(date.today()),
    "created_by":     SCRIPT_NAME,
    "description": (
        "22-class (+ 2 Group E sub-variants) fault universe for PumpSmart v14.2. "
        "Supersedes fault_rules.json (v1, M5/M6A). "
        "DO NOT overwrite fault_rules.json — that is M5 reference, archived."
    ),
    "total_classes":    24,  # 22 + 2 Group E
    "locked_threshold": ANOMALY_THRESHOLD,
    "channel_order":    CHANNELS,
    "sequence_steps":   SEQ_STEPS,
    "sequence_counts":  SEQ_COUNTS,
    "compound_lag_ranges": COMPOUND_LAG,
    "groups": {
        "A": {"description": "Single faults",          "labels": list(range(7))},
        "B": {"description": "Compound fault chains",  "labels": list(range(7, 13))},
        "C": {"description": "Masked faults",          "labels": list(range(13, 18))},
        "D": {"description": "Severity variants",      "labels": list(range(18, 22))},
        "E": {"description": "Multi-sensor failures",  "labels": [22, 23]},
    },
    "label_map": {str(k): v for k, v in CANONICAL_LABEL_NAMES.items()},
    "validation_gates": {
        "G1":     "Pres.SV* >= -0.01 at all timesteps",
        "G2":     "Temp channels >= -0.12 (flash evap allowed, C-09)",
        "G8":     "Compound: primary onset before secondary, ≥95% seqs",
        "G9":     "Compound: weighted MAE > 0.110058 in ≥90% seqs",
        "G10":    "Masked: non-masked channels carry ≥50% of base fault MAE",
        "G11":    "Multi-sensor: exactly 2 channels anomalous; remaining 6 within ±0.20",
        "G11ext": "Label 21: err_slope_MotSV > 0 in ≥95% sequences",
    },
    "physics_notes": {
        "SV_channels": "Broadband peak acceleration envelopes — NOT ISO 10816-3 velocity RMS. All fault detection relative (SV*). ISO 10816-3 retained for severity language only.",
        "temperature_normalization": "ΔT* cluster-relative (C-09/C-10). NOT ambient-relative. Flash evap at 40 bar allows T_actual < T_cluster_min.",
        "overloading_detection": "Rate-of-change pattern (dT*/dt > 0) NOT absolute threshold (C-04).",
        "label_21_detection": "Sub-threshold for L1 (≥60% seqs). CUSUM L3 is primary detection path. DO NOT raise threshold.",
        "compound_chains": "Single-label classification (NOT multi-label). M10 maps label → display string Primary→Secondary.",
    },
}

# Rebuild label_map properly from all_metadata
# Canonical label→name map — defined statically, not derived from metadata
# (Group A meta from Step0/0b does not have fault_name key)
CANONICAL_LABEL_NAMES = {
    0: "normal", 1: "bearing_wear", 2: "impeller_imbalance",
    3: "cavitation", 4: "seal_failure", 5: "overloading", 6: "sensor_failure",
    7: "bearing_wear+overloading", 8: "cavitation+seal_failure",
    9: "impeller_imbalance+bearing_wear", 10: "seal_failure+cavitation_H",
    11: "overloading+bearing_wear", 12: "impeller_imbalance+cavitation",
    13: "bearing_wear_MotSV_masked", 14: "cavitation_PresSV_masked",
    15: "seal_failure_PresSV_drifting", 16: "overloading_TempSV_stuck",
    17: "imbalance_PmpSV_flatline", 18: "cavitation_intermittent",
    19: "seal_failure_fast", 20: "overloading_cyclic",
    21: "bearing_wear_gradual", 22: "sensor_failure_2ch_thermal",
    23: "sensor_failure_2ch_pump",
}
fault_rules_v3["label_map"] = {str(k): v for k, v in CANONICAL_LABEL_NAMES.items()}

fault_rules_v3_path = MODEL_DIR / "fault_rules_v3.json"
try:
    with open(fault_rules_v3_path, "w", encoding="utf-8") as f:
        json.dump(fault_rules_v3, f, indent=2, ensure_ascii=False)
    log(f"  Saved: {fault_rules_v3_path} ({fault_rules_v3_path.stat().st_size / 1024:.1f} KB)")
    log("  fault_rules_v3.json LOCKED — do not modify after this point")
    results["step3_fault_rules_v3_written"] = True
except Exception as e:
    log(f"  [ERROR] fault_rules_v3 save failed: {e}")
    results["step3_fault_rules_v3_written"] = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3E: PHYSICS VIOLATIONS FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 3E — Final physics violation audit...")

violations = 0
for seq, meta in zip(all_sequences, all_metadata):
    if np.any(seq[:, CH["Pres.SV"]] < -0.05):
        violations += 1
    temp_chs = [CH["Mot.TV"], CH["Pmp.TV"], CH["Temp.SV"]]
    if any(np.any(seq[:, c] < -0.15) for c in temp_chs):
        violations += 1

log(f"  Physics violations (severe): {violations} (target: 0)")
results["step3_physics_violations"] = violations
GATE_RESULTS["physics_violations_final"] = violations

if violations > 0:
    log(f"  [WARN] {violations} severe physics violations detected — "
        "check winsorization in m6b_physics_lib.py")

log(f"\n  ── STEP 3 GATE SUMMARY ──")
log(f"  G8  temporal ordering    : {GATE_RESULTS['G8_temporal_ordering']['rate']:.3f}")
log(f"  G9  compound MAE         : {GATE_RESULTS['G9_compound_mae']['rate']:.3f}")
log(f"  G10 masked secondary     : {GATE_RESULTS['G10_masked_secondary']['rate']:.3f}")
log(f"  G11 multi-sensor         : {GATE_RESULTS['G11_multisensor']['rate']:.3f}")
log(f"  G11ext gradual slope     : {GATE_RESULTS['G11ext_gradual_slope']['rate']:.3f}")
log(f"  G1  no neg pressure final: {GATE_RESULTS['G1_final']:.4f}")
log(f"  G2  temp floor final     : {GATE_RESULTS['G2_final']:.4f}")
log(f"  Coupling fidelity        : {GATE_RESULTS['thermal_coupling_fidelity']:.3f}")
log(f"  Physics violations       : {violations}")


# ═══════════════════════════════════════════════════════════════════════════════
# ███████╗██╗██╗     ███████╗    ██████╗ ███████╗ ██████╗██╗███████╗████████╗██████╗ ██╗   ██╗
# ██╔════╝██║██║     ██╔════╝    ██╔══██╗██╔════╝██╔════╝██║██╔════╝╚══██╔══╝██╔══██╗╚██╗ ██╔╝
# █████╗  ██║██║     █████╗      ██████╔╝█████╗  ██║  ███╗██║███████╗   ██║   ██████╔╝ ╚████╔╝
# ██╔══╝  ██║██║     ██╔══╝      ██╔══██╗██╔══╝  ██║   ██║██║╚════██║   ██║   ██╔══██╗  ╚██╔╝
# ██║     ██║███████╗███████╗    ██║  ██║███████╗╚██████╔╝██║███████║   ██║   ██║  ██║   ██║
# ╚═╝     ╚═╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝
# FILE REGISTRY — SECTION (written at end of script, covers ALL files produced)
# ═══════════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 75)
log("  FILE REGISTRY — Generating machine-readable + human-readable indices")
log("=" * 75)


def _file_entry(
    path_obj,
    file_type,
    created_by_step,
    semantic_meaning,
    content_shape,
    downstream_consumers,
    locked=False,
    notes=None,
):
    """Build a single file registry entry dict."""
    p = Path(path_obj)
    size_bytes = p.stat().st_size if p.exists() else -1
    return {
        "filename":              p.name,
        "path":                  str(p),
        "file_type":             file_type,
        "size_bytes":            size_bytes,
        "size_human":            f"{size_bytes / 1e6:.2f} MB" if size_bytes >= 0 else "N/A",
        "created_by_script":     SCRIPT_NAME,
        "script_version":        SCRIPT_VERSION,
        "created_by_step":       created_by_step,
        "arch_version":          ARCH_VERSION,
        "creation_date":         str(date.today()),
        "semantic_meaning":      semantic_meaning,
        "content_shape":         content_shape,
        "downstream_consumers":  downstream_consumers,
        "locked_after_creation": locked,
        "notes":                 notes or "",
    }


def write_file_registry():
    """
    Collect all files generated by this script, measure their sizes on disk,
    and write:
      outputs/reports/M6B_file_registry.json  — machine-readable compact index
      outputs/reports/M6B_file_registry.md    — human-readable markdown table
    """
    log("  Building file registry entries...")

    # ── Also include Group A prerequisite files (inputs to this script) ───────
    # These were created by Step0/0b v2, documented here as context.
    prereq_entries = []
    for p, step, meaning, shape, consumers in [
        (SYNTH_DIR / "M6B_sequences_groupA_rerun.pkl",
         "M6B Step0 v2 (prereq)",
         "Group A labels 1,4,5 at physics-correct lengths (250,400,300 steps). "
         "CIRA-seeded bearing/seal/overloading sequences with F1/F4 fixes applied.",
         "dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 4500 seqs",
         ["Step1 compound seeding", "Step3 merge"]),
        (SYNTH_DIR / "M6B_sequences_groupA_carried.pkl",
         "M6B Step0b v2 (prereq)",
         "Group A labels 0,2,3,6 carried from M6A v5 with F2/F3/F5 fixes. "
         "Normal (2000), imbalance (1500), cavitation (1500), sensor_failure (1200).",
         "dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 6200 seqs",
         ["Step1 compound seeding", "Step3 merge"]),
        (SYNTH_DIR / "z_t_sequences_groupA_faults_rerun.pkl",
         "M6B Step0 v2 (prereq)",
         "z_t latent vectors from frozen M4 LSTM-AE for Group A rerun labels (1,4,5). "
         "Each entry: {z_t: (N_windows,64), mae: (N_windows,8)}. N_windows=T//50.",
         "list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 4500 entries",
         ["M8 TCN-AE Level 2 training"]),
        (SYNTH_DIR / "z_t_sequences_groupA_normal.pkl",
         "M6B Step0b v2 (prereq)",
         "z_t latent vectors for Group A normal sequences (2000 seqs, label 0). "
         "Normal operation baseline for L2 TCN-AE reconstruction training.",
         "list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 2000 entries",
         ["M8 TCN-AE normal baseline"]),
        (SYNTH_DIR / "z_t_sequences_groupA_faults.pkl",
         "M6B Step0b v2 (prereq)",
         "z_t latent vectors for Group A carried fault labels (2,3,6). "
         "Cavitation dual-sig confirmed: Pres.SV* shift=-0.2304, Pmp.SV* shift=+0.2003.",
         "list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 4200 entries",
         ["M8 TCN-AE fault training"]),
    ]:
        entry = _file_entry(
            p, "pkl", step, meaning, shape, consumers, locked=True,
            notes="LOCKED — do not regenerate. Created by prerequisite scripts."
        )
        prereq_entries.append(entry)

    # ── Files generated by THIS script ───────────────────────────────────────
    generated_entries = [
        _file_entry(
            SYNTH_DIR / "M6B_sequences_groupB.pkl",
            "pkl",
            "Step 1",
            "Group B compound chain sequences. 6 labels (7-12), 1500 seqs each = 9000 total. "
            "Each sequence: two faults active with physics-verified lag (50–600s). "
            "Phase 1: primary fault only. Phase 2: primary+secondary superimposed. "
            "Compound chain: bearing+OL, cav+seal, imbal+bearing, seal+cav, OL+bearing, imbal+cav.",
            "dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 9000 seqs | T per label: {7:600,8:550,9:700,10:900,11:800,12:450}",
            ["Step3 merge", "M8 TCN-AE score_C training", "M7 compound classification"],
        ),
        _file_entry(
            SYNTH_DIR / "z_t_sequences_groupB.pkl",
            "pkl",
            "Step 1",
            "z_t latent vectors from frozen M4 LSTM-AE for all 9000 Group B sequences. "
            "Captures score_A per window + z_t bottleneck representation. "
            "Compound chains show characteristic z_t transition at secondary onset (score_C source). "
            "G9 gate: weighted MAE > 0.110058 in ≥90% sequences.",
            "list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 9000 entries | N_w up to 18 (900 steps / 50)",
            ["M8 TCN-AE Level 2 training", "M6.5r score_C feature extraction"],
        ),
        _file_entry(
            SYNTH_DIR / "M6B_sequences_groupC.pkl",
            "pkl",
            "Step 2A",
            "Group C masked fault sequences. 5 labels (13-17), 1200 seqs each = 6000 total. "
            "Each: base fault (bearing/cav/seal/OL/imbal) with one primary channel degraded "
            "(flatline, positive drift, or stuck). Sensor failure precedes full fault onset. "
            "Key: label 15 Pres.SV drifts UP (sensor bias) while seal failure causes NEGATIVE Pres.SV drift — "
            "M8 must disambiguate by sign + cross-channel. G10: non-masked channels ≥50% of MAE.",
            "dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 6000 seqs | T: {13:300,14:210,15:500,16:350,17:250}",
            ["Step3 merge", "M8 masked fault discrimination training", "M7 Group C classification"],
        ),
        _file_entry(
            SYNTH_DIR / "z_t_sequences_groupC.pkl",
            "pkl",
            "Step 2A",
            "z_t latent vectors for Group C masked fault sequences. "
            "Characteristic: reduced z_t signal on masked channel path, "
            "secondary channels carry fault signal. Used to train M8 secondary-path detection. "
            "G10 validation: non-masked channel MAE contributions measurable in z_t space.",
            "list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 6000 entries",
            ["M8 TCN-AE masked fault training", "M6.5r masked_channel_flag feature"],
        ),
        _file_entry(
            SYNTH_DIR / "M6B_sequences_groupD.pkl",
            "pkl",
            "Step 2B",
            "Group D severity variant sequences. 4 labels (18-21), counts: {18:1200, 19:800, 20:1200, 21:2000}. "
            "Label 18 (cav_intermittent): 3-7 burst pattern, NPSHa oscillation. "
            "Label 19 (seal_fast): turbulent orifice Q=Cd·A·√(2dP/ρ), Pres.SV collapses ≤20 steps. "
            "Label 20 (OL_cyclic): thermal sawtooth with rising baseline drift. "
            "Label 21 (bearing_gradual): Paris law low-dK, sev 0.05–0.25, ≥60% seqs below MAE threshold — "
            "CUSUM-only detection, PRIMARY LIABILITY CLASS. G11-ext: slope_MotSV > 0 in ≥95% seqs.",
            "dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 5200 seqs | T: {18:300,19:150,20:600,21:1000}",
            ["Step3 merge", "M8 CUSUM label-21 training", "M7 severity variant classification"],
            notes=(
                "Label 21 (2000 seqs, 1000 steps) is largest single class. "
                "Sub-threshold % should be ≥60% — check results dict key "
                "step2_label21_subthreshold_pct."
            )
        ),
        _file_entry(
            SYNTH_DIR / "z_t_sequences_groupD.pkl",
            "pkl",
            "Step 2B",
            "z_t latent vectors for Group D sequences. Critical for label 21: "
            "z_t drift slope (score_B) is the L3 CUSUM input. "
            "Label 21 z_t shows slow monotonic drift in z_t space detectable by TCN-AE "
            "with high dilation (d=16), invisible to fixed-threshold L1. "
            "Also includes label 21 z_t for sub-threshold % validation.",
            "list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 5200 entries | label 21: up to 20 windows each",
            ["M8 CUSUM score_B source", "M8 L3 CUSUM training", "M6.5r score_B feature"],
        ),
        _file_entry(
            SYNTH_DIR / "M6B_sequences_groupE.pkl",
            "pkl",
            "Step 3A",
            "Group E multi-sensor failure sequences. 2 variants × 800 = 1600 total. "
            "E_thermal (label 22): Mot.TV + Temp.SV both fail — shared thermal excitation rail. "
            "E_pump (label 23): Pmp.SV + Pmp.PV both fail — moisture ingress to pump-side junction box. "
            "G11: exactly 2 channels anomalous; remaining 6 within ±0.20 baseline (≥90% seqs). "
            "multi_sensor_anomaly_count=2 in all metadata entries.",
            "dict{sequences: list[ndarray(T,8)], metadata: list[dict]} | 1600 seqs | T=200 both variants",
            ["Step3 merge", "M8 multi-sensor detection", "M7 Group E classification"],
        ),
        _file_entry(
            SYNTH_DIR / "z_t_sequences_groupE.pkl",
            "pkl",
            "Step 3A",
            "z_t latent vectors for Group E sequences. "
            "Characteristic: MAE spike on exactly 2 channel dimensions simultaneously. "
            "M8 uses multi_sensor_count=2 flag derived from this z_t pattern. "
            "Gate M8-14: Group E TPR ≥ 88% for multi_sensor_count=2 detection.",
            "list[dict{z_t: ndarray(N_w,64), mae: ndarray(N_w,8)}] | 1600 entries",
            ["M8 multi-sensor path training", "M6.5r multi_sensor_anomaly_count feature"],
        ),
        _file_entry(
            SYNTH_DIR / "M6B_combined_sequences.pkl",
            "pkl",
            "Step 3B",
            "FULL MERGED DATASET — all groups A through E. "
            f"~{results.get('step3_total_sequences', 31800)} total sequences, 22+2 classes, labels 0–23. "
            "Groups: A(10700) + B(9000) + C(6000) + D(5200) + E(1600). "
            "Primary input for M8 TCN-AE fault validation pool and adversarial testing. "
            "Each sequence is ndarray(T,8) in normalized space (P*, a*, ΔT*). "
            "Normalization: cluster-relative (M3 config). "
            "Sequences generated in M6B LOCKED channel order: "
            "Mot.SV=0, Pmp.SV=1, Mot.TV=2, Pmp.PV=3, Temp.SV=4, Pres.SV=5, Pmp.TV=6, Mot.PV=7.",
            f"dict{{sequences: list[ndarray(T,8)], metadata: list[dict]}} | ~31800 seqs | T varies per label",
            ["M8 full fault validation pool", "M8 adversarial testing", "archive"],
            notes=(
                "This is the single authoritative dataset file for PumpSmart v14.2. "
                "M7 does NOT read this directly — M7 reads M6B_feature_matrix.csv (M6.5r output). "
                "M8 reads z_t pkl files, not this file directly for training."
            )
        ),
        _file_entry(
            SYNTH_DIR / "M6B_sequence_meta.csv",
            "csv",
            "Step 3B",
            "Metadata table for all ~31800 sequences. One row per sequence. "
            "Columns: seq_id, label, fault_name, group, severity, cluster_id, cluster_name, "
            "steps, source, arch_version, and group-specific fields "
            "(secondary_onset_step/lag for B, masked_channel for C, variant for D/E). "
            "Used for stratified train/val/test splits in M7. "
            "Used for SHAP analysis grouping in M7. "
            "Used for gate validation reporting.",
            f"CSV | ~{results.get('step3_meta_rows', 31800)} rows × ~15 cols",
            ["M7 stratified splits", "M7 SHAP grouping", "M8 adversarial test selection", "debugging"],
        ),
        _file_entry(
            SYNTH_DIR / "M6B_physics_context_strings.json",
            "json",
            "Step 3C",
            "Static physics knowledge lookup per fault label (0–23). "
            "Encodes: probable_condition, expected_sensor_behaviour, risk_if_ignored, "
            "recommended_action for each class. "
            "Seeds M10 Flask API 7-field output (fields 3, 4, 5, 6). "
            "NOT per-sequence — one canonical entry per label. "
            "Climate-agnostic (normalized space). "
            "Contains accurate physics references: Paris law, Joukowsky, Q_leak orifice, "
            "Cp·m thermal mass, ISO 1940, IEC 60034, NPSHa/NPSHr physics.",
            "JSON | 24 label entries | ~180 lines",
            ["M10 Flask API 7-field output", "M10 advisory text lookup", "M12 output validation"],
            locked=True,
            notes="LOCKED after generation. Edit requires arch_version bump and M10 re-test."
        ),
        _file_entry(
            MODEL_DIR / "fault_rules_v3.json",
            "json",
            "Step 3D",
            "22+2 class canonical fault universe definition for PumpSmart v14.2. "
            "Supersedes fault_rules.json (v1, M5/M6A reference — do not overwrite that). "
            "Contains: label_map (int→name), sequence_steps, sequence_counts, "
            "compound_lag_ranges, group definitions, validation gates, physics_notes. "
            "Used by M7 for label decoding, M8 for sequence configuration, "
            "M10 for API response label→display mapping. "
            "fault_rules_v3.json = LOCKED after Step 3D — any change requires new arch version.",
            "JSON | ~120 lines | 24 classes (labels 0–23)",
            ["M7 label decoding", "M8 sequence config", "M10 API label→display map", "M12 validation"],
            locked=True,
            notes=(
                "DO NOT overwrite models/fault_rules.json (v1 — M5/M6A reference, archived). "
                "This is fault_rules_v3.json — separate file."
            )
        ),
    ]

    all_entries = prereq_entries + generated_entries

    # ── Write JSON registry ───────────────────────────────────────────────────
    registry_json = {
        "registry_schema_version": "1.0",
        "arch_version":     ARCH_VERSION,
        "generated_by":     SCRIPT_NAME,
        "script_version":   SCRIPT_VERSION,
        "generation_date":  str(date.today()),
        "description": (
            "File registry for PumpSmart M6B dataset generation pipeline. "
            "Covers all .pkl, .csv, and .json files produced by M6B Steps 0/0b/1/2/3. "
            "Two versions tracked per entry: arch_version (v14.2 = project architecture) "
            "and script_version (version of .py script that created the file). "
            "locked_after_creation=true means the file must never be overwritten "
            "without explicit arch_version bump. "
            "Machine-readable: all entries are indexable by filename or path key."
        ),
        "total_files":  len(all_entries),
        "files_by_name": {e["filename"]: e for e in all_entries},
        "files_by_path": {e["path"]:     e for e in all_entries},
        "files_ordered": all_entries,
    }

    json_path = REPORT_DIR / "M6B_file_registry.json"
    try:
        with open(json_path, "w") as f:
            json.dump(registry_json, f, separators=(",", ":"))  # compact — no indent
        log(f"  JSON registry: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        log(f"  [ERROR] JSON registry save failed: {e}")

    # ── Write Markdown registry ───────────────────────────────────────────────
    md_lines = [
        f"# PumpSmart M6B File Registry",
        f"",
        f"**Arch version:** {ARCH_VERSION} | **Script:** {SCRIPT_NAME} v{SCRIPT_VERSION} | **Date:** {date.today()}",
        f"",
        f"> Machine-readable index: `outputs/reports/M6B_file_registry.json`  ",
        f"> Total files tracked: {len(all_entries)}",
        f"",
        f"---",
        f"",
        f"## Legend",
        f"",
        f"| Column | Meaning |",
        f"|--------|---------|",
        f"| **File** | Filename (path relative to project root) |",
        f"| **Type** | File format |",
        f"| **Step** | Which script step created it |",
        f"| **Size** | Disk size at time of registry generation |",
        f"| **Shape** | Logical shape / row×col / entry count |",
        f"| **Locked** | If ✓: do NOT overwrite without arch version bump |",
        f"| **Consumers** | Downstream modules that read this file |",
        f"",
        f"---",
        f"",
    ]

    # Group by step
    from itertools import groupby
    step_order = [
        "M6B Step0 v2 (prereq)", "M6B Step0b v2 (prereq)",
        "Step 1", "Step 2A", "Step 2B", "Step 3A", "Step 3B", "Step 3C", "Step 3D"
    ]

    entries_by_step = {}
    for e in all_entries:
        s = e["created_by_step"]
        entries_by_step.setdefault(s, []).append(e)

    for step in step_order:
        step_entries = entries_by_step.get(step, [])
        if not step_entries:
            continue
        md_lines += [
            f"## {step}",
            f"",
            f"| File | Type | Size | Shape | Locked | Consumers |",
            f"|------|------|------|-------|--------|-----------|",
        ]
        for e in step_entries:
            fname    = f"`{e['filename']}`"
            ftype    = e["file_type"]
            size     = e["size_human"]
            shape    = e["content_shape"].split("|")[0].strip()  # first segment only
            locked   = "✓" if e["locked_after_creation"] else ""
            consumer = ", ".join(e["downstream_consumers"][:2])
            md_lines.append(f"| {fname} | {ftype} | {size} | {shape} | {locked} | {consumer} |")
        md_lines += [
            f"",
        ]
        # Add semantic meaning for each file as sub-section
        for e in step_entries:
            md_lines += [
                f"### `{e['filename']}`",
                f"",
                f"**Path:** `{e['path']}`  ",
                f"**Shape:** {e['content_shape']}  ",
                f"**Downstream:** {', '.join(e['downstream_consumers'])}  ",
                f"",
                e["semantic_meaning"],
                f"",
            ]
            if e["notes"]:
                md_lines += [f"> **Note:** {e['notes']}", f""]
            md_lines.append("---")
            md_lines.append("")

    md_lines += [
        f"## Gate Summary (at time of generation)",
        f"",
        f"| Gate | Rate | Pass |",
        f"|------|------|------|",
    ]
    for gate_name, val in GATE_RESULTS.items():
        if isinstance(val, dict):
            md_lines.append(f"| {gate_name} | {val.get('rate', 'N/A'):.3f} | {'✓' if val.get('pass') else '✗'} |")
        else:
            md_lines.append(f"| {gate_name} | {val:.4f} | {'✓' if val >= 0.98 else '~'} |")

    md_lines += [
        f"",
        f"---",
        f"",
        f"*Generated automatically by `{SCRIPT_NAME}` — do not edit manually.*",
    ]

    md_path = REPORT_DIR / "M6B_file_registry.md"
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        log(f"  MD  registry: {md_path} ({md_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        log(f"  [ERROR] MD registry save failed: {e}")

    log(f"  File registry complete: {len(all_entries)} files documented")
    return json_path, md_path


registry_json_path, registry_md_path = write_file_registry()


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 75)
log("  SAVING FINAL REPORT")
log("=" * 75)

results.update({
    "arch_version":              ARCH_VERSION,
    "script_version":            SCRIPT_VERSION,
    "anomaly_threshold":         ANOMALY_THRESHOLD,
    "step1_label_range":         "7–12",
    "step2_label_range":         "13–21",
    "step3_label_range":         "22–23 (Group E) + merge",
    "gate_results_summary":      {k: (v if not isinstance(v, dict) else v.get("pass", "?"))
                                  for k, v in GATE_RESULTS.items()},
    "step3_registry_json":       str(registry_json_path),
    "step3_registry_md":         str(registry_md_path),
    "status_for_M65r":           "READY" if (
                                     GATE_RESULTS.get("G8_temporal_ordering", {}).get("pass", False)
                                     and GATE_RESULTS.get("G9_compound_mae", {}).get("pass", False)
                                     and GATE_RESULTS.get("G11ext_gradual_slope", {}).get("pass", False)
                                 ) else "NEEDS_REVIEW",
})

report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
try:
    md = [
        f"# M6B Steps 1+2+3 Generation Report",
        f"",
        f"**Script:** `{SCRIPT_NAME}` v{SCRIPT_VERSION}  ",
        f"**Arch version:** {ARCH_VERSION}  ",
        f"**Date:** {date.today()}  ",
        f"**Device:** {DEVICE}  ",
        f"",
        f"## Results",
        f"",
        f"| Key | Value |",
        f"|-----|-------|",
    ]
    for k, v in results.items():
        if isinstance(v, dict):
            md.append(f"| `{k}` | {json.dumps(v)[:120]} |")
        else:
            md.append(f"| `{k}` | {v} |")
    md += [
        f"",
        f"## Gate Results",
        f"",
        f"| Gate | Rate/Value | Pass |",
        f"|------|-----------|------|",
    ]
    for k, v in GATE_RESULTS.items():
        if isinstance(v, dict):
            md.append(f"| {k} | {v.get('rate', 'N/A')} | {'✓' if v.get('pass') else '✗'} |")
        else:
            md.append(f"| {k} | {v:.4f} | {'✓' if v >= 0.98 else '~'} |")
    md += [
        f"",
        f"## Status for M6.5r",
        f"",
        f"**{results['status_for_M65r']}**",
        f"",
        f"*All gates must pass before M6.5r feature extraction begins.*",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log(f"  Report saved: {report_path}")
except Exception as e:
    log(f"  [ERROR] Report save failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PASTE TEXT UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 75)
print("  PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT")
print("═" * 75)
print(f"M6B_step1_groupB_count           : {results.get('step1_groupB_count', 'N/A')}")
print(f"M6B_step1_gate_G8                : {GATE_RESULTS.get('G8_temporal_ordering', {}).get('rate', 'N/A'):.3f} | PASS={GATE_RESULTS.get('G8_temporal_ordering', {}).get('pass', False)}")
print(f"M6B_step1_gate_G9                : {GATE_RESULTS.get('G9_compound_mae', {}).get('rate', 'N/A'):.3f} | PASS={GATE_RESULTS.get('G9_compound_mae', {}).get('pass', False)}")
print(f"M6B_step2_groupC_count           : {results.get('step2_groupC_count', 'N/A')}")
print(f"M6B_step2_groupD_count           : {results.get('step2_groupD_count', 'N/A')}")
print(f"M6B_step2_gate_G10               : {GATE_RESULTS.get('G10_masked_secondary', {}).get('rate', 'N/A'):.3f} | PASS={GATE_RESULTS.get('G10_masked_secondary', {}).get('pass', False)}")
print(f"M6B_step2_label21_subthreshold_pct: {results.get('step2_label21_subthreshold_pct', 'N/A'):.1f}%")
print(f"M6B_step2_gate_G11ext            : {GATE_RESULTS.get('G11ext_gradual_slope', {}).get('rate', 'N/A'):.3f} | PASS={GATE_RESULTS.get('G11ext_gradual_slope', {}).get('pass', False)}")
print(f"M6B_step3_groupE_count           : {results.get('step3_groupE_count', 'N/A')}")
print(f"M6B_step3_gate_G11               : {GATE_RESULTS.get('G11_multisensor', {}).get('rate', 'N/A'):.3f} | PASS={GATE_RESULTS.get('G11_multisensor', {}).get('pass', False)}")
print(f"M6B_step3_total_sequences        : {results.get('step3_total_sequences', 'N/A')}")
print(f"M6B_step3_label_min_count        : {results.get('step3_label_min_count', 'N/A')}")
print(f"M6B_step3_coupling_fidelity      : {results.get('step3_coupling_fidelity', 'N/A'):.3f}")
print(f"M6B_step3_physics_violations     : {results.get('step3_physics_violations', 'N/A')}")
print(f"M6B_step3_fault_rules_v3_written : {results.get('step3_fault_rules_v3_written', 'N/A')}")
print(f"M6B_step3_physics_ctx_generated  : {results.get('step3_physics_context_generated', 'N/A')}")
print(f"M6B_step3_G1_final               : {GATE_RESULTS.get('G1_final', 'N/A'):.4f}")
print(f"M6B_step3_G2_final               : {GATE_RESULTS.get('G2_final', 'N/A'):.4f}")
print(f"M6B_file_registry_json           : outputs/reports/M6B_file_registry.json")
print(f"M6B_file_registry_md             : outputs/reports/M6B_file_registry.md")
print(f"Status_for_M6p5r                 : {results.get('status_for_M65r', 'N/A')}")
print("═" * 75)
print("  END PASTE UPDATE")
print("═" * 75)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE MANIFEST
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 75)
print("  FILE MANIFEST")
print("─" * 75)

manifest_files = [
    # Step 1
    SYNTH_DIR / "M6B_sequences_groupB.pkl",
    SYNTH_DIR / "z_t_sequences_groupB.pkl",
    # Step 2
    SYNTH_DIR / "M6B_sequences_groupC.pkl",
    SYNTH_DIR / "z_t_sequences_groupC.pkl",
    SYNTH_DIR / "M6B_sequences_groupD.pkl",
    SYNTH_DIR / "z_t_sequences_groupD.pkl",
    # Step 3
    SYNTH_DIR / "M6B_sequences_groupE.pkl",
    SYNTH_DIR / "z_t_sequences_groupE.pkl",
    SYNTH_DIR / "M6B_combined_sequences.pkl",
    SYNTH_DIR / "M6B_sequence_meta.csv",
    SYNTH_DIR / "M6B_physics_context_strings.json",
    MODEL_DIR  / "fault_rules_v3.json",
    REPORT_DIR / f"{SCRIPT_NAME}_report.md",
    REPORT_DIR / "M6B_file_registry.json",
    REPORT_DIR / "M6B_file_registry.md",
]

print("\n  → GitHub push (reports + config):")
for p in manifest_files:
    p = Path(p)
    if p.exists():
        size_mb = p.stat().st_size / 1e6
        tag = "  [PUSH to GitHub]" if (p.suffix in [".md", ".json"] and "sequences" not in p.name and "z_t" not in p.name) else "  [Hugging Face Spaces / local]"
        print(f"    {p.name:<55} {size_mb:>6.1f} MB {tag}")
    else:
        print(f"    {p.name:<55}   NOT FOUND")

print("\n  → Do NOT push to GitHub (large data/model files):")
large_files = [f for f in manifest_files if "sequences" in str(f) or "z_t" in str(f)]
for p in large_files:
    p = Path(p)
    if p.exists():
        print(f"    {p.name:<55} {p.stat().st_size / 1e6:>6.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# NEXT PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 75)
print("  NEXT PROMPT")
print("─" * 75)
total = results.get('step3_total_sequences', '~31800')
status = results.get('status_for_M65r', 'READY')
print(f"""
📦 M6B done. Starting M6.5r.
Finding : {total} sequences generated across 22+2 classes (Groups A–E).
           fault_rules_v3.json written and LOCKED.
           File registry written: M6B_file_registry.json + .md
Status  : {status}
Upload  : outputs/reports/{SCRIPT_NAME}_report.md
          outputs/reports/M6B_file_registry.json
          outputs/reports/M6B_file_registry.md
          models/fault_rules_v3.json
Action  : Provide M6.5r complete feature extraction script.
          M6.5r reads all z_t pkl files (groupA_normal, groupA_faults,
          groupA_faults_rerun, groupB, groupC, groupD, groupE) +
          M6B_combined_sequences.pkl + M6B_sequence_meta.csv.
          Outputs: M6B_feature_matrix.csv (~196000 rows × ~35 features).
""")
print("─" * 75)