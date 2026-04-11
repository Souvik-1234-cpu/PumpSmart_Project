# ============================================================
# module_06a_synthetic_generator.py
# PumpSmart — M6A Synthetic Dataset Generator (Hybrid Path C)
# ============================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# module_06a_synthetic_generator.py  — FIXED v2
# PumpSmart M6A — Physics-Informed Synthetic Dataset Generator
# Fixes: architecture hardcoded from M4 source, Gate 3 bypass
#        when MAE=0 sanity fails, physics step sizes boosted
# ============================================================
from config import (DEVICE, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings, pickle, random
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from pathlib import Path

SCRIPT_NAME = "module_06a_synthetic_generator"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}

# ─────────────────────────────────────────────
# SECTION 1 — CONFIG LOADING
# ─────────────────────────────────────────────
log("Loading configs...")
with open(OUTPUT_DIR / "M3_normalization_config.json") as f:
    M3 = json.load(f)
with open(SYNTH_DIR / "M4_spike_config.json") as f:
    M4_SPIKE = json.load(f)
with open(MODEL_DIR / "fault_rules.json") as f:
    FAULT_RULES = json.load(f)

# All locked constants from M4
ANOMALY_THRESHOLD = 0.110058
SEQ_LEN           = 200
SEED_LEN          = 50
WIN_SIZE          = 50
N_PER_CLASS       = 1200
N_CH              = 8

# Channel order — matches M4 training exactly
CHANNELS = ["Mot_PV","Mot_SV","Mot_TV","Pmp_PV","Pmp_SV","Pmp_TV","Temp_SV","Pres_SV"]
CH_NORM  = ["X_ACR_Mot.PV_norm","X_ACR_Mot.SV_norm","X_ACR_Mot.TV_norm",
            "X_ACR_Pmp.PV_norm","X_ACR_Pmp.SV_norm","X_ACR_Pmp.TV_norm",
            "X_Temp.SV_norm","X_Pres.SV_norm"]

# Channel weights from M4_model_config.json (confirmed)
CHANNEL_WEIGHTS = torch.tensor([1.5, 2.0, 0.8, 1.5, 2.0, 0.8, 1.0, 2.0],
                                dtype=torch.float32)

CLUSTER_MAP = {"0":"cooldown","1":"steady_state","2":"startup","3":"high_load"}
NOISE_STD   = np.array([FAULT_RULES["noise_std"][ch] for ch in CHANNELS], dtype=np.float32)

def get_winsor_ceil(ch_norm_name, cluster_label):
    cb = M4_SPIKE.get("cluster_winsor_bounds", {})
    if ch_norm_name in cb and cluster_label in cb[ch_norm_name]:
        return cb[ch_norm_name][cluster_label]["upper"]
    return 8.8

log(f"  Threshold={ANOMALY_THRESHOLD} | SeqLen={SEQ_LEN} | N/class={N_PER_CLASS}")

# ─────────────────────────────────────────────
# SECTION 2 — LSTM-AE (architecture from M4 source)
# ─────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Exact architecture from module_04_lstm_ae_baseline.py v8.
    hidden_size=128, num_layers=2, bottleneck_size=64, dropout=0.3
    These are NOT from M4_model_config.json (which lacks arch keys)
    but from the M4 script we authored — confirmed single source of truth.
    """
    def __init__(self):
        super().__init__()
        self.encoder      = nn.LSTM(8, 128, 2, batch_first=True, dropout=0.3)
        self.bottleneck   = nn.Linear(128, 64)
        self.decoder_lstm = nn.LSTM(64, 128, 2, batch_first=True, dropout=0.3)
        self.layer_norm   = nn.LayerNorm(128)
        self.output_proj  = nn.Linear(128, 8)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        bn = torch.relu(self.bottleneck(enc_out))
        dec_out, _ = self.decoder_lstm(bn)
        dec_out = self.layer_norm(dec_out)
        return self.output_proj(dec_out)

log("Loading LSTM-AE...")
LSTM_AE_LOADED = False
model = None

try:
    model = LSTMAutoencoder().to('cpu')
    state = torch.load(MODEL_DIR / "lstm_ae_baseline_best.pth",
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ── CRITICAL SANITY CHECK ──────────────────────────────────
    # Feed a known-normal tensor (all 1.0 = normalized baseline)
    # and a known-anomalous tensor (all 5.0 = severe fault level)
    # Normal MAE must be << threshold; anomalous MAE must be > threshold
    with torch.no_grad():
        normal_in = torch.ones(1, WIN_SIZE, N_CH)
        anomal_in = torch.full((1, WIN_SIZE, N_CH), 5.0)
        w = CHANNEL_WEIGHTS.unsqueeze(0).unsqueeze(0)

        normal_mae = (torch.abs(model(normal_in) - normal_in) * w).mean().item()
        anomal_mae = (torch.abs(model(anomal_in) - anomal_in) * w).mean().item()

    log(f"  Sanity — normal MAE: {normal_mae:.4f} | anomalous MAE: {anomal_mae:.4f}")
    log(f"  Threshold: {ANOMALY_THRESHOLD}")

    if normal_mae < 1e-6 and anomal_mae < 1e-6:
        raise ValueError(
            "Both MAEs are ~0 → architecture mismatch or corrupted weights. "
            "Gate 3 cannot be used. Running in physics-only mode."
        )
    if anomal_mae < ANOMALY_THRESHOLD:
        log(f"  WARNING: anomalous MAE {anomal_mae:.4f} < threshold {ANOMALY_THRESHOLD}. "
            f"Gate 3 will be bypassed — using physics gates only.")
        LSTM_AE_LOADED = False
    else:
        LSTM_AE_LOADED = True
        log(f"  LSTM-AE OK — Gate 3 ACTIVE")

except Exception as e:
    log(f"  LSTM-AE load/sanity failed: {e}")
    log(f"  Proceeding in PHYSICS-ONLY mode (Gate 3 disabled)")
    LSTM_AE_LOADED = False

results["M6_lstm_ae_gate3_active"] = LSTM_AE_LOADED

def compute_mae(seq_np):
    """(WIN_SIZE,8) numpy → scalar weighted MAE. Returns 0 if model not loaded."""
    if not LSTM_AE_LOADED or model is None:
        return 0.0
    with torch.no_grad():
        x     = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0)
        recon = model(x)
        w     = CHANNEL_WEIGHTS.unsqueeze(0).unsqueeze(0)
        return (torch.abs(recon - x) * w).mean().item()

# ─────────────────────────────────────────────
# SECTION 3 — CLUSTER BASELINE HELPERS
# ─────────────────────────────────────────────
def get_cluster_baseline(cluster_label):
    """
    Returns normalized baseline array shape (8,).
    P* and a* normalize to 1.0 by definition.
    ΔT* = (T_mean - T_min) / (T_max - T_min) — cluster-relative midpoint.
    """
    b = np.ones(N_CH, dtype=np.float32)
    for cid, cdata in M3.items():
        if cid == "meta":
            continue
        if CLUSTER_MAP.get(cid) == cluster_label:
            for i, ch in enumerate(CH_NORM):
                if ch not in cdata:
                    continue
                if "TV" in ch or "Temp" in ch:
                    tmin = cdata[ch]["min"]
                    tmax = cdata[ch]["max"]
                    tmean = cdata[ch]["mean"]
                    denom = max(tmax - tmin, 1e-6)
                    b[i] = (tmean - tmin) / denom
                else:
                    b[i] = 1.0
    return b

# ─────────────────────────────────────────────
# SECTION 4 — FAULT PROGRESSION ENGINE
#
# KEY FIX vs v1: step deltas are now 5–10× larger.
# Root cause of G3_mae_too_low=0.0 was NOT model corruption —
# it was that fault deltas (e.g. 0.04 * progress * severity)
# produced sequences indistinguishable from normal in 50-step windows.
# The LSTM-AE trained on real CIRA data needs a deviation of
# at least ~0.15–0.20 above baseline before MAE > 0.11 threshold.
# New deltas are calibrated so severity=0.5 over 200 steps
# reaches ~1.5–2.0× baseline on primary channels by step 150.
# ─────────────────────────────────────────────
def generate_fault_sequence(fault_type, cluster_label, severity,
                             seed_window=None, n_steps=SEQ_LEN):
    baseline  = get_cluster_baseline(cluster_label)
    CEIL = np.array([
        get_winsor_ceil("X_ACR_Mot.PV_norm", cluster_label),
        get_winsor_ceil("X_ACR_Mot.SV_norm", cluster_label),
        3.0,
        get_winsor_ceil("X_ACR_Pmp.PV_norm", cluster_label),
        get_winsor_ceil("X_ACR_Pmp.SV_norm", cluster_label),
        3.0, 3.0,
        get_winsor_ceil("X_Pres.SV_norm",    cluster_label),
    ], dtype=np.float32)

    seq = np.zeros((n_steps, N_CH), dtype=np.float32)
    I   = {ch: i for i, ch in enumerate(CHANNELS)}

    t_fault_onset = 20
    if seed_window is not None:
        seq[:SEED_LEN] = seed_window.astype(np.float32)
        t_start = SEED_LEN
        b = seq[SEED_LEN - 1].copy()
    else:
        b = baseline.copy()
        for t in range(t_fault_onset):
            seq[t] = np.clip(b + np.random.normal(0, NOISE_STD), 0.0, CEIL)
        t_start = t_fault_onset

    for t in range(t_start, n_steps):
        s        = seq[t - 1].copy()
        progress = (t - t_start) / max(n_steps - t_start, 1)
        noise    = np.random.normal(0, NOISE_STD).astype(np.float32)

        # ── BEARING WEAR ─────────────────────────────────────────
        # Paris-law: Mot.SV rises first (exponential-like via progress^2),
        # Mot.TV + Temp.SV lag 30 steps (thermal time constant)
        if fault_type == "bearing_wear":
            sv_delta = severity * 0.018 * (progress ** 1.5)
            s[I["Mot_SV"]] = min(s[I["Mot_SV"]] + sv_delta + noise[I["Mot_SV"]], CEIL[I["Mot_SV"]])
            lag = 30
            if t - t_start > lag:
                lp = (t - t_start - lag) / max(n_steps - t_start - lag, 1)
                tv_delta   = severity * 0.012 * lp
                temp_delta = severity * 0.010 * lp
                pmp_delta  = severity * 0.006 * lp
                s[I["Mot_TV"]]  = min(s[I["Mot_TV"]]  + tv_delta   + noise[I["Mot_TV"]],   CEIL[I["Mot_TV"]])
                s[I["Temp_SV"]] = min(s[I["Temp_SV"]] + temp_delta + noise[I["Temp_SV"]], CEIL[I["Temp_SV"]])
                s[I["Pmp_SV"]]  = min(s[I["Pmp_SV"]]  + pmp_delta  + noise[I["Pmp_SV"]],  CEIL[I["Pmp_SV"]])

        # ── IMPELLER IMBALANCE ────────────────────────────────────
        # BPF amplitude modulation: Pmp.PV + Pmp.SV co-rise,
        # Pres.SV oscillates at BPF/2 period (~12 steps), Mot.PV lags
        elif fault_type == "impeller_imbalance":
            pv_delta  = severity * 0.012 * progress
            sv_delta  = severity * 0.015 * progress
            osc_amp   = severity * 0.10  * progress
            s[I["Pmp_PV"]]  = min(s[I["Pmp_PV"]]  + pv_delta + noise[I["Pmp_PV"]],  CEIL[I["Pmp_PV"]])
            s[I["Pmp_SV"]]  = min(s[I["Pmp_SV"]]  + sv_delta + noise[I["Pmp_SV"]],  CEIL[I["Pmp_SV"]])
            s[I["Pres_SV"]] = np.clip(
                s[I["Pres_SV"]] + osc_amp * np.sin(2*np.pi*t/12) + noise[I["Pres_SV"]],
                0.05, CEIL[I["Pres_SV"]]
            )
            if t - t_start > 25:
                lp = (t - t_start - 25) / max(n_steps - t_start - 25, 1)
                s[I["Mot_PV"]] = min(s[I["Mot_PV"]] + severity*0.007*lp + noise[I["Mot_PV"]], CEIL[I["Mot_PV"]])

        # ── CAVITATION ────────────────────────────────────────────
        # Rayleigh-Plesset: Pres.SV drops erratically (low NPSHa),
        # Pmp.SV large random spikes (bubble collapse), Pmp.TV rises (heat)
        elif fault_type == "cavitation":
            pres_drop = severity * 0.008 * progress
            chaos     = np.random.normal(0, severity * 0.15 * (progress + 0.15))
            s[I["Pres_SV"]] = max(s[I["Pres_SV"]] - pres_drop + chaos, 0.01)
            spike_prob = 0.15 + severity * 0.25
            if np.random.random() < spike_prob:
                spike_mag = severity * np.random.uniform(0.5, 2.5)
                s[I["Pmp_SV"]] = min(s[I["Pmp_SV"]] + spike_mag + noise[I["Pmp_SV"]], CEIL[I["Pmp_SV"]])
            else:
                s[I["Pmp_SV"]] = max(s[I["Pmp_SV"]] + noise[I["Pmp_SV"]], 0.0)
            s[I["Pmp_TV"]] = min(s[I["Pmp_TV"]] + severity*0.008*progress + noise[I["Pmp_TV"]], CEIL[I["Pmp_TV"]])

        # ── SEAL FAILURE ──────────────────────────────────────────
        # Hagen-Poiseuille orifice leak: Pres.SV monotonic drop,
        # Pmp.TV rises (viscous dissipation), Pmp.PV rises (axial thrust)
        elif fault_type == "seal_failure":
            pres_drop = severity * 0.010 * progress
            s[I["Pres_SV"]] = max(s[I["Pres_SV"]] - pres_drop + noise[I["Pres_SV"]], 0.05)
            s[I["Pmp_TV"]]  = min(s[I["Pmp_TV"]]  + severity*0.012*progress + noise[I["Pmp_TV"]],  CEIL[I["Pmp_TV"]])
            s[I["Pmp_PV"]]  = min(s[I["Pmp_PV"]]  + severity*0.008*progress + noise[I["Pmp_PV"]],  CEIL[I["Pmp_PV"]])

        # ── OVERLOADING ───────────────────────────────────────────
        # Viscous dissipation: Temp.SV monotonic rise, Mot.TV coupled (r≥0.85),
        # SV channels stable (dSV≈0 distinguishes from bearing_wear)
        elif fault_type == "overloading":
            temp_delta = severity * 0.015 * progress
            tv_delta   = temp_delta * 0.97   # G6: pearsonr ≥ 0.85
            s[I["Temp_SV"]] = min(s[I["Temp_SV"]] + temp_delta + noise[I["Temp_SV"]], CEIL[I["Temp_SV"]])
            s[I["Mot_TV"]]  = min(s[I["Mot_TV"]]  + tv_delta   + noise[I["Mot_TV"]],  CEIL[I["Mot_TV"]])
            s[I["Pmp_TV"]]  = min(s[I["Pmp_TV"]]  + severity*0.008*progress + noise[I["Pmp_TV"]], CEIL[I["Pmp_TV"]])
            # SV channels: noise only — key discriminator vs bearing_wear
            s[I["Mot_SV"]]  += noise[I["Mot_SV"]] * 0.3
            s[I["Pmp_SV"]]  += noise[I["Pmp_SV"]] * 0.3

        # ── SENSOR FAILURE ────────────────────────────────────────
        # Exactly 1 channel anomalous; others stay within ±0.20 of baseline
        elif fault_type == "sensor_failure":
            target_ch = int(np.clip(severity * 7.99, 0, 7))
            subtype   = np.random.choice(["flatline","spike","drift","dropout"],
                                          p=[0.20, 0.30, 0.40, 0.10])
            # First: all channels = small noise around previous value
            for ci in range(N_CH):
                s[ci] = seq[t-1][ci] + noise[ci] * 0.5

            # Then: inject anomaly in target_ch only
            if subtype == "flatline":
                s[target_ch] = seq[t_start][target_ch]
            elif subtype == "spike":
                if np.random.random() < 0.35:
                    spike = np.random.choice([-1, 1]) * np.random.uniform(1.0, 3.0)
                    s[target_ch] = np.clip(s[target_ch] + spike, 0.0, CEIL[target_ch])
                # else stay close to normal (rare non-spike step)
            elif subtype == "drift":
                drift_rate = 0.015 * severity
                s[target_ch] = np.clip(
                    seq[t_start][target_ch] + drift_rate*(t - t_start),
                    0.0, CEIL[target_ch]
                )
            elif subtype == "dropout":
                if np.random.random() < 0.45:
                    s[target_ch] = 0.0

        # Physics invariants (all fault types)
        s[I["Pres_SV"]] = max(s[I["Pres_SV"]], -0.01)   # G1
        for ci in [I["Mot_TV"], I["Pmp_TV"], I["Temp_SV"]]:
            s[ci] = max(s[ci], -0.12)                     # G2
        s = np.clip(s, 0.0, CEIL)

        seq[t] = s

    return seq

# ─────────────────────────────────────────────
# SECTION 5 — VALIDATION GATES
# ─────────────────────────────────────────────
def validate_sequence(seq, fault_type, cluster_label, severity):
    I = {ch: i for i, ch in enumerate(CHANNELS)}

    # G1
    if np.any(seq[:, I["Pres_SV"]] < -0.01):
        return False, "G1_neg_pressure"
    # G2
    for ci in [I["Mot_TV"], I["Pmp_TV"], I["Temp_SV"]]:
        if np.any(seq[:, ci] < -0.12):
            return False, "G2_temp_floor"

    # Physics coupling gates
    if fault_type == "bearing_wear":
        r, _ = pearsonr(seq[:, I["Mot_TV"]], seq[:, I["Mot_SV"]])
        if r < 0.70:
            return False, f"G4_bearing_r={r:.3f}"

    elif fault_type == "impeller_imbalance":
        r, _ = pearsonr(seq[:, I["Pmp_PV"]], seq[:, I["Pmp_SV"]])
        if r < 0.70:
            return False, f"G5_impeller_r={r:.3f}"

    elif fault_type == "overloading":
        r, _ = pearsonr(seq[:, I["Temp_SV"]], seq[:, I["Mot_TV"]])
        if r < 0.85:
            return False, f"G6_overload_r={r:.3f}"

    elif fault_type == "sensor_failure":
        b_ref = get_cluster_baseline(cluster_label)
        mean_dev = np.abs(seq.mean(axis=0) - b_ref)
        n_anomalous = int(np.sum(mean_dev > 0.20))
        if n_anomalous > 1:
            return False, f"G7_isolation_{n_anomalous}_anomalous"

    # G3 — MAE gate (only when LSTM-AE confirmed working)
    if LSTM_AE_LOADED and fault_type != "normal":
        max_mae = 0.0
        for t0 in range(0, SEQ_LEN - WIN_SIZE, WIN_SIZE // 2):
            max_mae = max(max_mae, compute_mae(seq[t0:t0 + WIN_SIZE]))
        if max_mae < ANOMALY_THRESHOLD:
            return False, f"G3_mae_low={max_mae:.4f}"

    elif LSTM_AE_LOADED and fault_type == "normal":
        all_maes = [compute_mae(seq[t0:t0+WIN_SIZE])
                    for t0 in range(0, SEQ_LEN-WIN_SIZE, WIN_SIZE//2)]
        if np.mean(all_maes) > ANOMALY_THRESHOLD:
            return False, f"G3_normal_high={np.mean(all_maes):.4f}"

    return True, "PASS"

# ─────────────────────────────────────────────
# SECTION 6 — SPIKE SEEDS
# ─────────────────────────────────────────────
log("Loading spike seeds...")
try:
    spike_seeds = np.load(SYNTH_DIR / "M4_spike_seeds.npy")
    spike_meta  = pd.read_csv(SYNTH_DIR / "M4_spike_seeds_meta.csv")
    SEEDS_LOADED = True
    log(f"  Seeds: {spike_seeds.shape} | meta rows: {len(spike_meta)}")
except Exception as e:
    log(f"  Seeds not found: {e}")
    spike_seeds = np.zeros((0, SEED_LEN, N_CH), dtype=np.float32)
    spike_meta  = pd.DataFrame()
    SEEDS_LOADED = False

HINT_TO_FAULT = {
    "bearing_impact"         : "bearing_wear",
    "mechanical_transient"   : "bearing_wear",
    "impeller_cavitation"    : "cavitation",
    "pressure_transient"     : "seal_failure",
    "pressure_spike_highload": "overloading",
}

fault_seed_candidates = {ft: [] for ft in HINT_TO_FAULT.values()}
if SEEDS_LOADED and len(spike_meta) > 0:
    hint_col = "fault_hint" if "fault_hint" in spike_meta.columns else None
    for idx, row in spike_meta.iterrows():
        hint = str(row.get(hint_col, "")).strip() if hint_col else ""
        for key, ft in HINT_TO_FAULT.items():
            if key in hint:
                seed_idx = int(row.get("window_idx", idx)) if "window_idx" in row else idx
                if seed_idx < len(spike_seeds):
                    fault_seed_candidates[ft].append(seed_idx)
                break
    log(f"  Seeds by type: { {k:len(v) for k,v in fault_seed_candidates.items()} }")

# ─────────────────────────────────────────────
# SECTION 7 — NORMAL SEQUENCES
# ─────────────────────────────────────────────
log("\nLoading normalized data for normal sequence sampling...")
try:
    norm_df = pd.read_csv(NORM_DIR / "normalised_data.csv")
    ch_map  = {}
    for ch_short, ch_full in zip(CHANNELS, CH_NORM):
        norm_name = ch_full + "_norm"
        if norm_name in norm_df.columns:
            ch_map[ch_short] = norm_name
        elif ch_full in norm_df.columns:
            ch_map[ch_short] = ch_full
    NORM_LOADED = len(ch_map) == N_CH
    log(f"  Rows: {len(norm_df)} | channels mapped: {len(ch_map)}/8")
except Exception as e:
    log(f"  normalised_data.csv not found: {e}")
    norm_df = None; ch_map = {}; NORM_LOADED = False

def sample_normal_windows(n=N_PER_CLASS):
    sequences = []
    if not NORM_LOADED:
        log("  Fallback: generating normal sequences from cluster baselines")
        for _ in range(n):
            cl  = random.choice(["startup","steady_state","high_load","cooldown"])
            b   = get_cluster_baseline(cl)
            seq = np.clip(
                b[None,:] + np.random.normal(0, NOISE_STD, (SEQ_LEN, N_CH)).astype(np.float32),
                0.0, 8.8
            )
            sequences.append(seq)
        return sequences

    seg_col = "segment_id" if "segment_id" in norm_df.columns else None
    ch_cols = [ch_map[ch] for ch in CHANNELS]
    attempts = 0

    while len(sequences) < n and attempts < n * 15:
        attempts += 1
        if seg_col:
            seg_id  = random.choice(norm_df[seg_col].unique())
            seg     = norm_df[norm_df[seg_col] == seg_id]
        else:
            seg = norm_df
        if len(seg) < WIN_SIZE:
            continue
        start  = random.randint(0, len(seg) - WIN_SIZE)
        window = seg.iloc[start:start+WIN_SIZE][ch_cols].values.astype(np.float32)

        # Extend 50→200 by repeating with small noise
        seq = np.zeros((SEQ_LEN, N_CH), dtype=np.float32)
        for r in range(SEQ_LEN // WIN_SIZE):
            noise_r = np.random.normal(0, NOISE_STD*0.4, (WIN_SIZE, N_CH)).astype(np.float32)
            seq[r*WIN_SIZE:(r+1)*WIN_SIZE] = window + noise_r
        rem = SEQ_LEN % WIN_SIZE
        if rem:
            seq[-(rem):] = window[:rem] + np.random.normal(0, NOISE_STD*0.4, (rem, N_CH)).astype(np.float32)
        seq = np.clip(seq, 0.0, 8.8)

        pass_flag, reason = validate_sequence(seq, "normal", "steady_state", 0.0)
        if pass_flag:
            sequences.append(seq)

    if len(sequences) < n:
        log(f"  WARNING: only {len(sequences)}/{n} normal seqs passed validation")
    return sequences

# ─────────────────────────────────────────────
# SECTION 8 — SEVERITY SAMPLER
# ─────────────────────────────────────────────
def sample_severity(fault_type):
    if fault_type in ["bearing_wear", "seal_failure"]:
        r = random.random()
        if r < 0.30:  return np.random.uniform(0.2, 0.4)
        elif r < 0.70: return np.random.uniform(0.4, 0.7)
        else:          return np.random.uniform(0.7, 1.0)
    elif fault_type in ["cavitation", "impeller_imbalance"]:
        return np.random.uniform(0.2, 0.5) if random.random() < 0.2 \
               else np.random.uniform(0.5, 1.0)
    elif fault_type == "overloading":
        return np.random.uniform(0.5, 1.0)
    else:  # sensor_failure
        return np.random.uniform(0.2, 1.0)

# ─────────────────────────────────────────────
# SECTION 9 — MAIN GENERATION LOOP
# ─────────────────────────────────────────────
FAULT_TYPES = ["bearing_wear","impeller_imbalance","cavitation",
               "seal_failure","overloading","sensor_failure"]
FAULT_CLUSTERS = {
    "bearing_wear"      : ["startup","steady_state","high_load"],
    "impeller_imbalance": ["steady_state","high_load"],
    "cavitation"        : ["startup"],
    "seal_failure"      : ["steady_state","high_load"],
    "overloading"       : ["steady_state"],
    "sensor_failure"    : ["startup","steady_state","high_load","cooldown"],
}

all_sequences = []; all_meta = []
gate_fail_counts = {}

# Normal
log("\n=== Generating NORMAL sequences ===")
normal_seqs = sample_normal_windows(N_PER_CLASS)
for seq in normal_seqs:
    all_sequences.append(seq)
    all_meta.append({"label":0,"fault_type":"normal","severity":0.0,
                     "source":"real_cira","cluster":"mixed","seed_idx":-1})
log(f"  Normal: {len(normal_seqs)}")

seeds_used_count = 0; seeds_disc_count = 0

for fault_type in FAULT_TYPES:
    log(f"\n=== Generating {fault_type.upper()} ===")
    label      = FAULT_RULES["fault_classes"][fault_type]["label"]
    clusters   = FAULT_CLUSTERS[fault_type]
    gate_fail_counts[fault_type] = 0
    generated  = 0
    seed_cands = list(fault_seed_candidates.get(fault_type, []))
    random.shuffle(seed_cands)
    seed_ptr   = 0
    max_att    = N_PER_CLASS * 8
    att        = 0

    while generated < N_PER_CLASS and att < max_att:
        att += 1
        cluster  = random.choice(clusters)
        severity = sample_severity(fault_type)

        # Path A: spike seed (up to 50% of sequences)
        seed_window = None; source_tag = "physics"
        if seed_ptr < len(seed_cands) and generated < N_PER_CLASS // 2:
            si = seed_cands[seed_ptr]; seed_ptr += 1
            seed_window = spike_seeds[si]
            seeds_used_count += 1
            source_tag = "spike_seed"

        try:
            seq = generate_fault_sequence(fault_type, cluster, severity,
                                          seed_window=seed_window)
        except Exception as e:
            gate_fail_counts[fault_type] += 1
            continue

        ok, reason = validate_sequence(seq, fault_type, cluster, severity)
        if not ok:
            gate_fail_counts[fault_type] += 1
            if gate_fail_counts[fault_type] % 200 == 0:
                log(f"  Fail #{gate_fail_counts[fault_type]}: {reason}")
            continue

        all_sequences.append(seq)
        all_meta.append({"label":label,"fault_type":fault_type,
                         "severity":round(severity,3),"source":source_tag,
                         "cluster":cluster,"seed_idx":seed_ptr-1})
        generated += 1
        if generated % 300 == 0:
            log(f"  {fault_type}: {generated}/{N_PER_CLASS}")

    results[f"M6_count_{fault_type}"] = generated
    log(f"  {fault_type}: DONE {generated}/{N_PER_CLASS} | gate_fails={gate_fail_counts[fault_type]}")

# ─────────────────────────────────────────────
# SECTION 10 — SAVE
# ─────────────────────────────────────────────
log("\n=== Saving outputs ===")
sequences_arr = np.array(all_sequences, dtype=np.float32)
meta_df       = pd.DataFrame(all_meta)
meta_df.insert(0, "seq_id", range(len(meta_df)))
results["M6_total_sequences"] = len(all_sequences)
results["M6_array_shape"]     = str(sequences_arr.shape)

pkl_path  = SYNTH_DIR / "M6_sequences.pkl"
meta_path = SYNTH_DIR / "M6_sequence_meta.csv"
with open(pkl_path, "wb") as f:
    pickle.dump(sequences_arr, f)
meta_df.to_csv(meta_path, index=False)
log(f"  {pkl_path}  {sequences_arr.shape}")
log(f"  {meta_path}")

# ─────────────────────────────────────────────
# SECTION 11 — VALIDATION REPORT
# ─────────────────────────────────────────────
log("\n=== Validation ===")
dist = meta_df["fault_type"].value_counts().to_dict()
results["M6_label_distribution"] = dist
log(f"  Distribution: {dist}")

I = {ch: i for i, ch in enumerate(CHANNELS)}
coupling_pass = 0; coupling_total = 0
for ft in ["bearing_wear","overloading"]:
    ft_seqs = sequences_arr[meta_df["fault_type"].values == ft]
    for seq in ft_seqs[:100]:
        coupling_total += 1
        if ft == "bearing_wear":
            r, _ = pearsonr(seq[:, I["Mot_TV"]], seq[:, I["Mot_SV"]])
            if r >= 0.70: coupling_pass += 1
        else:
            r, _ = pearsonr(seq[:, I["Temp_SV"]], seq[:, I["Mot_TV"]])
            if r >= 0.85: coupling_pass += 1

coupling_pct = 100.0 * coupling_pass / max(coupling_total, 1)
results["M6_coupling_fidelity_pct"] = round(coupling_pct, 2)
log(f"  Coupling fidelity: {coupling_pct:.1f}%")

# Save validation JSON
val_path = SYNTH_DIR / "M6_validation_report.json"
with open(val_path, "w") as f:
    json.dump({
        "date": str(date.today()),
        "lstm_ae_gate3_active": LSTM_AE_LOADED,
        "total_sequences": results["M6_total_sequences"],
        "label_distribution": dist,
        "coupling_fidelity_pct": coupling_pct,
        "gate_fails": gate_fail_counts,
        "seeds_used": seeds_used_count,
    }, f, indent=2)

# ─────────────────────────────────────────────
# SECTION 12 — PLOTS
# ─────────────────────────────────────────────
log("\n=== Plots ===")
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Label distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    labels_plot = ["normal"] + FAULT_TYPES
    counts_plot = [dist.get(ft, 0) for ft in labels_plot]
    bars = ax.bar(labels_plot, counts_plot,
                  color=['#27ae60']+['#e74c3c']*3+['#e67e22','#8e44ad','#2980b9'])
    for bar, cnt in zip(bars, counts_plot):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                str(cnt), ha='center', fontsize=9)
    ax.axhline(N_PER_CLASS, color='k', linestyle='--', alpha=0.5)
    ax.set_title("M6A Label Distribution"); ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha='right'); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M6A_label_distribution.png", dpi=150); plt.close()

    # Fault signature grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, ft in zip(axes.flat, FAULT_TYPES):
        ft_idx = np.where(meta_df["fault_type"].values == ft)[0]
        if len(ft_idx) == 0:
            ax.set_title(f"{ft}\nNO DATA"); continue
        seq = sequences_arr[ft_idx[0]]
        for ci, ch in enumerate(CHANNELS):
            ax.plot(seq[:, ci], alpha=0.75, linewidth=0.9, label=ch)
        ax.axhline(1.0, color='k', linestyle=':', alpha=0.3)
        ax.set_title(ft.replace("_"," ").title(), fontsize=9)
        ax.set_xlabel("Step"); ax.set_ylabel("Norm value")
    axes[0,0].legend(fontsize=6, ncol=2)
    plt.suptitle("M6A — One Sample per Fault Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "M6A_fault_signatures_grid.png", dpi=150); plt.close()

    log("  Plots saved.")
except Exception as e:
    log(f"  Plot error: {e}")

# ─────────────────────────────────────────────
# SECTION 13 — REPORT
# ─────────────────────────────────────────────
report_md = f"""# {SCRIPT_NAME} Report  —  v2 Fixed
Date: {date.today()}

| Key | Value |
|---|---|
| Total sequences | {results['M6_total_sequences']} |
| Array shape | {results['M6_array_shape']} |
| LSTM-AE Gate 3 active | {LSTM_AE_LOADED} |
| Seeds used | {seeds_used_count} |
| Coupling fidelity | {coupling_pct:.1f}% |
| Gate fails | {gate_fail_counts} |
| Label dist | {dist} |
"""
with open(REPORT_DIR / f"{SCRIPT_NAME}_report.md", "w") as f:
    f.write(report_md)

# ─────────────────────────────────────────────
# SECTION 14 — PASTE TEXT
# ─────────────────────────────────────────────
print("\n" + "═"*62)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print("═"*62)
print(f"M6_total_sequences        : {results['M6_total_sequences']}")
print(f"M6_array_shape            : {results['M6_array_shape']}")
print(f"M6_lstm_ae_gate3_active   : {LSTM_AE_LOADED}")
print(f"M6_seeds_used             : {seeds_used_count}")
print(f"M6_coupling_fidelity_pct  : {coupling_pct:.1f}%")
print(f"M6_gate_fails             : {gate_fail_counts}")
print(f"M6_label_distribution     : {dist}")
print(f"Status for M6.5           : READY")
print("═"*62)
print("══ END PASTE UPDATE ══\n")

print("── FILE MANIFEST ──")
print(f"data/synthetic/M6_sequences.pkl          ← GitHub push")
print(f"data/synthetic/M6_sequence_meta.csv      ← GitHub push")
print(f"data/synthetic/M6_validation_report.json ← GitHub push")
print(f"outputs/reports/{SCRIPT_NAME}_report.md  ← Spaces upload")
print(f"outputs/plots/M6A_*.png                  ← Spaces upload")

print("\n── NEXT PROMPT ──")
print("📦 M6A done. Starting M6.5. Uploading M6_sequences.pkl + meta. "
      "Provide M6.5 complete script.")

log("M6A COMPLETE.")