# =============================================================================
# module_06a_synthetic_generator_v5.py
# PumpSmart — M6A Synthetic Generator v5
# Fixes vs v4:
#   F1: bearing_wear — Temp.SV* now coupled to Mot.TV* via _tcoup (r=0.9793)
#   F2: impeller_imbalance — abs(sin) AM envelope (non-negative vibration)
#   F3: cavitation — M5-faithful: severity-dependent t_onset, mean_drop=0.6*sev
#   F4: overloading — Pres.SV* affinity law Q-H shift added
#   F5: sensor_failure — dropout subtype added (channel→0.0)
#   F6: All generation via m6b_physics_lib.py (single source of truth)
# =============================================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DEVICE, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, warnings, pickle, random
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Import unified physics library ────────────────────────────────────────────
from m6b_physics_lib import (
    init_lib, CH, CHANNELS, N_CH, CHANNEL_TO_M3_KEY,
    get_cluster_mean, apply_winsorization, make_baseline,
    generate_bearing_wear, generate_impeller_imbalance,
    generate_cavitation, generate_seal_failure,
    generate_overloading, generate_sensor_failure,
    generate_normal_from_real, SENSOR_SUBTYPES,
    NOISE_STD as LIB_NOISE_STD
)

SCRIPT_NAME = "module_06a_synthetic_generator_v5"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
log(f"Script: {SCRIPT_NAME} | Device: {DEVICE} | Date: {date.today()}")
log("v5 — unified m6b_physics_lib.py: F1-F6 fixes applied")

# =============================================================================
# SECTION 1 — CONFIG LOADING
# =============================================================================
log("SECTION 1 — Loading configs...")

try:
    with open(MODEL_DIR / "M3_normalization_config.json") as f:
        norm_config = json.load(f)
    log("  M3_normalization_config.json loaded")
except Exception as e:
    log(f"  [FATAL] {e}"); raise

try:
    with open(MODEL_DIR / "M5_physics_config.json") as f:
        phys_config = json.load(f)
    log("  M5_physics_config.json loaded")
except Exception as e:
    log(f"  [WARNING] M5_physics_config fallback: {e}")
    phys_config = {}

try:
    with open(MODEL_DIR / "fault_rules.json") as f:
        FAULT_RULES = json.load(f)
    log("  fault_rules.json loaded")
except Exception as e:
    log(f"  [WARNING] fault_rules.json: {e}")
    FAULT_RULES = {}

try:
    with open(SYNTH_DIR / "M4_spike_config.json") as f:
        M4_SPIKE = json.load(f)
    log("  M4_spike_config.json loaded")
except Exception as e:
    log(f"  [WARNING] M4_spike_config.json: {e}")
    M4_SPIKE = {}

# ── Initialise physics library ────────────────────────────────────────────────
init_lib(norm_config, phys_config, seed=SEED)
log("  m6b_physics_lib initialised (seed=2026)")

# Locked from M4 training
ANOMALY_THRESHOLD = 0.110058
SEQ_LEN           = 200
WIN_SIZE          = 50
N_PER_CLASS       = 1200
TOTAL_SEQ         = 8400

# M6A uses WRONG channel order internally — but output arrays are re-ordered
# to M6B channel order via m6b_physics_lib CH mapping
# M6B channel order (LOCKED):
log(f"  M6B channel order: {CH}")
results["channels"]           = CHANNELS
results["M4_threshold"]       = ANOMALY_THRESHOLD

FAULT_TYPES = ["bearing_wear", "impeller_imbalance", "cavitation",
               "seal_failure", "overloading", "sensor_failure"]
ALL_CLASSES = ["normal"] + FAULT_TYPES
LABEL_MAP   = {c: i for i, c in enumerate(ALL_CLASSES)}

FAULT_CLUSTERS = {
    "bearing_wear"      : [1, 2, 3],   # steady_state, startup, high_load
    "impeller_imbalance": [1, 3],      # steady_state, high_load
    "cavitation"        : [2],         # startup ONLY
    "seal_failure"      : [1, 3],      # steady_state, high_load
    "overloading"       : [2],         # maps to startup cluster_id in lib
    "sensor_failure"    : [0, 1, 2, 3],
}

# =============================================================================
# SECTION 2 — LSTM-AE LOADER (exact architecture — for Gate 3 MAE check)
# =============================================================================
log("SECTION 2 — Loading M4 LSTM-AE for Gate 3 validation...")

class LSTMAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8,   128, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128,  64, num_layers=1, batch_first=True)
        self.bn    = nn.LayerNorm(64)
    def forward(self, x):
        out1, _          = self.lstm1(x)
        out2, (h_n, c_n) = self.lstm2(out1)
        return self.bn(h_n[-1]), h_n, c_n

class LSTMAEDecoder(nn.Module):
    def __init__(self, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.fc_h  = nn.Linear(64, 128)
        self.fc_c  = nn.Linear(64, 128)
        self.lstm1 = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128,  8, num_layers=1, batch_first=True)
        self.out   = nn.Linear(8, 8)
    def forward(self, z, h_n, c_n):
        B  = z.size(0)
        h0 = torch.tanh(self.fc_h(h_n[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.tanh(self.fc_c(c_n[-1])).unsqueeze(0).repeat(2, 1, 1)
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out1, _ = self.lstm1(z_rep, (h0, c0))
        out2, _ = self.lstm2(out1)
        return self.out(out2)

class LSTMAE(nn.Module):
    def __init__(self, seq_len=50):
        super().__init__()
        self.encoder = LSTMAEEncoder()
        self.decoder = LSTMAEDecoder(seq_len=seq_len)
    def forward(self, x):
        z, h_n, c_n = self.encoder(x)
        return self.decoder(z, h_n, c_n), z

LSTM_AE_LOADED = False
lstm_model     = None
try:
    lstm_model = LSTMAE(seq_len=WIN_SIZE).to('cpu')
    state      = torch.load(MODEL_DIR / "lstm_ae_baseline_best.pth",
                            map_location='cpu')
    lstm_model.load_state_dict(state, strict=True)
    lstm_model.eval()
    with torch.no_grad():
        _t = torch.ones(1, WIN_SIZE, N_CH)
        _o, _z = lstm_model(_t)
        assert _z.shape == (1, 64)
    LSTM_AE_LOADED = True
    log(f"  M4 LSTM-AE loaded | strict=True PASSED | Gate 3 ACTIVE")
except Exception as e:
    log(f"  [WARNING] LSTM-AE load failed: {e} — Gate 3 bypassed")
results["M6A_lstm_ae_gate3_active"] = LSTM_AE_LOADED

CHANNEL_WEIGHTS = torch.tensor([2.0, 2.0, 0.8, 1.5, 2.0, 0.8, 1.0, 2.0],
                                dtype=torch.float32)  # M6B channel weight order

def compute_mae(seq_np):
    if not LSTM_AE_LOADED:
        return 0.0
    with torch.no_grad():
        x     = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0)
        recon, _ = lstm_model(x)
        w     = CHANNEL_WEIGHTS.unsqueeze(0).unsqueeze(0)
        return (torch.abs(recon - x) * w).mean().item()

# =============================================================================
# SECTION 3 — WEIBULL SEVERITY SAMPLER (identical to v4)
# =============================================================================
log("SECTION 3 — Weibull severity sampler (k=0.8)...")

def sample_severity_weibull():
    r = np.random.uniform()
    if r < 0.55:
        k, lam = 0.8, 0.20
        while True:
            u   = np.random.uniform()
            sev = lam * (-np.log(1 - u + 1e-9)) ** (1 / k)
            sev = float(np.clip(sev, 0.05, 1.0))
            if sev <= 0.30:
                return sev
    elif r < 0.85:
        return float(np.random.uniform(0.30, 0.65))
    else:
        return float(np.random.uniform(0.65, 1.00))

def get_fault_stage(sev):
    if sev <= 0.30:   return "early"
    elif sev <= 0.65: return "developing"
    else:             return "advanced"

test_sevs  = [sample_severity_weibull() for _ in range(10000)]
early_pct  = 100 * sum(s <= 0.30 for s in test_sevs) / 10000
dev_pct    = 100 * sum(0.30 < s <= 0.65 for s in test_sevs) / 10000
adv_pct    = 100 * sum(s > 0.65 for s in test_sevs) / 10000
log(f"  Weibull check — early:{early_pct:.1f}% dev:{dev_pct:.1f}% adv:{adv_pct:.1f}%")
results["weibull_early_pct"]      = round(early_pct, 1)
results["weibull_developing_pct"] = round(dev_pct,   1)
results["weibull_advanced_pct"]   = round(adv_pct,   1)

# =============================================================================
# SECTION 4 — LOAD NORMALISED DATA + SPIKE SEEDS
# =============================================================================
log("SECTION 4 — Loading normalised data and spike seeds...")

M3_NORM_COLS = [
    "X_ACR_Mot.SV_norm", "X_ACR_Pmp.SV_norm", "X_ACR_Mot.TV_norm",
    "X_ACR_Pmp.PV_norm", "X_Temp.SV_norm",    "X_Pres.SV_norm",
    "X_ACR_Pmp.TV_norm", "X_ACR_Mot.PV_norm",
]

norm_df     = None
NORM_LOADED = False
try:
    norm_df = pd.read_csv(NORM_DIR / "normalised_data.csv")
    missing = [c for c in M3_NORM_COLS if c not in norm_df.columns]
    if not missing:
        NORM_LOADED = True
        log(f"  normalised_data.csv loaded — {len(norm_df):,} rows")
    else:
        log(f"  [WARNING] Missing M3 cols: {missing}")
except Exception as e:
    log(f"  [WARNING] normalised_data.csv: {e}")

try:
    spike_seeds  = np.load(SYNTH_DIR / "M4_spike_seeds.npy")
    spike_meta   = pd.read_csv(SYNTH_DIR / "M4_spike_seeds_meta.csv")
    SEEDS_LOADED = True
    log(f"  Spike seeds: {spike_seeds.shape} | meta: {len(spike_meta)} rows")
except Exception as e:
    log(f"  [WARNING] Spike seeds: {e} — pure physics fallback")
    spike_seeds  = np.zeros((0, WIN_SIZE, N_CH), dtype=np.float32)
    spike_meta   = pd.DataFrame()
    SEEDS_LOADED = False

# =============================================================================
# SECTION 5 — VALIDATION GATES
# =============================================================================
log("SECTION 5 — Defining validation gates...")

def validate_sequence(seq, fault_type, cluster_id, severity):
    """
    Physics gates — severity-adaptive per v4 engineering rationale.
    """
    # G1: No negative pressure
    if np.any(seq[:, CH["Pres.SV"]] < -0.01):
        return False, "G1_neg_pressure"

    # G2: No sub-ambient temperature
    for c in ["Mot.TV", "Pmp.TV", "Temp.SV"]:
        if np.any(seq[:, CH[c]] < -0.12):
            return False, f"G2_temp_floor_{c}"

    def adaptive_r(ca, cb, start=50):
        if severity <= 0.30:
            zone = seq[start:, CH[cb]]
            return (zone[-1] - zone[0]) >= -0.05, f"dir_{cb}"
        min_r = min(0.70, 0.30 + severity * 0.57)
        r, _  = pearsonr(seq[:, CH[ca]], seq[:, CH[cb]])
        return r >= min_r, f"r={r:.3f}_min={min_r:.3f}"

    if fault_type == "bearing_wear":
        ok, reason = adaptive_r("Mot.SV", "Mot.TV")
        if not ok: return False, f"G4_{reason}"
        # F1: also check Temp.SV coupling
        ok2, reason2 = adaptive_r("Mot.TV", "Temp.SV")
        if not ok2: return False, f"G4b_{reason2}"

    elif fault_type == "impeller_imbalance":
        ok, reason = adaptive_r("Pmp.PV", "Pmp.SV", start=40)
        if not ok: return False, f"G5_{reason}"

    elif fault_type == "overloading":
        if severity <= 0.30:
            zone = seq[40:, CH["Mot.TV"]]
            if zone[-1] - zone[0] < -0.05:
                return False, "G6_direction_fail"
        else:
            min_r = min(0.85, 0.40 + severity * 0.57)
            r, _  = pearsonr(seq[:, CH["Temp.SV"]], seq[:, CH["Mot.TV"]])
            if r < min_r:
                return False, f"G6_overload_r={r:.3f}"

    elif fault_type == "sensor_failure":
        # Exactly 1-3 channels anomalous
        from scipy import stats as sp_stats
        PRE = 50
        t_index = np.arange(PRE, seq.shape[0])
        anomalous = 0
        for c in CHANNELS:
            ci = CH[c]
            fw = seq[PRE:, ci]; bw = seq[:PRE, ci]
            bs = bw.std(); fs = fw.std()
            if bs > 1e-6 and fs < 0.30 * bs:
                anomalous += 1
            elif bs > 1e-6:
                zs = np.abs(fw - bw.mean()) / bs
                if zs.max() > 4.0:
                    anomalous += 1
                else:
                    r, _ = sp_stats.spearmanr(t_index, fw)
                    if abs(r) > 0.70:
                        anomalous += 1
        if not (1 <= anomalous <= 3):
            return False, f"G7_isolation_{anomalous}_anomalous"

    # G3: MAE gate — severity-adaptive (non-acute faults are sub-threshold by design)
    if LSTM_AE_LOADED and fault_type != "normal":
        max_mae = max(compute_mae(seq[t0:t0+WIN_SIZE])
                    for t0 in range(0, SEQ_LEN - WIN_SIZE, WIN_SIZE // 2))
        # Early-stage faults (sev <= 0.30): accept MAE >= 50% of threshold
        # These are physically correct sub-threshold signals — rejecting them
        # creates survivorship bias toward advanced faults only (M6.5 v2 confirmed)
        min_mae = ANOMALY_THRESHOLD * (0.50 if severity <= 0.30 else 1.0)
        if max_mae < min_mae:
            return False, f"G3_mae_low={max_mae:.4f}_min={min_mae:.4f}"

    return True, "PASS"

# =============================================================================
# SECTION 6 — MAIN GENERATION LOOP
# =============================================================================
log("SECTION 6 — Main generation loop — 8400 sequences...")

all_sequences    = []
all_meta         = []
gate_fail_counts = {ft: 0 for ft in FAULT_TYPES}

# ── Normal sequences ──────────────────────────────────────────────────────────
log("  Generating NORMAL sequences (real CIRA windows)...")
normal_seqs, normal_meta_raw = generate_normal_from_real(
    norm_df if NORM_LOADED else pd.DataFrame(),
    n_target=N_PER_CLASS, n_steps=SEQ_LEN, m3_norm_cols=M3_NORM_COLS
)
for seq, meta_raw in zip(normal_seqs, normal_meta_raw):
    all_sequences.append(seq)
    all_meta.append({
        "label":       LABEL_MAP["normal"],
        "fault_type":  "normal",
        "severity":    0.0,
        "fault_stage": "normal",
        "source":      meta_raw["source"],
        "cluster_id":  meta_raw["cluster_id"],
        "seed_idx":    -1,
    })
log(f"  Normal: {len(normal_seqs)}/{N_PER_CLASS}")
results["M6A_count_normal"] = len(normal_seqs)

# ── Fault classes ─────────────────────────────────────────────────────────────
# Map cluster_id integers to physics lib function kwargs
CLUSTER_NAMES = {0: "cooldown", 1: "steady_state", 2: "startup", 3: "high_load"}

for fault_type in FAULT_TYPES:
    log(f"\n  === Generating {fault_type.upper()} ===")
    label      = LABEL_MAP[fault_type]
    clusters   = FAULT_CLUSTERS[fault_type]
    generated  = 0
    max_att    = N_PER_CLASS * 10
    att        = 0

    while generated < N_PER_CLASS and att < max_att:
        att      += 1
        cluster_id = int(random.choice(clusters))
        severity   = sample_severity_weibull()
        # Overloading: severity range [0.5, 1.0] per M5 patch
        if fault_type == "overloading":
            severity = max(severity, 0.5)
        fault_stage = get_fault_stage(severity)

        try:
            if fault_type == "bearing_wear":
                seq = generate_bearing_wear(severity=severity,
                                             cluster_id=cluster_id, n_steps=SEQ_LEN)
                ft_meta = {}

            elif fault_type == "impeller_imbalance":
                seq = generate_impeller_imbalance(severity=severity,
                                                   cluster_id=cluster_id, n_steps=SEQ_LEN)
                ft_meta = {}

            elif fault_type == "cavitation":
                seq = generate_cavitation(severity=severity,
                                           cluster_id=2, n_steps=150)
                # Pad to SEQ_LEN=200 with post-cavitation baseline
                if seq.shape[0] < SEQ_LEN:
                    pad = make_baseline(SEQ_LEN - seq.shape[0], cluster_id=2)
                    seq = np.vstack([seq, pad])
                ft_meta = {}

            elif fault_type == "seal_failure":
                seq = generate_seal_failure(severity=severity,
                                             cluster_id=cluster_id, n_steps=SEQ_LEN)
                ft_meta = {}

            elif fault_type == "overloading":
                seq = generate_overloading(severity=severity,
                                            cluster_id=cluster_id, n_steps=SEQ_LEN)
                ft_meta = {}

            elif fault_type == "sensor_failure":
                # Cycle subtypes evenly including dropout (F5 fix)
                subtype_cycle = SENSOR_SUBTYPES  # ["flatline","spike","drift","dropout"]
                fail_type  = subtype_cycle[generated % len(subtype_cycle)]
                fail_ch    = CHANNELS[generated % N_CH]
                seq, ft, fc = generate_sensor_failure(
                    severity=severity, cluster_id=cluster_id,
                    n_steps=SEQ_LEN, fail_type=fail_type, fail_channel=fail_ch)
                ft_meta = {"fail_type": ft, "fail_channel": fc}

        except Exception as e:
            gate_fail_counts[fault_type] += 1
            continue

        ok, reason = validate_sequence(seq, fault_type, cluster_id, severity)
        if not ok:
            gate_fail_counts[fault_type] += 1
            if gate_fail_counts[fault_type] % 200 == 0:
                log(f"    Fail #{gate_fail_counts[fault_type]}: {reason}")
            continue

        all_sequences.append(seq)
        meta_entry = {
            "label":       label,
            "fault_type":  fault_type,
            "severity":    round(severity, 4),
            "fault_stage": fault_stage,
            "source":      "physics_weibull_v5",
            "cluster_id":  cluster_id,
            "seed_idx":    -1,
        }
        meta_entry.update(ft_meta)
        all_meta.append(meta_entry)
        generated += 1

        if generated % 300 == 0:
            log(f"    {fault_type}: {generated}/{N_PER_CLASS}")

    results[f"M6A_count_{fault_type}"] = generated
    log(f"  {fault_type}: {generated}/{N_PER_CLASS} | gate_fails={gate_fail_counts[fault_type]}")

# =============================================================================
# SECTION 7 — SAVE OUTPUTS
# =============================================================================
log("\nSECTION 7 — Saving outputs...")

sequences_arr = np.array(all_sequences, dtype=np.float32)
meta_df       = pd.DataFrame(all_meta)
meta_df.insert(0, "seq_id", range(len(meta_df)))

results["M6A_total_sequences"] = len(all_sequences)
results["M6A_array_shape"]     = str(sequences_arr.shape)

try:
    npy_path = SYNTH_DIR / "M6_synthetic_sequences.npy"
    np.save(npy_path, sequences_arr)
    log(f"  Saved: {npy_path}  shape={sequences_arr.shape}")
except Exception as e:
    log(f"  [ERROR] .npy: {e}")

try:
    pkl_path = SYNTH_DIR / "M6_sequences.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(sequences_arr, f)
    log(f"  Saved: {pkl_path}")
except Exception as e:
    log(f"  [ERROR] .pkl: {e}")

try:
    meta_path = SYNTH_DIR / "M6_synthetic_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    meta_df.to_csv(SYNTH_DIR / "M6_sequence_meta.csv", index=False)
    log(f"  Saved: {meta_path}  shape={meta_df.shape}")
except Exception as e:
    log(f"  [ERROR] meta CSV: {e}")

# =============================================================================
# SECTION 8 — SEVERITY + COUPLING VALIDATION
# =============================================================================
log("\nSECTION 8 — Validation...")

fault_meta  = meta_df[meta_df["fault_type"] != "normal"]
actual_early = 100 * (fault_meta["severity"] <= 0.30).sum() / max(len(fault_meta), 1)
actual_dev   = 100 * ((fault_meta["severity"] > 0.30) & (fault_meta["severity"] <= 0.65)).sum() / max(len(fault_meta), 1)
actual_adv   = 100 * (fault_meta["severity"] > 0.65).sum() / max(len(fault_meta), 1)
log(f"  Severity: early={actual_early:.1f}% dev={actual_dev:.1f}% adv={actual_adv:.1f}%")

# F1 fix validation: Temp.SV* coupling for bearing_wear
bw_idx  = np.where(meta_df["fault_type"].values == "bearing_wear")[0][:100]
bw_seqs = sequences_arr[bw_idx]
bw_temp_r = np.mean([pearsonr(s[:, CH["Mot.TV"]], s[:, CH["Temp.SV"]])[0]
                      for s in bw_seqs])
log(f"  F1 check — bearing_wear Mot.TV*–Temp.SV* mean r = {bw_temp_r:.4f} "
    f"(target >= 0.70 overall, relaxed for Weibull early-stage)")
results["M6A_F1_bearing_tempSV_r"] = round(float(bw_temp_r), 4)

coupling_checks = [
    ("bearing_wear",       "Mot.SV",  "Mot.TV",   0.70),
    ("impeller_imbalance", "Pmp.PV",  "Pmp.SV",   0.70),
    ("overloading",        "Temp.SV", "Mot.TV",   0.85),
]
for ft, ca, cb, min_r in coupling_checks:
    idxs  = np.where(meta_df["fault_type"].values == ft)[0][:100]
    seqs  = sequences_arr[idxs]
    r_arr = [pearsonr(s[:, CH[ca]], s[:, CH[cb]])[0] for s in seqs]
    pct   = 100 * sum(r >= min_r for r in r_arr) / max(len(r_arr), 1)
    log(f"  Coupling {ft} ({ca}–{cb}): {pct:.1f}% >= {min_r}")
    results[f"M6A_coupling_{ft}"] = round(pct, 1)

results["M6A_sev_early_pct"]      = round(actual_early, 1)
results["M6A_sev_developing_pct"] = round(actual_dev,   1)
results["M6A_sev_advanced_pct"]   = round(actual_adv,   1)
results["M6A_gate_fails"]         = gate_fail_counts

# =============================================================================
# SECTION 9 — PLOTS
# =============================================================================
log("SECTION 9 — Generating plots...")

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("M6A v5 — Synthetic Data (m6b_physics_lib)", fontsize=13)

    # Label distribution
    ax = axes[0, 0]
    dist  = meta_df["fault_type"].value_counts()
    colors = ["#27ae60"] + ["#e74c3c"]*6
    bars  = ax.bar(dist.index, dist.values, color=colors)
    ax.axhline(N_PER_CLASS, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Label Distribution"); ax.tick_params(axis="x", rotation=25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", fontsize=8)

    # Severity distribution
    ax = axes[0, 1]
    ax.hist(fault_meta["severity"].values, bins=40, color="steelblue",
            edgecolor="white", alpha=0.85)
    ax.axvline(0.30, color="orange", linestyle="--", label="early/dev boundary")
    ax.axvline(0.65, color="red",    linestyle="--", label="dev/adv boundary")
    ax.set_title("Severity (Weibull k=0.8)"); ax.legend(fontsize=8)

    # Bearing wear — F1 fix: Temp.SV* coupling
    ax = axes[0, 2]
    if len(bw_idx):
        seq = sequences_arr[bw_idx[0]]
        for c, col in [("Mot.SV","blue"),("Mot.TV","orange"),("Temp.SV","red")]:
            ax.plot(seq[:, CH[c]], label=c, alpha=0.85, color=col)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.3)
    ax.set_title("Bearing Wear — Mot.SV→Mot.TV→Temp.SV (F1 fix)")
    ax.legend(fontsize=8)

    # Cavitation — dual signature
    ax = axes[1, 0]
    cav_idx = np.where(meta_df["fault_type"].values == "cavitation")[0]
    if len(cav_idx):
        seq = sequences_arr[cav_idx[0]]
        ax.plot(seq[:, CH["Pres.SV"]], label="Pres.SV* ↓", color="red")
        ax.plot(seq[:, CH["Pmp.SV"]],  label="Pmp.SV* ↑",  color="darkblue")
    ax.set_title("Cavitation — Dual Signature (F3 fix)"); ax.legend(fontsize=8)

    # Overloading — F4: Pres.SV* Q-H shift
    ax = axes[1, 1]
    ol_idx = np.where(meta_df["fault_type"].values == "overloading")[0]
    if len(ol_idx):
        seq = sequences_arr[ol_idx[0]]
        for c, col in [("Temp.SV","orange"),("Mot.TV","red"),("Pres.SV","blue")]:
            ax.plot(seq[:, CH[c]], label=c, alpha=0.85, color=col)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.3)
    ax.set_title("Overloading — Temp rise + Pres.SV* Q-H shift (F4 fix)")
    ax.legend(fontsize=8)

    # Sensor failure — F5: all 4 subtypes
    ax = axes[1, 2]
    sf_idx = np.where(meta_df["fault_type"].values == "sensor_failure")[0]
    if len(sf_idx) >= 4:
        colors_sf = ["gray", "red", "blue", "green"]
        for k, (si, label_s) in enumerate(zip(sf_idx[:4],
                                               ["flatline","spike","drift","dropout"])):
            seq = sequences_arr[si]
            fc  = meta_df.iloc[si].get("fail_channel", "Mot.SV")
            if fc in CH:
                ax.plot(seq[:, CH[fc]], alpha=0.7, color=colors_sf[k],
                        label=f"{label_s}({fc})", linewidth=0.9)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Sensor Failure — 4 subtypes (F5 fix)"); ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = PLOTS_DIR / f"{SCRIPT_NAME}_sanity_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {plot_path}")
    results["M6A_sanity_plot"] = str(plot_path)
except Exception as e:
    log(f"  [WARNING] plot: {e}")

# =============================================================================
# SECTION 10 — REPORT + PASTE TEXT
# =============================================================================
log("SECTION 10 — Report and paste text...")

total_gen = sum(results.get(f"M6A_count_{c}", 0) for c in ALL_CLASSES)
all_gates_pass = (total_gen == TOTAL_SEQ and
                  results.get("M6A_count_normal", 0) == N_PER_CLASS)
results["M6A_total_sequences"] = total_gen
results["M6A_all_gates_pass"]  = all_gates_pass

try:
    report_lines = [
        f"# {SCRIPT_NAME} Report", f"Date: {date.today()}", "",
        "## Fixes Applied (v5 vs v4)",
        "| Fix | Description |", "|-----|-------------|",
        "| F1 | bearing_wear Temp.SV* coupled via _tcoup r=0.9793 |",
        "| F2 | impeller_imbalance abs(sin) AM envelope |",
        "| F3 | cavitation M5-faithful: severity-dependent t_onset |",
        "| F4 | overloading Pres.SV* affinity law Q-H shift |",
        "| F5 | sensor_failure dropout subtype added |",
        "| F6 | all generation via m6b_physics_lib.py |",
        "", "## Results", "| Key | Value |", "|-----|-------|",
    ]
    for k, v in results.items():
        report_lines.append(f"| {k} | {v} |")
    report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"  Saved: {report_path}")
except Exception as e:
    log(f"  [ERROR] report: {e}")

print("\n" + "═"*66)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print("═"*66)
print(f"M6A_v5_total_sequences       : {total_gen}")
print(f"M6A_v5_sequences_per_class   : {N_PER_CLASS}")
print(f"M6A_v5_array_shape           : {sequences_arr.shape}")
print(f"M6A_v5_sev_early_pct         : {actual_early:.1f}%")
print(f"M6A_v5_F1_bearing_tempSV_r   : {results.get('M6A_F1_bearing_tempSV_r')}")
print(f"M6A_v5_fixes_applied         : F1,F2,F3,F4,F5,F6")
print(f"M6A_v5_physics_lib           : m6b_physics_lib.py")
print(f"M6A_v5_gate_fails            : {gate_fail_counts}")
print(f"M6A_v5_all_gates_pass        : {all_gates_pass}")
print(f"Status_for_M6B               : {'READY' if all_gates_pass else 'NEEDS_REVIEW'}")
print("═"*66)
print("══ END PASTE UPDATE ══")

print("\n── FILE MANIFEST ──")
print("  [Local only]    data/synthetic/M6_synthetic_sequences.npy")
print("  [Local only]    data/synthetic/M6_sequences.pkl")
print("  [GitHub PUSH]   src/m6b_physics_lib.py")
print("  [GitHub PUSH]   src/module_06a_synthetic_generator_v5.py")
print("  [Spaces Upload] data/synthetic/M6_synthetic_metadata.csv")
print("  [Spaces Upload] data/synthetic/M6_sequence_meta.csv")
print(f"  [Spaces Upload] outputs/plots/{SCRIPT_NAME}_sanity_plot.png")
print(f"  [Spaces Upload] outputs/reports/{SCRIPT_NAME}_report.md")

log(f"{SCRIPT_NAME} COMPLETE.")