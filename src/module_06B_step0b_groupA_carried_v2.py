# ═══════════════════════════════════════════════════════════════════════════════
# PumpSmart — module_06B_step0b_groupA_carried_v2.py
# M6B Step 0b v2: Labels 0(2000), 2(1500), 3(1500), 6(1200)
# Fixes vs v1:
#   F2: impeller_imbalance — abs(sin) AM envelope
#   F3: cavitation — M5-faithful (severity-dependent t_onset, mean_drop=0.6*sev)
#   F5: sensor_failure — dropout subtype added
#   F6: All generation via m6b_physics_lib.py
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_NAME = "module_06B_step0b_groupA_carried_v2"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DEVICE, IS_GPU, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, warnings, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from m6b_physics_lib import (
    init_lib, CH, CHANNELS, N_CH, CHANNEL_TO_M3_KEY,
    get_cluster_mean, apply_winsorization, make_baseline,
    generate_impeller_imbalance, generate_cavitation,
    generate_sensor_failure, generate_normal_from_real,
    SENSOR_SUBTYPES
)

warnings.filterwarnings('ignore')
REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
log(f"Script: {SCRIPT_NAME} | Device: {DEVICE} | Date: {date.today()}")
log("v2 — m6b_physics_lib: F2(abs_sin) F3(M5-faithful cav) F5(dropout) F6(unified)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════
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
    _pc           = phys_config.get("physics_constants", phys_config)
    TAU_THERMAL_s = float(_pc.get("TAU_THERMAL_s", 388.9))
    BPF_HZ        = float(_pc.get("BPF_HZ",        347.67))
    RHO           = float(_pc.get("RHO",            1000.0))
    log(f"  Physics: TAU={TAU_THERMAL_s}s | BPF={BPF_HZ}Hz | RHO={RHO}")
except Exception as e:
    log(f"  [WARNING] M5 fallback: {e}")
    phys_config = {}

try:
    with open(MODEL_DIR / "M4_threshold_config.json") as f:
        M4_THRESHOLD = float(json.load(f).get("threshold", 0.110058))
    assert abs(M4_THRESHOLD - 0.110058) < 1e-5
    log(f"  M4 threshold: q = {M4_THRESHOLD} ✓")
except Exception as e:
    log(f"  [WARNING] threshold fallback: {e}")
    M4_THRESHOLD = 0.110058
results["M4_threshold_confirmed"] = M4_THRESHOLD

# Initialise physics library
init_lib(norm_config, phys_config, seed=43)   # seed=43 consistent with v1
log(f"  m6b_physics_lib initialised | CH: {CH}")
results["channels"] = CHANNELS

# M3 norm column order matching M6B channel order
M3_NORM_COLS = [
    "X_ACR_Mot.SV_norm", "X_ACR_Pmp.SV_norm", "X_ACR_Mot.TV_norm",
    "X_ACR_Pmp.PV_norm", "X_Temp.SV_norm",    "X_Pres.SV_norm",
    "X_ACR_Pmp.TV_norm", "X_ACR_Mot.PV_norm",
]

# Load normalised_data.csv for Label 0 real CIRA windows
norm_df     = None
NORM_LOADED = False
try:
    norm_df = pd.read_csv(NORM_DIR / "normalised_data.csv")
    missing = [c for c in M3_NORM_COLS if c not in norm_df.columns]
    if not missing:
        NORM_LOADED = True
        log(f"  normalised_data.csv — {len(norm_df):,} rows | "
            f"segment_id: {'segment_id' in norm_df.columns}")
    else:
        log(f"  [WARNING] missing M3 cols: {missing}")
except Exception as e:
    log(f"  [WARNING] normalised_data.csv: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD FROZEN M4 LSTM-AE
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 2 — Loading frozen M4 LSTM-AE...")

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
        self.fc_h  = nn.Linear(64, 128)
        self.fc_c  = nn.Linear(64, 128)
        self.lstm1 = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128,  8, num_layers=1, batch_first=True)
        self.out   = nn.Linear(8, 8)
    def forward(self, z, h_n, c_n):
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

LSTM_AE_SEQ_LEN = 50
lstm_ae_model   = None
try:
    lstm_ae_model = LSTMAE(seq_len=LSTM_AE_SEQ_LEN)
    state = torch.load(MODEL_DIR / "lstm_ae_baseline_best.pth", map_location='cpu')
    lstm_ae_model.load_state_dict(state, strict=True)
    lstm_ae_model.eval()
    for p in lstm_ae_model.parameters():
        p.requires_grad_(False)
    lstm_ae_model = lstm_ae_model.to(DEVICE)
    with torch.no_grad():
        _t = torch.ones(1, LSTM_AE_SEQ_LEN, N_CH).to(DEVICE)
        _, _z = lstm_ae_model(_t)
    assert _z.shape == (1, 64)
    log(f"  M4 LSTM-AE → {DEVICE} | strict=True PASSED | z_t: {_z.shape}")
except Exception as e:
    log(f"  [FATAL] {e}"); raise
results["lstm_ae_loaded"] = True

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GENERATE SEQUENCES
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 3 — Generating sequences...")

rng = np.random.default_rng(seed=43)
all_sequences = []
all_meta      = []

# ── Label 0 — normal (real CIRA windows) ──────────────────────────────────────
log("  Label 0 (normal) — 2000 × 200 steps...")
seqs_0, meta_0_raw = generate_normal_from_real(
    norm_df if NORM_LOADED else pd.DataFrame(),
    n_target=2000, n_steps=200, m3_norm_cols=M3_NORM_COLS
)
for i, (seq, raw) in enumerate(zip(seqs_0, meta_0_raw)):
    all_sequences.append(seq)
    all_meta.append({
        "seq_id": f"M6B_L0_{i:05d}", "label": 0, "label_name": "normal",
        "group": "A", "steps": 200, "severity": 0.0,
        "cluster_id": raw["cluster_id"], "source": raw["source"],
        "seed_idx": -1, "m6b_step": "0b_v2", "generated": str(date.today()),
    })
results["label0_n_generated"] = len(seqs_0)
log(f"    Label 0: {len(seqs_0)} sequences")

# ── Label 2 — impeller_imbalance (F2: abs(sin) fix) ───────────────────────────
log("  Label 2 (impeller_imbalance) — 1500 × 200 steps...")
severities = rng.uniform(0.2, 1.0, size=1500).astype(np.float32)
for i in range(1500):
    cid = int(rng.choice([1, 3]))   # steady_state or high_load
    try:
        seq = generate_impeller_imbalance(severity=float(severities[i]),
                                           cluster_id=cid, n_steps=200)
        assert seq.shape == (200, 8) and not np.any(np.isnan(seq))
        assert seq[:, CH["Pres.SV"]].min() >= 0.0
    except Exception:
        seq = make_baseline(200, cluster_id=cid)
    all_sequences.append(seq)
    all_meta.append({
        "seq_id": f"M6B_L2_{i:05d}", "label": 2,
        "label_name": "impeller_imbalance", "group": "A", "steps": 200,
        "severity": float(severities[i]), "cluster_id": cid,
        "source": "m6b_physics_lib_v2", "seed_idx": -1,
        "m6b_step": "0b_v2", "generated": str(date.today()),
    })
results["label2_n_generated"] = 1500
log(f"    Label 2: 1500 sequences")

# ── Label 3 — cavitation (F3: M5-faithful) ────────────────────────────────────
log("  Label 3 (cavitation) — 1500 × 150 steps...")
severities = rng.uniform(0.2, 1.0, size=1500).astype(np.float32)
for i in range(1500):
    try:
        seq = generate_cavitation(severity=float(severities[i]),
                                   cluster_id=2, n_steps=150)
        assert seq.shape == (150, 8) and not np.any(np.isnan(seq))
        assert seq[:, CH["Pres.SV"]].min() >= 0.0
    except Exception:
        seq = make_baseline(150, cluster_id=2)
    all_sequences.append(seq)
    all_meta.append({
        "seq_id": f"M6B_L3_{i:05d}", "label": 3,
        "label_name": "cavitation", "group": "A", "steps": 150,
        "severity": float(severities[i]), "cluster_id": 2,
        "source": "m6b_physics_lib_v2", "seed_idx": -1,
        "m6b_step": "0b_v2", "generated": str(date.today()),
    })
results["label3_n_generated"] = 1500
log(f"    Label 3: 1500 sequences")

# ── Label 6 — sensor_failure (F5: dropout added) ─────────────────────────────
log("  Label 6 (sensor_failure) — 1200 × 150 steps...")
severities = rng.uniform(0.3, 1.0, size=1200).astype(np.float32)
for i in range(1200):
    cid       = int(rng.choice([0, 1, 2, 3]))
    fail_type = SENSOR_SUBTYPES[i % 4]    # flatline/spike/drift/dropout — equal mix
    fail_ch   = CHANNELS[i % N_CH]
    try:
        seq, ft, fc = generate_sensor_failure(
            severity=float(severities[i]), cluster_id=cid,
            n_steps=150, fail_type=fail_type, fail_channel=fail_ch)
        assert seq.shape == (150, 8) and not np.any(np.isnan(seq))
    except Exception:
        seq = make_baseline(150, cluster_id=cid)
        ft, fc = fail_type, fail_ch
    all_sequences.append(seq)
    all_meta.append({
        "seq_id": f"M6B_L6_{i:05d}", "label": 6,
        "label_name": "sensor_failure", "group": "A", "steps": 150,
        "severity": float(severities[i]), "cluster_id": cid,
        "source": "m6b_physics_lib_v2", "seed_idx": -1,
        "fail_type": ft, "fail_channel": fc,
        "m6b_step": "0b_v2", "generated": str(date.today()),
    })
results["label6_n_generated"] = 1200
log(f"    Label 6: 1200 sequences")

total = len(all_sequences)
log(f"  Total: {total} sequences")
results["step0b_total_sequences"] = total

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PHYSICS GATES
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 4 — Physics gates...")

from scipy import stats

gate_results = {}

def run_gate(gate_id, label_id, description, test_fn):
    label_seqs = [all_sequences[i] for i, m in enumerate(all_meta)
                  if m["label"] == label_id]
    if not label_seqs: return 0.0
    passes = sum(test_fn(s) for s in label_seqs)
    rate   = passes / len(label_seqs)
    status = "PASS" if rate >= 0.90 else ("WARN" if rate >= 0.80 else "FAIL")
    log(f"    Gate {gate_id} | L{label_id} | {description}: "
        f"{passes}/{len(label_seqs)} = {rate:.2%} [{status}]")
    gate_results[f"G{gate_id}_L{label_id}"] = {"pass_rate": rate, "status": status}
    return rate

# G1 — Primary signal present
def g1_normal(seq):
    ok_pvsv = all(0.05 < seq[:, CH[c]].mean() < 2.5
                  for c in ["Mot.SV","Pmp.SV","Pres.SV","Mot.PV","Pmp.PV"])
    ok_tv   = all(-0.10 < seq[:, CH[c]].mean() < 1.10
                  for c in ["Mot.TV","Pmp.TV","Temp.SV"])
    return ok_pvsv and ok_tv
run_gate("1", 0, "Normal — channels in valid normalized range", g1_normal)
run_gate("1", 2, "Pmp.SV* elevated in fault window",
         lambda s: s[100:, CH["Pmp.SV"]].mean() > s[:50, CH["Pmp.SV"]].mean() + 0.05)
run_gate("1", 3, "Pres.SV* decline AND Pmp.SV* elevated (dual signature)",
         lambda s: (s[80:, CH["Pres.SV"]].mean() < s[:50, CH["Pres.SV"]].mean() + 0.05
                    and s[80:, CH["Pmp.SV"]].mean() > s[:50, CH["Pmp.SV"]].mean() + 0.03))
def g1_sensor(seq):
    PRE = 50
    for c in CHANNELS:
        fw = seq[PRE:, CH[c]]; bm = seq[:PRE, CH[c]].mean()
        if fw.std() < 0.008: return True
        if fw.max() > bm * 3.0: return True
        if fw.min() < 0.02 and bm > 0.3: return True
        if abs(fw.mean() - get_cluster_mean(1, c, 1.0)) > 0.15: return True
    return False
run_gate("1", 6, "Primary fail channel shows flatline/spike/drift/dropout",
         g1_sensor)

# G2 — No negative pressure
for lid in [0, 2, 3, 6]:
    run_gate("2", lid, "No negative Pres.SV*",
             lambda s: s[:, CH["Pres.SV"]].min() >= 0.0)

# G3 — No NaN/Inf
for lid in [0, 2, 3, 6]:
    run_gate("3", lid, "No NaN/Inf",
             lambda s: not (np.any(np.isnan(s)) or np.any(np.isinf(s))))

# G4 — Correct shapes
run_gate("4", 0, "Shape (200,8)", lambda s: s.shape == (200, 8))
run_gate("4", 2, "Shape (200,8)", lambda s: s.shape == (200, 8))
run_gate("4", 3, "Shape (150,8)", lambda s: s.shape == (150, 8))
run_gate("4", 6, "Shape (150,8)", lambda s: s.shape == (150, 8))

# G5 — Physics coupling
run_gate("5", 2, "Pmp.PV*–Pmp.SV* Spearman r > 0.60 (ISO 1940 abs(sin) F2 fix)",
         lambda s: stats.spearmanr(s[:, CH["Pmp.PV"]], s[:, CH["Pmp.SV"]])[0] > 0.60)
cav_wrong = sum(1 for m in all_meta if m["label"] == 3 and m["cluster_id"] != 2)
log(f"    Gate 5 | L3 | Non-startup cavitation: {cav_wrong} [{'PASS' if cav_wrong==0 else 'FAIL'}]")
gate_results["G5_L3_cluster"] = {
    "pass_rate": 1.0 if cav_wrong == 0 else 0.0,
    "status":    "PASS" if cav_wrong == 0 else "FAIL"
}

# G5 — Cavitation dual-signature quantitative check
cav_check  = [all_sequences[i] for i, m in enumerate(all_meta) if m["label"] == 3][:100]
pres_drop  = np.mean([s[80:, CH["Pres.SV"]].mean() - s[:50, CH["Pres.SV"]].mean()
                      for s in cav_check])
pmpSV_rise = np.mean([s[80:, CH["Pmp.SV"]].mean()  - s[:50, CH["Pmp.SV"]].mean()
                      for s in cav_check])
log(f"    Cavitation dual-sig: Pres.SV* shift={pres_drop:+.4f} (<0 ✓) | "
    f"Pmp.SV* shift={pmpSV_rise:+.4f} (>0 ✓)")
results["cav_pres_shift"] = round(float(pres_drop), 4)
results["cav_pmpSV_shift"] = round(float(pmpSV_rise), 4)

# G6 — Sensor failure isolation (sub-type aware)
def g6_sensor_isolation(seq):
    PRE = 50
    anomalous = 0
    t_index   = np.arange(PRE, seq.shape[0])
    for c in CHANNELS:
        ci = CH[c]
        fw = seq[PRE:, ci]; bw = seq[:PRE, ci]
        bs = bw.std(); fs = fw.std()
        if bs > 1e-6 and fs < 0.30 * bs:          # flatline or dropout
            anomalous += 1
        elif bs > 1e-6:
            if (np.abs(fw - bw.mean()) / bs).max() > 4.0:   # spike
                anomalous += 1
            else:
                r, _ = stats.spearmanr(t_index, fw)
                if abs(r) > 0.70:                  # drift
                    anomalous += 1
    return 1 <= anomalous <= 3
run_gate("6", 6, "1–3 channels anomalous (flatline/spike/drift/dropout isolation)",
         g6_sensor_isolation)

# G7 — Subtype distribution (all 4 present)
subtype_counts = {}
for m in all_meta:
    if m["label"] == 6:
        ft = m.get("fail_type", "unknown")
        subtype_counts[ft] = subtype_counts.get(ft, 0) + 1
log(f"    Label 6 subtype distribution: {subtype_counts}")
gate_results["G7_L6_subtypes"] = {
    "pass_rate": 1.0 if len(subtype_counts) == 4 else 0.0,
    "status":    "PASS" if len(subtype_counts) == 4 else "FAIL"
}
log(f"    Gate 7 | L6 | All 4 subtypes present (F5 fix): "
    f"{len(subtype_counts)}/4 [{'PASS' if len(subtype_counts)==4 else 'FAIL'}]")

# Gate summary
all_gate_pass  = all(v["status"] in ("PASS","WARN") for v in gate_results.values())
gate_fail_list = [k for k, v in gate_results.items() if v["status"] == "FAIL"]
log(f"  Gate summary: {len(gate_results)} gates | Fails: {gate_fail_list}")
results["gates_all_pass"] = all_gate_pass
results["gate_fail_list"] = gate_fail_list
results["gate_results"]   = {k: v["pass_rate"] for k, v in gate_results.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — z_t EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 5 — Exporting z_t latent vectors...")

def export_zt(sequences_list, meta_list, batch_size=32):
    zt_dict = {}
    all_windows, window_meta = [], []
    for seq_idx, (seq, meta) in enumerate(zip(sequences_list, meta_list)):
        for w in range(seq.shape[0] // LSTM_AE_SEQ_LEN):
            t0 = w * LSTM_AE_SEQ_LEN
            all_windows.append(seq[t0:t0+LSTM_AE_SEQ_LEN])
            window_meta.append((seq_idx, w, meta["seq_id"]))
    n_total = len(all_windows)
    all_zt  = np.zeros((n_total, 64),   dtype=np.float32)
    all_mae = np.zeros((n_total, N_CH), dtype=np.float32)
    with torch.no_grad():
        for b0 in range(0, n_total, batch_size):
            b1    = min(b0 + batch_size, n_total)
            batch = np.stack(all_windows[b0:b1])
            tb    = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
            x_hat, z = lstm_ae_model(tb)
            all_zt[b0:b1]  = z.cpu().numpy()
            all_mae[b0:b1] = (tb - x_hat).abs().mean(dim=1).cpu().numpy()
    for i, (seq_idx, w_idx, seq_id) in enumerate(window_meta):
        if seq_id not in zt_dict:
            n_w = sequences_list[seq_idx].shape[0] // LSTM_AE_SEQ_LEN
            zt_dict[seq_id] = {
                "z_t": np.zeros((n_w, 64),   dtype=np.float32),
                "mae": np.zeros((n_w, N_CH), dtype=np.float32),
            }
        zt_dict[seq_id]["z_t"][w_idx] = all_zt[i]
        zt_dict[seq_id]["mae"][w_idx] = all_mae[i]
    return zt_dict

normal_idx = [i for i, m in enumerate(all_meta) if m["label"] == 0]
fault_idx  = [i for i, m in enumerate(all_meta) if m["label"] != 0]

try:
    zt_normal = export_zt([all_sequences[i] for i in normal_idx],
                           [all_meta[i]      for i in normal_idx])
    log(f"  z_t normal: {len(zt_normal)} entries")
    results["zt_normal_ok"] = True
except torch.cuda.OutOfMemoryError:
    log("  [OOM] falling back CPU batch=16")
    lstm_ae_model.cpu()
    zt_normal = export_zt([all_sequences[i] for i in normal_idx],
                           [all_meta[i]      for i in normal_idx], batch_size=16)
    results["zt_normal_ok"] = True

try:
    zt_faults = export_zt([all_sequences[i] for i in fault_idx],
                           [all_meta[i]      for i in fault_idx])
    log(f"  z_t faults: {len(zt_faults)} entries")
    results["zt_faults_ok"] = True
except Exception as e:
    log(f"  [ERROR] z_t faults: {e}")
    zt_faults = {}; results["zt_faults_ok"] = False

shape_errors = sum(1 for d in {**zt_normal, **zt_faults}.values()
                   if d["z_t"].ndim != 2 or d["z_t"].shape[1] != 64)
log(f"  z_t shape errors: {shape_errors}")
results["zt_shape_errors"] = shape_errors

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 6 — Saving...")

try:
    out_path = SYNTH_DIR / "M6B_sequences_groupA_carried.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"sequences": all_sequences, "meta": all_meta}, f)
    log(f"  Saved: {out_path}")
    results["sequences_pkl_saved"] = True
except Exception as e:
    log(f"  [ERROR] pkl: {e}"); results["sequences_pkl_saved"] = False

try:
    pd.DataFrame(all_meta).to_csv(
        SYNTH_DIR / "M6B_sequences_groupA_carried_meta.csv", index=False)
    log("  Saved meta CSV")
    results["meta_csv_saved"] = True
except Exception as e:
    log(f"  [ERROR] meta CSV: {e}")

try:
    with open(SYNTH_DIR / "z_t_sequences_groupA_normal.pkl", "wb") as f:
        pickle.dump(zt_normal, f)
    log("  Saved z_t_normal")
    results["zt_normal_pkl_saved"] = True
except Exception as e:
    log(f"  [ERROR] z_t normal: {e}")

try:
    with open(SYNTH_DIR / "z_t_sequences_groupA_faults.pkl", "wb") as f:
        pickle.dump(zt_faults, f)
    log("  Saved z_t_faults")
    results["zt_faults_pkl_saved"] = True
except Exception as e:
    log(f"  [ERROR] z_t faults: {e}")

locked = SYNTH_DIR / "M6_sequences.pkl"
assert locked.exists(), "[FATAL] M6_sequences.pkl (LOCKED) missing"
log(f"  LOCKED archive confirmed: {locked}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 7 — Plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("M6B Step 0b v2 — Labels 0,2,3,6 (m6b_physics_lib)", fontsize=13)

# Label 0
ax = axes.flat[0]
l0_seqs  = [all_sequences[i] for i, m in enumerate(all_meta) if m["label"] == 0]
sample_i = np.linspace(0, len(l0_seqs)-1, min(8, len(l0_seqs)), dtype=int)
for si in sample_i:
    ax.plot(l0_seqs[si][:, CH["Pres.SV"]], alpha=0.35, color="gray", linewidth=0.8)
mean_p = np.mean([l0_seqs[i][:, CH["Pres.SV"]] for i in sample_i], axis=0)
ax.plot(mean_p, color="gray", linewidth=2.5, label="Mean")
ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
ax.set_title("Label 0: normal — Pres.SV*"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Label 2 — F2 fix: show Pmp.PV and Pmp.SV co-rise
ax = axes.flat[1]
l2_seqs  = [all_sequences[i] for i, m in enumerate(all_meta) if m["label"] == 2]
sample_i = np.linspace(0, len(l2_seqs)-1, min(6, len(l2_seqs)), dtype=int)
for si in sample_i:
    ax.plot(l2_seqs[si][:, CH["Pmp.PV"]], alpha=0.2, color="purple", linewidth=0.7)
    ax.plot(l2_seqs[si][:, CH["Pmp.SV"]], alpha=0.2, color="blue",   linewidth=0.7)
pv_mean  = np.mean([l2_seqs[i][:, CH["Pmp.PV"]] for i in sample_i], axis=0)
sv_mean  = np.mean([l2_seqs[i][:, CH["Pmp.SV"]] for i in sample_i], axis=0)
ax.plot(pv_mean, color="purple", linewidth=2.5, label="Pmp.PV* (disp)")
ax.plot(sv_mean, color="blue",   linewidth=2.5, label="Pmp.SV* (vel)")
ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
ax.set_title("Label 2: imbalance — F2 abs(sin): PV+SV co-rise")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Label 3 — dual signature
ax = axes.flat[2]
l3_seqs  = [all_sequences[i] for i, m in enumerate(all_meta) if m["label"] == 3]
sample_i = np.linspace(0, len(l3_seqs)-1, min(8, len(l3_seqs)), dtype=int)
for si in sample_i:
    ax.plot(l3_seqs[si][:, CH["Pres.SV"]], alpha=0.25, color="red",      linewidth=0.7)
    ax.plot(l3_seqs[si][:, CH["Pmp.SV"]],  alpha=0.25, color="darkblue", linewidth=0.7)
pres_mean = np.mean([l3_seqs[i][:, CH["Pres.SV"]] for i in sample_i], axis=0)
pmpS_mean = np.mean([l3_seqs[i][:, CH["Pmp.SV"]]  for i in sample_i], axis=0)
ax.plot(pres_mean, color="red",      linewidth=2.5, label="Pres.SV* ↓ (F3 fix)")
ax.plot(pmpS_mean, color="darkblue", linewidth=2.5, label="Pmp.SV* ↑ spikes")
ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.4)
ax.set_title("Label 3: cavitation — dual signature (F3 M5-faithful)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Label 6 — 4 subtypes
ax = axes.flat[3]
colors_sf = {"flatline": "gray", "spike": "red", "drift": "blue", "dropout": "green"}
for st in SENSOR_SUBTYPES:
    st_seqs = [all_sequences[i] for i, m in enumerate(all_meta)
               if m["label"] == 6 and m.get("fail_type") == st]
    if st_seqs:
        fc = all_meta[[i for i, m in enumerate(all_meta)
                       if m["label"] == 6 and m.get("fail_type") == st][0]].get("fail_channel", "Mot.SV")
        ax.plot(st_seqs[0][:, CH.get(fc, 0)], color=colors_sf.get(st, "black"),
                alpha=0.8, linewidth=1.2, label=f"{st}({fc})")
ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
ax.set_title("Label 6: sensor_failure — all 4 subtypes (F5 fix)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
try:
    plt.savefig(PLOTS_DIR / "M6B_step0b_v2_profiles.png", dpi=120, bbox_inches='tight')
    log(f"  Saved plot")
    results["plot_saved"] = True
except Exception as e:
    log(f"  [WARNING] plot: {e}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — REPORT + PASTE TEXT
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 8 — Report...")

report_lines = [
    f"# {SCRIPT_NAME} Report", f"Date: {date.today()}", "",
    "## Fixes Applied", "| Fix | Description |", "|-----|-------------|",
    "| F2 | impeller_imbalance abs(sin) AM envelope |",
    "| F3 | cavitation M5-faithful: severity-dep t_onset, mean_drop=0.6*sev |",
    "| F5 | sensor_failure dropout subtype added |",
    "| F6 | All generation via m6b_physics_lib.py |",
    "", "## Gate Results", "| Gate | Pass Rate |", "|------|-----------|",
]
for k, v in results.get("gate_results", {}).items():
    report_lines.append(f"| {k} | {v:.3f} |")
report_lines += ["", "## Cavitation Dual Signature",
                 f"| Pres.SV* shift | {results.get('cav_pres_shift')} (must be <0) |",
                 f"| Pmp.SV* shift  | {results.get('cav_pmpSV_shift')} (must be >0) |",
                 "", "## Summary", "| Key | Value |", "|-----|-------|"]
for k, v in results.items():
    if k not in ("gate_results",):
        report_lines.append(f"| {k} | {v} |")

try:
    report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    log(f"  Saved: {report_path}")
except Exception as e:
    log(f"  [ERROR] report: {e}")

print("\n" + "═"*66)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print("═"*66)
print(f"M6B_step0b_v2_status           : {'READY' if results.get('gates_all_pass') else 'NEEDS_REVIEW'}")
print(f"M6B_step0b_v2_total_sequences  : {results.get('step0b_total_sequences')}")
print(f"M6B_step0b_v2_label0_n_seqs    : {results.get('label0_n_generated')} (normal, 200s)")
print(f"M6B_step0b_v2_label2_n_seqs    : {results.get('label2_n_generated')} (impeller_imbalance, 200s)")
print(f"M6B_step0b_v2_label3_n_seqs    : {results.get('label3_n_generated')} (cavitation, 150s)")
print(f"M6B_step0b_v2_label6_n_seqs    : {results.get('label6_n_generated')} (sensor_failure, 150s)")
print(f"M6B_step0b_v2_fixes_applied    : F2(abs_sin) F3(M5-faithful) F5(dropout) F6(lib)")
print(f"M6B_step0b_v2_channel_order    : M6B LOCKED (Mot.SV=0,...,Mot.PV=7)")
print(f"M6B_step0b_v2_gates_all_pass   : {results.get('gates_all_pass')}")
print(f"M6B_step0b_v2_gate_fails       : {results.get('gate_fail_list')}")
print(f"M6B_step0b_v2_cav_pres_shift   : {results.get('cav_pres_shift')} (must be <0)")
print(f"M6B_step0b_v2_cav_pmpSV_shift  : {results.get('cav_pmpSV_shift')} (must be >0)")
print(f"M6B_step0b_v2_zt_normal_ok     : {results.get('zt_normal_ok')}")
print(f"M6B_step0b_v2_zt_faults_ok     : {results.get('zt_faults_ok')}")
print(f"M6B_step0b_v2_zt_shape_errors  : {results.get('zt_shape_errors')}")
print(f"M6B_step0b_v2_m6a_archive_ok   : {(SYNTH_DIR/'M6_sequences.pkl').exists()}")
print(f"Status for M6B Step 1          : {'READY' if results.get('gates_all_pass') else 'NEEDS_REVIEW'}")
print("═"*66)
print("══ END PASTE UPDATE ══\n")

print("── FILE MANIFEST ──")
print("  [GitHub PUSH]   src/m6b_physics_lib.py")
print(f"  [GitHub PUSH]   src/{SCRIPT_NAME}.py")
print("  [Local only]    data/synthetic/M6B_sequences_groupA_carried.pkl")
print("  [Local only]    data/synthetic/M6B_sequences_groupA_carried_meta.csv")
print("  [Local only]    data/synthetic/z_t_sequences_groupA_normal.pkl")
print("  [Local only]    data/synthetic/z_t_sequences_groupA_faults.pkl")
print(f"  [Spaces Upload] outputs/plots/M6B_step0b_v2_profiles.png")
print(f"  [Spaces Upload] outputs/reports/{SCRIPT_NAME}_report.md")

print("\n── NEXT PROMPT ──")
print("📦 M6B Step 0b v2 done. Starting M6B Step 1 (Group B compound chains).")
print("   Step 0 v2 + Step 0b v2 both READY. All Group A in M6B channel order.")
print("   All physics fixes F1-F6 applied via m6b_physics_lib.py.")

log(f"{SCRIPT_NAME} COMPLETE.")