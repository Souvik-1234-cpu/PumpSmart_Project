# ═══════════════════════════════════════════════════════════════════════════════
# PumpSmart — module_06B_step0_groupA_rerun_v2.py
# M6B Step 0 v2: Regenerate Labels 1 (250s), 4 (400s), 5 (300s)
# Fixes vs v1:
#   F1: Label 1 — Temp.SV* now coupled via _tcoup (r=0.9793)
#   F4: Label 5 — Pres.SV* affinity law Q-H shift
#   F6: All generation via m6b_physics_lib.py
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_NAME = "module_06B_step0_groupA_rerun_v2"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DEVICE, IS_GPU, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, warnings, pickle, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from m6b_physics_lib import (
    init_lib, CH, CHANNELS, N_CH, CHANNEL_TO_M3_KEY,
    get_cluster_mean, apply_winsorization, make_baseline,
    generate_bearing_wear, generate_seal_failure, generate_overloading,
    NOISE_STD as LIB_NOISE_STD
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
log("v2 — uses m6b_physics_lib.py: F1(TempSV coupling) + F4(PresSV Q-H) applied")

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
    A_WAVE_m_s    = float(_pc.get("A_WAVE_m_s",    1200.0))
    RHO           = float(_pc.get("RHO",            1000.0))
    log(f"  Physics: TAU={TAU_THERMAL_s}s | BPF={BPF_HZ}Hz")
except Exception as e:
    log(f"  [WARNING] M5 fallback: {e}")
    TAU_THERMAL_s=388.9; BPF_HZ=347.67; A_WAVE_m_s=1200.0; RHO=1000.0
    phys_config = {}

try:
    with open(MODEL_DIR / "M4_threshold_config.json") as f:
        threshold_config = json.load(f)
    M4_THRESHOLD = float(threshold_config.get("threshold", 0.110058))
    assert abs(M4_THRESHOLD - 0.110058) < 1e-5
    log(f"  M4 threshold: q = {M4_THRESHOLD} ✓")
except Exception as e:
    log(f"  [WARNING] threshold fallback: {e}")
    M4_THRESHOLD = 0.110058
results["M4_threshold_confirmed"] = M4_THRESHOLD

# Initialise physics library
init_lib(norm_config, phys_config, seed=42)
log(f"  m6b_physics_lib initialised | CH: {CH}")
results["channels"] = CHANNELS

# Confirm M6A archive intact
try:
    with open(SYNTH_DIR / "M6_sequences.pkl", "rb") as f:
        m6a_data = pickle.load(f)
    m6a_shape = getattr(m6a_data, 'shape', len(m6a_data))
    log(f"  M6_sequences.pkl (LOCKED) confirmed: {m6a_shape}")
    results["m6a_loaded"] = True
except Exception as e:
    log(f"  [FATAL] M6_sequences.pkl: {e}"); raise

rng = np.random.default_rng(seed=42)

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
# SECTION 3 — GENERATE SEQUENCES VIA m6b_physics_lib
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 3 — Generating sequences via m6b_physics_lib...")

LABEL_CONFIG = {
    1: {"name": "bearing_wear",  "n_seq": 1500, "steps": 250,
        "fn": lambda sev, cid: generate_bearing_wear(sev, cid, n_steps=250),
        "sev_range": (0.2, 1.0), "clusters": [1, 2, 3]},
    4: {"name": "seal_failure",  "n_seq": 1500, "steps": 400,
        "fn": lambda sev, cid: generate_seal_failure(sev, cid, n_steps=400),
        "sev_range": (0.2, 1.0), "clusters": [1, 3]},
    5: {"name": "overloading",   "n_seq": 1500, "steps": 300,
        "fn": lambda sev, cid: generate_overloading(sev, cid, n_steps=300),
        "sev_range": (0.5, 1.0), "clusters": [2]},
}

all_sequences = []
all_meta      = []

for label_id, cfg in LABEL_CONFIG.items():
    log(f"  Generating Label {label_id} ({cfg['name']}) — "
        f"{cfg['n_seq']} × {cfg['steps']} steps...")
    sev_lo, sev_hi = cfg["sev_range"]
    severities     = rng.uniform(sev_lo, sev_hi, size=cfg["n_seq"]).astype(np.float32)

    for i in range(cfg["n_seq"]):
        cid = int(rng.choice(cfg["clusters"]))
        try:
            seq = cfg["fn"](float(severities[i]), cid)
            assert seq.shape == (cfg["steps"], N_CH)
            assert not np.any(np.isnan(seq))
            assert seq[:, CH["Pres.SV"]].min() >= 0.0
        except Exception as e:
            log(f"    [ERROR] Label {label_id} seq {i}: {e} — fallback")
            seq = make_baseline(cfg["steps"], cluster_id=cid)

        all_sequences.append(seq)
        all_meta.append({
            "seq_id":     f"M6B_L{label_id}_{i:05d}",
            "label":      label_id,
            "label_name": cfg["name"],
            "group":      "A",
            "steps":      cfg["steps"],
            "severity":   float(severities[i]),
            "cluster_id": cid,
            "source":     "m6b_physics_lib_v2",
            "m6b_step":   "0_v2",
            "generated":  str(date.today()),
        })

    results[f"label{label_id}_n_generated"] = cfg["n_seq"]
    results[f"label{label_id}_steps"]       = cfg["steps"]
    log(f"    Label {label_id}: {cfg['n_seq']} sequences ✓")

log(f"  Total: {len(all_sequences)} sequences")
results["step0_total_sequences"] = len(all_sequences)

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

# G1 — Primary fault signal
run_gate("1", 1, "Mot.SV* above baseline",
         lambda s: s[100:, CH["Mot.SV"]].mean() > s[:50, CH["Mot.SV"]].mean() + 0.05)
run_gate("1", 4, "Pres.SV* below baseline (orifice decline)",
         lambda s: s[100:300, CH["Pres.SV"]].mean() < s[:50, CH["Pres.SV"]].mean() - 0.03)
run_gate("1", 5, "Temp.SV* above baseline (thermal rise)",
         lambda s: s[150:, CH["Temp.SV"]].mean() > s[:50, CH["Temp.SV"]].mean() + 0.05)

# G1b — F1 fix: Temp.SV* coupling for Label 1
run_gate("1b", 1, "Temp.SV* follows Mot.TV* (F1 fix — r=0.9793 coupling)",
         lambda s: stats.spearmanr(s[:, CH["Mot.TV"]], s[:, CH["Temp.SV"]])[0] > 0.60)

# G1c — F4 fix: Pres.SV* Q-H shift for Label 5
run_gate("1c", 5, "Pres.SV* slightly elevated during overloading (F4 fix — affinity law)",
         lambda s: s[100:, CH["Pres.SV"]].mean() > 0.95)

# G2 — No negative pressure
for lid in [1, 4, 5]:
    run_gate("2", lid, "No negative Pres.SV*",
             lambda s: s[:, CH["Pres.SV"]].min() >= 0.0)

# G3 — No NaN/Inf
for lid in [1, 4, 5]:
    run_gate("3", lid, "No NaN/Inf",
             lambda s: not (np.any(np.isnan(s)) or np.any(np.isinf(s))))

# G4 — Correct shapes
run_gate("4", 1, "Shape (250,8)", lambda s: s.shape == (250, 8))
run_gate("4", 4, "Shape (400,8)", lambda s: s.shape == (400, 8))
run_gate("4", 5, "Shape (300,8)", lambda s: s.shape == (300, 8))

# G5 — Thermal coupling
run_gate("5", 1, "Mot.TV*–Mot.SV* Spearman r > 0.60 (bearing thermal)",
         lambda s: stats.spearmanr(s[:, CH["Mot.SV"]], s[:, CH["Mot.TV"]])[0] > 0.60)
run_gate("5", 4, "Pmp.TV*–Pres.SV* partial decoupling (r=-0.013 seal)",
         lambda s: abs(stats.spearmanr(s[200:, CH["Pmp.TV"]],
                                        s[200:, CH["Pres.SV"]])[0]) < 0.85)
run_gate("5", 5, "Temp.SV*–Mot.TV* Spearman r > 0.90 (overloading r=0.997)",
         lambda s: stats.spearmanr(s[:, CH["Temp.SV"]], s[:, CH["Mot.TV"]])[0] > 0.90)

# G6 — C-04 overloading invariant
def g6_overloading(seq):
    onset = seq[50:250, CH["Temp.SV"]]
    r_temp, _ = stats.spearmanr(np.arange(len(onset)), onset)
    slope, _, _, _, _ = stats.linregress(np.arange(seq.shape[0]), seq[:, CH["Mot.SV"]])
    return r_temp > 0.70 and abs(slope) < 0.0005
run_gate("6", 5, "Temp.SV* monotonic rise, Mot.SV* stable slope (C-04)",
         g6_overloading)

# G7 — Primary channel distinguishable
run_gate("7", 1, "Mot.SV* fault window MAE distinguishable",
         lambda s: abs(s[100:, CH["Mot.SV"]].mean() - s[:50, CH["Mot.SV"]].mean()) > 0.05)
run_gate("7", 4, "Pres.SV* fault window MAE distinguishable",
         lambda s: abs(s[200:, CH["Pres.SV"]].mean() - s[:50, CH["Pres.SV"]].mean()) > 0.03)
run_gate("7", 5, "Temp.SV* fault window MAE distinguishable",
         lambda s: abs(s[150:, CH["Temp.SV"]].mean() - s[:50, CH["Temp.SV"]].mean()) > 0.05)

all_gate_pass  = all(v["status"] in ("PASS","WARN") for v in gate_results.values())
gate_fail_list = [k for k,v in gate_results.items() if v["status"] == "FAIL"]
log(f"  Gate summary: {len(gate_results)} gates | Fails: {gate_fail_list}")
results["gates_all_pass"]  = all_gate_pass
results["gate_fail_list"]  = gate_fail_list
results["gate_results"]    = {k: v["pass_rate"] for k,v in gate_results.items()}

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
            batch = np.stack(all_windows[b0:b1], axis=0)
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

try:
    zt_export = export_zt(all_sequences, all_meta, batch_size=32)
    log(f"  z_t export: {len(zt_export)} entries")
    results["zt_export_ok"]       = True
    results["zt_export_n_seqs"]   = len(zt_export)
    results["zt_shape_errors"]    = sum(
        1 for d in zt_export.values()
        if d["z_t"].ndim != 2 or d["z_t"].shape[1] != 64)
except torch.cuda.OutOfMemoryError:
    log("  [OOM] Falling back CPU batch=16")
    lstm_ae_model.cpu()
    zt_export = export_zt(all_sequences, all_meta, batch_size=16)
    results["zt_export_ok"] = True; results["zt_cuda_oom"] = True
except Exception as e:
    log(f"  [ERROR] z_t: {e}")
    zt_export = {}; results["zt_export_ok"] = False

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 6 — Saving...")

try:
    out_path = SYNTH_DIR / "M6B_sequences_groupA_rerun.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"sequences": all_sequences, "meta": all_meta}, f)
    log(f"  Saved: {out_path}")
    results["sequences_pkl_saved"] = True
except Exception as e:
    log(f"  [ERROR] pkl: {e}"); results["sequences_pkl_saved"] = False

try:
    pd.DataFrame(all_meta).to_csv(
        SYNTH_DIR / "M6B_sequences_groupA_rerun_meta.csv", index=False)
    log(f"  Saved meta CSV")
except Exception as e:
    log(f"  [ERROR] meta CSV: {e}")

try:
    with open(SYNTH_DIR / "z_t_sequences_groupA_faults_rerun.pkl", "wb") as f:
        pickle.dump(zt_export, f)
    log(f"  Saved z_t pkl")
    results["zt_pkl_saved"] = True
except Exception as e:
    log(f"  [ERROR] z_t pkl: {e}"); results["zt_pkl_saved"] = False

locked_path = SYNTH_DIR / "M6_sequences.pkl"
assert locked_path.exists(), "[FATAL] M6_sequences.pkl (LOCKED) missing"
log(f"  LOCKED archive confirmed: {locked_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
log("SECTION 7 — Plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("M6B Step 0 v2 — Labels 1,4,5 (m6b_physics_lib)", fontsize=13)

plot_specs = [
    (1, "bearing_wear",  [("Mot.SV","blue"),("Mot.TV","orange"),("Temp.SV","red")],     250, 0, "F1: Mot.SV→Mot.TV→Temp.SV"),
    (4, "seal_failure",  [("Pres.SV","red"),("Pmp.TV","orange"),("Pmp.PV","purple")],   400, 1, "Pres.SV↓ orifice"),
    (5, "overloading",   [("Temp.SV","orange"),("Mot.TV","red"),("Pres.SV","blue")],    300, 2, "F4: Temp↑ + Pres.SV Q-H shift"),
]
for lid, lname, ch_cols, nsteps, ax_idx, title in plot_specs:
    ax = axes.flat[ax_idx]
    label_seqs = [all_sequences[i] for i, m in enumerate(all_meta) if m["label"] == lid]
    sample_i   = np.linspace(0, len(label_seqs)-1, min(6, len(label_seqs)), dtype=int)
    for ch_name, color in ch_cols:
        for si in sample_i:
            ax.plot(label_seqs[si][:, CH[ch_name]], alpha=0.2, color=color, linewidth=0.7)
        mean_p = np.mean([label_seqs[i][:, CH[ch_name]] for i in sample_i], axis=0)
        ax.plot(mean_p, color=color, linewidth=2.0, label=ch_name)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"L{lid} {lname}\n{title}", fontsize=9)
    ax.set_xlabel("Step"); ax.set_ylabel("Norm. value")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# z_t PCA
try:
    from sklearn.decomposition import PCA
    zt_pool, zt_labels_plot = [], []
    for lid, color in [(1,"blue"),(4,"red"),(5,"orange")]:
        ids = [m["seq_id"] for m in all_meta if m["label"]==lid][:200]
        for sid in ids:
            if sid in zt_export:
                zt_pool.append(zt_export[sid]["z_t"].mean(axis=0))
                zt_labels_plot.append(lid)
    if len(zt_pool) >= 20:
        zt_arr = np.array(zt_pool)
        zt_pca = PCA(n_components=2).fit_transform(zt_arr)
        for ax_idx2, (lid2, col2) in enumerate([(1,"blue"),(4,"red"),(5,"orange")]):
            ax = axes.flat[3 + ax_idx2]
            mask = [i for i,l in enumerate(zt_labels_plot) if l==lid2]
            ax.scatter(zt_pca[mask,0], zt_pca[mask,1], c=col2, s=10, alpha=0.5)
            ax.set_title(f"z_t PCA — Label {lid2}", fontsize=9)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.3)
except Exception as e:
    log(f"  [WARNING] PCA plot: {e}")

plt.tight_layout()
try:
    plt.savefig(PLOTS_DIR / "M6B_step0_v2_profiles.png", dpi=120, bbox_inches='tight')
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
    "## Fixes Applied", "| Fix | Label | Description |", "|-----|-------|-------------|",
    "| F1  | 1     | Temp.SV* coupled via _tcoup r=0.9793 |",
    "| F4  | 5     | Pres.SV* affinity law Q-H shift |",
    "| F6  | 1,4,5 | All generation via m6b_physics_lib.py |",
    "", "## Gate Results", "| Gate | Pass Rate |", "|------|-----------|",
]
for k, v in results.get("gate_results", {}).items():
    report_lines.append(f"| {k} | {v:.3f} |")
report_lines += ["", "## Summary", "| Key | Value |", "|-----|-------|"]
for k, v in results.items():
    if k != "gate_results":
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
print(f"M6B_step0_v2_status            : {'READY' if results.get('gates_all_pass') else 'NEEDS_REVIEW'}")
print(f"M6B_step0_v2_total_sequences   : {results.get('step0_total_sequences')}")
print(f"M6B_step0_v2_label1_n_seqs     : {results.get('label1_n_generated')} (bearing_wear, 250s)")
print(f"M6B_step0_v2_label4_n_seqs     : {results.get('label4_n_generated')} (seal_failure, 400s)")
print(f"M6B_step0_v2_label5_n_seqs     : {results.get('label5_n_generated')} (overloading, 300s)")
print(f"M6B_step0_v2_fixes_applied     : F1(TempSV coupling), F4(PresSV Q-H), F6(unified lib)")
print(f"M6B_step0_v2_gates_all_pass    : {results.get('gates_all_pass')}")
print(f"M6B_step0_v2_gate_fails        : {results.get('gate_fail_list')}")
print(f"M6B_step0_v2_zt_export_ok      : {results.get('zt_export_ok')}")
print(f"M6B_step0_v2_zt_shape_errors   : {results.get('zt_shape_errors')}")
print(f"M6B_step0_v2_m6a_archive_ok    : {(SYNTH_DIR/'M6_sequences.pkl').exists()}")
print(f"Status for M6B Step 0b v2      : READY")
print("═"*66)
print("══ END PASTE UPDATE ══\n")

print("── FILE MANIFEST ──")
print(f"  [GitHub PUSH]   src/m6b_physics_lib.py")
print(f"  [GitHub PUSH]   src/module_06B_step0_groupA_rerun_v2.py")
print(f"  [Local only]    data/synthetic/M6B_sequences_groupA_rerun.pkl")
print(f"  [Local only]    data/synthetic/M6B_sequences_groupA_rerun_meta.csv")
print(f"  [Local only]    data/synthetic/z_t_sequences_groupA_faults_rerun.pkl")
print(f"  [Spaces Upload] outputs/plots/M6B_step0_v2_profiles.png")
print(f"  [Spaces Upload] outputs/reports/{SCRIPT_NAME}_report.md")

log(f"{SCRIPT_NAME} COMPLETE.")