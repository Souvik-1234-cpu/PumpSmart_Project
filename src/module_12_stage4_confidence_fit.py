# =============================================================================
# src/module_12_stage4_confidence_fit.py
# PumpSmart v14.2 — M12 Stage 4: physics-honest confidence calibration FIT.
#
# PURPOSE
# -------
# M7 v3 (softprob, no calibration) saturates to 100% on clean synthetic data.
# This script fits a single TEMPERATURE T on held-out windows so that the
# DISPLAYED confidence tracks real windowed accuracy (reliability), then writes
# T + the physics-honest ceiling to models/M7_confidence_calibration.json.
#
# It does NOT retrain M7 and does NOT touch any locked artifact. It rebuilds the
# held-out feature matrix using the SAME single-source builder as inference
# (build_m7_features), so the fitted T is valid for the serve path.
#
# METHOD
# ------
# Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural
# Networks") adapted for XGBoost softprob: recover pseudo-logits log(p),
# minimise negative-log-likelihood over T on held-out windows. NLL is convex in
# 1/T, so a 1-D scan is exact and fast. Report Expected Calibration Error (ECE)
# before/after so the improvement is auditable.
#
# The CEILING is a separate physics-honesty policy (default 0.94), NOT fitted —
# it encodes "a single-pump synthetic-trained model never claims certainty"
# (C-26). Fitting governs the SPREAD; the ceiling governs the CAP.
#
# GATES
# -----
# G_CF_1  Held-out matrix rebuilt with build_m7_features (train==serve)
# G_CF_2  ECE improves after temperature scaling (calibration is real)
# G_CF_3  Post-calib mean top-confidence < raw mean top-confidence (softened)
# G_CF_4  Label decision unchanged by calibration (argmax invariant)
#
# OUTPUT
# ------
# models/M7_confidence_calibration.json   ← {temperature, ceiling, ECE, ...}
# outputs/reports/module_12_stage4_confidence_fit_report.md
# outputs/plots/stage4_reliability_diagram.png
# =============================================================================

import sys, json, pickle, warnings
from pathlib import Path
from datetime import date, datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from config import (DEVICE, IS_GPU, SYNTH_DIR, MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

from app.runtime.feature_builder import build_m7_features

SCRIPT_NAME = "module_12_stage4_confidence_fit"
REPORT_DIR  = OUTPUT_DIR / "reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
results = {}; GATES = {}

# Physics-honesty cap (NOT fitted — policy). C-26: never claim certainty.
CONF_CEILING = 0.94
WINDOW_SIZE  = 50
N_CLASSES    = 24
EPS          = 1e-12

M4_MODEL_PATH = MODEL_DIR / "lstm_ae_baseline_final.pth"
M7_CPU_PATH   = MODEL_DIR / "M7_xgboost_classifier_cpu.json"
CALIB_OUT     = MODEL_DIR / "M7_confidence_calibration.json"

M6B_SEQUENCES_FILES = {
    "A_rerun"  : SYNTH_DIR / "M6B_sequences_groupA_rerun.pkl",
    "A_carried": SYNTH_DIR / "M6B_sequences_groupA_carried.pkl",
    "B"        : SYNTH_DIR / "M6B_sequences_groupB.pkl",
    "C"        : SYNTH_DIR / "M6B_sequences_groupC.pkl",
    "D"        : SYNTH_DIR / "M6B_sequences_groupD.pkl",
    "E"        : SYNTH_DIR / "M6B_sequences_groupE.pkl",
}

log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}  |  Device: {DEVICE}")
log("  Fit confidence temperature (no retrain) + physics-honest ceiling")
log("=" * 72)


# ── M4 architecture (mirrors model_registry exactly) ─────────────────────────
class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8, 128, 2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, 1, batch_first=True)
        self.bn = nn.LayerNorm(64)
    def forward(self, x):
        o1, _ = self.lstm1(x); o2, (h, c) = self.lstm2(o1)
        return self.bn(h[-1]), h, c
class _Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_h = nn.Linear(64, 128); self.fc_c = nn.Linear(64, 128)
        self.lstm1 = nn.LSTM(64, 128, 2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 8, 1, batch_first=True)
        self.out = nn.Linear(8, 8)
    def forward(self, z, seq_len, h, c):
        h0 = torch.tanh(self.fc_h(h[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.tanh(self.fc_c(c[-1])).unsqueeze(0).repeat(2, 1, 1)
        xr = z.unsqueeze(1).repeat(1, seq_len, 1)
        o, _ = self.lstm1(xr, (h0, c0)); o, _ = self.lstm2(o)
        return self.out(o)
class _AE(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = _Enc(); self.decoder = _Dec()
    def forward(self, x):
        z, h, c = self.encoder(x); return self.decoder(z, x.size(1), h, c), z, h, c


@torch.no_grad()
def run_m4_batch(m4, wins):
    x = torch.from_numpy(wins).float()
    enc = m4.encoder
    o1, _ = enc.lstm1(x); o2, (h, c) = enc.lstm2(o1); z = enc.bn(h[-1])
    recon = m4.decoder(z, x.size(1), h, c)
    mae = (x - recon).abs().mean(dim=1)
    return mae.numpy(), z.numpy()


def _extract_label(meta):
    if not isinstance(meta, dict): return None
    for k in ("label", "label_int", "y", "target"):
        if k in meta:
            try: return int(meta[k])
            except Exception: continue
    return None


def softmax_T(logits, T):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())


def ece(probs, y, n_bins=15):
    """Expected Calibration Error."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc  = (pred == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


# ── Load models ──────────────────────────────────────────────────────────────
log("\nSECTION 1 — Load M4 + M7")
m4 = _AE()
m4.load_state_dict(torch.load(M4_MODEL_PATH, map_location="cpu", weights_only=True), strict=True)
m4.eval()
for p in m4.parameters(): p.requires_grad_(False)
clf = xgb.XGBClassifier(); clf.load_model(str(M7_CPU_PATH))
log(f"  M4 + M7 loaded ({clf.n_classes_} classes)")

# ── Build held-out matrix (same builder as serve) ────────────────────────────
log("\nSECTION 2 — Rebuild held-out feature matrix (build_m7_features)")
rows, labels, seqids = [], [], []
gid = 0
for key in ["A_rerun", "A_carried", "B", "C", "D", "E"]:
    p = M6B_SEQUENCES_FILES[key]
    if not p.exists():
        log(f"  WARN missing {p.name} — skipping"); continue
    with open(p, "rb") as f: data = pickle.load(f)
    if isinstance(data, dict) and "sequences" in data:
        seqs = data["sequences"]
        meta = data.get("meta", data.get("metadata", [{}] * len(seqs)))
        lab2seq = defaultdict(list)
        for s, m in zip(seqs, meta):
            lb = _extract_label(m)
            if lb is not None and lb >= 0: lab2seq[lb].append(s)
    else:
        lab2seq = {int(k): v for k, v in data.items() if str(k).lstrip("-").isdigit()}
    for lb, slist in sorted(lab2seq.items()):
        for s in slist:
            s = np.asarray(s, dtype=np.float32)
            if s.shape[0] < WINDOW_SIZE: gid += 1; continue
            nw = s.shape[0] // WINDOW_SIZE
            wb = np.stack([s[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE] for i in range(nw)])
            mae_b, zt_b = run_m4_batch(m4, wb)
            for j in range(nw):
                fv = build_m7_features(mae_per_ch_np=mae_b[j], window_np=wb[j], z_t_np=zt_b[j])
                rows.append(fv); labels.append(lb); seqids.append(gid)
            gid += 1
X = np.stack(rows).astype(np.float32); y = np.array(labels); g = np.array(seqids)
GATES["G_CF_1_matrix_rebuilt"] = {"pass": True, "n_windows": int(len(y))}
log(f"  Rebuilt {len(y):,} windows from {gid:,} sequences")

# Held-out split (no seq leakage)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
_, test_idx = list(sgkf.split(X, y, groups=g))[0]
Xt, yt = X[test_idx], y[test_idx]
log(f"  Held-out: {len(yt):,} windows")

# ── Recover pseudo-logits and fit T ──────────────────────────────────────────
log("\nSECTION 3 — Fit temperature (1-D NLL scan)")
proba = clf.predict_proba(Xt)                       # (N, 24) softprob
logits = np.log(np.clip(proba, EPS, 1.0))           # pseudo-logits

raw_ece = ece(proba, yt); raw_nll = nll(proba, yt)
raw_meanconf = float(proba.max(axis=1).mean())

best_T, best_nll = 1.0, raw_nll
for T in np.linspace(1.0, 6.0, 251):
    pl = softmax_T(logits, T)
    n = nll(pl, yt)
    if n < best_nll:
        best_nll, best_T = n, float(T)

cal_proba = softmax_T(logits, best_T)
cal_ece = ece(cal_proba, yt)
cal_meanconf = float(cal_proba.max(axis=1).mean())
capped_meanconf = float(np.minimum(cal_proba.max(axis=1), CONF_CEILING).mean())

# argmax invariance check
label_same = bool((proba.argmax(1) == cal_proba.argmax(1)).all())

log(f"  Raw  : meanconf={raw_meanconf:.4f}  ECE={raw_ece:.4f}  NLL={raw_nll:.4f}")
log(f"  Fit T={best_T:.3f}")
log(f"  Cal  : meanconf={cal_meanconf:.4f}  ECE={cal_ece:.4f}  NLL={best_nll:.4f}")
log(f"  Cal+ceiling({CONF_CEILING}): meanconf={capped_meanconf:.4f}")
log(f"  Label unchanged by calibration: {label_same}")

GATES["G_CF_2_ece_improves"]   = {"pass": bool(cal_ece <= raw_ece + 1e-6),
                                  "raw_ece": round(raw_ece,4), "cal_ece": round(cal_ece,4)}
GATES["G_CF_3_softened"]        = {"pass": bool(cal_meanconf < raw_meanconf),
                                  "raw": round(raw_meanconf,4), "cal": round(cal_meanconf,4)}
GATES["G_CF_4_label_invariant"] = {"pass": label_same}

# ── Write calibration config ─────────────────────────────────────────────────
log("\nSECTION 4 — Write calibration config")
cfg_out = {
    "_purpose": "Physics-honest confidence calibration for M7 served probability. "
                "Temperature softens saturated softprob; ceiling caps the display.",
    "_generated_by": SCRIPT_NAME,
    "_generated_utc": datetime.utcnow().isoformat() + "Z",
    "_method": "temperature scaling (Guo 2017) on XGBoost softprob pseudo-logits; "
               "ceiling is a separate C-26 honesty policy, not fitted",
    "temperature": round(best_T, 4),
    "ceiling": CONF_CEILING,
    "floor": 0.0,
    "fitted": True,
    "diagnostics": {
        "n_heldout_windows": int(len(yt)),
        "raw_mean_top_confidence": round(raw_meanconf, 4),
        "calibrated_mean_top_confidence": round(cal_meanconf, 4),
        "calibrated_capped_mean_top_confidence": round(capped_meanconf, 4),
        "raw_ece": round(raw_ece, 4),
        "calibrated_ece": round(cal_ece, 4),
        "raw_nll": round(raw_nll, 4),
        "calibrated_nll": round(best_nll, 4),
        "label_unchanged_by_calibration": label_same,
    },
}
with open(CALIB_OUT, "w", encoding="utf-8") as f:
    json.dump(cfg_out, f, indent=2)
log(f"  Written → {CALIB_OUT}")

# ── Reliability diagram ──────────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, (pr, ttl, ee) in zip(ax, [(proba, "Raw (uncalibrated)", raw_ece),
                                       (cal_proba, f"Calibrated T={best_T:.2f}", cal_ece)]):
        conf = pr.max(1); pred = pr.argmax(1); acc = (pred == yt).astype(float)
        bins = np.linspace(0, 1, 16); accs = []; confs = []
        for i in range(15):
            m = (conf > bins[i]) & (conf <= bins[i+1])
            if m.sum() > 0:
                accs.append(acc[m].mean()); confs.append(conf[m].mean())
            else:
                accs.append(np.nan); confs.append((bins[i]+bins[i+1])/2)
        a.plot([0,1],[0,1],"k--",lw=1,label="perfect")
        a.bar(np.array(confs), np.array(accs), width=0.05, alpha=0.7,
              color="#1976D2", edgecolor="white")
        a.set_title(f"{ttl}\nECE={ee:.4f}"); a.set_xlabel("Confidence")
        a.set_ylabel("Accuracy"); a.set_xlim(0,1); a.set_ylim(0,1); a.legend()
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "stage4_reliability_diagram.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("  Reliability diagram saved")
except Exception as e:
    log(f"  WARN: plot failed — {e}")

# ── Report + gate summary ────────────────────────────────────────────────────
n_pass = sum(1 for v in GATES.values() if v["pass"])
log(f"\nSECTION 5 — Gates: {n_pass}/{len(GATES)} pass")
for k, v in GATES.items():
    log(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")

with open(REPORT_DIR / f"{SCRIPT_NAME}_report.md", "w", encoding="utf-8") as f:
    f.write(f"# M12 Stage 4 — Confidence Calibration Fit\n\n")
    f.write(f"**Date:** {date.today()}\n\n")
    f.write(f"Temperature scaling (no retrain) + physics-honest ceiling {CONF_CEILING}.\n\n")
    f.write(f"| Metric | Raw | Calibrated | Cal+Ceiling |\n|---|---|---|---|\n")
    f.write(f"| Mean top-confidence | {raw_meanconf:.4f} | {cal_meanconf:.4f} | {capped_meanconf:.4f} |\n")
    f.write(f"| ECE | {raw_ece:.4f} | {cal_ece:.4f} | — |\n")
    f.write(f"| NLL | {raw_nll:.4f} | {best_nll:.4f} | — |\n\n")
    f.write(f"**Fitted T = {best_T:.3f}** | Label unchanged: {label_same}\n\n")
    f.write(f"## Gates\n\n| Gate | Pass |\n|---|---|\n")
    for k, v in GATES.items():
        f.write(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} |\n")

print("\n" + "═"*60)
print("══ PASTE TEXT UPDATE ══")
print(f"M12_conf_fit_temperature   : {best_T:.3f}")
print(f"M12_conf_fit_ceiling       : {CONF_CEILING}")
print(f"M12_conf_raw_meanconf      : {raw_meanconf:.4f}")
print(f"M12_conf_cal_meanconf      : {cal_meanconf:.4f}")
print(f"M12_conf_capped_meanconf   : {capped_meanconf:.4f}")
print(f"M12_conf_raw_ece           : {raw_ece:.4f}")
print(f"M12_conf_cal_ece           : {cal_ece:.4f}")
print(f"M12_conf_label_invariant   : {label_same}")
print(f"M12_conf_gates             : {n_pass}/{len(GATES)}")
print(f"Status                     : {'READY' if n_pass==len(GATES) else 'REVIEW'}")
print("══ END PASTE UPDATE ══")
print("═"*60)
print(f"\n📦 Confidence fit done. Restart server to load M7_confidence_calibration.json.")
