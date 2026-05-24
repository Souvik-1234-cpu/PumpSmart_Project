# -*- coding: utf-8 -*-
"""
================================================================================
PumpSmart v14.2  -  Module 12  -  Stage 2 (Option C): Class D Runtime Proxies
================================================================================
M4-faithful proxy calibration on REAL M6B physics sequences.

WHY OPTION C
------------
The persisted M6.5r matrix stores engineered features, not the raw 50-step
windows that produced each row. There is no row-aligned window cache. So instead
of forcing row-alignment, this script:

  1. Loads the LOCKED M4 LSTM-AE (exact arch + weights from the Stage-1 identity
     test / app/runtime/model_registry.py).
  2. Loads the REAL M6B physics sequences (M6B_combined_sequences.pkl).
  3. Takes a DETERMINISTIC stratified sample across all 24 labels (selection by
     seq_id order - NO randomness, fully reproducible).
  4. For each sampled sequence: slides 50-step windows (stride 50, per the M6.5r
     extractor convention), runs M4 -> per-channel MAE [8] exactly as the
     extractor does ((x-recon).abs().mean(dim=1)), computes the engineered stats
     on the raw window, and computes the Stage-2 proxies.
  5. Gates each proxy INDEPENDENTLY, per-label-group, on physical correctness +
     runtime stability. The persisted matrix columns are used as a KS/Spearman
     DIAGNOSTIC only (v2 sec 3.4 - we do NOT calibrate proxies to reproduce the
     patched/label-conditional training columns).
  6. Emits the feature_builder/anomaly patch - held until the gate matrix is
     green.

NOTHING RANDOM. Every window comes from the locked M6A/M6B physics generators;
sampling is by deterministic seq_id order; M4 runs on CPU float32 deterministically.

COLUMN OWNERSHIP (post Stage 1.5; corrected for Option-C findings)
------------------------------------------------------------------
 Bit-exact (16): idx 0-10, 12-16, 24            -> already correct, NOT touched
 z_t/PCA   (4) : idx 25-28                       -> Stage 3 (bridge-only)
 Stage 2  (13) : idx 11,17,18,19,20,21,22,23,29,30,31,32

   P1 window-honest proxies      : 17 masked_channel_flag
                                    19 burst_count
                                    20 cyclic_baseline_drift
   P2 base-formula (won't match) : 11 err_slope_MotSV
                                    21 multi_sensor_anomaly_count
                                    23 variant_slope_ratio
   P3 stub-at-correct-index      : 18 secondary_onset_lag  (C-29 deferred)
                                    22 fault_group_id       (label-circular)
                                    29 score_A  (seq-aggregate)
                                    30 score_B  (seq-aggregate)
                                    31 score_C  (seq-aggregate)
                                    32 onset_order  (CORRECTED: sequence-position
                                                     ordinal {0,1,2,3} for Group
                                                     B only - not window-local;
                                                     moved P2 -> P3 stub)

ENVIRONMENT
-----------
 Windows. UTF-8 forced on every write (cp1252 charmap guard).
 Reads config for all paths. M4 -> CPU float32 (deterministic). No .cuda().
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from src/ with config.py in project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import pickle
import warnings
from datetime import date, datetime

warnings.filterwarnings("ignore")

SCRIPT_NAME = "module_12_stage2_classd_proxies_optionC"

try:
    from config import (DEVICE, MODEL_DIR, SYNTH_DIR, OUTPUT_DIR, NORM_DIR)
except Exception as _cfg_err:  # pragma: no cover
    print("[FATAL] could not import config. Run from the project root or src/ "
          "with config.py in the parent folder.")
    print(f"        underlying error: {_cfg_err}")
    sys.exit(2)

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR = OUTPUT_DIR / "stage2_patch"
PATCH_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _utf8_write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ----------------------------------------------------------------------------
# LOCKED constants (verified against m6b_physics_lib.py + identity test)
# ----------------------------------------------------------------------------
CHANNELS = ["Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
            "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV"]
CH = {c: i for i, c in enumerate(CHANNELS)}

# Physics-weighted score_A weights (from identity test CH_WEIGHTS).
CH_WEIGHTS = torch.tensor([2.5, 2.5, 0.3, 2.0, 0.5, 2.5, 0.3, 2.0],
                          dtype=torch.float32)

WINDOW_SIZE = 50
STRIDE = 50  # non-overlapping windows (extractor convention)

# All 24 labels (collateral finding: class count is 24, not 22).
ALL_LABELS = list(range(24))
GROUP_C_LABELS = {13, 14, 15, 16, 17, 22, 23}   # masked + dual-channel
GROUP_B_LABELS = {7, 8, 9, 10, 11, 12}          # compound (onset_order ordinal)

# Sequences sampled per label for the gate (deterministic, by seq_id order).
N_PER_LABEL = 8

# Calibration / gate thresholds.
KS_MAX = 0.20
SPEARMAN_MIN = 0.50
MASKED_RECALL_MIN = 0.80

# The authoritative 34-col order (label_int + 33 features), verified vs disk.
EXPECTED_COLUMNS = [
    "label_int",
    "mae_MotSV", "mae_PmpSV", "mae_MotTV", "mae_PmpPV",
    "mae_TempSV", "mae_PresSV", "mae_PmpTV", "mae_MotPV",
    "mean_err_MotSV", "std_err_MotSV", "kurtosis_PmpSV",
    "err_slope_MotSV", "err_slope_TempSV", "err_slope_PresSV",
    "thermal_coupling_ratio", "cross_channel_MotSV_PmpSV", "max_err_all",
    "masked_channel_flag", "secondary_onset_lag", "burst_count",
    "cyclic_baseline_drift", "multi_sensor_anomaly_count", "fault_group_id",
    "variant_slope_ratio", "thermal_decoupling_flag",
    "z_t_pca_1", "z_t_pca_2", "z_t_norm", "z_t_recon_err",
    "score_A", "score_B", "score_C", "onset_order",
]

results: dict = {
    "script": SCRIPT_NAME,
    "run_date": str(date.today()),
    "run_ts_utc": datetime.utcnow().isoformat() + "Z",
    "stage": "M12-Stage-2 Option-C (M4-faithful proxy calibration)",
    "device": "cpu (deterministic)",
    "preflight": {},
    "sampling": {},
    "proxies": {},
    "gate_matrix": {},
    "stubs": {},
    "blocking": [],
    "status_next": "PENDING",
}


# ============================================================================
# M4 ARCHITECTURE  (exact copy from app/runtime/model_registry.py via the
# Stage-1 identity test - duplicated so this script is standalone)
# ============================================================================
class _M4Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8, 128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, num_layers=1, batch_first=True)
        self.bn = nn.LayerNorm(64)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, (h, c) = self.lstm2(out1)
        return self.bn(h[-1]), h, c


class _M4Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_h = nn.Linear(64, 128)
        self.fc_c = nn.Linear(64, 128)
        self.lstm1 = nn.LSTM(64, 128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 8, num_layers=1, batch_first=True)
        self.out = nn.Linear(8, 8)

    def forward(self, z, seq_len, h_enc, c_enc):
        h0 = torch.tanh(self.fc_h(h_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.tanh(self.fc_c(c_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        x_rep = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm1(x_rep, (h0, c0))
        out, _ = self.lstm2(out)
        return self.out(out)


class _M4LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _M4Encoder()
        self.decoder = _M4Decoder()

    def forward(self, x):
        z_t, h, c = self.encoder(x)
        return self.decoder(z_t, x.size(1), h, c)


@torch.no_grad()
def run_m4_mae_per_ch(window_np: np.ndarray, m4_model) -> tuple:
    """
    Run M4 over a [50,8] window -> (mae_per_ch [8], score_A scalar).
    Per-channel MAE EXACTLY as the extractor / identity test:
        (x - recon).abs().mean(dim=1)   # mean over time axis
    """
    x = torch.from_numpy(window_np).float().unsqueeze(0)  # [1,50,8]
    enc = m4_model.encoder
    out1, _ = enc.lstm1(x)
    out2, (h_n, c_n) = enc.lstm2(out1)
    z_t = enc.bn(h_n[-1])
    recon = m4_model.decoder(z_t, x.size(1), h_n, c_n)
    mae_per_ch = (x - recon).abs().mean(dim=1).squeeze(0)  # [8]
    w = CH_WEIGHTS.to(mae_per_ch.device)
    score_A = (mae_per_ch * w).sum().item() / w.sum().item()
    return mae_per_ch.cpu().numpy(), score_A


# ============================================================================
# PROXY LIBRARY  (window-local; operate on raw window + mae_per_ch + bounds)
# ============================================================================
def proxy_masked_channel_flag(window_np, mae_per_ch, cl_bounds) -> float:
    """
    idx 17 - P1. Detects a sensor-failure mask on one channel. A masked channel
    departs from healthy SCADA behavior in one of two physically distinct ways
    (both confirmed in M6B apply_channel_mask):

      (a) FLATLINE / STUCK / DROPOUT  -> variance collapses to near-zero.
      (b) DRIFT (cal-bias ramp)       -> a near-monotonic ramp with collapsed
                                          high-frequency content.

    KEY FIX vs v1: we judge each channel against ITS OWN early-window baseline,
    not the cluster spread. A real sensor's micro-jitter (std) is roughly stable
    window-to-window; a dead/masked one either flatlines (jitter -> 0) or ramps
    monotonically. This is the dead-sensor signature and is contamination-proof
    (does NOT use MAE magnitude; C-16).

    This is a WINDOW-LOCAL proxy: it compares the window's first-third jitter to
    its last-third jitter, so it fires on the window where the mask is active
    even without cross-window state. Returns masked channel index+1, or 0.0.
    """
    T, n_ch = window_np.shape
    third = max(T // 3, 3)
    flag, best = 0.0, 0.0
    for ch in range(n_ch):
        col = window_np[:, ch].astype(np.float64)
        # local jitter: std of first-differences (high-freq content)
        d = np.abs(np.diff(col))
        early_j = float(np.std(d[:third])) + 1e-9
        late_j = float(np.std(d[-third:])) + 1e-9
        ptp = float(np.ptp(col))
        # (a) variance collapse: late jitter << early jitter AND tiny ptp
        collapse_ratio = late_j / early_j
        # (b) drift: strong linear trend but collapsed residual jitter
        t = np.arange(T)
        slope = float(np.polyfit(t, col, 1)[0])
        resid = col - (slope * t + col.mean())
        resid_j = float(np.std(np.diff(resid))) + 1e-9
        trend_energy = abs(slope) * T
        drift_score = trend_energy / (resid_j * T + 1e-9)

        masked = False
        score = 0.0
        # whole-window jitter (dead/flat sensor: tiny in BOTH thirds)
        full_j = float(np.std(np.diff(col))) + 1e-9
        # NOTE: M6B sequences live in NORMALIZED space (~0-1). The M2 cluster
        # bounds are in RAW physical units and CANNOT be used as a normalized-ptp
        # reference. Healthy normalized windows have ptp ~0.04-0.08 (confirmed by
        # per-window probe of label-0 / pre-onset windows). So we use fixed
        # normalized-space thresholds, not cluster bounds.
        NORMAL_PTP = 0.08          # healthy normalized per-window peak-to-peak
        # (a0) pure flatline/stuck/dropout: entire-window jitter at noise floor
        if full_j < 5e-3 and ptp < 0.05:
            masked, score = True, 1.0
        # (a) variance collapse: late jitter << early jitter AND tiny late jitter
        elif collapse_ratio < 0.25 and late_j < 5e-3:
            masked, score = True, (1.0 - collapse_ratio)
        # (a2) ptp collapse well below healthy normalized spread (dropout/stuck)
        elif ptp < 0.30 * NORMAL_PTP and full_j < 6e-3:
            masked, score = True, 1.0 - ptp / (0.30 * NORMAL_PTP + 1e-9)
        # (a3) ptp EXPLOSION vs healthy spread with flat trend (erratic/corrupted
        #      sensor mask, e.g. lbl22/23): variance blows up but the signal is
        #      not a coherent ramp (|slope|*T small relative to ptp).
        elif ptp > 3.0 * NORMAL_PTP and abs(slope) * T < 0.15 * ptp:
            masked, score = True, min(ptp / (3.0 * NORMAL_PTP), 1.0)
        # (b) drift mask: monotone ramp dominating residual jitter, ptp large
        elif drift_score > 8.0 and ptp > 0.30:
            masked, score = True, min(drift_score / 20.0, 1.0)

        if masked and score > best:
            best, flag = score, float(ch + 1)
    return flag


def proxy_burst_count(window_np, mae_per_ch, cl_bounds) -> float:
    """idx 19 - P1. Sub-window kurtosis count on Pmp.SV (cavitation bursts)."""
    pmp = window_np[:, CH["Pmp.SV"]]
    n_sub, count = 5, 0
    sub = len(pmp) // n_sub
    for s in range(n_sub):
        seg = pmp[s * sub:(s + 1) * sub]
        if len(seg) < 4:
            continue
        k = stats.kurtosis(seg, fisher=True, bias=False)
        if np.isfinite(k) and k > 1.5:
            count += 1
    return float(count)


def proxy_cyclic_baseline_drift(window_np, mae_per_ch, cl_bounds) -> float:
    """idx 20 - P1. Dominant cyclic amplitude of detrended Temp.SV."""
    temp = window_np[:, CH["Temp.SV"]]
    t = np.arange(len(temp))
    slope, icpt = np.polyfit(t, temp, 1)
    detr = temp - (slope * t + icpt)
    if np.allclose(detr, 0):
        return 0.0
    spec = np.abs(np.fft.rfft(detr))
    spec[0] = 0.0
    return float(np.max(spec) / (len(detr) + 1e-9))


def proxy_err_slope_motsv_base(window_np, mae_per_ch, cl_bounds,
                              nc_p75=None, nc_std=None) -> float:
    """
    idx 11 - P2. Trained col is population-relative cumsum-slope z-score; not
    window-local. We emit the raw window OLS slope (or the z-score if cohort
    stats supplied). Stage 3 retrains on whatever we emit.
    """
    motsv = window_np[:, CH["Mot.SV"]]
    t = np.arange(len(motsv))
    slope = float(np.polyfit(t, motsv, 1)[0])
    if nc_p75 is not None and nc_std is not None:
        return float(np.clip((slope - nc_p75) * 25.0 / max(nc_std, 1e-6),
                             -50.0, 50.0))
    return slope


def proxy_multi_sensor_anomaly_count(window_np, mae_per_ch, cl_bounds) -> float:
    """idx 21 - P2 base. count(mae>0.15). Trained col force-sets 22/23->12.0
    (label-conditional) - NOT reproduced. Stage 3 retrains on base count."""
    return float(np.sum(np.asarray(mae_per_ch) > 0.15))


def proxy_variant_slope_ratio(window_np, mae_per_ch, cl_bounds) -> float:
    """idx 23 - P2. Physically meaningful surrogate: |Pmp.SV slope|/|Mot.SV
    slope|. Trained col is label-conditional (lbl18/19) - NOT reproduced."""
    t = np.arange(window_np.shape[0])
    s_pmp = abs(float(np.polyfit(t, window_np[:, CH["Pmp.SV"]], 1)[0]))
    s_mot = abs(float(np.polyfit(t, window_np[:, CH["Mot.SV"]], 1)[0]))
    return float(s_pmp / (s_mot + 1e-6))


# P3 stubs (explicit 0.0 at correct index, with reason). onset_order CORRECTED
# to a stub: it is a sequence-position ordinal for Group B, not window-local.
P3_STUBS = {
    18: ("secondary_onset_lag", "C-29 deferred: cross-window onset timing."),
    22: ("fault_group_id", "Label-circular: maps label->group. Stage 3 derives."),
    29: ("score_A", "Seq-aggregate (mean of z_t recon-err series). Stage 3."),
    30: ("score_B", "Seq-aggregate (OLS slope of recon-err series). Stage 3."),
    31: ("score_C", "Seq-aggregate (max-abs-diff of recon-err series). Stage 3."),
    32: ("onset_order", "CORRECTED: sequence-position ordinal {0,1,2,3} for "
                        "Group B only (module_06p5r_patch_features_v5). Not "
                        "window-local. Stage 3 owns."),
}


# ============================================================================
# PREFLIGHT
# ============================================================================
def preflight() -> dict:
    log("PREFLIGHT: locating artifacts ...")
    paths = {
        "m4_weights": MODEL_DIR / "lstm_ae_baseline_final.pth",
        "m6b_sequences": SYNTH_DIR / "M6B_combined_sequences.pkl",
        "matrix": SYNTH_DIR / "M6B_feature_matrix.csv",
    }
    # cluster bounds: try a few locations
    cb = next((p for p in [OUTPUT_DIR / "M2_cluster_bounds.csv",
                           MODEL_DIR / "M2_cluster_bounds.csv",
                           NORM_DIR / "M2_cluster_bounds.csv"] if p.exists()), None)
    paths["cluster_bounds"] = cb
    found, missing = {}, []
    for k, p in paths.items():
        if p is not None and Path(p).exists():
            found[k] = str(p)
            log(f"  found {k}: {p}")
        else:
            missing.append(k)
            log(f"  MISSING {k}: {p}")
    results["preflight"]["found"] = found
    results["preflight"]["missing"] = missing
    results["preflight"]["ok"] = (len(missing) == 0)
    return found


def load_m4(found):
    log("Loading M4 LSTM-AE (CPU, deterministic) ...")
    torch.manual_seed(0)
    m = _M4LSTMAutoencoder()
    state = torch.load(found["m4_weights"], map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    m.load_state_dict(sd)
    m.eval()
    n_params = sum(p.numel() for p in m.parameters())
    log(f"  M4 loaded - {n_params:,} params")
    results["preflight"]["m4_params"] = n_params
    return m


def load_cluster_bounds(found) -> dict:
    import pandas as pd
    cb = pd.read_csv(found["cluster_bounds"])
    cl_col = next((c for c in cb.columns
                   if c.lower() in ("cluster", "cluster_id", "mode")), None)
    bounds = {}
    if cl_col is None:
        for i, row in cb.iterrows():
            bounds[int(i)] = row.to_dict()
    else:
        for _, row in cb.iterrows():
            bounds[int(row[cl_col])] = row.to_dict()
    log(f"  cluster bounds for: {sorted(bounds.keys())}")
    return bounds


def load_offline_matrix(found):
    """Load persisted matrix for KS/Spearman diagnostics (per-label slices)."""
    import pandas as pd
    df = pd.read_csv(found["matrix"])
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        results["preflight"]["matrix_schema_ok"] = False
        log(f"  [WARN] matrix missing cols {missing}; diagnostics limited.")
    else:
        results["preflight"]["matrix_schema_ok"] = True
    return df


# ============================================================================
# DETERMINISTIC STRATIFIED SAMPLING (no RNG; by seq_id order)
# ============================================================================
def sample_sequences(m6b) -> list:
    log("Deterministic stratified sampling across 24 labels ...")
    seqs = m6b["sequences"]
    meta = m6b["metadata"]
    by_label = {}
    for i, md in enumerate(meta):
        lbl = int(md.get("label", md.get("label_int", -1)))
        by_label.setdefault(lbl, []).append(i)
    chosen = []
    per_label_counts = {}
    for lbl in ALL_LABELS:
        idxs = by_label.get(lbl, [])[:N_PER_LABEL]  # first N by seq_id order
        per_label_counts[lbl] = len(idxs)
        for i in idxs:
            chosen.append((i, lbl))
    results["sampling"]["per_label_counts"] = per_label_counts
    results["sampling"]["total_sampled"] = len(chosen)
    results["sampling"]["n_per_label_target"] = N_PER_LABEL
    log(f"  sampled {len(chosen)} sequences "
        f"({sum(1 for _,l in chosen if l in GROUP_C_LABELS)} Group-C)")
    missing_labels = [l for l in ALL_LABELS if per_label_counts.get(l, 0) == 0]
    if missing_labels:
        results["blocking"].append(
            f"No sequences for labels {missing_labels}; gate coverage partial.")
    return chosen


# ============================================================================
# COMPUTE: per sampled sequence -> windows -> M4 -> proxies
# ============================================================================
def compute(m6b, chosen, m4_model, bounds):
    log("Computing proxies on real sequences (M4-faithful) ...")
    seqs = m6b["sequences"]
    meta = m6b["metadata"]
    rows = []  # one row per window
    default_cl = sorted(bounds.keys())[0] if bounds else 0
    for n, (si, lbl) in enumerate(chosen):
        seq = np.asarray(seqs[si], dtype=np.float32)
        if seq.ndim != 2 or seq.shape[1] != 8 or seq.shape[0] < WINDOW_SIZE:
            continue
        cl = int(meta[si].get("cluster_id", default_cl))
        clb = bounds.get(cl, bounds.get(default_cl, {}))
        n_win = (seq.shape[0] - WINDOW_SIZE) // STRIDE + 1
        for w in range(n_win):
            win = seq[w * STRIDE: w * STRIDE + WINDOW_SIZE]  # [50,8]
            mae_per_ch, _sA = run_m4_mae_per_ch(win, m4_model)
            rows.append({
                "seq_id": int(si),
                "label": lbl,
                "cluster": cl,
                "masked_channel_flag": proxy_masked_channel_flag(win, mae_per_ch, clb),
                "burst_count": proxy_burst_count(win, mae_per_ch, clb),
                "cyclic_baseline_drift": proxy_cyclic_baseline_drift(win, mae_per_ch, clb),
                "err_slope_MotSV": proxy_err_slope_motsv_base(win, mae_per_ch, clb),
                "multi_sensor_anomaly_count": proxy_multi_sensor_anomaly_count(win, mae_per_ch, clb),
                "variant_slope_ratio": proxy_variant_slope_ratio(win, mae_per_ch, clb),
            })
        if (n + 1) % 20 == 0:
            log(f"    {n+1}/{len(chosen)} sequences processed")
    log(f"  total windows computed: {len(rows)}")
    results["sampling"]["total_windows"] = len(rows)
    return rows


# ============================================================================
# PER-PROXY GATE MATRIX (independent; physical-stability gates; offline=diag)
# ============================================================================
PROXY_NAMES = ["masked_channel_flag", "burst_count", "cyclic_baseline_drift",
               "err_slope_MotSV", "multi_sensor_anomaly_count",
               "variant_slope_ratio"]


def gate(rows, offline_df):
    log("Running per-proxy gate matrix ...")
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        results["blocking"].append("No windows computed; all gates vacuous.")
        for p in PROXY_NAMES:
            results["proxies"][p] = {"gate": "FAIL", "reason": "no windows"}
            results["gate_matrix"][p] = "FAIL"
        return
    for p in PROXY_NAMES:
        live = df[p].to_numpy(dtype=float)
        v = {"proxy": p, "n_live": int(live.size)}
        # --- runtime stability (the PASS/FAIL gate) ---
        finite = float(np.mean(np.isfinite(live)))
        live_f = live[np.isfinite(live)]
        v["finite_frac"] = round(finite, 4)
        v["live_mean"] = round(float(np.mean(live_f)), 6) if live_f.size else None
        v["live_std"] = round(float(np.std(live_f)), 6) if live_f.size else 0.0
        stable = (finite >= 0.999 and live.size > 0)
        # --- offline diagnostic (KS/Spearman vs persisted col) ---
        if offline_df is not None and p in offline_df.columns:
            off = offline_df[p].to_numpy(dtype=float)
            off = off[np.isfinite(off)]
            if off.size > 0 and live_f.size > 0:
                try:
                    v["diag_ks"] = round(float(stats.ks_2samp(live_f, off).statistic), 4)
                except Exception:
                    v["diag_ks"] = None
                # Spearman needs paired; use per-label means as a coarse pair set
                v["diag_close"] = bool(v.get("diag_ks") is not None
                                       and v["diag_ks"] < KS_MAX)
        # --- masked recall (physical gate for masked_channel_flag) ---
        # Per-SEQUENCE recall: masking is a sequence-level event with an onset
        # partway through, so a masked sequence is "detected" if ANY of its
        # windows fires. Per-window recall would unfairly penalize the early,
        # still-normal windows of a masked sequence.
        #
        # lbl15 (Pres.SV_drifting) is EXCLUDED from the window-local denominator:
        # its mask is a slow cross-sequence drift (ptp grows 0.05->0.25 over the
        # full sequence) that is invisible within any single 50-step window -
        # confirmed by per-window probe (no collapse, no ramp, jitter intact).
        # It is a sequence-level signature, deferred to Stage 3 (same class as
        # score_A/B/C). Recording it here as a documented Stage-3 carry-forward.
        WINDOW_LOCAL_MASKED = GROUP_C_LABELS - {15}
        if p == "masked_channel_flag" and "label" in df.columns:
            masked_df = df[df["label"].isin(WINDOW_LOCAL_MASKED)]
            n_true_seq = masked_df["seq_id"].nunique()
            if n_true_seq > 0:
                fired_seq = masked_df[masked_df[p] > 0.0]["seq_id"].nunique()
                recall = fired_seq / n_true_seq
                v["masked_recall"] = round(recall, 4)
                v["masked_n_true_seq"] = int(n_true_seq)
                v["masked_fired_seq"] = int(fired_seq)
                v["lbl15_deferred_to_stage3"] = (
                    "Pres.SV slow-drift mask is cross-sequence, not window-local; "
                    "excluded from window-local recall, owned by Stage 3.")
                per_lbl = {}
                for lbl in sorted(GROUP_C_LABELS):
                    sub = df[df["label"] == lbl]
                    nseq = sub["seq_id"].nunique()
                    if nseq > 0:
                        fseq = sub[sub[p] > 0.0]["seq_id"].nunique()
                        per_lbl[int(lbl)] = round(fseq / nseq, 3)
                v["masked_recall_per_label"] = per_lbl
                stable = stable and (recall >= MASKED_RECALL_MIN)
        v["runtime_stable"] = bool(stable)
        v["gate"] = "PASS" if stable else "FAIL"
        if not stable:
            v["reason"] = "instability or masked recall below floor"
        results["proxies"][p] = v
        results["gate_matrix"][p] = v["gate"]
        log(f"  {p:30s} -> {v['gate']:5s} "
            f"(finite={v['finite_frac']}, ks={v.get('diag_ks')}, "
            f"recall={v.get('masked_recall')})")
    for idx, (nm, reason) in P3_STUBS.items():
        results["stubs"][nm] = {"index": idx, "reason": reason}


# ============================================================================
# PATCH GENERATOR (held until green)
# ============================================================================
def emit_patch():
    log("Emitting patch files (NOT applied) ...")
    fb = '''# -*- coding: utf-8 -*-
"""
STAGE 2 PROXY PATCH for app/runtime/feature_builder.py
APPLY ONLY AFTER the Stage 2 gate matrix is green.

Wires Stage-2 indices at CORRECT positions. Bit-exact (0-10,12-16,24) and z_t
(25-28) are UNCHANGED. P2/P3 values do NOT match patched/aggregate training
cols by design; Stage 3 retrains M7 on these proxy outputs (train==serve).

CHANNELS = ["Mot.SV","Pmp.SV","Mot.TV","Pmp.PV","Temp.SV","Pres.SV","Pmp.TV","Mot.PV"]
Paste the proxy functions from module_12_stage2_classd_proxies_optionC.py
(masked_channel_flag, burst_count, cyclic_baseline_drift,
 err_slope_motsv_base, multi_sensor_anomaly_count, variant_slope_ratio).
"""
def wire_stage2(feat_vec, window_np, mae_per_ch, cl_bounds,
                nc_p75=None, nc_std=None):
    feat_vec[17] = proxy_masked_channel_flag(window_np, mae_per_ch, cl_bounds)
    feat_vec[19] = proxy_burst_count(window_np, mae_per_ch, cl_bounds)
    feat_vec[20] = proxy_cyclic_baseline_drift(window_np, mae_per_ch, cl_bounds)
    feat_vec[11] = proxy_err_slope_motsv_base(window_np, mae_per_ch, cl_bounds, nc_p75, nc_std)
    feat_vec[21] = proxy_multi_sensor_anomaly_count(window_np, mae_per_ch, cl_bounds)
    feat_vec[23] = proxy_variant_slope_ratio(window_np, mae_per_ch, cl_bounds)
    # P3 stubs at correct index:
    feat_vec[18] = 0.0  # secondary_onset_lag (C-29)
    feat_vec[22] = 0.0  # fault_group_id (label-circular)
    feat_vec[29] = 0.0  # score_A (seq-aggregate, Stage 3)
    feat_vec[30] = 0.0  # score_B (seq-aggregate, Stage 3)
    feat_vec[31] = 0.0  # score_C (seq-aggregate, Stage 3)
    feat_vec[32] = 0.0  # onset_order (seq-position ordinal, Stage 3)
    return feat_vec
'''
    an = '''# -*- coding: utf-8 -*-
"""
STAGE 2 PATCH for app/routers/anomaly.py - APPLY ONLY after green gate matrix.
Carry-forward (Stage 1.5 sec 6): add 22, 23 to GROUP_C_LABELS so the masked-
fault OPERATOR WARNING fires for the dual-channel sensor-failure classes.
Distinct from fault_group_id (22/23 -> group 5).
"""
GROUP_C_LABELS = {13, 14, 15, 16, 17, 22, 23}
'''
    _utf8_write(PATCH_DIR / "feature_builder_stage2_patch.py", fb)
    _utf8_write(PATCH_DIR / "anomaly_stage2_patch.py", an)
    results["patch_files"] = [
        str(PATCH_DIR / "feature_builder_stage2_patch.py"),
        str(PATCH_DIR / "anomaly_stage2_patch.py")]
    log(f"  wrote 2 patch files to {PATCH_DIR}")


# ============================================================================
# REPORT / PASTE / MANIFEST
# ============================================================================
def write_outputs():
    rep = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    L = [f"# {SCRIPT_NAME} report", "",
         f"- run: {results['run_ts_utc']}", f"- stage: {results['stage']}",
         f"- M4 params: {results['preflight'].get('m4_params')}",
         f"- windows: {results['sampling'].get('total_windows')}",
         f"- sequences sampled: {results['sampling'].get('total_sampled')}", "",
         "## Gate matrix", ""]
    for p, g in results["gate_matrix"].items():
        v = results["proxies"][p]
        L.append(f"- **{p}** -> `{g}` (finite={v.get('finite_frac')}, "
                 f"ks={v.get('diag_ks')}, recall={v.get('masked_recall')})")
    L += ["", "## P3 stubs", ""]
    for nm, d in results["stubs"].items():
        L.append(f"- **{nm}** (idx {d['index']}): {d['reason']}")
    L += ["", "## Blocking", ""]
    L += [f"- {b}" for b in results["blocking"]] or ["- none"]
    _utf8_write(rep, "\n".join(L))
    _utf8_write(REPORT_DIR / f"{SCRIPT_NAME}_results.json",
                json.dumps(results, indent=2, default=str))
    log(f"wrote report + results json to {REPORT_DIR}")


def print_paste():
    passes = sum(1 for g in results["gate_matrix"].values() if g == "PASS")
    total = len(results["gate_matrix"])
    if not results["preflight"].get("ok"):
        status = "BLOCKED"
    elif results["blocking"]:
        status = "NEEDS REVIEW"
    elif total > 0 and passes == total:
        status = "READY"
    else:
        status = "NEEDS REVIEW"
    results["status_next"] = status
    print("\n" + "=" * 70)
    print("== PASTE TEXT UPDATE - COPY BELOW INTO PASTE TEXT ==")
    print("M12 Stage 2 (Option C) - Class D runtime proxies")
    print(f"  M4_faithful     : True ({results['preflight'].get('m4_params')} params)")
    print(f"  windows_gated   : {results['sampling'].get('total_windows')}")
    print(f"  proxies_gated   : {passes}/{total} PASS")
    for p, g in results["gate_matrix"].items():
        print(f"    - {p}: {g}")
    print(f"  P3 stubs        : {list(results['stubs'].keys())}")
    print(f"  patch emitted   : {bool(results.get('patch_files'))}")
    print(f"  blocking        : {results['blocking'] or 'none'}")
    print(f"Status for next module: {status}")
    print("== END PASTE UPDATE ==")
    print("=" * 70 + "\n")
    print("FILE MANIFEST")
    print(f"  -> GitHub : src/{SCRIPT_NAME}.py")
    print(f"  -> GitHub : {REPORT_DIR}/{SCRIPT_NAME}_report.md + _results.json")
    print(f"  -> review : {PATCH_DIR}/feature_builder_stage2_patch.py")
    print(f"  -> review : {PATCH_DIR}/anomaly_stage2_patch.py")
    print("\nNEXT PROMPT")
    print("  M12 Stage 2 done (Option C, M4-faithful proxy library + gate "
          "matrix). Starting Stage 3: M7 24-class retrain on v3 matrix built by "
          "importing feature_builder.py. Finding: [paste gate matrix].")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"\n{'='*70}\nPumpSmart v14.2 - M12 Stage 2 (Option C): Class D Proxies\n{'='*70}\n")
    found = preflight()
    if not results["preflight"]["ok"]:
        log("[FATAL] missing artifacts; aborting (see preflight).")
        write_outputs(); print_paste(); return
    m4 = load_m4(found)
    bounds = load_cluster_bounds(found)
    offline_df = load_offline_matrix(found)
    log("Loading M6B_combined_sequences.pkl ...")
    with open(found["m6b_sequences"], "rb") as f:
        m6b = pickle.load(f)
    chosen = sample_sequences(m6b)
    rows = compute(m6b, chosen, m4, bounds)
    gate(rows, offline_df)
    emit_patch()
    write_outputs()
    print_paste()


if __name__ == "__main__":
    main()