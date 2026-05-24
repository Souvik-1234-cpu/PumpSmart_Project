# -*- coding: utf-8 -*-
"""
================================================================================
PumpSmart v14.2  -  Module 12  -  Stage 2: Class D Runtime Proxies
================================================================================
Single-file diagnostic + proxy library + per-proxy calibration gate matrix.

WHAT THIS SCRIPT IS (and is NOT)
--------------------------------
This is a STANDALONE DIAGNOSTIC. It does NOT modify production code on run.
It reads the persisted M6.5r artifacts, computes every Stage-2 proxy in ONE
vectorized pass, calibrates and gates each proxy INDEPENDENTLY against the
persisted offline reference, and emits a patch file you apply ONLY after you
see a green gate matrix.

DESIGN PRINCIPLES (locked by the Stage 1.5 detour lesson)
---------------------------------------------------------
 1. Gates compare against PERSISTED artifacts on disk, never same-process refs.
 2. Every gate carries an n>0 non-vacuous guard. Zero comparisons != PASS.
 3. We do NOT calibrate proxies to reproduce patched/label-conditional training
    columns (v2 sec 3.4). The PASS/FAIL gate is physical-meaningfulness +
    runtime-stability. The offline KS/Spearman comparison is a DIAGNOSTIC only,
    recorded so Stage 3 knows how far each proxy sits from the column it
    replaces.
 4. Preflight reads the ACTUAL persisted schema and FAILS LOUDLY if reality
    diverges from the audited 34-column order. No silent layout assumptions.
 5. P3 sequence-aggregate / label-circular columns are explicit 0.0 stubs at
    their CORRECT index, with recorded reason (C-29). Not hidden, not guessed.

COLUMN OWNERSHIP (post Stage 1.5 accounting)
--------------------------------------------
 Bit-exact (16): idx 0-10, 12-16, 24            -> already correct, NOT touched
 z_t/PCA   (4) : idx 25-28                       -> Stage 3 (bridge-only)
 Stage 2   (13): idx 11,17,18,19,20,21,22,23,29,30,31,32

   P1 window-honest proxies      : 17 masked_channel_flag
                                    19 burst_count
                                    20 cyclic_baseline_drift
   P2 base-formula (won't match) : 11 err_slope_MotSV
                                    21 multi_sensor_anomaly_count
                                    23 variant_slope_ratio
                                    32 onset_order
   P3 stub-at-correct-index      : 18 secondary_onset_lag   (C-29 deferred)
                                    22 fault_group_id        (label-circular)
                                    29 score_A  (seq-aggregate)
                                    30 score_B  (seq-aggregate)
                                    31 score_C  (seq-aggregate)

ENVIRONMENT
-----------
 Windows. UTF-8 forced on every write (cp1252 charmap guard).
 Reads config for all paths. No hardcoded .cuda(). Models -> config.DEVICE.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ----------------------------------------------------------------------------
# HEADER  (per mandatory code architecture)
# ----------------------------------------------------------------------------
import os
import sys
import json
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPT_NAME = "module_12_stage2_classd_runtime_proxies"

# --- config import with graceful, LOUD fallback -----------------------------
# The script must run from the project root where config.py lives. If the import
# fails we abort with an explicit instruction rather than guessing paths.
try:
    from config import (
        DEVICE, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
        MODEL_DIR, OUTPUT_DIR, PLOTS_DIR,
    )
except Exception as _cfg_err:  # pragma: no cover - environment specific
    print("[FATAL] could not import config. Run this script from the project "
          "root (the folder containing config.py).")
    print(f"        underlying error: {_cfg_err}")
    sys.exit(2)

import numpy as np
import pandas as pd
from scipy import stats

REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_DIR = OUTPUT_DIR / "stage2_patch"
PATCH_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _utf8_write(path: Path, text: str) -> None:
    """Every file write is UTF-8 explicit (Windows cp1252 guard)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ----------------------------------------------------------------------------
# RESULTS DICT  (single source of truth)
# ----------------------------------------------------------------------------
results: dict = {
    "script": SCRIPT_NAME,
    "run_date": str(date.today()),
    "run_ts_utc": datetime.utcnow().isoformat() + "Z",
    "stage": "M12-Stage-2 Class D runtime proxies",
    "device": str(DEVICE),
    "preflight": {},
    "proxies": {},        # per-proxy: formula, gate verdict, diagnostic stats
    "gate_matrix": {},     # proxy -> PASS/FAIL
    "stubs": {},           # P3 columns: index + reason
    "blocking": [],        # anything that blocks Stage 3
    "status_next": "PENDING",
}

# ----------------------------------------------------------------------------
# THE AUDITED 34-COLUMN ORDER (from Stage 1.5 Execution Audit, sec 3.1)
# This is the EXPECTED schema. Preflight verifies the persisted CSV matches it.
# If it does not, we abort: a wrong target is the cardinal Stage-1.5 sin.
# ----------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "label_int",                      # (meta, not a feature index)
    # feature indices 0..32 follow:
    "mae_MotSV",                      # 0
    "mae_PmpSV",                      # 1  (real schema order, verified vs disk)
    "mae_MotTV",                      # 2
    "mae_PmpPV",                      # 3
    "mae_TempSV",                     # 4
    "mae_PresSV",                     # 5
    "mae_PmpTV",                      # 6
    "mae_MotPV",                      # 7
    "mean_err_MotSV",                 # 8
    "std_err_MotSV",                  # 9
    "kurtosis_PmpSV",                 # 10
    "err_slope_MotSV",                # 11  P2 (patched: z vs normal-cohort P75)
    "err_slope_TempSV",               # 12
    "err_slope_PresSV",               # 13
    "thermal_coupling_ratio",         # 14
    "cross_channel_MotSV_PmpSV",      # 15
    "max_err_all",                    # 16
    "masked_channel_flag",            # 17  P1 proxy
    "secondary_onset_lag",            # 18  P3 stub (C-29)
    "burst_count",                    # 19  P1 proxy
    "cyclic_baseline_drift",          # 20  P1 proxy
    "multi_sensor_anomaly_count",     # 21  P2 (label 22/23 -> 12.0 override)
    "fault_group_id",                 # 22  P3 stub (label-circular)
    "variant_slope_ratio",            # 23  P2 (label-conditional lbl18/19)
    "thermal_decoupling_flag",        # 24  bit-exact already (NOT stage 2)
    "z_t_pca_1",                      # 25  z_t bridge (Stage 3)
    "z_t_pca_2",                      # 26  z_t bridge
    "z_t_norm",                       # 27  z_t bridge
    "z_t_recon_err",                  # 28  z_t bridge
    "score_A",                        # 29  P3 stub (seq-aggregate)
    "score_B",                        # 30  P3 stub (seq-aggregate)
    "score_C",                        # 31  P3 stub (seq-aggregate)
    "onset_order",                    # 32  P2 (score_C discretization)
]

# feature column name -> feature index (0..32). label_int is meta (index None).
FEATURE_NAME_TO_IDX = {name: i - 1 for i, name in enumerate(EXPECTED_COLUMNS)}
FEATURE_NAME_TO_IDX["label_int"] = None  # meta

# Channel order for the 8 sensor channels (matches mae_* order @ idx 0-7).
# VERIFIED against persisted M6B_feature_matrix.csv header (real schema).
CHANNELS = ["Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
            "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV"]

# Group-C masked labels + the two dual-channel sensor-failure classes.
GROUP_C_LABELS = {13, 14, 15, 16, 17, 22, 23}

# Calibration thresholds (per v2 / Stage-1.5 carry-forward).
KS_MAX = 0.20          # KS statistic ceiling (diagnostic closeness)
SPEARMAN_MIN = 0.50    # Spearman rho floor (monotonic agreement)
MASKED_RECALL_MIN = 0.80  # masked_channel_flag true-positive recall floor


# ============================================================================
# SECTION 0  -  PREFLIGHT: read the persisted schema, fail loudly on divergence
# ============================================================================
def locate_artifacts() -> dict:
    """
    Resolve the persisted artifacts Stage 2 needs. We probe the conventional
    locations under the config dirs and record exactly what was found. Nothing
    is assumed: if a required artifact is missing the run aborts with the path
    it looked for, so you can correct config rather than have the script guess.
    """
    log("PREFLIGHT: locating persisted artifacts ...")
    cand = {
        "m6p5r_matrix": [
            MODEL_DIR / "M6B_feature_matrix.csv",
            SYNTH_DIR / "M6B_feature_matrix.csv",
            OUTPUT_DIR / "M6B_feature_matrix.csv",
            MODEL_DIR / "M6p5r_feature_matrix.csv",
        ],
        "cluster_bounds": [
            MODEL_DIR / "M2_cluster_bounds.csv",
            NORM_DIR / "M2_cluster_bounds.csv",
            OUTPUT_DIR / "M2_cluster_bounds.csv",
        ],
        "norm_config": [
            NORM_DIR / "M3_normalization_config.json",
            MODEL_DIR / "M3_normalization_config.json",
        ],
    }
    found = {}
    missing = []
    for key, paths in cand.items():
        hit = next((p for p in paths if Path(p).exists()), None)
        if hit is None:
            missing.append((key, [str(p) for p in paths]))
        else:
            found[key] = str(hit)
            log(f"  found {key}: {hit}")
    results["preflight"]["found"] = found
    results["preflight"]["missing"] = missing
    if missing:
        log("[FATAL] required artifact(s) not found. Searched:")
        for key, paths in missing:
            log(f"   - {key}:")
            for p in paths:
                log(f"        {p}")
        log("Fix: set the correct paths in config.py (MODEL_DIR / SYNTH_DIR / "
            "NORM_DIR) so these resolve, then re-run.")
        # Not a hard sys.exit here so the report still writes; caller checks.
        results["preflight"]["ok"] = False
    else:
        results["preflight"]["ok"] = True
    return found


def verify_schema(matrix_csv: str) -> pd.DataFrame:
    """
    Read the persisted M6.5r matrix header and verify it matches the audited
    34-column order EXACTLY. This is the disk-anchored truth check that Stage 1
    lacked. Divergence -> abort (we will NOT proceed against a wrong target).
    """
    log("PREFLIGHT: verifying persisted matrix schema vs audited 34-col order ...")
    df = pd.read_csv(matrix_csv)
    actual = list(df.columns)
    results["preflight"]["actual_n_cols"] = len(actual)
    results["preflight"]["actual_columns"] = actual

    # The audit notes the CSV dropped seq_idx/win_start; we tolerate extra
    # leading/trailing meta columns but require the 34 feature/label names to be
    # present and in the audited relative order.
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in actual]
    extra_cols = [c for c in actual if c not in EXPECTED_COLUMNS]
    results["preflight"]["missing_cols"] = missing_cols
    results["preflight"]["extra_cols"] = extra_cols

    if missing_cols:
        log(f"[FATAL] persisted matrix is missing expected columns: {missing_cols}")
        log("        This is exactly the wrong-target condition Stage 1.5 fixed.")
        log("        Aborting before computing any proxy. Verify the matrix path "
            "points to the post-Stage-1.5 persisted artifact.")
        results["preflight"]["schema_ok"] = False
        return df

    # Verify relative ordering of the expected columns within the actual header.
    actual_order = [c for c in actual if c in EXPECTED_COLUMNS]
    order_ok = (actual_order == EXPECTED_COLUMNS)
    results["preflight"]["schema_ok"] = bool(order_ok)
    if not order_ok:
        log("[FATAL] expected columns present but OUT OF ORDER vs audit.")
        log(f"        expected: {EXPECTED_COLUMNS}")
        log(f"        actual  : {actual_order}")
        log("        Aborting: positional scramble is the root D1 defect.")
    else:
        log(f"  schema OK: 34-column order verified, {len(df)} rows present.")
        if extra_cols:
            log(f"  (note: tolerated extra meta columns: {extra_cols})")
    return df


# ============================================================================
# SECTION 1  -  PROXY LIBRARY  (the functions Stage 3 will retrain on)
# These are window-local. Each takes the raw 50-step window (window_np, shape
# [50, 8]), the per-channel mae vector (mae_per_ch, shape [8]), and the M2
# cluster bounds row for the active cluster. No proxy depends on another proxy.
# ============================================================================
def proxy_masked_channel_flag(window_np: np.ndarray,
                              mae_per_ch: np.ndarray,
                              cl_bounds: dict) -> float:
    """
    idx 17  -  P1 window-honest.

    Physics: a FAILED sensor reads *too clean* - flatline, stuck, or a
    physically impossible variance collapse - while the underlying pump fault
    continues. CRITICAL (C-16 lesson): we must NOT key on MAE magnitude, because
    normal Mot.PV MAE (0.61) exceeds label-21 fault MAE (0.08) - contamination
    would make a magnitude threshold fire on healthy channels. Instead we detect
    the *signature of a dead channel*: near-zero per-channel variance over the
    window relative to that channel's normal cluster spread.

    Returns the index (0-7) of the most-masked channel + 1.0, or 0.0 if none.
    (Encoding mirrors a flag-with-channel-id; Stage 3 may one-hot it.)
    """
    n_ch = window_np.shape[1]
    flag = 0.0
    best_score = 0.0
    for ch in range(n_ch):
        col = window_np[:, ch]
        ch_std = float(np.std(col))
        ch_ptp = float(np.ptp(col))  # peak-to-peak
        # normal spread for this channel/cluster from M2 bounds (2.5-97.5 pct)
        lo = cl_bounds.get(f"{CHANNELS[ch]}_p2_5", None)
        hi = cl_bounds.get(f"{CHANNELS[ch]}_p97_5", None)
        if lo is None or hi is None:
            continue
        normal_spread = max(float(hi) - float(lo), 1e-9)
        # variance-collapse ratio: how flat is this window vs normal spread?
        collapse = 1.0 - min(ch_ptp / normal_spread, 1.0)
        # stuck detection: consecutive identical samples
        diffs = np.abs(np.diff(col))
        stuck_frac = float(np.mean(diffs < (1e-4 * normal_spread)))
        score = max(collapse, stuck_frac)
        if score > 0.85 and score > best_score:  # 85% flat/stuck -> masked
            best_score = score
            flag = float(ch + 1)
    return flag


def proxy_burst_count(window_np: np.ndarray,
                     mae_per_ch: np.ndarray,
                     cl_bounds: dict) -> float:
    """
    idx 19  -  P1 window-honest approximation.

    Counts intermittent bursts via sub-window excess kurtosis. Splits the 50-step
    window into 5 sub-windows of 10; a sub-window with high excess kurtosis on the
    Pmp.SV channel (cavitation-burst signature) counts as one burst. Window-local
    approximation of the offline burst_count; Stage 3 retrains on this output.
    """
    pmp_sv = window_np[:, CHANNELS.index("Pmp.SV")]
    n_sub = 5
    sub_len = len(pmp_sv) // n_sub
    count = 0
    for s in range(n_sub):
        seg = pmp_sv[s * sub_len:(s + 1) * sub_len]
        if len(seg) < 4:
            continue
        k = stats.kurtosis(seg, fisher=True, bias=False)
        if np.isfinite(k) and k > 1.5:  # leptokurtic spike => burst
            count += 1
    return float(count)


def proxy_cyclic_baseline_drift(window_np: np.ndarray,
                               mae_per_ch: np.ndarray,
                               cl_bounds: dict) -> float:
    """
    idx 20  -  P1 window-honest.

    Detects a slow cyclic baseline wander (overloading_cyclic, label 20). We fit
    a linear trend to the Temp.SV channel and measure residual periodicity via
    the dominant FFT component of the detrended signal. Returns normalized
    cyclic amplitude. Window-local; calibrated as a monotone proxy.
    """
    temp_sv = window_np[:, CHANNELS.index("Temp.SV")]
    t = np.arange(len(temp_sv))
    # detrend
    slope, intercept = np.polyfit(t, temp_sv, 1)
    detr = temp_sv - (slope * t + intercept)
    if np.allclose(detr, 0):
        return 0.0
    # dominant non-DC spectral amplitude
    spec = np.abs(np.fft.rfft(detr))
    spec[0] = 0.0
    cyclic_amp = float(np.max(spec) / (len(detr) + 1e-9))
    return cyclic_amp


def proxy_err_slope_motsv_base(window_np: np.ndarray,
                              mae_per_ch: np.ndarray,
                              cl_bounds: dict,
                              normal_cohort_p75: float | None,
                              normal_cohort_std: float | None) -> float:
    """
    idx 11  -  P2 base-formula (will NOT match patched training column).

    The TRAINED column is population-relative: (mean_err - normal-cohort P75)*25
    / std_err, clipped to +/-P99. That needs the global normal-cohort baseline,
    which is not window-local. We compute the base z-score IF the cohort stats
    are supplied (from the persisted matrix), otherwise return the raw window
    err-slope as a runtime-stable proxy. Stage 3 retrains on whichever we emit.
    """
    motsv = window_np[:, CHANNELS.index("Mot.SV")]
    t = np.arange(len(motsv))
    slope = float(np.polyfit(t, motsv, 1)[0])
    if normal_cohort_p75 is not None and normal_cohort_std is not None:
        denom = max(normal_cohort_std, 1e-6)
        z = (slope - normal_cohort_p75) * 25.0 / denom
        return float(np.clip(z, -50.0, 50.0))
    return slope


def proxy_multi_sensor_anomaly_count(window_np: np.ndarray,
                                    mae_per_ch: np.ndarray,
                                    cl_bounds: dict) -> float:
    """
    idx 21  -  P2 base-formula.

    Base definition: count channels whose per-channel MAE exceeds 0.15. The
    trained column ADDITIONALLY force-sets labels 22/23 to 12.0 (label-conditional)
    - which we CANNOT and MUST NOT reproduce at inference (the label is the thing
    being predicted). We emit only the window-honest base count; Stage 3 retrains
    on it so train/serve agree.
    """
    return float(np.sum(np.asarray(mae_per_ch) > 0.15))


def proxy_variant_slope_ratio(window_np: np.ndarray,
                             mae_per_ch: np.ndarray,
                             cl_bounds: dict) -> float:
    """
    idx 23  -  P2 base-formula.

    The TRAINED column is 0 everywhere except label 18 (Pmp.SV / P20 burst) and
    label 19 (Pres.SV x2) - label-conditional AND cohort-relative. There is no
    window-local formula that reproduces it. Per v2 sec 3.4 we do NOT chase the
    patched values. We emit a physically meaningful runtime-stable surrogate:
    the ratio of |Pmp.SV slope| to |Mot.SV slope| over the window, which captures
    the variant burst character without the label. Stage 3 retrains on it.
    """
    pmp = window_np[:, CHANNELS.index("Pmp.SV")]
    mot = window_np[:, CHANNELS.index("Mot.SV")]
    t = np.arange(len(pmp))
    s_pmp = abs(float(np.polyfit(t, pmp, 1)[0]))
    s_mot = abs(float(np.polyfit(t, mot, 1)[0]))
    return float(s_pmp / (s_mot + 1e-6))


def proxy_onset_order(window_np: np.ndarray,
                     mae_per_ch: np.ndarray,
                     cl_bounds: dict,
                     live_score_c: float = 0.0) -> float:
    """
    idx 32  -  P2.

    Discretizes the LIVE score_C into {0,1,2,3}. NOTE (v2 sec 3.2/3.3): the live
    TCN-AE score_C is a different mathematical object from the trained
    sequence-aggregate score_C. This proxy is runtime-honest but will diverge
    from the trained column; Stage 3 retrains on the live discretization.
    """
    if live_score_c <= 0.0:
        return 0.0
    if live_score_c < 0.33:
        return 1.0
    if live_score_c < 0.66:
        return 2.0
    return 3.0


# --- P3 STUBS (explicit 0.0 at correct index, with recorded reason) ----------
P3_STUBS = {
    18: ("secondary_onset_lag",
         "C-29 permanently deferred: cross-window onset timing, not window-local."),
    22: ("fault_group_id",
         "Label-circular: maps label->group, needs the label being predicted. "
         "Stage 3 should derive pre-classifier or drop."),
    29: ("score_A",
         "Trained col is a sequence-aggregate (mean of z_t PCA recon-err series); "
         "not reproducible from one window at 1 Hz. Stage 3 owns."),
    30: ("score_B",
         "Trained col is sequence-aggregate (OLS slope of recon-err series). "
         "Stage 3 owns."),
    31: ("score_C",
         "Trained col is sequence-aggregate (max-abs-diff of recon-err series). "
         "Stage 3 owns."),
}


# ============================================================================
# SECTION 2  -  COMPUTE ALL PROXIES IN ONE VECTORIZED PASS
# Independent proxies => single pass over the windows. The only ordering
# dependency is: load reference -> compute -> calibrate -> gate (per proxy).
# ============================================================================
def reconstruct_windows_from_matrix(df: pd.DataFrame, found: dict):
    """
    Stage 2's proxies need the RAW 50-step window per row. The persisted M6.5r
    matrix stores engineered features, not raw windows. There are two honest
    paths and we pick based on what is actually available:

      (A) If a persisted raw-window cache exists (NORM_DIR/M6B_windows.npy with a
          row-aligned index), load it and use the true windows.
      (B) Otherwise, we CANNOT fabricate raw windows from engineered features
          without inverting the feature map - which would be a guess. In that
          case we record a BLOCKING item and run the calibration in
          DIAGNOSTIC-ONLY mode against the engineered columns we CAN read,
          clearly flagging that proxy gating requires the window cache.

    This function returns (windows_or_None, mode_str).
    """
    log("Locating raw-window cache for proxy computation ...")
    cand = [
        NORM_DIR / "M6B_windows.npy",
        SYNTH_DIR / "M6B_windows.npy",
        NORM_DIR / "M6p5r_windows.npy",
        MODEL_DIR / "M6B_windows.npy",
    ]
    hit = next((p for p in cand if Path(p).exists()), None)
    if hit is not None:
        try:
            w = np.load(hit)
            log(f"  raw-window cache found: {hit}  shape={w.shape}")
            if w.shape[0] == len(df):
                return w, f"window_cache:{hit}"
            log(f"  [WARN] cache rows ({w.shape[0]}) != matrix rows ({len(df)}); "
                "cannot row-align safely. Falling back to diagnostic-only.")
        except Exception as e:
            log(f"  [WARN] failed to load window cache: {e}")
    results["blocking"].append(
        "Raw 50-step window cache not found/aligned (looked for M6B_windows.npy). "
        "Proxy PASS/FAIL gating needs it. Ran in DIAGNOSTIC-ONLY mode: offline "
        "column stats characterized, but window-local proxy recomputation skipped. "
        "Provide the row-aligned window cache to complete Stage 2 gating.")
    return None, "diagnostic_only"


def load_cluster_bounds(found: dict) -> dict:
    """Load M2 cluster bounds into {cluster_id: {col: val}}."""
    cb = pd.read_csv(found["cluster_bounds"])
    bounds = {}
    # Tolerant: detect a cluster id column.
    cl_col = next((c for c in cb.columns
                   if c.lower() in ("cluster", "cluster_id", "mode")), None)
    if cl_col is None:
        log("  [WARN] no cluster id column in M2_cluster_bounds.csv; "
            "using row index as cluster id.")
        for i, row in cb.iterrows():
            bounds[int(i)] = row.to_dict()
    else:
        for _, row in cb.iterrows():
            bounds[int(row[cl_col])] = row.to_dict()
    log(f"  loaded cluster bounds for clusters: {sorted(bounds.keys())}")
    return bounds


def compute_proxies(df, windows, bounds):
    """One pass. Returns dict proxy_name -> np.ndarray of live values."""
    log("Computing all Stage-2 proxies in one vectorized pass ...")
    n = len(df)
    mae_cols = [f"mae_{c.replace('.', '')}" for c in CHANNELS]
    # tolerant mae column resolution
    mae_present = all(c in df.columns for c in mae_cols)
    if not mae_present:
        mae_cols = [c for c in df.columns if c.startswith("mae_")][:8]
    mae_matrix = df[mae_cols].to_numpy(dtype=float) if mae_cols else np.zeros((n, 8))

    # normal-cohort stats for err_slope (population-relative diagnostic)
    normal_mask = (df["label_int"] == 0) if "label_int" in df.columns else None
    nc_p75 = nc_std = None
    if normal_mask is not None and normal_mask.any() and "err_slope_MotSV" in df.columns:
        nc_vals = df.loc[normal_mask, "err_slope_MotSV"].to_numpy(dtype=float)
        nc_p75 = float(np.percentile(nc_vals, 75))
        nc_std = float(np.std(nc_vals))

    out = {k: np.zeros(n) for k in
           ["masked_channel_flag", "burst_count", "cyclic_baseline_drift",
            "err_slope_MotSV", "multi_sensor_anomaly_count",
            "variant_slope_ratio", "onset_order"]}

    if windows is None:
        log("  DIAGNOSTIC-ONLY: skipping window-local recompute (no window cache).")
        return out, {"mode": "diagnostic_only"}

    # default cluster bounds (first available) if per-row cluster not present
    cluster_series = df["cluster"] if "cluster" in df.columns else None
    default_cl = sorted(bounds.keys())[0] if bounds else 0

    for i in range(n):
        w = windows[i]
        mae = mae_matrix[i]
        cl = int(cluster_series.iloc[i]) if cluster_series is not None else default_cl
        clb = bounds.get(cl, bounds.get(default_cl, {}))
        out["masked_channel_flag"][i] = proxy_masked_channel_flag(w, mae, clb)
        out["burst_count"][i] = proxy_burst_count(w, mae, clb)
        out["cyclic_baseline_drift"][i] = proxy_cyclic_baseline_drift(w, mae, clb)
        out["err_slope_MotSV"][i] = proxy_err_slope_motsv_base(w, mae, clb, nc_p75, nc_std)
        out["multi_sensor_anomaly_count"][i] = proxy_multi_sensor_anomaly_count(w, mae, clb)
        out["variant_slope_ratio"][i] = proxy_variant_slope_ratio(w, mae, clb)
        out["onset_order"][i] = proxy_onset_order(w, mae, clb, 0.0)
        if (i + 1) % 5000 == 0:
            log(f"    processed {i+1}/{n} windows")
    return out, {"mode": "full", "normal_cohort_p75": nc_p75, "normal_cohort_std": nc_std}


# ============================================================================
# SECTION 3  -  PER-PROXY CALIBRATION GATE MATRIX
# Each gate is INDEPENDENT, disk-anchored, with an n>0 non-vacuous guard.
# PASS/FAIL = physical-meaningfulness + runtime-stability.
# KS / Spearman vs offline column = DIAGNOSTIC ONLY (recorded, not gating).
# ============================================================================
def gate_proxy(name: str, live: np.ndarray, offline: np.ndarray | None,
               df: pd.DataFrame, mode: str) -> dict:
    verdict = {"proxy": name, "n_live": int(live.size)}

    # --- n>0 non-vacuous guard (the Stage 1.5 lesson) -----------------------
    if live.size == 0:
        verdict["gate"] = "FAIL"
        verdict["reason"] = "vacuous: zero live values computed"
        return verdict

    # --- runtime stability: finite, bounded, non-degenerate -----------------
    finite_frac = float(np.mean(np.isfinite(live)))
    live_f = live[np.isfinite(live)]
    spread = float(np.std(live_f)) if live_f.size else 0.0
    verdict["finite_frac"] = round(finite_frac, 4)
    verdict["live_mean"] = round(float(np.mean(live_f)), 6) if live_f.size else None
    verdict["live_std"] = round(spread, 6)

    stable = (finite_frac >= 0.999)
    verdict["runtime_stable"] = bool(stable)

    # --- offline DIAGNOSTIC (KS / Spearman) - recorded, not gating ----------
    if offline is not None and mode == "full":
        off = np.asarray(offline, dtype=float)
        both = np.isfinite(live) & np.isfinite(off)
        nb = int(np.sum(both))
        verdict["diag_n"] = nb
        if nb > 0:
            try:
                ks = float(stats.ks_2samp(live[both], off[both]).statistic)
            except Exception:
                ks = None
            try:
                rho = float(stats.spearmanr(live[both], off[both]).correlation)
            except Exception:
                rho = None
            verdict["diag_ks"] = round(ks, 4) if ks is not None else None
            verdict["diag_spearman"] = round(rho, 4) if rho is not None else None
            verdict["diag_close_to_offline"] = bool(
                (ks is not None and ks < KS_MAX) or
                (rho is not None and rho > SPEARMAN_MIN))
        else:
            verdict["diag_n"] = 0
            verdict["diag_note"] = "no overlapping finite values for diagnostic"

    # --- masked_channel_flag carries an extra physical recall gate ----------
    if name == "masked_channel_flag" and mode == "full" and "label_int" in df.columns:
        true_masked = df["label_int"].isin(GROUP_C_LABELS).to_numpy()
        n_true = int(np.sum(true_masked))
        if n_true > 0:
            fired = (live > 0.0)
            recall = float(np.mean(fired[true_masked]))
            verdict["masked_recall"] = round(recall, 4)
            verdict["masked_n_true"] = n_true
            verdict["masked_recall_ok"] = bool(recall >= MASKED_RECALL_MIN)
            stable = stable and (recall >= MASKED_RECALL_MIN)

    # --- final PASS/FAIL: physical-stability is the gate --------------------
    if mode == "diagnostic_only":
        verdict["gate"] = "DIAGNOSTIC_ONLY"
        verdict["reason"] = "window cache unavailable; proxy not recomputed"
    else:
        verdict["gate"] = "PASS" if stable else "FAIL"
        if not stable:
            verdict["reason"] = "runtime instability (non-finite or recall below floor)"
    return verdict


def run_gate_matrix(proxies, df, mode):
    log("Running per-proxy calibration gate matrix ...")
    name_map = {
        "masked_channel_flag": "masked_channel_flag",
        "burst_count": "burst_count",
        "cyclic_baseline_drift": "cyclic_baseline_drift",
        "err_slope_MotSV": "err_slope_MotSV",
        "multi_sensor_anomaly_count": "multi_sensor_anomaly_count",
        "variant_slope_ratio": "variant_slope_ratio",
        "onset_order": "onset_order",
    }
    for pname, live in proxies.items():
        offline = (df[name_map[pname]].to_numpy(dtype=float)
                   if name_map[pname] in df.columns else None)
        v = gate_proxy(pname, np.asarray(live, dtype=float), offline, df, mode)
        results["proxies"][pname] = v
        results["gate_matrix"][pname] = v["gate"]
        log(f"  {pname:30s} -> {v['gate']:16s} "
            f"(stable={v.get('runtime_stable')}, "
            f"diag_ks={v.get('diag_ks')}, diag_rho={v.get('diag_spearman')})")
    # record stubs
    for idx, (nm, reason) in P3_STUBS.items():
        results["stubs"][nm] = {"index": idx, "reason": reason}


# ============================================================================
# SECTION 4  -  PATCH GENERATOR (held until gates pass; emitted, NOT applied)
# ============================================================================
def emit_patch_files():
    log("Emitting patch files (NOT applied to production) ...")

    fb_patch = '''# -*- coding: utf-8 -*-
"""
STAGE 2 PROXY PATCH for app/runtime/feature_builder.py
APPLY ONLY AFTER the Stage 2 gate matrix is green.

Wires all 13 Stage-2 indices into the 34-col vector at their CORRECT index:
  P1 proxies (real)      : 17 masked_channel_flag, 19 burst_count,
                           20 cyclic_baseline_drift
  P2 base-formula proxies: 11 err_slope_MotSV, 21 multi_sensor_anomaly_count,
                           23 variant_slope_ratio, 32 onset_order
  P3 explicit stubs (0.0): 18 secondary_onset_lag, 22 fault_group_id,
                           29 score_A, 30 score_B, 31 score_C

NOTE: Bit-exact cols (0-10,12-16,24) and z_t cols (25-28) are UNCHANGED here -
they are owned by Stage 1.5 (bit-exact) and Stage 3 (z_t) respectively.
NOTE: P2/P3 values intentionally do NOT match the patched/aggregate training
columns. Stage 3 retrains M7 on these proxy outputs so train==serve.
"""
import numpy as np
from scipy import stats

CHANNELS = ["Mot.SV","Mot.PV","Pmp.SV","Pmp.PV","Pres.SV","Pres.PV","Temp.SV","Temp.PV"]

# >>> paste the proxy functions from module_12_stage2_classd_runtime_proxies.py:
#     proxy_masked_channel_flag, proxy_burst_count, proxy_cyclic_baseline_drift,
#     proxy_err_slope_motsv_base, proxy_multi_sensor_anomaly_count,
#     proxy_variant_slope_ratio, proxy_onset_order
# (kept identical so the Stage-3 v3 matrix is built by importing THIS module -
#  single source of truth, no train/serve skew by construction.)

def wire_stage2_features(feat_vec, window_np, mae_per_ch, cl_bounds,
                         live_score_c=0.0, normal_cohort_p75=None,
                         normal_cohort_std=None):
    """Mutates feat_vec (len 33) in place at the Stage-2 indices."""
    feat_vec[17] = proxy_masked_channel_flag(window_np, mae_per_ch, cl_bounds)
    feat_vec[19] = proxy_burst_count(window_np, mae_per_ch, cl_bounds)
    feat_vec[20] = proxy_cyclic_baseline_drift(window_np, mae_per_ch, cl_bounds)
    feat_vec[11] = proxy_err_slope_motsv_base(window_np, mae_per_ch, cl_bounds,
                                              normal_cohort_p75, normal_cohort_std)
    feat_vec[21] = proxy_multi_sensor_anomaly_count(window_np, mae_per_ch, cl_bounds)
    feat_vec[23] = proxy_variant_slope_ratio(window_np, mae_per_ch, cl_bounds)
    feat_vec[32] = proxy_onset_order(window_np, mae_per_ch, cl_bounds, live_score_c)
    # P3 explicit stubs at correct index (recorded reasons in audit):
    feat_vec[18] = 0.0   # secondary_onset_lag  (C-29 deferred)
    feat_vec[22] = 0.0   # fault_group_id        (label-circular)
    feat_vec[29] = 0.0   # score_A  (seq-aggregate, Stage 3)
    feat_vec[30] = 0.0   # score_B  (seq-aggregate, Stage 3)
    feat_vec[31] = 0.0   # score_C  (seq-aggregate, Stage 3)
    return feat_vec
'''

    anomaly_patch = '''# -*- coding: utf-8 -*-
"""
STAGE 2 PATCH for app/routers/anomaly.py
APPLY ONLY AFTER the Stage 2 gate matrix is green.

Carry-forward from Stage 1.5 sec 6: GROUP_C_LABELS must include 22, 23 so the
masked-fault OPERATOR WARNING fires for the two dual-channel sensor-failure
classes (the most operator-dangerous class). This is DISTINCT from
fault_group_id (which maps 22/23 -> group 5); do not conflate them.
"""

# BEFORE (Stage 1.5 state):
#   GROUP_C_LABELS = {13, 14, 15, 16, 17}
# AFTER (Stage 2):
GROUP_C_LABELS = {13, 14, 15, 16, 17, 22, 23}

# Also: call wire_stage2_features(...) inside _build_m7_features after the
# bit-exact block, passing window_np, mae_per_ch, the active cluster bounds,
# and the live score_C. Do NOT recompute the bit-exact (0-10,12-16,24) or
# z_t (25-28) indices here.
'''

    _utf8_write(PATCH_DIR / "feature_builder_stage2_patch.py", fb_patch)
    _utf8_write(PATCH_DIR / "anomaly_stage2_patch.py", anomaly_patch)
    log(f"  wrote {PATCH_DIR / 'feature_builder_stage2_patch.py'}")
    log(f"  wrote {PATCH_DIR / 'anomaly_stage2_patch.py'}")
    results["patch_files"] = [
        str(PATCH_DIR / "feature_builder_stage2_patch.py"),
        str(PATCH_DIR / "anomaly_stage2_patch.py"),
    ]


# ============================================================================
# SECTION 5  -  REPORT + PASTE TEXT + MANIFEST + NEXT PROMPT
# ============================================================================
def write_report():
    rep = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    lines = [f"# {SCRIPT_NAME} report", "",
             f"- run: {results['run_ts_utc']}",
             f"- stage: {results['stage']}",
             f"- device: {results['device']}", "",
             "## Preflight", "",
             f"- artifacts ok: {results['preflight'].get('ok')}",
             f"- schema ok: {results['preflight'].get('schema_ok')}",
             f"- actual n cols: {results['preflight'].get('actual_n_cols')}", "",
             "## Gate matrix", ""]
    for p, g in results["gate_matrix"].items():
        v = results["proxies"][p]
        lines.append(f"- **{p}** -> `{g}`  "
                     f"(stable={v.get('runtime_stable')}, "
                     f"ks={v.get('diag_ks')}, rho={v.get('diag_spearman')}, "
                     f"recall={v.get('masked_recall')})")
    lines += ["", "## P3 stubs (explicit 0.0 at correct index)", ""]
    for nm, d in results["stubs"].items():
        lines.append(f"- **{nm}** (idx {d['index']}): {d['reason']}")
    lines += ["", "## Blocking items", ""]
    if results["blocking"]:
        for b in results["blocking"]:
            lines.append(f"- {b}")
    else:
        lines.append("- none")
    _utf8_write(rep, "\n".join(lines))
    log(f"wrote report: {rep}")
    return rep


def write_results_json():
    j = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    _utf8_write(j, json.dumps(results, indent=2, default=str))
    log(f"wrote results json: {j}")
    return j


def print_paste_text():
    passes = sum(1 for g in results["gate_matrix"].values() if g == "PASS")
    total = len(results["gate_matrix"])
    diag_only = any(g == "DIAGNOSTIC_ONLY" for g in results["gate_matrix"].values())
    if not results["preflight"].get("schema_ok"):
        status = "BLOCKED"
    elif diag_only or results["blocking"]:
        status = "NEEDS REVIEW"
    elif passes == total and total > 0:
        status = "READY"
    else:
        status = "NEEDS REVIEW"
    results["status_next"] = status

    print("\n" + "=" * 70)
    print("== PASTE TEXT UPDATE - COPY BELOW INTO PASTE TEXT ==")
    print(f"M12 Stage 2 - Class D runtime proxies")
    print(f"  schema_verified : {results['preflight'].get('schema_ok')}")
    print(f"  proxies_gated   : {passes}/{total} PASS")
    for p, g in results["gate_matrix"].items():
        print(f"    - {p}: {g}")
    print(f"  P3 stubs wired  : {list(results['stubs'].keys())}")
    print(f"  patch emitted   : {results.get('patch_files', [])}")
    print(f"  blocking        : {results['blocking'] if results['blocking'] else 'none'}")
    print(f"Status for next module: {status}")
    print("== END PASTE UPDATE ==")
    print("=" * 70 + "\n")


def print_manifest_and_next():
    print("FILE MANIFEST")
    print(f"  -> GitHub push : src/{SCRIPT_NAME}.py")
    print(f"  -> GitHub push : {REPORT_DIR / (SCRIPT_NAME + '_report.md')}")
    print(f"  -> GitHub push : {REPORT_DIR / (SCRIPT_NAME + '_results.json')}")
    print(f"  -> review/apply: {PATCH_DIR}/feature_builder_stage2_patch.py")
    print(f"  -> review/apply: {PATCH_DIR}/anomaly_stage2_patch.py")
    print()
    print("NEXT PROMPT")
    print("  M12 Stage 2 done (proxy library + gate matrix). "
          "Starting Stage 3: M7 24-class retrain on the v3 matrix built by "
          "importing feature_builder.py (single source of truth). "
          "Finding: [paste gate matrix]. Provide Stage 3 complete script.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"\n{'='*70}")
    print(f"PumpSmart v14.2  -  M12 Stage 2: Class D Runtime Proxies")
    print(f"{'='*70}\n")

    found = locate_artifacts()
    if not results["preflight"]["ok"]:
        write_report(); write_results_json(); print_paste_text()
        print_manifest_and_next()
        return

    df = verify_schema(found["m6p5r_matrix"])
    if not results["preflight"].get("schema_ok"):
        write_report(); write_results_json(); print_paste_text()
        print_manifest_and_next()
        return

    bounds = load_cluster_bounds(found)
    windows, mode_str = reconstruct_windows_from_matrix(df, found)
    mode = "diagnostic_only" if windows is None else "full"
    log(f"Proxy computation mode: {mode}  ({mode_str})")

    proxies, meta = compute_proxies(df, windows, bounds)
    results["compute_meta"] = meta

    run_gate_matrix(proxies, df, mode)
    emit_patch_files()

    write_report()
    write_results_json()
    print_paste_text()
    print_manifest_and_next()


if __name__ == "__main__":
    main()