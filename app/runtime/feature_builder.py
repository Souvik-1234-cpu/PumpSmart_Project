# =============================================================================
# app/runtime/feature_builder.py
# M12 Stage 2 — proxy-wired 33-feature M7 input builder
# =============================================================================
# Builds on Stage 1.5 (bit-exact repointed column map). Reproduces the EXACT
# column order and per-column formulas from
#   src/module_06p5r_feature_retrain.py  (the authoritative extractor)
# against which M7 (models/M7_xgboost_classifier_cpu.json) was trained.
#
# STAGE 2 CHANGE (this version):
#   6 columns previously stubbed 0.0 are now filled by the gate-proven proxy
#   library in app/runtime/stage2_proxies.py (SINGLE SOURCE OF TRUTH, imported
#   here AND by the Stage 3 offline matrix builder so train == serve):
#       idx 11 err_slope_MotSV          (was 0.0 stub)
#       idx 17 masked_channel_flag      (was 0.0 stub)
#       idx 19 burst_count              (was 0.0 stub)
#       idx 20 cyclic_baseline_drift    (was 0.0 stub)
#       idx 21 multi_sensor_anomaly_count (was 0.0 stub)
#       idx 23 variant_slope_ratio      (was 0.0 stub)
#
# UNCHANGED bit-exact (Stage 1.5 gate-proven):
#       idx 0-10, 12-16, 24  + z_t cols 25-28 (current-window exact)
#
# DELIBERATE 0.0 STUBS (Stage 3 / deferred — NOT live-injectable):
#       idx 18 secondary_onset_lag (C-29), 22 fault_group_id (label-circular),
#       29 score_A, 30 score_B, 31 score_C (sequence-aggregate of z_t recon-err),
#       32 onset_order (sequence-position ordinal {0,1,2,3}, Group B).
#       Live 4-layer score_A/B/C are NOT injected at 29-31 (would be train/serve
#       skew). Stage 3 retrain closes these.
#
# IMPORTANT (selftest): the C-30 startup identity guard checks 16 bit-exact
# indices [0-10,12-16,24]. None of the Stage-2 proxy indices (11,17,19,20,21,23)
# are in that set, so the selftest remains valid and still passes.
# =============================================================================

from pathlib import Path
import numpy as np
import pickle
from scipy.stats import kurtosis, linregress, pearsonr

# Stage-2 proxy library — SINGLE SOURCE OF TRUTH (also imported by Stage 3).
from app.runtime.stage2_proxies import (
    proxy_err_slope_motsv_base,
    proxy_masked_channel_flag,
    proxy_burst_count,
    proxy_cyclic_baseline_drift,
    proxy_multi_sensor_anomaly_count,
    proxy_variant_slope_ratio,
)

# Channel order — LOCKED, matches extractor CHANNELS
# Mot.SV(0) Pmp.SV(1) Mot.TV(2) Pmp.PV(3) Temp.SV(4) Pres.SV(5) Pmp.TV(6) Mot.PV(7)
_CH = {"Mot.SV": 0, "Pmp.SV": 1, "Mot.TV": 2, "Pmp.PV": 3,
       "Temp.SV": 4, "Pres.SV": 5, "Pmp.TV": 6, "Mot.PV": 7}

_WINDOW_SIZE = 50
_T_AXIS = np.arange(_WINDOW_SIZE)          # extractor: np.arange(WINDOW_SIZE)
_N_FEATURES = 33

# ── PCA artifacts (loaded once at import) ────────────────────────────────────
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_PCA = None
_PCA_ORIGIN = None


def _load_pca():
    global _PCA, _PCA_ORIGIN
    if _PCA is None:
        with open(_MODELS_DIR / "M6p5r_zt_pca.pkl", "rb") as f:
            _PCA = pickle.load(f)
        _PCA_ORIGIN = _PCA.transform(np.zeros((1, 64), dtype=np.float32))  # (1,2)
    return _PCA, _PCA_ORIGIN


def build_m7_features(
    mae_per_ch_np: np.ndarray,   # [8]    M4 per-channel reconstruction MAE
    window_np: np.ndarray,       # [50,8] raw M3-normalised window
    z_t_np: np.ndarray,          # [64]   M4 encoder bottleneck for THIS window
    score_A_live: float = 0.0,   # accepted for signature stability; NOT used
    score_B_live: float = 0.0,
    score_C_live: float = 0.0,
) -> np.ndarray:
    """
    Build the 33-feature vector in the EXACT trained column order.
    Bit-exact for idx 0-10,12-16,24-28; Stage-2 proxies at 11,17,19,20,21,23;
    deliberate 0.0 stubs at 18,22,29,30,31,32.
    """
    pca, pca_origin = _load_pca()
    feats = np.zeros(_N_FEATURES, dtype=np.float32)

    motsv = window_np[:, _CH["Mot.SV"]]
    pmpsv = window_np[:, _CH["Pmp.SV"]]
    mottv = window_np[:, _CH["Mot.TV"]]
    tempsv = window_np[:, _CH["Temp.SV"]]
    pressv = window_np[:, _CH["Pres.SV"]]

    # ── idx 0-7: per-channel M4 reconstruction MAE (D1a fix) ─────────────────
    feats[0:8] = mae_per_ch_np

    # ── idx 8-9: Mot.SV window mean / std ────────────────────────────────────
    feats[8] = float(np.mean(motsv))
    feats[9] = float(np.std(motsv))

    # ── idx 10: Pmp.SV kurtosis, Fisher ──────────────────────────────────────
    feats[10] = float(kurtosis(pmpsv, fisher=True))

    # ── idx 11: err_slope_MotSV — STAGE 2 PROXY (window-honest OLS slope) ─────
    feats[11] = proxy_err_slope_motsv_base(window_np, mae_per_ch_np)

    # ── idx 12-13: per-channel linregress slopes over arange(50) ─────────────
    feats[12] = float(linregress(_T_AXIS, tempsv).slope)
    feats[13] = float(linregress(_T_AXIS, pressv).slope)

    # ── idx 14: thermal_coupling_ratio = pearsonr(Mot.TV, Temp.SV) ───────────
    if np.std(mottv) > 1e-9 and np.std(tempsv) > 1e-9:
        tcr = float(pearsonr(mottv, tempsv)[0])
    else:
        tcr = 1.0
    feats[14] = tcr

    # ── idx 15: cross_channel_MotSV_PmpSV = pearsonr(Mot.SV, Pmp.SV) ─────────
    if np.std(motsv) > 1e-9 and np.std(pmpsv) > 1e-9:
        feats[15] = float(pearsonr(motsv, pmpsv)[0])
    else:
        feats[15] = 0.0

    # ── idx 16: max_err_all = max over per-channel MAE ───────────────────────
    feats[16] = float(mae_per_ch_np.max())

    # ── idx 17: masked_channel_flag — STAGE 2 PROXY ──────────────────────────
    feats[17] = proxy_masked_channel_flag(window_np, mae_per_ch_np)

    # ── idx 18: secondary_onset_lag — DELIBERATE STUB (C-29 deferred) ────────
    feats[18] = 0.0

    # ── idx 19: burst_count — STAGE 2 PROXY ──────────────────────────────────
    feats[19] = proxy_burst_count(window_np, mae_per_ch_np)

    # ── idx 20: cyclic_baseline_drift — STAGE 2 PROXY ────────────────────────
    feats[20] = proxy_cyclic_baseline_drift(window_np, mae_per_ch_np)

    # ── idx 21: multi_sensor_anomaly_count — STAGE 2 PROXY (base count) ───────
    feats[21] = proxy_multi_sensor_anomaly_count(window_np, mae_per_ch_np)

    # ── idx 22: fault_group_id — DELIBERATE STUB (label-circular) ────────────
    feats[22] = 0.0

    # ── idx 23: variant_slope_ratio — STAGE 2 PROXY (surrogate) ──────────────
    feats[23] = proxy_variant_slope_ratio(window_np, mae_per_ch_np)

    # ── idx 24: thermal_decoupling_flag = int(tcr < 0.5) ─────────────────────
    feats[24] = float(int(tcr < 0.5))

    # ── idx 25-28: z_t features from CURRENT window ──────────────────────────
    z_t = np.asarray(z_t_np, dtype=np.float32).reshape(1, 64)
    pca_proj = pca.transform(z_t)                      # (1,2)
    feats[25] = float(pca_proj[0, 0])                  # z_t_pca_1
    feats[26] = float(pca_proj[0, 1])                  # z_t_pca_2
    feats[27] = float(np.linalg.norm(z_t[0]))          # z_t_norm
    feats[28] = float(np.linalg.norm(pca_proj - pca_origin))  # z_t_recon_err

    # ── idx 29-31: trained=seq-aggregate zt_rerr stats; live≠trained — STUB ──
    feats[29] = 0.0   # score_A
    feats[30] = 0.0   # score_B
    feats[31] = 0.0   # score_C

    # ── idx 32: onset_order — DELIBERATE STUB (seq-position ordinal, Stage 3) ─
    feats[32] = 0.0

    return feats