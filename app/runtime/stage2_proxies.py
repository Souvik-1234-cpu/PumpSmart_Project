# =============================================================================
# app/runtime/stage2_proxies.py
# M12 Stage 2 — Class D runtime proxy library (SINGLE SOURCE OF TRUTH)
# =============================================================================
# These functions are imported by BOTH:
#   - app/runtime/feature_builder.py  (live inference)
#   - the Stage 3 offline matrix builder (training)
# so that train == serve by construction. NEVER fork or copy these bodies.
#
# Gate-proven in src/module_12_stage2_classd_proxies_optionC.py (6/6 PASS,
# masked_channel_flag recall 0.9167 on real M4-faithful windows).
#
# CHANNEL ORDER (LOCKED, matches m6b_physics_lib + extractor):
#   Mot.SV(0) Pmp.SV(1) Mot.TV(2) Pmp.PV(3) Temp.SV(4) Pres.SV(5) Pmp.TV(6) Mot.PV(7)
#
# WHICH INDICES THESE FILL (Stage 2 owns 6 real proxies):
#   idx 11 err_slope_MotSV          -> proxy_err_slope_motsv_base
#   idx 17 masked_channel_flag      -> proxy_masked_channel_flag
#   idx 19 burst_count              -> proxy_burst_count
#   idx 20 cyclic_baseline_drift    -> proxy_cyclic_baseline_drift
#   idx 21 multi_sensor_anomaly_count -> proxy_multi_sensor_anomaly_count
#   idx 23 variant_slope_ratio      -> proxy_variant_slope_ratio
#
# DELIBERATE 0.0 STUBS (NOT this module's job — Stage 3 / deferred):
#   idx 18 secondary_onset_lag   (C-29 cross-window; deferred)
#   idx 22 fault_group_id        (label-circular)
#   idx 29 score_A  30 score_B  31 score_C  (sequence-aggregate of z_t recon-err)
#   idx 32 onset_order           (sequence-position ordinal {0,1,2,3}, Group B)
#
# HONESTY NOTE (C-26 spirit): proxies at idx 11,21,23 do NOT reproduce the
# patched/label-conditional TRAINED columns. They are window-honest surrogates.
# Stage 3 retrains M7 on THESE outputs, closing the train/serve gap. Detection
# F1 does not improve until that retrain.
#
# WINDOW-LOCAL LIMITATION (logged Stage-3 carry-forward): label 15
# (Pres.SV slow-drift mask) is a cross-sequence signature invisible in any
# single 50-step window; masked_channel_flag cannot catch it. Stage 3 needs
# sequence-level detection for label 15.
# =============================================================================

import numpy as np
from scipy import stats

CHANNELS = ["Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
            "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV"]
_CH = {c: i for i, c in enumerate(CHANNELS)}

# Healthy normalized per-window peak-to-peak reference. M6B sequences live in
# normalized space (~0-1); a healthy channel window has ptp ~0.04-0.08
# (confirmed by per-window probe of label-0 / pre-onset windows). The M2 cluster
# bounds are RAW physical units and CANNOT serve as a normalized-ptp reference.
_NORMAL_PTP = 0.08


def proxy_masked_channel_flag(window_np, mae_per_ch, cl_bounds=None) -> float:
    """
    idx 17 — sensor-failure mask detector. A masked channel departs from healthy
    SCADA behavior in physically distinct ways (all confirmed in M6B
    apply_channel_mask): flatline / stuck / dropout (variance collapse) OR drift
    (cal-bias ramp) OR erratic ptp explosion. Keys on the dead-sensor signature
    relative to the channel's own behavior — NOT on MAE magnitude (C-16
    contamination defense). Returns masked channel index+1, or 0.0.
    """
    T, n_ch = window_np.shape
    third = max(T // 3, 3)
    flag, best = 0.0, 0.0
    for ch in range(n_ch):
        col = window_np[:, ch].astype(np.float64)
        d = np.abs(np.diff(col))
        early_j = float(np.std(d[:third])) + 1e-9
        late_j = float(np.std(d[-third:])) + 1e-9
        full_j = float(np.std(d)) + 1e-9
        ptp = float(np.ptp(col))
        collapse_ratio = late_j / early_j
        t = np.arange(T)
        slope = float(np.polyfit(t, col, 1)[0])
        resid = col - (slope * t + col.mean())
        resid_j = float(np.std(np.diff(resid))) + 1e-9
        drift_score = (abs(slope) * T) / (resid_j * T + 1e-9)

        masked, score = False, 0.0
        # (a0) pure flatline/stuck/dropout: whole-window jitter at noise floor
        if full_j < 5e-3 and ptp < 0.05:
            masked, score = True, 1.0
        # (a) variance collapse: late jitter << early jitter AND tiny late jitter
        elif collapse_ratio < 0.25 and late_j < 5e-3:
            masked, score = True, (1.0 - collapse_ratio)
        # (a2) ptp collapse well below healthy normalized spread
        elif ptp < 0.30 * _NORMAL_PTP and full_j < 6e-3:
            masked, score = True, 1.0 - ptp / (0.30 * _NORMAL_PTP + 1e-9)
        # (a3) ptp EXPLOSION vs healthy spread with flat trend (erratic mask)
        elif ptp > 3.0 * _NORMAL_PTP and abs(slope) * T < 0.15 * ptp:
            masked, score = True, min(ptp / (3.0 * _NORMAL_PTP), 1.0)
        # (b) drift mask: monotone ramp dominating residual jitter, ptp large
        elif drift_score > 8.0 and ptp > 0.30:
            masked, score = True, min(drift_score / 20.0, 1.0)

        if masked and score > best:
            best, flag = score, float(ch + 1)
    return flag


def proxy_burst_count(window_np, mae_per_ch, cl_bounds=None) -> float:
    """idx 19 — intermittent burst count via sub-window kurtosis on Pmp.SV."""
    pmp = window_np[:, _CH["Pmp.SV"]]
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


def proxy_cyclic_baseline_drift(window_np, mae_per_ch, cl_bounds=None) -> float:
    """idx 20 — dominant cyclic amplitude of detrended Temp.SV (cyclic load)."""
    temp = window_np[:, _CH["Temp.SV"]]
    t = np.arange(len(temp))
    slope, icpt = np.polyfit(t, temp, 1)
    detr = temp - (slope * t + icpt)
    if np.allclose(detr, 0):
        return 0.0
    spec = np.abs(np.fft.rfft(detr))
    spec[0] = 0.0
    return float(np.max(spec) / (len(detr) + 1e-9))


def proxy_err_slope_motsv_base(window_np, mae_per_ch, cl_bounds=None,
                              nc_p75=None, nc_std=None) -> float:
    """
    idx 11 — window-honest Mot.SV OLS slope (or population-relative z-score if
    normal-cohort stats supplied). NOT the patched cumsum-zscore trained column.
    Stage 3 retrains on whatever this emits.
    """
    motsv = window_np[:, _CH["Mot.SV"]]
    t = np.arange(len(motsv))
    slope = float(np.polyfit(t, motsv, 1)[0])
    if nc_p75 is not None and nc_std is not None:
        return float(np.clip((slope - nc_p75) * 25.0 / max(nc_std, 1e-6),
                             -50.0, 50.0))
    return slope


def proxy_multi_sensor_anomaly_count(window_np, mae_per_ch, cl_bounds=None) -> float:
    """idx 21 — base count of channels with per-channel MAE > 0.15. Does NOT
    apply the trained label-22/23 force-set (label-circular). Stage 3 retrains."""
    return float(np.sum(np.asarray(mae_per_ch) > 0.15))


def proxy_variant_slope_ratio(window_np, mae_per_ch, cl_bounds=None) -> float:
    """idx 23 — |Pmp.SV slope| / |Mot.SV slope| surrogate. NOT the trained
    label-conditional column (lbl18/19). Stage 3 retrains on this surrogate."""
    t = np.arange(window_np.shape[0])
    s_pmp = abs(float(np.polyfit(t, window_np[:, _CH["Pmp.SV"]], 1)[0]))
    s_mot = abs(float(np.polyfit(t, window_np[:, _CH["Mot.SV"]], 1)[0]))
    return float(s_pmp / (s_mot + 1e-6))
