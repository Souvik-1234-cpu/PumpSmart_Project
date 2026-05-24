# -*- coding: utf-8 -*-
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
