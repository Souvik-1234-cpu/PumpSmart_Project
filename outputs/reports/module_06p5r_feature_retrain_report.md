# module_06p5r_feature_retrain Report  v2.0
**Date:** 2026-04-30

## Gate Summary
| Gate | Result |
|------|--------|
| Z1_pca_variance | PASS |
| W1_boundary | PASS |
| W2_onset_split | PASS |
| W3_compound_lag | PASS |
| D1_class_balance | PASS |
| D2_masked_flag | PASS |
| D3_multisensor | WARN |
| D4_burst_count | PASS |
| D5_label21_slope | WARN |
| Z2_score_C_group_B | WARN |
| Z3_score_B_label21 | PASS |
| F1_fisher | WARN — 13 features < 0.5 |

## Results
| Key | Value |
|-----|-------|
| M4_threshold | 0.110058 |
| fault_rules_v3_labels | 24 |
| m4_params | 505096 |
| M6p5r_z_t_pca_variance_explained | 0.6923263072967529 |
| M6p5r_n_sequences_in | 32500 |
| M6p5r_n_windows_out | 526300 |
| M6p5r_boundary_violations | 0 |
| M6p5r_n_classes | 24 |
| M6p5r_feature_matrix_rows | 526300 |
| M6p5r_gate_D2_pct | 100.0 |
| M6p5r_gate_D3_pct | 47.169642857142854 |
| M6p5r_label21_slope_pct_positive | 68.6891891891892 |
| M6p5r_score_C_group_B_pct | 72.5116116466064 |
| M6p5r_score_C_groupA_p50 | 0.7413935661315918 |
| M6p5r_score_B_label21_pct_positive | 99.35000000000001 |
| M6p5r_feature_matrix_cols | 34 |
| M6p5r_top_fisher_feature | onset_order |
| M6p5r_output_file | C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_feature_matrix.csv |
| M6p5r_window_size | 50 |
| M6p5r_stride | 25 |
| M6p5r_gate_Z1_pca_variance | PASS |
| M6p5r_gate_W1_boundary | PASS |
| M6p5r_gate_W2_onset_split | PASS |
| M6p5r_gate_W3_compound_lag | PASS |
| M6p5r_gate_D1_class_balance | PASS |
| M6p5r_gate_D2_masked_flag | PASS |
| M6p5r_gate_D3_multisensor | WARN |
| M6p5r_gate_D4_burst_count | PASS |
| M6p5r_gate_D5_label21_slope | WARN |
| M6p5r_gate_Z2_score_C_group_B | WARN |
| M6p5r_gate_Z3_score_B_label21 | PASS |
| M6p5r_gate_F1_fisher | WARN — 13 features < 0.5 |
| Status_for_M7 | READY |

## Feature Columns (33 features)
- [01] `mae_MotSV` | Fisher: 0.3239
- [02] `mae_PmpSV` | Fisher: 0.4292
- [03] `mae_MotTV` | Fisher: 0.2085
- [04] `mae_PmpPV` | Fisher: 1.2312
- [05] `mae_TempSV` | Fisher: 0.9709
- [06] `mae_PresSV` | Fisher: 0.4738
- [07] `mae_PmpTV` | Fisher: 0.6122
- [08] `mae_MotPV` | Fisher: 0.8464
- [09] `mean_err_MotSV` | Fisher: 0.7676
- [10] `std_err_MotSV` | Fisher: 0.0987
- [11] `kurtosis_PmpSV` | Fisher: 0.2289
- [12] `err_slope_MotSV` | Fisher: 0.0564
- [13] `err_slope_TempSV` | Fisher: 0.0253
- [14] `err_slope_PresSV` | Fisher: 0.0493
- [15] `thermal_coupling_ratio` | Fisher: 0.7228
- [16] `cross_channel_MotSV_PmpSV` | Fisher: 0.2966
- [17] `max_err_all` | Fisher: 0.4751
- [18] `masked_channel_flag` | Fisher: 61275698217749.5938
- [19] `secondary_onset_lag` | Fisher: 1.4062
- [20] `burst_count` | Fisher: 37637087381491.4375
- [21] `cyclic_baseline_drift` | Fisher: 2.8451
- [22] `multi_sensor_anomaly_count` | Fisher: 1.3284
- [23] `fault_group_id` | Fisher: 15.1273
- [24] `variant_slope_ratio` | Fisher: 0.0276
- [25] `thermal_decoupling_flag` | Fisher: 0.7435
- [26] `z_t_pca_1` | Fisher: 1.8725
- [27] `z_t_pca_2` | Fisher: 1.3539
- [28] `z_t_norm` | Fisher: 1.1181
- [29] `z_t_recon_err` | Fisher: 0.4632
- [30] `score_A` | Fisher: 0.5218
- [31] `score_B` | Fisher: 0.5500
- [32] `score_C` | Fisher: 1.2184
- [33] `onset_order` | Fisher: 92714213339969.5156

## Output Files
- `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_feature_matrix.csv`
- `C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_feature_matrix_metadata.json`

## Visualizations
- `module_06p5r_physics_viz_A_single.png`
- `module_06p5r_physics_viz_B_compound.png`
- `module_06p5r_physics_viz_C_masked.png`
- `module_06p5r_physics_viz_D_variant.png`
- `module_06p5r_physics_viz_E_multisens.png`
- `module_06p5r_mae_all_classes_summary.png`
- Per-label: `module_06p5r_physics_label##_*.png` (24 files)

## Status for M7: **READY**
