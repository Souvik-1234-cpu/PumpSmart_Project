# module_06B_step0b_groupA_carried_v2 Report
Date: 2026-04-26

## Fixes Applied
| Fix | Description |
|-----|-------------|
| F2 | impeller_imbalance abs(sin) AM envelope |
| F3 | cavitation M5-faithful: severity-dep t_onset, mean_drop=0.6*sev |
| F5 | sensor_failure dropout subtype added |
| F6 | All generation via m6b_physics_lib.py |

## Gate Results
| Gate | Pass Rate |
|------|-----------|
| G1_L0 | 0.926 |
| G1_L2 | 1.000 |
| G1_L3 | 0.996 |
| G1_L6 | 0.966 |
| G2_L0 | 1.000 |
| G2_L2 | 1.000 |
| G2_L3 | 1.000 |
| G2_L6 | 1.000 |
| G3_L0 | 1.000 |
| G3_L2 | 1.000 |
| G3_L3 | 1.000 |
| G3_L6 | 1.000 |
| G4_L0 | 1.000 |
| G4_L2 | 1.000 |
| G4_L3 | 1.000 |
| G4_L6 | 1.000 |
| G5_L2 | 1.000 |
| G5_L3_cluster | 1.000 |
| G6_L6 | 1.000 |
| G7_L6_subtypes | 1.000 |

## Cavitation Dual Signature
| Pres.SV* shift | -0.2304 (must be <0) |
| Pmp.SV* shift  | 0.2003 (must be >0) |

## Summary
| Key | Value |
|-----|-------|
| M4_threshold_confirmed | 0.110058 |
| channels | ['Mot.SV', 'Pmp.SV', 'Mot.TV', 'Pmp.PV', 'Temp.SV', 'Pres.SV', 'Pmp.TV', 'Mot.PV'] |
| lstm_ae_loaded | True |
| label0_n_generated | 2000 |
| label2_n_generated | 1500 |
| label3_n_generated | 1500 |
| label6_n_generated | 1200 |
| step0b_total_sequences | 6200 |
| cav_pres_shift | -0.2304 |
| cav_pmpSV_shift | 0.2003 |
| gates_all_pass | True |
| gate_fail_list | [] |
| zt_normal_ok | True |
| zt_faults_ok | True |
| zt_shape_errors | 0 |
| sequences_pkl_saved | True |
| meta_csv_saved | True |
| zt_normal_pkl_saved | True |
| zt_faults_pkl_saved | True |
| plot_saved | True |