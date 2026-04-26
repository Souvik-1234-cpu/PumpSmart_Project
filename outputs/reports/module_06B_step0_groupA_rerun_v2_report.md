# module_06B_step0_groupA_rerun_v2 Report
Date: 2026-04-26

## Fixes Applied
| Fix | Label | Description |
|-----|-------|-------------|
| F1  | 1     | Temp.SV* coupled via _tcoup r=0.9793 |
| F4  | 5     | Pres.SV* affinity law Q-H shift |
| F6  | 1,4,5 | All generation via m6b_physics_lib.py |

## Gate Results
| Gate | Pass Rate |
|------|-----------|
| G1_L1 | 1.000 |
| G1_L4 | 0.861 |
| G1_L5 | 1.000 |
| G1b_L1 | 1.000 |
| G1c_L5 | 1.000 |
| G2_L1 | 1.000 |
| G2_L4 | 1.000 |
| G2_L5 | 1.000 |
| G3_L1 | 1.000 |
| G3_L4 | 1.000 |
| G3_L5 | 1.000 |
| G4_L1 | 1.000 |
| G4_L4 | 1.000 |
| G4_L5 | 1.000 |
| G5_L1 | 1.000 |
| G5_L4 | 1.000 |
| G5_L5 | 1.000 |
| G6_L5 | 1.000 |
| G7_L1 | 1.000 |
| G7_L4 | 0.999 |
| G7_L5 | 1.000 |

## Summary
| Key | Value |
|-----|-------|
| M4_threshold_confirmed | 0.110058 |
| channels | ['Mot.SV', 'Pmp.SV', 'Mot.TV', 'Pmp.PV', 'Temp.SV', 'Pres.SV', 'Pmp.TV', 'Mot.PV'] |
| m6a_loaded | True |
| lstm_ae_loaded | True |
| label1_n_generated | 1500 |
| label1_steps | 250 |
| label4_n_generated | 1500 |
| label4_steps | 400 |
| label5_n_generated | 1500 |
| label5_steps | 300 |
| step0_total_sequences | 4500 |
| gates_all_pass | True |
| gate_fail_list | [] |
| zt_export_ok | True |
| zt_export_n_seqs | 4500 |
| zt_shape_errors | 0 |
| sequences_pkl_saved | True |
| zt_pkl_saved | True |
| plot_saved | True |