# module_06a_synthetic_generator_v5 Report
Date: 2026-04-26

## Fixes Applied (v5 vs v4)
| Fix | Description |
|-----|-------------|
| F1 | bearing_wear Temp.SV* coupled via _tcoup r=0.9793 |
| F2 | impeller_imbalance abs(sin) AM envelope |
| F3 | cavitation M5-faithful: severity-dependent t_onset |
| F4 | overloading Pres.SV* affinity law Q-H shift |
| F5 | sensor_failure dropout subtype added |
| F6 | all generation via m6b_physics_lib.py |

## Results
| Key | Value |
|-----|-------|
| channels | ['Mot.SV', 'Pmp.SV', 'Mot.TV', 'Pmp.PV', 'Temp.SV', 'Pres.SV', 'Pmp.TV', 'Mot.PV'] |
| M4_threshold | 0.110058 |
| M6A_lstm_ae_gate3_active | True |
| weibull_early_pct | 55.0 |
| weibull_developing_pct | 30.1 |
| weibull_advanced_pct | 14.9 |
| M6A_count_normal | 1200 |
| M6A_count_bearing_wear | 1200 |
| M6A_count_impeller_imbalance | 1200 |
| M6A_count_cavitation | 1200 |
| M6A_count_seal_failure | 1200 |
| M6A_count_overloading | 1200 |
| M6A_count_sensor_failure | 1200 |
| M6A_total_sequences | 8400 |
| M6A_array_shape | (8400, 200, 8) |
| M6A_F1_bearing_tempSV_r | 0.7286 |
| M6A_coupling_bearing_wear | 61.0 |
| M6A_coupling_impeller_imbalance | 68.0 |
| M6A_coupling_overloading | 100.0 |
| M6A_sev_early_pct | 51.2 |
| M6A_sev_developing_pct | 32.9 |
| M6A_sev_advanced_pct | 15.9 |
| M6A_gate_fails | {'bearing_wear': 264, 'impeller_imbalance': 17, 'cavitation': 0, 'seal_failure': 355, 'overloading': 0, 'sensor_failure': 107} |
| M6A_sanity_plot | C:\Users\user\Desktop\PumpSmart_Project\outputs\plots\module_06a_synthetic_generator_v5_sanity_plot.png |
| M6A_all_gates_pass | True |