# module_06a_synthetic_generator_v4 Report
**Date:** 2026-04-12

## Gate Summary
| Gate | Actual | Target | Status |
|------|--------|--------|--------|
| Total sequences | 8400 | == 8400 | PASS |
| Sequences per class | 1200 | == 1200 | PASS |
| Sev early % | 55.291666666666664 | >= 45.0 | PASS |

## Severity Distribution (Weibull k=0.8)
| Stage | Actual % | Target % |
|-------|----------|----------|
| early (≤0.30)       | 55.3% | ~55% |
| developing (0.30–0.65) | 28.6% | ~30% |
| advanced (>0.65)    | 16.1% | ~15% |

## Results
| Key | Value |
|-----|-------|
| M6_lstm_ae_gate3_active | False |
| weibull_early_pct | 55.0 |
| weibull_developing_pct | 30.1 |
| weibull_advanced_pct | 14.9 |
| M6_count_normal | 1200 |
| M6_count_bearing_wear | 1200 |
| M6_count_impeller_imbalance | 1200 |
| M6_count_cavitation | 1200 |
| M6_count_seal_failure | 1200 |
| M6_count_overloading | 1200 |
| M6_count_sensor_failure | 1200 |
| M6_total_sequences | 8400 |
| M6_array_shape | (8400, 200, 8) |
| M6_metadata_columns | ['seq_id', 'label', 'fault_type', 'severity', 'fault_stage', 'source', 'cluster', 'seed_idx'] |
| sequences_npy_path | C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6_synthetic_sequences.npy |
| sequences_pkl_path | C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6_sequences.pkl |
| metadata_path | C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6_synthetic_metadata.csv |
| metadata_shape | (8400, 8) |
| actual_sev_early_pct | 55.3 |
| actual_sev_developing_pct | 28.6 |
| actual_sev_advanced_pct | 16.1 |
| fault_stage_counts | {'early': 3980, 'developing': 2058, 'normal': 1200, 'advanced': 1162} |
| M6_coupling_fidelity_pct | 52.33 |
| separation_ratio | SKIPPED |
| sanity_plot | C:\Users\user\Desktop\PumpSmart_Project\outputs\plots\module_06a_synthetic_generator_v4_sanity_plot.png |
| all_gates_pass | True |

## Output Files
- `data/synthetic/M6_synthetic_sequences.npy` — shape (8400, 200, 8)
- `data/synthetic/M6_synthetic_metadata.csv`  — 8 columns
- `data/synthetic/M6_sequences.pkl`            — legacy compat
- `data/synthetic/M6_sequence_meta.csv`        — legacy compat
- `data/synthetic/M6_validation_report.json`
- `outputs/plots/module_06a_synthetic_generator_v4_sanity_plot.png`
- `outputs/reports/module_06a_synthetic_generator_v4_report.md`