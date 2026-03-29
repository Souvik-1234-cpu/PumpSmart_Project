# M4 LSTM-AE Baseline Report v8
**Date:** 2026-03-28 | **Version:** v8 — Cluster-Conditional Winsorization

## Physics Fixes vs v7
- **FIX-1**: `Pmp.PV` startup ceiling raised to 3.2x (ISO 13373-3: BPF harmonics 2-4x steady-state during ramp-up)
- **FIX-2**: `Pres.SV` cluster-conditional ceilings: startup=3.0x, steady_state=5.6x, high_load=2.0x, cooldown=3.0x (Joukowsky water hammer prevention)
- **FIX-3**: `pressure_transient` spike ratio vs high_load mean (42 bar reference, not startup 0.62 bar)
- **FIX-4**: Full cluster bounds saved to M4_spike_config.json for M6 direct consumption

## Training Results
| Metric | Value |
|--------|-------|
| Best val loss | 0.026862 |
| Best epoch | 141 |
| Training time | 51.1s |
| Overfit triggered | False |

## Threshold
| Metric | Value |
|--------|-------|
| Mean MAE | 0.026765 |
| Std MAE | 0.025972 |
| P99 | 0.102045 |
| Threshold | 0.110058 |
| Separation ratio | 4.11x |
| False alarms | 8 |
| Spike rows excluded | 12620 |

## Cluster-Conditional Winsor Bounds (v8 key output)

### X_Pres.SV_norm
| Mode | Mean | Upper (normalised) | Multiplier |
|------|------|--------------------|------------|
| startup | 1.0000 | 3.0000 | 3.0x |
| steady_state | 1.0000 | 5.6000 | 5.6x |
| high_load | 1.0000 | 2.0000 | 2.0x |
| cooldown | 1.0000 | 3.0000 | 3.0x |

### X_ACR_Pmp.PV_norm
| Mode | Mean | Upper (normalised) | Multiplier |
|------|------|--------------------|------------|
| startup | 0.9999 | 3.1997 | 3.2x |
| steady_state | 0.9999 | 2.5997 | 2.6x |
| high_load | 1.0000 | 2.5999 | 2.6x |
| cooldown | 1.0006 | 2.6015 | 2.6x |

## Spike Seeds (M6 input)
| Fault Hint | Windows |
|------------|---------|
| pressure_transient | 408 |
| pressure_spike_high_load | 7 |
| impeller_cavitation | 113 |
| bearing_impact | 44 |
| mechanical_transient | 472 |

## Validation Gates
| Gate | Result |
|------|--------|
| GATE1_no_overfit | PASS |
| GATE2_mae_lt_006 | PASS |
| GATE3_threshold_range | PASS |
| GATE4_separation_gt3 | PASS |
| GATE5_false_alarms_lt1pct | PASS |
| GATE6_tv_channels_ok | PASS |
| GATE7_spike_seeds_saved | PASS |
| GATE8_val_loss_lt_005 | PASS |
| GATE9_pmpPV_startup_ceiling_correct | PASS |
| GATE10_pres_cluster_ceilings_ordered | PASS |