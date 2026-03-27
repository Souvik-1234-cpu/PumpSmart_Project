# M4 LSTM-AE Baseline Report (v3 — Physics Integrity)
**Date:** 2026-03-27  
**Script:** module_04_lstm_ae_baseline  

## v3 Changes vs v2
| Change | v2 | v3 |
|---|---|---|
| Epochs | 60 | 200 (early-stop aware) |
| LR Scheduler | ReduceOnPlateau | CosineAnnealingWarmRestarts T0=20 |
| Dropout | 0.2 | 0.3 |
| Gradient clip | 1.0 | 0.5 |
| Loss | 0.6MAE+0.4MSE | Physics-weighted channels |
| Threshold metric | Combined loss | Pure MAE (scale-free) |
| Overfit guard | None | Hard stop if val-train gap > 0.15 |
| Encoder | No norm | LayerNorm on bottleneck |

## Channel Physics Weights
| Channel | Weight | Reason |
|---|---|---|
| Mot.SV | 2.0 | ISO 10816 vibration — bearing fault indicator |
| Pmp.SV | 2.0 | Pump vibration — impeller cavitation |
| Pres.SV | 2.0 | Discharge pressure — NPSH violation |
| Mot.PV | 1.5 | Motor power — overload detection |
| Pmp.PV | 1.5 | Pump power — hydraulic efficiency |
| Temp.SV | 1.0 | Process temperature — baseline |
| Mot.TV | 0.8 | Motor temp — slow thermal lag |
| Pmp.TV | 0.8 | Pump temp — slow thermal lag |

## Training Results
| Metric | Value |
|---|---|
| Best val loss (weighted) | 0.251071 |
| Best epoch | 297 |
| Training time | 98s |
| Peak VRAM | 0.2 GB |
| Overfit triggered | False |

## Anomaly Threshold (Pure MAE)
| Metric | Value |
|---|---|
| Mean error | 0.049839 |
| Std error | 0.198503 |
| P95 | 0.099678 |
| P99 | 0.518297 |
| **Threshold** | **0.645347** |