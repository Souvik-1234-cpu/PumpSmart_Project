# M2 EDA + Clustering Report
**Date:** 2026-03-26 | **Audit-patched:** 2026-03-28  
**Script:** module_02_eda_clustering  

> **AUDIT NOTE (2026-03-28):** Unit annotations, cluster bounds companion JSON, and physics paradox documentation added. No data values changed — all statistics remain as originally computed.

## Summary
| Metric | Value |
|---|---|
| Usable rows | 117,970 |
| Usable segments | 25 |
| Optimal K | 4 |
| Silhouette score | 0.5458 |
| Optimal window size | 50s |
| Stationary sensors | 8/8 |
| Top correlation | X_ACR_Mot.TV ↔ X_Temp.SV: r=0.9793 |

## Sensor Unit Reference
| Channel | Description | Unit | Role in ML |
|---|---|---|---|
| X_ACR_Mot.PV | Motor bearing velocity (peak) | mm/s | Secondary |
| X_ACR_Mot.SV | Motor bearing acceleration (broadband RMS envelope) | m/s2 | **Primary** |
| X_ACR_Mot.TV | Motor bearing temperature | degC | Corroborating only |
| X_ACR_Pmp.PV | Pump bearing velocity (peak) | mm/s | Secondary |
| X_ACR_Pmp.SV | Pump bearing acceleration (broadband RMS envelope) | m/s2 | **Primary** |
| X_ACR_Pmp.TV | Pump bearing temperature | degC | Corroborating only |
| X_Temp.SV | Process fluid temperature | degC | T_ambient proxy in M3 |
| X_Pres.SV | Discharge pressure | bar | **Primary** |
| Barometer | Atmospheric pressure | hPa | Environmental — NOT in clustering |
| Temperature | Ambient air temperature | degC | T_ambient per-row in M3 |

## Operating Mode Assignments
| Cluster | Mode | Rows | Mot.SV mean (m/s2) | Pres.SV mean (bar) | Mot.TV mean (degC) |
|---|---|---|---|---|---|
| C0 | cooldown | 26,851 | 0.88 | 8.31 | 22.98 |
| C2 | startup | 49,884 | 0.48 | 0.62 | 39.61 |
| C1 | steady_state | 14,635 | 16.08 | 35.79 | 36.50 |
| C3 | high_load | 26,600 | 36.26 | 42.02 | 35.09 |

## K-Means Thermal Run-In Paradox — Audit Documentation
**Observation:** Startup cluster (C2) mean temperature 39.61 degC > High-load (C3) mean 35.09 degC.  
**Engineering explanation:** During startup, bearings heat rapidly from cold while flow/pressure is still ramping up. High-load steady operation sees turbulent cooling from high volumetric flow, settling bearings at a lower equilibrium temperature. This is physically correct industrial pump behaviour.  
**Consequence:** Temperature channels are **corroborating only**. Vibration (SV) and pressure (Pres.SV) are primary operating mode discriminators. The original mode labelling algorithm correctly used vibration+pressure rank. The STEP 8 code comment ("High temp + high vibration -> high_load") was misleading and has been corrected in the script.  
**Safe for M3:** M3 normalisation uses per-cluster baselines from M2_cluster_bounds.csv. The rate-of-change (dX*/dt) approach in M5/M6 fault rules is immune to absolute temperature ordering.

## ADF Stationarity Results
  - X_ACR_Mot.PV: Stationary (p=0.0)
  - X_ACR_Mot.SV: Stationary (p=0.0)
  - X_ACR_Mot.TV: Stationary (p=0.00044)
  - X_ACR_Pmp.PV: Stationary (p=0.0)
  - X_ACR_Pmp.SV: Stationary (p=0.0)
  - X_ACR_Pmp.TV: Stationary (p=1e-06)
  - X_Temp.SV: Stationary (p=0.000857)
  - X_Pres.SV: Stationary (p=2e-06)

## Top Correlations
  - X_ACR_Mot.TV ↔ X_Temp.SV: r=0.9793
  - X_ACR_Pmp.PV ↔ X_ACR_Pmp.SV: r=0.8882
  - X_ACR_Pmp.PV ↔ X_Pres.SV: r=0.8779

## Cluster Bounds (units documented in M2_cluster_bounds_units.json)
| cluster_id | operating_mode | n_rows | X_ACR_Mot.SV_mean (m/s2) | X_ACR_Mot.TV_mean (degC) | X_Pres.SV_mean (bar) | X_ACR_Mot.SV_p97_5 | X_Pres.SV_p97_5 |
|---|---|---|---|---|---|---|---|
| 0 | cooldown | 26,851 | 0.878 | 22.98 | 8.31 | 12.07 | 44.43 |
| 1 | steady_state | 14,635 | 16.08 | 36.50 | 35.79 | 21.53 | 43.28 |
| 2 | startup | 49,884 | 0.475 | 39.61 | 0.62 | 0.582 | 0.848 |
| 3 | high_load | 26,600 | 36.26 | 35.09 | 42.02 | 77.67 | 44.81 |

> Full bounds table (all 48 statistics columns) available in outputs/M2_cluster_bounds.csv

## Output Files
- `outputs/M2_cluster_bounds.csv` -> Used by M3 for normalization baselines
- `outputs/M2_cluster_bounds_units.json` -> **NEW** Unit annotations for all cluster bounds columns
- `outputs/unit_registry.json` -> **NEW** Master sensor unit registry (authoritative reference)
- `outputs/M2_labelled_data.csv` -> Full dataset with cluster labels
- `outputs/plots/M2_kmeans_selection.png`
- `outputs/plots/M2_cluster_pca.png`
- `outputs/plots/M2_cluster_centroids.png`
- `outputs/plots/M2_timeseries_clusters.png`
- `outputs/plots/M2_correlation_matrix.png`
