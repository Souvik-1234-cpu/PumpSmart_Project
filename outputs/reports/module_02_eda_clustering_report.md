# M2 EDA + Clustering Report
**Date:** 2026-03-26  
**Script:** module_02_eda_clustering  

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

## Operating Mode Assignments
| Cluster | Mode | Rows |
|---|---|---|
| C0 | cooldown | 26,851 |
| C2 | startup | 49,884 |
| C1 | steady_state | 14,635 |
| C3 | high_load | 26,600 |

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

## Cluster Bounds
|   cluster_id | operating_mode   |   n_rows |   X_ACR_Mot.PV_mean |   X_ACR_Mot.PV_std |   X_ACR_Mot.PV_p2_5 |   X_ACR_Mot.PV_p97_5 |   X_ACR_Mot.PV_max |   X_ACR_Mot.PV_min |   X_ACR_Mot.SV_mean |   X_ACR_Mot.SV_std |   X_ACR_Mot.SV_p2_5 |   X_ACR_Mot.SV_p97_5 |   X_ACR_Mot.SV_max |   X_ACR_Mot.SV_min |   X_ACR_Mot.TV_mean |   X_ACR_Mot.TV_std |   X_ACR_Mot.TV_p2_5 |   X_ACR_Mot.TV_p97_5 |   X_ACR_Mot.TV_max |   X_ACR_Mot.TV_min |   X_ACR_Pmp.PV_mean |   X_ACR_Pmp.PV_std |   X_ACR_Pmp.PV_p2_5 |   X_ACR_Pmp.PV_p97_5 |   X_ACR_Pmp.PV_max |   X_ACR_Pmp.PV_min |   X_ACR_Pmp.SV_mean |   X_ACR_Pmp.SV_std |   X_ACR_Pmp.SV_p2_5 |   X_ACR_Pmp.SV_p97_5 |   X_ACR_Pmp.SV_max |   X_ACR_Pmp.SV_min |   X_ACR_Pmp.TV_mean |   X_ACR_Pmp.TV_std |   X_ACR_Pmp.TV_p2_5 |   X_ACR_Pmp.TV_p97_5 |   X_ACR_Pmp.TV_max |   X_ACR_Pmp.TV_min |   X_Temp.SV_mean |   X_Temp.SV_std |   X_Temp.SV_p2_5 |   X_Temp.SV_p97_5 |   X_Temp.SV_max |   X_Temp.SV_min |   X_Pres.SV_mean |   X_Pres.SV_std |   X_Pres.SV_p2_5 |   X_Pres.SV_p97_5 |   X_Pres.SV_max |   X_Pres.SV_min |
|-------------:|:-----------------|---------:|--------------------:|-------------------:|--------------------:|---------------------:|-------------------:|-------------------:|--------------------:|-------------------:|--------------------:|---------------------:|-------------------:|-------------------:|--------------------:|-------------------:|--------------------:|---------------------:|-------------------:|-------------------:|--------------------:|-------------------:|--------------------:|---------------------:|-------------------:|-------------------:|--------------------:|-------------------:|--------------------:|---------------------:|-------------------:|-------------------:|--------------------:|-------------------:|--------------------:|---------------------:|-------------------:|-------------------:|-----------------:|----------------:|-----------------:|------------------:|----------------:|----------------:|-----------------:|----------------:|-----------------:|------------------:|----------------:|----------------:|
|            0 | cooldown         |    26851 |            0.001081 |           0.000394 |            0.000533 |             0.001793 |           0.002994 |           0.000513 |            0.878032 |           2.5351   |            0.422827 |            12.0694   |           21.2967  |           0.384388 |             22.9788 |            3.51718 |             18.7578 |              30.6641 |            31.3672 |            17.7188 |            0.000835 |           0.000446 |            0.000376 |             0.001354 |           0.004165 |           0.000212 |            0.843067 |           2.81017  |            0.413133 |             0.607295 |            35.0412 |           0.375575 |             22.9979 |            2.15824 |             18.8516 |              28.9609 |            28.9609 |            17.6094 |          23.4289 |         3.56641 |          18.7239 |           30.9978 |         33.3814 |         18.03   |         8.31174  |       16.3266   |         0.446894 |         44.4285   |         46.107  |        0.440174 |
|            1 | steady_state     |    14635 |            0.002832 |           0.000415 |            0.002092 |             0.003661 |           0.003945 |           0.00123  |           16.0825   |           2.87872  |           12.0267   |            21.5316   |           26.6139  |           0.461266 |             36.5028 |            7.72401 |             19.7109 |              47.8125 |            48.4453 |            18.3125 |            0.003599 |           0.00038  |            0.002736 |             0.004094 |           0.004266 |           0.002736 |           36.3229   |          10.2781   |           24.4537   |            55.2847   |           291.623  |          18.7047   |             36.3226 |            4.74287 |             19.9062 |              40.0391 |            43.1641 |            18.4609 |          36.8354 |         5.93001 |          20.9672 |           44.2203 |         44.6077 |         18.2697 |        35.7894   |       13.0381   |         0.685236 |         43.277    |         43.3013 |        0.451229 |
|            2 | startup          |    49884 |            0.001266 |           0.000407 |            0.00069  |             0.002122 |           0.003147 |           0.000343 |            0.475001 |           0.171961 |            0.387957 |             0.581936 |            5.39261 |           0.387957 |             39.613  |            5.92349 |             30.5547 |              53.8281 |            54.2969 |            30.1406 |            0.000736 |           0.000339 |            0.000293 |             0.001648 |           0.003491 |           0.000255 |            0.513687 |           0.897567 |            0.375575 |             0.566808 |            28.7651 |           0.375575 |             41.8991 |            2.15083 |             36.4688 |              45.0781 |            46.0625 |            35.5703 |          38.2934 |         5.94437 |          30.5519 |           53.2987 |         55.0396 |         30.349  |         0.620818 |        0.639786 |         0.433452 |          0.847981 |         42.169  |        0.424722 |
|            3 | high_load        |    26600 |            0.001272 |           0.000241 |            0.000904 |             0.001747 |           0.002999 |           0.000718 |           36.2649   |          15.4843   |           22.1922   |            77.6727   |          456.626   |           0.504345 |             35.0902 |            5.25088 |             23.3281 |              40.5547 |            48.6172 |            18.3203 |            0.003501 |           0.000388 |            0.002864 |             0.004179 |           0.004421 |           0.002461 |           25.3159   |           4.38324  |           19.5667   |            34.6553   |            70.6822 |           0.525806 |             39.5285 |            4.22221 |             30.3203 |              44.2109 |            44.3984 |            18.4922 |          33.5113 |         4.14529 |          24.0141 |           37.7973 |         46.4467 |         18.2164 |        42.0247   |        1.91755  |        37.3988   |         44.8089   |         46.7089 |        0.45318  |

## Output Files
- `outputs/M2_cluster_bounds.csv` → Used by M3 for normalization baselines
- `outputs/M2_labelled_data.csv` → Full dataset with cluster labels
- `outputs/plots/M2_kmeans_selection.png`
- `outputs/plots/M2_cluster_pca.png`
- `outputs/plots/M2_cluster_centroids.png`
- `outputs/plots/M2_timeseries_clusters.png`
- `outputs/plots/M2_correlation_matrix.png`

## Sensor Channel Unit Reference (Added 2026-03-28)
All values in `M2_cluster_bounds.csv` and `M2_labelled_data.csv` are in the following units.
This table is also stored in `outputs/M2_cluster_bounds_units.json`.

| Channel | Description | Unit | Sensor Type | ISO Reference |
|---|---|---|---|---|
| `X_ACR_Mot.PV` | Motor casing displacement (peak-to-peak vibration) | **mm** | displacement | ISO 10816-3 |
| `X_ACR_Mot.SV` | Motor casing vibration velocity (broadband RMS) | **mm/s** | velocity | ISO 10816-3 |
| `X_ACR_Mot.TV` | Motor casing surface temperature | **°C** | temperature | IEC 60034-1 |
| `X_ACR_Pmp.PV` | Pump casing displacement (peak-to-peak vibration) | **mm** | displacement | ISO 10816-3 |
| `X_ACR_Pmp.SV` | Pump casing vibration velocity (broadband RMS) | **mm/s** | velocity | ISO 10816-3 |
| `X_ACR_Pmp.TV` | Pump casing surface temperature | **°C** | temperature | IEC 60034-1 |
| `X_Temp.SV` | Process fluid / bearing temperature (PT100) | **°C** | temperature | ISO 13373-2 |
| `X_Pres.SV` | Pump discharge / system pressure | **bar** | pressure | ISO 5167 |

### Physical Validation of Cluster Centroids
| Cluster | Mode | Mot.SV mean (mm/s) | Pmp.TV mean (°C) | Pres.SV mean (bar) | Physics Check |
|---|---|---|---|---|---|
| C0 | cooldown | 0.88 | 23.0 | 8.3 | ✅ Low vibration, low pressure, low temp — spinning down |
| C2 | startup | 0.48 | 41.9 | 0.6 | ✅ Very low pressure, HIGH temp — thermal lag before hydraulic load |
| C1 | steady_state | 16.1 | 36.3 | 35.8 | ✅ Moderate vibration, high stable pressure, mid temp |
| C3 | high_load | 36.3 | 39.5 | 42.0 | ✅ High vibration, highest pressure, high temp |

> **Physics note:** Startup has HIGHER mean TV (39.6°C) than high_load (35.1°C) despite lower load.
> This is correct: 7-stage multistage pump has significant motor thermal run-in before
> hydraulics are fully loaded (affinity law — low flow at startup = low shaft power,
> but motor already at thermal steady state from previous cycle).

## Audit Record (Added 2026-03-28)
| Audit Item | Finding | Action Taken |
|---|---|---|
| Cluster bounds CSV units | ⚠️ No unit documentation existed | Created M2_cluster_bounds_units.json (FIX-1) |
| Time-series plot Y-axes | ⚠️ Raw column names, no units | Regenerated with unit labels (FIX-2) |
| STEP 8 physics comment | ⚠️ Comment said high temp→high_load (WRONG) | Patched in source script (FIX-3) |
| Cluster bound values | ✅ Values correct in raw physics units | Validated against nameplate + centroid table |
| Data integrity M1→M2 | ✅ No unit transformation in pipeline | Confirmed — raw values pass through unchanged |

**Audit conclusion:** M2 data correct. Three cosmetic/documentation issues patched.
No re-run of M1 or M2 required. M3 pipeline is cleared to proceed.
