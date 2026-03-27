# M3 Normalisation Report
**Date:** 2026-03-27  
**Script:** module_03_normalization  

## Summary
| Metric | Value |
|---|---|
| Normalised rows | 117,970 |
| Clusters used | 4 |
| Pressure cols | X_Pres.SV |
| Vibration cols | X_ACR_Mot.PV, X_ACR_Mot.SV, X_ACR_Pmp.PV, X_ACR_Pmp.SV |
| Temperature cols | X_ACR_Mot.TV, X_ACR_Pmp.TV, X_Temp.SV |
| Range issues | X_Pres.SV_norm, X_ACR_Mot.SV_norm, X_ACR_Pmp.SV_norm |
| Config saved | M3_normalization_config.json |

## Normalisation Formulas
- **Pressure:** P\* = P_actual / P_cluster_mean
- **Vibration:** a\* = a_actual / a_cluster_mean
- **Temperature:** ΔT\* = (T − T_ambient) / (T_cluster_max − T_ambient)

## Normal Operating Range
- All normalised values expected in **0.0 – 1.0**
- Values > 1.0 indicate elevated condition
- Values > 2.0 indicate potential fault

## Output Files
- `data/normalized/normalised_data.csv`
- `outputs/M3_normalization_config.json` → used by M4, M8, M10
- `outputs/plots/M3_raw_vs_norm_distributions.png`
- `outputs/plots/M3_norm_heatmap_by_cluster.png`
- `outputs/plots/M3_normalised_timeseries.png`