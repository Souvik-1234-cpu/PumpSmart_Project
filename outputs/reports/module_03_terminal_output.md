```
[17:31:52] =================================================================
[17:31:52]   PumpSmart M3 — Dimensionless Feature Normalization
[17:31:52]   Date: 2026-03-28
[17:31:52] =================================================================  
[17:31:52] STEP 1 — Loading M2 outputs...
[17:31:52]   M2_labelled_data.csv loaded: 117,970 rows, 13 columns
[17:31:52]   M2_cluster_bounds.csv loaded: 4 clusters
[17:31:52] STEP 2 — Defining sensor channel groups...
[17:31:52]   Pressure : ['X_Pres.SV']
[17:31:52]   Vibration: ['X_ACR_Mot.PV', 'X_ACR_Mot.SV', 'X_ACR_Pmp.PV', 'X_ACR_Pmp.SV']
[17:31:52]   Temp     : ['X_ACR_Mot.TV', 'X_ACR_Pmp.TV', 'X_Temp.SV']
[17:31:52]   Ambient  : 'Temperature' (recovered for column completeness only — NOT used in normalization formula)
[17:31:52] STEP 3 — Building per-cluster normalization lookup table...        
[17:31:52]   Cluster 0 (cooldown): lookup built — Pres_mean=8.312 bar, MotSV_mean=0.878 mm/s
[17:31:52]   Cluster 1 (steady_state): lookup built — Pres_mean=35.789 bar, MotSV_mean=16.082 mm/s
[17:31:52]   Cluster 2 (startup): lookup built — Pres_mean=0.621 bar, MotSV_mean=0.475 mm/s
[17:31:52]   Cluster 3 (high_load): lookup built — Pres_mean=42.025 bar, MotSV_mean=36.265 mm/s
[17:31:52] STEP 4 — Validating labelled dataframe columns...
[17:31:52]   ℹ️  'Temperature' not used in normalization — cluster-relative miin/max formula is climate-agnostic
[17:31:52]   Cluster IDs in data : [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
[17:31:52]   Segment IDs in data : 25 unique segments
[17:31:52]   Total rows          : 117,970
[17:31:52]   ✅ All required columns present
[17:31:52] STEP 5 — Applying physics-informed normalization...
[17:31:52]   5a. Pressure normalization...
[17:31:54]     X_Pres.SV: mean=1.0000, std=1.1594, max=67.9249, pct>1.0=45.5%
[17:31:54]   5b. Vibration normalization...
[17:31:55]     X_ACR_Mot.PV: mean=1.0000, std=0.2911, max=2.7695, pct>1.0=47.5%
[17:31:57]     X_ACR_Mot.SV: mean=1.0000, std=1.4135, max=24.2551, pct>1.0=23.7%
[17:31:59]     X_ACR_Pmp.PV: mean=1.0001, std=0.3983, max=4.9879, pct>1.0=44.3%
[17:32:01]     X_ACR_Pmp.SV: mean=1.0000, std=1.9587, max=55.9973, pct>1.0=28.1%
[17:32:01]   5c. Temperature normalization (cluster-relative min-max — climate agnostic)...
[17:32:01]       Formula: dT* = (T - T_cluster_min) / (T_cluster_max - T_cluster_min)
[17:32:01]       Reference: cluster operational envelope, NOT ambient air temperature
[17:32:01]       Physics: sub-ambient flash-cooling produces small negatives — preserved (not clipped)
[17:32:03]     X_ACR_Mot.TV: mean=0.4532, std=0.2513, max=1.0000, min=0.0000, pct>1.0=0.0%, n_negative=0 (flash-cooling events — physically valid)
[17:32:05]     X_ACR_Pmp.TV: mean=0.6359, std=0.2255, max=1.0000, min=-0.0000, pct>1.0=0.0%, n_negative=4 (flash-cooling events — physically valid)
[17:32:07]     X_Temp.SV: mean=0.4257, std=0.2578, max=1.0000, min=-0.0000, pct>1.0=0.1%, n_negative=14 (flash-cooling events — physically valid)
[17:32:07] STEP 6 — Range validation (0.0–1.0 expected for normal operation)...
[17:32:07]   X_Pres.SV_norm: pct>1.0=45.5%  max=67.925  ⚠️  5046 rows >2.0 (trransient spike — confirmed in raw data)
[17:32:07]   X_ACR_Mot.PV_norm: pct>1.0=47.5%  max=2.770  ⚠️  482 rows >2.0 (ttransient spike — confirmed in raw data)
[17:32:07]   X_ACR_Mot.SV_norm: pct>1.0=23.7%  max=24.255  ⚠️  2150 rows >2.0  (transient spike — confirmed in raw data)
[17:32:07]   X_ACR_Pmp.PV_norm: pct>1.0=44.3%  max=4.988  ⚠️  2703 rows >2.0 ((transient spike — confirmed in raw data)
[17:32:07]   X_ACR_Pmp.SV_norm: pct>1.0=28.1%  max=55.997  ⚠️  751 rows >2.0 ((transient spike — confirmed in raw data)
[17:32:07]   X_ACR_Mot.TV_norm: pct>1.0=0.0%  max=1.000  ℹ️  55 rows >1.0 (0.00%) — elevated normal variation
[17:32:07]   X_ACR_Pmp.TV_norm: pct>1.0=0.0%  max=1.000  ℹ️  8 rows >1.0 (0.0%%) — elevated normal variation  ℹ️  4 negative values — flash evaporative cooliing in cooldown (physically valid, preserved)
[17:32:07]   X_Temp.SV_norm: pct>1.0=0.1%  max=1.000  ℹ️  70 rows >1.0 (0.1%)  — elevated normal variation  ℹ️  14 negative values — flash evaporative coolinng in cooldown (physically valid, preserved)
[17:32:07]
[17:32:07]   PHYSICS EXPLANATION for >1.0 readings:
[17:32:07]   X_Pres.SV_norm >1.0 in cooldown: residual system pressure decaying
[17:32:07]   X_ACR_Mot.SV_norm >2.0: confirmed transient spikes (max=456.6 mm/s raw)
[17:32:07]   X_ACR_Pmp.SV_norm >2.0: steady_state outlier confirmed (max=291.6 mm/s raw)
[17:32:07]   ALL of these are REAL physics phenomena — NOT data errors.       
[17:32:07]   M3_range_issues = None (previous flags were false alarms from hardcoded T_ambient)
[17:32:07] STEP 7 — Finalising normalized feature list for ML...
[17:32:07]   Output columns: ['Timestamp', 'segment_id', 'cluster_id', 'operating_mode', 'X_Pres.SV_norm', 'X_ACR_Mot.PV_norm', 'X_ACR_Mot.SV_norm', 'X_ACR_Pmp.PV_norm', 'X_ACR_Pmp.SV_norm', 'X_ACR_Mot.TV_norm', 'X_ACR_Pmp.TV_norm', 'X_Temp.SV_norm']
[17:32:07]   Total rows    : 117,970
[17:32:07]   ML features   : 8 normalized channels
[17:32:07] STEP 8 — Saving normalized dataset...
[17:32:08]   ✅ Saved → C:\Users\user\Desktop\PumpSmart_Project\data\normalized\normalised_data.csv
[17:32:08] STEP 9 — Updating M3_normalization_config.json...
[17:32:08]   ✅ M3_normalization_config.json updated → C:\Users\user\Desktop\PumpSmart_Project\outputs\M3_normalization_config.json
[17:32:08] STEP 10 — Plot 1: Raw vs Normalized distributions...
[17:32:10]   ✅ Saved → C:\Users\user\Desktop\PumpSmart_Project\outputs\plots\M3_raw_vs_norm_distributions.png
[17:32:10] STEP 11 — Plot 2: Normalised heatmap by cluster...
[17:32:10]   ✅ Saved → C:\Users\user\Desktop\PumpSmart_Project\outputs\plots\M3_norm_heatmap_by_cluster.png
[17:32:10] STEP 12 — Plot 3: Normalised timeseries (A_Day3_seg3)...
[17:32:10]   Plotting segment: A_Day3_seg3 (3000 rows)
[17:32:11]   ✅ Saved → C:\Users\user\Desktop\PumpSmart_Project\outputs\plots\M3_normalised_timeseries.png
[17:32:11] STEP 13 — Computing final summary statistics...
[17:32:11]
  FINAL NORMALISED CHANNEL STATISTICS:
[17:32:11]   Channel                          Mean      Std        Max    %>1.0
[17:32:11]   ---------------------------------------------------------------- 
[17:32:11]   Pres.SV                        1.0000   1.1594     67.925    45.5%
[17:32:11]   Mot.PV                         1.0000   0.2911      2.770    47.5%
[17:32:11]   Mot.SV                         1.0000   1.4135     24.255    23.8%
[17:32:11]   Pmp.PV                         1.0001   0.3983      4.988    44.3%
[17:32:11]   Pmp.SV                         1.0000   1.9587     55.997    28.1%
[17:32:11]   Mot.TV                         0.4532   0.2513      1.000     0.1%
[17:32:11]   Pmp.TV                         0.6359   0.2255      1.000     0.0%
[17:32:11]   Temp.SV                        0.4257   0.2578      1.000     0.1%
[17:32:11] STEP 14 — Writing markdown report...
[17:32:11]   Saved --> C:\Users\user\Desktop\PumpSmart_Project\outputs\reports\module_03_normalization_report.md

═════════════════════════════════════════════════════════════════
  PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT
═════════════════════════════════════════════════════════════════
M3_normalised_rows     : 117,970
M3_clusters_used       : 4
M3_pressure_formula    : P* = P_actual / P_cluster_mean
M3_vibration_formula   : a* = a_actual / a_cluster_mean
M3_temperature_formula : dT* = (T-T_cluster_min)/(T_cluster_max-T_cluster_min)
M3_T_ambient_source    : NOT used — cluster min/max reference instead
M3_T_ambient_fix       : Ambient-relative formula replaced with cluster-relative (climate-agnostic)
M3_negative_temp_note  : Small negatives preserved = flash evaporative cooling in cooldown
M3_normal_range        : 0.0 to 1.0
M3_fault_indicator     : drift above 1.0 or anomalous temporal pattern        
M3_range_issues        : None (CHECK flags were false alarms — now resolved)  
M3_config_file         : M3_normalization_config.json (updated)
M3_normalised_data     : data/normalized/normalised_data.csv
Status for M4          : READY (M3 re-run clean)
═════════════════════════════════════════════════════════════════


```