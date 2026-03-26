```
[22:23:45] STEP 1 — Scanning raw data directory...
[22:23:45] Found 9 files: ['Pump_A_Day1.csv', 'Pump_A_Day2.csv', 'Pump_A_Day3.csv', 'Pump_B_Day1.csv', 'Pump_B_Day2.csv', 'Pump_B_Day3.csv', 'Pump_C_Day



y1.csv', 'Pump_C_Day2.csv', 'Pump_C_Day3.csv']
[22:23:45] STEP 2 — Loading, cleaning, segmenting all files...
[22:23:45] Loading Pump_A_Day1.csv...
[22:23:45] 4,981 raw → 4,859 clean (2.45% dropped) | Worst null: X_Temp.SV (1.53%)
[22:23:45] Segments: 11 | Median Δt: 1.0s | Gap threshold: 8.0s
[22:23:45] Usable segments (≥70 rows): 3/11
[22:23:45] Saved → Pump_A_Day1_clean.csv
[22:23:45] Loading Pump_A_Day2.csv...
[22:23:45] 21,429 raw → 19,164 clean (10.57% dropped) | Worst null: Temperature (10.32%)
[22:23:45] Segments: 7 | Median Δt: 1.0s | Gap threshold: 2.0s
[22:23:45] Usable segments (≥70 rows): 4/7
[22:23:45] Saved → Pump_A_Day2_clean.csv
[22:23:45] Loading Pump_A_Day3.csv...
[22:23:45] 21,600 raw → 21,592 clean (0.04% dropped) | Worst null: X_ACR_Pmp.PV (0.03%)
[22:23:45] Segments: 3 | Median Δt: 1.0s | Gap threshold: 2.0s
[22:23:45] Usable segments (≥70 rows): 2/3
[22:23:45] Saved → Pump_A_Day3_clean.csv
[22:23:45] Loading Pump_B_Day1.csv...
[22:23:45] 4,981 raw → 4,636 clean (6.93% dropped) | Worst null: X_Temp.SV (3.45%)
[22:23:45] Segments: 19 | Median Δt: 1.0s | Gap threshold: 8.0s
[22:23:45] Usable segments (≥70 rows): 4/19
[22:23:45] Saved → Pump_B_Day1_clean.csv
[22:23:45] Loading Pump_B_Day2.csv...
[22:23:45] 35,829 raw → 33,956 clean (5.23% dropped) | Worst null: X_ACR_Pmp.TV (5.02%)
[22:23:45] Segments: 13 | Median Δt: 1.0s | Gap threshold: 2.0s
[22:23:45] Usable segments (≥70 rows): 5/13
[22:23:46] Saved → Pump_B_Day2_clean.csv
[22:23:46] Loading Pump_B_Day3.csv...
[22:23:46] 29,700 raw → 10,187 clean (65.7% dropped) | Worst null: Barometer (65.46%)
[22:23:46] Segments: 3 | Median Δt: 1.0s | Gap threshold: 2.0s
[22:23:46] Usable segments (≥70 rows): 2/3
[22:23:46] Saved → Pump_B_Day3_clean.csv
[22:23:46] Loading Pump_C_Day1.csv...
[22:23:46] 4,981 raw → 4,871 clean (2.21% dropped) | Worst null: X_Temp.SV (1.55%)
[22:23:46] Segments: 5 | Median Δt: 1.0s | Gap threshold: 8.0s
[22:23:46] Usable segments (≥70 rows): 3/5
[22:23:46] Saved → Pump_C_Day1_clean.csv
[22:23:46] Loading Pump_C_Day2.csv...
[22:23:46] 21,429 raw → 19,218 clean (10.32% dropped) | Worst null: Temperature (10.32%)
[22:23:46] Segments: 2 | Median Δt: 1.0s | Gap threshold: 2.0s
[22:23:46] Usable segments (≥70 rows): 2/2
[22:23:46] Saved → Pump_C_Day2_clean.csv
[22:23:46] Loading Pump_C_Day3.csv...
[22:23:46] 28,800 raw → 28,734 clean (0.23% dropped) | Worst null: X_ACR_Pmp.PV (0.19%)
[22:23:46] Segments: 3 | Median Δt: 1.0s | Gap threshold: 2.0s
[22:23:46] Usable segments (≥70 rows): 2/3
[22:23:46] Saved → Pump_C_Day3_clean.csv
[22:23:46] STEP 3 — Saving segment registry and file summary...
[22:23:46] Total segments : 66
[22:23:46] Usable (≥70r): 27
[22:23:46] Unusable : 39
[22:23:46] TOTAL: 173,730 raw → 147,217 clean | 15.26% dropped
[22:23:46] STEP 4 — Generating null heatmap...
[22:23:49] Saved → M1_null_heatmap.png
[22:23:49] STEP 5 — Generating segment timeline...
[22:23:49] Saved → M1_segment_timeline.png
[22:23:49] STEP 6 — Generating drop percentage chart...
[22:23:50] Saved → M1_drop_percentage.png
[22:23:50] STEP 7 — Generating usability chart...
[22:23:50] Saved → M1_segment_usability.png
[22:23:50] STEP 8 — Writing markdown report...
[22:23:50] Report saved → module_01_cleaning_report.md

═══════════════════════════════════════════════════════
PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT
═══════════════════════════════════════════════════════
M1_total_raw_rows : 173730
M1_total_clean_rows : 147217
M1_total_dropped : 26513
M1_overall_drop_pct : 15.26%
M1_total_segments : 66
M1_usable_segments : 27
M1_unusable_segments : 39
M1_worst_null_col : Barometer
M1_worst_null_pct : 65.46%
M1_sampling_interval : 1s uniform across all files
M1_A_Day3_fix : semicolon+comma-decimal+col-rename
M1_B_Day3_note : sensor fault (not timestamp gap)
M1_Day1_gap_threshold : 8s (above 5s natural jitter)
M1_Day2_gap_threshold : 344s (above 172s operational pauses)
M1_Day3_gap_threshold : 2s (continuous 1s data)
Status for M2 : READY
═══════════════════════════════════════════════════════

── FILE MANIFEST ──────────────────────────────────────
→ GitHub push (large data files):
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\Pump_A_Day1_clean.csv
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\Pump_A_Day2_clean.csv
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\Pump_A_Day3_clean.csv
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\Pump_B_Day1_clean.csv
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\Pump_B_Day2_clean.csv
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\Pump_B_Day3_clean.csv
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\Pump_C_Day1_clean.csv
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\Pump_C_Day2_clean.csv
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\Pump_C_Day3_clean.csv
C:\\Users\\user\\Desktop\\PumpSmart_Project\\data\\clean\\segment_registry.csv
→ Spaces upload (plots + report):
C:\\Users\\user\\Desktop\\PumpSmart_Project\\outputs\\plots\\M1_drop_percentage.png
C:\\Users\\user\\Desktop\\PumpSmart_Project\\outputs\\plots\\M1_null_heatmap.png
C:\\Users\\user\\Desktop\\PumpSmart_Project\\outputs\\plots\\M1_segment_timeline.png
C:\\Users\\user\\Desktop\\PumpSmart_Project\\outputs\\plots\\M1_segment_usability.png
C:\\Users\\user\\Desktop\\PumpSmart_Project\\outputs\\reports\\module_01_cleaning_report.md
───────────────────────────────────────────────────────

📦 M1 done. Starting M2.
```