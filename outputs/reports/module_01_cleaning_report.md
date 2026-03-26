# M1 Cleaning Report
**Date:** 2026-03-25  
**Script:** module_01_cleaning  

## Summary
| Metric | Value |
|---|---|
| Raw files processed | 9 |
| Total raw rows | 173,730 |
| Total clean rows | 147,217 |
| Total dropped rows | 26,513 |
| Overall drop % | 15.26% |
| Total segments | 66 |
| Usable segments (≥70 rows) | 27 |
| Unusable segments | 39 |
| Worst null column | Barometer |
| Worst null % | 65.46% |

## Per-File Breakdown
| file            | pump_id   | day_id   |   raw_rows |   clean_rows |   dropped_rows |   drop_pct |   n_segments |   usable_segments | worst_null_col   |   worst_null_pct |   median_interval |   gap_threshold_s |
|:----------------|:----------|:---------|-----------:|-------------:|---------------:|-----------:|-------------:|------------------:|:-----------------|-----------------:|------------------:|------------------:|
| Pump_A_Day1.csv | A         | Day1     |       4981 |         4859 |            122 |       2.45 |           11 |                 3 | X_Temp.SV        |             1.53 |                 1 |                 8 |
| Pump_A_Day2.csv | A         | Day2     |      21429 |        19164 |           2265 |      10.57 |            7 |                 4 | Temperature      |            10.32 |                 1 |                 2 |
| Pump_A_Day3.csv | A         | Day3     |      21600 |        21592 |              8 |       0.04 |            3 |                 2 | X_ACR_Pmp.PV     |             0.03 |                 1 |                 2 |
| Pump_B_Day1.csv | B         | Day1     |       4981 |         4636 |            345 |       6.93 |           19 |                 4 | X_Temp.SV        |             3.45 |                 1 |                 8 |
| Pump_B_Day2.csv | B         | Day2     |      35829 |        33956 |           1873 |       5.23 |           13 |                 5 | X_ACR_Pmp.TV     |             5.02 |                 1 |                 2 |
| Pump_B_Day3.csv | B         | Day3     |      29700 |        10187 |          19513 |      65.7  |            3 |                 2 | Barometer        |            65.46 |                 1 |                 2 |
| Pump_C_Day1.csv | C         | Day1     |       4981 |         4871 |            110 |       2.21 |            5 |                 3 | X_Temp.SV        |             1.55 |                 1 |                 8 |
| Pump_C_Day2.csv | C         | Day2     |      21429 |        19218 |           2211 |      10.32 |            2 |                 2 | Temperature      |            10.32 |                 1 |                 2 |
| Pump_C_Day3.csv | C         | Day3     |      28800 |        28734 |             66 |       0.23 |            3 |                 2 | X_ACR_Pmp.PV     |             0.19 |                 1 |                 2 |

## Segment Registry (first 20 rows)
| segment_id   | pump_id   | day_id   | source_file     |   n_rows | start_time          | end_time            |   duration_s |   duration_min | usable_for_windowing   |
|:-------------|:----------|:---------|:----------------|---------:|:--------------------|:--------------------|-------------:|---------------:|:-----------------------|
| A_Day1_seg1  | A         | Day1     | Pump_A_Day1.csv |       83 | 2024-04-10 12:00:00 | 2024-04-10 12:06:50 |          410 |           6.83 | True                   |
| A_Day1_seg10 | A         | Day1     | Pump_A_Day1.csv |     3000 | 2024-04-10 12:20:40 | 2024-04-10 13:31:16 |         4236 |          70.6  | True                   |
| A_Day1_seg11 | A         | Day1     | Pump_A_Day1.csv |     1662 | 2024-04-10 13:32:18 | 2024-04-10 13:59:59 |         1661 |          27.68 | True                   |
| A_Day1_seg2  | A         | Day1     | Pump_A_Day1.csv |       23 | 2024-04-10 12:07:00 | 2024-04-10 12:08:50 |          110 |           1.83 | False                  |
| A_Day1_seg3  | A         | Day1     | Pump_A_Day1.csv |        3 | 2024-04-10 12:09:00 | 2024-04-10 12:09:10 |           10 |           0.17 | False                  |
| A_Day1_seg4  | A         | Day1     | Pump_A_Day1.csv |       23 | 2024-04-10 12:10:25 | 2024-04-10 12:12:15 |          110 |           1.83 | False                  |
| A_Day1_seg5  | A         | Day1     | Pump_A_Day1.csv |       11 | 2024-04-10 12:12:25 | 2024-04-10 12:13:15 |           50 |           0.83 | False                  |
| A_Day1_seg6  | A         | Day1     | Pump_A_Day1.csv |        8 | 2024-04-10 12:14:45 | 2024-04-10 12:15:20 |           35 |           0.58 | False                  |
| A_Day1_seg7  | A         | Day1     | Pump_A_Day1.csv |       19 | 2024-04-10 12:15:30 | 2024-04-10 12:17:00 |           90 |           1.5  | False                  |
| A_Day1_seg8  | A         | Day1     | Pump_A_Day1.csv |       21 | 2024-04-10 12:17:10 | 2024-04-10 12:18:50 |          100 |           1.67 | False                  |
| A_Day1_seg9  | A         | Day1     | Pump_A_Day1.csv |        6 | 2024-04-10 12:19:00 | 2024-04-10 12:19:25 |           25 |           0.42 | False                  |
| A_Day2_seg1  | A         | Day2     | Pump_A_Day2.csv |      671 | 2024-06-11 10:00:00 | 2024-06-11 10:11:10 |          670 |          11.17 | True                   |
| A_Day2_seg2  | A         | Day2     | Pump_A_Day2.csv |      293 | 2024-06-11 10:21:56 | 2024-06-11 10:26:49 |          293 |           4.88 | True                   |
| A_Day2_seg3  | A         | Day2     | Pump_A_Day2.csv |    13328 | 2024-06-11 10:26:53 | 2024-06-11 14:09:00 |        13327 |         222.12 | True                   |
| A_Day2_seg4  | A         | Day2     | Pump_A_Day2.csv |        2 | 2024-06-11 14:09:06 | 2024-06-11 14:09:07 |            1 |           0.02 | False                  |
| A_Day2_seg5  | A         | Day2     | Pump_A_Day2.csv |       37 | 2024-06-11 14:09:12 | 2024-06-11 14:09:48 |           36 |           0.6  | False                  |
| A_Day2_seg6  | A         | Day2     | Pump_A_Day2.csv |        5 | 2024-06-11 14:10:04 | 2024-06-11 14:10:08 |            4 |           0.07 | False                  |
| A_Day2_seg7  | A         | Day2     | Pump_A_Day2.csv |     4828 | 2024-06-11 14:10:25 | 2024-06-11 15:30:52 |         4827 |          80.45 | True                   |
| A_Day3_seg1  | A         | Day3     | Pump_A_Day3.csv |     2040 | 2024-10-30 10:30:00 | 2024-10-30 11:03:59 |         2039 |          33.98 | True                   |
| A_Day3_seg2  | A         | Day3     | Pump_A_Day3.csv |        5 | 2024-10-30 11:04:02 | 2024-10-30 11:04:06 |            4 |           0.07 | False                  |

## Key Engineering Findings
- All 9 files: 1-second sampling confirmed uniform
- Day1 files: 8× gap threshold (above 5s natural jitter)
- Day2 files: 2× gap threshold (above 172s operational pauses)
- Day3 files: 2× gap threshold (perfectly continuous 1s data)
- Pump_B_Day3: Barometer+Temperature sensor failure — 65.5% rows dropped
  Pump ran continuously; sensor logged NaN not timestamp gaps
  Clean segments before/after fault block are valid training data
- Segments < 70 rows flagged unusable for LSTM windowing
- Hard NaN policy: zero interpolation enforced

## Output Files
- `data/clean/Pump_*_clean.csv` — 9 cleaned CSVs with segment_id
- `data/clean/segment_registry.csv` — master segment index with usability flag
- `outputs/M1_file_summary.csv`
- `outputs/plots/M1_null_heatmap.png`
- `outputs/plots/M1_segment_timeline.png`
- `outputs/plots/M1_drop_percentage.png`
- `outputs/plots/M1_segment_usability.png`