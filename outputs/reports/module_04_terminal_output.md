```
[22:34:45] =================================================================
[22:34:45] PHASE 0 — Loading M3 normalised data + cluster config
[22:34:45] =================================================================
[22:34:45] Loaded: 117,970 rows | 25 segments
[22:34:45] Registry: 66 segments | 25 usable
[22:34:45] M3 cluster modes: {'0': 'cooldown', '1': 'steady_state', '2': 'startup', '3': 'high_load'}
[22:34:45] Operating mode distribution:
{'startup': 49884, 'cooldown': 26851, 'high_load': 26600, 'steady_state': 14635}
[22:34:45]
=================================================================
[22:34:45] PHASE 1 — Cluster-conditional winsor bounds + spike thresholds
[22:34:45] =================================================================
[22:34:45] X_Pres.SV_norm [startup]: mean=1.0000 std=1.0306 upper=3.0000 (3.0x mean)
[22:34:45] X_Pres.SV_norm [steady_state]: mean=1.0000 std=0.3643 upper=5.6000 (5.6x mean)
[22:34:45] X_Pres.SV_norm [high_load]: mean=1.0000 std=0.0456 upper=2.0000 (2.0x mean)
[22:34:45] X_Pres.SV_norm [cooldown]: mean=1.0000 std=1.9643 upper=3.0000 (3.0x mean)
[22:34:45] X_ACR_Mot.SV_norm [startup]: mean=1.0000 std=0.3620 upper=6.7000 (6.7x mean)
[22:34:45] X_ACR_Mot.SV_norm [steady_state]: mean=1.0000 std=0.1790 upper=6.7000 (6.7x mean)
[22:34:45] X_ACR_Mot.SV_norm [high_load]: mean=1.0000 std=0.4270 upper=6.7000 (6.7x mean)
[22:34:45] X_ACR_Mot.SV_norm [cooldown]: mean=1.0000 std=2.8873 upper=6.7000 (6.7x mean)
[22:34:45] X_ACR_Pmp.SV_norm [startup]: mean=1.0000 std=1.7473 upper=8.8000 (8.8x mean)
[22:34:45] X_ACR_Pmp.SV_norm [steady_state]: mean=1.0000 std=0.2830 upper=8.8000 (8.8x mean)
[22:34:45] X_ACR_Pmp.SV_norm [high_load]: mean=1.0000 std=0.1731 upper=8.8000 (8.8x mean)
[22:34:45] X_ACR_Pmp.SV_norm [cooldown]: mean=1.0000 std=3.3333 upper=8.8000 (8.8x mean)
[22:34:45] X_ACR_Mot.PV_norm [startup]: mean=1.0001 std=0.3215 upper=2.2003 (2.2x mean)
[22:34:45] X_ACR_Mot.PV_norm [steady_state]: mean=1.0001 std=0.1465 upper=2.2001 (2.2x mean)
[22:34:45] X_ACR_Mot.PV_norm [high_load]: mean=0.9999 std=0.1894 upper=2.1998 (2.2x mean)
[22:34:45] X_ACR_Mot.PV_norm [cooldown]: mean=0.9999 std=0.3649 upper=2.1998 (2.2x mean)
[22:34:45] X_ACR_Pmp.PV_norm [startup]: mean=0.9999 std=0.4600 upper=3.1997 (3.2x mean)
[22:34:45] X_ACR_Pmp.PV_norm [steady_state]: mean=0.9999 std=0.1057 upper=2.5997 (2.6x mean)
[22:34:45] X_ACR_Pmp.PV_norm [high_load]: mean=1.0000 std=0.1110 upper=2.5999 (2.6x mean)
[22:34:45] X_ACR_Pmp.PV_norm [cooldown]: mean=1.0006 std=0.5345 upper=2.6015 (2.6x mean)
[22:34:45]
FIX-3: pressure_transient spike ratio reference = high_load Pres.SV mean = 1.0000
[22:34:45] (Water hammer energy governed by operating pressure ~42 bar, not startup ~0.62 bar)
[22:34:45]
=================================================================
[22:34:45] PHASE 2 — Spike extraction → M4_spike_seeds (cluster-aware)
[22:34:45] =================================================================
[22:34:45] Spike windows extracted: 1,044
[22:34:45] → pressure_transient: 408
[22:34:45] → pressure_spike_high_load: 7
[22:34:45] → impeller_cavitation: 113
[22:34:45] → bearing_impact: 44
[22:34:45] → mechanical_transient: 472
[22:34:45] Spike row indices: 12,620 rows flagged for exclusion
[22:34:45] Saved: M4_spike_seeds.npy shape=(1044, 50, 8)
[22:34:45] Saved: M4_spike_seeds_meta.csv rows=1044
[22:34:45] Saved: M4_spike_config.json (v8 — cluster-conditional bounds)
[22:34:45]
=================================================================
[22:34:45] PHASE 3 — Cluster-conditional winsorization (v8 physics-correct)
[22:34:45] =================================================================
[22:34:45] X_Pres.SV_norm [startup]: upper=3.0000 (3.0x) | clipped=62
[22:34:45] X_Pres.SV_norm [steady_state]: upper=5.6000 (5.6x) | clipped=0
[22:34:45] X_Pres.SV_norm [high_load]: upper=2.0000 (2.0x) | clipped=0
[22:34:45] X_Pres.SV_norm [cooldown]: upper=3.0000 (3.0x) | clipped=4984
[22:34:45] X_Pres.SV_norm total clipped: 5046 rows

[22:34:45] X_ACR_Mot.SV_norm [startup]: upper=6.7000 (6.7x) | clipped=78
[22:34:45] X_ACR_Mot.SV_norm [steady_state]: upper=6.7000 (6.7x) | clipped=0
[22:34:45] X_ACR_Mot.SV_norm [high_load]: upper=6.7000 (6.7x) | clipped=1
[22:34:45] X_ACR_Mot.SV_norm [cooldown]: upper=6.7000 (6.7x) | clipped=688
[22:34:45] X_ACR_Mot.SV_norm total clipped: 767 rows

[22:34:45] X_ACR_Pmp.SV_norm [startup]: upper=8.8000 (8.8x) | clipped=56
[22:34:45] X_ACR_Pmp.SV_norm [steady_state]: upper=8.8000 (8.8x) | clipped=0
[22:34:45] X_ACR_Pmp.SV_norm [high_load]: upper=8.8000 (8.8x) | clipped=0
[22:34:45] X_ACR_Pmp.SV_norm [cooldown]: upper=8.8000 (8.8x) | clipped=402
[22:34:45] X_ACR_Pmp.SV_norm total clipped: 458 rows

[22:34:45] X_ACR_Mot.PV_norm [startup]: upper=2.2003 (2.2x) | clipped=79
[22:34:45] X_ACR_Mot.PV_norm [steady_state]: upper=2.2001 (2.2x) | clipped=0
[22:34:45] X_ACR_Mot.PV_norm [high_load]: upper=2.1998 (2.2x) | clipped=1
[22:34:45] X_ACR_Mot.PV_norm [cooldown]: upper=2.1998 (2.2x) | clipped=230
[22:34:45] X_ACR_Mot.PV_norm total clipped: 310 rows

[22:34:45] X_ACR_Pmp.PV_norm [startup]: upper=3.1997 (3.2x) | clipped=49
[22:34:45] X_ACR_Pmp.PV_norm [steady_state]: upper=2.5997 (2.6x) | clipped=0
[22:34:45] X_ACR_Pmp.PV_norm [high_load]: upper=2.5999 (2.6x) | clipped=0
[22:34:45] X_ACR_Pmp.PV_norm [cooldown]: upper=2.6015 (2.6x) | clipped=395
[22:34:45] X_ACR_Pmp.PV_norm total clipped: 444 rows

[22:34:45] Total rows modified: 7,025 (5.955% of dataset)
[22:34:45] Temperature channels untouched (cluster-relative min-max, max=1.0 by design)
[22:34:45]
Post-winsorization channel maxima:
[22:34:45] X_ACR_Mot.PV_norm: max=2.2003 | min=0.271162
[22:34:45] X_ACR_Mot.SV_norm: max=6.7000 | min=0.013907
[22:34:45] X_ACR_Mot.TV_norm: max=1.0000 | min=0.000000
[22:34:45] X_ACR_Pmp.PV_norm: max=3.1997 | min=0.253700
[22:34:45] X_ACR_Pmp.SV_norm: max=8.8000 | min=0.020770
[22:34:45] X_ACR_Pmp.TV_norm: max=1.0000 | min=-0.000000
[22:34:45] X_Temp.SV_norm: max=1.0000 | min=-0.000000
[22:34:45] X_Pres.SV_norm: max=3.0000 | min=0.010784
[22:34:45]
PHASE 3.5 — Spike row exclusion
[22:34:45] Rows removed: 12,620
[22:34:45] Clean rows remaining: 105,350
[22:34:45]
=================================================================
[22:34:45] PHASE 4 — Window generation on clean data
[22:34:45] =================================================================
[22:34:45] SKIP A_Day1_seg1: only 0 rows after warmup
[22:34:46] A_Day1_seg10: 1360 rows → 132 windows
[22:34:46] A_Day1_seg11: 1362 rows → 132 windows
[22:34:46] A_Day2_seg1: 181 rows → 14 windows
[22:34:46] SKIP A_Day2_seg2: only 0 rows after warmup
[22:34:46] A_Day2_seg3: 10448 rows → 1040 windows
[22:34:46] A_Day2_seg7: 4528 rows → 448 windows
[22:34:46] A_Day3_seg1: 1410 rows → 137 windows
[22:34:46] A_Day3_seg3: 18397 rows → 1835 windows
[22:34:46] SKIP B_Day1_seg1: only 0 rows after warmup
[22:34:46] SKIP B_Day1_seg15: only 0 rows after warmup
[22:34:46] B_Day1_seg17: 1974 rows → 193 windows
[22:34:46] B_Day1_seg19: 1163 rows → 112 windows
[22:34:46] B_Day2_seg1: 6195 rows → 615 windows
[22:34:46] SKIP B_Day2_seg5: only 0 rows after warmup
[22:34:46] B_Day2_seg10: 4994 rows → 495 windows
[22:34:46] B_Day2_seg11: 12910 rows → 1287 windows
[22:34:46] B_Day2_seg13: 4359 rows → 431 windows
[22:34:46] B_Day3_seg1: 478 rows → 43 windows
[22:34:46] B_Day3_seg3: 8217 rows → 817 windows
[22:34:46] SKIP C_Day1_seg1: only 0 rows after warmup
[22:34:46] C_Day1_seg4: 2164 rows → 212 windows
[22:34:46] C_Day1_seg5: 1162 rows → 112 windows
[22:34:46] C_Day2_seg1: 371 rows → 33 windows
[22:34:46] C_Day2_seg2: 16277 rows → 1623 windows
[22:34:46]
Total windows (clean): 9,711 | Shape: (9711, 50, 8)
[22:34:46] Spike windows held out: 1,044 → M4_spike_seeds.npy
[22:34:46] Post-winsorization window pool max: 5.5959
[22:34:46]
=================================================================
[22:34:46] PHASE 5 — Train / Val split
[22:34:46] =================================================================
[22:34:46] Train: 8,254 | Val: 1,457
[22:34:46]
Model on cuda | Parameters: 505,096
[22:34:46]
=================================================================
[22:34:46] PHASE 7 — Training
[22:34:46] =================================================================
[22:34:47] Epochs=150 | Batch=256 | Device=cuda | AMP=True | OverfitGap=0.12
[22:34:48] Epoch 1/150 | Train=0.704613 | Val=0.452421 | Gap=-0.2522 | LR=1.00e-03 | Pat=0/25 | 1s
[22:34:48] VRAM after epoch 1: 0.03 GB
[22:34:51] Epoch 10/150 | Train=0.161932 | Val=0.156426 | Gap=-0.0055 | LR=5.82e-04 | Pat=0/25 | 4s
[22:34:54] Epoch 20/150 | Train=0.143692 | Val=0.141997 | Gap=-0.0017 | LR=1.61e-05 | Pat=0/25 | 7s
[22:34:58] Epoch 30/150 | Train=0.093753 | Val=0.090385 | Gap=-0.0034 | LR=5.82e-04 | Pat=0/25 | 10s
[22:35:01] Epoch 40/150 | Train=0.075963 | Val=0.075130 | Gap=-0.0008 | LR=1.61e-05 | Pat=0/25 | 14s
[22:35:04] Epoch 50/150 | Train=0.058511 | Val=0.057213 | Gap=-0.0013 | LR=5.82e-04 | Pat=0/25 | 17s
[22:35:08] Epoch 60/150 | Train=0.048881 | Val=0.048035 | Gap=-0.0008 | LR=1.61e-05 | Pat=0/25 | 20s
[22:35:11] Epoch 70/150 | Train=0.041853 | Val=0.040410 | Gap=-0.0014 | LR=5.82e-04 | Pat=1/25 | 24s
[22:35:15] Epoch 80/150 | Train=0.037008 | Val=0.035997 | Gap=-0.0010 | LR=1.61e-05 | Pat=0/25 | 27s
[22:35:18] Epoch 90/150 | Train=0.034563 | Val=0.033655 | Gap=-0.0009 | LR=5.82e-04 | Pat=0/25 | 31s
[22:35:21] Epoch 100/150 | Train=0.032241 | Val=0.031184 | Gap=-0.0011 | LR=1.61e-05 | Pat=0/25 | 34s
[22:35:25] Epoch 110/150 | Train=0.032357 | Val=0.032291 | Gap=-0.0001 | LR=5.82e-04 | Pat=9/25 | 37s
[22:35:28] Epoch 120/150 | Train=0.029985 | Val=0.029052 | Gap=-0.0009 | LR=1.61e-05 | Pat=0/25 | 41s
[22:35:31] Epoch 130/150 | Train=0.029777 | Val=0.028877 | Gap=-0.0009 | LR=5.82e-04 | Pat=0/25 | 44s
[22:35:35] Epoch 140/150 | Train=0.027762 | Val=0.026888 | Gap=-0.0009 | LR=1.61e-05 | Pat=0/25 | 47s
[22:35:38] Epoch 150/150 | Train=0.028761 | Val=0.028538 | Gap=-0.0002 | LR=5.82e-04 | Pat=9/25 | 51s
[22:35:38]
Training done — 51.1s | Best val: 0.026862 @ epoch 141
[22:35:38]
=================================================================
[22:35:38] PHASE 8 — Threshold calibration on clean val set
[22:35:38] =================================================================
[22:35:38] Mean MAE: 0.026765
[22:35:38] Std MAE: 0.025972
[22:35:38] P95: 0.074673
[22:35:38] P99: 0.102045
[22:35:38] Threshold: 0.110058 (was 0.110058, delta=+0.0%)
[22:35:38] Separation ratio: 4.1x
[22:35:38] Per-channel MAE (val):
[22:35:38] X_ACR_Mot.PV_norm: 0.058402
[22:35:38] X_ACR_Mot.SV_norm: 0.030760
[22:35:38] X_ACR_Mot.TV_norm: 0.017054
[22:35:38] X_ACR_Pmp.PV_norm: 0.046601
[22:35:38] X_ACR_Pmp.SV_norm: 0.020482
[22:35:38] X_ACR_Pmp.TV_norm: 0.014864
[22:35:38] X_Temp.SV_norm: 0.015985
[22:35:38] X_Pres.SV_norm: 0.009306
[22:35:38]
v8 physics check — Pmp.PV MAE: 0.046601
[22:35:38] Expected: similar or slightly lower than v7 (0.0466)
[22:35:38] If higher: startup BPF harmonics now in training — model adjusting (acceptable)
[22:35:38]
--- Validation Gates ---
[22:35:38] GATE1_no_overfit: PASS
[22:35:38] GATE2_mae_lt_006: PASS
[22:35:38] GATE3_threshold_range: PASS
[22:35:38] GATE4_separation_gt3: PASS
[22:35:38] GATE5_false_alarms_lt1pct: PASS
[22:35:38] GATE6_tv_channels_ok: PASS
[22:35:38] GATE7_spike_seeds_saved: PASS
[22:35:38] GATE8_val_loss_lt_005: PASS
[22:35:38] GATE9_pmpPV_startup_ceiling_correct: PASS
[22:35:38] GATE10_pres_cluster_ceilings_ordered: PASS
[22:35:38]
All gates: ALL PASS — READY FOR M5
[22:35:38] Saved: M4_threshold_config.json (v8)
[22:35:38] Saved: lstm_ae_baseline_meta.json (v8)
[22:35:38]
Generating plots...
[22:35:39] Saved: M4_training_curve.png
[22:35:39] Saved: M4_error_distribution.png
[22:35:39] Saved: M4_per_channel_mae.png
[22:35:39] Saved: M4_spike_seeds_distribution.png
[22:35:40] Saved: M4_reconstruction_sample.png
[22:35:40] Saved: M4_v8_cluster_winsor_bounds.png (NEW — v8 fix verification)
[22:35:40] Peak VRAM: 0.20 GB
[22:35:40] Writing report...
[22:35:40] Saved: module_04_lstm_ae_baseline_report.md

============================================================
══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══
============================================================
M4_version : v8 (cluster-conditional winsor — physics fix)
M4_total_windows : 9711
M4_train_windows : 8254
M4_val_windows : 1457
M4_best_val_loss : 0.026862 (physics-weighted)
M4_best_epoch : 141
M4_mean_recon_error : 0.026765 (pure MAE)
M4_anomaly_threshold : 0.110058 (mean+3sigma | P99)
M4_separation_ratio : 4.11x
M4_threshold_delta_pct : +0.0% (was 0.110058)
M4_false_alarms_val : 8
M4_spike_rows_excluded : 12620
M4_peak_vram_gb : 0.2
M4_training_time_s : 51.1
M4_overfit_triggered : False
M4_all_gates_pass : True
M4_winsor_method : cluster_conditional_mean_multiplier
M4_pres_sv_ceilings : startup=3.0x, ss=5.6x, hl=2.0x, cd=3.0x
M4_pmpPV_startup_ceil : 3.2x (was 2.6x global — BPF harmonics preserved)
M4_pressure_ref_mode : high_load (42 bar — Joukowsky reference)
M4_spike_windows : 1044 → M4_spike_seeds.npy
M4_spike_fault_hints : {'pressure_transient': 408, 'pressure_spike_high_load': 7, 'impeller_cavitation': 113, 'bearing_impact': 44, 'mechanical_transient': 472}
M4_model_version : v8 (cluster-conditional | layernorm | hidden-seeded)
M4_AUDIT_violations : 2 (Pmp.PV startup BPF + Pres.SV water hammer)
M4_AUDIT_status : FIXED IN v8 — downstream modules receive clean physics
Status for M5 : READY
============================================================
══ END PASTE UPDATE ══
```