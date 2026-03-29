```
[15:53:50] Device: cuda | GPU: True
[15:53:50] Window=50s | Step=10s | Batch=256 | Epochs=300
[15:53:50] STEP 1 — Loading normalised data and segment registry...
[15:53:50] Normalised data: 117,970 rows | 25 segments
[15:53:50] Registry: 66 segments | 25 usable
[15:53:50] STEP 2 — Generating windows per segment (warmup-aware)...
[15:53:50] SKIP A_Day1_seg1: only 0 rows after warmup
[15:53:50] A_Day1_seg10: 2700 rows → 266 windows (warmup=300)
[15:53:50] A_Day1_seg11: 1362 rows → 132 windows (warmup=300)
[15:53:50] A_Day2_seg1: 371 rows → 33 windows (warmup=300)
[15:53:50] SKIP A_Day2_seg2: only 0 rows after warmup
[15:53:50] A_Day2_seg3: 13028 rows → 1298 windows (warmup=300)
[15:53:50] A_Day2_seg7: 4528 rows → 448 windows (warmup=300)
[15:53:50] A_Day3_seg1: 1740 rows → 170 windows (warmup=300)
[15:53:50] A_Day3_seg3: 19247 rows → 1920 windows (warmup=300)
[15:53:50] SKIP B_Day1_seg1: only 0 rows after warmup
[15:53:50] SKIP B_Day1_seg15: only 0 rows after warmup
[15:53:50] B_Day1_seg17: 2514 rows → 247 windows (warmup=300)
[15:53:50] B_Day1_seg19: 1163 rows → 112 windows (warmup=300)
[15:53:50] B_Day2_seg1: 7625 rows → 758 windows (warmup=300)
[15:53:50] SKIP B_Day2_seg5: only 0 rows after warmup
[15:53:50] B_Day2_seg10: 4994 rows → 495 windows (warmup=600)
[15:53:50] B_Day2_seg11: 13050 rows → 1301 windows (warmup=600)
[15:53:50] B_Day2_seg13: 6249 rows → 620 windows (warmup=300)
[15:53:50] B_Day3_seg1: 1358 rows → 131 windows (warmup=300)
[15:53:50] B_Day3_seg3: 8217 rows → 817 windows (warmup=300)
[15:53:50] SKIP C_Day1_seg1: only 0 rows after warmup
[15:53:50] C_Day1_seg4: 2944 rows → 290 windows (warmup=300)
[15:53:50] C_Day1_seg5: 1162 rows → 112 windows (warmup=300)
[15:53:50] C_Day2_seg1: 371 rows → 33 windows (warmup=300)
[15:53:50] C_Day2_seg2: 17947 rows → 1790 windows (warmup=600)
[15:53:50]
Total windows: 10,973 | Shape: (10973, 50, 8)
[15:53:50] STEP 3 — Creating Dataset and DataLoader...
[15:53:50] Train: 9,328 windows | Val: 1,645 windows
[15:53:50] STEP 4 — Building LSTM-AE model (v3)...
[15:53:51] Model parameters: 505,096
[15:53:51] Encoder bottleneck: 64 dims
[15:53:51] STEP 5 — Training LSTM-AE v3...
[15:53:51] Mixed precision (AMP): True
[15:53:52] Epoch 1/300 | Train=1.396509 | Val=0.952538 | Gap=-0.4440 | LR=1.00e-03 | Patience=0/25 | Elapsed=0s
[15:53:55] Epoch 10/300 | Train=0.882648 | Val=0.666975 | Gap=-0.2157 | LR=5.82e-04 | Patience=0/25 | Elapsed=3s
[15:53:58] Epoch 20/300 | Train=0.828816 | Val=0.622366 | Gap=-0.2065 | LR=1.61e-05 | Patience=0/25 | Elapsed=6s
[15:54:02] Epoch 30/300 | Train=0.731231 | Val=0.562159 | Gap=-0.1691 | LR=5.82e-04 | Patience=0/25 | Elapsed=10s
[15:54:05] Epoch 40/300 | Train=0.704223 | Val=0.543219 | Gap=-0.1610 | LR=1.61e-05 | Patience=0/25 | Elapsed=13s
[15:54:08] Epoch 50/300 | Train=0.671499 | Val=0.512883 | Gap=-0.1586 | LR=5.82e-04 | Patience=0/25 | Elapsed=16s
[15:54:11] Epoch 60/300 | Train=0.640456 | Val=0.493699 | Gap=-0.1468 | LR=1.61e-05 | Patience=0/25 | Elapsed=19s
[15:54:14] Epoch 70/300 | Train=0.612707 | Val=0.471135 | Gap=-0.1416 | LR=5.82e-04 | Patience=0/25 | Elapsed=22s
[15:54:17] Epoch 80/300 | Train=0.590668 | Val=0.465190 | Gap=-0.1255 | LR=1.61e-05 | Patience=0/25 | Elapsed=25s
[15:54:21] Epoch 90/300 | Train=0.575519 | Val=0.438757 | Gap=-0.1368 | LR=5.82e-04 | Patience=0/25 | Elapsed=29s
[15:54:24] Epoch 100/300 | Train=0.547042 | Val=0.421205 | Gap=-0.1258 | LR=1.61e-05 | Patience=0/25 | Elapsed=32s
[15:54:27] Epoch 110/300 | Train=0.523422 | Val=0.406170 | Gap=-0.1173 | LR=5.82e-04 | Patience=0/25 | Elapsed=35s
[15:54:30] Epoch 120/300 | Train=0.491737 | Val=0.386933 | Gap=-0.1048 | LR=1.61e-05 | Patience=0/25 | Elapsed=38s
[15:54:33] Epoch 130/300 | Train=0.491442 | Val=0.389399 | Gap=-0.1020 | LR=5.82e-04 | Patience=9/25 | Elapsed=41s
[15:54:36] Epoch 140/300 | Train=0.458925 | Val=0.374343 | Gap=-0.0846 | LR=1.61e-05 | Patience=0/25 | Elapsed=44s
[15:54:40] Epoch 150/300 | Train=0.485859 | Val=0.367591 | Gap=-0.1183 | LR=5.82e-04 | Patience=3/25 | Elapsed=48s
[15:54:43] Epoch 160/300 | Train=0.433205 | Val=0.359926 | Gap=-0.0733 | LR=1.61e-05 | Patience=6/25 | Elapsed=51s
[15:54:46] Epoch 170/300 | Train=0.422331 | Val=0.348504 | Gap=-0.0738 | LR=5.82e-04 | Patience=0/25 | Elapsed=54s
[15:54:49] Epoch 180/300 | Train=0.395725 | Val=0.337685 | Gap=-0.0580 | LR=1.61e-05 | Patience=2/25 | Elapsed=57s
[15:54:53] Epoch 190/300 | Train=0.413435 | Val=0.339894 | Gap=-0.0735 | LR=5.82e-04 | Patience=12/25 | Elapsed=60s
[15:54:56] Epoch 200/300 | Train=0.369164 | Val=0.326969 | Gap=-0.0422 | LR=1.61e-05 | Patience=3/25 | Elapsed=64s
[15:54:59] Epoch 210/300 | Train=0.379197 | Val=0.332668 | Gap=-0.0465 | LR=5.82e-04 | Patience=1/25 | Elapsed=67s
[15:55:03] Epoch 220/300 | Train=0.345368 | Val=0.306889 | Gap=-0.0385 | LR=1.61e-05 | Patience=0/25 | Elapsed=71s
[15:55:06] Epoch 230/300 | Train=0.360234 | Val=0.305393 | Gap=-0.0548 | LR=5.82e-04 | Patience=2/25 | Elapsed=74s
[15:55:10] Epoch 240/300 | Train=0.323797 | Val=0.288929 | Gap=-0.0349 | LR=1.61e-05 | Patience=1/25 | Elapsed=78s
[15:55:13] Epoch 250/300 | Train=0.342267 | Val=0.300712 | Gap=-0.0416 | LR=5.82e-04 | Patience=11/25 | Elapsed=81s
[15:55:17] Epoch 260/300 | Train=0.305609 | Val=0.286649 | Gap=-0.0190 | LR=1.61e-05 | Patience=0/25 | Elapsed=85s
[15:55:20] Epoch 270/300 | Train=0.308274 | Val=0.277947 | Gap=-0.0303 | LR=5.82e-04 | Patience=0/25 | Elapsed=88s
[15:55:24] Epoch 280/300 | Train=0.284235 | Val=0.268675 | Gap=-0.0156 | LR=1.61e-05 | Patience=0/25 | Elapsed=92s
[15:55:27] Epoch 290/300 | Train=0.292102 | Val=0.285736 | Gap=-0.0064 | LR=5.82e-04 | Patience=3/25 | Elapsed=95s
[15:55:31] Epoch 300/300 | Train=0.265032 | Val=0.253337 | Gap=-0.0117 | LR=1.61e-05 | Patience=3/25 | Elapsed=98s
[15:55:31]
Training complete: 98s | Best val loss: 0.251071 at epoch 297
[15:55:31] Overfit guard triggered: False
[15:55:31] STEP 6 — Computing reconstruction error distribution (pure MAE, normalised space)...
[15:55:31] Mean recon error (pure MAE) : 0.049839
[15:55:31] Std recon error : 0.198503
[15:55:31] P95 error : 0.099678
[15:55:31] P99 error : 0.518297
[15:55:31] Anomaly threshold : 0.645347 (mean + 3σ ∪ P99)
[15:55:31] Threshold config saved → M4_threshold_config.json
[15:55:31] STEP 7 — Peak VRAM used: 0.20 GB
[15:55:31] STEP 8 — Generating training loss curve...
[15:55:31] Saved → M4_training_curve.png
[15:55:31] STEP 9 — Generating error distribution plot...
[15:55:31] Saved → M4_error_distribution.png
[15:55:31] STEP 10 — Generating reconstruction overlay plot...
[15:55:32] Saved → M4_reconstruction_sample.png
[15:55:32] STEP 11 — Saving final model state dict...
[15:55:32] Model saved → lstm_ae_baseline_final.pth
[15:55:32] Metadata → lstm_ae_baseline_meta.json
[15:55:32] STEP 12 — Writing markdown report...
[15:55:32] Report saved → module_04_lstm_ae_baseline_report.md

════════════════════════════════════════════════════════════
PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT
════════════════════════════════════════════════════════════
M4_total_windows : 10973
M4_train_windows : 9328
M4_val_windows : 1645
M4_best_val_loss : 0.251071 (physics-weighted combined)
M4_best_epoch : 297
M4_mean_recon_error : 0.049839 (pure MAE)
M4_anomaly_threshold : 0.645347 (pure MAE | mean+3sigma∪P99)
M4_vram_peak_gb : 0.2
M4_training_time_s : 98
M4_overfit_triggered : False
M4_model_version : v3 (physics-weighted | cosine-LR | layernorm)
M4_model_file : lstm_ae_baseline_best.pth
M4_threshold_config : M4_threshold_config.json
Status for M5 : READY
════════════════════════════════════════════════════════════
```