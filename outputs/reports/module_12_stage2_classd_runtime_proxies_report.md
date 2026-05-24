# module_12_stage2_classd_runtime_proxies report

- run: 2026-05-23T03:39:47.434875Z
- stage: M12-Stage-2 Class D runtime proxies
- device: cuda

## Preflight

- artifacts ok: True
- schema ok: True
- actual n cols: 34

## Gate matrix

- **masked_channel_flag** -> `DIAGNOSTIC_ONLY`  (stable=True, ks=None, rho=None, recall=None)
- **burst_count** -> `DIAGNOSTIC_ONLY`  (stable=True, ks=None, rho=None, recall=None)
- **cyclic_baseline_drift** -> `DIAGNOSTIC_ONLY`  (stable=True, ks=None, rho=None, recall=None)
- **err_slope_MotSV** -> `DIAGNOSTIC_ONLY`  (stable=True, ks=None, rho=None, recall=None)
- **multi_sensor_anomaly_count** -> `DIAGNOSTIC_ONLY`  (stable=True, ks=None, rho=None, recall=None)
- **variant_slope_ratio** -> `DIAGNOSTIC_ONLY`  (stable=True, ks=None, rho=None, recall=None)
- **onset_order** -> `DIAGNOSTIC_ONLY`  (stable=True, ks=None, rho=None, recall=None)

## P3 stubs (explicit 0.0 at correct index)

- **secondary_onset_lag** (idx 18): C-29 permanently deferred: cross-window onset timing, not window-local.
- **fault_group_id** (idx 22): Label-circular: maps label->group, needs the label being predicted. Stage 3 should derive pre-classifier or drop.
- **score_A** (idx 29): Trained col is a sequence-aggregate (mean of z_t PCA recon-err series); not reproducible from one window at 1 Hz. Stage 3 owns.
- **score_B** (idx 30): Trained col is sequence-aggregate (OLS slope of recon-err series). Stage 3 owns.
- **score_C** (idx 31): Trained col is sequence-aggregate (max-abs-diff of recon-err series). Stage 3 owns.

## Blocking items

- Raw 50-step window cache not found/aligned (looked for M6B_windows.npy). Proxy PASS/FAIL gating needs it. Ran in DIAGNOSTIC-ONLY mode: offline column stats characterized, but window-local proxy recomputation skipped. Provide the row-aligned window cache to complete Stage 2 gating.