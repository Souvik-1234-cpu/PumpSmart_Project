# module_12_stage2_classd_proxies_optionC report

- run: 2026-05-23T09:52:13.929788Z
- stage: M12-Stage-2 Option-C (M4-faithful proxy calibration)
- M4 params: 505096
- windows: 1552
- sequences sampled: 192

## Gate matrix

- **masked_channel_flag** -> `PASS` (finite=1.0, ks=0.2874, recall=0.9167)
- **burst_count** -> `PASS` (finite=1.0, ks=0.4328, recall=None)
- **cyclic_baseline_drift** -> `PASS` (finite=1.0, ks=0.5874, recall=None)
- **err_slope_MotSV** -> `PASS` (finite=1.0, ks=0.7678, recall=None)
- **multi_sensor_anomaly_count** -> `PASS` (finite=1.0, ks=0.251, recall=None)
- **variant_slope_ratio** -> `PASS` (finite=1.0, ks=0.7392, recall=None)

## P3 stubs

- **secondary_onset_lag** (idx 18): C-29 deferred: cross-window onset timing.
- **fault_group_id** (idx 22): Label-circular: maps label->group. Stage 3 derives.
- **score_A** (idx 29): Seq-aggregate (mean of z_t recon-err series). Stage 3.
- **score_B** (idx 30): Seq-aggregate (OLS slope of recon-err series). Stage 3.
- **score_C** (idx 31): Seq-aggregate (max-abs-diff of recon-err series). Stage 3.
- **onset_order** (idx 32): CORRECTED: sequence-position ordinal {0,1,2,3} for Group B only (module_06p5r_patch_features_v5). Not window-local. Stage 3 owns.

## Blocking

- none