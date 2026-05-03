# PumpSmart — Module M8 Report
**Date:** 2026-05-03  
**Architecture:** v14.2  
**Pump:** 110 kW | 7-stage | 40 bar | 2980 RPM | 45 m³/h | 450 m head  

---

## Training Results

| Metric | Value |
|--------|-------|
| Best val loss | 0.006933 |
| Training time (min) | 0.77 |
| Peak VRAM (GB) | 0.036 |
| TCN-AE params | 255,840 |
| Normal train seqs | 1700 |

## Score Statistics

| Score | Normal Mean | Normal Std | Normal P95 |
|-------|-------------|------------|------------|
| score_A | 0.51579 | 0.42636 | - |
| score_B | -0.00954 | 0.01457 | - |
| score_C | - | - | 0.31008 |

## CUSUM Parameters (Layer 3)

| Parameter | Value |
|-----------|-------|
| mu0_B | -0.00954 |
| k | 0.02186 |
| H (control limit) | 5.0 |
| WATCH rate mild | 1.0 |
| CUSUM FPR | 0.0 |

## Layer 4 Adaptive Threshold

| Parameter | Value |
|-----------|-------|
| θ_initial | 1.881275 |
| Crosspoint lock | 2.821913 |
| Rolling window (calls) | 432 |
| Warmup (calls) | 216 |
| L4 WARN rate label 21 | 1.0 |

## Gate Results

| Gate | Status |
|------|--------|
| M8-10_presv_drift_first | PASS ✓ |
| M8-11_thermal_lag | PASS ✓ |
| M8-12_cav_cluster_ok | PASS ✓ |
| M8-13_groupC_tpr | PASS ✓ |
| M8-14_groupB_tpr | PASS ✓ |
| M8-14_groupD_tpr | PASS ✓ |
| M8-14_groupE_tpr | PASS ✓ |
| M8-14_overall | PASS ✓ |
| M8-14ext_cusum_fpr_ok | PASS ✓ |
| M8-14ext_cusum_watch | PASS ✓ |
| M8-14ext_l4_warn | PASS ✓ |
| M8-15_scoreC_calib | PASS ✓ |
| M8-1_groupA_tpr | PASS ✓ |
| M8-1_val_loss | PASS ✓ |
| M8-1_val_loss_ok | PASS ✓ |
| M8-2_fpr_ok | PASS ✓ |
| M8-3_vram_ok | PASS ✓ |
| M8-4_separation | PASS ✓ |
| M8-5_fa_abs | PASS ✓ |
| M8-6_fuzzy_valid | PASS ✓ |
| M8-7_overloading_mechC | PASS ✓ |
| M8-8_seam_ratio | PASS ✓ |
| M8-9B_lbl14_mechC | PASS ✓ |
| M8-9_seal_mechC | PASS ✓ |
| M8-J_youden | PASS ✓ |
| M8-lbl5_overload | PASS ✓ |
| M8-lbl6_cv | PASS ✓ |

**Total:** 27 PASS | 0 FAIL | 0 SKIP  
**Block M9:** False

## Detection Performance

| Group | TPR |
|-------|-----|
| Group A single fault | 0.97 |
| Group B compound | 1.0 |
| Group D variant | 1.0 |
| Group E multi-sensor | 0.915 |
| FPR normal pool | 0.05 |
| Separation ratio | 3.35 |
| Youden J | 0.9196 |

## Invariant 19 — Score Routing (ENFORCED)

| Score | Routes To |
|-------|-----------|
| score_A | Layer 4 Rolling Baseline ONLY |
| score_B | Layer 3 CUSUM ONLY |
| score_C | XGBoost M7 / output ONLY |

## Limitations

1. Synthetic-to-real gap: trained on CIRA-anchored physics-synthetic data.
2. 1 Hz sampling: BPF at 348 Hz captured as envelope statistics only.
3. Label 21 detection latency: earliest reliable detection ~Week 5 (Layer 4 slope shift).
4. Household pump OOD: `if pump_type=='household': return physics_advisory_only()` enforced at M10.

---

*Model: `models/tcn_ae_level2_best.pth`*  
*Config: `models/M8_threshold_config.json`*  
*M4 threshold (LOCKED): 0.110058*
