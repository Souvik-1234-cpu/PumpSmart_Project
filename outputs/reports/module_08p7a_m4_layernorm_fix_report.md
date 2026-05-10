# module_08p7a_m4_layernorm_fix — Report
**Date:** 2026-05-09
**Status:** COMPLETE

## Root Cause

T1.6b defined `encoder.bn = nn.BatchNorm1d(64)`.
Real M4 (`module_04_lstm_ae_baseline.py`) uses `self.bn = nn.LayerNorm(bottleneck)`.

LayerNorm has no `running_mean`/`running_var` — those keys were never in the
checkpoint. T1.6b's `strict=False` + zeros/ones init fabricated normalization
statistics, producing biased z_t values.

Fix: `nn.LayerNorm(BOTTLE)` + `strict=True` load. No fabrication needed.
M4 does NOT need retraining — checkpoint is correct.

## Results

| Metric | Value |
|---|---|
| M4 load | strict=True (LayerNorm) |
| z_t mean(|z|) | 0.7711 |
| z_t norm mean (new) | 8.0306 |
| z_t norm std (new) | 0.06 |
| z_t norm delta vs v1 | 0.23% (target <5%) |
| Feature rows updated | 120,000 |
| FINAL M7 macro F1 | 0.9980 |
| FINAL M7 accuracy | 0.9990 |
| Train time | 0.67 min |
| M7 saved | live |

## Group B F1

| Label | Class | F1 |
|---|---|---|
| 7 | bearing_wear+overloading | 0.9994 |
| 8 | cavitation+seal_failure | 0.9997 |
| 9 | impeller_imbalance+bearing_wear | 0.9969 |
| 10 | seal_failure+cavitation_H | 0.9982 |
| 11 | overloading+bearing_wear | 0.9995 |
| 12 | impeller_imbalance+cavitation | 0.9978 |

## Gates

| Gate | Status | Detail |
|---|---|---|
| T1.5.1_G1_zt_norm_agreement | PASS |  |
| T1.5.1_G2_zt_pkl_saved | PASS |  |
| T1.5.1_G3_fm_updated | PASS |  |
| T1.5.1_G4_m4_strict_load | PASS | strict=True load succeeded — LayerNorm confirmed |
| T1.5.1_G5_zt_nonzero | PASS | mean(|z|)=0.7711 |
| T1.5.1_G6_macro_f1 | PASS | F1=0.9980 (target >=0.82) |
| T1.5.1_G7_groupB_floor | PASS | min=0.9969 |

---
*module_08p7a_m4_layernorm_fix | PumpSmart v14.2 | 2026-05-09*
