# M6.5 Sequence Quality Audit Report

**Generated:** 2026-04-11 21:26:54

## Shape & Balance
- Shape: `(8400, 200, 8)` — OK
- NaN: `0` | Inf: `0` — OK
- Balance: `{'normal': 1200, 'bearing_wear': 1200, 'impeller_imbalance': 1200, 'cavitation': 1200, 'seal_failure': 1200, 'overloading': 1200, 'sensor_failure': 1200}` — OK

## Gate 3 Re-Audit
- `normal`: MAE=0.0965  pass=88.33%
- `bearing_wear`: MAE=0.1012  pass=19.17%
- `impeller_imbalance`: MAE=0.1103  pass=47.50%
- `cavitation`: MAE=0.6430  pass=100.00%
- `seal_failure`: MAE=0.2809  pass=57.50%
- `overloading`: MAE=0.0980  pass=2.50%
- `sensor_failure`: MAE=0.1805  pass=96.67%

## Top 5 Fisher Features
- `Pmp_SV_mean`
- `Pmp_SV_std`
- `Temp_SV_mean`
- `Mot_TV_std`
- `Mot_TV_mean`

## Temporal Coherence (Final)
- `bearing_wear`: pass=94.25%  flagged=69
- `impeller_imbalance`: pass=99.75%  flagged=3
- `cavitation`: pass=91.25%  flagged=105
- `seal_failure`: pass=100.00%  flagged=0
- `overloading`: pass=100.00%  flagged=0
- `sensor_failure`: pass=92.75%  flagged=87

## Seal Failure Patch
- Replaced: `220` sequences
- Coherence after patch: `100.00%`
- Physics reason: Real 40-bar multistage pump seal fails over hours.
  High-severity early saturation (< 60 steps) is physically unrealistic.
  Severity capped at 0.5 to ensure full 200-step progression.

## Overall Verdict
**M7 Ready: YES — PROCEED**
