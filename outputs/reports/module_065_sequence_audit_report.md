# M6.5 Sequence Quality Audit Report (v2)

**Generated:** 2026-04-11 22:02:02

**v2 Fix:** Gate 3 window slice corrected to `:50` (was `:60`, now matches M4 `WINDOW_SIZE=50`)

## Shape & Balance
- Shape: `(8400, 200, 8)` — OK
- NaN: `0` | Inf: `0` — OK
- Balance: `{'normal': 1200, 'bearing_wear': 1200, 'impeller_imbalance': 1200, 'cavitation': 1200, 'seal_failure': 1200, 'overloading': 1200, 'sensor_failure': 1200}` — OK

## Gate 3 Re-Audit (window=50 steps)
- `normal`: MAE=0.1202  pass=86.67%
- `bearing_wear`: MAE=0.0979  pass=13.33%
- `impeller_imbalance`: MAE=0.1031  pass=30.00%
- `cavitation`: MAE=0.6747  pass=100.00%
- `seal_failure`: MAE=0.1961  pass=29.17%
- `overloading`: MAE=0.0930  pass=0.00%
- `sensor_failure`: MAE=0.1696  pass=93.33%

## Top 5 Fisher Features
- `Pmp_SV_mean`
- `Pmp_SV_std`
- `Temp_SV_mean`
- `Mot_TV_mean`
- `Mot_TV_std`

## Temporal Coherence (Final)
- `bearing_wear`: pass=94.25%  flagged=69
- `impeller_imbalance`: pass=99.75%  flagged=3
- `cavitation`: pass=91.25%  flagged=105
- `seal_failure`: pass=100.00%  flagged=0
- `overloading`: pass=100.00%  flagged=0
- `sensor_failure`: pass=92.75%  flagged=87

## Overall Verdict
**M7 Ready: YES — PROCEED**
