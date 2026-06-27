# module_12_stage4_step1_score_c_calibration

PumpSmart v14.2 — M12 Stage 4, Step 4.1 — live score_C calibration (v2 fill-matched)

- Date: 2026-05-24 | Device: cuda | GPU: True
- Overall status: **PASS** | BLOCK_M11: True
- Output config: `C:\Users\user\Desktop\PumpSmart_Project\models\M8_alert_thresholds.json` (M8_threshold_config.json untouched)

## Gates

| Gate | Description | Status |
|------|-------------|--------|
| G4_1_1 | Production TCN-AE loads (strict=True, D8-fixed) | PASS |
| G4_1_2 | Non-vacuous: fill-matched pools >= 30 | PASS |
| G4_1_3 | Distributions measured: fill-matched + serve-s50 | PASS |
| G4_1_4 | Fill-matched AUC + reliability tier | PASS |
| G4_1_5 | M8_alert_thresholds.json persisted | PASS |

## Measured score_C ladder (BINDING = fill-matched adaptive stride)

- score_C_normal_p95 : 0.105989
- score_C_warn       : 0.077469
- score_C_danger     : 0.083763
- **reliability**    : **STRONG**
- AUC normal-vs-GroupB (fill-matched, binding) : 0.9509
- AUC normal-vs-GroupB (serve s50, reality)    : None
- thresholds ordered (fill-matched)            : True

### Serve-reality note

- At live stride=50, normal (200-step) sequences cannot fill MIN_READY windows, so score_C does NOT establish a normal baseline on the 1 Hz route. score_C is a slow chain signal (full 63-window buffer ~= 52 min at 1 Hz). Step 4.3 must NOT expect score_C to fire on short/acute windows; score_A (MAE) is the acute path.
- serve normal pool fires: False (n=0)

## Honest reading

- v1 reported a spurious UNUSABLE: a fixed stride-50 starved the 200-step normal pool to 0 samples. v2's fill-matched adaptive stride compares all pools at the same buffer length (~30 windows), so the AUC measures score_C's intrinsic separability.
- v3 polarity correction: the fill-matched AUC is high, but the warn/danger ladder is INVERTED (Group-B score_C is LOWER than normal). AUC is direction-agnostic; the alarm logic assumes faults score higher. A high-AUC-but-inverted signal would SUPPRESS on faults if used as a high-side trigger, so it is demoted to UNUSABLE (not WEAK).
- Root cause (physics/ML): score_C = temporal std of per-window MAE over the buffer. Normal operation spans cluster transitions (startup->steady->load) -> genuine MAE variance; an established compound fault is a SUSTAINED elevated-but-stable error -> LOWER temporal variance. score_C measures error-wobble, not fault magnitude.
- Consequence for Step 4.3: score_C drives NOTHING. CUSUM (L3), rolling-mean floors, and Mech-A/B/C are the primary detectors; score_A (MAE) is the acute path. This matches the long-standing documented weakness of score_C (M6.5r Gate Z2).