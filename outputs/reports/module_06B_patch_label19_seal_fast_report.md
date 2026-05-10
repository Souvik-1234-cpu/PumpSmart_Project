# PumpSmart — Label 19 Patch Report
**Date:** 2026-05-03  
**Script:** module_06B_patch_label19_seal_fast  
**Arch:** v14.2

## Root Cause
`generate_seal_failure_fast()` had `onset = rng.integers(20, 50)`.
M6B dispatcher embeds CIRA spike seed into steps 0–49.
Onset inside spike window → drop applied inside spike data →
"hold at minimum" locked in spike noise value (0.9654) not orifice drop (0.48).
Pres.SV drop magnitude: **0.035 units** (was) vs **0.35–0.52 units** (expected).
LSTM-AE z_t ≈ normal → score_A ≈ normal → M8 TPR = 0%.

## Fix Applied
1. `onset = rng.integers(55, 85)` — forces onset after spike seed window
2. `frac = (t - onset + 1) / drop_steps` — reaches 1.0 at final step
3. `target_min = 1.0 - max_drop` held explicitly in post-drop window

## Results
| Metric | Before | After |
|--------|--------|-------|
| Pres.SV min mean | 0.9738 | 0.6715 |
| Fraction < 0.85 | ~0% | 99.75% |
| Physics gate pass | FAIL | PASS ✓ |

## Files Modified
- `data/synthetic/M6B_sequences_groupD.pkl` — Label 19 positions 1200–1999 replaced
- `data/synthetic/z_t_sequences_groupD.pkl` — z_t for same positions replaced

## Backups Created
- `data/synthetic/M6B_sequences_groupD_backup_pre_label19_patch.pkl`
- `data/synthetic/z_t_sequences_groupD_backup_pre_label19_patch.pkl`

## Next Step
Re-run `module_08_tcn_ae_detection_stack.py`
Expected: Label 19 TPR substantially above 0%.
