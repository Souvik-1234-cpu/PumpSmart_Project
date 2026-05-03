# PumpSmart v14.2 — PATCH MANIFEST
*All in-place data fixes, source code locations, and regeneration instructions*
*If any data file is lost, follow regeneration instructions in order.*

---

## PATCH 001 — Label 19 (seal_failure_fast) Physics Fix
**Date:** 2026-05-03
**Status:** APPLIED ✓
**Priority:** CRITICAL — affected M7 training data and M8 validation

### Root Cause
`generate_seal_failure_fast()` in `src/m6b_physics_lib.py` had three bugs:
1. `onset = rng.integers(20, 50)` — onset fell inside M4 spike seed window (steps 0–49). Dispatcher embeds spike seed into steps 0–49; onset inside this range meant the Pres.SV drop was computed on spike data, then "hold at minimum" locked in spike noise value (~0.9654) instead of physics drop (~0.48–0.70).
2. `frac = (t - onset) / drop_steps` — off-by-one, frac never reached 1.0.
3. `max_drop = min(0.8, Q_leak_m3s * 1500)` — A_ref version mismatch (1e-4 vs 1e-6) made Q_leak calculation produce max_drop ~0.013 instead of ~0.30–0.48.

**Effect:** Label 19 Pres.SV drop = 0.035 units (should be 0.30–0.52). LSTM-AE produced near-normal z_t → M8 score_A ≈ normal → TPR = 0%.

### Fix Applied
```
src/m6b_physics_lib.py
  Function: generate_seal_failure_fast()
  Change 1: onset = rng.integers(55, 85)   [was: rng.integers(20, 50)]
  Change 2: frac = (t - onset + 1) / drop_steps   [was: (t - onset) / ...]
  Change 3: max_drop = severity_local * 0.60   [was: Q_leak_m3s * 1500]
  Change 4: target_min held explicitly in post-drop window
  Fix script: src/apply_fix_m6b_physics_lib.py
```

Same fix applied to inline copy:
```
src/module_06B_steps1to3_combined.py
  Fix script: src/apply_fix_module_06B.py
```

### Data Files Patched In-Place
| File | What changed | Backup |
|------|-------------|--------|
| `data/synthetic/M6B_sequences_groupD.pkl` | Positions 1200–1999 (Label 19) replaced | `*_backup_pre_label19_patch.pkl` |
| `data/synthetic/z_t_sequences_groupD.pkl` | Same positions replaced | `*_backup_pre_label19_patch.pkl` |
| `data/synthetic/M6B_feature_matrix.csv` | Label 19 rows replaced | `*_pre_label19_feature_patch.csv` |
| Script used | `src/module_06B_patch_label19_seal_fast.py` | — |
| Feature patch script | `src/module_06p5r_patch_label19_features.py` | — |

### Verification
- Pres.SV min mean (Label 19, after patch): **0.6715** (was ~0.9738)
- Physics gate (Pres.SV < 0.85): **99.75%** (was ~0%)
- M8 Label 19 TPR after patch: **100%** (was 0%)

### Regeneration Instructions (if data lost)
1. Fix source: `python src/apply_fix_m6b_physics_lib.py`
2. Fix inline copy: `python src/apply_fix_module_06B.py`
3. Regenerate sequences: `python src/module_06B_patch_label19_seal_fast.py`
4. Regenerate features: `python src/module_06p5r_patch_label19_features.py`
5. Re-run M7: `python src/module_07_xgboost_classifier.py`
6. Re-run M8: `python src/module_08_tcn_ae_detection_stack.py`

---

## PATCH 002 — M6B_sequence_meta.csv seq_id Sort Order Fix
**Date:** 2026-05-02
**Status:** APPLIED ✓ (in M8 script only)
**Priority:** MEDIUM

### Root Cause
`attach_meta()` in M8 loaded `meta_by_group` subsets without sorting by `seq_id`. The meta CSV stores labels in blocked order (all Label 18, then all 19, etc.) but this order didn't match the pkl sequence order causing positional mismatch.

### Fix Applied
```
src/module_08_tcn_ae_detection_stack.py
  Section 2, meta_by_group dict:
  Added .sort_values('seq_id') to all 7 group meta subsets
  Added low_memory=False to pd.read_csv()
```
**Note:** This fix is in M8 only. The upstream M6.5r and M6B scripts generate data in a consistent order — the mismatch was only in M8's loading logic.

---

## PATCH 003 — M6B_feature_matrix.csv score_C Distribution Fix
**Date:** 2026-05-01
**Status:** APPLIED ✓ (in feature matrix data — v4b→v5 patches)
**Priority:** HIGH — affected M7 SHAP ranks

### Root Cause
score_C was initially assigned as per-label mean constant value → Fisher score = 8×10¹⁵ → dominated SHAP for all classes (leakage). Multiple patch iterations (v1–v5) fixed this to per-sequence SNR values.

### Fix Applied
Patches v1–v5 applied via `src/module_06p5r_patch_features_v5.py`.
Final state: score_C uses per-sequence SNR via linspace boundaries (100% row coverage).
backup: `data/synthetic/M6B_feature_matrix_pre_patch_backup.csv` — DO NOT RESTORE.

---

## MODULE STATUS AFTER ALL PATCHES

| Module | Status | Re-run needed? |
|--------|--------|----------------|
| M1–M4 | LOCKED — no changes | No |
| M5 | LOCKED — no changes | No |
| M6A | LOCKED — no changes | No |
| M6B (source: m6b_physics_lib.py) | PATCHED via apply_fix scripts | Only if regenerating data |
| M6B (data files) | PATCHED in-place | No (backups exist) |
| M6.5r (feature matrix) | PATCHED in-place | No (run feature patch script if Label 19 rows lost) |
| M7 | **NEEDS RE-RUN** — trained on corrupt Label 19 features | **YES** |
| M8 | LOCKED — 27/27 gates PASS | No |
| M9–M12 | PENDING | After M7 re-run |

---

## REGENERATION DEPENDENCY CHAIN

```
If m6b_physics_lib.py is correct (apply_fix applied):

m6b_physics_lib.py (fixed)
    ↓
module_06B_steps1to3_combined.py → regenerates M6B_sequences_groupD.pkl
    ↓
module_06B_patch_label19_seal_fast.py → patches z_t_sequences_groupD.pkl
    ↓
module_06p5r_patch_label19_features.py → patches M6B_feature_matrix.csv
    ↓
module_07_xgboost_classifier.py → re-trains M7 on corrected features
    ↓
module_08_tcn_ae_detection_stack.py → re-validates M8 (should still pass 27/27)
```

**NEVER** regenerate M6B without applying the physics lib fix first.
**NEVER** restore `M6B_feature_matrix_pre_patch_backup.csv` — it predates all v1-v5 patches.
**ALWAYS** run patches in order listed above.

---

*Maintained by: PumpSmart v14.2 development*
*Last updated: 2026-05-03*
