# module_12_stage4_step0_c30_selftest_verify

PumpSmart v14.2 — M12 Stage 4, Step 4.0 — C-30 startup self-test verification (v2)

- Date: 2026-05-24
- Device: cuda | GPU: True
- M4: C:\Users\user\Desktop\PumpSmart_Project\models\lstm_ae_baseline_final.pth
- M6B present: True | v3 matrix present: True
- Overall status: **PASS**
- BLOCK_M11: True

## Gate results

| Gate | Description | Status |
|------|-------------|--------|
| G4_0_1  | Production M4 forward() returns recon-only Tensor [B,50,8] | PASS |
| G4_0_2a | DIAGNOSIS: per-column live-vs-reference divergence | PASS |
| G4_0_2b | DISK TRUTH: builder matches M6B_feature_matrix_v3.csv (1e-5) | PASS |
| G4_0_2c | Regenerate stale reference + clean selftest passes | PASS |
| G4_0_3  | POISON: corrupted bit-exact column makes selftest RAISE | PASS |

## Evidence

- **m4_params**: 505096
- **m4_out_type**: Tensor
- **m4_out_shape**: (1, 50, 8)
- **m6b_n_sequences**: 32500
- **per_column_max_diff**: {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.0, '9': 0.0, '10': 0.0, '12': 0.0, '13': 0.0, '14': 0.0, '15': 0.0, '16': 0.0, '24': 0.0}
- **n_reference_windows**: 20
- **disk_truth_checked_rows**: 40
- **disk_truth_unmatched**: 0
- **disk_truth_max_abs_diff**: 2.23145e-07
- **selftest_backup**: C:\Users\user\Desktop\PumpSmart_Project\app\runtime\feature_builder_selftest.py.bak_20260524_141856
- **references_regenerated**: 20
- **poison**: raised as required — RuntimeError

## Interpretation

- The live builder matches the persisted v3 training matrix at all 16 bit-exact columns, so the v1 selftest divergence was a STALE embedded reference (not a feature-path defect). The reference was regenerated and the C-30 guard now passes clean and fails on poison. Feature path verified. **Safe to proceed to Step 4.1 (live score_C calibration).**

## Carry-forward to M11 (full deployment)

- HF Spaces ships `models/` without `data/synthetic/M6B_combined_sequences.pkl`; the production registry SKIPs the C-30 selftest there. M11 should ship a small frozen reference bundle (e.g. `models/M6B_selftest_refset.npz`, ~20 windows) so the guard runs in production too — negligible size — or accept the documented SKIP and rely on this build-time (local) verification.