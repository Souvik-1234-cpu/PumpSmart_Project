# M12 Stage 3 — M7 v3 Retrain Report

**Date:** 2026-05-24  |  **Device:** cuda  |  **Status:** PASS

## Summary

Stage 3 eliminates train/serve skew by rebuilding the feature matrix using identical code to live inference (`build_m7_features` + `stage2_proxies`), then retraining M7 on it.

| Metric | Value |
|---|---|
| v3 matrix rows | 279,400 |
| v3 matrix features | 33 |
| n_classes | 24 |
| Train windows | 223,520 |
| Test windows | 55,880 |
| Train sequences | 26,000 |
| Test sequences | 6,500 |
| Window macro F1 | 0.9202 |
| **Seq macro F1 (onset-aware)** | **0.9529** |
| Seq macro F1 (vote) | 0.9994 |
| Train time (min) | 0.71 |

## Per-Group Sequence F1

| Group | F1 | Target |
|---|---|---|
| A | 0.9949 | 0.75 |
| B | 0.8241 | 0.65 |
| C | 0.9979 | 0.7 |
| D | 0.9974 | 0.6 |
| E | 0.9905 | 0.65 |

## Gate Matrix

| Gate | Pass | Detail |
|---|---|---|
| G_V3_1_row_count | ✓ | {'n_rows': 279400, 'expected_stride50_approx': 263150, 'original_stride25_rows': |
| G_V3_2_col_count | ✓ | {'n_features': 33, 'expected': 33} |
| G_V3_3_no_leakage | ✓ | {'overlap_count': 0, 'train_seqs': 26000, 'test_seqs': 6500} |
| G_V3_4_d1_collapse_resolved | ✓ | {'n_label22_test': 640, 'p_predict_correct': 0.8609, 'threshold': 0.5} |
| G_V3_6_seq_macro_f1 | ✓ | {'value': 0.9529, 'threshold': 0.85, 'note': 'Synthetic-domain target per Soluti |
| G_V3_7_per_group_f1 | ✓ | {'detail': {'A': {'pass': True, 'value': 0.9949, 'threshold': 0.75}, 'B': {'pass |
| G_V3_8_label21_seq_f1 | ✓ | {'value': 1.0, 'threshold': 0.6, 'note': 'Partial credit only — L3 CUSUM is the  |
| G_V3_5_shap_no_leakage | ✓ | {'top_5_global_feats': ['masked_channel_flag', 'mae_PresSV', 'std_err_MotSV', 'm |
| G_V3_9_c30_selftest | ✓ | {} |

## Known Residual Gaps

- idx 18 secondary_onset_lag: 0.0 stub (C-29 deferred)
- idx 22 fault_group_id: 0.0 stub (label-circular)
- idx 29-31 score_A/B/C: 0.0 stubs (sequence-aggregate)
- idx 32 onset_order: 0.0 stub (Group B seq-position ordinal)
- Label 15: window-local gap logged; sequence-level detection deferred to Stage 4

## C-26 Disclaimer

Trained on CIRA-anchored physics-synthetic data for 110 kW 7-stage pump at 2980 RPM, 40 bar. Sequence-level F1 cited above is synthetic-domain only. Real-world performance expected 0.65–0.85 per C-26 until active learning loop completes first retrain (~50 confirmed real faults).
