# M8 Patch 4 — Out-of-Distribution Detector
**Date:** 2026-05-09
**Status:** COMPLETE

## Why this patch existed
The current M7 confidence threshold (UNKNOWN if max_proba < 0.70) catches
uncertainty WITHIN the 22 trained classes. It does NOT catch the most
common real-world failure mode of fault classifiers: a fault that doesn't
match any of the 22 classes (shaft misalignment, foundation looseness,
parallel-pump coupling) producing a CONFIDENT but WRONG classification.

## What this patch did
1. Loaded all normal training z_t (M4 LSTM-AE 64-dim latent vectors).
2. Estimated mean and Tikhonov-regularised covariance (ridge=0.000053).
3. Computed Mahalanobis distance from every normal z_t to the centroid.
4. Calibrated tau_p99 = 22.0319 (the 99th percentile — gives 1% FPR on normal training data).
5. Verified that synthetic faults are appropriately positioned in z_t space.

## Calibration numbers
| Statistic | Value |
|---|---|
| z_t latent dimension | 64 |
| Normal training windows used | 8,000 |
| Mahalanobis on normal — mean | 6.539 |
| Mahalanobis on normal — std | 3.564 |
| Mahalanobis on normal — P99 (= tau_p99) | **22.0319** |
| Mahalanobis on fault — mean | 16.193 |
| Fraction of fault windows above tau_p99 | 21.86% |

## Runtime decision logic (for M10 implementation)

```python
def detect_ood(z_t_window, score_A_value, xgb_proba):
    cfg = json.load(open('models/M8p4_ood_detector_config.json'))
    m_cfg = cfg['mahalanobis_detector']
    zt_mean = np.array(m_cfg['zt_mean'])
    zt_cov_inv = np.array(m_cfg['zt_cov_inv'])
    tau_p99 = m_cfg['tau_p99']

    # Mahalanobis distance
    centered = z_t_window - zt_mean
    mahal = np.sqrt(centered @ zt_cov_inv @ centered)

    # Score_A guard from M8 threshold config
    m8_cfg = json.load(open('models/M8_threshold_config.json'))
    score_A_p95 = m8_cfg.get('M8_score_A_p95_on_normal', float('inf'))

    # Triple condition
    is_ood = (
        mahal > tau_p99
        and score_A_value > score_A_p95
        and xgb_proba.max() < 0.85
    )
    return is_ood, mahal, m_cfg
```

## M10 7-field response when OOD fires
See `runtime_decision_logic` and `m10_response_when_ood` blocks in
`models/M8p4_ood_detector_config.json`. Field 1 = "OUT_OF_DISTRIBUTION".
Field 7 explicitly tells the operator the system has refused to classify —
this is the system working correctly.

## Why this is the correct fix
The OOD detector converts the silent-confident-wrong failure mode into a
loud "I don't know" — the only honest output for an unmodelled fault on a
40-lakh asset. It is computationally cheap (one matrix-vector product per
inference), requires no model retraining, and integrates as a wrapper on
the existing M10 prediction path.

## Gates
| Gate | Status | Detail |
|---|---|---|
| M8p4-1_tau_p99_valid | ✓ PASS | tau_p99=22.0319 |
| M8p4-2_cov_well_conditioned | ✓ PASS | N=8000 > 10*D=640 |
| M8p4-3_fault_overlap_sensible | ✓ PASS | 21.9% of fault windows above tau_p99 (target <50% — trained faults should mostly be IN-distribution) |

## Files written
- `models/M8p4_ood_detector_config.json`
- `outputs/M8p4_mahal_distribution.png`

## Paste text update

══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══
M8p4_ood_detector_config           : models/M8p4_ood_detector_config.json
M8p4_zt_dim                        : 64
M8p4_n_normal_zt_windows           : 8000
M8p4_cov_ridge                     : 5.3060675194136736e-05
M8p4_tau_p99_mahal                 : 22.031909914678387
M8p4_normal_mahal_mean             : 6.539
M8p4_fault_above_tau_p99_pct       : 21.86
M8p4_runtime_logic_specified       : True (see report Section: Runtime decision logic)
M8p4_M10_implementation_required   : True (wrapper on /api/predict path)
Status_for_M8p5                    : READY
══ END PASTE UPDATE ══
