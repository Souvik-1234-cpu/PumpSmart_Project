# M12 Adversarial Validation Report  v3.0
**Date:** 2026-05-21  |  **Mode:** smoke
**Architecture:** severity-stratified, detection-classification separated
**BLOCK_M11:** **True**

## Critical gates
| Gate | Result | Spec |
|---|---|---|
| G0_hash_integrity | PASS | Model hash integrity |
| G6_cpu_latency_p95 | PASS | Latency p95 ≤ 1.0s |
| G6_cpu_latency_p99 | PASS | Latency p99 ≤ 3.0s |
| G6_cpu_latency_max | PASS | Latency max ≤ 5.0s |
| G1_normal_fpr | FAIL | Normal FPR (adapted θ_t) ≤ 1% |
| G2_in_env_detect | PASS | Acute in-envelope DETECTION ≥ 85%/label |
| G2_in_env_classify | PASS | Acute in-envelope CLASSIFICATION ≥ 85%/label |
| G3_in_env_detect | PASS | Compound in-envelope DETECTION ≥ 80% macro |
| G3_in_env_classify | PASS | Compound in-envelope CLASSIFICATION ≥ 70% macro |
| G4a_label21_cusum | PASS | Label 21 CUSUM WATCH within 1500 windows ≥ 75% |
| G4b_label21_cusum_spec | FAIL | CUSUM < 2.0 on 1000-win normals ≥ 99% |
| G7_7field_completeness | PASS | 7-field completeness 100% |
| G9_masked_detect | PASS | Masked DETECTION ≥ 80% per label |
| G10_sensor_interruption | PASS | Sensor interruption 3/3 states |
| G12_score_routing | PASS | Invariant 19 score routing |
| G14_physics_context | PASS | physics_context on non-normals |

## Advisory gates
| Gate | Result | Spec |
|---|---|---|
| G2_mild_cascade | PASS | Acute mild-extreme cascade response ≥ 30% (advisory) |
| G3_mild_cascade | PASS | Compound mild-extreme cascade response ≥ 30% (advisory) |
| G5_cross_cluster | PASS | Cross-cluster TPR ≥ 60% (advisory) |
| G8_crosspoint_lock | FAIL | L4 crosspoint guard ≥ 80% (advisory) |
| G9_masked_classify | PASS | Masked CLASSIFICATION ≥ 50% (advisory) |
| G9_mild_cascade | PASS | Masked mild-extreme cascade ≥ 20% (advisory) |
| G11_groupE_tpr | PASS | Group E TPR ≥ 55% (advisory) |
| G13_ood_responsive | FAIL | OOD flag ≥ 20% adversarial (diagnostic) |

## Details
```json
{
  "_latency": {
    "p95": 0.0331,
    "p99": 0.0383,
    "max": 0.0529
  },
  "_g1_fpr": 1.0,
  "_g2_detect_per_lbl": {
    "1": 1.0,
    "2": 1.0,
    "3": 1.0,
    "4": 1.0,
    "5": 1.0,
    "6": 1.0,
    "19": 1.0
  },
  "_g2_classify_per_lbl": {
    "1": 1.0,
    "2": 1.0,
    "3": 1.0,
    "4": 1.0,
    "5": 1.0,
    "6": 1.0,
    "19": 1.0
  },
  "_g2_mild_cascade": 1.0,
  "_g3_detect_macro": 1.0,
  "_g3_classify_macro": 1.0,
  "_g3_mild_cascade": 1.0,
  "_g4a": {
    "smoke_note": "S_n>0.01 proxy (insufficient windows for S_n\u22652.0 in smoke)",
    "accumulation_rate": 1.0,
    "cusum_max_mean": 75.88802
  },
  "_g4b_spec": 0.0,
  "_g5": 1.0,
  "_g7": 1.0,
  "_g8": {
    "lock_rate": 0.0,
    "sA_max_mean": 0.1677
  },
  "_g9_detect_per_lbl": {
    "13": 1.0,
    "14": 1.0,
    "15": 1.0,
    "16": 1.0,
    "17": 1.0
  },
  "_g9_classify_per_lbl": {
    "13": 1.0,
    "14": 1.0,
    "15": 1.0,
    "16": 1.0,
    "17": 1.0
  },
  "_g9_mild_cascade": 1.0,
  "_g10": {
    "partial": {
      "responding_rate": 1.0,
      "sA_max_mean": 0.1631
    },
    "full": {
      "responding_rate": 1.0,
      "sA_max_mean": 0.4505
    },
    "off": {
      "responding_rate": 1.0,
      "sA_max_mean": 0.4084
    }
  },
  "_g11": 1.0,
  "_g12_violations": 0,
  "_g13": 0.0,
  "_g14": 1.0
}
```