# module_12_stage4_step5_response_assembly_d2_closure

Date: 2026-05-29  
Overall: **PASS**

## Phase A_response_text

Status: **PASS**

| Gate | Status |
|---|---|
| c1_normal_label0 | PASS |
| c2_watch_label0 | PASS |
| c3_warn_label0 | PASS |
| c4_danger_label0 | PASS |
| c5_warn_cavitation | PASS |
| c6_danger_label21 | PASS |
| c7_normal_cusum_high | PASS |

## Phase B_floor_derivation

Status: **PASS**

| Gate | Status |
|---|---|
| floor_above_p99_rm100 | PASS |
| floor_above_p99_rm200 | PASS |
| floor_below_step1d_0.6157 | PASS |

## Phase C_behaviour_regression

Status: **PASS**

| Gate | Status |
|---|---|
| normal_sustained_stays_normal | PASS |
| acute_still_danger | PASS |
| sustained_floor_warns | PASS |
| mode_transition_stays_normal | PASS |

## Derived floors

```json
{
  "baseline_centre_score_A": 0.157,
  "baseline_sigma_score_A": 0.012,
  "N_pool": 2000,
  "rm100_p95": 0.15819570514992398,
  "rm100_p99": 0.15856722108295485,
  "rm200_p95": 0.1575482571579805,
  "rm200_p99": 0.1579305219610248,
  "rolling_mean_100_floor": 0.1740152756649164,
  "rolling_mean_200_floor": 0.17330308287377855
}
```
