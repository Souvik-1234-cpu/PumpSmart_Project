# module_08p7b_slope_continuity_gate — Report
**Date:** 2026-05-09
**Gate status:** FAIL (9.67% sequences)

## Method

Tests primary-channel slope continuity at Phase 2 boundary.
For each compound sequence, on primary channels only:
- slope_pre  = linear slope over [p2_start-10 : p2_start-1]
- slope_post = linear slope over [p2_start : p2_start+10]
- ratio = slope_post / slope_pre — must be in [0.3, 3.0]

**Why primary channels only:** secondary onset affects secondary-fault channels.
Primary channels continue their own fault progression through Phase 2 unaffected.
If np.tile freeze bug were present, slope_post ≈ 0 → ratio → 0 → FAIL.

## Per-Label Results

| Label | Class | Pass | Pass Rate |
|---|---|---|---|
| 7 | bearing_wear+overloading | 11/1500 | 0.7% |
| 8 | cavitation+seal_failure | 107/1500 | 7.1% |
| 9 | impeller_imbalance+bearing_wear | 36/1500 | 2.4% |
| 10 | seal_failure+cavitation_H | 57/1500 | 3.8% |
| 11 | overloading+bearing_wear | 27/1500 | 1.8% |
| 12 | impeller_imbalance+cavitation | 632/1500 | 42.1% |

## Gates

| Gate | Status | Detail |
|---|---|---|
| T1.5.2_G_L7 | FAIL | 11/1500 (0.7%) target >=95% |
| T1.5.2_G_L8 | FAIL | 107/1500 (7.1%) target >=95% |
| T1.5.2_G_L9 | FAIL | 36/1500 (2.4%) target >=95% |
| T1.5.2_G_L10 | FAIL | 57/1500 (3.8%) target >=95% |
| T1.5.2_G_L11 | FAIL | 27/1500 (1.8%) target >=95% |
| T1.5.2_G_L12 | FAIL | 632/1500 (42.1%) target >=95% |
| T1.5.2_G_overall | FAIL | 9.67% overall (target >=95%) |

## Interpretation

- Ratios near 1.0: primary fault progressing at same rate before/after secondary onset
- Ratios 0.3–3.0: primary rate changed (e.g. secondary interaction coupling) but continuous
- Ratios outside bounds: abrupt freeze or reversal — artifact present

---
*module_08p7b_slope_continuity_gate | PumpSmart v14.2 | 2026-05-09*
