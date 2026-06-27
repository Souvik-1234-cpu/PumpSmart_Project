# M12 Stage 5 — Honest Validation Rig Report
**Date:** 2026-05-31  |  **Mode:** smoke  |  reps/label: 1
**Drive:** server (http://localhost:8000) | fast-forward: True

## Headline
- Overall detection (faults): **100.0%**
- Overall classification: **52.2%**
- Breakdown lead-time PASS: **39.1%**
- Normal-data false-fire: **100.0%**
- **Normal-phase integrity gate: FAIL** (2 PASS / 1 PARTIAL / 21 FAIL of 24 labels)
- **Classification integrity gate: FAIL** (12 PASS / 11 FAIL of 23 fault classes)

## Classification integrity gate (correct fault named)
Detection alone is insufficient — a safe alert must name the RIGHT fault so the operator takes the correct action. Per fault class: M7 emits the correct label during the fault run = **PASS**; detected but never correctly classified, or not detected = **FAIL** (misclassified). Normal (label 0) is N/A. Any misclassified fault fails this gate and blocks M11 readiness.

## Normal-phase integrity gate (no false fire on healthy data)
Every test runs switch-on → startup → transition → steady BEFORE any fault is injected. The model must stay NORMAL throughout this healthy run. Graded verdict: NORMAL kept = **PASS**; reached WATCH = **PARTIAL_FAIL** (over-sensitive); reached WARN/DANGER = **FAIL** (false alarm on good data — unsafe). This guarantees the model does not fire randomly on correct data.

## Dual-timer design
- **Timer 1 (detection latency):** from fault injection to first WATCH/WARN/DANGER and to first-correct M7 label. Smaller is better.
- **Timer 2 (breakdown lead-time):** from pump switch-on to physical breakdown; scored gap = DANGER→breakdown. PASS needs ≥ per-fault margin (≥60 s floor; larger for high-mechanical-damage faults).

## Per-label results
|   label | name                              | group   |   detect_rate |   classify_rate |   t1_watch_s_med | t1_danger_s_med   |   t1_correct_s_med |   t2_lead_s_med |   lead_req_s |   t2_pass_rate |   mean_conf_pct |   normal_false_fire_rate | normal_phase_gate   | normal_phase_worst   | classify_gate   |
|--------:|:----------------------------------|:--------|--------------:|----------------:|-----------------:|:------------------|-------------------:|----------------:|-------------:|---------------:|----------------:|-------------------------:|:--------------------|:---------------------|:----------------|
|       0 | normal                            | A       |             1 |             nan |              nan |                   |                nan |             nan |           60 |            nan |           nan   |                        1 | PARTIAL_FAIL        | WATCH                | N/A             |
|       1 | bearing_wear                      | A       |             1 |               1 |               50 |                   |                 50 |              -1 |          180 |              0 |            84.7 |                        1 | FAIL                | WARN                 | PASS            |
|       2 | impeller_imbalance                | A       |             1 |               1 |               50 |                   |                200 |             nan |          120 |              1 |            90.2 |                        1 | FAIL                | WARN                 | PASS            |
|       3 | cavitation                        | A       |             1 |               1 |               50 |                   |                150 |              -1 |          180 |              0 |            94   |                        1 | FAIL                | WARN                 | PASS            |
|       4 | seal_failure                      | A       |             1 |               1 |               50 |                   |                 50 |              -1 |          180 |              0 |            87.7 |                        1 | FAIL                | WARN                 | PASS            |
|       5 | overloading                       | A       |             1 |               1 |               50 |                   |                 50 |             nan |          120 |              1 |            89.6 |                        1 | FAIL                | WARN                 | PASS            |
|       6 | sensor_failure                    | A       |             1 |               1 |               50 |                   |                100 |             nan |           60 |              1 |            94   |                        1 | FAIL                | WARN                 | PASS            |
|       7 | bearing_wear__overloading         | B       |             1 |               1 |               50 |                   |                300 |              -1 |          180 |              0 |            93.4 |                        1 | FAIL                | WARN                 | PASS            |
|       8 | cavitation__seal_failure          | B       |             1 |               0 |               50 |                   |                nan |              -1 |          180 |              0 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |
|       9 | impeller_imbalance__bearing_wear  | B       |             1 |               1 |               50 |                   |                950 |              -1 |          180 |              0 |            80   |                        1 | FAIL                | WARN                 | PASS            |
|      10 | seal_failure__cavitation          | B       |             1 |               0 |              nan |                   |                nan |              -1 |          180 |              0 |           nan   |                        0 | PASS                | NORMAL               | FAIL            |
|      11 | overloading__bearing_wear         | B       |             1 |               1 |               50 |                   |                450 |             nan |          180 |              1 |            85.9 |                        1 | FAIL                | WARN                 | PASS            |
|      12 | impeller_imbalance__cavitation    | B       |             1 |               0 |               50 |                   |                nan |              -1 |          150 |              0 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |
|      13 | bearing_wear_MotSV_masked         | C       |             1 |               0 |               50 |                   |                nan |              -1 |          180 |              0 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |
|      14 | cavitation_PresSV_masked          | C       |             1 |               1 |               50 |                   |                 50 |              -1 |          180 |              0 |            84.2 |                        1 | FAIL                | WARN                 | PASS            |
|      15 | seal_failure_PresSV_drifting      | C       |             1 |               0 |               50 |                   |                nan |              -1 |          180 |              0 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |
|      16 | overloading_TempSV_stuck          | C       |             1 |               0 |               50 |                   |                nan |             nan |          120 |              1 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |
|      17 | impeller_imbalance_PmpSV_flatline | C       |             1 |               1 |             1750 |                   |                550 |             nan |          120 |              1 |            90.8 |                        0 | PASS                | NORMAL               | PASS            |
|      18 | cavitation_intermittent           | D       |             1 |               0 |               50 |                   |                nan |              -1 |           60 |              0 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |
|      19 | seal_failure_fast                 | D       |             1 |               0 |               50 |                   |                nan |              -1 |          180 |              0 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |
|      20 | overloading_cyclic                | D       |             1 |               0 |               50 |                   |                nan |             nan |           60 |              1 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |
|      21 | bearing_wear_gradual              | D       |             1 |               1 |               50 |                   |                400 |             nan |          300 |              0 |            43   |                        1 | FAIL                | WARN                 | PASS            |
|      22 | multi_sensor_vibration            | E       |             1 |               0 |               50 |                   |                nan |             nan |           60 |              1 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |
|      23 | multi_sensor_pressure_temp        | E       |             1 |               0 |               50 |                   |                nan |             nan |           60 |              1 |           nan   |                        1 | FAIL                | WARN                 | FAIL            |

## Physics provenance (chemical-engineering panel)
All sequences composed from `m6b_physics_lib` M5-faithful generators: Paris-Erdogan crack growth (bearing), orifice-discharge leak (seal), first-order thermal (overloading), Rayleigh-Plesset (cavitation), ISO 1940 unbalance (impeller), M2 thermal coupling r=0.9793. Amplification toward breakdown is a bounded ≤1.5× envelope applied to the generator's deviation amplitude only — all signatures, phases and couplings preserved. Breakdown = physical destructive-level crossing.

## C-26 disclaimer
Synthetic-domain results. Real-world F1 expected 0.65–0.85 per C-26 until active-learning first retrain (≥50 confirmed real faults). Advisory only — verify physically.