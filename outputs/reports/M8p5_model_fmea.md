# PumpSmart v14.2 — Model FMEA Matrix
**Date:** 2026-05-09
**Methodology:** IEC 60812
**Asset under protection:** 110 kW 7-stage centrifugal pump (INR 30-40 lakh)

## Scoring scale
- **Severity (S):** 1 (negligible) to 10 (catastrophic, no warning)
- **Occurrence (O):** 1 (improbable) to 10 (very high)
- **Detection (D):** 1 (almost certain to detect) to 10 (cannot detect)
- **RPN = S × O × D** (1 to 1000)

## Action thresholds
- RPN > 300 → 🔴 BLOCKER — must mitigate before any deployment
- RPN 200–300 → 🟡 HIGH — mitigate before M11 production deployment
- RPN < 200 → 🟢 ACCEPTABLE — monitor through deployment lifecycle

## Aggregate
| | Total RPN |
|---|---|
| Pre-mitigation | **2107** |
| Post-mitigation | **753** |
| Reduction | 64.3% |

All six failure modes are reduced to RPN ≤ 200 (acceptable for shadow-mode
deployment). Remaining residual risk is the irreducible synthetic-to-real gap,
which is bounded by the active learning loop (C-26 mitigation) and the OOD
detector (M8p4).

---

## Failure modes

### FM-01 — SCADA -> PumpSmart input pipeline halted  🔴 BLOCKER

| | Severity | Occurrence | Detection | RPN |
|---|---|---|---|---|
| Pre-mitigation | 9 | 4 | 9 | **324** |
| Post-mitigation | 9 | 4 | 2 | **72** ✓ ACCEPTABLE |

**Description:** Network partition / SCADA outage / OPC server crash. Inference pipeline starves silently.

**Effect on asset:** No alerts issued. Silent monitoring loss while pump continues operating.

**Mitigation:**

(1) M10 must implement a heartbeat: if no /api/predict call received for > 5 minutes, an INDEPENDENT watchdog process raises an alert via separate channel (email, Slack, SMS). (2) /api/health must include 'last_predict_call_age_seconds' field. (3) M11 deployment must register an external monitor pinging /api/health every 60s and alerting on staleness.

**Recovery procedure:** Restore SCADA link, M10 auto-resumes when calls arrive. CUSUM state preserved (was paused, not reset).

---


### FM-02 — High-confidence wrong fault label  🔴 BLOCKER

| | Severity | Occurrence | Detection | RPN |
|---|---|---|---|---|
| Pre-mitigation | 10 | 5 | 7 | **350** |
| Post-mitigation | 10 | 5 | 3 | **150** ✓ ACCEPTABLE |

**Description:** M7 produces a high-probability prediction for the wrong class. Operator performs wrong maintenance action; underlying fault progresses.

**Effect on asset:** Wrong maintenance applied. Underlying real fault continues to progression. If real fault is bearing seizure, asset destroyed within 2-4 hours.

**Mitigation:**

(1) M8p4 OOD detector active in M10 - rejects classifications for inputs that don't look like training distribution. (2) 7-field output Field 4 (Expected Sensor Behavior) lets operator cross-check prediction against physical observation BEFORE acting. (3) M10 confidence threshold below 0.85 prompts manual verification. (4) M12 adversarial validation (planned) explicitly tests for this failure mode.

**Recovery procedure:** Operator-initiated /api/operator_ack with rejection flag. Active learning queue records the mistake for next M7 retrain.

---


### FM-03 — Missed gradual bearing wear (Label 21 false negative)  🟡 HIGH

| | Severity | Occurrence | Detection | RPN |
|---|---|---|---|---|
| Pre-mitigation | 10 | 3 | 8 | **240** |
| Post-mitigation | 10 | 3 | 5 | **150** ✓ ACCEPTABLE |

**Description:** Real Paris-law bearing degradation in progress but CUSUM never crosses H. Bearing seizes within 2-4 hours of crossing the unobserved fault threshold.

**Effect on asset:** Catastrophic bearing failure. Replacement INR 35-50 lakh + downtime.

**Mitigation:**

(1) CUSUM auto-decay (this patch) preserves long accumulation while preventing alarm fatigue from suppressing real WATCH. (2) Independent inspection schedule: every 30 days physical check of bearing condition regardless of PumpSmart status. PumpSmart never replaces inspection. (3) M9 Industrial Pump Selector flags pumps operating outside BEP envelope - those have higher Paris-law L10 hours and need tighter inspection intervals.

**Recovery procedure:** Physical inspection reveals fault. Bearing replaced. Sequence added to active learning queue with operator-confirmed Label 21 ground truth.

---


### FM-04 — CUSUM stuck WATCH (alarm fatigue induction)  🔴 BLOCKER

| | Severity | Occurrence | Detection | RPN |
|---|---|---|---|---|
| Pre-mitigation | 10 | 7 | 9 | **630** |
| Post-mitigation | 10 | 7 | 2 | **140** ✓ ACCEPTABLE |

**Description:** Operator investigates WATCH alerts repeatedly with no fault found. Without a non-maintenance reset path, S_n stays elevated. Eventually WATCH fires every call. Operator stops responding.

**Effect on asset:** Real Label 21 alert ignored when it eventually fires. Same downstream consequence as FM-03.

**Mitigation:**

(1) CUSUM geometric decay (this patch): lambda = 5.73e-05 applied per quiet call. Half-life = 7 days. (2) /api/cusum_quiet_review endpoint: explicit 'investigated, no fault' reset to 0.3 * S_n. (3) Alarm fatigue protection: if WATCH fires > 24h with no operator interaction, downgrade to daily summary email.

**Recovery procedure:** Decay returns S_n to baseline naturally over 7 days quiet operation. Operator review reset returns immediately to safe baseline.

---


### FM-05 — Silent TCN-AE (or any model) numerical degradation  🟡 HIGH

| | Severity | Occurrence | Detection | RPN |
|---|---|---|---|---|
| Pre-mitigation | 9 | 3 | 9 | **243** |
| Post-mitigation | 9 | 3 | 3 | **81** ✓ ACCEPTABLE |

**Description:** GPU->CPU casting bug, library version mismatch in deployment, NaN poisoning from a bad sensor sample, etc. Model produces low-quality scores with no observable change to operator.

**Effect on asset:** Detection performance silently drops. False negatives rise. Same downstream consequence as FM-03 if a real fault occurs in the degraded window.

**Mitigation:**

(1) Daily self-test routine (M11): inject 10 known synthetic test sequences from a frozen golden-set, compare outputs to expected score_A/B/C ranges. Halt if any deviation exceeds 5%. (2) NaN guard at every model boundary - reject any input/output with NaN values. (3) Version-pin all torch/sklearn/xgboost/numpy versions in Docker image with hash check.

**Recovery procedure:** Halt inference, alert operations team, roll back to prior known-good Docker image.

---


### FM-06 — L4 adaptive threshold chasing real slow degradation  🔴 BLOCKER

| | Severity | Occurrence | Detection | RPN |
|---|---|---|---|---|
| Pre-mitigation | 10 | 4 | 8 | **320** |
| Post-mitigation | 10 | 4 | 4 | **160** ✓ ACCEPTABLE |

**Description:** A real slow drift is in progress that should fire L3. L4's 6-hour rolling baseline tracks the drift up before L3 accumulates - threshold rises with the fault, suppressing the score_A alert.

**Effect on asset:** Fault concealment. Same downstream consequence as FM-03.

**Mitigation:**

(1) Crosspoint guard: theta_t locked if it exceeds 1.5 * theta_initial. Existing M8 design - verify implementation in M10 runtime. (2) L3 CUSUM operates on score_B (drift slope), independent of L4's score_A baseline. Invariant 19 (no cross-routing) ensures L4 cannot suppress L3. (3) Add a metric to /api/threshold_status: theta_t / theta_initial ratio. If > 1.3 for > 24h, raise an engineering review notice.

**Recovery procedure:** Engineering review identifies whether drift is real fault (commission Label 21 inspection) or operating-point shift (re-cluster via commissioning mode).

---


## Relationship to project challenge log
| Failure mode | Related challenge |
|---|---|
| FM-01 | New — not in C-01..C-27 |
| FM-02 | C-26 (synthetic-to-real domain gap) — addressed by M8p4 OOD detector |
| FM-03 | C-27 (sequence length physics), C-15 (false sense of security) |
| FM-04 | C-25 (adaptive threshold paradox) — solved here for the operational case |
| FM-05 | New — not in C-01..C-27 |
| FM-06 | C-25 (adaptive threshold paradox) — operational guardrail |

## Required M10/M11 implementation work to realize the post-mitigation scores
1. **M10:** /api/cusum_quiet_review endpoint with audit log
2. **M10:** OOD detector (M8p4) wired into /api/predict
3. **M10:** Heartbeat field on /api/health + age tracking
4. **M11:** External monitor pinging /api/health every 60s
5. **M11:** Daily golden-set self-test routine
6. **M11:** NaN guards at all model boundaries
7. **M11:** Version-pinned Docker image with hash check
8. **M11:** Independent maintenance inspection schedule documentation (handover)

Until all 8 implementation items are complete, the post-mitigation RPN scores
above are DESIGN INTENT, not realised. The deployment posture should be
shadow-mode-only until items 1-7 are coded and tested.
