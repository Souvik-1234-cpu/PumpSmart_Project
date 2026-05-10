# PumpSmart v14.2 — Scope Statement

**Date:** 2026-05-09
**Audit reference:** PumpSmart Industrial Audit v3.0, Section 10.3, Concern E (T1.5.3)

---

## Asset Envelope

PumpSmart v14.2 is designed and validated for:

- **Asset type:** Multi-stage centrifugal pump (inline, end-suction)
- **Nameplate:** 110 kW · 7-stage · 40 bar · 450 m · 2980 RPM · 45 m³/h
- **Motor:** IEC frame 315 mm · 400 V · 2-pole
- **Dataset anchor:** CIRA SACIP (Zenodo) — same pump class
- **Deployment posture:** Shadow-mode advisory only · Human-in-the-loop mandatory

---

## Fault Classes Detected (22 Classes)

| Group | Labels | Description |
|---|---|---|
| A — Single faults | 0,1,2,3,4,5,6,19 | Normal, bearing wear, imbalance, cavitation, seal failure, overloading, sensor failure, seal_failure_fast |
| B — Compound chains | 7,8,9,10,11,12 | Physically causal compound faults (Fault A → Fault B, 400–900 steps) |
| C — Masked faults | 13,14,15,16,17 | Sensor failure masking underlying process fault |
| D — Cyclic/Gradual | 18,20,21 | Intermittent, cyclic, gradual bearing wear |
| E — Sensor anomaly | 22,23 | **Single-channel** sensor anomaly with secondary indicator disturbance |

---

## ⚠ OUT OF SCOPE — MULTI-SENSOR COMMON-CAUSE FAILURE

> **PumpSmart v14.2 detects single-sensor anomalies (Labels 22, 23). Common-cause multi-sensor failures (shared excitation rail, EMI burst, moisture ingress affecting multiple sensors) are OUT OF SCOPE and require separate detection mechanisms.**

Labels 22 (sensor_anomaly_thermal) and 23 (sensor_anomaly_pump) detect
**single-channel** sensor faults only. The following multi-sensor scenarios
are NOT detected:

- Shared excitation rail loss affecting two sensors simultaneously
- EMI burst corrupting multiple SCADA channels at once
- Moisture ingress into a junction box affecting both Pmp.SV and Pmp.PV
- Power supply fault affecting entire instrument loop

**Why:** The original "two-sensor simultaneous failure" class definition was
aspirational — the generator produced single-channel anomalies (visualization
audit T1.7, 2026-05-09). Without real-data anchoring, a valid two-channel
generator cannot be built without fabricating physics. Path B (reclassification)
was selected as the honest engineering choice.

---

## ⚠ OUT OF SCOPE — OTHER CONDITIONS

| Condition | Reason |
|---|---|
| Household monoblock pumps | Cross-domain ML = out-of-distribution inference = safety risk |
| Axial / positive displacement pumps | Different hydraulic physics |
| Parallel pump interaction | Single-pump monitoring only |
| Shaft misalignment | Not in 22-class taxonomy — OOD detector flags these |
| Foundation looseness | Not in 22-class taxonomy — OOD detector flags these |
| Autonomous trip authority | IEC 61511 requires 1oo2/2oo3 voting — PumpSmart is one voter only |

---

## F1 Citation Format

When citing performance figures, always use this complete form:

> *5-fold sequence-stratified cross-validation on physics-synthetic data,
> macro F1 = 0.9965 ± 0.0005. Real-world F1 expected to be 0.65–0.85 per
> C-26 and published transfer-learning literature on rotating equipment.
> This number is not production-validated.*

**Do not cite 0.9965 alone** — without the C-26 disclaimer it implies
production-validated performance.

---

*PumpSmart v14.2 · T1.5.3 scope statement · 2026-05-09*
