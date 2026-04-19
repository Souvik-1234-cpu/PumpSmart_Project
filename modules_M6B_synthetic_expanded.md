# PumpSmart — Module M6B: Expanded Synthetic Generator
## Part 1 of 2 — Fault Universe, Physics Rules, Dataset Targets, Gates

**Document version:** v4.0
**Date:** 2026-04-19
**Architecture:** PumpSmart v14.2 (TCN-AE Level 2 — LSTM v2 superseded)
**Prerequisite:** M6A complete (Labels 2, 3, 6, normal windows VALID; Labels 1, 4, 5 RERUN in Step 0)
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Output (PENDING):** data/synthetic/M6B_combined_sequences.pkl (~31,800 sequences, 22 classes)

> **v14.2 EXECUTION STATUS:**
> - M6B = NEXT ACTIVE — spec locked (v14.2), script not yet run, no output files exist
> - M6.5r = NOT STARTED — blocked until M6B completes
> - M7 = NOT STARTED — blocked until M6.5r completes
> - M8 = NOT STARTED — TCN-AE architecture locked
>
> Sequence lengths in this file are PHYSICS-VERIFIED (2026-04-19 audit).
> Previous 200-step default was physically wrong for most classes — corrected here.
>
> M6A rerun status:
> - Labels 0, 2, 3, 6 + normal windows: VALID — carry forward from M6A
> - Labels 1, 4, 5: RERUN_REQUIRED — regenerated in M6B Step 0 at corrected lengths
>
> Canonical source of truth: pasted-text.txt (v14.2, 2026-04-19)
> Script plan: modules_M6B_script_plan.md

---

## Why M6B Exists — Engineering Rationale

M6A produced a clean single-fault baseline (7 classes). M6B extends the training universe
to cover four critical real-world failure modes that M6A deliberately excluded:

1. Compound chain faults — one fault physically triggers a second (causal propagation)
2. Masked faults — the primary detection sensor fails exactly when a real fault is present
3. Severity variants — same fault, different progression rate (fast/slow/intermittent/cyclic/gradual)
4. Multi-sensor failures — two sensors fail simultaneously due to common-cause hardware

Without M6B, M7 and M8 have zero training coverage for these scenarios.
On a 110 kW capital asset, these are not edge cases — they are the most dangerous
and most likely real-world failure modes.

---

## CIRA Anchor Rule — Mandatory Before Any Sequence is Generated

A synthetic fault sequence is PHYSICALLY VALID only if:
  (a) Its normalization baselines exist in M3_normalization_config.json, AND
  (b) At least ONE of:
      — spike seeds (from M4_spike_seeds.npy), OR
      — coupling constants confirmed in M5, OR
      — rate-of-change parameters anchored to real CIRA observations
      is present.

If BOTH (a) and (b) fail: sequence is fabricated, not physics-informed.
Including it = scientific fraud in training data. DO NOT include.

Removed from scope (no CIRA anchor):
- dry_running — fluid heat capacity, impeller clearance not in CIRA
- shaft_misalignment — coupling stiffness, bearing span not measured
- stator_winding_fault — motor-specific, not observable from 8 sensors
- impeller_erosion — indistinguishable from impeller_imbalance at 1 Hz
- flow_blockage — absence-of-pattern not learnable by LSTM-AE
- lubrication_degradation — clinically identical to bearing_wear early-stage
- Compound chains C5, C7 — parent faults unanchored
- Transition fault T3 (surge) — Q-H positive slope not in CIRA normal data
- Thermal shock T4-T2 — rate parameters unanchored; Layer 2 physics advisory only

---

## Two-Layer Architecture — Scope Boundary

LAYER 1 — ML INFERENCE (M6B trains this)
  Input:  8 sensors, 1 Hz, CIRA-anchored
  Output: 22 confirmed fault classes + confidence scores
  Basis:  TCN-AE (M8 Level 2) + XGBoost (M7) on physics-synthetic data
  Scope:  Everything DETECTABLE from sensor space

LAYER 2 — PHYSICS ADVISORY (M10 UI only)
  Input:  Layer 1 output (fault class, severity, cluster)
  Output: Beyond-scope suggestions, maintenance actions, disclaimers
  Basis:  Pure physics equations + fuzzy logic rules
  Scope:  Real-world practical but sensor-undetectable faults

if pump_type == 'household': return physics_advisory_only()
else: return ml_prediction()

---

## Group A — Single Faults (Labels 0–6)

M6B Step 0: Labels 1, 4, 5 are REGENERATED here at corrected lengths.
Labels 0, 2, 3, 6 and normal windows are CARRIED FORWARD from M6A unchanged.

Label | Class                | Steps | Count | M6A Status          | TCN Detection Scale
------+----------------------+-------+-------+---------------------+---------------------
  0   | normal               |  200  | 2,000 | VALID               | Level 1 (static threshold)
  1   | bearing_wear         |  250  | 1,500 | RERUN M6B Step 0    | Glass 2-3 (d=2,4)
  2   | impeller_imbalance   |  200  | 1,500 | VALID               | Glass 1-2 (d=1,2)
  3   | cavitation           |  150  | 1,500 | VALID               | Glass 1 (d=1)
  4   | seal_failure         |  400  | 1,500 | RERUN M6B Step 0    | Glass 2-3 (d=2,4)
  5   | overloading          |  300  | 1,500 | RERUN M6B Step 0    | Glass 2-3 (d=2,4)
  6   | sensor_failure       |  150  | 1,200 | VALID               | Glass 1 (d=1)

Physics basis for corrected lengths:
- Label 1 (250s): Paris law exponential Mot.SV rise — 150s to reach 2x baseline + 50s pre-fault + 50s post-onset
- Label 4 (400s): Pres.SV decline — 300s for gradual orifice leak at 40 bar to reach 15% drop + 50s margins
- Label 5 (300s): Thermal runaway — 200s to reach Temp.SV* > 1.3 (Cp x m thermal mass) + 50s margins
- Label 3 (150s): Bubble inception to full collapse chain: 30-60s onset — shorter is adequate
- Label 6 (150s): Flatline/dropout = instantaneous onset, 100s persistence window needed

z_t export: M4 LSTM-AE runs in sliding window mode over every sequence.
Per 50-step window: z_t in R^64 exported to z_t_sequences_groupA.pkl

---

## Group B — Compound Chain Faults (Labels 7–12)

Each compound chain = unique integer label (single-label XGBoost).
M10 API maps label to "Primary: X -> Secondary: Y" in UI display.

Structure: pre-fault baseline (50s) + primary onset + physics-verified lag +
           secondary onset + secondary persistence

Label | Class                            | Steps | Count | Lag (steps) | Primary Signal              | Secondary Signal              | TCN Scale
------+----------------------------------+-------+-------+-------------+-----------------------------+-------------------------------+-----------
  7   | bearing_wear->overloading        |  600  | 1,500 | 200-400     | Mot.SV rises first          | Temp.SV + Mot.TV rise after   | Glass 3-4 (d=4,8)
  8   | cavitation->seal_failure         |  550  | 1,500 |  50-150     | Pmp.SV spikes first         | Pres.SV monotonic drop        | Glass 3-4 (d=4,8)
  9   | impeller_imbalance->bearing_wear |  700  | 1,500 | 300-600     | Pmp.SV BPF-like first       | Mot.SV exponential drift      | Glass 4 (d=8)
 10   | seal_failure->cavitation         |  900  | 1,500 | 400-800     | Pres.SV smooth decline first| Pmp.SV erratic spikes appear  | Glass 4-5 (d=8,16)
 11   | overloading->bearing_wear        |  800  | 1,500 | 400-600     | Temp.SV rises first         | Mot.SV drift begins           | Glass 4-5 (d=8,16)
 12   | impeller_imbalance->cavitation   |  450  | 1,500 | 100-300     | Pmp.SV BPF-like first       | Pres.SV erratic + spike bursts| Glass 3 (d=4)

Physics-Verified Lag Derivations:

Label 7 — bearing_wear -> overloading (lag 200-400 steps):
  Mechanism: Bearing heat -> oil viscosity drop -> friction torque spike -> thermal runaway
  Physics: Thermal runaway from bearing heat takes 200-400s at 1 Hz
  Old blanket 30-80 steps = physically impossible for this mechanism

Label 8 — cavitation -> seal_failure (lag 50-150 steps):
  Mechanism: Joukowsky pressure shock (dP = rho x a_wave x dV = 19.1 bar) -> axial thrust -> seal face damage
  Physics: Acute mechanical event: 50-150s is physically correct

Label 9 — impeller_imbalance -> bearing_wear (lag 300-600 steps):
  Mechanism: BPF fatigue crack growth via Paris law (dK accumulation)
  Physics: Paris fatigue crack propagation to detectable level: 300-600s
  Old blanket 30-80 = completely wrong for fatigue mechanism

Label 10 — seal_failure -> cavitation (lag 400-800 steps):
  *** MOST CRITICAL CORRECTION ***
  Mechanism: Q_leak = Cd x A_gap x sqrt(2*dP/rho) -> Q_leak ~0.0019 m3/s at 40 bar
  Step 1: Operating point shifts left on Q-H curve: 45 -> ~40 m3/h
  Step 2: Internal recirculation -> hf_suction rises 1.5-2.5 m
  Step 3: NPSHa crosses NPSHr ~4.5 m (7-stage, 2980 RPM)
  Timescale: NPSHa margin loss over 400-800s at 1 Hz
  Old 100-200 step spec = physically impossible. Cavitation CANNOT onset
  in <100s via this hydraulic pathway. At 200 total steps, entire chain undetectable.

Label 11 — overloading -> bearing_wear (lag 400-600 steps):
  Mechanism: Thermal creep -> bearing metal fatigue -> load increase
  Physics: 400-600s thermal creep to reach bearing load threshold
  Old 40-80 steps = physically impossible

Label 12 — impeller_imbalance -> cavitation (lag 100-300 steps):
  Mechanism: BPF pressure oscillation -> local low-pressure zone at blade tips -> bubble nucleation
  Physics: 100-300s for BPF to establish persistent low-P zone

z_t export: Per sequence -> z_t sequences exported to z_t_sequences_groupB.pkl

---

## Group C — Masked Faults (Labels 13–17)

Primary detection sensor fails exactly when a real fault is present.
Remaining 7 sensors carry secondary signals — model must detect via secondary path only.

Label | Class                              | Steps | Count | Real Fault         | Failed Sensor      | TCN Scale
------+------------------------------------+-------+-------+--------------------+--------------------+----------
 13   | bearing_wear_MotSV_masked          |  300  | 1,200 | bearing_wear       | Mot.SV flatline    | Glass 2-3
 14   | cavitation_PresSV_masked           |  210  | 1,200 | cavitation         | Pres.SV dropout    | Glass 1-2
 15   | seal_failure_PresSV_drifting       |  500  | 1,200 | seal_failure       | Pres.SV drifting   | Glass 3-4
 16   | overloading_TempSV_stuck           |  350  | 1,200 | overloading        | Temp.SV stuck      | Glass 2-3
 17   | impeller_imbalance_PmpSV_flatline  |  250  | 1,200 | impeller_imbalance | Pmp.SV flatline    | Glass 2

Physics basis for lengths:
- Label 13 (300s): 50s baseline + instant mask + 150s bearing degradation + 100s detection window
- Label 14 (210s): 50s baseline + instant mask + 60s cavitation + 100s detection — minimum adequate
- Label 15 (500s): 50s baseline + gradual drift 100s + 300s hydraulic seal decline + 100s detection window
- Label 16 (350s): 50s baseline + instant mask + 200s thermal overloading + 100s detection window
- Label 17 (250s): 50s baseline + instant mask + 80s BPF imbalance + 100s detection window

Max achievable alert = WARN (not DANGER) if secondary signal only.
Gate M8-13: Group C TPR >= 65% via secondary detection path.

z_t export: Per sequence -> z_t sequences exported to z_t_sequences_groupC.pkl

---

## Group D — Severity Variants (Labels 18–21)

Label | Class                   | Steps | Count | Physics Mechanism                                     | TCN Scale
------+-------------------------+-------+-------+-------------------------------------------------------+----------
 18   | cavitation_intermittent |  300  | 1,200 | NPSHa oscillates above/below NPSHr — 3+ burst cycles: 15-30s on + 20-40s off | Glass 2-3
 19   | seal_failure_fast       |  150  |   800 | Catastrophic blowout — turbulent orifice discharge; Pres.SV drops in <=20 steps -> DANGER | Glass 1
 20   | overloading_cyclic      |  600  | 1,200 | Duty-cycle load variation; thermal sawtooth with rising baseline; 3 cycles x ~150s rise + 50s recovery | Glass 3-4
 21   | bearing_wear_gradual    | 1,000 | 2,000 | Paris-Erdogan low dK regime: da/dN = C*dK^m, sev=0.05; Mot.SV* rise rate ~0.0002/step; needs 500s min for detectable slope; CUSUM needs 600-1000s | Glass 5 (d=16)

Label 21 critical note:
  At severity 0.05, old 200-step sequences produced effectively ZERO drift —
  indistinguishable from normal. This is why M6.5 showed sub-threshold MAE for
  gradual bearing. The sequences were too short to encode the fault, not the model wrong.
  Fix: 1,000 steps + 2,000 sequences (highest count — hardest class).

Label 21 detection pathway:
  Layer 1 (LSTM-AE 50-step):  INSUFFICIENT for sev < 0.15 — MAE below threshold
  Layer 2 (TCN-AE Glass 5):   score_B (drift slope) accumulates
  Layer 3 (CUSUM on score_B): S_n rising from ~Week 5.5 (~30% bearing degraded)
  Layer 4 (Rolling Baseline): slope shift detectable ~Week 5 (pre-threshold)
  XGBoost output: "bearing_wear_gradual — plan inspection within 7-14 days"
  DO NOT raise global MAE threshold to compensate for label 21.

z_t export: Per sequence -> z_t sequences exported to z_t_sequences_groupD.pkl

---

## Group E — Multi-Sensor Failures (2 sub-classes)

Class                      | Steps | Count | Failed Channels   | Physics Basis
---------------------------+-------+-------+-------------------+----------------------------------------------
sensor_failure_2ch_thermal |  250  |   800 | Mot.TV + Temp.SV  | Common power rail failure — both temperature channels share excitation circuit
sensor_failure_2ch_pump    |  250  |   800 | Pmp.SV + Pmp.PV   | Moisture ingress to pump-side junction box / shared conduit path for both accelerometers

NOTE: Group E exact label integers assigned in fault_rules_v3.json written by M6B Step 3.
Total 22 classes = labels 0-21. Do not hardcode Group E label numbers here.

Gate G11: exactly 2 channels anomalous; remaining 6 within +/-0.20 normalized baseline.

z_t export: Per sequence -> z_t sequences exported to z_t_sequences_groupE.pkl

---

## Dataset Totals — v14.2 Locked Targets

Group                | Classes | Sequences per Class         | Group Total | Labels   | Status
---------------------+---------+-----------------------------+-------------+----------+-------------------
A — Normal           |    1    | 2,000                       |    2,000    | 0        | VALID from M6A
A — Single faults    |    6    | 1500/1500/1500/1500/1500/1200 |  9,200    | 1-6      | 1,4,5: RERUN; 2,3,6: VALID
B — Compound chains  |    6    | 1,500 each                  |    9,000    | 7-12     | PENDING
C — Masked faults    |    5    | 1,200 each                  |    6,000    | 13-17    | PENDING
D — Severity variants|    4    | 1200/800/1200/2000          |    5,200    | 18-21    | PENDING
E — Multi-sensor     |    2    | 800 each                    |    1,600    | 0-21     | PENDING
TOTAL                |   22    | —                           |  ~31,800    | 0-21     | PENDING

RAM check: ~31,800 x max_steps(900) x 8 x float32 = worst-case buffer ~825 MB
Actual average steps ~400 -> ~367 MB average — well within 16 GB RAM
M6.5r feature matrix target: ~196,000 rows x ~35 columns -> M7 input (22-class)
VRAM: M4 sliding window inference batch_size=32 -> ~2 GB VRAM within 8 GB RTX 4060

---

## Physics Gates — M6B Specific (G8–G11-ext)

In addition to M6A gates G1-G7 which remain active.

Gate    | Group       | Test                                          | Pass Criterion
--------+-------------+-----------------------------------------------+-------------------------------
G8      | B           | Temporal ordering — primary onset before secondary | Primary anomaly at t=50; secondary at t=50+lag; Spearman ordering correct >=95% sequences
G9      | B           | Compound MAE — both channels above threshold  | Weighted MAE > 0.110058 >=90% sequences; secondary contributes detectable delta-MAE
G10     | C           | Masked secondary signal strength              | Non-masked channels carry >=50% of base fault MAE
G11     | E           | Multi-sensor failure isolation                | Exactly 2 channels anomalous; remaining 6 within +/-0.20 normalized baseline
G11-ext | D (label 21)| Gradual slope confirmation                    | err_slope_MotSV > 0 in >=95% of label 21 sequences

---

## z_t Export Requirements — All Groups

For every sequence in every group, after synthetic generation:
1. Run M4 LSTM-AE (lstm_ae_baseline.pth, FROZEN) in sliding window mode
2. Per 50-step window: export z_t in R^64 + MAE in R^8
3. Stack per sequence: shape (N_windows x 64) where N_windows = steps / 50
4. Save per group:
   z_t_sequences_groupA_normal.pkl
   z_t_sequences_groupA_faults.pkl
   z_t_sequences_groupB.pkl
   z_t_sequences_groupC.pkl
   z_t_sequences_groupD.pkl
   z_t_sequences_groupE.pkl

These z_t sequences feed Level 2 TCN-AE (M8).
Raw sensor data NEVER enters Level 2 directly. (Invariant 16)

---

## Locked Files — DO NOT OVERWRITE in M6B

models/fault_rules.json              — M5/M6A original 6-class reference (LOCKED)
data/synthetic/M6_sequences.pkl      — M6A output (LOCKED)
data/synthetic/M6A_sequences.pkl     — archived M6A copy (LOCKED)
models/M3_normalization_config.json  — LOCKED baselines
models/M4_spike_config.json          — LOCKED winsor bounds
models/M4_threshold_config.json      — threshold=0.110058 (LOCKED — Level 1 only)
models/lstm_ae_baseline.pth          — LOCKED frozen weights

NOTE: fault_rules_v3.json does NOT exist yet.
      It will be WRITTEN by M6B Step 3.
      Do not attempt to load it before M6B Step 3 completes.

---

## Document Revision History

Version | Date       | Change
--------+------------+--------
v1.0    | 2026-04-15 | Original monolithic M6B spec
v2.0    | 2026-04-15 | Split: script plan moved to modules_M6B_script_plan.md
v3.0    | 2026-04-16 | v14.0 upgrade: 22-class, Label 21 added
v3.1    | 2026-04-18 | v14.1 physics corrections: Group B/C/E label corrections
v4.0    | 2026-04-19 | v14.2 PHYSICS AUDIT: All sequence lengths corrected per
                       first-principles derivation. Group B lags corrected from
                       blanket 30-80 to physics-verified per-label values.
                       Sequence counts updated to ~31,800 total.
                       TCN detection scale column added. z_t export requirements
                       added. M6A rerun status added per label.

---

Fault universe physics rules, CIRA anchor rationale, dataset targets, physics gates.
Script plan (Steps 0-3, dispatcher, API design): modules_M6B_script_plan.md
Canonical source of truth: pasted-text.txt (v14.2, 2026-04-19)
Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m3/h, 450 m head — CIRA SACIP# PumpSmart — Module M6B: Expanded Synthetic Generator
## Part 1 of 2 — Fault Universe, Physics Rules, Dataset Targets, Gates

**Document version:** v4.0
**Date:** 2026-04-19
**Architecture:** PumpSmart v14.2 (TCN-AE Level 2 — LSTM v2 superseded)
**Prerequisite:** M6A complete (Labels 2, 3, 6, normal windows VALID; Labels 1, 4, 5 RERUN in Step 0)
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Output (PENDING):** data/synthetic/M6B_combined_sequences.pkl (~31,800 sequences, 22 classes)

> **v14.2 EXECUTION STATUS:**
> - M6B = NEXT ACTIVE — spec locked (v14.2), script not yet run, no output files exist
> - M6.5r = NOT STARTED — blocked until M6B completes
> - M7 = NOT STARTED — blocked until M6.5r completes
> - M8 = NOT STARTED — TCN-AE architecture locked
>
> Sequence lengths in this file are PHYSICS-VERIFIED (2026-04-19 audit).
> Previous 200-step default was physically wrong for most classes — corrected here.
>
> M6A rerun status:
> - Labels 0, 2, 3, 6 + normal windows: VALID — carry forward from M6A
> - Labels 1, 4, 5: RERUN_REQUIRED — regenerated in M6B Step 0 at corrected lengths
>
> Canonical source of truth: pasted-text.txt (v14.2, 2026-04-19)
> Script plan: modules_M6B_script_plan.md

---

## Why M6B Exists — Engineering Rationale

M6A produced a clean single-fault baseline (7 classes). M6B extends the training universe
to cover four critical real-world failure modes that M6A deliberately excluded:

1. Compound chain faults — one fault physically triggers a second (causal propagation)
2. Masked faults — the primary detection sensor fails exactly when a real fault is present
3. Severity variants — same fault, different progression rate (fast/slow/intermittent/cyclic/gradual)
4. Multi-sensor failures — two sensors fail simultaneously due to common-cause hardware

Without M6B, M7 and M8 have zero training coverage for these scenarios.
On a 110 kW capital asset, these are not edge cases — they are the most dangerous
and most likely real-world failure modes.

---

## CIRA Anchor Rule — Mandatory Before Any Sequence is Generated

A synthetic fault sequence is PHYSICALLY VALID only if:
  (a) Its normalization baselines exist in M3_normalization_config.json, AND
  (b) At least ONE of:
      — spike seeds (from M4_spike_seeds.npy), OR
      — coupling constants confirmed in M5, OR
      — rate-of-change parameters anchored to real CIRA observations
      is present.

If BOTH (a) and (b) fail: sequence is fabricated, not physics-informed.
Including it = scientific fraud in training data. DO NOT include.

Removed from scope (no CIRA anchor):
- dry_running — fluid heat capacity, impeller clearance not in CIRA
- shaft_misalignment — coupling stiffness, bearing span not measured
- stator_winding_fault — motor-specific, not observable from 8 sensors
- impeller_erosion — indistinguishable from impeller_imbalance at 1 Hz
- flow_blockage — absence-of-pattern not learnable by LSTM-AE
- lubrication_degradation — clinically identical to bearing_wear early-stage
- Compound chains C5, C7 — parent faults unanchored
- Transition fault T3 (surge) — Q-H positive slope not in CIRA normal data
- Thermal shock T4-T2 — rate parameters unanchored; Layer 2 physics advisory only

---

## Two-Layer Architecture — Scope Boundary

LAYER 1 — ML INFERENCE (M6B trains this)
  Input:  8 sensors, 1 Hz, CIRA-anchored
  Output: 22 confirmed fault classes + confidence scores
  Basis:  TCN-AE (M8 Level 2) + XGBoost (M7) on physics-synthetic data
  Scope:  Everything DETECTABLE from sensor space

LAYER 2 — PHYSICS ADVISORY (M10 UI only)
  Input:  Layer 1 output (fault class, severity, cluster)
  Output: Beyond-scope suggestions, maintenance actions, disclaimers
  Basis:  Pure physics equations + fuzzy logic rules
  Scope:  Real-world practical but sensor-undetectable faults

if pump_type == 'household': return physics_advisory_only()
else: return ml_prediction()

---

## Group A — Single Faults (Labels 0–6)

M6B Step 0: Labels 1, 4, 5 are REGENERATED here at corrected lengths.
Labels 0, 2, 3, 6 and normal windows are CARRIED FORWARD from M6A unchanged.

Label | Class                | Steps | Count | M6A Status          | TCN Detection Scale
|------|----------------------|-------|-------|---------------------|---------------------|
  0   | normal               |  200  | 2,000 | VALID               | Level 1 (static threshold)
  1   | bearing_wear         |  250  | 1,500 | RERUN M6B Step 0    | Glass 2-3 (d=2,4)
  2   | impeller_imbalance   |  200  | 1,500 | VALID               | Glass 1-2 (d=1,2)
  3   | cavitation           |  150  | 1,500 | VALID               | Glass 1 (d=1)
  4   | seal_failure         |  400  | 1,500 | RERUN M6B Step 0    | Glass 2-3 (d=2,4)
  5   | overloading          |  300  | 1,500 | RERUN M6B Step 0    | Glass 2-3 (d=2,4)
  6   | sensor_failure       |  150  | 1,200 | VALID               | Glass 1 (d=1)

Physics basis for corrected lengths:
- Label 1 (250s): Paris law exponential Mot.SV rise — 150s to reach 2x baseline + 50s pre-fault + 50s post-onset
- Label 4 (400s): Pres.SV decline — 300s for gradual orifice leak at 40 bar to reach 15% drop + 50s margins
- Label 5 (300s): Thermal runaway — 200s to reach Temp.SV* > 1.3 (Cp x m thermal mass) + 50s margins
- Label 3 (150s): Bubble inception to full collapse chain: 30-60s onset — shorter is adequate
- Label 6 (150s): Flatline/dropout = instantaneous onset, 100s persistence window needed

z_t export: M4 LSTM-AE runs in sliding window mode over every sequence.
Per 50-step window: z_t in R^64 exported to z_t_sequences_groupA.pkl

---

## Group B — Compound Chain Faults (Labels 7–12)

Each compound chain = unique integer label (single-label XGBoost).
M10 API maps label to "Primary: X -> Secondary: Y" in UI display.

Structure: pre-fault baseline (50s) + primary onset + physics-verified lag +
           secondary onset + secondary persistence

Label | Class                            | Steps | Count | Lag (steps) | Primary Signal              | Secondary Signal              | TCN Scale
|------|----------------------------------|-------|-------|-------------|-----------------------------|-------------------------------|-----------|
  7   | bearing_wear->overloading        |  600  | 1,500 | 200-400     | Mot.SV rises first          | Temp.SV + Mot.TV rise after   | Glass 3-4 (d=4,8)
  8   | cavitation->seal_failure         |  550  | 1,500 |  50-150     | Pmp.SV spikes first         | Pres.SV monotonic drop        | Glass 3-4 (d=4,8)
  9   | impeller_imbalance->bearing_wear |  700  | 1,500 | 300-600     | Pmp.SV BPF-like first       | Mot.SV exponential drift      | Glass 4 (d=8)
 10   | seal_failure->cavitation         |  900  | 1,500 | 400-800     | Pres.SV smooth decline first| Pmp.SV erratic spikes appear  | Glass 4-5 (d=8,16)
 11   | overloading->bearing_wear        |  800  | 1,500 | 400-600     | Temp.SV rises first         | Mot.SV drift begins           | Glass 4-5 (d=8,16)
 12   | impeller_imbalance->cavitation   |  450  | 1,500 | 100-300     | Pmp.SV BPF-like first       | Pres.SV erratic + spike bursts| Glass 3 (d=4)

Physics-Verified Lag Derivations:

Label 7 — bearing_wear -> overloading (lag 200-400 steps):
  Mechanism: Bearing heat -> oil viscosity drop -> friction torque spike -> thermal runaway
  Physics: Thermal runaway from bearing heat takes 200-400s at 1 Hz
  Old blanket 30-80 steps = physically impossible for this mechanism

Label 8 — cavitation -> seal_failure (lag 50-150 steps):
  Mechanism: Joukowsky pressure shock (dP = rho x a_wave x dV = 19.1 bar) -> axial thrust -> seal face damage
  Physics: Acute mechanical event: 50-150s is physically correct

Label 9 — impeller_imbalance -> bearing_wear (lag 300-600 steps):
  Mechanism: BPF fatigue crack growth via Paris law (dK accumulation)
  Physics: Paris fatigue crack propagation to detectable level: 300-600s
  Old blanket 30-80 = completely wrong for fatigue mechanism

Label 10 — seal_failure -> cavitation (lag 400-800 steps):
  *** MOST CRITICAL CORRECTION ***
  Mechanism: Q_leak = Cd x A_gap x sqrt(2*dP/rho) -> Q_leak ~0.0019 m3/s at 40 bar
  Step 1: Operating point shifts left on Q-H curve: 45 -> ~40 m3/h
  Step 2: Internal recirculation -> hf_suction rises 1.5-2.5 m
  Step 3: NPSHa crosses NPSHr ~4.5 m (7-stage, 2980 RPM)
  Timescale: NPSHa margin loss over 400-800s at 1 Hz
  Old 100-200 step spec = physically impossible. Cavitation CANNOT onset
  in <100s via this hydraulic pathway. At 200 total steps, entire chain undetectable.

Label 11 — overloading -> bearing_wear (lag 400-600 steps):
  Mechanism: Thermal creep -> bearing metal fatigue -> load increase
  Physics: 400-600s thermal creep to reach bearing load threshold
  Old 40-80 steps = physically impossible

Label 12 — impeller_imbalance -> cavitation (lag 100-300 steps):
  Mechanism: BPF pressure oscillation -> local low-pressure zone at blade tips -> bubble nucleation
  Physics: 100-300s for BPF to establish persistent low-P zone

z_t export: Per sequence -> z_t sequences exported to z_t_sequences_groupB.pkl

---

## Group C — Masked Faults (Labels 13–17)

Primary detection sensor fails exactly when a real fault is present.
Remaining 7 sensors carry secondary signals — model must detect via secondary path only.

Label | Class                              | Steps | Count | Real Fault         | Failed Sensor      | TCN Scale
|------|------------------------------------|-------|-------|--------------------|--------------------|----------|
 13   | bearing_wear_MotSV_masked          |  300  | 1,200 | bearing_wear       | Mot.SV flatline    | Glass 2-3
 14   | cavitation_PresSV_masked           |  210  | 1,200 | cavitation         | Pres.SV dropout    | Glass 1-2
 15   | seal_failure_PresSV_drifting       |  500  | 1,200 | seal_failure       | Pres.SV drifting   | Glass 3-4
 16   | overloading_TempSV_stuck           |  350  | 1,200 | overloading        | Temp.SV stuck      | Glass 2-3
 17   | impeller_imbalance_PmpSV_flatline  |  250  | 1,200 | impeller_imbalance | Pmp.SV flatline    | Glass 2

Physics basis for lengths:
- Label 13 (300s): 50s baseline + instant mask + 150s bearing degradation + 100s detection window
- Label 14 (210s): 50s baseline + instant mask + 60s cavitation + 100s detection — minimum adequate
- Label 15 (500s): 50s baseline + gradual drift 100s + 300s hydraulic seal decline + 100s detection window
- Label 16 (350s): 50s baseline + instant mask + 200s thermal overloading + 100s detection window
- Label 17 (250s): 50s baseline + instant mask + 80s BPF imbalance + 100s detection window

Max achievable alert = WARN (not DANGER) if secondary signal only.
Gate M8-13: Group C TPR >= 65% via secondary detection path.

z_t export: Per sequence -> z_t sequences exported to z_t_sequences_groupC.pkl

---

## Group D — Severity Variants (Labels 18–21)

Label | Class                   | Steps | Count | Physics Mechanism                                     | TCN Scale
|------|-------------------------|-------|-------|-------------------------------------------------------|----------|
 18   | cavitation_intermittent |  300  | 1,200 | NPSHa oscillates above/below NPSHr — 3+ burst cycles: 15-30s on + 20-40s off | Glass 2-3
 19   | seal_failure_fast       |  150  |   800 | Catastrophic blowout — turbulent orifice discharge; Pres.SV drops in <=20 steps -> DANGER | Glass 1
 20   | overloading_cyclic      |  600  | 1,200 | Duty-cycle load variation; thermal sawtooth with rising baseline; 3 cycles x ~150s rise + 50s recovery | Glass 3-4
 21   | bearing_wear_gradual    | 1,000 | 2,000 | Paris-Erdogan low dK regime: da/dN = C*dK^m, sev=0.05; Mot.SV* rise rate ~0.0002/step; needs 500s min for detectable slope; CUSUM needs 600-1000s | Glass 5 (d=16)

Label 21 critical note:
  At severity 0.05, old 200-step sequences produced effectively ZERO drift —
  indistinguishable from normal. This is why M6.5 showed sub-threshold MAE for
  gradual bearing. The sequences were too short to encode the fault, not the model wrong.
  Fix: 1,000 steps + 2,000 sequences (highest count — hardest class).

Label 21 detection pathway:
  Layer 1 (LSTM-AE 50-step):  INSUFFICIENT for sev < 0.15 — MAE below threshold
  Layer 2 (TCN-AE Glass 5):   score_B (drift slope) accumulates
  Layer 3 (CUSUM on score_B): S_n rising from ~Week 5.5 (~30% bearing degraded)
  Layer 4 (Rolling Baseline): slope shift detectable ~Week 5 (pre-threshold)
  XGBoost output: "bearing_wear_gradual — plan inspection within 7-14 days"
  DO NOT raise global MAE threshold to compensate for label 21.

z_t export: Per sequence -> z_t sequences exported to z_t_sequences_groupD.pkl

---

## Group E — Multi-Sensor Failures (2 sub-classes)

Class                      | Steps | Count | Failed Channels   | Physics Basis
|---------------------------|-------|-------|-------------------|----------------------------------------------|
sensor_failure_2ch_thermal |  250  |   800 | Mot.TV + Temp.SV  | Common power rail failure — both temperature channels share excitation circuit
sensor_failure_2ch_pump    |  250  |   800 | Pmp.SV + Pmp.PV   | Moisture ingress to pump-side junction box / shared conduit path for both accelerometers

NOTE: Group E exact label integers assigned in fault_rules_v3.json written by M6B Step 3.
Total 22 classes = labels 0-21. Do not hardcode Group E label numbers here.

Gate G11: exactly 2 channels anomalous; remaining 6 within +/-0.20 normalized baseline.

z_t export: Per sequence -> z_t sequences exported to z_t_sequences_groupE.pkl

---

## Dataset Totals — v14.2 Locked Targets

Group                | Classes | Sequences per Class         | Group Total | Labels   | Status
|---------------------|---------|-----------------------------|-------------|----------|-------------------|
A — Normal           |    1    | 2,000                       |    2,000    | 0        | VALID from M6A
A — Single faults    |    6    | 1500/1500/1500/1500/1500/1200 |  9,200    | 1-6      | 1,4,5: RERUN; 2,3,6: VALID
B — Compound chains  |    6    | 1,500 each                  |    9,000    | 7-12     | PENDING
C — Masked faults    |    5    | 1,200 each                  |    6,000    | 13-17    | PENDING
D — Severity variants|    4    | 1200/800/1200/2000          |    5,200    | 18-21    | PENDING
E — Multi-sensor     |    2    | 800 each                    |    1,600    | 0-21     | PENDING
TOTAL                |   22    | —                           |  ~31,800    | 0-21     | PENDING

RAM check: ~31,800 x max_steps(900) x 8 x float32 = worst-case buffer ~825 MB
Actual average steps ~400 -> ~367 MB average — well within 16 GB RAM
M6.5r feature matrix target: ~196,000 rows x ~35 columns -> M7 input (22-class)
VRAM: M4 sliding window inference batch_size=32 -> ~2 GB VRAM within 8 GB RTX 4060

---

## Physics Gates — M6B Specific (G8–G11-ext)

In addition to M6A gates G1-G7 which remain active.

Gate    | Group       | Test                                          | Pass Criterion
|--------|-------------|-----------------------------------------------|-------------------------------|
G8      | B           | Temporal ordering — primary onset before secondary | Primary anomaly at t=50; secondary at t=50+lag; Spearman ordering correct >=95% sequences
G9      | B           | Compound MAE — both channels above threshold  | Weighted MAE > 0.110058 >=90% sequences; secondary contributes detectable delta-MAE
G10     | C           | Masked secondary signal strength              | Non-masked channels carry >=50% of base fault MAE
G11     | E           | Multi-sensor failure isolation                | Exactly 2 channels anomalous; remaining 6 within +/-0.20 normalized baseline
G11-ext | D (label 21)| Gradual slope confirmation                    | err_slope_MotSV > 0 in >=95% of label 21 sequences

---

## z_t Export Requirements — All Groups

For every sequence in every group, after synthetic generation:
1. Run M4 LSTM-AE (lstm_ae_baseline.pth, FROZEN) in sliding window mode
2. Per 50-step window: export z_t in R^64 + MAE in R^8
3. Stack per sequence: shape (N_windows x 64) where N_windows = steps / 50
4. Save per group:
   z_t_sequences_groupA_normal.pkl
   z_t_sequences_groupA_faults.pkl
   z_t_sequences_groupB.pkl
   z_t_sequences_groupC.pkl
   z_t_sequences_groupD.pkl
   z_t_sequences_groupE.pkl

These z_t sequences feed Level 2 TCN-AE (M8).
Raw sensor data NEVER enters Level 2 directly. (Invariant 16)

---

## Locked Files — DO NOT OVERWRITE in M6B

models/fault_rules.json              — M5/M6A original 6-class reference (LOCKED)
data/synthetic/M6_sequences.pkl      — M6A output (LOCKED)
data/synthetic/M6A_sequences.pkl     — archived M6A copy (LOCKED)
models/M3_normalization_config.json  — LOCKED baselines
models/M4_spike_config.json          — LOCKED winsor bounds
models/M4_threshold_config.json      — threshold=0.110058 (LOCKED — Level 1 only)
models/lstm_ae_baseline.pth          — LOCKED frozen weights

NOTE: fault_rules_v3.json does NOT exist yet.
      It will be WRITTEN by M6B Step 3.
      Do not attempt to load it before M6B Step 3 completes.

---

## Document Revision History

Version | Date       | Change
|--------|------------|--------|
v1.0    | 2026-04-15 | Original monolithic M6B spec
v2.0    | 2026-04-15 | Split: script plan moved to modules_M6B_script_plan.md
v3.0    | 2026-04-16 | v14.0 upgrade: 22-class, Label 21 added
v3.1    | 2026-04-18 | v14.1 physics corrections: Group B/C/E label corrections
v4.0    | 2026-04-19 | v14.2 PHYSICS AUDIT: All sequence lengths corrected per
                       first-principles derivation. Group B lags corrected from
                       blanket 30-80 to physics-verified per-label values.
                       Sequence counts updated to ~31,800 total.
                       TCN detection scale column added. z_t export requirements
                       added. M6A rerun status added per label.

---

Fault universe physics rules, CIRA anchor rationale, dataset targets, physics gates.

Script plan (Steps 0-3, dispatcher, API design): modules_M6B_script_plan.md

Canonical source of truth: pasted-text.txt (v14.2, 2026-04-19)

Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m3/h, 450 m head — CIRA SACIP
