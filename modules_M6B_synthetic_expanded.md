# ⛔ SUPERSEDED — DO NOT USE FOR SCRIPTING

> **This file is the original M6B architecture specification (v3.1, 2026-04-18).**
> It contains the fault universe physics rules, CIRA anchor rationale, dataset targets,
> and physics gates. The script plan has been split into a companion file.
>
> **⚠️ EXECUTION STATUS — READ BEFORE REFERENCING:**
> - M6B = 🔴 NEXT ACTIVE — spec locked (v14.1), script **not yet run**, no output files exist
> - M6.5r = ⬜ NOT STARTED — blocked until M6B completes
> - M7 = ⬜ NOT STARTED — blocked until M6.5r completes
>
> **Locked sequence targets (canonical — match `completed_modules_M5_to_M6p5r.md` v14.1):**
> - Group A = 7 classes, ~1,200 each → ~8,400 total
> - Group B = **6 compound classes, 1,200 each → ~7,200 total** (labels 7–12)
> - Group C = **5 masked classes, 800 each → ~4,000 total** (labels 13–17)
> - Group D = **4 variant classes → ~2,800 total** (labels 18–21; label 21 = 1,000 sequences)
> - Group E = 2 multi-sensor classes, ~400 each → ~800 total
> - **Grand total: ~26,000–28,000 sequences, 22 classes (labels 0–21)**
>
> **`sensor_failure_2ch_pump` = Pmp.SV + Pmp.PV** (pump-side junction box moisture ingress).
> NOT Pres.SV + Pmp.TV — that pairing was a previous physics error and is now corrected.
>
> **Label 21 = `bearing_wear_gradual` (NEW v14.0)** — Paris–Erdogan slow crack propagation.
> Sub-threshold MAE in severity < 0.15 sequences is PHYSICALLY CORRECT — do NOT raise threshold.
> Detected via CUSUM (Layer 3) + Rolling Baseline (Layer 4) in M10, NOT Layer 1 alone.
>
> **Canonical source of truth:** [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md) (Part 2A)
> **Script plan (Steps 1–3, dispatcher, pre-flight, API design):** [`modules_M6B_script_plan.md`](./modules_M6B_script_plan.md)
>
> This file is retained for historical audit trail and physics rule reference only.
> **Do not reference for any script generation — use `modules_M6B_script_plan.md` instead.**

---

# PumpSmart — Module M6B: Expanded Synthetic Generator
## Part 1 of 2 — Fault Universe, Physics Rules, Dataset Targets, Gates

**Document version:** v3.1
**Date:** 2026-04-18
**Prerequisite:** M6A complete (`M6A_total_sequences = 8,400`, 7 classes locked)
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Output file (PENDING):** `data/synthetic/M6B_combined_sequences.pkl` (~26,000–28,000 sequences, **22 classes**)

---

## Why M6B Exists — Engineering Rationale

M6A produced a clean single-fault baseline (7 classes, 1,200 sequences each).
M6B extends the training universe to cover four critical real-world failure modes
that M6A deliberately excluded:

1. **Compound chain faults** — one fault physically triggers a second (causal propagation)
2. **Masked faults** — the primary detection sensor fails exactly when a real fault is present
3. **Severity variants** — same fault, different progression rate (fast/slow/intermittent/cyclic/gradual)
4. **Multi-sensor failures** — two sensors fail simultaneously due to common-cause hardware

Without M6B, M7 and M8 have zero training coverage for these scenarios.
On a 110 kW capital asset, these are not edge cases — they are the most
dangerous and most likely real-world failure modes.

---

## CIRA Anchor Rule — Mandatory Before Any Sequence is Generated

```
A synthetic fault sequence is PHYSICALLY VALID only if:
  (a) Its normalization baselines exist in M3_normalization_config.json, AND
  (b) At least ONE of:
      — spike seeds (from M4_spike_seeds.npy), OR
      — coupling constants confirmed in M5, OR
      — rate-of-change parameters anchored to real CIRA observations
      is present.

If BOTH (a) and (b) fail → sequence is fabricated, not physics-informed.
Including it = scientific fraud in training data. DO NOT include.
```

### Removed from scope (no CIRA anchor):
- `dry_running` — PmpTV rate-of-rise depends on fluid heat capacity, impeller clearance geometry — not in CIRA or nameplate
- `shaft_misalignment` — BPF sideband spacing depends on coupling stiffness, shaft diameter, bearing span — not measured
- `stator_winding_fault` — 2× line frequency modulation amplitude depends on winding resistance imbalance, pole geometry — motor-specific
- `impeller_erosion` — Finnie equation requires particle velocity V and impact angle θ — neither observable from 8 sensors; signature indistinguishable from `impeller_imbalance` at 1 Hz sampling
- `flow_blockage` — PresSV sustained low is also `seal_failure` early stage and cavitation inter-spike baseline; absence-of-pattern not learnable by LSTM-AE
- `lubrication_degradation` — Stribeck curve rate parameters depend on oil grade, bearing clearance, lubrication interval — none in CIRA; clinically identical to `bearing_wear` early-stage (model as severity 0.1–0.3 instead)
- Compound chains C5 (`dry_running → bearing_wear`) and C7 (`shaft_misalignment → bearing_wear`) — parent faults unanchored
- Transition fault T3 (surge near-shutoff) — Q-H curve positive slope not observable in CIRA normal data
- Thermal shock (T4-T2) — rate parameters unanchored; moved to Layer 2 physics advisory

---

## Two-Layer Architecture — Scope Boundary

```
LAYER 1 — ML INFERENCE (M6B trains this)
  Input:  8 sensors, 1 Hz, CIRA-anchored
  Output: 22 confirmed fault classes + confidence scores
  Basis:  LSTM-AE (M8) + XGBoost (M7) on physics-synthetic data
  Scope:  Everything DETECTABLE from sensor space

LAYER 2 — PHYSICS ADVISORY (M10 UI only)
  Input:  Layer 1 output (fault class, severity, cluster)
  Output: Beyond-scope suggestions, maintenance actions, disclaimers
  Basis:  Pure physics equations + fuzzy logic rules
  Scope:  Real-world practical but sensor-undetectable faults
          (dry_running, shaft_misalignment, lubrication_degradation,
           thermal_shock, water_hammer as condition advisory)

if pump_type == 'household': return physics_advisory_only()
else: return ml_prediction()
```

---

## Complete M6B Fault Universe — 15 New Classes (Labels 7–21)

### Group B — Compound Chain Faults (6 classes, 1,200 sequences each → ~7,200 total)

Each compound chain = **unique integer label** (single-label XGBoost).
M10 API maps label → `"Primary: X → Secondary: Y"` in UI display.
Physics discriminator: secondary signal on **causally unrelated channels** = compound.
Causally explained secondary signal (e.g., thermal lag after vibration rise) = single propagating fault.

| Label | Class | Primary → Secondary | Lag (steps) | Primary Signal | Secondary Signal | CIRA Anchor |
|-------|-------|---------------------|-------------|----------------|------------------|-------------|
| 7 | `bearing_wear+overloading` | Bearing friction → motor excess current → thermal runaway | 30–80 | Mot.SV rises first | Temp.SV + Mot.TV both rise after | Both parent faults CIRA-anchored |
| 8 | `cavitation+seal_failure` | Joukowsky ΔP=19.1 bar shock → axial thrust → seal face blow | 30–80 | Pmp.SV spikes first | Pres.SV monotonic progressive drop | Both anchored; Joukowsky + gradual hydraulic leak physics |
| 9 | `impeller_imbalance+bearing_wear` | BPF radial load → lateral bearing fatigue crack (Paris law, ISO 281) | 30–80 | Pmp.SV BPF-like broadband first | Mot.SV exponential drift after | Both anchored; BPF + Paris law |
| 10 | `seal_failure+cavitation` | Leakage → NPSHa drops below NPSHr → bubble collapse | 30–80 | Pres.SV smooth decline first | Pmp.SV erratic spikes appear | Both anchored; gradual hydraulic leak + R-P |
| 11 | `overloading+bearing_wear` | Thermal overload first, then bearing degradation accumulates | 40–80 | Temp.SV rises first | Mot.SV begins drift after | Both anchored |
| 12 | `impeller_imbalance+cavitation` | BPF pressure oscillation → localised low-pressure zone | 30–80 | Pmp.SV BPF-like first | Pres.SV erratic + Pmp.SV changes character to spike-bursts | Both anchored |

**Sequence length:** All sequences are 200 steps.
`secondary_onset_lag` drawn from fault-specific range per label map above.
**Target:** 1,200 sequences per class → **~7,200 sequences total** for Group B.

#### Compound Fault Discriminator (BIAS-01 vs BIAS-04 Resolution)

```
BIAS-01 resolution: Single fault confirmed
  Early stage → ONE fault signature strengthens → ALERT (single label)

BIAS-04 resolution: Compound fault confirmed
  Early stage → SECOND signature on physically UNRELATED channels → CRITICAL (compound label)

Physics rules:
  Mot.SV rises → Mot.TV rises 30 steps later          = SINGLE (thermal lag, causal)
  Mot.SV rises + Pres.SV drops SIMULTANEOUSLY          = COMPOUND (uncorrelated channels)
  Pres.SV drops → Pmp.SV spikes appear AFTER           = SINGLE (seal→cavitation chain)
  Pres.SV drops + Mot.TV rises SIMULTANEOUSLY          = COMPOUND (no physical link)
```

---

### Group C — Masked Faults (5 classes, 800 sequences each → ~4,000 total)

Primary detection sensor fails exactly when a real fault is present.
Remaining 7 sensors carry secondary signals — model must detect via secondary path only.

| Label | Class | Real Fault | Failed Sensor | Danger | What Remains Visible | CIRA Anchor |
|-------|-------|-----------|---------------|--------|----------------------|-------------|
| 13 | `bearing_wear_MotSV_masked` | bearing_wear | Mot.SV (flatlined) | CRITICAL | Mot.TV + Temp.SV thermal secondary path only | Both anchored |
| 14 | `cavitation_PresSV_masked` | cavitation | Pres.SV (dropout) | CRITICAL | Pmp.SV spikes alone — strongest signal | Both anchored |
| 15 | `seal_failure_PresSV_drifting` | seal_failure | Pres.SV (drifting) | CRITICAL | Secondary hydraulic channels only | Both anchored |
| 16 | `overloading_TempSV_stuck` | overloading | Temp.SV (stuck) | HIGH | Mot.TV alone; r=0.997 coupling carries it (M5 confirmed) | Both anchored |
| 17 | `impeller_imbalance_PmpSV_flatline` | impeller_imbalance | Pmp.SV (flatlined) | HIGH | Pmp.PV + cross-channel correlation change | Both anchored |

**Target:** 800 sequences per class → **~4,000 sequences total** for Group C.
Max achievable alert state = WARN (not DANGER) if secondary signal only.
Gate M8-13: Group C TPR ≥ 65% via secondary Mech C path.

---

### Group D — Severity Variants (4 classes → ~2,800 total)

Same fault, different progression rate. Each is a new class because the sensor signature
shape is structurally different — not just scaled amplitude.

| Label | Class | Base Fault | Variant Type | Physics Mechanism | Sensor Pattern |
|-------|-------|-----------|--------------|-------------------|----------------|
| 18 | `cavitation_intermittent` | cavitation | Intermittent | NPSHa oscillates above/below NPSHr boundary | Pmp.SV spikes appear → vanish → reappear; `burst_interval` from Uniform(15, 30) steps |
| 19 | `seal_failure_fast` | seal_failure | Fast | Catastrophic blowout; turbulent orifice discharge through enlarged effective leak area | Pres.SV drops in ≤20 steps (not slowly) → DANGER within 1–3 windows |
| 20 | `overloading_cyclic` | overloading | Cyclic | Duty-cycle load variation; thermal sawtooth with RISING baseline | Temp.SV sawtooth + rising baseline each cycle; Spearman > 0.70 on detrended signal |
| 21 | `bearing_wear_gradual` | bearing_wear | Gradual | **Paris–Erdogan low ΔK regime: da/dN = C·ΔKᵐ, β=1.5, sev=0.05–0.25** | Mot.SV barely above baseline; err_slope_MotSV > 0 consistently over 150+ steps; MAE < threshold for sev < 0.15 (PHYSICALLY CORRECT) |

**Label 21 detailed physics:**
```
Mechanism   : Paris–Erdogan law at small ΔK (early-stage bearing crack)
              da/dN = C · ΔK^m, m=3 (steel), ΔK = 0.05–0.25 (normalized surrogate)
CIRA anchor : Same 44 bearing-impact spike seeds as label 1 (bearing_wear)
              Amplitude attenuated by factor 0.05–0.25 per severity
Weibull     : β=1.5, shape→long tail (chronic low-level degradation)
Detection   :
  Layer 1 (LSTM-AE 50-step): INSUFFICIENT for sev < 0.15 — MAE below threshold
  Layer 2 (Accumulator)    : WATCH triggered ~Week 6
  Layer 3 (CUSUM)          : S_n rising from ~Week 5.5 (≈30% bearing degraded)
  Layer 4 (Rolling Baseline): slope shift from ~Week 5 (pre-threshold)
XGBoost output: "bearing_wear_gradual — plan inspection within 7–14 days"
Do NOT raise global MAE threshold to compensate for label 21.
```

**Sequences per class:**
- Labels 18–20: ~600 sequences each
- Label 21: **1,000 sequences** (higher count — harder to learn; more training coverage)
- **Group D total: ~2,800 sequences**

---

### Group E — Multi-Sensor Failures (2 classes, ~400 sequences each → ~800 total)

Beyond M6A’s single-sensor failure. Common-cause hardware failures affect 2 sensors simultaneously.
Gate G11: exactly 2 channels anomalous, no mechanical fault signature in remaining 6.

| Label | Class | Failed Channels | Physics Basis | Pattern |
|-------|-------|----------------|---------------|---------|
| [fault_rules_v3.json] | `sensor_failure_2ch_thermal` | Mot.TV + Temp.SV | Common power rail failure — both temperature channels share excitation circuit | Both thermal channels anomalous; vibration + pressure remain within ±0.20 |
| [fault_rules_v3.json] | `sensor_failure_2ch_pump` | **Pmp.SV + Pmp.PV** | **Moisture ingress to pump-side junction box / shared conduit path for both accelerometers** | Both pump vibration channels simultaneously degrade; motor-side sensors remain normal; `multi_sensor_anomaly_count = 2` |

> ⚠️ NOTE: Group E exact label integers are assigned in
> `fault_rules_v3.json` written by M6B Step 3.
> Total 22 classes = labels 0–21. Do not hardcode Group E label numbers here.

**Target:** ~400 sequences per class → **~800 sequences total** for Group E.

---

## Dataset Totals — M6B Locked Targets (v14.1)

> ⚠️ These are TARGETS. Actual counts populate after M6B script runs.
> Actuals go into paste keys in `completed_modules_M5_to_M6p5r.md`.

| Group | Classes | Sequences per Class | Group Total | Labels | Status |
|-------|---------|---------------------|-------------|--------|--------|
| A (M6A — frozen) | 7 | ~1,200 | ~8,400 | 0–6 | ✅ EXISTS — do not modify |
| B (Compound chains) | **6** | **1,200** | **~7,200** | 7–12 | ⏳ PENDING |
| C (Masked faults) | **5** | **800** | **~4,000** | 13–17 | ⏳ PENDING |
| D (Severity variants) | **4** | 600 / 600 / 600 / **1,000** | **~2,800** | 18–21 | ⏳ PENDING |
| E (Multi-sensor failure) | 2 | **~400** | **~800** | within 0–21 | ⏳ PENDING |
| **TOTAL** | **22** | — | **~26,000–28,000** | 0–21 | ⏳ PENDING |

```
RAM check: ~27,000 × 200 × 8 × float32 ≈ 173 MB ✅ within 16 GB RAM
M6.5r feature matrix target: ~196,000 rows × 26 columns → M7 input (22-class)
```

---

## Physics Gates — M6B Specific (G8–G11 + G11-ext)

These are IN ADDITION to M6A gates G1–G7 which remain active.

| Gate | Group | Test | Pass Criterion |
|------|-------|------|----------------|
| **G8** | B | Temporal ordering — primary signal onset before secondary | Primary channel anomaly at t=50; secondary onset at t=50+lag; Spearman ordering correct ≥95% sequences |
| **G9** | B | Compound MAE — both channels above threshold | Weighted MAE > 0.110058 (M4 threshold) ≥90% sequences; secondary fault contributes detectable Δ-MAE |
| **G10** | C | Masked secondary signal strength | Non-masked channels carry ≥50% of base fault MAE — model can still detect with masked channel absent |
| **G11** | E | Multi-sensor failure isolation | Exactly 2 channels anomalous; remaining 6 channels within ±0.20 normalized baseline; no mechanical fault signature |
| **G11-ext** | D (label 21) | Gradual slope confirmation | err_slope_MotSV > 0 in ≥95% of label 21 sequences; confirms Paris–Erdogan monotonic progression |

---

## Locked Files — DO NOT OVERWRITE in M6B

```
models/fault_rules.json              — M5/M6A original 6-class reference (LOCKED per Invariant 16)
data/synthetic/M6_sequences.pkl      — M6A output (LOCKED after M6A completion)
data/synthetic/M6A_sequences.pkl     — archived M6A copy (LOCKED)
models/M3_normalization_config.json  — LOCKED baselines
models/M4_spike_config.json          — LOCKED winsor bounds
models/M4_threshold_config.json      — threshold=0.110058 (LOCKED)
```

```
NOTE: fault_rules_v3.json does NOT exist yet.
      It will be WRITTEN by M6B Step 3.
      Do not attempt to load it before M6B Step 3 completes.
      It will contain ALL 22 classes (labels 0–21).
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-15 | Original monolithic M6B spec — fault universe, script plan, API design, paste keys |
| v2.0 | 2026-04-15 | **SPLIT + CORRECTION.** Script plan, dispatcher, API design, paste keys moved to `modules_M6B_script_plan.md`. |
| v3.0 | 2026-04-16 | **v14.0 UPGRADE**: 22-class everywhere. Label 21 `bearing_wear_gradual` added to Group D. Group B and Group C expanded. |
| v3.1 | 2026-04-18 | **v14.1 PHYSICS CORRECTION**: Group B labels 11/12 corrected to canonical map. Group C labels 15/16/17 corrected to canonical map. Group E pump-side multi-sensor pair corrected to `Pmp.SV + Pmp.PV`. `seal_failure_fast` physics corrected from Hagen–Poiseuille wording to turbulent orifice discharge. Group E labels clarified as assigned within `fault_rules_v3.json` and total class count remains 22 (labels 0–21). |

---

*This file covers: fault universe physics rules, CIRA anchor rationale, dataset targets, physics gates.*
*For script plan (Steps 1–3), dispatcher design, API spec, paste keys → see [`modules_M6B_script_plan.md`](./modules_M6B_script_plan.md)*
*Canonical source of truth → [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
