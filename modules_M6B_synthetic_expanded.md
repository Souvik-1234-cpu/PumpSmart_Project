# ⛔ SUPERSEDED — DO NOT USE FOR SCRIPTING

> **This file is the original M6B architecture specification (v1.0, 2026-04-15).**
> It contains the fault universe physics rules, CIRA anchor rationale, dataset targets,
> and physics gates. The script plan has been split into a companion file.
>
> **⚠️ EXECUTION STATUS — READ BEFORE REFERENCING:**
> - M6B = 🔴 NEXT ACTIVE — spec locked, script **not yet run**, no output files exist
> - M6.5r = ⬜ NOT STARTED — blocked until M6B completes
> - M7 = ⬜ NOT STARTED — blocked until M6.5r completes
>
> **Locked sequence targets (canonical — match `completed_modules_M5_to_M6p5r.md`):**
> - Group A = 7 classes, ~1,200–1,500 each → ~9,000 total
> - Group B = 5 compound classes, **1,500 each → ~7,500 total**
> - Group C = 4 masked classes, **~1,200 each → ~4,800 total**
> - Group D = 3 variant classes, **~1,200 each → ~3,600 total**
> - Group E = 2 multi-sensor classes, **~1,000 each → ~2,000 total**
> - **Grand total: ~26,900 sequences, 21 classes**
>
> **Label 20 = `sensor_failure_2ch_pumpside` = Pres.SV + Pmp.TV (pump-side junction box moisture ingress)**
> NOT Pmp.SV + Pmp.PV — that was a previous error, now corrected everywhere.
>
> **Canonical source of truth:** [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md) (Part 2)
> **Script plan (Steps 1–3, dispatcher, pre-flight, API design):** [`modules_M6B_script_plan.md`](./modules_M6B_script_plan.md)
>
> This file is retained for historical audit trail and physics rule reference only.
> **Do not reference for any script generation — use `modules_M6B_script_plan.md` instead.**

---

# PumpSmart — Module M6B: Expanded Synthetic Generator
## Part 1 of 2 — Fault Universe, Physics Rules, Dataset Targets, Gates

**Document version:** v2.0 — Corrected (split from v1.0 monolithic)
**Date:** 2026-04-15
**Prerequisite:** M6A complete (`M6A_total_sequences = 8,400`, 7 classes locked)
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Output file (PENDING):** `data/synthetic/M6B_combined_sequences.pkl` (~26,900 sequences, 21 classes)

---

## Why M6B Exists — Engineering Rationale

M6A produced a clean single-fault baseline (7 classes, 1,200 sequences each).
M6B extends the training universe to cover four critical real-world failure modes
that M6A deliberately excluded:

1. **Compound chain faults** — one fault physically triggers a second (causal propagation)
2. **Masked faults** — the primary detection sensor fails exactly when a real fault is present
3. **Severity variants** — same fault, different progression rate (fast/slow/intermittent/cyclic)
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
  Output: 21 confirmed fault classes + confidence scores
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

## Complete M6B Fault Universe — 14 New Classes (Labels 7–20)

### Group B — Compound Chain Faults (5 classes, 1,500 sequences each → ~7,500 total)

Each compound chain = **unique integer label** (single-label XGBoost).
M10 API maps label → `"Primary: X → Secondary: Y"` in UI display.
Physics discriminator: secondary signal on **causally unrelated channels** = compound.
Causally explained secondary signal (e.g., thermal lag after vibration rise) = single propagating fault.

| Label | Class | Primary → Secondary | Lag (steps) | Primary Signal | Secondary Signal | CIRA Anchor |
|-------|-------|---------------------|-------------|----------------|------------------|-------------|
| 7 | `bearing_wear+overloading` | Bearing friction → motor excess current → thermal runaway | 30–80 | Mot.SV rises first | Temp.SV + Mot.TV both rise after | Both parent faults CIRA-anchored |
| 8 | `cavitation+seal_failure` | Joukowsky ΔP=19.1 bar shock → axial thrust → seal face blow | 30–80 | Pmp.SV spikes first | Pres.SV monotonic progressive drop | Both anchored; Joukowsky + H-P |
| 9 | `impeller_imbalance+bearing_wear` | BPF radial load → lateral bearing fatigue crack (Paris law, ISO 281) | 30–80 | Pmp.SV BPF-like broadband first | Mot.SV exponential drift after | Both anchored; BPF + Paris law |
| 10 | `seal_failure+cavitation` | Leakage → NPSHa drops below NPSHr → bubble collapse | 30–80 | Pres.SV smooth decline first | Pmp.SV erratic spikes appear | Both anchored; H-P + R-P |
| 11 | `impeller_imbalance+cavitation` | BPF pressure oscillation → localised low-pressure zone | 30–80 | Pmp.SV BPF-like first | Pres.SV erratic + Pmp.SV changes character to spike-bursts | Both anchored |

**Sequence length:** All sequences are 200 steps.
`secondary_onset_lag` drawn from `Uniform(30, 80)` steps for all Group B classes.
**Target:** 1,500 sequences per class → **~7,500 sequences total** for Group B.

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

### Group C — Masked Faults (4 classes, ~1,200 sequences each → ~4,800 total)

Primary detection sensor fails exactly when a real fault is present.
Remaining 7 sensors carry secondary signals — model must detect via secondary path only.

| Label | Class | Real Fault | Failed Sensor | Danger | What Remains Visible | CIRA Anchor |
|-------|-------|-----------|---------------|--------|----------------------|-------------|
| 12 | `bearing_wear_MotSV_masked` | bearing_wear | Mot.SV (flatlined) | CRITICAL | Mot.TV + Temp.SV thermal secondary path only | Both anchored |
| 13 | `cavitation_PresSV_masked` | cavitation | Pres.SV (dropout) | CRITICAL | Pmp.SV spikes alone — strongest signal | Both anchored |
| 14 | `overloading_TempSV_masked` | overloading | Temp.SV (stuck) | HIGH | Mot.TV alone; r=0.997 coupling carries it (M5 confirmed) | Both anchored |
| 15 | `impeller_imbalance_PmpSV_masked` | impeller_imbalance | Pmp.SV (flatlined) | HIGH | Pmp.PV + Pres.SV oscillation secondary path | Both anchored |

**Hard case (flagged, not a 5th class):**
- `seal_failure_PresSV_drifting` — Pres.SV is BOTH the primary fault indicator AND the drifting sensor. Ambiguous: is Pres.SV drifting because of a sensor fault or a real seal leak? Flagged separately in script output with extra Gate G10 secondary signal strength check. NOT a training class.

**Target:** ~1,200 sequences per class → **~4,800 sequences total** for Group C.
Max achievable alert state = WARN (not DANGER) if secondary signal only.
Gate M8-13: Group C TPR ≥ 65% via secondary Mech C path.

---

### Group D — Severity Variants (3 classes, ~1,200 sequences each → ~3,600 total)

Same fault, different progression rate. Each is a new class because the sensor signature
shape is structurally different — not just scaled amplitude.

| Label | Class | Base Fault | Variant Type | Physics Mechanism | Sensor Pattern |
|-------|-------|-----------|--------------|-------------------|----------------|
| 16 | `cavitation_intermittent` | cavitation | Intermittent | NPSHa oscillates above/below NPSHr boundary | Pmp.SV spikes appear → vanish → reappear; `burst_interval` from Uniform(15, 30) steps |
| 17 | `seal_failure_fast` | seal_failure | Fast | Catastrophic blowout; Hagen-Poiseuille at large orifice diameter d | Pres.SV drops in ≤20 steps (not slowly) → DANGER within 1–3 windows |
| 18 | `overloading_cyclic` | overloading | Cyclic | Duty-cycle load variation; thermal sawtooth with RISING baseline | Temp.SV sawtooth + rising baseline each cycle; Spearman > 0.70 on detrended signal |

**Target:** ~1,200 sequences per class → **~3,600 sequences total** for Group D.

---

### Group E — Multi-Sensor Failures (2 classes, ~1,000 sequences each → ~2,000 total)

Beyond M6A's single-sensor failure. Common-cause hardware failures affect 2 sensors simultaneously.
Gate G11: exactly 2 channels anomalous, no mechanical fault signature in remaining 6.

| Label | Class | Failed Channels | Physics Basis | Pattern |
|-------|-------|----------------|---------------|---------|
| 19 | `sensor_failure_2ch_thermal` | Mot.TV + Temp.SV | Common power rail failure — both temperature channels share excitation circuit | Both thermal channels anomalous; vibration + pressure remain within ±0.20 |
| 20 | `sensor_failure_2ch_pumpside` | **Pres.SV + Pmp.TV** | **Moisture ingress to pump-side junction box** — both pump-side sensors affected; motor-side sensors (Mot.PV, Mot.SV, Mot.TV) remain normal | Pres.SV + Pmp.TV simultaneously degrade; `multi_sensor_anomaly_count = 2` |

**Not included:** `sensor_failure_2ch_vibration` (Mot.SV + Pmp.SV) — removed post physics audit.
Both vibration channels failing simultaneously masks ALL mechanical fault signatures.
XGBoost cannot distinguish this from normal + dead sensor stack. Too high ambiguity.

**Target:** ~1,000 sequences per class → **~2,000 sequences total** for Group E.

---

## Dataset Totals — M6B Locked Targets

> ⚠️ These are TARGETS. Actual counts populate after M6B script runs.
> Actuals go into paste keys in `completed_modules_M5_to_M6p5r.md`.

| Group | Classes | Sequences per Class | Group Total | Status |
|-------|---------|---------------------|-------------|--------|
| A (M6A — frozen) | 7 | ~1,200 | ~8,400 | ✅ EXISTS — do not modify |
| B (Compound chains) | 5 | **1,500** | **~7,500** | ⏳ PENDING |
| C (Masked faults) | 4 | **~1,200** | **~4,800** | ⏳ PENDING |
| D (Severity variants) | 3 | **~1,200** | **~3,600** | ⏳ PENDING |
| E (Multi-sensor failure) | 2 | **~1,000** | **~2,000** | ⏳ PENDING |
| **TOTAL** | **21** | — | **~26,900** | ⏳ PENDING |

```
RAM check: ~27,000 × 200 × 8 × float32 ≈ 173 MB ✅ within 16 GB RAM
M6.5r feature matrix target: ~189,000 rows × 26 columns → M7 input
```

---

## Physics Gates — M6B Specific (G8–G11)

These are IN ADDITION to M6A gates G1–G7 which remain active.

| Gate | Group | Test | Pass Criterion |
|------|-------|------|----------------|
| **G8** | B | Temporal ordering — primary signal onset before secondary | Primary channel anomaly at t=50; secondary onset at t=50+lag; Spearman ordering correct ≥95% sequences |
| **G9** | B | Compound MAE — both channels above threshold | Weighted MAE > 0.110058 (M4 threshold) ≥90% sequences; secondary fault contributes detectable Δ-MAE |
| **G10** | C | Masked secondary signal strength | Non-masked channels carry ≥50% of base fault MAE — model can still detect with masked channel absent |
| **G11** | E | Multi-sensor failure isolation | Exactly 2 channels anomalous; remaining 6 channels within ±0.20 normalized baseline; no mechanical fault signature |

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
      It will be WRITTEN by M6B Step 3 (not read).
      Do not attempt to load it before M6B Step 3 completes.
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-15 | Original monolithic M6B spec — fault universe, script plan, API design, paste keys |
| v2.0 | 2026-04-15 | **SPLIT + CORRECTION.** Script plan, dispatcher, API design, paste keys moved to `modules_M6B_script_plan.md`. Fixes: M6B/M6.5r/M7 status corrected (none executed). Group B sequences corrected: 1,500 each / ~7,500 total. Group C: ~1,200 each / ~4,800 total. Group D: ~1,200 each / ~3,600 total. Group E: ~1,000 each / ~2,000 total. **Label 20 sensor pair corrected: Pres.SV + Pmp.TV** (moisture ingress, NOT Pmp.SV + Pmp.PV). All ✅ COMPLETE status markers in superseded banner corrected. fault_rules_v3.json noted as PENDING M6B output. |

---

*This file covers: fault universe physics rules, CIRA anchor rationale, dataset targets, physics gates.*
*For script plan (Steps 1–3), dispatcher design, API spec, paste keys → see [`modules_M6B_script_plan.md`](./modules_M6B_script_plan.md)*
*Canonical source of truth → [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
