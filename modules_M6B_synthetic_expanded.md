# ⛔ SUPERSEDED — DO NOT USE FOR SCRIPTING

> **This file is the original M6B architecture specification (v1.0, 2026-04-15).**
> It contains stale sequence counts, group sizes, and module-pathway status that have been superseded by the v12.0 architecture.
>
> **Current architecture (v12.0) actuals:**
> - M6B = 21 classes (labels 0–20), **~25,000–27,000 total sequences** (Groups A–E combined)
> - Group B = 5 compound chain classes (labels 7–11), ~1,200 sequences each
> - Group C = 4 masked fault classes (labels 12–15), ~800 sequences each
> - Group D = 3 severity variant classes (labels 16–18), ~600 sequences each
> - Group E = 2 multi-sensor failure classes (labels 19–20), ~400 sequences each
> - All sequences are 200-step; longer-lag compounds clipped to 200-step windows in M6.5r
> - M6B ✅ COMPLETE 2026-04-14
> - M6.5r ✅ COMPLETE 2026-04-14 → output: `M6B_feature_matrix.csv` (~189,000 × 26)
> - M7 🔴 NEXT ACTIVE → input: `M6B_feature_matrix.csv` (~189,000 × 26, 21 classes)
>
> **Canonical source of truth:** [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md) (Part 2)
>
> This file is retained for historical audit trail only. **Do not reference for any script generation.**

---

# PumpSmart — Module M6B: Expanded Synthetic Generator
## 14 New Classes · Groups B / C / D / E · 3-Step Sequential Script

**Document version:** v1.0 — Final Architecture Lock  ~~→ SUPERSEDED by v12.0~~
**Date:** 2026-04-15
**Prerequisite:** M6A complete (`M6A_total_sequences = 8,400`, 7 classes locked)
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Output file:** `data/synthetic/M6B_combined_sequences.pkl` (21 classes, ~~27,000~~ **~25,000–27,000** sequences)

---

## Why M6B Exists — Engineering Rationale

M6A produced a clean single-fault baseline (7 classes, 1,200 sequences each).
M6B extends the training universe to cover three critical real-world failure modes
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
  Output: 20 confirmed fault classes + confidence scores
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

### Group B — Compound Chain Faults (5 classes)

Each compound chain = **unique integer label** (single-label XGBoost).
M10 API maps label → `"Primary: X → Secondary: Y"` in UI display.
Physics discriminator: secondary signal on **causally unrelated channels** = compound.
Causally explained secondary signal (e.g., thermal lag after vibration rise) = single propagating fault.

| Label | Class | Primary → Secondary | Lag (steps) | Primary Signal | Secondary Signal | CIRA Anchor |
|-------|-------|---------------------|-------------|----------------|------------------|-------------|
| 7 | `bearing_wear→overloading` | Bearing friction → motor excess current → thermal runaway | 200–400 | MotSV rises first | TempSV + MotTV both rise after | Both parent faults CIRA-anchored |
| 8 | `cavitation→seal_failure` | Joukowsky ΔP=19.1 bar shock → axial thrust → seal face blow | 50–150 | PmpSV spikes first | PresSV monotonic progressive drop | Both anchored; Joukowsky + H-P |
| 9 | `impeller_imbalance→bearing_wear` | BPF radial load → lateral bearing fatigue crack (Paris law, ISO 281) | 300–600 | PmpSV BPF-like broadband first | MotSV exponential drift after | Both anchored; BPF + Paris law |
| 10 | `seal_failure→cavitation` | Leakage → NPSHa drops below NPSHr → bubble collapse | 100–200 | PresSV smooth decline first | PmpSV erratic spikes appear | Both anchored; H-P + R-P |
| 11 | `impeller_imbalance→cavitation` | BPF pressure oscillation → localised low-pressure zone | 100–300 | PmpSV BPF-like first | PresSV erratic + PmpSV changes character to spike-bursts | Both anchored |

**Sequence length:** ~~400 steps for labels 7 and 9 (lag 200–600), 200 steps for labels 8, 10, 11.~~ **All sequences are 200 steps. Longer-lag compounds (labels 7, 9) are clipped to 200-step windows during M6.5r windowing — physically honest (Option A confirmed).**
**Target:** ~~1,500 sequences per class → 7,500 sequences total~~ **~1,200 sequences per class → ~6,000 sequences total** for Group B.

#### Compound Fault Discriminator (BIAS-01 vs BIAS-04 Resolution)

```
BIAS-01 resolution: Single fault confirmed
  Early stage → ONE fault signature strengthens → ALERT (single label)

BIAS-04 resolution: Compound fault confirmed
  Early stage → SECOND signature on physically UNRELATED channels → CRITICAL (compound label)

Physics rules:
  MotSV rises → MotTV rises 30 steps later           = SINGLE (thermal lag, causal)
  MotSV rises + PresSV drops SIMULTANEOUSLY           = COMPOUND (uncorrelated channels)
  PresSV drops → PmpSV spikes appear AFTER            = SINGLE (seal→cavitation chain)
  PresSV drops + MotTV rises SIMULTANEOUSLY           = COMPOUND (no physical link)
```

---

### Group C — Masked Faults (4 classes + 1 hard case)

Primary detection sensor fails exactly when a real fault is present.
Remaining 7 sensors carry secondary signals — model must detect via secondary path only.

| Label | Class | Real Fault | Failed Sensor | Danger | What Remains Visible | CIRA Anchor |
|-------|-------|-----------|---------------|--------|----------------------|-------------|
| 12 | `bearing_wear_MotSV_masked` | bearing_wear | MotSV (flatlined) | CRITICAL | MotTV + TempSV thermal secondary path only | Both anchored |
| 13 | `cavitation_PresSV_masked` | cavitation | PresSV (dropout) | CRITICAL | PmpSV spikes alone — strongest signal | Both anchored |
| 14 | `overloading_TempSV_masked` | overloading | TempSV (stuck) | HIGH | MotTV alone; r=0.997 coupling carries it (M5 confirmed) | Both anchored |
| 15 | `impeller_imbalance_PmpSV_masked` | impeller_imbalance | PmpSV (flatlined) | HIGH | PmpPV + PresSV oscillation secondary path | Both anchored |

**Hard case (flagged, not removed):**
- `seal_failure_PresSV_drifting` — PresSV is BOTH the primary fault indicator AND the drifting sensor. Ambiguous: is PresSV drifting because of a sensor fault or a real seal leak? Needs extra Gate G10 secondary signal strength check. Flagged as hard case in script output.

**Target:** ~~1,200 sequences per class → 4,800 sequences total~~ **~800 sequences per class → ~3,200 sequences total** for Group C (excl. hard case).

---

### Group D — Severity Variants (3 classes)

Same fault, different progression rate. Each is a new class because the sensor signature
shape is structurally different — not just scaled amplitude.

| Label | Class | Base Fault | Variant Type | Physics Mechanism | Sensor Pattern |
|-------|-------|-----------|--------------|-------------------|----------------|
| 16 | `cavitation_intermittent` | cavitation | Intermittent | NPSHa oscillates above/below NPSHr boundary | PmpSV spikes appear → vanish → reappear; intermittent on-off |
| 17 | `seal_failure_fast` | seal_failure | Fast | Catastrophic blowout; Hagen-Poiseuille at large orifice diameter d | PresSV drops rapidly within ≤20 steps (not slowly) |
| 18 | `overloading_cyclic` | overloading | Cyclic | Duty-cycle load variation; thermal sawtooth | TempSV sawtooth with rising baseline each cycle; not monotonic |

**Target:** ~~1,200 sequences per class → 3,600 sequences total~~ **~600 sequences per class → ~1,800 sequences total** for Group D.

---

### Group E — Multi-Sensor Failures (2 classes)

Beyond M6A's single-sensor failure. Common-cause hardware failures affect 2 sensors simultaneously.
Gate G11: exactly 2 channels anomalous, no mechanical fault signature in remaining 6.

| Label | Class | Failed Channels | Physics Basis | Pattern |
|-------|-------|----------------|---------------|---------|
| 19 | `sensor_failure_2ch_thermal` | MotTV + TempSV | Common power rail failure — both temperature channels share excitation circuit | Both thermal channels anomalous; vibration + pressure remain within 0.20 |
| 20 | `sensor_failure_2ch_pumpside` | PresSV + PmpTV | Moisture ingress to pump-side junction box | Pump-side sensors both fail; motor-side sensors normal |

**Not included:** `sensor_failure_2ch_vibration` (MotSV + PmpSV) — removed post physics audit.
Both vibration channels failing simultaneously masks ALL mechanical fault signatures.
XGBoost cannot distinguish this from normal + dead sensor stack. Too high ambiguity.

**Target:** ~~1,200 sequences per class → 2,400 sequences total~~ **~400 sequences per class → ~800 sequences total** for Group E.

---

## Dataset Totals — M6B Final

> ⚠️ **Stale targets below — actual run totals are in [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md)**

| Group | Classes | ~~Target Sequences~~ | Actual (v12.0) | Status |
|-------|---------|-----------|-----------|--------|
| A (M6A — frozen) | 7 | 8,400 | 8,400 | ✅ COMPLETE — DO NOT MODIFY |
| B (Compound chains) | 5 | ~~7,500~~ | ~6,000 | ✅ COMPLETE |
| C (Masked faults) | 4 | ~~4,800~~ | ~3,200 | ✅ COMPLETE |
| D (Severity variants) | 3 | ~~3,600~~ | ~1,800 | ✅ COMPLETE |
| E (Multi-sensor failure) | 2 | ~~2,400~~ | ~800 | ✅ COMPLETE |
| **TOTAL** | **21** | ~~**~26,700**~~ | **~25,000–27,000** | ✅ COMPLETE |

**RAM check:** ~27,000 × 200 × 8 × float32 ≈ 173 MB ✅ within 16 GB RAM
**M6.5r feature matrix:** ~~27,000 × 25~~ **~189,000 rows × 26 columns** ✅ M7 input

---

## Physics Gates — M6B Specific (G8–G11)

These are IN ADDITION to M6A gates G1–G7 which remain active.

| Gate | Group | Test | Pass Criterion |
|------|-------|------|----------------|
| **G8** | B | Temporal ordering — primary signal onset before secondary | Primary channel anomaly at t=50; secondary onset at t=50+lag; Spearman ordering correct ≥95% sequences |
| **G9** | B | Compound MAE — both above threshold | Weighted MAE > 0.110058 (M4 threshold) ≥90% sequences; secondary fault contributes detectable Δ-MAE |
| **G10** | C | Masked secondary signal strength | Non-masked channels carry ≥50% of base fault MAE (i.e., model can still detect with masked channel absent) |
| **G11** | E | Multi-sensor failure isolation | Exactly 2 channels anomalous; remaining 6 channels within ±0.20 normalized baseline; no mechanical fault signature |

---

## 3-Step Sequential Script Plan

**Script filename:** `module_06B_synthetic_generator.py` (~1,800–2,200 lines total)
**Pattern:** Each step appended to the same script file — identical to M6A approach.
**Execution:** You run Step 1 → paste output → I write Step 2 → repeat.

### Step 1 — Group B: Compound Chain Faults (~700 lines)

**Covers:**
- Full script header: imports, config, logging, results dict (mandatory architecture)
- `fault_rules_v3.json` loader with all 21-class definitions
- New physics gates G8 (temporal ordering), G9 (compound MAE both above threshold)
- M6A sequence pool loader (`data/synthetic/M6_sequences.pkl`) — used as seed source
- Compound chain generator: `generate_compound_sequence(primary, secondary, lag_steps, severity, cluster)`
  - Physics: primary fault runs t=50→(50+lag), secondary onset at t=(50+lag)→200
  - Temporal seam continuity enforcement (same as M6A spike seed seam)
  - All sequences 200-step; longer-lag compounds clipped at windowing stage
- 5 compound classes generated (labels 7–11)
- Sanity plots for Group B
- Intermediate save: `data/synthetic/M6B_group_B_sequences.pkl`

**Output after Step 1:**
- ~~5 compound classes × 1,500 sequences = 7,500 sequences saved~~ **5 compound classes × ~1,200 sequences = ~6,000 sequences saved**
- Compound chain physics validation plots
- Gate G8 temporal ordering pass rates per class
- Gate G9 compound MAE pass rates per class

---

### Step 2 — Groups C and D: Masked Faults + Severity Variants (~700 lines appended)

**Covers:**
- Masked fault generator: `generate_masked_sequence(base_fault, masked_channel, severity, cluster)`
  - Physics: base fault runs normally; `masked_channel` replaced with flatline/dropout at onset_step=50
  - Hard case flagged: `seal_failure_PresSV_drifting` — extra G10 secondary signal check
- 4 masked classes generated (labels 12–15)
- Severity variant generator: `generate_variant_sequence(base_fault, variant_type, severity, cluster)`
  - `variant_type` ∈ {`intermittent`, `fast`, `cyclic`}
  - Physics per variant:
    - `intermittent`: NPSHa oscillation around NPSHr → spike ON/OFF pattern
    - `fast`: Hagen-Poiseuille large-d → PresSV drops in ≤20 steps
    - `cyclic`: Thermal sawtooth — load ON/OFF with rising baseline each cycle
- 3 variant classes generated (labels 16–18)
- Sanity plots for Groups C and D
- Intermediate save: `data/synthetic/M6B_group_CD_sequences.pkl`

**Output after Step 2:**
- ~~4 masked × 1,200 + 3 variant × 1,200 = 8,400 sequences saved~~ **4 masked × ~800 + 3 variant × ~600 = ~5,000 sequences saved**
- Masked fault secondary-signal validation plots
- Gate G10 pass rates per masked class

---

### Step 3 — Group E, Full Merge, Validation, Report (~700 lines appended)

**Covers:**
- Multi-sensor failure generator: `generate_multi_sensor_failure(failed_channels, failure_type, severity, cluster)`
  - Physics: 2 channels simultaneously anomalous; 6 others stay within ±0.20
  - Gate G11: exactly 2 channels anomalous, no mechanical fault signature
- 2 multi-sensor classes generated (labels 19–20)
- **Full merge:**
  - Load `M6_sequences.pkl` (8,400 sequences, 7 classes — Group A)
  - Load M6B Group B, C+D, E pkl files
  - Concatenate → `M6B_combined_sequences.pkl` (21 classes, ~25,000–27,000 sequences)
- **Full validation suite:**
  - Physics coupling fidelity: Mot.TV ↔ Temp.SV r≥0.87 in all thermal faults
  - Conservation check: all values in physically valid normalized ranges
  - MAE distribution plot: all 21 classes vs threshold 0.110058
  - Label distribution: 21-bar chart
  - Severity distribution per fault group
  - Physics gate summary: G1–G11 pass rates per class
  - Compound temporal ordering verification plot
  - Masked secondary signal strength plot
- Writes `models/fault_rules_v3.json` (all 21-class definitions — LOCKED for M6.5r, M7, M8)
- Writes `outputs/reports/module_06B_synthetic_report.md`
- Paste Text update, report file manifest, next prompt

**Output after Step 3:**
- `data/synthetic/M6B_combined_sequences.pkl` — 21 classes, ~25,000–27,000 sequences
- `models/fault_rules_v3.json` — LOCKED
- Full sanity check plots (6-panel equivalent of M6A plots)
- `outputs/reports/module_06B_synthetic_report.md`

---

## Single Dispatcher Design Decision

To keep M6B as one clean script (~700 lines per step, not 4 separate modules):

```python
def generate_sequence(fault_config: dict) -> np.ndarray:
    """
    Unified dispatcher. fault_config carries:
      - primary_fault        : str
      - secondary_fault      : str | None   (compound chains only)
      - masked_channels      : list | None   (masked faults only)
      - variant_type         : str | None    (fast / intermittent / cyclic)
      - onset_step_primary   : int = 50      (same as M6A)
      - onset_step_secondary : int | None    (50 + lag for compounds)
      - severity             : float         (Weibull-sampled)
      - cluster              : str           (startup / steady_state / high_load / cooldown)
    One physics engine call handles all 4 groups — parameterized dispatch.
    No 4 separate functions — just parameterized.
    """
```

---

## Pre-Flight Confirmations (Locked)

1. **Sequence length for compound chains:** ~~400-step sequences for labels 7 (lag 200–400) and 9 (lag 300–600).~~ **All sequences are 200-step.** Longer-lag compounds clipped to 200-step windows during M6.5r windowing. **Option A — physically honest.** ✅ Confirmed.

2. **M6A pkl file path:** `data/synthetic/M6_sequences.pkl` ✅ Confirmed.

3. **fault_rules_v3.json timing:** Steps 1 and 2 work from in-memory config dict. Step 3 writes the JSON. No intermediate JSON files needed. ✅ Confirmed.

---

## Locked Files — DO NOT OVERWRITE in M6B

```
models/fault_rules.json          — v1 (M5/M6A reference, frozen per Invariant 16)
data/synthetic/M6_sequences.pkl  — M6A output, frozen after M6A completion
models/M3_normalization_config.json
models/M4_spike_config.json
```

---

## Confidence-Gated Early Warning System (M8/M10 Design)

Three alert states — governed by rolling accumulator + fuzzy membership + XGBoost confidence:

| State | Trigger | XGBoost Condition | Output |
|-------|---------|-------------------|--------|
| **WATCH** | LSTM-AE MAE crosses fuzzy lower bound (0.07–0.09) OR rolling score 2.0–3.5 | Top-class prob < 0.65 OR top-2 combined < 0.80 | `WATCH — Anomaly detected. Type uncertain. Increase inspection frequency.` |
| **WARN** | MAE in fuzzy mid-zone OR accumulator 3.5–5.0 | Top-class prob 0.65–0.80 | `WARN — Likely [class]. Confidence X%. Monitoring for confirmation.` |
| **FAULT** | MAE above fuzzy upper bound AND accumulator ≥5.0 | Top-class prob ≥0.80 | Single: `FAULT — [class] confirmed. Severity LOW/MED/HIGH.` Compound: `COMPOUND FAULT — Primary [A] → Secondary [B].` Masked: `FAULT + SENSOR ISSUE — fault detected via secondary sensors. Verify [masked_channel].` |

---

## M10 API JSON Response Structure (Locked Design)

```json
{
  "timestamp": "2026-04-14T21:40:00",
  "operating_cluster": "steady_state",
  "alert_state": "WARN",
  "ml_inference": {
    "fault_class": "bearing_wear",
    "fault_group": "single_source",
    "confidence": 0.72,
    "severity": "DEVELOPING",
    "mae_score": 0.143,
    "fuzzy_membership": 0.61,
    "rolling_score": 4.1,
    "shap_top3": ["mean_err_MotSV", "err_slope_MotTV", "kurtosis_PmpSV"]
  },
  "compound_check": {
    "secondary_fault_possible": true,
    "secondary_candidate": "overloading",
    "secondary_confidence": 0.31,
    "compound_class_score": 0.28,
    "verdict": "MONITOR — insufficient confidence for compound confirmation"
  },
  "early_warning": {
    "state": "WARN",
    "message": "Likely bearing_wear. Confidence 72%. MotSV trend will confirm within 15 min.",
    "watch_triggered_at": "2026-04-14T21:25:00",
    "estimated_confirmation_steps": 18
  },
  "physics_advisory": {
    "in_scope_actions": [
      "Increase vibration monitoring frequency",
      "Check bearing lubrication schedule"
    ],
    "beyond_scope_suggestions": [
      "If L10 bearing life exceeded, dry lubrication may be root cause",
      "Check coupling condition — overloading risk if shaft misalignment present"
    ],
    "fuzzy_maintenance_window": "Plan maintenance within 48-72 hrs",
    "basis": "Severity DEVELOPING, MotSV slope 0.0032/step"
  },
  "disclaimer": "Detection based on discharge-side sensors only. Suction conditions, BPF harmonics, and shaft geometry are inferred from consequence patterns, not directly measured."
}
```

---

## Module Pathway — Actual Status (v12.0)

```
M6A ✅ COMPLETE (8,400 seq, 7 classes)
  ↓
M6B ✅ COMPLETE 2026-04-14 (~25,000–27,000 seq, 21 classes)
  ↓
M6.5r ✅ COMPLETE 2026-04-14
        Output: data/synthetic/M6B_feature_matrix.csv (~189,000 rows × 26 columns)
  ↓
M7 🔴 NEXT ACTIVE — 21-class XGBoost
        Input:  data/synthetic/M6B_feature_matrix.csv (~189,000 × 26)
        Target: label_int (0–20)
        Output: models/M7_xgboost_classifier.json
  ↓
M8  — LSTM-AE v2 + Fuzzy Logic (threshold unchanged: 0.110058)
  ↓
M9 → M10 → M11 → M12
```

**M6C: CANCELLED** — all valid content absorbed into M6B Groups C, D, E.
No residual content requires a separate module.

---

## Paste Keys — Actual Values (v12.0)

> ⚠️ These keys reflect PLANNED targets from v1.0. Actual run values are in `pasted-text.txt` and `completed_modules_M5_to_M6p5r.md`.

```
M6B_step1_group_B_sequences:      ~6,000 (actual — was target 7,500)
M6B_step1_gate_G8_temporal:       PASS (all 5 compound classes)
M6B_step1_gate_G9_compound_mae:   PASS (all 5 compound classes)
M6B_step2_group_C_sequences:      ~3,200 (actual — was target 4,800)
M6B_step2_group_D_sequences:      ~1,800 (actual — was target 3,600)
M6B_step2_gate_G10_masked:        PASS (all 4 masked classes)
M6B_step3_group_E_sequences:      ~800 (actual — was target 2,400)
M6B_step3_gate_G11_multisensor:   PASS
M6B_step3_total_sequences:        ~25,000–27,000 incl. M6A 8,400
M6B_step3_fault_rules_v3_written: True
M6B_step3_physics_violations:     NONE
M6B_step3_coupling_fidelity_all:  r≥0.87 in all thermal faults
Status_for_M6p5r:                 COMPLETE
```

---

*This file is SUPERSEDED. GitHub canonical source: [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md) (Part 2).*
*Do NOT reference any Spaces .md pathway files — all outdated.*
*Next file updated: `modules_M6p5r_feature_retrain.md`*
