# PumpSmart — Module M6B: Script Plan + API Design
## Part 2 of 2 — 3-Step Script Plan, Dispatcher, Pre-Flight, API Spec, Paste Keys

**Document version:** v1.0
**Date:** 2026-04-15
**Split from:** `modules_M6B_synthetic_expanded.md` (Part 1 — fault universe, physics rules, gates)
**Canonical source of truth:** [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md)

> ⚠️ **EXECUTION STATUS:**
> - M6B = 🔴 NEXT ACTIVE — spec locked, script **not yet run**
> - M6.5r = ⬜ NOT STARTED — blocked until M6B completes
> - M7 = ⬜ NOT STARTED — blocked until M6.5r completes
> - **No output files exist yet.** `fault_rules_v3.json`, all `M6B_*.pkl`, `M6B_sequence_meta.csv`,
>   `M6B_feature_matrix.csv` are ALL pending — created when scripts run.

---

## 3-Step Sequential Script Plan

**Script filename:** `module_06B_synthetic_generator.py` (~1,800–2,200 lines total)
**Pattern:** Each step appended to the same script file — identical to M6A approach.
**Execution:** Run Step 1 → paste output → Step 2 written → repeat.

---

### Step 1 — Group B: Compound Chain Faults (~700 lines)

**Covers:**
- Full script header: imports, config, logging, results dict (mandatory architecture)
- `fault_rules_v3.json` config dict loaded **in-memory** (JSON not written until Step 3)
- New physics gates G8 (temporal ordering), G9 (compound MAE both above threshold)
- M6A sequence pool loader (`data/synthetic/M6_sequences.pkl`) — used as seed source
- Compound chain generator: `generate_compound_sequence(primary, secondary, lag_steps, severity, cluster)`
  - Phase 1: primary fault only (t=0 to t=50+lag)
  - Phase 2: both faults active simultaneously (t=50+lag to t=199)
  - `secondary_onset_lag` drawn from `Uniform(30, 80)` steps
  - Temporal seam continuity enforcement (same as M6A spike seed seam)
  - All sequences 200-step (no variable-length sequences)
- 5 compound classes generated (labels 7–11)
- Sanity plots for Group B (temporal ordering + compound MAE distribution)
- Intermediate save: `data/synthetic/M6B_sequences_groupB.pkl`

**Output after Step 1:**
```
5 compound classes × 1,500 sequences = 7,500 sequences saved
data/synthetic/M6B_sequences_groupB.pkl
Gate G8 temporal ordering pass rates per class (target: ≥95%)
Gate G9 compound MAE pass rates per class (target: ≥90%)
Compound chain physics validation plots
```

---

### Step 2 — Groups C and D: Masked Faults + Severity Variants (~700 lines appended)

**Covers:**
- Masked fault generator: `generate_masked_sequence(base_fault, masked_channel, severity, cluster)`
  - Physics: base fault runs normally; `masked_channel` replaced with flatline/dropout at onset_step=50
  - Hard case flagged separately: `seal_failure_PresSV_drifting` — extra G10 secondary signal check
  - `masked_channel_flag = True` set in sequence metadata
- 4 masked classes generated (labels 12–15)
- Severity variant generator: `generate_variant_sequence(base_fault, variant_type, severity, cluster)`
  - `variant_type` ∈ {`intermittent`, `fast`, `cyclic`}
  - Physics per variant:
    - `intermittent`: NPSHa oscillation around NPSHr → Pmp.SV spike ON/OFF; `burst_interval` from Uniform(15,30)
    - `fast`: Hagen-Poiseuille large-d → Pres.SV drops in ≤20 steps
    - `cyclic`: Thermal sawtooth — load ON/OFF with RISING baseline each cycle; Spearman > 0.70 on detrended signal
- 3 variant classes generated (labels 16–18)
- Sanity plots for Groups C and D
- Intermediate save: `data/synthetic/M6B_sequences_groupCD.pkl`

**Output after Step 2:**
```
4 masked × ~1,200 + 3 variant × ~1,200 = ~8,400 sequences saved
data/synthetic/M6B_sequences_groupCD.pkl
Gate G10 masked secondary signal pass rates per class (target: ≥95%)
Masked fault secondary-signal validation plots
```

---

### Step 3 — Group E, Full Merge, Validation, Report (~700 lines appended)

**Covers:**
- Multi-sensor failure generator: `generate_multi_sensor_failure(failed_channels, failure_type, severity, cluster)`
  - Physics: 2 channels simultaneously anomalous; 6 others stay within ±0.20
  - Gate G11: exactly 2 channels anomalous, no mechanical fault signature in remaining 6
  - `multi_sensor_anomaly_count = 2` set in metadata
- 2 multi-sensor classes generated (labels 19–20)
  - Label 19: Mot.TV + Temp.SV (common thermal power rail)
  - Label 20: **Pres.SV + Pmp.TV** (pump-side junction box moisture ingress)
- **Individual group saves:**
  - `data/synthetic/M6B_sequences_groupA.pkl` ← copied from `M6_sequences.pkl` (Group A frozen)
  - `data/synthetic/M6B_sequences_groupE.pkl` ← ~2,000 Group E sequences
- **Full merge:**
  - Load Group A (`M6_sequences.pkl`, 8,400 seq, 7 classes)
  - Load Group B (`M6B_sequences_groupB.pkl`, ~7,500 seq)
  - Load Group CD (`M6B_sequences_groupCD.pkl`, ~8,400 seq)
  - Load Group E (`M6B_sequences_groupE.pkl`, ~2,000 seq)
  - Concatenate → `M6B_combined_sequences.pkl` (~26,900 sequences, 21 classes)
  - `M6B_sequence_meta.csv` ← seq_id, label, group, severity, cluster, source, masked_channel_flag, secondary_onset_lag
- **Full validation suite:**
  - Physics coupling fidelity: Mot.TV ↔ Temp.SV r≥0.87 in all thermal faults
  - Conservation check: all values in physically valid normalized ranges
  - MAE distribution plot: all 21 classes vs threshold 0.110058
  - Label distribution: 21-bar chart
  - Severity distribution per fault group
  - Physics gate summary: G1–G11 pass rates per class
  - Compound temporal ordering verification plot
  - Masked secondary signal strength plot
- **Writes `models/fault_rules_v3.json`** ← all 21-class definitions (LOCKED for M6.5r, M7, M8)
- Writes `outputs/reports/module_06b_synthetic_report.md`
- Paste Text update block printed to console
- File manifest printed
- Next prompt printed

**Output after Step 3:**
```
data/synthetic/M6B_sequences_groupA.pkl      ← ~9,000 Group A (copied)
data/synthetic/M6B_sequences_groupB.pkl      ← ~7,500 Group B compound
data/synthetic/M6B_sequences_groupCD.pkl     ← ~8,400 Group C+D (intermediate)
data/synthetic/M6B_sequences_groupC.pkl      ← ~4,800 Group C split
data/synthetic/M6B_sequences_groupD.pkl      ← ~3,600 Group D split
data/synthetic/M6B_sequences_groupE.pkl      ← ~2,000 Group E
data/synthetic/M6B_combined_sequences.pkl    ← ALL groups merged (~26,900 seq, 21 classes)
data/synthetic/M6B_sequence_meta.csv         ← full metadata table
models/fault_rules_v3.json                   ← 21-class label map (LOCKED)
outputs/reports/module_06b_synthetic_report.md
```

---

## Single Dispatcher Design Decision

To keep M6B as one clean script (~700 lines per step, not 4 separate modules):

```python
def generate_sequence(fault_config: dict) -> np.ndarray:
    """
    Unified dispatcher. fault_config carries:
      - primary_fault        : str
      - secondary_fault      : str | None    (compound chains only)
      - masked_channels      : list | None   (masked faults only)
      - variant_type         : str | None    (fast / intermittent / cyclic)
      - onset_step_primary   : int = 50      (same as M6A)
      - onset_step_secondary : int | None    (50 + lag for compounds)
      - secondary_onset_lag  : int | None    (drawn from Uniform(30, 80))
      - severity             : float         (Weibull-sampled)
      - cluster              : str           (startup / steady_state / high_load / cooldown)
    One physics engine call handles all 4 groups — parameterized dispatch.
    No 4 separate functions — just parameterized.
    """
```

**Why single dispatcher:**
- Consistent seam continuity logic across all groups
- Single severity sampling call (Weibull) — no per-group overrides
- Physics gate checks (G1–G11) can run on every sequence from one loop
- Simpler to debug: one function, one trace

---

## Pre-Flight Confirmations (Locked)

1. **Sequence length for all groups:** All sequences are 200 steps.
   Compound chains with `secondary_onset_lag` from Uniform(30, 80) fit comfortably within 200 steps.
   No clipping required — lag range was chosen to guarantee Phase 2 is always visible. ✅

2. **M6A pkl file path:** `data/synthetic/M6_sequences.pkl` ✅ Confirmed.

3. **`fault_rules_v3.json` timing:** Steps 1 and 2 work from in-memory config dict.
   Step 3 writes the JSON to `models/fault_rules_v3.json`.
   No intermediate JSON files needed. ✅ Confirmed.

4. **Group B sequences per class: 1,500** (not 1,200) — compound patterns are harder to learn;
   higher count improves M7 decision boundary separation. ✅ Confirmed.

5. **Label 20 sensor pair: Pres.SV + Pmp.TV** (pump-side junction box moisture ingress).
   NOT Pmp.SV + Pmp.PV. Motor-side sensors (Mot.PV, Mot.SV, Mot.TV) remain normal. ✅ Confirmed.

6. **`fault_group_id` field in metadata:** Written to `M6B_sequence_meta.csv` per row.
   Values: {0: normal, 1: single_source, 2: compound, 3: masked, 4: variant, 5: multi_sensor}.
   M6.5r reads this field to generate the `fault_group_id` feature column. ✅ Confirmed.

---

## Confidence-Gated Early Warning System (M8/M10 Design)

Three alert states — governed by rolling accumulator + fuzzy membership + XGBoost confidence:

| State | Trigger | XGBoost Condition | Output |
|-------|---------|-------------------|--------|
| **WATCH** | LSTM-AE MAE crosses fuzzy lower bound (0.07–0.09) OR rolling score 2.0–3.5 | Top-class prob < 0.65 OR top-2 combined < 0.80 | `WATCH — Anomaly detected. Type uncertain. Increase inspection frequency.` |
| **WARN** | MAE in fuzzy mid-zone OR accumulator 3.5–5.0 | Top-class prob 0.65–0.80 | `WARN — Likely [class]. Confidence X%. Monitoring for confirmation.` |
| **FAULT** | MAE above fuzzy upper bound AND accumulator ≥5.0 | Top-class prob ≥0.80 | Single: `FAULT — [class] confirmed. Severity LOW/MED/HIGH.` Compound: `COMPOUND FAULT — Primary [A] → Secondary [B].` Masked: `FAULT + SENSOR ISSUE — fault detected via secondary sensors. Verify [masked_channel].` |

**Special cases:**
- Cavitation → FAULT immediately (MAE=0.675, 6.1× threshold) — skip WATCH/WARN
- Overloading → Mech C Temp.SV Spearman drift PRIMARY — NOT single-window MAE
- Seal failure → Pres.SV Spearman (NEGATIVE) over 300 windows = `seal_failure_early` flag
- Group C (masked) → max state = WARN unless secondary signal very strong

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
    "fault_group_id": 1,
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
  "masked_sensor_check": {
    "masked_channel_flag": false,
    "masked_channel": null,
    "detection_path": "primary"
  },
  "multi_sensor_check": {
    "multi_sensor_anomaly_count": 0,
    "anomalous_channels": []
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

## Module Pathway — Corrected Status

```
M6A ✅ COMPLETE (8,400 seq, 7 classes) — LOCKED
  ↓
M6B 🔴 NEXT ACTIVE — spec locked, script not yet run
  Target: ~26,900 seq, 21 classes
  Outputs: M6B_*.pkl, M6B_sequence_meta.csv, fault_rules_v3.json
  ↓
M6.5r ⬜ NOT STARTED — blocked until M6B_combined_sequences.pkl exists
  Target: ~189,000 rows × 26 columns → M6B_feature_matrix.csv
  ↓
M7 ⬜ NOT STARTED — blocked until M6B_feature_matrix.csv exists
  Input:  data/synthetic/M6B_feature_matrix.csv (~189,000 × 26)
  Target: label_int (0–20)
  Output: models/M7_xgboost_classifier.json
  ↓
M8 ⬜ NOT STARTED — LSTM-AE v2 + Fuzzy Logic (threshold unchanged: 0.110058)
  ↓
M9 → M10 → M11 → M12
```

**M6C: CANCELLED** — all valid content absorbed into M6B Groups C, D, E.
No residual content requires a separate module.

---

## Paste Keys (⚠️ ALL PENDING — populate after each step runs, do not fill in advance)

```
══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT AFTER M6B COMPLETES ══

M6B_step1_group_B_sequences      : [fill after Step 1 — target ~7,500]
M6B_step1_gate_G8_temporal        : [PASS/FAIL per class]
M6B_step1_gate_G9_compound_mae    : [PASS/FAIL per class]
M6B_step2_group_C_sequences       : [fill after Step 2 — target ~4,800]
M6B_step2_group_D_sequences       : [fill after Step 2 — target ~3,600]
M6B_step2_gate_G10_masked         : [PASS/FAIL per class]
M6B_step3_group_E_sequences       : [fill after Step 3 — target ~2,000]
M6B_step3_gate_G11_multisensor    : [PASS/FAIL]
M6B_step3_total_sequences         : [fill after Step 3 — target ~26,900 incl. M6A 8,400]
M6B_step3_fault_rules_v3_written  : [True/False]
M6B_step3_physics_violations      : [fill after Step 3 — target: NONE]
M6B_step3_coupling_fidelity_all   : [fill after Step 3 — target: r≥0.87 in all thermal faults]
Status_for_M6p5r                  : PENDING — set to READY after Step 3 passes all gates

══ END PASTE UPDATE ══
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-15 | Created as Part 2 split from `modules_M6B_synthetic_expanded.md` v1.0. Contains script plan, dispatcher, pre-flight, API spec, module pathway, paste keys. All sequence counts corrected: Group B=1,500/class, Group C=1,200/class, Group D=1,200/class, Group E=1,000/class. Module pathway corrected: M6B=🔴 NEXT ACTIVE, M6.5r=⬜ NOT STARTED, M7=⬜ NOT STARTED. All paste keys set to PENDING. Label 20 = Pres.SV + Pmp.TV confirmed. fault_group_id pre-flight note added. |

---

*This file covers: 3-step script plan, dispatcher design, pre-flight confirmations, API spec, paste keys.*
*For fault universe physics rules, CIRA anchor rationale, dataset targets, physics gates → see [`modules_M6B_synthetic_expanded.md`](./modules_M6B_synthetic_expanded.md)*
*Canonical source of truth → [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
