# PumpSmart — Module M6B: Script Plan + API Design
## Part 2 of 2 — 3-Step Script Plan, Dispatcher, Pre-Flight, API Spec, Paste Keys

**Document version:** v2.0
**Date:** 2026-04-16
**Split from:** `modules_M6B_synthetic_expanded.md` (Part 1 — fault universe, physics rules, gates)
**Canonical source of truth:** [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md)

> ⚠️ **EXECUTION STATUS:**
> - M6B = 🔴 NEXT ACTIVE — spec locked (v14.0), script **not yet run**
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
  - `secondary_onset_lag` drawn from fault-specific range per label map (Part 2A)
  - Temporal seam continuity enforcement (same as M6A spike seed seam)
  - All sequences 200-step (no variable-length sequences)
- 6 compound classes generated (labels 7–12)
- Sanity plots for Group B (temporal ordering + compound MAE distribution)
- Intermediate save: `data/synthetic/M6B_sequences_groupB.pkl`

**Output after Step 1:**
```
6 compound classes × 1,200 sequences = 7,200 sequences saved
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
- 5 masked classes generated (labels 13–17)
- Severity variant generator: `generate_variant_sequence(base_fault, variant_type, severity, cluster)`
  - `variant_type` ∈ {`intermittent`, `fast`, `cyclic`, `gradual`}
  - Physics per variant:
    - `intermittent`: NPSHa oscillation around NPSHr → Pmp.SV spike ON/OFF; `burst_interval` from Uniform(15,30)
    - `fast`: Hagen-Poiseuille large-Δ → Pres.SV drops in ≤20 steps
    - `cyclic`: Thermal sawtooth — load ON/OFF with RISING baseline each cycle; Spearman > 0.70 on detrended signal
    - `gradual`: Paris–Erdogan small ΔK → Mot.SV rises barely above baseline over 150+ steps
      - Weibull β=1.5, severity=0.05–0.25
      - CIRA anchor: same 44 bearing-impact spike seeds as label 1
      - err_slope_MotSV > 0 required in ≥95% of sequences (Gate G11-ext)
      - Sequences at severity < 0.15 will have MAE < 0.110058 — PHYSICALLY CORRECT
        (CUSUM Layer 3 + Rolling Baseline Layer 4 in M10 catch these pre-threshold sequences)
- 4 variant classes generated (labels 18–21):
  - Label 18: cavitation_intermittent
  - Label 19: seal_failure_fast
  - Label 20: overloading_cyclic
  - Label 21: bearing_wear_gradual  ← NEW v14.0 (1,000 sequences — higher count, harder to learn)
- Sanity plots for Groups C and D
- Intermediate saves:
  - `data/synthetic/M6B_sequences_groupC.pkl`
  - `data/synthetic/M6B_sequences_groupD.pkl`

**Output after Step 2:**
```
5 masked × 800 sequences    = ~4,000 sequences (Group C)
3 variant × 600 sequences   = ~1,800 sequences (labels 18–20)
1 gradual × 1,000 sequences = ~1,000 sequences (label 21 bearing_wear_gradual)
Total Step 2: ~6,800 sequences saved
data/synthetic/M6B_sequences_groupC.pkl
data/synthetic/M6B_sequences_groupD.pkl
Gate G10 masked secondary signal pass rates per class (target: ≥95%)
Gate G11-ext: err_slope_MotSV > 0 in ≥95% of label 21 sequences
Masked fault secondary-signal validation plots
Label 21 slope distribution plot (err_slope_MotSV histogram)
```

---

### Step 3 — Group E, Full Merge, Validation, Report (~700 lines appended)

**Covers:**
- Multi-sensor failure generator: `generate_multi_sensor_failure(failed_channels, failure_type, severity, cluster)`
  - Physics: 2 channels simultaneously anomalous; 6 others stay within ±0.20
  - Gate G11: exactly 2 channels anomalous, no mechanical fault signature in remaining 6
  - `multi_sensor_anomaly_count = 2` set in metadata
- 2 multi-sensor classes generated (labels per fault_rules_v3.json):
  - sensor_failure_2ch_thermal: Mot.TV + Temp.SV (common thermal power rail)
  - sensor_failure_2ch_pump: **Pres.SV + Pmp.TV** (pump-side junction box moisture ingress)
- **Individual group save:**
  - `data/synthetic/M6B_sequences_groupE.pkl` ← ~800 Group E sequences (400 × 2)
- **Full merge:**
  - Load Group A (`M6_sequences.pkl`, 8,400 seq, 7 classes, labels 0–6)
  - Load Group B (`M6B_sequences_groupB.pkl`, ~7,200 seq, labels 7–12)
  - Load Group C (`M6B_sequences_groupC.pkl`, ~4,000 seq, labels 13–17)
  - Load Group D (`M6B_sequences_groupD.pkl`, ~2,800 seq, labels 18–21)
  - Load Group E (`M6B_sequences_groupE.pkl`, ~800 seq)
  - Concatenate → `M6B_combined_sequences.pkl` (~26,000–28,000 sequences, **22 classes**)
  - `M6B_sequence_meta.csv` ← seq_id, label, group, severity, cluster, source, masked_channel_flag,
    secondary_onset_lag, err_slope_MotSV (label 21 rows only)
- **Full validation suite:**
  - Physics coupling fidelity: Mot.TV ↔ Temp.SV r≥0.87 in all thermal faults
  - Conservation check: all values in physically valid normalized ranges
  - MAE distribution plot: all **22 classes** vs threshold 0.110058
  - Label 21 sub-threshold confirmation: ≥60% of label 21 sequences have MAE < 0.110058 (expected)
  - Label distribution: 22-bar chart
  - Severity distribution per fault group
  - Physics gate summary: G1–G11 + G11-ext pass rates per class
  - Compound temporal ordering verification plot
  - Masked secondary signal strength plot
  - Label 21 slope distribution: err_slope_MotSV histogram
- **Writes `models/fault_rules_v3.json`** ← all **22-class** definitions (LOCKED for M6.5r, M7, M8)
- Writes `outputs/reports/module_06b_synthetic_report.md`
- Paste Text update block printed to console
- File manifest printed
- Next prompt printed

**Output after Step 3:**
```
data/synthetic/M6B_sequences_groupA.pkl      ← ~8,400 Group A (from M6A, labels 0–6)
data/synthetic/M6B_sequences_groupB.pkl      ← ~7,200 Group B compound (labels 7–12)
data/synthetic/M6B_sequences_groupC.pkl      ← ~4,000 Group C masked (labels 13–17)
data/synthetic/M6B_sequences_groupD.pkl      ← ~2,800 Group D variants (labels 18–21)
data/synthetic/M6B_sequences_groupE.pkl      ← ~800 Group E multi-sensor
data/synthetic/M6B_combined_sequences.pkl    ← ALL groups merged (~26,000–28,000 seq, 22 classes)
data/synthetic/M6B_sequence_meta.csv         ← full metadata table
models/fault_rules_v3.json                   ← 22-class label map (LOCKED)
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
      - variant_type         : str | None    (fast / intermittent / cyclic / gradual)
      - onset_step_primary   : int = 50      (same as M6A)
      - onset_step_secondary : int | None    (50 + lag for compounds)
      - secondary_onset_lag  : int | None    (drawn from fault-specific range)
      - severity             : float         (Weibull-sampled; β=1.5 for label 21)
      - cluster              : str           (startup / steady_state / high_load / cooldown)
    One physics engine call handles all 4 groups — parameterized dispatch.
    No 4 separate functions — just parameterized.
    """
```

**Why single dispatcher:**
- Consistent seam continuity logic across all groups
- Single severity sampling call (Weibull) — no per-group overrides
- Physics gate checks (G1–G11 + G11-ext) can run on every sequence from one loop
- Simpler to debug: one function, one trace

---

## Pre-Flight Confirmations (Locked — v14.0)

1. **Sequence length for all groups:** All sequences are 200 steps.
   Compound chains fit comfortably within 200 steps for all lag ranges. ✅

2. **M6A pkl file path:** `data/synthetic/M6_sequences.pkl` ✅ Confirmed.

3. **`fault_rules_v3.json` timing:** Steps 1 and 2 work from in-memory config dict.
   Step 3 writes the JSON to `models/fault_rules_v3.json`. ✅ Confirmed.

4. **Group B sequences per class: 1,200** (6 compound classes). ✅

5. **Label 20 sensor pair: Pres.SV + Pmp.TV** (pump-side junction box moisture ingress).
   NOT Pmp.SV + Pmp.PV. Motor-side sensors remain normal. ✅ Confirmed.

6. **`fault_group_id` field in metadata:** Written to `M6B_sequence_meta.csv` per row.
   Values: {0: normal, 1: single_source, 2: compound, 3: masked, 4: variant, 5: multi_sensor}.
   M6.5r reads this field to generate the `fault_group_id` feature column. ✅ Confirmed.

7. **Label 21 `bearing_wear_gradual` sequences: 1,000** (higher count — harder to learn).
   Weibull β=1.5, severity=0.05–0.25. CIRA anchor: same 44 bearing seeds as label 1.
   Gate G11-ext: err_slope_MotSV > 0 in ≥95% of sequences. ✅ Confirmed.

8. **Label 21 sub-threshold behaviour is EXPECTED and CORRECT:**
   Sequences at severity < 0.15 → MAE < 0.110058 in most windows.
   LSTM-AE Layer 1 alone cannot detect these — this is physically correct.
   CUSUM (Layer 3) + Rolling Baseline (Layer 4) in M10 handle pre-threshold detection.
   Do NOT raise global threshold to compensate. ✅ Confirmed.

---

## Confidence-Gated Early Warning System (M8/M10 Design)

Four-layer detection cascade — all layers must be implemented in M8/M10:

| State | Trigger | XGBoost Condition | Output |
|-------|---------|-------------------|--------|
| **WATCH** | LSTM-AE MAE crosses fuzzy lower bound (0.07–0.09) OR rolling score 2.0–3.5 OR CUSUM S_n rising | Top-class prob < 0.65 OR top-2 combined < 0.80 | `WATCH — Anomaly detected. Type uncertain. Increase inspection frequency.` |
| **WARN** | MAE in fuzzy mid-zone OR accumulator 3.5–5.0 OR CUSUM S_n > control limit | Top-class prob 0.65–0.80 | `WARN — Likely [class]. Confidence X%. Monitoring for confirmation.` |
| **FAULT** | MAE above fuzzy upper bound AND accumulator ≥5.0 | Top-class prob ≥0.80 | Single: `FAULT — [class] confirmed. Severity LOW/MED/HIGH.` Compound: `COMPOUND FAULT — Primary [A] → Secondary [B].` Masked: `FAULT + SENSOR ISSUE — fault detected via secondary sensors. Verify [masked_channel].` |
| **TREND ALERT** | Rolling Baseline: 30-window rolling mean of err_slope > μ_normal + 2σ | Any | `TREND ALERT — [channel] baseline drift detected. Plan inspection within 7–14 days.` |

**Special cases:**
- Cavitation → FAULT immediately (MAE=0.675, 6.1× threshold) — skip WATCH/WARN
- Overloading → Mech C Temp.SV Spearman drift PRIMARY — NOT single-window MAE
- Seal failure → Pres.SV Spearman (NEGATIVE) over 300 windows = `seal_failure_early` flag
- Group C (masked) → max state = WARN unless secondary signal very strong
- **Label 21 bearing_wear_gradual** ← NEW v14.0:
  - Layer 1 alone: INSUFFICIENT (MAE below threshold for most of duration)
  - Layer 3 (CUSUM): fires at ~Week 5.5 (~30% bearing degraded)
  - Layer 4 (Rolling Baseline): shift detected at ~Week 5 (pre-threshold)
  - Layer 2 (Accumulator): WATCH triggered at ~Week 6
  - Layer 1: threshold crossing at ~Week 7 (too late alone)
  - XGBoost output: `"bearing_wear_gradual — plan inspection within 7–14 days"`

---

## M10 API JSON Response Structure (Locked Design — v14.0)

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
  "cusum_state": {
    "cusum_MotSV": 0.0,
    "cusum_PresSV": 0.0,
    "cusum_TempSV": 0.0,
    "cusum_alert": false
  },
  "rolling_baseline": {
    "slope_MotSV_30w_mean": 0.0,
    "slope_PresSV_30w_mean": 0.0,
    "slope_TempSV_30w_mean": 0.0,
    "drift_alert": false,
    "drift_message": null
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

## Module Pathway — Corrected Status (v14.0)

```
M6A ✅ COMPLETE (8,400 seq, 7 classes) — LOCKED
  ↓
M6B 🔴 NEXT ACTIVE — spec locked (v14.0), script not yet run
  Target: ~26,000–28,000 seq, 22 classes (labels 0–21)
  Outputs: M6B_*.pkl, M6B_sequence_meta.csv, fault_rules_v3.json (22-class)
  ↓
M6.5r ⬜ NOT STARTED — blocked until M6B_combined_sequences.pkl exists
  Target: ~196,000 rows × 26 columns → M6B_feature_matrix.csv
  ↓
M7 ⬜ NOT STARTED — blocked until M6B_feature_matrix.csv exists
  Input:  data/synthetic/M6B_feature_matrix.csv (~196,000 × 26)
  Target: label_int (0–21), 22-class XGBoost
  Output: models/M7_xgboost_classifier.json
  ↓
M8 ⬜ NOT STARTED — LSTM-AE v2 + 4-Layer Detection (threshold unchanged: 0.110058)
  Layer 1: LSTM-AE 50-step window
  Layer 2: Fuzzy Logic + Rolling Accumulator
  Layer 3: CUSUM runtime state (M10 persistent)
  Layer 4: Rolling Baseline Comparator (M10 persistent)
  ↓
M9 → M10 → M11 → M12
```

**M6C: CANCELLED** — all valid content absorbed into M6B Groups C, D, E.
No residual content requires a separate module.

---

## Paste Keys (⚠️ ALL PENDING — populate after each step runs, do not fill in advance)

```
══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT AFTER M6B COMPLETES ══

M6B_step1_group_B_sequences       : [fill after Step 1 — target ~7,200]
M6B_step1_gate_G8_temporal         : [PASS/FAIL per class]
M6B_step1_gate_G9_compound_mae     : [PASS/FAIL per class]
M6B_step2_group_C_sequences        : [fill after Step 2 — target ~4,000]
M6B_step2_group_D_sequences        : [fill after Step 2 — target ~2,800 incl. label 21]
M6B_step2_label21_sequences        : [fill after Step 2 — target 1,000]
M6B_step2_gate_G10_masked          : [PASS/FAIL per class]
M6B_step2_gate_G11ext_label21_slope: [PASS/FAIL — err_slope_MotSV > 0 in ≥95% label 21 seqs]
M6B_step3_group_E_sequences        : [fill after Step 3 — target ~800]
M6B_step3_gate_G11_multisensor     : [PASS/FAIL]
M6B_step3_total_sequences          : [fill after Step 3 — target ~26,000–28,000 incl. M6A 8,400]
M6B_step3_classes                  : 22 (labels 0–21)
M6B_step3_fault_rules_v3_written   : [True/False]
M6B_step3_physics_violations       : [fill after Step 3 — target: NONE]
M6B_step3_coupling_fidelity_all    : [fill after Step 3 — target: r≥0.87 in all thermal faults]
M6B_step3_label21_subthreshold_pct : [% label 21 sequences with MAE < 0.110058 — expect ≥60%]
Status_for_M6p5r                   : PENDING — set to READY after Step 3 passes all gates

══ END PASTE UPDATE ══
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-15 | Created as Part 2 split from `modules_M6B_synthetic_expanded.md` v1.0. Script plan, dispatcher, pre-flight, API spec, module pathway, paste keys. Group B=1,500/class, Label 20=Pres.SV+Pmp.TV confirmed. |
| v2.0 | 2026-04-16 | **v14.0 UPGRADE**: Label 21 `bearing_wear_gradual` added to Step 2 Group D (1,000 sequences, Weibull β=1.5, Gate G11-ext). Group B corrected to 6 classes (labels 7–12), 1,200/class. Group C corrected to 5 classes (labels 13–17), 800/class. Group D updated to 4 variants (labels 18–21). All sequence counts updated. Step 3 merge updated to 22-class. Module pathway: 22 classes, ~196k rows, label_int 0–21. API response: cusum_state + rolling_baseline fields added. Early warning table: TREND ALERT state added. Label 21 special case added. Paste keys: label21 slope gate + subthreshold pct added. Pre-flight confirmation 7+8 added for label 21. |

---

*This file covers: 3-step script plan, dispatcher design, pre-flight confirmations, API spec, paste keys.*
*For fault universe physics rules, CIRA anchor rationale, dataset targets, physics gates → see [`modules_M6B_synthetic_expanded.md`](./modules_M6B_synthetic_expanded.md)*
*Canonical source of truth → [`completed_modules_M5_to_M6p5r.md`](./completed_modules_M5_to_M6p5r.md)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
