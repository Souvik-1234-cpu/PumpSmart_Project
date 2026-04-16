# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
# PART 1A OF 3 — LSTM-AE Architecture + Fuzzy Membership Layer

**Document version:** v3.0 — 22-class M6B alignment (v14.0)
**Date:** 2026-04-16
**Part 1B (Mechanisms):** `module_M8_lstm_ae_v2_mechanisms.md` — Stage 3 Mech A/B/C, Detection Map, Training Data
**Part 2 (Gates + Outputs):** `module_M8_lstm_ae_v2_gates_and_outputs.md`
**Prerequisite:** M7 all 16 gates passed | `M7_all_16_gates_pass = True`
**Asset:** 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP)
**Status:** NOT STARTED — begins only after M7 gates confirmed

> ⚠️ READ ALL THREE PARTS BEFORE WRITING ANY M8 CODE.
> Part 1A = LSTM-AE + Fuzzy layer (this file).
> Part 1B = Slow drift mechanisms, detection map, training data.
> Part 2  = Gates, outputs, paste keys, adaptive actions.

---

## Safety Mandate (NON-NEGOTIABLE)

```
Asset              : 110 kW, 7-stage, 40 bar, 450m head multistage centrifugal pump
Replacement cost   : >₹50 lakh (industrial capital asset)
Failure consequence: Process shutdown + secondary damage + personnel injury risk
Standard           : ISO 10816-3 (vibration), ISO 13373-3 (condition monitoring)
Pipeline level     : ISO 13374 Level 3 condition monitoring

M8 is NOT an experiment. M4 was the baseline — M8 is the PRODUCTION model.
Every architectural decision must be physically justified.
Every gate must pass before M12 adversarial validation begins.
False negative on this asset = catastrophic failure.

PREREQUISITE: M7_all_16_gates_pass = True before M8 starts.
If M7 gates fail → fix M6.5r → rerun M7 → only then start M8.
```

---

## v2.0 → v3.0 Change Summary (v14.0)

| Item | v2.0 | v3.0 (CURRENT — USE THIS) | Reason |
|------|------|--------------------------|--------|
| M7 prerequisite gate count | 15 gates | **16 gates** | Gate M7-14-ext added for label 21 |
| Classes | 21 | **22** | Label 21 `bearing_wear_gradual` added |
| Group B compound TPR tracking | 5 classes (labels 7–11) | **6 classes (labels 7–12)** | Label 12 added |
| Group C masked TPR tracking | 4 classes (labels 12–15) | **5 classes (labels 13–17)** | Label 17 added; all C labels shifted |
| Group D variant TPR tracking | 3 classes (labels 16–18) | **4 classes (labels 18–21)** | Labels renumbered; label 21 added |
| Fuzzy calibration exclusion | overloading mild + seal mild | **+ label 21 mild (sev 0.05–0.15)** | Sub-threshold by Paris–Erdogan design |
| Detection for label 21 | not covered | **Layer 3 CUSUM + Layer 4 Rolling Baseline** | New detection layers — see Part 1B |
| File structure | 2 files (Part 1 + Part 2) | **3 files (Part 1A + Part 1B + Part 2)** | Size management + modularity |
| Gate count (M8) | 14 gates | **15 gates** (Gate M8-14-ext: label 21 CUSUM) | Label 21 detection gate |
| Upstream feature matrix | ~189,000 × 26 | **~196,000 × 26** | Label 21 adds ~7,000 windows |

---

## v1.0 → v2.0 Change Summary (retained for audit)

| Item | v1.0 (OLD — INVALID) | v2.0 | Reason |
|------|----------------------|------|--------|
| M7 prerequisite gate count | 10 gates | 15 gates (per-group A–E) | M7 updated to 21-class |
| Fault validation pool | M6A 7200 + M6B 1600 compound | M6B_combined_sequences: ~27,000 sequences windowed | M6B expanded dataset |
| Upstream reference | `M6_feature_matrix.csv` 10000×29 | `M6B_feature_matrix.csv` ~189,000×26 | M6.5r output |
| Gate count | 13 gates | 14 gates | Group D/E TPR gate added |

> v1.0 is INVALID. Do not reference v1.0 gate numbers or upstream file names.

---

## M6.5 Audit Findings — Critical Inputs for M8 Design

**Read before writing a single line of M8 code. Every finding directly constrains
threshold calibration, channel weighting, and detection strategy.**

| Finding | Root Cause | M8 Action |
|---------|-----------|----------|
| **F1** Overloading Gate 3 = 0.00% (MAE=0.093) | Temp.SV weight=0.5 → sub-threshold weighted MAE | Mech C Temp.SV drift = PRIMARY detection path. Gate M8-7: ≥80% TPR via Mech C ONLY |
| **F2** Seal failure Gate 3 = 29.17% (MAE=0.196) | Pres.SV gradual decline — per-window MAE low | Mech C Pres.SV drift (negative Spearman) = PRIMARY. Gate M8-9: WATCH ≤20 min |
| **F3** Bearing seam discontinuity 5.75% | Spike seed t=49→50 step change | Attention must NOT peak at seam. Gate M8-8: seam_ratio < 1.0 |
| **F4** Fisher rank 1 = PmpSV_mean | Pmp.SV dominant fault channel | Weight increase 2.0→2.5 Fisher-validated |
| **F5** Cavitation MAE = 0.675 (6.1×) | Hydraulic shock — always acute | Bypass WATCH/WARN → DANGER immediately at startup |
| **F6** Normal probe 86.67% NOT FPR problem | Edge-case probe sampling artifact | Gate M8-2 on full 9711-window pool ONLY — never on 30-window probe |
| **F7** Label 21 MAE sub-threshold by design | Paris–Erdogan low-ΔK regime sev 0.05–0.15 | Do NOT raise threshold. Layer 3 CUSUM + Layer 4 Rolling Baseline are detection mechanisms. |

---

## M8 Architecture — Stage 1

*(Stage 3 — Slow Drift Mechanisms, Detection Map, Training Data → Part 1B)*
*(Stage 4 — Alert Machine + Stage 5 — Cluster Thresholds → Part 2)*

---

### STAGE 1 — LSTM-AE Reconstruction

```
Input    : (batch, 50, 8) — 50-timestep windows, 8 normalized channels
           Window size = 50 (M2 optimal, M6.5r fixed — NEVER change)
Encoder  : LSTM(8→128, layers=2, dropout=0.3)
           → Multi-head temporal attention over encoder outputs
           → Bottleneck(128→64)
Decoder  : LSTM(64→128) → LayerNorm → Output(128→8)
           Hidden state seeded from encoder bottleneck

Loss function (3-component — physics-weighted):
  total_loss = 0.5×MAE + 0.3×MSE + 0.2×grad_penalty

  grad_penalty = mean(|dX_reconstructed/dt − dX_input/dt|)
  Physics basis: penalizes unphysical rate-of-change in reconstruction.
  Critical for cavitation — highly erratic pressure signal.
  Without grad_penalty: model produces smooth output for erratic cavitation input
  → reconstruction error underestimated → cavitation MAE artificially lowered.

Optimizer : AdamW
Scheduler : CosineAnnealingWarmRestarts (T0=20)
AMP       : GradScaler + autocast (CUDA — RTX 4060 Laptop)
MC Dropout: N=20 forward passes at inference → mean_MAE + uncertainty_std
Parameters: ~505,096 (same order as M4 — no architecture bloat)
```

#### Channel Weights — Fisher Validated from M6.5

```
Channel     M4 Weight   M8 Weight   Reason
─────────────────────────────────────────────────────────────────────
Mot.SV      2.0         2.5         Fisher rank 2 confirmed — vibration dominant
Pmp.SV      2.0         2.5         Fisher rank 1 confirmed — HIGHEST discriminability
Pres.SV     2.0         2.5         Primary seal + cavitation channel
Mot.PV      1.5         2.0         Displacement — secondary vibration
Pmp.PV      1.5         2.0         Displacement — BPF harmonics
Temp.SV     1.0         0.5         LOW WEIGHT — but Mech C monitors UNWEIGHTED
Mot.TV      0.8         0.3         Placement-dependent — low weight
Pmp.TV      0.8         0.3         Placement-dependent — low weight

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DESIGN NOTE — WHY Temp.SV WEIGHT IS LOW BUT STILL DETECTABLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temp.SV weight = 0.5 → weighted MAE contribution suppressed.
Overloading raises ONLY thermal channels → weighted MAE stays sub-threshold.
This is EXPECTED and CORRECT behaviour (Finding 1 from M6.5).

Mech C (Stage 3C, see Part 1B) operates on RAW channel reconstruction error,
BYPASSING the weight matrix entirely.
Temp.SV at weight 0.5 retains FULL Mech C monitoring sensitivity.

Without this design: overloading invisible to model.
With this design:    Mech C sees unweighted Temp.SV drift → overloading_early fires.
This is the architectural solution to Gate 3 = 0.00% in M6.5.
```

---

### STAGE 2 — Fuzzy Membership Layer

```
PURPOSE:
  Convert continuous MAE value to fault probability [0, 1].
  Handles the transition zone between clearly normal and clearly fault.
  Captures early-stage faults where MAE hovers near threshold — hard threshold misses.

CALIBRATION PROTOCOL:

  NORMAL POPULATION:
    Full 9711-window real CIRA normal pool [Finding F6 — never use probe subset]
    P95 of normal MAE distribution → lower_bound (fuzzy onset)
    Expected range: [0.07, 0.09]

  FAULT POPULATION — SELECTIVE EXCLUSION (physics-derived):
    EXCLUDE: overloading mild (sev 0.2–0.5)
      Reason: MAE=0.093 sub-threshold [Finding F1]
      Including drags upper_bound toward normal → narrows fuzzy zone → raises FPR
      Handled by Mech C (Temp.SV drift), NOT fuzzy layer
    EXCLUDE: seal_failure mild (sev 0.2–0.4)
      Reason: MAE near 0.12, too close to normal boundary [Finding F2]
      Handled by Mech C (Pres.SV drift), NOT fuzzy layer
    EXCLUDE: bearing_wear_gradual ALL severities (label 21, sev 0.05–0.25)
      Reason: Paris–Erdogan low-ΔK regime — MAE sub-threshold by design [Finding F7]
      Layer 1 fuzzy layer CANNOT detect label 21 — this is CORRECT, NOT a failure
      Handled by Layer 3 CUSUM + Layer 4 Rolling Baseline (see Part 1B)
      Including label 21 sequences drags upper_bound toward normal → raises FPR
    INCLUDE: cavitation ALL severities         (MAE = 0.675)
    INCLUDE: bearing_wear ALL severities       (standard, not gradual)
    INCLUDE: sensor_failure ALL severities
    INCLUDE: impeller_imbalance ALL severities
    INCLUDE: overloading severe (sev 0.5–1.0) only
    INCLUDE: seal_failure severe (sev 0.5–1.0) only
    INCLUDE: ALL Group B compound sequences (M6B, labels 7–12)
    INCLUDE: ALL Group C masked sequences (M6B, labels 13–17)
    INCLUDE: ALL Group D variant sequences EXCEPT label 21 (labels 18–20)
    INCLUDE: ALL Group E multi-sensor sequences (M6B)
    P5 of included fault MAE distribution → upper_bound
    Expected range: [0.15, 0.50] (dominated by cavitation MAE=0.675)

  FUZZY FUNCTION:
    μ_fault(e) = 0.0                                      if e < lower_bound
               = (e − lower_bound) / (upper − lower)      if lower ≤ e ≤ upper
               = 1.0                                      if e > upper_bound

  WHY EXCLUSION IS CRITICAL:
    Including overloading/seal/label-21 mild sequences in fault population
    drags upper_bound DOWN toward normal territory.
    Narrower fuzzy zone → more windows in transition → higher FPR.
    On a 110 kW asset: elevated FPR = operators ignore alerts = missed real faults.
    Excluded sequences are NOT lost — Mech C catches overloading/seal mild;
    Layer 3 CUSUM + Layer 4 Rolling Baseline catch label 21.
    This exclusion is physics-driven, not ad-hoc.

  GROUP B–E SEQUENCES IN FUZZY CALIBRATION:
    Compound (Group B, labels 7–12): both fault channels active → MAE well above threshold
    Masked (Group C, labels 13–17): secondary path only → MAE moderate but above lower_bound
    Variants (Group D, labels 18–20): follow base fault MAE character
    Label 21 (bearing_wear_gradual): EXCLUDED — see above
    Multi-sensor (Group E): 2-channel anomaly → MAE additive → clearly above threshold
    All eligible sequences included in fault population for upper_bound.
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic file — bias-audit updates incorporated |
| v2.0 | 2026-04-15 | Split into Part 1 + Part 2. Part 1 = Architecture. M7 prerequisite 10→15 gates. M6B fault validation pool. Group B–E coverage in Mech C + detection map. |
| v3.0 | 2026-04-16 | **v14.0 UPGRADE + FILE SPLIT**: This file = Part 1A (Stage 1 + Stage 2 only). Stage 3 Mech A/B/C + detection map + training data → new `module_M8_lstm_ae_v2_mechanisms.md` (Part 1B). M7 prerequisite 15→16 gates. 22-class. v2→v3 table added. Finding F7 added for label 21. Fuzzy exclusion: label 21 bearing_wear_gradual ALL severities excluded (Paris–Erdogan sub-threshold by design). Group B 6 classes, Group C 5 classes, Group D labels 18–20 (label 21 excluded from fuzzy). Mech C cross-refs updated to Part 1B. Gate M8-14-ext noted. |

---

*GitHub is the ONLY source of truth for this spec.*
*Part 1B (Mechanisms): `module_M8_lstm_ae_v2_mechanisms.md`*
*Part 2 (Gates + Outputs): `module_M8_lstm_ae_v2_gates_and_outputs.md`*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
