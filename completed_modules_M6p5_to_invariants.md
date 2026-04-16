# PumpSmart — Completed Modules Reference: M6.5, M6.5r, Invariants
# Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring
# PART 2B OF 3 — M6.5 results + 6 audit findings + M6.5r 22-class spec + invariants
#
# Status: M6.5 section LOCKED. M6.5r section = SPEC ONLY — NOT STARTED (blocked until M6B runs).
# Updated: 2026-04-16 | Author: Souvik
# Split from: completed_modules_M5_to_M6p5r.md (v3.0, was too large)
#
# THIS FILE CONTAINS:
#   - M6.5 LSTM-AE feature extractor v2 results (LOCKED — govern M7 + M8 design)
#   - M6.5 6 critical audit findings in FULL detail (LOCKED — DO NOT MODIFY)
#   - M6.5r updated spec for M6B 22-class (⬜ NOT STARTED — blocked until M6B runs)
#   - Cross-module invariants 1–16 (v14.0 — invariant 16 added for slow-fault detection)
#   - File structure (updated for M6B v14.0 — PENDING files clearly marked)
#   - Module progress tracker
#
# COMPANION FILES:
#   completed_modules_context_and_M1_to_M4.md  → Part 1 LOCKED: context, M1–M4 results
#   completed_modules_M5_to_M6p5r.md           → Part 2A: M5, M6A, M6B v14.0 spec
#
# GitHub is the ONLY source of truth. Spaces .md files are OUTDATED — do not use.

---

## ╔══════════════════════════════════════════════════╗
## M6.5 — LSTM-AE FEATURE EXTRACTOR v2 (M6A BRIDGE)
## Status: ✅ COMPLETED v2 (2026-04-11) — LOCKED — results govern M7 + M8 design
## ╚══════════════════════════════════════════════════╝

### Why M6.5 Exists

```
PROBLEM: XGBoost cannot consume raw time-series (shape 200×8).
         Flattening destroys temporal ordering → 1600-dim sparse feature space.
SOLUTION: Run M6A sequences through M4 LSTM-AE (inference only).
          Extract statistical features from reconstruction error array (200×8)
          → one static row per sequence → XGBoost trains on rows.
RESULT: XGBoost sees tabular data but every feature carries temporal
        meaning from LSTM-AE reconstruction behaviour.
```

### v2 Bug Fix (CRITICAL — ALL NUMBERS BELOW ARE FROM v2)
```
BUG IN v1: Gate 3 sliced sequences as [:60] instead of [:50]
           → created 60-step windows fed to M4 LSTM-AE (expects 50) → invalid MAE
FIX IN v2: Corrected to [:50] — matches M4 WINDOW_SIZE=50 (config.py)
IMPACT   : ALL Gate 3 numbers below are from v2 and AUTHORITATIVE.
           v1 Gate 3 numbers are INVALID — discard entirely.
```

### M6.5 Feature Set — 24 Features (LOCKED)

```
Per-channel mean reconstruction error : 8 features
Per-channel max reconstruction error  : 8 features
Temporal evolution features           : 5 features
  error_onset_lag, err_slope_primary, err_auc_primary,
  kurtosis_err_PmpSV, kurtosis_err_PresSV
Cross-channel features                : 2 features
  corr_delta_PmpSV_PresSV, thermal_decoupling_flag
Fuzzy fault membership                : 1 feature (fuzzy_fault_membership)
Label                                 : labels 0–6 (7 classes from M6A)
Output: data/synthetic/M6_feature_matrix.csv — 8400 rows × 25 columns
```

### Gate 3 Results — MAE Threshold Check (v2 AUTHORITATIVE)

| Class | Mean MAE | Gate 3 Pass% | Interpretation |
|---|---|---|---|
| normal | 0.1202 | 86.67% | Probe only — NOT false alarm [Finding 6] |
| bearing_wear | 0.0979 | 13.33% | Mild sev near-threshold — expected |
| impeller_imbalance | 0.1031 | 30.00% | Mild sequences dominate |
| cavitation | 0.6747 | 100.00% | ✅ MAE=0.675, 6.1× threshold |
| seal_failure | 0.1961 | 29.17% | Slow hydraulic — Mech C PRIMARY |
| overloading | 0.0930 | 0.00% | Thermal-dominant — Mech C PRIMARY |
| sensor_failure | 0.1696 | 93.33% | ✅ Flatline clearly anomalous |

### Top 5 Fisher Discriminant Features (LOCKED)

| Rank | Feature | Physics Validation |
|---|---|---|
| 1 | Pmp_SV_mean | Pump vibration dominant ✅ |
| 2 | Pmp_SV_std | Vibration variance ✅ |
| 3 | Temp_SV_mean | Thermal drift — overloading discriminator ✅ |
| 4 | Mot_TV_mean | Motor temp — bearing/overloading ✅ |
| 5 | Mot_TV_std | Temperature variance ✅ |

### M6.5 Outputs (LOCKED)
```
data/synthetic/M6_feature_matrix.csv        ← 8400 rows × 25 columns (M6A, 7-class)
src/module_065_sequence_audit.py            ← v2 (Gate 3 :60→:50 fix applied)
outputs/reports/module_065_sequence_audit_report.md
```

### M6.5 Paste Text Keys (LOCKED)
```
M6p5_feature_matrix_rows        : 8400
M6p5_features_per_row           : 24 + label = 25 columns
M6p5_gate3_normal_probe         : 86.67% (probe only — NOT FPR)
M6p5_gate3_cavitation           : 100.00% (MAE=0.675, 6.1×)
M6p5_gate3_overloading          : 0.00% (thermal-dominant — Mech C)
M6p5_gate3_seal_failure         : 29.17% (slow fault — Mech C)
M6p5_top_fisher_feature         : Pmp_SV_mean (rank 1)
M6p5_window_fix                 : v2 corrected :60→:50
M6p5_seal_patch                 : 165→220 sequences accepted
```

---

## ⚠️ M6.5 AUDIT — 6 CRITICAL FINDINGS (Govern M7 and M8 Design) ⚠️
## THESE ARE LOCKED — DO NOT MODIFY

### FINDING 1 — OVERLOADING IS THERMAL-DOMINANT (Gate 3 pass = 0.00%)
```
Observed : mean MAE = 0.093 — BELOW threshold 0.110058
Root cause: M4 Temp.SV weight=1.0, Mot.TV weight=0.8 (lowest weights).
            Overloading raises ONLY thermal channels — weighted MAE sub-threshold.

M7 implication : XGBoost classifies correctly via mean_err_TempSV (Fisher rank 3).
M8 MANDATORY  : Mech C Temp.SV Spearman drift > 0.70 = PRIMARY detection.
                Gate M8-7: overloading TPR ≥ 80% via Mech C ONLY.
                Do NOT measure via single-window MAE threshold crossing.
DO NOT raise global threshold to compensate.
```

### FINDING 2 — SEAL FAILURE IS A SLOW HYDRAULIC FAULT (Gate 3 pass = 29.17%)
```
Observed : mean MAE = 0.1961 — above threshold ON AVERAGE but 29.17% of
           individual 50-step windows cross threshold.
Root cause: Pres.SV* decline very gradual. Single window shows only small drop.

Seal patch applied (M6.5 v2):
  Original: only 165/1200 sequences exceeded MAE threshold
  Fix: severity distribution rebalanced toward [0.4, 0.7] band
  Final accepted: 220 sequences (padded to 1200 with physics variants)

M8 MANDATORY : Pres.SV Spearman drift (NEGATIVE) > 0.70 over 300 windows
               = seal_failure_early flag (PRIMARY).
               Gate M8-9: WATCH fires ≤ 20 min of onset.
               Gate M8-10: Pres.SV drift fires BEFORE total MAE reaches WARN.
DO NOT raise global threshold to accommodate seal_failure.
```

### FINDING 3 — BEARING WEAR TEMPORAL COHERENCE = 94.25% (69 flagged sequences)
```
Observed : 69 sequences have dX/dt discontinuity at seam (t=49→50)
           between spike seed onset and M5 physics continuation.
Decision : Sequences KEPT — represent realistic mechanical shock events.
M8 implication: Monitor attention heatmap — peaks at fault onset, not seam.
               Gate M8-8: seam_ratio = mean_attention(t=49,50) / mean_attention(t=10,40)
               Gate: seam_ratio < 1.0 (fault onset dominates over seam artifact).
M12 implication: Config 1–3 adversarial must use SMOOTH sequences only.
```

### FINDING 4 — TOP FISHER FEATURES CONFIRM M8 CHANNEL WEIGHT DIRECTION
```
Fisher rank 1: Pmp_SV_mean  → confirms M8 weight Pmp.SV 2.0→2.5 ✅
Fisher rank 2: Pmp_SV_std   → variance is discriminator ✅
Fisher rank 3: Temp_SV_mean → low M4 weight but survives → M7 uses for overloading
Fisher rank 4: Mot_TV_mean  → M4 weight 0.8 but signal survives ✅
Fisher rank 5: Mot_TV_std   → temperature variance discriminative ✅

M8 weight decision VALIDATED:
  Increase: Mot.SV=2.5, Pmp.SV=2.5, Pres.SV=2.5 (from 2.0)
  Increase: Mot.PV=2.0, Pmp.PV=2.0 (from 1.5)
  Decrease: Temp.SV=0.5, Mot.TV=0.3, Pmp.TV=0.3
  Mech C monitors Temp.SV UNWEIGHTED via raw channel error — NOT weight matrix.
```

### FINDING 5 — CAVITATION STRONGLY ANOMALOUS (Gate 3 = 100%, MAE = 0.675)
```
Observed : MAE = 0.675 — 6.1× above threshold. Every window crosses.
M8 implication: Cavitation → DANGER immediately (skip WATCH/WARN).
               Gate M8-12: ZERO cavitation DANGER outside startup cluster.
REPORT SEPARATELY: Do NOT lump cavitation with overloading in TPR.
  If overloading TPR=50%, cavitation TPR=100% → overall=75% hides the gap.
  Gate M8-7 (overloading) measured independently.
```

### FINDING 6 — NORMAL PROBE 86.67% = NOT A FALSE ALARM PROBLEM
```
Observed : 86.67% of 30 probed normal windows crossed MAE threshold.
Why NOT a problem:
  30-window probe deliberately samples near-boundary edge cases.
  Full M4 val set (1457 windows) → 0.55% FPR (8/1457) — confirmed.
M8 action:
  Do NOT adjust threshold based on probe.
  Gate M8-2 (FPR < 5%) measured on FULL 9711-window pool ONLY.
  Cluster-conditional thresholds handle remaining boundary cases.
```

---

## ╔══════════════════════════════════════════════════╗
## M6.5r — UPDATED FEATURE EXTRACTOR FOR M6B (22-CLASS)  ← v14.0
## Status: ⬜ NOT STARTED — BLOCKED until M6B script runs successfully
## ╚══════════════════════════════════════════════════╝

> ⚠️ M6.5r cannot start until M6B_combined_sequences.pkl and fault_rules_v3.json exist.
> This section is the LOCKED SPEC that governs the M6.5r script.

### Why M6.5r Was Created

```
M6.5 (original): processed M6A only — 7 classes, 24 features, 8400 rows.
M6B adds Groups B, C, D, E → 22 classes, ~26,000–28,000 sequences.
M6.5r runs M6B_combined_sequences through M4 LSTM-AE and extracts
an EXPANDED 25-feature set (26 columns including label).

New features added for Groups B–E:
  masked_channel_flag       : 1 if primary detection channel = constant (Group C)
  secondary_onset_lag       : steps until secondary fault channel activates (Group B)
  burst_count               : number of MAE spikes in 200 steps (Group D label 18)
  cyclic_baseline_drift     : Temp.SV baseline slope (Group D label 20)
  multi_sensor_anomaly_count: 0/1/2 — how many channels simultaneously anomalous (Group E)
  fault_group_id            : {0:normal,1:single,2:compound,3:masked,4:variant,5:multi_sensor}
                              — M7 group-level regularizer (derived from metadata, not label)

NOTE on err_slope_* features (label 21 bearing_wear_gradual):
  err_slope_MotSV, err_slope_TempSV, err_slope_PresSV already exist in Domain 2.
  These are computed as linear regression slope over each 50-step window.
  They ARE the primary discriminator for label 21 — no new column needed.
  err_slope_MotSV > 0 consistently across windows = slowly progressing bearing fault.
```

### M6.5r Feature Set — 25 Features + Label (26 columns TOTAL)

```
Inherited from M6.5 (24 features):
  Per-channel mean reconstruction error   : 8 features
  Per-channel max reconstruction error    : 8 features
  Temporal evolution                      : 5 features
    error_onset_lag, err_slope_primary, err_auc_primary,
    kurtosis_err_PmpSV, kurtosis_err_PresSV
  Cross-channel                           : 2 features
    corr_delta_PmpSV_PresSV, thermal_decoupling_flag
  Fuzzy fault membership                  : 1 feature

New features for M6B Groups B–E (1 net additional — 25 total unique + label):
  masked_channel_flag         : bool ← Group C primary detection channel absent
  secondary_onset_lag         : int  ← Group B secondary fault onset step
  burst_count                 : int  ← Group D label 18 cavitation_intermittent burst count
  cyclic_baseline_drift       : float ← Group D label 20 overloading_cyclic baseline slope
  multi_sensor_anomaly_count  : int  ← Group E number of simultaneously anomalous channels
  fault_group_id              : int  ← {0–5} group regularizer for M7 tree splitting

NOTE: fault_group_id is set from fault_rules_v3.json group metadata field.
      It does NOT leak label information — group is broader than label.
      If M7 SHAP top-1 = fault_group_id for ANY class → FAIL (label leakage).

CUSUM features (cusum_MotSV, cusum_PresSV, cusum_TempSV):
  NOT in feature matrix — rejected (train-serve skew, see 4-layer architecture in Part 2A).
  CUSUM lives in M10 runtime only.

Label: 0–21 (22 classes, fault_rules_v3.json)
Total columns: 26 (25 features + label_int)

Output: M6B_feature_matrix.csv — ~196,000 rows × 26 columns → M7 input
```

### M6.5r Validation Gates

| Gate | Check | Target | Physics Basis |
|---|---|---|---|
| W1 | Group A Gate 3 MAE pass rates match M6.5 (locked) | Within 2% | Consistency check |
| W2 | All Group B sequences: both fault channels elevated | 100% | Compound causal physics |
| W3 | All Group B: MAE > 0.110058 in Phase 2 | ≥90% | Both channels active = high MAE |
| W4 | Group C: masked channel std < 0.001 confirmed | 100% | Flatline verification |
| W5 | Group C: secondary channel detectable (MAE>0 on secondary) | ≥95% | Secondary path exists |
| W6 | Group D label 18: burst_count ≥ 2 per sequence | 100% | Intermittent by definition |
| W7 | Group D label 19: Pres.SV slope > 3× Group A seal_failure | 100% | "Fast" validated |
| W8 | Group D label 20: cyclic_baseline_drift > 0 | 100% | Rising baseline confirmed |
| W9 | Group E: multi_sensor_anomaly_count = 2 | 100% | Dual degradation |
| W10 | secondary_onset_lag in valid range for all Group B | 100% | Physics plausible range |
| W11 | Group D label 21: err_slope_MotSV > 0 in ≥95% of sequences | 95% | Gradual bearing drift |

### M6.5r Planned Outputs (⚠️ NONE EXIST YET — written when M6.5r script runs)
```
data/synthetic/M6B_feature_matrix.csv     ← ~196,000 rows × 26 cols → M7 input
src/module_065r_feature_extractor.py      ← extended feature extraction script
outputs/reports/module_065r_feature_extractor_report.md
```

### M6.5r Paste Text Keys (⚠️ Populate AFTER script runs — do not fill in advance)
```
M6p5r_rows                    : [fill after run — target ~196,000]
M6p5r_cols                    : 26 (25 features + label)
M6p5r_labels                  : 22 (0–21, fault_rules_v3.json)
M6p5r_gate_W1_pass            : [PASS/FAIL]
M6p5r_gate_W2_pass            : [PASS/FAIL]
M6p5r_gate_W3_pass            : [PASS/FAIL]
M6p5r_gate_W4_pass            : [PASS/FAIL]
M6p5r_gate_W5_pass            : [PASS/FAIL]
M6p5r_gate_W6_to_W10_pass     : [PASS/FAIL]
M6p5r_gate_W11_label21_slope  : [PASS/FAIL — err_slope_MotSV > 0 in ≥95% label 21 seqs]
M6p5r_masked_channel_pct      : [% Group C rows with masked_channel_flag=1]
M6p5r_mean_secondary_lag      : [mean secondary_onset_lag for Group B]
M6p5r_mean_burst_count        : [mean burst_count for Group D label 18]
M6p5r_multi_sensor_count_2    : [% Group E rows with multi_sensor_anomaly_count=2]
M6p5r_label21_slope_pass_pct  : [% label 21 sequences with err_slope_MotSV > 0]
Status_for_M7                 : PENDING — populate after M6.5r script runs successfully
```

---

## CROSS-MODULE INVARIANTS (Enforced M1 → M12 — ALL 16)

```
1.  segment_id preserved in ALL dataframes through M6B
2.  Windows NEVER cross segment boundaries
3.  Normalization baselines LOCKED at M3_normalization_config.json
4.  Winsor ceilings LOCKED at M4_spike_config.json (M6B + M12 read, do not override)
5.  M4 threshold=0.110058 is the fault/normal boundary for M6B validation gate
6.  M8 cluster-conditional thresholds are the production boundary for M12
7.  Physical couplings (r>0.87) must hold in ALL synthetic sequences (M6B + M12)
8.  Conservation of energy + mass in all synthetic sequences (Groups A–E)
9.  Household pump → physics_advisory_only() always — no ML inference
10. XGBoost: device="cuda" train | device="cpu" deploy
11. All models: torch.save(state_dict) | torch.load(map_location="cpu")
12. M7 trains on M6.5r M6B_feature_matrix.csv (26 cols, 22-class) — NOT M6.5 M6A matrix (25 cols)
13. Fuzzy logic is a core M8 detection component — not just M10 display
14. M12 MUST pass PRODUCTION_VALIDATED before deployment on 110 kW asset
15. M8 MUST detect slow drift faults (sev 0.2–0.3) via trend accumulator
    within 20 minutes of fault onset — this is the Category 3 liability gate.
    M6B severity 0.2–0.3 sequences are the TRAINING DATA for this requirement.
16. bearing_wear_gradual (label 21) requires all 4 detection layers to be useful.  ← NEW v14.0
    LSTM-AE Layer 1 alone is INSUFFICIENT for label 21 — MAE < threshold for most of duration.
    CUSUM (Layer 3) + Rolling Baseline (Layer 4) are MANDATORY in M10 for label 21 coverage.
    M8 gate M8-gradual_detection_week: system must detect label 21 within ~6 weeks of onset.
    XGBoost output for label 21: "bearing_wear_gradual — plan inspection within 7–14 days"
```

---

## FILE STRUCTURE (v14.0 — M6B outputs marked PENDING — do not exist yet)

```
PumpSmart_Project/
├── config.py
├── data/
│   ├── raw/                           ← 9 original CSVs (never modified)
│   ├── clean/                         ← M1 output
│   ├── normalized/                    ← M3 output
│   └── synthetic/                     ← M4 seeds + M6A archive + M6B PENDING
│       ├── M4_spike_seeds.npy         ← shape=(1044, 50, 8) — LOCKED ✅ EXISTS
│       ├── M4_spike_seeds_meta.csv    ← ✅ EXISTS
│       ├── M4_spike_config.json       ← LOCKED winsor bounds ✅ EXISTS
│       ├── M6A_sequences.pkl          ← 8400 seq (archived, superseded) ✅ EXISTS
│       ├── M6A_sequence_meta.csv      ← archived ✅ EXISTS
│       ├── M6_feature_matrix.csv      ← M6.5 output (M6A, 8400×25) ✅ EXISTS
│       ├── M6B_sequences_groupA.pkl   ← ⏳ PENDING — created by M6B script (locked from M6A)
│       ├── M6B_sequences_groupB.pkl   ← ⏳ PENDING — created by M6B script (~7,200)
│       ├── M6B_sequences_groupC.pkl   ← ⏳ PENDING — created by M6B script (~4,000)
│       ├── M6B_sequences_groupD.pkl   ← ⏳ PENDING — created by M6B script (~2,800 incl. label 21)
│       ├── M6B_sequences_groupE.pkl   ← ⏳ PENDING — created by M6B script (~800)
│       ├── M6B_combined_sequences.pkl ← ⏳ PENDING — ALL groups merged → M8 fault validation pool
│       ├── M6B_sequence_meta.csv      ← ⏳ PENDING — seq_id, label, group, severity, cluster, source
│       └── M6B_feature_matrix.csv     ← ⏳ PENDING — created by M6.5r script (~196,000 × 26)
├── models/
│   ├── lstm_ae_baseline_best.pth      ← M4 model (LOCKED) ✅ EXISTS
│   ├── M3_normalization_config.json   ← LOCKED baselines ✅ EXISTS
│   ├── M4_threshold_config.json       ← threshold=0.110058 (LOCKED) ✅ EXISTS
│   ├── fault_rules.json               ← M5 original 6-class (LOCKED — archived) ✅ EXISTS
│   ├── fault_rules_v3.json            ← ⏳ PENDING — written by M6B Step 3 (22-class, labels 0–21)
│   ├── M5_physics_config.json         ← ✅ EXISTS
│   └── unit_registry.json             ← ✅ EXISTS
├── outputs/
│   ├── reports/                       ← one .md per module
│   └── plots/                         ← one set of plots per module
├── src/                               ← module_01 through module_12 scripts
└── app/                               ← Flask web app (M10)
```

---

## MODULE PROGRESS TRACKER (v14.0)

```
M1    Data Ingestion & Cleaning          : ✅ COMPLETED (2026-03-25)
M2    EDA + Operating Mode Clustering    : ✅ COMPLETED (2026-03-26)
M3    Dimensionless Normalization        : ✅ COMPLETED (2026-03-28)
M4    LSTM-AE Baseline (v8)              : ✅ COMPLETED (2026-03-28)
M5    Physics Engine                     : ✅ COMPLETED (2026-03-29)
M6A   Synthetic Dataset (7-class)        : ✅ COMPLETED (2026-04-11) — SUPERSEDED by M6B
M6.5  LSTM-AE Feature Extractor v2      : ✅ COMPLETED (2026-04-11) — LOCKED
M6B   Expanded Synthetic (22-class)     : 🔴 NEXT ACTIVE — spec locked (v14.0), script not yet run
M6.5r Updated Feature Extractor (M6B)   : ⬜ NOT STARTED — blocked until M6B completes
M7    XGBoost Fault Classifier (22-class): ⬜ NOT STARTED — awaits M6B + M6.5r output files
M8    LSTM-AE v2 + 4-Layer Detection    : ⬜ NOT STARTED — see module_M8_lstm_ae_v2_architecture.md
M9    Pump Selector + Household Advisor  : ⬜ NOT STARTED
M10   Flask Web App + CUSUM + Baseline  : ⬜ NOT STARTED
M11   Docker + Hugging Face Deployment   : ⬜ NOT STARTED
M12   Physics-Governed Validation Suite  : ⬜ NOT STARTED (post-M11)
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Content originally part of monolithic `completed_modules_M1_to_M6p5.md` |
| v2.0 | 2026-04-15 | Split into Part 1 + Part 2. M6.5r 26-feature spec, 15 invariants added |
| v3.0 | 2026-04-15 | M6.5r status corrected. Gate W10 added. fault_group_id leakage note added |
| v4.0 | 2026-04-16 | **v14.0 UPGRADE**: Split into standalone Part 2B (this file). M6.5r updated to 22-class (~196,000 rows). Gate W11 added for label 21 err_slope_MotSV. Invariant 16 added for bearing_wear_gradual 4-layer requirement. CUSUM explicitly excluded from feature matrix with reasoning. M6.5r paste keys updated with label21 slope gate. Progress tracker updated: M6B=22-class, M7=22-class, M8=4-layer, M10=CUSUM+baseline. |

---

*GitHub is the ONLY source of truth for this spec.*
*Companion Part 1: `completed_modules_context_and_M1_to_M4.md` (LOCKED — context + M1–M4)*
*Companion Part 2A: `completed_modules_M5_to_M6p5r.md` (M5 + M6A + M6B v14.0 spec)*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration, ISO 13373-3 condition monitoring*
