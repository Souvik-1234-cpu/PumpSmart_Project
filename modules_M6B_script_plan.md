# PumpSmart — Module M6B: Script Plan + API Design
## Part 2 of 2 — Step 0 + 3-Step Script Plan, Dispatcher, Pre-Flight, API Spec, Paste Keys

| Field | Value |
|-------|-------|
| **Document version** | v3.0 — v14.2 physics audit + Step 0 + z_t export + corrected counts |
| **Date** | 2026-04-19 |
| **Split from** | `modules_M6B_synthetic_expanded.md` (Part 1 — fault universe, physics rules, gates) |
| **Canonical source of truth** | `pasted-text.txt` (v14.2, 2026-04-19) |

> **NOTE:** Read `modules_M6B_synthetic_expanded.md` (Part 1) BEFORE this file.
> - Part 1 = fault universe, physics rules, group tables, dataset targets, gates
> - Part 2 = Step 0 + 3-step script plan, dispatcher, pre-flight, API spec, paste keys *(this file)*

> **EXECUTION STATUS:**
> - M6B Step 0 v2  = COMPLETE LOCKED 2026-04-26 (Labels 1,4,5 | 4,500 seqs | 21/21 gates)
> - M6B Step 0b v2 = COMPLETE LOCKED 2026-04-26 (Labels 0,2,3,6 | 6,200 seqs | 20/20 gates)
> - M6B Step 1     = NEXT ACTIVE — Group B compound chains (Labels 7-12)
> - M6B Steps 2-3  = PENDING (blocked on Step 1)
> - Physics lib    = src/m6b_physics_lib.py LOCKED v1.0
> - M6.5r = NOT STARTED — blocked until M6B completes
> - M7 = NOT STARTED — blocked until M6.5r completes
> - M8 = NOT STARTED — TCN-AE architecture locked (v14.2)
>
> No output files exist yet. `fault_rules_v3.json`, all `M6B_*.pkl`, `z_t_sequences_group*.pkl`, `M6B_sequence_meta.csv`, `M6B_feature_matrix.csv` are ALL pending — created when scripts run.

---

## 4-Step Sequential Script Plan

> **Step 0 is NEW in v3.0**

| Property | Value |
|----------|-------|
| **Script filename** | `module_06B_synthetic_generator.py` (~2,200–2,600 lines total) |
| **Pattern** | Each step appended to the same script file — identical to M6A approach |
| **Execution** | Run Step 0 → paste output → Step 1 written → repeat |

### VRAM/RAM Budget (v14.2 corrected sequence lengths)

| Item | Size |
|------|------|
| Longest sequence (Label 10) | 900 steps × 8 × float32 = 28.8 KB per sequence |
| Label 10 pool | 1,500 × 900 × 8 × float32 = ~43.2 MB |
| Full combined pool | ~31,800 × avg 400 steps × 8 × float32 = ~407 MB RAM |
| M4 sliding window z_t export | batch_size=32, VRAM ~2 GB — within 8 GB RTX 4060 |
| Label 21 z_t export | 2,000 × 20 windows × 64 × float32 = ~10.2 MB |
| All z_t exports combined | ~196,000 windows × 64 × float32 = ~50 MB |

---

## Step 0 — Group A Labels 1, 4, 5 Rerun (~400 lines)

### Why Step 0 Exists

M6A generated Labels 1, 4, 5 at 200 steps (blanket default). Physics audit (v14.2) found these are **physically wrong:**

| Label | Class | Old Steps | Correct Steps | Physics Reason |
|-------|-------|-----------|--------------|----------------|
| 1 | `bearing_wear` | 200 | **250** | Paris law rise requires 150s + margins |
| 4 | `seal_failure` | 200 | **400** | NPSHa-safe orifice leak at 40 bar needs 300s + margins |
| 5 | `overloading` | 200 | **300** | Thermal runaway needs 200s + margins |

Step 0 regenerates ONLY these 3 labels. All others (0, 2, 3, 6) carried forward from M6A unchanged.

### Covers

- Script header: imports, config, logging, results dict (mandatory architecture)
- Load M6A pool: `data/synthetic/M6_sequences.pkl` (LOCKED — source seeds only)
- **Re-generate Label 1 at 250 steps:**
  - Same CIRA bearing spike seeds (44 seeds from `M4_spike_seeds.npy`)
  - Paris law: `da/dN = C * dK^m`, Mot.SV\* rise rate calibrated to 250-step envelope
  - Phase: 50s pre-fault baseline + 150s onset + 50s post-detection
- **Re-generate Label 4 at 400 steps:**
  - Same Pres.SV declining-rate from M5 orifice equation: `Q_leak = Cd*A*sqrt(2*dP/rho)`
  - Phase: 50s pre-fault + 300s gradual pressure decline + 50s post-detection
- **Re-generate Label 5 at 300 steps:**
  - Thermal runaway: `Cp*m*dT/dt = Q_friction - Q_ambient`
  - Phase: 50s pre-fault + 200s Temp.SV\* rise + 50s detection window
- Run M6A physics gates G1–G7 on re-generated sequences to confirm validity
- z_t export for new Labels 1, 4, 5:
  - Run M4 `lstm_ae_baseline.pth` (FROZEN) in sliding window mode
  - Per 50-step window: z_t in ℝ⁶⁴ + MAE in ℝ⁸
  - Export: `z_t_sequences_groupA_faults_rerun.pkl`
- Intermediate save: `data/synthetic/M6B_sequences_groupA_rerun.pkl` (contains ONLY new Labels 1, 4, 5 at corrected lengths)
- Print physics gate summary for rerun labels
- **DO NOT overwrite `M6_sequences.pkl` (LOCKED M6A archive)**

### Output After Step 0

```
data/synthetic/M6B_sequences_groupA_rerun.pkl  ← Labels 1, 4, 5 at corrected lengths
z_t_sequences_groupA_faults_rerun.pkl          ← z_t for rerun labels
Gate G1-G7 pass rates for Labels 1, 4, 5       (target: all PASS)
Label 1 Mot.SV* profile plot (250 steps — Paris law rise visible)
Label 4 Pres.SV* profile plot (400 steps — gradual orifice decline)
Label 5 Temp.SV* profile plot (300 steps — thermal runaway)
```

---

## Step 1 — Group B: Compound Chain Faults (~600 lines appended)

### Covers

- `fault_rules_v3.json` config dict loaded in-memory (JSON not written until Step 3 — **do NOT write earlier**)
- Physics gates G8 (temporal ordering), G9 (compound MAE both above threshold)
- M6A sequence pool loader (`data/synthetic/M6_sequences.pkl`) — seed source
- **Compound chain generator:**

```python
generate_compound_sequence(primary, secondary, lag_steps, severity, cluster)
  Phase 1: primary fault only (t=0 to t=50+lag)
  Phase 2: both faults active simultaneously (t=50+lag to end)
  secondary_onset_lag drawn from PHYSICS-VERIFIED per-label range
  Temporal seam continuity enforcement (same as M6A spike seed seam)
  Variable-length sequences per label (NEW v14.2 — NOT blanket 200 steps)
```

- **6 compound classes generated (labels 7–12), counts 1,500/class:**

| Label | Class | Steps | Lag Range |
|-------|-------|-------|-----------|
| 7 | `bearing_wear -> overloading` | 600 | 200–400 |
| 8 | `cavitation -> seal_failure` | 550 | 50–150 |
| 9 | `impeller_imbalance -> bearing_wear` | 700 | 300–600 |
| 10 | `seal_failure -> cavitation` | 900 | 400–800 ⚠️ CRITICAL |
| 11 | `overloading -> bearing_wear` | 800 | 400–600 |
| 12 | `impeller_imbalance -> cavitation` | 450 | 100–300 |

- `secondary_onset_lag` stored per sequence in metadata dict for Step 3 merge
- `score_C` column placeholder in metadata (filled by M8 — not M6B)
- Sanity plots: temporal ordering + compound MAE distribution per class
- Intermediate save: `data/synthetic/M6B_sequences_groupB.pkl`
- **z_t export:** Run M4 `lstm_ae_baseline.pth` (FROZEN) sliding window on every Group B sequence → export `z_t_sequences_groupB.pkl`. Shape per sequence: (N_windows × 64) where N_windows = steps/50

### Output After Step 1

```
data/synthetic/M6B_sequences_groupB.pkl    ← 6 classes × 1,500 = 9,000 sequences
z_t_sequences_groupB.pkl                   ← z_t for all Group B sequences
Gate G8 temporal ordering pass rates       (target: >=95% per class)
Gate G9 compound MAE pass rates            (target: >=90% per class)
Compound chain physics validation plots
```

> **VRAM Note (Label 10 = 900 steps):** Label 10 is longest sequence. z_t windows per sequence = 900/50 = 18 windows. z_t export for Label 10: 1,500 seq × 18 windows × 64 = 1.73M floats = ~6.6 MB. Fits comfortably in 8 GB VRAM at batch_size=32.

---

## Step 2 — Groups C and D: Masked Faults + Severity Variants (~700 lines appended)

### Covers

**Masked fault generator:**

```python
generate_masked_sequence(base_fault, masked_channel, severity, cluster)
  Physics: base fault runs normally; masked_channel replaced with
  flatline/dropout at onset_step=50
  masked_channel_flag = True set in sequence metadata
```

**5 masked classes (labels 13–17), counts 1,200/class:**

| Label | Class | Steps | Note |
|-------|-------|-------|------|
| 13 | `bearing_wear + Mot.SV flatline` | 300 | — |
| 14 | `cavitation + Pres.SV dropout` | 210 | — |
| 15 | `seal_failure + Pres.SV drifting` | 500 | — |
| 16 | `overloading + Temp.SV stuck` | 350 | — |
| 17 | `impeller_imbalance + Pmp.SV flatline` | 250 | Secondary = Pmp.PV only |

**Severity variant generator:**

```python
generate_variant_sequence(base_fault, variant_type, severity, cluster)
  variant_type in {intermittent, fast, cyclic, gradual}
```

**Physics per variant:**

| Variant | Physics |
|---------|---------|
| `intermittent` | NPSHa oscillates above/below NPSHr. Pmp.SV spike ON/OFF; `burst_interval = Uniform(15,30)` |
| `fast` | Turbulent orifice discharge through enlarged effective seal leak area. Pres.SV drops in ≤20 steps. `Q_leak = Cd * A_orifice * sqrt(2*dP/rho)` — **NOT Hagen-Poiseuille** (blowout ≠ laminar) |
| `cyclic` | Duty-cycle load variation. Temp.SV sawtooth with RISING baseline each cycle. Mech B slope detection: `baseline_drift > 0.0002/window`. Spearman > 0.70 on baseline-detrended signal. |
| `gradual` | Paris-Erdogan low dK regime: `da/dN = C*dK^m`. Weibull beta=1.5, severity=0.05–0.25. CIRA anchor: same 44 bearing seeds as Label 1. Mot.SV\* rise rate ~0.0002/step. `err_slope_MotSV > 0` required in ≥95% seqs (Gate G11-ext). Sequences at sev <0.15: MAE <0.110058 — **PHYSICALLY CORRECT**. |

**4 variant classes (labels 18–21):**

| Label | Class | Steps | Sequences |
|-------|-------|-------|-----------|
| 18 | `cavitation_intermittent` | 300 | 1,200 |
| 19 | `seal_failure_fast` | 150 | 800 |
| 20 | `overloading_cyclic` | 600 | 1,200 |
| 21 | `bearing_wear_gradual` | 1,000 | **2,000** — highest count, hardest class |

> **Label 21 sub-threshold behaviour:** CUSUM Layer 3 on `score_B` + Rolling Baseline Layer 4 on `score_A` handle pre-threshold detection — NOT raw MAE. (Invariant 19: `score_B → CUSUM only`, `score_A → Rolling Baseline only`.)

**Validation gates:**
- Gate G10: masked secondary signal ≥50% of base fault MAE (Group C)
- Gate G11-ext: `err_slope_MotSV > 0` in ≥95% of Label 21 sequences

**Intermediate saves:**
```
data/synthetic/M6B_sequences_groupC.pkl
data/synthetic/M6B_sequences_groupD.pkl
```

**z_t export:** Run M4 `lstm_ae_baseline.pth` (FROZEN) sliding window on every sequence. Export: `z_t_sequences_groupC.pkl`, `z_t_sequences_groupD.pkl`. Label 21 z_t: 2,000 seqs × 20 windows × 64 = ~10.2 MB.

### Output After Step 2

```
data/synthetic/M6B_sequences_groupC.pkl    ← 5 classes × 1,200 = 6,000 sequences
data/synthetic/M6B_sequences_groupD.pkl    ← 1200+800+1200+2000 = 5,200 sequences
z_t_sequences_groupC.pkl
z_t_sequences_groupD.pkl
Gate G10 masked secondary signal pass rates  (target: >=50% per class)
Gate G11-ext: err_slope_MotSV > 0 in >=95% Label 21 seqs  (target: PASS)
Label 21 slope distribution plot (err_slope_MotSV histogram)
Masked fault secondary-signal validation plots
```

---

## Step 3 — Group E, Full Merge, z_t Final Export, Validation, Report (~700 lines appended)

### Covers

**Multi-sensor failure generator:**

```python
generate_multi_sensor_failure(failed_channels, failure_type, severity, cluster)
  Physics: 2 channels simultaneously anomalous; 6 others stay within +/-0.20
  Gate G11: exactly 2 channels anomalous — no mechanical fault in remaining 6
  multi_sensor_anomaly_count = 2 set in metadata
```

**2 multi-sensor classes (800 seqs each), label integers from `fault_rules_v3.json`:**

| Class | Failed Channels | Physics |
|-------|----------------|---------|
| `sensor_failure_2ch_thermal` | Mot.TV + Temp.SV | Common thermal power rail |
| `sensor_failure_2ch_pump` | Pmp.SV + Pmp.PV | Pump-side junction box / shared conduit moisture ingress |

> **NOTE:** Group E exact label integers assigned in `fault_rules_v3.json`. **DO NOT hardcode Group E integers here.**

Individual Group E save: `data/synthetic/M6B_sequences_groupE.pkl`. z_t export: `z_t_sequences_groupE.pkl`.

### Full Merge (in this exact order)

```
1. Load M6A carried-forward: Labels 0, 2, 3, 6 from data/synthetic/M6_sequences.pkl
   (normal: 2,000 seqs; Labels 2,3,6: 1,200+1,500+1,200 = 3,900 seqs)
2. Load Step 0 rerun:  M6B_sequences_groupA_rerun.pkl
   (Labels 1,4,5: 1,500+1,500+1,500 = 4,500 seqs)
3. Load Step 1 output: M6B_sequences_groupB.pkl (Labels 7-12: 9,000 seqs)
4. Load Step 2 output: M6B_sequences_groupC.pkl (Labels 13-17: 6,000 seqs)
5. Load Step 2 output: M6B_sequences_groupD.pkl (Labels 18-21: 5,200 seqs)
6. Load Step 3 output: M6B_sequences_groupE.pkl (Group E: 1,600 seqs)
Concatenate → data/synthetic/M6B_combined_sequences.pkl (~31,800 seqs, 22 classes)
```

### z_t Final Combined Export

```
z_t_sequences_groupA_normal.pkl          ← Label 0 normal windows
z_t_sequences_groupA_faults.pkl          ← Labels 2,3,6 carried from M6A
z_t_sequences_groupA_faults_rerun.pkl    ← Labels 1,4,5 Step 0 rerun
z_t_sequences_groupB.pkl                 ← Group B (Step 1)
z_t_sequences_groupC.pkl                 ← Group C (Step 2)
z_t_sequences_groupD.pkl                 ← Group D (Step 2)
z_t_sequences_groupE.pkl                 ← Group E (Step 3)
```

> These z_t files feed Level 2 TCN-AE (M8). **Raw sensor data NEVER enters Level 2.** (Invariant 16 — NEVER VIOLATE)

### M6B_sequence_meta.csv Columns

```
seq_id, label, label_name, group, group_id, severity, cluster, source,
masked_channel_flag, masked_channel, secondary_onset_lag,
err_slope_MotSV, n_steps, n_windows_z_t, physics_context_str
```

> **`physics_context_str` column:** Plain-language fault description per sequence. Seeds M10 `/api/physics_context` lookup table. Format per `fault_rules_v3.json`. Example: *"bearing_wear: Mot.SV rises due to Paris law fatigue crack growth. Risk: impeller contact within 7–14 days if unaddressed."*

### Full Validation Suite

- Physics coupling fidelity: Mot.TV ↔ Temp.SV r≥0.87 in all thermal faults
- Conservation check: all values in physically valid normalized ranges
- MAE distribution plot: all 22 classes vs threshold 0.110058
- Label 21 sub-threshold confirmation: ≥60% Label 21 seqs MAE <0.110058 (EXPECTED)
- Label distribution: 22-bar chart (target: no class <800 seqs)
- Severity distribution per fault group
- Physics gate summary: G1–G11 + G11-ext pass rates per class
- Compound temporal ordering verification plot (Group B)
- Masked secondary signal strength plot (Group C)
- Label 21 slope distribution: `err_slope_MotSV` histogram
- z_t export verification: shape check (N_windows × 64) per group pkl

### Writes `fault_rules_v3.json` → `models/fault_rules_v3.json`

- All 22-class definitions (LOCKED for M6.5r, M7, M8)
- Group E label integers assigned here (first and only assignment)
- `physics_context` dict per label (Invariant 18 — mandatory for M10 alerts)
- **Source of truth: `fault_rules_v3.json` (NOT `pasted-text.txt` for label integers)**

### Output After Step 3

```
data/synthetic/M6B_sequences_groupA_rerun.pkl    ← Labels 1,4,5 rerun (Step 0)
data/synthetic/M6B_sequences_groupB.pkl          ← Labels 7-12 compound (9,000 seqs)
data/synthetic/M6B_sequences_groupC.pkl          ← Labels 13-17 masked (6,000 seqs)
data/synthetic/M6B_sequences_groupD.pkl          ← Labels 18-21 variants (5,200 seqs)
data/synthetic/M6B_sequences_groupE.pkl          ← Group E multi-sensor (1,600 seqs)
data/synthetic/M6B_combined_sequences.pkl        ← ALL groups (~31,800 seqs, 22 classes)
data/synthetic/M6B_sequence_meta.csv             ← full metadata table
z_t_sequences_groupA_normal.pkl
z_t_sequences_groupA_faults.pkl
z_t_sequences_groupA_faults_rerun.pkl
z_t_sequences_groupB.pkl
z_t_sequences_groupC.pkl
z_t_sequences_groupD.pkl
z_t_sequences_groupE.pkl
models/fault_rules_v3.json                       ← 22-class label map (LOCKED)
outputs/reports/module_06b_synthetic_report.md
```

---

## Single Dispatcher Design Decision

```python
def generate_sequence(fault_config: dict) -> np.ndarray:
    """
    Unified dispatcher. fault_config carries:
      primary_fault           : str
      secondary_fault         : str | None     (compound chains only)
      masked_channels         : list | None    (masked faults only)
      variant_type            : str | None     (fast / intermittent / cyclic / gradual)
      n_steps                 : int            (per-label corrected length — NOT blanket 200)
      onset_step_primary      : int = 50       (same as M6A)
      onset_step_secondary    : int | None     (50 + lag for compounds)
      secondary_onset_lag     : int | None     (drawn from physics-verified per-label range)
      severity                : float          (Weibull-sampled; beta=1.5 for Label 21)
      cluster                 : str            (startup/steady_state/high_load/cooldown)
    One physics engine call handles all 4 groups — parameterized dispatch.
    No separate functions per group — just parameterized.
    n_steps is now variable per label — never hardcode 200 steps.
    """
```

**Why single dispatcher:**
- Consistent seam continuity logic across all groups
- Single severity sampling call (Weibull) — no per-group overrides
- Physics gate checks (G1–G11 + G11-ext) can run on every sequence from one loop
- Variable `n_steps` per label handled by config dict, not hard-coded branches
- Simpler to debug: one function, one trace

---

## Pre-Flight Confirmations (Locked — v14.2)

| # | Confirmation |
|---|-------------|
| 1 | Sequence lengths are **PER-LABEL** (from Part 1 tables) — NOT blanket 200 steps. `n_steps` is read from `fault_config` dict loaded from canonical label map. **DO NOT hardcode 200 anywhere in the generator.** |
| 2 | M6A pkl file path: `data/synthetic/M6_sequences.pkl` confirmed. **DO NOT overwrite this file.** Step 0 reads it as seed source only. |
| 3 | `fault_rules_v3.json` timing: Steps 0–2 work from in-memory config dict. Step 3 writes the JSON to `models/fault_rules_v3.json`. **Write ONCE only.** |
| 4 | Group B sequences per class: **1,500**. Total Group B: **9,000** sequences. (Previous v2.x spec said 1,200/class — corrected to 1,500 in v14.2.) |
| 5 | Group C sequences per class: **1,200**. Total Group C: **6,000** sequences. (Previous v2.x spec said 800/class — corrected to 1,200 in v14.2.) |
| 6 | Label 21 sequences: **2,000**. Steps: **1,000**. (Previous: 1,000 seqs, 200 steps — both corrected.) Highest count — hardest class. Weibull beta=1.5, sev=0.05–0.25. CIRA anchor: same 44 bearing spike seeds as Label 1. |
| 7 | Label 21 sub-threshold behaviour is EXPECTED and CORRECT. Sequences at sev <0.15 → MAE <0.110058 in most windows. LSTM-AE Layer 1 alone cannot detect these. **Do NOT raise threshold.** CUSUM on `score_B` (Layer 3) + Rolling Baseline on `score_A` (Layer 4) handle detection. |
| 8 | Group E pump-side sensor pair: **Pmp.SV + Pmp.PV** (NOT Pres.SV + Pmp.TV). Common-cause physics: both pump accelerometers share junction box/conduit. Label 17 secondary signal: Pmp.PV only (weakest masked path — max alert = WARN). |
| 9 | `fault_group_id` field in `M6B_sequence_meta.csv`: values `{0: normal, 1: single_source, 2: compound, 3: masked, 4: variant, 5: multi_sensor}`. M6.5r reads this field to generate `fault_group_id` feature column. |
| 10 | `physics_context_str` column in `M6B_sequence_meta.csv`: generated per sequence. Seeds M10 `/api/physics_context` endpoint. Mandatory — no alert exits M10 without physics context. (Invariant 18) |
| 11 | All z_t exports use M4 `lstm_ae_baseline.pth` (FROZEN, `map_location='cpu'` for safety). z_t shape per sequence = (N_windows × 64). N_windows = n_steps / 50. **Raw sensor data NEVER enters Level 2 TCN-AE.** (Invariant 16 — NEVER VIOLATE) |
| 12 | M8 TCN-AE expects z_t input — NOT raw sensor sequences. Module pathway: M6B generates z_t → M8 trains TCN-AE on z_t → M10 runs both. |

---

## Confidence-Gated Early Warning System (M8/M10 Design)

Four-layer detection cascade — all layers implemented in M8/M10:

| State | LSTM-AE Trigger | XGBoost Condition | Output Message |
|-------|----------------|------------------|----------------|
| **WATCH** | MAE crosses fuzzy lower bound (0.07–0.09) OR rolling score 2.0–3.5 OR CUSUM S_pos rising on `score_B` OR `score_C > score_C_normal_p95` | Top-class prob <0.65 OR top-2 combined <0.80 | *"WATCH — Anomaly detected. Type uncertain. Increase inspection frequency."* |
| **WARN** | MAE in fuzzy mid-zone OR accumulator 3.5–5.0 OR CUSUM S_pos > H on `score_B` OR `score_C > score_C_warn_threshold` | Top-class prob 0.65–0.80 | *"WARN — Likely [class]. Confidence X%. Monitoring for confirm."* |
| **DANGER** | MAE above fuzzy upper bound AND accumulator ≥5.0 OR rolling_score >3.5 OR `score_C > score_C_danger_threshold` | Top-class prob ≥0.80 | Single: *"DANGER — [class] confirmed."* Compound: *"COMPOUND FAULT — Primary [A] → Secondary [B]"* |
| **TREND ALERT** | Rolling Baseline on `score_A`: `drift_ratio > 1.10` (Layer 4 — disabled during burn-in ≤5,000 windows) | Any | *"TREND ALERT — [channel] drift. Plan inspection 7–14 days."* |

**Special cases:**

| Class/Label | Behaviour |
|-------------|-----------|
| Cavitation (label 3) | DANGER immediately (MAE=0.675, 6.1× threshold) — skip WATCH/WARN, bypass rolling accumulator |
| Overloading | Mech C Temp.SV Spearman drift PRIMARY — NOT single-window MAE |
| Seal failure | Pres.SV Spearman NEGATIVE over 300 windows = `seal_failure_early` |
| Group C (masked) | Max state = WARN unless secondary MAE independently crosses threshold |
| Label 17 | Max state = WARN (Pmp.PV only — weakest secondary path) |
| Label 21 | Layer 1 INSUFFICIENT for sev <0.15. Layer 3 CUSUM on `score_B` → WATCH at ~500 windows. Layer 4 Rolling Baseline on `score_A` → WARN at ~800 windows. XGBoost: *"bearing_wear_gradual — inspect within 7–14 days"* |
| Group B compound | `score_C` elevates at `secondary_onset_lag`. Phase 2 DANGER within 200 windows of secondary onset. |
| Label 10 (`seal→cav`) | `score_C` accumulates slowly then spikes at NPSHa/NPSHr crossing |
| Label 12 (`imbal→cav`) | `score_C` spikes at BPF-to-bubble-nucleation transition |

---

## M10 API JSON Response Structure (Locked Design — v14.2)

```json
{
  "timestamp": "2026-04-19T22:00:00",
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
    "score_A": 0.31,
    "score_B": 0.018,
    "score_C": 0.04,
    "shap_top3": ["mean_err_MotSV", "err_slope_MotTV", "kurtosis_PmpSV"]
  },
  "compound_check": {
    "secondary_fault_possible": true,
    "secondary_candidate": "overloading",
    "secondary_confidence": 0.31,
    "score_C_value": 0.04,
    "score_C_threshold": 0.12,
    "verdict": "MONITOR — score_C below compound threshold"
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
    "cusum_S_pos_MotSV": 0.0,
    "cusum_bearing_gradual_flag": false,
    "cusum_S_pos_current": 0.0,
    "cusum_alert": false
  },
  "rolling_baseline": {
    "drift_ratio": 1.02,
    "score_A_short_mean": 0.31,
    "score_A_long_mean": 0.30,
    "rolling_baseline_drift_flag": false,
    "drift_alert": false,
    "drift_message": null,
    "layer4_burn_in_complete": false
  },
  "early_warning": {
    "state": "WARN",
    "message": "Likely bearing_wear. Confidence 72%. MotSV trend will confirm within 15 min.",
    "watch_triggered_at": "2026-04-19T21:45:00",
    "estimated_confirmation_steps": 18
  },
  "physics_context": {
    "fault_class": "bearing_wear",
    "what": "Rolling element fatigue crack propagating via Paris-Erdogan law",
    "why": "Elevated Mot.SV* indicates mechanical vibration above baseline",
    "timeline": "Bearing typically fails within 48-120 hours at DEVELOPING severity",
    "action": "Increase vibration monitoring; schedule bearing inspection within 48h",
    "risk": "Unaddressed: bearing seizure -> shaft damage -> impeller contact",
    "disclaimer": "Detection based on discharge-side sensors. Suction conditions inferred."
  },
  "limitation_disclaimer": "Detection based on 8 discharge-side sensors at 1 Hz. Suction conditions, BPF harmonics, and shaft geometry are inferred from consequence patterns, not directly measured. Confidence scores reflect training data distribution. Confirm with physical inspection before maintenance shutdown."
}
```

---

## Module Pathway — v14.2 Status

```
M6A COMPLETE (Labels 0,2,3,6 + normal: LOCKED; Labels 1,4,5: RERUN in Step 0)
  │
M6B NEXT ACTIVE — spec locked v14.2, script not yet run
  Step 0: Re-gen Labels 1,4,5 at corrected lengths → M6B_sequences_groupA_rerun.pkl
  Step 1: Group B compound (9,000 seqs) + z_t → M6B_sequences_groupB.pkl + z_t_groupB.pkl
  Step 2: Groups C+D (11,200 seqs) + z_t → M6B_sequences_groupC/D.pkl + z_t files
  Step 3: Group E (1,600 seqs) + full merge (~31,800) + z_t final + fault_rules_v3.json
  Targets: ~31,800 sequences, 22 classes, 6 z_t pkl files per group
  │
M6.5r NOT STARTED — blocked until M6B_combined_sequences.pkl + z_t files exist
  Target: ~196,000 rows × ~35 columns → M6B_feature_matrix.csv
  New features: score_A, score_B, score_C (from z_t), zt_drift_slope,
                mean_zt_magnitude, std_zt_magnitude, onset_order
  │
M7 NOT STARTED — blocked until M6B_feature_matrix.csv (~196,000 × ~35)
  Input: M6B_feature_matrix.csv
  Target: label_int 0-21, 22-class XGBoost
  Output: models/M7_xgboost_classifier.json
  SHAP: score_C expected rank 1 for Group B compound classes
  │
M8 NOT STARTED — TCN-AE architecture locked v14.2
  Level 1: LSTM-AE 50-step window (frozen M4 weights as starting point)
  Level 2: TCN-AE on z_t sequences (5-layer, dilation=1,2,4,8,16, RF=63 windows)
  Level 3: CUSUM on score_B only
  Level 4: Rolling Baseline on score_A only
  score_C → XGBoost only (Invariant 19 — NEVER cross-route)
  │
M9 → M10 → M11 → M12

M6C: CANCELLED — all valid content absorbed into M6B Groups C, D, E.
```

---

## Paste Keys (ALL PENDING — fill after each step runs, do NOT fill in advance)

> **══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT AFTER M6B COMPLETES ══**

### Step 0 — Group A Rerun

| Key | Target / Value |
|-----|---------------|
| `M6B_step0_label1_seqs` | 1,500 (bearing_wear 250s) |
| `M6B_step0_label4_seqs` | 1,500 (seal_failure 400s) |
| `M6B_step0_label5_seqs` | 1,500 (overloading 300s) |
| `M6B_step0_gate_G1_G7_rerun_labels` | ALL PASS (21/21 gates) |
| `M6B_step0_zt_export_rerun` | True (4,500 entries, 0 shape errors) |
| `M6B_step0_fixes_applied` | F1 (Temp.SV* coupling r=0.9793), F4 (Pres.SV* Q-H shift) |
| `M6B_step0_script` | module_06B_step0_groupA_rerun_v2.py |
| `M6B_step0b_label0_seqs` | 2,000 (normal 200s) |
| `M6B_step0b_label2_seqs` | 1,500 (impeller_imbalance 200s) |
| `M6B_step0b_label3_seqs` | 1,500 (cavitation 150s) |
| `M6B_step0b_label6_seqs` | 1,200 (sensor_failure 150s) |
| `M6B_step0b_gate_all_pass` | True (20/20 gates) |
| `M6B_step0b_cav_pres_shift` | -0.2304 (must be less than 0) |
| `M6B_step0b_cav_pmpSV_shift` | +0.2003 (must be greater than 0) |
| `M6B_step0b_label6_subtypes` | flatline/spike/drift/dropout 300 each |
| `M6B_step0b_zt_normal` | 2,000 entries | 0 shape errors |
| `M6B_step0b_zt_faults` | 4,200 entries | 0 shape errors |
| `M6B_step0b_fixes_applied` | F2 (abs_sin), F3 (M5-faithful cav), F5 (dropout) |
| `M6B_step0b_script` | module_06B_step0b_groupA_carried_v2.py |
| `Status_for_M6B_Step1` | READY |

### Step 1 — Group B

| Key | Target / Value |
|-----|---------------|
| `M6B_step1_group_B_sequences` | [fill — target 9,000] |
| `M6B_step1_gate_G8_temporal` | [PASS/FAIL per class] |
| `M6B_step1_gate_G9_compound_mae` | [PASS/FAIL per class] |
| `M6B_step1_zt_export_groupB` | [True/False] |
| `M6B_step1_label10_max_lag_used` | [actual max lag value — should be 400–800] |

### Step 2 — Groups C + D

| Key | Target / Value |
|-----|---------------|
| `M6B_step2_group_C_sequences` | [fill — target 6,000] |
| `M6B_step2_group_D_sequences` | [fill — target 5,200] |
| `M6B_step2_label21_sequences` | [fill — target 2,000] |
| `M6B_step2_label21_steps` | 1,000 |
| `M6B_step2_gate_G10_masked` | [PASS/FAIL per class] |
| `M6B_step2_gate_G11ext_label21_slope` | [PASS/FAIL — `err_slope_MotSV > 0` in ≥95% seqs] |
| `M6B_step2_zt_export_groupC` | [True/False] |
| `M6B_step2_zt_export_groupD` | [True/False] |
| `M6B_step2_label21_subthreshold_pct` | [% Label 21 seqs MAE <0.110058 — expect ≥60%] |

### Step 3 — Group E + Merge

| Key | Target / Value |
|-----|---------------|
| `M6B_step3_group_E_sequences` | [fill — target 1,600] |
| `M6B_step3_gate_G11_multisensor` | [PASS/FAIL] |
| `M6B_step3_total_sequences` | [fill — target ~31,800] |
| `M6B_step3_classes` | 22 (labels 0–21) |
| `M6B_step3_fault_rules_v3_written` | [True/False] |
| `M6B_step3_zt_export_all_groups` | [True/False] |
| `M6B_step3_physics_context_str_generated` | [True/False] |
| `M6B_step3_physics_violations` | [fill — target: NONE] |
| `M6B_step3_coupling_fidelity_all` | [fill — target: r≥0.87 all thermal faults] |
| `M6B_step3_label_distribution_min` | [min sequences per class — target ≥800] |
| `M6B_step3_sequence_meta_rows` | [fill — should match total sequences] |
| `Status_for_M6p5r` | PENDING → set READY after Step 3 all gates PASS |

> **══ END PASTE UPDATE ══**

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-15 | Created as Part 2 split from `modules_M6B_synthetic_expanded.md` v1.0. Script plan, dispatcher, pre-flight, API spec, module pathway, paste keys. |
| v2.0 | 2026-04-16 | v14.0: Label 21 `bearing_wear_gradual` added to Step 2 Group D (1,000 seqs, Weibull beta=1.5, Gate G11-ext). Group B 6 classes labels 7–12, 1,200/class. Group C 5 classes labels 13–17, 800/class. Group D 4 variants labels 18–21. Step 3 merge 22-class. API response: `cusum_state` + `rolling_baseline` fields added. Early warning table: TREND ALERT state added. Label 21 special case added. Paste keys: label21 slope gate + subthreshold pct added. Pre-flight 7+8 added. |
| v2.1 | 2026-04-18 | v14.1 physics corrections. Group B label map explicit (label 11 = `overloading->bearing_wear`, label 12 = `impeller_imbalance->cavitation`). Group C label map explicit. `seal_failure_fast` corrected from Hagen-Poiseuille to turbulent orifice discharge. Group E pump-side pair corrected to Pmp.SV + Pmp.PV. Pre-flight item 5 corrected. |
| v3.0 | 2026-04-19 | v14.2 MAJOR UPDATE: Step 0 added (re-generate Labels 1, 4, 5 at corrected physics-verified lengths). All sequence counts corrected: Group B 1,200→1,500/class; Group C 800→1,200/class; Label 21 1,000→2,000 seqs, 200→1,000 steps. Total ~26,000→~31,800 sequences. z_t export added to ALL steps. Dispatcher `n_steps` now variable per label. `physics_context_str` column added to `M6B_sequence_meta.csv`. API response: `score_A`, `score_B`, `score_C` fields added; `physics_context` block added; `limitation_disclaimer` added. Module pathway: z_t→M8 TCN-AE noted. Paste keys expanded 17→32. |

---

> This file covers: Step 0 + 3-step script plan, dispatcher, pre-flight confirmations, API spec, paste keys.
> For fault universe physics rules, CIRA anchor rationale, dataset targets, physics gates:
> → `modules_M6B_synthetic_expanded.md` (Part 1)
>
> **Canonical source of truth:** `pasted-text.txt` (v14.2, 2026-04-19)
> **Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset
