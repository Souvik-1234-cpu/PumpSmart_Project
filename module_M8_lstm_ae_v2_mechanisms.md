# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
## Part 1B of 3 — TCN-AE Mechanisms, Detection Map, Training Data

| Field | Value |
|-------|-------|
| **Document version** | v2.0 — v14.2 TCN-AE mechanisms |
| **Date** | 2026-04-19 |
| **Part 1A (Architecture)** | `module_M8_lstm_ae_v2_architecture.md` — Level 1 LSTM-AE + Level 2 TCN-AE + Fuzzy layer |
| **Part 2 (Gates + Outputs)** | `module_M8_lstm_ae_v2_gates_and_outputs.md` |
| **Prerequisite** | M7 all 16 gates passed — `M7_all_16_gates_pass = True` |
| **Asset** | 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP) |

> **NOTE:** This file covers Level 2 TCN-AE mechanisms, the Glass analogy, Mech A/B/C slow drift detectors, Layer 3 CUSUM, Layer 4 Rolling Baseline, detection map, and training data composition.
>
> - Read **Part 1A** first for LSTM-AE architecture, TCN-AE architecture overview, channel weights, fuzzy layer, score routing rules, and adaptive threshold.
> - Read **Part 2** for gates, alert state machine, paste keys.

---

## Detection Layer Architecture (M8 Full Stack)

| Layer | Component | Covered In |
|-------|-----------|-----------|
| Level 1 | LSTM-AE per-window anomaly | Stage 1 + Stage 2 (Part 1A) |
| Level 2 | TCN-AE cross-window pattern | Stage 3A: TCN mechanism *(this file)* |
| Layer 2 — Mech A | Rolling Mean Gate | Stage 3B *(this file)* |
| Layer 2 — Mech B | Slope Detector | Stage 3C *(this file)* |
| Layer 2 — Mech C | Per-Channel Drift Monitor | Stage 3D *(this file)* |
| Layer 3 | CUSUM on score_B | Stage 3E *(this file)* — label 21 primary |
| Layer 4 | Rolling Baseline on score_A | Stage 3F *(this file)* — label 21 confirm |
| Layer 5 | Alert State Machine | Stage 4 (Part 2) |
| Layer 6 | Cluster-Conditional Thresholds | Stage 5 (Part 2) |

> **Layers 3 and 4 are the primary detection path for label 21 (`bearing_wear_gradual`). No other fault class REQUIRES them — do NOT apply them as general detectors.**

---

## Liability Basis (ISO 13374 Level 3)

> **Category 3 fault = progressive degradation = MODEL'S RESPONSIBILITY.**
> Fault developing over days/weeks → per-window MAE too small for threshold. Without cross-window detection → fault missed entirely → liability exposure. M6B severity 0.2–0.3 sequences generated SPECIFICALLY to calibrate these layers.
>
> For label 21 (`bearing_wear_gradual`): severity 0.05–0.25 means MAE **NEVER** crosses threshold even in severe windows. Layers 3 and 4 are mandatory.

---

## Stage 3A — Level 2: TCN-AE Mechanism

> **⭐ THIS SECTION IS NEW IN v2.0 — TCN-AE REPLACES LSTM v2**

### Dilated Causal Convolution — Core Equation

```
y[t] = sum(k=0 to K-1) f[k] * x[t - d*k]

Where:
  y[t]    = output at position t
  f[k]    = filter weight at position k
  x[t-d*k]= input at position t minus dilation*k (PAST only — causal)
  K       = kernel size = 3
  d       = dilation factor (1, 2, 4, 8, 16 per layer)
```

> **Causal constraint:** `x[t - d*k]` uses ONLY positions ≤ t. No future leakage — valid for real-time streaming inference.

### Receptive Field — Glass Analogy

Each TCN layer is a "Glass" pane looking back over a different window:

| Glass | Dilation | RF (windows) | RF (raw seconds) | Catches | Physics |
|-------|----------|-------------|-----------------|---------|---------|
| Glass 1 | d=1 | 3 | 150s | Labels 3 (cavitation), 6 (sensor_failure) | Acute hydraulic shock or instantaneous sensor death |
| Glass 2 | d=2 | 5 | 250s | Labels 1 (bearing_wear), 2 (impeller_imbalance), 17 (masked) | Bearing Paris-law early rise, BPF pattern establishment |
| Glass 3 | d=4 | 9 | 450s | Labels 7, 8, 12 (compound), 13, 16, 18 (masked/intermittent) | Compound fault primary-phase establishment, masked secondary path |
| Glass 4 | d=8 | 17 | 850s | Labels 9, 10, 11 (compound), 14, 15 (masked), 20 (cyclic) | Compound fault chain transition (lag 400–800 steps), cyclic baseline drift |
| Glass 5 | d=16 | 33 (63 full RF) | 3,150s | Label 21 (gradual bearing, 1,000 steps = 20 windows) | Paris-Erdogan low-dK crack growth — only visible at multi-week timescale |

**Full receptive field across all 5 layers:**

```
RF = 1 + (K-1) × sum(dilations) = 1 + 2 × (1+2+4+8+16) = 63 windows
```

### Vanishing Gradient Comparison

| Architecture | Gradient Path at N_windows=16 | Implication |
|-------------|------------------------------|-------------|
| **LSTM v2** (rejected) | 0.9¹⁶ = 0.19 (19% signal remaining) | Chain onset at window 1 nearly invisible at window 16 |
| **TCN-AE** (current) | Constant 5 layers regardless of length | 100% signal preserved at window 16 |
| **Transformer** (rejected) | N/A — attention degrades at N=6–20 | Weights degenerate to near-uniform (1/6 each); no meaningful position weighting |

> TCN receptive field is deterministic and physics-aligned — preferred over Transformer for this sequence length range.

### Stateful vs Geometric Memory

| Property | LSTM (stateful) | TCN (geometric) |
|----------|----------------|----------------|
| Memory mechanism | Hidden state h_t carries memory forward | Dilation pattern creates fixed receptive field |
| Memory decay | Exponential with sequence length | Deterministic geometric coverage (RF = 63 windows) |
| Initialization | Careful init required, prone to vanishing gradient | No hidden state to reset or initialize |
| Window boundaries | Reset = inter-window amnesia | Slide one window, reuse cached activations |

### Rolling Buffer Architecture — Real-Time Streaming Inference

```
At M10 inference time:
  Maintain circular buffer: z_t_buffer of shape (63, 64)
  On each new 50-step sensor window:
    1. Run Level 1 LSTM-AE → get z_t (64-dim)
    2. Append z_t to circular buffer (shift oldest out)
    3. When buffer has >= min_windows (e.g., 3): run TCN-AE forward pass
    4. Get score_A, score_B, score_C
    5. Route scores to CUSUM / Rolling Baseline / XGBoost per routing rules

Memory cost: 63 × 64 × float32 = 16 KB per pump — negligible
```

---

## Stage 3B — Mechanism A: Rolling Mean Gate (~3 minute horizon)

### Computation

```
rolling_mean_MAE_200 = mean(MAE, last 200 windows)
rolling_mean_MAE_100 = mean(MAE, last 100 windows)
```

### Thresholds (calibrated on mild-severity M6B sequences)

| Condition | Alert State |
|-----------|-------------|
| `rolling_mean_MAE_200 > 0.085` | WATCH |
| `rolling_mean_MAE_100 > 0.095` | WARN |

**Physics basis:** 200 windows = ~3 min at 1 Hz sampling. Seal wear at severity 0.2 raises mean MAE by ~0.008 / 100 windows. Detectable in WATCH within ~10 minutes of onset.

### Calibration Targets

| Scenario | Target |
|----------|--------|
| Mild bearing sev 0.2–0.3 | WATCH fires ≤10 min of simulated onset |
| Mild seal sev 0.2–0.3 | WATCH fires ≤15 min of simulated onset |
| Normal pool | `rolling_mean_MAE_200` stays below 0.085 in steady_state/high_load |

> **Label 21 (`bearing_wear_gradual`) — NOT detected by Mech A:** MAE never accumulates above 0.085 rolling mean at sev 0.05–0.25. Mech A will NOT fire for label 21. This is **CORRECT**. Layer 3 CUSUM (Stage 3E) is the mandatory detection path. **Do NOT lower Mech A threshold to detect label 21 — FPR will rise.**

---

## Stage 3C — Mechanism B: Slope Detector (~8 minute horizon)

### Computation

```
slope = linear_regression_slope(MAE_values, last 500 windows)
```

### Threshold

| Condition | Action |
|-----------|--------|
| `slope > 0.0003/window` | Escalate alert state by 1 level |

**Physics basis:** Bearing degradation over 8h produces slope ~0.0001–0.0005/window. At 0.0003/window threshold: 500 windows = ~8 min to confirm trend. Never used in isolation — always combined with Mech A or Mech C.

### Group D Severity Variant Implications

| Label | Class | Mech B Behaviour |
|-------|-------|-----------------|
| 18 | `cavitation_intermittent` | Slope NOT monotonic (on-off bursts) |
| 20 | `overloading_cyclic` | Slope of BASELINE drift, not instantaneous MAE. Use `cyclic_baseline_drift` feature from M6.5r to distinguish sawtooth from trend. |
| 19 | `seal_failure_fast` | Slope strongly negative, fires rapidly |

> **Label 21 (`bearing_wear_gradual`) — Mech B PARTIAL signal only:** `err_slope_MotSV` (the M6.5r feature) is positive and monotonic. However, MAE-based slope at 0.0003/window will NOT fire reliably for sev 0.05–0.15. Mech B may fire at sev 0.20–0.25 (later stage gradual wear). Layer 3 CUSUM (Stage 3E) is the **PRIMARY** detection mechanism for label 21. Mech B is **SECONDARY** confirmation — not relied upon for gate.

---

## Stage 3D — Mechanism C: Per-Channel Drift Monitor

### Computation

```python
channel_error[ch] = |reconstructed[ch] - input[ch]|   # RAW — bypasses weight matrix

spearman_r[ch] = spearman_correlation(
    channel_error[ch], time_index, last 300 consecutive windows
)
if spearman_r[ch] > 0.70:
    channel_drift_flag[ch] = True
```

> **⚠️ IMPLEMENTATION MANDATE:**
> ```python
> channel_error = abs(model_output[ch] - model_input[ch])  # BEFORE weight matrix
> # NOT: weighted_channel_error
> ```
> Reason: Temp.SV weight=0.5 suppresses in MAE but Mech C needs raw signal. Pres.SV weight=2.5 would exaggerate — raw error gives honest channel signal.

### Channel → Fault Type Mapping

**Temp.SV drift** (Spearman_r > 0.70, POSITIVE trend):
- → `overloading_early` flag
- Finding F1 — **PRIMARY and ONLY** reliable detection path for overloading
- Physics: overloading = motor overheating → Temp.SV\* rises monotonically
- Gate M8-7: overloading TPR ≥80% measured via THIS flag ONLY
- Also fires for: `overloading_cyclic` (label 20) — Mech B slope of baseline drift used to distinguish cyclic from sustained overload

**Pres.SV drift** (Spearman_r > 0.70, NEGATIVE slope):
- → `seal_failure_early` flag
- Finding F2 — **PRIMARY** detection path for mild seal failure
- Physics: seal failure = progressive pressure loss → Pres.SV\* falls
- Cross-check: `thermal_decoupling` must ALSO be True simultaneously
- Gate M8-9: WATCH fires ≤20 min via this flag
- Gate M8-10: This flag fires BEFORE total MAE reaches WARN level
- Also fires for: `cavitation->seal_failure` (label 8) — Pres.SV drift begins AFTER `secondary_onset_lag` timesteps

**Mot.SV drift** (Spearman_r > 0.70, POSITIVE trend):
- → `bearing_wear_early` flag
- Physics: bearing degradation → Mot.SV\* rises before Mot.TV (20–40s lag)
- Thermal coupling must ALSO be preserved (r > 0.85) simultaneously
- Both: Mot.SV drift + thermal coupling preserved = bearing confirmed
- Also fires for: `impeller_imbalance->bearing_wear` (label 9) — Mot.SV drift appears at `secondary_onset_lag` after PmpSV initial spike
- Also fires for: `bearing_wear->overloading` (label 7) — Mot.SV drift (primary) precedes Temp.SV rise (secondary) by lag steps

**Mot.SV VERY SLOW drift** (Spearman_r > 0.65, POSITIVE, window=500):
- → `bearing_wear_gradual_early` flag (label 21 SPECIFIC)
- Finding F7 — **PRIMARY** Mech C path for label 21
- Physics: Paris-Erdogan low-dK regime — crack growth rate sub-critical
- Mot.SV\* rises at ~0.002–0.005 per 100 windows (vs 0.01–0.03 for standard BW)
- LOWER Spearman threshold (0.65 not 0.70) — signal is very weak
- LONGER window (500 not 300) — need more data to accumulate trend
- Layer 3 CUSUM (Stage 3E) fires BEFORE this flag at typical severities — this flag is CONFIRMATION, not primary
- **DO NOT apply 500-window Spearman to other fault classes**

**Single channel flatline:**
- `std(channel_error[ch], last 100 windows) < 0.001` → `sensor_failure` flag
- Physics: dead sensor → value locked → reconstruction error constant → std collapses
- Group E (multi-sensor): TWO channels flatline simultaneously → `multi_sensor_anomaly_count = 2` → `sensor_failure_2ch` variant

### Group C — Masked Fault Detection via Secondary Path (5 classes)

When primary fault channel is flatline (`masked_channel_flag = True`):

| Label | Class | Secondary Detection Path | Notes |
|-------|-------|--------------------------|-------|
| 13 | `bearing_wear_MotSV_masked` | Mech C: Mot.TV + Temp.SV drift | — |
| 14 | `cavitation_PresSV_masked` | Mech C: Pmp.SV kurtosis bursts | — |
| 15 | `seal_failure_PresSV_drifting` | Mech C: secondary hydraulic channels | — |
| 16 | `overloading_TempSV_stuck` | Mech C: Mot.TV (r=0.997 coupling) | — |
| 17 | `impeller_imbalance_PmpSV_flatline` | Mech C: Pmp.PV + cross-channel | Spearman_r threshold lowered to 0.60 (weaker signal) |

> Gate M8-13 covers all 5 Group C classes. Label 17 expected weakest.

---

## Stage 3E — Layer 3: CUSUM on score_B (label 21 primary detector)

> **SCOPE:** Label 21 (`bearing_wear_gradual`) PRIMARY detection. Operates on `score_B` (drift slope from TCN-AE) — NOT raw MAE. (`score_B → CUSUM only` — Invariant 19)

### Physics Basis

Paris-Erdogan law: `da/dN = C*(dK)^m`. At low dK (sub-critical regime): crack growth is sub-threshold per cycle. Per-window MAE never crosses threshold at sev 0.05–0.15. But `score_B` (drift slope) is consistently positive — direction is monotonic. CUSUM detects cumulative directional deviation — not magnitude. This is EXACTLY what sub-threshold progressive degradation looks like.

### Computation

```
target  = mean(normal_pool_score_B)          # computed from CIRA normal pool
k       = 0.5 × sigma(normal_pool_score_B)  # allowance = half std dev
S_pos   = 0                                  # accumulator

for each new window w:
    b_w   = score_B[w]                       # drift slope from TCN-AE
    S_pos = max(0, S_pos + (b_w - target - k))
    if S_pos > H:
        fire cusum_bearing_gradual_flag = True
        emit WATCH alert
```

**Threshold H calibration:**
- Target: fires within 300–500 windows (~5–8 min) for sev 0.10 sequences
- Does NOT fire for 500 consecutive normal pool windows
- Start with `H = 5 × sigma(normal_pool_score_B)`
- Tune on M6B label 21 mild sequences

### Reset Policy

Reset `S_pos = 0` only if 100 consecutive windows give `b_w < target`. Prevents false reset on brief normal interludes within gradual wear.

### Integration with Alert State Machine

| Condition | Alert State |
|-----------|-------------|
| `cusum_bearing_gradual_flag = True` | WATCH |
| `cusum_bearing_gradual_flag` + Mech C Mot.SV slow drift flag | WARN |
| `cusum_bearing_gradual_flag` + Layer 4 rolling baseline flag | WARN |
| All three simultaneously (CUSUM + rolling baseline + Mech C slow drift) | **DANGER** (escalate, do not wait for MAE threshold) |

---

## Stage 3F — Layer 4: Rolling Baseline on score_A (label 21 confirm)

> **SCOPE:** Label 21 (`bearing_wear_gradual`) SECONDARY confirmation. Operates on `score_A` (severity from TCN-AE) — NOT raw MAE. (`score_A → Rolling Baseline only` — Invariant 19)

### Physics Basis

Bearing degradation over days → the BASELINE of `score_A` drifts up. Even if each individual window looks near-normal, the 6hr mean is higher than the 24hr mean. This is the hallmark of Paris-Erdogan slow accumulation.

### Computation

```
Adaptive threshold: theta_t = mu_rolling(6hr) + 3*sigma_rolling(6hr)
Update interval: every 50 seconds in M10 runtime

baseline_short = mean(score_A, last 6 hours of operation)
baseline_long  = mean(score_A, last 24 hours of operation)
drift_ratio    = baseline_short / baseline_long

if drift_ratio > 1.10:
    fire rolling_baseline_drift_flag = True
    emit WATCH (if not already in WATCH from CUSUM)
if drift_ratio > 1.25:
    escalate to WARN
```

### Calibration Targets

| Scenario | Target |
|----------|--------|
| Normal pool | `drift_ratio` stays in [0.95, 1.05] for 95% of windows |
| Label 21 sev 0.10 | `drift_ratio` crosses 1.10 within 800–1,200 windows |
| Label 21 sev 0.20 | `drift_ratio` crosses 1.25 within 600–900 windows |

> **NOTE:** Layer 4 requires minimum 5,000 windows of operational history. Do NOT activate at machine startup — enable after burn-in = 5,000 windows. Use CUSUM only during burn-in period.

### Integration

| Condition | Alert State |
|-----------|-------------|
| `rolling_baseline_drift_flag` alone | WATCH (soft alert, low confidence) |
| `rolling_baseline_drift_flag` + `cusum_bearing_gradual_flag` | WARN (high confidence) |
| All three (CUSUM + rolling baseline + Mech C slow drift) | **DANGER** |

**Two-speed adaptation rationale (why both Layer 3 and Layer 4 are needed):**

- **Fast (Layer 4, 6hr rolling):** Adapts to operating-point shifts (load changes). Prevents false alarms when pump load changes shift normal `score_A` baseline.
- **Slow (Layer 3, CUSUM weeks):** Detects secular drift immune to baseline creep. CUSUM on `score_B` (rate) is orthogonal to `score_A` (level). Operating point shift changes `score_A` level but NOT `score_B` slope. Therefore CUSUM on `score_B` is immune to load changes.
- **Together:** Layer 4 handles level noise, Layer 3 handles drift signal.

---

## M8 Detection Coverage Map (22 Fault Classes)

| Fault | Label | Level 1 (MAE+Fuzzy) | Mech A (Rolling) | Mech B (Slope) | Mech C (Per-Ch) | Layer 3 (CUSUM) | Layer 4 (Baseline) | TCN Glass |
|-------|-------|---------------------|-----------------|----------------|-----------------|-----------------|-------------------|-----------|
| normal | 0 | below threshold | stable | flat | flat | S=0 | ratio~1.0 | — |
| bearing_wear sev 0.8 | 1 | DANGER | yes | yes | Mot.SV drift | — | — | 2–3 |
| bearing_wear sev 0.2 | 1 | MAE~0.098 sub-threshold | WATCH ~10min | ~8min | Mot.SV drift | — | — | 2–3 |
| impeller_imbalance | 2 | yes | yes | yes | Pmp.SV | — | — | 1–2 |
| cavitation severe | 3 | DANGER (bypass) | bypassed | bypassed | Pres.SV | — | — | 1 |
| seal_failure slow | 4 | 29% windows | WATCH ~15min | yes | Pres.SV PRIMARY | — | — | 2–3 |
| overloading mild | 5 | MAE~0.093 sub-threshold | slow | yes | Temp.SV PRIMARY | — | — | 2–3 |
| sensor_failure | 6 | DANGER MAE~0.170 | yes | yes | flatline std<0.001 | — | — | 1 |
| bearing_wear->overloading | 7 | both channels | yes | yes | Mot.SV then Temp.SV | — | — | 3–4 |
| cavitation->seal_failure | 8 | Pmp.SV dominant | yes | yes | Pmp.SV+Pres.SV | — | — | 3–4 |
| impeller->bearing_wear | 9 | yes | yes | yes | Pmp.SV then Mot.SV | — | — | 4 |
| seal_failure->cavitation | 10 | yes | yes | yes | Pres.SV then Pmp.SV | — | — | 4–5 |
| overloading->bearing_wear | 11 | yes | yes | yes | Temp.SV then Mot.SV | — | — | 4–5 |
| impeller->cavitation | 12 | yes | yes | yes | Pmp.SV kurtosis | — | — | 3 |
| bearing_wear_MotSV_masked | 13 | MotSV absent | secondary | yes | Mot.TV+Temp.SV | — | — | 2–3 |
| cavitation_PresSV_masked | 14 | PresSV absent | Pmp.SV | yes | Pmp.SV kurtosis | — | — | 1–2 |
| seal_fail_PresSV_drifting | 15 | PresSV drifting | slow | yes | secondary hydraulic | — | — | 3–4 |
| overloading_TempSV_stuck | 16 | TempSV absent | no | yes | Mot.TV r=0.997 | — | — | 2–3 |
| impeller_PmpSV_flatline | 17 | PmpSV absent | PmpPV | yes | Pmp.PV+cross-ch (0.60) | — | — | 2 |
| cavitation_intermittent | 18 | burst windows | not monotonic | no | Pmp.SV bursts | — | — | 2–3 |
| seal_failure_fast | 19 | high MAE quickly | rapid | yes | Pres.SV sharp drop | — | — | 1 |
| overloading_cyclic | 20 | sawtooth | ambiguous | baseline drift | Temp.SV cyclic | — | — | 3–4 |
| **bearing_wear_gradual (\*)** | **21** | **SUB-THRESHOLD (correct)** | **NO** | partial sev≥0.20 | Mot.SV slow (0.65, 500w) | **PRIMARY** | **CONFIRM** | **5** |
| sensor_failure_2ch_thermal | E-a | additive MAE | yes | yes | 2× flatline Mot.TV+Temp.SV | — | — | 1 |
| sensor_failure_2ch_pump | E-b | additive MAE | yes | yes | 2× flatline Pmp.SV+Pmp.PV | — | — | 1 |

> **(\*) Label 21 is the ONLY class using Layer 3 + Layer 4.** Both are mandatory for Gate M8-14-ext. Level 1 SUB-THRESHOLD for label 21 is **PHYSICALLY CORRECT** — NOT a model failure.

---

## M8 Training Data Composition

### Normal Training Pool (model learns ONLY normal — faults never appear in training)

| Source | Windows | Share |
|--------|---------|-------|
| Real CIRA normal (M3 normalized) | 9,711 windows | ~30% |
| Synthetic normal (M6B Type-A) | from ~2,000 normal sequences windowed | ~70% |
| **Total normal training pool** | **~33,000 windows (approx)** | — |

> **BIAS 2 RATIONALE — WHY 30:70 REAL:SYNTHETIC:**
> Pure synthetic training → model learns only physics-idealized normal patterns → too sensitive to real-world deviations → elevated FPR in field. 30% real CIRA anchors the model to actual pump behaviour: manufacturing tolerances, sensor placement variation, ambient noise. 70% synthetic provides coverage of all 4 operating modes with controlled severity distribution (Weibull-skewed from M6B).

Cluster distribution maintained: startup 42.3%, cooldown 22.8%, etc.

### Validation Only — Fault Sequences (never in training)

All M6B fault sequences (Groups A–E, 22 classes) → windowed → validation pool.

| Subset | Purpose |
|--------|---------|
| All M6B fault sequences | Calibrate thresholds + fuzzy bounds + measure TPR/FPR + gate checks |
| Severity 0.2–0.3 subset | Calibrate Mech A/B/C rolling thresholds |
| Group B compound (labels 7–12) | Verify both fault channels produce high MAE |
| Group C masked (labels 13–17) | Verify secondary detection path fires |
| Group D variants (labels 18–20) | Verify detection character matches variant type |
| Label 21 mild (sev 0.05–0.15) | Calibrate CUSUM threshold H and rolling baseline |
| Label 21 moderate (sev 0.15–0.25) | Verify Layer 4 `drift_ratio` crosses 1.10 |
| Group E multi-sensor | Verify multi-channel flatline detection |

> **WHY FAULTS NEVER IN TRAINING:** LSTM-AE is anomaly detector, NOT classifier. Training on faults = model learns to reconstruct faults as normal = complete failure of the anomaly detection purpose. Faults only appear in validation to calibrate the boundary.

### z_t Sequence Inputs for Level 2 (TCN-AE)

All M6B sequences windowed by Level 1 → z_t sequences per group:

```
z_t_sequences_groupA_normal.pkl  → Level 2 normal pool training
z_t_sequences_groupA_faults.pkl  → Level 2 fault validation
z_t_sequences_groupB.pkl         → compound chain validation (score_C key)
z_t_sequences_groupC.pkl         → masked fault validation
z_t_sequences_groupD.pkl         → severity variant validation (score_B key for label 21)
z_t_sequences_groupE.pkl         → multi-sensor validation
```

> **Raw sensor data NEVER enters Level 2 directly.** (Invariant 16)

### Mech C Calibration Subset

Use ONLY mild severity (sev 0.2–0.4) sequences for Mech C tuning.

| Target | Condition |
|--------|-----------|
| ≥80% mild overloading sequences | Temp.SV drift fires ≤15 min (Spearman threshold 0.70) |
| ≥80% mild seal sequences | Pres.SV drift fires ≤20 min (Spearman threshold 0.70) |
| ≥60% label 17 sequences | Pmp.PV drift fires ≤25 min (Spearman threshold 0.65) |
| ≥70% label 21 sev≥0.10 sequences | Slow drift fires ≤20 min (Spearman 0.65, window=500) |

### Label 21 CUSUM Calibration Subset

Use ONLY label 21 mild (sev 0.05–0.15) for CUSUM H tuning.
- Target: CUSUM fires within 300–500 windows for sev 0.10
- Normal pool: `S_pos` stays below H for 500 consecutive normal windows
- Adjust `k` and `H` until both conditions simultaneously satisfied

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-16 | NEW FILE (v14.0 split). Stage 3 Mech A/B/C + detection map + training data. Stage 3D CUSUM Layer 3 (label 21 primary). Stage 3E Rolling Baseline Layer 4. Detection map expanded to 24 rows. Group C 5 classes. |
| v2.0 | 2026-04-19 | v14.2 TCN-AE MECHANISMS: Stage 3A added (TCN dilated causal convolution equation). Glass analogy added (5 panes, d=1/2/4/8/16, RF per pane). Full receptive field: 63 windows = 3,150 raw seconds. Vanishing gradient comparison: LSTM 0.9¹⁶=0.19 vs TCN constant 5 layers. Transformer rejection rationale: N_windows=6–20 too short for attention. Stateful vs geometric memory distinction added. Rolling buffer architecture for real-time streaming added. Layer 3 CUSUM: now operates on score_B (not raw MAE) per Invariant 19. Layer 4 Rolling Baseline: now operates on score_A per Invariant 19. Two-speed adaptation rationale: why Layer 3 and Layer 4 are orthogonal. z_t sequence inputs section added to training data. Detection map: TCN Glass column added, score routing reflected. Stage numbering updated: 3E (CUSUM) and 3F (Rolling Baseline). |

---

> **GitHub is the ONLY source of truth for this spec.**
>
> - Part 1A (Architecture): `module_M8_lstm_ae_v2_architecture.md`
> - Part 2 (Gates + Outputs): `module_M8_lstm_ae_v2_gates_and_outputs.md`
>
> **Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset
> **Standard:** ISO 10816-3 vibration, ISO 13373-3 condition monitoring
> **Canonical source of truth:** `pasted-text.txt` (v14.2, 2026-04-19)
