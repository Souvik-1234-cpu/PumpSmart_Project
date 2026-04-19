# PumpSmart — Module M8: LSTM-AE v2 Production Anomaly Detector
## Part 1A of 3 — Architecture Overview + Level 1 LSTM-AE + Level 2 TCN-AE + Fuzzy Layer
 
| Field | Value |
|-------|-------|
| **Document version** | v4.0 — v14.2 TCN-AE Level 2 architecture |
| **Date** | 2026-04-19 |
| **Part 1B (Mechanisms)** | `module_M8_lstm_ae_v2_mechanisms.md` — TCN mechanisms, detection map, training data |
| **Part 2 (Gates + Outputs)** | `module_M8_lstm_ae_v2_gates_and_outputs.md` |
| **Prerequisite** | M7 all 16 gates passed — `M7_all_16_gates_pass = True` |
| **Asset** | 110 kW, 7-stage, 40 bar, 2980 RPM multistage centrifugal pump (CIRA SACIP) |
| **Status** | NOT STARTED — begins only after M7 gates confirmed |
 
> **NOTE:** READ ALL THREE PARTS BEFORE WRITING ANY M8 CODE.
> - Part 1A = Architecture overview + Level 1 LSTM-AE + Level 2 TCN-AE + Fuzzy layer *(this file)*
> - Part 1B = TCN mechanisms, detection map, training data, Glass analogy
> - Part 2 = Gates, outputs, paste keys, score routing, adaptive threshold
 
---
 
## Safety Mandate (NON-NEGOTIABLE)
 
| Field | Value |
|-------|-------|
| **Asset** | 110 kW, 7-stage, 40 bar, 450 m head multistage centrifugal pump |
| **Replacement cost** | >Rs 50 lakh (industrial capital asset) |
| **Failure consequence** | Process shutdown + secondary damage + personnel injury risk |
| **Standard** | ISO 10816-3 (vibration), ISO 13373-3 (condition monitoring) |
| **Pipeline level** | ISO 13374 Level 3 condition monitoring |
 
> M8 is **NOT** an experiment. M4 was the baseline — M8 is the **PRODUCTION** model.
> Every architectural decision must be physically justified.
> Every gate must pass before M12 adversarial validation begins.
> **False negative on this asset = catastrophic failure.**
 
**PREREQUISITE:** `M7_all_16_gates_pass = True` before M8 starts.
If M7 gates fail → fix M6.5r → rerun M7 → only then start M8.
 
---
 
## v3.0 → v4.0 Change Summary (v14.2)
 
| Item | v3.0 (SUPERSEDED) | v4.0 (CURRENT — USE THIS) |
|------|------------------|--------------------------|
| Level 2 architecture | Hierarchical LSTM v2 | 5-layer dilated causal TCN-AE |
| Level 2 input | Raw sensor windows (50 × 8) | z_t sequences (N_windows × 64) from Level 1 |
| Level 2 output | Fuzzy accumulator score | score_A (severity), score_B (drift), score_C (chain) |
| Layer 3 CUSUM operates on | Raw MAE | score_B (drift slope from TCN-AE) |
| Layer 4 Rolling Baseline on | err_slope per channel | score_A (severity from TCN-AE) |
| Receptive field | ~40 windows (LSTM hidden state) | 63 windows = 3,150 raw seconds |
| Upstream feature count | 26 columns | ~35 columns (adds z_t + score_A/B/C features) |
| Adaptive threshold | Not implemented | `theta_t = mu_rolling(6hr) + 3*sigma_rolling(6hr)` |
| Static threshold 0.110058 | Used by all layers | Level 1 ONLY — never applied to TCN-AE output |
| Label 21 count | 1,000 sequences | 2,000 sequences (physics audit correction) |
| Total training sequences | ~26,000–28,000 | ~31,800 |
 
## v2.0 → v3.0 Change Summary (retained for audit)
 
| Item | v2.0 | v3.0 |
|------|------|------|
| M7 prerequisite gate count | 15 gates | 16 gates (Gate M7-14-ext added for label 21) |
| Classes | 21 | 22 (label 21 `bearing_wear_gradual` added) |
| Group B compound TPR | 5 classes (labels 7–11) | 6 classes (labels 7–12) |
| Group C masked TPR | 4 classes (labels 12–15) | 5 classes (labels 13–17) |
| Group D variant TPR | 3 classes (labels 16–18) | 4 classes (labels 18–21) |
| Fuzzy exclusion | overloading mild + seal mild | + label 21 mild (sev 0.05–0.15) |
| Gate count (M8) | 14 gates | 15 gates |
 
> **v1.0 is INVALID.** Do not reference v1.0 gate numbers or upstream file names.
 
---
 
## M6.5 Audit Findings — Critical Inputs for M8 Design
 
> Read before writing a single line of M8 code. Every finding directly constrains threshold calibration, channel weighting, and detection strategy.
 
| Finding | Root Cause | M8 Action |
|---------|-----------|-----------|
| **F1** | Overloading Gate 3 = 0.00% (MAE=0.093). Temp.SV weight=0.5 → sub-threshold weighted MAE. | Mech C Temp.SV drift = PRIMARY detection path. Gate M8-7: ≥80% TPR via Mech C ONLY. |
| **F2** | Seal failure Gate 3 = 29.17% (MAE=0.196). Pres.SV gradual decline — per-window MAE low. | Mech C Pres.SV drift (negative Spearman) = PRIMARY. Gate M8-9: WATCH ≤20 min. |
| **F3** | Bearing seam discontinuity 5.75%. Spike seed t=49→50 step change. | Attention must NOT peak at seam. Gate M8-8: `seam_ratio < 1.0`. |
| **F4** | Fisher rank 1 = PmpSV_mean. | Weight increase 2.0→2.5 Fisher-validated. |
| **F5** | Cavitation MAE = 0.675 (6.1×). | Bypass WATCH/WARN → DANGER immediately at startup. |
| **F6** | Normal probe 86.67% NOT FPR problem. Edge-case probe sampling artifact. | Gate M8-2 on full 9,711-window pool ONLY — never on 30-window probe. |
| **F7** | Label 21 MAE sub-threshold by design. Paris-Erdogan low-dK regime sev 0.05–0.15. | Do NOT raise threshold. TCN-AE score_B + CUSUM + Rolling Baseline are detection. |
 
---
 
## M8 Two-Level Architecture Overview
 
M8 operates as two sequential detection levels. They are **NOT** alternatives — both run.
 
### Level 1 — LSTM-AE (per-window anomaly detection)
 
| Property | Value |
|----------|-------|
| **Input** | (batch, 50, 8) — 50-timestep sensor windows, 8 normalized channels |
| **Output** | Per-channel MAE in ℝ⁸ + bottleneck z_t in ℝ⁶⁴ |
| **Question** | "Is THIS 50-step window anomalous?" |
| **Memory** | 50 steps only — hidden state resets each window |
| **Threshold** | 0.110058 (STATIC — locked from M4, Level 1 ONLY) |
| **Blind to** | Cross-window trends, compound fault chains, gradual drift |
 
### Level 2 — TCN-AE (cross-window pattern detection) ⭐ *NEW v14.2*
 
| Property | Value |
|----------|-------|
| **Input** | z_t sequences (N_windows × 64) — NEVER raw sensor data (Invariant 16) |
| **Output** | score_A (severity), score_B (drift slope), score_C (chain transition) |
| **Question** | "What PATTERN exists across the last 63 windows?" |
| **Memory** | 63 windows = 3,150 raw seconds (receptive field) |
| **Threshold** | Adaptive — `theta_t = mu_rolling(6hr) + 3*sigma_rolling(6hr)` |
| **Catches** | Compound chains, gradual drift, severity variants |
 
**Why Level 1 alone is insufficient:**
 
- Label 10 (`seal->cavitation`) lag = 400–800 steps = 8–16 windows. LSTM-AE at window 16 has ZERO memory of window 1 (hidden state reset). Chain transition signal completely lost.
- Label 21 (`bearing_wear_gradual`) at severity 0.05: MAE = 0.072 (below 0.110058). Level 1 CANNOT detect this by design — not a model failure, a physics reality.
**Why Level 2 was redesigned from LSTM v2 to TCN-AE:**
 
LSTM v2 had the same inter-window amnesia problem as Level 1. At N_windows=16, LSTM v2 gradient path = 0.9¹⁶ ≈ 0.19. Signal from window 1 reduces to 19% by window 16. Compound chain onset at window 1 = effectively invisible at window 16.
 
TCN-AE has a constant gradient path = 5 layers regardless of sequence length. No hidden state to reset, no vanishing gradient. Full architectural reasoning: see `module_M8_lstm_ae_v2_mechanisms.md` (Part 1B).
 
---
 
## Stage 1 — Level 1: LSTM-AE Reconstruction
 
| Parameter | Value |
|-----------|-------|
| **Input** | (batch, 50, 8) — 50-timestep windows, 8 normalized channels |
| **Window size** | 50 (M2 optimal, M6.5r fixed — **NEVER change**) |
| **Encoder** | LSTM(8→128, layers=2, dropout=0.3) → Multi-head temporal attention → Bottleneck(128→64) |
| **Decoder** | LSTM(64→128) → LayerNorm → Output(128→8). Hidden state seeded from encoder bottleneck. |
| **Optimizer** | AdamW |
| **Scheduler** | CosineAnnealingWarmRestarts (T0=20) |
| **AMP** | GradScaler + autocast (CUDA — RTX 4060 Laptop) |
| **MC Dropout** | N=20 forward passes at inference → mean_MAE + uncertainty_std |
| **Parameters** | ~505,096 (same order as M4 — no architecture bloat) |
 
### Loss Function (3-component — physics-weighted)
 
```
total_loss = 0.5*MAE + 0.3*MSE + 0.2*grad_penalty
 
grad_penalty = mean(|dX_reconstructed/dt - dX_input/dt|)
```
 
**Physics basis:** Penalizes unphysical rate-of-change in reconstruction. Critical for cavitation — highly erratic pressure signal. Without `grad_penalty`: model produces smooth output for erratic cavitation input → reconstruction error underestimated → cavitation MAE artificially lowered.
 
### Level 1 Outputs per Window
 
| Output | Type | Description |
|--------|------|-------------|
| `mae_per_channel` | ℝ⁸ | One MAE per sensor channel |
| `z_t` | ℝ⁶⁴ | Bottleneck latent vector — fed to Level 2 |
| `weighted_mae` | scalar | Channel-weighted composite |
| `uncertainty_std` | scalar | MC Dropout spread |
 
### Channel Weights — Fisher Validated from M6.5
 
| Channel | M4 Weight | M8 Weight | Reason |
|---------|-----------|-----------|--------|
| Mot.SV | 2.0 | 2.5 | Fisher rank 2 confirmed — vibration dominant |
| Pmp.SV | 2.0 | 2.5 | Fisher rank 1 confirmed — HIGHEST discriminability |
| Pres.SV | 2.0 | 2.5 | Primary seal + cavitation channel |
| Mot.PV | 1.5 | 2.0 | Displacement — secondary vibration |
| Pmp.PV | 1.5 | 2.0 | Displacement — BPF harmonics |
| Temp.SV | 1.0 | **0.5** | LOW WEIGHT — but Mech C monitors UNWEIGHTED |
| Mot.TV | 0.8 | 0.3 | Placement-dependent — low weight |
| Pmp.TV | 0.8 | 0.3 | Placement-dependent — low weight |
 
> **⚠️ CRITICAL DESIGN NOTE — WHY Temp.SV WEIGHT IS LOW BUT STILL DETECTABLE:**
>
> Temp.SV weight = 0.5 → weighted MAE contribution suppressed. Overloading raises ONLY thermal channels → weighted MAE stays sub-threshold. This is **EXPECTED and CORRECT** behaviour (Finding F1 from M6.5).
>
> Mech C (Stage 3C — see Part 1B) operates on **RAW** channel reconstruction error, **BYPASSING** the weight matrix entirely. Temp.SV at weight 0.5 retains FULL Mech C monitoring sensitivity.
>
> Without this design: overloading invisible to model.
> With this design: Mech C sees unweighted Temp.SV drift → `overloading_early` fires.
> This is the architectural solution to Gate 3 = 0.00% in M6.5.
 
---
 
## Stage 2 — Level 2: TCN-AE Cross-Window Pattern Detection
 
> **⭐ THIS ENTIRE SECTION IS NEW IN v4.0 — REPLACES HIERARCHICAL LSTM v2**
 
### Architecture
 
| Component | Specification |
|-----------|--------------|
| **Input** | z_t sequences of shape (N_windows × 64), where N_windows = total_steps / 50 per sequence. Ranges: 3 (Label 3, 150 steps) to 20 (Label 21, 1,000 steps). |
| **Output** | 3 scalar scores per sequence |
 
**Encoder — 5-layer dilated causal TCN-AE:**
 
| Layer | Dilation | Kernel | Filters | Context |
|-------|----------|--------|---------|---------|
| Conv1D layer 1 | 1 | 3 | 64 | Local anomaly detection |
| Conv1D layer 2 | 2 | 3 | 64 | 2-window context |
| Conv1D layer 3 | 4 | 3 | 64 | 4-window context |
| Conv1D layer 4 | 8 | 3 | 64 | 8-window context |
| Conv1D layer 5 | 16 | 3 | 64 | 16-window context |
| Bottleneck | — | — | — | Dense(64→32) → z_seq in ℝ³² |
 
**Decoder:** Transpose Conv1D layers (mirror of encoder). Output: reconstructed z_t sequence (N_windows × 64).
 
### Receptive Field Calculation
 
```
RF = 1 + (kernel-1) × sum(dilations)
   = 1 + (3-1) × (1+2+4+8+16)
   = 1 + 2 × 31
   = 63 windows = 3,150 raw seconds at 1 Hz
```
 
Covers Label 10 (longest lag: 400–800 steps = 8–16 windows) with margin. Covers Label 21 (N_windows = 20) completely within single receptive field.
 
### TCN-AE Outputs — score_A, score_B, score_C
 
**score_A (severity):**
- Derived from: reconstruction error magnitude across N_windows
- Formula: `score_A = mean(||z_t_actual - z_t_reconstructed||_2 over N_windows)`
- Physical meaning: how far the overall z_t trajectory is from normal reconstruction
- Feeds: Layer 4 Rolling Baseline (adaptive threshold)
- Example: Label 19 (`seal_failure_fast`) → high score_A in first 3 windows
**score_B (drift slope):**
- Derived from: linear trend in per-window reconstruction error over time
- Formula: `score_B = slope of OLS fit to [reconstruction_error[0], ..., reconstruction_error[N]]`
- Physical meaning: is the anomaly GROWING? (Paris law drift = monotonic positive slope)
- Feeds: Layer 3 CUSUM only
- Example: Label 21 (`bearing_wear_gradual`) → small but consistently positive score_B
**score_C (chain transition):**
- Derived from: discontinuity in z_t trajectory — sudden change in reconstruction pattern
- Formula: `score_C = max(||z_t_reconstructed[n] - z_t_reconstructed[n-1]||_2) over N_windows`
- Physical meaning: did the fault CHARACTER change? (compound fault transition event)
- Feeds: XGBoost M7 feature input (onset_order + score_C as top features for Group B)
- Example: Label 10 (`seal->cavitation`) → low score_C early, spike at lag window
### Score Routing Rules — INVARIANT 19 — NEVER CROSS
 
| Score | Routed To |
|-------|-----------|
| `score_B` | CUSUM only |
| `score_A` | Rolling baseline only |
| `score_C` | XGBoost only |
 
> **Cross-routing = architecture violation.**
 
### Adaptive Threshold (Level 2 — NOT Level 1)
 
> Static threshold **0.110058** is **Level 1 ONLY**. It is LOCKED and NEVER changes.
 
**Level 2 adaptive threshold for score_A:**
 
```
theta_t = mu_rolling(6hr) + 3*sigma_rolling(6hr)
 
mu_rolling(6hr)    : rolling mean of score_A over last 6 hours of operation
sigma_rolling(6hr) : rolling std of score_A over last 6 hours of operation
Update interval    : every 50 seconds in M10 runtime
Implemented in     : M10 Flask API (NOT in M8 training)
```
 
**Two-speed adaptation — why both are needed simultaneously:**
 
*Fast (6hr rolling) purpose:* Operating point shifts (load change, startup, valve adjustment) change baseline score_A. Without fast adaptation: score_A threshold stale → false alarms on every load change. Fast adaptation zeroes out operating-point noise in 6 hours.
 
*Slow (CUSUM weeks) purpose:* Paris law bearing degradation takes weeks to accumulate. If threshold adapts to the drift (fast rolling tracks it) → CUSUM S_n never grows. This is the **adaptive threshold paradox** for Label 21. Solution: CUSUM operates on `score_B` (drift slope) NOT `score_A`. `score_B` measures RATE of change, not absolute level. Rate does not increase with operating point shifts → immune to baseline creep. Fast rolling handles level shifts; slow CUSUM handles rate-of-change accumulation. Both are needed simultaneously — they monitor orthogonal signal components.
 
### Physics Context Layer
 
Static lookup table generated per sequence during M6B (physics context string). Stored in `fault_rules_v3.json` under key `"physics_context"` per label. Mandatory output for every M10 alert — no alert exits M10 without it. (Invariant 18)
 
**Format per fault class (all 22 classes):**
 
```json
{
  "label": <int>,
  "class_name": "<string>",
  "what": "<1 sentence: what is happening physically>",
  "why": "<1 sentence: root cause mechanism>",
  "timeline": "<estimated time to critical if untreated>",
  "action": "<recommended maintenance action>",
  "risk": "<consequence if ignored>",
  "disclaimer": "Advisory only. Confirm with certified engineer before maintenance."
}
```
 
**Examples:**
 
*Label 10 (`seal_failure->cavitation`):*
 
| Field | Value |
|-------|-------|
| what | Seal leak has reduced suction pressure below NPSHr causing bubble collapse |
| why | Q_leak reduces operating flow, shifts Q-H curve left, NPSHa crosses NPSHr |
| timeline | 2–6 hours to impeller pitting if untreated |
| action | Inspect mechanical seal. Reduce flow or increase suction head immediately. |
| risk | Impeller surface damage, vibration escalation, catastrophic failure |
 
*Label 21 (`bearing_wear_gradual`):*
 
| Field | Value |
|-------|-------|
| what | Early-stage bearing crack growth detected via CUSUM drift accumulation |
| why | Paris-Erdogan low stress intensity crack propagation (`da/dN = C*dK^m`) |
| timeline | 7–14 days to detectable vibration threshold (Layer 1 alert) |
| action | Plan bearing inspection within 7 days. Do not operate at high load. |
| risk | Fatigue fracture if untreated. Sudden seizure at high load. |
 
> **Real-world conditions note (mandatory in every alert):**
> *"This advisory is based on physics-synthetic training data anchored to CIRA SACIP. Real-world conditions including pipe corrosion, fluid contamination, cross-pump hydraulic interactions, and installation geometry may affect sensor signatures. All alerts require confirmation by a certified process engineer before maintenance."*
 
---
 
## Stage 3 — Fuzzy Membership Layer (Level 1 output)
 
**Purpose:** Convert continuous Level 1 MAE value to fault probability [0, 1]. Handles the transition zone between clearly normal and clearly fault. Captures early-stage faults where MAE hovers near threshold.
 
### Calibration Protocol
 
**Normal population:**
- Full 9,711-window real CIRA normal pool [Finding F6 — never use probe subset]
- P95 of normal MAE distribution → `lower_bound` (fuzzy onset)
- Expected range: [0.07, 0.09]
**Fault population — selective exclusion (physics-derived):**
 
| Action | Class | Reason |
|--------|-------|--------|
| **EXCLUDE** | overloading mild (sev 0.2–0.5) | MAE=0.093 sub-threshold [Finding F1]. Handled by Mech C (Temp.SV drift), NOT fuzzy layer. |
| **EXCLUDE** | seal_failure mild (sev 0.2–0.4) | MAE near 0.12, too close to normal boundary [Finding F2]. Handled by Mech C (Pres.SV drift), NOT fuzzy layer. |
| **EXCLUDE** | bearing_wear_gradual ALL severities (label 21, sev 0.05–0.25) | Paris-Erdogan low-dK regime — MAE sub-threshold by design [Finding F7]. Handled by TCN-AE score_B → CUSUM + score_A → Rolling Baseline. Layer 1 fuzzy layer CANNOT detect label 21 — CORRECT, NOT a failure. |
| **INCLUDE** | cavitation ALL severities | MAE = 0.675 |
| **INCLUDE** | bearing_wear ALL severities (standard, not gradual) | — |
| **INCLUDE** | sensor_failure ALL severities | — |
| **INCLUDE** | impeller_imbalance ALL severities | — |
| **INCLUDE** | overloading severe (sev 0.5–1.0) only | — |
| **INCLUDE** | seal_failure severe (sev 0.5–1.0) only | — |
| **INCLUDE** | ALL Group B compound sequences (labels 7–12) | — |
| **INCLUDE** | ALL Group C masked sequences (labels 13–17) | — |
| **INCLUDE** | ALL Group D variant sequences EXCEPT label 21 (labels 18–20) | — |
| **INCLUDE** | ALL Group E multi-sensor sequences | — |
 
P5 of included fault MAE distribution → `upper_bound`. Expected range: [0.15, 0.50] (dominated by cavitation MAE=0.675).
 
### Fuzzy Function
 
```
mu_fault(e) = 0.0                                      if e < lower_bound
            = (e - lower_bound) / (upper - lower)      if lower <= e <= upper
            = 1.0                                      if e > upper_bound
```
 
> **Why exclusion is critical:** Including overloading/seal/label-21 mild sequences drags `upper_bound` DOWN. Narrower fuzzy zone → more windows in transition → higher FPR. On a 110 kW asset: elevated FPR = operators ignore alerts = missed real faults. Excluded sequences are NOT lost — Mech C + TCN-AE catch them via their specific paths. **This exclusion is physics-driven, not ad-hoc.**
 
---
 
## Known Limitations Registry
 
| # | Limitation | Current Mitigation | Long-Term Roadmap |
|---|-----------|-------------------|-------------------|
| 1 | Static threshold 0.110058 calibrated on M4 normal CIRA data. As pump ages, normal operating envelope shifts → threshold becomes stale. | Level 2 adaptive threshold (score_A rolling baseline). | Periodic M3 re-normalization + threshold recalibration. |
| 2 | Synthetic-to-real domain gap. All fault sequences are physics-synthetic anchored to CIRA. Real faults may have additional noise, cross-pump hydraulic interactions, or sensor placement effects. | CIRA anchor rule + physics gates G1–G11. | Active learning queue in M10 (operator-confirmed alerts feed retraining). |
| 3 | 1 Hz sampling rate. Blade pass frequency (BPF) at 2980 RPM, 7 impellers = 348 Hz. 1 Hz captures only low-frequency envelope of BPF — not spectral content. | Envelope statistics (mean, std, kurtosis) capture BPF signatures. | 10 Hz or higher sampling if sensor upgrade available. |
| 4 | 8-sensor coverage. Pipe corrosion, cross-pump hydraulic interactions, lubrication degradation, and shaft misalignment are not detectable from 8 sensors at 1 Hz. These appear in Physics Context Layer (Layer 2 advisory) only. | — | Expand sensor set to include flow meter, shaft proximity probe. |
| 5 | Label 21 (`bearing_wear_gradual`) detection latency. Earliest reliable detection = ~Week 5 (Layer 4 Rolling Baseline slope shift). For bearing cracks propagating faster than Paris law low-dK regime, detection may occur later than ideal. | 2,000 training sequences + 4-layer cascade. | Acoustic emission sensor for sub-threshold crack detection. |
| 6 | Household pump out-of-distribution warning. Model trained exclusively on industrial 110 kW multistage centrifugal pump data. `if pump_type == 'household': return physics_advisory_only()` Applying M8 to household monoblock pumps = OOD inference = **safety risk**. | Check enforced at M10 API layer — see `modules_M9_M10_M11_deployment.md`. | — |
 
---
 
## M8 Training Data Summary
 
**Input to M8 training:**
- `M6B_combined_sequences.pkl`: ~31,800 sequences across 22 classes
- z_t sequences (N_windows × 64): exported by M6B after M4 LSTM-AE sliding window
- M6.5r feature matrix: ~196,000 rows × ~35 columns (XGBoost input)
- M4 model weights: `lstm_ae_baseline.pth` (FROZEN — Level 1 pre-trained)
**Level 1 training:** Fine-tune LSTM-AE on M6B normal sequences (Group A label 0: 2,000 sequences). Freeze M4 weights as starting point; fine-tune encoder only. Validate: normal reconstruction MAE < 0.110058 on CIRA holdout (9,711 windows).
 
**Level 2 training:** Train TCN-AE on z_t sequences from ALL 22 classes. Loss: reconstruction loss on z_t + score_A/B/C regression head losses. Validate per gate set (Part 2).
 
---
 
## Document Revision History
 
| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Original monolithic file — bias-audit updates incorporated. |
| v2.0 | 2026-04-15 | Split into Part 1 + Part 2. M7 prerequisite 10→15 gates. |
| v3.0 | 2026-04-16 | v14.0: Part 1A + Part 1B split. 22-class. Label 21 detection layers. Finding F7. Fuzzy exclusion label 21. Gate M8-14-ext noted. |
| v4.0 | 2026-04-19 | v14.2 TCN-AE ARCHITECTURE: Level 2 LSTM v2 REPLACED by 5-layer dilated causal TCN-AE. TCN-AE architecture fully specified. score_A/B/C derivation and routing rules added. Adaptive threshold formula specified (Level 2 only). Two-speed adaptation explanation added. Physics Context Layer section added (all 22 classes, 8 fields). Known Limitations Registry added (6 limitations + roadmap). M8 Two-Level Architecture Overview section added. Static threshold 0.110058 clearly marked Level 1 ONLY. Real-world conditions disclaimer added. Training data summary updated to ~31,800 sequences + z_t inputs. |
 
---
 
> **GitHub is the ONLY source of truth for this spec.**
>
> - Part 1B (TCN Mechanisms): `module_M8_lstm_ae_v2_mechanisms.md`
> - Part 2 (Gates + Outputs): `module_M8_lstm_ae_v2_gates_and_outputs.md`
>
> **Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset
> **Standard:** ISO 10816-3 vibration, ISO 13373-3 condition monitoring
> **Canonical source of truth:** `pasted-text.txt` (v14.2, 2026-04-19)
