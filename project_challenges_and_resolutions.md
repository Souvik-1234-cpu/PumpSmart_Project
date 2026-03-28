# PumpSmart — Project Challenges & Resolutions
# Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring
# Complete Audit Trail: M1 through M4
# Asset: 110 kW | 7-stage | 40 bar | 450 m head | 2980 RPM | 45 m³/h
# Document version: 1.0 | Date: 2026-03-28 | Author: Souvik

---

## Purpose of This Document

This document records every **non-trivial engineering or ML challenge** encountered
during the PumpSmart project from M1 through M4 — excluding syntax errors and
environment setup. Each entry includes:

- The module where it was discovered
- The exact nature of the problem (physics, thermodynamics, transport phenomena,
  ISO standard, ML architecture, data integrity)
- Why it is a problem (the physical or mathematical reasoning)
- The resolution adopted and why it is correct

This document serves as a technical audit trail for presentation, publication,
and onboarding. It demonstrates that every architectural decision was physically
justified, not arbitrary.

---

## Challenge Index

| # | Module | Category | Challenge Title | Severity |
|---|---|---|---|---|
| C-01 | M1 | Data Integrity | Pump_B_Day3 Sensor Failure vs Data Corruption | HIGH |
| C-02 | M1 | Data Integrity | Gap Threshold Inconsistency Across Day Files | MEDIUM |
| C-03 | M1 | Data Integrity | Pump_C_Day3 Exclusion Decision | MEDIUM |
| C-04 | M2 | Physics — Thermodynamics | Startup Cluster Higher Temperature Than High-Load | HIGH |
| C-05 | M2 | Physics — ISO Standards | .SV Channel Misidentification (ISO 10816-3) | CRITICAL |
| C-06 | M2 | Physics — Transport Phenomena | Pressure Range Paradox in Startup Cluster | HIGH |
| C-07 | M2 | Data Documentation | No Unit Registry for Sensor Channels | MEDIUM |
| C-08 | M2 | Physics — ISO Standards | M2 Plot Y-Axes Without Physical Units | LOW |
| C-09 | M3 | Physics — Thermodynamics | Flash Evaporative Cooling Producing ~20,000 Negative ΔT* Values | CRITICAL |
| C-10 | M3 | Physics — Transport Phenomena | Climate-Dependent Ambient Reference Making Model Non-Portable | HIGH |
| C-11 | M3 | Data Integrity | False Alarm Flags on Pressure Normalization Range | MEDIUM |
| C-12 | M3 | Physics | unit_registry.json Premature Specification | LOW |
| C-13 | M4 | ML — Architecture | LSTM-AE Decoder Collapse (v1) | CRITICAL |
| C-14 | M4 | ML — Architecture | Physics-Naive Equal-Weight Loss Function | HIGH |
| C-15 | M4 | ML — Training | Threshold Enormously High: 0.645 → Separation 12.9× (False Sense of Security) | HIGH |
| C-16 | M4 | ML — Training | Spike Windows Contaminating Normal Validation Set | CRITICAL |
| C-17 | M4 | Physics — ISO 13373-3 | Pmp.PV Startup Winsorization Ceiling Physically Too Low | HIGH |
| C-18 | M4 | Physics — Fluid Mechanics | Pressure Winsorization Using Global σ Across Clusters | CRITICAL |
| C-19 | M4 | Physics — Fluid Mechanics | pressure_transient Spike Ratio Referenced to Wrong Cluster Mean | HIGH |
| C-20 | M4 | ML — Architecture | LSTM Phase-Lag on Mode Transitions (Deferred Flag) | MEDIUM |
| C-21 | M4 | ML — Architecture | Single Global Threshold Across All Operating Modes (Deferred Flag) | MEDIUM |
| C-22 | Cross-Module | Physics | Motor Power Misidentified as 10 kW Instead of 110 kW | CRITICAL |

---

## C-01 — Pump_B_Day3: Sensor Failure vs Data Corruption

**Module:** M1 — Data Ingestion & Cleaning
**Category:** Data Integrity / Physics of Sensor Systems
**Severity:** HIGH

### The Problem

Pump_B_Day3 had 65.5% rows with NaN values in the Barometer and Temperature
columns. The naive interpretation was to discard the entire file as corrupted
data.

### Why This Matters Physically

There is a critical distinction between:
- **Sensor failure**: The sensor hardware fails while the pump continues to
  operate normally. The NaN rows are a hardware event, not a pump event.
- **Process failure**: The pump itself fails, producing meaningless readings.

Pump_B_Day3 showed continuous, physically consistent readings in all 8 pump
sensor channels (Mot.PV, Mot.SV, Mot.TV, Pmp.PV, Pmp.SV, Pmp.TV, Temp.SV,
Pres.SV) throughout the NaN period. Only the environmental sensors (Barometer,
Temperature) went silent. This is the exact signature of a barometric sensor
hardware dropout — the pump ran normally while the sensor stopped logging.

This is directly analogous to Fault Type C (Sensor Dropout) which M6 will
synthetically generate. Discarding the file would have destroyed valid training
data AND misclassified sensor failure as process failure — a safety error.

### Resolution

Hard drop all rows with any NaN (policy unchanged — no interpolation). After
dropping, re-segment by timestamp gap detection. The clean segments before and
after the NaN block are structurally valid. Two usable segments recovered from
B_Day3. Post-dropout segments assigned 600-row warmup (vs 300 standard) to
allow sensor equilibration before windowing.

**Lesson locked:** Sensor failure ≠ process failure. Physical context must be
applied before discarding data. This distinction is now a formal fault type in M6.

---

## C-02 — Gap Threshold Inconsistency Across Day Files

**Module:** M1 — Data Ingestion & Cleaning
**Category:** Data Integrity / Time Series Segmentation
**Severity:** MEDIUM

### The Problem

A single global gap threshold applied uniformly across all 9 files produced
incorrect segment splits. Day1 files had natural timestamp jitter up to ~5s
between consecutive records despite being 1s nominal sampling. Day2/Day3 files
were perfectly continuous at 1s.

Applying a 2s threshold to Day1 files would fragment single continuous
operational runs into dozens of unusable micro-segments. Applying an 8s
threshold to Day2/Day3 would merge genuinely separate operational sessions.

### Physics Basis

The gap threshold defines what constitutes a "new operating state" vs natural
measurement noise. For a 110 kW, 2980 RPM pump:
- A 5s gap in a Day1 file at 1s sampling = SCADA clock drift, not a pump event
- A 5s gap in a Day3 perfectly-continuous file = genuine process interruption
  (startup–shutdown boundary)

Using the same threshold for both = either over-segmenting Day1 (losing training
data) or under-segmenting Day3 (creating windows across operational boundaries —
a time-series integrity violation).

### Resolution

File-specific gap thresholds derived from the actual median sampling interval
per file:
- Day1 files: gap threshold = 8s (8× median — above natural 1s jitter)
- Day2/Day3 files: gap threshold = 2s (2× median — catches operational pauses)

This preserved 27 usable segments vs the ~8 that uniform thresholding would
have produced.

---

## C-03 — Pump_C_Day3 Exclusion Decision

**Module:** M1 — Data Ingestion & Cleaning
**Category:** Data Integrity
**Severity:** MEDIUM

### The Problem

Pump_C_Day3 initially appeared to have valid data. However, cross-referencing
with B_Day3 revealed 100% Barometer corruption in linked session files. Unlike
B_Day3 where the pump data remained physically consistent, C_Day3 showed
anomalous behaviour in process sensor channels correlated with the environmental
sensor failure period — suggesting the NaN propagation affected more than just
the barometric sensor.

### Resolution

C_Day3 excluded entirely from the training pipeline. Segments from 8 remaining
files (25 usable segments) provide sufficient representation across all 4
operating modes. Exclusion documented in segment_registry.csv with reason flag.

---

## C-04 — Startup Cluster Higher Temperature Than High-Load

**Module:** M2 — EDA & Clustering
**Category:** Physics — Thermodynamics (Heat Transfer)
**Severity:** HIGH

### The Problem

K-Means clustering assigned clusters, and the temperature analysis showed:
- Startup cluster mean Pmp.TV: **41.9°C**
- High-load cluster mean Pmp.TV: **39.5°C**

The initial script comment flagged this as physically suspicious — conventional
understanding suggests high-load = high temperature. If interpreted as an error,
it would have prompted a re-clustering attempt that would have destroyed valid
operating mode identification.

### Physics Explanation — Thermodynamics

This is correct physics for a 7-stage multistage centrifugal pump at 2980 RPM.

The **thermal time constant** for a 110 kW IEC 315 frame motor is approximately:
```
τ = m·Cp / (h·A) ≈ 400–600 seconds
```

The pump operates in cycles: startup → steady-state → high-load → cooldown.
When the pump enters the startup cluster, it is coming out of a warm cooldown
phase. The motor casing retains heat from the previous operational cycle. The
pump has NOT yet reached hydraulic full-load (affinity law: Q ∝ N, H ∝ N² —
flow and head build up from zero), so shaft power is low, but **residual
thermal mass** from the prior cycle keeps casing temperature elevated.

By the time the pump reaches high-load, the convective heat transfer from
forced airflow over the motor casing (and process fluid cooling in the pump
casing) has partially offset the increased shaft heat generation. Net result:
high-load steady-state temperature is LOWER than the peak thermal transient
during startup run-in.

This has a direct consequence for M5 (Overloading fault): defining overloading
as "temperature above high-load maximum" would be WRONG. Overloading is defined
as monotonically rising temperature DURING stable vibration — a temporal pattern,
not an absolute threshold.

### Resolution

Script comment corrected (FIX-3 in M2 audit). Cluster assignment validated.
No re-clustering. Overloading fault definition revised in M5 specification to
use temporal rate-of-change, not absolute temperature threshold.

---

## C-05 — .SV Channel Misidentification (ISO 10816-3 Violation)

**Module:** M2 — EDA & Clustering
**Category:** Physics — ISO Standards / Vibration Measurement
**Severity:** CRITICAL

### The Problem

The CIRA dataset documentation and initial project descriptions labeled the
.SV channels (X_ACR_Mot.SV, X_ACR_Pmp.SV) as single-tone ISO acceleration
values in m/s². This led to early attempts to apply ISO 10816-3 absolute
alarm thresholds directly:
- Zone A (new machine): < 2.3 mm/s
- Zone B (acceptable): 2.3–4.5 mm/s
- Zone C (alarm): 4.5–7.1 mm/s
- Zone D (danger): > 7.1 mm/s

The data showed Motor SV values at:
- Startup: 0.39–0.58 mm/s (well below Zone A)
- High-load: 22.2–77.7 mm/s (far above Zone D)

If these were true ISO 10816-3 velocity values, the pump would permanently
be in danger zone during high-load operation — physically implausible for a
well-maintained industrial pump in service.

### Physics Diagnosis

Cross-referencing the CIRA SACIP technical documentation confirmed these are
**broadband peak acceleration envelopes** — a SCADA composite metric that
integrates across multiple frequency bands and takes the peak envelope value.
This is NOT the same as ISO 10816-3 velocity RMS. The scale and units are
different. The absolute values cannot be compared to ISO 10816 tables.

Furthermore, the Mot.SV transient spike of **456.6 mm/s** in the dataset
(high-load cluster) is consistent with a broadband peak envelope during an
impulsive event (bearing impact or cavitation shock) — not a physically
sustained velocity that would destroy any known pump bearing.

### Resolution

All fault detection re-defined as **relative change from cluster mean**,
not absolute ISO threshold comparison:
- Normal: SV* = 0.8–1.2 (near cluster mean)
- Early fault: SV* = 1.3–1.5
- Active fault: SV* = 1.5–2.5
- Severe: SV* > 2.5

ISO 10816-3 retained as a **reference framework** for fault severity
classification in M5/M6 but NOT applied as absolute cutoffs.
Channel weights in M4 set high (2.0) for SV channels to preserve their
relative fault-detection value while removing absolute-threshold dependence.
This is documented in the pathway as a locked architectural decision.

---

## C-06 — Pressure Range Paradox in Startup Cluster

**Module:** M2 — EDA & Clustering
**Category:** Physics — Fluid Mechanics / Transport Phenomena
**Severity:** HIGH

### The Problem

The startup cluster showed outlet pressure ranging from **0.43 to 0.85 bar**,
while the cooldown cluster showed **0.45 to 44.4 bar**. The cooldown cluster
therefore contains both very low AND very high pressure states.

This appeared contradictory: how can a pump in cooldown (shutting down) have
pressures up to 44.4 bar, higher than steady-state (0.69–43.3 bar)?

### Physics Explanation — Fluid Mechanics (Joukowsky Water Hammer)

When a 110 kW pump at 40 bar operational pressure shuts down, the rapid
deceleration of the fluid column creates a **water hammer pressure surge**
(Joukowsky equation):
```
ΔP = ρ·c·Δv
```
where c = acoustic wave speed in pressurized fluid (~1200 m/s for water), and
Δv = velocity change at shutoff. For a 45 m³/h flow rate in the pump discharge
line, this produces transient pressure spikes well above steady-state.

This is why cooldown cluster has higher pressure than steady-state cluster:
the cooldown cluster **includes the moment of shutdown** when water hammer
transiently pushes discharge pressure above the steady-state 40 bar nameplate.
The observed transient maximum of 46.7 bar is consistent with Joukowsky
predictions for this system.

The startup cluster has LOW pressure (0.43–0.85 bar) because it captures the
moment the pump comes online from rest — pressure builds from atmospheric as
the impellers accelerate. This is also the **NPSH-critical zone**: at these
low pressures, NPSHa may fall below NPSHr, creating cavitation risk.

### Resolution

Cluster assignments confirmed as physically correct. Pressure transient data
preserved in cooldown cluster (not filtered as outliers). Startup cluster
flagged as the **cavitation risk zone** — M5 physics engine specifies that
all cavitation fault sequences must begin in startup cluster context.
Joukowsky bounds used to set the Pres.SV winsorization ceiling in M4.

---

## C-07 — No Unit Registry for Sensor Channels

**Module:** M2 — EDA & Clustering (discovered), Fixed M2 Audit
**Category:** Data Documentation / ISO Compliance
**Severity:** MEDIUM

### The Problem

All cluster bounds in M2_cluster_bounds.csv were stored as raw numbers with no
unit documentation. Column names like X_ACR_Mot.SV gave no indication of
whether values were in m/s, mm/s, m/s², or mm/s². This created ambiguity in
every subsequent module that reads this file.

For a safety-critical application on a 110 kW pump, unit ambiguity in sensor
data is unacceptable — a confusion between mm/s and m/s represents a factor
of 1000 error in vibration severity assessment.

### Resolution

Created M2_cluster_bounds_units.json as authoritative unit registry:

| Channel | Unit | Standard |
|---|---|---|
| X_ACR_Mot.PV | mm (displacement) | ISO 10816-3 |
| X_ACR_Mot.SV | mm/s (broadband peak envelope) | ISO 10816-3 |
| X_ACR_Mot.TV | °C | IEC 60034-1 |
| X_ACR_Pmp.PV | mm | ISO 10816-3 |
| X_ACR_Pmp.SV | mm/s | ISO 10816-3 |
| X_ACR_Pmp.TV | °C | IEC 60034-1 |
| X_Temp.SV | °C | ISO 13373-2 |
| X_Pres.SV | bar | ISO 5167 |

All downstream modules reference this file. M5 will extend it with
fault-specific unit metadata in unit_registry.json.

---

## C-08 — M2 Plot Y-Axes Without Physical Units

**Module:** M2 — EDA & Clustering
**Category:** Documentation / ISO Compliance
**Severity:** LOW

### The Problem

All M2 time-series plots had raw column names on Y-axes (e.g., "X_ACR_Mot.SV")
with no unit labels. For a project targeting maintenance engineers and
plant operators, unlabelled axes are a presentation failure and violate
basic engineering documentation standards.

### Resolution

All M2 time-series plots regenerated with proper unit labels:
"X_ACR_Mot.SV (mm/s — broadband peak)", "X_Pres.SV (bar)", etc.
FIX-2 in M2 audit record.

---

## C-09 — Flash Evaporative Cooling Producing ~20,000 Negative ΔT* Values

**Module:** M3 — Dimensionless Normalization
**Category:** Physics — Thermodynamics / Phase Change (Transport Phenomena)
**Severity:** CRITICAL

### The Problem

The initial temperature normalization formula (v5 specification) used a
per-row ambient temperature reference:
```
ΔT* = (T - T_ambient_per_row) / (T_cluster_max - T_ambient_per_row)
```

When this was executed during M3, approximately **20,000 rows** in the cooldown
cluster produced **negative ΔT* values** (down to −0.113). The immediate
interpretation was a formula error — temperatures should be above ambient,
so negative normalized values should not exist.

### Physics Explanation — Thermodynamics (Phase Change / Evaporative Cooling)

This was NOT a formula error. It is a real physical phenomenon:

When a 40 bar pressurized pump shuts down, the internal fluid is at high
pressure and temperature (~40–50°C). As discharge valves close and pressure
drops during cooldown, the pressurized fluid undergoes **flash evaporation** —
a fraction of the fluid instantaneously vaporizes as pressure drops below the
saturation pressure at the local temperature. This adiabatic phase change
absorbs latent heat from the surrounding metal:

```
Q_evap = ṁ_flash × h_fg   (h_fg ≈ 2257 kJ/kg for water at 100°C)
```

The net effect: pump casing metal temperature drops **BELOW ambient air
temperature** within minutes of shutdown. For CIRA data:
- Ambient temperature (Naples, Italy): 19.0–28.8°C
- Cooldown cluster sensor minimum: 17.6°C
- Sub-ambient: (17.6 - 19.0) = −1.4°C below ambient, lasting 5–15 minutes

This is the same physics as a refrigerator evaporator coil — pressurised
fluid expands → cools below surroundings. The 20,000 negative rows are
entirely correct physical data, not errors.

**The bug was in the normalization formula, not the data.**

The ambient-relative formula breaks when the physical process being measured
can cool below the reference baseline. The formula implicitly assumes:
T_actual ≥ T_ambient always — a false assumption for phase-change systems.

### Resolution

Temperature normalization formula changed to **cluster-relative** (v6/M3 re-run):
```
ΔT* = (T - T_cluster_min) / (T_cluster_max - T_cluster_min)
```

T_cluster_min is the coldest observed temperature in each operating mode —
it represents the physical floor of that mode including flash evaporative
events. The sub-ambient readings (18 extreme rows) produce small negative
values (−0.02 to −0.05) under the new formula, which is acceptable and
physically meaningful. They are preserved — not clipped. The LSTM-AE must
learn these as valid extreme-normal cooldown states.

**Secondary benefit:** The cluster-relative formula is **climate-agnostic**.
The ambient-relative formula was calibrated to Naples, Italy (19–28°C ambient).
Deploying on a pump in Kolkata, India (22–38°C ambient) would shift the entire
temperature baseline, producing systematically different normalized values for
the same physical pump state. The cluster-relative formula uses the pump's own
operational envelope as reference — it works identically regardless of
installation climate.

---

## C-10 — Climate-Dependent Ambient Reference Breaking Model Portability

**Module:** M3 — Dimensionless Normalization
**Category:** Physics — Transport Phenomena / Deployment Engineering
**Severity:** HIGH

### The Problem

Directly related to C-09. Even if the flash evaporation issue did not exist,
the ambient-relative temperature formula creates a deployment portability
problem. A model trained on CIRA data from Naples (ambient 19–28°C) would
be miscalibrated when deployed on the same pump type in a different climate
(Kolkata: 22–38°C; industrial desert facility: 35–48°C).

The normalized temperature at the SAME physical pump state would be different
depending on where the model is deployed — making the learned normality
baseline geography-dependent.

### Resolution

Identical to C-09 — cluster-relative formula. Documented as a locked
architectural decision in the pathway. The model's normality baseline
is the pump's own operational physics, not the local weather.

---

## C-11 — False Alarm Flags on Pressure Normalization Range

**Module:** M3 — Dimensionless Normalization
**Category:** Data Integrity / Statistical Misinterpretation
**Severity:** MEDIUM

### The Problem

After applying ratio normalization (P* = P_actual / P_cluster_mean), the
output statistics showed ~45.5% of Pres.SV_norm values above 1.0. An
automated range-check script flagged this as an anomaly: "45% of values
outside normal range [0,1] — potential normalization error."

### Physics Explanation — Statistics

This is a mathematical certainty, not an error. Ratio normalization around the
cluster mean (P* = P / P_mean) produces a distribution where:
- Mean = 1.0 by definition
- All values below the mean produce P* < 1.0
- All values above the mean produce P* > 1.0
- For a right-skewed pressure distribution: ~45–55% of values will be above 1.0

The [0,1] range is not a hard constraint for pressure and vibration channels —
it is the **expected operating zone** where 1.0 = cluster mean (normal operating
point). The LSTM-AE is trained to reconstruct patterns in this space. Values
above 1.0 are not automatically anomalous — they are above-mean normal
operation. Values significantly above 1.0 (>1.5) AND with anomalous temporal
patterns = anomalous.

The only channels that hard-clip to [0,1] are the temperature channels,
because the cluster-relative formula uses min/max bounds.

### Resolution

Range-check logic corrected to distinguish channel types:
- Pressure/vibration channels: range CHECK is informational only; values above
  1.0 are expected and normal
- Temperature channels: hard range [−0.1, 1.05] expected; violations warrant
  investigation

False alarm flags removed from M3 report. Correct statistical interpretation
documented.

---

## C-12 — unit_registry.json Premature Specification in M3

**Module:** M3 — Normalization
**Category:** Project Architecture
**Severity:** LOW

### The Problem

The v5 pathway document specified that unit_registry.json should be created
as an M3 output. During M3 execution, it became clear this was premature:
the physics engine in M5 would define fault-specific unit metadata
(ISO alarm thresholds, coupling coefficients, fault envelope boundaries)
that require knowledge not yet available at M3 time.

Creating a partial unit registry in M3 and then extending it in M5 creates
a fragmented, hard-to-maintain documentation artifact.

### Resolution

unit_registry.json creation deferred to M5. M3 produces M3_normalization_config.json
(normalization baselines only). M5 produces unit_registry.json as its first
output step, with complete metadata for all channels including fault context.

---

## C-13 — LSTM-AE Decoder Collapse (v1)

**Module:** M4 — LSTM-AE Baseline
**Category:** ML Architecture — LSTM Sequence-to-Sequence
**Severity:** CRITICAL

### The Problem

M4 v1 trained successfully (loss decreased), but the reconstruction sample
plots showed the decoder producing a **flat constant output** — approximately
the mean of the training data — for all input sequences regardless of content.
This is called decoder collapse: the model learns that outputting a constant
near the data mean minimises reconstruction loss on average, so it stops
learning to reconstruct individual sequences.

Key symptoms:
- val loss: 0.251 (appeared reasonable)
- Per-channel MAE: all channels showing near-identical errors
- Reconstruction plot: orange dashed line = horizontal constant across all 50 timesteps

### Root Cause

The decoder LSTM was initialized with **zero hidden state** (h0=0, c0=0).
The encoder produced a meaningful latent vector in its bottleneck, but the
decoder, starting from zero state, could not effectively use the latent
information to seed its sequential reconstruction. The decoder defaulted
to learning a global mean predictor, which is a local minimum for MSE-based
loss.

For a sequence-to-sequence autoencoder, the decoder's initial hidden state
must be seeded from the encoder's final hidden state — this is the
information bridge. Without it, the bottleneck is effectively disconnected.

### Resolution

**hidden_state_seeded decoder**: The encoder's final hidden state (h_n, c_n)
is passed directly as the decoder's initial state (h0, c0):
```python
# Encoder
_, (h_n, c_n) = self.encoder_lstm(x)
bottleneck = self.fc_encoder(h_n[-1])  # compress to 64 dims

# Decoder — seeded from encoder state
h0 = self.fc_decoder_h(bottleneck).unsqueeze(0).repeat(n_layers, 1, 1)
c0 = self.fc_decoder_c(bottleneck).unsqueeze(0).repeat(n_layers, 1, 1)
out, _ = self.decoder_lstm(decoder_input, (h0, c0))
```

Result: val MAE improved from 0.251 (v1) to 0.050 (v3) — a 5× improvement.
Threshold tightened from 1.362 to 0.645. Decoder collapse completely resolved.
This fix has been present from v3 onwards and is locked in the architecture.

---

## C-14 — Physics-Naive Equal-Weight Loss Function

**Module:** M4 — LSTM-AE Baseline
**Category:** ML Architecture — Loss Function Design
**Severity:** HIGH

### The Problem

The initial loss function weighted all 8 sensor channels equally. This is
physically naive: a 1 mm/s change in Mot.SV (vibration velocity) has
completely different fault significance than a 1°C change in Pmp.TV (contact
temperature).

Equal weighting means the model optimizes reconstruction quality by spending
capacity on whichever channels are easiest to reconstruct (typically slow-moving
temperature channels). The high-frequency, physically informative vibration
and pressure channels — which carry the earliest fault signatures — get
under-weighted.

### Physics Basis for Channel Importance Hierarchy

For a 110 kW, 2980 RPM, 7-stage pump:

1. **Vibration velocity (SV channels)**: Directly governed by ISO 10816-3.
   First to show bearing wear, imbalance, cavitation. Broadband RMS changes
   within 2–3 shaft rotations of fault onset. Placement-robust.

2. **Vibration displacement (PV channels)**: Related to SV by frequency
   content. Highly informative for impeller imbalance (1×RPM harmonic).
   Placement-robust.

3. **Discharge pressure (Pres.SV)**: Governed by pump affinity laws.
   Cavitation, seal failure, and impeller damage all manifest as pressure
   anomalies. Direct process variable.

4. **Casing temperature (Temp.SV)**: Governed by heat transfer:
   dT/dt = Q_loss/(m·Cp) - h·A·(T-T_amb)
   Lags fault onset by the thermal time constant τ ≈ 400–600s.
   Secondary fault indicator — important but slow.

5. **Contact temperatures (Mot.TV, Pmp.TV)**: Most placement-dependent.
   Valid for the CIRA installation geometry, but cannot be guaranteed
   for different sensor mounting configurations. Lowest fault-detection
   reliability for general deployment.

### Resolution

Physics-weighted channel loss function:
```python
channel_weights = {
    'Mot.SV': 2.0,  # placement-robust, ISO 10816-3 primary
    'Pmp.SV': 2.0,  # placement-robust, ISO 10816-3 primary
    'Pres.SV': 2.0, # direct process variable
    'Mot.PV': 1.5,  # placement-robust, secondary vibration
    'Pmp.PV': 1.5,  # placement-robust, secondary vibration
    'Temp.SV': 1.0, # single casing sensor, thermal lag
    'Mot.TV': 0.8,  # placement-dependent
    'Pmp.TV': 0.8   # placement-dependent
}
total_loss = 0.6×MAE + 0.4×MSE  (physics-weighted per-channel)
```

The 0.6/0.4 MAE/MSE split: MAE is robust to outliers (better for spike-heavy
vibration data), MSE penalises large deviations more heavily (important for
catching pressure drops and thermal runaway). Combined loss captures both
failure modes.

---

## C-15 — Threshold 0.645 Giving 12.9× Separation (False Sense of Security)

**Module:** M4 — LSTM-AE Baseline (v3/v4/v5/v6)
**Category:** ML Training / Threshold Calibration
**Severity:** HIGH

### The Problem

M4 versions v3 through v6 produced an anomaly threshold of 0.645 with a
separation ratio of 12.9× (threshold / mean MAE = 0.645 / 0.050). While this
appeared excellent, it was discovered that the validation set was **contaminated
with spike windows** — windows containing genuine transient fault events that
elevated the mean MAE and pushed the threshold unrealistically high.

A threshold of 0.645 means the model requires a fault to elevate reconstruction
error by 12.9× before raising an alarm. For a 110 kW, 40 bar pump, early-stage
bearing wear or incipient cavitation might only elevate MAE by 3–5× before
becoming dangerous. A 12.9× threshold would miss these entirely.

The high threshold was not evidence of model quality — it was evidence that the
"normal" baseline was polluted with anomalous events, artificially elevating the
mean and pulling the threshold up with it.

### Resolution

**Spike row exclusion before windowing** (v7/v8): All windows containing rows
where any channel exceeded cluster-conditional winsorization bounds (i.e., the
spike rows) were excluded from both training and validation pools.

The result:
- 12,620 spike rows excluded from training data
- 1,044 spike windows saved separately as M4_spike_seeds.npy for M6
- Clean val set of 1,457 normal-only windows
- Recalibrated threshold: **0.110058** (mean=0.0268, separation=4.11×)

A 4.11× separation on a clean normal baseline is more honest and more
sensitive than 12.9× on a contaminated baseline. The model now requires only
4.11× elevation to trigger an alarm — appropriate for early-fault detection
on a high-value asset.

---

## C-16 — Spike Windows Contaminating Normal Validation Set

**Module:** M4 — LSTM-AE Baseline (v6→v7 fix)
**Category:** ML Training — Data Pipeline Integrity
**Severity:** CRITICAL

### The Problem

The fundamental purpose of an LSTM-AE anomaly detector is to learn the normal
operating manifold and flag deviations. The validation set must contain ONLY
clean normal operation. If spike windows (containing real transient fault events)
enter the validation set, the model learns that these spikes are "normal" —
directly undermining the detection objective.

In v1–v6, the training/validation split was performed on the full window pool
before any spike filtering. This meant ~8% of validation windows contained
genuine anomalous events. The threshold was therefore calibrated not on the
normal manifold but on a contaminated distribution.

For a 110 kW industrial pump, this is a safety issue: the model was unknowingly
trained to consider certain fault events as normal.

### Resolution

Strict two-phase pipeline in v7/v8:
1. **PHASE 1**: Scan all rows, identify spike rows (per-channel, per-cluster bounds)
2. **PHASE 2.5**: Drop ALL spike rows from the dataframe BEFORE windowing
3. **PHASE 3**: Window the cleaned dataframe → clean window pool
4. **PHASE 4**: Save spike rows separately as M4_spike_seeds for M6
5. **PHASE 5**: Train/val split on clean pool only

Spike rows are not wasted — they become the real-data seeds for synthetic
fault generation in M6. Every anomalous event in the dataset has been
deliberately preserved and redirected to its correct use: fault examples,
not normal examples.

---

## C-17 — Pmp.PV Startup Winsorization Ceiling Physically Too Low

**Module:** M4 — LSTM-AE Baseline (v7→v8 fix)
**Category:** Physics — ISO 13373-3 / Vibration Mechanics
**Severity:** HIGH

### The Problem

In M4 v7, the Pmp.PV (pump casing displacement) winsorization ceiling was
set uniformly at 2.6× the cluster mean across all clusters including startup.
During v8 physics audit, this was identified as physically incorrect.

During pump startup at 2980 RPM, 7 impellers:
- Blade Passing Frequency (BPF) = RPM/60 × N_blades = 2980/60 × 7 = **348.7 Hz**
- During ramp-up from 0 to 2980 RPM, the rotational frequency sweeps through
  ALL resonant frequencies of the pump casing and piping system
- ISO 13373-3 documents that BPF harmonics during speed ramp-up can reach
  **2–4× the steady-state displacement amplitude**

A 2.6× ceiling during startup would cause the winsorizer to clip legitimate
startup resonance events as spikes, mislabeling them as fault seeds in M4
and injecting physically incorrect fault data into M6.

### Resolution

Startup cluster Pmp.PV ceiling raised to **3.2×** (ISO 13373-3 BPF harmonic
headroom). All other clusters remain at 2.6×. This is now stored in
M4_spike_config.json as a cluster-conditional ceiling. M6 reads this file
and cannot override it.

---

## C-18 — Pressure Winsorization Using Global σ Across Clusters

**Module:** M4 — LSTM-AE Baseline (v7→v8 fix)
**Category:** Physics — Fluid Mechanics / Dimensional Analysis
**Severity:** CRITICAL

### The Problem

M4 v7 used a uniform pressure winsorization ceiling derived from the global
pressure standard deviation across all clusters. The global pressure std was
dominated by the wide multi-modal distribution across clusters.

The fundamental problem:

| Cluster | Pres.SV mean | Pres.SV std |
|---|---|---|
| Startup | 0.621 bar | 1.03 bar (normalized to mean) |
| High-load | 42.0 bar | 1.92 bar |
| Steady-state | 35.8 bar | 13.0 bar |

A pressure deviation of 5 bar during startup (P = 0.43–0.85 bar) is physically
catastrophic — it represents a Joukowsky water hammer exceeding the entire
operating pressure range. The same 5 bar deviation during high-load (P ≈ 42 bar)
is a routine ±12% fluctuation well within normal operation.

Applying a global σ-based ceiling treats these identically — a critical physics
violation. It either misses genuine startup pressure transients (ceiling too
high) or flags normal high-load pressure variation as faults (ceiling too low).

This is dimensionally analogous to trying to use a single Reynolds number
criterion for both laminar pipe flow and turbulent jet flow simultaneously —
the physics are qualitatively different.

### Resolution

**Cluster-conditional pressure winsorization** in v8:
```
Startup      : ceiling = 3.0× cluster mean  (Joukowsky transient headroom)
Steady-state : ceiling = 5.6× cluster mean  (wide valid operating range)
High-load    : ceiling = 2.0× cluster mean  (tight — any deviation is fault)
Cooldown     : ceiling = 3.0× cluster mean  (depressurization transients)
```

The high-load ceiling of 2.0× reflects the physics: a 110 kW pump at 40 bar
steady high-load operation has near-constant pressure. Any deviation >2× the
expected pressure is a genuine fault event (water hammer, valve closure, seal
failure). The tighter ceiling means fault detection is MORE sensitive at
high-load — exactly correct for a high-consequence operating state.

---

## C-19 — pressure_transient Spike Ratio Referenced to Wrong Cluster Mean

**Module:** M4 — LSTM-AE Baseline (v7→v8 fix)
**Category:** Physics — Fluid Mechanics (Joukowsky Water Hammer)
**Severity:** HIGH

### The Problem

M4 v7 computed the spike ratio for pressure_transient fault seeds using
the startup cluster mean (0.621 bar) as denominator:
```
spike_ratio = P_spike / P_startup_mean = 46.7 / 0.621 = 75.2×
```

This produced spike ratios of 70–80× for pressure transient events. While
mathematically accurate, it is **physically meaningless** as a fault severity
indicator. A water hammer event in a 40 bar system does not derive its energy
from the startup pressure — it derives it from the HIGH-LOAD operating pressure
at the moment of shutoff.

Joukowsky equation:
```
ΔP = ρ·c·Δv
```
The momentum of the fluid column at shutoff is governed by the **high-load
flow conditions** (Q ≈ 45 m³/h at ~42 bar), not by the startup conditions.
Using the startup mean as denominator inflates the spike ratio by a factor of
42/0.621 = **67×** — a purely artefactual amplification.

M6 using these inflated spike ratios would generate synthetic pressure
transient faults with completely unphysical magnitudes.

### Resolution

pressure_transient spike ratio recalculated using **high-load cluster mean**
(42.0 bar) as reference:
```
spike_ratio = P_spike / P_highload_mean = 46.7 / 42.0 = 1.11×
```

This is physically correct: the Joukowsky pressure surge of 46.7 bar represents
a **+11% overpressure** relative to normal high-load operating pressure —
a realistic and physically significant transient. Documented in M4_spike_config.json.

---

## C-20 — LSTM Phase-Lag on Mode Transitions (Deferred to M8)

**Module:** M4 — LSTM-AE Baseline v8 (identified, deferred)
**Category:** ML Architecture — LSTM Temporal Dynamics
**Severity:** MEDIUM (M4 context) / HIGH (M8 context)

### The Problem

M4 v8 reconstruction sample plots showed phase-lag artifacts on Pmp.PV
(MAE=0.079) and Mot.PV (MAE=0.079) during operating mode transitions. The
LSTM reconstructed a smooth decay curve instead of tracking the sharp
step-change when the pump crosses from one operating regime to another.

Root cause: LSTM hidden state integrates temporal history by design. When the
pump transitions abruptly (startup→steady-state), the hidden state retains
memory of the prior regime for several timesteps — producing slow adaptation.

### Why Deferred to M8 (Not Fixed in M4)

For M4 (baseline), this behavior is actually **conservative**: the model
underestimates reconstruction quality on transitions → MAE is slightly elevated
→ less false-alarm risk on normal transitions. The 8 false alarms in the v8
val set confirm this is not causing alarm inflation.

More critically: the transition zone (startup→steady-state) at low pressure
(0.43–0.85 bar) is the **highest cavitation risk zone** for this pump. A model
that is slightly elevated in sensitivity during this transition is correct
behavior for a baseline safety system.

For M8 (production), this is inadequate. See C-21 and M8 Safety Mandate in
module_pathway_M1_to_M8_v7.md.

### M8 Resolution (Mandated)

Temporal attention + gradient penalty loss. Full specification in M8 Safety
Mandate section of module_pathway_M1_to_M8_v7.md.

---

## C-21 — Single Global Threshold Across All Operating Modes (Deferred to M8)

**Module:** M4 — LSTM-AE Baseline v8 (identified, deferred)
**Category:** ML Architecture — Threshold Calibration
**Severity:** MEDIUM (M4 context) / HIGH (M8 context)

### The Problem

M4 uses a single threshold of 0.110058 for all 4 operating modes. The
normalized reconstruction error distributions have different noise floors
per cluster:

| Cluster | Expected MAE noise floor | Physics reason |
|---|---|---|
| Startup | Relatively high | Rapid hydraulic and thermal changes |
| Steady-state | Low | Stable conditions, slowest dynamics |
| High-load | Very low | Tight pressure control, consistent vibration |
| Cooldown | Medium | Depressurization transients |

A single threshold that works for steady-state is too sensitive for startup
(false alarms) and not sensitive enough for high-load (missed early faults).
For a 110 kW pump where high-load is the maximum stress condition, reduced
sensitivity at high-load is a safety gap.

### Why Deferred to M8

M4 is a baseline — its false alarm count of 8 on 1,457 val windows (0.55%)
is acceptable for calibration purposes. Cluster-conditional thresholds require
sufficient labeled data per cluster for robust calibration, which M4's clean
window pool (post-spike-exclusion) barely supports.

### M8 Resolution (Mandated)

Cluster-conditional threshold calibration using M4_spike_config.json cluster
distributions as threshold priors. Full specification in M8 Safety Mandate.

---

## C-22 — Motor Power Misidentified as 10 kW Instead of 110 kW

**Module:** Cross-module (source: Zenodo documentation, propagated M1→M4)
**Category:** Physics — Nameplate Specification
**Severity:** CRITICAL

### The Problem

The CIRA SACIP Zenodo record (DOI: 10.5281/zenodo.15301820) contains a
reference to "10 kW" in its pump specification text. This was initially
used as the motor power in all early physics calculations, including
hydraulic power checks and thermodynamic balance estimates.

The error only became apparent during M5 physics engine planning when
the hydraulic power equation was applied:
```
P_hyd = ρgQH/η = 1000 × 9.81 × (45/3600) × 450 / 0.65 = ~55 kW
```

A 55 kW hydraulic power requirement CANNOT be delivered by a 10 kW motor.
This is a fundamental violation of energy conservation: the motor output
power must exceed the hydraulic power by at least 1/η ≈ 1.54×.
A 10 kW motor driving a 55 kW hydraulic load is physically impossible.

### Diagnosis

Cross-referencing the physical evidence:
- IEC Frame 315mm → standard motor frame for **90–200 kW** motors
- 400V, 2-pole, 2980 RPM → consistent with 110 kW industrial motors
- 7 impellers, 450 m head, 45 m³/h → requires ~55 kW hydraulic power
  → shaft power = 55/0.5 = **110 kW** (assuming 50% overall efficiency)
- The "10 kW" in the Zenodo text likely refers to a **sub-duty point** or
  a specific test condition, not the rated motor power

The nameplate is definitively: **Motor shaft power = 110 kW**.

### Impact on Completed Modules

- M1, M2, M3: No physics calculations used the motor power value directly.
  Zero impact on data pipeline.
- M4: Channel weights and loss function are dimensionless — no motor power
  dependency. Zero impact on model.
- M5 onwards: All hydraulic power, efficiency, and thermal calculations
  MUST use 110 kW. This is now hardcoded as a verified constant in the
  pathway document and Paste Text tracker.

### Resolution

Motor power corrected to 110 kW in all documentation, Paste Text, and
pathway file. A warning note added to the project specification:
"'10 kW' in Zenodo source = sub-duty test point. Nameplate motor = 110 kW.
Use 110 kW for ALL physics calculations in M5+."

---

## Summary Statistics

| Category | Count | Severity Distribution |
|---|---|---|
| Data Integrity / Segmentation | 4 | 1 HIGH, 2 MEDIUM, 1 LOW |
| Physics — Thermodynamics | 2 | 1 CRITICAL, 1 HIGH |
| Physics — Fluid Mechanics | 3 | 1 CRITICAL, 2 HIGH |
| Physics — ISO Standards / Vibration | 2 | 1 CRITICAL, 1 HIGH |
| Physics — Transport Phenomena | 1 | HIGH |
| Physics — Nameplate / Specification | 1 | CRITICAL |
| Data Documentation | 2 | 1 MEDIUM, 1 LOW |
| ML Architecture | 4 | 2 CRITICAL, 2 HIGH |
| ML Training / Data Pipeline | 2 | 1 CRITICAL, 1 HIGH |
| ML Deferred (→ M8) | 2 | 2 MEDIUM |
| **TOTAL** | **22** | **6 CRITICAL, 10 HIGH, 4 MEDIUM, 2 LOW** |

---

## Key Engineering Principles Established

The 22 challenges above collectively established the following principles
that now govern all remaining modules (M5–M11):

1. **Rate-of-change over absolute thresholds**: Sensor placement dependence
   means absolute vibration values cannot be compared across installations.
   All fault signatures defined as dX*/dt patterns (from C-05).

2. **Cluster-conditional normalization and thresholds**: A single global
   statistical parameter fails a multi-modal process. Every statistical
   parameter must respect operating mode (from C-09, C-18).

3. **Physics-weighted ML**: Loss functions and feature importance must
   reflect the physical information hierarchy of sensor channels (from C-14).

4. **Temporal pattern > absolute level**: Overloading cannot be detected
   by temperature magnitude alone — the temporal signature of rising T*
   during stable vibration is the discriminating feature (from C-04).

5. **Conservative baseline = safer production**: M4's phase-lag conservatism
   is deliberate. A false-negative on a normal transition is safer than a
   false-positive that triggers unnecessary shutdown of a 110 kW pump (from C-20).

6. **Sensor failure ≠ process failure**: Isolated single-channel anomalies
   must be classified separately from multi-channel process faults (from C-01).

7. **Climate-agnostic normalization**: Any model targeting deployment beyond
   its training geography must use an internal reference frame (from C-10).

8. **Clean normal baseline is non-negotiable**: The anomaly threshold quality
   is bounded by the purity of the normal training set (from C-15, C-16).

---

*Document version: 1.0 | Covers: M1 through M4 | Next update after M5 completion*
*Asset: 110 kW, 7-stage, 40 bar centrifugal pump | CIRA SACIP dataset*
*Author: Souvik | PumpSmart Physics-Informed ML Digital Twin*
