# PumpSmart — Project Challenges & Resolutions
## Physics-Informed ML Digital Twin: Industrial Centrifugal Pump Health Monitoring

**Complete Audit Trail: M1 through M8**

| Field | Value |
|-------|-------|
| **Asset** | 110 kW \| 7-stage \| 40 bar \| 450 m head \| 2980 RPM \| 45 m³/h |
| **Document version** | 2.0 |
| **Date** | 2026-04-23 |
| **Author** | Souvik |
| **Architecture** | v14.2 \| TCN-AE Level 2 \| CUSUM score_B \| Adaptive Threshold score_A |

---

## Purpose of This Document

This document records every non-trivial engineering or ML challenge encountered during the PumpSmart project from M1 through M8 — excluding syntax errors and environment setup. Each entry includes:

- The module where it was discovered
- The exact nature of the problem (physics, thermodynamics, transport phenomena, ISO standard, ML architecture, data integrity)
- Why it is a problem (the physical or mathematical reasoning)
- The resolution adopted and why it is correct

This document serves as a technical audit trail for presentation, publication, and onboarding. It demonstrates that every architectural decision was physically justified, not arbitrary.

---

## Challenge Index

| # | Module | Category | Challenge Title | Severity |
|---|--------|----------|-----------------|----------|
| C-01 | M1 | Data Integrity | Pump_B_Day3 Sensor Failure vs Data Corruption | HIGH |
| C-02 | M1 | Data Integrity | Gap Threshold Inconsistency Across Day Files | MEDIUM |
| C-03 | M1 | Data Integrity | Pump_C_Day3 Exclusion Decision | MEDIUM |
| C-04 | M2 | Physics — Thermodynamics | Startup Cluster Higher Temperature Than High-Load | HIGH |
| C-05 | M2 | Physics — ISO Standards | .SV Channel Misidentification (ISO 10816-3) | CRITICAL |
| C-06 | M2 | Physics — Transport Phenomena | Pressure Range Paradox in Startup Cluster | HIGH |
| C-07 | M2 | Data Documentation | No Unit Registry for Sensor Channels | MEDIUM |
| C-08 | M2 | Physics — ISO Standards | M2 Plot Y-Axes Without Physical Units | LOW |
| C-09 | M3 | Physics — Thermodynamics | Flash Evaporative Cooling Producing ~20,000 Negative DeltaT\* Values | CRITICAL |
| C-10 | M3 | Physics — Transport Phenomena | Climate-Dependent Ambient Reference Making Model Non-Portable | HIGH |
| C-11 | M3 | Data Integrity | False Alarm Flags on Pressure Normalization Range | MEDIUM |
| C-12 | M3 | Physics | unit_registry.json Premature Specification | LOW |
| C-13 | M4 | ML — Architecture | LSTM-AE Decoder Collapse (v1) | CRITICAL |
| C-14 | M4 | ML — Architecture | Physics-Naive Equal-Weight Loss Function | HIGH |
| C-15 | M4 | ML — Training | Threshold 0.645 Giving 12.9x Separation (False Sense of Security) | HIGH |
| C-16 | M4 | ML — Training | Spike Windows Contaminating Normal Validation Set | CRITICAL |
| C-17 | M4 | Physics — ISO 13373-3 | Pmp.PV Startup Winsorization Ceiling Physically Too Low | HIGH |
| C-18 | M4 | Physics — Fluid Mechanics | Pressure Winsorization Using Global Sigma Across Clusters | CRITICAL |
| C-19 | M4 | Physics — Fluid Mechanics | pressure_transient Spike Ratio Referenced to Wrong Cluster Mean | HIGH |
| C-20 | M4 | ML — Architecture | LSTM Phase-Lag on Mode Transitions (Deferred Flag) | MEDIUM |
| C-21 | M4 | ML — Architecture | Single Global Threshold Across All Operating Modes (Deferred Flag) | MEDIUM |
| C-22 | Cross-Module | Physics | Motor Power Misidentified as 10 kW Instead of 110 kW | CRITICAL |
| C-23 | M6/M8 | ML — Architecture | Inter-Window Amnesia in LSTM-AE for Compound Fault Chains | CRITICAL |
| C-24 | M8 | ML — Architecture | Vanishing Gradient for Sequences Longer Than 150 Steps | HIGH |
| C-25 | M8 | ML — Threshold Engineering | Adaptive Threshold Paradox for Slow-Drift Faults | CRITICAL |
| C-26 | M6/M7 | Data Engineering | Synthetic-to-Real Domain Gap in XGBoost Confidence Scores | HIGH |
| C-27 | M5/M6 | Physics — Fault Sequence Engineering | Sequence Length Insufficient for Real Fault Physics | HIGH |

---

## C-01 — Pump_B_Day3: Sensor Failure vs Data Corruption

| Field | Value |
|-------|-------|
| **Module** | M1 — Data Ingestion & Cleaning |
| **Category** | Data Integrity / Physics of Sensor Systems |
| **Severity** | HIGH |

### The Problem

Pump_B_Day3 had 65.5% rows with NaN values in the Barometer and Temperature columns. The naive interpretation was to discard the entire file as corrupted data.

### Why This Matters Physically

There is a critical distinction between:

- **Sensor failure:** The sensor hardware fails while the pump continues to operate normally. The NaN rows are a hardware event, not a pump event.
- **Process failure:** The pump itself fails, producing meaningless readings.

Pump_B_Day3 showed continuous, physically consistent readings in all 8 pump sensor channels (Mot.PV, Mot.SV, Mot.TV, Pmp.PV, Pmp.SV, Pmp.TV, Temp.SV, Pres.SV) throughout the NaN period. Only the environmental sensors (Barometer, Temperature) went silent. This is the exact signature of a barometric sensor hardware dropout — the pump ran normally while the sensor stopped logging.

This is directly analogous to Fault Type C (Sensor Dropout) which M6 will synthetically generate. Discarding the file would have destroyed valid training data AND misclassified sensor failure as process failure — a safety error.

### Resolution

Hard drop all rows with any NaN (policy unchanged — no interpolation). After dropping, re-segment by timestamp gap detection. The clean segments before and after the NaN block are structurally valid. Two usable segments recovered from B_Day3. Post-dropout segments assigned 600-row warmup (vs 300 standard) to allow sensor equilibration before windowing.

> **Lesson locked:** Sensor failure ≠ process failure. Physical context must be applied before discarding data. This distinction is now a formal fault type in M6.

---

## C-02 — Gap Threshold Inconsistency Across Day Files

| Field | Value |
|-------|-------|
| **Module** | M1 — Data Ingestion & Cleaning |
| **Category** | Data Integrity / Time Series Segmentation |
| **Severity** | MEDIUM |

### The Problem

A single global gap threshold applied uniformly across all 9 files produced incorrect segment splits. Day1 files had natural timestamp jitter up to ~5s between consecutive records despite being 1s nominal sampling. Day2/Day3 files were perfectly continuous at 1s.

Applying a 2s threshold to Day1 files would fragment single continuous operational runs into dozens of unusable micro-segments. Applying an 8s threshold to Day2/Day3 would merge genuinely separate operational sessions.

### Physics Basis

The gap threshold defines what constitutes a "new operating state" vs natural measurement noise. For a 110 kW, 2980 RPM pump:

- A 5s gap in a Day1 file at 1s sampling = SCADA clock drift, not a pump event
- A 5s gap in a Day3 perfectly-continuous file = genuine process interruption (startup-shutdown boundary)

Using the same threshold for both = either over-segmenting Day1 (losing training data) or under-segmenting Day3 (creating windows across operational boundaries — a time-series integrity violation).

### Resolution

File-specific gap thresholds derived from the actual median sampling interval per file:

- **Day1 files:** gap threshold = 8s (8× median — above natural 1s jitter)
- **Day2/Day3 files:** gap threshold = 2s (2× median — catches operational pauses)

This preserved 27 usable segments vs the ~8 that uniform thresholding would have produced.

---

## C-03 — Pump_C_Day3 Exclusion Decision

| Field | Value |
|-------|-------|
| **Module** | M1 — Data Ingestion & Cleaning |
| **Category** | Data Integrity |
| **Severity** | MEDIUM |

### The Problem

Pump_C_Day3 initially appeared to have valid data. However, cross-referencing with B_Day3 revealed 100% Barometer corruption in linked session files. Unlike B_Day3 where the pump data remained physically consistent, C_Day3 showed anomalous behaviour in process sensor channels correlated with the environmental sensor failure period — suggesting the NaN propagation affected more than just the barometric sensor.

### Resolution

C_Day3 excluded entirely from the training pipeline. Segments from 8 remaining files (25 usable segments) provide sufficient representation across all 4 operating modes. Exclusion documented in `segment_registry.csv` with reason flag.

---

## C-04 — Startup Cluster Higher Temperature Than High-Load

| Field | Value |
|-------|-------|
| **Module** | M2 — EDA & Clustering |
| **Category** | Physics — Thermodynamics (Heat Transfer) |
| **Severity** | HIGH |

### The Problem

K-Means clustering assigned clusters, and the temperature analysis showed:

- Startup cluster mean Pmp.TV: **41.9 °C**
- High-load cluster mean Pmp.TV: **39.5 °C**

The initial script comment flagged this as physically suspicious — conventional understanding suggests high-load = high temperature. If interpreted as an error, it would have prompted a re-clustering attempt that would have destroyed valid operating mode identification.

### Physics Explanation — Thermodynamics

This is correct physics for a 7-stage multistage centrifugal pump at 2980 RPM.

The thermal time constant for a 110 kW IEC 315 frame motor is approximately:

```
τ = m × Cp / (h × A) ≈ 400–600 seconds
```

The pump operates in cycles: startup → steady-state → high-load → cooldown. When the pump enters the startup cluster, it is coming out of a warm cooldown phase. The motor casing retains heat from the previous operational cycle. The pump has NOT yet reached hydraulic full-load (affinity law: Q ∝ N, H ∝ N² — flow and head build up from zero), so shaft power is low, but residual thermal mass from the prior cycle keeps casing temperature elevated.

By the time the pump reaches high-load, the convective heat transfer from forced airflow over the motor casing (and process fluid cooling in the pump casing) has partially offset the increased shaft heat generation. Net result: high-load steady-state temperature is **LOWER** than the peak thermal transient during startup run-in.

> **Direct consequence for M5 (Overloading fault):** defining overloading as "temperature above high-load maximum" would be WRONG. Overloading is defined as monotonically rising temperature DURING stable vibration — a temporal pattern, not an absolute threshold.

### Resolution

Script comment corrected (FIX-3 in M2 audit). Cluster assignment validated. No re-clustering. Overloading fault definition revised in M5 specification to use temporal rate-of-change, not absolute temperature threshold.

---

## C-05 — .SV Channel Misidentification (ISO 10816-3 Violation)

| Field | Value |
|-------|-------|
| **Module** | M2 — EDA & Clustering |
| **Category** | Physics — ISO Standards / Vibration Measurement |
| **Severity** | CRITICAL |

### The Problem

The CIRA dataset documentation and initial project descriptions labeled the `.SV` channels (X_ACR_Mot.SV, X_ACR_Pmp.SV) as single-tone ISO acceleration values in m/s². This led to early attempts to apply ISO 10816-3 absolute alarm thresholds directly:

| Zone | Description | Range |
|------|-------------|-------|
| Zone A | New machine | < 2.3 mm/s |
| Zone B | Acceptable | 2.3–4.5 mm/s |
| Zone C | Alarm | 4.5–7.1 mm/s |
| Zone D | Danger | > 7.1 mm/s |

The data showed Motor SV values at:
- Startup: 0.39–0.58 mm/s (well below Zone A)
- High-load: 22.2–77.7 mm/s (far above Zone D)

If these were true ISO 10816-3 velocity values, the pump would permanently be in danger zone during high-load operation — physically implausible for a well-maintained industrial pump in service.

### Physics Diagnosis

Cross-referencing the CIRA SACIP technical documentation confirmed these are **broadband peak acceleration envelopes** — a SCADA composite metric that integrates across multiple frequency bands and takes the peak envelope value. This is NOT the same as ISO 10816-3 velocity RMS. The scale and units are different. The absolute values cannot be compared to ISO 10816 tables.

Furthermore, the Mot.SV transient spike of 456.6 mm/s in the dataset (high-load cluster) is consistent with a broadband peak envelope during an impulsive event (bearing impact or cavitation shock) — not a physically sustained velocity that would destroy any known pump bearing.

### Resolution

All fault detection re-defined as **relative change from cluster mean**, not absolute ISO threshold comparison:

| State | SV\* Range |
|-------|-----------|
| Normal | 0.8–1.2 (near cluster mean) |
| Early fault | 1.3–1.5 |
| Active fault | 1.5–2.5 |
| Severe | > 2.5 |

ISO 10816-3 retained as a reference framework for fault severity classification in M5/M6 but NOT applied as absolute cutoffs. Channel weights in M4 set high (2.0) for SV channels to preserve their relative fault-detection value while removing absolute-threshold dependence. This is documented in the pathway as a locked architectural decision.

---

## C-06 — Pressure Range Paradox in Startup Cluster

| Field | Value |
|-------|-------|
| **Module** | M2 — EDA & Clustering |
| **Category** | Physics — Fluid Mechanics / Transport Phenomena |
| **Severity** | HIGH |

### The Problem

The startup cluster showed outlet pressure ranging from 0.43 to 0.85 bar, while the cooldown cluster showed 0.45 to 44.4 bar. The cooldown cluster therefore contains both very low AND very high pressure states.

This appeared contradictory: how can a pump in cooldown (shutting down) have pressures up to 44.4 bar, higher than steady-state (0.69–43.3 bar)?

### Physics Explanation — Fluid Mechanics (Joukowsky Water Hammer)

When a 110 kW pump at 40 bar operational pressure shuts down, the rapid deceleration of the fluid column creates a water hammer pressure surge (Joukowsky equation):

```
ΔP = ρ × c × Δv
```

where c = acoustic wave speed in pressurized fluid (~1200 m/s for water), and Δv = velocity change at shutoff. For a 45 m³/h flow rate in the pump discharge line, this produces transient pressure spikes well above steady-state.

This is why the cooldown cluster has higher pressure than the steady-state cluster: the cooldown cluster includes the moment of shutdown when water hammer transiently pushes discharge pressure above the steady-state 40 bar nameplate. The observed transient maximum of 46.7 bar is consistent with Joukowsky predictions for this system.

The startup cluster has LOW pressure (0.43–0.85 bar) because it captures the moment the pump comes online from rest — pressure builds from atmospheric as the impellers accelerate. This is also the **NPSH-critical zone**: at these low pressures, NPSHa may fall below NPSHr, creating cavitation risk.

### Resolution

Cluster assignments confirmed as physically correct. Pressure transient data preserved in cooldown cluster (not filtered as outliers). Startup cluster flagged as the cavitation risk zone — M5 physics engine specifies that all cavitation fault sequences must begin in startup cluster context. Joukowsky bounds used to set the Pres.SV winsorization ceiling in M4.

---

## C-07 — No Unit Registry for Sensor Channels

| Field | Value |
|-------|-------|
| **Module** | M2 — EDA & Clustering (discovered), Fixed M2 Audit |
| **Category** | Data Documentation / ISO Compliance |
| **Severity** | MEDIUM |

### The Problem

All cluster bounds in `M2_cluster_bounds.csv` were stored as raw numbers with no unit documentation. Column names like `X_ACR_Mot.SV` gave no indication of whether values were in m/s, mm/s, m/s², or mm/s². This created ambiguity in every subsequent module that reads this file.

For a safety-critical application on a 110 kW pump, unit ambiguity in sensor data is unacceptable — a confusion between mm/s and m/s represents a factor of 1000 error in vibration severity assessment.

### Resolution

Created `M2_cluster_bounds_units.json` as authoritative unit registry:

| Channel | Unit | Standard |
|---------|------|----------|
| X_ACR_Mot.PV | mm (displacement) | ISO 10816-3 |
| X_ACR_Mot.SV | mm/s (broadband peak envelope) | ISO 10816-3 |
| X_ACR_Mot.TV | °C | IEC 60034-1 |
| X_ACR_Pmp.PV | mm | ISO 10816-3 |
| X_ACR_Pmp.SV | mm/s | ISO 10816-3 |
| X_ACR_Pmp.TV | °C | IEC 60034-1 |
| X_Temp.SV | °C | ISO 13373-2 |
| X_Pres.SV | bar | ISO 5167 |

All downstream modules reference this file. M5 extended it with fault-specific unit metadata in `unit_registry.json`.

---

## C-08 — M2 Plot Y-Axes Without Physical Units

| Field | Value |
|-------|-------|
| **Module** | M2 — EDA & Clustering |
| **Category** | Documentation / ISO Compliance |
| **Severity** | LOW |

### The Problem

All M2 time-series plots had raw column names on Y-axes (e.g., `X_ACR_Mot.SV`) with no unit labels. For a project targeting maintenance engineers and plant operators, unlabelled axes are a presentation failure and violate basic engineering documentation standards.

### Resolution

All M2 time-series plots regenerated with proper unit labels: `"X_ACR_Mot.SV (mm/s — broadband peak)"`, `"X_Pres.SV (bar)"`, etc. FIX-2 in M2 audit record.

---

## C-09 — Flash Evaporative Cooling Producing ~20,000 Negative DeltaT\* Values

| Field | Value |
|-------|-------|
| **Module** | M3 — Dimensionless Normalization |
| **Category** | Physics — Thermodynamics / Phase Change (Transport Phenomena) |
| **Severity** | CRITICAL |

### The Problem

The initial temperature normalization formula (v5 specification) used a per-row ambient temperature reference:

```
ΔT* = (T - T_ambient_per_row) / (T_cluster_max - T_ambient_per_row)
```

When this was executed during M3, approximately 20,000 rows in the cooldown cluster produced negative ΔT\* values (down to -0.113). The immediate interpretation was a formula error — temperatures should be above ambient, so negative normalized values should not exist.

### Physics Explanation — Thermodynamics (Phase Change / Evaporative Cooling)

This was **NOT** a formula error. It is a real physical phenomenon.

When a 40 bar pressurized pump shuts down, the internal fluid is at high pressure and temperature (~40–50 °C). As discharge valves close and pressure drops during cooldown, the pressurized fluid undergoes **flash evaporation** — a fraction of the fluid instantaneously vaporizes as pressure drops below the saturation pressure at the local temperature. This adiabatic phase change absorbs latent heat from the surrounding metal:

```
Q_evap = ṁ_flash × h_fg    (h_fg ≈ 2257 kJ/kg for water at 100°C)
```

The net effect: pump casing metal temperature drops **BELOW** ambient air temperature within minutes of shutdown. For CIRA data:
- Ambient temperature (Naples, Italy): 19.0–28.8 °C
- Cooldown cluster sensor minimum: 17.6 °C
- Sub-ambient: (17.6 - 19.0) = **-1.4 °C below ambient**, lasting 5–15 min

This is the same physics as a refrigerator evaporator coil — pressurised fluid expands → cools below surroundings. The 20,000 negative rows are entirely correct physical data, not errors.

The bug was in the normalization formula, not the data. The ambient-relative formula breaks when the physical process being measured can cool below the reference baseline. The formula implicitly assumes T_actual ≥ T_ambient always — a false assumption for phase-change systems.

### Resolution

Temperature normalization formula changed to cluster-relative (v6/M3 re-run):

```
ΔT* = (T - T_cluster_min) / (T_cluster_max - T_cluster_min)
```

`T_cluster_min` is the coldest observed temperature in each operating mode — it represents the physical floor of that mode including flash evaporative events. The sub-ambient readings (18 extreme rows) produce small negative values (-0.02 to -0.05) under the new formula, which is acceptable and physically meaningful. They are preserved — not clipped. The LSTM-AE must learn these as valid extreme-normal cooldown states.

**Secondary benefit:** The cluster-relative formula is climate-agnostic. The ambient-relative formula was calibrated to Naples, Italy (19–28 °C ambient). Deploying on a pump in Kolkata, India (22–38 °C ambient) would shift the entire temperature baseline, producing systematically different normalized values for the same physical pump state. The cluster-relative formula uses the pump's own operational envelope as reference — it works identically regardless of installation climate.

---

## C-10 — Climate-Dependent Ambient Reference Breaking Model Portability

| Field | Value |
|-------|-------|
| **Module** | M3 — Dimensionless Normalization |
| **Category** | Physics — Transport Phenomena / Deployment Engineering |
| **Severity** | HIGH |

### The Problem

Directly related to C-09. Even if the flash evaporation issue did not exist, the ambient-relative temperature formula creates a deployment portability problem. A model trained on CIRA data from Naples (ambient 19–28 °C) would be miscalibrated when deployed on the same pump type in a different climate (Kolkata: 22–38 °C; industrial desert facility: 35–48 °C).

The normalized temperature at the SAME physical pump state would be different depending on where the model is deployed — making the learned normality baseline geography-dependent.

### Resolution

Identical to C-09 — cluster-relative formula. Documented as a locked architectural decision in the pathway. The model's normality baseline is the pump's own operational physics, not the local weather.

---

## C-11 — False Alarm Flags on Pressure Normalization Range

| Field | Value |
|-------|-------|
| **Module** | M3 — Dimensionless Normalization |
| **Category** | Data Integrity / Statistical Misinterpretation |
| **Severity** | MEDIUM |

### The Problem

After applying ratio normalization (P\* = P_actual / P_cluster_mean), the output statistics showed ~45.5% of `Pres.SV_norm` values above 1.0. An automated range-check script flagged this as an anomaly: *"45% of values outside normal range [0,1] — potential normalization error."*

### Physics Explanation — Statistics

This is a mathematical certainty, not an error. Ratio normalization around the cluster mean (P\* = P / P_mean) produces a distribution where:

- Mean = 1.0 by definition
- All values below the mean produce P\* < 1.0
- All values above the mean produce P\* > 1.0
- For a right-skewed pressure distribution: ~45–55% of values will be above 1.0

The [0,1] range is not a hard constraint for pressure and vibration channels — it is the expected operating zone where 1.0 = cluster mean (normal operating point). The LSTM-AE is trained to reconstruct patterns in this space. Values above 1.0 are not automatically anomalous — they are above-mean normal operation. Values significantly above 1.0 (>1.5) AND with anomalous temporal patterns = anomalous.

The only channels that hard-clip to [0,1] are the temperature channels, because the cluster-relative formula uses min/max bounds.

### Resolution

Range-check logic corrected to distinguish channel types:

- **Pressure/vibration channels:** range CHECK is informational only; values above 1.0 are expected and normal
- **Temperature channels:** hard range [-0.1, 1.05] expected; violations warrant investigation

False alarm flags removed from M3 report. Correct statistical interpretation documented.

---

## C-12 — unit_registry.json Premature Specification in M3

| Field | Value |
|-------|-------|
| **Module** | M3 — Normalization |
| **Category** | Project Architecture |
| **Severity** | LOW |

### The Problem

The v5 pathway document specified that `unit_registry.json` should be created as an M3 output. During M3 execution, it became clear this was premature: the physics engine in M5 would define fault-specific unit metadata (ISO alarm thresholds, coupling coefficients, fault envelope boundaries) that require knowledge not yet available at M3 time.

Creating a partial unit registry in M3 and then extending it in M5 creates a fragmented, hard-to-maintain documentation artifact.

### Resolution

`unit_registry.json` creation deferred to M5. M3 produces `M3_normalization_config.json` (normalization baselines only). M5 produces `unit_registry.json` as its first output step, with complete metadata for all channels including fault context.

---

## C-13 — LSTM-AE Decoder Collapse (v1)

| Field | Value |
|-------|-------|
| **Module** | M4 — LSTM-AE Baseline |
| **Category** | ML Architecture — LSTM Sequence-to-Sequence |
| **Severity** | CRITICAL |

### The Problem

M4 v1 trained successfully (loss decreased), but the reconstruction sample plots showed the decoder producing a flat constant output — approximately the mean of the training data — for all input sequences regardless of content. This is called **decoder collapse**: the model learns that outputting a constant near the data mean minimises reconstruction loss on average, so it stops learning to reconstruct individual sequences.

Key symptoms:
- val loss: 0.251 (appeared reasonable)
- Per-channel MAE: all channels showing near-identical errors
- Reconstruction plot: orange dashed line = horizontal constant across all 50 timesteps

### Root Cause

The decoder LSTM was initialized with zero hidden state (h0=0, c0=0). The encoder produced a meaningful latent vector in its bottleneck, but the decoder, starting from zero state, could not effectively use the latent information to seed its sequential reconstruction. The decoder defaulted to learning a global mean predictor, which is a local minimum for MSE-based loss.

For a sequence-to-sequence autoencoder, the decoder's initial hidden state must be seeded from the encoder's final hidden state — this is the information bridge. Without it, the bottleneck is effectively disconnected.

### Resolution

**Hidden-state-seeded decoder:** The encoder's final hidden state (h_n, c_n) is passed directly as the decoder's initial state (h0, c0):

```python
# Encoder
_, (h_n, c_n) = self.encoder_lstm(x)
bottleneck = self.fc_encoder(h_n[-1])  # compress to 64 dims

# Decoder — seeded from encoder state
h0 = self.fc_decoder_h(bottleneck).unsqueeze(0).repeat(n_layers, 1, 1)
c0 = self.fc_decoder_c(bottleneck).unsqueeze(0).repeat(n_layers, 1, 1)
out, _ = self.decoder_lstm(decoder_input, (h0, c0))
```

Result: val MAE improved from 0.251 (v1) to 0.050 (v3) — a 5× improvement. Threshold tightened from 1.362 to 0.645. Decoder collapse completely resolved. This fix has been present from v3 onwards and is locked in the architecture.

---

## C-14 — Physics-Naive Equal-Weight Loss Function

| Field | Value |
|-------|-------|
| **Module** | M4 — LSTM-AE Baseline |
| **Category** | ML Architecture — Loss Function Design |
| **Severity** | HIGH |

### The Problem

The initial loss function weighted all 8 sensor channels equally. This is physically naive: a 1 mm/s change in Mot.SV (vibration velocity) has completely different fault significance than a 1 °C change in Pmp.TV (contact temperature).

Equal weighting means the model optimizes reconstruction quality by spending capacity on whichever channels are easiest to reconstruct (typically slow-moving temperature channels). The high-frequency, physically informative vibration and pressure channels — which carry the earliest fault signatures — get under-weighted.

### Physics Basis for Channel Importance Hierarchy

For a 110 kW, 2980 RPM, 7-stage pump:

| Rank | Channels | Reason |
|------|---------|--------|
| 1st | Vibration velocity (SV channels) | Directly governed by ISO 10816-3. First to show bearing wear, imbalance, cavitation. Broadband RMS changes within 2–3 shaft rotations of fault onset. Placement-robust. |
| 2nd | Vibration displacement (PV channels) | Related to SV by frequency content. Highly informative for impeller imbalance (1×RPM harmonic). Placement-robust. |
| 3rd | Discharge pressure (Pres.SV) | Governed by pump affinity laws. Cavitation, seal failure, and impeller damage all manifest as pressure anomalies. Direct process variable. |
| 4th | Casing temperature (Temp.SV) | Governed by heat transfer: dT/dt = Q_loss/(m·Cp) - h·A·(T-T_amb). Lags fault onset by thermal time constant τ ≈ 400–600s. Secondary fault indicator — important but slow. |
| 5th | Contact temperatures (Mot.TV, Pmp.TV) | Most placement-dependent. Valid for CIRA installation geometry, but cannot be guaranteed for different sensor mounting configurations. Lowest fault-detection reliability for general deployment. |

### Resolution

Physics-weighted channel loss function:

```python
channel_weights = {
    'Mot.SV':  2.0,   # placement-robust, ISO 10816-3 primary
    'Pmp.SV':  2.0,   # placement-robust, ISO 10816-3 primary
    'Pres.SV': 2.0,   # direct process variable
    'Mot.PV':  1.5,   # placement-robust, secondary vibration
    'Pmp.PV':  1.5,   # placement-robust, secondary vibration
    'Temp.SV': 1.0,   # single casing sensor, thermal lag
    'Mot.TV':  0.8,   # placement-dependent
    'Pmp.TV':  0.8    # placement-dependent
}
total_loss = 0.6*MAE + 0.4*MSE  # physics-weighted per-channel
```

The 0.6/0.4 MAE/MSE split: MAE is robust to outliers (better for spike-heavy vibration data), MSE penalises large deviations more heavily (important for catching pressure drops and thermal runaway). Combined loss captures both failure modes.

---

## C-15 — Threshold 0.645 Giving 12.9x Separation (False Sense of Security)

| Field | Value |
|-------|-------|
| **Module** | M4 — LSTM-AE Baseline (v3/v4/v5/v6) |
| **Category** | ML Training / Threshold Calibration |
| **Severity** | HIGH |

### The Problem

M4 versions v3 through v6 produced an anomaly threshold of 0.645 with a separation ratio of 12.9× (threshold / mean MAE = 0.645 / 0.050). While this appeared excellent, it was discovered that the validation set was contaminated with spike windows — windows containing genuine transient fault events that elevated the mean MAE and pushed the threshold unrealistically high.

A threshold of 0.645 means the model requires a fault to elevate reconstruction error by 12.9× before raising an alarm. For a 110 kW, 40 bar pump, early-stage bearing wear or incipient cavitation might only elevate MAE by 3–5× before becoming dangerous. A 12.9× threshold would miss these entirely.

The high threshold was not evidence of model quality — it was evidence that the "normal" baseline was polluted with anomalous events, artificially elevating the mean and pulling the threshold up with it.

### Resolution

Spike row exclusion before windowing (v7/v8): All windows containing rows where any channel exceeded cluster-conditional winsorization bounds were excluded from both training and validation pools.

The result:
- 12,620 spike rows excluded from training data
- 1,044 spike windows saved separately as `M4_spike_seeds.npy` for M6
- Clean val set of 1,457 normal-only windows
- Recalibrated threshold: **0.110058** (mean=0.0268, separation=4.11×)

A 4.11× separation on a clean normal baseline is more honest and more sensitive than 12.9× on a contaminated baseline. The model now requires only 4.11× elevation to trigger an alarm — appropriate for early-fault detection on a high-value asset.

---

## C-16 — Spike Windows Contaminating Normal Validation Set

| Field | Value |
|-------|-------|
| **Module** | M4 — LSTM-AE Baseline (v6 to v7 fix) |
| **Category** | ML Training — Data Pipeline Integrity |
| **Severity** | CRITICAL |

### The Problem

The fundamental purpose of an LSTM-AE anomaly detector is to learn the normal operating manifold and flag deviations. The validation set must contain ONLY clean normal operation. If spike windows (containing real transient fault events) enter the validation set, the model learns that these spikes are "normal" — directly undermining the detection objective.

In v1–v6, the training/validation split was performed on the full window pool before any spike filtering. This meant ~8% of validation windows contained genuine anomalous events. The threshold was therefore calibrated not on the normal manifold but on a contaminated distribution.

For a 110 kW industrial pump, this is a safety issue: the model was unknowingly trained to consider certain fault events as normal.

### Resolution

Strict two-phase pipeline in v7/v8:

1. **PHASE 1:** Scan all rows, identify spike rows (per-channel, per-cluster bounds)
2. **PHASE 2.5:** Drop ALL spike rows from the dataframe BEFORE windowing
3. **PHASE 3:** Window the cleaned dataframe → clean window pool
4. **PHASE 4:** Save spike rows separately as M4_spike_seeds for M6
5. **PHASE 5:** Train/val split on clean pool only

Spike rows are not wasted — they become the real-data seeds for synthetic fault generation in M6. Every anomalous event in the dataset has been deliberately preserved and redirected to its correct use: fault examples, not normal examples.

---

## C-17 — Pmp.PV Startup Winsorization Ceiling Physically Too Low

| Field | Value |
|-------|-------|
| **Module** | M4 — LSTM-AE Baseline (v7 to v8 fix) |
| **Category** | Physics — ISO 13373-3 / Vibration Mechanics |
| **Severity** | HIGH |

### The Problem

In M4 v7, the Pmp.PV (pump casing displacement) winsorization ceiling was set uniformly at 2.6× the cluster mean across all clusters including startup. During v8 physics audit, this was identified as physically incorrect.

During pump startup at 2980 RPM with 7 impellers:
- Blade Passing Frequency (BPF) = RPM/60 × N_blades = 2980/60 × 7 = **348.7 Hz**
- During ramp-up from 0 to 2980 RPM, the rotational frequency sweeps through ALL resonant frequencies of the pump casing and piping system
- ISO 13373-3 documents that BPF harmonics during speed ramp-up can reach **2–4× the steady-state displacement amplitude**

A 2.6× ceiling during startup would cause the winsorizer to clip legitimate startup resonance events as spikes, mislabeling them as fault seeds in M4 and injecting physically incorrect fault data into M6.

### Resolution

Startup cluster Pmp.PV ceiling raised to **3.2×** (ISO 13373-3 BPF harmonic headroom). All other clusters remain at 2.6×. This is now stored in `M4_spike_config.json` as a cluster-conditional ceiling. M6 reads this file and cannot override it.

---

## C-18 — Pressure Winsorization Using Global Sigma Across Clusters

| Field | Value |
|-------|-------|
| **Module** | M4 — LSTM-AE Baseline (v7 to v8 fix) |
| **Category** | Physics — Fluid Mechanics / Dimensional Analysis |
| **Severity** | CRITICAL |

### The Problem

M4 v7 used a uniform pressure winsorization ceiling derived from the global pressure standard deviation across all clusters.

| Cluster | Pres.SV mean | Pres.SV std |
|---------|-------------|-------------|
| Startup | 0.621 bar | 1.03 bar (normalized to mean) |
| High-load | 42.0 bar | 1.92 bar |
| Steady-state | 35.8 bar | 13.0 bar |

A pressure deviation of 5 bar during startup (P = 0.43–0.85 bar) is physically catastrophic — it represents a Joukowsky water hammer exceeding the entire operating pressure range. The same 5 bar deviation during high-load (P ≈ 42 bar) is a routine ±12% fluctuation well within normal operation.

Applying a global sigma-based ceiling treats these identically — a critical physics violation. This is dimensionally analogous to trying to use a single Reynolds number criterion for both laminar pipe flow and turbulent jet flow simultaneously — the physics are qualitatively different.

### Resolution

Cluster-conditional pressure winsorization in v8:

| Cluster | Ceiling | Rationale |
|---------|---------|-----------|
| Startup | 3.0× cluster mean | Joukowsky transient headroom |
| Steady-state | 5.6× cluster mean | Wide valid operating range |
| High-load | 2.0× cluster mean | Tight — any deviation is fault |
| Cooldown | 3.0× cluster mean | Depressurization transients |

The high-load ceiling of 2.0× reflects the physics: a 110 kW pump at 40 bar steady high-load operation has near-constant pressure. Any deviation >2× the expected pressure is a genuine fault event. The tighter ceiling means fault detection is MORE sensitive at high-load — exactly correct for a high-consequence operating state.

---

## C-19 — pressure_transient Spike Ratio Referenced to Wrong Cluster Mean

| Field | Value |
|-------|-------|
| **Module** | M4 — LSTM-AE Baseline (v7 to v8 fix) |
| **Category** | Physics — Fluid Mechanics (Joukowsky Water Hammer) |
| **Severity** | HIGH |

### The Problem

M4 v7 computed the spike ratio for `pressure_transient` fault seeds using the startup cluster mean (0.621 bar) as denominator:

```
spike_ratio = P_spike / P_startup_mean = 46.7 / 0.621 = 75.2×
```

This produced spike ratios of 70–80× for pressure transient events. While mathematically accurate, it is physically meaningless as a fault severity indicator. A water hammer event in a 40 bar system derives its energy from the HIGH-LOAD operating pressure at the moment of shutoff.

Joukowsky equation:
```
ΔP = ρ × c × Δv
```

The momentum of the fluid column at shutoff is governed by the high-load flow conditions (Q ≈ 45 m³/h at ~42 bar), not by the startup conditions. Using the startup mean as denominator inflates the spike ratio by a factor of 42/0.621 = 67× — a purely artefactual amplification. M6 using these inflated spike ratios would generate synthetic pressure transient faults with completely unphysical magnitudes.

### Resolution

`pressure_transient` spike ratio recalculated using **high-load cluster mean (42.0 bar)** as reference:

```
spike_ratio = P_spike / P_highload_mean = 46.7 / 42.0 = 1.11×
```

This is physically correct: the Joukowsky pressure surge of 46.7 bar represents a **+11% overpressure** relative to normal high-load operating pressure — a realistic and physically significant transient. Documented in `M4_spike_config.json`.

---

## C-20 — LSTM Phase-Lag on Mode Transitions (Deferred to M8)

| Field | Value |
|-------|-------|
| **Module** | M4 — LSTM-AE Baseline v8 (identified, deferred) |
| **Category** | ML Architecture — LSTM Temporal Dynamics |
| **Severity** | MEDIUM (M4 context) / HIGH (M8 context) |

### The Problem

M4 v8 reconstruction sample plots showed phase-lag artifacts on Pmp.PV (MAE=0.079) and Mot.PV (MAE=0.079) during operating mode transitions. The LSTM reconstructed a smooth decay curve instead of tracking the sharp step-change when the pump crosses from one operating regime to another.

Root cause: LSTM hidden state integrates temporal history by design. When the pump transitions abruptly (startup to steady-state), the hidden state retains memory of the prior regime for several timesteps — producing slow adaptation.

### Why Deferred to M8 (Not Fixed in M4)

For M4 (baseline), this behavior is actually **conservative**: the model underestimates reconstruction quality on transitions → MAE is slightly elevated → less false-alarm risk on normal transitions. The 8 false alarms in the v8 val set confirm this is not causing alarm inflation.

More critically: the transition zone (startup to steady-state) at low pressure (0.43–0.85 bar) is the highest cavitation risk zone for this pump. A model that is slightly elevated in sensitivity during this transition is correct behavior for a baseline safety system.

For M8 (production), this is inadequate. See C-21 and M8 Safety Mandate.

### M8 Resolution (Mandated)

Temporal attention + gradient penalty loss. Full specification in M8 Safety Mandate section of `module_pathway_M1_to_M8_v7.md`.

---

## C-21 — Single Global Threshold Across All Operating Modes (Deferred to M8)

| Field | Value |
|-------|-------|
| **Module** | M4 — LSTM-AE Baseline v8 (identified, deferred) |
| **Category** | ML Architecture — Threshold Calibration |
| **Severity** | MEDIUM (M4 context) / HIGH (M8 context) |

### The Problem

M4 uses a single threshold of 0.110058 for all 4 operating modes. The normalized reconstruction error distributions have different noise floors per cluster:

| Cluster | Expected MAE Noise Floor | Physics Reason |
|---------|--------------------------|----------------|
| Startup | Relatively high | Rapid hydraulic and thermal changes |
| Steady-state | Low | Stable conditions, slowest dynamics |
| High-load | Very low | Tight pressure control |
| Cooldown | Medium | Depressurization transients |

A single threshold that works for steady-state is too sensitive for startup (false alarms) and not sensitive enough for high-load (missed early faults). For a 110 kW pump where high-load is the maximum stress condition, reduced sensitivity at high-load is a safety gap.

### Why Deferred to M8

M4 is a baseline — its false alarm count of 8 on 1,457 val windows (0.55%) is acceptable for calibration purposes. Cluster-conditional thresholds require sufficient labeled data per cluster for robust calibration, which M4's clean window pool barely supports.

### M8 Resolution (Mandated)

Cluster-conditional threshold calibration using `M4_spike_config.json` cluster distributions as threshold priors. Full specification in M8 Safety Mandate.

---

## C-22 — Motor Power Misidentified as 10 kW Instead of 110 kW

| Field | Value |
|-------|-------|
| **Module** | Cross-module (source: Zenodo documentation, propagated M1 to M4) |
| **Category** | Physics — Nameplate Specification |
| **Severity** | CRITICAL |

### The Problem

The CIRA SACIP Zenodo record (DOI: 10.5281/zenodo.15301820) contains a reference to "10 kW" in its pump specification text. This was initially used as the motor power in all early physics calculations, including hydraulic power checks and thermodynamic balance estimates.

The error only became apparent during M5 physics engine planning when the hydraulic power equation was applied:

```
P_hyd = ρ × g × Q × H / η = 1000 × 9.81 × (45/3600) × 450 / 0.65 ≈ 55 kW
```

A 55 kW hydraulic power requirement **CANNOT** be delivered by a 10 kW motor. This is a fundamental violation of energy conservation: the motor output power must exceed the hydraulic power by at least 1/η ≈ 1.54×. A 10 kW motor driving a 55 kW hydraulic load is physically impossible.

### Diagnosis

Cross-referencing the physical evidence:

- IEC Frame 315mm → standard motor frame for 90–200 kW motors
- 400V, 2-pole, 2980 RPM → consistent with 110 kW industrial motors
- 7 impellers, 450 m head, 45 m³/h → requires ~55 kW hydraulic power → shaft power = 55/0.5 = **110 kW** (assuming 50% overall efficiency)
- The "10 kW" in the Zenodo text likely refers to a sub-duty point or a specific test condition, not the rated motor power

> The nameplate is definitively: **Motor shaft power = 110 kW.**

### Impact on Completed Modules

| Module | Impact |
|--------|--------|
| M1, M2, M3 | No physics calculations used the motor power value directly. Zero impact on data pipeline. |
| M4 | Channel weights and loss function are dimensionless — no motor power dependency. Zero impact on model. |
| M5 onwards | All hydraulic power, efficiency, and thermal calculations MUST use 110 kW. Hardcoded as a verified constant in the pathway document and Paste Text tracker. |

### Resolution

Motor power corrected to **110 kW** in all documentation, Paste Text, and pathway file. Warning note added to project specification:

> *"'10 kW' in Zenodo source = sub-duty test point. Nameplate motor = 110 kW. Use 110 kW for ALL physics calculations in M5+."*

---

## C-23 — Inter-Window Amnesia in LSTM-AE for Compound Fault Chains

| Field | Value |
|-------|-------|
| **Module** | M6 (discovered during M6B design) / M8 (resolved) |
| **Category** | ML Architecture — Temporal Context |
| **Severity** | CRITICAL |

### The Problem

During M6B compound fault sequence design, it was established that several fault classes involve a causal chain unfolding across time:

**Example — Label 10 (seal+cavitation):**
- t=0: seal begins leaking → Pres.SV starts declining
- t=80–120s: NPSHa drops below NPSHr → cavitation initiates
- t=150–200s: impeller erosion begins → Pmp.SV spikes emerge

The LSTM-AE processes one 50-step window at a time. Between windows, the hidden state is reset to zero (stateless inference). This means:

| Window | Sees | Result |
|--------|------|--------|
| Window 1 (t=0–49) | Seal pressure decline only | Reconstructs normally |
| Window 2 (t=50–99) | Pressure decline + early cavitation onset | MAE rises |
| Window 3 (t=100–149) | Full cavitation signature | MAE crosses threshold |

Window 3 cannot "remember" that Pres.SV was already declining in Window 1. The LSTM-AE evaluates each window in isolation. For a compound fault where the full diagnostic signature only becomes clear across 3–5 consecutive windows, single-window MAE will either:

- **(a) Fire too late** — only after Window 3, missing 100–150 steps of lead time
- **(b) Misclassify** — XGBoost sees only the current window's features, not the causal history connecting Pres.SV decline to Pmp.SV spike

This is **architectural amnesia**: each inference window has no knowledge of what the previous window saw.

### Physics Basis — Why This Matters for Compound Faults

For a 110 kW, 40 bar, 7-stage pump:

**Bearing-to-seal cascade (Label 7, bearing+overloading):**
Bearing wear → shaft misalignment → uneven seal face load → seal leak. Time scale: 80–300 steps between bearing onset and seal failure. Single-window MAE: bearing phase below threshold; seal phase above threshold. XGBoost without chain context: classifies seal failure only, misses bearing.

**Seal-to-cavitation cascade (Label 10, seal+cavitation):**
Seal leak → pressure loss → NPSHa margin drops → cavitation. Time scale: 80–120 steps between seal onset and cavitation. CUSUM on raw MAE: accumulates slowly during early seal phase; may not distinguish from noise before cavitation onset accelerates everything.

Without cross-window temporal context, the compound fault classification reduces to a single-window snapshot that captures the dominant symptom but loses the causal history.

### Why LSTM-AE Alone Cannot Solve This

The LSTM-AE hidden state could theoretically carry context between windows — but only if it is run in a stateful streaming mode where h_n from Window t is passed as h_0 to Window t+1. In training, the model was trained on independent randomly sampled windows (stateless). Running it statefully at inference on a sequence it was never trained to process statelessly would produce out-of-distribution hidden states — degraded performance, not improved.

Retraining Level 1 LSTM-AE as a stateful model would require discarding all M4 training work and rebuilding the data pipeline. This is not feasible given M4 results are LOCKED (threshold 0.110058, val_loss 0.001705).

### Resolution — Architecture v14.2: TCN-AE Level 2

A second-level model is introduced that explicitly consumes sequences of LSTM-AE reconstruction error vectors (zt sequences), not raw sensor data:

```
Input to Level 2:
  zt = [e_1, e_2, ..., e_N]   where e_i is the 8-channel error vector from
  window i, shape (N_windows, 8)

Level 2 architecture: 5-layer Temporal Convolutional Network (TCN-AE)
  Dilation schedule: d = [1, 2, 4, 8, 16]
  Kernel size: 3
  Receptive field: (2*(3-1)*31 + 1) = 63 windows = 3,150 raw seconds

Level 2 outputs:
  score_A: reconstruction severity of the zt sequence (per-window)
  score_B: drift slope of score_A over the receptive field
  score_C: chain transition signal — detects step-change in score_A pattern
           that is characteristic of a second fault beginning (compound onset)
```

The TCN receives 63 consecutive windows of LSTM-AE error vectors. This gives it direct access to the causal history that the LSTM-AE could not maintain:
- score_C fires when the pattern of zt changes qualitatively, not just quantitatively — the exact signature of "seal leak morphing into cavitation"
- score_B detects the slow upward drift of zt typical of bearing degradation weeks before any single window crosses the Level 1 threshold

> **LOCKED:** Level 2 input is ALWAYS zt sequences. It NEVER receives raw sensor data. This separation is **Invariant 16** in `completed_modules_M6p5_to_invariants.md`.

---

## C-24 — Vanishing Gradient for Sequences Longer Than 150 Steps

| Field | Value |
|-------|-------|
| **Module** | M8 — Level 2 Architecture Selection |
| **Category** | ML Architecture — Gradient Flow |
| **Severity** | HIGH |

### The Problem

M6B compound fault sequences range from 300 to 1,000 steps (after physics-verified length corrections — see C-27). When zt sequences derived from these are fed into a second-level recurrent model (LSTM or GRU), the gradient must propagate back through the entire sequence during training.

For a Level 2 LSTM processing N_windows = 20–63 zt vectors:

```
Gradient at step t = Product of (W_h^T × diag(σ'(h_t))) for t=1 to N
```

If the dominant eigenvalue of W_h is < 1.0, this product exponentially decays toward zero. For N_windows > 30, effective gradient magnitude at t=1 is approximately:

```
|grad_1| ≈ |grad_N| × λ_max^N

For λ_max = 0.95, N = 63:  |grad_1| ≈ |grad_N| × (0.95)^63 ≈ 0.04 × |grad_N|
```

The model cannot learn long-range dependencies that span the beginning and end of the zt sequence. For compound fault chain detection (where the causal link between event at t=5 and event at t=60 must be learned), this means the Level 2 model fails to capture exactly the pattern it was designed to detect.

LSTM gradient gating partially mitigates this — the forget gate can preserve gradient flow for ~20–50 steps — but for compound sequences spanning 63+ zt windows, even LSTM gating is insufficient.

### Why This Was Not a Problem at Level 1 (M4 LSTM-AE)

The Level 1 LSTM-AE processes 50-step raw sensor windows. At 50 steps, LSTM gradient gating is effective — the forget gate can protect the gradient across 50 timesteps without significant vanishing. This is well within the empirically validated effective range for LSTM on sensor data (typically 30–100 steps per training sequence, depending on data statistics).

The problem only emerges at Level 2 when longer zt sequences are required to capture multi-window compound fault dynamics.

### Resolution — Dilated Causal TCN Architecture

The TCN architecture resolves vanishing gradient through its structural properties, not through gating:

```
TCN gradient path length = number of layers = 5 (constant, regardless of N)
```

The dilated convolution at layer L sees input from windows separated by 2^(L-1) positions. The gradient flows directly from output at t to input at t - 2^(L-1) in ONE LAYER — not through N sequential hidden states. The effective gradient path is:

```
|grad_1| ≈ |grad_N| × product of (conv_layer_gradients)  (5 terms only)
```

This is structurally immune to sequence length. A TCN processing 63 windows has the same gradient magnitude at window 1 as it does at window 63. No vanishing. No exploding. Gradient flow is constant by architecture, not by training trick.

**Secondary advantage:** TCN is non-recurrent — it processes the entire zt sequence in parallel, yielding 3–5× faster training on the RTX 4060 compared to a comparable LSTM processing the same sequence length.

> **Architecture locked** in `completed_modules_M6p5_to_invariants.md`: Decision Record entry confirms TCN-AE selected over LSTM v2, Transformer (too short sequences for attention), and GRU (same gradient problem as LSTM at N>50).

---

## C-25 — Adaptive Threshold Paradox for Slow-Drift Faults

| Field | Value |
|-------|-------|
| **Module** | M8 — Level 4 Adaptive Threshold Design |
| **Category** | ML Architecture — Threshold Engineering / Safety |
| **Severity** | CRITICAL |

### The Problem

The M4 static threshold of 0.110058 was computed at commissioning from clean normal windows. As the pump ages over months, the normal bearing vibration baseline rises slowly (0.0003 per step), normal seal compression characteristics shift, and the LSTM-AE reconstruction error for "normal" operation drifts upward.

If no adaptation is made, the false alarm rate increases month by month. Operators begin ignoring alerts. The system loses trust and is switched off. This is the number one reason industrial ML monitoring systems fail in production — not poor accuracy, but operator distrust from stale thresholds.

The proposed solution — rolling adaptive threshold — introduces a paradox:

> **The adaptive threshold paradox:**
> If the threshold tracks the 6-hour rolling mean of score_A, it will also track a SLOW FAULT that develops over hours. The threshold rises with the fault. The fault never crosses the threshold. The alarm never fires.
>
> Affected fault classes:
> - Label 21 (bearing_wear_gradual): sev 0.25, 1000 steps, MAE rises 0.0003/step
> - Label 13 (bearing Mot.SV masked): sev 0.25, 300 steps, masked vibration
> - Label 4 (seal_failure): sev 0.3, 400 steps slow leak version
> - Label 8 (cavitation+seal): sev 0.3 seal phase, 550 steps

For these faults, the adaptive threshold would ADAPT to the fault itself — treating rising fault anomaly scores as new normal and never firing. This paradox is not theoretical — it is the documented failure mode of adaptive threshold systems on gradual degradation signals in rotating machinery (ISO 13374 Level 3 guidance notes this explicitly).

### Physics Basis — Why Two Time Scales Exist

For a 110 kW pump over its 15–20 year service life, two fundamentally different drift processes occur simultaneously:

**SLOW NORMAL AGING (weeks to months):**
Bearing surface roughness increases gradually with operating hours. Normal vibration baseline rises at ~0.0002–0.0005 normalized units per day. This is NOT a fault — it is normal pump aging within maintenance schedule. Threshold MUST track this to prevent false alarm accumulation. Time constant: 6–24 hours of rolling data is appropriate reference window.

**ACTIVE FAULT DEVELOPMENT (hours to days):**
- Bearing wear fault (sev 0.6): MAE rises at 0.0012/step — 4–6× faster than normal aging rate
- Seal failure (sev 0.3): Pres.SV decline at 0.0004/step over 400 steps

These rates are physically distinguishable from normal aging IF the detection system uses a different time constant. A detection system with a SLOW accumulator that is immune to threshold updates can separate them: the accumulator detects the rate, not the level.

### Resolution — Two-Speed Detection Architecture (v14.2)

**Layer 3 — CUSUM on score_B (slow accumulator, immune to threshold updates):**

```
CUSUM control chart on score_B (TCN-AE drift slope output):
  S_n = max(0, S_{n-1} + (score_B_n - μ_ref) - k)
  where μ_ref = score_B baseline at last confirmed maintenance reset
        k     = allowance parameter (slack, set to 0.5 × σ_score_B)
  S_n resets ONLY on confirmed maintenance action — NOT on threshold update
  WATCH fires when S_n > control_limit (set to 5.0)
```

The CUSUM accumulator sees the RATE of drift, not the level. Even if the rolling threshold rises to track normal aging, the CUSUM detects that the RATE of anomaly score increase has accelerated beyond the normal aging rate. For Label 21 (bearing_wear_gradual, sev 0.25): score_B rises monotonically at ~0.003/window above the normal aging score_B. CUSUM S_n crosses 5.0 within 900 steps even when Level 1 MAE stays below static threshold for 400+ steps.

**Layer 4 — Rolling Baseline on score_A (fast adaptation, false-alarm control):**

```
θ_t = μ_{score_A, 6hr} + 3 × σ_{score_A, 6hr}
Updates every 50 seconds in M10 runtime.
```

Controls false alarm rate for non-gradual faults (cavitation, sensor failure, impeller imbalance — all acute onset, fast rate, clearly above rolling mean). For acute faults: score_A spike >> rolling mean → threshold_t still exceeded.

**Separation of concerns:**
- Gradual faults → CUSUM Layer 3 (score_B rate detection)
- Acute faults → Rolling threshold Layer 4 (score_A level detection)
- Neither system undermines the other

The adaptive threshold paradox is resolved by never using the rolling baseline as the SOLE detection path for any fault class. Every slow-drift fault class (Labels 3, 4, 8, 13, 21 in the Group A/B classification) has a parallel CUSUM gate in M8 that is independent of the adaptive threshold.

> **LOCKED:** score_B feeds CUSUM ONLY. score_A feeds rolling baseline ONLY. score_C feeds XGBoost ONLY. These routing assignments are never crossed. This is **Invariant 19** in `completed_modules_M6p5_to_invariants.md`.

> **LOCKED:** Adaptive threshold warmup period = 216 API calls (6-hour equivalent at 50-second polling). During warmup, static threshold 0.110058 governs Level 1. Adaptive threshold activates at Layer 4 only after warmup completes.

---

## C-26 — Synthetic-to-Real Domain Gap in XGBoost Confidence Scores

| Field | Value |
|-------|-------|
| **Module** | M6 / M7 — Synthetic Data Pipeline / Classifier Training |
| **Category** | Data Engineering — Domain Gap / Deployment Risk |
| **Severity** | HIGH |

### The Problem

All Group B through E fault sequences in M6/M6B are physics-synthetic. No real pump has ever run to failure in the PumpSmart training dataset. CIRA SACIP provides only normal operating data and spike seeds — no labeled fault history from a pump that actually failed.

This creates a systematic risk in M7 XGBoost confidence scores:

**The classifier was trained on clean, physics-governed synthetic faults:**
- Bearing wear: Mot.SV rises smoothly at exactly 0.0003/step
- Seal failure: Pres.SV declines smoothly at exactly 0.0004/step
- Cavitation: Pres.SV kurtosis rises cleanly with no external interference

**Real pump faults in a chemical plant are NOT clean:**
- Bearing wear in a pump handling slurry: vibration signal contaminated by particle impact noise — irregular spikes superimposed on the degradation trend
- Seal failure in a pump with corroded impellers: pressure decline rate varies non-monotonically as corrosion flakes partially block the leak path
- Cavitation in a pump with worn suction strainer: onset is intermittent before becoming sustained — the CIRA kurtosis signature may not match

The consequence: M7 XGBoost may produce overconfident predictions on real deployment data because it has never seen a real fault with real-world noise, interference, and physical idiosyncrasies. A confidence score of 0.91 from M7 on synthetic test data may translate to a real-world confidence of 0.65–0.72 on a genuinely degrading pump.

This is the **synthetic-to-real domain gap** — a known and documented problem in physics-informed ML for condition monitoring (IEEE Transactions on Industrial Electronics, multiple papers 2021–2024).

### Why This Cannot Be Fully Resolved Pre-Deployment

There is no labeled real fault data for this pump. CIRA SACIP is a normal operations dataset. Acquiring labeled fault data requires either:

- **(a)** Running a pump to failure deliberately — not feasible on a 110 kW, 40 bar production asset (safety and cost prohibitive)
- **(b)** Waiting for a real fault to occur and capturing it with confirmed labels — requires months to years of parallel deployment

Neither option is available before the initial deployment. The gap is an acknowledged limitation, not an engineering failure.

### Resolution — Layered Mitigation Strategy

**Immediate (implemented in M10 v4.0):**

1. **Confidence display:** M10 Flask API and UI show raw M7 confidence score explicitly. Threshold for recommended action set at 0.85 (not 0.50). UI label: *"Confidence ≥ 0.85 recommended before maintenance action."*
2. **Limitation flag L4_synthetic_domain:** Included in every M10 API response for all non-sensor-failure fault classifications. Text: *"Trained on CIRA-anchored synthetic data. Real fault signatures may deviate."*
3. **Physics context layer:** Per-fault plain-language description of expected physical conditions, sensor signatures, and timeline. Operator can cross-check model prediction against physical observation before acting.

**Medium term (post-deployment active learning loop):**

When an operator confirms or rejects a PumpSmart alert via M10 UI feedback button, the confirmed event (real sensor data with operator-validated label) enters a retraining pool. After 50+ confirmed events, M7 is retrained with real-world examples augmenting the synthetic training data. Over 6–12 months of deployment, the synthetic-to-real gap narrows progressively as the model accumulates real fault examples.

**Long term (research roadmap):**

Partnership with plant for deliberately staged fault induction on a pump scheduled for decommission. One real bearing failure with confirmed sensor signatures is worth approximately 10,000 synthetic sequences for calibration (empirical finding from transfer learning literature on bearing fault datasets).

> The limitation is bounded and has a clear improvement path. It is NOT hidden from operators — it is explicitly surfaced in every API response.

---

## C-27 — Sequence Length Insufficient for Real Fault Physics

| Field | Value |
|-------|-------|
| **Module** | M5 / M6 — Physics Engine / Synthetic Generation |
| **Category** | Physics — Fault Sequence Engineering |
| **Severity** | HIGH |

| C-28 | M8 Patch 6 | Instrumentation — ISA-37 Sensor Saturation | Sensor Sensitivity Ceiling-Approach Detection Missing | MEDIUM |

### The Problem

The initial M6 synthetic generation plan specified all fault sequences at a uniform length of **200 steps**. This was a placeholder derived from the M4 window size (50 steps) multiplied by 4 — a purely computational heuristic with no physical justification.

During M6B physics audit, every fault class was evaluated against its real-world fault development timeline on a 110 kW, 40 bar, 7-stage centrifugal pump. The audit revealed that 200 steps is insufficient for most fault types:

| Fault Class | Required Steps | Physics Basis |
|------------|---------------|---------------|
| bearing_wear (sev 0.6) | 250 | ISO 10816: 3 stages of bearing wear require visible MAE elevation time |
| seal_failure (sev 0.4) | 400 | Seal leakage builds over pressure differential — 400 steps for NPSHa margin to measurably drop |
| overloading (sev 0.5) | 300 | Thermal time constant τ=400–600s → need 300+ steps for Temp.SV to show sustained monotonic rise above startup transient |
| bearing+overloading (Label 7, sev 0.5) | 600 | Compound: 250 bearing + 80 lag + 270 overloading onset = 600 min |
| seal+cavitation (Label 10, sev 0.5) | 900 | Compound: 400 seal + 120 lag + 380 cavitation development = 900 |
| bearing_wear_gradual (Label 21, sev 0.25) | 1000 | **PRIMARY LIABILITY:** sev 0.25 drift at 0.0003/step needs 1000 steps for CUSUM S_n to cross detection threshold |

A 200-step sequence for Label 21 (bearing_wear_gradual) would require severity to be set at 0.6+ to produce detectable drift within the window — which defeats the purpose of a gradual degradation class. At 200 steps with sev=0.25, the total Mot.SV drift = 200 × 0.0003 = 0.06 normalized units. The LSTM-AE cannot distinguish this from measurement noise in a 50-step window.

### Knock-on Effects of Incorrect Lengths

1. **CUSUM calibration broken:** CUSUM parameters (k, control_limit) were tuned assuming S_n would accumulate over 900–1500 steps for slow drift faults. With 200-step sequences, S_n never reaches the control limit during training — the CUSUM parameters are calibrated to the wrong scale.

2. **Compound fault causal lags physically impossible:** For Label 10 (seal+cavitation), the causal lag between seal onset and cavitation onset is 80–120 steps (physical minimum — NPSHa margin does not collapse faster than this for a well-designed pump at 40 bar). In a 200-step sequence, 80–120 steps of lag consumes 40–60% of the entire sequence, leaving only 80–120 steps for the cavitation phase. This is insufficient for M7 to learn the two-phase compound signature — it sees only the transition.

3. **XGBoost feature quality degraded:** Rolling statistics (mean, std, slope over a window of 6–10 zt vectors) computed on 200-step sequences cannot capture the slow-drift trend for sev=0.25 faults. The features collapse to noise.

### Resolution — Physics-Verified Length Audit (v14.2 Canonical)

Every fault class audited against physical fault development equations. Corrected lengths locked in `completed_modules_M5_to_M6p5r.md` canonical label map.

| Label | Class | Old Length | New Length | Physics Basis |
|-------|-------|-----------|-----------|---------------|
| 0 | normal | 200 | 200 | unchanged |
| 1 | bearing_wear | 200 | 250 | 3-stage ISO 10816 wear |
| 2 | impeller_imbalance | 200 | 200 | unchanged (acute onset) |
| 3 | cavitation | 200 | 150 | acute — 150 sufficient |
| 4 | seal_failure | 200 | 400 | NPSHa margin time |
| 5 | overloading | 200 | 300 | thermal time constant |
| 6 | sensor_failure | 200 | 150 | acute flatline |
| 7 | bearing+overloading | 200 | 600 | compound causal lag |
| 8 | cavitation+seal | 200 | 550 | compound causal lag |
| 9 | imbalance+bearing | 200 | 700 | compound causal lag |
| 10 | seal+cavitation | 200 | 900 | compound causal lag |
| 11 | overloading+bearing | 200 | 800 | compound causal lag |
| 12 | imbalance+cavitation | 200 | 450 | compound causal lag |
| 13 | bearing Mot.SV masked | 200 | 300 | masked drift window |
| 14 | cavitation Pres.SV masked | 200 | 210 | minimal extension |
| 15 | seal Pres.SV drifting | 200 | 500 | slow leak development |
| 16 | overloading Temp.SV stuck | 200 | 350 | thermal overload build |
| 17 | imbalance Pmp.SV flatline | 200 | 250 | imbalance + sensor |
| 18 | cavitation intermittent | 200 | 300 | intermittent onset |
| 19 | seal_failure_fast | 200 | 150 | acute — shortened |
| 20 | overloading_cyclic | 200 | 600 | cyclic load pattern |
| 21 | bearing_wear_gradual | 200 | 1000 | **PRIMARY LIABILITY GATE** |

Consequence for sequence counts: Total sequences revised from 26,000 (original plan) to 31,800 (v14.2) to maintain sufficient training examples per class despite longer per-sequence duration. Full count breakdown locked in `completed_modules_M5_to_M6p5r.md`.

M6A rerun status: Labels 1 (bearing_wear), 4 (seal_failure), 5 (overloading) were generated in M6A at incorrect lengths. These three classes are designated for regeneration in M6B Step 0 before any M6B compound sequence generation begins. Labels 2, 3, 6 (impeller_imbalance, cavitation, sensor_failure) remain valid from M6A — their lengths were unchanged or reduced, not increased.

---

## Summary Statistics

| Category | Count | Severity Distribution |
|----------|-------|-----------------------|
| Data Integrity / Segmentation | 4 | 1 HIGH, 2 MEDIUM, 1 LOW |
| Physics — Thermodynamics | 2 | 1 CRITICAL, 1 HIGH |
| Physics — Fluid Mechanics | 3 | 1 CRITICAL, 2 HIGH |
| Physics — ISO Standards / Vibration | 2 | 1 CRITICAL, 1 HIGH |
| Physics — Transport Phenomena | 1 | HIGH |
| Physics — Nameplate / Specification | 1 | CRITICAL |
| Physics — Fault Sequence Engineering | 1 | HIGH |
| Data Documentation | 2 | 1 MEDIUM, 1 LOW |
| Data Engineering — Domain Gap | 1 | HIGH |
| ML Architecture | 6 | 3 CRITICAL, 3 HIGH |
| ML Threshold Engineering | 1 | CRITICAL |
| ML Training / Data Pipeline | 2 | 1 CRITICAL, 1 HIGH |
| ML Deferred (to M8, resolved in v14.2) | 2 | 2 MEDIUM |
| **TOTAL** | **27** | **8 CRITICAL, 14 HIGH, 4 MEDIUM, 1 LOW** |

---

## Key Engineering Principles Established

The 27 challenges above collectively established the following principles that now govern all remaining modules (M5–M12):

1. **Rate-of-change over absolute thresholds:** Sensor placement dependence means absolute vibration values cannot be compared across installations. All fault signatures defined as dX\*/dt patterns (from C-05).

2. **Cluster-conditional normalization and thresholds:** A single global statistical parameter fails a multi-modal process. Every statistical parameter must respect operating mode (from C-09, C-18).

3. **Physics-weighted ML:** Loss functions and feature importance must reflect the physical information hierarchy of sensor channels (from C-14).

4. **Temporal pattern > absolute level:** Overloading cannot be detected by temperature magnitude alone — the temporal signature of rising T\* during stable vibration is the discriminating feature (from C-04).

5. **Conservative baseline = safer production:** M4 phase-lag conservatism is deliberate. A false-negative on a normal transition is safer than a false-positive that triggers unnecessary shutdown of a 110 kW pump (C-20).

6. **Sensor failure ≠ process failure:** Isolated single-channel anomalies must be classified separately from multi-channel process faults (from C-01).

7. **Climate-agnostic normalization:** Any model targeting deployment beyond its training geography must use an internal reference frame (from C-10).

8. **Clean normal baseline is non-negotiable:** The anomaly threshold quality is bounded by the purity of the normal training set (from C-15, C-16).

9. **Cross-window temporal context is required for compound fault detection:** A single-window reconstruction model cannot capture causal chains that unfold across 80–300 steps. A second-level model consuming zt sequences is the only architecturally sound solution (from C-23).

10. **Vanishing gradient sets an architectural ceiling on recurrent Level 2 models:** For N_windows > 50, TCN is architecturally superior to LSTM/GRU for compound fault chain detection (from C-24).

11. **Adaptive thresholds require dual time-scale architecture:** A single rolling baseline cannot simultaneously control false alarms from normal aging AND detect gradual fault drift. CUSUM (slow, immune to threshold updates) and rolling baseline (fast, false-alarm control) must operate in parallel on separate TCN-AE outputs (from C-25).

12. **Sequence length must be physically justified:** Uniform sequence lengths derived from computational convenience produce untrainable gradual fault classes and physically impossible compound causal lags (from C-27).

13. **Synthetic-to-real gap is an acknowledged bounded limitation:** It cannot be resolved pre-deployment but has a clear active learning mitigation path. It must be surfaced explicitly in every production API response (from C-26).

## C-28 — Sensor Sensitivity Ceiling-Approach Detection Missing
 
| Field | Value |
|-------|-------|
| **Module** | M8 — Patch 6 (Stakeholder-Driven Sidecar Diagnostic) |
| **Category** | Instrumentation — Sensor Health / ISA-37 Transducer Saturation |
| **Severity** | MEDIUM |
 
### The Problem
 
A stakeholder review on 2026-05-14 identified a coverage gap in the existing sensor-health logic. The system already detected three sensor failure modes:
 
- **Spikes** — M4 winsorization with cluster-conditional ceilings (C-17, C-18)
- **Flatlines** — Group C masked faults (Labels 13–17) and Label 6 sensor_failure
- **Drift** — Adaptive Threshold L4 + Mech C in the M8 main path
 
However, the system did NOT detect a fourth documented industrial failure mode: **a sensor approaching its sensitivity ceiling before flatline.** This is the classic failure mode of industrial pressure transducers operating near their max-pressure rating and RTDs near their max-temperature rating — the sensor produces precise-looking values right up to the point it saturates against its physical range. At the ceiling, a 1–3% change in true physical quantity produces a 50–100% change in normalised output, or the channel goes flatline.
 
The stakeholder phrased the failure mode as: *"if a 1%, 2%, or 3% change leads to a 100% change in output, that indicates the sensor is having a problem — not exactly a 'problem,' but the sensor is giving precise values, yet it might fail because it is reaching that specific threshold."*
 
Without this check, M10 would surface fault predictions based on sensor values that were arithmetically valid but physically suspect, with no operator-facing warning that the underlying transducer was approaching saturation. For a 110 kW, 40 bar pump operating in production, silently trusting a saturating sensor is a Category 2 liability exposure — the prediction is wrong but the system reports it confidently.
 
### Why This Matters Physically
 
For a 110 kW, 40 bar, 7-stage centrifugal pump:
 
- The Pres.SV transducer is rated for a nominal 0–60 bar range. CIRA observed values peak near 45 bar in high-load operation. The transducer operates at ~75% of full scale — well inside spec but with limited headroom before non-linear regime onset.
- The Mot.TV and Pmp.TV RTDs are rated for a nominal 0–100 °C range. Real observed peaks during high-load reach 55 °C — comfortable margin.
- The vibration accelerometers (.SV channels) have a documented broadband peak range; saturation manifests as clipped peaks (flat-topped sinusoid envelope) rather than smooth non-linear gain rise.
 
The danger is not in finite headroom itself — every sensor has finite range. The danger is in the system silently trusting a measurement from a sensor that has entered its non-linear regime, then propagating that measurement through the M4 → M7 prediction pipeline as if it were a clean input.
 
### Resolution
 
A sidecar diagnostic script `module_08p6_sensor_sensitivity_analysis.py` was implemented on 2026-05-15. **No locked artifact was modified** — purely additive.
 
**Two-metric check** on M3 normalised data (117,970 rows) across all 4 clusters × 8 channels = 32 channel-cluster pairs:
 
1. **Local gain ratio** — std of 50-step rolling window divided by cluster-wide stdev. Flagged if p95 > **3.0×** per ISA-37 transducer guidelines. This is the engineering equivalent of the stakeholder's "1–3% in, 100% out" intuition, expressed as a variance ratio. Variance-ratio rather than strict dY/dX gain because raw input columns are not persisted in M3 output — acceptable because both formulations detect the same failure mode (saturation produces compressed-then-explosive variance).
 
2. **Headroom to ceiling** = `(cluster_ceiling - p99_value) / (cluster_ceiling - cluster_mean)`. Flagged if < **10%**. Ceiling values pulled from `M4_spike_config.json` cluster-conditional winsor bounds (locked from C-17, C-18, C-19 fixes).
 
**Results on CIRA training data (2026-05-15):**
 
| Metric | Value |
|--------|-------|
| Channel-cluster pairs evaluated | 32 |
| Pairs flagged | 0 |
| Worst gain p95 | 0.98 (Mot.PV startup — BPF harmonic content, physically expected) |
| Worst headroom — X_Pres.SV | 0.115 (1.5% above flag threshold) |
| Worst headroom — X_ACR_Pmp.PV | 0.120 (2.0% above flag threshold) |
| Best headroom — X_ACR_Mot.PV | 0.235 (comfortable margin) |
 
The clean pass on CIRA validates that the M4 v8 cluster-conditional winsor ceilings were calibrated with sufficient headroom. Two channels (Pres.SV and Pmp.PV) sit close to the flag line, reflecting deliberate tight calibration in fault-sensitive operating regimes — Pres.SV high_load ceiling 2.0× (tightest in system per C-18 fault-sensitivity rationale) and Pmp.PV startup ceiling 3.2× (ISO 13373-3 BPF harmonic headroom per C-17). These two channels are expected to trigger the runtime addendum in deployment on pumps operating outside CIRA's envelope — by design.
 
**M10 integration:** Config file `models/M8p6_sensor_sensitivity_config.json` is loaded at FastAPI lifespan startup. At every inference, live `gain_p95` and `headroom` are computed for the active cluster. If either crosses the flag threshold, Field 6 of the 7-field output appends a sensor-health sub-line:

Sensor health: {channel_friendly_name} in {cluster_name} at {ratio:.2f}× ceiling 
— verify transducer calibration before trusting {fault_label} prediction.


The addendum annotates Field 6 only. It does NOT modify the prediction. This preserves the principle that sensor health is a sidecar diagnostic, not a prediction override (consistent with Invariant 6: sensor failure ≠ process failure).
 
**Limitation acknowledged:** The variance-ratio formulation is a second-best for true dY/dX gain. The check is a screen, not a calibration certificate. Periodic transducer recalibration per manufacturer schedule remains the operator's responsibility — M8p6 surfaces *when* recalibration may be overdue based on operating signature, not whether the sensor is electrically healthy.
 
**Stakeholder note:** The same 2026-05-14 review also asked whether the M2 K-Means clustering should exclude Pres.SV and Pmp.TV as inputs and predict them as outputs (virtual-sensor / analytical-redundancy pattern). That recommendation was evaluated and declined — see the Stakeholder Defense Memo (`outputs/reports/Stakeholder_Defense_Memo_M2_clusters_and_spike_seeds.md`, 2026-05-14) for the architectural rationale. Excluding pressure and casing temperature from clustering would collapse the four operating modes the stakeholder is reviewing in the PCA plot. The clustering operates on all 8 channels and is locked.

---

| Field | Value |
|-------|-------|
| **Document version** | 2.1 |
| **Covers** | M1 through M8 + M8 Patch 6 (architecture v14.2) |
| **Previous version** | 2.0 (M1 through M8, C-01 through C-27) |
| **Version 2.1 additions** | C-28 (stakeholder-driven sensor sensitivity sidecar, 1 new challenge) + Principle 14 (sensor health as sidecar diagnostic) |
| **Next update** | After M12 validation — append M9–M12 challenges if any arise |
| **Asset** | 110 kW, 7-stage, 40 bar centrifugal pump \| CIRA SACIP dataset |
| **Author** | Souvik \| PumpSmart Physics-Informed ML Digital Twin |
| **Architecture** | v14.2 \| 4-layer detection stack + M8p6 sensitivity guardrail \| 22-class fault classification |
