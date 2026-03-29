# PumpSmart — Non-Technical Stakeholder Report
**Project:** Physics-Informed AI for Industrial Pump Health Monitoring
**Version:** 1.0 | **Date:** 29 March 2026 | **Author:** Souvik
**Status:** Active Development — Modules 1–4 Complete

---

## At a Glance

> **PumpSmart is an AI-powered early warning system for large industrial pumps.**
> It monitors sensor data in real time, detects the first signs of mechanical failure before they become catastrophic, names the specific fault, and explains *why* it raised an alarm — all grounded in the proven laws of engineering physics.

---

## 1. What Problem Does This Solve?

### The Machine at the Centre of This Project

The pump PumpSmart is designed for is not a household appliance. It is a **7-stage, high-pressure centrifugal pump** running at 2,980 rotations per minute, delivering fluid at pressures up to **40 bar** — roughly 40 times atmospheric pressure — powered by a **110 kilowatt motor**. In an Indian industrial context, a pump of this class costs **more than ₹50 lakh** to replace. In a process plant, it often cannot be substituted quickly: when it fails, the line stops.

### How Pumps Fail Today

Large industrial pumps fail in predictable, physics-driven ways. The most common causes are:

- **Bearing wear** — rotating bearings degrade over time, generating heat and increasing vibration
- **Impeller imbalance** — the spinning wheel inside the pump develops an uneven mass distribution, causing oscillations that worsen rapidly
- **Cavitation** — at low inlet pressures, the fluid locally vaporises and collapses inside the pump, causing violent micro-shocks that pit and erode the impeller
- **Seal failure** — the mechanical seal between rotating and stationary parts degrades, causing internal leakage, pressure loss, and heat build-up
- **Overloading** — the motor is driven beyond its rated duty point for extended periods, causing thermal damage
- **Sensor failure** — the monitoring hardware itself malfunctions, which must be distinguished from actual pump faults

The traditional response to these faults is either **reactive maintenance** (fix it after it breaks, at maximum cost and disruption) or **time-based preventive maintenance** (replace parts on a fixed schedule, regardless of actual condition, wasting serviceable components). Both approaches are expensive, imprecise, and — in the case of sudden failures — dangerous.

### What Industry Needs

The ideal system would:
1. Watch the pump continuously, 24 hours a day
2. Detect the *earliest* sign of a developing fault — before it becomes audible, visible, or catastrophic
3. Tell the operator not just *that* something is wrong, but *what* is wrong and *which sensor* is showing it
4. Be explainable enough that a maintenance engineer trusts it and acts on it
5. Avoid false alarms that cause unnecessary shutdowns

This is what PumpSmart is being built to do.

---

## 2. What Makes PumpSmart Different?

### Most ML Projects and Why They Fall Short

The vast majority of machine learning projects for industrial equipment take the following approach: collect historical sensor data, train a model on it, and deploy. This approach has a fundamental problem in the context of pump fault detection: **real fault data is almost never available**.

Industrial operators do not deliberately run pumps to failure to generate training labels. When a fault does occur, it is typically an uncontrolled emergency — the fault type is ambiguous, the sensor data is incomplete, and the ground truth label is debated. Standard ML models trained without labeled fault data cannot tell the difference between a bearing wearing out and a pump starting up.

### PumpSmart's Approach — Physics First

PumpSmart takes a fundamentally different path. Every fault signature in the system is derived from the **laws of thermodynamics, fluid mechanics, and mechanical engineering** — not guessed from historical incident reports.

For example:
- Bearing wear is encoded as an exponential rise in vibration governed by the Paris fatigue law, followed by heat build-up in the motor casing after a physically-derived thermal lag — because that is what the heat transfer equation `Q = mCp × dT/dt` dictates for a 110 kW motor frame
- Cavitation is encoded as chaotic, erratic pressure drops at the pump inlet during low-pressure startup conditions — because that is when the Net Positive Suction Head Available (NPSHa) falls below the pump's minimum requirement
- Overloading is encoded as a monotonically rising temperature during a period of *stable* vibration — because rising temperature at constant speed can only come from sustained excess power input, not from mechanical noise

This means every synthetic training example the AI learns from is **physically correct by construction**. There is no guesswork about what a fault looks like.

### Complete Control Over Ground Truth

Because PumpSmart generates its own fault training data using physics equations, it has something rare in industrial AI: **perfect labels**. The system knows exactly:
- Which fault type is present
- At which moment in time the fault began
- How severe it is
- Which sensors should respond and in which order

This makes it possible to test the model against scenarios it has never seen before — and to know with certainty whether it passes or fails. The final module of the project (M12) is dedicated entirely to this adversarial testing.

---

## 3. How Does the System Work? (Plain Language)

Think of PumpSmart as having three layers:

### Layer 1 — The Librarian (What Is Normal?)

The first job of the AI is to learn — with extreme precision — what a healthy pump looks like at every stage of its operation cycle: starting up, running at full load, running at partial load, and cooling down after shutdown.

It does this by studying over **117,000 real sensor readings** from three identical industrial pumps recorded over multiple days at the CIRA research institute in Italy. It finds that pumps naturally operate in four distinct modes (startup, high-load, steady-state, cooldown) and builds a precise mathematical fingerprint of each mode.

When a new sensor reading arrives, it asks: *"Does this look like what I expect from a healthy pump in this operating mode?"* If the answer is no — and the answer is meaningfully, consistently no over a window of 50 seconds — it raises an alarm.

This is the **LSTM-Autoencoder** (the core AI model). LSTM stands for Long Short-Term Memory — it is a type of AI architecture specifically designed for data that flows through time, like sensor readings. The autoencoder part means it is trained to reproduce its input perfectly; anything it cannot reproduce well is, by definition, unusual.

### Layer 2 — The Detective (What Is Wrong?)

Knowing that something is wrong is useful. Knowing *what* is wrong is actionable.

Once the LSTM-Autoencoder flags an anomaly, a second model — an **XGBoost classifier** — takes a snapshot of the sensor data from that moment and classifies it into one of seven categories: normal, bearing wear, impeller imbalance, cavitation, seal failure, overloading, or sensor failure.

XGBoost is a well-established, highly interpretable AI technique used widely in engineering and finance. Crucially, it can explain its own decisions through a method called **SHAP** (SHapley Additive exPlanations). When it says "this is bearing wear", it can also say "the primary evidence is that Motor Vibration rose by 40% over the last 30 seconds, followed 25 seconds later by Motor Temperature rising at a rate inconsistent with normal thermal dynamics." A maintenance engineer can look at that explanation and decide whether to act.

### Layer 3 — The Examiner (How Do We Know It Works?)

This is the most unusual part of PumpSmart and the one that most distinguishes it from academic demonstrations.

After the model is trained and deployed, a dedicated **Physics-Governed Validation Suite (M12)** is used to stress-test it against completely fresh fault scenarios it has never seen during training. These scenarios are generated by the same physics engine, but with different parameter combinations — a bearing fault with a faster growth rate, a cavitation event with higher sensor noise, a seal failure starting from a different operating pressure.

The key metric is **detection latency**: how many seconds after a fault begins does the model raise an alarm? For a bearing fault, physics dictates that from first detectable vibration to mechanical failure is typically 600–1,800 seconds. The operator needs at least 300 seconds to respond. That leaves a maximum tolerable detection lag of roughly 300 seconds. PumpSmart's safety gate is set at **60 seconds** — a 5× safety margin.

For cavitation — the most time-critical fault, because impeller pitting begins within 60–180 seconds of onset — the gate is tighter: **30 seconds**.

If the model fails any of these gates, it is automatically flagged for retraining. Only when all gates pass does the system receive a **Production Validated** certificate.

---

## 4. What Has Been Built So Far?

Four of the twelve development modules are complete.

### ✅ Module 1 — Data Cleaning (Complete)

All 9 raw sensor data files from the CIRA dataset (3 pumps × 3 operational days) were processed. Out of 173,730 raw readings, 147,217 clean readings were retained after removing corrupted, missing, and anomalous data. The data was carefully segmented so that no two time segments are accidentally joined — a critical data integrity requirement for time-series AI.

One particularly noteworthy finding: Pump B on Day 3 had its atmospheric pressure sensor fail completely, logging no valid readings for 65% of the recording. However, the pump itself continued running normally. This real-world sensor failure event was preserved as a genuine example of "sensor fails, pump works" — exactly the kind of Sensor Failure fault type the model must learn to recognize.

### ✅ Module 2 — Understanding the Pump's Operating Modes (Complete)

Using a clustering algorithm (K-Means), the project identified that the pump naturally operates in exactly **4 distinct modes** throughout its duty cycle. These modes — startup, steady-state, high-load, and cooldown — have measurably different sensor signatures. Knowing which mode the pump is in is essential for fair fault detection: a vibration level that is perfectly normal at high load would be alarming during a quiet cooldown.

The analysis also confirmed three strong physical relationships in the data — for example, the motor temperature and the fluid temperature track each other with a Pearson correlation of 0.979, a near-perfect relationship that must be preserved in all synthetic training data. If a fault breaks this coupling, that itself is a diagnostic signal.

### ✅ Module 3 — Standardizing the Data (Complete)

Raw sensor values — pressure in bar, vibration in mm/s, temperature in °C — cannot be directly compared or combined in a machine learning model. Module 3 converted all readings into dimensionless ratios relative to each operating mode's normal range. A pressure of 40 bar at high load and a pressure of 0.7 bar at startup are both "normal" for their respective modes; after normalization, both appear as a value near 1.0. A fault manifests as a value drifting above 1.0.

This normalization is also **climate-agnostic**: the formulas use the pump's own operating range as the reference, not the ambient air temperature. This means the same model can be deployed in the heat of Kolkata or the climate of Naples, Italy (where the training data was recorded) without recalibration.

### ✅ Module 4 — Training the Anomaly Detection AI (Complete)

The core LSTM-Autoencoder model was trained on over 9,700 windows of normal pump operation. Key results:

| What Was Measured | Result | What It Means |
|---|---|---|
| Normal reconstruction error | 0.027 (on a 0–1 scale) | The model reproduces healthy pump data almost perfectly |
| Anomaly detection threshold | 0.110 | Any window with error above this is flagged as unusual |
| Separation ratio | 4.1× | Fault signals are 4× more "surprising" to the model than normal signals |
| False alarms on validation data | 8 out of 1,457 windows (0.55%) | Less than 1 in 200 healthy readings incorrectly flagged |
| Training time | ~51 seconds | GPU-accelerated on RTX 4060 |

The model also extracted 1,044 real anomalous windows from the sensor data — transient spikes and pressure surges — which will be used as seeds for generating realistic synthetic fault sequences in a later module.

---

## 5. What Is Still Being Built?

### 🔲 Module 5 — Physics Equations Engine

This module encodes all the governing equations of the pump's physics into Python functions. Every fault type will have a mathematically derived "signature" — a precise description of how each sensor should respond, in what sequence, and at what rate. This is the foundation that makes synthetic fault data physically honest rather than statistically fabricated.

### 🔲 Module 6 — Synthetic Fault Data Generator

Using the physics engine from Module 5, this module will generate 660 labeled fault sequences — covering all six fault types at three severity levels. This is the training library for the fault classifier, and it is built entirely from first principles.

### 🔲 Modules 7 & 8 — The Fault Classifier and Production AI Model

Module 7 trains the XGBoost fault classifier (the "Detective" described above). Module 8 retrains the LSTM-Autoencoder with a more sophisticated architecture — including the ability to express uncertainty in its predictions and to pay more attention to the exact moments within a time window where something goes wrong, rather than treating every second equally.

### 🔲 Modules 9–11 — Application and Deployment

Module 9 builds a pump selection tool (for engineers specifying a new pump installation) and a household pump advisor (for non-industrial users needing sizing guidance — clearly labelled as advisory only, with no AI fault prediction). Modules 10 and 11 package everything into a web application, containerize it with Docker, and deploy it to Hugging Face Spaces for public access.

### 🔲 Module 12 — Physics-Governed Validation Suite

The final and most safety-critical module. As described in Section 3, this stress-tests the deployed model against unseen fault scenarios with known ground truth. It runs a mandatory suite of 12 test configurations — including multi-fault scenarios (two faults developing simultaneously) and a deliberately broken sensor coupling (to test robustness when sensors are relocated to a different installation). The model receives a Production Validated certificate only when all safety gates pass.

---

## 6. Who Is This For?

### Industrial Scope (AI-Backed)
The full anomaly detection and fault classification capability applies to **large industrial centrifugal pumps** matching the class of the training data — multi-stage, high-pressure, motor-driven. Target users are plant operators and maintenance engineers who upload sensor CSV files or connect live sensor feeds to the web application.

### Household Scope (Physics-Only Advisory)
A separate section of the application provides pump sizing and maintenance guidance for small household or agricultural pumps. This section uses only physics calculations — no AI inference — and is clearly labelled as advisory guidance. This boundary is firm: household monoblock pumps are mechanically different enough that applying the industrial AI model to them would be scientifically unsound and potentially misleading.

---

## 7. Safety and Trustworthiness

Every design decision in PumpSmart has been made with the following principle: **the consequences of a missed fault on a ₹50 lakh, 110 kW, 40 bar industrial pump are too severe to accept any shortcut in validation**.

Specific commitments built into the system:

- **No raw sensor values ever enter the AI** — all data is converted to physics-derived dimensionless ratios first
- **No fault data from M6 training is reused in M12 validation** — the adversarial test suite uses completely fresh, unseen parameter configurations
- **Explainability is non-negotiable** — SHAP values accompany every XGBoost prediction so engineers can interrogate the reasoning, not just the conclusion
- **The model knows when it is uncertain** — MC Dropout uncertainty quantification gives a confidence interval on every anomaly flag, not just a binary yes/no
- **A UI disclaimer is mandatory** — the web application must display, before any inference, that the model is trained on one specific CIRA sensor installation, that sensor placement must follow ISO 13373 guidelines, and that model outputs are advisory pending qualified engineering review
- **The household boundary is enforced in code** — the application contains a hard-coded check: `if pump_type == 'household': return physics_advisory_only()`. Household pumps cannot reach the industrial AI model, regardless of user input

---

## 8. Technical Foundations

| Aspect | Detail |
|---|---|
| Training data | CIRA SACIP dataset (Zenodo record 15301820) — real Italian industrial pump sensor data |
| Pump nameplate | 110 kW motor, 7-stage impeller, 2,980 RPM, 40 bar max, 45 m³/h flow, 450 m head |
| Standards followed | ISO 10816-3 (vibration), ISO 13373-3 (condition monitoring) |
| AI architecture | LSTM-Autoencoder (anomaly detection) + XGBoost (fault classification) |
| Explainability | SHAP (SHapley Additive exPlanations) — industry-standard interpretability method |
| Uncertainty | MC Dropout — 20 inference passes per prediction, outputs confidence interval |
| Hardware | NVIDIA RTX 4060 GPU training; CPU-only deployment (standard cloud compatible) |
| Deployment | Docker container on Hugging Face Spaces; Flask web application |
| Physics basis | Thermodynamics, fluid mechanics, ISO pump standards, Paris fatigue law, NPSH theory |
| Developer | Souvik — 2nd year Chemical Engineering undergraduate, CIRA-validated training pipeline |

---

## 9. Project Repository and Data Sources

- **GitHub Repository:** [github.com/Souvik-1234-cpu/PumpSmart_Project](https://github.com/Souvik-1234-cpu/PumpSmart_Project)
- **Training Dataset:** [zenodo.org/records/15301820](https://zenodo.org/records/15301820) — CIRA SACIP Industrial Pump Dataset
- **Module Pathway Document (Technical):** `module_pathway_M1_to_M12_v8.md` in the repository — full specification of all 12 modules, validation gates, and cross-module invariants

---

## 10. Summary of Current Status

| Module | Name | Status | Key Output |
|---|---|---|---|
| M1 | Data Cleaning | ✅ Complete | 117,970 clean, segmented sensor readings |
| M2 | Operating Mode Analysis | ✅ Complete | 4 operating modes identified; physical couplings confirmed |
| M3 | Data Standardization | ✅ Complete | Climate-agnostic normalization locked |
| M4 | Anomaly Detection AI (Baseline) | ✅ Complete | Threshold 0.110; separation ratio 4.1×; <1% false alarms |
| M5 | Physics Equations Engine | 🔲 Active | 10 pump physics equations; fault causal chains |
| M6 | Synthetic Fault Generator | 🔲 Pending M5 | 660 labeled fault sequences from physics |
| M7 | Fault Classifier (XGBoost) | 🔲 Pending M6 | 7-class fault identification + SHAP explanations |
| M8 | Production AI Model | 🔲 Pending M6 | Upgraded model with uncertainty + attention mechanism |
| M9 | Pump Selector + Advisor | 🔲 Pending M8 | Industrial selector + household advisory tool |
| M10 | Web Application (Flask) | 🔲 Pending M9 | Full web interface for all tools |
| M11 | Deployment (Docker + HF) | 🔲 Pending M10 | Public web deployment |
| M12 | Physics Validation Suite | 🔲 Pending M11 | Production-validated certificate or recalibration flag |

---

*PumpSmart — Physics-Informed AI for Industrial Pump Health Monitoring*
*Report version 1.0 | 29 March 2026 | For questions, contact project author Souvik*
