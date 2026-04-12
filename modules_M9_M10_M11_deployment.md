# PumpSmart — M9 + M10 + M11: Deployment Modules
## Pump Selector | Flask Web Application | Docker + Hugging Face Deployment

**Document version:** v2.0 — Post Bias-Audit Cascade  
**Date:** 2026-04-12  
**Prerequisites:** M8 all_13_gates_pass = True | M7 all_10_gates_pass = True  
**Status:** All NOT STARTED — begin after M8 gates confirmed  
**Pump:** 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP

---

## Prerequisite Chain

```
M7 all_10_gates_pass = True
  → M8 all_13_gates_pass = True
      → M9 (physics tools) — can run in parallel with M8 after M7 completes
      → M10 (Flask app) — requires M8 models + M7 models + M9 physics tools
      → M11 (deployment) — requires M10 fully tested locally
      → M12 adversarial validation — run AFTER M11 deployment confirmed
```

---

## What Changed in v2.0 (Bias-Audit Cascade)

| Item | v1.0 | v2.0 | Source |
|------|------|------|--------|
| M7 classify route output | Single fault + confidence | **Progressive Stage 1/2/3 + secondary faults** | M7 multi-label arch |
| classify_fault input | 24 features | **25 features (+ compound_interaction_flag)** | M6.5 v3 |
| XGBoost model loading | `xgb.Booster()` | **`MultiOutputClassifier` + 7 binary models** | M7 arch change |
| WARN state UI detail | Fault type if conf > 0.7 | **Stage 2 output: probable fault + secondary candidates** | M7 Stage 2 |
| DANGER state UI detail | Single fault class | **Stage 3: confirmed + secondary faults + fault_stage** | M7 Stage 3 |
| Compound fault display | Not present | **Both faults shown in output if secondary prob > 0.30** | Bias 4 |
| Docker requirements | `xgboost>=2.0` | **+ `scikit-learn>=1.3` (for MultiOutputClassifier)** | M7 arch |
| M10 local tests | 10 tests | **+2 compound fault tests = 12 tests** | Bias 4 |

---

## M9 — Pump Selector + Household Advisor

**Status:** 🔲 NOT STARTED (can begin after M7 completes)  
**Nature:** PHYSICS ONLY — no ML inference in this module

```
SCOPE BOUNDARY (NEVER VIOLATE):
  if pump_type == 'household': return physics_advisory_only()
  else: return ml_prediction()   ← routes to M8 + M7 in M10

Household monoblock pump ≠ industrial multistage pump.
Cross-domain ML = out-of-distribution inference = safety risk.
Household advisor label in M10 UI: "Advisory guidance only — not a monitoring tool"
```

---

### M9 Part A — Industrial Pump Selector

#### Physics Equations Implemented

```
1. HYDRAULIC POWER:
   P_hyd = ρ × g × Q × H
   where: ρ = fluid density (kg/m³), g = 9.81 m/s²,
          Q = flow rate (m³/s), H = total head (m)
   Nameplate check: P_hyd = 1000 × 9.81 × (45/3600) × 450 = ~55.2 kW ✓

2. SHAFT POWER (motor requirement):
   P_shaft = P_hyd / η_pump
   where η_pump = pump hydraulic efficiency (default 0.65 for multistage)
   P_shaft = 55.2 / 0.65 = ~84.9 kW → motor selection: 110 kW (standard IEC frame)

3. TOTAL HEAD (multi-component):
   H_total = H_static + H_friction + H_velocity
   H_static   = elevation difference (m)
   H_friction = f × (L/D) × (v²/2g)   [Darcy-Weisbach]
   H_velocity = v² / 2g

4. NPSH AVAILABLE (NPSHa):
   NPSHa = (P_atm - P_vapour) / (ρg) + H_suction - H_friction_suction
   where P_vapour = vapour pressure at operating temperature

5. NPSH REQUIRED (NPSHr):
   NPSHr = pump-specific value from manufacturer curve
   CAVITATION RISK FLAG: if NPSHa < NPSHr + 0.5m (safety margin) → flag

6. NPSH MARGIN CHECK:
   margin = NPSHa - NPSHr
   if margin < 0.5m  → CAVITATION_RISK = True  (startup vulnerable)
   if margin < 0.0m  → CAVITATION_CERTAIN = True (do not operate)

7. AFFINITY LAWS (speed change):
   Q2/Q1 = N2/N1
   H2/H1 = (N2/N1)²
   P2/P1 = (N2/N1)³
   Use: estimate performance at variable speed / partial load

8. SPECIFIC SPEED (pump type selector):
   Ns = N × Q^0.5 / H^0.75   [SI units: RPM, m³/s, m]
   Ns < 25       : Radial flow (high head, low flow) → multistage centrifugal
   25 < Ns < 70  : Mixed flow
   Ns > 70       : Axial flow (low head, high flow)
   Nameplate: Ns = 2980 × (45/3600)^0.5 / 450^0.75 = ~10.2 → radial ✓

9. JOUKOWSKY WATER HAMMER (transient pressure check):
   ΔP = ρ × a × Δv
   where a = pressure wave velocity (~1200 m/s in steel pipe)
   Δv = sudden velocity change (valve closure)
   Max transient: P_operating + ΔP must not exceed pipe pressure rating
   Nameplate transient observed: 46.7 bar (confirmed M2 data)

10. STAGE HEAD (multistage):
    H_per_stage = H_total / n_stages
    Nameplate: 450 / 7 = 64.3 m per stage
    Verify: per-stage head within impeller design range
```

#### Industrial Selector I/O

```
INPUT:
  flow_rate_m3h    : float  (required m³/h)
  total_head_m     : float  (required head in m)
  fluid_density    : float  (kg/m³, default 1000)
  fluid_temp_c     : float  (°C, for vapour pressure lookup)
  suction_head_m   : float  (positive = flooded, negative = suction lift)
  pipe_length_m    : float
  pipe_diameter_m  : float
  speed_rpm        : float  (default 2980 for 50Hz 2-pole)

OUTPUT:
  hydraulic_power_kw    : float
  required_shaft_kw     : float
  recommended_motor_kw  : float  (next IEC standard frame above shaft)
  npsha                 : float
  npshr_margin          : float
  cavitation_risk       : bool
  specific_speed        : float
  pump_type             : str  ("multistage_centrifugal" / "mixed_flow" / "axial")
  stage_head_m          : float
  water_hammer_dp_bar   : float
  warnings              : list[str]
  recommendation        : str
```

#### Validation Test Cases (M9 Gate)

```
TEST-M9-1: Nameplate reproduction
  Input : Q=45 m³/h, H=450m, ρ=1000, η=0.65, N=2980
  Expect: P_hyd ≈55.2 kW, P_shaft ≈84.9 kW, motor=110 kW, Ns≈10.2

TEST-M9-2: Cavitation risk flag
  Input : NPSHa=3.2m, NPSHr=3.0m (margin=0.2m < 0.5m)
  Expect: cavitation_risk = True, warning issued

TEST-M9-3: Affinity law speed reduction
  Input : N1=2980, Q1=45, H1=450, N2=2500
  Expect: Q2 = 45 × (2500/2980) = 37.75 m³/h
          H2 = 450 × (2500/2980)² = 317.8 m
          P2/P1 = (2500/2980)³ = 0.591

TEST-M9-4: Water hammer transient
  Input : ρ=1000, a=1200 m/s, Δv=2.5 m/s, P_operating=40 bar
  Expect: ΔP = 1000×1200×2.5/100000 = 30 bar
          P_transient = 70 bar → WARNING: exceeds 40 bar nameplate

TEST-M9-5: Specific speed pump type
  Input : N=2980, Q=45 m³/h, H=450m
  Expect: Ns ≈10.2 → pump_type = "multistage_centrifugal"

GATE-M9-1: All 5 test cases PASS
GATE-M9-2: No unphysical outputs (negative pressure, T below ambient, Ns < 0)
GATE-M9-3: Household pump_type → physics_advisory_only() returns, no ML call
```

---

### M9 Part B — Household Advisor (Physics Only)

```
SCOPE: Domestic water supply, agricultural irrigation, small booster systems.
NO ML INFERENCE. NO MONITORING. Advisory guidance only.
UI label: "Advisory guidance only — not a monitoring tool"

INPUT:
  usage_type       : str   ("domestic" / "agricultural" / "booster")
  daily_demand_lph : float (litres per hour)
  static_head_m    : float
  pipe_length_m    : float
  pipe_diameter_mm : float

OUTPUT:
  recommended_flow_lph  : float
  recommended_head_m    : float
  recommended_motor_kw  : float  (standard sizes: 0.5, 0.75, 1.0, 1.5 kW)
  pipe_velocity_ms      : float  (warn if > 2.0 m/s)
  friction_head_m       : float
  estimated_runtime_h   : float  (hours/day to meet demand)
  recommendations       : list[str]
  advisory_disclaimer   : str    (always appended — see below)

PHYSICS:
  Flow velocity : v = Q / A  (A = πD²/4)
  Friction head : Darcy-Weisbach (simplified for small pipes)
  Motor sizing  : P = ρgQH / (η×1000) → round up to next standard size

ADVISORY DISCLAIMER (always appended):
  "This is advisory guidance based on simplified hydraulic calculations.
   Actual pump selection should be verified by a qualified engineer.
   This tool does not monitor pump health and cannot detect faults."
```

### M9 Outputs

```
src/module_09_pump_selector.py
outputs/reports/module_09_pump_selector_report.md
```

### M9 Paste Text Keys

```
M9_industrial_test_cases_pass : [X/5]
M9_household_advisory_tested  : True/False
M9_scope_boundary_enforced    : True/False
M9_all_gates_pass             : True/False
Status_for_M10                : READY/BLOCKED
```

---

## M10 — Flask Web Application

**Status:** 🔲 NOT STARTED (requires M8 + M7 + M9 complete)

---

### Model Loading (at startup — ALL map_location='cpu')

```python
import torch, xgboost as xgb, json, pickle
from sklearn.multioutput import MultiOutputClassifier

# M8 LSTM-AE v2
lstm_ae = LSTMAEv2(...)
lstm_ae.load_state_dict(
    torch.load('models/lstm_ae_v2_best.pth', map_location='cpu')
)
lstm_ae.eval()

# M7 XGBoost — MultiOutputClassifier (7 binary heads)
# NOTE: MultiOutputClassifier is sklearn wrapper — saved as pickle, NOT .json
with open('models/xgboost_fault_classifier_cpu.pkl', 'rb') as f:
    xgb_model = pickle.load(f)  # MultiOutputClassifier instance
# Each internal estimator: xgb.XGBClassifier with device='cpu'
# NEVER call device='cuda' in any estimator at inference

# Configs
with open('models/M8_threshold_config.json') as f:
    threshold_config = json.load(f)
with open('models/M8_fuzzy_config.json') as f:
    fuzzy_config = json.load(f)
with open('models/M3_normalization_config.json') as f:
    norm_config = json.load(f)
```

---

### API Routes (6 Routes)

#### Route 1: POST /api/anomaly_detect

```
Purpose  : Real-time anomaly detection on incoming sensor window
Input    : JSON or CSV upload — 50 rows × 8 sensor columns (raw values)
Process  :
  1. Normalize using M3_normalization_config.json (cluster-aware)
  2. Detect cluster from M2 KMeans model
  3. Run M8 8-step inference protocol
  4. Return 4-state alert + full output dict
Output   : {
    alert_state       : "NORMAL" / "WATCH" / "WARN" / "DANGER"
    anomaly_flag      : bool
    fuzzy_membership  : float
    rolling_mean_mae  : float
    mae_slope         : float
    channel_drift     : {per-channel bool flags}
    early_fault_type  : None / "overloading_early" / "seal_failure_early" /
                        "bearing_wear_early" / "sensor_failure"
    severity          : "LOW" / "MEDIUM" / "HIGH"
    uncertainty_std   : float
    confidence        : float
    attention_heatmap : array(50,)
    cluster           : str
  }

SCOPE CHECK:
  if pump_type == 'household': return physics_advisory_only()
  else: run M8 inference
```

#### Route 2: POST /api/classify_fault

```
Purpose  : Classify fault type from feature snapshot
           Returns progressive confidence output (M7 multi-label architecture)
Input    : JSON — 25 feature values (24 original + compound_interaction_flag)
           OR raw 200-step sequence → extract features on-the-fly via M8 LSTM-AE
Process  :
  1. If raw sequence: run M8 LSTM-AE → extract 25 features
  2. Run M7 MultiOutputClassifier (device='cpu')
  3. Get predict_proba → 7 probability arrays
  4. Apply Stage 1/2/3 progressive confidence logic (threshold=0.75)
  5. Run SHAP TreeExplainer on X_input → top-3 features per predicted fault
Output   :
  Stage 1 (primary_conf < 0.50):
  {
    stage             : 1
    message           : "Minor anomaly — multiple causes possible"
    top3_candidates   : [(fault, prob), (fault, prob), (fault, prob)]
    action            : "Monitor all channels closely"
  }
  Stage 2 (0.50 ≤ primary_conf < 0.75):
  {
    stage             : 2
    message           : "Probable fault: <name> (XX%)"
    secondary_faults  : {fault: prob} for all faults with prob > 0.30
    action            : "Schedule inspection within 48h"
  }
  Stage 3 (primary_conf ≥ 0.75):
  {
    stage             : 3
    message           : "CONFIRMED: <name> (XX%)"
    secondary_faults  : {fault: prob}  ← compound faults shown here
    fault_stage       : "early" / "developing" / "advanced"
    shap_top3         : [{feature, value, direction}, ...]
    physical_meaning  : str  (plain language per fault)
    action            : urgency string based on fault_stage
  }

SCOPE CHECK:
  if pump_type == 'household': return physics_advisory_only()
```

#### Route 3: POST /api/select_pump

```
Purpose  : Industrial pump sizing and selection
Input    : JSON — flow_rate_m3h, total_head_m, fluid_density,
           fluid_temp_c, suction_head_m, pipe_length_m, pipe_diameter_m
Process  : M9 industrial pump selector (physics only)
Output   : Full M9 industrial output dict
No ML    : Pure physics — no model inference
```

#### Route 4: GET /api/household

```
Purpose  : Household pump advisory
Input    : Query params — usage_type, daily_demand_lph, static_head_m,
           pipe_length_m, pipe_diameter_mm
Process  : M9 household advisor (physics only)
Output   : M9 household output dict + advisory_disclaimer always appended
No ML    : physics_advisory_only() — scope boundary enforced
UI label : "Advisory guidance only — not a monitoring tool"
```

#### Route 5: POST /api/validate_model

```
Purpose  : M12 adversarial validation entry point
Input    : JSON — config_id (1–16), sequence data OR auto-generate flag
Process  : Trigger M12 validation suite for specified config
Output   : {
    config_id     : int
    alert_state   : str
    detection_lag : int  (timesteps)
    gate_pass     : bool
    details       : str
  }
Access   : Internal only — not exposed in production UI
```

#### Route 6: GET /health

```
Purpose  : Docker / Hugging Face health check
Output   : {
    status        : "healthy"
    models_loaded : ["lstm_ae_v2", "xgboost_multioutput", "m9_physics"]
    device        : "cpu"
    version       : "2.0"
  }
Used by  : Docker HEALTHCHECK + Hugging Face Spaces liveness probe
```

---

### 4-State Alert UI Rendering

```
🟢 NORMAL  (rolling_score < 2.0, no drift flags)
  Display  : "System operating within normal parameters"
  Colour   : Green
  Action   : None

🟡 WATCH   (Mech A/B/C trigger — early drift)
  Display  : "Early anomaly trend detected — monitor closely"
  Colour   : Yellow
  Details shown:
    — Which channel is drifting (from channel_drift dict)
    — Trend duration (how many windows in WATCH)
    — Slope value (Mech B reading)
    — Stage 2 classify_fault output: probable fault + secondary candidates
  Fault-specific channel messages:
    Temp.SV drift  → "Thermal overload trend — check motor loading" [Finding 1]
    Pres.SV drift  → "Pressure loss trend — possible seal degradation" [Finding 2]
    Mot.SV drift   → "Vibration rising on motor side — possible bearing wear"
    Flatline        → "Sensor signal lost — verify sensor hardware"

🟠 WARN    (rolling mean > 0.095 OR rolling_score 2.0–3.5)
  Display  : "Sustained anomaly — schedule maintenance inspection"
  Colour   : Orange
  Details shown:
    — Estimated time to DANGER at current trend rate
    — Stage 2 output from classify_fault: probable primary fault
    — Secondary fault candidates (compound fault visible here)
    — SHAP top-3 feature explanation (if Stage 2 or 3)

🔴 DANGER  (single window MAE > threshold OR rolling_score > 3.5)
  Display  : "Fault confirmed — immediate maintenance action required"
  Colour   : Red
  Details shown:
    — Stage 3 output from classify_fault (always at DANGER)
    — PRIMARY fault: confirmed name + confidence %
    — SECONDARY faults: compound pair shown if prob > 0.30
    — fault_stage: early / developing / advanced
    — SHAP top-3 feature explanation
    — Physical meaning in plain language
    — Recommended action based on fault_stage
    — MC Dropout uncertainty_std (confidence proxy)

CAVITATION DANGER specific display:
  — "CAVITATION DETECTED — STOP PUMP IMMEDIATELY"
  — "Impeller damage risk within 60–180 seconds of continued operation"
  — "Check inlet valve, suction line, and NPSH conditions before restart"
  — Stage 3 output shown (cavitation always returns Stage 3 — MAE 6.1× threshold)

COMPOUND FAULT display example (bearing_wear + seal_failure):
  PRIMARY  : "bearing_wear (87%)"
  SECONDARY: "seal_failure (61%) — likely cascade from bearing degradation"
  ACTION   : "Shutdown recommended — both mechanical seal and bearing require inspection"
```

---

### Mandatory Disclaimers (before ANY industrial inference)

```
DISCLAIMER 1 — Model Scope:
"This model is trained on CIRA SACIP dataset (1 specific installation).
 Sensor placement must follow ISO 13373 guidelines.
 r=0.9793 coupling between Mot.TV and Temp.SV is installation-specific.
 Model outputs are advisory — consult a qualified engineer before action."

DISCLAIMER 2 — Sensor Dependency:
"Inference quality depends entirely on sensor hardware integrity.
 Sensor malfunction or miscalibration will affect model output.
 Verify sensor health independently before acting on any alert."

DISCLAIMER 3 — Safety System Boundary:
"PumpSmart is a condition monitoring tool (ISO 13374 Level 3).
 It is NOT a Safety Instrumented System (SIS) per IEC 61511.
 Hardwired process trips remain the primary safety barrier.
 PumpSmart alerts are advisory and do not replace hardwired protection."
```

---

### M10 Local Testing Protocol (12 Tests)

```
Before pushing to M11, verify locally:

  [CORE ROUTES]
  1.  flask run → server starts without error
  2.  GET /health → all 3 models listed: [lstm_ae_v2, xgboost_multioutput, m9_physics]

  [ANOMALY DETECTION]
  3.  POST /api/anomaly_detect → normal window → alert_state = NORMAL
  4.  POST /api/anomaly_detect → known fault window → alert_state = DANGER
  5.  POST /api/anomaly_detect → mild fault window → alert_state = WATCH or WARN

  [FAULT CLASSIFICATION — MULTI-LABEL]
  6.  POST /api/classify_fault → cavitation features → Stage 3, fault_class = cavitation
  7.  POST /api/classify_fault → overloading mild features → Stage 1 or 2 output
  8.  POST /api/classify_fault → compound bearing+seal features
         → Stage 3, secondary_faults contains seal_failure with prob > 0.30

  [PHYSICS TOOLS]
  9.  POST /api/select_pump → nameplate inputs → motor = 110 kW
  10. GET /api/household → advisory_disclaimer present in output

  [SCOPE BOUNDARY]
  11. Household route → physics_advisory_only() fires, no ML model called

  [UI]
  12. All 3 disclaimers visible in dashboard before industrial inference
      WATCH state renders correctly with channel drift detail
      DANGER state renders compound fault secondary section if applicable
```

### M10 Outputs

```
app/
  app.py
  routes/
    anomaly.py      ← /api/anomaly_detect
    classify.py     ← /api/classify_fault  (multi-label, Stage 1/2/3)
    selector.py     ← /api/select_pump + /api/household
    validate.py     ← /api/validate_model
    health.py       ← /health
  templates/
    index.html      ← main dashboard (4-state UI)
    household.html  ← household advisor UI
    selector.html   ← industrial selector UI
  static/
    style.css
    dashboard.js
outputs/reports/module_10_flask_app_report.md
```

### M10 Paste Text Keys

```
M10_routes_registered           : [list of 6 routes]
M10_health_check_response       : healthy/error
M10_models_loaded_at_startup    : [lstm_ae_v2, xgboost_multioutput, m9_physics]
M10_normal_window_test          : NORMAL/error
M10_fault_window_test           : DANGER/error
M10_mild_fault_test             : WATCH/WARN/error
M10_compound_fault_test         : secondary_fault_visible/error
M10_household_scope_enforced    : True/False
M10_disclaimers_displayed       : True/False
M10_local_tests_pass            : [X/12]
Status_for_M11                  : READY/BLOCKED
```

---

## M11 — Docker + Hugging Face Deployment

**Status:** 🔲 NOT STARTED (requires M10 local tests pass)

---

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/
COPY src/module_09_pump_selector.py ./src/
COPY config.py .

# Hugging Face Spaces uses port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:7860", \
     "--workers", "1", "--timeout", "120"]
```

### requirements.txt (Deployment — CPU Only)

```
torch==2.6.0+cpu          # CPU-only — no CUDA in container
xgboost>=2.0
flask>=3.0
gunicorn>=21.0
scikit-learn>=1.3         # required for MultiOutputClassifier (M7 arch)
numpy>=1.24
pandas>=2.0
shap>=0.44
scipy>=1.11
```

### Model Loading Rules (Deployment — NON-NEGOTIABLE)

```python
# LSTM-AE v2: map_location='cpu' always
lstm_ae.load_state_dict(
    torch.load('models/lstm_ae_v2_best.pth', map_location='cpu')
)

# XGBoost: MultiOutputClassifier saved as pickle — load with pickle
import pickle
with open('models/xgboost_fault_classifier_cpu.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
# Each internal XGBClassifier must have device='cpu'
# Verify at load: assert all e.device == 'cpu' for e in xgb_model.estimators_

# MC Dropout inference: N=20 on CPU — acceptable latency (~200ms per window)
# NEVER call .cuda() or .to('cuda') anywhere in deployment code
```

### GitHub Actions CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Hugging Face Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Push to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config --global user.email "ci@pumpsmart"
          git config --global user.name "PumpSmart CI"
          git remote add hf https://souvik:$HF_TOKEN@huggingface.co/spaces/Souvik-1234-cpu/PumpSmart
          git push hf main --force
```

### Hugging Face Spaces Configuration

```yaml
# README.md front matter
---
title: PumpSmart Industrial Pump Health Monitor
emoji: 🔧
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
license: mit
---
```

### Deployment Validation Checklist (10 Checks)

```
1.  Docker build locally → no errors
2.  docker run -p 7860:7860 pumpsmart → container starts
3.  GET http://localhost:7860/health → {"status": "healthy"}
4.  POST /api/anomaly_detect → valid response (not 500)
5.  POST /api/classify_fault → valid Stage 3 fault class returned
6.  POST /api/classify_fault compound window → secondary_faults populated
7.  Image size < 2GB (Hugging Face free tier limit)
8.  Startup time < 60s (within HEALTHCHECK start-period)
9.  Push to Hugging Face Spaces → Space builds successfully
10. HF Space URL /health → {"status": "healthy"}
    GitHub Actions workflow runs without failure on push to main
```

### M11 Outputs

```
Dockerfile
requirements.txt                    (deployment version — CPU only)
.github/workflows/deploy.yml        (GitHub Actions CI/CD)
README.md                           (Hugging Face Spaces front matter)
outputs/reports/module_11_deployment_report.md
```

### M11 Paste Text Keys

```
M11_docker_build_status      : SUCCESS/FAILED
M11_container_startup_time_s : [seconds — gate < 60]
M11_image_size_mb            : [MB — gate < 2000]
M11_health_check_local       : healthy/error
M11_hf_deployment_url        : [URL]
M11_hf_health_check          : healthy/error
M11_github_actions_status    : PASS/FAIL
M11_compound_fault_route_test: PASS/FAIL  ← NEW
M11_all_checks_pass          : True/False
Status_for_M12               : READY/BLOCKED
```

---

## Module Dependency Summary

```
M7 XGBoost (MultiOutputClassifier) ────────────────────────┬─────────────┐
M8 LSTM-AE v2 (4-state alert) ───────────────────────┬─────────────┐
M9 Physics Tools ─────────────────────────────────────────┤          │
                                                         ↓          │
                                                      M10 Flask     │
                                                         ↓        models/
                                                      M11 Docker+HF
                                                         ↓
                                                      M12 Adversarial
                                                      Validation

SEQUENCING LAW:
  M7 gates pass        → M8 starts
  M8 gates pass        → M9 finalised + M10 starts
  M10 12/12 tests pass → M11 starts
  M11 deployment OK    → M12 starts
  M12 PRODUCTION_VALIDATED → system live
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Initial creation — split from module_pathway_M1_to_M12_v10.md |
| v2.0 | 2026-04-12 | Bias-audit cascade: multi-label classify route, Stage 1/2/3 API schema, compound fault UI display, MultiOutputClassifier pickle loading, scikit-learn deployment dependency, 12-test local protocol, M11 compound route check |

---

*Next file: `module_M12_validation_suite.md`*  
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*  
*Standard: ISO 10816-3 vibration | ISO 13373-3 monitoring | ISO 13374 Level 3 | IEC 61511 boundary*
