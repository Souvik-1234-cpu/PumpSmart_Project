# PumpSmart — M9 + M10 + M11: Deployment Modules
## Pump Selector | Flask Web Application | Docker + Hugging Face Deployment

**Document version:** v3.0 — Architecture v14.0 (CUSUM + Rolling Baseline + 22-class)
**Date:** 2026-04-16
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

## What Changed in v3.0 (Architecture v14.0 — CUSUM + 22-class)

| Item | v2.0 | v3.0 | Source |
|------|------|------|--------|
| classify_fault input | 25 features (24 original + compound_interaction_flag) | **25 features (M6.5r 26-col matrix, label excluded at inference)** | M6.5r spec |
| Fault classes | 7 binary heads (M6.5 matrix) | **22-class single-label XGBoost (M6B labels 0–21)** | v14.0 fault universe |
| anomaly_detect output | No CUSUM fields | **+ cusum_state, rolling_baseline_alert, bearing_wear_gradual_advisory** | v14.0 Layer 3+4 |
| Model loading (M10 startup) | LSTM-AE + XGBoost + configs | **+ CUSUM state init (3 channels) + rolling baseline init (M3 baselines)** | v14.0 Layer 3+4 |
| WATCH UI rendering | Mech A/B/C trigger only | **+ CUSUM trigger source + rolling baseline drift message** | v14.0 Layer 3+4 |
| bearing_wear_gradual alert | Not present | **Advisory: "Plan bearing inspection within 7–14 days"** | label 21 v14.0 |
| Local test protocol | 12 tests | **+1 label 21 slow drift test = 13 tests** | label 21 v14.0 |
| /health route version | "2.0" | **"3.0"** | this update |
| API routes total | 6 | **7 (+ /api/acknowledge)** | CUSUM reset requirement |
| M10 paste keys | 10 keys | **+3 CUSUM keys = 13 keys** | v14.0 |
| M11 local checks | 10 checks | **+1 CUSUM state persistence check = 11 checks** | v14.0 |
| Changelog | v1.0, v2.0 | **+ v3.0 row** | this update |

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

### Model + Runtime State Loading (at startup — ALL map_location='cpu')

```python
import torch, json, pickle
from sklearn.multioutput import MultiOutputClassifier

# ── M8 LSTM-AE v2 ─────────────────────────────────────────────────
lstm_ae = LSTMAEv2(...)
lstm_ae.load_state_dict(
    torch.load('models/lstm_ae_v2_best.pth', map_location='cpu')
)
lstm_ae.eval()

# ── M7 XGBoost — 22-class single-label ────────────────────────────
# Trains on M6B_feature_matrix.csv (M6.5r output) — 25 features at inference
# (26-col training matrix has label in col 25 — excluded at inference)
# NOTE: saved as pickle (sklearn wrapper) NOT .json
with open('models/xgboost_fault_classifier_cpu.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
# Each internal XGBClassifier must have device='cpu'
# Verify: assert all(e.device == 'cpu' for e in xgb_model.estimators_)
# NEVER call device='cuda' in any estimator at inference

# ── Configs ───────────────────────────────────────────────────────
with open('models/M8_threshold_config.json') as f:
    threshold_config = json.load(f)
with open('models/M8_fuzzy_config.json') as f:
    fuzzy_config = json.load(f)
with open('models/M3_normalization_config.json') as f:
    norm_config = json.load(f)   # READ-ONLY — used for CUSUM μ0 + Layer 4 baselines

# ── Layer 3: CUSUM Runtime State Initialisation ───────────────────
# Formula: S_n = max(0, S_{n-1} + (mae_channel_n − μ0) − k)
# k = 0.5 × (threshold − μ0) per channel (from threshold_config)
# Reference μ0: per-channel normal MAE from M3_normalization_config.json
# PERSISTENT across API calls. Resets ONLY on operator ACK or pump restart.
cusum_state = {
    "X_ACR_Mot.SV": 0.0,   # S_n initialised to 0
    "X_Pres.SV":    0.0,
    "X_Temp.SV":    0.0,
}
cusum_control_limit = 5.0   # default — may be tuned from M8_fuzzy_config.json
cusum_mu0 = {               # loaded from norm_config at startup
    "X_ACR_Mot.SV": norm_config["normal_mae"]["X_ACR_Mot.SV"],
    "X_Pres.SV":    norm_config["normal_mae"]["X_Pres.SV"],
    "X_Temp.SV":    norm_config["normal_mae"]["X_Temp.SV"],
}
cusum_k = {ch: 0.5 * (threshold_config["base"] - cusum_mu0[ch])
           for ch in cusum_state}

# ── Layer 4: Rolling Baseline Comparator Initialisation ──────────
# Monitors 30-window rolling mean of err_slope per channel
# Fires when rolling mean > μ_normal + 2σ_normal (from M3 baselines)
# PERSISTENT across API calls. Resets on operator ACK or pump restart.
rolling_slope_buffer = {
    "X_ACR_Mot.SV": [],   # appended each inference call, max len=30
    "X_Pres.SV":    [],
    "X_Temp.SV":    [],
}
rolling_baseline_ref = {    # loaded from norm_config at startup
    ch: {
        "mu":    norm_config["normal_slope"][ch]["mean"],
        "sigma": norm_config["normal_slope"][ch]["std"],
    }
    for ch in rolling_slope_buffer
}
```

---

### API Routes (7 Routes)

#### Route 1: POST /api/anomaly_detect

```
Purpose  : Real-time anomaly detection on incoming sensor window
Input    : JSON or CSV upload — 50 rows × 8 sensor columns (raw values)
Process  :
  1. Normalize using M3_normalization_config.json (cluster-aware)
  2. Detect cluster from M2 KMeans model
  3. Run M8 4-layer inference protocol:
       Layer 1: LSTM-AE single-window MAE vs M4_THRESHOLD (0.110058)
       Layer 2: Fuzzy logic + rolling accumulator (last 20–40 windows)
       Layer 3: Update CUSUM state for MotSV, PresSV, TempSV channels
       Layer 4: Update rolling slope buffer; check vs M3 baseline + 2σ
  4. Return 4-state alert + full output dict
Output   : {
    alert_state                  : "NORMAL" / "WATCH" / "WARN" / "DANGER"
    anomaly_flag                 : bool
    fuzzy_membership             : float
    rolling_mean_mae             : float
    mae_slope                    : float
    channel_drift                : {per-channel bool flags}
    early_fault_type             : None / "overloading_early" /
                                   "seal_failure_early" /
                                   "bearing_wear_early" /
                                   "bearing_wear_gradual" /
                                   "sensor_failure"
    severity                     : "LOW" / "MEDIUM" / "HIGH"
    uncertainty_std              : float   (MC Dropout N=20)
    confidence                   : float
    attention_heatmap            : array(50,)
    cluster                      : str
    cusum_state                  : {                        ← NEW v3.0 (Layer 3)
        "X_ACR_Mot.SV": float,   (current S_n)
        "X_Pres.SV":    float,
        "X_Temp.SV":    float,
        "fired":        bool,    (True if any S_n > control_limit=5.0)
        "fired_channels": list[str]
    }
    rolling_baseline_alert       : {                        ← NEW v3.0 (Layer 4)
        "X_ACR_Mot.SV": bool,   (True if rolling slope > μ+2σ)
        "X_Pres.SV":    bool,
        "X_Temp.SV":    bool,
        "any_alert":    bool
    }
    bearing_wear_gradual_advisory: str | None               ← NEW v3.0
        None if no signal.
        "Plan bearing inspection within 7–14 days." if
        CUSUM fired on MotSV AND rolling_baseline_alert MotSV = True
        AND early_fault_type = "bearing_wear_gradual"
  }

CUSUM RESET RULE:
  cusum_state resets to 0.0 per channel on:
    (a) Explicit POST /api/acknowledge — operator confirms action taken
    (b) Pump restart signal detected (PresSV drops to near 0 → startup cluster)
  Do NOT auto-reset on NORMAL windows — gradual drift must accumulate.

SCOPE CHECK:
  if pump_type == 'household': return physics_advisory_only()
  else: run M8 inference
```

#### Route 2: POST /api/classify_fault

```
Purpose  : Classify fault type from feature snapshot (22-class, labels 0–21)
           Returns progressive confidence output
Input    : JSON — 25 feature values (M6.5r feature set, label excluded)
           Feature order: same as M6B_feature_matrix.csv columns 0–24
           OR raw 200-step sequence → extract features on-the-fly via M8 LSTM-AE
           NOTE: M6.5r training matrix = 26 cols (25 features + label col 25).
                 At inference: send 25 features only. Label col NEVER sent by client.
Process  :
  1. If raw sequence: run M8 LSTM-AE → extract 25 features
  2. Run M7 22-class XGBoost (device='cpu')
  3. Get predict_proba → probability array over 22 classes (labels 0–21)
  4. Apply Stage 1/2/3 progressive confidence logic (threshold=0.75)
  5. Map label integer → fault name via fault_rules_v3.json
  6. Map compound label (7–12) → causal chain string for UI display
  7. Run SHAP TreeExplainer on X_input → top-3 features per predicted fault
Output   :
  Stage 1 (primary_conf < 0.50):
  {
    stage             : 1
    message           : "Minor anomaly — multiple causes possible"
    top3_candidates   : [(fault_name, prob), ...]
    action            : "Monitor all channels closely"
  }
  Stage 2 (0.50 ≤ primary_conf < 0.75):
  {
    stage             : 2
    message           : "Probable fault: <name> (XX%)"
    secondary_faults  : {fault_name: prob} for all labels with prob > 0.30
    action            : "Schedule inspection within 48h"
  }
  Stage 3 (primary_conf ≥ 0.75):
  {
    stage             : 3
    message           : "CONFIRMED: <name> (XX%)"
    secondary_faults  : {fault_name: prob}
    fault_stage       : "early" / "developing" / "advanced"
    shap_top3         : [{feature, value, direction}, ...]
    physical_meaning  : str  (plain language per fault)
    action            : urgency string based on fault_stage
    causal_chain      : str | None  ← populated for Group B labels 7–12
                        e.g. "bearing_wear → overloading"
  }

COMPOUND FAULT LABEL MAP (Group B — single integer → causal chain display):
  7  → "Primary: bearing_wear        → Secondary: overloading"
  8  → "Primary: cavitation          → Secondary: seal_failure"
  9  → "Primary: impeller_imbalance  → Secondary: bearing_wear"
  10 → "Primary: seal_failure        → Secondary: cavitation"
  11 → "Primary: overloading         → Secondary: bearing_wear"
  12 → "Primary: impeller_imbalance  → Secondary: cavitation"
  Source: fault_rules_v3.json (written by M6B Step 3 — authoritative label map)

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

#### Route 5: POST /api/acknowledge

```
Purpose  : Operator acknowledgement — resets CUSUM + rolling baseline state
           for specified channels after maintenance action taken
Input    : JSON — {
    channels     : list[str]  ("all" OR specific channel names)
    pump_id      : str        (for logging)
    action_taken : str        (free text — maintenance notes)
  }
Process  :
  Reset cusum_state[ch] = 0.0 for each acknowledged channel
  Reset rolling_slope_buffer[ch] = [] for each acknowledged channel
  Log acknowledgement with timestamp
Output   : {
    acknowledged_channels : list[str]
    cusum_reset           : True
    rolling_reset         : True
    timestamp             : str
  }
Access   : Production UI — operator must explicitly trigger
           Do NOT auto-call from anomaly_detect route
```

#### Route 6: POST /api/validate_model

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

#### Route 7: GET /health

```
Purpose  : Docker / Hugging Face health check
Output   : {
    status          : "healthy"
    models_loaded   : ["lstm_ae_v2", "xgboost_22class", "m9_physics"]
    device          : "cpu"
    version         : "3.0"
    cusum_active    : true        ← confirms Layer 3 initialised
    n_fault_classes : 22
  }
Used by  : Docker HEALTHCHECK + Hugging Face Spaces liveness probe
```

---

### 4-State Alert UI Rendering

```
🟢 NORMAL  (rolling_score < 2.0, no drift flags, all cusum S_n < 1.0)
  Display  : "System operating within normal parameters"
  Colour   : Green
  Action   : None

🟡 WATCH   (Mech A/B/C trigger OR CUSUM S_n rising OR rolling baseline drift)
  Display  : "Early anomaly trend detected — monitor closely"
  Colour   : Yellow
  Details shown:
    — Which channel is drifting (from channel_drift dict)
    — Trend duration (how many windows in WATCH)
    — Slope value (Mech B / Layer 4 reading)
    — Stage 2 classify_fault output: probable fault + secondary candidates
    — CUSUM trigger: if cusum_state.fired = True → show current S_n per channel
    — Rolling baseline: if rolling_baseline_alert.any_alert = True →
        show rolling slope vs μ+2σ reference
  Fault-specific channel messages:
    Temp.SV drift  → "Thermal overload trend — check motor loading" [Finding 1]
    Pres.SV drift  → "Pressure loss trend — possible seal degradation" [Finding 2]
    Mot.SV drift   → "Vibration rising on motor side — possible bearing wear"
    Mot.SV CUSUM   → "Gradual bearing wear accumulating — CUSUM S_n = X.X" ← NEW
    Flatline        → "Sensor signal lost — verify sensor hardware"

  BEARING WEAR GRADUAL (label 21) — WATCH display:         ← NEW v3.0
    Trigger: CUSUM S_n(MotSV) > 3.0 AND rolling_baseline_alert(MotSV) = True
    Display: "⚠ Gradual bearing wear trend detected"
             "CUSUM accumulator S_n = X.X (control limit = 5.0)"
             "Rolling slope drift: X.XXX (ref: μ+2σ = X.XXX)"
    Advisory: "Plan bearing inspection within 7–14 days."
    Note: This alert fires BEFORE Layer 1 threshold is crossed.
          LSTM-AE single window MAE will be below threshold — this is expected.
          CUSUM + rolling baseline are the PRIMARY detection path for label 21.

🟠 WARN    (rolling mean > 0.095 OR rolling_score 2.0–3.5)
  Display  : "Sustained anomaly — schedule maintenance inspection"
  Colour   : Orange
  Details shown:
    — Estimated time to DANGER at current trend rate
    — Stage 2 output from classify_fault: probable primary fault
    — Secondary fault candidates (compound fault visible here)
    — SHAP top-3 feature explanation (if Stage 2 or 3)
    — CUSUM state shown if any channel S_n > 2.0

🔴 DANGER  (single window MAE > threshold OR rolling_score > 3.5
            OR CUSUM S_n > control_limit on any channel)
  Display  : "Fault confirmed — immediate maintenance action required"
  Colour   : Red
  Details shown:
    — Stage 3 output from classify_fault (always at DANGER)
    — PRIMARY fault: confirmed name + confidence %
    — SECONDARY faults: compound pair shown if prob > 0.30
    — fault_stage: early / developing / advanced
    — causal_chain string for Group B labels (7–12)
    — SHAP top-3 feature explanation
    — Physical meaning in plain language
    — Recommended action based on fault_stage
    — MC Dropout uncertainty_std (confidence proxy)
    — CUSUM state if Layer 3 triggered DANGER independently

CAVITATION DANGER specific display:
  — "CAVITATION DETECTED — STOP PUMP IMMEDIATELY"
  — "Impeller damage risk within 60–180 seconds of continued operation"
  — "Check inlet valve, suction line, and NPSH conditions before restart"
  — Stage 3 output shown (cavitation always returns Stage 3 — MAE 6.1× threshold)

COMPOUND FAULT display example (label 7: bearing_wear → overloading):
  PRIMARY  : "bearing_wear (87%)"
  SECONDARY: "overloading (61%) — thermal cascade from bearing degradation"
  CHAIN    : "bearing_wear → overloading"
  ACTION   : "Shutdown recommended — bearing + motor thermal inspection required"
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

### M10 Local Testing Protocol (13 Tests)

```
Before pushing to M11, verify locally:

  [CORE ROUTES]
  1.  flask run → server starts without error
  2.  GET /health → models_loaded = [lstm_ae_v2, xgboost_22class, m9_physics]
                    cusum_active = true, n_fault_classes = 22, version = "3.0"

  [ANOMALY DETECTION — LAYERS 1+2]
  3.  POST /api/anomaly_detect → normal window → alert_state = NORMAL
  4.  POST /api/anomaly_detect → known hard fault window → alert_state = DANGER
  5.  POST /api/anomaly_detect → mild fault window → alert_state = WATCH or WARN

  [ANOMALY DETECTION — LAYERS 3+4: CUSUM + ROLLING BASELINE]
  6.  Simulate 40 consecutive label-21 windows (bearing_wear_gradual severity=0.15):
      → cusum_state["X_ACR_Mot.SV"] S_n must rise above 0 across calls
      → rolling_baseline_alert["X_ACR_Mot.SV"] = True after ~30 windows
      → bearing_wear_gradual_advisory = "Plan bearing inspection within 7–14 days."
      → alert_state = WATCH (NOT DANGER — Layer 1 MAE below threshold — expected)
  7.  POST /api/acknowledge → channels=["X_ACR_Mot.SV"]
      → cusum_state["X_ACR_Mot.SV"] resets to 0.0
      → rolling_slope_buffer["X_ACR_Mot.SV"] resets to []

  [FAULT CLASSIFICATION — 22-CLASS]
  8.  POST /api/classify_fault → cavitation features → Stage 3, label=3
  9.  POST /api/classify_fault → overloading mild features → Stage 1 or 2
  10. POST /api/classify_fault → compound bearing+overloading features
        → Stage 3, label=7, causal_chain="bearing_wear → overloading"
        → secondary_faults populated

  [PHYSICS TOOLS]
  11. POST /api/select_pump → nameplate inputs → motor = 110 kW
  12. GET /api/household → advisory_disclaimer present in output

  [SCOPE BOUNDARY]
  13. Household route → physics_advisory_only() fires, no ML model called

  [UI — verify manually]
  All 3 disclaimers visible before industrial inference
  WATCH state renders CUSUM S_n values when cusum_state.fired = True
  WATCH state renders bearing_wear_gradual_advisory for label 21
  DANGER state renders causal_chain for Group B labels (7–12)
```

### M10 Outputs

```
app/
  app.py
  routes/
    anomaly.py      ← /api/anomaly_detect  (4-layer inference, CUSUM state)
    classify.py     ← /api/classify_fault  (22-class, Stage 1/2/3, causal chain)
    selector.py     ← /api/select_pump + /api/household
    acknowledge.py  ← /api/acknowledge     (CUSUM + rolling baseline reset)
    validate.py     ← /api/validate_model
    health.py       ← /health
  runtime/
    cusum_state.py   ← CUSUM S_n persistent state class
    rolling_state.py ← Rolling baseline buffer class
  templates/
    index.html      ← main dashboard (4-state UI, CUSUM panel)
    household.html  ← household advisor UI
    selector.html   ← industrial selector UI
  static/
    style.css
    dashboard.js
outputs/reports/module_10_flask_app_report.md
```

### M10 Paste Text Keys

```
M10_routes_registered           : [list of 7 routes]
M10_health_check_response       : healthy/error
M10_models_loaded_at_startup    : [lstm_ae_v2, xgboost_22class, m9_physics]
M10_normal_window_test          : NORMAL/error
M10_fault_window_test           : DANGER/error
M10_mild_fault_test             : WATCH/WARN/error
M10_compound_fault_test         : causal_chain_visible/error
M10_label21_cusum_test          : WATCH+advisory_visible/error   ← NEW
M10_cusum_reset_test            : reset_confirmed/error           ← NEW
M10_cusum_active_at_startup     : True/False                      ← NEW
M10_household_scope_enforced    : True/False
M10_disclaimers_displayed       : True/False
M10_local_tests_pass            : [X/13]
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

# XGBoost 22-class: saved as pickle — load with pickle
import pickle
with open('models/xgboost_fault_classifier_cpu.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
# Each internal XGBClassifier must have device='cpu'
# Verify at load: assert all(e.device == 'cpu' for e in xgb_model.estimators_)

# CUSUM + Rolling Baseline: initialised from norm_config at startup (see M10 loading)
# State is in-memory Python dicts — NOT stored in model weights
# Persists across API calls within one container lifecycle
# Resets on container restart OR explicit /api/acknowledge call

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

### Deployment Validation Checklist (11 Checks)

```
1.  Docker build locally → no errors
2.  docker run -p 7860:7860 pumpsmart → container starts
3.  GET http://localhost:7860/health → {"status":"healthy", "cusum_active":true,
                                        "n_fault_classes":22, "version":"3.0"}
4.  POST /api/anomaly_detect → valid response (not 500), cusum_state in output
5.  POST /api/classify_fault → valid Stage 3 fault class returned (label 0–21)
6.  POST /api/classify_fault compound window → causal_chain populated for Group B label
7.  POST /api/acknowledge → cusum_state resets to 0.0 per channel confirmed   ← NEW
8.  Image size < 2GB (Hugging Face free tier limit)
9.  Startup time < 60s (within HEALTHCHECK start-period)
10. Push to Hugging Face Spaces → Space builds successfully
11. HF Space URL /health → {"status": "healthy"}
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
M11_docker_build_status         : SUCCESS/FAILED
M11_container_startup_time_s    : [seconds — gate < 60]
M11_image_size_mb               : [MB — gate < 2000]
M11_health_check_local          : healthy/error
M11_cusum_active_in_container   : True/False                ← NEW
M11_hf_deployment_url           : [URL]
M11_hf_health_check             : healthy/error
M11_github_actions_status       : PASS/FAIL
M11_compound_fault_route_test   : PASS/FAIL
M11_all_checks_pass             : True/False
Status_for_M12                  : READY/BLOCKED
```

---

## Module Dependency Summary

```
M7 XGBoost (22-class, M6B_feature_matrix.csv) ─────────────────┬──────────────┐
M8 LSTM-AE v2 (4-layer: LSTM+Fuzzy+CUSUM+Rolling) ───────┬─────────────┐      │
M9 Physics Tools ──────────────────────────────────────────────┤            │    │
                                                              ↓            │    │
                                                          M10 Flask        │    │
                                                     (CUSUM runtime state) │  models/
                                                              ↓            │
                                                          M11 Docker+HF ───┘
                                                              ↓
                                                          M12 Adversarial
                                                          Validation

SEQUENCING LAW:
  M7 gates pass        → M8 starts
  M8 gates pass        → M9 finalised + M10 starts
  M10 13/13 tests pass → M11 starts
  M11 deployment OK    → M12 starts
  M12 PRODUCTION_VALIDATED → system live
```

---

## Document Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-04-12 | Initial creation — split from module_pathway_M1_to_M12_v10.md |
| v2.0 | 2026-04-12 | Bias-audit cascade: multi-label classify route, Stage 1/2/3 API schema, compound fault UI display, MultiOutputClassifier pickle loading, scikit-learn deployment dependency, 12-test local protocol, M11 compound route check |
| v3.0 | 2026-04-16 | Architecture v14.0: CUSUM runtime state (Layer 3) + rolling baseline comparator (Layer 4) added to model loading, Route 1 output schema, WATCH UI, Tests 6+7; label 21 bearing_wear_gradual advisory; /api/acknowledge added as Route 5 (routes renumbered, total 7); 22-class XGBoost throughout; feature count clarified (25 features at inference from 26-col M6.5r matrix); /health version→3.0 + cusum_active + n_fault_classes; 13 local tests; 11 M11 checks; M10 paste keys +3 CUSUM keys (total 13); M11 paste key cusum_active_in_container added |

---

*Next file: `module_M12_validation_suite.md`*
*Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP dataset*
*Standard: ISO 10816-3 vibration | ISO 13373-3 monitoring | ISO 13374 Level 3 | IEC 61511 boundary*
*Architecture: v14.0 | Classes: 22 (labels 0–21) | Detection layers: 4*
