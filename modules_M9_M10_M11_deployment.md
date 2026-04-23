# PumpSmart --- M9 + M10 + M11: Deployment Modules

**Pump Selector \| Flask Web Application \| Docker + Hugging Face
Deployment**

  -----------------------------------------------------------------------
  Field                               Value
  ----------------------------------- -----------------------------------
  Document version                    v4.0 --- Architecture v14.2 (TCN-AE
                                      Level 2 + score_A/B/C +
                                      physics_context + adaptive
                                      threshold)

  Date                                2026-04-21

  Prerequisites                       M8 all_gates_pass = True \| M7
                                      all_gates_pass = True

  Status                              All NOT STARTED --- begin after M8
                                      gates confirmed

  Pump                                110 kW, 7-stage, 40 bar, 2980 RPM,
                                      45 m3/h, 450 m head --- CIRA SACIP
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## What Changed in v4.0 (Architecture v14.2)

  ------------------------------------------------------------------------------------------------
  Item                    v3.0 (arch v14.0)       v4.0 (arch v14.2)
  ----------------------- ----------------------- ------------------------------------------------
  Level 2 model           LSTM-AE v2              TCN-AE Level 2 `tcn_ae_level2_best.pth` ---
                          `lstm_ae_v2_best.pth`   5-layer dilated causal, dilation=\[1,2,4,8,16\]

  Level 2 input           Raw MAE channels        z_t sequences (N_windows x 64) --- NEVER raw
                                                  sensor data (Invariant 16)

  Level 2 output          Single MAE score        score_A (severity), score_B (drift slope),
                                                  score_C (chain transition)

  CUSUM input             Raw per-channel MAE     score_B ONLY (Invariant 19)

  Rolling baseline input  err_slope per channel   score_A ONLY (Invariant 19)

  XGBoost input at        25 features             \~35 features including score_A/B/C,
  runtime                                         onset_order, z_t features

  Adaptive threshold      Not implemented         theta_t = mu_rolling(6hr) +
                                                  3\*sigma_rolling(6hr), updates every 50s in M10

  z_t rolling buffer      Not present             N_windows z_t vectors maintained in memory per
                                                  streaming call

  physics_context field   Not present             Added to Route 1 + Route 2 output ---
                                                  what/why/timeline/action/if_ignored/disclaimer

  /api/physics_context    Not present             New Route 8 --- plain-language fault description
                                                  per label, all 22 classes

  Commissioning mode      Not documented here     48hr data collection then re-run M2/M3 then lock
                                                  config. M4 + TCN-AE weights FROZEN

  Limitation disclaimer   3 global disclaimers    \+ per-alert limitation flag in UI linking to 6
                                                  known limitations (File 3 registry)

  models_loaded at        lstm_ae_v2,             lstm_ae_l1, tcn_ae_l2, xgboost_22class,
  startup                 xgboost_22class,        m9_physics
                          m9_physics              

  /health version         "3.0"                   "4.0"

  API routes total        7                       8 (+ /api/physics_context)

  M10 local tests         13                      15

  M11 deployment checks   11                      12

  M10 paste keys          13                      18
  ------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## Prerequisite Chain

    M7 all_gates_pass = True
      -> M8 all_gates_pass = True
           -> M9 (physics tools) — can run in parallel with M8 after M7 completes
           -> M10 (Flask app) — requires M8 models + M7 models + M9 physics tools
           -> M11 (deployment) — requires M10 fully tested locally
           -> M12 adversarial validation — run AFTER M11 deployment confirmed

------------------------------------------------------------------------

## M9 --- Pump Selector + Household Advisor

**Status:** NOT STARTED (can begin after M7 completes) **Nature:**
PHYSICS ONLY --- no ML inference in this module

### Scope Boundary (NEVER VIOLATE)

``` python
if pump_type == 'household': return physics_advisory_only()
else: return ml_prediction()   # routes to M8 + M7 in M10
```

> Household monoblock pump ≠ industrial multistage pump. Cross-domain ML
> = out-of-distribution inference = safety risk. Household advisor label
> in M10 UI: **"Advisory guidance only --- not a monitoring tool"**

------------------------------------------------------------------------

### M9 Part A --- Industrial Pump Selector

#### Physics Equations Implemented

**1. Hydraulic Power**

    P_hyd = rho x g x Q x H
    rho = fluid density (kg/m3), g = 9.81 m/s2,
    Q = flow rate (m3/s), H = total head (m)
    Nameplate check: P_hyd = 1000 x 9.81 x (45/3600) x 450 = ~55.2 kW  PASS

**2. Shaft Power (motor requirement)**

    P_shaft = P_hyd / eta_pump
    eta_pump = pump hydraulic efficiency (default 0.65 for multistage)
    P_shaft = 55.2 / 0.65 = ~84.9 kW -> motor selection: 110 kW (IEC frame)

**3. Total Head (multi-component)**

    H_total = H_static + H_friction + H_velocity
    H_static   = elevation difference (m)
    H_friction = f x (L/D) x (v2/2g)   [Darcy-Weisbach]
    H_velocity = v2 / 2g

**4. NPSH Available (NPSHa)**

    NPSHa = (P_atm - P_vapour) / (rho x g) + H_suction - H_friction_suction
    P_vapour = vapour pressure at operating temperature

**5. NPSH Required (NPSHr)**

    NPSHr = pump-specific value from manufacturer curve
    CAVITATION RISK FLAG: if NPSHa < NPSHr + 0.5m (safety margin) -> flag

**6. NPSH Margin Check**

    margin = NPSHa - NPSHr
    if margin < 0.5m  -> CAVITATION_RISK = True  (startup vulnerable)
    if margin < 0.0m  -> CAVITATION_CERTAIN = True (do not operate)

**7. Affinity Laws (speed change)**

    Q2/Q1 = N2/N1
    H2/H1 = (N2/N1)^2
    P2/P1 = (N2/N1)^3
    Use: estimate performance at variable speed / partial load

**8. Specific Speed (pump type selector)**

    Ns = N x Q^0.5 / H^0.75   [SI units: RPM, m3/s, m]
    Ns < 25       : Radial flow -> multistage centrifugal
    25 < Ns < 70  : Mixed flow
    Ns > 70       : Axial flow
    Nameplate: Ns = 2980 x (45/3600)^0.5 / 450^0.75 = ~10.2 -> radial  PASS

**9. Joukowsky Water Hammer (transient pressure check)**

    Delta_P = rho x a x Delta_v
    a = pressure wave velocity (~1200 m/s in steel pipe)
    Delta_v = sudden velocity change (valve closure)
    Max transient: P_operating + Delta_P must not exceed pipe pressure rating
    Nameplate transient observed: 46.7 bar (confirmed M2 data)

**10. Stage Head (multistage)**

    H_per_stage = H_total / n_stages
    Nameplate: 450 / 7 = 64.3 m per stage
    Verify: per-stage head within impeller design range

------------------------------------------------------------------------

#### Industrial Selector Input

  Parameter         Type    Description
  ----------------- ------- ---------------------------------------------
  flow_rate_m3h     float   required m3/h
  total_head_m      float   required head in m
  fluid_density     float   kg/m3, default 1000
  fluid_temp_c      float   degrees C, for vapour pressure lookup
  suction_head_m    float   positive = flooded, negative = suction lift
  pipe_length_m     float   ---
  pipe_diameter_m   float   ---
  speed_rpm         float   default 2980 for 50Hz 2-pole

#### Industrial Selector Output

  Field                  Type          Description
  ---------------------- ------------- ---------------------------------------------
  hydraulic_power_kw     float         ---
  required_shaft_kw      float         ---
  recommended_motor_kw   float         next IEC standard frame above shaft
  npsha                  float         ---
  npshr_margin           float         ---
  cavitation_risk        bool          ---
  specific_speed         float         ---
  pump_type              str           multistage_centrifugal / mixed_flow / axial
  stage_head_m           float         ---
  water_hammer_dp_bar    float         ---
  warnings               list of str   ---
  recommendation         str           ---

------------------------------------------------------------------------

#### Validation Test Cases (M9 Gate)

**TEST-M9-1: Nameplate reproduction** - Input: Q=45 m3/h, H=450m,
rho=1000, eta=0.65, N=2980 - Expect: P_hyd \~55.2 kW, P_shaft \~84.9 kW,
motor=110 kW, Ns\~10.2 **TEST-M9-2: Cavitation risk flag** - Input:
NPSHa=3.2m, NPSHr=3.0m (margin=0.2m \< 0.5m) - Expect: cavitation_risk =
True, warning issued **TEST-M9-3: Affinity law speed reduction** -
Input: N1=2980, Q1=45, H1=450, N2=2500 - Expect: Q2 = 45 x (2500/2980) =
37.75 m3/h, H2 = 450 x (2500/2980)\^2 = 317.8 m, P2/P1 = (2500/2980)\^3
= 0.591 **TEST-M9-4: Water hammer transient** - Input: rho=1000, a=1200
m/s, Delta_v=2.5 m/s, P_operating=40 bar - Expect: Delta_P = 1000 x 1200
x 2.5 / 100000 = 30 bar --- P_transient = 70 bar -\> WARNING: exceeds 40
bar nameplate **TEST-M9-5: Specific speed pump type** - Input: N=2980,
Q=45 m3/h, H=450m - Expect: Ns \~10.2 -\> pump_type =
multistage_centrifugal **Gates:** - GATE-M9-1: All 5 test cases PASS -
GATE-M9-2: No unphysical outputs (negative pressure, T below ambient, Ns
\< 0) - GATE-M9-3: Household pump_type -\> physics_advisory_only()
returns, no ML call ---

### M9 Part B --- Household Advisor (Physics Only)

**Scope:** Domestic water supply, agricultural irrigation, small booster
systems. **NO ML INFERENCE. NO MONITORING. Advisory guidance only.**
**UI label:** "Advisory guidance only --- not a monitoring tool"

#### Input

  Parameter          Type    Description
  ------------------ ------- -----------------------------------
  usage_type         str     domestic / agricultural / booster
  daily_demand_lph   float   litres per hour
  static_head_m      float   ---
  pipe_length_m      float   ---
  pipe_diameter_mm   float   ---

#### Output

  -----------------------------------------------------------------------
  Field                   Type                    Description
  ----------------------- ----------------------- -----------------------
  recommended_flow_lph    float                   ---

  recommended_head_m      float                   ---

  recommended_motor_kw    float                   standard sizes: 0.5,
                                                  0.75, 1.0, 1.5 kW

  pipe_velocity_ms        float                   warn if \> 2.0 m/s

  friction_head_m         float                   ---

  estimated_runtime_h     float                   hours per day to meet
                                                  demand

  recommendations         list of str             ---

  advisory_disclaimer     str                     always appended --- see
                                                  below
  -----------------------------------------------------------------------

#### Physics

    Flow velocity : v = Q / A  (A = pi x D^2 / 4)
    Friction head : Darcy-Weisbach (simplified for small pipes)
    Motor sizing  : P = rho x g x Q x H / (eta x 1000) -> round up to next standard size

#### Advisory Disclaimer (always appended)

> "This is advisory guidance based on simplified hydraulic calculations.
> Actual pump selection should be verified by a qualified engineer. This
> tool does not monitor pump health and cannot detect faults."

------------------------------------------------------------------------

### M9 Outputs

-   `src/module_09_pump_selector.py`
-   `outputs/reports/module_09_pump_selector_report.md` \### M9 Paste
    Text Keys

  Key                             Value
  ------------------------------- ---------------
  M9_industrial_test_cases_pass   \[X/5\]
  M9_household_advisory_tested    True/False
  M9_scope_boundary_enforced      True/False
  M9_all_gates_pass               True/False
  Status_for_M10                  READY/BLOCKED

------------------------------------------------------------------------

## M10 --- Flask Web Application

**Status:** NOT STARTED (requires M8 + M7 + M9 complete)

------------------------------------------------------------------------

### Commissioning Mode (MANDATORY --- First 48 Hours on New Installation)

> **COMMISSIONING MODE RULE --- NEVER SKIP**

On first deployment at a new pump or plant:

**Phase 1 --- Data Collection (Hours 0 to 48)**

All inference routes respond with:

``` json
{
  "commissioning_mode": true,
  "message": "Data collection in progress. Predictions unavailable for 48 hours."
}
```

No anomaly alerts. No fault classifications. No CUSUM accumulation. Raw
sensor data logged to `commissioning_buffer/` at 1 Hz.

**Phase 2 --- Recalibration (After 48 hours)**

-   Re-run M2 KMeans clustering on `commissioning_buffer/` data
-   Re-compute M3 normalization baselines
-   Lock new `M3_normalization_config.json`
-   Re-initialize CUSUM mu0 and rolling baseline references from new M3
    config
-   CUSUM state reset to 0.0
-   Rolling slope buffer cleared **Phase 3 --- Lock Config
    (Post-Recalibration)**

  Component                Action
  ------------------------ ---------------------------
  M4 LSTM-AE weights       FROZEN --- DO NOT RETRAIN
  TCN-AE Level 2 weights   FROZEN --- DO NOT RETRAIN
  XGBoost M7 weights       FROZEN --- DO NOT RETRAIN

Only M2 cluster labels + M3 normalization baselines are recalibrated.
`commissioning_mode` flag -\> False. Full inference enabled.

**Invariant:** - M4 threshold 0.110058 is NOT recalibrated during
commissioning. It is the Level 1 static threshold, frozen at M4
commissioning. - The adaptive threshold theta_t (Level 4) IS
recalibrated using new M3 baselines after Phase 2. ---

### Model and Runtime State Loading (at startup --- ALL map_location='cpu')

``` python
from config import DEVICE, MODEL_DIR
import torch, json, pickle
```

**M4 LSTM-AE --- Level 1 (LOCKED --- inference only)**

``` python
lstm_ae_l1 = LSTMAEModel(config)
lstm_ae_l1.load_state_dict(
    torch.load(MODEL_DIR / 'lstm_ae_baseline_best.pth', map_location='cpu')
)
lstm_ae_l1.eval()
M4_THRESHOLD = 0.110058
# NOTE: STATIC — Level 1 only. NEVER apply to TCN-AE output.
```

**TCN-AE Level 2 (NEW v4.0 --- replaces LSTM-AE v2 from v3.0)**

``` python
# 5-layer dilated causal TCN, dilation=[1,2,4,8,16], kernel=3
# Input  : z_t sequence (N_windows x 64)
# Output : score_A (severity), score_B (drift slope), score_C (chain transition)
tcn_ae_l2 = TCNAutoencoder(config)
tcn_ae_l2.load_state_dict(
    torch.load(MODEL_DIR / 'tcn_ae_level2_best.pth', map_location='cpu')
)
tcn_ae_l2.eval()
# NEVER call .cuda() or .to('cuda') on tcn_ae_l2 in deployment.
```

**M7 XGBoost --- 22-class single-label**

``` python
# Trained on M6B_feature_matrix.csv (~35 features at inference, label excluded)
# Saved as pickle (sklearn wrapper). device='cpu' at inference.
with open(MODEL_DIR / 'xgboost_fault_classifier_cpu.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
assert all(e.device == 'cpu' for e in xgb_model.estimators_)
```

**Configs**

``` python
with open(MODEL_DIR / 'M8_threshold_config.json') as f:
    threshold_config = json.load(f)
with open(MODEL_DIR / 'M3_normalization_config.json') as f:
    norm_config = json.load(f)    # READ-ONLY
with open(MODEL_DIR / 'fault_rules_v3.json') as f:
    fault_rules = json.load(f)    # 22-class label map + physics_context strings
```

**z_t Rolling Buffer --- Level 2 TCN-AE Streaming Input (NEW v4.0)**

``` python
# Maintains last N_windows z_t vectors (each 64-dim) in memory per call.
# Each inference call: append new z_t -> feed sequence to TCN-AE Level 2.
# Min windows before TCN fires : 6  (Glass 1 minimum)
# Max buffer length             : 20 (sliding window — oldest dropped)
# Persists across API calls within one container lifecycle.
 
ZT_BUFFER_MIN = 6
ZT_BUFFER_MAX = 20
zt_rolling_buffer = []    # list of np.ndarray shape (64,)
# Resets on pump restart signal OR /api/acknowledge with channels="all".
```

**Layer 3 --- CUSUM Runtime State**

``` python
# Operates on score_B (drift slope from TCN-AE) — NEVER on raw MAE channels.
# Invariant 19: score_B -> CUSUM ONLY. Never route score_A or score_C here.
# Formula: S_n = max(0, S_{n-1} + (score_B_n - mu0_B) - k)
# mu0_B = normal score_B mean (from M8_threshold_config.json)
# k     = 0.5 x (score_B_cusum_threshold - mu0_B)
# Control limit H = 5.0
# PERSISTENT across API calls.
# Resets ONLY on /api/acknowledge or pump restart.
 
cusum_state = {
    "score_B_Sn": 0.0,
    "fired": False,
    "n_consecutive": 0
}
cusum_mu0_B = threshold_config["score_B_normal_mean"]
cusum_k     = 0.5 * (threshold_config["score_B_cusum_threshold"] - cusum_mu0_B)
CUSUM_CONTROL_LIMIT = 5.0
```

**Layer 4 --- Adaptive Threshold + Rolling Baseline (NEW v4.0)**

``` python
# Operates on score_A (severity from TCN-AE) — NEVER on score_B or score_C.
# Invariant 19: score_A -> rolling baseline ONLY.
# Adaptive threshold formula:
#   theta_t = mean(score_A_rolling_buffer) + 3 x std(score_A_rolling_buffer)
# Updates every 50 seconds (every inference call at 1Hz x 50-step window).
# 6-hour rolling window = 6 x 3600 / 50 = 432 inference calls.
# Warmup: theta_t activates after 216 calls (half window).
# Until warmup complete: M4_THRESHOLD (0.110058) governs Level 1 gate as fallback.
# NOTE: adaptive theta_t is Level 4 ONLY.
#       M4_THRESHOLD = 0.110058 remains Level 1 static gate — NEVER changed.
 
ROLLING_WINDOW_CALLS  = 432
score_A_rolling_buffer = []    # list of float, max len = 432
adaptive_threshold     = 0.110058    # initialised to static threshold at startup
```

**Physics Context Lookup (NEW v4.0)**

``` python
# Static lookup table loaded from fault_rules_v3.json at startup.
# Maps label_int (0–21) -> physics_context dict.
# NOT ML inference — pure static lookup. Safe for all pump types.
physics_context_table = {int(k): v for k, v in fault_rules["physics_context"].items()}
# Structure per label: what, why, timeline, action, if_ignored, disclaimer
```

------------------------------------------------------------------------

### Score Routing --- Invariant 19 (ENFORCED IN M10 RUNTIME --- NEVER CROSS)

  -----------------------------------------------------------------------
  Score                   Routes To               Never To
  ----------------------- ----------------------- -----------------------
  score_A                 Layer 4 Rolling         CUSUM, XGBoost directly
                          Baseline only           

  score_B                 Layer 3 CUSUM only      Rolling Baseline,
                                                  XGBoost directly

  score_C                 XGBoost M7 (as          CUSUM, Rolling Baseline
                          onset_order / chain     
                          feature) only           
  -----------------------------------------------------------------------

> Cross-routing = architecture violation. Any code path routing score_B
> to rolling baseline, score_A to CUSUM, or either to XGBoost directly =
> **BLOCK**. Fix before M10 testing begins.

------------------------------------------------------------------------

### API Routes (8 Routes)

#### Route 1 --- POST /api/anomaly_detect

**Purpose:** Real-time anomaly detection on incoming sensor window
**Input:** JSON or CSV upload --- 50 rows x 8 sensor columns (raw
values)

**Process:**

-   Step 1: Normalize using `M3_normalization_config.json`
    (cluster-aware)

-   Step 2: Detect cluster from M2 KMeans model

-   Step 3: Run 4-layer inference protocol **Layer 1 --- M4 LSTM-AE
    single-window**

        MAE vs M4_THRESHOLD (0.110058 — STATIC, Level 1 only)
        Output: per-channel MAE (8 values), z_t (64-dim bottleneck vector)
        Append z_t to zt_rolling_buffer (max len = ZT_BUFFER_MAX = 20)

    **Layer 2 --- TCN-AE Level 2 on z_t sequence**

        Fires when len(zt_rolling_buffer) >= ZT_BUFFER_MIN (6)
        Input: zt_rolling_buffer as array shape (N_windows, 64)
        Output:
          score_A = mean MAE over N_windows (overall severity)
          score_B = OLS slope of per-window recon error (drift slope)
          score_C = max delta of consecutive recon errors (chain transition)
        If buffer < 6 windows: skip Layer 2, set score_A/B/C = None

    **Layer 3 --- CUSUM update on score_B (Invariant 19 --- score_B
    ONLY)**

        S_n = max(0, S_{n-1} + (score_B - cusum_mu0_B) - cusum_k)
        cusum_state["score_B_Sn"] = S_n
        cusum_state["fired"] = (S_n > CUSUM_CONTROL_LIMIT)

    **Layer 4 --- Adaptive threshold update + rolling baseline on
    score_A**

        Append score_A to score_A_rolling_buffer (max len = 432)
        If len(buffer) >= 216 (warmup complete):
          adaptive_threshold = mean(buffer) + 3 x std(buffer)
        rolling_baseline_alert = (score_A > adaptive_threshold)
        NOTE: adaptive_threshold updates every 50 seconds automatically.

-   Step 4: Fetch `physics_context` from
    `physics_context_table[predicted_label]`

-   Step 5: Return full output dict **Output:**

  --------------------------------------------------------------------------------------------------
  Field                                       Type                    Description
  ------------------------------------------- ----------------------- ------------------------------
  alert_state                                 str                     NORMAL / WATCH / WARN / DANGER

  anomaly_flag                                bool                    ---

  mae_per_channel                             dict channel -\> float  Layer 1

  z_t_norm                                    float                   L2 norm of z_t

  score_A                                     float or None           Layer 2

  score_B                                     float or None           Layer 2

  score_C                                     float or None           Layer 2

  zt_buffer_len                               int                     ---

  cusum_state.score_B_Sn                      float                   ---

  cusum_state.fired                           bool                    ---

  cusum_state.n_consecutive                   int                     ---

  rolling_baseline_alert.score_A_current      float                   ---

  rolling_baseline_alert.adaptive_threshold   float                   ---

  rolling_baseline_alert.alert                bool                    ---

  rolling_baseline_alert.buffer_len           int                     ---

  adaptive_threshold_active                   bool                    False until 216-call warmup

  bearing_wear_gradual_advisory               str or None             "Plan bearing inspection
                                                                      within 7--14 days." if
                                                                      cusum_state.fired AND
                                                                      rolling_baseline_alert.alert =
                                                                      True AND predicted_label = 21

  physics_context.what                        str                     plain-language fault
                                                                      description

  physics_context.why                         str                     physical causal mechanism

  physics_context.timeline                    str                     how fast this fault develops

  physics_context.action                      str                     recommended immediate action

  physics_context.if_ignored                  str                     consequence of inaction

  physics_context.disclaimer                  str                     always: Advisory only ---
                                                                      consult qualified engineer

  limitation_flags                            list of str             applicable limitation IDs from
                                                                      File 3 registry

  cluster                                     str                     ---

  confidence                                  float                   ---

  uncertainty_std                             float                   MC Dropout N=20, CPU

  commissioning_mode                          bool                    True during first 48h
  --------------------------------------------------------------------------------------------------

> **Note:** `physics_context` = None if `alert_state = NORMAL`

**CUSUM Reset Rule:** - `cusum_state` resets to 0.0 ONLY on: - (a) POST
`/api/acknowledge` --- operator confirms action taken - (b) Pump restart
signal: PresSV drops to near 0 -\> startup cluster detected - Do NOT
auto-reset on NORMAL windows --- gradual drift MUST accumulate. - Do NOT
reset on WATCH -\> NORMAL transition --- this destroys the CUSUM signal.
**ZT Buffer Reset Rule:** - `zt_rolling_buffer` clears ONLY on: - (a)
POST `/api/acknowledge` with `channels="all"` - (b) Pump restart signal
detected - Do NOT clear on every API call --- the buffer IS the temporal
memory. **Scope Check:**

``` python
if pump_type == 'household': return physics_advisory_only()
else: run full 4-layer inference
```

------------------------------------------------------------------------

#### Route 2 --- POST /api/classify_fault

**Purpose:** Classify fault type from feature snapshot (22-class, labels
0--21) **Input:** JSON --- \~35 feature values (M6.5r feature set, label
excluded). Feature order: same as `M6B_feature_matrix.csv` (all columns
except label_int). `score_C` must be included in input features
(Invariant 19 routing). OR: raw 50-step window -\> run Layer 1 + Layer 2
inference on-the-fly.

**Process:** 1. If raw window: run Layer 1 + Layer 2 -\> extract \~35
features 2. Run M7 22-class XGBoost (device='cpu') 3. get
`predict_proba` -\> probability array over 22 classes (labels 0--21) 4.
Apply Stage 1/2/3 progressive confidence logic (threshold=0.75) 5. Map
`label_int` -\> fault name via `fault_rules_v3.json` 6. Map compound
label (7--12) -\> causal chain string for UI 7. Run SHAP TreeExplainer
on X_input -\> top-3 features per predicted fault 8. Fetch
`physics_context` from `physics_context_table[predicted_label]` \> All
stages output includes `physics_context` (same structure as Route 1) and
`limitation_flags` (list of applicable limitation IDs).

**Stage 1 output** (primary_conf \< 0.50)

  Field              Value
  ------------------ ----------------------------------------------
  stage              1
  message            "Minor anomaly --- multiple causes possible"
  top3_candidates    list of (fault_name, prob)
  action             "Monitor all channels closely"
  physics_context    physics_context_table\[top_candidate_label\]
  limitation_flags   list of str

**Stage 2 output** (0.50 \<= primary_conf \< 0.75)

  ------------------------------------------------------------------------------
  Field                               Value
  ----------------------------------- ------------------------------------------
  stage                               2

  message                             "Probable fault: \<n\> (XX%)"

  secondary_faults                    dict fault_name -\> prob (all labels with
                                      prob \> 0.30)

  action                              "Schedule inspection within 48h"

  physics_context                     physics_context_table\[predicted_label\]

  limitation_flags                    list of str
  ------------------------------------------------------------------------------

**Stage 3 output** (primary_conf \>= 0.75)

  Field              Value
  ------------------ ------------------------------------------
  stage              3
  message            "CONFIRMED: \<n\> (XX%)"
  secondary_faults   dict fault_name -\> prob
  fault_stage        early / developing / advanced
  shap_top3          list of dict (feature, value, direction)
  physical_meaning   str
  action             urgency string based on fault_stage
  causal_chain       str or None (Group B labels 7--12)
  physics_context    physics_context_table\[predicted_label\]
  limitation_flags   list of str

**Compound Fault Label Map (Group B --- single integer -\> causal chain
display)**

  Label   Primary              Secondary
  ------- -------------------- --------------
  7       bearing_wear         overloading
  8       cavitation           seal_failure
  9       impeller_imbalance   bearing_wear
  10      seal_failure         cavitation
  11      overloading          bearing_wear
  12      impeller_imbalance   cavitation

*Source: `fault_rules_v3.json` (written by M6B Step 3 ---
authoritative)*

**Scope Check:**

``` python
if pump_type == 'household': return physics_advisory_only()
```

------------------------------------------------------------------------

#### Route 3 --- POST /api/select_pump

**Purpose:** Industrial pump sizing and selection **Input:** JSON ---
`flow_rate_m3h`, `total_head_m`, `fluid_density`, `fluid_temp_c`,
`suction_head_m`, `pipe_length_m`, `pipe_diameter_m` **Process:** M9
industrial pump selector (physics only) **Output:** Full M9 industrial
output dict **No ML:** Pure physics --- no model inference

------------------------------------------------------------------------

#### Route 4 --- GET /api/household

**Purpose:** Household pump advisory **Input:** Query params ---
`usage_type`, `daily_demand_lph`, `static_head_m`, `pipe_length_m`,
`pipe_diameter_mm` **Process:** M9 household advisor (physics only)
**Output:** M9 household output dict + `advisory_disclaimer` always
appended **No ML:** `physics_advisory_only()` --- scope boundary
enforced **UI label:** "Advisory guidance only --- not a monitoring
tool"

------------------------------------------------------------------------

#### Route 5 --- POST /api/acknowledge

**Purpose:** Operator acknowledgement --- resets CUSUM + z_t buffer +
rolling baseline state after maintenance action confirmed

**Input:**

  Field          Type                   Description
  -------------- ---------------------- ---------------------------------
  channels       list of str OR "all"   ---
  pump_id        str                    ---
  action_taken   str                    free text --- maintenance notes

**Process:**

``` python
if channels == "all":
    cusum_state["score_B_Sn"] = 0.0
    cusum_state["fired"]      = False
    zt_rolling_buffer.clear()
    score_A_rolling_buffer.clear()
    adaptive_threshold reset to M4_THRESHOLD  # warmup restarts
else:
    # Reset only specified channel accumulators (partial ACK)
    pass
# Log acknowledgement with timestamp and action_taken text.
```

**Output:**

  Field                      Type
  -------------------------- -------------
  acknowledged_channels      list of str
  cusum_reset                True
  zt_buffer_reset            True
  rolling_baseline_reset     True
  adaptive_threshold_reset   bool
  timestamp                  str

**Access:** Production UI --- operator must explicitly trigger. Do NOT
auto-call from anomaly_detect route.

------------------------------------------------------------------------

#### Route 6 --- POST /api/validate_model

**Purpose:** M12 adversarial validation entry point **Input:** JSON ---
`config_id` (1--16), sequence data OR auto-generate flag **Process:**
Trigger M12 validation suite for specified config

**Output:**

  Field           Type
  --------------- -----------------
  config_id       int
  alert_state     str
  detection_lag   int (timesteps)
  gate_pass       bool
  details         str

**Access:** Internal only --- not exposed in production UI

------------------------------------------------------------------------

#### Route 7 --- GET /health

**Purpose:** Docker / Hugging Face health check

**Output:**

  Field                Value
  -------------------- --------------------------------------------------------
  status               healthy
  models_loaded        \[lstm_ae_l1, tcn_ae_l2, xgboost_22class, m9_physics\]
  device               cpu
  version              4.0
  cusum_active         true
  tcn_ae_active        true
  zt_buffer_len        int (current buffer depth)
  adaptive_threshold   float (current theta_t value)
  n_fault_classes      22
  commissioning_mode   bool

*Used by: Docker HEALTHCHECK + Hugging Face Spaces liveness probe*

------------------------------------------------------------------------

#### Route 8 --- GET /api/physics_context (NEW v4.0)

**Purpose:** Return plain-language fault description for any label
(0--21). Useful for UI tooltip, operator training, standalone lookup.
**Input:** Query param --- `label=<int>` (0--21) OR `label="all"` -\>
return full table for all 22 classes **Process:** Static lookup in
`physics_context_table`. Loaded at startup from `fault_rules_v3.json`.
NOT ML inference --- safe for all pump types.

**Output:**

  -----------------------------------------------------------------------
  Field                   Type                    Description
  ----------------------- ----------------------- -----------------------
  label_int               int                     ---

  label_str               str                     ---

  what                    str                     plain-language fault
                                                  description

  why                     str                     physical causal
                                                  mechanism --- pump
                                                  physics

  timeline                str                     how fast this fault
                                                  develops --- seconds to
                                                  weeks

  action                  str                     recommended immediate
                                                  action

  if_ignored              str                     consequence of inaction
                                                  --- physical outcome

  disclaimer              str                     always: Advisory only
                                                  --- consult qualified
                                                  engineer
  -----------------------------------------------------------------------

**Example --- Label 10 (seal_failure -\> cavitation):**

    label_int  : 10
    label_str  : seal_failure -> cavitation
    what       : Mechanical seal degrading, allowing internal leakage that
                 reduces net suction head. NPSHa is approaching NPSHr.
    why        : Seal gap (A_gap) allows Q_leak = Cd x A_gap x sqrt(2*delta_P/rho).
                 Leakage shifts operating point left on Q-H curve, raising
                 recirculation head losses until NPSHa < NPSHr.
    timeline   : Full progression: 300s seal phase + 400–800s hydraulic lag +
                 60s cavitation onset. Total: 760–1160 seconds at 40 bar.
    action     : Reduce pump speed immediately. Inspect mechanical seal faces.
                 Check suction pressure. Do not restart until seal replaced.
    if_ignored : Impeller cavitation damage within 60–180s of NPSHa crossing NPSHr.
                 Bubble collapse at 7-stage impeller tips causes pitting and
                 catastrophic efficiency loss.
    disclaimer : Advisory only — consult qualified engineer before action.

**Access:** Public --- exposed in production UI as fault tooltip and
operator guide. No authentication required. Read-only static lookup.

------------------------------------------------------------------------

### 4-State Alert UI Rendering

#### NORMAL

-   **Condition:** score_A \< adaptive_threshold, CUSUM S_n \< 1.0, no
    rolling alert
-   **Display:** "System operating within normal parameters"
-   **Colour:** 🟢 Green
-   **Action:** None
-   **Note:** If adaptive_threshold not yet active (warmup incomplete),
    Level 1 static threshold (0.110058) governs NORMAL gate. \#### WATCH
-   **Condition:** score_A rising OR CUSUM S_n rising OR rolling
    baseline drift
-   **Display:** "Early anomaly trend detected --- monitor closely"
-   **Colour:** 🟡 Yellow
-   **Details shown:**
    -   score_A current value vs adaptive_threshold
    -   zt_buffer_len (how many windows in memory)
    -   score_B trend (CUSUM S_n rising)
    -   Stage 2 classify_fault: probable fault + secondary candidates
    -   physics_context for probable fault (plain language)
    -   Limitation flags applicable to this prediction
    -   CUSUM panel: if cusum_state.fired = True -\> show S_n and
        n_consecutive **BEARING WEAR GRADUAL (label 21) --- WATCH
        display:**
-   **Trigger:** CUSUM S_n(score_B) \> 3.0 AND
    rolling_baseline_alert.alert = True
-   **Display:** "Warning: Gradual bearing wear trend detected" / "CUSUM
    accumulator S_n = X.X (control limit = 5.0)" / "Adaptive threshold:
    theta_t = X.XXXX (static was 0.110058)"
-   **Advisory:** "Plan bearing inspection within 7--14 days."
-   **Note:** Level 1 MAE will be BELOW threshold --- this is EXPECTED.
    CUSUM (score_B) + Rolling Baseline (score_A) are the PRIMARY
    detection path for label 21. Do NOT treat sub-threshold Level 1 MAE
    as a false alarm for this class. \#### WARN
-   **Condition:** score_A \> adaptive_threshold sustained, score_B
    slope positive
-   **Display:** "Sustained anomaly --- schedule maintenance inspection"
-   **Colour:** 🟠 Orange
-   **Details shown:**
    -   Estimated time to DANGER at current score_A trend rate
    -   Stage 2 classify_fault: probable primary fault
    -   Secondary fault candidates (compound fault visible here)
    -   SHAP top-3 feature explanation (Stage 2 or 3)
    -   physics_context for primary predicted fault
    -   Limitation flags \#### DANGER
-   **Condition:** score_A \>\> adaptive_threshold OR CUSUM S_n \> 5.0
    OR Level 1 MAE \> 0.110058
-   **Display:** "Fault confirmed --- immediate maintenance action
    required"
-   **Colour:** 🔴 Red
-   **Details shown:**
    -   Stage 3 classify_fault (always at DANGER)
    -   PRIMARY fault: confirmed name + confidence %
    -   SECONDARY faults: compound pair shown if prob \> 0.30
    -   fault_stage: early / developing / advanced
    -   causal_chain string for Group B labels (7--12)
    -   SHAP top-3 feature explanation
    -   physics_context (full --- what/why/timeline/action/if_ignored)
    -   Limitation flags (all applicable)
    -   MC Dropout uncertainty_std
    -   score_C value shown for Group B: "Chain transition confirmed at
        window N" **CAVITATION DANGER specific display:**

    ```{=html}
    <!-- -->
    ```
        "CAVITATION DETECTED — STOP PUMP IMMEDIATELY"
        "Impeller damage risk within 60–180 seconds of continued operation"
        "Check inlet valve, suction line, and NPSH conditions before restart"
        physics_context for label 3 or label 10 (as applicable)
        Stage 3 output always fires for cavitation

**COMPOUND FAULT display example (label 10: seal_failure -\>
cavitation):**

    PRIMARY  : seal_failure (84%)
    SECONDARY: cavitation (79%) — NPSHa crossed NPSHr after hydraulic lag
    CHAIN    : seal_failure -> cavitation
    score_C  : Chain transition detected at window 13 of 18
    ACTION   : Shutdown immediately — seal replacement + impeller inspection required
    PHYSICS  : physics_context.what + physics_context.timeline

------------------------------------------------------------------------

### Physics Context Layer --- Per-Alert Limitation Flags (NEW v4.0)

Every industrial alert (WATCH / WARN / DANGER) carries a
`limitation_flags` list. Source: File 3
(`module_M8_lstm_ae_v2_architecture.md`) Limitation Registry.

#### Limitation IDs Displayed in UI

  -----------------------------------------------------------------------
  ID                                  Description
  ----------------------------------- -----------------------------------
  L1_single_installation              "Trained on 1 CIRA installation.
                                      Sensor placement must follow ISO
                                      13373. Verify calibration on your
                                      specific installation."

  L2_no_rul                           "No Remaining Useful Life estimate.
                                      timeline field is physics-based
                                      approximation only."

  L3_static_threshold                 "Level 1 threshold (0.110058) may
                                      drift false-positive over pump
                                      lifespan. Adaptive threshold (Level
                                      4) compensates but requires 6hr
                                      warmup on new installation."

  L4_synthetic_domain                 "Compound and masked fault data is
                                      physics-synthetic. Real compound
                                      fault characteristics may differ
                                      from training distribution."

  L5_label21_early                    "Gradual bearing wear (label 21)
                                      detection relies on CUSUM
                                      accumulation (score_B). Early-stage
                                      S_n \< 3.0 may not yet trigger
                                      WATCH. Continue monitoring."

  L6_confidence_proxy                 "MC Dropout uncertainty_std is a
                                      confidence proxy only. Not a
                                      calibrated probability interval."
  -----------------------------------------------------------------------

#### Which Flags Appear Per Alert

  Alert Level              Flags
  ------------------------ ---------------------------------------------------------------
  WATCH                    L3_static_threshold, L5_label21_early (if label 21)
  WARN                     L3_static_threshold, L4_synthetic_domain, L6_confidence_proxy
  DANGER                   All applicable --- determined by predicted label group
  Group B labels (7--12)   L4_synthetic_domain always included
  Label 21                 L5_label21_early always included

------------------------------------------------------------------------

### Mandatory Disclaimers (before ANY industrial inference --- all 3 required)

**Disclaimer 1 --- Model Scope** \> "This model is trained on CIRA SACIP
dataset (1 specific installation). Sensor placement must follow ISO
13373 guidelines. r=0.9793 coupling between Mot.TV and Temp.SV is
installation-specific. Model outputs are advisory --- consult a
qualified engineer before action."

**Disclaimer 2 --- Sensor Dependency** \> "Inference quality depends
entirely on sensor hardware integrity. Sensor malfunction or
miscalibration will affect model output. Verify sensor health
independently before acting on any alert."

**Disclaimer 3 --- Safety System Boundary** \> "PumpSmart is a condition
monitoring tool (ISO 13374 Level 3). It is NOT a Safety Instrumented
System (SIS) per IEC 61511. Hardwired process trips remain the primary
safety barrier. PumpSmart alerts are advisory and do not replace
hardwired protection."

------------------------------------------------------------------------

### M10 Local Testing Protocol (15 Tests)

Before pushing to M11, verify locally:

**Core Routes**

  -----------------------------------------------------------------------
  Test                    Description             Expected
  ----------------------- ----------------------- -----------------------
  Test 1                  `flask run`             server starts without
                                                  error

  Test 2                  GET /health             models_loaded =
                                                  \[lstm_ae_l1,
                                                  tcn_ae_l2,
                                                  xgboost_22class,
                                                  m9_physics\],
                                                  tcn_ae_active = true,
                                                  cusum_active = true,
                                                  n_fault_classes = 22,
                                                  version = "4.0"
  -----------------------------------------------------------------------

**Anomaly Detection --- Layer 1**

  -----------------------------------------------------------------------
  Test                    Description             Expected
  ----------------------- ----------------------- -----------------------
  Test 3                  POST                    alert_state = NORMAL
                          /api/anomaly_detect -\> 
                          normal window           

  Test 4                  POST                    alert_state = DANGER
                          /api/anomaly_detect -\> 
                          known hard fault        

  Test 5                  POST                    alert_state = WATCH or
                          /api/anomaly_detect -\> WARN
                          mild fault              
  -----------------------------------------------------------------------

**Anomaly Detection --- Layer 2 TCN-AE z_t Buffer (NEW)**

  ---------------------------------------------------------------------------
  Test                    Description             Expected
  ----------------------- ----------------------- ---------------------------
  Test 6                  Send 7 consecutive      zt_buffer_len must reach 7;
                          windows (same mild      score_A, score_B, score_C
                          fault)                  must be non-None from call
                                                  6 onward;
                                                  adaptive_threshold_active =
                                                  False (warmup not yet done
                                                  --- expected); all 3 scores
                                                  present in output JSON

  ---------------------------------------------------------------------------

**Anomaly Detection --- Layers 3+4 CUSUM and Rolling Baseline**

  -----------------------------------------------------------------------------------
  Test                    Description             Expected
  ----------------------- ----------------------- -----------------------------------
  Test 7                  Simulate 40 consecutive cusum_state\["score_B_Sn"\] must
                          label-21 windows        rise above 0 across calls;
                          (severity=0.15)         rolling_baseline_alert\["alert"\] =
                                                  True after \~30 windows;
                                                  bearing_wear_gradual_advisory =
                                                  "Plan bearing inspection within
                                                  7--14 days."; alert_state = WATCH
                                                  (NOT DANGER --- Level 1 MAE below
                                                  0.110058 --- expected)

  Test 8                  POST /api/acknowledge   cusum_state\["score_B_Sn"\] resets
                          -\> channels="all"      to 0.0; zt_buffer_len resets to 0;
                                                  adaptive_threshold reset confirmed
                                                  in /health response
  -----------------------------------------------------------------------------------

**Fault Classification --- 22-Class**

  -----------------------------------------------------------------------
  Test                    Description             Expected
  ----------------------- ----------------------- -----------------------
  Test 9                  POST                    Stage 3, label=3
                          /api/classify_fault -\> 
                          cavitation features     

  Test 10                 POST                    Stage 1 or 2
                          /api/classify_fault -\> 
                          overloading mild        

  Test 11                 POST                    Stage 3, label=10,
                          /api/classify_fault -\> causal_chain =
                          compound                "seal_failure -\>
                          seal+cavitation (label  cavitation"; score_C
                          10)                     present in input
                                                  features and top SHAP
                                                  feature;
                                                  secondary_faults
                                                  populated
  -----------------------------------------------------------------------

**Physics Context (NEW)**

  ----------------------------------------------------------------------------------------------------------
  Test                    Description                      Expected
  ----------------------- -------------------------------- -------------------------------------------------
  Test 12                 GET                              Response contains
                          /api/physics_context?label=10    what/why/timeline/action/if_ignored/disclaimer;
                                                           if_ignored contains reference to NPSHa/NPSHr
                                                           crossing

  Test 13                 GET                              22 entries returned, labels 0--21 all present
                          /api/physics_context?label=all   
  ----------------------------------------------------------------------------------------------------------

**Physics Tools**

  Test      Description                                  Expected
  --------- -------------------------------------------- ---------------------------------------
  Test 14   POST /api/select_pump -\> nameplate inputs   motor = 110 kW
  Test 15   GET /api/household                           advisory_disclaimer present in output

**Scope Boundary --- verify across all tests:** - All industrial routes:
3 disclaimers visible before inference - Household route:
`physics_advisory_only()` fires, no ML model called - All
WATCH/WARN/DANGER alerts: `limitation_flags` non-empty in response - All
WATCH/WARN/DANGER alerts: `physics_context` non-null in response ---

### M10 Outputs

    app/
      app.py
      routes/
        anomaly.py      <- /api/anomaly_detect  (4-layer inference, score routing)
        classify.py     <- /api/classify_fault  (22-class, Stage 1/2/3, causal chain)
        selector.py     <- /api/select_pump + /api/household
        acknowledge.py  <- /api/acknowledge     (CUSUM + z_t + rolling baseline reset)
        validate.py     <- /api/validate_model
        health.py       <- /health
        physics.py      <- /api/physics_context (NEW v4.0 — static lookup, 22 classes)
      runtime/
        cusum_state.py      <- CUSUM S_n on score_B — persistent state class
        rolling_state.py    <- score_A rolling buffer + adaptive threshold updater
        zt_buffer.py        <- z_t rolling buffer for TCN-AE streaming (NEW v4.0)
        physics_context.py  <- static lookup loader from fault_rules_v3.json (NEW v4.0)
      templates/
        index.html      <- main dashboard (4-state UI, CUSUM panel, physics context)
        household.html  <- household advisor UI
        selector.html   <- industrial selector UI
      static/
        style.css
        dashboard.js
    outputs/reports/module_10_flask_app_report.md

### M10 Paste Text Keys

  ---------------------------------------------------------------------------------
  Key                                 Value
  ----------------------------------- ---------------------------------------------
  M10_routes_registered               \[list of 8 routes\]

  M10_health_check_response           healthy/error

  M10_models_loaded_at_startup        \[lstm_ae_l1, tcn_ae_l2, xgboost_22class,
                                      m9_physics\]

  M10_tcn_ae_active_at_startup        True/False

  M10_normal_window_test              NORMAL/error

  M10_fault_window_test               DANGER/error

  M10_mild_fault_test                 WATCH/WARN/error

  M10_zt_buffer_layer2_test           score_A_B_C_non_null_at_window_6/error

  M10_compound_fault_test             causal_chain_visible/score_C_top_SHAP/error

  M10_label21_cusum_test              WATCH+advisory_visible/error

  M10_cusum_reset_test                reset_confirmed/error

  M10_cusum_active_at_startup         True/False

  M10_adaptive_threshold_active       True/False

  M10_physics_context_route_test      22_labels_returned/error

  M10_limitation_flags_in_response    True/False

  M10_commissioning_mode_documented   True/False

  M10_household_scope_enforced        True/False

  M10_disclaimers_displayed           True/False

  M10_local_tests_pass                \[X/15\]

  Status_for_M11                      READY/BLOCKED
  ---------------------------------------------------------------------------------

------------------------------------------------------------------------

## M11 --- Docker + Hugging Face Deployment

**Status:** NOT STARTED (requires M10 15/15 local tests pass)

------------------------------------------------------------------------

### Dockerfile

``` dockerfile
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
 
EXPOSE 7860
 
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1
 
CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:7860", \
     "--workers", "1", "--timeout", "120"]
```

------------------------------------------------------------------------

### requirements.txt (Deployment --- CPU Only)

    torch==2.6.0+cpu
    xgboost>=2.0
    flask>=3.0
    gunicorn>=21.0
    scikit-learn>=1.3
    numpy>=1.24
    pandas>=2.0
    shap>=0.44
    scipy>=1.11

> **Note:** TCN-AE is implemented in PyTorch --- already covered by
> `torch==2.6.0+cpu`. No new library required vs v3.0.
> `requirements.txt` UNCHANGED.

------------------------------------------------------------------------

### Model Loading Rules (Deployment --- NON-NEGOTIABLE)

**M4 LSTM-AE Level 1**

``` python
lstm_ae_l1.load_state_dict(
    torch.load('models/lstm_ae_baseline_best.pth', map_location='cpu')
)
```

**TCN-AE Level 2**

``` python
tcn_ae_l2.load_state_dict(
    torch.load('models/tcn_ae_level2_best.pth', map_location='cpu')
)
# NEVER call .cuda() or .to('cuda') on either model in deployment code.
```

**XGBoost 22-class**

``` python
import pickle
with open('models/xgboost_fault_classifier_cpu.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
assert all(e.device == 'cpu' for e in xgb_model.estimators_)
```

**CUSUM (score_B), z_t buffer, rolling baseline (score_A):** - All
in-memory Python state --- NOT stored in model weights. - Persist across
API calls within one container lifecycle. - Reset on container restart
OR `/api/acknowledge`. - Score routing Invariant 19 enforced in
`anomaly.py` --- NEVER crossed. ---

### GitHub Actions CI/CD Pipeline

**File:** `.github/workflows/deploy.yml`

``` yaml
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

------------------------------------------------------------------------

### Hugging Face Spaces Configuration

**File:** `README.md` front matter

``` yaml
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

------------------------------------------------------------------------

### Deployment Validation Checklist (12 Checks)

  --------------------------------------------------------------------------------------------------------------
  Check                   Description                           Expected
  ----------------------- ------------------------------------- ------------------------------------------------
  Check 1                 Docker build locally                  no errors

  Check 2                 `docker run -p 7860:7860 pumpsmart`   container starts

  Check 3                 GET http://localhost:7860/health      status = healthy, version = 4.0; models_loaded =
                                                                \[lstm_ae_l1, tcn_ae_l2, xgboost_22class,
                                                                m9_physics\]; cusum_active = true, tcn_ae_active
                                                                = true; n_fault_classes = 22, commissioning_mode
                                                                = false

  Check 4                 POST /api/anomaly_detect              valid response (not 500); score_A/B/C present in
                                                                output

  Check 5                 POST /api/classify_fault              valid Stage 3 fault (label 0--21)

  Check 6                 POST /api/classify_fault compound     causal_chain = "seal_failure -\> cavitation";
                          window (label 10)                     score_C top SHAP feature confirmed

  Check 7                 POST /api/acknowledge                 cusum_state resets, zt_buffer_len = 0

  Check 8                 GET /api/physics_context?label=10     what/why/timeline/action/if_ignored/disclaimer
                                                                all non-empty

  Check 9                 Image size                            \< 2GB (Hugging Face free tier limit)

  Check 10                Startup time                          \< 60s (within HEALTHCHECK start-period)

  Check 11                Push to Hugging Face Spaces           Space builds successfully

  Check 12                HF Space URL /health                  status = healthy, version = 4.0; GitHub Actions
                                                                workflow passes on push to main
  --------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

### M11 Outputs

-   `Dockerfile`
-   `requirements.txt` (deployment version --- CPU only)
-   `.github/workflows/deploy.yml` (GitHub Actions CI/CD)
-   `README.md` (Hugging Face Spaces front matter)
-   `outputs/reports/module_11_deployment_report.md` \### M11 Paste Text
    Keys

  Key                              Value
  -------------------------------- ----------------------------
  M11_docker_build_status          SUCCESS/FAILED
  M11_container_startup_time_s     \[seconds --- gate \< 60\]
  M11_image_size_mb                \[MB --- gate \< 2000\]
  M11_health_check_local           healthy/error
  M11_tcn_ae_active_in_container   True/False
  M11_cusum_active_in_container    True/False
  M11_physics_context_route_test   PASS/FAIL
  M11_hf_deployment_url            \[URL\]
  M11_hf_health_check              healthy/error
  M11_github_actions_status        PASS/FAIL
  M11_compound_fault_route_test    PASS/FAIL
  M11_all_checks_pass              True/False
  Status_for_M12                   READY/BLOCKED

------------------------------------------------------------------------

## Module Dependency Summary

    M7 XGBoost (22-class, ~35 features, M6B_feature_matrix.csv)
    M8 Level 1 LSTM-AE + Level 2 TCN-AE + Layer 3 CUSUM + Layer 4 Rolling Baseline
    M9 Physics Tools
        |
        v
    M10 Flask App
        score routing (Invariant 19)
        z_t rolling buffer
        adaptive threshold theta_t
        physics context lookup
        commissioning mode
        |
        v
    M11 Docker + Hugging Face
        |
        v
    M12 Adversarial Validation

### Sequencing Law

  Gate                       Unlocks
  -------------------------- ---------------------------
  M7 gates pass              M8 starts
  M8 gates pass              M9 finalised + M10 starts
  M10 15/15 tests pass       M11 starts
  M11 deployment OK          M12 starts
  M12 PRODUCTION_VALIDATED   system live

------------------------------------------------------------------------

## Document Revision History

  -----------------------------------------------------------------------------------
  Version                 Date                    Changes
  ----------------------- ----------------------- -----------------------------------
  v1.0                    2026-04-12              Initial creation --- split from
                                                  `module_pathway_M1_to_M12_v10.md`

  v2.0                    2026-04-12              Bias-audit cascade: multi-label
                                                  classify route, Stage 1/2/3 API
                                                  schema, compound fault UI display,
                                                  MultiOutputClassifier pickle
                                                  loading, scikit-learn deployment
                                                  dependency, 12-test local protocol,
                                                  M11 compound route check

  v3.0                    2026-04-16              Architecture v14.0: CUSUM runtime
                                                  state (Layer 3, raw MAE) + rolling
                                                  baseline (Layer 4) +
                                                  /api/acknowledge + 22-class XGBoost
                                                  throughout + label 21 advisory + 7
                                                  routes + 13 local tests + 11 M11
                                                  checks

  v4.0                    2026-04-21              Architecture v14.2: TCN-AE Level 2
                                                  replaces LSTM-AE v2. z_t rolling
                                                  buffer added (streaming).
                                                  score_A/B/C routing per Invariant
                                                  19 enforced in M10. Adaptive
                                                  threshold theta_t =
                                                  mu_rolling(6hr) +
                                                  3\*sigma_rolling(6hr) added (Layer
                                                  4). /api/physics_context added as
                                                  Route 8. physics_context field in
                                                  Route 1 + Route 2 output.
                                                  limitation_flags per alert.
                                                  Commissioning mode 48hr documented.
                                                  /health updated: version=4.0,
                                                  tcn_ae_active, zt_buffer_len,
                                                  adaptive_threshold. 8 routes, 15
                                                  local tests, 12 M11 checks, 18 M10
                                                  paste keys.
  -----------------------------------------------------------------------------------

------------------------------------------------------------------------



  -----------------------------------------------------------------------
  Field                               Value
  ----------------------------------- -----------------------------------
  Pump                                110 kW, 7-stage, 40 bar, 2980 RPM,
                                      45 m3/h, 450 m head --- CIRA SACIP

  Standards                           ISO 10816-3 vibration \| ISO
                                      13373-3 monitoring \| ISO 13374
                                      Level 3 \| IEC 61511 boundary

  Architecture                        v14.2

  Classes                             22 (labels 0--21)

  Detection layers                    4
  -----------------------------------------------------------------------
