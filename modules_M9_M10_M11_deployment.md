# PumpSmart — M9 + M10 + M11: Deployment Modules
 
**Pump Selector | FastAPI Web Application | Docker + Hugging Face Deployment**
 
> **ARCHITECTURE CHANGE — 2026-05-10**
> Flask → FastAPI throughout. Reason: PumpSmart M10 requires real-time continuous
> sensor data inflow, persistent state (CUSUM accumulator, rolling score_A buffer,
> z_t streaming buffer), and async-native concurrent request handling — all of which
> FastAPI + uvicorn handles natively without threading hacks. gunicorn/WSGI replaced
> by uvicorn/ASGI. All route signatures, state management, and validation updated
> accordingly. This change propagates through M10, M11, Dockerfile, requirements.txt,
> GitHub Actions, and all dependency diagrams below.
 
| Field | Value |
|---|---|
| Document version | v5.0 — Architecture v14.2 (FastAPI migration) |
| Supersedes | v4.0 (Flask) |
| Date | 2026-05-10 |
| Prerequisites | M8 all_gates_pass = True \| M7 all_gates_pass = True \| M9 24/24 PASS |
| Status | M9 COMPLETE LOCKED — M10 IN DEVELOPMENT |
| Pump | 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP |
 
---
 
## What Changed in v5.0 (FastAPI Migration)
 
| Item | v4.0 (Flask) | v5.0 (FastAPI) |
|---|---|---|
| Web framework | `flask>=3.0` | `fastapi>=0.111.0` |
| WSGI/ASGI server | `gunicorn` | `uvicorn[standard]>=0.29.0` |
| Request handlers | `def` (synchronous) | `async def` (non-blocking) |
| Input validation | Manual / none | Pydantic `BaseModel` — automatic |
| Route decorator | `@app.route('/path', methods=['POST'])` | `@app.post('/path')` |
| Auto API docs | None (manual Swagger needed) | Built-in at `/docs` and `/redoc` |
| Concurrent sensor streams | Blocking — requires threading | Native async — no threading needed |
| Persistent state | `global` variables + lock hacks | FastAPI `lifespan` + dependency injection |
| WebSocket (future) | `flask-socketio` extension | Built-in `WebSocket` support |
| Dockerfile CMD | `gunicorn app.app:app ...` | `uvicorn app.main:app --host 0.0.0.0 --port 7860` |
| Entry point file | `app/app.py` | `app/main.py` |
| Health check | `GET /health` returns dict | `GET /health` — same contract, async handler |
| Startup model loading | `@app.before_first_request` | `lifespan` context manager (FastAPI standard) |
 
**Why these changes matter for PumpSmart specifically:**
 
1. **1 Hz sensor polling** — Flask WSGI blocks thread per request. At 1 Hz continuous inflow from
   a SCADA system plus concurrent operator UI queries, Flask needs multiple workers or threading.
   FastAPI's async model handles both on a single uvicorn worker without blocking.
2. **Persistent CUSUM + rolling buffer** — `S_n` accumulator, `score_A_rolling_buffer`, and
   `zt_buffer` must survive across requests without race conditions. FastAPI's dependency injection
   (`Depends()`) provides clean shared state with proper async locking — no `global` variables.
3. **Pydantic sensor payload validation** — incoming 8-channel sensor windows are validated
   structurally at the API boundary before any ML code runs. Malformed payloads return 422
   immediately with field-level error detail. Flask required manual `request.json` parsing.
4. **Real-time UI** — FastAPI supports WebSocket endpoints natively for pushing live alert state
   to the dashboard without polling overhead. This is optional at M10 but required for Tier 3
   shadow deployment UI (T3-3 input distribution drift monitoring dashboard).
---
 
## Prerequisite Chain
 
```
M7 all_gates_pass = True
  -> M8 all_gates_pass = True
       -> M9 (physics tools) — COMPLETE LOCKED (24/24 PASS)
       -> M10 (FastAPI app) — requires M8 models + M7 models + M9 physics tools
       -> M12 adversarial validation — MUST run BEFORE M11 (T2-1)
       -> M11 (deployment) — requires M10 15/15 tests pass + M12 ≥80% detection rate
```
 
---
 
## M9 — Pump Selector + Household Advisor
 
**Status: COMPLETE LOCKED (24/24 PASS — 2026-05-10)**
 
Entry point for M10: `from src.module_09_pump_selector import pump_selector_dispatch`
 
Framework-agnostic pure Python — works identically under FastAPI, Flask, or CLI.
 
### Scope Boundary (NEVER VIOLATE)
 
```python
# Enforced in M10 FastAPI Route 3 + Route 4
if pump_type == 'household':
    return physics_advisory_only()   # NO ML inference
else:
    return ml_prediction()           # routes to M8 + M7 stack
```
 
### Routing (T2-3 — Physical Envelope — NEVER string-based)
 
```python
def route_pump(power_kW: float, head_m: float,
               stages: int, pressure_bar: float) -> str:
    is_industrial = (power_kW >= 30 and head_m >= 80
                     and stages >= 3 and pressure_bar >= 8)
    if is_industrial:
        return 'industrial_ml_pipeline'
    elif power_kW <= 5 and stages == 1 and pressure_bar <= 5:
        return 'household_physics_advisory'
    else:
        return 'OUT_OF_SCOPE'   # 5–30 kW commercial gap — explicit refusal
```
 
### Validated Physics (M9 locked outputs)
 
| Parameter | Value | Status |
|---|---|---|
| P_hydraulic | 55.181 kW | PASS (expect ~55.2) |
| Motor | 110 kW IEC | PASS |
| Ns | 26.41 (m³/min convention) | PASS (radial < 50) |
| Pump type | multistage_centrifugal | PASS |
| H/stage | 64.29 m | PASS |
| Water hammer ΔP | 30 bar → 70 bar transient | PASS |
| Cavitation flag | Fires at NPSHa < NPSHr + 0.5 m | PASS |
| Affinity laws | Q₂/Q₁=N₂/N₁, H₂/H₁=(N₂/N₁)² | PASS |
 
---
 
## M10 — FastAPI Application
 
**Status: IN DEVELOPMENT — UNBLOCKED**
 
### FastAPI Application Structure
 
```
app/
  main.py               <- FastAPI app instance, lifespan, startup model loading
  routers/
    anomaly.py          <- POST /api/anomaly_detect  (4-layer inference, score routing)
    classify.py         <- POST /api/classify_fault  (22-class XGBoost, causal chain)
    selector.py         <- POST /api/select_pump + GET /api/household
    acknowledge.py      <- POST /api/acknowledge     (CUSUM + z_t + rolling reset)
    validate.py         <- GET  /api/validate_model
    health.py           <- GET  /health
    physics.py          <- GET  /api/physics_context (static lookup, 22 classes)
    websocket.py        <- WS   /ws/live_alerts      (optional — Tier 3 shadow UI)
  runtime/
    cusum_state.py      <- CUSUM S_n on score_B — async-safe persistent state class
    rolling_state.py    <- score_A rolling buffer + adaptive threshold updater
    zt_buffer.py        <- z_t rolling buffer for TCN-AE streaming
    physics_context.py  <- static lookup loader from fault_rules_v3.json
    model_registry.py   <- all model loading in lifespan — single source of truth
  schemas/
    sensor_input.py     <- Pydantic BaseModel: SensorWindow(8 float channels × 50 steps)
    fault_output.py     <- Pydantic BaseModel: FaultPrediction(7 mandatory fields)
    selector_input.py   <- Pydantic BaseModel: PumpSpec(flow, head, density, temp, ...)
    household_input.py  <- Pydantic BaseModel: HouseholdSpec(usage_type, demand, ...)
  templates/
    index.html          <- main dashboard (4-state UI, CUSUM panel, physics context)
    household.html      <- household advisor UI
    selector.html       <- industrial selector UI
  static/
    style.css
    dashboard.js        <- async fetch() for continuous 1 Hz sensor polling
outputs/reports/module_10_fastapi_app_report.md
```
 
### FastAPI App Entry Point (`app/main.py`)
 
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routers import anomaly, classify, selector, acknowledge, validate, health, physics
from app.runtime.model_registry import load_all_models
from app.runtime.cusum_state import CUSUMState
from app.runtime.rolling_state import RollingState
from app.runtime.zt_buffer import ZTBuffer
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP — load all models once, store in app.state
    app.state.models = load_all_models()        # M4 LSTM-AE + TCN-AE + XGBoost
    app.state.cusum  = CUSUMState()             # CUSUM S_n — score_B ONLY (Invariant 19)
    app.state.rolling = RollingState()          # rolling score_A — Layer 4 (Invariant 19)
    app.state.zt_buf  = ZTBuffer()              # z_t streaming buffer — Layer 2 input
    yield
    # SHUTDOWN — clean up (optional)
 
app = FastAPI(
    title="PumpSmart v14.2 — Industrial Pump Health Monitor",
    version="5.0",
    lifespan=lifespan,
)
 
app.include_router(health.router)
app.include_router(anomaly.router,    prefix="/api")
app.include_router(classify.router,   prefix="/api")
app.include_router(selector.router,   prefix="/api")
app.include_router(acknowledge.router,prefix="/api")
app.include_router(validate.router,   prefix="/api")
app.include_router(physics.router,    prefix="/api")
 
app.mount("/static", StaticFiles(directory="app/static"), name="static")
```
 
### Pydantic Sensor Input Schema (`app/schemas/sensor_input.py`)
 
```python
from pydantic import BaseModel, Field, validator
from typing import List
 
class SensorWindow(BaseModel):
    """
    Single 50-step window of 8 normalised sensor channels.
    Order (LOCKED from M6B): Mot.SV, Pmp.SV, Mot.TV, Pmp.PV,
                              Temp.SV, Pres.SV, Pmp.TV, Mot.PV
    Values must be cluster-normalised (M3 config) before submission.
    Raw sensor values are NEVER fed directly — physics invariant.
    """
    window: List[List[float]] = Field(
        ...,
        description="50 timesteps × 8 channels, cluster-normalised",
    )
    pump_id: str = Field(default="pump_001")
    cluster: str = Field(default="steady_state",
                         description="startup|steady_state|high_load|cooldown")
 
    @validator('window')
    def check_shape(cls, v):
        assert len(v) == 50, f"Expected 50 timesteps, got {len(v)}"
        assert all(len(row) == 8 for row in v), "Each timestep must have 8 channels"
        return v
```
 
### Pydantic Fault Output Schema (`app/schemas/fault_output.py`)
 
```python
from pydantic import BaseModel
from typing import List, Optional, Dict
 
class FaultPrediction(BaseModel):
    """
    Mandatory 7-field output — NEVER reduce to fewer fields.
    All 7 fields required in every non-normal prediction.
    """
    fault_label: str                        # Field 1: specific class name
    confidence_pct: float                   # Field 2: M7 predict_proba max
    unknown_fault_flag: bool                # Field 2b: True if confidence < 70%
    probable_physical_condition: str        # Field 3: what is happening inside
    expected_sensor_behavior: str          # Field 4: trajectory if correct
    operational_risk_if_ignored: str       # Field 5: consequence timeline
    recommended_action: str                # Field 6: specific, sequenced steps
    model_limitation_disclaimer: str       # Field 7: MANDATORY — never omit
 
    # Additional M10 v5.0 fields
    alert_state: str                       # NORMAL|WATCH|WARN|DANGER
    score_A: float                         # TCN-AE severity (Layer 4 routing)
    score_B: float                         # TCN-AE drift slope (Layer 3 routing)
    score_C: float                         # TCN-AE chain transition (M7 routing)
    cusum_Sn: float                        # Current CUSUM accumulator value
    adaptive_threshold: float              # Current θ_t
    physics_context: Optional[Dict]        # from fault_rules_v3.json
    causal_chain: Optional[str]            # Group B labels only
    limitation_flags: List[str]            # per-alert known limitations
    ood_suspected: bool                    # Mahalanobis > tau_p99
```
 
### Route Definitions (8 routes)
 
```python
# Route 1 — POST /api/anomaly_detect
# Input:  SensorWindow (50-step × 8-channel normalised)
# Output: FaultPrediction (7-field mandatory)
# Process: M4 LSTM-AE → z_t → TCN-AE → score_A/B/C → L3 CUSUM → L4 adaptive θ
#          → OOD check → XGBoost classify → 7-field render
# INVARIANT 19: score_A → L4 ONLY | score_B → CUSUM ONLY | score_C → XGBoost ONLY
 
@router.post("/anomaly_detect", response_model=FaultPrediction)
async def anomaly_detect(window: SensorWindow, request: Request):
    ...
 
# Route 2 — POST /api/classify_fault
# Input:  SensorWindow
# Output: FaultPrediction (Stage 1/2/3 label, causal chain for Group B)
# Scope:  if pump_type == 'household': return physics_advisory_only()
 
@router.post("/classify_fault", response_model=FaultPrediction)
async def classify_fault(window: SensorWindow, request: Request):
    ...
 
# Route 3 — POST /api/select_pump
# Input:  PumpSpec (flow_rate_m3h, total_head_m, fluid_density, fluid_temp_c,
#                   suction_head_m, pipe_length_m, pipe_diameter_m)
# Output: industrial_pump_selector() output dict
# Process: M9 physics — NO ML inference
 
@router.post("/select_pump")
async def select_pump(spec: PumpSpec):
    ...
 
# Route 4 — GET /api/household
# Input:  HouseholdSpec (usage_type, daily_demand_lph, static_head_m,
#                        pipe_length_m, pipe_diameter_mm)
# Output: household_physics_advisory() dict + advisory_disclaimer always present
# Process: M9 physics ONLY — ZERO ML inference
# UI label: "Advisory guidance only — not a monitoring tool"
 
@router.get("/household")
async def household_advisory(spec: HouseholdSpec = Depends()):
    ...
 
# Route 5 — POST /api/acknowledge
# Input:  AcknowledgeRequest(channels: list|"all", pump_id: str, action_taken: str)
# Process: Reset CUSUM S_n, z_t buffer, rolling score_A buffer
#          Log with timestamp + action_taken text
# INVARIANT: CUSUM resets ONLY on confirmed maintenance — NEVER on threshold update
 
@router.post("/acknowledge")
async def acknowledge(ack: AcknowledgeRequest, request: Request):
    ...
 
# Route 6 — GET /api/validate_model
# Output: model hash verification against M9_selector_config.json
# Gate: all model files present + SHA-256 match
 
@router.get("/validate_model")
async def validate_model(request: Request):
    ...
 
# Route 7 — GET /health
# Output: {"status":"healthy","version":"5.0",
#          "models_loaded":["lstm_ae_l1","tcn_ae_l2","xgboost_22class","m9_physics"],
#          "tcn_ae_active":true, "cusum_active":true,
#          "zt_buffer_len": N, "adaptive_threshold": θ_t,
#          "commissioning_mode": false}
 
@router.get("/health")
async def health_check(request: Request):
    ...
 
# Route 8 — GET /api/physics_context?label={0-21}
# Output: static lookup from fault_rules_v3.json
#         {what, why, timeline, action, if_ignored, disclaimer}
# NOT ML inference — pure static dict lookup
 
@router.get("/physics_context")
async def physics_context(label: int, request: Request):
    ...
```
 
### Score Routing — Invariant 19 (ENFORCED IN M10 RUNTIME — NEVER CROSS)
 
| Score | Routes To | Never To |
|---|---|---|
| score_A | Layer 4 Rolling Baseline ONLY → `rolling_state.update(score_A)` | CUSUM, XGBoost |
| score_B | Layer 3 CUSUM ONLY → `cusum_state.update(score_B)` | Rolling Baseline, XGBoost |
| score_C | XGBoost M7 feature ONLY → `xgb_model.predict_proba(features)` | CUSUM, Rolling Baseline |
 
Cross-routing is an architecture violation. Enforced in `app/routers/anomaly.py`.
 
### Layer 3 — CUSUM Runtime State (`app/runtime/cusum_state.py`)
 
```python
import asyncio
 
class CUSUMState:
    """
    Thread-safe async CUSUM accumulator for score_B (drift slope).
    Invariant 19: score_B → CUSUM ONLY.
    Resets ONLY on confirmed maintenance (/api/acknowledge).
    NEVER resets on adaptive threshold update — C-25 Adaptive Threshold Paradox.
    Formula: S_n = max(0, S_{n-1} + (score_B_n - mu0_B) - k)
    """
    def __init__(self):
        self._Sn       = 0.0
        self._fired    = False
        self._lock     = asyncio.Lock()
 
    async def update(self, score_B: float, mu0_B: float,
                     k: float, H: float = 5.0) -> dict:
        async with self._lock:
            self._Sn = max(0.0, self._Sn + (score_B - mu0_B) - k)
            if self._Sn > H:
                self._fired = True
            return {"Sn": self._Sn, "fired": self._fired}
 
    async def reset(self):
        async with self._lock:
            self._Sn    = 0.0
            self._fired = False
```
 
### Layer 4 — Adaptive Threshold (`app/runtime/rolling_state.py`)
 
```python
import asyncio
import numpy as np
 
class RollingState:
    """
    score_A rolling buffer + adaptive threshold θ_t.
    Invariant 19: score_A → Rolling Baseline ONLY.
    θ_t = μ_rolling(6hr) + 3σ_rolling(6hr)
    Updates every inference call (50 seconds of pump data at 1 Hz).
    6-hour window = 432 calls. Warmup = 216 calls.
    Crosspoint guard: θ_t > 1.5 × θ_initial → LOCK + DRIFT ALERT.
    NEVER modifies CUSUM state — the two operate in parallel (C-25).
    """
    ROLLING_WINDOW = 432
    WARMUP_CALLS   = 216
    CROSSPOINT_GUARD = 1.5
 
    def __init__(self, theta_initial: float = 0.110058):
        self._buffer        = []
        self._theta_initial = theta_initial
        self._theta_t       = theta_initial
        self._locked        = False
        self._lock          = asyncio.Lock()
 
    async def update(self, score_A: float) -> dict:
        async with self._lock:
            self._buffer.append(score_A)
            if len(self._buffer) > self.ROLLING_WINDOW:
                self._buffer.pop(0)
            if len(self._buffer) >= self.WARMUP_CALLS and not self._locked:
                arr = np.array(self._buffer)
                new_theta = float(arr.mean() + 3 * arr.std())
                if new_theta > self.CROSSPOINT_GUARD * self._theta_initial:
                    self._locked  = True
                    self._theta_t = self._theta_initial * self.CROSSPOINT_GUARD
                    return {"theta_t": self._theta_t, "drift_alert": True,
                            "locked": True}
                self._theta_t = new_theta
            return {"theta_t": self._theta_t, "drift_alert": False,
                    "locked": self._locked}
 
    async def reset(self):
        async with self._lock:
            self._buffer  = []
            self._theta_t = self._theta_initial
            self._locked  = False
```
 
### Model Loading in Lifespan (`app/runtime/model_registry.py`)
 
```python
import torch
import xgboost as xgb
import json
from pathlib import Path
 
def load_all_models() -> dict:
    """
    Load all models at startup via FastAPI lifespan.
    ALL models loaded to CPU — NEVER .cuda() in deployment.
    """
    models = {}
 
    # M4 LSTM-AE Level 1 (FROZEN — do not retrain)
    from src.module_09_pump_selector import LSTMAutoencoder
    m4 = LSTMAutoencoder(seq_len=50)
    m4.load_state_dict(
        torch.load('models/lstm_ae_baseline_final.pth', map_location='cpu'))
    m4.eval()
    for p in m4.parameters():
        p.requires_grad_(False)
    models['lstm_ae_l1'] = m4
 
    # TCN-AE Level 2
    from src.module_09_pump_selector import TCNAutoencoder
    tcn = TCNAutoencoder()
    tcn.load_state_dict(
        torch.load('models/tcn_ae_level2_best.pth', map_location='cpu'))
    tcn.eval()
    for p in tcn.parameters():
        p.requires_grad_(False)
    models['tcn_ae_l2'] = tcn
 
    # XGBoost 22-class (CPU deploy version)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/M7_xgboost_classifier_cpu.json')
    models['xgboost_22class'] = xgb_model
 
    # M9 physics config
    with open('models/M9_selector_config.json') as f:
        models['m9_physics'] = json.load(f)
 
    # M8 threshold config
    with open('models/M8_threshold_config.json') as f:
        models['m8_config'] = json.load(f)
 
    # OOD detector config
    with open('models/M8p4_ood_config.json') as f:
        models['ood_config'] = json.load(f)
 
    # Fault rules (physics context lookup)
    with open('models/fault_rules_v3.json') as f:
        models['fault_rules'] = json.load(f)
 
    return models
```
 
### M10 Local Test Protocol (15 tests)
 
| # | Test | Input | Expected |
|---|---|---|---|
| 1 | Health check | GET /health | status=healthy, version=5.0, all 4 models loaded |
| 2 | Normal window | POST /api/anomaly_detect (normal data) | alert_state=NORMAL |
| 3 | Fault window | POST /api/anomaly_detect (high MAE) | alert_state=DANGER |
| 4 | Mild fault | POST /api/anomaly_detect (moderate MAE) | WATCH or WARN |
| 5 | z_t buffer Layer 2 | 6 consecutive windows | score_A/B/C non-null at window 6 |
| 6 | Compound fault (label 10) | POST /api/classify_fault | causal_chain visible, score_C top SHAP |
| 7 | Label 21 CUSUM | Repeated low-score_B windows | WATCH + advisory visible |
| 8 | CUSUM reset | POST /api/acknowledge {"channels":"all"} | Sn=0, zt_buffer_len=0 |
| 9 | Physics context | GET /api/physics_context?label=10 | what/why/timeline/action/if_ignored all non-empty |
| 10 | Industrial selector | POST /api/select_pump (nameplate params) | P_hyd≈55.2 kW, motor=110 kW |
| 11 | Household advisory | GET /api/household | advisory_disclaimer present, no ML fields |
| 12 | OOD detection | POST /api/anomaly_detect (out-of-distribution) | ood_suspected=True |
| 13 | Limitation flags | Any WATCH/WARN/DANGER | limitation_flags non-empty |
| 14 | 7-field completeness | Any fault prediction | all 7 mandatory fields present |
| 15 | Scope boundary | Household pump via /api/classify_fault | physics_advisory_only() response, no ML call |
 
### M10 Paste Text Keys
 
| Key | Value |
|---|---|
| M10_framework | FastAPI v5.0 |
| M10_routes_registered | [/health, /api/anomaly_detect, /api/classify_fault, /api/select_pump, /api/household, /api/acknowledge, /api/validate_model, /api/physics_context] |
| M10_health_check_response | healthy/error |
| M10_models_loaded_at_startup | [lstm_ae_l1, tcn_ae_l2, xgboost_22class, m9_physics] |
| M10_tcn_ae_active_at_startup | True/False |
| M10_normal_window_test | NORMAL/error |
| M10_fault_window_test | DANGER/error |
| M10_mild_fault_test | WATCH/WARN/error |
| M10_zt_buffer_layer2_test | score_A_B_C_non_null_at_window_6/error |
| M10_compound_fault_test | causal_chain_visible/score_C_top_SHAP/error |
| M10_label21_cusum_test | WATCH+advisory_visible/error |
| M10_cusum_reset_test | reset_confirmed/error |
| M10_cusum_active_at_startup | True/False |
| M10_adaptive_threshold_active | True/False |
| M10_physics_context_route_test | 22_labels_returned/error |
| M10_limitation_flags_in_response | True/False |
| M10_commissioning_mode_documented | True/False |
| M10_household_scope_enforced | True/False |
| M10_pydantic_validation_active | True/False |
| M10_async_handlers_confirmed | True/False |
| M10_local_tests_pass | [X/15] |
| Status_for_M11 | READY/BLOCKED |
 
---
 
## M11 — Docker + Hugging Face Deployment
 
**Status: NOT STARTED — requires M10 15/15 tests pass + M12 ≥80% detection rate**
 
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
 
EXPOSE 7860
 
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1
 
# FastAPI via uvicorn — replaces gunicorn/WSGI from v4.0
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "120"]
```
 
**Why `--workers 1`:** PyTorch models are not fork-safe. Multiple uvicorn workers would each load
their own copy of the models into RAM, exceeding Hugging Face Spaces free-tier memory (16 GB).
One worker with async handlers provides adequate concurrency for shadow-mode deployment.
 
### requirements.txt (Deployment — CPU Only)
 
```
# Framework — v4.0 Flask replaced by FastAPI + uvicorn
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.0           # FastAPI v0.100+ requires Pydantic v2
httpx>=0.27.0           # async HTTP client (used in tests)
jinja2>=3.1.0           # templating (same as Flask)
python-multipart>=0.0.9 # FastAPI form data support
 
# ML stack — UNCHANGED from v4.0
torch==2.6.0+cpu
xgboost>=2.0
scikit-learn>=1.3
numpy>=1.24
pandas>=2.0
shap>=0.44
scipy>=1.11
 
# REMOVED from v4.0
# flask>=3.0       <- REPLACED by fastapi
# gunicorn>=21.0   <- REPLACED by uvicorn
```
 
**Delta from v4.0:** Remove `flask`, `gunicorn`. Add `fastapi`, `uvicorn[standard]`,
`pydantic>=2.0`, `httpx`, `python-multipart`. All ML dependencies unchanged.
 
### Model Loading Rules (NON-NEGOTIABLE — unchanged from v4.0)
 
```python
# M4 LSTM-AE Level 1
lstm_ae_l1.load_state_dict(
    torch.load('models/lstm_ae_baseline_final.pth', map_location='cpu'))
 
# TCN-AE Level 2
tcn_ae_l2.load_state_dict(
    torch.load('models/tcn_ae_level2_best.pth', map_location='cpu'))
 
# NEVER call .cuda() or .to('cuda') on any model in deployment code.
# XGBoost CPU version: models/M7_xgboost_classifier_cpu.json
# CUSUM S_n, z_t buffer, rolling baseline: in-memory Python state — NOT model weights
# Persist across API calls within one container lifecycle.
# Reset on container restart OR POST /api/acknowledge.
```
 
### GitHub Actions CI/CD Pipeline (`.github/workflows/deploy.yml`)
 
```yaml
name: Deploy PumpSmart to Hugging Face Spaces
 
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
 
### Hugging Face Spaces Configuration (`README.md` front matter)
 
```yaml
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
 
### Deployment Validation Checklist (12 Checks)
 
| Check | Description | Expected |
|---|---|---|
| 1 | Docker build locally | No errors |
| 2 | `docker run -p 7860:7860 pumpsmart` | Container starts |
| 3 | GET http://localhost:7860/health | status=healthy, version=5.0, models_loaded=[lstm_ae_l1,tcn_ae_l2,xgboost_22class,m9_physics], cusum_active=true, tcn_ae_active=true |
| 4 | POST /api/anomaly_detect | Valid FaultPrediction response; score_A/B/C present; Pydantic 422 on bad input |
| 5 | POST /api/classify_fault | Valid Stage 3 fault (label 0–21) |
| 6 | POST /api/classify_fault compound (label 10) | causal_chain = "seal_failure → cavitation"; score_C top SHAP |
| 7 | POST /api/acknowledge | cusum_state Sn=0, zt_buffer_len=0 |
| 8 | GET /api/physics_context?label=10 | what/why/timeline/action/if_ignored/disclaimer all non-empty |
| 9 | Image size | < 2 GB (HF free tier limit) |
| 10 | Startup time | < 60 s (within HEALTHCHECK start-period) |
| 11 | Push to Hugging Face Spaces | Space builds successfully; uvicorn starts |
| 12 | HF Space URL /health | status=healthy, version=5.0; GitHub Actions workflow passes on push to main |
 
### M11 Paste Text Keys
 
| Key | Value |
|---|---|
| M11_docker_build_status | SUCCESS/FAILED |
| M11_server | uvicorn (ASGI) |
| M11_container_startup_time_s | [seconds — gate < 60] |
| M11_image_size_mb | [MB — gate < 2000] |
| M11_health_check_local | healthy/error |
| M11_tcn_ae_active_in_container | True/False |
| M11_cusum_active_in_container | True/False |
| M11_physics_context_route_test | PASS/FAIL |
| M11_pydantic_422_on_bad_input | PASS/FAIL |
| M11_hf_deployment_url | [URL] |
| M11_hf_health_check | healthy/error |
| M11_github_actions_status | PASS/FAIL |
| M11_compound_fault_route_test | PASS/FAIL |
| M11_all_checks_pass | True/False |
| Status_for_M12 | READY/BLOCKED |
 
---
 
## Module Dependency Summary
 
```
M7 XGBoost (22-class, ~35 features, M6B_feature_matrix.csv)
M8 Level 1 LSTM-AE + Level 2 TCN-AE + Layer 3 CUSUM + Layer 4 Rolling Baseline
M9 Physics Tools [COMPLETE LOCKED]
    |
    v
M10 FastAPI App
    score routing (Invariant 19)
    z_t rolling buffer (async-safe)
    adaptive threshold θ_t (async-safe)
    physics context lookup
    Pydantic input validation
    commissioning mode
    |
    v
M12 Adversarial Validation  ← MUST precede M11 (T2-1)
    |
    v
M11 Docker + Hugging Face (uvicorn ASGI)
    |
    v
M12.5 Post-deployment validation
```
 
### Sequencing Law
 
| Gate | Unlocks |
|---|---|
| M7 gates pass | M8 starts |
| M8 gates pass | M9 finalised + M10 starts |
| M10 15/15 tests pass | M12 starts |
| M12 detection rate ≥80% | M11 starts |
| M11 deployment OK | T3 shadow operation |
 
---
 
## Tier 2 — Required before M11
 
### T2-1 — M12 before M11 (procedural reorder)
 
Pathway: M9 → M10 → **M12** → M11. Deploying M11 before M12 publishes an unvalidated model.
Gate: `T2-1_M12_before_M11_PASS`. Block M11 if M12 macro detection rate < 80%.
 
### T2-2 — CPU inference benchmark
 
Measure full inference path in Docker `--cpus=1 --memory=2g`:
M4 forward + TCN-AE forward + OOD check + XGBoost predict_proba + CUSUM update + 7-field render.
**Total must be < 5 seconds.** If not, ONNX export of LSTM-AE and TCN-AE required.
Script: `module_08p6_cpu_inference_benchmark.py`.
 
**Note:** FastAPI's async handlers mean the 5-second CPU inference budget applies per-request,
not per-concurrent-user. Multiple operators can poll simultaneously without queuing.
 
### T2-3 — Physical-parameter routing (COMPLETE — M9 locked)
 
Physical envelope routing implemented and gate-tested (24/24 PASS). T2-3 closed.
 
### T2-4 — Baseline LSTM-AE comparison
 
Train vanilla LSTM-AE with single fixed threshold. Measure on M12 adversarial set.
Report: detection rate Group A / B / Label 21 / Group C / FPR — four-column comparison table.
Publishable headline: Label 21 detection 0% (vanilla) vs >60% (PumpSmart).
 
### T2-5 — Threshold sensitivity audit
 
Sweep: q=0.110058 (M4), H=5.0 (CUSUM), tau_p99 (OOD), θ_initial (L4).
Include FPR vs threshold sweep for v2 slope-continuity gate on healthy CIRA windows.
Report per-gate TPR/FPR trade-off table. Document sensitivity bounds.
 
### T2-6 — Configuration drift hash registry
 
SHA-256 lock all config files: `M8_threshold_config.json`, `M9_selector_config.json`,
`M8p4_ood_config.json`, `fault_rules_v3.json`. Alert on mismatch at startup.
 
### T2-7 — Cluster assignment hysteresis
 
Prevent sawtooth false alerts at startup/steady-state boundary.
Implement hysteresis: cluster assignment requires N consecutive windows in new cluster before switching.
 
### T2-8 — Operator UI honesty controls
 
Confidence displayed as range not point. Disclaimers non-dismissible on first view.
Advisory-only label persistent on household path.
 
### T2-9 — Group B v1↔v2 cross-evaluation
 
Report per-class F1 deltas in both directions (not just macro). Required for publication-grade
artifact independence claim.
 
---
 
## Document Revision History
 
| Version | Date | Changes |
|---|---|---|
| v1.0 | 2026-04-12 | Initial creation — split from `module_pathway_M1_to_M12_v10.md` |
| v2.0 | 2026-04-12 | Bias-audit cascade: multi-label classify route, Stage 1/2/3 API schema |
| v3.0 | 2026-04-16 | Architecture v14.0: CUSUM runtime state + rolling baseline + /api/acknowledge + 22-class XGBoost |
| v4.0 | 2026-04-21 | Architecture v14.2: TCN-AE Level 2, z_t rolling buffer, score_A/B/C routing (Invariant 19), adaptive threshold θ_t, /api/physics_context, limitation_flags |
| v5.0 | 2026-05-10 | **FastAPI migration.** Flask → FastAPI throughout. gunicorn → uvicorn. Sync → async handlers. global state → lifespan + dependency injection. Manual validation → Pydantic BaseModel. requirements.txt updated. Dockerfile CMD updated. All route signatures updated. Async-safe CUSUM and rolling state classes. New: schemas/ directory, model_registry.py lifespan loader. |
 
---
 
*PumpSmart v14.2 | Architecture v5.0 | FastAPI + uvicorn + Pydantic v2*
*Deployment target: Hugging Face Spaces Docker SDK, port 7860*