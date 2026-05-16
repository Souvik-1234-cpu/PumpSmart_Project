# =============================================================================
# PumpSmart v14.2 — Module 10 Phase 1
# app/main.py — FastAPI entry point
# =============================================================================
# Architecture: v14.2 + M8p6 | FastAPI + uvicorn (migrated from Flask v5.0)
# Deployment target: Hugging Face Spaces Docker, port 7860
#
# Invariant 19 (CRITICAL — enforced in runtime state classes, NOT here):
#   score_A  → Layer 4 RollingState ONLY
#   score_B  → Layer 3 CUSUMState  ONLY
#   score_C  → M7 XGBoost         ONLY
#
# Scope guard (enforced in selector router):
#   if pump_type == 'household': physics_advisory_only()
#   else:                        ml_prediction()
# =============================================================================

from app.routers import operator_verdict          # add this import
from contextlib import asynccontextmanager
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import (
    health,
    anomaly,
    classify,
    selector,
    acknowledge,
    validate,
    physics,
)
from app.runtime.model_registry import load_all_models
from app.runtime.cusum_state   import CUSUMState
from app.runtime.rolling_state import RollingState
from app.runtime.zt_buffer     import ZTBuffer

SCRIPT_NAME  = "module_10_fastapi_app"
ARCH_VERSION = "v14.2"
APP_VERSION  = "5.0"
START_TIME   = datetime.utcnow()


def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# LIFESPAN — model loading + persistent state initialisation
# All models loaded ONCE at startup and stored in app.state.
# If any required artifact is missing → hard failure, no silent fallback.
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────────────────────
    log("=== PumpSmart v14.2 startup initiated ===")
    log(f"Architecture: {ARCH_VERSION} | App version: {APP_VERSION}")

    # Load all ML models + configs (raises FileNotFoundError if any missing)
    log("Loading model artifacts via model_registry ...")
    app.state.models = load_all_models()
    log(f"  M4 LSTM-AE        : loaded  (threshold q={app.state.models['m4_threshold']:.6f})")
    log(f"  M8 TCN-AE         : loaded")
    log(f"  M7 XGBoost        : loaded  ({app.state.models['xgb_n_classes']} classes)")
    log(f"  M8p4 OOD config   : loaded  (tau_p99={app.state.models['ood_tau_p99']:.4f})")
    log(f"  M8p6 sensor cfg   : loaded  ({app.state.models['m8p6_n_channels']} channels)")
    log(f"  fault_rules_v3    : loaded  ({app.state.models['n_fault_labels']} labels)")
    log(f"  M3 norm config    : loaded")
    log(f"  M2 cluster bounds : loaded")

    # Initialise persistent runtime state (async-safe, survives across requests)
    log("Initialising runtime state ...")

    # L3 CUSUM — receives score_B ONLY (Invariant 19)
    app.state.cusum = CUSUMState(
        H     = app.state.models['cusum_H'],       # alarm threshold = 5.0
        k     = app.state.models['cusum_k'],       # reference value (slack)
        lam   = app.state.models['cusum_lambda'],  # decay λ = 5.73e-05 (7-day half-life)
    )
    log(f"  CUSUMState        : H={app.state.cusum.H}, λ={app.state.cusum.lam:.2e}")

    # L4 Rolling baseline — receives score_A ONLY (Invariant 19)
    app.state.rolling = RollingState(
        window_size   = app.state.models['rolling_window_size'],   # 6-hr rolling window
        theta_initial = app.state.models['theta_initial'],         # locked θ_initial
        lock_factor   = 1.5,    # crosspoint guard: θ_t > 1.5×θ_initial → DRIFT ALERT
    )
    log(f"  RollingState      : window={app.state.rolling.window_size} steps, "
        f"θ_initial={app.state.rolling.theta_initial:.6f}")

    # z_t buffer — feeds L2 TCN-AE (63-window rolling buffer)
    app.state.zt_buf = ZTBuffer(
        max_len = app.state.models['zt_buffer_len'],   # 63 consecutive z_t vectors
        z_dim   = app.state.models['zt_dim'],          # L1 encoder hidden dim
    )
    log(f"  ZTBuffer          : max_len={app.state.zt_buf.max_len}, z_dim={app.state.zt_buf.z_dim}")

    # Commissioning flag — set True on first deploy; set False after 48-hr baseline
    app.state.commissioning_mode = False
    app.state.monitoring_start   = START_TIME

    log("=== PumpSmart v14.2 startup COMPLETE — all systems ready ===")

    yield   # application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    log("=== PumpSmart v14.2 shutdown ===")


# =============================================================================
# APP INSTANCE
# =============================================================================
app = FastAPI(
    title       = "PumpSmart v14.2 — Industrial Pump Health Monitor",
    description = (
        "4-layer hybrid fault detection: LSTM-AE (L1) + TCN-AE (L2) + "
        "CUSUM (L3) + Adaptive Threshold (L4). "
        "22-class XGBoost fault classifier (M7). "
        "Asset: 110 kW | 7-stage | 40 bar | 2980 RPM | 45 m³/h. "
        "ADVISORY ONLY — predictions must be verified physically by a "
        "qualified engineer before any maintenance action. "
        "Trained on CIRA-anchored physics-synthetic data. "
        "Expected real-world F1: 0.65–0.85 (C-26)."
    ),
    version     = APP_VERSION,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.include_router(operator_verdict.router, prefix="/api")   # add after other routers
# =============================================================================
# ROUTER REGISTRATION — all 8 routes
# =============================================================================
# Route 1  — GET  /health
app.include_router(health.router)

# Routes 2–8 — all under /api prefix
app.include_router(anomaly.router,     prefix="/api")   # POST /api/anomaly_detect
app.include_router(classify.router,    prefix="/api")   # POST /api/classify_fault
app.include_router(selector.router,    prefix="/api")   # POST /api/select_pump
                                                        # GET  /api/household
app.include_router(acknowledge.router, prefix="/api")   # POST /api/acknowledge
app.include_router(validate.router,    prefix="/api")   # GET  /api/validate_model
app.include_router(physics.router,     prefix="/api")   # GET  /api/physics_context

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {
        "request": request, "arch_version": ARCH_VERSION
    })

@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, "arch_version": ARCH_VERSION,
        "app_version": APP_VERSION, "api_base": "",
    })
# =============================================================================
# STATIC FILES + TEMPLATES
# =============================================================================
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# =============================================================================
# ROOT ROUTE — serves main React dashboard via Jinja2
# =============================================================================
from fastapi import Request
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    """Serve main dashboard. React JSX rendered client-side."""
    return templates.TemplateResponse("index.html", {
        "request"      : request,
        "arch_version" : ARCH_VERSION,
        "app_version"  : APP_VERSION,
    })


@app.get("/household", response_class=HTMLResponse, include_in_schema=False)
async def household_ui(request: Request):
    """Serve household advisor UI (physics-only, no ML)."""
    return templates.TemplateResponse("household.html", {
        "request"      : request,
        "arch_version" : ARCH_VERSION,
    })
