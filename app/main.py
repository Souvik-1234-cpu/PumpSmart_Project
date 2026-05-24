# =============================================================================
# app/main.py
# PumpSmart v14.2 — FastAPI application entry point
# Version 5.1 — adds SensorHistoryBuffer (M10 Phase 2.5) for multi-client
# server-side history retention. Required before M12 adversarial validation.
# =============================================================================
from contextlib import asynccontextmanager
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import (
    health, anomaly, classify, selector,
    acknowledge, validate, physics, operator_verdict,
    history,                       # NEW — Phase 2.5
)
from app.runtime.model_registry   import load_all_models
from app.runtime.cusum_state      import CUSUMState
from app.runtime.rolling_state    import RollingState
from app.runtime.zt_buffer        import ZTBuffer
from app.runtime.sensor_history   import SensorHistoryBuffer   # NEW

ARCH_VERSION = "v14.2"
APP_VERSION  = "5.1"
START_TIME   = datetime.utcnow()


def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("=== PumpSmart v14.2 startup initiated ===")
    app.state.models = load_all_models()
    log(f"  M4 LSTM-AE    : loaded (q={app.state.models['m4_threshold']:.6f})")
    log(f"  M8 TCN-AE     : loaded")
    log(f"  M7 XGBoost    : loaded ({app.state.models['xgb_n_classes']} classes)")
    log(f"  M8p4 OOD      : loaded (tau_p99={app.state.models['ood_tau_p99']:.4f})")
    log(f"  M8p6 sensor   : loaded ({app.state.models['m8p6_n_channels']} channels)")
    log(f"  fault_rules   : loaded ({app.state.models['n_fault_labels']} labels)")
    log(f"  M3 norm + M2 bounds : loaded")

    app.state.cusum = CUSUMState(
        H   = app.state.models['cusum_H'],
        k   = app.state.models['cusum_k'],
        lam = app.state.models['cusum_lambda'],
    )
    app.state.rolling = RollingState(
        window_size   = app.state.models['rolling_window_size'],
        theta_initial = app.state.models['theta_initial'],
        lock_factor   = 1.5,
    )
    app.state.zt_buf = ZTBuffer(
        max_len = app.state.models['zt_buffer_len'],
        z_dim   = app.state.models['zt_dim'],
    )
    app.state.history = SensorHistoryBuffer(max_len=86_400)   # 24 h @ 1 Hz
    log(f"  sensor history: initialised (capacity 86,400 @ 1 Hz = 24 h)")

    app.state.commissioning_mode = False
    app.state.monitoring_start   = START_TIME
    log("=== PumpSmart v14.2 startup COMPLETE ===")
    yield
    log("=== PumpSmart v14.2 shutdown ===")


app = FastAPI(
    title    = "PumpSmart v14.2",
    version  = APP_VERSION,
    lifespan = lifespan,
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(anomaly.router,          prefix="/api")
app.include_router(classify.router,         prefix="/api")
app.include_router(selector.router,         prefix="/api")
app.include_router(acknowledge.router,      prefix="/api")
app.include_router(validate.router,         prefix="/api")
app.include_router(physics.router,          prefix="/api")
app.include_router(operator_verdict.router, prefix="/api")
app.include_router(history.router,          prefix="/api")    # NEW

# ── Static + templates ──────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(request=request, name="landing.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/household", response_class=HTMLResponse)
async def household(request: Request):
    return templates.TemplateResponse(request=request, name="household.html")
