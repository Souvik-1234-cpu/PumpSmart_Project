# app/routers/health.py — GET /health
from fastapi import APIRouter, Request
from datetime import datetime, timezone

router = APIRouter(tags=["health"])

@router.get("/health", summary="System health + runtime state")
async def health_check(request: Request):
    """
    Polled every 30 seconds by the dashboard.
    Returns model load status, CUSUM S_n, theta_t, z_t buffer fill,
    sensor count, and commissioning mode flag.
    """
    models   = request.app.state.models
    cusum_st = await request.app.state.cusum.get_state()
    roll_st  = await request.app.state.rolling.get_state()
    zt_st    = await request.app.state.zt_buf.get_state()

    uptime_s = (
        datetime.now(timezone.utc) - request.app.state.monitoring_start.replace(tzinfo=timezone.utc)
    ).total_seconds()

    return {
        "status"            : "healthy",
        "arch_version"      : "v14.2",
        "app_version"       : "5.0",
        "uptime_seconds"    : round(uptime_s),
        "commissioning_mode": request.app.state.commissioning_mode,
        "models_loaded"     : {
            "m4_lstm_ae"  : models["m4_model"] is not None,
            "m8_tcn_ae"   : models["m8_model"] is not None,
            "m7_xgboost"  : models["xgb_model"] is not None,
            "fault_rules" : len(models["label_map"]) == 22,
        },
        "cusum_state"       : cusum_st,
        "rolling_state"     : roll_st,
        "zt_buffer_state"   : zt_st,
        "m4_threshold_locked": round(models["m4_threshold"], 6),
        "timestamp_utc"     : datetime.utcnow().isoformat() + "Z",
    }
