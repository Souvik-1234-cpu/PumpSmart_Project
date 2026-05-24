# =============================================================================
# app/routers/history.py
# GET  /api/sensor_history     — last_n_seconds range, adaptive downsample
# GET  /api/sensor_history/state — lightweight buffer status
# POST /api/sensor_history/reset — admin hard wipe (kept off /api/acknowledge)
#
# Multi-client safety: SensorHistoryBuffer is a single instance shared across
# all clients via app.state. All clients see identical history.
# =============================================================================
from typing import Literal
from fastapi import APIRouter, Request, Query

router = APIRouter(tags=["history"])


@router.get("/sensor_history", summary="Sensor + inference history for plotting")
async def sensor_history(
    request: Request,
    last_n_seconds: int = Query(3600, ge=1, le=86_400,
        description="Look-back window. Default 1 h. Max 24 h (buffer capacity)."),
    downsample: Literal["adaptive", "full", "lttb", "stride"] = Query(
        "adaptive",
        description="adaptive = last 5 min full-res + LTTB on older (DS-C)"),
    max_points: int = Query(500, ge=10, le=5000,
        description="Upper bound on returned points per array"),
):
    """
    Returns timestamps, 8-channel sensor history, and derived metrics
    (score_A, score_B, cusum_Sn, theta_t, alert_state, label_int, confidence_pct).

    Recommended polling intervals (frontend):
      - Dashboard live chart  : 1 s   (last_n_seconds=300, max_points=300)
      - Analytics tab         : 5 s   (last_n_seconds=3600, max_points=500)
      - History tab on-demand : button click (last_n_seconds up to 86400)
    """
    data = await request.app.state.history.get_range(
        last_n_seconds = last_n_seconds,
        downsample     = downsample,
        max_points     = max_points,
    )
    return data


@router.get("/sensor_history/state", summary="Buffer fill status")
async def sensor_history_state(request: Request):
    return await request.app.state.history.get_state()


@router.post("/sensor_history/reset", summary="Admin: wipe sensor history buffer")
async def sensor_history_reset(request: Request):
    """
    Hard wipe of the history ring buffer. NOT called by /api/acknowledge
    (operators must retain pre-acknowledge history for forensic review).
    """
    return await request.app.state.history.reset()
