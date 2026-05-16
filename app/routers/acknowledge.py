# app/routers/acknowledge.py — POST /api/acknowledge
# Operational reset ONLY (v5.0-A).
# Does NOT write to active-learning data store.
# Resets: CUSUM S_n, z_t buffer, rolling baseline.
# =============================================================================

from datetime import datetime
from fastapi import APIRouter, Request
from app.schemas.sensor_input import AcknowledgeRequest

router = APIRouter(tags=["operations"])

@router.post("/acknowledge", summary="Operational alarm reset after confirmed maintenance")
async def acknowledge(payload: AcknowledgeRequest, request: Request):
    """
    Resets CUSUM S_n, z_t buffer, and rolling baseline.
    Called when operator has physically investigated and confirmed maintenance.

    v5.0-A: This endpoint does NOT write to the active-learning data store.
    Operator prediction verdict (Correct/Incorrect/Unsure) is a SEPARATE
    action submitted from the Predictions tab — see POST /api/operator_verdict.

    C-25: L4 adaptive threshold updates must never call this endpoint.
    """
    ts = datetime.utcnow().isoformat() + "Z"

    cusum_after   = await request.app.state.cusum.reset(reason="maintenance_acknowledged")
    rolling_after = await request.app.state.rolling.reset()
    await request.app.state.zt_buf.reset()

    return {
        "acknowledged"     : True,
        "timestamp_utc"    : ts,
        "pump_id"          : payload.pump_id,
        "operator_id"      : payload.operator_id,
        "action_taken"     : payload.action_taken,
        "cusum_after_reset": cusum_after,
        "rolling_after_reset": rolling_after,
        "active_learning_write": False,   # explicitly documented — v5.0-A
        "note": (
            "Alarm state reset. To record your prediction verdict for model improvement, "
            "use the Predictions tab (Correct / Incorrect / Unsure)."
        ),
    }
