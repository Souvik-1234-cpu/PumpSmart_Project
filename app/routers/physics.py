# app/routers/physics.py — GET /api/physics_context
from fastapi import APIRouter, Request, Query, HTTPException

router = APIRouter(tags=["reference"])

@router.get("/physics_context",
            summary="Static physics context lookup for fault labels 0–21")
async def physics_context(
    request: Request,
    label  : int = Query(..., ge=0, le=23),
):
    ctx   = request.app.state.models["physics_ctx"].get("labels", {})
    label_map = request.app.state.models["label_map"]
    entry     = ctx.get(str(label))
    if entry is None:
        raise HTTPException(status_code=404,
                            detail=f"Label {label} not found. Valid range: 0–21.")
    return {
        "label_int"                 : label,
        "fault_name"                : label_map.get(label, "unknown"),
        "probable_condition"        : entry.get("probable_condition", ""),
        "expected_sensor_behaviour" : entry.get("expected_sensor_behaviour", ""),
        "risk_if_ignored"           : entry.get("risk_if_ignored", ""),
        "recommended_action"        : entry.get("recommended_action", ""),
        "source"                    : "M6B_physics_context_strings.json (locked)",
    }
