# app/routers/classify.py — POST /api/classify_fault
import uuid
from datetime import datetime
from fastapi import APIRouter, Request
from app.schemas.sensor_input import SensorWindow
from app.schemas.fault_output import MODEL_DISCLAIMER_TEXT

router = APIRouter(tags=["inference"])

GROUP_B_LABELS = {7, 8, 9, 10, 11, 12}   # compound fault chains

@router.post("/classify_fault", summary="22-class XGBoost fault classification")
async def classify_fault(payload: SensorWindow, request: Request):
    """
    Delegates to anomaly_detect inference pipeline.
    Additionally returns causal_chain description for Group B labels.
    Phase 1: placeholder response with correct schema.
    """
    models     = request.app.state.models
    label_int  = 0
    label_name = models["label_map"].get(label_int, "normal")
    conf_pct   = 98.2
    phys       = models["physics_ctx"].get(str(label_int), {})

    causal_chain = None
    if label_int in GROUP_B_LABELS:
        rules = models["fault_rules"]
        causal_chain = rules.get("compound_chains", {}).get(str(label_int), {}).get(
            "description", "Compound chain — verify causal direction physically."
        )

    return {
        "prediction_id"              : str(uuid.uuid4()),
        "fault_label"                : label_name,
        "label_int"                  : label_int,
        "confidence_pct"             : conf_pct,
        "unknown_fault_flag"         : conf_pct < 70.0,
        "causal_chain"               : causal_chain,
        "probable_physical_condition": phys.get("probable_condition", "[Phase 1]"),
        "expected_sensor_behavior"   : phys.get("expected_sensor_behaviour", "[Phase 1]"),
        "operational_risk_if_ignored": phys.get("risk_if_ignored", "None."),
        "recommended_action"         : phys.get("recommended_action", "Routine monitoring."),
        "model_limitation_disclaimer": MODEL_DISCLAIMER_TEXT,
        "timestamp_utc"              : datetime.utcnow().isoformat() + "Z",
        "pump_id"                    : payload.pump_id,
    }
