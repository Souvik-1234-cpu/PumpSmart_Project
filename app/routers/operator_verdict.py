# app/routers/operator_verdict.py
# POST /api/operator_verdict
# v5.0-A: ONLY active-learning write point.
# Called from Predictions tab Correct/Incorrect/Unsure buttons ONLY.
# /api/acknowledge does NOT call this.
# Pushes 27-column row to HF Datasets API (v5.0-C).
# =============================================================================

import uuid
import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request, BackgroundTasks

from app.schemas.sensor_input import OperatorVerdict

router = APIRouter(tags=["active_learning"])

# HF Datasets API persistence (v5.0-C)
# Falls back to local JSONL if HF not configured (local dev)
HF_DATASET_REPO = "pumpsmart-active-learning"
LOCAL_FALLBACK   = Path("outputs") / "active_learning_queue.jsonl"


async def _push_to_hf(row: dict) -> bool:
    """Push one 27-column row to HF Datasets API."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        content = json.dumps(row, ensure_ascii=False).encode("utf-8")
        path_in_repo = f"data/{row['timestamp_utc'][:10]}/{row['prediction_id']}.json"
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=path_in_repo,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
        )
        return True
    except Exception as e:
        # Fallback: append to local JSONL
        LOCAL_FALLBACK.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCAL_FALLBACK, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return False


@router.post("/operator_verdict", summary="Record operator verdict (active-learning write point)")
async def operator_verdict(
    payload: OperatorVerdict,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    v5.0-A: ONLY active-learning data write point.
    Called from Predictions tab after physical investigation.
    NOT called by /api/acknowledge (which is operational reset only).

    Builds 27-column schema row v1.0 and pushes to HF Datasets API.
    data_source = SHADOW_REAL (production shadow operation).
    """
    models  = request.app.state.models
    ts_now  = datetime.utcnow().isoformat() + "Z"

    # Retrieve latest prediction state for context columns
    cusum_st   = await request.app.state.cusum.get_state()
    rolling_st = await request.app.state.rolling.get_state()

    # Build 27-column row (schema v1.0 — LOCKED)
    row = {
        # Identity
        "timestamp_utc"            : ts_now,
        "pump_id"                  : request.headers.get("X-Pump-ID", "PUMP-0032"),
        "prediction_id"            : payload.prediction_id,
        "cluster_id"               : None,          # filled by frontend if available
        # Prediction
        "predicted_label_int"      : None,          # from prediction_id lookup (Phase 4)
        "predicted_label_name"     : None,
        "confidence_pct"           : None,
        # Scores (Invariant 19)
        "score_A"                  : rolling_st.get("latest_score_A"),
        "score_B"                  : None,
        "score_C"                  : None,
        "cusum_s_n"                : cusum_st["cusum_Sn"],
        "theta_t"                  : rolling_st["theta_t"],
        "alert_state"              : cusum_st["cusum_alert"],
        # M8p6
        "m8p6_sensor_flag"         : False,         # populated from cached prediction
        "m8p6_flagged_channels"    : "",
        # OOD
        "mahal_dist"               : None,
        "ood_flag"                 : False,
        # Raw data (populated from cached prediction in Phase 4)
        "raw_sensor_window"        : None,
        "top_3_shap_features"      : None,
        # Operator verdict
        "operator_verdict"         : payload.verdict,
        "operator_correct_label"   : payload.operator_correct_label,
        "verdict_timestamp_utc"    : ts_now,
        "time_to_verdict_seconds"  : None,          # calculated in Phase 4 with prediction_id lookup
        "physical_inspection_done" : payload.physical_inspection_done,
        "inspection_notes"         : payload.inspection_notes,
        # Source
        "data_source"              : "SHADOW_REAL",
        "consent_granted_by"       : payload.consent_granted_by or payload.operator_id,
    }

    # Push in background (non-blocking)
    background_tasks.add_task(_push_to_hf, row)

    return {
        "recorded"           : True,
        "prediction_id"      : payload.prediction_id,
        "verdict"            : payload.verdict,
        "timestamp_utc"      : ts_now,
        "data_source"        : "SHADOW_REAL",
        "persistence"        : "huggingface_datasets_api",
        "active_learning_write": True,   # explicitly documented — v5.0-A
        "note": (
            "Verdict recorded. Thank you — your feedback contributes to model improvement. "
            "Prediction accuracy builds the synthetic-to-real bridge (C-26)."
        ),
    }
