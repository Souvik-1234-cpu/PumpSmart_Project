# =============================================================================
# app/schemas/fault_output.py
# Pydantic output schema — 7-field mandatory output (NEVER reduce).
# C-26 disclaimer is a required field, never omit.
# =============================================================================

from pydantic import BaseModel
from typing import List, Optional, Dict

# Locked disclaimer text (C-26 — never alter)
MODEL_DISCLAIMER_TEXT = (
    "Trained on CIRA-anchored physics-synthetic data for 110 kW 7-stage centrifugal pump "
    "at 2980 RPM, 40 bar, 45 m³/h. Advisory only — verify predictions physically by a "
    "qualified mechanical or instrumentation engineer before any maintenance action. "
    "5-fold sequence-stratified CV: macro F1 = 0.9965 ± 0.0005 (synthetic domain). "
    "Real-world F1 expected 0.65–0.85 per C-26. Single-pump monitoring — cross-pump "
    "effects not modelled. Confidence scores may be lower on real-world faults than "
    "on simulated training data."
)

CLUSTER_NAMES = {
    "startup"      : "Cluster 0 — Startup",
    "steady_state" : "Cluster 1 — Steady-state",
    "high_load"    : "Cluster 2 — High-load",
    "cooldown"     : "Cluster 3 — Cooldown",
}


class M8p6Addendum(BaseModel):
    """
    Sensor-health sidecar annotation for Field 6 (C-28 / Principle 14).
    NEVER overrides fault_label or confidence_pct.
    override_existing_prediction: always False (locked).
    """
    triggered              : bool
    flagged_channels       : List[str] = []
    addendum_text          : str = ""
    override_existing_prediction: bool = False  # LOCKED — must always be False


class FaultPrediction(BaseModel):
    """
    Mandatory 7-field output for every inference call.
    All 7 fields present in every response — never reduced.
    """
    # ── 7 mandatory fields ──────────────────────────────────────────────────
    fault_label                    : str    # Field 1
    confidence_pct                 : float  # Field 2
    unknown_fault_flag             : bool   # Field 2b — True if confidence < 70%
    probable_physical_condition    : str    # Field 3
    expected_sensor_behavior       : str    # Field 4
    operational_risk_if_ignored    : str    # Field 5
    recommended_action             : str    # Field 6  (M8p6 addendum appended if triggered)
    model_limitation_disclaimer    : str = MODEL_DISCLAIMER_TEXT  # Field 7 — LOCKED

    # ── Detection layer outputs (Invariant 19) ──────────────────────────────
    score_A          : float        # L1 LSTM-AE reconstruction error → L4
    score_B          : float        # L2 TCN-AE drift slope → L3 CUSUM
    score_C          : float        # L2 TCN-AE chain transition → M7 XGBoost
    cusum_Sn         : float        # L3 current accumulator value
    adaptive_threshold: float       # L4 current θ_t
    alert_state      : str          # NORMAL | WATCH | WARN | DANGER

    # ── Supplementary fields ─────────────────────────────────────────────────
    prediction_id    : str          # UUID — key for active-learning row
    pump_id          : str
    cluster          : str
    timestamp_utc    : str
    ood_suspected    : bool         # Mahalanobis > tau_p99
    mahal_dist       : float
    causal_chain     : Optional[str] = None   # Group B only
    limitation_flags : List[str]    = []
    top_shap_features: Optional[Dict[str, float]] = None
    m8p6_addendum    : Optional[M8p6Addendum]     = None

    class Config:
        json_schema_extra = {
            "example": {
                "fault_label"                  : "cavitation",
                "confidence_pct"               : 82.7,
                "unknown_fault_flag"           : False,
                "probable_physical_condition"  : "Suction-side vapour bubble collapse — NPSHa likely below NPSHr at current flow rate.",
                "expected_sensor_behavior"     : "Discharge pressure oscillating ±2–4 bar. Suction pressure dropping. Motor vibration rising.",
                "operational_risk_if_ignored"  : "Hours to days — impeller erosion accelerates at 40 bar. Seal faces at risk from pressure cycling.",
                "recommended_action"           : "Check suction valve fully open. Reduce flow rate. Verify NPSHa ≥ NPSHr + 0.5 m.",
                "model_limitation_disclaimer"  : MODEL_DISCLAIMER_TEXT,
                "score_A"                      : 0.138,
                "score_B"                      : 0.031,
                "score_C"                      : 0.54,
                "cusum_Sn"                     : 2.14,
                "adaptive_threshold"           : 0.121,
                "alert_state"                  : "WARN",
                "prediction_id"                : "a1b2c3d4-...",
                "pump_id"                      : "PUMP-0032",
                "cluster"                      : "steady_state",
                "timestamp_utc"                : "2026-05-16T10:00:00Z",
                "ood_suspected"                : False,
                "mahal_dist"                   : 4.2,
            }
        }
