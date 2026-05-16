# =============================================================================
# app/schemas/sensor_input.py
# Pydantic input validation for sensor window payloads.
# Raw sensor values NEVER enter — must be M3 cluster-normalised first.
# Channel order LOCKED from M6B: Mot.SV, Pmp.SV, Mot.TV, Pmp.PV,
#                                 Temp.SV, Pres.SV, Pmp.TV, Mot.PV
# =============================================================================

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal


CHANNEL_ORDER = [
    "Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
    "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV",
]
N_CHANNELS = 8
N_TIMESTEPS = 50


class SensorWindow(BaseModel):
    """
    50-step × 8-channel cluster-normalised sensor window.
    All values must be M3-normalised BEFORE submission.
    Normal operation produces values in 0–1.
    Fault signatures produce drift above 1.0.
    """
    window: List[List[float]] = Field(
        ...,
        description=(
            f"{N_TIMESTEPS} timesteps × {N_CHANNELS} channels. "
            "Cluster-normalised (M3). Channel order: " + ", ".join(CHANNEL_ORDER)
        ),
    )
    pump_id: str = Field(
        default="PUMP-0032",
        description="Pump identifier — single-pump deployment v14.2",
    )
    cluster: Literal["startup", "steady_state", "high_load", "cooldown"] = Field(
        default="steady_state",
        description="Active K-Means cluster from M2 (auto-assigned at runtime)",
    )
    timestamp_utc: str = Field(
        default="",
        description="ISO 8601 UTC timestamp of window start",
    )

    @field_validator("window")
    @classmethod
    def check_shape(cls, v):
        if len(v) != N_TIMESTEPS:
            raise ValueError(
                f"Expected {N_TIMESTEPS} timesteps, got {len(v)}. "
                f"Window size is locked from M4 training."
            )
        for i, row in enumerate(v):
            if len(row) != N_CHANNELS:
                raise ValueError(
                    f"Timestep {i}: expected {N_CHANNELS} channels, got {len(row)}. "
                    f"Channel order: {CHANNEL_ORDER}"
                )
        return v


class AcknowledgeRequest(BaseModel):
    """POST /api/acknowledge — operational reset after confirmed maintenance."""
    pump_id     : str = Field(default="PUMP-0032")
    action_taken: str = Field(
        ...,
        description="Free-text description of maintenance action performed",
    )
    operator_id : str = Field(default="", description="Optional operator identifier")
    timestamp_utc: str = Field(default="")


class OperatorVerdict(BaseModel):
    """Operator prediction verdict — written to active-learning data store."""
    prediction_id         : str
    verdict               : Literal["CORRECT", "INCORRECT", "UNSURE"]
    operator_correct_label: int | None = Field(
        default=None,
        ge=0, le=21,
        description="If INCORRECT: actual fault label (0–21). Else None.",
    )
    physical_inspection_done: bool = Field(default=False)
    inspection_notes         : str = Field(default="")
    operator_id              : str = Field(default="")
    consent_granted_by       : str = Field(default="")


class PumpSpec(BaseModel):
    """POST /api/select_pump — industrial pump sizing request (M9 physics)."""
    flow_rate_m3h   : float = Field(..., gt=0, description="Required flow rate (m³/h)")
    total_head_m    : float = Field(..., gt=0, description="Required total head (m)")
    fluid_density   : float = Field(default=1000.0, description="kg/m³")
    fluid_temp_c    : float = Field(default=20.0,   description="°C")
    suction_head_m  : float = Field(default=5.0,    description="Available suction head (m)")
    npsh_margin_m   : float = Field(default=0.5,    description="NPSHa safety margin (m)")
    power_kW        : float | None = Field(default=None, description="If known")
    stages          : int   | None = Field(default=None, ge=1)
    pressure_bar    : float | None = Field(default=None, gt=0)


class HouseholdSpec(BaseModel):
    """GET /api/household — household pump advisory (physics-only, no ML)."""
    usage_type     : Literal["domestic", "irrigation", "borehole", "pressure_boost"]
    daily_demand_L : float = Field(..., gt=0, description="Litres per day")
    pipe_length_m  : float = Field(..., gt=0)
    elevation_m    : float = Field(default=0.0)
    pipe_diameter_mm: float = Field(default=25.0, gt=0)
