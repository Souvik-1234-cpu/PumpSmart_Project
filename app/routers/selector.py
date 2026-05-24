# app/routers/selector.py
# POST /api/select_pump  — M9 physics-only industrial pump sizing
# GET  /api/household    — physics-only household advisor (zero ML)
#
# T2-3 physical-parameter routing (LOCKED — never string-based):
#   is_industrial → ml_pipeline
#   is_household  → physics_advisory_only()
#   else          → OUT_OF_SCOPE (explicit refusal, no ML)
# =============================================================================

from fastapi import APIRouter, HTTPException, Request, Query
from app.schemas.sensor_input import PumpSpec, HouseholdSpec
import math

router = APIRouter(tags=["selector"])

ADVISORY_DISCLAIMER = (
    "Physics-only advisory. No ML inference. Results are estimates only — "
    "verify with a qualified engineer before purchasing or installing equipment."
)


def route_pump(power_kW: float, head_m: float, stages: int, pressure_bar: float) -> str:
    """T2-3 physical envelope routing — NEVER string-based."""
    is_industrial = (power_kW >= 30 and head_m >= 80 and
                     stages >= 3 and pressure_bar >= 8)
    if is_industrial:
        return "industrial_ml_pipeline"
    elif power_kW <= 5 and stages == 1 and pressure_bar <= 5:
        return "household_physics_advisory"
    else:
        return "OUT_OF_SCOPE"


# ── Darcy-Weisbach friction loss helper ─────────────────────────────────────
def darcy_head_loss(flow_m3h: float, length_m: float, diameter_mm: float,
                    roughness_m: float = 1.5e-5) -> float:
    """Returns friction head loss in metres."""
    d   = diameter_mm / 1000
    q   = flow_m3h / 3600         # m³/s
    A   = math.pi * d**2 / 4
    v   = q / A
    Re  = 1000 * v * d / 1e-3     # ρvD/μ, water at ~20°C
    if Re < 1:
        return 0.0
    # Colebrook-White (one iteration approximation)
    f  = 0.25 / (math.log10(roughness_m / (3.7 * d) + 5.74 / Re**0.9))**2
    hf = f * (length_m / d) * (v**2 / (2 * 9.81))
    return round(hf, 3)


# ── POST /api/select_pump ────────────────────────────────────────────────────
@router.post("/select_pump", summary="Industrial pump sizing (M9 physics)")
async def select_pump(spec: PumpSpec, request: Request):
    """
    Physics-only industrial pump sizing from M9.
    Delegates to pump_selector_dispatch from src.module_09_pump_selector.
    Phase 1: returns key sizing equations directly (M9 already locked).
    """
    # M9 script runs full training on import — use inline sizing directly
    pass

    # ── Inline M9 sizing (Phase 1 fallback) ────────────────────────────────
    Q   = spec.flow_rate_m3h
    H   = spec.total_head_m
    rho = spec.fluid_density
    g   = 9.81

    P_hydraulic = (rho * g * (Q / 3600) * H) / 1000   # kW
    eta_assumed = 0.75
    P_shaft     = P_hydraulic / eta_assumed
    P_motor     = P_shaft / 0.92   # motor efficiency

    stages = spec.stages or max(1, math.ceil(H / 70))
    H_per_stage = H / stages

    Ns = (2980 * (Q / 60)**0.5) / (H_per_stage**0.75)   # specific speed

    # Physical routing check
    route = route_pump(P_motor, H, stages, spec.pressure_bar or (rho*g*H/1e5))
    if route == "OUT_OF_SCOPE":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Pump specification ({P_motor:.1f} kW, {stages}-stage, {H:.0f} m head) "
                "is outside both the industrial ML envelope (≥30 kW, ≥3 stage, ≥80 m, ≥8 bar) "
                "and the household advisory envelope (≤5 kW, 1-stage, ≤5 bar). "
                "This commercial mid-size range is not supported — consult a pump engineer."
            ),
        )

    NPSHa = spec.suction_head_m - darcy_head_loss(Q, 5.0, 50.0) - 0.24   # approx vapour head at 20°C
    cavitation_risk = NPSHa < (1.2 + spec.npsh_margin_m)

    return {
        "route"            : route,
        "P_hydraulic_kW"   : round(P_hydraulic, 3),
        "P_shaft_kW"       : round(P_shaft, 3),
        "P_motor_kW"       : round(P_motor, 3),
        "recommended_stages": stages,
        "H_per_stage_m"    : round(H_per_stage, 2),
        "specific_speed_Ns": round(Ns, 2),
        "NPSHa_m"          : round(NPSHa, 3),
        "cavitation_risk"  : cavitation_risk,
        "notes"            : (
            "Estimates assume η_pump=0.75, η_motor=0.92, water at 20°C. "
            "Verify with pump curve from manufacturer."
        ),
        "advisory_disclaimer": ADVISORY_DISCLAIMER,
    }


# ── GET /api/household ───────────────────────────────────────────────────────
@router.get("/household", summary="Household pump advisor (physics-only, zero ML)")
async def household_advisor(
    usage_type     : str   = Query(...),
    daily_demand_L : float = Query(..., gt=0),
    pipe_length_m  : float = Query(..., gt=0),
    elevation_m    : float = Query(0.0),
    pipe_diameter_mm: float = Query(25.0, gt=0),
):
    """
    Physics-only household pump sizing. Zero ML inference.
    advisory_disclaimer always present in response (scope guard).
    """
    # Peak flow estimate: 3× average demand, 4-hour peak window
    avg_flow_lps   = daily_demand_L / (24 * 3600)
    peak_flow_lps  = avg_flow_lps * 3
    peak_flow_m3h  = peak_flow_lps * 3.6

    hf = darcy_head_loss(peak_flow_m3h, pipe_length_m, pipe_diameter_mm)
    total_head_m   = elevation_m + hf + 5.0   # 5 m residual pressure

    P_hydraulic    = (1000 * 9.81 * (peak_flow_lps / 1000) * total_head_m) / 1000
    P_motor_kW     = P_hydraulic / 0.55   # household pump typical efficiency

    return {
        "usage_type"         : usage_type,
        "peak_flow_m3h"      : round(peak_flow_m3h, 3),
        "total_head_m"       : round(total_head_m, 2),
        "friction_loss_m"    : round(hf, 3),
        "P_hydraulic_kW"     : round(P_hydraulic, 4),
        "recommended_motor_kW": round(P_motor_kW, 3),
        "pump_type_suggestion": (
            "Submersible pump" if usage_type == "borehole"
            else "Pressure booster" if usage_type == "pressure_boost"
            else "Surface centrifugal pump"
        ),
        "advisory_disclaimer": ADVISORY_DISCLAIMER,
        "ml_inference"       : False,   # scope guard — always False for household
    }
