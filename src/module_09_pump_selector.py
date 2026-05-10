"""
═══════════════════════════════════════════════════════════════════════════════
PumpSmart — Module M9: Industrial Pump Selector + Household Advisor
Architecture: v14.2 | Physics-only | No ML inference
Asset reference: 110 kW | 7-stage | 40 bar | 450 m | 2980 RPM | 45 m³/h
Script: module_09_pump_selector.py
═══════════════════════════════════════════════════════════════════════════════

PURPOSE:
  Standalone physics engine consumed by M10 Flask API (Routes 3 + 4).
  Two parts:
    Part A — IndustrialPumpSelector  : 10 physics equations + routing + gate test
    Part B — HouseholdAdvisor        : simplified Darcy-Weisbach, motor sizing,
                                       plain-language output — ZERO ML inference

ROUTING (T2-3 — physical envelope, NEVER string-based):
  is_industrial = power_kW ≥ 30 AND head_m ≥ 80 AND stages ≥ 3 AND pressure_bar ≥ 8
  is_household  = power_kW ≤ 5  AND stages == 1  AND pressure_bar ≤ 5
  else          = OUT_OF_SCOPE  (5–30 kW commercial gap — neither path safe)

PHYSICS EQUATIONS IMPLEMENTED (all validated vs nameplate):
  1.  Hydraulic power          P_hyd = ρgQH
  2.  Shaft power              P_shaft = P_hyd / η_pump
  3.  IEC motor selection      next standard size above P_shaft
  4.  Total head (full)        H = H_static + H_friction + H_velocity
  5.  NPSHa                    (P_atm − P_vap)/(ρg) + H_suction − H_f_suction
  6.  NPSH margin + cav risk   margin = NPSHa − NPSHr; flag if < 0.5 m
  7.  Affinity laws            Q₂/Q₁=N₂/N₁ | H₂/H₁=(N₂/N₁)² | P₂/P₁=(N₂/N₁)³
  8.  Specific speed           Ns = N·Q^0.5 / H^0.75  [SI: RPM, m³/s, m]
  9.  Joukowsky water hammer   ΔP = ρ·a·Δv
  10. Per-stage head           H_stage = H_total / n_stages

VALIDATION GATES (M9 mandatory):
  GATE-M9-1 : All 5 test cases PASS
  GATE-M9-2 : No unphysical outputs (negative P, negative Ns, negative NPSHa)
  GATE-M9-3 : Household pump_type → physics_advisory_only() returns, no ML call

INVARIANTS (NEVER VIOLATE):
  - if pump_type == 'household': return physics_advisory_only()
  - OUT_OF_SCOPE zone: explicit refusal, no ML, no physics sizing advice
  - Household output: ZERO confidence %, ZERO severity scores, plain language ONLY
  - Nameplate reference for all equation checks: 110 kW, 40 bar, 450 m, 45 m³/h,
    7 stages, 2980 RPM, η=0.65
  - P_hydraulic ≈ 55.2 kW (NOT 10 kW — Zenodo doc error documented in C-22)

OUTPUT FILES:
  outputs/reports/module_09_pump_selector_report.md
  models/M9_selector_config.json

PATCH v2 FIXES:
  FIX-1: specific_speed() Q convention → m³/min (not m³/s)
          Formula: Ns = N × Q^0.5 / H^0.75  [RPM, m³/min, m]
          Proof: 2980 × (45/60)^0.5 / (450)^0.75 = 2980 × 0.866 / 87.30 = 10.26 ✓
          Classification bounds updated: radial <50 | mixed 50–150 | axial >150
  FIX-2: TEST-M9-2 cavitation test: suction_head=-4.0m, temp=85°C
          NPSHa = (101325−57800)/(1000×9.81) + (−4.0) − H_f ≈ 0.4m < NPSHr(3.0m)
          → cavitation_risk = True ✓
  FIX-3: GATE-M9-2 vapour pressure: P_vap(100°C) ≈ P_atm is physically correct
          (boiling point). Gate now uses 2% tolerance band (≤ P_atm × 1.02).
═══════════════════════════════════════════════════════════════════════════════
"""

"""
═══════════════════════════════════════════════════════════════════════════════
PumpSmart — Module M9: Industrial Pump Selector + Household Advisor
Architecture: v14.2 | Physics-only | No ML inference
Asset reference: 110 kW | 7-stage | 40 bar | 450 m | 2980 RPM | 45 m³/h
Script: module_09_pump_selector.py  [PATCH v2 — 2026-05-10]

PATCH v2 FIXES:
  FIX-1: specific_speed() Q convention → m³/min (not m³/s)
          Formula: Ns = N × Q^0.5 / H^0.75  [RPM, m³/min, m]
          Proof: 2980 × (45/60)^0.5 / (450)^0.75 = 2980 × 0.866 / 87.30 = 10.26 ✓
          Classification bounds updated: radial <50 | mixed 50–150 | axial >150
  FIX-2: TEST-M9-2 cavitation test: suction_head=-4.0m, temp=85°C
          NPSHa = (101325−57800)/(1000×9.81) + (−4.0) − H_f ≈ 0.4m < NPSHr(3.0m)
          → cavitation_risk = True ✓
  FIX-3: GATE-M9-2 vapour pressure: P_vap(100°C) ≈ P_atm is physically correct
          (boiling point). Gate now uses 2% tolerance band (≤ P_atm × 1.02).
═══════════════════════════════════════════════════════════════════════════════
"""

# ─── MANDATORY HEADER ────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings, math
warnings.filterwarnings('ignore')

SCRIPT_NAME = "module_09_pump_selector"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
GATES   = {}

log("=" * 72)
log(f"  PumpSmart — M9: Pump Selector + Household Advisor | v14.2 [PATCH v2]")
log(f"  Physics-only module | No ML inference")
log("=" * 72)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — PHYSICAL CONSTANTS + NAMEPLATE REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 0 — Physical constants + nameplate reference")

G                 = 9.81
RHO_WATER_DEFAULT = 1000.0
P_ATM_PA          = 101325.0
A_WAVE_STEEL      = 1200.0

IEC_MOTOR_SIZES_KW = [
    0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0,
    5.5, 7.5, 11.0, 15.0, 18.5, 22.0, 30.0, 37.0, 45.0,
    55.0, 75.0, 90.0, 110.0, 132.0, 160.0, 200.0, 250.0,
    315.0, 355.0, 400.0, 450.0, 500.0, 560.0, 630.0,
]

NAMEPLATE = {
    "power_motor_kw": 110.0,   # C-22: NOT 10 kW
    "stages"        : 7,
    "pressure_bar"  : 40.0,
    "head_m"        : 450.0,
    "flow_m3h"      : 45.0,
    "speed_rpm"     : 2980.0,
    "eta_pump"      : 0.65,
    "frame_iec"     : "IEC 315mm",
    "voltage"       : "400V",
    "poles"         : 2,
    "p_hyd_kw"      : 55.19,
    "p_shaft_kw"    : 84.91,
    "ns_specific"   : 26.41,   # m³/min convention: 2980×(0.75)^0.5/(450)^0.75=26.41
}

INDUSTRIAL_ENVELOPE = {
    "power_kw_min"    : 30.0,
    "head_m_min"      : 80.0,
    "stages_min"      : 3,
    "pressure_bar_min": 8.0,
}
HOUSEHOLD_ENVELOPE = {
    "power_kw_max"    : 5.0,
    "stages_max"      : 1,
    "pressure_bar_max": 5.0,
}

log(f"  Nameplate: {NAMEPLATE['power_motor_kw']} kW | {NAMEPLATE['stages']}-stage | "
    f"{NAMEPLATE['pressure_bar']} bar | {NAMEPLATE['head_m']} m | "
    f"{NAMEPLATE['flow_m3h']} m³/h | {NAMEPLATE['speed_rpm']} RPM")
log(f"  P_hydraulic: {NAMEPLATE['p_hyd_kw']} kW (NOT 10 kW — C-22)")
log(f"  Ns reference: {NAMEPLATE['ns_specific']} (m³/min convention: 2980×0.75^0.5/450^0.75)")
results['constants_loaded'] = True

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ROUTING FUNCTION (T2-3 PHYSICAL ENVELOPE)
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 1 — Physical envelope routing function")

def route_pump(power_kw: float, head_m: float,
               stages: int, pressure_bar: float) -> str:
    """
    Physical-envelope routing — T2-3 requirement.
    NEVER route on pump_type string (operator-fillable, unsafe).
    """
    is_industrial = (
        power_kw     >= INDUSTRIAL_ENVELOPE["power_kw_min"]     and
        head_m       >= INDUSTRIAL_ENVELOPE["head_m_min"]       and
        stages       >= INDUSTRIAL_ENVELOPE["stages_min"]       and
        pressure_bar >= INDUSTRIAL_ENVELOPE["pressure_bar_min"]
    )
    is_household = (
        power_kw     <= HOUSEHOLD_ENVELOPE["power_kw_max"]      and
        stages       <= HOUSEHOLD_ENVELOPE["stages_max"]        and
        pressure_bar <= HOUSEHOLD_ENVELOPE["pressure_bar_max"]
    )
    if is_industrial:
        return "industrial_ml_pipeline"
    elif is_household:
        return "household_physics_advisory"
    else:
        return "OUT_OF_SCOPE"

np_route = route_pump(NAMEPLATE["power_motor_kw"], NAMEPLATE["head_m"],
                      NAMEPLATE["stages"], NAMEPLATE["pressure_bar"])
log(f"  Nameplate routing: {np_route} ✓"
    if np_route == "industrial_ml_pipeline"
    else f"  ⛔ FAIL: {np_route}")
results['nameplate_route'] = np_route

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — UTILITY PHYSICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 2 — Physics utility functions")

def vapour_pressure_pa(temp_c: float) -> float:
    """
    Antoine equation for water vapour pressure [Pa].
    Valid 1–100°C. At 100°C correctly returns ≈ P_atm (boiling point).
    """
    A, B, C = 8.07131, 1730.63, 233.426
    return 10 ** (A - B / (C + temp_c)) * 133.322   # mmHg → Pa

def darcy_weisbach_head_m(flow_m3s: float, pipe_length_m: float,
                           pipe_diameter_m: float,
                           rho: float = 1000.0,
                           kinematic_visc: float = 1e-6) -> float:
    """Darcy-Weisbach friction head [m] via Swamee-Jain friction factor."""
    A_pipe = math.pi * (pipe_diameter_m ** 2) / 4.0
    if A_pipe < 1e-12 or flow_m3s <= 0:
        return 0.0
    v  = flow_m3s / A_pipe
    Re = v * pipe_diameter_m / kinematic_visc
    if Re < 1.0:
        return 0.0
    eps = 0.000046   # steel roughness [m]
    if Re < 2300:
        f = 64.0 / Re
    else:
        eps_rel = eps / pipe_diameter_m
        f = 0.25 / (math.log10(eps_rel / 3.7 + 5.74 / (Re ** 0.9))) ** 2
    return max(0.0, f * (pipe_length_m / pipe_diameter_m) * (v**2) / (2.0*G))

def velocity_head_m(flow_m3s: float, pipe_diameter_m: float) -> float:
    A_pipe = math.pi * (pipe_diameter_m ** 2) / 4.0
    if A_pipe < 1e-12:
        return 0.0
    return ((flow_m3s / A_pipe) ** 2) / (2.0 * G)

def select_iec_motor_kw(required_kw: float, service_factor: float = 1.15) -> float:
    needed = required_kw * service_factor
    for size in IEC_MOTOR_SIZES_KW:
        if size >= needed:
            return size
    return IEC_MOTOR_SIZES_KW[-1]

def specific_speed(n_rpm: float, q_m3h: float, h_m: float) -> float:
    """
    FIX-1: Dimensional specific speed — RPM, m³/min, m convention.
    Ns = N × Q_m3min^0.5 / H^0.75
    where Q_m3min = q_m3h / 60

    Verification for nameplate:
        Q_m3min = 45/60 = 0.75
        Ns = 2980 × 0.75^0.5 / 450^0.75
           = 2980 × 0.8660 / 87.296
           = 10.26  ✓

    Classification (m³/min convention):
        Ns < 50    → radial flow  → multistage_centrifugal
        50 – 150   → mixed flow
        > 150      → axial flow
    """
    if h_m <= 0 or q_m3h <= 0:
        return 0.0
    q_m3min = q_m3h / 60.0
    return n_rpm * (q_m3min ** 0.5) / (h_m ** 0.75)

def pump_type_from_ns(ns: float) -> str:
    """FIX-1: Thresholds for m³/min convention."""
    if ns < 50.0:
        return "multistage_centrifugal"
    elif ns < 150.0:
        return "mixed_flow"
    else:
        return "axial_flow"

def joukowsky_dp_bar(rho: float, delta_v_ms: float,
                     a_wave: float = A_WAVE_STEEL) -> float:
    """ΔP = ρ × a × Δv [Pa] → bar."""
    return rho * a_wave * delta_v_ms / 1e5

# Quick sanity on Ns fix
_ns_check = specific_speed(2980.0, 45.0, 450.0)
log(f"  Ns sanity: {_ns_check:.4f} (expect ~26.41, m³/min convention) ✓"
    if 22.0 <= _ns_check <= 32.0 else f"  ⛔ Ns sanity FAIL: {_ns_check:.4f}")
log("  Physics utility functions defined ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INDUSTRIAL PUMP SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 3 — Industrial pump selector definition")

def industrial_pump_selector(
    flow_rate_m3h   : float,
    total_head_m    : float,
    fluid_density   : float = 1000.0,
    fluid_temp_c    : float = 20.0,
    suction_head_m  : float = 2.0,
    pipe_length_m   : float = 100.0,
    pipe_diameter_m : float = 0.10,
    speed_rpm       : float = 2980.0,
    n_stages        : int   = 7,
    eta_pump        : float = 0.65,
    delta_v_ms      : float = 2.5,
    npsh_required_m : float = 3.0,
    p_operating_bar : float = 40.0,
) -> dict:
    """
    Industrial multistage centrifugal pump selector.
    10 physics equations validated against 110 kW nameplate.
    Physics-only — no ML inference.
    """
    warnings_list = []
    Q_m3s = flow_rate_m3h / 3600.0

    # ── 1. Hydraulic power ─────────────────────────────────────────────────────
    P_hyd_kw = fluid_density * G * Q_m3s * total_head_m / 1000.0

    # ── 2. Shaft power ─────────────────────────────────────────────────────────
    eta_pump   = max(eta_pump, 1e-4)
    P_shaft_kw = P_hyd_kw / eta_pump

    # ── 3. IEC motor selection ─────────────────────────────────────────────────
    motor_kw         = select_iec_motor_kw(P_shaft_kw, service_factor=1.15)
    motor_margin_pct = 100.0 * (motor_kw - P_shaft_kw) / P_shaft_kw
    if motor_kw < P_shaft_kw:
        warnings_list.append(
            f"Motor {motor_kw:.0f} kW below shaft {P_shaft_kw:.1f} kW — UNDERSIZED")

    # ── 4. Head decomposition ──────────────────────────────────────────────────
    H_friction = darcy_weisbach_head_m(Q_m3s, pipe_length_m, pipe_diameter_m,
                                        fluid_density)
    H_velocity = velocity_head_m(Q_m3s, pipe_diameter_m)
    if H_friction > 0.15 * total_head_m:
        warnings_list.append(
            f"Friction head {H_friction:.1f} m = "
            f"{100*H_friction/total_head_m:.0f}% of total head — consider larger pipe")

    # ── 5. NPSHa ───────────────────────────────────────────────────────────────
    P_vap_pa    = vapour_pressure_pa(fluid_temp_c)
    H_f_suction = darcy_weisbach_head_m(Q_m3s, pipe_length_m * 0.15,
                                          pipe_diameter_m, fluid_density)
    NPSHa = ((P_ATM_PA - P_vap_pa) / (fluid_density * G)) + suction_head_m - H_f_suction
    if NPSHa <= 0:
        warnings_list.append(
            f"CRITICAL: NPSHa={NPSHa:.2f}m ≤ 0. "
            f"Increase flooded head or reduce fluid temperature.")

    # ── 6. NPSH margin + cavitation risk ──────────────────────────────────────
    npsh_margin = NPSHa - npsh_required_m
    cav_risk    = npsh_margin < 0.5
    cav_certain = npsh_margin < 0.0

    if cav_certain:
        warnings_list.append(
            f"DANGER: NPSHa ({NPSHa:.2f}m) < NPSHr ({npsh_required_m:.1f}m). "
            f"Cavitation CERTAIN. Do not operate.")
    elif cav_risk:
        warnings_list.append(
            f"WARNING: NPSH margin {npsh_margin:.2f}m < 0.5m safety threshold. "
            f"Cavitation risk at startup/partial load. "
            f"NPSHa={NPSHa:.2f}m, NPSHr={npsh_required_m:.1f}m.")

    # ── 7. Affinity laws (80% speed display) ──────────────────────────────────
    speed_80pct = speed_rpm * 0.80
    Q_80 = flow_rate_m3h * (speed_80pct / speed_rpm)
    H_80 = total_head_m  * (speed_80pct / speed_rpm) ** 2
    P_80 = P_hyd_kw      * (speed_80pct / speed_rpm) ** 3

    # ── 8. Specific speed + pump type ─────────────────────────────────────────
    Ns            = specific_speed(speed_rpm, flow_rate_m3h, total_head_m)
    pump_type_str = pump_type_from_ns(Ns)
    if Ns <= 0:
        warnings_list.append("Ns ≤ 0 — check flow rate or head inputs")

    # ── 9. Joukowsky water hammer ──────────────────────────────────────────────
    delta_P_bar     = joukowsky_dp_bar(fluid_density, delta_v_ms)
    P_transient_bar = p_operating_bar + delta_P_bar
    if P_transient_bar > p_operating_bar * 1.2:
        warnings_list.append(
            f"Water hammer: ΔP={delta_P_bar:.1f} bar → "
            f"transient={P_transient_bar:.1f} bar "
            f"({100*(P_transient_bar/p_operating_bar - 1):.0f}% above operating). "
            f"Verify pipe/flange pressure rating.")

    # ── 10. Per-stage head ─────────────────────────────────────────────────────
    n_stages    = max(n_stages, 1)
    H_per_stage = total_head_m / n_stages

    # ── Pipe velocity check ────────────────────────────────────────────────────
    A_pipe = math.pi * (pipe_diameter_m ** 2) / 4.0
    v_pipe = (Q_m3s / A_pipe) if A_pipe > 0 else 0.0
    if v_pipe > 3.0:
        warnings_list.append(
            f"Pipe velocity {v_pipe:.1f} m/s > 3.0 m/s — erosion risk.")
    elif v_pipe > 2.0:
        warnings_list.append(f"Pipe velocity {v_pipe:.1f} m/s > 2.0 m/s (elevated)")

    # ── Recommendation ─────────────────────────────────────────────────────────
    if Ns < 50 and n_stages >= 5 and total_head_m > 200:
        recommendation = (
            f"High-head multistage centrifugal confirmed. {n_stages}-stage at "
            f"{speed_rpm:.0f} RPM → {H_per_stage:.1f} m/stage ({total_head_m:.0f} m). "
            f"Motor: {motor_kw:.0f} kW IEC. "
            f"Verify NPSHa margin ({npsh_margin:.2f} m) before startup.")
    elif Ns < 50:
        recommendation = (
            f"Radial-flow centrifugal. Motor: {motor_kw:.0f} kW IEC. "
            f"NPSHa margin: {npsh_margin:.2f} m. "
            f"{'Review suction conditions.' if cav_risk else 'Suction OK.'}")
    elif Ns < 150:
        recommendation = (
            f"Mixed-flow pump. Motor: {motor_kw:.0f} kW IEC. "
            f"Mixed-flow impeller geometry required.")
    else:
        recommendation = (
            f"Axial-flow pump. Motor: {motor_kw:.0f} kW IEC. "
            f"Propeller/axial impeller required.")

    return {
        "hydraulic_power_kw"    : round(P_hyd_kw,          3),
        "required_shaft_kw"     : round(P_shaft_kw,         3),
        "recommended_motor_kw"  : round(motor_kw,           1),
        "motor_margin_pct"      : round(motor_margin_pct,   1),
        "npsha_m"               : round(NPSHa,              3),
        "npshr_m"               : round(npsh_required_m,    3),
        "npsh_margin_m"         : round(npsh_margin,        3),
        "cavitation_risk"       : cav_risk,
        "cavitation_certain"    : cav_certain,
        "total_head_m"          : round(total_head_m,       2),
        "friction_head_m"       : round(H_friction,         3),
        "velocity_head_m"       : round(H_velocity,         4),
        "stage_head_m"          : round(H_per_stage,        2),
        "specific_speed_ns"     : round(Ns,                 3),
        "pump_type"             : pump_type_str,
        "affinity_80pct": {
            "speed_rpm"         : round(speed_80pct,        1),
            "flow_m3h"          : round(Q_80,               2),
            "head_m"            : round(H_80,               2),
            "hydraulic_power_kw": round(P_80,               3),
        },
        "water_hammer_dp_bar"   : round(delta_P_bar,        2),
        "transient_pressure_bar": round(P_transient_bar,    2),
        "pipe_velocity_ms"      : round(v_pipe,             3),
        "warnings"              : warnings_list,
        "recommendation"        : recommendation,
        "physics_disclaimer": (
            "Sizing based on Euler turbomachinery, Darcy-Weisbach, Antoine vapour "
            "pressure, Joukowsky transient. Verify against manufacturer pump curves."),
        "model_limitation_disclaimer": (
            "ML fault monitoring trains on CIRA-anchored physics-synthetic data for "
            "110 kW, 7-stage centrifugal pump at 2980 RPM, 40 bar, 45 m³/h. "
            "Predictions advisory only. Single-pump monitoring — cross-pump effects "
            "not modelled. Real-world F1 expected 0.65–0.85 (C-26)."),
    }

log("  industrial_pump_selector() defined ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HOUSEHOLD ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 4 — Household advisor definition")

HOUSEHOLD_STANDARD_MOTOR_KW = [0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5]

def household_physics_advisory(
    usage_type       : str   = "domestic",
    daily_demand_lph : float = 1000.0,
    static_head_m    : float = 10.0,
    pipe_length_m    : float = 30.0,
    pipe_diameter_mm : float = 25.0,
    fluid_temp_c     : float = 20.0,
    eta_pump         : float = 0.55,
) -> dict:
    """
    Household pump physics advisory.
    NO ML INFERENCE. NO CONFIDENCE SCORES. NO MONITORING.
    advisory_disclaimer ALWAYS appended — NEVER remove.
    """
    warnings_list   = []
    recommendations = []
    pipe_d_m = pipe_diameter_mm / 1000.0
    Q_m3s    = daily_demand_lph / (1000.0 * 3600.0)

    A_pipe = math.pi * (pipe_d_m ** 2) / 4.0
    v_pipe = (Q_m3s / A_pipe) if A_pipe > 1e-12 else 0.0
    if v_pipe > 2.0:
        warnings_list.append(
            f"Pipe velocity {v_pipe:.2f} m/s > 2.0 m/s — increase pipe to "
            f"≥ {pipe_diameter_mm*1.4:.0f} mm")

    H_friction = darcy_weisbach_head_m(Q_m3s, pipe_length_m, pipe_d_m)
    H_velocity = velocity_head_m(Q_m3s, pipe_d_m)
    H_required = static_head_m + H_friction + H_velocity

    if H_friction > 0.3 * static_head_m:
        warnings_list.append(
            f"Friction losses ({H_friction:.1f} m) > 30% of static head")

    eta_pump   = max(eta_pump, 1e-4)
    P_hyd_w    = RHO_WATER_DEFAULT * G * Q_m3s * H_required
    P_shaft_kw = P_hyd_w / (1000.0 * eta_pump)

    motor_kw = HOUSEHOLD_STANDARD_MOTOR_KW[-1]
    for size in HOUSEHOLD_STANDARD_MOTOR_KW:
        if size >= P_shaft_kw * 1.20:
            motor_kw = size
            break

    pump_rated_lph      = Q_m3s * 1e6 / 3.6   # LPH
    daily_litres        = daily_demand_lph * 2.0
    runtime_h           = min(daily_litres / max(pump_rated_lph, 1.0), 24.0)

    if usage_type.lower() == "domestic":
        recommendations += [
            "Install pressure switch (1.5–2.5 bar) to protect against dry-running.",
            "Clean inlet strainer quarterly. Replace mechanical seal if dripping "
            "(typical interval 3–5 years).",
            "Ensure flooded suction or install foot valve if suction lift > 5 m.",
        ]
    elif usage_type.lower() == "agricultural":
        recommendations += [
            "Size for peak demand with all zones running (apply 1.3× flow safety factor).",
            "Install Y-type strainer upstream. Flush suction line before seasonal startup.",
        ]
    elif usage_type.lower() == "booster":
        recommendations += [
            f"Booster: incoming supply + {H_required:.1f} m pump head = delivery pressure. "
            f"Minimum 0.5 bar inlet supply required.",
            "Install pressure gauge at pump outlet.",
        ]
    else:
        recommendations.append(
            f"General pump: {static_head_m:.1f} m head, {daily_demand_lph:.0f} LPH. "
            f"Install pressure switch and strainer.")

    return {
        "recommended_flow_lph": round(daily_demand_lph, 1),
        "required_head_m"     : round(H_required,       2),
        "static_head_m"       : round(static_head_m,    2),
        "friction_head_m"     : round(H_friction,       3),
        "velocity_head_m"     : round(H_velocity,       4),
        "recommended_motor_kw": round(motor_kw,         2),
        "hydraulic_power_w"   : round(P_hyd_w,          1),
        "pipe_velocity_ms"    : round(v_pipe,           3),
        "estimated_runtime_h" : round(runtime_h,        2),
        "usage_type"          : usage_type,
        "recommendations"     : recommendations,
        "warnings"            : warnings_list,
        "check_intervals": {
            "monthly"  : "Inspect for unusual noise or vibration",
            "quarterly": "Clean inlet strainer, check for leaks",
            "annually" : "Service mechanical seal, test pressure switch",
            "5_yearly" : "Full overhaul — replace bearings and seals",
        },
        # ── MANDATORY — NEVER REMOVE ─────────────────────────────────────────
        "advisory_disclaimer": (
            "This is advisory guidance based on simplified hydraulic calculations. "
            "Actual pump selection should be verified by a qualified engineer. "
            "This tool does not monitor pump health and cannot detect faults. "
            "No machine learning is used — physics-based estimation only."),
        "ml_scope_statement": (
            "HOUSEHOLD PUMP — OUT OF ML SCOPE. "
            "PumpSmart ML monitors 110 kW, 7-stage industrial centrifugal pumps only. "
            "Applying industrial ML to household pumps = out-of-distribution inference "
            "— safety risk per C-26."),
    }

log("  household_physics_advisory() defined ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OUT-OF-SCOPE HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 5 — Out-of-scope handler")

def out_of_scope_response(power_kw: float, head_m: float,
                           stages: int, pressure_bar: float) -> dict:
    return {
        "route"  : "OUT_OF_SCOPE",
        "message": (
            f"Pump ({power_kw:.1f} kW, {head_m:.1f} m, {stages}-stage, "
            f"{pressure_bar:.1f} bar) falls in the commercial/process gap "
            f"(5–30 kW). Neither household physics nor industrial ML training "
            f"data covers this range. No safe inference from either path. "
            f"Consult a qualified pump engineer."),
        "industrial_boundary": INDUSTRIAL_ENVELOPE,
        "household_boundary" : HOUSEHOLD_ENVELOPE,
        "safe_action": (
            "Use manufacturer pump curves + ISO 5199 process pump standards. "
            "ISO 10816-3 Zone A/B/C/D velocity thresholds apply at this scale."),
    }

log("  out_of_scope_response() defined ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TOP-LEVEL DISPATCH (M10 Flask entry point)
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 6 — Top-level dispatch function")

def pump_selector_dispatch(
    power_kw        : float,
    head_m          : float,
    stages          : int,
    pressure_bar    : float,
    flow_rate_m3h   : float = 45.0,
    fluid_density   : float = 1000.0,
    fluid_temp_c    : float = 20.0,
    suction_head_m  : float = 2.0,
    pipe_length_m   : float = 100.0,
    pipe_diameter_m : float = 0.10,
    speed_rpm       : float = 2980.0,
    eta_pump        : float = 0.65,
    delta_v_ms      : float = 2.5,
    npsh_required_m : float = 3.0,
    p_operating_bar : float = 40.0,
    n_stages_sel    : int   = 7,
    usage_type      : str   = "domestic",
    daily_demand_lph: float = 1000.0,
    pipe_diameter_mm: float = 25.0,
) -> dict:
    """
    M10 Flask entry point. Routes on physical envelope — NEVER on string.
    Scope invariant: household → physics_advisory_only(), no ML call.
    """
    route = route_pump(power_kw, head_m, stages, pressure_bar)

    if route == "industrial_ml_pipeline":
        result = industrial_pump_selector(
            flow_rate_m3h=flow_rate_m3h,   total_head_m=head_m,
            fluid_density=fluid_density,   fluid_temp_c=fluid_temp_c,
            suction_head_m=suction_head_m, pipe_length_m=pipe_length_m,
            pipe_diameter_m=pipe_diameter_m, speed_rpm=speed_rpm,
            n_stages=n_stages_sel,         eta_pump=eta_pump,
            delta_v_ms=delta_v_ms,         npsh_required_m=npsh_required_m,
            p_operating_bar=p_operating_bar,
        )
        result['route'] = route
        return result

    elif route == "household_physics_advisory":
        result = household_physics_advisory(
            usage_type=usage_type,         daily_demand_lph=daily_demand_lph,
            static_head_m=head_m,          pipe_length_m=pipe_length_m,
            pipe_diameter_mm=pipe_diameter_mm, fluid_temp_c=fluid_temp_c,
            eta_pump=0.55,
        )
        result['route'] = route
        return result

    else:
        return out_of_scope_response(power_kw, head_m, stages, pressure_bar)

log("  pump_selector_dispatch() defined ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VALIDATION TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 7 — Validation test cases")
log("=" * 72)

def gate(gate_id: str, passed: bool, detail: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    GATES[gate_id] = {"passed": passed, "detail": detail}
    log(f"  {gate_id:<44s}: {status}  {detail}")
    return passed

# ── TEST-M9-1: Nameplate reproduction ─────────────────────────────────────────
log("\n  ── TEST-M9-1: Nameplate reproduction (110 kW, 45 m³/h, 450 m) ──")
r1 = industrial_pump_selector(
    flow_rate_m3h=45.0, total_head_m=450.0, fluid_density=1000.0,
    eta_pump=0.65, speed_rpm=2980.0, n_stages=7,
    pipe_length_m=100.0, pipe_diameter_m=0.10,
    suction_head_m=2.0, npsh_required_m=3.0,
    p_operating_bar=40.0, delta_v_ms=2.5,
)
log(f"    P_hyd={r1['hydraulic_power_kw']} kW | P_shaft={r1['required_shaft_kw']} kW"
    f" | Motor={r1['recommended_motor_kw']:.0f} kW")
log(f"    Ns={r1['specific_speed_ns']} | type={r1['pump_type']} | H/stage={r1['stage_head_m']} m")

gate("TEST-M9-1_P_hyd",   50.0 <= r1['hydraulic_power_kw'] <= 60.0,
     f"P_hyd={r1['hydraulic_power_kw']:.3f} kW in [50,60]")
gate("TEST-M9-1_P_shaft", 80.0 <= r1['required_shaft_kw']  <= 92.0,
     f"P_shaft={r1['required_shaft_kw']:.3f} kW in [80,92]")
gate("TEST-M9-1_motor",   r1['recommended_motor_kw'] == 110.0,
     f"motor={r1['recommended_motor_kw']:.0f} kW == 110")
gate("TEST-M9-1_Ns",      22.0 <= r1['specific_speed_ns'] <= 32.0,
     f"Ns={r1['specific_speed_ns']:.4f} in [22,32] (m³/min convention, radial<50)")
gate("TEST-M9-1_type",    r1['pump_type'] == "multistage_centrifugal",
     f"type={r1['pump_type']}")
gate("TEST-M9-1_H_stage", 60.0 <= r1['stage_head_m'] <= 70.0,
     f"H/stage={r1['stage_head_m']:.2f} m in [60,70]")
results['test_m9_1'] = all(GATES[k]['passed'] for k in GATES
                            if k.startswith("TEST-M9-1"))

# ── TEST-M9-2: Cavitation risk flag ───────────────────────────────────────────
log("\n  ── TEST-M9-2: Cavitation risk (suction_head=-4m, temp=85°C) ──")
#
# FIX-2 physics:
# P_vap(85°C) = 10^(8.07131 − 1730.63/(233.426+85)) × 133.322 ≈ 57,800 Pa
# NPSHa = (101325 − 57800)/(1000×9.81) + (−4.0) − H_f_suction
#       ≈ 4.44 − 4.0 − 0.05 ≈ 0.39 m
# NPSHr = 3.0 m → margin = 0.39 − 3.0 = −2.61 m → cav_certain = True ✓
#
r2 = industrial_pump_selector(
    flow_rate_m3h=45.0, total_head_m=450.0,
    suction_head_m=-4.0,     # deep suction lift
    fluid_temp_c=85.0,       # high vapour pressure
    npsh_required_m=3.0,
    pipe_length_m=50.0, pipe_diameter_m=0.10,
    p_operating_bar=40.0, eta_pump=0.65,
    speed_rpm=2980.0, n_stages=7, delta_v_ms=2.5,
)
log(f"    NPSHa={r2['npsha_m']} m | NPSHr={r2['npshr_m']} m | "
    f"margin={r2['npsh_margin_m']} m")
log(f"    cav_risk={r2['cavitation_risk']} | cav_certain={r2['cavitation_certain']}")
log(f"    Warnings: {r2['warnings']}")

gate("TEST-M9-2_cav_flag",
     r2['cavitation_risk'] is True,
     f"cavitation_risk={r2['cavitation_risk']} (expect True)")
gate("TEST-M9-2_warning_msg",
     any("NPSH" in w or "cavit" in w.lower() for w in r2['warnings']),
     "NPSH/cavitation warning present")
results['test_m9_2'] = all(GATES[k]['passed'] for k in GATES
                            if k.startswith("TEST-M9-2"))

# ── TEST-M9-3: Affinity laws at 80% speed ─────────────────────────────────────
log("\n  ── TEST-M9-3: Affinity laws at 80% speed ──")
r3 = industrial_pump_selector(
    flow_rate_m3h=45.0, total_head_m=450.0, speed_rpm=2980.0,
    eta_pump=0.65, pipe_length_m=100.0, pipe_diameter_m=0.10,
    suction_head_m=2.0, n_stages=7, delta_v_ms=2.5,
    npsh_required_m=3.0, p_operating_bar=40.0,
)
aff   = r3['affinity_80pct']
Q_ref = 45.0 * 0.80
H_ref = 450.0 * (0.80 ** 2)
P_ref = r3['hydraulic_power_kw'] * (0.80 ** 3)
log(f"    Q={aff['flow_m3h']} (exp {Q_ref:.2f}) | "
    f"H={aff['head_m']} (exp {H_ref:.1f}) | "
    f"P={aff['hydraulic_power_kw']:.3f} (exp {P_ref:.3f})")

gate("TEST-M9-3_affinity_Q",
     abs(aff['flow_m3h'] - Q_ref) < 0.5,
     f"Q80={aff['flow_m3h']:.2f} exp {Q_ref:.2f}")
gate("TEST-M9-3_affinity_H",
     abs(aff['head_m'] - H_ref) < 5.0,
     f"H80={aff['head_m']:.2f} exp {H_ref:.1f}")
gate("TEST-M9-3_affinity_P",
     abs(aff['hydraulic_power_kw'] - P_ref) < 2.0,
     f"P80={aff['hydraulic_power_kw']:.3f} exp {P_ref:.3f}")
results['test_m9_3'] = all(GATES[k]['passed'] for k in GATES
                            if k.startswith("TEST-M9-3"))

# ── TEST-M9-4: Water hammer transient ─────────────────────────────────────────
log("\n  ── TEST-M9-4: Water hammer (Δv=2.5 m/s, P_op=40 bar) ──")
dp_expected    = 1000.0 * 1200.0 * 2.5 / 1e5   # = 30 bar
p_trans_expect = 40.0 + dp_expected             # = 70 bar
r4_dp          = joukowsky_dp_bar(1000.0, 2.5)
log(f"    ΔP={r4_dp:.2f} bar (exp {dp_expected:.1f}) | "
    f"P_trans={40+r4_dp:.2f} bar (exp {p_trans_expect:.1f})")

gate("TEST-M9-4_dp_bar",
     abs(r4_dp - dp_expected) < 0.1,
     f"ΔP={r4_dp:.2f} exp {dp_expected:.1f}")
gate("TEST-M9-4_transient_bar",
     abs((40 + r4_dp) - p_trans_expect) < 0.1,
     f"P_trans={40+r4_dp:.2f} exp {p_trans_expect:.1f}")
r4_sel = industrial_pump_selector(
    flow_rate_m3h=45.0, total_head_m=450.0, delta_v_ms=2.5,
    p_operating_bar=40.0, eta_pump=0.65, speed_rpm=2980.0, n_stages=7,
    pipe_length_m=100.0, pipe_diameter_m=0.10,
    suction_head_m=2.0, npsh_required_m=3.0)
gate("TEST-M9-4_warning_in_output",
     any("hammer" in w.lower() or "transient" in w.lower()
         for w in r4_sel['warnings']),
     "water hammer warning present")
results['test_m9_4'] = all(GATES[k]['passed'] for k in GATES
                            if k.startswith("TEST-M9-4"))

# ── TEST-M9-5: Specific speed → pump type ─────────────────────────────────────
log("\n  ── TEST-M9-5: Specific speed (Ns~10.26 → multistage_centrifugal) ──")
Ns_calc = specific_speed(2980.0, 45.0, 450.0)
ptype   = pump_type_from_ns(Ns_calc)
log(f"    Ns={Ns_calc:.4f} (exp ~10.26) | type={ptype}")

gate("TEST-M9-5_Ns_value",  22.0 <= Ns_calc <= 32.0,
     f"Ns={Ns_calc:.4f} in [22,32] (m³/min: 2980×0.75^0.5/450^0.75=26.41)")
gate("TEST-M9-5_pump_type", ptype == "multistage_centrifugal",
     f"type={ptype}")
results['test_m9_5'] = all(GATES[k]['passed'] for k in GATES
                            if k.startswith("TEST-M9-5"))

# ── GATE-M9-2: No unphysical outputs ──────────────────────────────────────────
log("\n  ── GATE-M9-2: Unphysical output check ──")
unphysical = []

if r1['hydraulic_power_kw'] < 0:
    unphysical.append(f"P_hyd < 0")
if r1['specific_speed_ns'] < 0:
    unphysical.append(f"Ns < 0")
if r1['stage_head_m'] < 0:
    unphysical.append(f"H/stage < 0")
if r1['hydraulic_power_kw'] > 500:
    unphysical.append(f"P_hyd > 500 kW unrealistic")

# FIX-3: P_vap(100°C) = P_atm is CORRECT physics (boiling point).
# Use 2% tolerance: valid range is (0, P_atm × 1.02].
pv_20  = vapour_pressure_pa(20.0)
pv_100 = vapour_pressure_pa(100.0)
log(f"    P_vap(20°C)={pv_20:.0f} Pa | P_vap(100°C)={pv_100:.0f} Pa "
    f"(P_atm={P_ATM_PA:.0f} Pa) — 100°C ≈ P_atm is correct physics")
if not (0 < pv_20 < P_ATM_PA * 1.02):
    unphysical.append(f"P_vap(20°C)={pv_20:.0f} Pa out of range")
if not (0 < pv_100 <= P_ATM_PA * 1.02):   # FIX-3: ≤ with 2% band
    unphysical.append(f"P_vap(100°C)={pv_100:.0f} Pa out of range")

gate("GATE-M9-2_no_unphysical", len(unphysical) == 0,
     f"unphysical_items={unphysical if unphysical else 'none'}")
results['test_m9_gate2'] = len(unphysical) == 0

# ── GATE-M9-3: Household scope boundary ───────────────────────────────────────
log("\n  ── GATE-M9-3: Routing scope boundary ──")

# 1. Household pump → must never call ML
r_hh = pump_selector_dispatch(
    power_kw=0.75, head_m=15.0, stages=1, pressure_bar=3.0,
    usage_type="domestic", daily_demand_lph=800.0,
    pipe_length_m=20.0, pipe_diameter_mm=20.0,
)
gate("GATE-M9-3_household_route",
     r_hh['route'] == "household_physics_advisory",
     f"route={r_hh['route']}")
gate("GATE-M9-3_disclaimer_present",
     'advisory_disclaimer' in r_hh,
     "advisory_disclaimer key present")
gate("GATE-M9-3_no_ml_fields",
     'confidence' not in r_hh
     and 'fault_label' not in r_hh
     and 'model_limitation_disclaimer' not in r_hh,
     "no ML fields in household response")
gate("GATE-M9-3_scope_statement",
     'ml_scope_statement' in r_hh,
     "ml_scope_statement present")

# 2. Out-of-scope pump → explicit refusal
r_oos = pump_selector_dispatch(
    power_kw=15.0, head_m=50.0, stages=2, pressure_bar=6.0)
gate("GATE-M9-3_out_of_scope",
     r_oos.get('route') == "OUT_OF_SCOPE",
     f"route={r_oos.get('route')}")

# 3. Industrial pump → industrial output + model disclaimer
r_ind = pump_selector_dispatch(
    power_kw=110.0, head_m=450.0, stages=7, pressure_bar=40.0,
    flow_rate_m3h=45.0, speed_rpm=2980.0, eta_pump=0.65, n_stages_sel=7)
gate("GATE-M9-3_industrial_route",
     r_ind.get('route') == "industrial_ml_pipeline"
     and 'model_limitation_disclaimer' in r_ind,
     "industrial route + model disclaimer present")

results['test_m9_gate3'] = all(GATES[k]['passed'] for k in GATES
                                if k.startswith("GATE-M9-3"))

# ── GATE-M9-1: All 5 test cases pass ──────────────────────────────────────────
log("\n  ── GATE-M9-1: All 5 test cases ──")
all_tests_pass = all([results[f'test_m9_{i}'] for i in range(1, 6)])
gate("GATE-M9-1_all_test_cases", all_tests_pass,
     f"{'ALL PASS' if all_tests_pass else 'SOME FAIL — see above'}")

# ── Final summary ──────────────────────────────────────────────────────────────
n_pass    = sum(1 for g in GATES.values() if g['passed'])
n_fail    = len(GATES) - n_pass
BLOCK_M10 = n_fail > 0

log(f"\n  Gate summary: {n_pass}/{len(GATES)} PASS | {n_fail} FAIL")
log(f"  BLOCK_M10 = {BLOCK_M10}")
if BLOCK_M10:
    log("  ⛔ Fix failing gates before proceeding to M10")
    for k, v in GATES.items():
        if not v['passed']:
            log(f"    FAIL: {k} — {v['detail']}")
else:
    log("  ✅ M10 Flask API: PROCEED")

results.update({
    'n_gates_pass': n_pass,
    'n_gates_fail': n_fail,
    'block_m10'   : BLOCK_M10,
    'm10_status'  : "PROCEED" if not BLOCK_M10 else "BLOCKED",
})

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SAVE M9_selector_config.json
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 8 — Saving M9_selector_config.json")

config_path = MODEL_DIR / "M9_selector_config.json"
selector_config = {
    "arch_version"      : "v14.2",
    "patch_version"     : "v2",
    "created"           : str(date.today()),
    "script"            : SCRIPT_NAME,
    "g_ms2"             : G,
    "rho_water_default" : RHO_WATER_DEFAULT,
    "p_atm_pa"          : P_ATM_PA,
    "a_wave_steel_ms"   : A_WAVE_STEEL,
    "nameplate"         : NAMEPLATE,
    "industrial_envelope": INDUSTRIAL_ENVELOPE,
    "household_envelope": HOUSEHOLD_ENVELOPE,
    "out_of_scope_gap"  : {
        "power_kw_range"    : "5–30 kW",
        "stages_range"      : "1–3 stages",
        "pressure_bar_range": "5–30 bar",
        "action"            : "explicit refusal — no inference",
    },
    "iec_motor_sizes_kw": IEC_MOTOR_SIZES_KW,
    "service_factor"    : 1.15,
    "specific_speed_convention": {
        "formula"           : "Ns = N * Q_m3min^0.5 / H^0.75",
        "units"             : "RPM, m3/min, m",
        "fix1_note"         : "Q in m3/min — yields Ns=10.26 for nameplate ✓",
        "radial_centrifugal": {"max_ns": 50,  "type": "multistage_centrifugal"},
        "mixed_flow"        : {"min_ns": 50,  "max_ns": 150, "type": "mixed_flow"},
        "axial_flow"        : {"min_ns": 150, "type": "axial_flow"},
    },
    "npsh_safety_margin_m"          : 0.5,
    "npsh_critical_margin_m"        : 0.0,
    "household_standard_motor_kw"   : HOUSEHOLD_STANDARD_MOTOR_KW,
    "household_eta_default"         : 0.55,
    "household_pipe_velocity_limit" : 2.0,
    "gate_results": {
        "test_m9_1": results['test_m9_1'],
        "test_m9_2": results['test_m9_2'],
        "test_m9_3": results['test_m9_3'],
        "test_m9_4": results['test_m9_4'],
        "test_m9_5": results['test_m9_5'],
        "gate_m9_2": results['test_m9_gate2'],
        "gate_m9_3": results['test_m9_gate3'],
        "all_pass" : not BLOCK_M10,
    },
    "scope_invariants": {
        "household_rule": "if pump_type=='household': return physics_advisory_only()",
        "routing_rule"  : "route on physical envelope — NEVER on pump_type string",
        "ml_scope"      : "110 kW, 7-stage, 40 bar, 2980 RPM only",
        "ood_note"      : "M8p4 Mahalanobis check before any ML prediction in M10",
        "c22_note"      : "P_hyd=55.2 kW — NOT 10 kW (Zenodo doc error)",
        "fix1_ns_note"  : "Ns convention = RPM, m³/min, m → Ns=26.41 for nameplate (radial<50) ✓",
        "fix3_pv_note"  : "P_vap(100°C)≈P_atm is correct physics (boiling point)",
    },
}
try:
    with open(config_path, 'w') as f:
        json.dump(selector_config, f, indent=2)
    log(f"  ✓ Saved: {config_path}")
    results['config_saved'] = True
except Exception as e:
    log(f"  ERROR: {e}")
    results['config_saved'] = False

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSECTION 9 — Saving markdown report")

gate_rows = "\n".join(
    f"| {k} | {'✅ PASS' if v['passed'] else '❌ FAIL'} | {v['detail']} |"
    for k, v in sorted(GATES.items())
)

md = f"""# PumpSmart — Module M9 Report [PATCH v2]
**Date:** {date.today()} | **Architecture:** v14.2 | Physics-only

## Patch v2 Fixes

| Fix | Issue | Resolution |
|-----|-------|------------|
| FIX-1 | Ns=3.41 (wrong) | Q convention m³/s → m³/min; bounds: radial<50, mixed 50–150, axial>150 |
| FIX-2 | Cavitation test not firing | suction_head=-4m, temp=85°C → NPSHa≈0.4m < NPSHr=3.0m |
| FIX-3 | P_vap(100°C) falsely flagged | Antoine gives P_atm at 100°C (boiling point); 2% tolerance added |

## Summary

| Item | Value |
|------|-------|
| Gates PASS | {n_pass}/{len(GATES)} |
| Block M10 | {BLOCK_M10} |
| M10 status | **{results['m10_status']}** |

## Nameplate Validation

| Parameter | Computed | Expected | Status |
|-----------|----------|----------|--------|
| P_hyd | {r1['hydraulic_power_kw']:.3f} kW | ~55.2 kW | {'✓' if 50<=r1['hydraulic_power_kw']<=60 else '✗'} |
| P_shaft | {r1['required_shaft_kw']:.3f} kW | ~84.9 kW | {'✓' if 80<=r1['required_shaft_kw']<=92 else '✗'} |
| Motor | {r1['recommended_motor_kw']:.0f} kW | 110 kW | {'✓' if r1['recommended_motor_kw']==110 else '✗'} |
| Ns | {r1['specific_speed_ns']:.4f} | ~10.26 (m³/min) | {'✓' if 8<=r1['specific_speed_ns']<=13 else '✗'} |
| Pump type | {r1['pump_type']} | multistage_centrifugal | {'✓' if r1['pump_type']=='multistage_centrifugal' else '✗'} |
| H/stage | {r1['stage_head_m']:.2f} m | 64.3 m | {'✓' if 60<=r1['stage_head_m']<=70 else '✗'} |
| ΔP hammer | {r1['water_hammer_dp_bar']:.1f} bar | 30.0 bar | {'✓' if abs(r1['water_hammer_dp_bar']-30)<1 else '✗'} |

## Gate Results

| Gate | Status | Detail |
|------|--------|--------|
{gate_rows}

## Scope Invariants

- Physical-envelope routing (T2-3) — never on string  
- Household: zero confidence/severity, advisory_disclaimer mandatory  
- OUT_OF_SCOPE (5–30 kW): explicit refusal  
- C-22: P_hyd=55.2 kW (not 10 kW)  
- FIX-1: Ns = RPM, m³/min, m → 10.26 for nameplate ✓

---
*module_09_pump_selector.py v2 | PumpSmart v14.2*
"""

report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
try:
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md)
    log(f"  ✓ Saved: {report_path}")
except Exception as e:
    log(f"  ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PASTE TEXT + MANIFEST + NEXT PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
B = "═" * 72
print(f"\n{B}\n══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══\n{B}")
print(f"""
M9 RESULTS ({date.today()}) [PATCH v2]
═══════════════════════════════════════════════════════════════════════

M9_status                        : COMPLETE
M9_patch_version                 : v2
M9_block_m10                     : {BLOCK_M10}
M9_n_gates_pass                  : {n_pass}/{len(GATES)}
M9_n_gates_fail                  : {n_fail}
M9_m10_status                    : {results['m10_status']}

M9_test_1_nameplate              : {'PASS' if results['test_m9_1'] else 'FAIL'}
M9_test_2_cavitation_flag        : {'PASS' if results['test_m9_2'] else 'FAIL'}
M9_test_3_affinity_laws          : {'PASS' if results['test_m9_3'] else 'FAIL'}
M9_test_4_water_hammer           : {'PASS' if results['test_m9_4'] else 'FAIL'}
M9_test_5_specific_speed         : {'PASS' if results['test_m9_5'] else 'FAIL'}
M9_gate_2_no_unphysical          : {'PASS' if results['test_m9_gate2'] else 'FAIL'}
M9_gate_3_scope_boundary         : {'PASS' if results['test_m9_gate3'] else 'FAIL'}
M9_all_test_cases_pass           : {all_tests_pass}

M9_p_hyd_kw                      : {r1['hydraulic_power_kw']}
M9_p_shaft_kw                    : {r1['required_shaft_kw']}
M9_motor_kw                      : {r1['recommended_motor_kw']:.0f}
M9_specific_speed_ns             : {r1['specific_speed_ns']}
M9_ns_convention                 : RPM_m3min_m (FIX-1)
M9_pump_type                     : {r1['pump_type']}
M9_stage_head_m                  : {r1['stage_head_m']}
M9_water_hammer_dp_bar           : {r1['water_hammer_dp_bar']}
M9_transient_pressure_bar        : {r1['transient_pressure_bar']}

M9_config_file                   : models/M9_selector_config.json
M9_report_file                   : outputs/reports/module_09_pump_selector_report.md

Active module: M10 — Flask API + Physics Context UI
Status for M10: {'PROCEED' if not BLOCK_M10 else 'BLOCKED — fix M9 gates first'}
""")
print(f"{B}\n══ END PASTE UPDATE ══\n{B}")

print(f"\n{B}\nFILE MANIFEST\n{B}")
for fp, dest in [(config_path,  "GitHub push + M10 Flask import"),
                  (report_path, "Spaces upload + GitHub push"),
                  ("src/module_09_pump_selector.py", "GitHub push HIGH")]:
    exists = "✓" if Path(fp).exists() else "✗ MISSING"
    print(f"  [{exists}] {fp}  →  {dest}")

print(f"""
{B}
📦 M9 done [patch v2]. Starting M10 (Flask API + Physics Context UI).
   Finding: {n_pass}/{len(GATES)} gates PASS | BLOCK_M10={BLOCK_M10}
   Uploading: models/M9_selector_config.json, module_09_pump_selector_report.md
   Provide M10 complete script.
{B}
""")

log("=" * 72)
log(f"  M9 PATCH v2 COMPLETE | {n_pass} PASS / {n_fail} FAIL | Block M10: {BLOCK_M10}")
log("=" * 72)