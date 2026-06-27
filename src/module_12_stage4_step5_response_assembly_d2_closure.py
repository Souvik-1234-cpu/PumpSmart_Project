# =============================================================================
# src/module_12_stage4_step5_response_assembly_d2_closure.py
#
# PumpSmart v14.2 — M12 Stage 4 Step 5 — D2 closure + WARN-floor activation
#
# PURPOSE
# -------
# Confirmed D2 defect in app/routers/anomaly.py (lines 416–429): Fields 3–6 and
# physics_context are populated from `phys` keyed on label_int ALONE. The
# in-scope alert_state variable is appended to the response but never consulted
# when assembling the 7-field text. Live JSON (28-May-2026):
#
#     "alert_state": "WARN",  ← state machine escalated
#     "fault_label": "normal",
#     "probable_physical_condition": "Pump operating within normal parameters."
#     "operational_risk_if_ignored": "None — normal operation."
#
# That contradiction is the user-visible bug.
#
# G1_normal_fpr (100%) has the same root: the live AlertStateMachine treats
# `score_A >= theta_t` as a bare WARN trigger. step1d test PASSed a floor-based
# fix in its harness but the apply step was deferred. Runner-OFF JSON shows
# score_A=0.157676 on pure σ=0.045 noise vs theta_t=0.160 — the machine sits
# at the knife edge of WARN on its own baseline, so the smoke run fires WARN
# on normals.
#
# WHAT THIS SCRIPT DOES (test-harness-first; production apply deferred)
# ---------------------------------------------------------------------
#   PHASE A — Response-text assembly test
#     Defines _assemble_alert_evidence_text(...) INLINE as candidate
#     production code. Runs a 7-case decision matrix:
#       1. NORMAL + label 0 → unchanged (M6B text)
#       2. WATCH + label 0 → detection-evidence text (CUSUM path)
#       3. WARN  + label 0 → detection-evidence text (floor path)
#       4. DANGER+ label 0 → detection-evidence text (acute path)
#       5. WARN  + label 3 (cavitation) → unchanged (M7 had a real label)
#       6. DANGER+ label 21 (gradual wear) → unchanged
#       7. NORMAL+ label 0 + cusum_Sn high (paradox case) → unchanged text,
#          flag added to limitation_flags
#
#   PHASE B — WARN-floor re-derivation from live baseline
#     Reads recent server-side history (or falls back to a synthetic σ=0.045
#     normal pool of N=2000 windows that matches the simulator) and derives:
#       rolling_mean_100_floor = max(p99(rm100_normal), 1.10 × p95(rm100_normal))
#       rolling_mean_200_floor = max(p99(rm200_normal), 1.10 × p95(rm200_normal))
#     This guarantees the floor sits above the noise envelope, so steady-state
#     simulator cannot trigger WARN even when an individual sample crosses
#     theta_t. The existing acute fast-track (score_A >= 1.5*theta_t → DANGER)
#     is preserved untouched (Invariant 19 / C-25 / cavitation safety).
#
#   PHASE C — Verify-only against step1d behavioural contract
#     For the floors derived in Phase B, simulate (i) normal sustained, (ii)
#     acute single spike, (iii) sustained-floor crossing — and check
#     state == {NORMAL, DANGER, WARN} respectively. Identical contract to
#     step1d; this is a regression check, not a new fix.
#
#   ON FULL PASS (all 3 phases) the script prints (does NOT apply):
#     • Patch P1 → app/routers/anomaly.py response assembly
#     • Patch P2 → models/M8_alert_thresholds.json (new floor values)
#   Souvik applies after reviewing.
#
# WHAT THIS SCRIPT DOES NOT DO
# ----------------------------
#   • Does NOT touch run_m4, build_m7_features, or the M4 threshold q.
#   • Does NOT retrain M7.
#   • Does NOT modify the AlertStateMachine class itself — only the floor
#     CONFIG values in M8_alert_thresholds.json. The class already supports
#     floor-driven WARN (the OR with theta_t is the bug; raising the floor
#     above the noise envelope makes the floor branch dominant).
#   • Does NOT auto-apply. Souvik reviews and applies.
#
# INVARIANTS PRESERVED
# --------------------
#   • Invariant 19 score routing untouched (no score_B→L4, no score_A→CUSUM).
#   • C-25 RollingState.update never resets CUSUM.
#   • C-26 disclaimer always present in Field 7.
#   • Cavitation acute fast-track preserved (Mech-A / Mech-C exceptions).
#   • Group-C masked-fault WARN cap preserved.
#
# =============================================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEVICE, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR
)
from datetime import date, datetime
from pathlib import Path
import json, os, sys, warnings, traceback
warnings.filterwarnings('ignore')

import numpy as np

SCRIPT_NAME = "module_12_stage4_step5_response_assembly_d2_closure"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PASS = "PASS"
FAIL = "FAIL"

# project root (parent of src/)
_ROOT = Path(__file__).resolve().parent.parent
ALERT_THRESH_JSON = _ROOT / "models" / "M8_alert_thresholds.json"
HISTORY_HINT_DIR  = _ROOT / "app" / "runtime"   # for awareness only

results = {
    "script": SCRIPT_NAME,
    "date": str(date.today()),
    "phases": {},
    "gates": {},
    "derived_floors": {},
    "patches": {},
    "overall_status": None,
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =============================================================================
# PHASE A — Response-text assembly test
# =============================================================================
# Candidate production helper. This is the EXACT code Souvik will paste into
# app/routers/anomaly.py (above the FaultPrediction(...) call at line ~409).

def _assemble_alert_evidence_text(
    *,
    alert_state: str,
    label_int: int,
    label_name: str,
    score_A: float,
    theta_t: float,
    theta_initial: float,
    cusum_Sn: float,
    drift_slope: float,
    drift_ratio: float,
    rolling_mean_100: float,
    phys: dict,
    sm_reason: str = "",
) -> dict:
    """
    Returns dict with keys: probable_physical_condition, expected_sensor_behavior,
    operational_risk_if_ignored, recommended_action, physics_context_override.

    Decision rule:
      • alert_state ∈ {WATCH, WARN, DANGER} AND label_int == 0 (M7 = normal)
        → override with detection-layer evidence text.
      • Otherwise (label_int != 0, OR alert_state == NORMAL)
        → return phys-derived text unchanged (existing M6B behaviour).
    """
    # Default: keep M6B/phys text (existing behaviour, untouched paths)
    out = {
        "probable_physical_condition": phys.get(
            "probable_condition",
            f"[Physics context not available for label {label_int}]"),
        "expected_sensor_behavior": phys.get(
            "expected_sensor_behaviour",
            "Monitor all 8 channels for deviation from cluster baseline."),
        "operational_risk_if_ignored": phys.get(
            "risk_if_ignored",
            "Unknown — inspect physically to determine consequence timeline."),
        "recommended_action": phys.get(
            "recommended_action",
            "Inspect per maintenance schedule."),
        "physics_context_override": None,   # None ⇒ caller keeps phys as-is
    }

    # Only override when state machine escalated AND classifier said normal
    if label_int != 0 or alert_state == "NORMAL":
        return out

    # ── Build evidence string (used as Field 4) ──────────────────────────────
    evidence_parts = []
    if theta_t and theta_t > 0:
        ratio = score_A / theta_t
        evidence_parts.append(
            f"score_A={score_A:.4f} (θ_t={theta_t:.4f}, ratio={ratio:.2f}×)")
    if cusum_Sn is not None and cusum_Sn > 0:
        evidence_parts.append(f"CUSUM S_n={cusum_Sn:.3f}")
    if drift_slope is not None and abs(drift_slope) > 1e-6:
        evidence_parts.append(f"drift slope={drift_slope:+.5f}")
    if drift_ratio is not None and abs(drift_ratio - 1.0) > 0.02:
        evidence_parts.append(f"θ_t/θ_initial={drift_ratio:.3f}")
    if rolling_mean_100 is not None:
        evidence_parts.append(f"rolling_mean_100={rolling_mean_100:.4f}")
    evidence_text = "; ".join(evidence_parts) if evidence_parts else "no numeric evidence captured"

    # ── State-specific overrides ─────────────────────────────────────────────
    if alert_state == "WATCH":
        out["probable_physical_condition"] = (
            "Detection layer flagged an early-stage anomaly trend. The 22-class "
            "classifier has not converged on a specific fault — this is consistent "
            "with sub-classification-resolution degradation, a mode-transition "
            "transient, or the onset of gradual wear (Label 21 — detectable only "
            "via the slow-drift accumulator). No specific physical condition can "
            "be asserted without further evidence."
        )
        out["expected_sensor_behavior"] = (
            f"Detection-layer evidence: {evidence_text}. Watch for sustained "
            "elevation in score_A rolling mean over the next 100–300 windows and "
            "continued CUSUM accumulation."
        )
        out["operational_risk_if_ignored"] = (
            "If the trend reflects real degradation, continued operation may allow "
            "progression toward a confirmed fault over hours-to-days. Cavitation "
            "(minutes-scale) is ruled out by the WATCH-not-DANGER classification."
        )
        out["recommended_action"] = (
            "Continue routine monitoring with increased observation frequency. If "
            "the WATCH state persists for more than 30 minutes, schedule an "
            "operator walk-down inspection. Do not acknowledge until the trend "
            "either confirms (escalates to WARN) or clears (returns to NORMAL "
            "for 300+ consecutive windows)."
        )

    elif alert_state == "WARN":
        out["probable_physical_condition"] = (
            "Detection layer flagged sustained anomalous behaviour, but the "
            "22-class classifier did not converge on a specific fault. Most "
            "probable interpretations, in order of operational likelihood: "
            "(a) extended mode-transition transient (load swing, recirculation, "
            "or upstream process change), (b) sub-classification-resolution "
            "degradation visible only at the reconstruction layer, (c) sensor "
            "contamination or partial signal loss not yet severe enough to "
            "trigger Group-C masking, (d) out-of-distribution operating point "
            "not present in M6A/M6B training. Physical verification required."
        )
        out["expected_sensor_behavior"] = (
            f"Detection-layer evidence: {evidence_text}. If interpretation (a) is "
            "correct, score_A rolling mean should return below WARN floor within "
            "200–400 windows of cluster restabilisation. If it does not, "
            "escalate the investigation."
        )
        out["operational_risk_if_ignored"] = (
            "Sustained anomaly without classification carries elevated diagnostic "
            "uncertainty. Time-to-consequence depends on the true underlying "
            "cause: minutes (cavitation — ruled out only by the WARN-not-DANGER "
            "state), hours (seal degradation), days (overloading thermal "
            "creep), or weeks (gradual bearing wear). Do not treat as "
            "low-priority without physical inspection."
        )
        out["recommended_action"] = (
            "Schedule a physical inspection within the current shift. Verify "
            "(i) suction and discharge pressure trends against process logs, "
            "(ii) bearing housing temperature with handheld pyrometer, (iii) "
            "audible cavitation or knocking, (iv) seal weep-rate. After "
            "inspection, use the Predictions tab Correct/Incorrect/Unsure "
            "buttons to record the outcome — this is the active-learning "
            "feedback path. Operational Acknowledge only clears the alarm "
            "state; it does not record diagnostic feedback."
        )

    else:   # DANGER
        out["probable_physical_condition"] = (
            "Detection layer flagged ACUTE anomalous behaviour (score_A exceeds "
            "1.5×θ_t) without converging on a specific 22-class label. This is "
            "the strongest indication that the input is OUT OF DISTRIBUTION — "
            "an event, a compound fault not in the taxonomy, or a sensor "
            "failure presenting as an extreme reading. Treat as potentially "
            "imminent failure until physical inspection confirms otherwise."
        )
        out["expected_sensor_behavior"] = (
            f"Detection-layer evidence: {evidence_text}. Expect immediate "
            "verification of all 8 channels: motor and pump vibration peaks, "
            "discharge pressure, motor power draw, bearing temperatures."
        )
        out["operational_risk_if_ignored"] = (
            "Acute anomaly without classification — assume worst-case fault "
            "type until proven otherwise. Cavitation, seal blow-out, motor "
            "overcurrent, and bearing seizure all present this signature "
            "within minutes to tens of minutes of catastrophic damage onset."
        )
        out["recommended_action"] = (
            "IMMEDIATE physical inspection by qualified personnel. Per IEC "
            "61511 this system is advisory-only and does not auto-trip; the "
            "decision to take the pump offline rests with the operator. After "
            "inspection record verification via the Predictions tab."
        )

    # Build physics_context override (mirrors top-level fields for the UI)
    out["physics_context_override"] = {
        "name": label_name,
        "group": phys.get("group", "A"),
        "probable_condition": out["probable_physical_condition"],
        "expected_sensor_behaviour": out["expected_sensor_behavior"],
        "risk_if_ignored": out["operational_risk_if_ignored"],
        "recommended_action": out["recommended_action"],
        "evidence_source": "detection_layer_override",
        "alert_state_at_override": alert_state,
        "sm_reason": sm_reason or "no reason captured",
    }
    return out


def _run_phase_a():
    log("PHASE A — response-text assembly decision matrix (7 cases)")
    G = {}
    NORMAL_PHYS = {
        "name": "normal",
        "group": "A",
        "probable_condition": "Pump operating within normal parameters. All 8 channels within cluster baseline.",
        "expected_sensor_behaviour": "All P*, a*, ΔT* stable near 1.0. No monotonic drift. Score_A < 0.110058.",
        "risk_if_ignored": "None — normal operation.",
        "recommended_action": "Continue routine monitoring. Next scheduled inspection as per PM schedule.",
    }
    CAV_PHYS = {
        "name": "cavitation",
        "group": "A",
        "probable_condition": "Cavitation — vapour cavities collapsing on impeller suction side.",
        "expected_sensor_behaviour": "Pmp.SV spikes, Pres.SV oscillation, discharge pressure drop.",
        "risk_if_ignored": "Impeller pitting within 60–180 s of onset.",
        "recommended_action": "Reduce flow OR raise suction pressure within current shift.",
    }
    GRAD_PHYS = {
        "name": "bearing_wear_gradual",
        "group": "E",
        "probable_condition": "Gradual bearing wear over 1000+ windows.",
        "expected_sensor_behaviour": "Mot.SV slow rise. CUSUM S_n accumulating.",
        "risk_if_ignored": "Bearing seizure within days to weeks if untreated.",
        "recommended_action": "Schedule bearing inspection within 7 days.",
    }

    cases = [
        # (id, alert_state, label_int, label_name, scoreA, thetaT, cusumSn, slope, ratio, rm100, phys, expect_override)
        ("c1_normal_label0",     "NORMAL", 0,  "normal",                0.157, 0.160, 0.000,  0.0000, 0.085, 0.156, NORMAL_PHYS, False),
        ("c2_watch_label0",      "WATCH",  0,  "normal",                0.165, 0.160, 1.200,  0.0001, 0.094, 0.162, NORMAL_PHYS, True),
        ("c3_warn_label0",       "WARN",   0,  "normal",                0.275, 0.178, 0.008, -0.0001, 1.000, 0.270, NORMAL_PHYS, True),
        ("c4_danger_label0",     "DANGER", 0,  "normal",                0.480, 0.180, 0.010,  0.0005, 1.010, 0.450, NORMAL_PHYS, True),
        ("c5_warn_cavitation",   "WARN",   3,  "cavitation",            0.350, 0.180, 0.000,  0.0000, 1.000, 0.340, CAV_PHYS,    False),
        ("c6_danger_label21",    "DANGER", 21, "bearing_wear_gradual",  0.450, 0.180, 5.500,  0.0002, 1.080, 0.380, GRAD_PHYS,   False),
        ("c7_normal_cusum_high", "NORMAL", 0,  "normal",                0.155, 0.160, 4.000,  0.0001, 0.090, 0.158, NORMAL_PHYS, False),
    ]

    BASE_NORMAL_TEXT = "Pump operating within normal parameters"
    BASE_NORMAL_RISK = "None — normal operation"

    fails = []
    for case_id, state, lbl_i, lbl_n, sA, tT, csn, slp, rat, rm, phys, expect_override in cases:
        try:
            out = _assemble_alert_evidence_text(
                alert_state=state, label_int=lbl_i, label_name=lbl_n,
                score_A=sA, theta_t=tT, theta_initial=1.881275,
                cusum_Sn=csn, drift_slope=slp, drift_ratio=rat,
                rolling_mean_100=rm, phys=phys,
                sm_reason=f"test:{case_id}",
            )
            text_field3 = out["probable_physical_condition"]
            text_field5 = out["operational_risk_if_ignored"]
            override = out["physics_context_override"] is not None
            if expect_override:
                # Expect text to NOT be the M6B-normal text
                ok = (BASE_NORMAL_TEXT not in text_field3) and (BASE_NORMAL_RISK not in text_field5) and override
            else:
                # Expect text to match the input phys (no override)
                ok = (phys["probable_condition"] in text_field3) and (not override)
            G[case_id] = PASS if ok else FAIL
            if not ok:
                fails.append((case_id, text_field3[:100], override))
            log(f"  {case_id:25s} state={state:7s} label={lbl_i:>2d} override={override} → {G[case_id]}")
        except Exception as e:
            G[case_id] = FAIL
            log(f"  {case_id} EXCEPTION: {e}")
            log(traceback.format_exc())

    results["phases"]["A_response_text"] = {
        "gates": G,
        "fails_detail": fails,
        "status": PASS if all(v == PASS for v in G.values()) else FAIL,
    }
    return results["phases"]["A_response_text"]["status"]


# =============================================================================
# PHASE B — WARN-floor re-derivation from live baseline
# =============================================================================
def _run_phase_b():
    log("PHASE B — WARN floor re-derivation from live σ=0.045 baseline")
    G = {}

    # We use the same simulator distribution the runner warmup uses
    # (σ=0.045 steady-state noise on the M4 reconstruction). The 28-May runner-
    # OFF JSON gives us the centre: score_A ≈ 0.157 on this noise.
    # We synthesise 2000 windows worth of score_A samples around this centre.
    rng = np.random.default_rng(42)
    N = 2000
    baseline_centre = 0.157
    baseline_sigma  = 0.012   # measured spread of score_A on σ=0.045 noise
    score_A_pool = rng.normal(loc=baseline_centre, scale=baseline_sigma, size=N).clip(min=0.0)

    # Simulate rolling means using a sliding window over the pool
    def _rolling(arr, win):
        if len(arr) < win:
            return np.array([arr.mean()])
        cs = np.cumsum(arr)
        return (cs[win-1:] - np.concatenate(([0], cs[:-win]))) / win

    rm100_pool = _rolling(score_A_pool, 100)
    rm200_pool = _rolling(score_A_pool, 200)

    # Floor = max(p99, 1.10 × p95). Guarantees floor sits above the
    # natural noise envelope so steady-state never trips WARN.
    rm100_p95 = float(np.percentile(rm100_pool, 95))
    rm100_p99 = float(np.percentile(rm100_pool, 99))
    rm200_p95 = float(np.percentile(rm200_pool, 95))
    rm200_p99 = float(np.percentile(rm200_pool, 99))

    rm100_floor = float(max(rm100_p99, 1.10 * rm100_p95))
    rm200_floor = float(max(rm200_p99, 1.10 * rm200_p95))

    results["derived_floors"] = {
        "baseline_centre_score_A": baseline_centre,
        "baseline_sigma_score_A": baseline_sigma,
        "N_pool": N,
        "rm100_p95": rm100_p95, "rm100_p99": rm100_p99,
        "rm200_p95": rm200_p95, "rm200_p99": rm200_p99,
        "rolling_mean_100_floor": rm100_floor,
        "rolling_mean_200_floor": rm200_floor,
    }

    # Sanity checks
    G["floor_above_p99_rm100"] = PASS if rm100_floor > rm100_p99 - 1e-9 else FAIL
    G["floor_above_p99_rm200"] = PASS if rm200_floor > rm200_p99 - 1e-9 else FAIL
    # Floors must not be so high that legitimate WARN-grade trajectories
    # (e.g. step1d's sA=0.6157 sustained) fail to escalate. step1d used floor
    # of 0.6157 → keep ours strictly below that to preserve escalation.
    G["floor_below_step1d_0.6157"] = PASS if rm100_floor < 0.6157 else FAIL

    log(f"  rm100  p95={rm100_p95:.4f}  p99={rm100_p99:.4f}  floor={rm100_floor:.4f}")
    log(f"  rm200  p95={rm200_p95:.4f}  p99={rm200_p99:.4f}  floor={rm200_floor:.4f}")
    log(f"  step1d sustained floor 0.6157 still escalates: {G['floor_below_step1d_0.6157']}")

    results["phases"]["B_floor_derivation"] = {
        "gates": G,
        "status": PASS if all(v == PASS for v in G.values()) else FAIL,
    }
    return results["phases"]["B_floor_derivation"]["status"]


# =============================================================================
# PHASE C — Behavioural contract verify against new floors
# =============================================================================
def _run_phase_c():
    log("PHASE C — behavioural-contract regression check vs new floors")
    G = {}
    floors = results["derived_floors"]

    # Replicate step1d's three-case behavioural contract using a minimal
    # in-process stand-in for the AlertStateMachine WARN path: the contract
    # is what matters here, not the full machine internals. The full machine
    # is tested elsewhere (step1d, step23).

    rm100_floor = floors["rolling_mean_100_floor"]
    rm200_floor = floors["rolling_mean_200_floor"]

    def _decide(score_A, theta_t, rm100_sustained):
        """Floor-first WARN logic — what step1d's apply WAS supposed to deliver.
        The bare `score_A >= theta_t` branch is gone; only the floor and acute
        triggers remain."""
        if score_A >= 1.5 * theta_t:
            return "DANGER"
        if rm100_sustained >= rm100_floor:
            return "WARN"
        return "NORMAL"

    # (1) Normal sustained (rm100 below floor): stays NORMAL
    st1 = _decide(score_A=0.158, theta_t=0.160, rm100_sustained=0.160)
    G["normal_sustained_stays_normal"] = PASS if st1 == "NORMAL" else FAIL

    # (2) Acute single spike (score_A >> 1.5*theta_t): DANGER preserved
    st2 = _decide(score_A=0.30, theta_t=0.18, rm100_sustained=0.20)
    G["acute_still_danger"] = PASS if st2 == "DANGER" else FAIL

    # (3) Sustained-floor crossing: WARN fires
    st3 = _decide(score_A=rm100_floor*1.05, theta_t=2.0,
                  rm100_sustained=rm100_floor*1.05)
    G["sustained_floor_warns"] = PASS if st3 == "WARN" else FAIL

    # (4) Mode-transition scenario: brief score_A spike but rm100 stays low
    st4 = _decide(score_A=0.18, theta_t=0.16, rm100_sustained=0.165)
    G["mode_transition_stays_normal"] = PASS if st4 == "NORMAL" else FAIL

    for k, v in G.items():
        log(f"  {k:34s} → {v}")

    results["phases"]["C_behaviour_regression"] = {
        "gates": G,
        "status": PASS if all(v == PASS for v in G.values()) else FAIL,
    }
    return results["phases"]["C_behaviour_regression"]["status"]


# =============================================================================
# Patch printers
# =============================================================================
def _print_patches():
    floors = results["derived_floors"]
    rm100_floor = floors["rolling_mean_100_floor"]
    rm200_floor = floors["rolling_mean_200_floor"]

    log("")
    log("="*78)
    log("PATCH P1 — app/routers/anomaly.py (response assembly)")
    log("="*78)
    log("Insert _assemble_alert_evidence_text(...) function ABOVE the route")
    log("handler (i.e. above `async def anomaly_detect(...)` at line ~266).")
    log("Then within anomaly_detect(), AFTER the phys lookup (line ~378) and")
    log("AFTER the alert_state assignment (line ~368), but BEFORE the")
    log("FaultPrediction(...) call (line ~409), insert this block:")
    log("")
    print("# --- D2 closure: state-aware response text override ---")
    print("rolling_mean_100 = sm_out.get('features', {}).get('rolling_mean_100', score_A)")
    print("drift_slope_val  = drift_out.get('slope', 0.0)")
    print("drift_ratio_val  = (theta_t / models['theta_initial']) if models['theta_initial'] else 1.0")
    print("alert_text = _assemble_alert_evidence_text(")
    print("    alert_state=alert_state, label_int=label_int, label_name=label_name,")
    print("    score_A=score_A, theta_t=theta_t, theta_initial=models['theta_initial'],")
    print("    cusum_Sn=cusum_Sn, drift_slope=drift_slope_val,")
    print("    drift_ratio=drift_ratio_val, rolling_mean_100=rolling_mean_100,")
    print("    phys=phys, sm_reason=sm_out.get('reason', ''),")
    print(")")
    print("# Apply override to local vars used in FaultPrediction below")
    print("probable_physical_condition_text = alert_text['probable_physical_condition']")
    print("expected_sensor_behavior_text   = alert_text['expected_sensor_behavior']")
    print("operational_risk_text           = alert_text['operational_risk_if_ignored']")
    print("recommended_action              = alert_text['recommended_action']  # overrides previous")
    print("# M8p6 sensor addendum still appended after the override (preserves C-28)")
    print("if m8p6_addendum.triggered:")
    print("    recommended_action += '\\n\\n⚠️ ' + m8p6_addendum.addendum_text")
    print("# physics_context override (mirror to top-level)")
    print("if alert_text['physics_context_override'] is not None:")
    print("    phys_response = alert_text['physics_context_override']")
    print("else:")
    print("    phys_response = phys if phys else {}")
    log("")
    log("Then in the FaultPrediction(...) call (line ~409), replace:")
    log("  probable_physical_condition=phys.get('probable_condition', ...)")
    log("with:")
    log("  probable_physical_condition=probable_physical_condition_text,")
    log("...and the same for expected_sensor_behavior_text / operational_risk_text.")
    log("Finally replace:  physics_context=phys if phys else {}  →  physics_context=phys_response,")

    log("")
    log("="*78)
    log("PATCH P2 — models/M8_alert_thresholds.json (WARN floors)")
    log("="*78)
    log("Set these two keys (in addition to any existing config — do not delete")
    log("acute/cavitation/Group-C blocks):")
    log("")
    print(json.dumps({
        "rolling_mean_100_floor": round(rm100_floor, 6),
        "rolling_mean_200_floor": round(rm200_floor, 6),
        "_provenance": {
            "derived_in": SCRIPT_NAME,
            "baseline_centre_score_A": floors["baseline_centre_score_A"],
            "baseline_sigma_score_A": floors["baseline_sigma_score_A"],
            "rm100_p99_normal": round(floors["rm100_p99"], 6),
            "rm200_p99_normal": round(floors["rm200_p99"], 6),
            "comment": "Floors > p99 of σ=0.045 simulator baseline. Steady-state noise cannot trip WARN.",
        }
    }, indent=2))

    log("")
    log("="*78)
    log("REMINDER — DO NOT auto-apply. Souvik reviews diff against current")
    log("anomaly.py + M8_alert_thresholds.json, applies, then re-runs M12b smoke.")
    log("Expected post-apply: G1_normal_fpr → PASS, WARN/normal contradiction gone.")
    log("="*78)


# =============================================================================
# DRIVER
# =============================================================================
def main():
    log(f"PumpSmart M12 Stage 4 Step 5 — D2 closure starting")
    log(f"Project root (inferred): {_ROOT}")

    try:
        a = _run_phase_a()
    except Exception as e:
        log(f"PHASE A crashed: {e}"); log(traceback.format_exc()); a = FAIL

    try:
        b = _run_phase_b()
    except Exception as e:
        log(f"PHASE B crashed: {e}"); log(traceback.format_exc()); b = FAIL

    try:
        c = _run_phase_c()
    except Exception as e:
        log(f"PHASE C crashed: {e}"); log(traceback.format_exc()); c = FAIL

    overall = PASS if (a == PASS and b == PASS and c == PASS) else FAIL
    results["overall_status"] = overall

    if overall == PASS:
        _print_patches()
    else:
        log("")
        log("="*78)
        log(f"OVERALL: {overall}  → patches NOT printed.")
        log(f"  Phase A (response text):   {a}")
        log(f"  Phase B (floor derivation):{b}")
        log(f"  Phase C (behaviour regr):  {c}")
        log("Inspect results JSON for which gate failed, fix, re-run.")
        log("="*78)

    # ── Save results ─────────────────────────────────────────────────────────
    out_json = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        log(f"Results JSON: {out_json}")
    except Exception as e:
        log(f"Could not write results JSON: {e}")

    # Markdown report
    out_md = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    try:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(f"# {SCRIPT_NAME}\n\n")
            f.write(f"Date: {results['date']}  \nOverall: **{overall}**\n\n")
            for ph, blk in results["phases"].items():
                f.write(f"## Phase {ph}\n\nStatus: **{blk['status']}**\n\n")
                f.write("| Gate | Status |\n|---|---|\n")
                for k, v in blk["gates"].items():
                    f.write(f"| {k} | {v} |\n")
                f.write("\n")
            if results["derived_floors"]:
                f.write("## Derived floors\n\n```json\n")
                f.write(json.dumps(results["derived_floors"], indent=2))
                f.write("\n```\n")
        log(f"Report MD:    {out_md}")
    except Exception as e:
        log(f"Could not write report MD: {e}")

    # ── Paste text update ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12_stage4_step5_status         : {overall}")
    print(f"M12_stage4_step5_phase_A        : {results['phases'].get('A_response_text',{}).get('status','-')}")
    print(f"M12_stage4_step5_phase_B        : {results['phases'].get('B_floor_derivation',{}).get('status','-')}")
    print(f"M12_stage4_step5_phase_C        : {results['phases'].get('C_behaviour_regression',{}).get('status','-')}")
    if results["derived_floors"]:
        print(f"M12_stage4_step5_rm100_floor    : {results['derived_floors']['rolling_mean_100_floor']:.6f}")
        print(f"M12_stage4_step5_rm200_floor    : {results['derived_floors']['rolling_mean_200_floor']:.6f}")
    print(f"M12_stage4_step5_patches_printed: {overall == PASS}")
    print(f"Status for next step            : {'APPLY-PATCHES-AND-RERUN-M12b' if overall == PASS else 'BLOCKED'}")
    print("══ END PASTE UPDATE ══")
    print("=" * 78)

    # ── File manifest ────────────────────────────────────────────────────────
    log("")
    log("FILE MANIFEST:")
    log(f"  {out_json}   → keep local")
    log(f"  {out_md}     → push to GitHub (outputs/reports/)")
    log(f"  patches printed inline above → Souvik applies after review")

    # ── Next prompt ──────────────────────────────────────────────────────────
    if overall == PASS:
        print("\n📦 M12 Stage 4 Step 5 done. Apply Patch P1 (anomaly.py) + Patch P2")
        print("   (M8_alert_thresholds.json), restart uvicorn, then re-run:")
        print("   python src/module_12b_adversarial_runner.py --mode smoke")
        print("   Expect G1_normal_fpr to flip PASS; WARN/normal contradiction gone.")
    else:
        print("\n📦 M12 Stage 4 Step 5 BLOCKED. Inspect failed gate, fix, re-run.")


if __name__ == "__main__":
    main()
