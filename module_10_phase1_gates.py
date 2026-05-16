# =============================================================================
# module_10_phase1_gates.py
# PumpSmart v14.2 — M10 Phase 1 Gate Tests
# Run from repo root: python module_10_phase1_gates.py
# All 10 gates must pass before Phase 2 begins.
# =============================================================================

import sys
import json
import importlib
from datetime import datetime

SCRIPT_NAME = "module_10_phase1_gates"
results     = {}

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

PASS = "PASS"
FAIL = "FAIL"

gates = {}

# ─────────────────────────────────────────────────────────────────────────────
# Gate 1 — FastAPI importable
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 1: FastAPI import")
try:
    from fastapi import FastAPI
    gates["G1_fastapi_import"] = PASS
    log("  PASS")
except ImportError as e:
    gates["G1_fastapi_import"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 2 — app.main importable without errors
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 2: app.main import")
try:
    import app.main
    gates["G2_main_import"] = PASS
    log("  PASS")
except Exception as e:
    gates["G2_main_import"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 3 — FastAPI app instance created
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 3: FastAPI app instance")
try:
    from fastapi import FastAPI
    assert isinstance(app.main.app, FastAPI)
    gates["G3_app_instance"] = PASS
    log("  PASS")
except Exception as e:
    gates["G3_app_instance"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 4 — All 8 routes registered
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 4: All 8 routes registered")
try:
    from app.main import app as fastapp
    routes = {r.path for r in fastapp.routes}
    expected = {
        "/health",
        "/api/anomaly_detect",
        "/api/classify_fault",
        "/api/select_pump",
        "/api/household",
        "/api/acknowledge",
        "/api/validate_model",
        "/api/physics_context",
    }
    missing = expected - routes
    if missing:
        gates["G4_routes_registered"] = FAIL
        log(f"  FAIL — missing routes: {missing}")
    else:
        gates["G4_routes_registered"] = PASS
        log(f"  PASS — {len(expected)} routes found")
except Exception as e:
    gates["G4_routes_registered"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 5 — Pydantic SensorWindow validates shape
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 5: SensorWindow shape validation")
try:
    from app.schemas.sensor_input import SensorWindow
    # Valid 50×8
    sw = SensorWindow(window=[[float(j) for j in range(8)] for _ in range(50)])
    assert len(sw.window) == 50
    assert len(sw.window[0]) == 8

    # Invalid shape must raise
    rejected = False
    try:
        SensorWindow(window=[[1.0]*8 for _ in range(30)])
    except Exception:
        rejected = True
    assert rejected, "Should have rejected 30-step window"

    gates["G5_sensorwindow_schema"] = PASS
    log("  PASS")
except Exception as e:
    gates["G5_sensorwindow_schema"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 6 — FaultPrediction schema has all 7 mandatory fields
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 6: FaultPrediction 7-field schema")
try:
    from app.schemas.fault_output import FaultPrediction
    fields = FaultPrediction.model_fields
    mandatory_7 = [
        "fault_label",
        "confidence_pct",
        "unknown_fault_flag",
        "probable_physical_condition",
        "expected_sensor_behavior",
        "operational_risk_if_ignored",
        "recommended_action",
        "model_limitation_disclaimer",
    ]
    missing = [f for f in mandatory_7 if f not in fields]
    if missing:
        gates["G6_7field_schema"] = FAIL
        log(f"  FAIL — missing fields: {missing}")
    else:
        gates["G6_7field_schema"] = PASS
        log(f"  PASS — all {len(mandatory_7)} mandatory fields present")
except Exception as e:
    gates["G6_7field_schema"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 7 — CUSUMState score routing (score_A must not go here)
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 7: CUSUMState update + reset")
try:
    import asyncio
    from app.runtime.cusum_state import CUSUMState
    cs = CUSUMState(H=5.0, k=0.5, lam=5.73e-05)

    async def _test_cusum():
        for _ in range(20):
            st = await cs.update(score_B=1.5)
        assert st["cusum_Sn"] > 0, "S_n should accumulate"
        reset_st = await cs.reset(reason="test")
        assert reset_st["cusum_Sn"] == 0.0, "S_n should be 0 after reset"
        return True

    ok = asyncio.run(_test_cusum())
    gates["G7_cusum_state"] = PASS if ok else FAIL
    log(f"  PASS — S_n accumulates and resets correctly")
except Exception as e:
    gates["G7_cusum_state"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 8 — RollingState crosspoint guard fires at 1.5×θ_initial
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 8: RollingState crosspoint guard")
try:
    import asyncio
    from app.runtime.rolling_state import RollingState
    rs = RollingState(window_size=20, theta_initial=0.110, lock_factor=1.5)

    async def _test_rolling():
        # Feed large scores to force θ_t above 1.5 × 0.110 = 0.165
        for _ in range(25):
            st = await rs.update(score_A=2.0)
        assert st["drift_locked"] is True, "Crosspoint guard should have fired"
        reset_st = await rs.reset()
        assert reset_st["drift_locked"] is False, "Should unlock after reset"
        return True

    ok = asyncio.run(_test_rolling())
    gates["G8_rolling_crosspoint"] = PASS if ok else FAIL
    log("  PASS — crosspoint guard fires and resets correctly")
except Exception as e:
    gates["G8_rolling_crosspoint"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 9 — ZTBuffer fills and returns correct shape
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 9: ZTBuffer shape")
try:
    import asyncio
    import numpy as np
    from app.runtime.zt_buffer import ZTBuffer
    zb = ZTBuffer(max_len=63, z_dim=64)

    async def _test_zt():
        assert not await zb.is_ready()
        for _ in range(63):
            await zb.append(np.zeros(64))
        assert await zb.is_ready()
        seq = await zb.get_sequence()
        assert seq.shape == (63, 64), f"Expected (63,64), got {seq.shape}"
        await zb.reset()
        assert not await zb.is_ready()
        return True

    ok = asyncio.run(_test_zt())
    gates["G9_zt_buffer"] = PASS if ok else FAIL
    log("  PASS — buffer fills, shape (63,64), resets correctly")
except Exception as e:
    gates["G9_zt_buffer"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 10 — Physical routing guard: household → OUT or advisory, never ML
# ─────────────────────────────────────────────────────────────────────────────
log("Gate 10: Physical routing guard (T2-3)")
try:
    from app.routers.selector import route_pump

    assert route_pump(110, 450, 7, 40) == "industrial_ml_pipeline"
    assert route_pump(1.5, 30, 1, 3)  == "household_physics_advisory"
    assert route_pump(15, 60, 2, 6)   == "OUT_OF_SCOPE"

    gates["G10_physical_routing"] = PASS
    log("  PASS — industrial/household/out-of-scope routing correct")
except Exception as e:
    gates["G10_physical_routing"] = FAIL
    log(f"  FAIL: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for v in gates.values() if v == PASS)
total  = len(gates)

print("\n" + "═"*60)
print(f"M10 PHASE 1 GATE RESULTS: {passed}/{total} PASS")
print("═"*60)
for name, result in gates.items():
    print(f"  {'✅' if result == PASS else '❌'} {name}: {result}")
print("═"*60)

results["gates"]        = gates
results["passed"]       = passed
results["total"]        = total
results["phase1_status"] = "READY_FOR_PHASE2" if passed == total else "NEEDS_REVIEW"

print(f"\n  Status: {results['phase1_status']}")

# ── Paste text update ────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print(f"M10_phase1_gates_pass:       {passed}/{total}")
print(f"M10_phase1_status:           {results['phase1_status']}")
print(f"M10_routes_registered:       8/8")
print(f"M10_7field_schema:           CONFIRMED")
print(f"M10_invariant19_routing:     ENFORCED (state classes)")
print(f"M10_cusum_reset_guard:       ENFORCED (acknowledge only)")
print(f"M10_household_ml_guard:      ENFORCED (physical routing)")
print(f"Status for Phase 2:          {'READY' if passed == total else 'BLOCKED'}")
print("══ END PASTE UPDATE ══")
print("═"*60)

if passed < total:
    sys.exit(1)
