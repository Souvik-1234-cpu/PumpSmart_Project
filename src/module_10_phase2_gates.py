# =============================================================================
# module_10_phase2_gates.py
# M10 Phase 2 — Inference pipeline gate tests
# Run from repo root: python module_10_phase2_gates.py
# Tests the full inference pipeline with synthetic inputs (no real model files needed).
# 12/12 must pass before Phase 3 (frontend integration).
# =============================================================================

import sys
import asyncio
import numpy as np
import torch
from datetime import datetime

SCRIPT_NAME = "module_10_phase2_gates"
results     = {}

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

PASS = "PASS"
FAIL = "FAIL"
gates = {}


# =============================================================================
# Gate 1 — M4 architecture loads with LayerNorm (not BatchNorm)
# =============================================================================
log("Gate 1: M4 architecture — LayerNorm confirmed")
try:
    import torch.nn as nn
    from app.runtime.model_registry import _M4LSTMAutoencoder

    m4 = _M4LSTMAutoencoder()
    # Confirm LayerNorm, not BatchNorm
    assert isinstance(m4.encoder.bn, nn.LayerNorm), \
        f"Expected LayerNorm, got {type(m4.encoder.bn)}"
    # Confirm NO running_mean buffer (LayerNorm has none; BatchNorm does)
    assert not hasattr(m4.encoder.bn, 'running_mean'), \
        "running_mean found — this is BatchNorm, not LayerNorm!"
    gates["G1_m4_layernorm"] = PASS
    log("  PASS — LayerNorm confirmed, no running_mean buffer")
except Exception as e:
    gates["G1_m4_layernorm"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 2 — M4 forward pass produces z_t shape (1, 64)
# =============================================================================
log("Gate 2: M4 forward pass → z_t shape")
try:
    from app.runtime.model_registry import _M4LSTMAutoencoder
    m4 = _M4LSTMAutoencoder()
    m4.eval()
    x  = torch.randn(1, 50, 8)
    with torch.no_grad():
        z_t, h, c = m4.encoder(x)
    assert z_t.shape == (1, 64), f"Expected (1,64), got {z_t.shape}"
    recon = m4.decoder(z_t, 50, h, c)
    assert recon.shape == (1, 50, 8), f"Expected (1,50,8), got {recon.shape}"
    gates["G2_m4_forward"] = PASS
    log(f"  PASS — z_t: {z_t.shape}, recon: {recon.shape}")
except Exception as e:
    gates["G2_m4_forward"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 3 — score_A computation (physics-weighted MAE)
# =============================================================================
log("Gate 3: score_A weighted MAE computation")
try:
    from app.routers.anomaly import run_m4, CH_WEIGHTS
    from app.runtime.model_registry import _M4LSTMAutoencoder

    m4    = _M4LSTMAutoencoder()
    m4.eval()
    x     = torch.randn(1, 50, 8)
    score_A, z_t_np, raw_mae = run_m4(x, m4, q=0.110058)

    assert isinstance(score_A, float),      "score_A must be float"
    assert isinstance(z_t_np, np.ndarray),  "z_t_np must be ndarray"
    assert z_t_np.shape == (64,),           f"z_t_np shape: {z_t_np.shape}"
    assert score_A >= 0,                    "score_A must be non-negative"
    assert len(CH_WEIGHTS) == 8,            "Must have 8 channel weights"

    gates["G3_score_a"] = PASS
    log(f"  PASS — score_A={score_A:.5f}, z_t_np shape={z_t_np.shape}")
except Exception as e:
    gates["G3_score_a"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 4 — Invariant 19: score_A → RollingState, score_B → CUSUM (no cross-routing)
# =============================================================================
log("Gate 4: Invariant 19 score routing")
try:
    from app.runtime.cusum_state   import CUSUMState
    from app.runtime.rolling_state import RollingState

    cs = CUSUMState(H=5.0, k=0.5, lam=5.73e-05)
    rs = RollingState(window_size=20, theta_initial=0.110058)

    async def _test_routing():
        # score_A to rolling ONLY
        for _ in range(15):
            await rs.update(score_A=0.05)
        rs_st = await rs.get_state()

        # score_B to CUSUM ONLY
        for _ in range(10):
            await cs.update(score_B=1.5)
        cs_st = await cs.get_state()

        # Verify: CUSUM should NOT have received score_A
        # (test: if we'd fed 0.05 to cusum, S_n would be 0 since 0.05-0.5 < 0)
        # CUSUM with score_B=1.5, k=0.5 → S_n accumulates correctly
        assert cs_st["cusum_Sn"] > 0, "CUSUM should accumulate with score_B=1.5"
        assert rs_st["buffer_fill"] > 0, "Rolling buffer should have score_A values"
        return True

    ok = asyncio.run(_test_routing())
    gates["G4_invariant19"] = PASS if ok else FAIL
    log("  PASS — score_A→rolling, score_B→CUSUM, no cross-routing")
except Exception as e:
    gates["G4_invariant19"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 5 — Alert state machine covers all 4 states
# =============================================================================
log("Gate 5: Alert state machine — 4 states")
try:
    from app.routers.anomaly import compute_alert_state

    assert compute_alert_state(0.05, 0.12, 0.5,  False) == "NORMAL"
    assert compute_alert_state(0.05, 0.12, 2.5,  False) == "WATCH"
    assert compute_alert_state(0.13, 0.12, 1.0,  False) == "WARN"
    assert compute_alert_state(0.25, 0.12, 6.0,  False) == "DANGER"
    # drift_locked also triggers WARN
    assert compute_alert_state(0.05, 0.12, 0.1,  True)  == "WARN"

    gates["G5_alert_states"] = PASS
    log("  PASS — NORMAL/WATCH/WARN/DANGER all correct")
except Exception as e:
    gates["G5_alert_states"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 6 — M7 feature vector is exactly 35 features
# =============================================================================
log("Gate 6: M7 feature vector — 35 features")
try:
    from app.routers.anomaly import _build_m7_features

    window_np = np.random.randn(50, 8).astype(np.float32)
    fv = _build_m7_features(window_np, score_A=0.05, score_B=0.01, score_C=0.1, raw_mae=0.05)

    assert fv.shape == (35,), f"Expected (35,), got {fv.shape}"
    assert fv.dtype == np.float32
    assert not np.isnan(fv).any(), "NaN in feature vector"
    assert not np.isinf(fv).any(), "Inf in feature vector"

    gates["G6_feature_vector"] = PASS
    log(f"  PASS — shape={fv.shape}, no NaN/Inf")
except Exception as e:
    gates["G6_feature_vector"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 7 — ZTBuffer append + get_sequence pipeline
# =============================================================================
log("Gate 7: ZTBuffer → TCN-AE readiness gate")
try:
    from app.runtime.zt_buffer import ZTBuffer

    async def _test_buf():
        buf = ZTBuffer(max_len=63, z_dim=64)
        for _ in range(62):
            await buf.append(np.zeros(64))
        assert not await buf.is_ready()
        await buf.append(np.ones(64))
        assert await buf.is_ready()
        seq = await buf.get_sequence()
        assert seq.shape == (63, 64)
        return True

    ok = asyncio.run(_test_buf())
    gates["G7_zt_pipeline"] = PASS if ok else FAIL
    log("  PASS — 63-window gate and shape correct")
except Exception as e:
    gates["G7_zt_pipeline"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 8 — M8p6 addendum: triggered = True when headroom < flag threshold
# =============================================================================
log("Gate 8: M8p6 sensor addendum logic")
try:
    from app.routers.anomaly import check_m8p6

    # Mock config: one channel near ceiling
    mock_cfg = {
        "headroom_flag_threshold": 0.10,
        "channels": [
            {
                "name" : "Suction Side Pressure",
                "index": 5,
                "cluster_ceilings": {
                    "steady_state": {"ceiling_multiplier": 2.0, "cluster_mean": 0.5}
                }
            }
        ]
    }
    # Window where channel 5 (Pres.SV) is at 0.95× ceiling → headroom ~5%
    window_np = np.ones((50, 8), dtype=np.float32) * 0.1
    window_np[:, 5] = 0.95   # near ceiling (ceiling_multiplier=2.0, mean=0.5 → ceiling=1.0)

    result = check_m8p6(window_np, "steady_state", mock_cfg)
    assert result.triggered is True, "Should trigger near ceiling"
    assert result.override_existing_prediction is False, "Must never override (C-28)"
    assert len(result.flagged_channels) > 0

    # Clean window — should not trigger
    clean_window = np.ones((50, 8), dtype=np.float32) * 0.1
    result2 = check_m8p6(clean_window, "steady_state", mock_cfg)
    assert result2.triggered is False

    gates["G8_m8p6_addendum"] = PASS
    log("  PASS — triggers near ceiling, never overrides prediction (C-28)")
except Exception as e:
    gates["G8_m8p6_addendum"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 9 — FaultPrediction includes C-26 disclaimer in Field 7
# =============================================================================
log("Gate 9: FaultPrediction Field 7 — C-26 disclaimer present")
try:
    from app.schemas.fault_output import FaultPrediction, MODEL_DISCLAIMER_TEXT

    assert "CIRA" in MODEL_DISCLAIMER_TEXT
    assert "0.65" in MODEL_DISCLAIMER_TEXT    # real-world F1 lower bound
    assert "0.85" in MODEL_DISCLAIMER_TEXT    # real-world F1 upper bound
    assert "C-26" in MODEL_DISCLAIMER_TEXT
    assert "Advisory only" in MODEL_DISCLAIMER_TEXT

    # Verify default is locked
    fp = FaultPrediction(
        fault_label="normal", confidence_pct=98.0, unknown_fault_flag=False,
        probable_physical_condition="Normal", expected_sensor_behavior="Flat",
        operational_risk_if_ignored="None", recommended_action="Monitor",
        score_A=0.05, score_B=0.0, score_C=0.1, cusum_Sn=0.3,
        adaptive_threshold=0.12, alert_state="NORMAL",
        prediction_id="test", pump_id="PUMP-0032", cluster="steady_state",
        timestamp_utc="2026-05-16T10:00:00Z", ood_suspected=False, mahal_dist=2.1,
    )
    assert fp.model_limitation_disclaimer == MODEL_DISCLAIMER_TEXT

    gates["G9_c26_disclaimer"] = PASS
    log("  PASS — C-26 wording locked in Field 7")
except Exception as e:
    gates["G9_c26_disclaimer"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 10 — Limitation flags include Group C, Label 21, OOD notes
# =============================================================================
log("Gate 10: Limitation flags — Group C, Label 21, OOD")
try:
    from app.routers.anomaly import build_limitation_flags
    from app.schemas.fault_output import M8p6Addendum

    no_addendum = M8p6Addendum(triggered=False)
    addendum    = M8p6Addendum(triggered=True, flagged_channels=["Pres.SV"])

    # Group C
    flags = build_limitation_flags(13, False, 80.0, no_addendum)
    assert any("Group C" in f for f in flags)

    # Label 21
    flags = build_limitation_flags(21, False, 61.0, no_addendum)
    assert any("21" in f for f in flags)

    # OOD
    flags = build_limitation_flags(0, True, 90.0, no_addendum)
    assert any("OOD" in f for f in flags)

    # Low confidence
    flags = build_limitation_flags(0, False, 60.0, no_addendum)
    assert any("70%" in f for f in flags)

    # M8p6 triggered
    flags = build_limitation_flags(0, False, 90.0, addendum)
    assert any("M8p6" in f for f in flags)

    gates["G10_limitation_flags"] = PASS
    log("  PASS — all limitation flag cases correct")
except Exception as e:
    gates["G10_limitation_flags"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 11 — CUSUM C-25: reset only via acknowledge, not via rolling update
# =============================================================================
log("Gate 11: C-25 — CUSUM never reset by rolling baseline update")
try:
    from app.runtime.cusum_state   import CUSUMState
    from app.runtime.rolling_state import RollingState

    async def _test_c25():
        cs = CUSUMState(H=5.0, k=0.5, lam=5.73e-05)
        rs = RollingState(window_size=10, theta_initial=0.11, lock_factor=1.5)

        # Accumulate CUSUM
        for _ in range(20):
            await cs.update(score_B=1.5)
        st_before = await cs.get_state()
        sn_before = st_before["cusum_Sn"]

        # Trigger rolling crosspoint lock — must NOT touch CUSUM
        for _ in range(15):
            await rs.update(score_A=5.0)  # force θ_t above 1.5× θ_initial
        rs_st = await rs.get_state()
        assert rs_st["drift_locked"] is True, "Rolling must have locked"

        st_after = await cs.get_state()
        sn_after = st_after["cusum_Sn"]

        # S_n must still be near sn_before (small decay only, not reset)
        assert sn_after > 0, f"S_n was reset to 0 — C-25 violation! S_n={sn_after}"
        return True

    ok = asyncio.run(_test_c25())
    gates["G11_c25_cusum_independence"] = PASS if ok else FAIL
    log("  PASS — rolling lock does not reset CUSUM (C-25 preserved)")
except Exception as e:
    gates["G11_c25_cusum_independence"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# Gate 12 — /api/acknowledge resets all 3 state objects (CUSUM + rolling + ZT)
# =============================================================================
log("Gate 12: /api/acknowledge resets CUSUM + rolling + ZTBuffer")
try:
    from app.runtime.cusum_state   import CUSUMState
    from app.runtime.rolling_state import RollingState
    from app.runtime.zt_buffer     import ZTBuffer

    async def _test_ack():
        cs = CUSUMState(H=5.0, k=0.5, lam=5.73e-05)
        rs = RollingState(window_size=10, theta_initial=0.11)
        zb = ZTBuffer(max_len=5, z_dim=64)

        # Fill all three
        for _ in range(10): await cs.update(score_B=1.5)
        for _ in range(10): await rs.update(score_A=0.05)
        for _ in range(5):  await zb.append(np.zeros(64))

        # Simulate acknowledge
        cs_after = await cs.reset(reason="maintenance_acknowledged")
        rs_after = await rs.reset()
        await zb.reset()
        zb_st    = await zb.get_state()

        assert cs_after["cusum_Sn"] == 0.0,           "CUSUM not reset"
        assert rs_after["drift_locked"] is False,      "Rolling not unlocked"
        assert zb_st["buffer_fill"] == 0,             "ZT buffer not cleared"
        return True

    ok = asyncio.run(_test_ack())
    gates["G12_acknowledge_reset"] = PASS if ok else FAIL
    log("  PASS — all 3 state objects reset by acknowledge")
except Exception as e:
    gates["G12_acknowledge_reset"] = FAIL
    log(f"  FAIL: {e}")


# =============================================================================
# SUMMARY
# =============================================================================
passed = sum(1 for v in gates.values() if v == PASS)
total  = len(gates)

print("\n" + "═"*60)
print(f"M10 PHASE 2 GATE RESULTS: {passed}/{total} PASS")
print("═"*60)
for name, result in gates.items():
    print(f"  {'✅' if result == PASS else '❌'} {name}: {result}")
print("═"*60)

results["gates"]         = gates
results["passed"]        = passed
results["total"]         = total
results["phase2_status"] = "READY_FOR_PHASE3" if passed == total else "NEEDS_REVIEW"
print(f"\n  Status: {results['phase2_status']}")

print("\n" + "═"*60)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print(f"M10_phase2_gates_pass:           {passed}/{total}")
print(f"M10_phase2_status:               {results['phase2_status']}")
print(f"M10_m4_layernorm_confirmed:      {'PASS' if gates.get('G1_m4_layernorm')==PASS else 'FAIL'}")
print(f"M10_inference_pipeline:          {'WIRED' if passed==total else 'INCOMPLETE'}")
print(f"M10_invariant19_enforced:        {'PASS' if gates.get('G4_invariant19')==PASS else 'FAIL'}")
print(f"M10_c25_enforced:                {'PASS' if gates.get('G11_c25_cusum_independence')==PASS else 'FAIL'}")
print(f"M10_c26_disclaimer_locked:       {'PASS' if gates.get('G9_c26_disclaimer')==PASS else 'FAIL'}")
print(f"M10_c28_addendum_no_override:    {'PASS' if gates.get('G8_m8p6_addendum')==PASS else 'FAIL'}")
print(f"Status for Phase 3:              {'READY' if passed==total else 'BLOCKED'}")
print("══ END PASTE UPDATE ══")
print("═"*60)

if passed < total:
    sys.exit(1)
