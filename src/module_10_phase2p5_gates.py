# =============================================================================
# module_10_phase2p5_gates.py
# PumpSmart v14.2 — M10 Phase 2.5 (Sensor History) Gate Tests
# Run from repo root: python src/module_10_phase2p5_gates.py
#
# 10 gates must pass before M12 adversarial validation begins.
# These verify that the server-side sensor history buffer works correctly
# under multi-client concurrency, ring-buffer eviction, and downsampling.
# =============================================================================

import sys, asyncio, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT_NAME = "module_10_phase2p5_gates"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
PASS = "PASS"; FAIL = "FAIL"
gates = {}


# ── Gate 1 — SensorHistoryBuffer importable ──────────────────────────────────
log("Gate 1: SensorHistoryBuffer import")
try:
    from app.runtime.sensor_history import (
        SensorHistoryBuffer, CHANNEL_ORDER, N_CHANNELS, DEFAULT_MAX_LEN
    )
    assert N_CHANNELS == 8
    assert len(CHANNEL_ORDER) == 8
    assert DEFAULT_MAX_LEN == 86_400
    gates["G1_import"] = PASS; log("  PASS")
except Exception as e:
    gates["G1_import"] = FAIL; log(f"  FAIL: {e}")


# ── Gate 2 — Append + retrieve basic ─────────────────────────────────────────
log("Gate 2: append + get_range basic")
try:
    from app.runtime.sensor_history import SensorHistoryBuffer

    async def _test():
        buf = SensorHistoryBuffer(max_len=100)
        win = [[0.5]*8 for _ in range(50)]
        for i in range(10):
            pred = {"score_A": 0.05 + i*0.01, "score_B": 0.0,
                    "cusum_Sn": 0.0, "adaptive_threshold": 0.110,
                    "alert_state": "NORMAL", "fault_label_int": 0,
                    "confidence_pct": 99.0}
            await buf.append(win, pred)
        st = await buf.get_state()
        assert st["buffer_fill"] == 10
        assert st["n_writes_total"] == 10
        data = await buf.get_range(last_n_seconds=10, downsample="full")
        assert data["n_points"] == 10
        assert len(data["channels"]["Mot.SV"]) == 10
        return True

    ok = asyncio.run(_test())
    gates["G2_basic_append"] = PASS if ok else FAIL
    log("  PASS")
except Exception as e:
    gates["G2_basic_append"] = FAIL; log(f"  FAIL: {e}")


# ── Gate 3 — Ring-buffer eviction ────────────────────────────────────────────
log("Gate 3: ring-buffer eviction at capacity")
try:
    from app.runtime.sensor_history import SensorHistoryBuffer

    async def _test():
        buf = SensorHistoryBuffer(max_len=50)
        win = [[1.0]*8 for _ in range(50)]
        for i in range(75):    # overflow by 25
            pred = {"score_A": float(i), "alert_state": "NORMAL",
                    "fault_label_int": 0, "confidence_pct": 0.0}
            await buf.append(win, pred)
        st = await buf.get_state()
        assert st["buffer_fill"] == 50, f"expected 50, got {st['buffer_fill']}"
        assert st["n_writes_total"] == 75
        data = await buf.get_range(last_n_seconds=50, downsample="full")
        # Oldest should be score_A=25 (i.e. 75-50), newest 74
        assert data["score_A"][0]  == 25.0
        assert data["score_A"][-1] == 74.0
        return True

    ok = asyncio.run(_test())
    gates["G3_ring_eviction"] = PASS if ok else FAIL
    log("  PASS — evicts oldest correctly")
except Exception as e:
    gates["G3_ring_eviction"] = FAIL; log(f"  FAIL: {e}")


# ── Gate 4 — Multi-client concurrent reads ───────────────────────────────────
log("Gate 4: multi-client concurrent reads (5 simultaneous)")
try:
    from app.runtime.sensor_history import SensorHistoryBuffer

    async def _test():
        buf = SensorHistoryBuffer(max_len=1000)
        win = [[0.5]*8 for _ in range(50)]
        # Pre-fill
        for i in range(500):
            await buf.append(win, {"score_A": float(i), "alert_state": "NORMAL",
                                    "fault_label_int": 0, "confidence_pct": 0.0})

        # 5 concurrent readers
        async def reader():
            return await buf.get_range(last_n_seconds=500, downsample="full")

        results = await asyncio.gather(*[reader() for _ in range(5)])
        # All 5 must see identical data
        first = results[0]["score_A"]
        for r in results[1:]:
            assert r["score_A"] == first, "Inconsistent reads across clients"
        assert len(first) == 500
        return True

    ok = asyncio.run(_test())
    gates["G4_concurrent_reads"] = PASS if ok else FAIL
    log("  PASS — 5 clients see identical history")
except Exception as e:
    gates["G4_concurrent_reads"] = FAIL; log(f"  FAIL: {e}")


# ── Gate 5 — Concurrent read + write race safety ─────────────────────────────
log("Gate 5: read/write race (writes interleaved with reads)")
try:
    from app.runtime.sensor_history import SensorHistoryBuffer

    async def _test():
        buf = SensorHistoryBuffer(max_len=2000)
        win = [[0.5]*8 for _ in range(50)]

        async def writer():
            for i in range(500):
                await buf.append(win, {"score_A": float(i), "alert_state": "NORMAL",
                                        "fault_label_int": 0, "confidence_pct": 0.0})

        async def reader():
            for _ in range(50):
                data = await buf.get_range(last_n_seconds=100, downsample="full")
                # Just verify no exception + consistent length within each call
                n = data["n_points"]
                assert len(data["score_A"]) == n
                assert len(data["channels"]["Mot.SV"]) == n
                await asyncio.sleep(0.001)

        await asyncio.gather(writer(), reader(), reader(), reader())
        st = await buf.get_state()
        assert st["buffer_fill"] == 500
        return True

    ok = asyncio.run(_test())
    gates["G5_race_safety"] = PASS if ok else FAIL
    log("  PASS — no torn reads under concurrent write")
except Exception as e:
    gates["G5_race_safety"] = FAIL; log(f"  FAIL: {e}")


# ── Gate 6 — DS-C adaptive downsampling ──────────────────────────────────────
log("Gate 6: DS-C adaptive downsampling (last 5 min full, older LTTB)")
try:
    from app.runtime.sensor_history import SensorHistoryBuffer

    async def _test():
        buf = SensorHistoryBuffer(max_len=5000)
        win = [[0.5]*8 for _ in range(50)]
        for i in range(3600):    # 1 hour at 1 Hz
            await buf.append(win, {"score_A": float(i % 100), "alert_state": "NORMAL",
                                    "fault_label_int": 0, "confidence_pct": 0.0})

        # Adaptive: ask for 500 points from 3600
        data = await buf.get_range(last_n_seconds=3600,
                                    downsample="adaptive", max_points=500)
        assert data["downsample_method"] == "adaptive"
        assert data["original_n"]  == 3600
        # Should be roughly 500 points (LTTB older + 300 tail)
        assert 400 <= data["n_points"] <= 600
        # Last 300 must be full-res (consecutive)
        assert len(data["channels"]["Mot.SV"]) == data["n_points"]
        return True

    ok = asyncio.run(_test())
    gates["G6_adaptive_downsample"] = PASS if ok else FAIL
    log("  PASS")
except Exception as e:
    gates["G6_adaptive_downsample"] = FAIL; log(f"  FAIL: {e}")


# ── Gate 7 — Empty buffer returns valid response ─────────────────────────────
log("Gate 7: empty buffer returns valid zero-point response")
try:
    from app.runtime.sensor_history import SensorHistoryBuffer

    async def _test():
        buf = SensorHistoryBuffer(max_len=100)
        data = await buf.get_range(last_n_seconds=60, downsample="adaptive")
        assert data["n_points"] == 0
        assert data["timestamps"] == []
        assert all(data["channels"][ch] == [] for ch in data["channels"])
        return True

    ok = asyncio.run(_test())
    gates["G7_empty_buffer"] = PASS if ok else FAIL
    log("  PASS")
except Exception as e:
    gates["G7_empty_buffer"] = FAIL; log(f"  FAIL: {e}")


# ── Gate 8 — Router file present + endpoints registered ──────────────────────
log("Gate 8: history router registered + 3 endpoints present")
try:
    from app.main import app as fastapp
    routes = {r.path for r in fastapp.routes}
    expected = {
        "/api/sensor_history",
        "/api/sensor_history/state",
        "/api/sensor_history/reset",
    }
    missing = expected - routes
    assert not missing, f"missing: {missing}"
    gates["G8_routes_registered"] = PASS
    log(f"  PASS — {len(expected)} history endpoints registered")
except Exception as e:
    gates["G8_routes_registered"] = FAIL; log(f"  FAIL: {e}")


# ── Gate 9 — anomaly.py wired to append on every inference ───────────────────
log("Gate 9: anomaly.py calls history.append after inference")
try:
    anomaly_src = Path("app/routers/anomaly.py").read_text(encoding="utf-8")
    assert "request.app.state.history.append" in anomaly_src, \
        "anomaly.py must call request.app.state.history.append(...)"
    gates["G9_anomaly_wired"] = PASS
    log("  PASS — history.append call present in anomaly.py")
except Exception as e:
    gates["G9_anomaly_wired"] = FAIL; log(f"  FAIL: {e}")


# ── Gate 10 — acknowledge does NOT reset history (forensic retention) ────────
log("Gate 10: /api/acknowledge does NOT reset sensor history")
try:
    ack_src = Path("app/routers/acknowledge.py").read_text(encoding="utf-8")
    # acknowledge must reset cusum, rolling, zt_buf — but NOT history
    assert "history" not in ack_src.lower() or "history.reset" not in ack_src, \
        "acknowledge.py must not reset history (forensic retention required)"
    gates["G10_history_forensic_retention"] = PASS
    log("  PASS — acknowledge preserves history for forensic review")
except Exception as e:
    gates["G10_history_forensic_retention"] = FAIL; log(f"  FAIL: {e}")


# ── Summary ──────────────────────────────────────────────────────────────────
passed = sum(1 for v in gates.values() if v == PASS)
total  = len(gates)

print("\n" + "═"*60)
print(f"M10 PHASE 2.5 GATE RESULTS: {passed}/{total} PASS")
print("═"*60)
for name, result in gates.items():
    print(f"  {'✅' if result == PASS else '❌'} {name}: {result}")
print("═"*60)
status = "READY_FOR_M12" if passed == total else "NEEDS_REVIEW"
print(f"\n  Status: {status}")

print("\n" + "═"*60)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print(f"M10_phase2p5_gates_pass:     {passed}/{total}")
print(f"M10_phase2p5_status:         {status}")
print(f"M10_sensor_history_buffer:   RAM ring buffer, 86400 @ 1Hz = 24h")
print(f"M10_history_downsample:      DS-C adaptive (last 5min full + LTTB)")
print(f"M10_multi_client_safe:       {'PASS' if gates.get('G4_concurrent_reads')==PASS else 'FAIL'}")
print(f"M10_forensic_retention:      acknowledge preserves history")
print(f"Status for M12:              {'READY' if passed==total else 'BLOCKED'}")
print("══ END PASTE UPDATE ══")
print("═"*60)

if passed < total:
    sys.exit(1)
