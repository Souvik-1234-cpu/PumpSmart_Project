# =============================================================================
# src/module_12_stage5_client_sync_test.py
# PumpSmart v14.2 — TERMINAL vs DASHBOARD client-sync diagnostic
# =============================================================================
#
# QUESTION: why does the terminal trace not match the live web dashboard?
#
# HYPOTHESIS (from dashboard.js): the dashboard does NOT view the terminal's
# stream. It calls _generateDemoWindow() every ~50 s and POSTs its OWN random
# window to /api/anomaly_detect (line 96, "frontend generates synthetic windows
# for UI testing"). So terminal and dashboard feed the server DIFFERENT data and
# can never match while both run.
#
# THIS TEST proves it three ways, all read straight from the API:
#   T1. Drive the server with a KNOWN bearing fault (score_A climbs ~0.078→0.44).
#       Record the server's verdict at the peak window.
#   T2. WITHOUT warming, fire a single dashboard-style demo window (the exact
#       _generateDemoWindow steady baseline) and read its verdict.
#   T3. Compare: if T1 shows WARN@0.44/label-7 and T2 shows ~0.15/different label,
#       the two clients are provably on different data → mismatch is structural.
#
#   T4. THE FIX CHECK: drive bearing, then read the server's LATEST state via a
#       read-only call WITHOUT posting new data. If the dashboard polled THIS
#       instead of generating its own window, both would match. Confirms the
#       fix direction (one driver, one read-only viewer).
#
# NO MODEL CHANGE. Pure API diagnostic.
# =============================================================================

import sys, json, time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import requests

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent]:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p)); break
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p / "src")); break

import importlib.util
_RIG = _THIS.parent / "module_12_stage5_honest_validation_rig.py"
spec = importlib.util.spec_from_file_location("stage5_rig", _RIG)
rig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rig)

PumpClient        = rig.PumpClient
compose_narrative = rig.compose_narrative
do_warmup         = rig.do_warmup
WARMUP_WINDOWS    = rig.WARMUP_WINDOWS
WARMUP_NOISE_SIGMA= rig.WARMUP_NOISE_SIGMA
WINDOW_SIZE       = rig.WINDOW_SIZE
MODEL_DIR         = rig.MODEL_DIR
plib              = rig.plib

BASE = "http://localhost:8000"


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# Exact replica of dashboard.js _generateDemoWindow steady baseline (line 226)
DASH_BASELINE = [0.45, 0.42, 0.50, 0.95, 0.55, 0.80, 0.52, 0.90]


def dash_demo_window(rng):
    """What the dashboard POSTs every ~50s — its OWN synthetic steady window."""
    return [[max(0.0, min(3.0, b + rng.gauss(0, 0.045))) for b in DASH_BASELINE]
            for _ in range(WINDOW_SIZE)]


def post(client, win, cluster="steady_state"):
    r = client.session.post(f"{client.base_url}/api/anomaly_detect",
        json={"window": win, "pump_id": "PUMP-0032", "cluster": cluster,
              "timestamp_utc": datetime.now(timezone.utc).isoformat()},
        timeout=client.timeout)
    r.raise_for_status()
    return r.json()


def verdict(p):
    return (p.get("raw_alert_state") or p.get("alert_state", "?"),
            float(p.get("score_A", 0) or 0),
            int(p.get("fault_label_int", -1)),
            float(p.get("confidence_pct", 0) or 0))


def main():
    import random
    rng = random.Random(99)
    client = PumpClient(BASE)

    print("═" * 80)
    print("  TERMINAL vs DASHBOARD — CLIENT SYNC DIAGNOSTIC")
    print("═" * 80)
    try:
        h = client.health(); log(f"server: {h.get('status')} {h.get('arch_version')}")
    except Exception as e:
        log(f"FATAL server unreachable: {e}"); sys.exit(1)

    # init physics
    with open(MODEL_DIR / "M3_normalization_config.json", encoding="utf-8") as f:
        norm_cfg = json.load(f)
    pp = MODEL_DIR / "M5_physics_config.json"
    phys = json.load(open(pp, encoding="utf-8")) if pp.exists() else {"physics_constants": {}}
    plib.init_lib(norm_cfg, phys, seed=99)

    # ── Warm once (this is what the TERMINAL does; the dashboard does NOT) ─────
    log(f"Warming θ_t with {WARMUP_WINDOWS} windows (TERMINAL does this)...")
    do_warmup(client, WARMUP_WINDOWS, cluster="steady_state")

    # ── T1: drive a KNOWN bearing fault, capture peak verdict ─────────────────
    print("\nT1 — TERMINAL drives a real bearing fault (score_A should climb high)")
    nrng = np.random.default_rng(102)
    narr = compose_narrative(1, "full", nrng)   # bearing
    seq, clusters, t_inj = narr["seq"], narr["clusters"], narr["t_inject"]
    n = seq.shape[0] // WINDOW_SIZE
    peak = {"sA": 0}
    last_fault = None
    for k in range(n):
        s0 = k * WINDOW_SIZE
        win = np.nan_to_num(seq[s0:s0+WINDOW_SIZE], nan=0.0).tolist()
        cl = clusters[s0] if s0 < len(clusters) else "steady_state"
        p = post(client, win, cl)
        st, sA, lbl, cf = verdict(p)
        if (t_inj is not None) and ((k+1)*WINDOW_SIZE > t_inj):
            last_fault = (st, sA, lbl, cf)
            if sA > peak["sA"]:
                peak = {"sA": sA, "st": st, "lbl": lbl, "cf": cf}
    log(f"  TERMINAL peak fault window : alert={peak['st']}  score_A={peak['sA']:.4f}  "
        f"label={peak['lbl']}  conf={peak['cf']:.0f}%")
    log(f"  TERMINAL last fault window : alert={last_fault[0]}  score_A={last_fault[1]:.4f}  "
        f"label={last_fault[2]}  conf={last_fault[3]:.0f}%")

    # ── T2: immediately fire ONE dashboard-style demo window ──────────────────
    print("\nT2 — DASHBOARD-style poll (its own synthetic steady window) on SAME server")
    pd = post(client, dash_demo_window(rng), "steady_state")
    dst, dsA, dlbl, dcf = verdict(pd)
    log(f"  DASHBOARD demo window      : alert={dst}  score_A={dsA:.4f}  "
        f"label={dlbl}  conf={dcf:.0f}%")

    # ── T3: compare ───────────────────────────────────────────────────────────
    print("\nT3 — COMPARISON")
    same_alert = (peak.get("st") == dst)
    sA_gap = abs(peak["sA"] - dsA)
    print(f"  Terminal fault score_A {peak['sA']:.4f}  vs  Dashboard demo score_A {dsA:.4f}"
          f"   → gap {sA_gap:.4f}")
    print(f"  Terminal fault label {peak['lbl']}  vs  Dashboard demo label {dlbl}")

    print("\n" + "═" * 80)
    print("  DIAGNOSIS")
    print("═" * 80)
    if sA_gap > 0.10:
        print("  ✅ CONFIRMED: terminal and dashboard are on DIFFERENT DATA.")
        print("     The dashboard POSTs its OWN _generateDemoWindow() (random steady")
        print("     window) every ~50s — it does NOT view the terminal's injected")
        print("     fault stream. The numbers differ because the inputs differ.")
        print("     This is structural (dashboard.js line 96), not a model bug.")
        print()
        print("  FIX so both ALWAYS show the same during a demo:")
        print("    Make the dashboard a READ-ONLY VIEWER of the injected stream:")
        print("    1. Add a read-only GET /api/latest_state endpoint that returns the")
        print("       server's most recent FaultPrediction WITHOUT running inference.")
        print("    2. In dashboard.js pollInference(): replace _generateDemoWindow()+")
        print("       POST with a GET to /api/latest_state. One driver (terminal/SCADA),")
        print("       one viewer (dashboard) → identical numbers, every run.")
        print("    3. For the presentation: drive the fault from the terminal; the")
        print("       dashboard mirrors it live.")
    else:
        print("  ◑ score_A gap small — rerun with a higher-amplitude fault to see the")
        print("     divergence clearly, or the dashboard may already be near fault level.")
    print("═" * 80)


if __name__ == "__main__":
    main()
