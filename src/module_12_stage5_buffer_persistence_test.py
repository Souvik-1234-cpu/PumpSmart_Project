# =============================================================================
# src/module_12_stage5_buffer_persistence_test.py
# PumpSmart v14.2 — M12 Stage 5 : z_t BUFFER PERSISTENCE + score_C WARM-UP TEST
# =============================================================================
#
# HYPOTHESIS UNDER TEST
# ---------------------
# The 52% classification is suspected to be a TEST-RIG ARTIFACT, not a model
# defect. M7 depends on score_C (the TCN-AE chain discriminator). score_C is
# 0.0 until the server-side z_t buffer is warmed (needs ~63 z_t windows). In
# real deployment the pump streams continuously → buffer is ALWAYS warm → M7
# always has score_C. In the rig, short isolated sequences (~22-35 windows)
# may start cold → score_C=0.0 → M7 flip-flops (cavitation→3 or →8).
#
# THIS TEST ANSWERS THREE QUESTIONS DEFINITIVELY:
#   Q1. Does the 432-window warmup actually fill the buffer? (score_C goes >0?)
#   Q2. Does the buffer PERSIST across requests (server-global state)?
#   Q3. After warmup, is score_C live (non-zero) from the FIRST fault window?
#
# If Q1-Q3 all YES → 52% is a cold-start rig artifact; deployment is fine; the
# fix is simply "warm the buffer before measuring", no retrain.
# If score_C stays 0.0 even after 432 warm windows → the buffer is being reset
# or min_fill never satisfied → a real server bug to fix.
#
# NO MODEL CHANGE. NO RETRAIN. Pure read-only diagnostic via /api/anomaly_detect
# (which returns score_C in its response, confirmed anomaly.py:438).
#
# USAGE
#   python src/module_12_stage5_buffer_persistence_test.py
#   python src/module_12_stage5_buffer_persistence_test.py --warmup 432 --probe 80
# =============================================================================

import sys, argparse, time, json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import requests

# Resolve project root + import rig building blocks (single source of truth)
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
WINDOW_SIZE       = rig.WINDOW_SIZE
WARMUP_NOISE_SIGMA= rig.WARMUP_NOISE_SIGMA


def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def steady_window(rng):
    """A clean steady-state window (M3-normalised ~1.0 baseline)."""
    base = [0.45, 0.42, 0.50, 0.95, 0.55, 0.80, 0.52, 0.90]
    return [[max(0.0, min(3.0, b + rng.gauss(0, WARMUP_NOISE_SIGMA))) for b in base]
            for _ in range(WINDOW_SIZE)]


def raw_post(client, win, cluster="steady_state"):
    """Direct POST so we can read score_C, score_B, label from the response."""
    r = client.session.post(
        f"{client.base_url}/api/anomaly_detect",
        json={"window": win, "pump_id": "PUMP-0032", "cluster": cluster,
              "timestamp_utc": datetime.now(timezone.utc).isoformat()},
        timeout=client.timeout)
    r.raise_for_status()
    return r.json()


def main():
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--warmup", type=int, default=432,
                    help="warm-up windows streamed before probing")
    ap.add_argument("--probe", type=int, default=80,
                    help="windows to probe score_C trajectory after warmup")
    ap.add_argument("--mode", default="full", choices=["smoke", "quick", "full"],
                    help="narrative length for Phase C cavitation (full ≈ 13 fault windows)")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    client = PumpClient(args.base_url)

    print("═" * 84)
    print("  M12 STAGE 5 — z_t BUFFER PERSISTENCE + score_C WARM-UP TEST")
    print(f"  warmup={args.warmup}  probe={args.probe}  url={args.base_url}")
    print("═" * 84)

    log("Server health...")
    try:
        h = client.health()
        log(f"  status={h.get('status')} arch={h.get('arch_version')}")
    except Exception as e:
        log(f"FATAL: server unreachable — {e}"); sys.exit(1)

    # init physics lib (compose_narrative needs plib._rng seeded)
    MODEL_DIR = rig.MODEL_DIR
    plib = rig.plib
    with open(MODEL_DIR / "M3_normalization_config.json", encoding="utf-8") as f:
        norm_cfg = json.load(f)
    phys_path = MODEL_DIR / "M5_physics_config.json"
    phys_cfg = (json.load(open(phys_path, encoding="utf-8")) if phys_path.exists()
                else {"physics_constants": {}})
    plib.init_lib(norm_cfg, phys_cfg, seed=args.seed)

    # ── PHASE A: stream warmup windows, watch when score_C first goes non-zero ─
    print("\nPHASE A — Warm-up: when does score_C first become non-zero?")
    print("  (score_C requires the z_t buffer to reach min_fill; 0.0 = cold)")
    print("  " + "-" * 70)
    print(f"  {'win#':>6}{'score_A':>10}{'score_B':>11}{'score_C':>11}{'M7lbl':>7}{'conf%':>7}  note")
    first_nonzero_C = None
    first_nonzero_B = None
    last_C = 0.0
    milestones = {1, 10, 30, 50, 60, 62, 63, 64, 70, 100, 150, 200, 300, 432}
    for i in range(1, args.warmup + 1):
        win = steady_window(rng)
        try:
            p = raw_post(client, win)
        except Exception as e:
            log(f"  win {i}: ERROR {e}"); continue
        sB = float(p.get("score_B", 0.0) or 0.0)
        sC = float(p.get("score_C", 0.0) or 0.0)
        sA = float(p.get("score_A", 0.0) or 0.0)
        lbl = p.get("fault_label_int", -1)
        conf = float(p.get("confidence_pct", 0.0) or 0.0)
        if first_nonzero_C is None and abs(sC) > 1e-9:
            first_nonzero_C = i
        if first_nonzero_B is None and abs(sB) > 1e-9:
            first_nonzero_B = i
        last_C = sC
        if i in milestones or (first_nonzero_C is not None and i <= first_nonzero_C + 1):
            note = ""
            if i == first_nonzero_C: note += "← score_C FIRST NON-ZERO "
            if i == first_nonzero_B: note += "← score_B first non-zero "
            print(f"  {i:>6}{sA:>10.4f}{sB:>11.5f}{sC:>11.5f}{str(lbl):>7}{conf:>7.1f}  {note}")

    print("  " + "-" * 70)
    log(f"  score_B first non-zero at warm-up window: {first_nonzero_B}")
    log(f"  score_C first non-zero at warm-up window: {first_nonzero_C}")
    log(f"  score_C at end of warm-up ({args.warmup}): {last_C:.5f}")

    # ── PHASE B: persistence — does score_C stay live across a STREAM PAUSE? ───
    # We do NOT reset. Simply continue probing more steady windows; score_C must
    # remain non-zero (buffer is server-global and persists across requests).
    print("\nPHASE B — Persistence: does score_C stay live across continued requests?")
    print("  (no reset between Phase A and B — buffer should already be warm)")
    stay_live = 0
    for i in range(1, 6):
        p = raw_post(client, steady_window(rng))
        sC = float(p.get("score_C", 0.0) or 0.0)
        live = abs(sC) > 1e-9
        stay_live += int(live)
        print(f"    probe {i}: score_C={sC:.5f}  {'LIVE' if live else 'COLD'}")
    persists = stay_live == 5

    # ── PHASE C: the decisive test — inject a CAVITATION fault on the WARM buffer
    # Now the buffer is warm (Phase A+B). Stream a cavitation fault and check:
    # is score_C live from the FIRST fault window, and does M7 classify correctly
    # AND STABLY (3 = cavitation, not flip-flopping to 8)?
    print("\nPHASE C — Cavitation on a WARM buffer (the decisive test)")
    print("  If buffer-warmth is the fix: score_C live from window 1, M7=3 stable")
    print("  " + "-" * 70)
    print(f"  {'win#':>6}{'score_A':>10}{'score_C':>11}{'M7lbl':>7}{'conf%':>7}  verdict")

    nrng = np.random.default_rng(args.seed + 3)
    narrative = compose_narrative(3, args.mode, nrng)   # 3 = cavitation
    seq = narrative["seq"]; clusters = narrative["clusters"]
    t_inject = narrative["t_inject"]
    n_win = seq.shape[0] // WINDOW_SIZE
    cav_correct = 0; cav_fault_windows = 0; cav_C_live_at_inject = None
    flips = []
    for k in range(n_win):
        s0 = k * WINDOW_SIZE
        win = np.nan_to_num(seq[s0:s0 + WINDOW_SIZE], nan=0.0).tolist()
        cl = clusters[s0] if s0 < len(clusters) else "steady_state"
        p = raw_post(client, win, cluster=cl)
        sC = float(p.get("score_C", 0.0) or 0.0)
        sA = float(p.get("score_A", 0.0) or 0.0)
        lbl = int(p.get("fault_label_int", -1))
        conf = float(p.get("confidence_pct", 0.0) or 0.0)
        win_end = (k + 1) * WINDOW_SIZE
        in_fault = (t_inject is not None) and (win_end > t_inject)
        if in_fault:
            cav_fault_windows += 1
            flips.append(lbl)
            if cav_C_live_at_inject is None:
                cav_C_live_at_inject = abs(sC) > 1e-9
            if lbl == 3:
                cav_correct += 1
            verdict = ("✓cav" if lbl == 3 else
                       f"✗={lbl}" + (" (chain)" if lbl == 8 else ""))
            print(f"  {k:>6}{sA:>10.4f}{sC:>11.5f}{lbl:>7}{conf:>7.1f}  {verdict}")

    cav_acc = (cav_correct / cav_fault_windows * 100) if cav_fault_windows else 0.0

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 84)
    print("  DIAGNOSIS")
    print("═" * 84)
    print(f"  Q1 — Warmup fills buffer (score_C goes non-zero)?    "
          f"{'YES @ win '+str(first_nonzero_C) if first_nonzero_C else 'NO — STAYS 0.0'}")
    print(f"  Q2 — Buffer persists across requests (no reset)?     "
          f"{'YES (5/5 live)' if persists else f'NO ({stay_live}/5 live)'}")
    print(f"  Q3 — score_C live at first cavitation fault window?  "
          f"{'YES' if cav_C_live_at_inject else 'NO — COLD AT INJECT'}")
    print(f"  Cavitation classification on WARM buffer:            "
          f"{cav_acc:.0f}% ({cav_correct}/{cav_fault_windows} windows = label 3)")
    uniq = sorted(set(flips))
    print(f"  M7 labels seen during cavitation fault:              {uniq}")
    print("  " + "-" * 80)

    if first_nonzero_C and persists and cav_C_live_at_inject and cav_acc >= 70:
        print("  ✅ HYPOTHESIS CONFIRMED — the 52% is a COLD-START RIG ARTIFACT.")
        print("     The buffer warms, persists, and score_C is live at inject when warm.")
        print("     Cavitation classifies correctly on a warm buffer. In deployment the")
        print("     buffer is ALWAYS warm (continuous stream), so field classification")
        print("     is unaffected. FIX = warm the buffer before measuring in the rig.")
        print("     NO RETRAIN NEEDED.")
    elif first_nonzero_C and not cav_C_live_at_inject:
        print("  ⚠ BUFFER WARMS BUT IS COLD AT INJECT — something clears it between")
        print("     warmup and the fault sequence (per-sequence reset, or the rig's")
        print("     cooldown_flush / compose path resets zt_buf). FIX = preserve buffer")
        print("     into the sequence. Still NO RETRAIN.")
    elif not first_nonzero_C:
        print("  ❌ score_C NEVER GOES NON-ZERO even after full warmup. This is a real")
        print("     server issue: TCN min_fill never satisfied, or m8_model is None, or")
        print("     get_sequence() returns None. Investigate zt_buf.min_fill + m8 load.")
    else:
        print("  ◑ MIXED — buffer warms but cavitation still misclassifies on warm buffer.")
        print("     score_C alone may not separate cav(3) from cav-chain(8); the chain")
        print(f"     discriminator is weak. M7 labels seen: {uniq}. May need score_C")
        print("     feature review or targeted retrain — but confirm Phase C trace first.")
    print("═" * 84)


if __name__ == "__main__":
    main()
