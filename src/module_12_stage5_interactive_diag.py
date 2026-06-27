# =============================================================================
# src/module_12_stage5_interactive_diag.py
# PumpSmart v14.2 — M12 Stage 5 : INTERACTIVE single-fault diagnostic
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# The batch rig runs all 24 labels at once — fires blur together and you can't
# see WHERE/WHY a single fault alerts. This runner does ONE thing per launch:
#   1. fresh reset of detector state (warmup from zero)
#   2. compose ONE selected fault narrative (real m6b physics)
#   3. stream it window-by-window through the LIVE server
#   4. print a FULL per-window TRACE: window# | phase | cluster | alert |
#      score_A | cusum | M7 label | confidence
#   5. print the dual-timer + gate verdict for that single fault
#
# You pick ONE label from the menu, watch its complete trace, terminate, inspect,
# relaunch for the next. Each launch is a clean phase-1-from-zero run.
#
# Reuses ALL composer/scorer/client logic from the batch rig by importing it,
# so physics + conventions stay identical. Nothing here re-implements the model.
#
# USAGE
#   python src/module_12_stage5_interactive_diag.py
#   (then follow the menu)
#   python src/module_12_stage5_interactive_diag.py --label 3   # skip menu
# =============================================================================

import sys, argparse, time
from pathlib import Path
from datetime import datetime

import numpy as np

# Resolve project root
_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent]:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p)); break
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p / "src")); break

# Import the batch rig's building blocks (single source of truth)
import importlib.util
_RIG = _THIS.parent / "module_12_stage5_honest_validation_rig.py"
spec = importlib.util.spec_from_file_location("stage5_rig", _RIG)
rig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rig)

# Pull the pieces we reuse
PumpClient        = rig.PumpClient
compose_narrative = rig.compose_narrative
do_warmup         = rig.do_warmup
WARMUP_WINDOWS    = rig.WARMUP_WINDOWS
WARMUP_NOISE_SIGMA= rig.WARMUP_NOISE_SIGMA
WINDOW_SIZE       = rig.WINDOW_SIZE
LABEL_NAME        = rig.LABEL_NAME
CLUSTER_NAMES     = rig.CLUSTER_NAMES
SEV_RANK          = rig.SEV_RANK
LEAD_MARGIN_S     = rig.LEAD_MARGIN_S
LEAD_MARGIN_FLOOR_S = rig.LEAD_MARGIN_FLOOR_S
MANDATORY_FIELDS  = rig.MANDATORY_FIELDS
plib              = rig.plib
MODEL_DIR         = rig.MODEL_DIR

import json
def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
PASS = "PASS"; FAIL = "FAIL"

# Seconds to wait between windows when --realtime is set. 1.0 matches the
# dashboard's 1 s /api/latest_state poll so the two advance in lockstep.
# Overridable on the CLI with --rt-sec (e.g. 2.0 for a slower, more dramatic
# climb, or 0.5 for a faster demo).
REALTIME_SEC_PER_WINDOW = 1.0


# =============================================================================
# Per-window TRACE runner for ONE label
# =============================================================================
def run_one_label(client, label, mode="full", seed=20260531, realtime=False):
    rng = np.random.default_rng(seed + label)
    narrative = compose_narrative(label, mode, rng)
    seq        = narrative["seq"]
    clusters   = narrative["clusters"]
    t_inject   = narrative["t_inject"]
    breakdown  = narrative["breakdown"]
    n_st       = narrative.get("n_startup", 0)
    n_tr       = narrative.get("n_trans", 0)
    fault_cl   = narrative["fault_cluster"]

    n_windows = seq.shape[0] // WINDOW_SIZE
    ln = LABEL_NAME.get(label, f"label_{label}")

    print("\n" + "═" * 96)
    print(f"  SINGLE-FAULT TRACE — L{label:02d}  {ln}")
    print(f"  fault cluster: {CLUSTER_NAMES[fault_cl]} | windows: {n_windows} "
          f"| fault injected at step {t_inject} | breakdown at step {breakdown}")
    print(f"  phases: startup≤{n_st}s  transition≤{n_st+n_tr}s  steady→inject  fault→end")
    print("═" * 96)
    hdr = (f"{'win':>4}{'step':>7}{'phase':>11}{'cluster':>13}"
           f"{'alert':>8}{'score_A':>9}{'cusum':>8}{'M7lbl':>7}{'conf%':>7}  note")
    print(hdr); print("─" * 96)

    # trackers
    first_watch = first_warn = first_danger = first_correct = None
    normal_fire_steady = False; normal_worst_steady = "NORMAL"
    sA_max = 0.0; cusum_max = 0.0; correct_any = False
    confidences = []

    for k in range(n_windows):
        s0 = k * WINDOW_SIZE
        win = np.nan_to_num(seq[s0:s0 + WINDOW_SIZE], nan=0.0).tolist()
        cl = clusters[s0] if s0 < len(clusters) else "steady_state"
        try:
            pred, _ = client.anomaly_detect(win, cluster=cl)
        except Exception as e:
            print(f"{k:>4}  ERROR: {e}"); continue
        win_end = (k + 1) * WINDOW_SIZE

        alert = pred.get("raw_alert_state") or pred.get("alert_state", "NORMAL")
        sA = float(pred.get("score_A", 0.0) or 0.0)
        cu = float(pred.get("cusum_Sn", 0.0) or 0.0)
        lbl = pred.get("fault_label_int")
        lbl = int(lbl) if lbl is not None else -1
        conf = float(pred.get("confidence_pct", 0.0) or 0.0)
        sA_max = max(sA_max, sA); cusum_max = max(cusum_max, cu)

        # phase tag
        if win_end <= n_st:
            phase = "startup"
        elif win_end <= n_st + n_tr:
            phase = "transition"
        elif t_inject is not None and win_end <= t_inject:
            phase = "steady"
        else:
            phase = "FAULT"

        # markers / notes
        note = ""
        after_inject = (t_inject is None) or (win_end > t_inject)
        if after_inject:
            if alert in ("WATCH", "WARN", "DANGER") and first_watch is None:
                first_watch = win_end; note += "←1st WATCH+ "
            if alert in ("WARN", "DANGER") and first_warn is None:
                first_warn = win_end; note += "←1st WARN+ "
            if alert == "DANGER" and first_danger is None:
                first_danger = win_end; note += "←1st DANGER "
            if lbl == label and label != 0:
                correct_any = True
                if first_correct is None:
                    first_correct = win_end; note += "←1st CORRECT "
                confidences.append(conf)
        # steady false-fire (gate scope)
        if phase == "steady" and alert != "NORMAL":
            normal_fire_steady = True
            if SEV_RANK.get(alert, 0) > SEV_RANK.get(normal_worst_steady, 0):
                normal_worst_steady = alert
            note += "⚠STEADY-FIRE "
        if breakdown is not None and win_end >= breakdown and "BREAK" not in note:
            note += "💥BREAKDOWN "

        # mark transitions visibly
        flag = " " if alert == "NORMAL" else ("•" if alert == "WATCH" else "!")
        print(f"{k:>4}{win_end:>7}{phase:>11}{cl:>13}"
              f"{alert:>7}{flag}{sA:>9.4f}{cu:>8.3f}{lbl:>7}{conf:>7.1f}  {note}")

        # ── Real-time pacing ────────────────────────────────────────────────
        # Without this, every window POSTs back-to-back (~3 s for the whole
        # sequence), so the dashboard (which polls /api/latest_state once a
        # second) only ever catches the final window — the "15-second jump"
        # you see is the dashboard polling AFTER the burst already finished.
        # With --realtime we sleep REALTIME_SEC_PER_WINDOW between windows so
        # the server's latest_prediction advances one window per second, in
        # lockstep with the 1 s dashboard poll. The panel then watches score_A
        # climb live (0.08 → 0.30 → 0.44) instead of a sudden end-of-run jump.
        # NOTE: this is demo time-compression. True plant time is 50× slower
        # (each window = 50 one-second samples); 1 window/s lets the full
        # ~73-window fault evolution show in ~73 s instead of ~60 min.
        if realtime:
            time.sleep(REALTIME_SEC_PER_WINDOW)

    # ── verdicts ────────────────────────────────────────────────────────────
    def lat(x): return None if (x is None or t_inject is None) else x - t_inject
    margin_req = max(LEAD_MARGIN_FLOOR_S, LEAD_MARGIN_S.get(label, 60))

    print("─" * 96)
    print(f"\n  RESULT — L{label:02d} {ln}")
    print(f"    Detection latency (after inject):  "
          f"WATCH={lat(first_watch)}s  WARN={lat(first_warn)}s  DANGER={lat(first_danger)}s")
    print(f"    First correct M7 label:            {lat(first_correct)}s  "
          f"(classified: {'YES' if correct_any else 'NO'})")
    if confidences:
        print(f"    Mean confidence (correct windows): {np.mean(confidences):.1f}%")
    print(f"    score_A max: {sA_max:.4f}   cusum max: {cusum_max:.3f}")

    # Timer 2
    if breakdown is None:
        print(f"    Breakdown lead-time: N/A (no in-window breakdown; slow fault)")
    elif first_danger is None:
        print(f"    Breakdown lead-time:  DANGER Level not Reached before breakdown "
              f"(step {breakdown})")
    else:
        gap = breakdown - first_danger
        verdict = PASS if gap >= margin_req else FAIL
        print(f"    Breakdown lead-time: DANGER@{first_danger}s, breakdown@{breakdown}s "
              f"→ gap {gap}s vs req {margin_req}s → {verdict}")

    # gates
    detected = first_watch is not None or sA_max > 0.05
    classify_gate = "N/A" if label == 0 else (PASS if correct_any else FAIL)
    norm_gate = (PASS if normal_worst_steady == "NORMAL"
                 else "PARTIAL_PASS" if normal_worst_steady == "WATCH" else FAIL)
    print(f"    Detection gate:      {'PASS' if detected else 'FAIL'}")
    print(f"    Classification gate: {classify_gate}")
    print(f"    Normal-phase gate (steady only): {norm_gate}"
          f"{'' if norm_gate == PASS else f' (worst steady alert: {normal_worst_steady})'}")
    print("═" * 96)


# =============================================================================
# Menu
# =============================================================================
def menu():
    print("\n" + "═" * 60)
    print("  M12 STAGE 5 — INTERACTIVE SINGLE-FAULT DIAGNOSTIC")
    print("═" * 60)
    print("  Select ONE item to run (fresh warmup, full per-window trace):\n")
    for i in range(0, 24, 2):
        left = f"  [{i:>2}] {LABEL_NAME.get(i, '')[:26]:<26}"
        right = (f"[{i+1:>2}] {LABEL_NAME.get(i+1, '')[:26]}"
                 if i + 1 < 24 else "")
        print(left + right)
    print("\n  [Q] quit")
    print("═" * 60)
    return input("  Enter label number (0-23) or Q: ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--label", type=int, default=None,
                    help="run this label directly, skip menu")
    ap.add_argument("--mode", default="full", choices=["smoke", "quick", "full"])
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--realtime", action="store_true",
                    help="pace one window per second (matches dashboard poll) "
                         "so the dashboard tracks the fault live")
    ap.add_argument("--rt-sec", type=float, default=None,
                    help="seconds per window in --realtime (default 1.0; "
                         "use 2.0 for a slower climb, 0.5 for faster)")
    args = ap.parse_args()

    if args.rt_sec is not None:
        global REALTIME_SEC_PER_WINDOW
        REALTIME_SEC_PER_WINDOW = args.rt_sec

    client = PumpClient(args.base_url)

    log("Server health check...")
    try:
        h = client.health()
        log(f"  status={h.get('status')} arch={h.get('arch_version')} "
            f"M4 q={h.get('m4_threshold_locked')}")
    except Exception as e:
        log(f"FATAL: server unreachable — {e}"); sys.exit(1)

    # init physics lib
    with open(MODEL_DIR / "M3_normalization_config.json", encoding="utf-8") as f:
        norm_cfg = json.load(f)
    phys_path = MODEL_DIR / "M5_physics_config.json"
    phys_cfg = (json.load(open(phys_path, encoding="utf-8")) if phys_path.exists()
                else {"physics_constants": {}})
    plib.init_lib(norm_cfg, phys_cfg, seed=args.seed)

    # Pick label (menu or arg)
    if args.label is not None:
        label = args.label
    else:
        sel = menu()
        if sel.upper() == "Q":
            print("  Bye."); return
        if not sel.isdigit() or not (0 <= int(sel) <= 23):
            print("  Invalid selection."); return
        label = int(sel)

    # FRESH warmup from zero (phase-1-from-zero, as requested)
    log(f"Fresh warmup: {WARMUP_WINDOWS} windows × σ={WARMUP_NOISE_SIGMA} (steady_state)")
    do_warmup(client, WARMUP_WINDOWS, cluster="steady_state")
    try:
        th = client.health().get("rolling_state", {}).get("theta_t")
        log(f"  θ_t after warmup → {th}")
    except Exception:
        pass

    run_one_label(client, label, mode=args.mode, seed=args.seed,
                  realtime=args.realtime)

    print("\n  Done. Relaunch the script to run another label from a clean reset.")


if __name__ == "__main__":
    main()
