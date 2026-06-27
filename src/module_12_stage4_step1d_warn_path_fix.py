# =============================================================================
# src/module_12_stage4_step1d_warn_path_fix.py
# PumpSmart v14.2 — M12 Stage 4, Step 1d: AlertStateMachine WARN-path fix (G1)
# =============================================================================
#
# WHY
# ---
# Smoke (post 4.1c): G1_normal_fpr still 1.0 even with calibrated floors
# (rm100_floor=0.616 > normal rm100 ~0.33). Root cause is a DIFFERENT WARN
# trigger: the bare single-window `score_A >= theta_t`. Measured: theta_t adapts
# to 0.178 but normal score_A peaks at 0.45 (_g10 full sA_max_mean=0.4505), so
# normal windows routinely exceed theta_t -> WARN -> 100% FPR.
#
# ISA-18.2: a single-window analog crossing must NOT drive an alarm STATE — that
# is chatter. Instantaneous score_A>=theta_t is a DETECTION signal (used by M4
# threshold / G2-G3 detect, unaffected here), not an alarm-state trigger. The
# alarm WARN state must be governed by SUSTAINED elevation (the calibrated
# rolling-mean floors) plus the mechanism triggers (Mech-B slope, Mech-C drift,
# soft drift_ratio). DANGER still fires on acute score_A>=1.5*theta_t and the
# confirmed crosspoint lock (Mech-A) — acute spikes are not suppressed.
#
# CHANGE (surgical, in app/runtime/alert_state_machine.py):
#   In AlertStateMachine.update(), the WARN-evidence block:
#       if score_A >= theta_t:
#           warn_hit = True; reasons.append(... ">= theta_t ...")
#   is REMOVED. WARN now requires sustained evidence:
#       rolling_mean_100/200 floors, Mech-B slope, drift_ratio>=warn, Mech-C.
#   Acute DANGER (score_A>=1.5*theta_t) and Mech-A are UNCHANGED.
#
# This affects only the alarm STATE; detection (G2/G3, score_A vs M4 q) is
# computed elsewhere and is unchanged (smoke shows G2/G3 detect = 100%).
#
# Backup written; change verified by re-import + a normal-vs-acute unit check.
#
# RUN:  python src/module_12_stage4_step1d_warn_path_fix.py
# =============================================================================

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import OUTPUT_DIR
from datetime import date, datetime
import json, shutil, importlib.util, warnings
warnings.filterwarnings("ignore")

SCRIPT_NAME = "module_12_stage4_step1d_warn_path_fix"
REPORT_DIR = OUTPUT_DIR / "reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)
PASS, FAIL = "PASS", "FAIL"
ASM = _ROOT / "app" / "runtime" / "alert_state_machine.py"


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


results = {"script": SCRIPT_NAME, "timestamp": datetime.now().isoformat(),
           "gates": {}, "evidence": {}, "overall_status": "UNKNOWN"}
G = results["gates"]

# The exact WARN-on-theta_t lines to remove (must match deployed file).
OLD_BLOCK = '''            if score_A >= theta_t:
                warn_hit = True; reasons.append(f"score_A {score_A:.4f} >= theta_t {theta_t:.4f}")
'''
NEW_BLOCK = '''            # Stage 4 Step 1d: bare single-window score_A>=theta_t WARN trigger
            # REMOVED (caused G1 chatter when adapted theta_t < normal score_A
            # range). WARN now requires SUSTAINED evidence (rolling-mean floors)
            # or a mechanism trigger. Acute DANGER (>=1.5*theta_t) is unchanged.
'''


def main():
    log("=" * 76)
    log(f"  {SCRIPT_NAME}  |  {date.today().isoformat()}")
    log("  Stage 4 Step 1d — remove bare score_A>=theta_t WARN trigger (G1 fix)")
    log("=" * 76)

    if not ASM.exists():
        log(f"  FATAL — {ASM} not found."); results["overall_status"] = FAIL; _finish(); return

    src = ASM.read_text(encoding="utf-8")

    if "Step 1d:" in src:
        log("  Already patched (Step 1d marker present). Verifying behavior only.")
        G["patch_applied"] = "SKIP"
    elif OLD_BLOCK not in src:
        log("  FAIL — could not find the exact WARN-on-theta_t block to remove.")
        log("  The deployed alert_state_machine.py differs from the gated source.")
        log("  Manual edit: in AlertStateMachine.update(), delete the two lines:")
        log("      if score_A >= theta_t:")
        log("          warn_hit = True; reasons.append(... '>= theta_t' ...)")
        G["patch_applied"] = FAIL
        results["overall_status"] = FAIL
        _finish(); return
    else:
        bak = ASM.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(ASM, bak)
        ASM.write_text(src.replace(OLD_BLOCK, NEW_BLOCK, 1), encoding="utf-8")
        results["evidence"]["backup"] = str(bak)
        G["patch_applied"] = PASS
        log(f"  PASS — WARN-on-theta_t trigger removed; backup {bak.name}")

    # ── Verify behavior: load patched module, check normal stays NORMAL while
    #    acute still reaches DANGER and sustained-floor still reaches WARN. ────
    try:
        spec = importlib.util.spec_from_file_location("_asm_patched", str(ASM))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        ASMclass = mod.AlertStateMachine
        cfg = mod.load_alert_thresholds(str(_ROOT / "models" / "M8_alert_thresholds.json"))
        T0 = 1.881275   # production theta_initial

        # (1) NORMAL: score_A=0.22 (> theta_t 0.178 but < 1.5*theta_t=0.267
        #     AND < floor). True sub-acute normal — must stay NORMAL.
        sm = ASMclass(cfg=cfg)
        r = None
        for _ in range(10):
            r = sm.update(score_A=0.22, theta_t=0.178, theta_initial=T0,
                          drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")
        normal_ok = (r["alert_state"] == "NORMAL")
        log(f"  normal (sA=0.22: >theta_t 0.178 but <1.5*theta_t 0.267 & <floor): "
            f"state={r['alert_state']} rm100={r['features']['rolling_mean_100']:.3f}")

        # (2) ACUTE DANGER preserved: score_A=0.30 >= 1.5*theta_t(0.18)=0.27
        sm2 = ASMclass(cfg=cfg)
        r2 = sm2.update(score_A=0.30, theta_t=0.18, theta_initial=T0,
                        drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")
        danger_ok = (r2["alert_state"] == "DANGER")
        log(f"  acute (sA=0.30 >= 1.5*0.18): state={r2['alert_state']}")

        # (3) Sustained-floor WARN preserved: feed above rm100 floor
        floor100 = cfg.get("rolling_mean_100_floor", 0.6157)
        sm3 = ASMclass(cfg=cfg)
        r3 = None
        for _ in range(120):
            r3 = sm3.update(score_A=floor100 * 1.05, theta_t=2.0, theta_initial=T0,
                            drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")
        warn_ok = (r3["alert_state"] == "WARN")
        log(f"  sustained floor (sA>{floor100:.3f}): state={r3['alert_state']} "
            f"rm100={r3['features']['rolling_mean_100']:.3f}")

        G["verify_normal_stays_normal"] = PASS if normal_ok else FAIL
        G["verify_acute_still_danger"]  = PASS if danger_ok else FAIL
        G["verify_floor_still_warn"]    = PASS if warn_ok else FAIL
    except Exception as e:
        import traceback
        log(f"  verify error: {e}"); log(traceback.format_exc())
        G["verify_normal_stays_normal"] = FAIL

    core = [G.get("verify_normal_stays_normal"), G.get("verify_acute_still_danger"),
            G.get("verify_floor_still_warn")]
    results["overall_status"] = PASS if all(s == PASS for s in core) else FAIL
    _finish()


def _finish():
    out = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    with open(out, "w", encoding="utf-8") as f: json.dump(results, f, indent=2, default=str)
    g, status = results["gates"], results["overall_status"]
    print("\n" + "=" * 76)
    print("══ PASTE TEXT UPDATE ══")
    print(f"M12 Stage 4 Step 1d (WARN-path fix): {status}")
    print(f"  patch_applied: {g.get('patch_applied','-')}")
    print(f"  normal stays NORMAL (sA>theta_t but sustained-low): {g.get('verify_normal_stays_normal','-')}")
    print(f"  acute still DANGER (>=1.5*theta_t): {g.get('verify_acute_still_danger','-')}")
    print(f"  sustained floor still WARN: {g.get('verify_floor_still_warn','-')}")
    if status == PASS:
        print("  WARN now governed by calibrated floors, not bare theta_t cross.")
        print("  Next: re-derive CUSUM k for H=5.0 regime (G4a+G4b joint), then re-smoke.")
    print("══ END PASTE UPDATE ══")
    print(f"  Report: {out}")
    if results["evidence"].get("backup"):
        print(f"  GitHub push: app/runtime/alert_state_machine.py (backup {Path(results['evidence']['backup']).name})")
    print("=" * 76)


if __name__ == "__main__":
    main()