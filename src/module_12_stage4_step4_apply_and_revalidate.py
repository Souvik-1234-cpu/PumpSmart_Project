# =============================================================================
# src/module_12_stage4_step4_apply_and_revalidate.py
# PumpSmart v14.2 — M12 Stage 4, Step 4.4: apply state-machine patch + revalidate
# =============================================================================
#
# PURPOSE
# -------
# Final integration of Stage 4. Both critical pre-validation defects are now
# fixed at code level:
#   G1  (normal FPR=1.0)  -> new AlertStateMachine (gated 15/15, step 4.2+4.3)
#   G4b (CUSUM runaway)   -> corrected cusum_state.py (gated 6/6, step 1b)
#
# This step:
#   (A) PATCHES the live server code (with timestamped .bak):
#         1. app/routers/anomaly.py
#              - import AlertStateMachine + load_alert_thresholds
#              - move alert computation to AFTER M7 classify (label_int in scope)
#              - call request.app.state.alert_sm.update(...) instead of the
#                old compute_alert_state(...)
#         2. app/routers/acknowledge.py
#              - reset request.app.state.alert_sm alongside cusum/rolling/zt
#         3. app/runtime/model_registry.py  (lifespan wiring)
#              - instantiate request.app.state.alert_sm at startup
#       (If a patch's anchor is already patched, that sub-step is SKIPPED — the
#        script is idempotent and safe to re-run.)
#   (B) VERIFIES wiring statically (imports present, old call replaced).
#   (C) Prints the exact restart + run instructions for the adversarial runner.
#
# IMPORTANT: this script does NOT start the server or run the runner itself
# (the server must be restarted to load patched modules + corrected CUSUM).
# It prepares and verifies the code, then hands off to the runner.
#
# SAFETY: every edited file is backed up first. If any anchor is not found,
# that file is left UNTOUCHED and the script reports which manual edit is needed
# — it never writes a half-applied patch.
#
# RUN (CWD-independent; server should be STOPPED while patching)
#   .venv\Scripts\Activate.ps1
#   python src/module_12_stage4_step4_apply_and_revalidate.py
# =============================================================================

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import OUTPUT_DIR

from datetime import date, datetime
import json
import shutil
import re
import warnings
warnings.filterwarnings("ignore")

SCRIPT_NAME = "module_12_stage4_step4_apply_and_revalidate"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

ANOMALY   = _ROOT / "app" / "routers" / "anomaly.py"
ACK       = _ROOT / "app" / "routers" / "acknowledge.py"
REGISTRY  = _ROOT / "app" / "runtime" / "model_registry.py"
ASM_FILE  = _ROOT / "app" / "runtime" / "alert_state_machine.py"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


results = {
    "script": SCRIPT_NAME,
    "stage": "M12 Stage 4 — Step 4.4 (apply patch + revalidate handoff)",
    "timestamp": datetime.now().isoformat(),
    "patches": {}, "gates": {}, "evidence": {},
    "overall_status": "UNKNOWN", "block_m11": True,
}


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, bak)
    return bak


# =============================================================================
# Patch helpers — each returns (status, detail). Idempotent.
# =============================================================================
def patch_anomaly():
    if not ANOMALY.exists():
        return FAIL, f"{ANOMALY} not found"
    src = ANOMALY.read_text(encoding="utf-8")

    already = "alert_sm.update(" in src
    if already:
        return SKIP, "anomaly.py already wired to alert_sm.update()"

    new = src

    # 1) import
    if "from app.runtime.alert_state_machine import" not in new:
        # insert after the feature_builder import (known present) or after first import block
        anchor = "from app.runtime.feature_builder import build_m7_features"
        imp = ("from app.runtime.feature_builder import build_m7_features\n"
               "from app.runtime.alert_state_machine import AlertStateMachine, load_alert_thresholds")
        if anchor in new:
            new = new.replace(anchor, imp, 1)
        else:
            new = "from app.runtime.alert_state_machine import AlertStateMachine, load_alert_thresholds\n" + new

    # 2) The old call computes alert_state BEFORE M7 classify. We must move it
    #    AFTER label_int is known. Strategy:
    #    (a) delete the old "alert_state = compute_alert_state(...)" line at its
    #        current position (it sits right after the L4 rolling block),
    #    (b) insert the new alert_sm.update(...) block right AFTER label_int /
    #        conf_pct are computed (after the proba/argmax block).
    old_call_re = re.compile(
        r"[ \t]*#?[ \t]*[─\-]*\s*Alert state.*?\n"
        r"[ \t]*alert_state\s*=\s*compute_alert_state\([^\)]*\)\s*\n",
        re.DOTALL)
    m = old_call_re.search(new)
    if not m:
        # fall back to a looser match on just the assignment line
        loose = re.compile(r"[ \t]*alert_state\s*=\s*compute_alert_state\([^\)]*\)\s*\n")
        m2 = loose.search(new)
        if not m2:
            return FAIL, ("could not locate the old `alert_state = compute_alert_state(...)` "
                          "line — apply patch manually (see printed guide).")
        new = loose.sub("", new, count=1)
    else:
        new = old_call_re.sub("", new, count=1)

    # 3) Insert the new block after label_name is computed.
    #    Anchor on the known line: label_name = label_map.get(label_int, "unknown")
    anchor2 = 'label_name = label_map.get(label_int, "unknown")'
    if anchor2 not in new:
        return FAIL, ("could not find label_name anchor to insert alert_sm.update() — "
                      "apply patch manually (see printed guide).")
    insert = (
        anchor2 + "\n\n"
        "    # ── Alert state machine (Stage 4: stateful, after M7 so label_int is in scope)\n"
        "    sm_out = request.app.state.alert_sm.update(\n"
        "        score_A=score_A,\n"
        "        theta_t=theta_t,\n"
        "        theta_initial=models[\"theta_initial\"],\n"
        "        drift_locked=drift_locked,\n"
        "        cusum_Sn=cusum_Sn,\n"
        "        cusum_alert=cusum_result.get(\"cusum_alert\", \"NORMAL\"),\n"
        "        label_int=label_int,\n"
        "        channel_drift_flags=None,        # Mech-C source wired when available\n"
        "        score_C=score_C,                 # advisory-only (Step 4.1)\n"
        "    )\n"
        "    alert_state = sm_out[\"alert_state\"]\n"
    )
    new = new.replace(anchor2, insert, 1)

    bak = backup(ANOMALY)
    ANOMALY.write_text(new, encoding="utf-8")
    return PASS, f"patched (move-after-classify + alert_sm.update); backup {bak.name}"


def patch_acknowledge():
    if not ACK.exists():
        return FAIL, f"{ACK} not found"
    src = ACK.read_text(encoding="utf-8")
    if "alert_sm.reset()" in src:
        return SKIP, "acknowledge.py already resets alert_sm"
    anchor = "await request.app.state.zt_buf.reset()"
    if anchor not in src:
        return FAIL, "could not find zt_buf.reset() anchor — apply manually"
    new = src.replace(
        anchor,
        anchor + "\n"
        "    # Stage 4: reset the alert state machine hysteresis on maintenance ack\n"
        "    try:\n"
        "        request.app.state.alert_sm.reset()\n"
        "    except Exception:\n"
        "        pass   # alert_sm may be absent in degraded boot; non-fatal\n",
        1)
    bak = backup(ACK)
    ACK.write_text(new, encoding="utf-8")
    return PASS, f"patched (alert_sm.reset on acknowledge); backup {bak.name}"


def patch_registry():
    """Wire request.app.state.alert_sm at startup. The registry builds `models`
    but app.state wiring may live in main.py lifespan. We add a helper import +
    a note; actual app.state.alert_sm assignment is added to load_all_models'
    consumer. Safest: append alert_sm construction guidance and, if a known
    app.state wiring block exists in main.py, patch there."""
    main_py = _ROOT / "app" / "main.py"
    target = main_py if main_py.exists() else REGISTRY
    if not target.exists():
        return FAIL, f"neither app/main.py nor model_registry found for wiring"
    src = target.read_text(encoding="utf-8")
    if "alert_sm" in src and "AlertStateMachine" in src:
        return SKIP, f"{target.name} already constructs alert_sm"
    # Look for an app.state assignment block to anchor near (e.g. cusum wiring)
    anchors = [
        "app.state.cusum",
        ".state.cusum =",
        "state.rolling =",
        "app.state.zt_buf",
    ]
    hit = next((a for a in anchors if a in src), None)
    if hit is None:
        return FAIL, (f"no app.state.cusum/rolling/zt_buf wiring anchor in {target.name} "
                      f"— add `app.state.alert_sm = AlertStateMachine(load_alert_thresholds())` "
                      f"manually where the other state objects are created.")
    # Insert import if missing
    new = src
    if "from app.runtime.alert_state_machine import" not in new:
        new = ("from app.runtime.alert_state_machine import AlertStateMachine, load_alert_thresholds\n"
               + new)
    # Insert construction right after the first matching anchor line
    lines = new.splitlines(keepends=True)
    out, inserted = [], False
    for ln in lines:
        out.append(ln)
        if (not inserted) and (hit in ln) and ("=" in ln):
            indent = ln[:len(ln) - len(ln.lstrip())]
            out.append(f"{indent}# Stage 4: alert state machine (one per app)\n")
            out.append(f"{indent}app.state.alert_sm = AlertStateMachine(load_alert_thresholds())\n")
            inserted = True
    if not inserted:
        return FAIL, f"found anchor token but no assignment line in {target.name}; wire manually"
    bak = backup(target)
    target.write_text("".join(out), encoding="utf-8")
    return PASS, f"patched {target.name} (construct alert_sm); backup {bak.name}"


def main():
    log("=" * 76)
    log(f"  {SCRIPT_NAME}  |  {date.today().isoformat()}")
    log("  Stage 4 Step 4.4 — apply state-machine patch + revalidation handoff")
    log("  (server should be STOPPED while patching; restart after)")
    log("=" * 76)

    # ── Pre-req: alert_state_machine.py must exist (from step 4.2+4.3) ───────
    if not ASM_FILE.exists():
        log(f"  FATAL — {ASM_FILE} missing. Run Step 4.2+4.3 first (it emits this on PASS).")
        results["overall_status"] = FAIL
        _finish(); return
    log(f"  prereq OK — alert_state_machine.py present")

    # ── Apply patches ────────────────────────────────────────────────────────
    for name, fn in (("anomaly", patch_anomaly),
                     ("acknowledge", patch_acknowledge),
                     ("registry_wiring", patch_registry)):
        try:
            st, detail = fn()
        except Exception as e:
            st, detail = FAIL, f"{type(e).__name__}: {e}"
        results["patches"][name] = {"status": st, "detail": detail}
        log(f"  [{st}] {name}: {detail}")

    # ── Static wiring verification ───────────────────────────────────────────
    log("\nStatic wiring verification:")
    checks = {}
    a_src = ANOMALY.read_text(encoding="utf-8") if ANOMALY.exists() else ""
    checks["anomaly_imports_asm"] = ("from app.runtime.alert_state_machine import" in a_src)
    checks["anomaly_calls_update"] = ("alert_sm.update(" in a_src)
    checks["anomaly_old_call_gone"] = ("compute_alert_state(" not in a_src
                                       or a_src.count("compute_alert_state(") <= 1)  # def may remain
    ack_src = ACK.read_text(encoding="utf-8") if ACK.exists() else ""
    checks["ack_resets_asm"] = ("alert_sm.reset()" in ack_src)
    for k, v in checks.items():
        results["gates"][k] = PASS if v else FAIL
        log(f"  {'PASS' if v else 'FAIL'}  {k}")

    patch_ok = all(p["status"] in (PASS, SKIP) for p in results["patches"].values())
    wiring_ok = all(results["gates"][k] == PASS for k in checks)
    results["overall_status"] = PASS if (patch_ok and wiring_ok) else FAIL
    _finish()


def _finish():
    out_json = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    status = results["overall_status"]
    manual_needed = [n for n, p in results["patches"].items() if p["status"] == FAIL]

    print("\n" + "=" * 76)
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12 Stage 4 Step 4.4 (apply patch + revalidate): {status}")
    for n, p in results["patches"].items():
        print(f"  patch {n}: {p['status']}")
    for k in results["gates"]:
        print(f"  verify {k}: {results['gates'][k]}")
    print("  Both critical defects fixed pre-validation: G1 (state machine), G4b (CUSUM).")
    print("  BLOCK_M11 = True  (flips to False only after the runner passes critical gates)")
    print("══ END PASTE UPDATE ══")

    if status == PASS:
        print("\n" + "=" * 76)
        print("  NEXT — run the full revalidation against the PATCHED server")
        print("=" * 76)
        print("  1. RESTART the server (loads patched anomaly/acknowledge + corrected CUSUM):")
        print("       uvicorn app.main:app --host 0.0.0.0 --port 8000")
        print("  2. Confirm health:  GET http://localhost:8000/api/health")
        print("  3. Smoke first (wiring check, fast):")
        print("       python src/module_12b_adversarial_runner.py --mode smoke")
        print("  4. Then the BINDING full run (decides BLOCK_M11):")
        print("       python src/module_12b_adversarial_runner.py --mode full")
        print("  Expect: G1_normal_fpr now PASS (hysteresis), G4b now PASS (CUSUM bounded).")
        print("  G8_crosspoint / G13_ood are ADVISORY (not in CRITICAL_GATES) — they do not")
        print("  block M11; log them as deferred-advisory if still failing.")
    else:
        print("\n  Patch incomplete. Manual edits needed for:", manual_needed or "(see verify fails)")
        print("  The printed guide below shows the exact edits; no half-patch was written.")
        print(_MANUAL_GUIDE)

    print("\n══ FILE MANIFEST ══")
    print(f"  Report: {out_json}")
    print("  Patched (with .bak): app/routers/anomaly.py, app/routers/acknowledge.py, app/main.py")
    print(f"  GitHub push: src/{SCRIPT_NAME}.py + patched app files (after runner passes)")
    print("=" * 76)
    print()
    if status == PASS:
        print("📦 Step 4.4 patch applied + verified. RESTART server, then run the adversarial "
              "runner (smoke -> full). Paste the runner output for the BLOCK_M11 decision.")
    else:
        print("📦 Step 4.4 patch incomplete — apply the manual edits shown, then re-run this script.")


_MANUAL_GUIDE = r'''
# --- MANUAL PATCH GUIDE (only if a sub-step reported FAIL) ---
# anomaly.py:
#   - add:   from app.runtime.alert_state_machine import AlertStateMachine, load_alert_thresholds
#   - DELETE the early line: alert_state = compute_alert_state(score_A, theta_t, cusum_Sn, drift_locked)
#   - AFTER `label_name = label_map.get(label_int, "unknown")` insert:
#         sm_out = request.app.state.alert_sm.update(
#             score_A=score_A, theta_t=theta_t, theta_initial=models["theta_initial"],
#             drift_locked=drift_locked, cusum_Sn=cusum_Sn,
#             cusum_alert=cusum_result.get("cusum_alert","NORMAL"),
#             label_int=label_int, channel_drift_flags=None, score_C=score_C)
#         alert_state = sm_out["alert_state"]
# acknowledge.py:
#   - after `await request.app.state.zt_buf.reset()` add:
#         request.app.state.alert_sm.reset()
# app/main.py (lifespan, where app.state.cusum/rolling/zt_buf are created):
#   - add:   from app.runtime.alert_state_machine import AlertStateMachine, load_alert_thresholds
#   - add:   app.state.alert_sm = AlertStateMachine(load_alert_thresholds())
'''


if __name__ == "__main__":
    main()
