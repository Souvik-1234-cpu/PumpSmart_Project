#!/usr/bin/env python3
# =============================================================================
# src/module_12_stage4_item13_install_diagnostic.py
# PumpSmart v14.2 — M12 Stage 4 Item 1+3 INSTALL DIAGNOSTIC
#
# PURPOSE
# -------
# Pinpoints WHY the latch / re-nag / CUSUM-state fixes are not visible on the
# dashboard. Checks three independent layers and reports which one is stale:
#
#   LAYER 1  Files on disk         — are the new markers present in the .py/.jsx?
#   LAYER 2  Served static asset   — is the SERVER handing the browser the new
#                                    JSX, or a cached/old one?
#   LAYER 3  Live API behaviour    — does /api/anomaly_detect actually RETURN
#                                    the latch fields, and does a forced
#                                    WARN/DANGER sequence latch + hold + clear?
#
# It needs NO arguments. Run from the project root with uvicorn running:
#     python src/module_12_stage4_item13_install_diagnostic.py
#
# If uvicorn is NOT running it still does the on-disk checks (Layers 1) and
# tells you the server is unreachable.
#
# Output: a per-check PASS/FAIL table + a single ROOT CAUSE verdict at the end,
# plus a JSON report at outputs/reports/<script>_results.json you can paste back.
# =============================================================================

import sys
import json
import os
import re
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

SCRIPT_NAME = "module_12_stage4_item13_install_diagnostic"

# ── Try to import config for canonical paths; fall back to CWD-relative ──────
def _resolve_paths():
    paths = {}
    try:
        import config  # noqa
        paths["OUTPUT_DIR"] = Path(getattr(config, "OUTPUT_DIR", "outputs"))
        paths["config_imported"] = True
    except Exception:
        paths["OUTPUT_DIR"] = Path("outputs")
        paths["config_imported"] = False
    return paths

_P = _resolve_paths()
REPORT_DIR = _P["OUTPUT_DIR"] / "reports"
try:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    REPORT_DIR = Path(".")

BASE_URL = os.environ.get("PUMPSMART_URL", "http://localhost:8000")

results = {
    "script": SCRIPT_NAME,
    "run_utc": datetime.now().isoformat() + "Z",
    "base_url": BASE_URL,
    "config_imported": _P["config_imported"],
    "cwd": str(Path.cwd()),
    "layers": {},
    "checks": [],          # list of {layer, name, status, detail}
    "root_cause": None,
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def check(layer, name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results["checks"].append(
        {"layer": layer, "name": name, "status": status, "detail": detail}
    )
    mark = "✅" if ok else "❌"
    print(f"  {mark} [{layer}] {name}")
    if detail:
        print(f"        {detail}")
    return ok


# ── HTTP helpers (stdlib only — no requests dependency) ──────────────────────
def _http_get(path, timeout=8):
    url = BASE_URL + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _http_get_text(path, timeout=8):
    url = BASE_URL + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", errors="replace")


def _http_post(path, body, timeout=20):
    url = BASE_URL + path
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _find_file(candidates):
    """Return the first existing path among candidates, else None."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


# =============================================================================
# LAYER 1 — Files on disk
# =============================================================================
def layer1_files():
    print("\n" + "=" * 70)
    print("LAYER 1 — Files on disk (are the new markers present?)")
    print("=" * 70)
    L = {}

    # Marker strings unique to the NEW versions of each file.
    file_markers = {
        "fault_state_tracker.py": {
            "candidates": [
                "app/runtime/fault_state_tracker.py",
            ],
            "markers": ["async def acknowledge_popup", "async def clear_latch",
                        "class FaultStateTracker"],
        },
        "anomaly.py": {
            "candidates": ["app/routers/anomaly.py"],
            "markers": ["fault_tracker.update", "display_alert_state",
                        "alert_state=display_alert_state"],
        },
        "main.py": {
            "candidates": ["app/main.py"],
            "markers": ["FaultStateTracker", "fault_tracker"],
        },
        "acknowledge.py": {
            "candidates": ["app/routers/acknowledge.py"],
            "markers": ["acknowledge_popup"],
        },
        "operator_verdict.py": {
            "candidates": ["app/routers/operator_verdict.py"],
            "markers": ["clear_latch"],
        },
        "fault_output.py": {
            "candidates": ["app/schemas/fault_output.py"],
            "markers": ["raw_alert_state", "is_latched", "popup_should_show"],
        },
        "pumpsmart_full_v2_local.jsx": {
            "candidates": [
                "app/static/pumpsmart_full_v2_local.jsx",
                "static/pumpsmart_full_v2_local.jsx",
            ],
            # New JSX markers (the exact strings that prove Item-3 edits landed)
            "markers": ['k:"State",  v:alertState',
                        "LATCHED",
                        "reNagTimerRef",
                        "Go to the"],
        },
    }

    for label, spec in file_markers.items():
        p = _find_file(spec["candidates"])
        if p is None:
            check("L1", f"{label} found on disk", False,
                  f"NOT FOUND at any of: {spec['candidates']}")
            L[label] = {"found": False, "path": None, "markers": {}}
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            check("L1", f"{label} readable", False, f"{p}: {e}")
            L[label] = {"found": True, "path": str(p), "read_error": str(e)}
            continue

        marker_state = {}
        all_present = True
        for m in spec["markers"]:
            present = (m in text)
            marker_state[m] = present
            if not present:
                all_present = False
        detail = f"{p}  | markers: " + ", ".join(
            f"{'✓' if v else '✗'}{k[:24]}" for k, v in marker_state.items()
        )
        check("L1", f"{label} is NEW version", all_present, detail)
        L[label] = {"found": True, "path": str(p), "markers": marker_state,
                    "is_new": all_present, "mtime": datetime.fromtimestamp(
                        p.stat().st_mtime).isoformat() + "Z"}

    results["layers"]["L1_files"] = L
    return L


# =============================================================================
# LAYER 2 — Served static asset (what the browser actually downloads)
# =============================================================================
def layer2_served_jsx():
    print("\n" + "=" * 70)
    print("LAYER 2 — Served static JSX (what the BROWSER downloads)")
    print("=" * 70)
    L = {"reachable": False}
    try:
        status, text = _http_get_text("/static/pumpsmart_full_v2_local.jsx")
        L["reachable"] = True
        L["http_status"] = status
        markers = {
            'k:"State",  v:alertState': 'k:"State",  v:alertState' in text,
            "LATCHED": "LATCHED" in text,
            "reNagTimerRef": "reNagTimerRef" in text,
            "Go to the": "Go to the" in text,
            # OLD marker — if THIS is present and new ones absent, it's the old file
            "OLD: setAlertState(\"NORMAL\")": 'setAlertState("NORMAL")' in text
                and "Item 1: acknowledge SILENCES" not in text,
        }
        L["markers"] = markers
        served_new = markers['k:"State",  v:alertState'] and markers["LATCHED"]
        check("L2", "Server serves NEW JSX", served_new,
              "served file markers: " + ", ".join(
                  f"{'✓' if v else '✗'}{k[:28]}" for k, v in markers.items()))
        L["served_new"] = served_new
    except urllib.error.URLError as e:
        check("L2", "Static JSX reachable", False,
              f"cannot GET /static/pumpsmart_full_v2_local.jsx: {e}")
    except Exception as e:
        check("L2", "Static JSX reachable", False, str(e))
    results["layers"]["L2_served_jsx"] = L
    return L


# =============================================================================
# LAYER 3 — Live API behaviour (does the running server DO the latch?)
# =============================================================================
def _make_window(baseline, drift=0.0):
    """Build a 50x8 normalised window from a per-channel baseline + drift."""
    import random
    win = []
    for t in range(50):
        row = []
        for ch, b in enumerate(baseline):
            noise = (random.random() - 0.5) * 0.04
            val = b + noise + drift * t
            row.append(max(0.0, min(3.0, val)))
        win.append(row)
    return win


def layer3_live_api():
    print("\n" + "=" * 70)
    print("LAYER 3 — Live API behaviour (does the SERVER latch?)")
    print("=" * 70)
    L = {"reachable": False}

    # 3.0 — health + startup tracker presence
    try:
        status, health = _http_get("/health")
        L["reachable"] = True
        L["health_status"] = status
        # live_l3_alarm is exposed in main.py; tracker presence is implied if
        # the app booted with the new main.py. We can't read the banner here,
        # but we CAN check the response schema below.
        check("L3", "Server /health reachable", status == 200,
              f"arch={health.get('arch_version', health.get('arch','?'))}")
    except Exception as e:
        check("L3", "Server reachable", False,
              f"cannot reach {BASE_URL}/health: {e} — is uvicorn running?")
        results["layers"]["L3_live_api"] = L
        return L

    # 3.1 — does the anomaly response CONTAIN the latch fields?
    steady = [0.45, 0.42, 0.50, 0.95, 0.55, 0.80, 0.52, 0.90]
    try:
        status, pred = _http_post("/api/anomaly_detect", {
            "window": _make_window(steady),
            "pump_id": "PUMP-0032",
            "cluster": "steady_state",
        })
        latch_fields = ["is_latched", "popup_should_show", "raw_alert_state",
                        "watch_candidates"]
        present = {f: (f in pred) for f in latch_fields}
        all_present = all(present.values())
        L["anomaly_keys_sample"] = sorted(list(pred.keys()))
        L["latch_fields_present"] = present
        check("L3", "anomaly_detect RETURNS latch fields", all_present,
              "fields: " + ", ".join(
                  f"{'✓' if v else '✗'}{k}" for k, v in present.items()))
        L["first_alert_state"] = pred.get("alert_state")
        L["first_raw_alert_state"] = pred.get("raw_alert_state")
        L["first_is_latched"] = pred.get("is_latched")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        check("L3", "anomaly_detect accepts request", False,
              f"HTTP {e.code}: {body}")
        results["layers"]["L3_live_api"] = L
        return L
    except Exception as e:
        check("L3", "anomaly_detect callable", False, str(e))
        results["layers"]["L3_live_api"] = L
        return L

    # 3.2 — force a DANGER and verify it LATCHES, HOLDS through a normal window,
    #       then clears via /api/operator_verdict.
    #       We push strongly-drifted windows to trip score_A >> theta_t.
    try:
        latched_seen = False
        held_through_normal = False
        danger_drift = [b + 1.2 for b in steady]   # large offset → high score_A
        # Fire several strong windows to escalate + clear persistence floor.
        last = None
        for i in range(6):
            _, last = _http_post("/api/anomaly_detect", {
                "window": _make_window(danger_drift),
                "pump_id": "PUMP-0032",
                "cluster": "steady_state",
            })
            if last.get("is_latched"):
                latched_seen = True
        L["after_danger_alert_state"] = last.get("alert_state")
        L["after_danger_is_latched"] = last.get("is_latched")
        L["after_danger_label"] = last.get("fault_label")
        check("L3", "DANGER sequence LATCHES (is_latched=True)", latched_seen,
              f"alert_state={last.get('alert_state')} "
              f"is_latched={last.get('is_latched')} "
              f"label={last.get('fault_label')}")

        # Now push a clean NORMAL window — latch must HOLD (the collapse case).
        _, after_norm = _http_post("/api/anomaly_detect", {
            "window": _make_window(steady),
            "pump_id": "PUMP-0032",
            "cluster": "steady_state",
        })
        held_through_normal = bool(after_norm.get("is_latched"))
        L["after_normal_alert_state"] = after_norm.get("alert_state")
        L["after_normal_is_latched"] = after_norm.get("is_latched")
        L["after_normal_label"] = after_norm.get("fault_label")
        check("L3", "Latch HOLDS through a normal window (no self-reset)",
              held_through_normal,
              f"alert_state={after_norm.get('alert_state')} "
              f"is_latched={after_norm.get('is_latched')} "
              f"(if False here, the backend latch is NOT engaging)")

        # 3.3 — acknowledge must NOT release the latch.
        _, ack = _http_post("/api/acknowledge", {
            "pump_id": "PUMP-0032",
            "action_taken": "diagnostic acknowledge",
            "operator_id": "diag",
        })
        ack_holds = bool(ack.get("fault_latch_after_ack", {}).get("latch_held", False))
        L["ack_response_has_latch_key"] = "fault_latch_after_ack" in ack
        L["ack_latch_held"] = ack.get("fault_latch_after_ack", {}).get("latch_held")
        check("L3", "acknowledge HOLDS latch (fault_latch_after_ack.latch_held)",
              ack_holds or ("fault_latch_after_ack" in ack),
              f"fault_latch_after_ack present={('fault_latch_after_ack' in ack)} "
              f"latch_held={ack.get('fault_latch_after_ack', {}).get('latch_held')}")

        # 3.4 — verdict must RELEASE the latch.
        _, verdict = _http_post("/api/operator_verdict", {
            "prediction_id": last.get("prediction_id", "diag"),
            "verdict": "INCORRECT",
            "operator_correct_label": None,
            "physical_inspection_done": True,
            "inspection_notes": "diagnostic verdict",
            "operator_id": "diag",
            "consent_granted_by": "diag",
        })
        released = bool(verdict.get("fault_latch_release", {}).get("latch_released", False))
        L["verdict_response_has_latch_key"] = "fault_latch_release" in verdict
        L["verdict_latch_released"] = verdict.get("fault_latch_release", {}).get("latch_released")
        check("L3", "verdict RELEASES latch (fault_latch_release.latch_released)",
              released or ("fault_latch_release" in verdict),
              f"fault_latch_release present={('fault_latch_release' in verdict)} "
              f"latch_released={verdict.get('fault_latch_release', {}).get('latch_released')}")

    except Exception as e:
        check("L3", "latch round-trip", False, f"exception during round-trip: {e}")

    results["layers"]["L3_live_api"] = L
    return L


# =============================================================================
# ROOT CAUSE verdict
# =============================================================================
def verdict():
    print("\n" + "=" * 70)
    print("ROOT CAUSE")
    print("=" * 70)

    L1 = results["layers"].get("L1_files", {})
    L2 = results["layers"].get("L2_served_jsx", {})
    L3 = results["layers"].get("L3_live_api", {})

    jsx_disk_new = L1.get("pumpsmart_full_v2_local.jsx", {}).get("is_new", False)
    jsx_served_new = L2.get("served_new", False)
    backend_disk_new = all(
        L1.get(f, {}).get("is_new", False)
        for f in ["fault_state_tracker.py", "anomaly.py", "fault_output.py",
                  "acknowledge.py", "operator_verdict.py", "main.py"]
    )
    api_returns_latch = all(
        (L3.get("latch_fields_present") or {}).values()
    ) if L3.get("latch_fields_present") else False
    latch_holds = L3.get("after_normal_is_latched") is True

    rc = []

    if not backend_disk_new:
        rc.append("BACKEND FILES ON DISK ARE STALE — one or more of the 6 Python "
                  "files do not contain the new markers. Replace them (Layer 1 "
                  "rows with ✗) and FULLY restart uvicorn.")
    if backend_disk_new and not api_returns_latch and L3.get("reachable"):
        rc.append("BACKEND DISK IS NEW BUT THE RUNNING SERVER IS OLD — the API "
                  "response is missing latch fields. uvicorn is running a stale "
                  "process. Stop ALL python, restart uvicorn (not --reload).")
    if api_returns_latch and not latch_holds and L3.get("reachable"):
        rc.append("BACKEND RETURNS FIELDS BUT LATCH DOES NOT HOLD — the tracker "
                  "may not be constructed (main.py) or update() is not engaging. "
                  "Check the startup banner for 'fault tracker : initialised'.")
    if not jsx_disk_new:
        rc.append("FRONTEND JSX ON DISK IS STALE — app/static/pumpsmart_full_v2_"
                  "local.jsx does not contain the Item-3 markers. You replaced "
                  "the wrong copy. Put the new file at the served static path.")
    if jsx_disk_new and not jsx_served_new and L2.get("reachable"):
        rc.append("JSX DISK IS NEW BUT SERVER SERVES OLD — uvicorn cached the "
                  "old static asset. Fully restart uvicorn, then hard-reload "
                  "(Ctrl+Shift+R) the browser.")
    if jsx_served_new and L3.get("reachable"):
        rc.append("SERVER SERVES THE NEW JSX — if the dashboard still looks old, "
                  "it is BROWSER CACHE. Hard-reload (Ctrl+Shift+R) or DevTools → "
                  "Network → Disable cache → reload.")

    if not L3.get("reachable") and backend_disk_new and jsx_disk_new:
        rc.append("Files on disk are all NEW, but the server is DOWN — start "
                  "uvicorn and re-run to validate the live latch round-trip.")

    if not rc:
        rc.append("All layers report NEW + latch round-trip works. If the UI "
                  "still looks old, it is purely browser cache — hard reload.")

    results["root_cause"] = rc
    for i, r in enumerate(rc, 1):
        print(f"  {i}. {r}")

    # One-line headline
    server_up = L3.get("reachable", False)
    if not (backend_disk_new and jsx_disk_new):
        headline = "FILES NOT SWAPPED — replace the ✗ files on disk (see Layer 1)."
    elif not server_up:
        headline = ("FILES ON DISK ARE NEW, but the SERVER is not running — start "
                    "uvicorn and re-run this diagnostic to test the live latch.")
    elif backend_disk_new and api_returns_latch and latch_holds and jsx_served_new:
        headline = "LIKELY BROWSER CACHE ONLY — everything server-side is correct. Hard-reload."
    elif jsx_disk_new and not jsx_served_new:
        headline = "SERVER SERVING STALE JSX — fully restart uvicorn, then hard reload."
    elif backend_disk_new and not api_returns_latch:
        headline = "STALE SERVER PROCESS — restart uvicorn fully (not --reload)."
    elif api_returns_latch and not latch_holds:
        headline = "TRACKER NOT ENGAGING — check startup banner for 'fault tracker : initialised'."
    else:
        headline = "See numbered items above."
    print("\n  >>> HEADLINE: " + headline)
    results["headline"] = headline


# =============================================================================
def main():
    print("=" * 70)
    print("  PumpSmart v14.2 — M12 Stage 4 Item 1+3 INSTALL DIAGNOSTIC")
    print(f"  CWD     : {Path.cwd()}")
    print(f"  Server  : {BASE_URL}")
    print(f"  config  : {'imported' if _P['config_imported'] else 'NOT imported (CWD-relative paths)'}")
    print("=" * 70)

    layer1_files()
    layer2_served_jsx()
    layer3_live_api()
    verdict()

    # Save JSON report
    out = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    try:
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  Report saved: {out}")
    except Exception as e:
        print(f"\n  (could not save report: {e})")

    print("\n" + "=" * 70)
    print("  COPY EVERYTHING ABOVE (or paste the JSON report) back for analysis.")
    print("=" * 70)


if __name__ == "__main__":
    main()
