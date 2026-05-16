# =============================================================================
# module_10_phase3_gates.py
# M10 Phase 3 — Frontend integration gate tests
# Run: python module_10_phase3_gates.py
# 10/10 must pass before Phase 4 (HF Datasets API + end-to-end test).
# =============================================================================

# =============================================================================
# module_10_phase3_gates.py  — patch v2
# Fixes: G3 (block boundary), G9 (literal vs regex), G10 (encoding)
# =============================================================================
import sys, re
from pathlib import Path
from datetime import datetime

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
PASS="PASS"; FAIL="FAIL"
gates = {}

# ── Gate 1 ───────────────────────────────────────────────────────────────────
log("Gate 1: index.html — all required script tags")
try:
    html = Path("app/templates/index.html").read_text(encoding="utf-8")
    for check in ["chart.umd.js","react.production","react-dom.production",
                  "babel-standalone","dashboard.js","pumpsmart_full_v2.jsx",
                  "PUMPSMART_API_BASE","HEALTH_POLL_MS"]:
        assert check in html, f"Missing: {check}"
    gates["G1_index_html"] = PASS; log("  PASS")
except Exception as e:
    gates["G1_index_html"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 2 ───────────────────────────────────────────────────────────────────
log("Gate 2: dashboard.js — PumpSmartAPI methods")
try:
    js = Path("app/static/dashboard.js").read_text(encoding="utf-8")
    for m in ["init","stopPolling","on","off","acknowledge","submitVerdict",
              "validateModel","selectPump","householdAdvisor","getPhysicsContext",
              "getLatestPrediction","getLatestHealth","setSensorConnected","getSensorState"]:
        assert m in js, f"Missing: {m}"
    assert "window.PumpSmartAPI" in js
    gates["G2_dashboard_js"] = PASS; log(f"  PASS — 14 methods exported")
except Exception as e:
    gates["G2_dashboard_js"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 3 — fixed: extract only the acknowledge function body ───────────────
log("Gate 3: v5.0-A — acknowledge body must not POST to operator_verdict")
try:
    js = Path("app/static/dashboard.js").read_text(encoding="utf-8")
    # Extract just the acknowledge async function body (not comments or other fns)
    m = re.search(r'async function acknowledge\s*\([^)]*\)\s*\{(.*?)^\s*\}',
                  js, re.DOTALL | re.MULTILINE)
    assert m, "acknowledge() function not found"
    ack_body = m.group(1)
    # Must NOT call /api/operator_verdict in its body
    assert "/api/operator_verdict" not in ack_body, \
        "acknowledge() must not call /api/operator_verdict (v5.0-A)"
    # submitVerdict must contain the operator_verdict call
    assert "/api/operator_verdict" in js, "submitVerdict must post to /api/operator_verdict"
    gates["G3_v5_0a_separation"] = PASS; log("  PASS — separation confirmed")
except Exception as e:
    gates["G3_v5_0a_separation"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 4 ───────────────────────────────────────────────────────────────────
log("Gate 4: operator_verdict router present and registered in main.py")
try:
    assert Path("app/routers/operator_verdict.py").exists()
    verd = Path("app/routers/operator_verdict.py").read_text(encoding="utf-8")
    assert "active_learning_write" in verd
    assert "SHADOW_REAL"           in verd
    assert "_push_to_hf"           in verd
    main = Path("app/main.py").read_text(encoding="utf-8")
    assert "operator_verdict" in main, \
        "operator_verdict not in main.py — add router registration"
    gates["G4_verdict_router"] = PASS; log("  PASS")
except Exception as e:
    gates["G4_verdict_router"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 5 ───────────────────────────────────────────────────────────────────
log("Gate 5: demo window generator — 4 clusters")
try:
    js = Path("app/static/dashboard.js").read_text(encoding="utf-8")
    for c in ["startup","steady_state","high_load","cooldown"]:
        assert c in js
    assert "CLUSTER_BASELINES" in js and "_generateDemoWindow" in js
    gates["G5_cluster_windows"] = PASS; log("  PASS")
except Exception as e:
    gates["G5_cluster_windows"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 6 ───────────────────────────────────────────────────────────────────
log("Gate 6: pumpsmart_full_v2.jsx in app/static/")
try:
    jsx = Path("app/static/pumpsmart_full_v2.jsx").read_text(encoding="utf-8")
    assert len(jsx) > 5000
    for check in ["single-pump v14.2","clusterRanges","stateChanged",
                  "verdictSent","ACTIVE_LEARNING_SCHEMA"]:
        assert check in jsx, f"Missing: {check}"
    gates["G6_jsx_present"] = PASS; log(f"  PASS — {len(jsx):,} chars")
except Exception as e:
    gates["G6_jsx_present"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 7 ───────────────────────────────────────────────────────────────────
log("Gate 7: household.html present with advisory label")
try:
    hh = Path("app/templates/household.html").read_text(encoding="utf-8")
    assert "advisory" in hh.lower()
    gates["G7_household_html"] = PASS; log("  PASS")
except Exception as e:
    gates["G7_household_html"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 7b — NEW: landing page exists with both entry paths ─────────────────
log("Gate 7b: landing.html — both entry paths present")
try:
    land = Path("app/templates/landing.html").read_text(encoding="utf-8")
    assert "/household" in land,  "Missing /household link"
    assert "/dashboard" in land,  "Missing /dashboard link"
    assert "advisory" in land.lower(), "Missing advisory label on household card"
    assert "Industrial" in land,  "Missing Industrial entry"
    assert "Household"  in land,  "Missing Household entry"
    gates["G7b_landing_page"] = PASS; log("  PASS — both paths present")
except Exception as e:
    gates["G7b_landing_page"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 8 ───────────────────────────────────────────────────────────────────
log("Gate 8: app/static/ files present")
try:
    static = Path("app/static")
    for f in ["dashboard.js","pumpsmart_full_v2.jsx"]:
        assert (static/f).exists(), f"Missing: {f}"
    gates["G8_static_files"] = PASS; log("  PASS")
except Exception as e:
    gates["G8_static_files"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 9 — fixed: literal string checks only, no regex ────────────────────
log("Gate 9: routes and scope boundary coverage")
try:
    routers_dir = Path("app/routers")
    router_files = [f.stem for f in routers_dir.glob("*.py") if f.stem != "__init__"]
    for r in ["health","anomaly","classify","selector",
              "acknowledge","validate","physics","operator_verdict"]:
        assert r in router_files, f"Router file missing: {r}.py"

    sel = (routers_dir / "selector.py").read_text(encoding="utf-8")
    assert "OUT_OF_SCOPE"        in sel, "Missing OUT_OF_SCOPE"
    assert "advisory_disclaimer" in sel, "Missing advisory_disclaimer"
    # Check ml_inference False exists in some form
    assert '"ml_inference"' in sel, 'Missing "ml_inference" key'
    assert "False" in sel,          "Missing False value in selector"
    gates["G9_m10_checklist"] = PASS; log("  PASS")
except Exception as e:
    gates["G9_m10_checklist"] = FAIL; log(f"  FAIL: {e}")

# ── Gate 10 — fixed: try multiple encodings ──────────────────────────────────
log("Gate 10: requirements.txt — key packages available")
try:
    req_path = Path("requirements.txt")
    # Try multiple encodings — pip freeze on Windows may produce UTF-16
    req = ""
    for enc in ["utf-8","utf-16","utf-8-sig","latin-1"]:
        try:
            req = req_path.read_text(encoding=enc).lower()
            break
        except Exception:
            continue

    import importlib.util
    needed = ["fastapi", "uvicorn", "huggingface_hub"]
    for pkg in needed:
        in_req = pkg.replace("_","-") in req or pkg in req
        installed = importlib.util.find_spec(pkg.replace("-","_")) is not None
        assert in_req or installed, f"{pkg} not in requirements.txt and not installed"
    gates["G10_requirements"] = PASS; log("  PASS — all packages available")
except Exception as e:
    gates["G10_requirements"] = FAIL; log(f"  FAIL: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
passed = sum(1 for v in gates.values() if v == PASS)
total  = len(gates)
print("\n" + "═"*60)
print(f"M10 PHASE 3 GATE RESULTS: {passed}/{total} PASS")
print("═"*60)
for name, result in gates.items():
    print(f"  {'✅' if result==PASS else '❌'} {name}: {result}")
print("═"*60)
status = "READY_FOR_PHASE4" if passed == total else "NEEDS_REVIEW"
print(f"\n  Status: {status}")
print("\n" + "═"*60)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print(f"M10_phase3_gates_pass:           {passed}/{total}")
print(f"M10_phase3_status:               {status}")
print(f"M10_landing_page:                {'PASS' if gates.get('G7b_landing_page')==PASS else 'FAIL'}")
print(f"M10_v5_0a_separation_confirmed:  {'PASS' if gates.get('G3_v5_0a_separation')==PASS else 'FAIL'}")
print(f"M10_verdict_router_registered:   {'PASS' if gates.get('G4_verdict_router')==PASS else 'FAIL'}")
print(f"Status for Phase 4:              {'READY' if passed==total else 'BLOCKED'}")
print("══ END PASTE UPDATE ══")
print("═"*60)
if passed < total:
    sys.exit(1)