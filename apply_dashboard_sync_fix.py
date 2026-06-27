# =============================================================================
# apply_dashboard_sync_fix.py
# PumpSmart v14.2 — make the dashboard MIRROR the injected stream.
#
# Run from project root:  python apply_dashboard_sync_fix.py
#
# Does two things, with backups:
#   1. app/routers/anomaly.py:
#        - stash latest prediction on app.state before `return prediction`
#        - append a read-only GET /api/latest_state route
#   2. app/static/dashboard.js:
#        - replace pollInference() to GET /api/latest_state (no demo windows)
#        - speed INFERENCE_POLL default to 2000ms
#
# Idempotent: re-running detects the markers and skips.
# =============================================================================
import shutil, sys, re
from pathlib import Path
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path.cwd()
ANOM = ROOT / "app" / "routers" / "anomaly.py"
DASH = ROOT / "app" / "static" / "dashboard.js"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(p):
    b = p.with_suffix(p.suffix + f".bak_{STAMP}")
    shutil.copy2(p, b)
    print(f"  backup: {b.name}")


# ── 1. anomaly.py ────────────────────────────────────────────────────────────
def patch_anomaly():
    if not ANOM.exists():
        print(f"SKIP anomaly.py — not found at {ANOM}")
        return
    src = ANOM.read_text(encoding="utf-8")
    if "latest_state" in src:
        print("anomaly.py already patched — skipping")
        return
    backup(ANOM)

    # (a) stash latest prediction right before the final `return prediction`
    # Find the LAST occurrence of a line that is exactly `    return prediction`
    lines = src.splitlines(keepends=True)
    idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].rstrip("\n").strip() == "return prediction":
            idx = i
            break
    if idx is None:
        print("  WARN: could not find `return prediction` — stash not inserted")
    else:
        indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip())]
        stash = (
            f"{indent}# Stash latest full prediction for read-only viewers "
            f"(dashboard sync — v5.2)\n"
            f"{indent}request.app.state.latest_prediction = prediction.model_dump()\n\n"
        )
        lines.insert(idx, stash)
        src = "".join(lines)
        print("  inserted latest_prediction stash before return")

    # (b) append the read-only route at end of file
    route = '''

# =============================================================================
# Read-only latest-state endpoint (v5.2 — dashboard sync)
# Returns the most recent FaultPrediction WITHOUT running inference or mutating
# any state. Lets the dashboard mirror whatever stream (terminal rig / SCADA)
# drives the server, instead of generating its own demo windows.
# One driver, one viewer -> identical numbers on terminal and dashboard.
# =============================================================================
@router.get("/latest_state")
async def latest_state(request: Request):
    latest = getattr(request.app.state, "latest_prediction", None)
    if latest is None:
        return {"status": "no_data",
                "detail": "No inference has run yet. Drive a window first."}
    return latest
'''
    if not src.endswith("\n"):
        src += "\n"
    src += route
    ANOM.write_text(src, encoding="utf-8")
    print("  appended GET /api/latest_state route")
    print("anomaly.py PATCHED")


# ── 2. dashboard.js ──────────────────────────────────────────────────────────
NEW_POLLINFERENCE = '''  // ── /api/latest_state polling (v5.2 — read-only viewer) ──────────────────
  // The dashboard MIRRORS whatever drives the server (terminal rig / SCADA).
  // It no longer generates its own synthetic windows (that produced numbers
  // unrelated to the actual injected stream). One driver, one viewer.
  async function pollInference() {
    try {
      const data = await _get('/api/latest_state');
      if (data && data.status === 'no_data') {
        emit('inference_paused', { reason: 'no_stream_yet' });
        return;
      }
      _latestPrediction = data;
      if (data.fault_label !== undefined) {
        const labelInt = _labelNameToInt(data.fault_label);
        if (labelInt !== null && !_physicsCache[labelInt]) {
          _fetchPhysicsContext(labelInt);
        }
      }
      emit('prediction', data);
    } catch (e) {
      emit('inference_error', { error: e.message });
    }
  }
'''


def patch_dashboard():
    if not DASH.exists():
        print(f"SKIP dashboard.js — not found at {DASH}")
        return
    src = DASH.read_text(encoding="utf-8")
    if "/api/latest_state" in src:
        print("dashboard.js already patched — skipping")
        return
    backup(DASH)

    # Replace the whole pollInference function body.
    # Match from `  async function pollInference() {` to its closing `  }\n`
    # that precedes `  function startInferencePolling`.
    start_marker = "  async function pollInference() {"
    end_marker = "  function startInferencePolling() {"
    s = src.find(start_marker)
    e = src.find(end_marker)
    if s == -1 or e == -1 or e < s:
        print("  WARN: could not locate pollInference block — dashboard NOT patched")
        return
    # keep any leading comment block immediately above start_marker? We replace
    # from start_marker to end_marker (exclusive), leaving end_marker intact.
    src = src[:s] + NEW_POLLINFERENCE + "\n" + src[e:]
    print("  replaced pollInference() with read-only viewer")

    # Speed the poll default 50000 -> 2000
    src2 = src.replace("window.INFERENCE_POLL_MS || 50000",
                       "window.INFERENCE_POLL_MS || 2000")
    if src2 != src:
        print("  inference poll default 50000ms -> 2000ms")
        src = src2

    DASH.write_text(src, encoding="utf-8")
    print("dashboard.js PATCHED")


if __name__ == "__main__":
    print("=== PumpSmart dashboard-sync fix ===")
    print(f"root: {ROOT}")
    patch_anomaly()
    patch_dashboard()
    print()
    print("NEXT STEPS:")
    print("  1. Restart uvicorn (loads new /api/latest_state route).")
    print("  2. Hard-reload the dashboard in browser (Ctrl+Shift+R).")
    print("  3. Run the interactive diag, inject bearing (label 1).")
    print("     Dashboard should now climb to score_A ~0.43, label 7, WARN —")
    print("     matching the terminal, because both read the same server state.")
    print("  Backups written with suffix .bak_%s" % STAMP)
