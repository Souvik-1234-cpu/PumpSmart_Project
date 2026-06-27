# =============================================================================
# apply_warmup_guard.py
# PumpSmart v14.2 — suppress alerts during θ_t warm-up; show "Calibrating…".
#
# Run from project root:  python apply_warmup_guard.py
#
# Turns the startup transient (cold-θ_t "Unclassified anomaly" flash) into an
# intentional, visible CALIBRATING phase — reads as engineering maturity.
#
# Changes (all backed up, idempotent):
#   1. app/main.py        : add app.state.warmup_target + warmup_count
#   2. app/routers/anomaly.py : increment counter; if warming, force the stashed
#                          latest_prediction alert_state -> "CALIBRATING"
#   3. app/static/dashboard.js : pass CALIBRATING through (no popup); add note
#   4. app/static/...jsx  : ALERT_COLORS + popup gate already ignore CALIBRATING
#                          (only WARN/DANGER pop). Add CALIBRATING color + label.
#
# Warm-up length = rolling_window_size (the θ_t adaptation window). After that
# many live windows, θ_t has settled and real alerts flow normally.
# =============================================================================
import shutil, sys
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
MAIN = ROOT / "app" / "main.py"
ANOM = ROOT / "app" / "routers" / "anomaly.py"
DASH = ROOT / "app" / "static" / "dashboard.js"
# jsx filename varies; find it
JSX_CANDIDATES = list((ROOT / "app" / "static").glob("pumpsmart_full*.jsx"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(p):
    b = p.with_suffix(p.suffix + f".bak_{STAMP}")
    shutil.copy2(p, b); print(f"  backup: {b.name}")


# ── 1. main.py — initialise warmup counters ──────────────────────────────────
def patch_main():
    if not MAIN.exists():
        print(f"SKIP main.py — not found"); return
    src = MAIN.read_text(encoding="utf-8")
    if "warmup_target" in src:
        print("main.py already patched — skipping"); return
    backup(MAIN)
    anchor = "    app.state.commissioning_mode = False"
    if anchor not in src:
        print("  WARN: commissioning_mode anchor not found — main.py NOT patched"); return
    inject = (
        "    # Warm-up guard (v5.2): suppress alerts until θ_t adapts. Length =\n"
        "    # rolling window (the θ_t adaptation horizon). Counter incremented in\n"
        "    # anomaly_detect; alert forced to CALIBRATING until target reached.\n"
        "    app.state.warmup_target = int(app.state.models.get('rolling_window_size', 432))\n"
        "    app.state.warmup_count = 0\n"
        "    log(f\"  warmup guard  : CALIBRATING for first {app.state.warmup_target} windows\")\n"
        + anchor
    )
    src = src.replace(anchor, inject, 1)
    MAIN.write_text(src, encoding="utf-8")
    print("  added warmup_target + warmup_count")
    print("main.py PATCHED")


# ── 2. anomaly.py — increment + force CALIBRATING in stash ───────────────────
def patch_anomaly():
    if not ANOM.exists():
        print(f"SKIP anomaly.py — not found"); return
    src = ANOM.read_text(encoding="utf-8")
    if "CALIBRATING" in src:
        print("anomaly.py already has warmup guard — skipping"); return
    if "latest_prediction" not in src:
        print("  WARN: run apply_dashboard_sync_fix.py FIRST (no latest_prediction stash)"); return
    backup(ANOM)

    # Replace the stash line with a warmup-aware version.
    old_stash = "    request.app.state.latest_prediction = prediction.model_dump()"
    if old_stash not in src:
        # try to find with any indentation
        import re
        m = re.search(r"^(\s*)request\.app\.state\.latest_prediction = prediction\.model_dump\(\)",
                      src, re.M)
        if not m:
            print("  WARN: stash line not found — anomaly.py NOT patched"); return
        old_stash = m.group(0)
        indent = m.group(1)
    else:
        indent = "    "

    new_stash = (
        f"{indent}# Warm-up guard (v5.2): while θ_t adapts, surface CALIBRATING\n"
        f"{indent}# instead of a spurious alert from the cold-threshold transient.\n"
        f"{indent}_pred_dict = prediction.model_dump()\n"
        f"{indent}try:\n"
        f"{indent}    request.app.state.warmup_count += 1\n"
        f"{indent}    if request.app.state.warmup_count <= request.app.state.warmup_target:\n"
        f"{indent}        _pred_dict['alert_state'] = 'CALIBRATING'\n"
        f"{indent}        _pred_dict['fault_label'] = 'calibrating'\n"
        f"{indent}        _pred_dict['warmup_progress'] = round(\n"
        f"{indent}            request.app.state.warmup_count / request.app.state.warmup_target, 3)\n"
        f"{indent}except AttributeError:\n"
        f"{indent}    pass  # counters not initialised (older main.py) — skip guard\n"
        f"{indent}request.app.state.latest_prediction = _pred_dict"
    )
    src = src.replace(old_stash, new_stash, 1)
    ANOM.write_text(src, encoding="utf-8")
    print("  warmup counter + CALIBRATING override added to stash")
    print("anomaly.py PATCHED")


# ── 3. dashboard.js — pass CALIBRATING through cleanly ───────────────────────
def patch_dashboard():
    if not DASH.exists():
        print(f"SKIP dashboard.js — not found"); return
    src = DASH.read_text(encoding="utf-8")
    if "CALIBRATING" in src:
        print("dashboard.js already has warmup guard — skipping"); return
    if "/api/latest_state" not in src:
        print("  WARN: run apply_dashboard_sync_fix.py FIRST"); return
    backup(DASH)
    # In pollInference, after _latestPrediction = data, emit a calibrating event
    anchor = "      _latestPrediction = data;"
    if anchor not in src:
        print("  WARN: anchor not found — dashboard.js NOT patched"); return
    inject = (
        "      _latestPrediction = data;\n"
        "      if (data && data.alert_state === 'CALIBRATING') {\n"
        "        emit('calibrating', { progress: data.warmup_progress || 0 });\n"
        "        emit('prediction', data);   // jsx shows CALIBRATING; no popup (not WARN/DANGER)\n"
        "        return;\n"
        "      }"
    )
    src = src.replace(anchor, inject, 1)
    DASH.write_text(src, encoding="utf-8")
    print("  CALIBRATING pass-through added to pollInference")
    print("dashboard.js PATCHED")


# ── 4. jsx — add CALIBRATING color + label (popup already gated to WARN/DANGER)
def patch_jsx():
    if not JSX_CANDIDATES:
        print("SKIP jsx — not found (add CALIBRATING color manually if needed)"); return
    jsx = JSX_CANDIDATES[0]
    src = jsx.read_text(encoding="utf-8")
    if "CALIBRATING" in src:
        print(f"{jsx.name} already has CALIBRATING — skipping"); return
    backup(jsx)
    old = ('const ALERT_COLORS = { NORMAL:"#00e676", WATCH:"#ffcc00", '
           'WARN:"#ff8800", DANGER:"#ff2244" };')
    new = ('const ALERT_COLORS = { NORMAL:"#00e676", WATCH:"#ffcc00", '
           'WARN:"#ff8800", DANGER:"#ff2244", CALIBRATING:"#3a9bdc" };')
    if old in src:
        src = src.replace(old, new, 1)
        jsx.write_text(src, encoding="utf-8")
        print(f"  added CALIBRATING color (#3a9bdc) to {jsx.name}")
        print(f"{jsx.name} PATCHED")
    else:
        print(f"  WARN: ALERT_COLORS line not found in {jsx.name} — add CALIBRATING color manually")


if __name__ == "__main__":
    print("=== PumpSmart warm-up guard ===")
    print(f"root: {ROOT}")
    patch_main()
    patch_anomaly()
    patch_dashboard()
    patch_jsx()
    print()
    print("NEXT STEPS:")
    print("  1. Restart uvicorn (logs 'warmup guard: CALIBRATING for first N windows').")
    print("  2. Hard-reload dashboard (Ctrl+Shift+R).")
    print("  3. Run interactive diag. During the 432 warmup windows the dashboard")
    print("     shows CALIBRATING (blue) — no spurious 'Unclassified anomaly'.")
    print("     After warmup it flows NORMAL/WATCH/WARN as the fault develops.")
    print(f"  Backups: .bak_{STAMP}")
