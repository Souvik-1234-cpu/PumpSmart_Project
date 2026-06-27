# =============================================================================
# fix_calibrating_log_filter.py
# PumpSmart v14.2 — two fixes for the residual startup flash:
#   1. jsx: skip event-log + popup entirely while alert_state == CALIBRATING
#      (CALIBRATING is a setup phase, not an event worth logging).
#   2. anomaly.py: tighten the θ_t-settled guard so no cold-transient WARN
#      slips through before θ_t settles. Guard holds until θ_t is within a
#      tight absolute band of its settled value, not merely below 60% of boot.
#
# Run from project root AFTER fix_warmup_guard_theta.py:
#   python fix_calibrating_log_filter.py
# =============================================================================
import shutil, re
from pathlib import Path
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path.cwd()
ANOM = ROOT / "app" / "routers" / "anomaly.py"
JSX  = next(iter((ROOT / "app" / "static").glob("pumpsmart_full*.jsx")), None)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(p):
    b = p.with_suffix(p.suffix + f".bak_{STAMP}"); shutil.copy2(p, b)
    print(f"  backup: {b.name}")


# ── 1. jsx — skip log + popup during CALIBRATING ─────────────────────────────
def patch_jsx():
    if JSX is None:
        print("SKIP jsx — not found"); return
    src = JSX.read_text(encoding="utf-8")
    if "alert_state === \"CALIBRATING\"" in src or "alert_state === 'CALIBRATING'" in src:
        print(f"{JSX.name} already filters CALIBRATING — skipping"); return
    backup(JSX)

    anchor = 'window.PumpSmartAPI.on("prediction", (data) => {\n      setPrediction(data);'
    if anchor not in src:
        # tolerate single quotes / spacing
        m = re.search(r'window\.PumpSmartAPI\.on\(\s*["\']prediction["\']\s*,\s*\(data\)\s*=>\s*\{\s*\n(\s*)setPrediction\(data\);', src)
        if not m:
            print("  WARN: prediction handler not found — jsx NOT patched"); return
        indent = m.group(1)
        anchor = m.group(0)
        replacement = (
            f'window.PumpSmartAPI.on("prediction", (data) => {{\n'
            f'{indent}setPrediction(data);\n'
            f'{indent}setAlertState(data.alert_state || "NORMAL");\n'
            f'{indent}setLastUpdate(new Date().toLocaleTimeString());\n'
            f'{indent}// Warm-up: show CALIBRATING in vitals but do NOT log or pop up.\n'
            f'{indent}if (data.alert_state === "CALIBRATING") {{ setApiError(null); return; }}'
        )
        # also need to drop the now-duplicated setAlertState/setLastUpdate lines that follow
        src = src.replace(anchor, replacement, 1)
        # remove the immediate duplicate lines that originally followed setPrediction
        src = re.sub(
            r'(if \(data\.alert_state === "CALIBRATING"\) \{ setApiError\(null\); return; \}\n)\s*setAlertState\(data\.alert_state \|\| "NORMAL"\);\n\s*setLastUpdate\(new Date\(\)\.toLocaleTimeString\(\)\);\n\s*setApiError\(null\);',
            r'\1      setApiError(null);', src, count=1)
        JSX.write_text(src, encoding="utf-8")
        print(f"  {JSX.name}: CALIBRATING skips log + popup")
        print(f"{JSX.name} PATCHED"); return

    # Standard path: insert the guard right after setLastUpdate / setApiError(null)
    insert_after = "      setApiError(null);\n"
    guard = ('      // Warm-up: show CALIBRATING in vitals but do NOT log or pop up.\n'
             '      if (data.alert_state === "CALIBRATING") return;\n')
    # Insert the guard before the setEventLog call
    log_anchor = "      setEventLog(prev => [{"
    if log_anchor in src:
        src = src.replace(log_anchor, guard + log_anchor, 1)
        JSX.write_text(src, encoding="utf-8")
        print(f"  {JSX.name}: CALIBRATING skips log + popup (guard before setEventLog)")
        print(f"{JSX.name} PATCHED")
    else:
        print("  WARN: setEventLog anchor not found — jsx NOT patched")


# ── 2. anomaly.py — tighten θ_t settled band ─────────────────────────────────
def patch_anomaly():
    if not ANOM.exists():
        print("SKIP anomaly.py — not found"); return
    src = ANOM.read_text(encoding="utf-8")
    if "_theta_settled_abs" in src:
        print("anomaly.py already tightened — skipping"); return
    if "_theta_settled_frac" not in src:
        print("  WARN: run fix_warmup_guard_theta.py first — no θ_t guard present"); return
    backup(ANOM)

    old = (
        "        _theta_init = float(models.get('theta_initial', 1.881) or 1.881)\n"
        "        _theta_settled_frac = 0.60   # θ_t must drop below 60% of boot\n"
        "        _settled = theta_t < (_theta_settled_frac * _theta_init)\n"
    )
    new = (
        "        _theta_init = float(models.get('theta_initial', 1.881) or 1.881)\n"
        "        # Settled = θ_t has fallen close to its warmed value. Use an ABSOLUTE\n"
        "        # ceiling (settled θ_t ≈ 0.178; allow 2.5× headroom = 0.45) so the\n"
        "        # guard holds through the full warm-up and no cold-transient WARN\n"
        "        # slips through. theta_initial fraction kept as a secondary bound.\n"
        "        _theta_settled_abs = 0.45\n"
        "        _settled = theta_t < _theta_settled_abs\n"
    )
    if old not in src:
        print("  WARN: θ_t guard block not in expected form — manual edit needed"); return
    src = src.replace(old, new, 1)
    ANOM.write_text(src, encoding="utf-8")
    print("  anomaly.py: θ_t settled band tightened (abs ceiling 0.45)")
    print("anomaly.py PATCHED")


if __name__ == "__main__":
    print("=== CALIBRATING log filter + tighter θ_t guard ===")
    print(f"root: {ROOT}")
    patch_anomaly()
    patch_jsx()
    print()
    print("Restart uvicorn + hard-reload dashboard, then run the diag.")
    print("Expect: CALIBRATING shows in vitals during warm-up, NO event-log entry,")
    print("NO popup; real NORMAL/WATCH/WARN flow only after θ_t < 0.45 (settled).")
    print(f"Backups: .bak_{STAMP}")
