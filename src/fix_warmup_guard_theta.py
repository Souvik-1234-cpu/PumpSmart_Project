# =============================================================================
# fix_warmup_guard_theta.py
# PumpSmart v14.2 — make the warm-up guard θ_t-based, not counter-based.
#
# WHY: the counter (warmup_count) is cumulative across diag runs and never
# resets except on server restart. After the first run exhausts it, later runs
# get no CALIBRATING guard → the cold-θ_t "Unclassified anomaly" flash returns.
#
# FIX: guard on θ_t itself. During warm-up θ_t sits near its boot value
# (theta_initial ≈ 1.88); once adapted it drops to ≈0.178. While θ_t is still
# above a "settled" fraction of theta_initial, surface CALIBRATING. This
# auto-re-arms on EVERY fresh warmup (θ_t resets with it) — no counter.
#
# Run from project root AFTER apply_dashboard_sync_fix.py (+ optionally after
# apply_warmup_guard.py — this supersedes its anomaly.py stash logic):
#   python fix_warmup_guard_theta.py
# =============================================================================
import shutil, re
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
ANOM = ROOT / "app" / "routers" / "anomaly.py"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(p):
    b = p.with_suffix(p.suffix + f".bak_{STAMP}")
    shutil.copy2(p, b); print(f"  backup: {b.name}")


def main():
    if not ANOM.exists():
        print("anomaly.py not found"); return
    src = ANOM.read_text(encoding="utf-8")
    if "_theta_settled_frac" in src:
        print("already θ_t-guarded — skipping"); return
    if "latest_prediction" not in src:
        print("run apply_dashboard_sync_fix.py first"); return
    backup(ANOM)

    # Find the stash region (either the plain sync-fix stash OR the counter
    # guard from apply_warmup_guard.py) and replace it with θ_t-based logic.
    # Anchor on the assignment to latest_prediction (last occurrence).
    lines = src.splitlines(keepends=True)

    # Remove any prior counter-guard block: from the "_pred_dict = prediction.model_dump()"
    # line through "request.app.state.latest_prediction = _pred_dict" if present.
    joined = src
    counter_block = re.search(
        r"[ \t]*# Warm-up guard.*?request\.app\.state\.latest_prediction = _pred_dict",
        joined, re.S)
    plain_stash = re.search(
        r"[ \t]*# Stash latest full prediction.*?\n[ \t]*request\.app\.state\.latest_prediction = prediction\.model_dump\(\)",
        joined, re.S)
    bare_stash = re.search(
        r"[ \t]*request\.app\.state\.latest_prediction = prediction\.model_dump\(\)",
        joined)

    if counter_block:
        target = counter_block.group(0)
        indent = target[:len(target) - len(target.lstrip())]
    elif plain_stash:
        target = plain_stash.group(0)
        indent = "    "
    elif bare_stash:
        target = bare_stash.group(0)
        indent = target[:len(target) - len(target.lstrip())]
    else:
        print("  WARN: stash block not found — no change"); return

    new_block = (
        f"{indent}# Warm-up guard (v5.2, θ_t-based — auto re-arms each fresh warmup).\n"
        f"{indent}# While θ_t is still near its boot value (theta_initial), the\n"
        f"{indent}# adaptive threshold has not settled → surface CALIBRATING instead\n"
        f"{indent}# of a spurious cold-threshold alert. Settled when θ_t has fallen\n"
        f"{indent}# below _theta_settled_frac × theta_initial.\n"
        f"{indent}_pred_dict = prediction.model_dump()\n"
        f"{indent}try:\n"
        f"{indent}    _theta_init = float(models.get('theta_initial', 1.881) or 1.881)\n"
        f"{indent}    _theta_settled_frac = 0.60   # θ_t must drop below 60% of boot\n"
        f"{indent}    _settled = theta_t < (_theta_settled_frac * _theta_init)\n"
        f"{indent}    if not _settled:\n"
        f"{indent}        _pred_dict['alert_state'] = 'CALIBRATING'\n"
        f"{indent}        _pred_dict['fault_label'] = 'calibrating'\n"
        f"{indent}        _pred_dict['warmup_theta_t'] = round(float(theta_t), 4)\n"
        f"{indent}except Exception:\n"
        f"{indent}    pass\n"
        f"{indent}request.app.state.latest_prediction = _pred_dict"
    )
    src = src.replace(target, new_block, 1)
    ANOM.write_text(src, encoding="utf-8")
    print("  replaced stash with θ_t-based CALIBRATING guard")
    print("anomaly.py PATCHED (θ_t-based warm-up guard)")
    print()
    print("Restart uvicorn, hard-reload dashboard, run the diag (any mode).")
    print("CALIBRATING now shows whenever θ_t is un-settled — every run, not just")
    print("the first. After θ_t drops below 60% of boot (~1.13), real states flow.")


if __name__ == "__main__":
    print("=== θ_t-based warm-up guard ==="); print(f"root: {ROOT}")
    main()
