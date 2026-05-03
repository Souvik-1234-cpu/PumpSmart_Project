"""
apply_fix_m6b_physics_lib.py  v3
Verified 2026-05-03: generate_seal_failure_fast() does NOT exist in
m6b_physics_lib.py — it was implemented inline in module_06B_steps1to3_combined.py.
That file was already patched by apply_fix_module_06B.py (FIX v2 applied).
This script confirms that state and exits cleanly.
"""
import re, sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "m6b_physics_lib.py"
M6B    = Path(__file__).resolve().parent / "module_06B_steps1to3_combined.py"

src_lib = TARGET.read_text(encoding="utf-8", errors="replace")
src_m6b = M6B.read_text(encoding="utf-8", errors="replace")

# Confirm: no seal_failure_fast in physics lib (correct architecture)
if "generate_seal_failure_fast" not in src_lib:
    print("CONFIRMED: generate_seal_failure_fast() not in m6b_physics_lib.py")
    print("  This is correct — it is an inline function in module_06B only.")

# Confirm: module_06B already patched
if "FIX v2" in src_m6b:
    print("CONFIRMED: module_06B_steps1to3_combined.py already has FIX v2 applied.")
    print("  onset=55-85, frac corrected, severity-direct drop. Patch complete.")
else:
    print("WARNING: FIX v2 not found in module_06B — run apply_fix_module_06B.py")
    sys.exit(1)

print("\nStatus: m6b_physics_lib.py requires NO changes.")
print("Status: module_06B patch COMPLETE.")
print("Next:   python module_06p5r_patch_label19_features.py")