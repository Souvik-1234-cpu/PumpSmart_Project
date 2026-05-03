"""
apply_fix_module_06B.py
Permanently fixes generate_seal_failure_fast() in module_06B_steps1to3_combined.py

Run once from project root:
    python src/apply_fix_module_06B.py

Safe: backup created before patching. Idempotent: skips if already patched.
"""
import sys, shutil
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "module_06B_steps1to3_combined.py"
if not TARGET.exists():
    print(f"[ERROR] Not found: {TARGET}")
    sys.exit(1)

src = TARGET.read_text(encoding="utf-8")

if "FIX v2 (2026-05-03)" in src:
    print("Already patched — skipping.")
    sys.exit(0)

backup = TARGET.with_suffix(".py.bak_pre_label19_fix")
if not backup.exists():
    shutil.copy2(TARGET, backup)
    print(f"Backup: {backup.name}")

# The inline version in module_06B has a slightly different docstring
# Identify the function by its unique signature line
import re

# Find the function start
func_pattern = re.compile(
    r'def generate_seal_failure_fast\(steps.*?return seq\.astype\(np\.float32\)',
    re.DOTALL
)
match = func_pattern.search(src)
if not match:
    print("[ERROR] generate_seal_failure_fast not found in module_06B.")
    print("The function may be imported from m6b_physics_lib — no inline fix needed.")
    print("Check: grep 'generate_seal_failure_fast' module_06B_steps1to3_combined.py")
    sys.exit(0)

old_func = match.group(0)

NEW_FUNC = '''def generate_seal_failure_fast(steps, rng=None, cluster_id=1):
    """
    Label 19: Catastrophic seal blowout — turbulent orifice discharge.
    FIX v2 (2026-05-03): Three bugs fixed — see m6b_physics_lib.py for details.
    onset=55-85 (post spike-seed), frac corrected, severity-direct drop used.
    Physics: sev=0.50 → Pres.SV drops to 0.70. sev=0.80 → drops to 0.52.
    """
    seq            = make_baseline(steps, cluster_id)
    onset          = int(rng.integers(55, 85))
    drop_steps     = int(rng.integers(10, 21))
    severity_local = float(rng.uniform(0.20, 0.80))
    max_drop       = float(severity_local * 0.60)
    target_min     = float(max(0.05, 1.0 - max_drop))

    for t in range(onset, min(onset + drop_steps, steps)):
        frac = (t - onset + 1) / drop_steps
        seq[t, CH["Pres.SV"]] = float(max(
            target_min,
            seq[t, CH["Pres.SV"]] - max_drop * frac
        ))
    for t in range(min(onset + drop_steps, steps), steps):
        seq[t, CH["Pres.SV"]] = float(target_min) + float(
            rng.normal(0, NOISE_STD.get("Pres.SV", 0.015)))

    t_sec_end = min(onset + 15, steps)
    seq[onset:t_sec_end, CH["Mot.PV"]] += float(rng.uniform(0.20, 0.35))
    return seq.astype(np.float32)'''

patched = src.replace(old_func, NEW_FUNC, 1)

if patched == src:
    print("[ERROR] Replacement failed — text unchanged.")
    sys.exit(1)

TARGET.write_text(patched, encoding="utf-8")
print(f"Patched: {TARGET.name}")
print("generate_seal_failure_fast() — FIX v2 applied.")
