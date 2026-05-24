"""
patch_fault_output_schema.py
Adds fault_label_int, physics_context, drift_locked, invariant19_violation
to FaultPrediction in app/schemas/fault_output.py.

Run ONCE from project root:
    python patch_fault_output_schema.py

Safe: idempotent (skips if already patched), backup created before edit.
"""
import shutil, sys
from pathlib import Path

TARGET = Path("app/schemas/fault_output.py")
if not TARGET.exists():
    print(f"[ERROR] Not found: {TARGET}"); sys.exit(1)

src = TARGET.read_text(encoding="utf-8")

if "fault_label_int" in src:
    print("Already patched — skipping."); sys.exit(0)

backup = TARGET.with_suffix(".py.bak_pre_m12")
if not backup.exists():
    shutil.copy2(TARGET, backup)
    print(f"Backup: {backup.name}")

# ── Strategy: find the last field declaration line before the class ends ──────
# Insert the new Optional fields right before the class's last field or
# after m8p6_addendum — whichever is found first.

NEW_FIELDS = '''
    # ── M10 Phase 2.5 / M12 additions ─────────────────────────────────────────
    # fault_label_int: raw int label from M7 — required for M12 gate scoring
    fault_label_int: Optional[int] = None
    # physics_context: label-specific physics lookup dict — required for G14
    physics_context: Optional[dict] = None
    # drift_locked: L4 crosspoint guard state — required for G8
    drift_locked: Optional[bool] = None
    # invariant19_violation: set True if score routing violated — required for G12
    invariant19_violation: Optional[bool] = False
'''

# Try to insert after m8p6_addendum field line
import re
pattern = re.compile(
    r'([ \t]*m8p6_addendum\s*:[^\n]+\n)',
    re.MULTILINE,
)
match = pattern.search(src)

if match:
    insert_pos = match.end()
    patched = src[:insert_pos] + NEW_FIELDS + src[insert_pos:]
    print(f"Inserting after m8p6_addendum at line {src[:insert_pos].count(chr(10))}")
else:
    # Fallback: append before the last blank line / end of class
    # Find last field-like line (    fieldname: type)
    field_lines = [
        (m.start(), m.end())
        for m in re.finditer(r'^    \w+\s*:[^\n]+\n', src, re.MULTILINE)
    ]
    if not field_lines:
        print("[ERROR] Could not find insertion point."); sys.exit(1)
    last_end = field_lines[-1][1]
    patched = src[:last_end] + NEW_FIELDS + src[last_end:]
    print(f"Fallback: inserting after last field")

# Ensure Optional is imported
if "Optional" not in patched:
    patched = patched.replace(
        "from typing import",
        "from typing import Optional, ",
        1,
    )
    if "Optional" not in patched:
        patched = "from typing import Optional\n" + patched

TARGET.write_text(patched, encoding="utf-8")
print("Patched: FaultPrediction now includes fault_label_int, physics_context,")
print("         drift_locked, invariant19_violation")
print(f"Verify: python -c \"from app.schemas.fault_output import FaultPrediction;"
      f" print([f for f in FaultPrediction.model_fields if 'fault_label_int' in f or 'drift' in f])\"")
