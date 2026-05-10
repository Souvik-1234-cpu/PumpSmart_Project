# =============================================================================
# module_08p7c_scope_statement.py
# PumpSmart v14.2 — Tier-1.5 Item T1.5.3: Group E Scope Statement
# =============================================================================
#
# WHY THIS EXISTS (Audit v3.0 §10.3 Concern E):
#   T1.7 Path B renamed Labels 22/23 from "two-sensor common-cause failure"
#   to "single-sensor anomaly". This removes a fault category from the
#   system's claimed coverage without updating user-facing documentation.
#   Anyone reading project documentation expecting multi-sensor common-cause
#   failure detection will not find that coverage.
#
# REQUIRED TEXT (audit-specified):
#   "PumpSmart v14.2 detects single-sensor anomalies (Labels 22, 23).
#    Common-cause multi-sensor failures (shared excitation rail, EMI burst,
#    moisture ingress affecting multiple sensors) are OUT OF SCOPE and
#    require separate detection mechanisms."
#
# WHAT THIS SCRIPT DOES:
#   1. Creates M10 scope disclaimer JSON (loaded by Flask API at startup)
#   2. Creates/updates project scope statement markdown
#   3. Updates fault_rules_v3.json with explicit out-of-scope note
#   4. Writes SCOPE_STATEMENT.md to project root level
#   5. Gates, report, paste-text
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (MODEL_DIR, OUTPUT_DIR)
from datetime import date, datetime
import json, warnings
warnings.filterwarnings('ignore')

SCRIPT_NAME = "module_08p7c_scope_statement"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

GATES   = {}
results = {}

log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}")
log("  T1.5.3 — Group E scope statement + M10 disclaimer update")
log("=" * 72)

# =============================================================================
# AUDIT-MANDATED SCOPE STATEMENT (exact text from audit v3.0 §10.3 Concern E)
# =============================================================================
SCOPE_STATEMENT = (
    "PumpSmart v14.2 detects single-sensor anomalies (Labels 22, 23). "
    "Common-cause multi-sensor failures (shared excitation rail, EMI burst, "
    "moisture ingress affecting multiple sensors) are OUT OF SCOPE and "
    "require separate detection mechanisms."
)
RECLASSIFICATION_NOTE = (
    "Labels 22 and 23 were originally defined as 'two-sensor simultaneous "
    "common-cause failure'. Visualization audit (T1.7, 2026-05-09) confirmed "
    "the generator produces single-channel anomalies only. Path B selected: "
    "class names updated to match generator output. No sequences regenerated."
)

# =============================================================================
# SECTION 1 — M10 SCOPE DISCLAIMER JSON
# =============================================================================
log("\nSECTION 1 — M10 scope disclaimer JSON")

m10_scope = {
    "_meta": {
        "date":           str(date.today()),
        "script":         SCRIPT_NAME,
        "audit_ref":      "PumpSmart Industrial Audit v3.0 §10.3 Concern E / T1.5.3",
        "required_by":    "Audit v3.0 — must appear in M10 Flask API /disclaimer endpoint",
    },
    "scope_statement":         SCOPE_STATEMENT,
    "reclassification_note":   RECLASSIFICATION_NOTE,
    "in_scope": {
        "description": "PumpSmart v14.2 detects the following fault classes on "
                       "a 110 kW 7-stage centrifugal pump matching CIRA SACIP envelope "
                       "(2980 RPM, 40 bar, 450 m, 45 m³/h).",
        "groups": {
            "A_single_faults": {
                "labels": [0,1,2,3,4,5,6,19],
                "description": "Single-mechanism faults: normal, bearing wear, "
                               "impeller imbalance, cavitation, seal failure, "
                               "overloading, sensor failure, seal_failure_fast",
            },
            "B_compound_chains": {
                "labels": [7,8,9,10,11,12],
                "description": "Physically causal compound faults where Fault A "
                               "triggers Fault B across 400–900 steps",
            },
            "C_masked_faults": {
                "labels": [13,14,15,16,17],
                "description": "Sensor failure masking underlying process fault — "
                               "most dangerous for operators",
            },
            "D_cyclic_gradual": {
                "labels": [18,20,21],
                "description": "Intermittent, cyclic, and gradual wear patterns",
            },
            "E_sensor_anomaly": {
                "labels": [22,23],
                "description": "Single-channel sensor anomaly with secondary "
                               "indicator disturbance. "
                               "NOT multi-sensor common-cause failure.",
            },
        },
    },
    "out_of_scope": {
        "multi_sensor_common_cause": {
            "description": SCOPE_STATEMENT,
            "examples": [
                "Shared excitation rail loss affecting two sensors simultaneously",
                "EMI burst corrupting multiple SCADA channels at once",
                "Moisture ingress into junction box affecting both Pmp.SV and Pmp.PV",
                "Power supply fault affecting entire instrument loop",
            ],
            "reason_not_detected": (
                "Labels 22 and 23 generators use 100% pure physics synthesis "
                "with zero CIRA spike-seed anchoring. Without empirical data for "
                "multi-sensor common-cause events, a valid two-channel failure "
                "generator cannot be constructed without fabricating physics. "
                "Path B (reclassification) chosen per T1.7 audit recommendation."
            ),
        },
        "other_pump_types": {
            "description": "Household monoblock pumps, axial flow pumps, "
                           "positive displacement pumps",
            "reason": "Cross-domain ML inference on out-of-distribution equipment "
                      "is a safety risk. Household advisory uses physics-only calculation.",
        },
        "cross_pump_effects": {
            "description": "Parallel pump interaction, recirculation effects",
            "reason": "Single-pump monitoring — cross-pump effects not modelled. "
                      "Tier-3 item T3-5.",
        },
        "novel_fault_classes": {
            "description": "Shaft misalignment, foundation looseness, "
                           "coupling wear, impeller erosion",
            "reason": "Not in 22-class training taxonomy. OOD detector (T1.4) "
                      "flags these as SUSPECTED_OOD with manual inspection required. "
                      "Will be added via active learning (T3-2) as real faults accumulate.",
        },
    },
    "f1_citation_format": (
        "5-fold sequence-stratified cross-validation on physics-synthetic data, "
        "macro F1 = 0.9965 ± 0.0005. "
        "Real-world F1 expected to be 0.65–0.85 per C-26 and published "
        "transfer-learning literature on rotating equipment. "
        "This number is not production-validated."
    ),
    "deployment_posture": (
        "Shadow-mode only with mandatory human-in-the-loop for minimum 6 months. "
        "Not cleared for autonomous alerting or closed-loop trip authority. "
        "Closed-loop trip requires 1oo2/2oo3 voting per IEC 61511 — "
        "PumpSmart can be one voter, not the only voter."
    ),
}

scope_path = MODEL_DIR / "M10_scope_disclaimer.json"
try:
    with open(scope_path, "w", encoding="utf-8") as f:
        json.dump(m10_scope, f, indent=2)
    log(f"  M10 scope disclaimer → {scope_path.name}")
    GATES["T1.5.3_G1_m10_scope_json"] = {"passed": True, "detail": str(scope_path)}
    results["scope_json_saved"] = True
except Exception as e:
    log(f"  [ERROR] {e}")
    GATES["T1.5.3_G1_m10_scope_json"] = {"passed": False, "detail": str(e)}

# =============================================================================
# SECTION 2 — UPDATE fault_rules_v3.json WITH OUT-OF-SCOPE NOTE
# =============================================================================
log("\nSECTION 2 — Update fault_rules_v3.json with scope note")

fault_rules_path = MODEL_DIR / "fault_rules_v3.json"
try:
    with open(fault_rules_path, "r", encoding="utf-8") as f:
        fault_rules = json.load(f)

    fault_rules["_scope_statement"] = {
        "date":              str(date.today()),
        "scope":             SCOPE_STATEMENT,
        "reclassification":  RECLASSIFICATION_NOTE,
        "audit_reference":   "PumpSmart Industrial Audit v3.0 T1.5.3",
        "out_of_scope_note": (
            "Labels 22 and 23 are single-sensor anomaly classes. "
            "Multi-sensor common-cause failures are NOT detected by this system."
        ),
    }

    with open(fault_rules_path, "w", encoding="utf-8") as f:
        json.dump(fault_rules, f, indent=2)
    log(f"  fault_rules_v3.json updated with scope note")
    GATES["T1.5.3_G2_fault_rules_updated"] = {"passed": True, "detail": "scope note added"}
except Exception as e:
    log(f"  [ERROR] {e}")
    GATES["T1.5.3_G2_fault_rules_updated"] = {"passed": False, "detail": str(e)}

# =============================================================================
# SECTION 3 — WRITE SCOPE_STATEMENT.md
# =============================================================================
log("\nSECTION 3 — Write SCOPE_STATEMENT.md")

scope_md = f"""# PumpSmart v14.2 — Scope Statement

**Date:** {date.today()}
**Audit reference:** PumpSmart Industrial Audit v3.0, Section 10.3, Concern E (T1.5.3)

---

## Asset Envelope

PumpSmart v14.2 is designed and validated for:

- **Asset type:** Multi-stage centrifugal pump (inline, end-suction)
- **Nameplate:** 110 kW · 7-stage · 40 bar · 450 m · 2980 RPM · 45 m³/h
- **Motor:** IEC frame 315 mm · 400 V · 2-pole
- **Dataset anchor:** CIRA SACIP (Zenodo) — same pump class
- **Deployment posture:** Shadow-mode advisory only · Human-in-the-loop mandatory

---

## Fault Classes Detected (22 Classes)

| Group | Labels | Description |
|---|---|---|
| A — Single faults | 0,1,2,3,4,5,6,19 | Normal, bearing wear, imbalance, cavitation, seal failure, overloading, sensor failure, seal_failure_fast |
| B — Compound chains | 7,8,9,10,11,12 | Physically causal compound faults (Fault A → Fault B, 400–900 steps) |
| C — Masked faults | 13,14,15,16,17 | Sensor failure masking underlying process fault |
| D — Cyclic/Gradual | 18,20,21 | Intermittent, cyclic, gradual bearing wear |
| E — Sensor anomaly | 22,23 | **Single-channel** sensor anomaly with secondary indicator disturbance |

---

## ⚠ OUT OF SCOPE — MULTI-SENSOR COMMON-CAUSE FAILURE

> **{SCOPE_STATEMENT}**

Labels 22 (sensor_anomaly_thermal) and 23 (sensor_anomaly_pump) detect
**single-channel** sensor faults only. The following multi-sensor scenarios
are NOT detected:

- Shared excitation rail loss affecting two sensors simultaneously
- EMI burst corrupting multiple SCADA channels at once
- Moisture ingress into a junction box affecting both Pmp.SV and Pmp.PV
- Power supply fault affecting entire instrument loop

**Why:** The original "two-sensor simultaneous failure" class definition was
aspirational — the generator produced single-channel anomalies (visualization
audit T1.7, 2026-05-09). Without real-data anchoring, a valid two-channel
generator cannot be built without fabricating physics. Path B (reclassification)
was selected as the honest engineering choice.

---

## ⚠ OUT OF SCOPE — OTHER CONDITIONS

| Condition | Reason |
|---|---|
| Household monoblock pumps | Cross-domain ML = out-of-distribution inference = safety risk |
| Axial / positive displacement pumps | Different hydraulic physics |
| Parallel pump interaction | Single-pump monitoring only |
| Shaft misalignment | Not in 22-class taxonomy — OOD detector flags these |
| Foundation looseness | Not in 22-class taxonomy — OOD detector flags these |
| Autonomous trip authority | IEC 61511 requires 1oo2/2oo3 voting — PumpSmart is one voter only |

---

## F1 Citation Format

When citing performance figures, always use this complete form:

> *5-fold sequence-stratified cross-validation on physics-synthetic data,
> macro F1 = 0.9965 ± 0.0005. Real-world F1 expected to be 0.65–0.85 per
> C-26 and published transfer-learning literature on rotating equipment.
> This number is not production-validated.*

**Do not cite 0.9965 alone** — without the C-26 disclaimer it implies
production-validated performance.

---

*PumpSmart v14.2 · T1.5.3 scope statement · {date.today()}*
"""

scope_md_path = OUTPUT_DIR / "SCOPE_STATEMENT.md"
try:
    with open(scope_md_path, "w", encoding="utf-8") as f:
        f.write(scope_md)
    log(f"  SCOPE_STATEMENT.md → {scope_md_path}")
    GATES["T1.5.3_G3_scope_md"] = {"passed": True, "detail": str(scope_md_path)}
except Exception as e:
    log(f"  [ERROR] {e}")
    GATES["T1.5.3_G3_scope_md"] = {"passed": False, "detail": str(e)}

# =============================================================================
# SECTION 4 — GATES
# =============================================================================
log("\nSECTION 4 — Gates")
for n, g in GATES.items():
    log(f"  {'PASS' if g['passed'] else 'FAIL'}  {n}: {g['detail']}")

n_pass = sum(1 for g in GATES.values() if g["passed"])
n_fail = len(GATES) - n_pass
log(f"\n  Gates: {n_pass}/{n_pass+n_fail} PASS")
results.update({"gates_passed": n_pass, "gates_failed": n_fail,
                 "all_pass": n_fail == 0})

# =============================================================================
# SECTION 5 — REPORT
# =============================================================================
gate_table = "\n".join(
    f"| {n} | {'PASS' if g['passed'] else 'FAIL'} | {g['detail']} |"
    for n, g in GATES.items()
)
report = f"""# {SCRIPT_NAME} — Report
**Date:** {date.today()}
**Status:** {"COMPLETE" if n_fail == 0 else "NEEDS REVIEW"}

## Purpose

Documents the Group E scope reduction following T1.7 Path B reclassification.
Ensures user-facing documentation discloses that multi-sensor common-cause
failures are out of scope, as required by Audit v3.0 §10.3 Concern E.

## Scope Statement

> {SCOPE_STATEMENT}

## Files Written

| File | Purpose |
|---|---|
| `models/M10_scope_disclaimer.json` | M10 Flask API /disclaimer endpoint |
| `models/fault_rules_v3.json` | Updated with `_scope_statement` key |
| `outputs/SCOPE_STATEMENT.md` | Human-readable project scope document |

## Gates

| Gate | Status | Detail |
|---|---|---|
{gate_table}

---
*{SCRIPT_NAME} | PumpSmart v14.2 | {date.today()}*
"""
rp = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
with open(rp, "w", encoding="utf-8") as f:
    f.write(report)
log(f"  Report → {rp}")

print()
print("=" * 72)
print("== PASTE TEXT UPDATE ==")
print(f"T1.5.3_status          = {'COMPLETE' if n_fail==0 else 'NEEDS_REVIEW'}")
print(f"T1.5.3_scope_json      = models/M10_scope_disclaimer.json")
print(f"T1.5.3_scope_md        = outputs/SCOPE_STATEMENT.md")
print(f"T1.5.3_fault_rules_upd = True")
print(f"T1.5.3_gates           = {n_pass}/{n_pass+n_fail}")
print()
print("## Tier-1.5 Queue")
print("T1.5.1: module_08p7a_m4_layernorm_fix.py")
print("T1.5.2: module_08p7b_slope_continuity_gate.py")
print("T1.5.3: module_08p7c_scope_statement.py  ← THIS SCRIPT")
print("After all 3 complete: M10 is UNBLOCKED")
print("== END PASTE UPDATE ==")
print()
print("-- FILE MANIFEST --")
print(f"NEW: {scope_path}")
print(f"NEW: {scope_md_path}")
print(f"NEW: {rp}")
print(f"UPDATED: {fault_rules_path}")
print("GitHub push: all .json, .md files above, this script")
print("M10 /disclaimer endpoint must load: models/M10_scope_disclaimer.json")

log("\n[DONE]")
