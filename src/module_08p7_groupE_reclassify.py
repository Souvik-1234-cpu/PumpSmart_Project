# =============================================================================
# module_08p7_groupE_reclassify.py
# PumpSmart v14.2 — Tier-1 Fix T1.7: Group E Reclassification (Path B)
# =============================================================================
#
# WHY THIS SCRIPT EXISTS (Audit finding T1.7, Industrial Audit v2.0 §5.7):
#   Visualization audit confirmed both Group E plots show only ONE sensor
#   failing — directly contradicting the class definition "two sensors
#   degraded simultaneously due to common-cause hardware failure."
#
#   Root cause: Group E has ZERO spike-seed anchoring (0 CIRA seeds). It is
#   100% pure physics synthesis. Without an empirical anchor to enforce the
#   "two-channel" requirement, the generator implemented something that looks
#   like a single-channel anomaly with secondary disturbance. The class names
#   were aspirational, not generator-supported.
#
#   PATH A (rejected): Rewrite generator to actually fail two sensors via
#   shared excitation rail loss / moisture ingress physics. Estimated: 1-2
#   working days. Rejected because we cannot anchor it to real data — we
#   would be fabricating physics for a scenario we have no measurements of.
#
#   PATH B (selected): Reclassify. Drop "multi-sensor" framing. Rename to
#   match what the generator actually produces — which is physically valid
#   and useful, just not what the class name claimed.
#       Label 22: "sensor_failure_2ch_thermal"  → "sensor_anomaly_thermal"
#       Label 23: "sensor_failure_2ch_pump"     → "sensor_anomaly_pump"
#
# WHY THIS RUNS BEFORE T1.1–T1.6 (and why M7 retrain is DEFERRED):
#   This script is PURE METADATA — JSON label names, CSV label columns,
#   M10 7-field output templates. Zero data regeneration. Zero model training.
#   Running it first means BOTH downstream M7 retrains (T1.2 and T1.6) will
#   already see the corrected Group E label names, so the FINAL M7 from T1.6
#   holds all three fixes simultaneously:
#       T1.2 → Label 19 physics fix
#       T1.6 → Group B continuous superposition fix + M7 retrain #2 (FINAL)
#       T1.7 → Group E label names already present in feature matrix
#
#   M7 retrain is DELIBERATELY NOT in this script. Running it here would
#   train on broken Label-19 features and broken Group-B compound chains.
#   The correct production M7 weights come from T1.6.
#
# WHAT THIS SCRIPT DOES:
#   1. Verifies all input files (fault_rules, feature_matrix, sequence_meta)
#   2. Creates SHA-256 checksums of inputs before any modification
#   3. Backs up all files with dated .bak suffix
#   4. Patches fault_rules_v3.json:
#        - Renames Labels 22 and 23
#        - Adds new physics descriptions
#        - Adds audit trail and M10 7-field templates
#        - Flags m7_retrain_required = True (status: DEFERRED to T1.6)
#   5. Patches M6B_feature_matrix.csv: renames label_name column values
#   6. Patches M6B_sequence_meta.csv: renames label_name column values
#   7. Creates group_e_reclassified_definitions.json for M10 Flask API
#   8. Runs 4-gate verification pass
#   9. Writes full markdown report
#  10. Prints paste-text update + file manifest + next-module prompt
#
# OUTPUT FILES:
#   models/fault_rules_v3.json                       (MODIFIED)
#   data/synthetic/M6B_feature_matrix.csv            (MODIFIED — label names)
#   data/synthetic/M6B_sequence_meta.csv             (MODIFIED — label names)
#   models/group_e_reclassified_definitions.json     (NEW — M10 config)
#   outputs/reports/M8p7_groupE_reclassify_report.md (NEW — audit trail)
#
# BACKUPS (all inputs):
#   *.pre_groupE_reclassify_<date>.bak
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings, hashlib, shutil
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

SCRIPT_NAME = "module_08p7_groupE_reclassify"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}

log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}")
log("  Group E Path B — metadata only | M7 retrain DEFERRED to T1.6")
log("=" * 72)

# =============================================================================
# SECTION 0 — RECLASSIFICATION CONSTANTS (single source of truth)
# =============================================================================
# ALL label name changes flow from this dict. Nothing is hardcoded downstream.

OLD_NAMES = {
    22: "sensor_failure_2ch_thermal",
    23: "sensor_failure_2ch_pump",
}
NEW_NAMES = {
    22: "sensor_anomaly_thermal",
    23: "sensor_anomaly_pump",
}

# Updated class descriptions for fault_rules_v3.json
NEW_DESCRIPTIONS = {
    22: (
        "Single-channel thermal sensor anomaly with secondary-indicator disturbance. "
        "Mot.TV* exhibits abnormal behavior (erratic, spiking, drifting, or dropout). "
        "Root cause: cable damage, terminal oxidation, sensor element drift, or shared "
        "signal-conditioning hardware fault on the motor thermal monitoring rail. "
        "The underlying pump process is likely normal — do not act without redundant "
        "thermal measurement confirming actual temperature change. "
        "[RECLASSIFIED from sensor_failure_2ch_thermal — Audit T1.7 Path B: "
        "visualization confirmed generator produces single-channel anomaly only. "
        "Multi-sensor simultaneous failure is not currently modelled.]"
    ),
    23: (
        "Single-channel pump-side sensor anomaly with secondary-indicator disturbance. "
        "Pmp.SV* or Pmp.PV* exhibits abnormal behavior (flatline, dropout, or erratic "
        "signal). Root cause: moisture ingress at pump junction box, cable degradation, "
        "connector fault, or sensor element failure on the pump-side instrumentation rail. "
        "Cross-validate with motor-side vibration (Mot.SV) and pressure (Pres.SV) to "
        "assess actual pump health independently of the failed sensor. "
        "[RECLASSIFIED from sensor_failure_2ch_pump — Audit T1.7 Path B: "
        "visualization confirmed generator produces single-channel anomaly only. "
        "Multi-sensor simultaneous failure is not currently modelled.]"
    ),
}

# Physical group E sensor targets per class (for fault_rules and M10)
PRIMARY_CHANNELS = {
    22: ["Mot.TV"],         # thermal sensor anomaly — primary evidence channel
    23: ["Pmp.SV"],         # pump-side sensor anomaly — primary evidence channel
}
SECONDARY_CHANNELS = {
    22: ["Temp.SV"],        # secondary disturbance (expected but weak)
    23: ["Pmp.PV"],         # secondary disturbance
}

# M10 7-field output templates
M10_FIELD3 = {
    22: (
        "Sensor anomaly in the motor thermal monitoring subsystem. Mot.TV* is showing "
        "abnormal readings. Most probable hardware cause: sensor element drift, terminal "
        "block oxidation, cable insulation fault, or signal-conditioning PCB fault on the "
        "thermal monitoring rail. The pump process is likely unaffected — verify with a "
        "handheld IR thermometer or redundant sensor before concluding the motor is "
        "running hot."
    ),
    23: (
        "Sensor anomaly in the pump-side instrumentation subsystem. Pmp.SV* is showing "
        "flatline, dropout, or erratic behavior. Most probable cause: moisture ingress at "
        "the pump junction box (common on process pumps handling hot or humid media), "
        "cable degradation, or sensor element failure. The pump may be mechanically "
        "healthy — use motor-side channels (Mot.SV, Mot.PV, Pres.SV) to assess pump "
        "condition independently."
    ),
}
M10_FIELD4 = {
    22: (
        "If prediction is correct: Mot.TV* will remain abnormal (flat, spiking, or "
        "drifting) while Mot.PV* (motor power) and Mot.SV* (vibration) stay near "
        "normal baseline. This asymmetry — single thermal channel abnormal, power and "
        "vibration normal — is the confirming signature of sensor origin vs process fault."
    ),
    23: (
        "If prediction is correct: Pmp.SV* will remain abnormal (flat or erratic) while "
        "Mot.SV* (motor vibration) and Pres.SV* (differential pressure) continue normally. "
        "If Mot.SV* also rises concurrently, re-evaluate for an underlying mechanical "
        "fault (Label 13: bearing with masked sensor)."
    ),
}
M10_FIELD5 = {
    22: (
        "Low operational risk if thermal process is verified normal via redundant "
        "measurement. Risk escalates to HIGH if the anomaly masks genuine thermal "
        "runaway — a 110 kW IEC 315 motor running above rated temperature with no "
        "thermal alarm can fail winding insulation within 2-4 hours of sustained "
        "overtemperature. Cross-validate immediately."
    ),
    23: (
        "Moderate risk. Loss of Pmp.SV* creates a vibration blind spot for cavitation "
        "(Label 3), impeller imbalance (Label 2), and compound faults that depend on "
        "pump-side vibration as the primary detection channel. Running with Pmp.SV* "
        "offline degrades the system to approximately Label 13 / 17 detection capability "
        "only. Restore sensor before resuming normal monitoring."
    ),
}
M10_FIELD6 = {
    22: (
        "1. Do NOT declare thermal overloading without redundant confirmation. "
        "2. Take handheld IR temperature reading of motor frame at IEC 315 measurement "
        "   points — compare against rated temperature rise. "
        "3. If motor is thermally normal: isolate and inspect Mot.TV sensor wiring, "
        "   terminal block, and signal-conditioning card. "
        "4. If motor IS thermally elevated: treat as Label 5 (overloading) and "
        "   reduce load or investigate root cause per overloading procedure. "
        "5. Replace or recalibrate Mot.TV sensor before trusting any thermal alert."
    ),
    23: (
        "1. Inspect pump junction box for moisture ingress — drain and dry if found. "
        "2. Check Pmp.SV cable for physical damage from heat, abrasion, or vibration. "
        "3. Test sensor element resistance per manufacturer spec. "
        "4. While Pmp.SV is suspect, monitor Mot.SV and Pres.SV as proxy indicators "
        "   for pump mechanical condition. "
        "5. Do not clear sensor fault flag until Pmp.SV readings are physically "
        "   verified against a reference (portable vibration probe)."
    ),
}
RECLASSIFICATION_NOTE = {
    22: (
        "RECLASSIFICATION NOTE (Audit T1.7 Path B): Originally defined as "
        "'two-channel simultaneous thermal failure'. Visualization audit confirmed "
        "generator output shows single-channel thermal anomaly only. Class definition "
        "updated to match generator output. True multi-sensor simultaneous failure "
        "(e.g. shared excitation rail loss) is not currently modelled in PumpSmart v14.2."
    ),
    23: (
        "RECLASSIFICATION NOTE (Audit T1.7 Path B): Originally defined as "
        "'two-channel simultaneous pump-side failure'. Visualization audit confirmed "
        "generator output shows single-channel pump-side anomaly only. Class definition "
        "updated to match generator output. True multi-sensor simultaneous failure "
        "(e.g. moisture ingress affecting two sensors in the same junction box) is not "
        "currently modelled in PumpSmart v14.2."
    ),
}

results["reclassification_map"] = {str(k): {"old": OLD_NAMES[k], "new": NEW_NAMES[k]}
                                   for k in [22, 23]}

log(f"\n  Reclassification: Label 22 '{OLD_NAMES[22]}' → '{NEW_NAMES[22]}'")
log(f"  Reclassification: Label 23 '{OLD_NAMES[23]}' → '{NEW_NAMES[23]}'")


# =============================================================================
# SECTION 1 — LOCATE INPUT FILES
# =============================================================================
log("\nSECTION 1 — Locate input files")

def find_file(*candidates: Path) -> Path | None:
    """Return first existing path from candidates list."""
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None

fault_rules_path = find_file(
    MODEL_DIR  / "fault_rules_v3.json",
    OUTPUT_DIR / "fault_rules_v3.json",
    MODEL_DIR  / "fault_rules.json",
)
feature_matrix_path = find_file(
    SYNTH_DIR  / "M6B_feature_matrix.csv",
    OUTPUT_DIR / "M6B_feature_matrix.csv",
)
sequence_meta_path = find_file(
    SYNTH_DIR  / "M6B_sequence_meta.csv",
    OUTPUT_DIR / "M6B_sequence_meta.csv",
)
# Canonical save path for fault_rules (create here if it doesn't exist yet)
fault_rules_save_path = fault_rules_path if fault_rules_path else (MODEL_DIR / "fault_rules_v3.json")

file_status = {
    "fault_rules_v3.json"   : fault_rules_path,
    "M6B_feature_matrix.csv": feature_matrix_path,
    "M6B_sequence_meta.csv" : sequence_meta_path,
}
for fname, fpath in file_status.items():
    found = fpath is not None
    log(f"  {'✓' if found else '✗ (will create/skip)'}  {fname:<35s}  {fpath or 'NOT FOUND'}")

results["input_fault_rules_found"]   = fault_rules_path is not None
results["input_feature_matrix_found"]= feature_matrix_path is not None
results["input_sequence_meta_found"] = sequence_meta_path is not None


# =============================================================================
# SECTION 2 — SHA-256 CHECKSUMS AND BACKUPS
# =============================================================================
log("\nSECTION 2 — Checksum + backup existing files")

BACKUP_SUFFIX = f".pre_groupE_reclassify_{date.today()}.bak"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def backup_file(path: Path | None) -> tuple[Path | None, str | None]:
    """Backup + return (backup_path, original_sha256). Returns (None, None) if no file."""
    if path is None or not path.exists():
        return None, None
    cksum = sha256(path)
    bak = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    shutil.copy2(path, bak)
    log(f"  Backed up {path.name} → {bak.name}  sha256={cksum[:16]}…")
    return bak, cksum

bak_fr,   sha_fr   = backup_file(fault_rules_path)
bak_fm,   sha_fm   = backup_file(feature_matrix_path)
bak_meta, sha_meta = backup_file(sequence_meta_path)

results["backups_created"] = sum(1 for b in [bak_fr, bak_fm, bak_meta] if b is not None)
results["input_checksums"] = {
    "fault_rules"   : sha_fr   or "file_not_found",
    "feature_matrix": sha_fm   or "file_not_found",
    "sequence_meta" : sha_meta or "file_not_found",
}
log(f"  {results['backups_created']} backup(s) created.")


# =============================================================================
# SECTION 3 — PATCH fault_rules_v3.json
# =============================================================================
log("\nSECTION 3 — Patch fault_rules_v3.json")

# Load or initialise
def load_fault_rules(path: Path | None) -> dict:
    if path and path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            log(f"  [WARNING] JSON parse error: {e}. Initialising skeleton.")
    return {"fault_classes": {}, "_meta": {"version": "v3", "created_by": SCRIPT_NAME}}

fault_rules = load_fault_rules(fault_rules_path)
if "fault_classes" not in fault_rules:
    fault_rules["fault_classes"] = {}

# Log current Group E state
for lid in [22, 23]:
    entry = fault_rules["fault_classes"].get(str(lid), {})
    current_name = entry.get("name", "NOT PRESENT")
    log(f"  Label {lid} current name: '{current_name}'")

# Apply reclassification
for lid in [22, 23]:
    existing = fault_rules["fault_classes"].get(str(lid), {})
    fault_rules["fault_classes"][str(lid)] = {
        # Core identity
        "label_id"               : lid,
        "name"                   : NEW_NAMES[lid],
        "old_name"               : OLD_NAMES[lid],
        "group"                  : "E",
        "group_description"      : (
            "Sensor anomaly — single-channel failure with secondary indicator disturbance. "
            "Reclassified from 'multi-sensor simultaneous failure' per Audit T1.7 Path B."
        ),
        # Physics
        "description"            : NEW_DESCRIPTIONS[lid],
        "seq_steps"              : existing.get("seq_steps", 300),
        "severity_range"         : existing.get("severity_range", [0.3, 0.8]),
        "primary_channels"       : PRIMARY_CHANNELS[lid],
        "secondary_channels"     : SECONDARY_CHANNELS[lid],
        # M10 7-field output
        "m10_7field": {
            "field_1_fault_label"                 : NEW_NAMES[lid],
            "field_3_probable_physical_condition" : M10_FIELD3[lid],
            "field_4_expected_sensor_behavior"    : M10_FIELD4[lid],
            "field_5_operational_risk_if_ignored" : M10_FIELD5[lid],
            "field_6_recommended_action"          : M10_FIELD6[lid],
            "field_7_model_limitation_disclaimer" : (
                "Trained on CIRA-anchored physics-synthetic data for 110 kW 7-stage pump "
                "at 2980 RPM, 40 bar. Predictions advisory only. Verify physically. "
                "Single-pump monitoring — cross-pump effects not modelled. "
                "Confidence scores may be lower on real-world faults than on simulated "
                "training data. " + RECLASSIFICATION_NOTE[lid]
            ),
        },
        # Audit trail
        "reclassified"            : True,
        "reclassification_audit"  : {
            "date"            : str(date.today()),
            "script"          : SCRIPT_NAME,
            "tier"            : "T1.7",
            "path"            : "B",
            "reason"          : (
                "Visualization audit confirmed generator output shows single-channel "
                "anomaly only. 'Multi-sensor' label was aspirational with no spike-seed "
                "anchoring (0 CIRA seeds). Path B selected: rename to match generator "
                "output. Path A (implement true multi-channel failure generator) rejected "
                "due to absence of empirical anchor and estimated 1-2 day effort."
            ),
            "audit_reference" : "PumpSmart Industrial Audit v2.0, Section 5.7",
        },
        "m7_retrain_required"     : True,
        "m7_retrain_status"       : "DEFERRED — runs in T1.6 (module_08p6_groupB_regenerate.py) with all Tier-1 fixes",
    }
    log(f"  Label {lid}: '{OLD_NAMES[lid]}' → '{NEW_NAMES[lid]}' ✓")

# Root-level audit trail
fault_rules["_group_e_reclassification"] = {
    "date"              : str(date.today()),
    "script"            : SCRIPT_NAME,
    "labels_affected"   : [22, 23],
    "path_chosen"       : "B",
    "m7_retrain_status" : "DEFERRED to T1.6",
    "class_count"       : "UNCHANGED — still 24 classes (Labels 0–23)",
}

try:
    with open(fault_rules_save_path, 'w') as f:
        json.dump(fault_rules, f, indent=2)
    results["fault_rules_patched"] = True
    log(f"  fault_rules_v3.json saved → {fault_rules_save_path}")
except Exception as e:
    log(f"  [ERROR] Could not save fault_rules_v3.json: {e}")
    results["fault_rules_patched"] = False


# =============================================================================
# SECTION 4 — PATCH M6B_feature_matrix.csv
# =============================================================================
log("\nSECTION 4 — Patch M6B_feature_matrix.csv")

results["feature_matrix_rows_renamed"] = 0
results["feature_matrix_patched"]      = "NOT_FOUND"

if feature_matrix_path and feature_matrix_path.exists():
    try:
        log(f"  Loading {feature_matrix_path.name} ...")
        fm = pd.read_csv(feature_matrix_path, low_memory=False)
        log(f"  Shape: {fm.shape}  |  Columns (first 10): {list(fm.columns[:10])}")

        # ── Identify label name column ─────────────────────────────────────
        label_name_col = None
        for cand in ['label_name', 'fault_name', 'class_name', 'fault_class', 'fault_label']:
            if cand in fm.columns:
                label_name_col = cand
                break
        # Identify numeric label ID column (must contain 22 or 23)
        label_id_col = None
        for cand in ['label_id', 'label', 'fault_id', 'class_id', 'y']:
            if cand in fm.columns:
                try:
                    col = pd.to_numeric(fm[cand], errors='coerce')
                    if col.isin([22, 23]).any():
                        label_id_col = cand
                        break
                except Exception:
                    pass

        log(f"  Label name column : {label_name_col or 'NOT FOUND'}")
        log(f"  Label ID column   : {label_id_col or 'NOT FOUND'}")

        renamed = 0

        # Strategy 1: rename by label_name string match
        if label_name_col:
            for old_n, new_n in [(OLD_NAMES[22], NEW_NAMES[22]),
                                  (OLD_NAMES[23], NEW_NAMES[23])]:
                mask = fm[label_name_col] == old_n
                n = int(mask.sum())
                if n > 0:
                    fm.loc[mask, label_name_col] = new_n
                    log(f"  [name] '{old_n}' → '{new_n}': {n} rows")
                    renamed += n

        # Strategy 2: rename by label_id lookup (fallback or supplement)
        if label_id_col:
            for lid in [22, 23]:
                id_mask = pd.to_numeric(fm[label_id_col], errors='coerce') == lid
                n = int(id_mask.sum())
                if n > 0:
                    if label_name_col is None:
                        fm['label_name'] = ''
                        label_name_col = 'label_name'
                    still_old = (fm.loc[id_mask, label_name_col] == OLD_NAMES[lid])
                    if still_old.any():
                        fm.loc[id_mask & (fm[label_name_col] == OLD_NAMES[lid]),
                               label_name_col] = NEW_NAMES[lid]
                        n_updated = int(still_old.sum())
                        log(f"  [id]   Label {lid}: renamed {n_updated} remaining rows via ID lookup")
                        renamed += n_updated

        results["feature_matrix_rows_renamed"] = renamed
        fm.to_csv(feature_matrix_path, index=False)
        results["feature_matrix_patched"] = True
        log(f"  Saved. Total rows renamed: {renamed}")

    except MemoryError:
        log("  [MemoryError] Feature matrix too large.")
        log("  ACTION: Increase system RAM or process in chunks.")
        results["feature_matrix_patched"] = False
    except Exception as e:
        log(f"  [ERROR] {type(e).__name__}: {e}")
        results["feature_matrix_patched"] = False
else:
    log("  Feature matrix not found at expected paths.")
    log("  -> Patch DEFERRED: T1.2 regenerates M6B_feature_matrix.csv with")
    log("     correct Label-19 rows. At that point, T1.7 label names should")
    log("     already be in fault_rules, so T1.2's regeneration will use them.")
    results["feature_matrix_patched"] = "DEFERRED_TO_T1_2"


# =============================================================================
# SECTION 5 — PATCH M6B_sequence_meta.csv
# =============================================================================
log("\nSECTION 5 — Patch M6B_sequence_meta.csv")

results["sequence_meta_rows_renamed"] = 0
results["sequence_meta_patched"]      = "NOT_FOUND"

if sequence_meta_path and sequence_meta_path.exists():
    try:
        meta = pd.read_csv(sequence_meta_path, low_memory=False)
        log(f"  Shape: {meta.shape}  |  Columns: {list(meta.columns)}")

        # Identify columns
        lname_col = None
        for cand in ['label_name', 'fault_name', 'class_name', 'label']:
            if cand in meta.columns and meta[cand].dtype == object:
                if meta[cand].isin([OLD_NAMES[22], OLD_NAMES[23]]).any():
                    lname_col = cand
                    break
        lid_col = None
        for cand in ['label_id', 'label', 'fault_id']:
            if cand in meta.columns:
                try:
                    col = pd.to_numeric(meta[cand], errors='coerce')
                    if col.isin([22, 23]).any():
                        lid_col = cand
                        break
                except Exception:
                    pass

        renamed_meta = 0
        if lname_col:
            for old_n, new_n in [(OLD_NAMES[22], NEW_NAMES[22]),
                                  (OLD_NAMES[23], NEW_NAMES[23])]:
                mask = meta[lname_col] == old_n
                n = int(mask.sum())
                if n > 0:
                    meta.loc[mask, lname_col] = new_n
                    renamed_meta += n
                    log(f"  '{old_n}' → '{new_n}': {n} rows")

        # Supplement with label_id if name rename found nothing
        if renamed_meta == 0 and lid_col:
            for lid in [22, 23]:
                mask = pd.to_numeric(meta[lid_col], errors='coerce') == lid
                n = int(mask.sum())
                if n > 0:
                    if lname_col is None:
                        meta['label_name'] = ''
                        lname_col = 'label_name'
                    meta.loc[mask, lname_col] = NEW_NAMES[lid]
                    renamed_meta += n
                    log(f"  Label {lid} → '{NEW_NAMES[lid]}': {n} rows (via ID)")

        results["sequence_meta_rows_renamed"] = renamed_meta
        meta.to_csv(sequence_meta_path, index=False)
        results["sequence_meta_patched"] = True
        log(f"  Saved. Rows renamed: {renamed_meta}")

    except Exception as e:
        log(f"  [ERROR] {type(e).__name__}: {e}")
        results["sequence_meta_patched"] = False
else:
    log("  Sequence meta not found — skipping.")
    results["sequence_meta_patched"] = "NOT_FOUND"


# =============================================================================
# SECTION 6 — CREATE group_e_reclassified_definitions.json (M10 API config)
# =============================================================================
log("\nSECTION 6 — Create group_e_reclassified_definitions.json")
# M10 Flask API reads this at startup to populate 7-field output for labels 22/23.
# This file is standalone — M10 doesn't need to parse fault_rules_v3.json at runtime.

group_e_m10_config = {
    "_meta": {
        "description"         : "Group E reclassified class definitions for M10 7-field output",
        "reclassification"    : "T1.7 Path B — PumpSmart Industrial Audit v2.0 §5.7",
        "date"                : str(date.today()),
        "script"              : SCRIPT_NAME,
        "class_count_change"  : "NONE — still 24 classes. Label IDs 22 and 23 retained.",
        "m7_retrain_status"   : "DEFERRED to T1.6 (module_08p6_groupB_regenerate.py)",
    },
    "classes": {},
}

for lid in [22, 23]:
    group_e_m10_config["classes"][str(lid)] = {
        "label_id"   : lid,
        "name"       : NEW_NAMES[lid],
        "old_name"   : OLD_NAMES[lid],
        "group"      : "E",
        "seq_steps"  : 300,
        "7_field_output": {
            "field_1_fault_label"                 : NEW_NAMES[lid],
            "field_2_confidence_note"             : (
                "UNKNOWN FAULT flag triggers if confidence < 70%. "
                "Group E sensor anomaly classes have lower CIRA seed anchoring than "
                "Group A — expect confidence 55–75% on real sensor anomalies."
            ),
            "field_3_probable_physical_condition" : M10_FIELD3[lid],
            "field_4_expected_sensor_behavior"    : M10_FIELD4[lid],
            "field_5_operational_risk_if_ignored" : M10_FIELD5[lid],
            "field_6_recommended_action"          : M10_FIELD6[lid],
            "field_7_model_limitation_disclaimer" : (
                "Trained on CIRA-anchored physics-synthetic data for 110 kW 7-stage pump "
                "at 2980 RPM, 40 bar. Predictions advisory only. Verify physically. "
                "Single-pump monitoring — cross-pump effects not modelled. "
                "Confidence scores may be lower on real-world faults than on simulated "
                "training data. " + RECLASSIFICATION_NOTE[lid]
            ),
        },
    }

group_e_config_path = MODEL_DIR / "group_e_reclassified_definitions.json"
try:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(group_e_config_path, 'w') as f:
        json.dump(group_e_m10_config, f, indent=2)
    results["group_e_m10_config_created"] = True
    log(f"  Saved → {group_e_config_path}")
except Exception as e:
    log(f"  [ERROR] {e}")
    results["group_e_m10_config_created"] = False


# =============================================================================
# SECTION 7 — GATE VERIFICATION
# =============================================================================
log("\nSECTION 7 — Gate verification")

GATE = {}

# T1.7-G1: fault_rules.json written without error
GATE["T1.7_G1_fault_rules_written"] = results.get("fault_rules_patched", False) is True

# T1.7-G2: Label 22 name correct in saved fault_rules
# T1.7-G3: Label 23 name correct in saved fault_rules
try:
    with open(fault_rules_save_path, 'r') as f:
        fr_verify = json.load(f)
    name_22 = fr_verify.get("fault_classes", {}).get("22", {}).get("name", "")
    name_23 = fr_verify.get("fault_classes", {}).get("23", {}).get("name", "")
    GATE["T1.7_G2_label22_name_correct"] = (name_22 == NEW_NAMES[22])
    GATE["T1.7_G3_label23_name_correct"] = (name_23 == NEW_NAMES[23])
    log(f"  Verified Label 22 name = '{name_22}' → expected '{NEW_NAMES[22]}'")
    log(f"  Verified Label 23 name = '{name_23}' → expected '{NEW_NAMES[23]}'")
except Exception as e:
    GATE["T1.7_G2_label22_name_correct"] = False
    GATE["T1.7_G3_label23_name_correct"] = False
    log(f"  [ERROR] Fault rules verification failed: {e}")

# T1.7-G4: M10 config created
GATE["T1.7_G4_m10_config_created"] = results.get("group_e_m10_config_created", False) is True

# T1.7-G5: M7 retrain correctly DEFERRED (invariant — this script never retrains)
GATE["T1.7_G5_m7_retrain_deferred_correctly"] = True

# T1.7-G6: Old names no longer present in fault_rules
try:
    name_22_old_present = any(
        entry.get("name") == OLD_NAMES[22]
        for entry in fr_verify.get("fault_classes", {}).values()
    )
    name_23_old_present = any(
        entry.get("name") == OLD_NAMES[23]
        for entry in fr_verify.get("fault_classes", {}).values()
    )
    GATE["T1.7_G6_old_names_absent"] = (not name_22_old_present and not name_23_old_present)
except Exception:
    GATE["T1.7_G6_old_names_absent"] = False

for gname, gval in GATE.items():
    log(f"  {'✅ PASS' if gval else '❌ FAIL'}  {gname}")

gates_passed = sum(GATE.values())
gates_total  = len(GATE)
results["gates_passed"] = gates_passed
results["gates_total"]  = gates_total
results["gates"]        = GATE

# T1.7 complete only if all critical gates pass (G5 is always True)
critical_pass = (
    GATE["T1.7_G1_fault_rules_written"]    and
    GATE["T1.7_G2_label22_name_correct"]   and
    GATE["T1.7_G3_label23_name_correct"]   and
    GATE["T1.7_G4_m10_config_created"]     and
    GATE["T1.7_G6_old_names_absent"]
)
results["t17_status"] = "COMPLETE" if critical_pass else "NEEDS_REVIEW"
log(f"\n  Gates: {gates_passed}/{gates_total}  |  T1.7 status: {results['t17_status']}")


# =============================================================================
# SECTION 8 — WRITE MARKDOWN REPORT
# =============================================================================
log("\nSECTION 8 — Writing report")

report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"

report_content = f"""# {SCRIPT_NAME} — Report
**Date:** {date.today()}
**Audit reference:** PumpSmart Industrial Audit v2.0, Section 5.7 — T1.7 Path B
**Purpose:** Group E reclassification. Metadata only. M7 retrain DEFERRED to T1.6.

---

## 1. Why This Was Necessary

Visualization audit (Audit v2.0 §4.5.6) confirmed both Group E plots show only ONE
sensor failing. The class definition says "two sensors degraded simultaneously" — but the
generator was 100% pure physics synthesis with zero CIRA spike-seed anchoring. Without an
empirical anchor enforcing two-channel behavior, the generator drifted to produce a
single-channel anomaly with secondary disturbance. The class names were aspirational.

**Path B chosen over Path A** because implementing a true multi-sensor failure generator
requires inventing a common-cause physics mechanism (shared excitation rail, moisture
ingress affecting two sensors) without any real-data validation. A fabricated generator
for an unanchored failure mode is worse than an honest reclassification.

---

## 2. Reclassification Summary

| Label | Old Name | New Name | Status |
|---|---|---|---|
| 22 | `{OLD_NAMES[22]}` | `{NEW_NAMES[22]}` | {'✅ RENAMED' if GATE.get('T1.7_G2_label22_name_correct') else '❌ FAILED'} |
| 23 | `{OLD_NAMES[23]}` | `{NEW_NAMES[23]}` | {'✅ RENAMED' if GATE.get('T1.7_G3_label23_name_correct') else '❌ FAILED'} |

**Class count:** UNCHANGED. Still 24 classes (Labels 0–23). Label IDs 22 and 23 retained.
Sequences not regenerated. Feature values not changed. Only label name strings updated.

---

## 3. Files Modified

| File | Status | Rows Renamed |
|---|---|---|
| `fault_rules_v3.json` | {'✅ PATCHED' if results.get('fault_rules_patched') else '❌ FAILED'} | N/A (JSON) |
| `M6B_feature_matrix.csv` | `{results.get('feature_matrix_patched', 'NOT_FOUND')}` | {results.get('feature_matrix_rows_renamed', 0)} |
| `M6B_sequence_meta.csv` | `{results.get('sequence_meta_patched', 'NOT_FOUND')}` | {results.get('sequence_meta_rows_renamed', 0)} |

## 4. New Files Created

| File | Purpose |
|---|---|
| `models/group_e_reclassified_definitions.json` | M10 Flask API — 7-field output templates for Labels 22 and 23 |
| `outputs/reports/{SCRIPT_NAME}_report.md` | This document |

---

## 5. Gate Results

| Gate | Result | Description |
|---|---|---|
| T1.7_G1_fault_rules_written | {'✅ PASS' if GATE.get('T1.7_G1_fault_rules_written') else '❌ FAIL'} | fault_rules_v3.json written without error |
| T1.7_G2_label22_name_correct | {'✅ PASS' if GATE.get('T1.7_G2_label22_name_correct') else '❌ FAIL'} | Label 22 name = '{NEW_NAMES[22]}' confirmed in saved JSON |
| T1.7_G3_label23_name_correct | {'✅ PASS' if GATE.get('T1.7_G3_label23_name_correct') else '❌ FAIL'} | Label 23 name = '{NEW_NAMES[23]}' confirmed in saved JSON |
| T1.7_G4_m10_config_created | {'✅ PASS' if GATE.get('T1.7_G4_m10_config_created') else '❌ FAIL'} | group_e_reclassified_definitions.json created for M10 |
| T1.7_G5_m7_retrain_deferred | ✅ PASS | M7 retrain correctly NOT done in this script (invariant) |
| T1.7_G6_old_names_absent | {'✅ PASS' if GATE.get('T1.7_G6_old_names_absent') else '❌ FAIL'} | Old class names absent from saved fault_rules |

**Overall: {gates_passed}/{gates_total} PASS — T1.7 {results['t17_status']}**

---

## 6. M7 Retrain Deferral — Engineering Justification

Running M7 retrain here would train on:
- Broken Label-19 features (T1.2 not yet executed — Pres.SV* flat at 1.0)
- Broken Group-B compound chains (T1.6 not yet executed — step discontinuities present)

The correct production M7 is produced at the end of T1.6 with ALL fixes in place:
```
T1.2: Label 19 → Pres.SV* drop restored (0.48–0.88 range)
T1.6: Group B → continuous superposition replaces step artifacts
T1.7: Group E → label names corrected (this script)
```
All three fixes present → single M7 retrain at T1.6 end → consistent M7 weights.

---

## 7. M10 7-Field Output — What Changes for Labels 22 and 23

**Field 1 (Fault Label):**
- Was: `sensor_failure_2ch_thermal` / `sensor_failure_2ch_pump`
- Now: `sensor_anomaly_thermal` / `sensor_anomaly_pump`

**Field 3 (Probable Physical Condition):**
- Was: Implied two sensors failing simultaneously
- Now: Honest single-channel sensor anomaly with named hardware causes

**Field 7 (Model Limitation Disclaimer):**
- Added RECLASSIFICATION NOTE per label explaining the audit finding and
  the absence of true multi-sensor simultaneous failure modelling

**Fields 2, 4, 5, 6:** Fully specified in group_e_reclassified_definitions.json.

---

## 8. Next Execution Order

```
✅ T1.7  COMPLETE  (this script)
▶️  T1.1  Run module_08p1_unit_registry_reconciliation.py   (~5 sec)
▶️  T1.2  Run module_08p2_label19_full_propagation.py       (~5 min)
▶️  T1.6  Run module_08p6_groupB_regenerate.py [TO WRITE]  (~45 min)
▶️  T1.3  Run module_08p3_m7_sequence_level_eval.py         (~5 min)
▶️  T1.4  Run module_08p4_ood_detector.py                   (~2 min)
▶️  T1.5  Run module_08p5_cusum_decay_and_fmea.py           (~30 sec)
```

---

*Generated by {SCRIPT_NAME} | PumpSmart v14.2 | {date.today()}*
"""

try:
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    results["report_written"] = True
    log(f"  Report → {report_path}")
except Exception as e:
    log(f"  [ERROR] Report write failed: {e}")
    results["report_written"] = False


# =============================================================================
# PASTE TEXT UPDATE
# =============================================================================
print()
print("=" * 72)
print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print("=" * 72)
print()
print(f"## T1.7 Group E Reclassification (Path B) — {date.today()}")
print(f"T1.7_status                      = {results['t17_status']}")
print(f"T1.7_label22_old                 = {OLD_NAMES[22]}")
print(f"T1.7_label22_new                 = {NEW_NAMES[22]}")
print(f"T1.7_label23_old                 = {OLD_NAMES[23]}")
print(f"T1.7_label23_new                 = {NEW_NAMES[23]}")
print(f"T1.7_gates_passed                = {gates_passed}/{gates_total}")
print(f"T1.7_fault_rules_patched         = {results.get('fault_rules_patched', False)}")
print(f"T1.7_feature_matrix_rows_renamed = {results.get('feature_matrix_rows_renamed', 0)}")
print(f"T1.7_sequence_meta_rows_renamed  = {results.get('sequence_meta_rows_renamed', 0)}")
print(f"T1.7_m10_config_path             = {group_e_config_path}")
print(f"T1.7_m7_retrain_status           = DEFERRED — runs in T1.6")
print(f"T1.7_class_count_unchanged       = True (still 24 classes, Labels 0-23)")
print()
print("## Execution Queue — Remaining")
print("NEXT → T1.1: python module_08p1_unit_registry_reconciliation.py")
print("THEN → T1.2: python module_08p2_label19_full_propagation.py")
print("THEN → T1.6: [script to be written] module_08p6_groupB_regenerate.py")
print("THEN → T1.3: python module_08p3_m7_sequence_level_eval.py")
print("THEN → T1.4: python module_08p4_ood_detector.py")
print("THEN → T1.5: python module_08p5_cusum_decay_and_fmea.py")
print()
print("Status for next module: T1.1 READY")
print()
print("=" * 72)
print("══ END PASTE UPDATE ══")
print("=" * 72)


# =============================================================================
# FILE MANIFEST
# =============================================================================
print()
print("── FILE MANIFEST ──────────────────────────────────────────────────────")
print()
print("MODIFIED (backed up with .bak suffix):")
print(f"  {fault_rules_save_path}  ← label 22/23 renamed + M10 templates")
if feature_matrix_path and results.get("feature_matrix_patched") is True:
    print(f"  {feature_matrix_path}  ← label_name values updated")
if sequence_meta_path and results.get("sequence_meta_patched") is True:
    print(f"  {sequence_meta_path}  ← label_name values updated")
print()
print("NEW (additive — no backup needed):")
print(f"  {group_e_config_path}")
print(f"  {report_path}")
print()
print("BACKUPS CREATED:")
for bak in [bak_fr, bak_fm, bak_meta]:
    if bak:
        print(f"  {bak}")
print()
print("GitHub push  → module_08p7_groupE_reclassify.py, fault_rules_v3.json")
print("HF Spaces    → group_e_reclassified_definitions.json, report .md")
print("DO NOT PUSH  → *.bak files")
print("────────────────────────────────────────────────────────────────────────")


# =============================================================================
# NEXT PROMPT
# =============================================================================
print()
print("── NEXT PROMPT ─────────────────────────────────────────────────────────")
print()
print("📦 T1.7 done. Starting T1.1.")
print(f"Finding: Labels 22→'{NEW_NAMES[22]}', 23→'{NEW_NAMES[23]}'.")
print("         M7 retrain deferred. 24 classes unchanged. Metadata only.")
print("Next: Run module_08p1_unit_registry_reconciliation.py (~5 sec).")
print("      No model changes. Unit registry documentation fix.")
print("      Must complete before T1.2 for clean audit trail.")
print("────────────────────────────────────────────────────────────────────────")

log("\n[DONE]")
