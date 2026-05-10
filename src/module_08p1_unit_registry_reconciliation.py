# =============================================================================
# module_08p1_unit_registry_reconciliation.py
# PumpSmart v14.2 — M8 Patch 1 of 5: Unit Registry Reconciliation
# =============================================================================
# WHY THIS SCRIPT EXISTS:
#   Three local unit files disagree with each other AND with the CIRA Zenodo
#   sensor documentation. For a 40-lakh asset, unit ambiguity is a deployment
#   killer — a SCADA system streaming raw m/s² into a model that assumes mm/s
#   produces values 1000× off scale and a silent catastrophic miscalibration.
#
# WHAT THIS SCRIPT DOES:
#   1. Reads all three legacy unit files (if present).
#   2. Writes ONE canonical unit registry that:
#       a. Records the CIRA-authoritative physical raw unit
#       b. Records the SCADA encoding (broadband_peak — your C-05 finding)
#       c. Carries an explicit ISO interpretation note per channel
#       d. Adds an SHA256 checksum to detect future tampering
#   3. Does NOT modify the trained models or normalisation config.
#      The model operates in normalised space (value / cluster_mean) which is
#      unit-invariant. The unit fix is DOCUMENTATION, not data.
#
# WHAT THIS SCRIPT DOES NOT DO:
#   - It does not retrain M4, M7, or M8.
#   - It does not change M3_normalization_config.json.
#   - It does not change any ISO threshold (no .SV channel was using ISO
#     thresholds anyway — that's the C-05 invariant).
#
# OUTPUT FILES (safe — additive only):
#   models/unit_registry_canonical.json      ← single source of truth
#   outputs/reports/M8p1_unit_reconciliation_report.md
#   models/unit_registry_legacy_<name>.bak.json   ← legacy files backed up
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings, hashlib, shutil
warnings.filterwarnings('ignore')

SCRIPT_NAME = "module_08p1_unit_registry_reconciliation"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}")
log("=" * 72)

# =============================================================================
# SECTION 1 — IDENTIFY LEGACY UNIT FILES
# =============================================================================
log("\nSECTION 1 — Identify legacy unit files")

LEGACY_FILES = {
    "outputs_unit_registry":       OUTPUT_DIR / "unit_registry.json",
    "outputs_m2_bounds_units":     OUTPUT_DIR / "M2_cluster_bounds_units.json",
    "models_unit_registry_m5":     MODEL_DIR  / "unit_registry.json",
}

legacy_present = {}
for nick, path in LEGACY_FILES.items():
    exists = path.exists()
    legacy_present[nick] = exists
    log(f"  {'✓' if exists else '✗'}  {nick:<35s} → {path}")

results["legacy_files_found"] = sum(legacy_present.values())

# =============================================================================
# SECTION 2 — BUILD CANONICAL UNIT REGISTRY
# =============================================================================
# Source-of-truth ordering:
#   1. CIRA Zenodo dataset description (authoritative for raw unit)
#   2. C-05 challenge resolution     (authoritative for SCADA encoding)
#   3. ISO 10816-3 / IEC 60034 / ISO 5167 (authoritative for interpretation
#      caveat on .SV channels — DO NOT apply RMS thresholds)
# =============================================================================
log("\nSECTION 2 — Build canonical registry")

CANONICAL = {
    "_meta": {
        "schema_version":     "2.0",
        "supersedes":         list(LEGACY_FILES.keys()),
        "created_by":         SCRIPT_NAME,
        "created":            str(date.today()),
        "authoritative":      True,
        "source_of_truth":    "CIRA SACIP Zenodo dataset description",
        "asset_nameplate": {
            "power_kW":     110,
            "flow_m3h":     45,
            "head_m":       450,
            "pressure_bar": 40,
            "impellers":    7,
            "speed_rpm":    2980,
        },
        "downstream_consumers": [
            "M3_normalization_config.json (raw unit reference only — normalised values are unit-invariant)",
            "M5_physics_engine.py (nameplate equation unit checking)",
            "M10 Flask API physics_context (per-class display units)",
            "M11 SCADA ingestion adapter (input unit validation)",
        ],
        "downstream_invariant_C05": (
            "Mot.SV and Pmp.SV are SCADA broadband peak ACCELERATION envelopes "
            "(per CIRA documentation), NOT ISO 10816-3 velocity RMS values. "
            "Apply ISO 10816-3 Zone A/B/C/D thresholds ONLY to derived velocity "
            "RMS computed from PV displacement channels at known frequency. "
            "All .SV fault detection in PumpSmart uses RELATIVE deviation from "
            "cluster mean (SV*) — never absolute ISO comparison."
        ),
    },
    "channels": {
        # ─── Motor accelerometer ─────────────────────────────────────────────
        "X_ACR_Mot.PV": {
            "description":               "Motor bearing — vibrational velocity (peak)",
            "raw_unit_per_CIRA":         "m/s",
            "raw_unit_symbol":           "m/s",
            "sensor_type":               "velocity",
            "sensor_location":           "motor_bearing",
            "scada_encoding":            "broadband_peak",
            "iso_reference":             "ISO 10816-3 (reference only — not directly comparable)",
            "iso_interpretation_note":   "Peak velocity, not RMS. ISO 10816 thresholds are RMS-based — do not compare directly.",
            "expected_normal_range":     [0.0005, 0.004],
            "expected_normal_unit":      "m/s",
        },
        "X_ACR_Mot.SV": {
            "description":               "Motor bearing — peak acceleration (broadband envelope)",
            "raw_unit_per_CIRA":         "m/s^2",
            "raw_unit_symbol":           "m/s²",
            "sensor_type":               "accelerometer",
            "sensor_location":           "motor_bearing",
            "scada_encoding":            "broadband_peak_envelope",
            "iso_reference":             "ISO 10816-3 (NOT applicable — see iso_interpretation_note)",
            "iso_interpretation_note":   "BROADBAND ENVELOPE, NOT SINGLE-TONE ACCELERATION. Do NOT apply ISO 10816-3 Zone A/B/C/D directly. Use cluster-relative SV* via M3 normalisation. C-05 invariant.",
            "expected_normal_range":     [0.39, 21.5],
            "expected_normal_unit":      "m/s^2",
            "channel_weight_M4":         2.0,
            "channel_weight_rationale":  "Primary mechanical fault indicator (bearing impact / cavitation shock signature)",
        },
        "X_ACR_Mot.TV": {
            "description":               "Motor bearing — contact temperature",
            "raw_unit_per_CIRA":         "degC",
            "raw_unit_symbol":           "°C",
            "sensor_type":               "thermocouple",
            "sensor_location":           "motor_bearing",
            "scada_encoding":            "instantaneous",
            "iso_reference":             "IEC 60034-1",
            "iso_interpretation_note":   "Corroborating sensor only. Subject to thermal time-constant lag (400–600s) per C-04.",
            "expected_normal_range":     [18.0, 55.0],
            "expected_normal_unit":      "°C",
        },
        # ─── Pump accelerometer ──────────────────────────────────────────────
        "X_ACR_Pmp.PV": {
            "description":               "Pump bearing — vibrational velocity (peak)",
            "raw_unit_per_CIRA":         "m/s",
            "raw_unit_symbol":           "m/s",
            "sensor_type":               "velocity",
            "sensor_location":           "pump_bearing",
            "scada_encoding":            "broadband_peak",
            "iso_reference":             "ISO 10816-3 (reference only)",
            "iso_interpretation_note":   "Peak velocity, not RMS.",
            "expected_normal_range":     [0.0003, 0.004],
            "expected_normal_unit":      "m/s",
        },
        "X_ACR_Pmp.SV": {
            "description":               "Pump bearing — peak acceleration (broadband envelope)",
            "raw_unit_per_CIRA":         "m/s^2",
            "raw_unit_symbol":           "m/s²",
            "sensor_type":               "accelerometer",
            "sensor_location":           "pump_bearing",
            "scada_encoding":            "broadband_peak_envelope",
            "iso_reference":             "ISO 10816-3 (NOT applicable)",
            "iso_interpretation_note":   "BROADBAND ENVELOPE, NOT SINGLE-TONE. C-05 invariant. Coupled to BPF=347.67 Hz aliased at 1 Hz sampling.",
            "expected_normal_range":     [0.38, 55.3],
            "expected_normal_unit":      "m/s^2",
            "channel_weight_M4":         2.0,
        },
        "X_ACR_Pmp.TV": {
            "description":               "Pump bearing — contact temperature",
            "raw_unit_per_CIRA":         "degC",
            "raw_unit_symbol":           "°C",
            "sensor_type":               "thermocouple",
            "sensor_location":           "pump_bearing",
            "scada_encoding":            "instantaneous",
            "iso_reference":             "IEC 60034-1",
            "expected_normal_range":     [18.0, 45.0],
            "expected_normal_unit":      "°C",
        },
        # ─── Process channels ────────────────────────────────────────────────
        "X_Temp.SV": {
            "description":               "Motor casing / process fluid temperature",
            "raw_unit_per_CIRA":         "degC",
            "raw_unit_symbol":           "°C",
            "sensor_type":               "PT100",
            "sensor_location":           "motor_casing",
            "scada_encoding":            "instantaneous",
            "iso_reference":             "ISO 13373-2",
            "iso_interpretation_note":   "Overloading detection uses dT*/dt rate, NOT absolute T threshold (per C-04). Cluster-relative ΔT* per C-09.",
            "expected_normal_range":     [18.0, 55.0],
            "expected_normal_unit":      "°C",
        },
        "X_Pres.SV": {
            "description":               "Pump outlet fluid pressure (discharge)",
            "raw_unit_per_CIRA":         "bar",
            "raw_unit_symbol":           "bar",
            "sensor_type":               "pressure_transducer",
            "sensor_location":           "pump_discharge",
            "scada_encoding":            "instantaneous",
            "iso_reference":             "ISO 5167",
            "iso_interpretation_note":   "Cluster-conditional winsorisation per C-18. Joukowsky transient ceiling 19.1 bar above operating point.",
            "expected_normal_range":     [0.4, 46.0],
            "expected_normal_unit":      "bar",
        },
        # ─── Environmental channels (NOT used in ML) ─────────────────────────
        "Barometer": {
            "description":               "Ambient atmospheric pressure (environmental — NOT used in ML)",
            "raw_unit_per_CIRA":         "mbar",
            "raw_unit_symbol":           "mbar",
            "sensor_type":               "environmental",
            "scada_encoding":            "instantaneous",
            "ml_usage":                  "EXCLUDED — only used for cleaning audit per C-01",
            "expected_normal_range":     [980.0, 1025.0],
            "expected_normal_unit":      "mbar",
        },
        "Temperature": {
            "description":               "Ambient air temperature (environmental — NOT used in ML)",
            "raw_unit_per_CIRA":         "degC",
            "raw_unit_symbol":           "°C",
            "sensor_type":               "environmental",
            "scada_encoding":            "instantaneous",
            "ml_usage":                  "EXCLUDED — climate-agnostic per C-10",
            "expected_normal_range":     [10.0, 40.0],
            "expected_normal_unit":      "°C",
        },
    },
    "scada_ingestion_validation": {
        "description": (
            "M11 SCADA adapter MUST validate incoming sensor units against this "
            "registry. If incoming unit string does not match raw_unit_per_CIRA "
            "for the channel → REJECT the sample with explicit error. "
            "Do NOT silently auto-convert."
        ),
        "validation_routine": "M11_unit_validator.py (must be implemented)",
        "fail_action":        "RAISE_AND_HALT — never silently rescale",
    },
}

# Compute checksum over the channels block (exclude _meta which contains date)
canon_str   = json.dumps(CANONICAL["channels"], sort_keys=True)
canon_hash  = hashlib.sha256(canon_str.encode("utf-8")).hexdigest()
CANONICAL["_meta"]["channels_sha256"] = canon_hash
log(f"  Channels SHA256: {canon_hash[:16]}...")

# =============================================================================
# SECTION 3 — BACKUP LEGACY FILES, WRITE CANONICAL
# =============================================================================
log("\nSECTION 3 — Backup legacy files + write canonical")

backup_log = []
for nick, path in LEGACY_FILES.items():
    if not path.exists():
        backup_log.append(f"{nick}: not present (skip)")
        continue
    bak_path = path.with_suffix(path.suffix + ".bak_pre_canonical")
    try:
        shutil.copy2(path, bak_path)
        backup_log.append(f"{nick}: backed up → {bak_path.name}")
    except Exception as e:
        backup_log.append(f"{nick}: backup FAILED — {e}")

for line in backup_log:
    log(f"  {line}")

CANON_PATH = MODEL_DIR / "unit_registry_canonical.json"
try:
    with open(CANON_PATH, "w", encoding="utf-8") as f:
        json.dump(CANONICAL, f, indent=2)
    log(f"  ✓ Wrote canonical: {CANON_PATH}")
    results["canonical_path"]  = str(CANON_PATH)
    results["canonical_size_kb"] = round(CANON_PATH.stat().st_size / 1024, 2)
except Exception as e:
    log(f"  ✗ FAILED to write canonical: {e}")
    results["canonical_path"] = None

# =============================================================================
# SECTION 4 — DOWNSTREAM-CODE PATCH GUIDE (manual, reviewed)
# =============================================================================
# The intent here is NOT to auto-edit downstream code. Auto-edits to scripts
# you depend on are dangerous. Instead, we generate a patch guide so you can
# review and apply with str_replace yourself.
log("\nSECTION 4 — Generating downstream patch guide")

patch_guide = """# Downstream code patch guide — apply manually after review

The canonical registry is now `models/unit_registry_canonical.json`.
The three legacy files have been backed up with `.bak_pre_canonical` suffix.
**Do not delete the legacy files yet** — keep them for one full sprint to confirm
no script silently reads them, then delete after M9.

## Required reads to update

### 1. `src/module_05_physics_engine.py`
Replace the inline UNIT_REGISTRY dict definition (Section 8) with a load:

```python
# OLD:
# UNIT_REGISTRY = { ... inline dict ... }
# with open(unit_reg_path, 'w') as f: json.dump(UNIT_REGISTRY, f, indent=2)

# NEW:
unit_reg_canonical = MODEL_DIR / "unit_registry_canonical.json"
if unit_reg_canonical.exists():
    with open(unit_reg_canonical) as f:
        UNIT_REGISTRY = json.load(f)
else:
    raise FileNotFoundError(
        "Canonical unit registry missing. Run module_08p1 first."
    )
```

### 2. M10 Flask `app/routes/physics.py` (when written)
Read units for display from the canonical registry only:

```python
with open(MODEL_DIR / "unit_registry_canonical.json") as f:
    REGISTRY = json.load(f)
display_unit = REGISTRY['channels'][channel_name]['raw_unit_symbol']
```

### 3. M11 SCADA ingestion adapter (when written)
This is the *critical* deployment-side check. Before any sample enters
the inference pipeline:

```python
def validate_incoming_units(scada_payload, registry):
    for channel, value_dict in scada_payload.items():
        if channel not in registry['channels']:
            continue   # ignore unknown channels
        expected = registry['channels'][channel]['raw_unit_per_CIRA']
        actual   = value_dict.get('unit', None)
        if actual is None:
            raise ValueError(
                f"SCADA payload for {channel} has no unit field. "
                f"Cannot validate against registry. HALT."
            )
        if actual != expected:
            raise ValueError(
                f"Unit mismatch on {channel}: SCADA reports {actual}, "
                f"registry expects {expected}. NEVER auto-convert. HALT."
            )
```

## NOT touched (intentional)
- `M3_normalization_config.json` — values are dimensionless ratios; unit-invariant
- M4 weights, M7 weights, M8 weights — trained on normalised inputs; unit-invariant
- All M5 nameplate equations — already in correct SI units
"""

PATCH_GUIDE_PATH = REPORT_DIR / "M8p1_downstream_patch_guide.md"
with open(PATCH_GUIDE_PATH, "w", encoding="utf-8") as f:
    f.write(patch_guide)
log(f"  ✓ Patch guide: {PATCH_GUIDE_PATH}")

# =============================================================================
# SECTION 5 — RECONCILIATION REPORT
# =============================================================================
log("\nSECTION 5 — Writing reconciliation report")

REPORT_PATH = REPORT_DIR / f"{SCRIPT_NAME}_report.md"

# Build comparison table — what each file said vs canonical
diff_rows = []
for ch in ["X_ACR_Mot.PV", "X_ACR_Mot.SV", "X_ACR_Pmp.PV", "X_ACR_Pmp.SV"]:
    canonical_unit = CANONICAL["channels"][ch]["raw_unit_symbol"]
    legacy_units = []
    for nick, path in LEGACY_FILES.items():
        if not path.exists():
            legacy_units.append(f"{nick}=N/A")
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            # Each legacy schema is different — try several keys
            unit_str = "?"
            if "channels" in data and ch in data["channels"]:
                unit_str = data["channels"][ch].get("unit", "?")
            elif "sensor_channels" in data and ch in data["sensor_channels"]:
                unit_str = data["sensor_channels"][ch].get("unit_symbol",
                          data["sensor_channels"][ch].get("unit", "?"))
            elif "sensors" in data:
                # M5 unit_registry uses Mot_SV form
                m5_key = ch.replace("X_ACR_", "").replace(".", "_")
                if m5_key in data["sensors"]:
                    unit_str = data["sensors"][m5_key].get("unit", "?")
            legacy_units.append(f"{nick}={unit_str}")
        except Exception as e:
            legacy_units.append(f"{nick}=ERROR({e})")
    diff_rows.append(f"| {ch} | **{canonical_unit}** (CIRA) | " + " · ".join(legacy_units) + " |")

diff_block = "\n".join(diff_rows)

report_md = f"""# M8 Patch 1 — Unit Registry Reconciliation Report
**Date:** {date.today()}
**Script:** {SCRIPT_NAME}.py
**Status:** {'COMPLETE' if results.get('canonical_path') else 'FAILED'}

---

## What this patch does
Resolves the contradiction between three local unit registry files and the
CIRA Zenodo dataset documentation. Writes one canonical registry as the
single source of truth, backs up legacy files, and generates a patch guide
for downstream code.

## Why the model is not retrained
PumpSmart operates entirely in normalised (dimensionless) space:
`P* = P / P_cluster_mean`, `a* = a / a_cluster_mean`, `ΔT* = (T - T_min) / (T_max - T_min)`.
These ratios are unit-invariant — the value of P/P_mean is the same whether
P is measured in bar or psi, m/s² or mm/s², as long as both are measured
the SAME way. Since M3 normalisation, M4/M7/M8 training, and all gate
thresholds operate exclusively on these dimensionless values, no model
weight or threshold needs to change.

The unit registry mismatch was a **documentation defect**, not a numerical
defect. This patch is the corresponding documentation fix.

## Channel unit comparison: CIRA vs legacy files

| Channel | Canonical (CIRA) | Legacy files said |
|---|---|---|
{diff_block}

> The CIRA values are now authoritative. Legacy files are backed up with
> `.bak_pre_canonical` suffix. Downstream code must read the canonical only.

## C-05 invariant preserved
The canonical registry carries an `iso_interpretation_note` for every
.SV channel stating that ISO 10816-3 RMS thresholds **DO NOT APPLY**
to broadband peak envelopes. This is the C-05 finding made explicit and
enforceable per channel.

## Files written
- `models/unit_registry_canonical.json` (single source of truth)
- `outputs/reports/M8p1_downstream_patch_guide.md` (manual review needed)
- Legacy backups: `<original>.bak_pre_canonical`

## Files NOT touched
- `M3_normalization_config.json` — dimensionless, unit-invariant
- `models/lstm_ae_baseline_best.pth` — trained on normalised data
- `models/M7_xgboost_classifier.json` — feature matrix is dimensionless
- `models/tcn_ae_level2_best.pth` — z_t latents are dimensionless

## Critical deployment requirement (M11)
M11 SCADA ingestion adapter **MUST** include a unit validation routine
that rejects samples whose unit string does not match the canonical
registry. NEVER auto-convert silently. See patch guide for code.

## Paste text update

══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══
M8p1_canonical_registry          : models/unit_registry_canonical.json
M8p1_channels_sha256             : {canon_hash}
M8p1_legacy_files_backed_up      : {sum(legacy_present.values())}
M8p1_canonical_size_kb           : {results.get('canonical_size_kb', 'N/A')}
M8p1_models_retrained            : False (unit fix is dimensionless-invariant)
M8p1_M11_scada_validator_required: True (see M8p1_downstream_patch_guide.md)
M8p1_C05_invariant_preserved     : True (per-channel ISO interpretation note)
Status_for_M8p2                  : READY
══ END PASTE UPDATE ══
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_md)
log(f"  ✓ Report: {REPORT_PATH}")

# =============================================================================
# SECTION 6 — FILE MANIFEST + NEXT
# =============================================================================
log("\n" + "=" * 72)
log("FILE MANIFEST")
log("=" * 72)
log(f"  GitHub push: {CANON_PATH}")
log(f"  GitHub push: {PATCH_GUIDE_PATH}")
log(f"  GitHub push: {REPORT_PATH}")
log("  Spaces upload: NONE (no model artifacts changed)")

log("\n" + "=" * 72)
log("📦 M8 Patch 1 done. Next: M8 Patch 2 — Label 19 propagation + M7 retrain.")
log("    Provide M8p2 complete script.")
log("=" * 72)
