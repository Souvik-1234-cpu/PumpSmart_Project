# M8 Patch 1 — Unit Registry Reconciliation Report
**Date:** 2026-05-09
**Script:** module_08p1_unit_registry_reconciliation.py
**Status:** COMPLETE

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
| X_ACR_Mot.PV | **m/s** (CIRA) | outputs_unit_registry=mm/s · outputs_m2_bounds_units=mm · models_unit_registry_m5=mm/s_RMS |
| X_ACR_Mot.SV | **m/s²** (CIRA) | outputs_unit_registry=m/s2 · outputs_m2_bounds_units=mm/s · models_unit_registry_m5=mm/s_RMS |
| X_ACR_Pmp.PV | **m/s** (CIRA) | outputs_unit_registry=mm/s · outputs_m2_bounds_units=mm · models_unit_registry_m5=mm/s_RMS |
| X_ACR_Pmp.SV | **m/s²** (CIRA) | outputs_unit_registry=m/s2 · outputs_m2_bounds_units=mm/s · models_unit_registry_m5=mm/s_RMS |

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
M8p1_channels_sha256             : fae26e769f07ea815f2d2db1f4041c454327a567dce8f38dcb17b7a07af685ab
M8p1_legacy_files_backed_up      : 3
M8p1_canonical_size_kb           : 7.45
M8p1_models_retrained            : False (unit fix is dimensionless-invariant)
M8p1_M11_scada_validator_required: True (see M8p1_downstream_patch_guide.md)
M8p1_C05_invariant_preserved     : True (per-channel ISO interpretation note)
Status_for_M8p2                  : READY
══ END PASTE UPDATE ══
