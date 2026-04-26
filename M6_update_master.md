# PumpSmart M6 Update Master — 2026-04-26
# Source of truth for Cowork automation task
# Scope: M6A v5 + M6B Step0 v2 + Step0b v2 + m6b_physics_lib.py

## WHAT CHANGED (summary for Cowork context)
- New file: src/m6b_physics_lib.py (unified physics library, single source of truth)
- M6A updated to v5 (was v4): 8400 seqs, physics fixes F1-F6
- M6B Step0 updated to v2: 4500 seqs, Labels 1/4/5 corrected
- M6B Step0b updated to v2: 6200 seqs, Labels 0/2/3/6 corrected
- Channel order bug fixed: M6A used wrong order, now unified under M6B LOCKED order
- Physics fixes: F1 Temp.SV coupling, F2 abs(sin) AM, F3 M5-faithful cavitation,
  F4 Pres.SV Q-H shift, F5 dropout subtype, F6 unified library

---

## FILE 1: completed_modules_M5_to_M6p5r.md
### LOCATION: GitHub repo root or docs/

### SECTION TO UPDATE: M6A status block
FIND:
  M6A status: COMPLETE (v4)
  Script: module_06a_synthetic_generator_v4.py

REPLACE WITH:
  M6A status: COMPLETE (v5) — LOCKED
  Script: module_06a_synthetic_generator_v5.py
  Physics lib: src/m6b_physics_lib.py (unified, F1-F6 fixes)
  Total sequences: 8,400 | shape (8400, 200, 8)
  Severity early%: 51.2% (Weibull k=0.8)
  Channel order: M6B LOCKED (Mot.SV=0, Pmp.SV=1, Mot.TV=2, Pmp.PV=3,
                              Temp.SV=4, Pres.SV=5, Pmp.TV=6, Mot.PV=7)
  Physics fixes applied:
    F1: bearing_wear — Temp.SV* coupled via _tcoup r=0.9793
    F2: impeller_imbalance — abs(sin) AM envelope
    F3: cavitation — M5-faithful severity-dep t_onset, mean_drop=0.6*sev
    F4: overloading — Pres.SV* affinity law Q-H shift
    F5: sensor_failure — dropout subtype added
    F6: all generation via m6b_physics_lib.py

### SECTION TO UPDATE: M6B Step 0 status block
FIND:
  M6B Step 0 status: COMPLETE
  Script: module_06B_step0_groupA_rerun.py

REPLACE WITH:
  M6B Step 0 status: COMPLETE (v2) — LOCKED
  Script: module_06B_step0_groupA_rerun_v2.py
  Labels: 1 (bearing_wear 250s), 4 (seal_failure 400s), 5 (overloading 300s)
  Total sequences: 4,500 | all 21 gates pass
  Fixes: F1 (Temp.SV* coupling), F4 (Pres.SV* Q-H shift)
  z_t export: 4500 entries, 0 shape errors

### SECTION TO UPDATE: M6B Step 0b status block
FIND:
  M6B Step 0b status: COMPLETE
  Script: module_06B_step0b_groupA_carried.py

REPLACE WITH:
  M6B Step 0b status: COMPLETE (v2) — LOCKED
  Script: module_06B_step0b_groupA_carried_v2.py
  Labels: 0 (normal 200s), 2 (imbalance 200s), 3 (cavitation 150s), 6 (sensor 150s)
  Total sequences: 6,200 | all 20 gates pass
  Fixes: F2 (abs_sin), F3 (M5-faithful cav), F5 (dropout)
  Cavitation dual-sig: Pres.SV* shift=-0.2304 | Pmp.SV* shift=+0.2003
  Label 6 subtypes: flatline/spike/drift/dropout 300 each
  z_t export: 2000 normal + 4200 faults, 0 shape errors

---

## FILE 2: modules_M6_synthetic_generation.md
### LOCATION: GitHub repo root or docs/

### SECTION TO UPDATE: M6A Group A Label Table
FIND the table row:
  | 0 | normal | ... | ✅ VALID |
  | 2 | impeller_imbalance | ... | ✅ VALID |
  | 3 | cavitation | ... | ✅ VALID |
  | 6 | sensor_failure | ... | ✅ VALID |

REPLACE STATUS COLUMN WITH:
  | 0 | normal | 200 | 2000 | Real CIRA | ✅ REGENERATED v2 (M6B Step0b) |
  | 1 | bearing_wear | 250 | 1500 | Physics lib | ✅ REGENERATED v2 (M6B Step0 — F1 fix) |
  | 2 | impeller_imbalance | 200 | 1500 | Physics lib | ✅ REGENERATED v2 (M6B Step0b — F2 fix) |
  | 3 | cavitation | 150 | 1500 | Physics lib | ✅ REGENERATED v2 (M6B Step0b — F3 fix) |
  | 4 | seal_failure | 400 | 1500 | Physics lib | ✅ REGENERATED v2 (M6B Step0 — orifice model) |
  | 5 | overloading | 300 | 1500 | Physics lib | ✅ REGENERATED v2 (M6B Step0 — F4 fix) |
  | 6 | sensor_failure | 150 | 1200 | Physics lib | ✅ REGENERATED v2 (M6B Step0b — F5 dropout) |

### SECTION TO ADD: after M6A label table, add new section
ADD:
  ## Physics Library — m6b_physics_lib.py (v1.0 LOCKED 2026-04-26)
  Single source of truth for all M6 fault generation.
  Location: src/m6b_physics_lib.py
  Used by: M6A v5, M6B Step0 v2, M6B Step0b v2, all future M6B Steps 1-3, M12

  | Fix | Fault | Description | M5 Ref |
  |-----|-------|-------------|--------|
  | F1 | bearing_wear | Temp.SV* coupled via _tcoup r=0.9793 | M2 r=0.9793 locked |
  | F2 | impeller_imbalance | abs(sin) AM envelope | ISO 1940 non-negative amplitude |
  | F3 | cavitation | M5-faithful: sev-dep t_onset, mean_drop=0.6*sev, noise=0.3*sev | M5 fault_cavitation() |
  | F4 | overloading | Pres.SV* = (Q/Q_BEP)^2*(1-sev*0.1) affinity law | M5 fault_overloading() |
  | F5 | sensor_failure | dropout subtype added (channel→0.0, cable cut) | M5 SENSOR_SUBTYPES |
  | F6 | all | Unified library replaces all inline generation | architecture |

  SCADA noise std (M5-calibrated, locked):
    Mot.SV: 0.035 | Pmp.SV: 0.040 | Mot.TV: 0.008 | Pmp.PV: 0.012
    Temp.SV: 0.010 | Pres.SV: 0.015 | Pmp.TV: 0.008 | Mot.PV: 0.012

### SECTION TO UPDATE: Channel order note
FIND any reference to:
  M6A channel order: Mot.PV=0, Mot.SV=1...

REPLACE WITH:
  M6A channel order (WRONG — historical bug, fixed in v5): Mot.PV=0, Mot.SV=1...
  M6B channel order (LOCKED — all v2 scripts): Mot.SV=0, Pmp.SV=1, Mot.TV=2,
    Pmp.PV=3, Temp.SV=4, Pres.SV=5, Pmp.TV=6, Mot.PV=7
  NOTE: M6A v5 now uses M6B channel order via m6b_physics_lib.py

---

## FILE 3: PASTE TEXT (config.py or paste_text tracker)
### Add/replace these keys:

M6A_v5_status                  : COMPLETE_LOCKED
M6A_v5_total_sequences         : 8400
M6A_v5_array_shape             : (8400, 200, 8)
M6A_v5_sev_early_pct           : 51.2
M6A_v5_physics_lib             : m6b_physics_lib.py
M6A_v5_fixes                   : F1,F2,F3,F4,F5,F6
M6B_step0_v2_status            : COMPLETE_LOCKED
M6B_step0_v2_total_sequences   : 4500
M6B_step0_v2_gates_all_pass    : True
M6B_step0_v2_zt_shape_errors   : 0
M6B_step0b_v2_status           : COMPLETE_LOCKED
M6B_step0b_v2_total_sequences  : 6200
M6B_step0b_v2_gates_all_pass   : True
M6B_step0b_v2_cav_pres_shift   : -0.2304
M6B_step0b_v2_cav_pmpSV_shift  : 0.2003
M6B_step0b_v2_label6_subtypes  : flatline/spike/drift/dropout 300each
M6B_step0b_v2_zt_shape_errors  : 0
m6b_physics_lib_status         : LOCKED_v1.0
m6b_physics_lib_fixes          : F1,F2,F3,F4,F5,F6
channel_order_M6B_LOCKED       : Mot.SV=0,Pmp.SV=1,Mot.TV=2,Pmp.PV=3,Temp.SV=4,Pres.SV=5,Pmp.TV=6,Mot.PV=7
Active_module                  : M6B_Step1_GroupB_compound_chains
Status_for_M6B_Step1           : READY

---

## COWORK TASK INSTRUCTIONS
Task type: File update automation
Files to update: 3 files listed above
Source: This master file (M6_update_master.md)
Method: FIND + REPLACE for existing sections, ADD for new sections
Do NOT delete any existing content not referenced above
After each file update: commit to GitHub with message format:
  "docs: update [filename] — M6A v5 + M6B Step0/0b v2 physics fixes"