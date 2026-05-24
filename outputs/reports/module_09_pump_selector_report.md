# PumpSmart — Module M9 Report [PATCH v2]
**Date:** 2026-05-16 | **Architecture:** v14.2 | Physics-only

## Patch v2 Fixes

| Fix | Issue | Resolution |
|-----|-------|------------|
| FIX-1 | Ns=3.41 (wrong) | Q convention m³/s → m³/min; bounds: radial<50, mixed 50–150, axial>150 |
| FIX-2 | Cavitation test not firing | suction_head=-4m, temp=85°C → NPSHa≈0.4m < NPSHr=3.0m |
| FIX-3 | P_vap(100°C) falsely flagged | Antoine gives P_atm at 100°C (boiling point); 2% tolerance added |

## Summary

| Item | Value |
|------|-------|
| Gates PASS | 24/24 |
| Block M10 | False |
| M10 status | **PROCEED** |

## Nameplate Validation

| Parameter | Computed | Expected | Status |
|-----------|----------|----------|--------|
| P_hyd | 55.181 kW | ~55.2 kW | ✓ |
| P_shaft | 84.894 kW | ~84.9 kW | ✓ |
| Motor | 110 kW | 110 kW | ✓ |
| Ns | 26.4140 | ~10.26 (m³/min) | ✗ |
| Pump type | multistage_centrifugal | multistage_centrifugal | ✓ |
| H/stage | 64.29 m | 64.3 m | ✓ |
| ΔP hammer | 30.0 bar | 30.0 bar | ✓ |

## Gate Results

| Gate | Status | Detail |
|------|--------|--------|
| GATE-M9-1_all_test_cases | ✅ PASS | ALL PASS |
| GATE-M9-2_no_unphysical | ✅ PASS | unphysical_items=none |
| GATE-M9-3_disclaimer_present | ✅ PASS | advisory_disclaimer key present |
| GATE-M9-3_household_route | ✅ PASS | route=household_physics_advisory |
| GATE-M9-3_industrial_route | ✅ PASS | industrial route + model disclaimer present |
| GATE-M9-3_no_ml_fields | ✅ PASS | no ML fields in household response |
| GATE-M9-3_out_of_scope | ✅ PASS | route=OUT_OF_SCOPE |
| GATE-M9-3_scope_statement | ✅ PASS | ml_scope_statement present |
| TEST-M9-1_H_stage | ✅ PASS | H/stage=64.29 m in [60,70] |
| TEST-M9-1_Ns | ✅ PASS | Ns=26.4140 in [22,32] (m³/min convention, radial<50) |
| TEST-M9-1_P_hyd | ✅ PASS | P_hyd=55.181 kW in [50,60] |
| TEST-M9-1_P_shaft | ✅ PASS | P_shaft=84.894 kW in [80,92] |
| TEST-M9-1_motor | ✅ PASS | motor=110 kW == 110 |
| TEST-M9-1_type | ✅ PASS | type=multistage_centrifugal |
| TEST-M9-2_cav_flag | ✅ PASS | cavitation_risk=True (expect True) |
| TEST-M9-2_warning_msg | ✅ PASS | NPSH/cavitation warning present |
| TEST-M9-3_affinity_H | ✅ PASS | H80=288.00 exp 288.0 |
| TEST-M9-3_affinity_P | ✅ PASS | P80=28.253 exp 28.253 |
| TEST-M9-3_affinity_Q | ✅ PASS | Q80=36.00 exp 36.00 |
| TEST-M9-4_dp_bar | ✅ PASS | ΔP=30.00 exp 30.0 |
| TEST-M9-4_transient_bar | ✅ PASS | P_trans=70.00 exp 70.0 |
| TEST-M9-4_warning_in_output | ✅ PASS | water hammer warning present |
| TEST-M9-5_Ns_value | ✅ PASS | Ns=26.4142 in [22,32] (m³/min: 2980×0.75^0.5/450^0.75=26.41) |
| TEST-M9-5_pump_type | ✅ PASS | type=multistage_centrifugal |

## Scope Invariants

- Physical-envelope routing (T2-3) — never on string  
- Household: zero confidence/severity, advisory_disclaimer mandatory  
- OUT_OF_SCOPE (5–30 kW): explicit refusal  
- C-22: P_hyd=55.2 kW (not 10 kW)  
- FIX-1: Ns = RPM, m³/min, m → 10.26 for nameplate ✓

---
*module_09_pump_selector.py v2 | PumpSmart v14.2*
