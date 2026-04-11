# M5 Physics Engine Report
**Date:** 2026-03-29

## Nameplate Verification
- **nameplate_P_hyd_kW**: 55.18
- **nameplate_eta_overall**: 0.5016
- **nameplate_NS_SI**: 3.41
- **nameplate_BPF_Hz**: 347.67
- **nameplate_joukowsky_bar**: 19.1
- **nameplate_tau_thermal_s**: 388.9
- **nameplate_omega_rad_s**: 312.0649
- **s5_all_pass**: True
- **s5_cases_tested**: 19
- **eq_all_pass**: True
- **eq_pass_count**: 20
- **eq_total**: 20
- **s7_sequences_total**: 38
- **s7_fault_pass_count**: 24
- **s7_fault_total**: 26
- **s7_all_pass**: False
- **fault_rules_saved**: True
- **unit_registry_saved**: True
- **m5_config_saved**: True
- **plot1_saved**: True
- **plot2_saved**: True
- **physics_engine_exported**: True
- **fault_rules_path**: C:\Users\user\Desktop\PumpSmart_Project\models\fault_rules.json
- **physics_engine_path**: C:\Users\user\Desktop\PumpSmart_Project\src\physics_engine.py

## Equation Check Results (EQ1-EQ20)
- EQ1_P_hyd_kW: ✅ PASS
- EQ2_affinity_H_ratio: ✅ PASS
- EQ3_Ns_SI: ✅ PASS
- EQ4_BEP_excess_kW_10pct: ✅ PASS
- EQ5_tau_thermal_s: ✅ PASS
- EQ6_joukowsky_bar: ✅ PASS
- EQ7_NPSHa_startup_m: ✅ PASS
- EQ8_thoma_sigma: ✅ PASS
- EQ9_BPF_Hz: ✅ PASS
- EQ10_RP_peak_GPa: ✅ PASS
- EQ11_flash_dT_C: ✅ PASS
- EQ12_seal_Q_leak_m3s: ✅ PASS
- EQ13_bernoulli_v_ms: ✅ PASS
- EQ14_NS_dissipation_W: ✅ PASS
- EQ15_thermal_coupling_enforce: ✅ PASS
- EQ16_bearing_Q0_W: ✅ PASS
- EQ17_ISO10816_zones: ✅ PASS
- EQ18_BEP_excess_25pct_kW: ✅ PASS
- EQ19_eta_continuity: ✅ PASS
- EQ20_L10_hours: ✅ PASS

## Section 5 Validation
- Cases tested: 19
- All pass: True

## Section 7 Full Generation
- Total sequences: 38
- Fault sequences pass: 24/26
- All pass: False

## Fail List
- bearing_wear_steady_state_s100
- overloading_steady_state_s140
