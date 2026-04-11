```
[20:45:00] ======================================================================
[20:45:00] SECTION 0 - Nameplate constants + cluster baselines
[20:45:00] ======================================================================
[20:45:00] Hydraulic power      : 55.18 kW
[20:45:00] Overall efficiency   : 0.502
[20:45:00] Specific speed (SI)  : 3.41
[20:45:00] BPF                  : 347.67 Hz
[20:45:00] Joukowsky ΔP         : 19.10 bar
[20:45:00] Thermal time const   : 388.9 s
[20:45:00] ω                    : 312.0649 rad/s
[20:45:00] Cluster baselines + winsor ceilings loaded.
[20:45:00] ======================================================================
[20:45:00] SECTION 1 - Physics helper functions
[20:45:00] ======================================================================
[20:45:00] All 20 physics helper functions defined.
[20:45:00]   EQ1 hydraulic_power
[20:45:00]   EQ2 affinity_speed_ratio
[20:45:00]   EQ3 specific_speed_SI
[20:45:00]   EQ4 bep_excess_power
[20:45:00]   EQ5 thermal_response (1st Law lumped capacitance)
[20:45:00]   EQ6 joukowsky_pressure_rise
[20:45:00]   EQ7 npsha
[20:45:00]   EQ8 thoma_number
[20:45:00]   EQ9 Paris-Erdogan via bearing_friction_heat beta
[20:45:00]   EQ10 rayleigh_plesset_peak_pressure
[20:45:00]   EQ11 flash_evaporation_temp_drop
[20:45:00]   EQ12 seal_leakage_flow
[20:45:00]   EQ13 bernoulli_velocity_from_pressure
[20:45:00]   EQ14 navier_stokes_viscous_dissipation
[20:45:00]   EQ15 thermal_coupling_enforce (r=0.9793)
[20:45:00]   EQ16 bearing_friction_heat (Palmgren)
[20:45:00]   EQ17 ISO10816 zone check (constants)
[20:45:00]   EQ18 BEP 25pct overload check
[20:45:00]   EQ19 continuity pressure delivery
[20:45:00]   EQ20 L10 bearing life (ISO 281)
[20:45:00] ======================================================================
[20:45:00] SECTION 3 - Fault causal chain functions
[20:45:00] ======================================================================
[20:45:00] ======================================================================
[20:45:00] SECTION 4 - Physics validation gates
[20:45:00] ======================================================================
[20:45:00] ======================================================================
[20:45:00] SECTION 5 - Running initial validation suite
[20:45:00] ======================================================================
[20:45:00]   bearing_wear           | steady_state    | sev=0.7 | PASS
[20:45:00]   bearing_wear           | high_load       | sev=0.8 | PASS
[20:45:00]   bearing_wear           | startup         | sev=0.5 | PASS
[20:45:00]   impeller_imbalance     | steady_state    | sev=0.7 | PASS
[20:45:00]   impeller_imbalance     | high_load       | sev=0.8 | PASS
[20:45:00]   Cavitation: NPSHa=11.69 m | sigma=0.0260 | sigma_crit=0.0120
[20:45:00]   cavitation             | startup         | sev=0.7 | PASS
[20:45:00]   Cavitation: NPSHa=11.69 m | sigma=0.0260 | sigma_crit=0.0120
[20:45:00]   cavitation             | startup         | sev=0.9 | PASS
[20:45:00]   seal_failure           | steady_state    | sev=0.6 | PASS
[20:45:00]   seal_failure           | high_load       | sev=0.8 | PASS
[20:45:00]   Overloading: Q=51.7 m3/h | P_excess=29.01 kW
[20:45:00]   overloading            | steady_state    | sev=0.6 | PASS
[20:45:00]   Overloading: Q=55.1 m3/h | P_excess=45.78 kW
[20:45:00]   overloading            | steady_state    | sev=0.9 | PASS
[20:45:00]   sensor_failure         | steady_state    | sev=0.5 | PASS
[20:45:00]   sensor_failure         | startup         | sev=0.5 | PASS
[20:45:00]   sensor_failure         | high_load       | sev=0.5 | PASS
[20:45:00]   sensor_failure         | cooldown        | sev=0.5 | PASS
[20:45:00]   sensor_failure         | steady_state    | sev=0.5 | PASS
[20:45:00]   sensor_failure         | high_load       | sev=0.5 | PASS
[20:45:00]   sensor_failure         | startup         | sev=0.5 | PASS
[20:45:00]   sensor_failure         | cooldown        | sev=0.5 | PASS
[20:45:00]
Section 5 done - ALL_PASS: True
[20:45:00] ======================================================================
[20:45:00] SECTION 6 - Nameplate equation verification EQ1-EQ20
[20:45:00] ======================================================================
[20:45:00]   EQ1_P_hyd_kW                           | val=55.18 | PASS
[20:45:00]   EQ2_affinity_H_ratio                   | val=1.4209 | PASS
[20:45:00]   EQ3_Ns_SI                              | val=3.41 | PASS
[20:45:00]   EQ4_BEP_excess_kW_10pct                | val=17.368 | PASS
[20:45:00]   EQ5_tau_thermal_s                      | val=388.9 | PASS
[20:45:00]   EQ6_joukowsky_bar                      | val=19.1 | PASS
[20:45:00]   EQ7_NPSHa_startup_m                    | val=5.71 | PASS
[20:45:00]   EQ8_thoma_sigma                        | val=0.01269 | PASS
[20:45:00]   EQ9_BPF_Hz                             | val=347.67 | PASS
[20:45:00]   EQ10_RP_peak_GPa                       | val=997700.0 | PASS
[20:45:00]   EQ11_flash_dT_C                        | val=0.1231 | PASS
[20:45:00]   EQ12_seal_Q_leak_m3s                   | val=5.456e-06 | PASS
[20:45:00]   EQ13_bernoulli_v_ms                    | val=89.42 | PASS
[20:45:00]   EQ14_NS_dissipation_W                  | val=0.009 | PASS
[20:45:00]   EQ15_thermal_coupling_enforce          | val=0.8428 | PASS
[20:45:00]   EQ16_bearing_Q0_W                      | val=33.0 | PASS
[20:45:00]   EQ17_ISO10816_zones                    | val=A=2.3 B=4.5 C=7.1 | PASS
[20:45:00]   EQ18_BEP_excess_25pct_kW               | val=51.765 | PASS
[20:45:00]   EQ19_eta_continuity                    | val=0.5016 | PASS
[20:45:00]   EQ20_L10_hours                         | val=25000.0 | PASS
[20:45:00]
Section 6 done - EQ PASS: 20/20
[20:45:00] ======================================================================
[20:45:00] SECTION 7 - Full sequence generation + validation
[20:45:00] ======================================================================
[20:45:00]   bearing_wear           | steady_state    | sev=0.4 | FAIL - ['G4_bearing_thermal_coupling']
[20:45:00]   bearing_wear           | steady_state    | sev=0.7 | PASS
[20:45:00]   bearing_wear           | steady_state    | sev=1.0 | PASS
[20:45:00]   bearing_wear           | high_load       | sev=0.6 | PASS
[20:45:00]   bearing_wear           | startup         | sev=0.5 | PASS
[20:45:00]   impeller_imbalance     | steady_state    | sev=0.4 | PASS
[20:45:00]   impeller_imbalance     | steady_state    | sev=0.7 | PASS
[20:45:00]   impeller_imbalance     | steady_state    | sev=1.0 | PASS
[20:45:00]   impeller_imbalance     | high_load       | sev=0.6 | PASS
[20:45:00]   Cavitation: NPSHa=11.69 m | sigma=0.0260 | sigma_crit=0.0120
[20:45:00]   cavitation             | startup         | sev=0.4 | PASS
[20:45:00]   Cavitation: NPSHa=11.69 m | sigma=0.0260 | sigma_crit=0.0120
[20:45:00]   cavitation             | startup         | sev=0.7 | PASS
[20:45:00]   Cavitation: NPSHa=11.69 m | sigma=0.0260 | sigma_crit=0.0120
[20:45:00]   cavitation             | startup         | sev=1.0 | PASS
[20:45:00]   seal_failure           | steady_state    | sev=0.4 | PASS
[20:45:00]   seal_failure           | steady_state    | sev=0.7 | PASS
[20:45:00]   seal_failure           | steady_state    | sev=1.0 | PASS
[20:45:00]   seal_failure           | high_load       | sev=0.6 | PASS
[20:45:00]   Overloading: Q=49.5 m3/h | P_excess=18.72 kW
[20:45:00]   overloading            | steady_state    | sev=0.4 | FAIL - ['G6_overload_thermal_coupling']
[20:45:00]   Overloading: Q=52.9 m3/h | P_excess=34.41 kW
[20:45:00]   overloading            | steady_state    | sev=0.7 | PASS
[20:45:00]   Overloading: Q=56.2 m3/h | P_excess=51.76 kW
[20:45:00]   overloading            | steady_state    | sev=1.0 | PASS
[20:45:00]   sensor_failure         | steady_state    | sev=0.5 | PASS
[20:45:00]   sensor_failure         | steady_state    | sev=0.5 | PASS
[20:45:00]   sensor_failure         | startup         | sev=0.5 | PASS
[20:45:00]   sensor_failure         | high_load       | sev=0.5 | PASS
[20:45:00]   sensor_failure         | cooldown        | sev=0.5 | PASS
[20:45:00]   sensor_failure         | steady_state    | sev=0.5 | PASS
[20:45:00]   sensor_failure         | high_load       | sev=0.5 | PASS
[20:45:00]   Generating Type-A normal sequences...
[20:45:00]
Section 7 done - 24/26 fault seqs PASS | Total seqs: 38
[20:45:00] ======================================================================
[20:45:00] SECTION 8 - Saving fault_rules.json + unit_registry.json
[20:45:00] ======================================================================
[20:45:00]   Saved: C:\Users\user\Desktop\PumpSmart_Project\models\fault_rules.json
[20:45:00]   Wrote fresh unit_registry.json → C:\Users\user\Desktop\PumpSmart_Project\models\unit_registry.json
[20:45:00] ======================================================================
[20:45:00] SECTION 9 - Saving M5 physics config for M6
[20:45:00] ======================================================================
[20:45:00]   Saved: C:\Users\user\Desktop\PumpSmart_Project\models\M5_physics_config.json
[20:45:00] ======================================================================
[20:45:00] SECTION 10 - Generating validation plots
[20:45:00] ======================================================================
[20:45:01]   Saved: C:\Users\user\Desktop\PumpSmart_Project\outputs\plots\M5_fault_signatures_validation.png
[20:45:02]   Saved: C:\Users\user\Desktop\PumpSmart_Project\outputs\plots\M5_thermal_coupling_validation.png
[20:45:02] ======================================================================
[20:45:02] SECTION 11 - Exporting src/physics_engine.py
[20:45:02] ======================================================================
[20:45:02]   Saved: C:\Users\user\Desktop\PumpSmart_Project\src\physics_engine.py
[20:45:02] ======================================================================
[20:45:02] END - Finalising report + paste text
[20:45:02] ======================================================================
[20:45:02]   Saved report: C:\Users\user\Desktop\PumpSmart_Project\outputs\reports\module_05_physics_engine_report.md

────────────────────────────────────────────────────────────
  PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT
────────────────────────────────────────────────────────────
M5_status                  : READY
M5_eq_pass                 : 20/20
M5_s5_all_pass             : True
M5_s7_fault_seqs_pass      : 24/26
M5_s7_total_seqs           : 38
M5_s7_all_pass             : False
M5_fail_list               : ['bearing_wear_steady_state_s100', 'overloading_steady_state_s140']
M5_fault_rules_saved       : True
M5_physics_engine_exported : True
M5_nameplate_P_hyd_kW      : 55.18
M5_nameplate_eta           : 0.5016
M5_nameplate_BPF_Hz        : 347.67
M5_tau_thermal_s           : 388.9
M5_joukowsky_bar           : 19.1
Status for next module     : READY
────────────────────────────────────────────────────────────

FILE MANIFEST
  [GitHub PUSH]   src/module_05_physics_engine.py
  [GitHub PUSH]   src/physics_engine.py
  [GitHub PUSH]   models/fault_rules.json
  [GitHub PUSH]   models/M5_physics_config.json
  [GitHub PUSH]   models/unit_registry.json
  [Spaces Upload] outputs/reports/module_05_physics_engine_report.md
  [Spaces Upload] outputs/plots/M5_fault_signatures_validation.png
  [Spaces Upload] outputs/plots/M5_thermal_coupling_validation.png

📦 M5 done.
```