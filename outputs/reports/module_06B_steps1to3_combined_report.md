# M6B Steps 1+2+3 Generation Report

**Script:** `module_06B_steps1to3_combined` v1.0  
**Arch version:** v14.2  
**Date:** 2026-04-28  
**Device:** cuda  

## Results

| Key | Value |
|-----|-------|
| `step1_groupB_count` | 9000 |
| `step1_groupB_file` | C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequences_groupB.pkl |
| `step1_zt_groupB_file` | C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\z_t_sequences_groupB.pkl |
| `step2_groupC_count` | 6000 |
| `step2_groupD_count` | 5200 |
| `step2_label21_subthreshold_pct` | 68.0 |
| `step2_groupC_file` | C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequences_groupC.pkl |
| `step2_groupD_file` | C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_sequences_groupD.pkl |
| `step3_groupE_count` | 1600 |
| `step3_total_sequences` | 32500 |
| `step3_label_distribution` | {"1": 1500, "4": 1500, "5": 1500, "0": 2000, "2": 1500, "3": 1500, "6": 1200, "7": 1500, "8": 1500, "9": 1500, "10": 150 |
| `step3_label_min_count` | 800 |
| `step3_coupling_fidelity` | 0.42657142857142855 |
| `step3_combined_file` | C:\Users\user\Desktop\PumpSmart_Project\data\synthetic\M6B_combined_sequences.pkl |
| `step3_meta_rows` | 32500 |
| `step3_physics_context_generated` | True |
| `step3_fault_rules_v3_written` | True |
| `step3_physics_violations` | 0 |
| `arch_version` | v14.2 |
| `script_version` | 1.0 |
| `anomaly_threshold` | 0.110058 |
| `step1_label_range` | 7–12 |
| `step2_label_range` | 13–21 |
| `step3_label_range` | 22–23 (Group E) + merge |
| `gate_results_summary` | {"G1_groupB": 1.0, "G2_groupB": 1.0, "G8_temporal_ordering": true, "G9_compound_mae": true, "G10_masked_secondary": true |
| `step3_registry_json` | C:\Users\user\Desktop\PumpSmart_Project\outputs\reports\M6B_file_registry.json |
| `step3_registry_md` | C:\Users\user\Desktop\PumpSmart_Project\outputs\reports\M6B_file_registry.md |
| `status_for_M65r` | READY |

## Gate Results

| Gate | Rate/Value | Pass |
|------|-----------|------|
| G1_groupB | 1.0000 | ✓ |
| G2_groupB | 1.0000 | ✓ |
| G8_temporal_ordering | 1.0 | ✓ |
| G9_compound_mae | 0.9997777777777778 | ✓ |
| G10_masked_secondary | 1.0 | ✓ |
| G1_groupC | 1.0000 | ✓ |
| G2_groupC | 1.0000 | ✓ |
| G11ext_gradual_slope | 1.0 | ✓ |
| G1_groupD | 1.0000 | ✓ |
| G2_groupD | 1.0000 | ✓ |
| G11_multisensor | 1.0 | ✓ |
| G1_final | 1.0000 | ✓ |
| G2_final | 1.0000 | ✓ |
| thermal_coupling_fidelity | 0.4266 | ~ |
| physics_violations_final | 0.0000 | ~ |

## Status for M6.5r

**READY**

*All gates must pass before M6.5r feature extraction begins.*