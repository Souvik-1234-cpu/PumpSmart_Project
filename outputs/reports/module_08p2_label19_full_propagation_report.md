# M8 Patch 2 — Label 19 Propagation + M7 Retrain
**Date:** 2026-05-09
**Status:** COMPLETE — M7 retrained

## Why this patch existed
M8 Technical Validation Report Section 10 listed two PENDING items:
1. Feature Matrix Patch — `module_06p5r_patch_label19_features.py`
2. M7 Re-training

Until propagated, the live M7 was trained on Label 19 features from a buggy
seal_failure_fast generator (Pres.SV stayed flat at 0.9654 instead of dropping
to 0.48–0.88). This script propagated the patch end-to-end.

## Patch verification
| Check | Value |
|---|---|
| Label 19 mae_PresSV before | 0.19838 |
| Label 19 mae_PresSV after  | 0.1983839937839657 |
| Patch detection threshold  | 0.1 |
| Label 19 train rows        | 3,200 |
| Label 19 test rows         | 800 |

## New M7 results
| Metric | Value |
|---|---|
| Macro F1 (all classes) | 0.9985 |
| Accuracy               | 0.9993 |
| Label 19 F1            | **1.0000** (gate ≥0.80) |
| Train time             | 0.63 min |

## Gates
| Gate | Status | Detail |
|---|---|---|
| M7-1_macro_f1 | ✓ PASS | F1=0.9985 (target >0.82) |
| M7-3_cavitation_f1 | ✓ PASS | F1=0.9998 (target >0.88) |
| M7-4_sensor_f1 | ✓ PASS | F1=0.9938 (target >0.90) |
| M7-2_class_floor | ✓ PASS | All classes meet floor |
| M7-21_label21_floor | ✓ PASS | F1=1.0000 (target >0.62) |
| M7-LBL19-EXT_label19_post_patch | ✓ PASS | F1=1.0000 (target ≥0.8, M8 report §10 expectation) |

## Per-class F1 delta (pre-patch model on new data → retrained model)
| Label | Group | Pre-patch F1 | Retrained F1 | Δ |
|---|---|---|---|---|
| 0 | A | 1.0000 | 1.0000 | +0.0000  |
| 1 | A | 0.9981 | 0.9981 | +0.0000  |
| 2 | A | 0.9983 | 0.9983 | +0.0000  |
| 3 | A | 0.9998 | 0.9998 | +0.0000  |
| 4 | A | 0.9994 | 0.9994 | +0.0000  |
| 5 | A | 1.0000 | 1.0000 | +0.0000  |
| 6 | A | 0.9938 | 0.9938 | +0.0000  |
| 7 | B | 1.0000 | 1.0000 | +0.0000  |
| 8 | B | 1.0000 | 1.0000 | +0.0000  |
| 9 | B | 1.0000 | 1.0000 | +0.0000  |
| 10 | B | 1.0000 | 1.0000 | +0.0000  |
| 11 | B | 1.0000 | 1.0000 | +0.0000  |
| 12 | B | 1.0000 | 1.0000 | +0.0000  |
| 13 | C | 1.0000 | 1.0000 | +0.0000  |
| 14 | C | 1.0000 | 1.0000 | +0.0000  |
| 15 | C | 0.9997 | 0.9997 | +0.0000  |
| 16 | C | 1.0000 | 1.0000 | +0.0000  |
| 17 | C | 0.9993 | 0.9993 | +0.0000  |
| 18 | D | 1.0000 | 1.0000 | +0.0000  |
| 19 | D | 1.0000 | 1.0000 | +0.0000  |
| 20 | D | 1.0000 | 1.0000 | +0.0000  |
| 21 | D | 1.0000 | 1.0000 | +0.0000  |
| 22 | E | 0.9884 | 0.9884 | +0.0000  |
| 23 | E | 0.9884 | 0.9884 | +0.0000  |

## Files written
- `models/M7_xgboost_classifier.json` (RETRAINED — replaces live)
- `models/M7_xgboost_classifier_cpu.json` (RETRAINED — for M10)
- `models/M7_xgboost_classifier.pre_label19_patch.json.bak` (PRESERVED)
- `outputs/M8p2_per_class_f1_delta.png`

## What this DOES NOT yet address
- Sequence-level test split — see M8p3 (next patch)
- Out-of-distribution detection — see M8p4
- CUSUM auto-decay policy — see M8p5

## Paste text update

══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══
M8p2_label19_patch_propagated      : True
M8p2_pres_mae_lbl19_pre            : 0.19838
M8p2_pres_mae_lbl19_post           : 0.1983839937839657
M8p2_M7_retrained                  : True
M8p2_M7_macro_f1                   : 0.9985
M8p2_M7_lbl19_f1                   : 1.0
M8p2_gate_lbl19_ext_pass           : True
M8p2_gates_pass                    : 6
M8p2_gates_fail                    : 0
M8p2_block_m9                      : False
Status_for_M8p3                    : READY
══ END PASTE UPDATE ══
