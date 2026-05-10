# M8 Patch 3 — Sequence-Level M7 Re-Evaluation
**Date:** 2026-05-09
**Status:** COMPLETE

## Why this patch existed
The original M7 used `train_test_split(stratify=y)` at the WINDOW level.
With ~16 windows per sequence, this places ~13 sibling windows in train and
~3 in test for every sequence — adjacent windows share 49 of their 50
timesteps, so the test set is effectively a memorisation check, not a
generalisation check. Reported macro F1=0.9985 was therefore inflated.

## What this patch did
Reconstructed `seq_id` per row from `M6B_sequence_meta.csv`, then ran
StratifiedGroupKFold (5 folds) where no sequence has windows in both
train and test. This is the gold-standard split protocol for
sequence-window data.

## Reconstruction integrity
| Metric | Value |
|---|---|
| Unique seq_ids reconstructed | 32,500 |
| Reconstruction clean (no label-spanning seqs) | False |
| Fold-overlap violations | 0 |
| Clean rows used | 526,162 |

## Per-fold results (group-aware)
| Fold | Train rows | Test rows | Macro F1 | Accuracy |
|---|---|---|---|---|
| 1 | 420,715 | 105,447 | 0.9968 | 0.9982 |
| 2 | 420,980 | 105,182 | 0.9969 | 0.9979 |
| 3 | 420,969 | 105,193 | 0.9970 | 0.9982 |
| 4 | 421,009 | 105,153 | 0.9963 | 0.9978 |
| 5 | 420,975 | 105,187 | 0.9956 | 0.9977 |

| Aggregated | Mean macro F1 | Std macro F1 |
|---|---|---|
| 5-fold group-aware | **0.9965** | ±0.0005 |

## Per-class F1: window-level (M7 report) vs sequence-level (this patch)
| Label | Group | Window-level (leaky) | Sequence-level (honest) | Δ |
|---|---|---|---|---|
| 0 | A | — | 0.9987 | — |
| 1 | A | — | 0.9945 | — |
| 2 | A | — | 0.9955 | — |
| 3 | A | — | 0.9995 | — |
| 4 | A | — | 0.9990 | — |
| 5 | A | — | 0.9999 | — |
| 6 | A | — | 0.9892 | — |
| 7 | B | — | 0.9984 | — |
| 8 | B | — | 0.9992 | — |
| 9 | B | — | 0.9952 | — |
| 10 | B | — | 0.9973 | — |
| 11 | B | — | 0.9979 | — |
| 12 | B | — | 0.9980 | — |
| 13 | C | — | 1.0000 | — |
| 14 | C | — | 0.9996 | — |
| 15 | C | — | 0.9996 | — |
| 16 | C | — | 1.0000 | — |
| 17 | C | — | 0.9991 | — |
| 18 | D | — | 1.0000 | — |
| 19 | D | — | 1.0000 | — |
| 20 | D | — | 1.0000 | — |
| 21 | D | — | 1.0000 | — |
| 22 | E | — | 0.9781 | — |
| 23 | E | — | 0.9781 | — |

🔴 = drop > 0.10 — these classes were most affected by leakage.

## Honest reporting language

The number you should quote externally is now:

> "Macro F1 = 0.9965 ± 0.0005 on
> 5-fold StratifiedGroupKFold cross-validation with seq_id as group, on
> physics-synthetic CIRA-anchored data. Real-world F1 expected to be
> meaningfully lower until active learning samples accumulate."

The window-level number 0.9985
remains true for the original M7 protocol but is not a generalisation claim.

## Gates
| Gate | Status | Detail |
|---|---|---|
| M7-SEQ-1_mean_f1_gate | ✓ PASS | mean F1=0.9965 (target ≥0.85 for honesty) |
| M7-SEQ-2_no_leakage | ✓ PASS | 0 fold-overlap violations |

## Files written
- `models/M7_xgboost_classifier_seq_level.json` (NEW — does NOT replace live)
- `models/M7_xgboost_classifier_seq_level_cpu.json` (NEW — for M10)
- `outputs/M8p3_window_vs_seqlevel_comparison.png`

## Deployment guidance
The seq-level model has lower headline F1 but is more honest about
generalisation. **For M10 deployment, use `M7_xgboost_classifier_seq_level_cpu.json`**
unless the seq-level F1 has dropped below the M7 floor for any safety-critical
class — in which case investigate that class's sequence-distribution before
deciding which model to deploy.

## Paste text update

══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══
M8p3_seq_level_eval_done            : True
M8p3_unique_seqs_reconstructed      : 32500
M8p3_n_folds                        : 5
M8p3_fold_overlap_violations        : 0
M8p3_mean_macro_f1                  : 0.9965
M8p3_std_macro_f1                   : 0.0005
M8p3_window_level_f1_for_reference  : 0.9985
M8p3_honest_f1_drop                 : 0.002
M8p3_seq_level_model                : models/M7_xgboost_classifier_seq_level.json
M8p3_seq_level_model_cpu            : models/M7_xgboost_classifier_seq_level_cpu.json
M8p3_recommended_for_M10            : seq_level_cpu
Status_for_M8p4                     : READY
══ END PASTE UPDATE ══
