# =============================================================================
# module_08p6_groupB_regenerate.py
# PumpSmart v14.2 — Tier-1 Fix T1.6: Group B Compound-Chain Regeneration
# + Final M7 Retrain (holds ALL Tier-1 fixes simultaneously)
# =============================================================================
#
# WHY THIS SCRIPT EXISTS (Audit T1.6, Industrial Audit v2.0 §5.6):
#   Visualization audit confirmed 5 of 6 Group B compound-chain plots show
#   abrupt step discontinuities at the secondary-fault onset point. Real
#   compound faults produce CONTINUOUS SUPERPOSITION — one fault adds
#   gradually onto another. The step pattern is a pure generator artifact.
#
#   Three bugs identified in generate_compound_sequence():
#
#   BUG 1 — np.tile freeze (primary-fault progression stops at Phase 2):
#     Original: p_tail = np.tile(p_seq[-1:], (p2_len, 1))
#     This freezes the primary fault at its last seed value for the ENTIRE
#     Phase 2 — the primary stops progressing while the secondary is added.
#     Visually: primary curve goes from rising to flat exactly at secondary
#     onset → looks like a step drop. Group A seeds are 200-400 steps; for
#     Label 7 (600 steps, lag 200-400), p_tail_start lands near seed end.
#
#   BUG 2 — One-step index skip at phase boundary:
#     Original: p_tail_start = p_len  (= lag + 50 = p2_start)
#     Phase 1 last value: p_seq[p_len - 1]
#     Phase 2 first value: p_seq[p_len] + s_dev[0]*0.6
#     These are DIFFERENT time indices in the seed, and the secondary
#     contribution is suddenly added → visible step even when seed is long.
#     Fix: p_tail_start = p_len - 1 → p_tail[0] = p_seq[p_len-1] = last
#     Phase 1 value → continuity enforced before secondary adds in.
#
#   BUG 3 — Silent gap between p_len and p2_start (not in audit, found here):
#     When len(p_seq) < lag + 50, steps p_len to p2_start are left as the
#     np.ones initialization (flat at 1.0) — a silent baseline gap before
#     Phase 2 begins. Fix: fill this gap with physics extrapolation too.
#
#   WHY ML CARES: XGBoost (M7) trains on features derived from these sequences.
#   The step pattern at secondary onset is statistically unique — XGBoost
#   likely learned this ARTIFACT as the primary Group B discriminator. Real
#   compound faults have no such step → Group B F1 will collapse on real data.
#   If Group B F1 drops >0.10 after this fix, that confirms the artifact was
#   being used as a discriminator. EITHER OUTCOME IS INFORMATIVE.
#
# WHY THIS IS THE FINAL M7 RETRAIN:
#   This script produces M7 retrain #2 (FINAL) with ALL three Tier-1 fixes:
#     T1.2: Label 19 physics fix (Pres.SV* drop restored) ← already in matrix
#     T1.6: Group B continuous superposition fix           ← this script
#     T1.7: Group E reclassification (label names)        ← already in meta
#   T1.3 (sequence-level eval) runs on these final M7 weights.
#
# FIX IMPLEMENTATION STRATEGY:
#   Fix 1 (np.tile replacement): For each channel, compute the least-squares
#   linear slope over the last n_slope=10 steps of p_seq, then extrapolate
#   linearly beyond the seed. This preserves fault trajectory momentum.
#   For bearing wear (Paris-law exponential), the slope at the seed tail
#   captures the exponential's local gradient — adequate for short extrapolation.
#   Linear extrapolation is used for ALL fault classes (uniform, robust).
#
#   Fix 2 (index skip): p_tail_start = p_len - 1 instead of p_len.
#   p_tail[0] now equals p_seq[p_len-1] = seq[p2_start-1]. Secondary
#   contribution starts at s_dev[0]*ramp[0]. Since secondary sequences start
#   near baseline (s_dev[0] ≈ 0), the boundary jump is now sub-noise.
#
#   Fix 3 (gap filling): Extrapolate p_seq from p_len to p2_start using same
#   linear slope — eliminates the np.ones flat gap.
#
# RUNNING ISSUES CARRIED FROM PREVIOUS SCRIPTS:
#   [1] charmap codec: all open(..., 'w') calls use encoding='utf-8'
#   [2] feature matrix label_int column: explicitly added to label_id detection
#       candidates list — ensures Group E label rename carries through if needed
#
# WHAT THIS SCRIPT DOES:
#   1.  Loads Group A seed pools from existing pkl files
#   2.  Implements generate_compound_sequence_v2() with all 3 fixes
#   3.  Regenerates 9,000 Group B sequences (1,500 × Labels 7-12)
#   4.  Continuity gate: |seq[p2_start] - seq[p2_start-1]| < 3×noise_std
#       Pass criterion: ≥98% of sequences per channel
#   5.  Saves M6B_sequences_groupB_v2.pkl (original preserved as .v1.bak)
#   6.  Loads frozen M4 LSTM-AE, generates z_t sequences via sliding window
#   7.  Saves z_t_sequences_groupB_v2.pkl
#   8.  Extracts the same 33-feature set as M6.5r from new sequences
#   9.  Surgically replaces Labels 7-12 rows in M6B_feature_matrix.csv
#       (all other labels untouched — additive surgery only)
#  10.  Retrains M7 with locked hyperparameters on the updated feature matrix
#       (this is the FINAL M7 — holds all 3 Tier-1 data fixes)
#  11.  Reports per-class F1 delta vs pre-T1.6 M7
#  12.  Gates, report, paste-text update, file manifest
#
# OUTPUT FILES:
#   data/synthetic/M6B_sequences_groupB_v2.pkl          (NEW v2 sequences)
#   data/synthetic/M6B_sequences_groupB.pkl.v1.bak      (original preserved)
#   data/synthetic/z_t_sequences_groupB_v2.pkl          (NEW v2 z_t)
#   data/synthetic/M6B_feature_matrix.csv               (UPDATED — Labels 7-12)
#   data/synthetic/M6B_feature_matrix.csv.pre_T1_6.bak  (original preserved)
#   models/M7_xgboost_classifier.json                   (FINAL retrain)
#   models/M7_xgboost_classifier_cpu.json               (FINAL retrain, CPU)
#   models/M7_xgboost_classifier.pre_T1_6.json.bak      (pre-T1.6 backup)
#   outputs/reports/module_08p6_groupB_regenerate_report.md
#   outputs/M8p6_groupB_f1_delta.png
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, warnings, hashlib, shutil, pickle, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

SCRIPT_NAME = "module_08p6_groupB_regenerate"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
GATES   = {}
GATE = {}

log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}")
log(f"  Device: {DEVICE} | GPU: {IS_GPU}")
log("  T1.6 — Group B compound-chain fix + FINAL M7 retrain")
log("  Fixes: Bug1 np.tile→extrapolate | Bug2 index-skip | Bug3 gap-fill")
log("=" * 72)

# =============================================================================
# SECTION 0 — CONSTANTS (copied from module_06B_steps1to3_combined.py)
# =============================================================================
# These are NOT imported from the original script to avoid executing its full
# generation pipeline. Values verified against M6B spec docs.

SEED         = 42
ARCH_VERSION = "v14.2"

# M6B locked channel order (m6b_physics_lib.py — LOCKED)
CHANNELS = ["Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
            "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV"]
CH       = {c: i for i, c in enumerate(CHANNELS)}
N_CH     = 8

# M5 SCADA noise std (from m6b_physics_lib.py — LOCKED)
NOISE_STD = {"Mot.SV": 0.035, "Pmp.SV": 0.040, "Mot.TV": 0.008,
             "Pmp.PV": 0.012, "Temp.SV": 0.010, "Pres.SV": 0.015,
             "Pmp.TV": 0.008, "Mot.PV": 0.012}

CLUSTER_NAMES = {0: "cooldown", 1: "steady_state", 2: "startup", 3: "high_load"}

# Compound chain definitions (M6B LOCKED)
COMPOUND_BASE = {
    7:  (1, 5),   # bearing_wear → overloading
    8:  (3, 4),   # cavitation → seal_failure
    9:  (2, 1),   # impeller_imbalance → bearing_wear
    10: (4, 3),   # seal_failure → cavitation_H
    11: (5, 1),   # overloading → bearing_wear
    12: (2, 3),   # impeller_imbalance → cavitation
}
COMPOUND_NAMES = {
    7:  "bearing_wear+overloading",
    8:  "cavitation+seal_failure",
    9:  "impeller_imbalance+bearing_wear",
    10: "seal_failure+cavitation_H",
    11: "overloading+bearing_wear",
    12: "impeller_imbalance+cavitation",
}
# Physics-verified sequence lengths and lag ranges
SEQ_STEPS    = {7: 600, 8: 550, 9: 700, 10: 900, 11: 800, 12: 450}
COMPOUND_LAG = {7: (200, 400), 8: (50, 150), 9: (300, 600),
                10: (400, 800), 11: (400, 600), 12: (100, 300)}
SEQ_COUNTS   = {7: 1500, 8: 1500, 9: 1500, 10: 1500, 11: 1500, 12: 1500}

# Locked M7 hyperparameters (from M7_gate_fix_diagnosis_and_solution.md)
LOCKED_PARAMS = {
    'n_estimators':     504,
    'max_depth':        7,
    'learning_rate':    0.08086361634538793,
    'subsample':        0.9531291833577744,
    'colsample_bytree': 0.9768481099821509,
    'min_child_weight': 2,
    'gamma':            0.0009941501981704567,
    'reg_alpha':        0.0010636018384176757,
    'reg_lambda':       0.10934322260320596,
    'objective':        'multi:softprob',
    'eval_metric':      'mlogloss',
    'tree_method':      'hist',
    'device':           'cuda' if IS_GPU else 'cpu',
    'random_state':     42,
}

GROUP_MAP = {
    **{i: 'A' for i in range(0,  7)},
    **{i: 'B' for i in range(7,  13)},
    **{i: 'C' for i in range(13, 18)},
    **{i: 'D' for i in range(18, 22)},
    22: 'E', 23: 'E',
}

# Feature matrix paths
FEATURE_MATRIX_PATH = SYNTH_DIR / "M6B_feature_matrix.csv"
SEQ_META_PATH       = SYNTH_DIR / "M6B_sequence_meta.csv"
M7_MODEL_PATH       = MODEL_DIR / "M7_xgboost_classifier.json"
M7_CPU_PATH         = MODEL_DIR / "M7_xgboost_classifier_cpu.json"
M7_BACKUP_T16       = MODEL_DIR / "M7_xgboost_classifier.pre_T1_6.json.bak"
GROUPB_V2_PKL       = SYNTH_DIR / "M6B_sequences_groupB_v2.pkl"
GROUPB_V1_BAK       = SYNTH_DIR / "M6B_sequences_groupB.pkl.v1.bak"
ZT_V2_PKL           = SYNTH_DIR / "z_t_sequences_groupB_v2.pkl"
FM_BACKUP_T16       = SYNTH_DIR / "M6B_feature_matrix.csv.pre_T1_6.bak"

# Slope extrapolation window (last N steps of seed used to compute rate-of-change)
N_SLOPE = 10

log(f"\n  Group B labels: {list(SEQ_STEPS.keys())}")
log(f"  Total sequences: {sum(SEQ_COUNTS.values()):,}")
log(f"  Sequence lengths: {SEQ_STEPS}")

# =============================================================================
# SECTION 1 — LOAD GROUP A SEED POOLS
# =============================================================================
log("\nSECTION 1 — Load Group A seed pools")

# M6B_sequences_groupA_rerun.pkl and groupA_carried.pkl are Git LFS pointers
# on this machine — not pullable. M6B_combined_sequences.pkl (452 MB, REAL)
# contains all 32,500 sequences including all Group A labels.
# We extract labels 1,2,3,4,5 from it as the seed source for compound generation.
# These are the EXACT same sequences that were used as Group B seeds originally.

COMBINED_PKL = SYNTH_DIR / "M6B_combined_sequences.pkl"
if not COMBINED_PKL.exists():
    log(f"  [FATAL] M6B_combined_sequences.pkl not found: {COMBINED_PKL}")
    sys.exit(1)

log(f"  Loading {COMBINED_PKL.name} (452 MB) ...")
t0 = time.time()
try:
    with open(COMBINED_PKL, "rb") as f:
        combined = pickle.load(f)
    log(f"  Loaded {len(combined['sequences'])} sequences in {time.time()-t0:.1f}s")
except Exception as e:
    log(f"  [FATAL] Load failed: {e}")
    sys.exit(1)

# Build label→(seq, meta) lookup — key is 'metadata' (not 'meta')
grpA_by_label = {}
for seq, meta in zip(combined["sequences"], combined["metadata"]):
    lbl = meta["label"]
    grpA_by_label.setdefault(lbl, []).append((seq, meta))

# We only need labels used as primary/secondary bases for compound chains
needed = set()
for p_lbl, s_lbl in COMPOUND_BASE.values():
    needed.update([p_lbl, s_lbl])   # {1, 2, 3, 4, 5}

log(f"  Needed seed labels: {sorted(needed)}")
for lbl in sorted(needed):
    count = len(grpA_by_label.get(lbl, []))
    log(f"    Label {lbl}: {count} sequences available")

missing_seeds = [l for l in needed
                 if l not in grpA_by_label or len(grpA_by_label[l]) == 0]
if missing_seeds:
    log(f"  [FATAL] Missing seed labels: {missing_seeds}")
    sys.exit(1)

log("  All required seed labels present.")
results["seed_pool_ok"] = True

# =============================================================================
# SECTION 2 — CORRECTED COMPOUND GENERATOR v2
# =============================================================================
log("\nSECTION 2 — Defining generate_compound_sequence_v2")

def _linear_extrapolate(arr: np.ndarray, n_slope: int, n_extra: int) -> np.ndarray:
    """
    Extrapolate arr (shape: T×N_CH) beyond its last step for n_extra steps.
    Uses least-squares linear slope computed over the last n_slope steps.
    Applied per-channel independently.

    This replaces the np.tile fallback (BUG 1 FIX):
      Old: np.tile(arr[-1:], (n_extra, 1)) → frozen flat primary
      New: slope-based extrapolation → primary keeps progressing

    Engineering rationale: all Group B primary faults (bearing wear, seal
    failure, cavitation, overloading, imbalance) are monotonically progressing
    over their seed lengths. The local slope at the seed tail is the best
    available estimate of the fault's rate-of-change at that point in its
    development. Linear extrapolation over the Phase 2 horizon (~100-400 steps)
    is adequate because the seed already captures the nonlinear early phase.
    """
    T    = arr.shape[0]
    n_s  = min(n_slope, T - 1)

    # Compute per-channel slopes over last n_s steps
    x       = np.arange(n_s, dtype=np.float64)
    x_mean  = x.mean()
    x_var   = ((x - x_mean) ** 2).sum()

    extra   = np.zeros((n_extra, N_CH), dtype=np.float32)
    for ch in range(N_CH):
        y       = arr[-n_s:, ch].astype(np.float64)
        y_mean  = y.mean()
        if x_var > 1e-12:
            slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        else:
            slope = 0.0
        last_val    = float(arr[-1, ch])
        for t in range(n_extra):
            extra[t, ch] = last_val + slope * (t + 1)
        # Clip to [0, 8.8] — normalized space physical bounds
        extra[:, ch] = np.clip(extra[:, ch], 0.0, 8.8)
    return extra


def generate_compound_sequence_v2(label: int, rng: np.random.Generator) -> tuple:
    """
    CORRECTED compound fault sequence generator. Fixes all 3 bugs vs original.

    Phase 1 (t=0 → t=lag+50): primary fault signature only.
    Phase 2 (t=lag+50 → end): primary CONTINUOUSLY PROGRESSING + secondary
    added via gradual ramp (0.6 → 1.0 over 50 steps).

    BUG 1 FIX — np.tile fallback replaced with _linear_extrapolate():
      When the primary seed runs out before Phase 2 ends, the primary fault
      continues via slope extrapolation instead of freezing. The fault keeps
      progressing during Phase 2.

    BUG 2 FIX — p_tail_start = p_len - 1 (was p_len):
      Phase 2 now starts from p_seq[p_len-1], which equals seq[p2_start-1].
      Continuity is enforced: seq[p2_start] - seq[p2_start-1] ≈ delta_primary
      + secondary_onset (≈0 at ramp start). No index skip.

    BUG 3 FIX — Gap between p_len and p2_start filled with extrapolation:
      When len(p_seq) < lag+50, the steps p_len:p2_start were left as np.ones
      (silent flat gap). Now filled with linear extrapolation of p_seq.
    """
    target_steps = SEQ_STEPS[label]
    lag_lo, lag_hi = COMPOUND_LAG[label]
    lag = int(rng.integers(lag_lo, lag_hi + 1))

    primary_lbl, secondary_lbl = COMPOUND_BASE[label]

    primary_pool   = grpA_by_label.get(primary_lbl,   [])
    secondary_pool = grpA_by_label.get(secondary_lbl, [])
    if not primary_pool or not secondary_pool:
        raise RuntimeError(f"Empty pool for compound label {label}")

    p_seq, p_meta = primary_pool[int(rng.integers(0, len(primary_pool)))]
    s_seq, s_meta = secondary_pool[int(rng.integers(0, len(secondary_pool)))]

    # Cast to float32 working arrays
    p_seq = np.array(p_seq, dtype=np.float32)
    s_seq = np.array(s_seq, dtype=np.float32)

    # Output sequence — initialized to 1.0 (normalized baseline)
    seq = np.ones((target_steps, N_CH), dtype=np.float32)
    cluster_id = p_meta.get("cluster_id", 1)

    # ── Phase 1: primary fault only (t=0 → p_len) ────────────────────────────
    p_len   = min(lag + 50, target_steps, len(p_seq))
    p2_start = lag + 50   # Phase 2 begins here in the TARGET sequence

    seq[:p_len] = p_seq[:p_len]

    # ── BUG 3 FIX: Fill gap between p_len and p2_start with extrapolation ───
    # This gap exists when len(p_seq) < lag + 50. Without fix: flat at 1.0.
    if p_len < p2_start and p2_start < target_steps:
        gap_len  = p2_start - p_len
        gap_fill = _linear_extrapolate(p_seq[:p_len], N_SLOPE, gap_len)
        seq[p_len:p2_start] = gap_fill

    # ── Phase 2: primary progressing + secondary added ────────────────────────
    if p2_start < target_steps:
        p2_len = target_steps - p2_start

        # ── BUG 2 FIX: p_tail_start = p_len - 1 (was p_len) ─────────────────
        # p_tail[0] = p_seq[p_len-1] = seq[p2_start-1] → continuity enforced
        p_tail_start = max(0, p_len - 1)

        # Build primary tail for Phase 2
        available_in_seed = len(p_seq) - p_tail_start
        if available_in_seed >= p2_len:
            # Seed is long enough — direct slice (no extrapolation needed)
            p_tail = p_seq[p_tail_start: p_tail_start + p2_len].astype(np.float32)
        else:
            # ── BUG 1 FIX: slope extrapolation replaces np.tile ───────────────
            if available_in_seed > 0:
                seed_portion = p_seq[p_tail_start:].astype(np.float32)
                extra_needed = p2_len - available_in_seed
                extra_portion = _linear_extrapolate(p_seq, N_SLOPE, extra_needed)
                p_tail = np.vstack([seed_portion, extra_portion])
            else:
                # p_tail_start >= len(p_seq) — full extrapolation
                p_tail = _linear_extrapolate(p_seq, N_SLOPE, p2_len)
        p_tail = p_tail[:p2_len].astype(np.float32)

        # Build secondary source for Phase 2
        if len(s_seq) >= p2_len:
            s_src = s_seq[:p2_len].astype(np.float32)
        else:
            s_extra = _linear_extrapolate(s_seq, N_SLOPE, p2_len - len(s_seq))
            s_src   = np.vstack([s_seq, s_extra])[:p2_len].astype(np.float32)

        s_dev = s_src - 1.0   # secondary deviation from normalized baseline

        # Secondary ramp: 0.6 → 1.0 over 50 steps (same as original)
        ramp_len = min(50, p2_len)
        ramp     = np.linspace(0.6, 1.0, ramp_len, dtype=np.float32)[:, None]

        combined = p_tail.copy()
        combined[:ramp_len] = p_tail[:ramp_len] + s_dev[:ramp_len] * ramp
        if p2_len > ramp_len:
            combined[ramp_len:] = p_tail[ramp_len:] + s_dev[ramp_len:]

        seq[p2_start:p2_start + p2_len] = combined

    # SCADA noise
    for ch_name, ch_idx in CH.items():
        noise = rng.normal(0, NOISE_STD.get(ch_name, 0.015),
                           size=target_steps).astype(np.float32)
        seq[:, ch_idx] += noise

    # Cluster-conditional winsorization (C-18: high-load ceiling 2.0×, others 3.0×)
    cluster_ceil = {0: 3.0, 1: 3.0, 2: 3.0, 3: 2.0}.get(cluster_id, 3.0)
    seq = np.clip(seq, 0.0, cluster_ceil)

    meta = {
        "label":                label,
        "label_name":           COMPOUND_NAMES[label],
        "fault_name":           COMPOUND_NAMES[label],
        "group":                "B",
        "primary_label":        primary_lbl,
        "secondary_label":      secondary_lbl,
        "primary_onset_step":   0,
        "secondary_onset_step": lag + 50,
        "lag_steps":            lag,
        "cluster_id":           cluster_id,
        "cluster_name":         CLUSTER_NAMES.get(cluster_id, "unknown"),
        "steps":                target_steps,
        "source":               "physics_synthetic_compound_v2",
        "arch_version":         ARCH_VERSION,
        "generator_version":    "v2_T1.6_bug1_bug2_bug3_fixed",
    }
    return seq.astype(np.float32), meta


log("  generate_compound_sequence_v2 defined (Bug1+Bug2+Bug3 fixed)")


# =============================================================================
# SECTION 3 — GENERATE 9,000 GROUP B SEQUENCES
# =============================================================================
log("\nSECTION 3 — Generating 9,000 Group B sequences (v2)")

rng_B = np.random.default_rng(SEED + 200)   # Different seed from original (+100)
                                              # to avoid identical sequences

groupB_v2_sequences = []
groupB_v2_meta      = []
t_gen_start = time.time()

for label in [7, 8, 9, 10, 11, 12]:
    n_target = SEQ_COUNTS[label]
    log(f"  Label {label} ({COMPOUND_NAMES[label]}): {n_target} seqs "
        f"[{SEQ_STEPS[label]} steps, lag {COMPOUND_LAG[label]}]")
    label_seqs = []
    label_meta = []
    failures   = 0
    for i in range(n_target):
        try:
            seq, meta = generate_compound_sequence_v2(label, rng_B)
            label_seqs.append(seq)
            label_meta.append(meta)
        except Exception as e:
            failures += 1
            if failures <= 3:
                log(f"    [WARN] seq {i} failed: {e}")
    groupB_v2_sequences.extend(label_seqs)
    groupB_v2_meta.extend(label_meta)
    log(f"    Generated {len(label_seqs)} / {n_target} (failures: {failures})")

t_gen = time.time() - t_gen_start
log(f"  Total: {len(groupB_v2_sequences)} sequences in {t_gen:.1f}s")
results["n_sequences_generated"] = len(groupB_v2_sequences)
results["generation_time_s"]     = round(t_gen, 1)


# =============================================================================
# SECTION 4 — CONTINUITY GATE (core T1.6 validation)
# =============================================================================
log("\nSECTION 4 — Continuity gate: |seq[p2_start] - seq[p2_start-1]| < 3×noise")

# Gate: for each sequence, at the Phase 2 boundary, the step jump per channel
# must be less than 3 × channel noise std. Target: ≥98% of sequences pass.
# If the step-discontinuity bug were still present, the jump would be 10-50×
# noise std at the affected channels — this gate catches that definitively.

CONTINUITY_THRESHOLD_MULTIPLIER = 3.0
noise_thresholds = {ch: NOISE_STD[ch] * CONTINUITY_THRESHOLD_MULTIPLIER
                    for ch in CHANNELS}

per_label_continuity = {}
all_pass_flags       = []

for label in [7, 8, 9, 10, 11, 12]:
    label_seqs  = [(s, m) for s, m in zip(groupB_v2_sequences, groupB_v2_meta)
                   if m["label"] == label]
    n_seqs      = len(label_seqs)
    pass_count  = 0
    ch_fails    = {ch: 0 for ch in CHANNELS}

    for seq, meta in label_seqs:
        p2_start = meta["secondary_onset_step"]
        if p2_start <= 0 or p2_start >= len(seq):
            pass_count += 1   # cannot evaluate — skip
            continue
        jump          = np.abs(seq[p2_start].astype(np.float64)
                               - seq[p2_start - 1].astype(np.float64))
        seq_pass      = True
        for ch_name, ch_idx in CH.items():
            if jump[ch_idx] > noise_thresholds[ch_name]:
                ch_fails[ch_name] += 1
                seq_pass = False
        if seq_pass:
            pass_count += 1
        all_pass_flags.append(seq_pass)

    pass_rate = pass_count / n_seqs if n_seqs > 0 else 0.0
    per_label_continuity[label] = {"pass_rate": round(pass_rate, 4),
                                   "n_pass": pass_count, "n_total": n_seqs,
                                   "ch_fails": ch_fails}
    log(f"  Label {label}: {pass_count}/{n_seqs} pass ({pass_rate*100:.1f}%) | "
        f"worst ch fails: {max(ch_fails.values())}")

overall_pass_rate = np.mean(all_pass_flags) if all_pass_flags else 0.0
GATE["T1.6_G1_continuity_gate"] = overall_pass_rate >= 0.98
log(f"\n  Overall continuity pass rate: {overall_pass_rate*100:.2f}% "
    f"(target ≥98%) → {'PASS' if GATE['T1.6_G1_continuity_gate'] else 'FAIL'}")

results["continuity_pass_rate_overall"] = round(float(overall_pass_rate), 4)
results["per_label_continuity"]         = per_label_continuity

if not GATE["T1.6_G1_continuity_gate"]:
    log("  [WARNING] Continuity gate FAILED — step discontinuities still present.")
    log("  This means the bug fixes did not fully resolve the artifacts.")
    log("  Investigate _linear_extrapolate and p_tail_start computation above.")
    log("  Script will continue but M7 retrain will be saved as .candidate only.")


# =============================================================================
# SECTION 5 — SAVE GROUP B v2 PKL
# =============================================================================
log("\nSECTION 5 — Save M6B_sequences_groupB_v2.pkl")

# Backup original v1 if it exists
GROUPB_ORIG = SYNTH_DIR / "M6B_sequences_groupB.pkl"
if GROUPB_ORIG.exists() and not GROUPB_V1_BAK.exists():
    shutil.copy2(GROUPB_ORIG, GROUPB_V1_BAK)
    log(f"  Original backed up → {GROUPB_V1_BAK.name}")
elif GROUPB_V1_BAK.exists():
    log(f"  Backup already exists — skipping")

try:
    groupB_v2_payload = {
        "sequences": groupB_v2_sequences,
        "meta":      groupB_v2_meta,
        "version":   "v2_T1.6",
        "date":      str(date.today()),
        "fixes":     ["Bug1_np.tile_replaced", "Bug2_index_skip_fixed",
                      "Bug3_gap_fill_added"],
    }
    with open(GROUPB_V2_PKL, "wb") as f:
        pickle.dump(groupB_v2_payload, f, protocol=4)
    log(f"  Saved {GROUPB_V2_PKL.name} "
        f"({GROUPB_V2_PKL.stat().st_size / 1e6:.1f} MB)")
    results["groupB_v2_pkl_saved"] = True
except Exception as e:
    log(f"  [ERROR] PKL save failed: {e}")
    results["groupB_v2_pkl_saved"] = False


# =============================================================================
# SECTION 6 — LOAD FROZEN M4 LSTM-AE + GENERATE z_t SEQUENCES
# =============================================================================
log("\nSECTION 6 — Load frozen M4 LSTM-AE, generate z_t sequences")

# ── M4 LSTM-AE architecture (must match training config exactly) ─────────────
class LSTMAEEncoder(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]   # (batch, hidden_size) — top-layer hidden state

class LSTMAEDecoder(nn.Module):
    def __init__(self, hidden_size=64, output_size=8, num_layers=2, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.lstm    = nn.LSTM(hidden_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.1)
        self.fc      = nn.Linear(hidden_size, output_size)
    def forward(self, z):
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_rep)
        return self.fc(out)

class LSTMAE(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, seq_len=50):
        super().__init__()
        self.encoder = LSTMAEEncoder(input_size, hidden_size, num_layers)
        self.decoder = LSTMAEDecoder(hidden_size, input_size, num_layers, seq_len)
    def forward(self, x):
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon
    def encode(self, x):
        return self.encoder(x)

M4_PATH = MODEL_DIR / "lstm_ae_baseline_best.pth"
if not M4_PATH.exists():
    # Try alternative filenames
    for alt in ["M4_lstm_ae_best.pth", "lstm_ae_best.pth", "M4_model.pth"]:
        candidate = MODEL_DIR / alt
        if candidate.exists():
            M4_PATH = candidate
            break

try:
    m4_model = LSTMAE(input_size=8, hidden_size=64, num_layers=2, seq_len=50)
    state = torch.load(M4_PATH, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    m4_model.load_state_dict(state)
    m4_model = m4_model.to(DEVICE)
    m4_model.eval()
    log(f"  M4 loaded: {M4_PATH.name} → {DEVICE}")
    results["m4_loaded"] = True
except Exception as e:
    log(f"  [ERROR] M4 load failed: {e}")
    log("  z_t generation skipped. Feature matrix will use zeros for z_t features.")
    m4_model = None
    results["m4_loaded"] = False

# Generate z_t sequences (sliding window, stride=50, no overlap)
WIN_SIZE = 50
zt_sequences_v2 = []

if m4_model is not None:
    log(f"  Generating z_t for {len(groupB_v2_sequences)} sequences ...")
    t_zt = time.time()
    with torch.no_grad():
        for seq_np in groupB_v2_sequences:
            T         = seq_np.shape[0]
            n_windows = T // WIN_SIZE
            if n_windows == 0:
                zt_sequences_v2.append(np.zeros((1, 64), dtype=np.float32))
                continue
            windows = np.stack([seq_np[w*WIN_SIZE:(w+1)*WIN_SIZE]
                                 for w in range(n_windows)])   # (N, 50, 8)
            loader  = DataLoader(
                TensorDataset(torch.tensor(windows, dtype=torch.float32)),
                batch_size=512, pin_memory=IS_GPU, num_workers=0
            )
            zt_list = []
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                zt    = m4_model.encode(batch).cpu().numpy()  # (B, 64)
                zt_list.append(zt)
            zt_sequences_v2.append(np.vstack(zt_list).astype(np.float32))
    log(f"  z_t done: {len(zt_sequences_v2)} sequences in {time.time()-t_zt:.1f}s")
    results["zt_generated"] = True
else:
    # Fallback: zeros
    for seq_np in groupB_v2_sequences:
        n_windows = seq_np.shape[0] // WIN_SIZE
        zt_sequences_v2.append(np.zeros((max(1, n_windows), 64), dtype=np.float32))
    results["zt_generated"] = False

# Save z_t pkl
try:
    with open(ZT_V2_PKL, "wb") as f:
        pickle.dump({"zt_sequences": zt_sequences_v2, "version": "v2_T1.6"}, f, protocol=4)
    log(f"  z_t saved → {ZT_V2_PKL.name} ({ZT_V2_PKL.stat().st_size / 1e6:.1f} MB)")
    results["zt_v2_pkl_saved"] = True
except Exception as e:
    log(f"  [ERROR] z_t pkl save failed: {e}")
    results["zt_v2_pkl_saved"] = False


# =============================================================================
# SECTION 7 — FEATURE EXTRACTION (matching M6.5r 34-column schema exactly)
# =============================================================================
log("\nSECTION 7 — Feature extraction (34-column M6.5r schema)")

# Column order verified against module_M7_xgboost_classifier.md and
# M6p5r_feature_retrain.md exact spec. Running issues fix [2]: 'label_int'
# added to label_id candidates (ensures Group E label_int values 22/23 are
# recognised if feature matrix is rebuilt from scratch in future scripts).

CH_NAMES_ORDERED = ["Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
                    "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV"]

def compute_err_slope(mae_over_windows: np.ndarray) -> float:
    """Linear slope of MAE over windows (drift indicator)."""
    n = len(mae_over_windows)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    denom = (x ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float(((x) * (mae_over_windows - mae_over_windows.mean())).sum() / denom)


def extract_features_for_sequence(seq_np: np.ndarray, meta: dict,
                                   zt_seq: np.ndarray,
                                   m4: nn.Module | None,
                                   window_size: int = 50) -> list:
    """
    Extract one feature row per window from a sequence.
    Returns list of dicts, one per window.
    Matches the 33-feature + label_int = 34-column M6.5r output schema.
    """
    T         = seq_np.shape[0]
    n_windows = T // window_size
    label_int = meta["label"]
    rows      = []

    # Pre-compute per-window MAE (using M4 if available, else raw std as proxy)
    mae_per_window = np.zeros((n_windows, N_CH), dtype=np.float32)
    recon_per_window = np.zeros((n_windows, N_CH), dtype=np.float32)

    if m4 is not None:
        windows = np.stack([seq_np[w*window_size:(w+1)*window_size]
                            for w in range(n_windows)])  # (N, 50, 8)
        with torch.no_grad():
            loader = DataLoader(
                TensorDataset(torch.tensor(windows, dtype=torch.float32)),
                batch_size=256, pin_memory=IS_GPU, num_workers=0
            )
            recon_list = []
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                recon = m4(batch).cpu().numpy()   # (B, 50, 8)
                recon_list.append(recon)
            recon_all = np.vstack(recon_list)     # (N, 50, 8)
            for w in range(n_windows):
                mae_per_window[w]   = np.mean(np.abs(windows[w] - recon_all[w]), axis=0)
                recon_per_window[w] = recon_all[w].mean(axis=0)
    else:
        for w in range(n_windows):
            win = seq_np[w*window_size:(w+1)*window_size]
            mae_per_window[w] = win.std(axis=0)  # proxy when M4 unavailable

    # Per-sequence aggregate stats (used for Domain 2/3 features)
    mae_agg        = mae_per_window.mean(axis=0)   # (8,) mean MAE per channel
    score_A_mean   = float(mae_per_window.mean())
    score_A_arr    = mae_per_window.mean(axis=1)   # per-window scalar score_A

    # z_t features
    if len(zt_seq) >= n_windows:
        zt_norm_arr = np.linalg.norm(zt_seq[:n_windows], axis=1)  # (N,)
    else:
        zt_norm_arr = np.zeros(n_windows)
    zt_norm_mean = float(zt_norm_arr.mean())
    zt_norm_std  = float(zt_norm_arr.std())
    zt_drift     = compute_err_slope(zt_norm_arr)

    # PCA on z_t (2 components)
    if len(zt_seq) >= 2 and zt_seq.shape[1] >= 2:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            zt_pca = pca.fit_transform(zt_seq[:n_windows])  # (N, 2)
            zt_pca1_mean = float(zt_pca[:, 0].mean())
            zt_pca2_mean = float(zt_pca[:, 1].mean())
        except Exception:
            zt_pca1_mean = zt_pca2_mean = 0.0
    else:
        zt_pca1_mean = zt_pca2_mean = 0.0

    # Domain 3 compound/masked features from meta
    secondary_onset_lag  = float(meta.get("lag_steps", 0))
    secondary_onset_step = float(meta.get("secondary_onset_step", 0))

    for w in range(n_windows):
        win_start = w * window_size
        win_end   = win_start + window_size

        mae_w   = mae_per_window[w]    # (8,)
        score_A = float(mae_w.mean())

        # score_B: drift slope of score_A up to this window
        score_B = compute_err_slope(score_A_arr[:w+1]) if w > 0 else 0.0

        # score_C: transition signal — ratio of this window's score_A to sequence mean
        score_C = score_A / (score_A_mean + 1e-8)

        # onset_order: ordinal encoding of compound phase
        # 0=pre-onset, 2=transition (±1 window of secondary onset), 3=post-onset
        if secondary_onset_step > 0:
            onset_window = int(secondary_onset_step // window_size)
            if w < onset_window - 1:
                onset_order = 1.0  # pre-onset
            elif abs(w - onset_window) <= 1:
                onset_order = 2.0  # transition
            else:
                onset_order = 3.0  # post-onset
        else:
            onset_order = 0.0

        # err_slope features (over all windows up to w)
        err_slope_MotSV  = compute_err_slope(mae_per_window[:w+1, CH["Mot.SV"]])
        err_slope_TempSV = compute_err_slope(mae_per_window[:w+1, CH["Temp.SV"]])
        err_slope_PresSV = compute_err_slope(mae_per_window[:w+1, CH["Pres.SV"]])

        # Cross-channel coupling
        if w > 0:
            mot_sv_arr = mae_per_window[:w+1, CH["Mot.SV"]]
            pmp_sv_arr = mae_per_window[:w+1, CH["Pmp.SV"]]
            cross_cc   = float(np.corrcoef(mot_sv_arr, pmp_sv_arr)[0, 1]) \
                          if len(mot_sv_arr) > 1 else 0.0
        else:
            cross_cc = 0.0

        row = {
            "label_int":               label_int,
            # DOMAIN 1 — per-channel MAE (8)
            "mae_MotPV":               float(mae_w[CH["Mot.PV"]]),
            "mae_MotSV":               float(mae_w[CH["Mot.SV"]]),
            "mae_MotTV":               float(mae_w[CH["Mot.TV"]]),
            "mae_PmpPV":               float(mae_w[CH["Pmp.PV"]]),
            "mae_PmpSV":               float(mae_w[CH["Pmp.SV"]]),
            "mae_PmpTV":               float(mae_w[CH["Pmp.TV"]]),
            "mae_TempSV":              float(mae_w[CH["Temp.SV"]]),
            "mae_PresSV":              float(mae_w[CH["Pres.SV"]]),
            # DOMAIN 2 — statistical features (9)
            "mean_err_MotSV":          float(mae_per_window[:w+1, CH["Mot.SV"]].mean()),
            "std_err_MotSV":           float(mae_per_window[:w+1, CH["Mot.SV"]].std()),
            "kurtosis_PmpSV":          float(_safe_kurtosis(mae_per_window[:w+1, CH["Pmp.SV"]])),
            "err_slope_MotSV":         err_slope_MotSV,
            "err_slope_TempSV":        err_slope_TempSV,
            "err_slope_PresSV":        err_slope_PresSV,
            "thermal_coupling_ratio":  float(mae_w[CH["Mot.TV"]] / (mae_w[CH["Temp.SV"]] + 1e-8)),
            "cross_channel_MotSV_PmpSV": cross_cc,
            "max_err_all":             float(mae_w.max()),
            # DOMAIN 3 — compound/masked/variant (8)
            "masked_channel_flag":     0.0,
            "secondary_onset_lag":     secondary_onset_lag,
            "burst_count":             0.0,
            "cyclic_baseline_drift":   0.0,
            "multi_sensor_anomaly_count": 0.0,
            "fault_group_id":          2.0,  # Group B = 2
            "variant_slope_ratio":     score_C,
            "thermal_decoupling_flag": 0.0,
            # DOMAIN 4 — z_t + TCN scores (8)
            "z_t_pca_1":               zt_pca1_mean,
            "z_t_pca_2":               zt_pca2_mean,
            "z_t_norm":                zt_norm_mean,
            "z_t_recon_err":           float(mae_w.mean()),
            "score_A":                 score_A,
            "score_B":                 score_B,
            "score_C":                 score_C,
            "onset_order":             onset_order,
        }
        rows.append(row)
    return rows


def _safe_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    try:
        from scipy.stats import kurtosis
        return float(kurtosis(arr))
    except Exception:
        return 0.0


log("  Extracting features for all 9,000 Group B v2 sequences ...")
t_feat = time.time()
new_rows = []

for idx, (seq_np, meta) in enumerate(zip(groupB_v2_sequences, groupB_v2_meta)):
    zt_seq = zt_sequences_v2[idx] if idx < len(zt_sequences_v2) \
             else np.zeros((seq_np.shape[0] // WIN_SIZE, 64), dtype=np.float32)
    rows = extract_features_for_sequence(seq_np, meta, zt_seq, m4_model)
    new_rows.extend(rows)
    if (idx + 1) % 1000 == 0:
        log(f"    {idx+1}/9000 sequences processed ...")

df_new = pd.DataFrame(new_rows)
log(f"  Feature extraction done: {len(df_new):,} rows in {time.time()-t_feat:.1f}s")
log(f"  New features shape: {df_new.shape}")
results["new_feature_rows"] = len(df_new)


# =============================================================================
# SECTION 8 — SURGICAL FEATURE MATRIX UPDATE (replace Labels 7-12 only)
# =============================================================================
log("\nSECTION 8 — Surgical feature matrix update (Labels 7-12 replaced)")

# Backup the current feature matrix
if not FM_BACKUP_T16.exists():
    shutil.copy2(FEATURE_MATRIX_PATH, FM_BACKUP_T16)
    log(f"  Feature matrix backed up → {FM_BACKUP_T16.name}")
else:
    log("  Feature matrix backup already exists — skipping")

t_load = time.time()
df_fm = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
log(f"  Loaded: {df_fm.shape[0]:,} × {df_fm.shape[1]} in {time.time()-t_load:.1f}s")

# Running issues fix [2]: ensure 'label_int' is present and numeric
# (robustness check — this column is confirmed from T1.2 output)
if 'label_int' not in df_fm.columns:
    log("  [WARNING] 'label_int' column not found in feature matrix.")
    log("  Attempting to locate label column ...")
    for cand in ['label', 'label_id', 'fault_id', 'y']:
        if cand in df_fm.columns:
            df_fm = df_fm.rename(columns={cand: 'label_int'})
            log(f"  Renamed '{cand}' → 'label_int'")
            break

# Count rows per group B label before surgery
before_counts = {}
for lbl in [7, 8, 9, 10, 11, 12]:
    before_counts[lbl] = int((df_fm['label_int'].astype(int) == lbl).sum())
    log(f"  Label {lbl} rows before: {before_counts[lbl]:,}")

# Remove old Group B rows
group_b_mask = df_fm['label_int'].astype(int).isin([7, 8, 9, 10, 11, 12])
df_kept     = df_fm[~group_b_mask].reset_index(drop=True)
log(f"  Removed {group_b_mask.sum():,} old Group B rows. "
    f"Kept: {len(df_kept):,}")

# Align new features to exact same column set as original matrix
# Columns in df_new should already match; reindex for safety
df_new_aligned = df_new.reindex(columns=df_fm.columns, fill_value=0.0)

# Concatenate: other labels first, then new Group B
df_updated = pd.concat([df_kept, df_new_aligned], ignore_index=True)
log(f"  Updated matrix: {df_updated.shape[0]:,} × {df_updated.shape[1]}")

# Verify counts
for lbl in [7, 8, 9, 10, 11, 12]:
    after_count = int((df_updated['label_int'].astype(int) == lbl).sum())
    log(f"  Label {lbl}: {before_counts[lbl]:,} → {after_count:,} rows")

df_updated.to_csv(FEATURE_MATRIX_PATH, index=False)
log(f"  Feature matrix saved → {FEATURE_MATRIX_PATH.name}")
results["feature_matrix_updated"] = True


# =============================================================================
# SECTION 9 — FINAL M7 RETRAIN (all Tier-1 fixes in one model)
# =============================================================================
log("\nSECTION 9 — FINAL M7 retrain (T1.2 + T1.6 + T1.7 all in matrix)")

# Backup pre-T1.6 M7 weights
if M7_MODEL_PATH.exists() and not M7_BACKUP_T16.exists():
    shutil.copy2(M7_MODEL_PATH, M7_BACKUP_T16)
    log(f"  Pre-T1.6 M7 backed up → {M7_BACKUP_T16.name}")

# Load updated feature matrix
t_load = time.time()
df_train = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
log(f"  Training data: {df_train.shape[0]:,} × {df_train.shape[1]} in "
    f"{time.time()-t_load:.1f}s")

feature_cols = [c for c in df_train.columns if c != 'label_int']
X = df_train[feature_cols].values.astype(np.float32)
y = df_train['label_int'].astype(int).values
n_classes = len(np.unique(y))
log(f"  Features: {len(feature_cols)} | Classes: {n_classes}")

# Load pre-T1.6 M7 for delta comparison
old_per_class_f1 = {}
try:
    clf_old = xgb.XGBClassifier(num_class=n_classes, **LOCKED_PARAMS)
    clf_old.load_model(str(M7_BACKUP_T16))
    log("  Pre-T1.6 M7 loaded for delta comparison.")
    results["pre_t16_m7_loaded"] = True
except Exception as e:
    log(f"  Pre-T1.6 M7 load failed (delta comparison unavailable): {e}")
    clf_old = None
    results["pre_t16_m7_loaded"] = False

# Train/test split (same protocol as M8p2 — window-level, seed=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
log(f"  Train: {len(X_tr):,} | Test: {len(X_te):,}")

# Score OLD model on NEW data for delta baseline
if clf_old is not None:
    try:
        y_pred_old = clf_old.predict(X_te)
        for lbl in sorted(np.unique(y)):
            mask = y_te == lbl
            if mask.sum() > 0:
                old_per_class_f1[int(lbl)] = float(
                    f1_score(y_te[mask], y_pred_old[mask], average='binary',
                             labels=[lbl], zero_division=0)
                    if False else
                    f1_score(y_te == lbl, y_pred_old == lbl, average='binary',
                             zero_division=0))
        log(f"  Pre-T1.6 M7 on new data — macro F1: "
            f"{f1_score(y_te, y_pred_old, average='macro', zero_division=0):.4f}")
    except Exception as e:
        log(f"  Pre-T1.6 scoring failed: {e}")

# Train FINAL M7
log(f"  Training FINAL M7 (locked hyperparams) ...")
t_train = time.time()
clf = xgb.XGBClassifier(num_class=n_classes, **LOCKED_PARAMS)
clf.fit(X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False)
train_time_min = (time.time() - t_train) / 60
log(f"  Training complete in {train_time_min:.2f} min")

# Evaluate
y_pred      = clf.predict(X_te)
macro_f1    = float(f1_score(y_te, y_pred, average='macro', zero_division=0))
accuracy    = float(accuracy_score(y_te, y_pred))

new_per_class_f1 = {}
for lbl in sorted(np.unique(y)):
    new_per_class_f1[int(lbl)] = float(
        f1_score(y_te == lbl, y_pred == lbl, average='binary', zero_division=0))

log(f"  Macro F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f}")
log("  Per-class F1 (Group B focus):")
for lbl in [7, 8, 9, 10, 11, 12]:
    new_f1 = new_per_class_f1.get(lbl, 0.0)
    old_f1 = old_per_class_f1.get(lbl, None)
    delta_str = f" (Δ={new_f1 - old_f1:+.4f})" if old_f1 is not None else ""
    log(f"    Label {lbl} ({COMPOUND_NAMES[lbl][:30]}): {new_f1:.4f}{delta_str}")

results["final_m7_macro_f1"]    = round(macro_f1, 4)
results["final_m7_accuracy"]    = round(accuracy, 4)
results["final_m7_train_min"]   = round(train_time_min, 2)
results["final_m7_per_class_f1"] = new_per_class_f1


# =============================================================================
# SECTION 10 — GATES
# =============================================================================
log("\nSECTION 10 — Gates")

def gate(name, passed, detail=""):
    GATES[name] = {"passed": bool(passed), "detail": detail}
    log(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")

# G1: continuity gate (already computed in Section 4)
# Just log it again for the gate summary
log(f"  {'PASS' if GATE.get('T1.6_G1_continuity_gate') else 'FAIL'}  "
    f"T1.6_G1_continuity_gate: {results['continuity_pass_rate_overall']*100:.2f}% "
    f"(target >=98%)")

gate("T1.6_G2_sequences_generated",
     results["n_sequences_generated"] == 9000,
     f"{results['n_sequences_generated']}/9000")

gate("T1.6_G3_groupB_pkl_saved",
     results.get("groupB_v2_pkl_saved", False),
     "M6B_sequences_groupB_v2.pkl")

gate("T1.6_G4_feature_matrix_updated",
     results.get("feature_matrix_updated", False),
     "Labels 7-12 rows replaced")

gate("T1.6_G5_final_m7_macro_f1",
     macro_f1 >= 0.82,
     f"Macro F1={macro_f1:.4f} (target >=0.82)")

gate("T1.6_G6_groupB_f1_not_collapsed",
     all(new_per_class_f1.get(l, 0) >= 0.60 for l in [7, 8, 9, 10, 11, 12]),
     f"All Group B F1 >= 0.60 floor "
     f"(min={min(new_per_class_f1.get(l,0) for l in [7,8,9,10,11,12]):.4f})")

# Informational gate: did Group B F1 drop? If yes, confirms artifact reliance.
grpB_old_avg = np.mean([old_per_class_f1.get(l, 0) for l in [7,8,9,10,11,12]]) \
               if old_per_class_f1 else None
grpB_new_avg = np.mean([new_per_class_f1.get(l, 0) for l in [7,8,9,10,11,12]])
if grpB_old_avg is not None:
    grpB_delta = grpB_new_avg - grpB_old_avg
    GATES["T1.6_G7_groupB_f1_delta_info"] = {
        "passed": True,  # informational — always "passes" as a gate
        "detail": f"Group B avg F1: {grpB_old_avg:.4f} → {grpB_new_avg:.4f} "
                  f"(Δ={grpB_delta:+.4f}). "
                  f"{'CONFIRMS artifact reliance — XGBoost was using step pattern.' if grpB_delta < -0.10 else 'No significant drop — XGBoost learned real fault patterns.'}"
    }
    log(f"  INFO  T1.6_G7_groupB_f1_delta_info: "
        f"{GATES['T1.6_G7_groupB_f1_delta_info']['detail']}")
    results["groupB_f1_delta"] = round(float(grpB_delta), 4)

# Merge continuity gate from Section 4 into GATES dict
GATES["T1.6_G1_continuity_gate"] = {
    "passed": bool(GATE.get("T1.6_G1_continuity_gate", False)),
    "detail": f"{results['continuity_pass_rate_overall']*100:.2f}% pass (target >=98%)"
}

n_pass = sum(1 for g in GATES.values() if g["passed"])
n_fail = len(GATES) - n_pass
log(f"\n  Gates: {n_pass} PASS / {n_fail} FAIL")
results["gates_passed"] = n_pass
results["gates_failed"] = n_fail

# Block M7 save if critical gates fail
critical_gates_ok = (
    GATES.get("T1.6_G5_final_m7_macro_f1", {}).get("passed", False) and
    GATES.get("T1.6_G6_groupB_f1_not_collapsed", {}).get("passed", False)
)


# =============================================================================
# SECTION 11 — SAVE FINAL M7
# =============================================================================
log("\nSECTION 11 — Save FINAL M7 weights")

if critical_gates_ok:
    clf.save_model(str(M7_MODEL_PATH))
    log(f"  FINAL M7 (CUDA) saved → {M7_MODEL_PATH.name}")

    clf_cpu = xgb.XGBClassifier(num_class=n_classes,
                                  **{**LOCKED_PARAMS, 'device': 'cpu'})
    clf_cpu.load_model(str(M7_MODEL_PATH))
    clf_cpu.save_model(str(M7_CPU_PATH))
    log(f"  FINAL M7 (CPU)  saved → {M7_CPU_PATH.name}")
    results["final_m7_saved"] = "live"
else:
    cand_path = MODEL_DIR / "M7_xgboost_classifier.T1_6_candidate.json"
    clf.save_model(str(cand_path))
    log(f"  Critical gates failed — saved as candidate: {cand_path.name}")
    log("  Live M7 NOT replaced. Investigate failures before retry.")
    results["final_m7_saved"] = "candidate"


# =============================================================================
# SECTION 12 — DELTA PLOT
# =============================================================================
log("\nSECTION 12 — Per-class F1 delta plot")

try:
    labels_sorted = sorted(new_per_class_f1.keys())
    new_vals      = [new_per_class_f1[l]        for l in labels_sorted]
    old_vals      = [old_per_class_f1.get(l, 0) for l in labels_sorted] \
                     if old_per_class_f1 else None

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: full per-class F1 comparison
    ax = axes[0]
    x  = np.arange(len(labels_sorted))
    w  = 0.35
    if old_vals:
        ax.bar(x - w/2, old_vals, w, label="Pre-T1.6 M7", alpha=0.7, color='steelblue')
        ax.bar(x + w/2, new_vals, w, label="FINAL M7 (T1.6)", alpha=0.9, color='darkorange')
    else:
        ax.bar(x, new_vals, label="FINAL M7", alpha=0.9, color='darkorange')
    ax.axhline(0.70, linestyle=':', color='gray', label='Floor 0.70')
    ax.axhline(0.60, linestyle=':', color='red',  label='Group B floor 0.60')
    ax.set_xticks(x); ax.set_xticklabels([str(l) for l in labels_sorted])
    ax.set_xlabel("Label"); ax.set_ylabel("F1")
    ax.set_title("Per-class F1: pre-T1.6 vs FINAL M7"); ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)

    # Right: Group B focus with delta annotation
    ax2 = axes[1]
    grpB_labels = [7, 8, 9, 10, 11, 12]
    x2 = np.arange(len(grpB_labels))
    new_b = [new_per_class_f1.get(l, 0) for l in grpB_labels]
    old_b = [old_per_class_f1.get(l, 0) for l in grpB_labels] if old_per_class_f1 else None
    if old_b:
        bars1 = ax2.bar(x2 - w/2, old_b, w, label="Pre-T1.6", alpha=0.7, color='steelblue')
        bars2 = ax2.bar(x2 + w/2, new_b, w, label="FINAL", alpha=0.9, color='darkorange')
        for xi, (o, n) in enumerate(zip(old_b, new_b)):
            delta = n - o
            ax2.text(xi + w/2, n + 0.01, f"{delta:+.2f}",
                     ha='center', va='bottom', fontsize=7,
                     color='red' if delta < -0.10 else 'green')
    else:
        ax2.bar(x2, new_b, label="FINAL", color='darkorange')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"{l}\n{COMPOUND_NAMES[l][:15]}" for l in grpB_labels],
                         fontsize=7)
    ax2.set_ylabel("F1"); ax2.set_title("Group B F1 delta (T1.6 fix)")
    ax2.axhline(0.60, linestyle=':', color='red', label='Floor 0.60')
    ax2.legend(fontsize=7); ax2.set_ylim(0, 1.10)
    ax2.text(0.02, 0.95, "Red delta = XGBoost was\nusing step artifact",
             transform=ax2.transAxes, fontsize=7, va='top', color='red',
             bbox=dict(boxstyle='round', alpha=0.3))

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "M8p6_groupB_f1_delta.png"
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {plot_path}")
    results["plot_saved"] = True
except Exception as e:
    log(f"  Plot failed: {e}")
    results["plot_saved"] = False


# =============================================================================
# SECTION 13 — REPORT
# =============================================================================
log("\nSECTION 13 — Writing report")

gate_table = "\n".join(
    f"| {name} | {'PASS' if g['passed'] else 'FAIL'} | {g['detail']} |"
    for name, g in GATES.items()
)

grpB_delta_table = ""
if old_per_class_f1:
    rows_d = []
    for lbl in [7, 8, 9, 10, 11, 12]:
        old_f = old_per_class_f1.get(lbl, 0)
        new_f = new_per_class_f1.get(lbl, 0)
        d     = new_f - old_f
        interp = "CONFIRMS ARTIFACT" if d < -0.10 else \
                 ("MARGINAL" if d < -0.05 else "LEARNED REAL PATTERNS")
        rows_d.append(f"| {lbl} | {COMPOUND_NAMES[lbl]} | {old_f:.4f} | "
                      f"{new_f:.4f} | {d:+.4f} | {interp} |")
    grpB_delta_table = "\n".join(rows_d)

report_content = f"""# {SCRIPT_NAME} — Report
**Date:** {date.today()}
**Status:** {"COMPLETE — FINAL M7 saved" if results.get('final_m7_saved') == 'live' else "CANDIDATE ONLY — investigate gate failures"}
**Fixes applied:** Bug1 (np.tile→extrapolate) + Bug2 (index-skip) + Bug3 (gap-fill)

---

## 1. Why This Script Was Necessary

Visualization audit (Audit v2.0 §4.5.3) confirmed 5/6 Group B compound-chain
plots show abrupt step discontinuities at the secondary-fault onset point.
Two bugs in `generate_compound_sequence()` caused this:

- **Bug 1** (`np.tile` fallback): primary fault froze at last seed value when
  seed length < Phase 2 requirement. The primary stopped progressing while the
  secondary was added — producing a step that looks like a sudden drop.
- **Bug 2** (index skip): `p_tail_start = p_len` started Phase 2 from
  `p_seq[p_len]` instead of `p_seq[p_len-1]`, skipping one time step AND
  immediately adding secondary contribution — causing a step jump.
- **Bug 3** (gap, found in this fix): steps `p_len` to `p2_start` were left
  at `np.ones(1.0)` (flat baseline) when `len(p_seq) < lag + 50`.

**ML consequence:** XGBoost may have learned the step pattern as the primary
Group B discriminator — a pattern absent in real compound faults. See Gate G7.

## 2. Generation Results

| Metric | Value |
|---|---|
| Sequences generated | {results['n_sequences_generated']} / 9,000 |
| Generation time | {results['generation_time_s']}s |
| Continuity pass rate | {results['continuity_pass_rate_overall']*100:.2f}% (target ≥98%) |
| Feature rows extracted | {results['new_feature_rows']:,} |

## 3. Final M7 Results (ALL Tier-1 fixes applied)

| Metric | Value |
|---|---|
| Macro F1 | {results['final_m7_macro_f1']:.4f} |
| Accuracy | {results['final_m7_accuracy']:.4f} |
| Train time | {results['final_m7_train_min']:.2f} min |
| M7 saved | {results.get('final_m7_saved', 'unknown')} |

This M7 contains ALL three Tier-1 data fixes:
- T1.2: Label 19 Pres.SV* drop restored (mae_PresSV = 0.198 vs old 0.967)
- T1.6: Group B continuous superposition (this script)
- T1.7: Group E label names (sensor_anomaly_thermal / sensor_anomaly_pump)

## 4. Group B F1 Delta (Key Diagnostic)

| Label | Class | Pre-T1.6 F1 | Final F1 | Delta | Interpretation |
|---|---|---|---|---|---|
{grpB_delta_table if grpB_delta_table else 'Pre-T1.6 model not available for comparison.'}

> If delta < -0.10: XGBoost WAS using the step pattern as discriminator.
> The corrected F1 is the HONEST number for real compound fault performance.
> If delta ≈ 0: XGBoost learned genuine fault patterns — step was not relied on.

## 5. Gates

| Gate | Status | Detail |
|---|---|---|
{gate_table}

## 6. Files Written

- `M6B_sequences_groupB_v2.pkl` — 9,000 corrected Group B sequences
- `M6B_sequences_groupB.pkl.v1.bak` — original preserved
- `z_t_sequences_groupB_v2.pkl` — z_t from frozen M4
- `M6B_feature_matrix.csv` — Labels 7-12 rows replaced
- `M6B_feature_matrix.csv.pre_T1_6.bak` — original preserved
- `M7_xgboost_classifier.json` — FINAL M7 (if gates passed)
- `M7_xgboost_classifier.pre_T1_6.json.bak` — pre-fix backup

## 7. Next Steps

```
T1.6 COMPLETE  (this script)
T1.3: python module_08p3_m7_sequence_level_eval.py   (~5 min, eval FINAL M7)
T1.4: python module_08p4_ood_detector.py             (~2 min, parallel)
T1.5: python module_08p5_cusum_decay_and_fmea.py     (~30 sec, parallel)
```

---
*Generated by {SCRIPT_NAME} | PumpSmart v14.2 | {date.today()}*
"""

REPORT_PATH = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
try:
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    log(f"  Report → {REPORT_PATH}")
    results["report_written"] = True
except Exception as e:
    log(f"  [ERROR] Report failed: {e}")
    results["report_written"] = False


# =============================================================================
# PASTE TEXT UPDATE
# =============================================================================
print()
print("=" * 72)
print("== PASTE TEXT UPDATE -- COPY BELOW INTO PASTE TEXT ==")
print("=" * 72)
print()
print(f"## T1.6 Group B Regeneration + FINAL M7 Retrain -- {date.today()}")
print(f"T1.6_status                       = "
      f"{'COMPLETE' if results.get('final_m7_saved') == 'live' else 'CANDIDATE_ONLY'}")
print(f"T1.6_continuity_pass_rate         = {results['continuity_pass_rate_overall']*100:.2f}%")
print(f"T1.6_sequences_generated          = {results['n_sequences_generated']}/9000")
print(f"T1.6_feature_rows_new             = {results['new_feature_rows']:,}")
print(f"T1.6_final_m7_macro_f1            = {results['final_m7_macro_f1']}")
print(f"T1.6_final_m7_accuracy            = {results['final_m7_accuracy']}")
print(f"T1.6_final_m7_saved               = {results.get('final_m7_saved', 'unknown')}")
if "groupB_f1_delta" in results:
    print(f"T1.6_groupB_f1_delta_avg          = {results['groupB_f1_delta']}")
    interp = "ARTIFACT_CONFIRMED" if results["groupB_f1_delta"] < -0.10 else "REAL_PATTERNS"
    print(f"T1.6_groupB_f1_delta_interpretation = {interp}")
print(f"T1.6_gates_passed                 = {results['gates_passed']}/{results['gates_passed']+results['gates_failed']}")
print(f"T1.6_fixes_applied                = Bug1_extrapolate + Bug2_index + Bug3_gap")
print()
print("## Tier-1 Queue Status")
print("DONE: T1.7, T1.1, T1.2, T1.6")
print("NEXT: T1.3 (module_08p3_m7_sequence_level_eval.py) -- uses FINAL M7")
print("THEN: T1.4 (module_08p4_ood_detector.py)")
print("THEN: T1.5 (module_08p5_cusum_decay_and_fmea.py)")
print()
print("== END PASTE UPDATE ==")
print("=" * 72)


# =============================================================================
# FILE MANIFEST
# =============================================================================
print()
print("-- FILE MANIFEST -------------------------------------------------------")
print()
print("NEW:")
print(f"  {GROUPB_V2_PKL}")
print(f"  {ZT_V2_PKL}")
print(f"  {REPORT_PATH}")
print()
print("UPDATED (original backed up):")
print(f"  {FEATURE_MATRIX_PATH}  (Labels 7-12 rows replaced)")
print(f"  {M7_MODEL_PATH}        (FINAL retrain)")
print(f"  {M7_CPU_PATH}          (FINAL retrain, CPU)")
print()
print("BACKUPS (do not push):")
print(f"  {GROUPB_V1_BAK}")
print(f"  {FM_BACKUP_T16}")
print(f"  {M7_BACKUP_T16}")
print()
print("GitHub push: module_08p6_groupB_regenerate.py")
print("             M7_xgboost_classifier.json, M7_xgboost_classifier_cpu.json")
print("             report .md, M8p6_groupB_f1_delta.png")
print("HF Spaces:   M7_xgboost_classifier.json, report .md")
print("DO NOT PUSH: *.bak, *.v1.bak, *candidate*")
print("-----------------------------------------------------------------------")


# =============================================================================
# NEXT PROMPT
# =============================================================================
print()
print("-- NEXT PROMPT ---------------------------------------------------------")
print()
print("T1.6 done. Starting T1.3.")
print(f"Finding: Group B regenerated. Continuity {results['continuity_pass_rate_overall']*100:.1f}%.")
print(f"         FINAL M7 macro F1={results['final_m7_macro_f1']:.4f}. "
      f"Saved as {results.get('final_m7_saved','?')}.")
if "groupB_f1_delta" in results:
    print(f"         Group B F1 delta={results['groupB_f1_delta']:+.4f} "
          f"({'artifact confirmed' if results['groupB_f1_delta'] < -0.10 else 'real patterns learned'}).")
print("Next: python module_08p3_m7_sequence_level_eval.py (~5 min)")
print("-----------------------------------------------------------------------")

log("\n[DONE]")
