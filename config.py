# config.py
# PumpSmart Project — Central Configuration
# Version: v14.0 | Updated: 2026-04-16
# Pump: 110 kW, 7-stage, 40 bar, 2980 RPM, 45 m³/h, 450 m head — CIRA SACIP
# Architecture: v14.0 — M6B 22-class, 4-layer detection, CUSUM + Rolling Baseline
# GitHub is ONLY source of truth. Spaces .md files are OUTDATED.

import torch
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# HARDWARE
# ══════════════════════════════════════════════════════════════════
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_GPU  = torch.cuda.is_available()
# Lenovo LOQ 15APH8 | RTX 4060 Laptop 8GB VRAM | CUDA 12.6 | PyTorch 2.6.0+cu126
# AMD Ryzen 7840HS | 16GB DDR5 RAM
# Rules: .to(DEVICE) | pin_memory=True | num_workers=4 | NEVER hardcode .cuda()

# ══════════════════════════════════════════════════════════════════
# DIRECTORY STRUCTURE
# ══════════════════════════════════════════════════════════════════
ROOT_DIR   = Path(__file__).resolve().parent
RAW_DIR    = ROOT_DIR / "data" / "raw"
CLEAN_DIR  = ROOT_DIR / "data" / "clean"
NORM_DIR   = ROOT_DIR / "data" / "normalized"
SYNTH_DIR  = ROOT_DIR / "data" / "synthetic"
MODEL_DIR  = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR  = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"
SRC_DIR    = ROOT_DIR / "src"
APP_DIR    = ROOT_DIR / "app"

for _dir in [RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
             MODEL_DIR, PLOTS_DIR, REPORT_DIR, SRC_DIR, APP_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# PUMP NAMEPLATE — 110 kW INDUSTRIAL MULTISTAGE CENTRIFUGAL
# LOCKED — DO NOT CHANGE — validated against CIRA SACIP dataset
# ══════════════════════════════════════════════════════════════════
MOTOR_KW        = 110          # IEC Frame 315mm, 400V, 2-pole  ← WAS WRONG (10) NOW FIXED
MOTOR_RPM       = 2980         # 50Hz, 2-pole synchronous speed
PUMP_STAGES     = 7            # 7 impellers (multistage centrifugal)
PUMP_FLOW_M3H   = 45           # rated flow, m³/h
PUMP_HEAD_M     = 450          # total head, m
PUMP_MAX_BAR    = 40           # max discharge pressure, bar
PUMP_HYD_KW     = 55.2         # P_hyd = ρgQH/η = 1000×9.81×(45/3600)×450/0.65
PUMP_EFFICIENCY = 0.65         # hydraulic efficiency (multistage centrifugal default)
PUMP_BPF_HZ     = 347.67       # blade pass frequency = (RPM/60) × stages = 49.67 × 7
PUMP_NPSH_R     = 5.71         # NPSHr at nameplate operating point, m
PUMP_SPECIFIC_SPEED = 10.2     # Ns = N×Q^0.5/H^0.75 → radial flow confirmed
REPLACEMENT_COST_INR = 5_000_000  # >₹50 lakh — capital asset

# NOTE: "10 kW" in original Zenodo source refers to a sub-duty point.
# USE MOTOR_KW = 110 FOR ALL PHYSICS CALCULATIONS IN M5+.
# Reference: completed_modules_context_and_M1_to_M4.md (Part 1 — LOCKED)

# ══════════════════════════════════════════════════════════════════
# DATASET — CIRA SACIP
# ══════════════════════════════════════════════════════════════════
DATASET_URL     = "https://zenodo.org/records/15301820"
DATASET_FILES   = 9            # 3 pumps × 3 operational days
DATASET_PUMPS   = ["A", "B", "C"]
SENSOR_COLUMNS  = [            # 8 ML-input channels (Barometer dropped, Temperature = T_ambient)
    "X_ACR_Mot.PV",            # Motor casing vibrational velocity, mm/s
    "X_ACR_Mot.SV",            # Motor casing broadband peak acceleration envelope, mm/s²
    "X_ACR_Mot.TV",            # Motor casing accelerometer contact temperature, °C
    "X_ACR_Pmp.PV",            # Pump casing vibrational velocity, mm/s
    "X_ACR_Pmp.SV",            # Pump casing broadband peak acceleration envelope, mm/s²
    "X_ACR_Pmp.TV",            # Pump casing accelerometer contact temperature, °C
    "X_Temp.SV",               # Motor casing surface temperature, °C
    "X_Pres.SV",               # Pump discharge pressure, bar
]
N_SENSORS       = 8
SAMPLING_HZ     = 1            # 1 second uniform sampling, all 9 files confirmed
AMBIENT_COL     = "Temperature"  # per-row ambient temperature (NOT hardcoded 20°C)
DROP_COL        = "Barometer"    # dropped globally before M3

# ══════════════════════════════════════════════════════════════════
# M1 — DATA CLEANING (LOCKED RESULTS)
# ══════════════════════════════════════════════════════════════════
M1_RAW_ROWS     = 173_730
M1_CLEAN_ROWS   = 147_217
M1_DROP_PCT     = 15.26
M1_TOTAL_SEGMENTS   = 66
M1_USABLE_SEGMENTS  = 25       # after Pump_C_Day3 exclusion (>50% nulls)
GAP_MULTIPLIER      = 2.0      # gap > 2× median sampling interval = new segment boundary

# ══════════════════════════════════════════════════════════════════
# M2 — EDA + CLUSTERING (LOCKED RESULTS)
# ══════════════════════════════════════════════════════════════════
M2_USABLE_ROWS      = 117_970
M2_OPTIMAL_K        = 4
M2_SILHOUETTE       = 0.5458
M2_PCA_PC1          = 47.37    # % variance
M2_PCA_PC2          = 32.97    # % variance
M2_TOP_CORR_R       = 0.9793   # Mot.TV ↔ Temp.SV — dominant coupling
WINDOW_SIZE         = 50       # LOCKED in M4 (overrides M2 default of 60)
WINDOW_STEP         = 10       # stride between windows

# Operating mode cluster labels
CLUSTER_NAMES = {
    0: "cooldown",
    1: "steady_state",
    2: "startup",
    3: "high_load",
}
CLUSTER_ROWS = {
    "startup":      49_884,    # 42.3%
    "cooldown":     26_851,    # 22.8%
    "high_load":    26_600,    # 22.5%
    "steady_state": 14_635,    # 12.4%
}

# ══════════════════════════════════════════════════════════════════
# M3 — NORMALIZATION (LOCKED — never change formulas)
# ══════════════════════════════════════════════════════════════════
# P*  = P_actual / P_cluster_mean
# a*  = a_actual / a_cluster_mean
# ΔT* = (T_row - T_ambient_row) / (T_cluster_max - T_ambient_row)
# T_ambient_row = per-row Temperature column (climate-agnostic)
# Small negatives in temperature (flash evaporative cooling): PRESERVED, not clipped
# Normal operation → [0, 1.0] | Fault → drift above 1.0
M3_CONFIG_FILE      = MODEL_DIR / "M3_normalization_config.json"   # LOCKED
M3_ROWS_NORMALIZED  = 117_970
M3_CHANNELS         = 8
M3_CLUSTERS         = 4

# ══════════════════════════════════════════════════════════════════
# M4 — LSTM-AE BASELINE v8 (LOCKED — all values from trained model)
# ══════════════════════════════════════════════════════════════════

# --- Architecture (matches trained lstm_ae_baseline_best.pth) ---
LSTM_INPUT_SIZE     = 8        # 8 normalized sensor channels
LSTM_HIDDEN         = 128      # encoder: 8→128 (bottleneck is 64)
LSTM_BOTTLENECK     = 64       # encoder bottleneck dimension
LSTM_LAYERS         = 2        # stacked LSTM layers
LSTM_DROPOUT        = 0.3      # ← WAS 0.2 (WRONG), NOW FIXED to match M4 v8
BATCH_SIZE          = 32
LEARNING_RATE       = 1e-3
EPOCHS              = 150      # ← WAS 50 (WRONG), NOW FIXED — M4 ran 150 epochs
BEST_EPOCH          = 141      # actual best epoch from M4 training

# --- Loss function weights ---
LOSS_MAE_WEIGHT     = 0.6
LOSS_MSE_WEIGHT     = 0.4

# --- M4 channel weights (used in weighted MAE loss) ---
# Source: completed_modules_context_and_M1_to_M4.md
CHANNEL_WEIGHTS = {
    "X_ACR_Mot.SV":  2.0,      # primary mechanical fault channel
    "X_ACR_Pmp.SV":  2.0,      # primary mechanical fault channel
    "X_Pres.SV":     2.0,      # primary hydraulic fault channel
    "X_ACR_Mot.PV":  1.5,      # displacement — correlated with SV
    "X_ACR_Pmp.PV":  1.5,      # displacement — correlated with SV
    "X_Temp.SV":     1.0,      # thermal — secondary signal
    "X_ACR_Mot.TV":  0.8,      # accelerometer temp — placement-dependent
    "X_ACR_Pmp.TV":  0.8,      # accelerometer temp — placement-dependent
}
# NOTE: Low weight on Temp.SV (1.0) means overloading has LOW weighted MAE
# → M8 Mech C (Temp.SV Spearman drift) is PRIMARY for overloading detection

# --- M8 v2 updated channel weights (Fisher rank validated) ---
# Source: M6.5 Finding 4 — Pmp_SV_mean rank 1, TV channels lowest
M8_CHANNEL_WEIGHTS = {
    "X_ACR_Mot.SV":  2.5,      # increased from 2.0 (Fisher rank 2)
    "X_ACR_Pmp.SV":  2.5,      # increased from 2.0 (Fisher rank 1)
    "X_Pres.SV":     2.5,      # increased from 2.0 (hydraulic faults)
    "X_ACR_Mot.PV":  2.0,      # increased from 1.5
    "X_ACR_Pmp.PV":  2.0,      # increased from 1.5
    "X_Temp.SV":     0.5,      # decreased — Mech C monitors UNWEIGHTED
    "X_ACR_Mot.TV":  0.3,      # decreased — thermal lag secondary
    "X_ACR_Pmp.TV":  0.3,      # decreased — thermal lag secondary
}

# --- Locked training results ---
M4_CLEAN_WINDOWS    = 9_711
M4_TRAIN_WINDOWS    = 8_254
M4_VAL_WINDOWS      = 1_457
M4_VAL_LOSS         = 0.026862
M4_MEAN_MAE         = 0.026765
M4_SPIKE_SEEDS      = 1_044    # shape: (1044, 50, 8)
M4_SPIKE_EXCLUDED   = 12_620
M4_FALSE_ALARMS     = 8        # 0.55% FPR on full val set
M4_SEPARATION       = 4.11     # anomaly/normal MAE separation ratio
M4_PARAMS           = 505_096  # total model parameters

# --- LOCKED THRESHOLD — DO NOT MODIFY ---
# Changing this value invalidates ALL M6B data and all downstream gates
M4_THRESHOLD        = 0.110058  # mean + 3σ ∪ P99 on val set — LOCKED FOREVER
M4_THRESHOLD_CONFIG = MODEL_DIR / "M4_threshold_config.json"
M4_SPIKE_CONFIG     = SYNTH_DIR / "M4_spike_config.json"  # winsor bounds — LOCKED

# ══════════════════════════════════════════════════════════════════
# LOCKED PHYSICAL COUPLINGS (M2 confirmed — must hold in ALL synthetic data)
# ══════════════════════════════════════════════════════════════════
COUPLING_MOT_TV_TEMP_SV     = 0.9793   # r — dominant thermal coupling
COUPLING_PMP_PV_PMP_SV      = 0.8882   # r — mechanical vibration coupling
COUPLING_PMP_PV_PRES_SV     = 0.8779   # r — impeller → pressure coupling
COUPLING_MIN_THRESHOLD      = 0.87     # Invariant 7: all synth must exceed this

# Thermal coupling by fault (M5 validated — LOCKED)
THERMAL_COUPLING_BY_FAULT = {
    "bearing_wear_steady_state":    0.972,   # PRESERVED
    "overloading_steady_state":     0.997,   # STRONGLY PRESERVED
    "seal_failure_steady_state":   -0.013,   # BROKEN (hydraulic)
    "cavitation_startup":           0.376,   # WEAK (hydraulic)
    "bearing_wear_high_load":       0.949,   # PRESERVED
    "normal_steady_state":         -0.062,   # baseline
}

# ══════════════════════════════════════════════════════════════════
# FAULT UNIVERSE — v14.0 (22 CLASSES, labels 0–21)
# ══════════════════════════════════════════════════════════════════
N_FAULT_CLASSES     = 22       # labels 0–21, Groups A–E
N_SENSORS_INPUT     = 8        # input to all ML models

FAULT_LABELS = {
    # Group A — Single Source (LOCKED from M6A)
    0:  "normal",
    1:  "bearing_wear",
    2:  "impeller_imbalance",
    3:  "cavitation",
    4:  "seal_failure",
    5:  "overloading",
    6:  "sensor_failure",
    # Group B — Compound Chain
    7:  "bearing_wear__overloading",
    8:  "cavitation__seal_failure",
    9:  "impeller_imbalance__bearing_wear",
    10: "seal_failure__cavitation",
    11: "overloading__bearing_wear",
    12: "impeller_imbalance__cavitation",
    # Group C — Masked Faults
    13: "bearing_wear_MotSV_masked",
    14: "cavitation_PresSV_masked",
    15: "seal_failure_PresSV_drifting",
    16: "overloading_TempSV_stuck",
    17: "impeller_imbalance_PmpSV_flatline",
    # Group D — Severity Variants
    18: "cavitation_intermittent",
    19: "seal_failure_fast",
    20: "overloading_cyclic",
    21: "bearing_wear_gradual",     # NEW v14.0 — Paris law small ΔK
    # Group E — labels confirmed in fault_rules_v3.json (written by M6B Step 3)
    # DO NOT hardcode Group E labels here
}

FAULT_GROUPS = {
    "A": list(range(0, 7)),
    "B": list(range(7, 13)),
    "C": list(range(13, 18)),
    "D": list(range(18, 22)),
    "E": [],   # populated from fault_rules_v3.json at runtime
}

# Target sequence counts per group
M6B_TARGET_SEQUENCES = {
    "group_A": 8_400,    # LOCKED from M6A
    "group_B": 7_200,    # 1,200 × 6 compound chains
    "group_C": 4_000,    # 800 × 5 masked scenarios
    "group_D": 2_800,    # 600×3 variants + 1,000×1 label 21
    "group_E": 800,      # 400 × 2 multi-sensor failures
    "total":   26_000,   # target minimum (~26,000–28,000)
}
M6B_LABEL21_TARGET  = 1_000   # bearing_wear_gradual — higher count (harder class)

# Cluster constraints for fault generation
FAULT_CLUSTER_CONSTRAINTS = {
    "cavitation":    ["startup"],          # ONLY in startup cluster
    "overloading":   ["steady_state"],     # ONLY in steady_state cluster
    "seal_failure":  ["steady_state", "high_load"],
    "bearing_wear":  ["steady_state", "high_load", "cooldown"],
}

# ══════════════════════════════════════════════════════════════════
# M4 WINSORIZATION BOUNDS — cluster-conditional (LOCKED)
# Source: M4_spike_config.json — DO NOT OVERRIDE in M6B or M12
# ══════════════════════════════════════════════════════════════════
WINSOR_MULTIPLIERS = {
    "X_Pres.SV_norm": {
        "startup":      3.0,   # Joukowsky transient headroom
        "steady_state": 5.6,   # wide valid range (std=13 bar)
        "high_load":    2.0,   # tight — faults caught immediately
        "cooldown":     3.0,   # depressurization transients
    },
    "X_ACR_Pmp.PV": {
        "startup":      3.2,   # ISO 13373-3 BPF harmonics
        "steady_state": 2.6,
        "high_load":    2.6,
        "cooldown":     2.6,
    },
    "X_ACR_Mot.SV":  {"all": 6.7},   # uniform — no cluster physics
    "X_ACR_Pmp.SV":  {"all": 8.8},   # uniform — broadband RMS spike
    "X_ACR_Mot.PV":  {"all": 2.2},   # uniform — displacement bounded
}

# ══════════════════════════════════════════════════════════════════
# M4 SPIKE SEED FAULT HINTS (LOCKED for M6B + M12)
# ══════════════════════════════════════════════════════════════════
SPIKE_SEED_HINTS = {
    "mechanical_transient":    472,   # → bearing_wear, impeller_imbalance
    "pressure_transient":      408,   # → cavitation, seal_failure
    "impeller_cavitation":     113,   # → cavitation (direct)
    "bearing_impact":           44,   # → bearing_wear (direct) + label 21
    "pressure_spike_highload":   7,   # → overloading
}

# ══════════════════════════════════════════════════════════════════
# M6.5 FEATURE MATRIX (LOCKED — 8,400 rows × 25 cols from M6A)
# ══════════════════════════════════════════════════════════════════
M65_ROWS            = 8_400
M65_COLS            = 25       # 24 features + label
M65_FEATURE_FILE    = SYNTH_DIR / "M6_feature_matrix.csv"
M65_THRESHOLD_PASS  = M4_THRESHOLD   # 0.110058

# M6.5 Gate 3 pass rates (LOCKED — v2 authoritative, v1 INVALID)
M65_GATE3 = {
    "normal":               86.67,  # probe only — NOT FPR
    "bearing_wear":         13.33,
    "impeller_imbalance":   30.00,
    "cavitation":          100.00,  # MAE=0.675, 6.1× threshold
    "seal_failure":         29.17,  # slow hydraulic — Mech C primary
    "overloading":           0.00,  # thermal dominant — Mech C primary
    "sensor_failure":       93.33,
}

# ══════════════════════════════════════════════════════════════════
# M6.5r FEATURE MATRIX (PENDING — ~196,000 rows × 26 cols from M6B)
# ══════════════════════════════════════════════════════════════════
M65R_TARGET_ROWS    = 196_000  # approximate — depends on M6B sequence count
M65R_COLS           = 26       # 25 features + label  ← M7 input
M65R_FEATURE_FILE   = SYNTH_DIR / "M6B_feature_matrix.csv"
# NOTE: M7 trains on M65R_FEATURE_FILE (26 cols) NOT M65_FEATURE_FILE (25 cols)
# Invariant 12: NEVER feed old M6.5 matrix to M7

# ══════════════════════════════════════════════════════════════════
# M6B FILE PATHS (all PENDING — created when M6B script runs)
# ══════════════════════════════════════════════════════════════════
M6B_SEQUENCES_A     = SYNTH_DIR / "M6B_sequences_groupA.pkl"
M6B_SEQUENCES_B     = SYNTH_DIR / "M6B_sequences_groupB.pkl"
M6B_SEQUENCES_C     = SYNTH_DIR / "M6B_sequences_groupC.pkl"
M6B_SEQUENCES_D     = SYNTH_DIR / "M6B_sequences_groupD.pkl"
M6B_SEQUENCES_E     = SYNTH_DIR / "M6B_sequences_groupE.pkl"
M6B_COMBINED        = SYNTH_DIR / "M6B_combined_sequences.pkl"
M6B_META            = SYNTH_DIR / "M6B_sequence_meta.csv"
FAULT_RULES_V1      = MODEL_DIR / "fault_rules.json"       # LOCKED — M5/M6A, do not overwrite
FAULT_RULES_V3      = MODEL_DIR / "fault_rules_v3.json"    # written by M6B Step 3

# ══════════════════════════════════════════════════════════════════
# M4 / M8 MODEL FILE PATHS
# ══════════════════════════════════════════════════════════════════
M4_MODEL_PATH       = MODEL_DIR / "lstm_ae_baseline_best.pth"
M8_MODEL_PATH       = MODEL_DIR / "lstm_ae_v2_best.pth"
M7_MODEL_PATH       = MODEL_DIR / "xgboost_fault_classifier_cpu.pkl"
M3_NORM_CONFIG      = MODEL_DIR / "M3_normalization_config.json"   # LOCKED
M5_PHYSICS_CONFIG   = MODEL_DIR / "M5_physics_config.json"
UNIT_REGISTRY       = MODEL_DIR / "unit_registry.json"
M8_THRESHOLD_CONFIG = MODEL_DIR / "M8_threshold_config.json"
M8_FUZZY_CONFIG     = MODEL_DIR / "M8_fuzzy_config.json"

# ══════════════════════════════════════════════════════════════════
# M8 — LSTM-AE v2 + 4-LAYER DETECTION ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

# --- Layer 1: LSTM-AE single-window threshold ---
M8_THRESHOLD_BASE   = 0.110058     # same as M4 — LOCKED
# Cluster-conditional thresholds (tuned in M8 — placeholders until M8 runs)
M8_THRESHOLD_STARTUP        = None  # filled after M8 training
M8_THRESHOLD_STEADY_STATE   = 0.110058
M8_THRESHOLD_HIGH_LOAD      = None
M8_THRESHOLD_COOLDOWN       = None

# --- Layer 2: Fuzzy Logic + Rolling Accumulator ---
M8_FUZZY_LOWER      = 0.07         # lower fuzzy bound (WATCH entry)
M8_FUZZY_UPPER_WARN = 0.09         # upper fuzzy bound (WARN entry)
M8_ACCUMULATOR_WATCH    = 2.0      # rolling score → WATCH state
M8_ACCUMULATOR_WARN     = 3.5      # rolling score → WARN state
M8_ACCUMULATOR_FAULT    = 5.0      # rolling score → FAULT/DANGER state
M8_ROLLING_WINDOW       = 30       # windows for rolling mean
M8_ROLLING_WATCH_MAE    = 0.085    # rolling mean MAE → WATCH
M8_ROLLING_WARN_MAE     = 0.095    # rolling mean MAE → WARN
M8_SLOPE_THRESHOLD      = 0.0003   # MAE slope per window → trend alert

# --- Layer 3: CUSUM Runtime State (M10 only — NOT in feature matrix) ---
# Formula: S_n = max(0, S_{n-1} + (mae_channel_n − μ0) − k)
# k = 0.5 × (threshold − μ0) per channel
# Fires when S_n > CUSUM_CONTROL_LIMIT
CUSUM_CHANNELS      = ["X_ACR_Mot.SV", "X_Pres.SV", "X_Temp.SV"]
CUSUM_CONTROL_LIMIT = 5.0          # configurable, default 5.0
# Reference μ0: M3_normalization_config.json (read-only at M10 startup)
# State: PERSISTENT across API calls. Resets on operator ACK or pump restart.
# Catches: label 21 (bearing_wear_gradual) at ~Week 5.5

# --- Layer 4: Rolling Baseline Comparator (M10 only — NOT in feature matrix) ---
ROLLING_BASELINE_WINDOW     = 30   # windows for rolling slope mean
ROLLING_BASELINE_SIGMA      = 2.0  # control limit: μ_normal + 2σ_normal
ROLLING_SLOPE_CHANNELS      = ["X_ACR_Mot.SV", "X_Pres.SV", "X_Temp.SV"]
# Reference: M3_normalization_config.json normal baselines (per cluster)
# Catches: pre-threshold drift weeks before Layer 1 fires

# --- M8 detection gates (targets for M8 validation) ---
M8_GATE_TPR_OVERALL         = 0.90     # ≥90% excluding overloading/seal mild
M8_GATE_TPR_CAVITATION      = 1.00     # 100% — direct DANGER, 1 window
M8_GATE_TPR_OVERLOADING     = 0.80     # ≥80% via Mech C ONLY (Finding 1)
M8_GATE_TPR_SEAL_WATCH_MIN  = 20       # WATCH ≤20 min of onset (Finding 2)
M8_GATE_TPR_BEARING_GRADUAL = 0.75     # label 21 — via accumulator
M8_GATE_TPR_COMPOUND        = 0.85     # Group B ≥85%
M8_GATE_TPR_MASKED          = 0.65     # Group C ≥65% via secondary Mech C
M8_GATE_FPR_FULL_POOL       = 0.05     # ≤5% FPR on full 9,711 windows
M8_GATE_YOUDEN_J            = 0.85
M8_GATE_SEPARATION          = 5.0      # ≥5.0× separation ratio
M8_SEAM_RATIO_GATE          = 1.0      # attention seam ratio < 1.0 (Finding 3)
M8_GRADUAL_DETECT_WEEK      = 6        # label 21 must detect by Week 6

# --- MC Dropout uncertainty ---
M8_MC_DROPOUT_SAMPLES       = 20       # N=20 forward passes for uncertainty

# ══════════════════════════════════════════════════════════════════
# M7 — XGBOOST FAULT CLASSIFIER
# ══════════════════════════════════════════════════════════════════
M7_INPUT_COLS       = 26       # 25 features + label (M6.5r output)
M7_N_CLASSES        = 22
M7_GATE_ACCURACY    = 0.85     # ≥85% overall accuracy
M7_GATE_F1_MIN      = 0.80     # ≥0.80 per class
M7_GATE_F1_CAVITATION   = 0.88
M7_GATE_F1_SENSOR   = 0.92
M7_GATE_F1_LABEL21  = 0.70     # bearing_wear_gradual (harder class)
M7_CONFIDENCE_STAGE3    = 0.75  # Stage 3 CONFIRMED threshold
M7_CONFIDENCE_STAGE2    = 0.50  # Stage 2 WARN threshold
M7_SECONDARY_FAULT_THRESH = 0.30  # compound secondary shown if prob > 0.30
# XGBoost: device='cuda' train | device='cpu' deploy (Invariant 10)

# ══════════════════════════════════════════════════════════════════
# M9 — PHYSICS CONSTANTS (pump hydraulics)
# ══════════════════════════════════════════════════════════════════
FLUID_DENSITY_KGM3  = 1000.0   # water
GRAVITY_MS2         = 9.81
PRESSURE_WAVE_VEL   = 1200.0   # m/s — steel pipe (Joukowsky)
VAPOUR_PRESSURE_BAR = 0.0234   # water at 20°C
NPSH_SAFETY_MARGIN  = 0.5      # m — cavitation risk flag
CAVITATION_CERTAIN_MARGIN = 0.0  # m — do not operate
AFFINITY_N1         = 2980     # reference speed RPM
STAGE_HEAD_M        = 64.3     # H_total / n_stages = 450/7

# ══════════════════════════════════════════════════════════════════
# M10 — FLASK APP + DEPLOYMENT
# ══════════════════════════════════════════════════════════════════
FLASK_PORT          = 7860     # Hugging Face Spaces port
APP_VERSION         = "2.0"
DEPLOY_DEVICE       = "cpu"    # ALWAYS cpu in deployment (Invariant 11)
# Model loading: torch.load(..., map_location='cpu')
# XGBoost: pickle.load() — MultiOutputClassifier
# NEVER call .cuda() in deployment code

# 4-state alert thresholds (used in M10 UI rendering)
ALERT_STATES = ["NORMAL", "WATCH", "WARN", "DANGER"]
ALERT_NORMAL_MAX_SCORE  = 2.0
ALERT_WATCH_MAX_SCORE   = 3.5
ALERT_WARN_MAX_SCORE    = 5.0
# > 5.0 or single window > M4_THRESHOLD → DANGER

# ══════════════════════════════════════════════════════════════════
# SCOPE BOUNDARY — NEVER VIOLATE (Invariant 9)
# ══════════════════════════════════════════════════════════════════
# if pump_type == 'household': return physics_advisory_only()
# else: return ml_prediction()
# Household monoblock ≠ industrial pump.
# Cross-domain ML inference = OOD = safety risk. No exceptions.
HOUSEHOLD_DISCLAIMER = (
    "Advisory guidance only — not a monitoring tool. "
    "This tool does not monitor pump health and cannot detect faults."
)
INDUSTRIAL_DISCLAIMER_COUNT = 3   # M10 must display all 3 disclaimers before inference

# ══════════════════════════════════════════════════════════════════
# STANDARDS COMPLIANCE
# ══════════════════════════════════════════════════════════════════
STANDARD_VIBRATION      = "ISO 10816-3"
STANDARD_MONITORING     = "ISO 13373-3"
STANDARD_CM_LEVEL       = "ISO 13374 Level 3"
STANDARD_SIS_BOUNDARY   = "IEC 61511"    # PumpSmart is NOT SIS
STANDARD_CUSUM          = "ISO 7870"     # Shewhart control chart basis

# ══════════════════════════════════════════════════════════════════
# ARCHITECTURE VERSION TRACKER
# ══════════════════════════════════════════════════════════════════
ARCH_VERSION        = "v14.0"
ACTIVE_MODULE       = "M6B"    # update this as modules complete
PROJECT_NAME        = "PumpSmart"
DATASET_NAME        = "CIRA SACIP"
ZENODO_RECORD       = "15301820"
