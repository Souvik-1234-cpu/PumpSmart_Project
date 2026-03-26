# config.py
import torch
from pathlib import Path

# ── Hardware ──────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_GPU     = torch.cuda.is_available()

# ── Root ──────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent

# ── Data Directories ──────────────────────────────────────
RAW_DIR    = ROOT_DIR / "data" / "raw"
CLEAN_DIR  = ROOT_DIR / "data" / "clean"
NORM_DIR   = ROOT_DIR / "data" / "normalized"
SYNTH_DIR  = ROOT_DIR / "data" / "synthetic"

# ── Model Directory ───────────────────────────────────────
MODEL_DIR  = ROOT_DIR / "models"

# ── Output Directories ────────────────────────────────────
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR  = OUTPUT_DIR / "plots"

# ── Pump Nameplate Constants ──────────────────────────────
MOTOR_KW        = 110
MOTOR_RPM       = 2980
PUMP_KW         = 10
PUMP_FLOW_M3H   = 45
PUMP_HEAD_M     = 450
PUMP_MAX_BAR    = 40
PUMP_IMPELLERS  = 7

# ── Segment Gap Threshold Multiplier ─────────────────────
GAP_MULTIPLIER  = 2.0          # gap > 2× median interval = new segment

# ── Windowing ─────────────────────────────────────────────
WINDOW_SIZE     = 60           # finalized in M2 based on autocorrelation
WINDOW_STEP     = 10           # stride (to be confirmed in M2)

# ── LSTM-AE Hyperparameters (M4 defaults) ────────────────
LSTM_HIDDEN     = 64
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.2
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
EPOCHS          = 50

# ── Auto-create all directories ───────────────────────────
for _dir in [RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
             MODEL_DIR, PLOTS_DIR, OUTPUT_DIR / "reports"]:
    _dir.mkdir(parents=True, exist_ok=True)
