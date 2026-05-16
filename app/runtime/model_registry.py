# =============================================================================
# app/runtime/model_registry.py  — Phase 2 update
# Loads ALL artifacts at lifespan startup. Hard fail if any missing.
# M4: LayerNorm (NOT BatchNorm) — T1.5.1 confirmed fix.
# Checkpoint names confirmed from project knowledge.
# =============================================================================

import json
import hashlib
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import xgboost as xgb
import pandas as pd
import numpy as np

MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs")
DATA_DIR   = Path("data") / "synthetic"

REQUIRED_ARTIFACTS = {
    "m4_lstm_ae"      : MODEL_DIR / "lstm_ae_baseline_final.pth",   # confirmed filename
    "m8_tcn_ae"       : MODEL_DIR / "tcn_ae_best.pth",              # confirmed filename
    "m7_xgboost"      : MODEL_DIR / "M7_xgboost.json",
    "m4_threshold"    : MODEL_DIR / "M4_threshold_config.json",
    "m8_threshold"    : MODEL_DIR / "M8_threshold_config.json",
    "m8p4_ood"        : MODEL_DIR / "M8p4_ood_config.json",
    "m8p6_sensor"     : MODEL_DIR / "M8p6_sensor_sensitivity_config.json",
    "fault_rules"     : MODEL_DIR / "fault_rules_v3.json",
    "m3_norm"         : MODEL_DIR / "M3_normalization_config.json",
    "m2_bounds"       : OUTPUT_DIR / "M2_cluster_bounds.csv",
    "physics_context" : DATA_DIR  / "M6B_physics_context_strings.json",
}


# =============================================================================
# M4 LSTM-AE Architecture — EXACT match to lstm_ae_baseline_final.pth
# LayerNorm confirmed (T1.5.1 fix — NOT BatchNorm1d).
# Source: src/module_04_lstm_ae_baseline.py + src/module_08_tcn_ae_detection_stack.py
# =============================================================================
class _M4Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8,   128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64,  num_layers=1, batch_first=True)
        self.bn    = nn.LayerNorm(64)   # 'bn' key matches checkpoint — LayerNorm NOT BatchNorm

    def forward(self, x):
        out1, _      = self.lstm1(x)
        out2, (h, c) = self.lstm2(out1)
        return self.bn(h[-1]), h, c    # z_t: [batch, 64]


class _M4Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_h  = nn.Linear(64, 128)
        self.fc_c  = nn.Linear(64, 128)
        self.lstm1 = nn.LSTM(64,  128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 8,   num_layers=1, batch_first=True)
        self.out   = nn.Linear(8, 8)

    def forward(self, z, seq_len, h_enc, c_enc):
        h0    = torch.tanh(self.fc_h(h_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0    = torch.tanh(self.fc_c(c_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        x_rep = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm1(x_rep, (h0, c0))
        out, _ = self.lstm2(out)
        return self.out(out)


class _M4LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _M4Encoder()
        self.decoder = _M4Decoder()

    def forward(self, x):
        z, h, c = self.encoder(x)
        return self.decoder(z, x.size(1), h, c)


# =============================================================================
# M8 TCN-AE Architecture stub — matches tcn_ae_best.pth
# Full architecture in src/module_08_tcn_ae_detection_stack.py
# Loaded here by importing from src if available, else inline.
# =============================================================================
def _load_tcn_ae(path: Path, m8_cfg: dict):
    try:
        from src.module_08_tcn_ae_detection_stack import TCNAutoencoder
        model = TCNAutoencoder(
            z_dim        = m8_cfg.get("zt_dim", 64),
            tcn_channels = m8_cfg.get("tcn_channels", [128, 128, 128, 128, 128]),
            kernel_size  = m8_cfg.get("kernel_size", 3),
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
    except (ImportError, Exception) as e:
        _log(f"  WARNING: TCN-AE load issue: {e} — using None (score_B/C = 0.0)")
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] [model_registry] {msg}", flush=True)


def load_all_models() -> dict:
    # ── 1. Verify all present ────────────────────────────────────────────────
    missing = [n for n, p in REQUIRED_ARTIFACTS.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"STARTUP FAILED — missing artifacts: {missing}. "
            f"Application will not start without all required files."
        )

    hashes = {n: _sha256(p) for n, p in REQUIRED_ARTIFACTS.items()}

    # ── 2. Load configs ──────────────────────────────────────────────────────
    with open(REQUIRED_ARTIFACTS["m4_threshold"]) as f: m4_cfg   = json.load(f)
    with open(REQUIRED_ARTIFACTS["m8_threshold"]) as f: m8_cfg   = json.load(f)
    with open(REQUIRED_ARTIFACTS["m8p4_ood"])     as f: ood_cfg  = json.load(f)
    with open(REQUIRED_ARTIFACTS["m8p6_sensor"])  as f: m8p6_cfg = json.load(f)
    with open(REQUIRED_ARTIFACTS["fault_rules"])  as f: fault_rules = json.load(f)
    with open(REQUIRED_ARTIFACTS["m3_norm"])      as f: norm_cfg = json.load(f)
    with open(REQUIRED_ARTIFACTS["physics_context"]) as f: physics_ctx = json.load(f)
    cluster_bounds = pd.read_csv(REQUIRED_ARTIFACTS["m2_bounds"])

    # ── 3. M4 LSTM-AE — LayerNorm architecture, map_location='cpu' ──────────
    m4_model = _M4LSTMAutoencoder()
    state = torch.load(REQUIRED_ARTIFACTS["m4_lstm_ae"], map_location="cpu", weights_only=True)
    m4_model.load_state_dict(state, strict=True)   # strict=True — LayerNorm has no buffers
    m4_model.eval()
    for p in m4_model.parameters():
        p.requires_grad_(False)
    _log(f"M4 LSTM-AE loaded (LayerNorm, strict=True) — {sum(p.numel() for p in m4_model.parameters()):,} params")

    # ── 4. M8 TCN-AE ─────────────────────────────────────────────────────────
    m8_model = _load_tcn_ae(REQUIRED_ARTIFACTS["m8_tcn_ae"], m8_cfg)
    if m8_model:
        _log(f"M8 TCN-AE loaded — {sum(p.numel() for p in m8_model.parameters()):,} params")
    else:
        _log("M8 TCN-AE NOT loaded — score_B/C will be 0.0 (Phase 2 fallback)")

    # ── 5. M7 XGBoost — CPU deploy ───────────────────────────────────────────
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(REQUIRED_ARTIFACTS["m7_xgboost"]))
    n_classes = xgb_model.n_classes_
    assert n_classes == 22, f"Expected 22 XGBoost classes, got {n_classes}"
    _log(f"M7 XGBoost loaded — {n_classes} classes, CPU deploy")

    # ── 6. Threshold validation ───────────────────────────────────────────────
    m4_threshold = float(m4_cfg["threshold"])
    assert abs(m4_threshold - 0.110058) < 1e-4, (
        f"CRITICAL: M4 threshold drifted to {m4_threshold:.6f}. Do NOT retrain M4."
    )
    _log(f"M4 threshold locked: q={m4_threshold:.6f} ✓")

    label_map = {int(k): v for k, v in fault_rules["label_map"].items()}

    return {
        "m4_model"          : m4_model,
        "m8_model"          : m8_model,
        "xgb_model"         : xgb_model,
        "m4_threshold"      : m4_threshold,
        "cusum_H"           : float(m8_cfg.get("cusum_H", 5.0)),
        "cusum_k"           : float(m8_cfg.get("cusum_k", 0.5)),
        "cusum_lambda"      : float(m8_cfg.get("cusum_lambda", 5.73e-05)),
        "rolling_window_size": int(m8_cfg.get("rolling_window_size", 432)),
        "theta_initial"     : float(m8_cfg.get("theta_initial", 1.881275)),
        "ood_tau_p99"       : float(ood_cfg.get("tau_p99", 15.0)),
        "score_c_threshold" : float(m8_cfg.get("score_c_threshold", 0.5)),
        "zt_buffer_len"     : int(m8_cfg.get("zt_buffer_len", 63)),
        "zt_dim"            : int(m8_cfg.get("zt_dim", 64)),
        "label_map"         : label_map,
        "fault_rules"       : fault_rules,
        "norm_cfg"          : norm_cfg,
        "cluster_bounds"    : cluster_bounds,
        "physics_ctx"       : physics_ctx,
        "m8p6_cfg"          : m8p6_cfg,
        "m8p4_cfg"          : ood_cfg,
        "xgb_n_classes"     : n_classes,
        "n_fault_labels"    : len(label_map),
        "m8p6_n_channels"   : len(m8p6_cfg.get("channels", [])),
        "artifact_hashes"   : hashes,
        "artifact_paths"    : {k: str(v) for k, v in REQUIRED_ARTIFACTS.items()},
    }
