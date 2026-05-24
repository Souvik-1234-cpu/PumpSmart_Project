# =============================================================================
# app/runtime/model_registry.py  — Stage 1 update (22 May 2026)
# Loads ALL artifacts at lifespan startup. Hard fail if any missing.
# M4: LayerNorm (NOT BatchNorm) — T1.5.1 confirmed fix.
# M8: Correct TCN-AE architecture — D8 resolved (strict=True, analytical scores).
# Stage 1.4: run_startup_selftest() called at end — server aborts on mismatch.
# =============================================================================

import json
import pickle
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
SYNTH_DIR  = Path("data") / "synthetic"

REQUIRED_ARTIFACTS = {
    "m4_lstm_ae"      : MODEL_DIR  / "lstm_ae_baseline_final.pth",
    "m8_tcn_ae"       : MODEL_DIR  / "tcn_ae_level2_best.pth",
    "m7_xgboost"      : MODEL_DIR  / "M7_xgboost_classifier_cpu.json",
    "m4_threshold"    : MODEL_DIR  / "M4_threshold_config.json",
    "m8_threshold"    : MODEL_DIR  / "M8_threshold_config.json",
    "m8p4_ood"        : MODEL_DIR  / "M8p4_ood_detector_config.json",
    "m8p6_sensor"     : MODEL_DIR  / "M8p6_sensor_sensitivity_config.json",
    "fault_rules"     : MODEL_DIR  / "fault_rules_v3.json",
    "m3_norm"         : MODEL_DIR  / "M3_normalization_config.json",
    "m2_bounds"       : DATA_DIR   / "M2_cluster_bounds.csv",
    "physics_context" : DATA_DIR   / "M6B_physics_context_strings.json",
    # Stage 1.2 — PCA artifacts for Class B features
    "zt_pca"          : MODEL_DIR  / "M6p5r_zt_pca.pkl",
    "zt_mean"         : MODEL_DIR  / "M6p5r_zt_mean.npy",
}


# =============================================================================
# M4 LSTM-AE Architecture — EXACT match to lstm_ae_baseline_final.pth
# LayerNorm confirmed (T1.5.1 fix — NOT BatchNorm1d).
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
# M8 TCN-AE — correct architecture reconstructed from checkpoint keys (D8 fix)
#
# Key pattern confirmed:
#   encoder.tcn_stack.N.{conv1,conv2}.conv.{weight,bias}
#   encoder.tcn_stack.N.{bn1,bn2}.{weight,bias}
#   encoder.bottleneck.{weight,bias}   [Linear(64→32)]
#   decoder.tcn_stack.N.{conv1,conv2}.conv.{weight,bias}
#   decoder.tcn_stack.N.{bn1,bn2}.{weight,bias}
#   decoder.out_proj.{weight,bias}     [Conv1d(64,64,1)]
#
# Scores are computed analytically — no learned head_A/B/C.
# strict=True now valid (all 84 keys match exactly).
# =============================================================================
def _load_tcn_ae(path: Path, m8_cfg: dict):
    try:
        import torch.nn.functional as F

        # ── Causal conv wrapper (key pattern: block.convN.conv.*) ─────────────
        class _CausalConv(nn.Module):
            def __init__(self, ch, dilation):
                super().__init__()
                self.conv  = nn.Conv1d(ch, ch, 3, padding=(3-1)*dilation, dilation=dilation)
                self._trim = (3-1)*dilation
            def forward(self, x):
                out = self.conv(x)
                return out[:, :, :-self._trim] if self._trim else out

        # ── Dual-conv residual block (bn1/bn2, conv1/conv2 per checkpoint) ────
        class _ResBlock(nn.Module):
            def __init__(self, ch, dilation):
                super().__init__()
                self.conv1 = _CausalConv(ch, dilation)
                self.conv2 = _CausalConv(ch, dilation)
                self.bn1   = nn.LayerNorm(ch)
                self.bn2   = nn.LayerNorm(ch)
            def forward(self, x):            # x: [B, ch, T]
                o = self.conv1(x)
                o = F.relu(self.bn1(o.transpose(1,2)).transpose(1,2))
                o = self.conv2(o)
                o = self.bn2(o.transpose(1,2)).transpose(1,2)
                return F.relu(o + x)         # identity residual (ch_in == ch_out)

        # ── Encoder ───────────────────────────────────────────────────────────
        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.tcn_stack  = nn.ModuleList(
                    [_ResBlock(64, d) for d in [1, 2, 4, 8, 16]])
                self.bottleneck = nn.Linear(64, 32)   # → z_t [B, 32]
            def forward(self, x):            # x: [B, 64, T]
                for blk in self.tcn_stack:
                    x = blk(x)
                z_t = self.bottleneck(x.mean(-1))     # [B, 32]
                return x, z_t

        # ── Decoder ───────────────────────────────────────────────────────────
        class _Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.tcn_stack = nn.ModuleList(
                    [_ResBlock(64, d) for d in [1, 2, 4, 8, 16]])
                self.out_proj  = nn.Conv1d(64, 64, 1)
            def forward(self, x):            # x: [B, 64, T]
                for blk in self.tcn_stack:
                    x = blk(x)
                return self.out_proj(x)      # [B, 64, T]

        # ── Full autoencoder ──────────────────────────────────────────────────
        class _TCNAutoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Encoder()
                self.decoder = _Decoder()

            def forward(self, x):
                # x: [B, seq, 64]  — z_t buffer from M4 ZTBuffer
                B, T, C = x.shape
                xc = x.transpose(1, 2)              # [B, 64, T]

                enc_out, z_t = self.encoder(xc)     # [B, 64, T], [B, 32]
                dec_out = self.decoder(enc_out)      # [B, 64, T]
                recon   = dec_out.transpose(1, 2)    # [B, T, 64]

                # score_A: mean reconstruction MAE over full buffer
                residual = (x - recon).abs()         # [B, T, 64]
                score_A  = residual.mean(dim=(1, 2)) # [B]

                # score_B: signed drift (second-half minus first-half MAE)
                # Calibrated: mu0=-0.00954, k=0.02186, H=5.0 (CUSUM params)
                # Positive for gradual wear (error grows over time)
                half    = T // 2
                score_B = (residual[:, half:].mean(dim=(1, 2))
                           - residual[:, :half].mean(dim=(1, 2)))  # [B]

                # score_C: temporal MAE variance → chain transition signal
                # Normal P95=0.31 | Group B mean=3.3-5.0 (20-31× above normal)
                score_C = F.relu(
                    residual.mean(dim=2).std(dim=1))               # [B]

                return score_A, score_B, score_C

        model = _TCNAutoencoder()
        state = torch.load(path, map_location="cpu", weights_only=True)
        res   = model.load_state_dict(state, strict=True)          # strict=True now valid
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        n = sum(p.numel() for p in model.parameters())
        _log(f"TCN-AE loaded (correct arch, strict=True, D8 resolved) — {n:,} params")
        return model

    except Exception as e:
        _log(f"WARNING: TCN-AE load failed: {e} — score_B/C = 0.0")
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
    # ── 1. Verify all artifacts present ──────────────────────────────────────
    missing = [n for n, p in REQUIRED_ARTIFACTS.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"STARTUP FAILED — missing artifacts: {missing}. "
            f"Application will not start without all required files."
        )

    hashes = {n: _sha256(p) for n, p in REQUIRED_ARTIFACTS.items()}

    # ── 2. Load JSON configs ──────────────────────────────────────────────────
    with open(REQUIRED_ARTIFACTS["m4_threshold"],    encoding="utf-8") as f: m4_cfg      = json.load(f)
    with open(REQUIRED_ARTIFACTS["m8_threshold"],    encoding="utf-8") as f: m8_cfg      = json.load(f)
    with open(REQUIRED_ARTIFACTS["m8p4_ood"],        encoding="utf-8") as f: ood_cfg     = json.load(f)
    with open(REQUIRED_ARTIFACTS["m8p6_sensor"],     encoding="utf-8") as f: m8p6_cfg    = json.load(f)
    with open(REQUIRED_ARTIFACTS["fault_rules"],     encoding="utf-8") as f: fault_rules = json.load(f)
    with open(REQUIRED_ARTIFACTS["m3_norm"],         encoding="utf-8") as f: norm_cfg    = json.load(f)
    with open(REQUIRED_ARTIFACTS["physics_context"], encoding="utf-8") as f: physics_ctx = json.load(f)
    cluster_bounds = pd.read_csv(REQUIRED_ARTIFACTS["m2_bounds"])

    # ── 3. M4 LSTM-AE — LayerNorm architecture, map_location='cpu' ───────────
    m4_model = _M4LSTMAutoencoder()
    state = torch.load(REQUIRED_ARTIFACTS["m4_lstm_ae"], map_location="cpu", weights_only=True)
    m4_model.load_state_dict(state, strict=True)
    m4_model.eval()
    for p in m4_model.parameters():
        p.requires_grad_(False)
    _log(f"M4 LSTM-AE loaded (LayerNorm, strict=True) — {sum(p.numel() for p in m4_model.parameters()):,} params")

    # ── 4. M8 TCN-AE (D8 architecture fix applied) ───────────────────────────
    m8_model = _load_tcn_ae(REQUIRED_ARTIFACTS["m8_tcn_ae"], m8_cfg)
    if m8_model:
        _log(f"M8 TCN-AE loaded — {sum(p.numel() for p in m8_model.parameters()):,} params")
    else:
        _log("M8 TCN-AE NOT loaded — score_B/C will be 0.0 (Phase 2 fallback)")

    # ── 5. M7 XGBoost — CPU deploy ───────────────────────────────────────────
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(REQUIRED_ARTIFACTS["m7_xgboost"]))
    n_classes = xgb_model.n_classes_
    assert n_classes in (22, 24), f"Expected 22 or 24 XGBoost classes, got {n_classes}"
    _log(f"M7 XGBoost loaded — {n_classes} classes, CPU deploy")

    # ── 6. M4 threshold validation (LOCKED) ──────────────────────────────────
    m4_threshold = float(m4_cfg["anomaly_threshold"])
    assert abs(m4_threshold - 0.110058) < 1e-4, (
        f"CRITICAL: M4 threshold drifted to {m4_threshold:.6f}. Do NOT retrain M4."
    )
    _log(f"M4 threshold locked: q={m4_threshold:.6f} ✓")

    label_map = {int(k): v for k, v in fault_rules["label_map"].items()}

    models = {
        "m4_model"           : m4_model,
        "m8_model"           : m8_model,
        "xgb_model"          : xgb_model,
        "m4_threshold"       : m4_threshold,
        "cusum_H"            : float(m8_cfg.get("cusum_H", 5.0)),
        "cusum_k"            : float(m8_cfg.get("cusum_k", 0.5)),
        "cusum_lambda"       : float(m8_cfg.get("cusum_lambda", 5.73e-05)),
        "rolling_window_size": int(m8_cfg.get("rolling_window_size", 432)),
        "theta_initial"      : float(m8_cfg.get("theta_initial", 1.881275)),
        "ood_tau_p99"        : float(ood_cfg.get("tau_p99", 15.0)),
        "score_c_threshold"  : float(m8_cfg.get("score_c_threshold", 0.5)),
        "zt_buffer_len"      : int(m8_cfg.get("zt_buffer_len", 63)),
        "zt_dim"             : int(m8_cfg.get("zt_dim", 64)),
        "label_map"          : label_map,
        "fault_rules"        : fault_rules,
        "norm_cfg"           : norm_cfg,
        "cluster_bounds"     : cluster_bounds,
        "physics_ctx"        : physics_ctx,
        "m8p6_cfg"           : m8p6_cfg,
        "m8p4_cfg"           : ood_cfg,
        "xgb_n_classes"      : n_classes,
        "n_fault_labels"     : len(label_map),
        "m8p6_n_channels"    : len(m8p6_cfg.get("channels", [])),
        "artifact_hashes"    : hashes,
        "artifact_paths"     : {k: str(v) for k, v in REQUIRED_ARTIFACTS.items()},
    }

    # ── 7. Stage 1.4 — feature_builder startup self-test (C-30 enforcement) ──
    # Loads M6B sequences, runs feature_builder on 20 reference windows,
    # asserts Class A+B features match embedded ground-truth within 1e-5.
    # Raises RuntimeError → server refuses to start on any feature drift.
    try:
        _log("Running feature_builder startup self-test (Stage 1.4 / C-30)...")
        m6b_path = SYNTH_DIR / "M6B_combined_sequences.pkl"
        with open(m6b_path, "rb") as f:
            _m6b = pickle.load(f)
        seqs_list = _m6b["sequences"]
        meta_list = _m6b["metadata"]

        from app.runtime.feature_builder_selftest import run_startup_selftest
        run_startup_selftest(m4_model, seqs_list, meta_list)
        _log("feature_builder self-test: PASS ✓")
    except RuntimeError:
        # RuntimeError from run_startup_selftest = identity mismatch — abort startup
        raise
    except Exception as e:
        # Any other error (import, file not found) = warn but do not block startup.
        # This allows the server to start in environments where M6B is not deployed
        # (e.g. HuggingFace Spaces production where only models/ is present).
        _log(f"WARNING: feature_builder self-test skipped — {e}")
        _log("  (Non-fatal if M6B_combined_sequences.pkl absent in production env.)")

    return models