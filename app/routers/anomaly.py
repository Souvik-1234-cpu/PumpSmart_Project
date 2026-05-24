# =============================================================================
# app/routers/anomaly.py — POST /api/anomaly_detect
# M10 Phase 2 — Full 4-layer inference pipeline
# =============================================================================
# Pipeline:
#   Raw 50×8 window (M3-normalised)
#     → M4 LSTM-AE encoder → score_A (MAE) + z_t (ℝ⁶⁴) + mae_per_ch (ℝ⁸)
#     → ZTBuffer.append(z_t)
#     → M8 TCN-AE (when buffer ready, 63 z_t) → score_B (drift) + score_C (chain)
#     → score_B → CUSUMState.update()     [L3 — Invariant 19]
#     → score_A → RollingState.update()   [L4 — Invariant 19]
#     → score_C → M7 XGBoost             [22-class classification]
#     → M8p4 OOD Mahalanobis distance
#     → M8p6 sensor sensitivity check
#     → Build FaultPrediction (7 mandatory fields)
#
# Invariant 19: score_A → L4, score_B → L3, score_C → M7. NEVER cross-routed.
# C-25: RollingState.update() NEVER calls CUSUMState.reset().
# C-26: MODEL_DISCLAIMER_TEXT always present in Field 7.
# C-28: M8p6 addendum annotates Field 6 only — never alters label/confidence.
#
# Stage 1.1 patch applied (22 May 2026):
#   run_m4 now returns mae_per_ch_np as 3rd element (was discarded — D1a fix).
#   _build_m7_features receives mae_per_ch_np directly from run_m4 output,
#   replacing the broken MAD-of-raw-window that was 100× off scale (D1a).
# =============================================================================

from app.runtime.feature_builder import build_m7_features
import uuid
import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, Request, HTTPException

from app.schemas.sensor_input import SensorWindow
from app.schemas.fault_output import (
    FaultPrediction, M8p6Addendum, MODEL_DISCLAIMER_TEXT
)

router = APIRouter(tags=["inference"])

# Channel weights for score_A (MAE) — M8 validated, Fisher-ranked
# Order: Mot.SV, Pmp.SV, Mot.TV, Pmp.PV, Temp.SV, Pres.SV, Pmp.TV, Mot.PV
CH_WEIGHTS = torch.tensor([2.5, 2.5, 0.3, 2.0, 0.5, 2.5, 0.3, 2.0], dtype=torch.float32)

# Group B labels — compound fault chains (causal chain returned in response)
GROUP_B_LABELS = {7, 8, 9, 10, 11, 12}

# Group C labels — masked faults (limitation flag added)
GROUP_C_LABELS = {13, 14, 15, 16, 17, 22, 23}

# Cluster name map
CLUSTER_NAMES = {
    "startup"      : "Cluster 0 — Startup",
    "steady_state" : "Cluster 1 — Steady-state",
    "high_load"    : "Cluster 2 — High-load",
    "cooldown"     : "Cluster 3 — Cooldown",
}


# =============================================================================
# L1 — M4 LSTM-AE forward pass
# =============================================================================
@torch.no_grad()
def run_m4(window_tensor: torch.Tensor, m4_model, q: float) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Forward pass through M4 LSTM-AE encoder.
    Returns: (score_A, z_t_np, mae_per_ch_np, raw_mae)
      score_A       = physics-weighted MAE scalar (routed to L4 RollingState)
      z_t_np        = encoder bottleneck vector ℝ⁶⁴ (routed to ZTBuffer → L2 TCN-AE)
      mae_per_ch_np = per-channel reconstruction MAE ℝ⁸ (routed to feature_builder)
                      Stage 1.1 patch: formerly discarded — now surfaced (D1a fix).
      raw_mae       = unweighted mean MAE (for OOD Mahalanobis feature)
    """
    # window_tensor: [1, 50, 8]
    x = window_tensor.float()

    # Encoder pass
    enc = m4_model.encoder
    out1, _ = enc.lstm1(x)                           # [1, 50, 128]
    out2, (h_n, c_n) = enc.lstm2(out1)               # h_n: [1, 1, 64]
    z_t = enc.bn(h_n[-1])                            # [1, 64] — LayerNorm (NOT BatchNorm)

    # Decoder reconstruction
    recon = m4_model.decoder(z_t, x.size(1), h_n, c_n)   # [1, 50, 8]

    # Per-channel reconstruction MAE — identical formula to M6.5r training-time computation.
    # Mean over time dimension (dim=1), squeeze batch dim → [8]
    mae_per_ch = (x - recon).abs().mean(dim=1).squeeze(0)   # [8]

    # Physics-weighted score_A (→ L4)
    weights = CH_WEIGHTS.to(mae_per_ch.device)
    score_A = (mae_per_ch * weights).sum().item() / weights.sum().item()

    # Raw unweighted MAE (for OOD feature)
    raw_mae = mae_per_ch.mean().item()

    return (
        score_A,
        z_t.squeeze(0).cpu().numpy(),
        mae_per_ch.cpu().numpy(),    # ← Stage 1.1 patch: was discarded, now returned (D1a fix)
        raw_mae,
    )


# =============================================================================
# L2 — M8 TCN-AE forward pass (when z_t buffer ready)
# =============================================================================
@torch.no_grad()
def run_tcn_ae(zt_sequence: np.ndarray, m8_model) -> tuple[float, float]:
    """
    Forward pass through M8 TCN-AE on 63-window z_t sequence.
    Returns: (score_B, score_C)
      score_B = drift slope output  → L3 CUSUM  (Invariant 19)
      score_C = chain transition    → M7 XGBoost (Invariant 19)
    """
    # zt_sequence: [63, 64]
    x = torch.from_numpy(zt_sequence).float().unsqueeze(0)   # [1, 63, 64]

    # TCN-AE forward — returns (score_A_tcn, score_B, score_C) from dual output heads
    _sA, score_B, score_C = m8_model(x)

    return float(score_B), float(score_C)


# =============================================================================
# OOD — Mahalanobis distance (M8p4)
# =============================================================================
def compute_mahalanobis(z_t_np: np.ndarray, ood_cfg: dict) -> float:
    """
    Computes Mahalanobis distance of z_t from training distribution centroid.
    Uses pre-computed centroid and precision matrix from M8p4 OOD config.
    """
    try:
        centroid  = np.array(ood_cfg["centroid"], dtype=np.float32)          # [64]
        precision = np.array(ood_cfg["precision_matrix"], dtype=np.float32)  # [64, 64]
        diff      = z_t_np - centroid
        dist      = float(np.sqrt(diff @ precision @ diff))
        return dist
    except (KeyError, Exception):
        return 0.0   # graceful fallback if config missing precision matrix


# =============================================================================
# M8p6 — Sensor sensitivity check (C-28 / Principle 14)
# =============================================================================
def check_m8p6(window_np: np.ndarray, cluster: str, m8p6_cfg: dict) -> M8p6Addendum:
    """
    Checks if any sensor channel is approaching ISA-37 ceiling in the active cluster.
    Annotates Field 6 only — NEVER overrides fault label or confidence.
    override_existing_prediction: always False (locked).
    """
    flagged_channels = []
    headroom_flag    = float(m8p6_cfg.get("headroom_flag_threshold", 0.10))

    channels_cfg = m8p6_cfg.get("channels", [])
    for ch_cfg in channels_cfg:
        ch_name     = ch_cfg.get("name", "")
        ch_idx      = ch_cfg.get("index", -1)
        cluster_cfg = ch_cfg.get("cluster_ceilings", {}).get(cluster, {})
        ceiling     = cluster_cfg.get("ceiling_multiplier", 3.0)
        mean_val    = cluster_cfg.get("cluster_mean", 1.0)

        if ch_idx < 0 or ch_idx >= 8:
            continue

        channel_vals = window_np[:, ch_idx]
        gain_p95     = float(np.percentile(np.abs(channel_vals), 95))
        ceiling_val  = ceiling * mean_val
        headroom     = 1.0 - (gain_p95 / ceiling_val) if ceiling_val > 0 else 1.0

        if headroom < headroom_flag:
            flagged_channels.append(
                f"{ch_name} ({headroom*100:.1f}% headroom in {cluster})"
            )

    if not flagged_channels:
        return M8p6Addendum(triggered=False)

    addendum_text = (
        "Sensor health: " + "; ".join(flagged_channels) +
        " — verify transducer calibration before trusting prediction."
    )
    return M8p6Addendum(
        triggered=True,
        flagged_channels=flagged_channels,
        addendum_text=addendum_text,
        override_existing_prediction=False,   # LOCKED — C-28 / Principle 14
    )


# =============================================================================
# Alert state machine
# =============================================================================
def compute_alert_state(
    score_A: float, theta_t: float, cusum_Sn: float, drift_locked: bool
) -> str:
    if score_A >= 1.5 * theta_t or (drift_locked and score_A >= theta_t):
        return "DANGER"
    elif score_A >= theta_t or drift_locked:
        return "WARN"
    elif cusum_Sn >= 2.0:
        return "WATCH"
    return "NORMAL"


# =============================================================================
# Build limitation flags
# =============================================================================
def build_limitation_flags(label_int: int, ood_suspected: bool,
                            conf_pct: float, m8p6: M8p6Addendum) -> list[str]:
    flags = []
    if label_int in GROUP_C_LABELS:
        flags.append(
            "Group C masked fault — sensor failure masking underlying condition. "
            "Physical inspection of masked channel mandatory."
        )
    if label_int == 21:
        flags.append(
            "Label 21 gradual bearing wear — earliest reliable detection ~Week 5. "
            "CUSUM S_n = 0 does NOT confirm bearing health."
        )
    if label_int in GROUP_B_LABELS:
        flags.append(
            "Group B compound chain — verify causal direction physically "
            "before acting on primary vs secondary fault."
        )
    if ood_suspected:
        flags.append(
            "OOD_SUSPECTED — Mahalanobis distance exceeds tau_p99. "
            "Input may be outside training distribution. Treat with caution."
        )
    if conf_pct < 70.0:
        flags.append(
            "UNKNOWN_FAULT — confidence below 70%. Multiple fault types plausible. "
            "Inspect regardless of label."
        )
    if m8p6.triggered:
        flags.append(
            "M8p6 sensor ceiling-approach — see Field 6 sensor health addendum."
        )
    flags.append(
        "Trained on physics-synthetic data. Verify physically before any action. (C-26)"
    )
    return flags


# =============================================================================
# MAIN ROUTE
# =============================================================================
@router.post(
    "/anomaly_detect",
    response_model=FaultPrediction,
    summary="Main 4-layer inference — returns 7-field mandatory output",
)
async def anomaly_detect(payload: SensorWindow, request: Request):
    """
    Full 4-layer inference. Accepts M3-normalised 50×8 window.
    Returns complete FaultPrediction with all 7 mandatory fields.
    score routing is strictly enforced (Invariant 19).
    """
    models = request.app.state.models

    # Household pump guard
    if payload.pump_id.startswith("HH-"):
        raise HTTPException(
            status_code=400,
            detail="Household pumps must use /api/household — ML inference not valid.",
        )

    # ── Prepare input tensor ─────────────────────────────────────────────────
    window_np     = np.array(payload.window, dtype=np.float32)   # [50, 8]
    window_tensor = torch.from_numpy(window_np).unsqueeze(0)      # [1, 50, 8]

    # ── L1: M4 LSTM-AE ───────────────────────────────────────────────────────
    m4_model = models["m4_model"]
    # Stage 1.1 patch: unpack 4 values — mae_per_ch_np now in scope (D1a fix)
    score_A, z_t_np, mae_per_ch_np, raw_mae = run_m4(
        window_tensor, m4_model, models["m4_threshold"]
    )

    # Feed z_t into buffer (async-safe)
    await request.app.state.zt_buf.append(z_t_np)

    # ── L2: TCN-AE (when z_t buffer full) ────────────────────────────────────
    score_B = 0.0
    score_C = 0.0
    m8_model = models["m8_model"]
    if m8_model is not None and await request.app.state.zt_buf.is_ready():
        zt_sequence = await request.app.state.zt_buf.get_sequence()   # [63, 64]
        # Run in executor to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        score_B, score_C = await loop.run_in_executor(
            None, run_tcn_ae, zt_sequence, m8_model
        )

    # ── L3: CUSUM update — score_B ONLY (Invariant 19) ───────────────────────
    cusum_result = await request.app.state.cusum.update(score_B)
    cusum_Sn     = cusum_result["cusum_Sn"]

    # ── L4: Rolling baseline update — score_A ONLY (Invariant 19) ────────────
    rolling_result = await request.app.state.rolling.update(score_A)
    theta_t        = rolling_result["theta_t"]
    drift_locked   = rolling_result["drift_locked"]

    # ── Alert state ───────────────────────────────────────────────────────────
    alert_state = compute_alert_state(score_A, theta_t, cusum_Sn, drift_locked)

    # ── M7 XGBoost classification — score_C routed here (Invariant 19) ───────
    xgb_model   = models["xgb_model"]
    label_map   = models["label_map"]
    fault_rules = models["fault_rules"]

    # Build feature vector for M7.
    # mae_per_ch_np is the true M4 reconstruction error per channel (D1a fix).
    # Class D sequence-aware features are stubbed at 0.0 — Stage 2 fills them.
    feature_vec = build_m7_features(mae_per_ch_np, window_np, z_t_np, score_A, score_B, score_C)
    feature_vec_2d = feature_vec.reshape(1, -1)

    proba      = xgb_model.predict_proba(feature_vec_2d)[0]   # [22]
    label_int  = int(np.argmax(proba))
    conf_pct   = float(proba[label_int]) * 100.0
    label_name = label_map.get(label_int, "unknown")

    # ── OOD detection (M8p4 Mahalanobis) ─────────────────────────────────────
    mahal_dist    = compute_mahalanobis(z_t_np, models["m8p4_cfg"])
    ood_suspected = mahal_dist > models["ood_tau_p99"]

    # ── M8p6 sensor sensitivity check (C-28) ─────────────────────────────────
    m8p6_addendum = check_m8p6(window_np, payload.cluster, models["m8p6_cfg"])

    # ── Physics context lookup ────────────────────────────────────────────────
    phys = models["physics_ctx"].get("labels", {}).get(str(label_int), {})
    recommended_action = phys.get("recommended_action", "Inspect per maintenance schedule.")

    # Append M8p6 addendum to Field 6 if triggered (sidecar — C-28 / Principle 14)
    if m8p6_addendum.triggered:
        recommended_action += "\n\n⚠️ " + m8p6_addendum.addendum_text

    # ── Causal chain (Group B only) ───────────────────────────────────────────
    causal_chain = None
    if label_int in GROUP_B_LABELS:
        causal_chain = fault_rules.get("compound_chains", {}).get(
            str(label_int), {}).get(
            "description",
            "Compound chain — verify causal direction physically before acting."
        )

    # ── SHAP top features (lightweight inline — full SHAP in Analytics tab) ──
    top_shap: Optional[dict] = None
    try:
        scores    = xgb_model.get_booster().get_score(importance_type="gain")
        sorted_ft = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_shap  = {k: round(v, 4) for k, v in sorted_ft}
    except Exception:
        top_shap = None

    # ── Limitation flags ──────────────────────────────────────────────────────
    limitation_flags = build_limitation_flags(
        label_int, ood_suspected, conf_pct, m8p6_addendum
    )

    # ── Build 7-field response ────────────────────────────────────────────────
    prediction = FaultPrediction(
        # Field 1
        fault_label=label_name,
        # Field 2
        confidence_pct=round(conf_pct, 2),
        unknown_fault_flag=conf_pct < 70.0,
        # Field 3
        probable_physical_condition=phys.get(
            "probable_condition",
            f"[Physics context not available for label {label_int}]"
        ),
        # Field 4
        expected_sensor_behavior=phys.get(
            "expected_sensor_behaviour",
            "Monitor all 8 channels for deviation from cluster baseline."
        ),
        # Field 5
        operational_risk_if_ignored=phys.get(
            "risk_if_ignored",
            "Unknown — inspect physically to determine consequence timeline."
        ),
        # Field 6 (M8p6 addendum appended if triggered)
        recommended_action=recommended_action,
        # Field 7 — LOCKED, always present
        model_limitation_disclaimer=MODEL_DISCLAIMER_TEXT,

        # Scores (Invariant 19 routing already applied above)
        score_A=round(score_A, 6),
        score_B=round(score_B, 6),
        score_C=round(score_C, 6),
        cusum_Sn=cusum_Sn,
        adaptive_threshold=theta_t,
        alert_state=alert_state,

        # Metadata
        prediction_id=str(uuid.uuid4()),
        pump_id=payload.pump_id,
        cluster=payload.cluster,
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
        ood_suspected=ood_suspected,
        mahal_dist=round(mahal_dist, 4),
        causal_chain=causal_chain,
        limitation_flags=limitation_flags,
        top_shap_features=top_shap,
        m8p6_addendum=m8p6_addendum,
        fault_label_int=label_int,
        physics_context=phys if phys else {},
    )

    # ── M10 Phase 2.5 — append to server-side sensor history (multi-client) ──
    try:
        await request.app.state.history.append(
            window=payload.window,
            prediction=prediction.model_dump(),
        )
    except Exception as _hist_err:
        import logging
        logging.getLogger("pumpsmart").warning(f"history append failed: {_hist_err}")

    return prediction
