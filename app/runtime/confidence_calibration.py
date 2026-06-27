# =============================================================================
# app/runtime/confidence_calibration.py
# M12 Stage 4 — physics-honest confidence calibration (NO retrain).
#
# WHY THIS EXISTS
# ---------------
# M7 v3 is XGBoost multi:softprob, 504 trees @ depth 7, trained on clean
# physics-synthetic data with NO calibration layer. On perfectly-separable
# synthetic classes the softmax saturates to 1.000 → the dashboard shows
# "100.0%" confidence. That is physically absurd: nothing governed by
# thermodynamics / fluid mechanics is ever certain, and C-26 explicitly states
# real-world performance is 0.65–0.85, far below the synthetic 0.9965 F1.
# A served 100% destroys stakeholder trust and contradicts the disclaimer.
#
# FIX (two stages, applied to the SERVED probability only — model untouched):
#   1. TEMPERATURE SCALING. Recover pseudo-logits z = log(p + eps), soften by
#      z/T with T > 1, re-softmax. This spreads the saturated 1.000 mass back
#      into a real distribution: a marginal/early-onset fault lands lower than a
#      textbook matured fault. RELATIVE ordering is preserved — that ordering is
#      what an operator reads as "how sure is it".
#   2. HARD CEILING. After softening, clip the displayed top-class confidence at
#      CONF_CEILING (default 0.94). This is the physics-honesty cap: the system
#      never claims near-certainty on a single-pump synthetic-trained model.
#
# Order matters: temperature FIRST (creates a real spread), ceiling SECOND
# (clips only the genuine top). A ceiling alone would make every fault show 94%
# — a different lie. Temperature alone can still touch ~0.99 on the cleanest
# synthetic windows. Both together = honest spread with an honest cap.
#
# T is CALIBRATED, not guessed — fit by module_12_stage4_confidence_fit.py
# against held-out M7 probabilities so that displayed confidence tracks real
# windowed accuracy (reliability). The fitted value is written to
# models/M7_confidence_calibration.json and loaded at startup. If the file is
# absent, a conservative default T is used and a warning is logged.
#
# This module changes NOTHING about the predicted LABEL (argmax is invariant
# under monotonic temperature scaling) — only the confidence NUMBER. Invariant
# 19, the 7-field contract, and M7's class decision are all untouched.
# =============================================================================

import json
import math
from pathlib import Path

import numpy as np

_CALIB_PATH = Path("models") / "M7_confidence_calibration.json"

# Conservative defaults if the fitted file is absent. T>1 softens; ceiling caps.
_DEFAULT_TEMPERATURE = 2.5      # softens saturated softprob into a real spread
_DEFAULT_CEILING     = 0.94     # physics-honest cap (≈ never claim certainty)
_DEFAULT_FLOOR       = 0.0      # no floor by default (UNKNOWN_FAULT < 70% logic
                                # in anomaly.py still applies on the calibrated %)
_EPS = 1e-12


def load_calibration(path: Path = _CALIB_PATH) -> dict:
    """
    Load fitted calibration. Falls back to conservative defaults if absent.
    Returns dict with: temperature, ceiling, floor, source, fitted (bool).
    """
    cfg = {
        "temperature": _DEFAULT_TEMPERATURE,
        "ceiling":     _DEFAULT_CEILING,
        "floor":       _DEFAULT_FLOOR,
        "source":      "defaults (calibration file absent)",
        "fitted":      False,
    }
    try:
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
            cfg["temperature"] = float(j.get("temperature", cfg["temperature"]))
            cfg["ceiling"]     = float(j.get("ceiling",     cfg["ceiling"]))
            cfg["floor"]       = float(j.get("floor",       cfg["floor"]))
            cfg["source"]      = str(p)
            cfg["fitted"]      = bool(j.get("fitted", True))
    except Exception:
        pass
    # Guard rails — a bad config must never produce a >1 or <0 confidence.
    cfg["temperature"] = max(1.0, float(cfg["temperature"]))   # T<1 would sharpen
    cfg["ceiling"]     = min(0.99, max(0.50, float(cfg["ceiling"])))
    cfg["floor"]       = min(cfg["ceiling"], max(0.0, float(cfg["floor"])))
    return cfg


def temperature_soften(proba: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to a probability vector from softprob.

    proba already sums to 1 (XGBoost softprob). Recover pseudo-logits via
    log(p), divide by T, re-softmax. argmax is invariant (label unchanged).
    """
    p = np.asarray(proba, dtype=np.float64)
    p = np.clip(p, _EPS, 1.0)
    logits = np.log(p)
    logits = logits / float(max(1.0, temperature))
    # stable softmax
    logits -= logits.max()
    ex = np.exp(logits)
    return ex / ex.sum()


def calibrate_confidence(proba: np.ndarray, cfg: dict) -> dict:
    """
    Full serve-time calibration of one prediction's probability vector.

    Returns dict:
      label_int            : argmax (UNCHANGED by calibration)
      raw_confidence       : original softprob top-class probability (0–1)
      calibrated_confidence: temperature-softened then ceiling-capped (0–1)
      calibrated_proba     : full softened vector (pre-ceiling, sums to 1)
      temperature, ceiling : applied values (for transparency/logging)

    The LABEL is taken from the RAW proba argmax so the model's decision is
    never altered. Only the confidence NUMBER is calibrated.
    """
    raw = np.asarray(proba, dtype=np.float64)
    label_int = int(np.argmax(raw))
    raw_conf  = float(raw[label_int])

    softened = temperature_soften(raw, cfg["temperature"])
    soft_conf = float(softened[label_int])

    # Hard ceiling — physics-honest cap. Floor only lifts genuinely tiny values
    # if a floor is configured (default 0.0 → no lift).
    capped = min(cfg["ceiling"], soft_conf)
    capped = max(cfg["floor"], capped)

    return {
        "label_int":             label_int,
        "raw_confidence":        round(raw_conf, 6),
        "calibrated_confidence": round(capped, 6),
        "calibrated_proba":      softened,
        "temperature":           cfg["temperature"],
        "ceiling":               cfg["ceiling"],
    }
