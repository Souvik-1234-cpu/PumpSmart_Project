# =============================================================================
# app/runtime/drift_signal.py  — M12 Stage 4 (label-21 detectability fix)
# Slow-drift signal for L3 CUSUM.
#
# WHY THIS EXISTS
# ---------------
# CUSUM was wired to score_B (TCN second-half-minus-first-half MAE delta). The
# live probe PROVED score_B cannot separate label-21 from normal:
#     normal  score_B p99 = 0.0285
#     label21 score_B p99 = 0.0260   (overlapping — no threshold separates them)
# score_B is a within-buffer CHANGE detector; gradual wear (0.0003/step over
# 1000 steps) barely moves reconstruction error within a 52-min buffer.
#
# The probe then proved the drift IS visible in score_A's SLOPE:
#     normal  score_A slope = 0.0  (flat, all 15 sequences)
#     label21 score_A slope = 0.00015 median (0.000045 min — every seq positive)
# Gradual wear = slow monotone creep in reconstruction-error LEVEL → its SLOPE
# is the separable signal. This class derives that slope and feeds it to CUSUM.
#
# INVARIANT 19 (preserved in intent): L3 remains the slow-drift accumulator.
# We only correct WHICH drift signal it integrates — score_A SLOPE (a true
# drift measure) instead of score_B (which carried no drift content). score_A
# itself still flows to L4 RollingState unchanged; DriftSignal observes a copy
# to derive the trend, it does not route score_A into CUSUM raw.
#
# CALIBRATION (from module_12_stage4_probe_scoreA_drift):
#     mu0_slope = 0.0       (normal slope floor, measured)
#     k_slope   = 0.00005   (above normal 0.0, below label21 p50 0.00015)
#   → normal evidence = slope - 0 - 0.00005 <= 0  (no false accumulate → G4b)
#   → label21 evidence = 0.00015 - 0.00005 = +0.0001 > 0  (accumulates → G4a)
#   Both gates separable SIMULTANEOUSLY — impossible with score_B.
# =============================================================================

import asyncio
from collections import deque
import numpy as np

# Slope window: label-21 ramps over ~20 windows (1000 steps / 50). 30 captures
# the ramp with margin while staying robust to single-window noise.
DEFAULT_SLOPE_WINDOW = 30
# smooth_window=1 => RAW score_A (probe proved normal raw slope = 0.0). Increase
# only if real-world score_A is noisier than synthetic and normal slope lifts
# off zero (defensive config; no code change needed to enable smoothing).
DEFAULT_SMOOTH_WINDOW = 1


class DriftSignal:
    """
    Async-safe slow-drift signal: ingests score_A, emits least-squares slope
    over the recent window. Output is the L3 CUSUM input (replaces score_B).
    """

    def __init__(self, slope_window: int = DEFAULT_SLOPE_WINDOW,
                 smooth_window: int = DEFAULT_SMOOTH_WINDOW):
        self.slope_window  = int(slope_window)
        self.smooth_window = max(1, int(smooth_window))
        self._lock = asyncio.Lock()
        self._sa   = deque(maxlen=self.slope_window + self.smooth_window)
        self._n    = 0

    async def update(self, score_A: float) -> dict:
        """Ingest one score_A; return {'slope': float, 'ready': bool, 'n': int}."""
        async with self._lock:
            self._sa.append(float(score_A))
            self._n += 1
            slope = self._compute_slope()
            # Need a full slope window before the slope is trustworthy; until
            # then report ready=False so CUSUM treats it as provisional.
            ready = len(self._sa) >= self.slope_window
            return {"slope": round(slope, 8), "ready": ready, "n": self._n}

    def _compute_slope(self) -> float:
        vals = list(self._sa)
        if len(vals) < 5:
            return 0.0
        # optional smoothing (default raw)
        if self.smooth_window > 1 and len(vals) >= self.smooth_window:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            vals = list(np.convolve(np.array(vals), kernel, mode="valid"))
            if len(vals) < 5:
                return 0.0
        y = np.array(vals[-self.slope_window:], dtype=float)
        x = np.arange(len(y), dtype=float)
        mx = x.mean(); my = y.mean()
        den = ((x - mx) ** 2).sum()
        if den <= 0:
            return 0.0
        return float(((x - mx) * (y - my)).sum() / den)

    async def reset(self) -> None:
        """Called from /api/acknowledge alongside the other L3/L4 resets."""
        async with self._lock:
            self._sa.clear()
            self._n = 0

    async def get_state(self) -> dict:
        async with self._lock:
            return {"slope": round(self._compute_slope(), 8),
                    "fill": len(self._sa), "slope_window": self.slope_window,
                    "smooth_window": self.smooth_window, "n": self._n}
