# =============================================================================
# app/runtime/rolling_state.py
# L4 Adaptive Threshold — score_A input ONLY (Invariant 19).
# θ_t = rolling_mean + 3 × rolling_std
# Crosspoint guard: if θ_t > 1.5 × θ_initial → LOCK + raise DRIFT ALERT
# NEVER resets CUSUM (C-25 — Adaptive Threshold Paradox).
# =============================================================================

import asyncio
from collections import deque
from datetime import datetime
import numpy as np


class RollingState:
    """
    Async-safe rolling baseline for score_A (Layer 4).

    Invariant 19: only score_A enters this class.
    C-25: threshold updates in this class NEVER reset CUSUMState.S_n.
    """

    def __init__(
        self,
        window_size   : int   = 432,       # 6-hour window @ 50-s intervals
        theta_initial : float = 1.881275,  # locked θ_initial from M8 config
        lock_factor   : float = 1.5,       # crosspoint guard multiplier
        sigma_mult    : float = 3.0,       # θ_t = μ + sigma_mult × σ
    ):
        self.window_size   = window_size
        self.theta_initial = theta_initial
        self.lock_factor   = lock_factor
        self.sigma_mult    = sigma_mult
        self._lock         = asyncio.Lock()

        self._buffer   : deque[float] = deque(maxlen=window_size)
        self._theta_t  : float = theta_initial
        self._locked   : bool  = False        # True when crosspoint guard fires
        self._lock_ts  : datetime | None = None
        self._n_updates: int = 0

    # ── Public async interface ───────────────────────────────────────────────

    async def update(self, score_A: float) -> dict:
        """
        Ingest one score_A value. Recompute θ_t. Apply crosspoint guard.
        Returns current state dict (does NOT touch CUSUMState).
        """
        async with self._lock:
            self._buffer.append(score_A)
            self._n_updates += 1

            if len(self._buffer) >= 10:   # need minimum samples for stable σ
                arr  = np.array(self._buffer)
                mu   = float(arr.mean())
                sig  = float(arr.std())
                new_theta = mu + self.sigma_mult * sig

                # Crosspoint guard — C-25 / C-16
                if not self._locked:
                    self._theta_t = new_theta
                    if new_theta > self.lock_factor * self.theta_initial:
                        self._locked  = True
                        self._lock_ts = datetime.utcnow()
                # If already locked: θ_t stays at lock value — operator must acknowledge

            return self._state_dict(score_A)

    async def reset(self) -> dict:
        """
        Reset rolling buffer and unlock. Called ONLY from /api/acknowledge.
        Never called by CUSUM or any other internal component.
        """
        async with self._lock:
            self._buffer.clear()
            self._theta_t = self.theta_initial
            self._locked  = False
            self._lock_ts = None
            return self._state_dict(None)

    async def get_state(self) -> dict:
        async with self._lock:
            return self._state_dict(None)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _state_dict(self, latest_score_A) -> dict:
        n = len(self._buffer)
        return {
            "theta_t"           : round(self._theta_t, 6),
            "theta_initial"     : self.theta_initial,
            "drift_locked"      : self._locked,
            "lock_timestamp_utc": self._lock_ts.isoformat() if self._lock_ts else None,
            "buffer_fill"       : n,
            "buffer_capacity"   : self.window_size,
            "latest_score_A"    : round(latest_score_A, 6) if latest_score_A is not None else None,
            "n_updates"         : self._n_updates,
        }
