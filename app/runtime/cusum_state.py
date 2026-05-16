# =============================================================================
# app/runtime/cusum_state.py
# L3 CUSUM accumulator — score_B input ONLY (Invariant 19).
# NEVER receives score_A. NEVER merged with rolling baseline (C-25).
# Resets ONLY on confirmed maintenance (/api/acknowledge).
# NEVER resets on adaptive threshold updates.
# =============================================================================

import asyncio
from datetime import datetime


class CUSUMState:
    """
    Async-safe persistent CUSUM accumulator.

    CUSUM update rule:
        S_n = max(0, S_{n-1} + score_B - k) * decay_factor

    Where:
        k           = reference value (slack) — filters low-magnitude drift
        H           = alarm threshold (default 5.0)
        decay_factor = exp(-λ) per step  — 7-day half-life (λ = 5.73e-05)

    Signal routing (Invariant 19):
        score_B → this class ONLY
        score_A → RollingState ONLY  (never here)
        score_C → M7 XGBoost ONLY   (never here)

    C-25 (Adaptive Threshold Paradox):
        L4 adaptive threshold updates NEVER reset S_n.
        Only /api/acknowledge (confirmed maintenance) resets S_n.
    """

    def __init__(self, H: float = 5.0, k: float = 0.5, lam: float = 5.73e-05):
        self.H   = H      # alarm threshold
        self.k   = k      # reference / slack
        self.lam = lam    # decay rate (7-day half-life)
        self._lock = asyncio.Lock()

        # State
        self._Sn            : float = 0.0
        self._n_updates     : int   = 0
        self._last_reset_ts : datetime | None = None
        self._watch_entered : datetime | None = None  # when Sn first crossed 2.0
        self._alarm_count   : int   = 0

        # Decay factor per step (pre-computed)
        import math
        self._decay = math.exp(-self.lam)

    # ── Public async interface ───────────────────────────────────────────────

    async def update(self, score_B: float) -> dict:
        """
        Ingest one score_B value. Returns current state dict.
        score_B must come from L2 TCN-AE drift slope output ONLY.
        """
        async with self._lock:
            prev_Sn = self._Sn

            # CUSUM with exponential decay
            self._Sn = max(0.0, (self._Sn + score_B - self.k) * self._decay)
            self._n_updates += 1

            # Track WATCH entry (S_n crossed 2.0 for first time)
            if prev_Sn < 2.0 and self._Sn >= 2.0:
                self._watch_entered = datetime.utcnow()

            # Count alarms (S_n crossed H)
            if prev_Sn < self.H and self._Sn >= self.H:
                self._alarm_count += 1

            return self._state_dict()

    async def reset(self, reason: str = "maintenance_acknowledged") -> dict:
        """
        Reset S_n to zero. ONLY callable from /api/acknowledge.
        C-25: adaptive threshold updates must NEVER call this.
        """
        async with self._lock:
            self._Sn            = 0.0
            self._watch_entered = None
            self._last_reset_ts = datetime.utcnow()
            return {**self._state_dict(), "reset_reason": reason}

    async def get_state(self) -> dict:
        async with self._lock:
            return self._state_dict()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _state_dict(self) -> dict:
        alert = (
            "DANGER" if self._Sn >= self.H * 1.5
            else "ALARM"  if self._Sn >= self.H
            else "WATCH"  if self._Sn >= 2.0
            else "NORMAL"
        )
        return {
            "cusum_Sn"          : round(self._Sn, 4),
            "cusum_H"           : self.H,
            "cusum_alert"       : alert,
            "n_updates"         : self._n_updates,
            "alarm_count"       : self._alarm_count,
            "last_reset_utc"    : self._last_reset_ts.isoformat() if self._last_reset_ts else None,
            "watch_entered_utc" : self._watch_entered.isoformat() if self._watch_entered else None,
        }
