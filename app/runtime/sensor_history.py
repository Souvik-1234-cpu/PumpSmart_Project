# =============================================================================
# app/runtime/sensor_history.py
# PumpSmart v14.2 — M10 Phase 2.5
#
# SensorHistoryBuffer: RAM-only ring buffer for dashboard plotting.
# Retains last 24 hours of (raw sensor window + derived per-window metrics).
# Async-safe, multi-client safe (single source of truth — all clients
# polling /api/sensor_history read from this same buffer).
#
# Storage:
#   - Raw: last value of each 50-step window per channel (8 floats / second)
#   - Derived: score_A, score_B, cusum_Sn, theta_t, alert_state,
#     predicted_label_int, confidence_pct
#
# Memory budget at 1 Hz × 86,400 s × (8 + 7) floats × 4 bytes ≈ 5.2 MB
#
# Survival:
#   - NOT reset on /api/acknowledge (operators need to see what happened
#     BEFORE they acknowledged the alarm)
#   - Reset only via explicit /api/sensor_history/reset (admin)
#   - Wiped on server restart (P1 RAM-only policy)
# =============================================================================

import asyncio
import math
from collections import deque
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import numpy as np

# 24-hour retention at 1 Hz
DEFAULT_MAX_LEN = 86_400

# Channel order — LOCKED from M6B
CHANNEL_ORDER = [
    "Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
    "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV",
]
N_CHANNELS = 8


class SensorHistoryBuffer:
    """
    Async-safe ring buffer for sensor + inference history.
    Single instance per server — all clients read from the same source.

    Append cadence: once per inference call (~1 Hz in production).
    """

    def __init__(self, max_len: int = DEFAULT_MAX_LEN):
        self.max_len = max_len
        self._lock = asyncio.Lock()

        # Per-second snapshot — one row = (timestamp, 8 channels, derived metrics)
        self._timestamps   : deque = deque(maxlen=max_len)
        self._channels     : List[deque] = [deque(maxlen=max_len) for _ in range(N_CHANNELS)]

        # Derived metrics from the inference pipeline
        self._score_A      : deque = deque(maxlen=max_len)
        self._score_B      : deque = deque(maxlen=max_len)
        self._cusum_Sn     : deque = deque(maxlen=max_len)
        self._theta_t      : deque = deque(maxlen=max_len)
        self._alert_state  : deque = deque(maxlen=max_len)
        self._label_int    : deque = deque(maxlen=max_len)
        self._confidence   : deque = deque(maxlen=max_len)

        self._n_writes     : int   = 0

    # ── Public async interface ───────────────────────────────────────────────

    async def append(
        self,
        window: List[List[float]],
        prediction: Dict[str, Any],
        timestamp_utc: Optional[str] = None,
    ) -> None:
        """
        Append one inference call's data.

        Args:
            window       : the 50×8 SensorWindow.window submitted
            prediction   : the FaultPrediction dict (or model_dump output)
            timestamp_utc: ISO 8601; auto-generated if None
        """
        if timestamp_utc is None:
            timestamp_utc = datetime.now(timezone.utc).isoformat()

        # Use the LAST row of the 50-step window as the "current" sensor snapshot
        # (this is the most recent 1-second reading represented in this inference)
        if len(window) == 0:
            return
        last_row = window[-1]
        if len(last_row) != N_CHANNELS:
            return

        async with self._lock:
            self._timestamps.append(timestamp_utc)
            for ch_idx in range(N_CHANNELS):
                val = float(last_row[ch_idx])
                if not math.isfinite(val):
                    val = 0.0
                self._channels[ch_idx].append(val)

            self._score_A.append(float(prediction.get("score_A", 0.0) or 0.0))
            self._score_B.append(float(prediction.get("score_B", 0.0) or 0.0))
            self._cusum_Sn.append(float(prediction.get("cusum_Sn", 0.0) or 0.0))
            self._theta_t.append(float(prediction.get("adaptive_threshold", 0.0) or 0.0))
            self._alert_state.append(str(prediction.get("alert_state", "NORMAL")))

            # label_int may be in fault_label_int or extracted from fault_label
            lbl = prediction.get("fault_label_int")
            if lbl is None:
                lbl = -1
            self._label_int.append(int(lbl))
            self._confidence.append(float(prediction.get("confidence_pct", 0.0) or 0.0))

            self._n_writes += 1

    async def get_range(
        self,
        last_n_seconds: int = 3600,
        downsample: str = "adaptive",
        max_points: int = 500,
    ) -> Dict[str, Any]:
        """
        Return history slice for plotting.

        downsample policy (DS-C adaptive):
          - 'adaptive' : full resolution for last 5 min, LTTB for older data
          - 'full'     : no downsampling (use sparingly — large payload)
          - 'lttb'     : LTTB across entire range to max_points
          - 'stride'   : every Nth point (fast but loses spikes)
        """
        async with self._lock:
            total = len(self._timestamps)
            if total == 0:
                return _empty_response()

            # Determine slice
            n_requested = min(last_n_seconds, total)
            start_idx = total - n_requested

            ts_slice = list(self._timestamps)[start_idx:]
            ch_slices = [list(self._channels[i])[start_idx:] for i in range(N_CHANNELS)]
            sA  = list(self._score_A)[start_idx:]
            sB  = list(self._score_B)[start_idx:]
            cs  = list(self._cusum_Sn)[start_idx:]
            tt  = list(self._theta_t)[start_idx:]
            als = list(self._alert_state)[start_idx:]
            lbl = list(self._label_int)[start_idx:]
            cf  = list(self._confidence)[start_idx:]

        # Downsample outside the lock (CPU work, no shared state)
        if downsample == "full" or len(ts_slice) <= max_points:
            return _build_response(
                ts_slice, ch_slices, sA, sB, cs, tt, als, lbl, cf,
                method="full", original_n=len(ts_slice),
            )

        if downsample == "stride":
            stride = max(1, len(ts_slice) // max_points)
            idx = list(range(0, len(ts_slice), stride))
        elif downsample == "adaptive":
            # DS-C: last 300 points full-res, older → LTTB
            FULL_RES_TAIL = 300
            n = len(ts_slice)
            if n <= FULL_RES_TAIL:
                idx = list(range(n))
            else:
                older_budget = max(1, max_points - FULL_RES_TAIL)
                older_idx = _lttb_indices(ts_slice[:n - FULL_RES_TAIL],
                                          ch_slices[0][:n - FULL_RES_TAIL],
                                          older_budget)
                tail_idx = list(range(n - FULL_RES_TAIL, n))
                idx = older_idx + tail_idx
        else:   # 'lttb'
            idx = _lttb_indices(ts_slice, ch_slices[0], max_points)

        def _pick(arr): return [arr[i] for i in idx]

        return _build_response(
            _pick(ts_slice),
            [_pick(ch_slices[i]) for i in range(N_CHANNELS)],
            _pick(sA), _pick(sB), _pick(cs), _pick(tt),
            _pick(als), _pick(lbl), _pick(cf),
            method=downsample,
            original_n=len(ts_slice),
        )

    async def get_state(self) -> Dict[str, Any]:
        """Lightweight status (for /health)."""
        async with self._lock:
            return {
                "buffer_fill"      : len(self._timestamps),
                "buffer_capacity"  : self.max_len,
                "fill_pct"         : round(100 * len(self._timestamps) / self.max_len, 2),
                "n_writes_total"   : self._n_writes,
                "oldest_ts"        : self._timestamps[0]  if self._timestamps else None,
                "newest_ts"        : self._timestamps[-1] if self._timestamps else None,
            }

    async def reset(self) -> Dict[str, Any]:
        """Hard wipe — admin only. Returns state after wipe."""
        async with self._lock:
            self._timestamps.clear()
            for d in self._channels:
                d.clear()
            for d in (self._score_A, self._score_B, self._cusum_Sn,
                      self._theta_t, self._alert_state, self._label_int,
                      self._confidence):
                d.clear()
            self._n_writes = 0
            return {"reset": True, "buffer_fill": 0}


# =============================================================================
# Helpers
# =============================================================================
def _empty_response() -> Dict[str, Any]:
    return {
        "n_points"        : 0,
        "downsample_method": "n/a",
        "original_n"      : 0,
        "channels"        : {ch: [] for ch in CHANNEL_ORDER},
        "timestamps"      : [],
        "score_A"         : [],
        "score_B"         : [],
        "cusum_Sn"        : [],
        "theta_t"         : [],
        "alert_state"     : [],
        "label_int"       : [],
        "confidence_pct"  : [],
    }


def _build_response(ts, channels, sA, sB, cs, tt, als, lbl, cf,
                    method: str, original_n: int) -> Dict[str, Any]:
    return {
        "n_points"         : len(ts),
        "downsample_method": method,
        "original_n"       : original_n,
        "channels"         : {CHANNEL_ORDER[i]: channels[i] for i in range(N_CHANNELS)},
        "timestamps"       : ts,
        "score_A"          : sA,
        "score_B"          : sB,
        "cusum_Sn"         : cs,
        "theta_t"          : tt,
        "alert_state"      : als,
        "label_int"        : lbl,
        "confidence_pct"   : cf,
    }


def _lttb_indices(timestamps: List[str], values: List[float],
                  threshold: int) -> List[int]:
    """
    Largest-Triangle-Three-Buckets downsampling.
    Returns indices preserved (not values), so all parallel arrays
    can be downsampled identically.
    Reference: Sveinn Steinarsson, 2013, "Downsampling time series for
    visual representation."
    """
    n = len(values)
    if threshold >= n or threshold <= 2:
        return list(range(n))

    # Convert timestamps to numeric x-axis (use index since they're regular 1 Hz)
    sampled = [0]              # always keep first
    every = (n - 2) / (threshold - 2)
    a = 0

    for i in range(threshold - 2):
        # Next bucket boundaries
        avg_range_start = int(math.floor((i + 1) * every) + 1)
        avg_range_end   = int(math.floor((i + 2) * every) + 1)
        avg_range_end   = min(avg_range_end, n)

        if avg_range_end <= avg_range_start:
            sampled.append(avg_range_start)
            a = avg_range_start
            continue

        # Average of next bucket (x, y)
        avg_x = (avg_range_start + avg_range_end - 1) / 2.0
        avg_y = sum(values[avg_range_start:avg_range_end]) / (avg_range_end - avg_range_start)

        # Current bucket range
        range_offs = int(math.floor((i + 0) * every) + 1)
        range_to   = int(math.floor((i + 1) * every) + 1)
        range_to   = min(range_to, n)

        # Point a in fixed position
        point_a_x = a
        point_a_y = values[a]

        # Find point in current bucket forming largest triangle with a and avg
        max_area = -1.0
        max_area_point = range_offs
        for j in range(range_offs, range_to):
            area = abs(
                (point_a_x - avg_x) * (values[j] - point_a_y) -
                (point_a_x - j)     * (avg_y - point_a_y)
            ) * 0.5
            if area > max_area:
                max_area = area
                max_area_point = j

        sampled.append(max_area_point)
        a = max_area_point

    sampled.append(n - 1)      # always keep last
    return sampled
