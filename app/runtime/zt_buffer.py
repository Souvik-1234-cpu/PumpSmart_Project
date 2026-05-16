# =============================================================================
# app/runtime/zt_buffer.py
# z_t streaming buffer — feeds L2 TCN-AE with 63 consecutive z_t vectors.
# z_t comes from M4 LSTM-AE encoder hidden state (h_n, c_n concatenated).
# Resets only on /api/acknowledge.
# =============================================================================

import asyncio
from collections import deque
import numpy as np


class ZTBuffer:
    """
    Async-safe rolling buffer of z_t vectors from L1 LSTM-AE encoder.
    TCN-AE requires exactly zt_buffer_len (63) consecutive z_t vectors.
    Buffer is ready when len(buffer) == max_len.
    """

    def __init__(self, max_len: int = 63, z_dim: int = 64):
        self.max_len = max_len
        self.z_dim   = z_dim
        self._lock   = asyncio.Lock()
        self._buffer : deque[np.ndarray] = deque(maxlen=max_len)
        self._n_appended: int = 0

    async def append(self, zt: np.ndarray) -> None:
        """Add one z_t vector (shape: [z_dim])."""
        assert zt.shape == (self.z_dim,), (
            f"z_t shape mismatch: expected ({self.z_dim},), got {zt.shape}"
        )
        async with self._lock:
            self._buffer.append(zt.copy())
            self._n_appended += 1

    async def is_ready(self) -> bool:
        async with self._lock:
            return len(self._buffer) == self.max_len

    async def get_sequence(self) -> np.ndarray | None:
        """
        Returns z_t sequence as np.ndarray of shape [max_len, z_dim]
        if buffer is full, else None.
        """
        async with self._lock:
            if len(self._buffer) < self.max_len:
                return None
            return np.stack(list(self._buffer), axis=0)  # [63, z_dim]

    async def reset(self) -> None:
        """Called only from /api/acknowledge."""
        async with self._lock:
            self._buffer.clear()
            self._n_appended = 0

    async def get_state(self) -> dict:
        async with self._lock:
            return {
                "buffer_fill"    : len(self._buffer),
                "buffer_capacity": self.max_len,
                "is_ready"       : len(self._buffer) == self.max_len,
                "n_appended"     : self._n_appended,
                "z_dim"          : self.z_dim,
            }
