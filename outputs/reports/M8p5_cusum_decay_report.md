# M8 Patch 5a — CUSUM Auto-Decay Policy
**Date:** 2026-05-09

## Why this patch existed
Per the original M8 spec, CUSUM resets only on confirmed maintenance event.
If an operator investigates a WATCH alert and finds nothing wrong (the most
common real outcome), there was no path through the spec for S_n to come
back down. It accumulated forever -> WATCH fires on every call -> alarm
fatigue -> real Label 21 ignored.

This is the canonical "alarm fatigue induction" failure mode. It is FM-04 in
the model FMEA.

## What this patch does
Adds three reset mechanisms (selectable per deployment):

1. **Geometric quiet decay (recommended, default):**
   `S_n_new = S_n_current * (1 - λ)` applied per call when no positive
   fault evidence is present. λ = 5.730219e-05 → 7-day half-life
   on a fully quiet pump.

2. **Operator-investigated reset (NEW endpoint /api/cusum_quiet_review):**
   Discrete reset to 0.3 * S_n when operator confirms WATCH was investigated
   and no fault found. Audit logged.

3. **Maintenance reset (existing /api/acknowledge):**
   Hard reset to 0.0. Unchanged from existing M10 spec.

## Why these are mathematically safe for Label 21 detection

Decay erodes S_n only on **quiet** calls. A persistent fault produces positive
evidence faster than decay erodes — accumulation continues. The asymptote
of S_n on a fault producing +0.01/call evidence with this λ is approximately
0.01/λ ≈ 174.5, far above H=5.0. Detection
latency for genuine slow drift is preserved.

## Tunable parameters
| Parameter | Default | Where to tune |
|---|---|---|
| λ (decay rate) | 5.730219e-05 | M8p5_cusum_runtime_policy.json |
| Half-life (calls) | 12,096 | derived from λ |
| Operator-reset factor | 0.30 | M8p5_cusum_runtime_policy.json |
| Quiet-detection condition | score_B ≤ μ₀ + k | hardcoded (mathematically required) |

## Validation plan (first 60 days of deployment)
- Log: WATCH count, /api/cusum_quiet_review count, /api/acknowledge count
- If WATCH count > 1/day and /api/acknowledge = 0 → alarm fatigue in progress, raise λ ×1.5
- If a real Label 21 is missed → reduce λ ×0.5 to preserve evidence longer
- Tune after 60 days of real operating data, not earlier

## Files written
- `models/M8p5_cusum_runtime_policy.json` (M10 reads at startup)

## M10 runtime code skeleton (must be implemented in app/)

```python
# In app/runtime/cusum_state.py
import json
from pathlib import Path

CUSUM_POLICY_PATH = Path('models/M8p5_cusum_runtime_policy.json')

class CUSUMState:
    def __init__(self):
        cfg = json.load(open(CUSUM_POLICY_PATH))
        p = cfg['cusum_parameters']
        d = cfg['decay_policy']['geometric_quiet_decay']
        self.mu0 = p['mu0_B']
        self.k   = p['k']
        self.H   = p['H']
        self.lam = d['lambda']
        self.S_n = 0.0
        self.fired = False

    def update(self, score_B):
        evidence = score_B - self.mu0 - self.k
        if evidence > 0:
            self.S_n = self.S_n + evidence
        else:
            self.S_n = self.S_n * (1 - self.lam)   # decay
        self.S_n = max(0.0, self.S_n)
        if self.S_n > self.H and not self.fired:
            self.fired = True
            return 'WATCH'
        return None

    def operator_quiet_review(self, factor=0.30):
        self.S_n = self.S_n * factor
        self.fired = False
        # MUST log: timestamp, operator_id, reason, S_n_before, S_n_after

    def maintenance_reset(self):
        self.S_n = 0.0
        self.fired = False
```
