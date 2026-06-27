# =============================================================================
# app/runtime/alert_state_machine.py
# M12 Stage 4 (Steps 4.2+4.3) — stateful 4-state alert machine (D2 fix).
#
# Reads (does NOT mutate) locked RollingState (L4) + CUSUMState (L3) state dicts.
# Owns its own short-horizon score_A buffers (rolling_mean_100/200, slope).
#
# Standards: ISA-18.2 / IEC 62682 (alarm states, asymmetric hysteresis,
# rationalization), ISO 13374/13379-1 (state detection + fault-family logic),
# IEC 61511 (advisory only; DANGER = manual inspection, never auto-trip).
#
# score_C: ADVISORY-ONLY (Step 4.1: STRONG offline but serve-normal n=0 -> no
# validated live normal baseline). Annotates reason; drives NO transition.
# =============================================================================
import json
from collections import deque
from datetime import datetime
from pathlib import Path

# ── Thresholds (loaded once; defaults are M8-spec fallbacks if config absent) ─
_DEFAULTS = {
    "rolling_mean_100_floor": 0.095,   # M8 spec: mild seal/bearing WATCH floor
    "rolling_mean_200_floor": 0.085,
    "slope_warn":             0.0003,  # M8 spec: per-call drift slope (Mech-B)
    "drift_ratio_warn":       1.10,    # theta_t/theta_initial soft drift -> WARN
    "drift_ratio_danger":     1.50,    # aligned to RollingState lock_factor=1.5
                                       # (crosspoint). Below this = WARN-class drift.
    "cusum_watch":            2.0,     # mirror CUSUMState WATCH entry
    "clear_dwell":            300,     # consecutive clear calls to de-escalate
    "escalate_calls":         1,       # qualifying calls to escalate (fast)
}

ESCALATION = {"NORMAL": 0, "WATCH": 1, "WARN": 2, "DANGER": 3}
DEESCALATION_TARGET = {"DANGER": "WARN", "WARN": "WATCH", "WATCH": "NORMAL"}

GROUP_C_LABELS = {13, 14, 15, 16, 17, 22, 23}
GROUP_B_LABELS = {7, 8, 9, 10, 11, 12}
CAVITATION_LABELS = {3, 14, 18}


def load_alert_thresholds(path="models/M8_alert_thresholds.json"):
    cfg = dict(_DEFAULTS)
    try:
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
            sm = j.get("state_machine", {})
            for k in _DEFAULTS:
                if k in sm:
                    cfg[k] = sm[k]
            sc = j.get("score_C", {})
            cfg["score_C_reliability"] = sc.get("score_C_reliability", "UNUSABLE")
        else:
            cfg["score_C_reliability"] = "UNUSABLE"
    except Exception:
        cfg["score_C_reliability"] = "UNUSABLE"
    return cfg


# ── Mechanism triggers (Mech-A/B/C) — pure functions, fully testable ─────────
def mech_A_crosspoint(drift_ratio, drift_locked, cfg):
    """Mech-A: L4 CONFIRMED crosspoint lock -> DANGER.
    Requires drift_locked=True (RollingState's own lock at lock_factor=1.5).
    A bare drift_ratio without the confirmed lock is NOT acute: drift in the
    1.10-1.5 band is slow-drift (WARN-class, Invariant 12 dual time-scale),
    handled by the drift_ratio_warn path. DANGER on drift requires the locked
    crosspoint to agree with the L4 detector — the alarm layer never out-runs
    the detector it reads from."""
    return bool(drift_locked)


def mech_B_slope(slope, cfg):
    """Mech-B: sustained positive drift slope over the short horizon."""
    return bool(slope is not None and slope >= cfg["slope_warn"])


def mech_C_channel_drift(channel_drift_flags):
    """Mech-C: any per-channel drift flag set (masked-fault secondary path).
    channel_drift_flags: iterable of bool (len 8) or None."""
    if not channel_drift_flags:
        return False
    return bool(any(channel_drift_flags))


class AlertStateMachine:
    """Stateful 4-state alarm machine with asymmetric hysteresis.
    Reads locked RollingState + CUSUMState dicts; owns short-horizon score_A.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or load_alert_thresholds()
        self._sa_100 = deque(maxlen=100)
        self._sa_200 = deque(maxlen=200)
        self._state = "NORMAL"
        self._clear_streak = 0
        self._entered_ts = datetime.utcnow()
        self._last_reason = "init"
        self._n = 0

    # ── short-horizon features owned by the alarm layer ──────────────────────
    def _ingest(self, score_A):
        self._sa_100.append(float(score_A))
        self._sa_200.append(float(score_A))

    def _rolling_mean_100(self):
        return float(sum(self._sa_100) / len(self._sa_100)) if self._sa_100 else 0.0

    def _rolling_mean_200(self):
        return float(sum(self._sa_200) / len(self._sa_200)) if self._sa_200 else 0.0

    def _slope(self):
        n = len(self._sa_100)
        if n < 10:
            return 0.0
        y = list(self._sa_100)
        x = list(range(n))
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den = sum((xi - mx) ** 2 for xi in x) or 1e-9
        return float(num / den)

    def reset(self):
        self._sa_100.clear(); self._sa_200.clear()
        self._state = "NORMAL"; self._clear_streak = 0
        self._entered_ts = datetime.utcnow(); self._last_reason = "reset"

    def update(self, score_A, theta_t, theta_initial, drift_locked,
               cusum_Sn, cusum_alert, label_int=None,
               channel_drift_flags=None, score_C=None):
        """Compute next alarm state. Returns dict (state, reason, features)."""
        self._n += 1
        self._ingest(score_A)
        cfg = self.cfg

        rm100 = self._rolling_mean_100()
        rm200 = self._rolling_mean_200()
        slope = self._slope()
        drift_ratio = (theta_t / theta_initial) if theta_initial else 1.0

        mA = mech_A_crosspoint(drift_ratio, drift_locked, cfg)
        mB = mech_B_slope(slope, cfg)
        mC = mech_C_channel_drift(channel_drift_flags)

        # ── Candidate (instantaneous) severity from evidence ─────────────────
        reasons = []
        cand = "NORMAL"

        # DANGER evidence: acute MAE >> threshold, or hard crosspoint/Mech-A
        acute = score_A >= 1.5 * theta_t
        if acute:
            cand = "DANGER"; reasons.append(f"score_A {score_A:.4f} >= 1.5*theta_t {1.5*theta_t:.4f}")
        if mA:
            cand = "DANGER"; reasons.append("Mech-A crosspoint/hard-drift")

        # WARN evidence: score_A over threshold, rolling-mean floors, slope/Mech-B,
        # soft drift, Mech-C channel drift
        if cand != "DANGER":
            warn_hit = False
            if score_A >= theta_t:
                warn_hit = True; reasons.append(f"score_A {score_A:.4f} >= theta_t {theta_t:.4f}")
            if rm100 >= cfg["rolling_mean_100_floor"]:
                warn_hit = True; reasons.append(f"rolling_mean_100 {rm100:.4f} >= floor")
            if rm200 >= cfg["rolling_mean_200_floor"]:
                warn_hit = True; reasons.append(f"rolling_mean_200 {rm200:.4f} >= floor")
            if mB:
                warn_hit = True; reasons.append(f"Mech-B slope {slope:.5f} >= {cfg['slope_warn']}")
            if drift_ratio >= cfg["drift_ratio_warn"]:
                warn_hit = True; reasons.append(f"drift_ratio {drift_ratio:.3f} >= warn")
            if mC:
                warn_hit = True; reasons.append("Mech-C per-channel drift")
            if warn_hit:
                cand = "WARN"

        # WATCH evidence: CUSUM slow accumulator (label-21 path — may fire even
        # when score_A is sub-threshold; this is the gradual-wear detector)
        if cand == "NORMAL":
            if cusum_Sn >= cfg["cusum_watch"] or cusum_alert in ("WATCH", "ALARM", "DANGER"):
                cand = "WATCH"; reasons.append(f"CUSUM S_n {cusum_Sn:.3f} (gradual-wear path)")

        # ── Fault-family exceptions (ISO 13379-1) ────────────────────────────
        # Cavitation acute-shock fast-track: cavitation family + acute MAE -> DANGER
        if label_int in CAVITATION_LABELS and acute:
            cand = "DANGER"; reasons.append("cavitation acute-shock fast-track")

        # Group-C masked faults: a masked sensor must NOT auto-escalate to DANGER
        # on its own (sensor masking != confirmed multi-channel fault). Cap at WARN.
        capped = False
        if label_int in GROUP_C_LABELS and cand == "DANGER" and not acute:
            cand = "WARN"; capped = True
            reasons.append("Group-C masked fault: DANGER capped to WARN (single masked channel)")

        # Group-B compound phase-2: if compound label AND (Mech-B or CUSUM), allow
        # escalation one notch (chain entering secondary phase).
        if label_int in GROUP_B_LABELS and cand == "WATCH" and (mB or mC):
            cand = "WARN"; reasons.append("Group-B phase-2 escalation")

        # score_C: ADVISORY-ONLY — annotate, never change `cand`
        if score_C is not None and cfg.get("score_C_reliability") != "STRONG_VALIDATED":
            reasons.append(f"score_C={score_C:.4f} (advisory-only; no live normal baseline)")

        # ── Asymmetric hysteresis: escalate fast, de-escalate slow ───────────
        cur = ESCALATION[self._state]
        nxt = ESCALATION[cand]
        if nxt > cur:
            # escalate immediately
            self._state = cand
            self._clear_streak = 0
            self._entered_ts = datetime.utcnow()
            self._last_reason = "; ".join(reasons) or "escalate"
        elif nxt < cur:
            # candidate is calmer than current: require sustained clear dwell
            if cand == "NORMAL":
                self._clear_streak += 1
            else:
                # partial calm still counts toward de-escalating one level only
                self._clear_streak += 1
            if self._clear_streak >= cfg["clear_dwell"]:
                target = DEESCALATION_TARGET.get(self._state, "NORMAL")
                # de-escalate ONE level per dwell satisfaction (graded)
                self._state = target
                self._clear_streak = 0
                self._entered_ts = datetime.utcnow()
                self._last_reason = (f"de-escalate to {target} after "
                                     f"{cfg['clear_dwell']} clear calls")
            # else: hold current state (hysteresis)
        else:
            # same level — refresh reason if any evidence
            self._clear_streak = 0
            if reasons:
                self._last_reason = "; ".join(reasons)

        return {
            "alert_state": self._state,
            "candidate":   cand,
            "reason":      self._last_reason,
            "group_c_capped": capped,
            "features": {
                "rolling_mean_100": round(rm100, 6),
                "rolling_mean_200": round(rm200, 6),
                "slope":            round(slope, 7),
                "drift_ratio":      round(drift_ratio, 4),
                "mech_A": mA, "mech_B": mB, "mech_C": mC,
                "acute":  acute,
            },
            "clear_streak": self._clear_streak,
            "n_updates":    self._n,
        }
