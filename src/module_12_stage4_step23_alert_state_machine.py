# =============================================================================
# src/module_12_stage4_step23_alert_state_machine.py
# PumpSmart v14.2 — M12 Stage 4, Steps 4.2 + 4.3 (COMBINED)
#   4.2  mech_triggers (Mech-A/B/C) + short-horizon rolling features
#   4.3  full AlertStateMachine (NORMAL/WATCH/WARN/DANGER) — the D2 fix
# =============================================================================
#
# WHY COMBINED
# ------------
# mech_triggers + rolling-mean/slope/drift_ratio features (4.2) exist ONLY to
# feed compute_alert_state (4.3). Testing them apart needs two throwaway gate
# harnesses. Combined => one coherent module + one transition-matrix gate.
#
# ARCHITECTURE DECISION (Souvik-approved: new class, no edits to locked code)
# --------------------------------------------------------------------------
# The locked production classes (RollingState L4, CUSUMState L3) are NOT edited
# — they carry the C-25 / Invariant-19 guarantees that the G11 gate proves. The
# alarm layer is a SEPARATE concern (ISO 13374 Health-Assessment boundary):
#   * AlertStateMachine owns its OWN short-horizon score_A buffers
#     (rolling_mean_100 / rolling_mean_200 / slope) — features the ALARM layer
#     derives, not detector state.
#   * It READS theta_t, theta_initial, drift_locked from RollingState's existing
#     state dict, and cusum_Sn / cusum_alert from CUSUMState's existing dict.
#   * drift_ratio = theta_t / theta_initial — both already exposed; no edit.
# => zero blast radius on locked classes; the G11 C-25 guarantee is untouched.
#
# INDUSTRY-STANDARD ALARM BEHAVIOUR (ISA-18.2 / IEC 62682, ISO 13379-1)
# ---------------------------------------------------------------------
#   * Four states with explicit entry/exit. No memoryless re-evaluation.
#   * ASYMMETRIC hysteresis: escalate fast (1 qualifying call), de-escalate slow
#     (sustained CLEAR_DWELL consecutive clear calls). Prevents chatter.
#   * Alarm rationalization: every transition traces to a named trigger
#     (rolling-mean floor, slope/Mech-B, CUSUM WATCH, crosspoint/Mech-A,
#     per-channel drift/Mech-C). Recorded in `reason`.
#   * Fault-family exceptions (ISO 13379-1): cavitation acute-shock fast-track;
#     Group-C masked faults capped at WARN (sensor masking => never auto-DANGER
#     on a single masked channel); Group-B phase-2 escalation; label-21 CUSUM
#     path (CUSUM may drive WATCH/WARN even when score_A is sub-threshold).
#   * IEC 61511: advisory only — DANGER = "immediate manual inspection", never
#     an automated trip. L5 unreachable for single-model (per role spec).
#
# score_C POLICY (from Step 4.1 — M8_alert_thresholds.json)
# ---------------------------------------------------------
# 4.1 measured score_C STRONG offline (fill-matched AUC 0.95) BUT serve-normal
# n=0: at live stride-50 the buffer never fills on normal ops, so score_C has NO
# validated live normal baseline. Per ISA-18.2 (a threshold you cannot validate
# against live-route normal data cannot drive a high-severity alarm), score_C is
# pinned ADVISORY-ONLY here regardless of the STRONG label: it annotates the
# reason string and the response, but drives NO state transition. If a future
# step validates a live normal baseline, reliability policy can promote it.
#
# DELIVERABLE (test-harness-first — production patch only on full gate PASS)
# -------------------------------------------------------------------------
# This script:
#   1. Defines AlertStateMachine + mech_triggers INLINE (candidate production code).
#   2. Runs a STATE-TRANSITION MATRIX gate over synthetic score_A/cusum/drift
#      trajectories exercising every transition + every fault-family exception +
#      a no-chatter test.
#   3. ONLY on full PASS, writes the production module
#      app/runtime/alert_state_machine.py and prints the exact anomaly.py patch
#      (NOT auto-applied — Souvik applies after reviewing the gate).
#
# RUN (CWD-independent)
#   .venv\Scripts\Activate.ps1
#   python src/module_12_stage4_step23_alert_state_machine.py
# =============================================================================

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import (DEVICE, IS_GPU, MODEL_DIR, OUTPUT_DIR)

from datetime import date, datetime
import json
import warnings
import traceback
from collections import deque
warnings.filterwarnings("ignore")

import numpy as np

SCRIPT_NAME = "module_12_stage4_step23_alert_state_machine"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def _abspath(p, default_rel):
    p = Path(p)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        alt = _ROOT / default_rel
        if alt.exists():
            return alt
    return p

ALERT_CFG_PATH = _abspath(MODEL_DIR / "M8_alert_thresholds.json", "models/M8_alert_thresholds.json")
M8_CFG_PATH    = _abspath(MODEL_DIR / "M8_threshold_config.json",  "models/M8_threshold_config.json")
PROD_MODULE    = _ROOT / "app" / "runtime" / "alert_state_machine.py"

GROUP_C_LABELS = {13, 14, 15, 16, 17, 22, 23}   # masked faults (cap at WARN)
GROUP_B_LABELS = {7, 8, 9, 10, 11, 12}           # compound chains (phase-2 escalate)
CAVITATION_LABELS = {3, 14, 18}                  # acute shock / cavitation family


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# CANDIDATE PRODUCTION CODE (inline here; written to app/ only on gate PASS)
# =============================================================================
# The body below is the SINGLE SOURCE for both the in-script gate and the
# emitted production module — they are byte-identical (the production file is
# generated from PROD_SRC). This guarantees test == serve.
# =============================================================================
PROD_SRC = r'''# =============================================================================
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
'''


# =============================================================================
# Import the candidate code into THIS process for the gate (exec PROD_SRC)
# =============================================================================
def _load_candidate():
    ns = {}
    exec(compile(PROD_SRC, "<alert_state_machine_candidate>", "exec"), ns)
    return ns["AlertStateMachine"], ns["load_alert_thresholds"]


# =============================================================================
# STATE-TRANSITION MATRIX GATE
# =============================================================================
def run_gate(AlertStateMachine, cfg):
    """Exercise every transition + fault-family exception + no-chatter.
    Returns dict of subgate -> (status, detail)."""
    g = {}
    THETA0 = 0.110058           # locked theta_initial
    THETA  = THETA0             # baseline theta_t (drift_ratio = 1.0)

    # Use a small clear_dwell for the gate so de-escalation is testable quickly,
    # but assert the PRODUCTION value separately.
    test_cfg = dict(cfg); test_cfg["clear_dwell"] = 5

    def fresh():
        return AlertStateMachine(cfg=test_cfg)

    # ── T1: NORMAL on quiet input ────────────────────────────────────────────
    try:
        sm = fresh()
        r = None
        for _ in range(20):
            r = sm.update(score_A=0.02, theta_t=THETA, theta_initial=THETA0,
                          drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")
        g["T1_normal_quiet"] = (PASS if r["alert_state"] == "NORMAL" else FAIL,
                                f"state={r['alert_state']}")
    except Exception as e:
        g["T1_normal_quiet"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T2: WATCH via CUSUM (label-21 gradual path, score_A sub-threshold) ────
    try:
        sm = fresh()
        r = sm.update(score_A=0.02, theta_t=THETA, theta_initial=THETA0,
                      drift_locked=False, cusum_Sn=3.0, cusum_alert="WATCH",
                      label_int=21)
        g["T2_watch_cusum"] = (PASS if r["alert_state"] == "WATCH" else FAIL,
                               f"state={r['alert_state']} reason={r['reason'][:50]}")
    except Exception as e:
        g["T2_watch_cusum"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T3: WARN via score_A >= theta_t ──────────────────────────────────────
    try:
        sm = fresh()
        r = sm.update(score_A=0.12, theta_t=THETA, theta_initial=THETA0,
                      drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")
        g["T3_warn_scoreA"] = (PASS if r["alert_state"] == "WARN" else FAIL,
                               f"state={r['alert_state']}")
    except Exception as e:
        g["T3_warn_scoreA"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T4: WARN via rolling_mean_100 floor (mild sustained, sub-instant) ─────
    # NOTE: theta_t held near theta_initial (drift_ratio~1.0) so Mech-A does NOT
    # fire — we are isolating the rolling-mean-floor path, not drift.
    try:
        sm = fresh()
        r = None
        # sustained ~0.097: above rm100 floor 0.095 but below theta_t 0.11
        for _ in range(60):
            r = sm.update(score_A=0.097, theta_t=0.110, theta_initial=THETA0,
                          drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")
        ok = r["alert_state"] in ("WARN",) and r["features"]["rolling_mean_100"] >= cfg["rolling_mean_100_floor"]
        g["T4_warn_rollmean"] = (PASS if ok else FAIL,
                                 f"state={r['alert_state']} rm100={r['features']['rolling_mean_100']:.4f} "
                                 f"drift_ratio={r['features']['drift_ratio']:.3f}")
    except Exception as e:
        g["T4_warn_rollmean"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T5: DANGER via acute score_A >= 1.5*theta_t ──────────────────────────
    try:
        sm = fresh()
        r = sm.update(score_A=0.20, theta_t=0.11, theta_initial=THETA0,
                      drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")
        g["T5_danger_acute"] = (PASS if r["alert_state"] == "DANGER" else FAIL,
                                f"state={r['alert_state']}")
    except Exception as e:
        g["T5_danger_acute"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T6: DANGER via Mech-A crosspoint (drift_locked) ──────────────────────
    try:
        sm = fresh()
        r = sm.update(score_A=0.05, theta_t=0.18, theta_initial=THETA0,
                      drift_locked=True, cusum_Sn=0.0, cusum_alert="NORMAL")
        g["T6_danger_mechA"] = (PASS if r["alert_state"] == "DANGER" else FAIL,
                                f"state={r['alert_state']}")
    except Exception as e:
        g["T6_danger_mechA"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T7: asymmetric hysteresis — escalate fast, hold then de-escalate slow ─
    try:
        sm = fresh()
        sm.update(score_A=0.20, theta_t=0.11, theta_initial=THETA0,
                  drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")  # -> DANGER
        # now quiet: must HOLD DANGER for < clear_dwell, then de-escalate one level
        held = []
        for _ in range(4):   # < clear_dwell(5)
            r = sm.update(score_A=0.01, theta_t=0.11, theta_initial=THETA0,
                          drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")
            held.append(r["alert_state"])
        r5 = sm.update(score_A=0.01, theta_t=0.11, theta_initial=THETA0,
                       drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")  # 5th clear
        ok = all(s == "DANGER" for s in held) and r5["alert_state"] == "WARN"
        g["T7_asym_hysteresis"] = (PASS if ok else FAIL,
                                   f"held={held} after_dwell={r5['alert_state']}")
    except Exception as e:
        g["T7_asym_hysteresis"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T8: no chatter — alternating mild input must not oscillate state ──────
    try:
        sm = fresh()
        sm.update(score_A=0.20, theta_t=0.11, theta_initial=THETA0,
                  drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")  # DANGER
        states = []
        for i in range(10):
            sa = 0.01 if i % 2 == 0 else 0.20   # alternate calm/acute
            r = sm.update(score_A=sa, theta_t=0.11, theta_initial=THETA0,
                          drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL")
            states.append(r["alert_state"])
        # must never drop below DANGER (acute re-arrives before dwell completes)
        ok = all(s == "DANGER" for s in states)
        g["T8_no_chatter"] = (PASS if ok else FAIL, f"states={states}")
    except Exception as e:
        g["T8_no_chatter"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T9: Group-C masked fault — DANGER (non-acute) capped to WARN ──────────
    try:
        sm = fresh()
        # drift_locked would push DANGER, but Group-C non-acute must cap at WARN
        r = sm.update(score_A=0.05, theta_t=0.18, theta_initial=THETA0,
                      drift_locked=True, cusum_Sn=0.0, cusum_alert="NORMAL",
                      label_int=13)
        ok = r["alert_state"] == "WARN" and r["group_c_capped"]
        g["T9_groupC_cap"] = (PASS if ok else FAIL,
                              f"state={r['alert_state']} capped={r['group_c_capped']}")
    except Exception as e:
        g["T9_groupC_cap"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T10: Group-C masked fault WITH acute MAE still allowed to DANGER ──────
    try:
        sm = fresh()
        r = sm.update(score_A=0.20, theta_t=0.11, theta_initial=THETA0,
                      drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL",
                      label_int=14)
        # label 14 is cavitation-masked; acute => cavitation fast-track DANGER
        g["T10_groupC_acute_danger"] = (PASS if r["alert_state"] == "DANGER" else FAIL,
                                        f"state={r['alert_state']}")
    except Exception as e:
        g["T10_groupC_acute_danger"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T11: cavitation acute-shock fast-track ────────────────────────────────
    try:
        sm = fresh()
        r = sm.update(score_A=0.18, theta_t=0.11, theta_initial=THETA0,
                      drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL",
                      label_int=3)
        g["T11_cavitation_fasttrack"] = (PASS if r["alert_state"] == "DANGER" else FAIL,
                                        f"state={r['alert_state']}")
    except Exception as e:
        g["T11_cavitation_fasttrack"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T12: Group-B phase-2 escalation (WATCH + Mech-B -> WARN) ──────────────
    # theta_t held near theta_initial (drift_ratio~1.0, no Mech-A); rising score_A
    # capped BELOW theta_t so the escalation comes from slope (Mech-B), not acute.
    try:
        sm = fresh()
        r = None
        for i in range(40):
            sa = 0.02 + i * 0.0008   # rising -> positive slope (Mech-B); max ~0.05 < 0.11
            r = sm.update(score_A=sa, theta_t=0.110, theta_initial=THETA0,
                          drift_locked=False, cusum_Sn=2.5, cusum_alert="WATCH",
                          label_int=7)
        g["T12_groupB_phase2"] = (PASS if r["alert_state"] == "WARN" else FAIL,
                                  f"state={r['alert_state']} slope={r['features']['slope']:.5f} "
                                  f"drift_ratio={r['features']['drift_ratio']:.3f}")
    except Exception as e:
        g["T12_groupB_phase2"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T13: score_C is advisory-only — never changes state ───────────────────
    try:
        sm = fresh()
        r_no = sm.update(score_A=0.02, theta_t=THETA, theta_initial=THETA0,
                         drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL",
                         score_C=None)
        sm2 = fresh()
        r_hi = sm2.update(score_A=0.02, theta_t=THETA, theta_initial=THETA0,
                          drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL",
                          score_C=99.0)   # absurd score_C must NOT change state
        ok = (r_no["alert_state"] == "NORMAL" and r_hi["alert_state"] == "NORMAL"
              and "advisory-only" in r_hi["reason"])
        g["T13_scoreC_advisory"] = (PASS if ok else FAIL,
                                    f"no={r_no['alert_state']} hi={r_hi['alert_state']}")
    except Exception as e:
        g["T13_scoreC_advisory"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T14: Mech-C per-channel drift -> at least WARN ────────────────────────
    # theta_t held near theta_initial (drift_ratio~1.0, no Mech-A); score_A well
    # below theta_t so the escalation comes from Mech-C only.
    try:
        sm = fresh()
        flags = [False, False, False, False, False, True, False, False]  # Pres.SV drift
        r = sm.update(score_A=0.03, theta_t=0.110, theta_initial=THETA0,
                      drift_locked=False, cusum_Sn=0.0, cusum_alert="NORMAL",
                      channel_drift_flags=flags)
        g["T14_mechC_warn"] = (PASS if r["alert_state"] == "WARN" else FAIL,
                               f"state={r['alert_state']} drift_ratio={r['features']['drift_ratio']:.3f}")
    except Exception as e:
        g["T14_mechC_warn"] = (FAIL, f"{type(e).__name__}: {e}")

    # ── T15: production clear_dwell is the M8-spec value (not the test value) ─
    try:
        prod = AlertStateMachine(cfg=cfg)   # real cfg
        ok = prod.cfg["clear_dwell"] >= 100   # M8 spec: hundreds of calls
        g["T15_prod_dwell"] = (PASS if ok else FAIL,
                               f"clear_dwell={prod.cfg['clear_dwell']}")
    except Exception as e:
        g["T15_prod_dwell"] = (FAIL, f"{type(e).__name__}: {e}")

    return g


# =============================================================================
# Main
# =============================================================================
results = {
    "script": SCRIPT_NAME,
    "stage": "M12 Stage 4 — Steps 4.2+4.3 (alert state machine)",
    "timestamp": datetime.now().isoformat(),
    "gates": {}, "evidence": {}, "overall_status": "UNKNOWN", "block_m11": True,
}


def main():
    log("=" * 76)
    log(f"  {SCRIPT_NAME}  |  {date.today().isoformat()}")
    log("  Stage 4 Steps 4.2+4.3 — AlertStateMachine + mech_triggers (D2 fix)")
    log("  test-harness-first; production patch emitted ONLY on full gate PASS")
    log("=" * 76)

    # Load thresholds: merge M8_alert_thresholds.json (score_C reliability) +
    # M8-spec state_machine defaults. We also persist resolved state_machine
    # params into M8_alert_thresholds.json so production reads one file.
    AlertStateMachine, load_alert_thresholds = _load_candidate()
    cfg = load_alert_thresholds(str(ALERT_CFG_PATH))
    results["evidence"]["score_C_reliability"] = cfg.get("score_C_reliability")
    results["evidence"]["clear_dwell"] = cfg.get("clear_dwell")
    log(f"  score_C_reliability (from 4.1 config) = {cfg.get('score_C_reliability')}")
    log(f"  score_C policy: ADVISORY-ONLY (serve-normal n=0 -> no live baseline)")

    # ── Run the transition-matrix gate ───────────────────────────────────────
    log("\nRunning state-transition matrix gate (15 subgates)...")
    try:
        sub = run_gate(AlertStateMachine, cfg)
    except Exception as e:
        log(f"  GATE CRASHED: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        results["overall_status"] = FAIL
        _finish(emit=False)
        return

    n_pass = 0
    for name, (status, detail) in sub.items():
        results["gates"][name] = status
        results["evidence"][name] = detail
        log(f"  {status}  {name}: {detail}")
        if status == PASS:
            n_pass += 1
    results["evidence"]["gate_pass_count"] = f"{n_pass}/{len(sub)}"

    all_pass = (n_pass == len(sub))
    results["overall_status"] = PASS if all_pass else FAIL

    # ── Persist resolved state_machine params into M8_alert_thresholds.json ──
    # (Adds a state_machine block; leaves score_C block untouched.)
    if all_pass:
        try:
            j = {}
            if ALERT_CFG_PATH.exists():
                with open(ALERT_CFG_PATH, encoding="utf-8") as f:
                    j = json.load(f)
            j["state_machine"] = {
                "rolling_mean_100_floor": cfg["rolling_mean_100_floor"],
                "rolling_mean_200_floor": cfg["rolling_mean_200_floor"],
                "slope_warn":             cfg["slope_warn"],
                "drift_ratio_warn":       cfg["drift_ratio_warn"],
                "drift_ratio_danger":     cfg["drift_ratio_danger"],
                "cusum_watch":            cfg["cusum_watch"],
                "clear_dwell":            cfg["clear_dwell"],
                "escalate_calls":         cfg["escalate_calls"],
                "_standards": "ISA-18.2/IEC 62682 asymmetric hysteresis; ISO 13374/13379-1; IEC 61511 advisory-only",
                "_score_C_policy": "advisory-only (Step 4.1: STRONG offline but serve-normal n=0; no validated live baseline)",
                "_generated_by": SCRIPT_NAME,
                "_generated_utc": datetime.utcnow().isoformat() + "Z",
            }
            with open(ALERT_CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(j, f, indent=2)
            results["evidence"]["state_machine_cfg_written"] = str(ALERT_CFG_PATH)
            log(f"\n  state_machine block written to {ALERT_CFG_PATH.name}")
        except Exception as e:
            log(f"  WARNING: could not persist state_machine cfg: {e}")

        # ── Emit production module (test == serve: byte-identical PROD_SRC) ──
        try:
            PROD_MODULE.parent.mkdir(parents=True, exist_ok=True)
            with open(PROD_MODULE, "w", encoding="utf-8") as f:
                f.write(PROD_SRC)
            results["evidence"]["prod_module_written"] = str(PROD_MODULE)
            log(f"  production module written -> {PROD_MODULE}")
        except Exception as e:
            log(f"  WARNING: could not write production module: {e}")

    _finish(emit=all_pass)


def _finish(emit):
    out_json = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    out_md = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    g = results["gates"]
    L = [f"# {SCRIPT_NAME}", "",
         "PumpSmart v14.2 — M12 Stage 4 Steps 4.2+4.3 — alert state machine (D2 fix)", "",
         f"- Date: {date.today().isoformat()}",
         f"- Overall: **{results['overall_status']}** | BLOCK_M11: {results['block_m11']}",
         f"- score_C: {results['evidence'].get('score_C_reliability')} -> ADVISORY-ONLY", "",
         "## Transition-matrix gate", "",
         "| Subgate | Status | Detail |", "|---|---|---|"]
    for name in g:
        L.append(f"| {name} | {g[name]} | {results['evidence'].get(name,'')} |")
    L += ["", "## Standards basis", "",
          "- ISA-18.2 / IEC 62682: four states with explicit entry/exit; asymmetric "
          "hysteresis (escalate in 1 call, de-escalate after clear_dwell consecutive "
          "clear calls); every transition traces to a named trigger (rationalization).",
          "- ISO 13374 / 13379-1: state detection + fault-family logic (cavitation "
          "fast-track, Group-C masked cap-to-WARN, Group-B phase-2, label-21 CUSUM path).",
          "- IEC 61511: advisory only — DANGER = immediate manual inspection, never auto-trip.",
          "", "## Architecture", "",
          "- New AlertStateMachine reads LOCKED RollingState (theta_t, drift_locked) + "
          "CUSUMState (cusum_Sn, cusum_alert) state dicts; owns its own short-horizon "
          "score_A buffers (rolling_mean_100/200, slope). ZERO edits to locked classes "
          "-> C-25 / Invariant-19 guarantees untouched.",
          "- score_C ADVISORY-ONLY: Step 4.1 found STRONG offline (AUC 0.95) but serve-"
          "normal n=0 (no validated live normal baseline), so per ISA-18.2 it cannot drive "
          "a high-severity alarm. It annotates reason only.",
          "", "## Production integration (apply ONLY after reviewing this gate)", ""]
    if emit:
        L += ["- Written: `app/runtime/alert_state_machine.py` (byte-identical to the gated code).",
              "- `state_machine` block written to `models/M8_alert_thresholds.json`.",
              "- anomaly.py patch: see the printed patch block in the run log.",
              "- Next: Step 4.4 — full M12 17-gate revalidation against M7 v3 + new state machine."]
    else:
        L += ["- Gate did NOT fully pass; production module NOT emitted. Fix failing subgates first."]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    status = results["overall_status"]
    print("\n" + "=" * 76)
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12 Stage 4 Steps 4.2+4.3 (alert state machine): {status}")
    print(f"  Transition-matrix gate: {results['evidence'].get('gate_pass_count','-')}")
    print(f"  score_C policy: ADVISORY-ONLY (STRONG offline, serve-normal n=0)")
    print(f"  Architecture: new AlertStateMachine reads locked RollingState+CUSUM (zero edits)")
    if emit:
        print("  Production emitted: app/runtime/alert_state_machine.py + state_machine cfg block")
        print("  Next: apply anomaly.py patch (below), then Step 4.4 — full 17-gate revalidation")
    else:
        print("  Production NOT emitted — fix failing subgates before integration.")
    print("  BLOCK_M11 = True  (Step 4.4 owns the flip)")
    print("══ END PASTE UPDATE ══")

    if emit:
        print("\n" + "=" * 76)
        print("  anomaly.py PATCH (apply manually after reviewing the gate)")
        print("=" * 76)
        print(_ANOMALY_PATCH)

    print("\n══ FILE MANIFEST ══")
    print(f"  Reports (Spaces upload):\n    {out_md}\n    {out_json}")
    if emit:
        print(f"  GitHub push (NEW production module): app/runtime/alert_state_machine.py")
        print(f"  GitHub push (config block added): models/M8_alert_thresholds.json")
        print(f"  Manual edit: app/routers/anomaly.py (apply printed patch)")
    print(f"  GitHub push: src/{SCRIPT_NAME}.py")
    print("=" * 76)
    print()
    if status == PASS:
        print("📦 M12 Stage 4 Steps 4.2+4.3 done — alert state machine gated 15/15. "
              "Apply the anomaly.py patch, then Step 4.4 (full 17-gate revalidation).")
    else:
        print("📦 Steps 4.2+4.3 gate did not fully pass. Fix failing subgates before integration.")


# =============================================================================
# anomaly.py patch (printed on PASS — manual apply, per gate-driven discipline)
# =============================================================================
_ANOMALY_PATCH = r'''
# --- in app/routers/anomaly.py ---
#
# 1) At module load (top, after other imports):
from app.runtime.alert_state_machine import AlertStateMachine, load_alert_thresholds
#
# 2) In the lifespan/startup where app.state is populated (e.g. model_registry
#    wiring), create ONE machine instance per pump stream:
#       request.app.state.alert_sm = AlertStateMachine(load_alert_thresholds())
#    (and call request.app.state.alert_sm.reset() inside /api/acknowledge,
#     alongside the existing zt_buf/cusum/rolling resets.)
#
# 3) REPLACE the old call:
#       alert_state = compute_alert_state(score_A, theta_t, cusum_Sn, drift_locked)
#    WITH:
#       sm_out = request.app.state.alert_sm.update(
#           score_A=score_A,
#           theta_t=theta_t,
#           theta_initial=models["theta_initial"],
#           drift_locked=drift_locked,
#           cusum_Sn=cusum_Sn,
#           cusum_alert=cusum_result.get("cusum_alert", "NORMAL"),
#           label_int=label_int,                 # available after M7 classify
#           channel_drift_flags=None,            # wire Mech-C source when available
#           score_C=score_C,                     # advisory-only
#       )
#       alert_state = sm_out["alert_state"]
#    NOTE: this requires moving the alert computation to AFTER M7 classification
#    so label_int is in scope (cavitation/Group-C/Group-B exceptions need it).
#    Keep the OLD compute_alert_state function in place as a fallback/no-op.
#
# 4) Optionally surface sm_out["reason"] and sm_out["features"] in the response
#    (e.g. FaultPrediction.alert_reason) for operator-facing rationalization.
'''


if __name__ == "__main__":
    main()