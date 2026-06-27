# =============================================================================
# src/module_12_stage5_honest_validation_rig.py
# PumpSmart v14.2 — M12 Stage 5 : Honest Labelled Validation Rig + Lead-Time
# =============================================================================
#
# WHAT THIS IS (the stakeholder-facing final-phase validation gate)
# -----------------------------------------------------------------
# For each of the 24 fault classes (+ normal), this rig composes ONE physically
# continuous failure narrative and streams it window-by-window through the LIVE
# FastAPI detection stack (/api/anomaly_detect), exactly as the deployed system
# sees it. It then scores the three things the panel cares about:
#
#   • DETECTION LATENCY  (Timer 1) — from fault-injection to first NORMAL→WATCH
#     →WARN→DANGER transition and to first-correct M7 label. SMALLER = better.
#   • BREAKDOWN LEAD-TIME (Timer 2) — from pump switch-on (window 0) to physical
#     breakdown (data truncation). The scored quantity is the GAP between first
#     DANGER and breakdown = the warning the operator gets to shut the pump down.
#     LARGER = better. PASS requires gap >= a per-fault minimum lead margin
#     (>= 60 s floor; larger for high-mechanical-damage faults).
#   • NORMAL-PHASE FALSE-FIRE — did the startup/steady prefix trip an alert
#     BEFORE the fault was injected? (the FPR cross-check, per spec).
#   • CLASSIFICATION ACCURACY + CALIBRATED CONFIDENCE per class.
#
# PHYSICS PROVENANCE (for the chemical-engineering panel)
# -------------------------------------------------------
# Every sequence is composed from the REAL M5-faithful generators in
# src/m6b_physics_lib.py (Paris-Erdogan crack growth, orifice-discharge seal
# leak, first-order thermal overloading, Rayleigh-Plesset cavitation, ISO 1940
# unbalance, M2 thermal coupling r=0.9793). Amplification toward breakdown uses
# a BOUNDED severity envelope (<= 1.5x) applied ONLY to the generator's deviation
# amplitude — all channel signatures, phases and couplings are preserved. The
# normal prefix (startup -> transition -> steady) is real cluster-baseline data.
# "Breakdown" is a PHYSICAL destructive-level crossing on the ground-truth
# signal (e.g. Pres.SV < 0.20 = 40-bar containment loss; Mot.SV >= 2.8x = bearing
# seizure), NOT an invented event.
#
# DRIVE MODE / FIDELITY
# ---------------------
# Server-driven (validates the ACTUAL deployed 24-class M7 + 4-layer stack via
# model_registry, not an offline reimplementation). Reuses module_12b v3.1
# conventions verbatim: warmup 432 windows @ sigma=0.045, raw_alert_state read,
# /api/test_reset_latch per sequence. FAST-FORWARD by default (no inter-window
# sleep) — full sequence fidelity, every window scored, only wall-clock
# compressed, so the full 24-class run fits a 10-15 min jury slot. --realtime
# inserts 1 s/window for a true-speed demo if ever wanted.
#
# OUTPUT
# ------
# Terminal tables (the jury reads these live) + CSV + JSON + report.md + plots.
# No live-web endpoint is added (the dashboard is the operator surface; these
# forensic timers are intentionally terminal-only per spec).
#
# C-26 honesty disclaimer printed on every result block.
#
# USAGE
#   python src/module_12_stage5_honest_validation_rig.py --mode smoke
#   python src/module_12_stage5_honest_validation_rig.py --mode full   # jury run
#   python src/module_12_stage5_honest_validation_rig.py --mode full --realtime
# =============================================================================

import sys, json, argparse, time, statistics, warnings
from pathlib import Path
from datetime import datetime, date, timezone

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Resolve project root + import config ─────────────────────────────────────
_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent]:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p)); break

from config import (SYNTH_DIR, MODEL_DIR, OUTPUT_DIR,
                    FAULT_LABELS, SAMPLING_HZ, MOTOR_RPM)

# Real physics generators (single source of truth for synthetic faults)
import m6b_physics_lib as plib

SCRIPT_NAME = "module_12_stage5_honest_validation_rig"
REPORT_DIR  = OUTPUT_DIR / "reports"
PLOTS_DIR   = OUTPUT_DIR / "plots"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
PASS = "PASS"; FAIL = "FAIL"

# =============================================================================
# CONVENTIONS — inherited verbatim from module_12b v3.1 (DO NOT change)
# =============================================================================
WINDOW_SIZE        = 50
WARMUP_WINDOWS     = 432
WARMUP_NOISE_SIGMA = 0.045
SEC_PER_STEP       = 1.0 / float(SAMPLING_HZ or 1)   # 1 s/step at 1 Hz

CLUSTER_NAMES = {0: "cooldown", 1: "steady_state", 2: "startup", 3: "high_load"}
# m6b channel order (LOCKED): Mot.SV Pmp.SV Mot.TV Pmp.PV Temp.SV Pres.SV Pmp.TV Mot.PV
CH = plib.CH

MANDATORY_FIELDS = [
    "fault_label", "confidence_pct", "unknown_fault_flag",
    "probable_physical_condition", "expected_sensor_behavior",
    "operational_risk_if_ignored", "recommended_action",
    "model_limitation_disclaimer",
]

# Label int -> str for reporting
LABEL_NAME = dict(FAULT_LABELS)
LABEL_NAME.setdefault(22, "multi_sensor_vibration")
LABEL_NAME.setdefault(23, "multi_sensor_pressure_temp")

# =============================================================================
# PHYSICAL BREAKDOWN CRITERIA  (ground-truth destructive level per fault family)
# -----------------------------------------------------------------------------
# (channel, direction, level) — evaluated on the composed GROUND-TRUTH signal.
# Levels are in normalized space (P*/a* baseline = 1.0; DeltaT* range 0..1 with
# fault drift > 1.0). These encode the real failure endpoint of a 110 kW 7-stage
# centrifugal pump:
#   • Mot.SV / Pmp.SV >= 2.8x : severe spalling / unbalance -> bearing or
#     rotor-dynamic seizure imminent (broadband acceleration envelope).
#   • Pres.SV <= 0.20 : discharge-pressure collapse -> on a 40-bar machine this
#     is loss of containment / total head loss (catastrophic).
#   • Pres.SV <= 0.30 : NPSH fully lost / sustained cavitation head collapse.
#   • Temp.SV >= 1.50 : thermal excursion beyond winding/bearing limit (trip).
# None -> instrument fault, no mechanical breakdown (Timer-2 not applicable).
# =============================================================================
BREAKDOWN_CRITERIA = {
    0:  None,                       # normal
    1:  ("Mot.SV", "above", 2.8),   # bearing wear -> seizure
    2:  ("Pmp.SV", "above", 2.8),   # imbalance -> rotor-dynamic failure
    3:  ("Pres.SV", "below", 0.30), # cavitation -> head collapse
    4:  ("Pres.SV", "below", 0.20), # seal failure -> 40-bar containment loss
    5:  ("Temp.SV", "above", 1.50), # overloading -> thermal trip
    6:  None,                       # sensor failure -> instrument only
    7:  ("Mot.SV", "above", 2.8),   # bearing+overloading
    8:  ("Pres.SV", "below", 0.25), # cavitation+seal
    9:  ("Pmp.SV", "above", 2.8),   # imbalance+bearing
    10: ("Pres.SV", "below", 0.25), # seal+cavitation_H
    11: ("Mot.SV", "above", 2.8),   # overloading+bearing
    12: ("Pmp.SV", "above", 2.8),   # imbalance+cavitation
    13: ("Mot.SV", "above", 2.8),   # bearing (Mot.SV masked)
    14: ("Pres.SV", "below", 0.30), # cavitation (Pres.SV masked)
    15: ("Pres.SV", "below", 0.20), # seal (Pres.SV drifting)
    16: ("Temp.SV", "above", 1.50), # overloading (Temp.SV stuck)
    17: ("Pmp.SV", "above", 2.8),   # imbalance (Pmp.SV flatline)
    18: ("Pres.SV", "below", 0.35), # cavitation intermittent (milder)
    19: ("Pres.SV", "below", 0.20), # seal failure fast -> containment
    20: ("Temp.SV", "above", 1.40), # overloading cyclic
    21: ("Mot.SV", "above", 2.8),   # bearing gradual (breakdown far -> CUSUM)
    22: ("Mot.SV", "above", 2.6),   # group E multi-sensor vibration
    23: ("Pres.SV", "below", 0.30), # group E pressure/temp
}

# =============================================================================
# PER-FAULT MINIMUM LEAD MARGIN (seconds). Floor = 60 s everywhere.
# Larger for high-mechanical-damage faults (you need more warning to prevent the
# worse outcome); 60 s floor for shutdown-only / recoverable / instrument faults.
# =============================================================================
LEAD_MARGIN_FLOOR_S = 60
LEAD_MARGIN_S = {
    0: 0,
    1: 180, 2: 120, 3: 180, 4: 180, 5: 120, 6: 60,
    7: 180, 8: 180, 9: 180, 10: 180, 11: 180, 12: 150,
    13: 180, 14: 180, 15: 180, 16: 120, 17: 120,
    18: 60, 19: 180, 20: 60, 21: 300,
    22: 60, 23: 60,
}

# =============================================================================
# COMPOSER — build ONE physically continuous failure narrative per label
# -----------------------------------------------------------------------------
# Phases (all real physics):
#   [startup]   real startup-cluster baseline (elevated vibration = correct
#               physics during shaft acceleration; NOT a fault)
#   [transition] linear blend startup -> steady baseline
#   [steady]    real steady-state baseline (model must read NORMAL here)
#   --- T1 fault-injection marker ---
#   [fault]     real m6b generator output for the label, with a BOUNDED rising
#               severity envelope carrying the signature to the physical
#               breakdown level. Truncated at the breakdown crossing.
# Returns: full (N,8), per-window cluster list, t_inject_step, breakdown_step
# =============================================================================

# How long each phase runs, in 50-step windows, by mode.
PHASE_WINDOWS = {
    "smoke": {"startup": 3, "transition": 2, "steady": 5,  "fault_max": 40},
    "quick": {"startup": 4, "transition": 3, "steady": 8,  "fault_max": 60},
    "full":  {"startup": 6, "transition": 4, "steady": 12, "fault_max": 80},
}

# Terminal severity per fault (realistic max within the generator's valid
# envelope — NOT exaggerated). Onset severity is low (sub-threshold by design).
ONSET_SEVERITY    = 0.10
TERMINAL_SEVERITY = 0.85
ENVELOPE_MAX      = 1.50   # bounded amplification of generator deviation

# Fault cluster per label (operating mode in which the fault physically occurs).
FAULT_CLUSTER = {
    0: 1, 1: 1, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1,
    7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1,
    13: 1, 14: 1, 15: 1, 16: 1, 17: 1,
    18: 2, 19: 1, 20: 2, 21: 1, 22: 1, 23: 1,
}


def _startup_character(seq, rng):
    """Startup vibration 2-4x steady on SV channels (correct physics, not a
    fault) — resonance during shaft acceleration. Applied to a baseline seq."""
    n = seq.shape[0]
    ramp = np.linspace(2.5, 1.2, n)  # high at switch-on, settling toward steady
    for c in ("Mot.SV", "Pmp.SV"):
        seq[:, CH[c]] *= ramp
    return seq


def _fault_via_m6b(label, severity, cluster, n_steps, rng):
    """Return a single-fault segment from the REAL m6b generators."""
    s = float(severity)
    if label in (1, 13):                       # bearing (13 = Mot.SV masked)
        seq = plib.generate_bearing_wear(s, cluster_id=cluster, n_steps=max(200, n_steps))
    elif label in (2, 17):                     # imbalance (17 = Pmp.SV flatline)
        seq = plib.generate_impeller_imbalance(s, cluster_id=cluster, n_steps=n_steps)
    elif label in (3, 14, 18):                 # cavitation family (cluster 2)
        seq = plib.generate_cavitation(s, cluster_id=2, n_steps=max(150, n_steps))
    elif label in (4, 15, 19):                 # seal family
        ns = max(150, n_steps) if label == 19 else max(400, n_steps)
        seq = plib.generate_seal_failure(s, cluster_id=cluster, n_steps=ns)
    elif label in (5, 16, 20):                 # overloading family (cluster 2)
        seq = plib.generate_overloading(s, cluster_id=2, n_steps=max(300, n_steps))
    elif label == 6:                           # sensor failure
        seq, _, _ = plib.generate_sensor_failure(s, cluster_id=cluster, n_steps=n_steps)
    else:
        seq = plib.make_baseline(n_steps, cluster_id=cluster)
    return np.asarray(seq, dtype=np.float32)


def _apply_masking(seq, label, rng):
    """Group C masking on top of the underlying real fault (sensor hides it)."""
    n = seq.shape[0]
    onset = int(rng.integers(n // 4, max(n // 4 + 1, n // 2)))
    masked = {13: "Mot.SV", 14: "Pres.SV", 15: "Pres.SV", 16: "Temp.SV", 17: "Pmp.SV"}
    c = CH[masked[label]]
    if label == 15:    # slow drift mask
        for t in range(onset, n):
            seq[t, c] = seq[onset, c] - 0.0001 * (t - onset)
    elif label == 17:  # flatline to ~1.0
        seq[onset:, c] = 1.0
    else:              # stuck-at-last (13,14,16)
        seq[onset:, c] = seq[onset, c]
    return seq


def _compound_segment(label, severity, cluster, n_steps, rng):
    """Group B causal chain: real primary m6b fault + lagged real secondary."""
    primary   = {7: 1, 8: 3, 9: 2, 10: 4, 11: 5, 12: 2}[label]
    secondary = {7: 5, 8: 4, 9: 1, 10: 3, 11: 1, 12: 3}[label]
    base = plib.make_baseline(n_steps, cluster_id=cluster)
    p = _fault_via_m6b(primary, severity, cluster, n_steps, rng)
    L = p.shape[0]
    full = np.asarray(p[:n_steps] if L >= n_steps else
                      np.vstack([p, np.repeat(p[-1:], n_steps - L, axis=0)]),
                      dtype=np.float32)
    lag = int(0.45 * n_steps)
    sec_len = n_steps - lag
    if sec_len > 60:
        s = _fault_via_m6b(secondary, severity * 0.85, cluster, sec_len, rng)
        sdev = s[:sec_len] - plib.make_baseline(s.shape[0], cluster_id=cluster)[:sec_len]
        full[lag:lag + sdev.shape[0]] += 0.7 * sdev
    return full


def _groupE_segment(label, severity, cluster, n_steps, rng):
    """Group E multi-sensor common-cause anomaly (real-baseline + coupled dev)."""
    seq = plib.make_baseline(n_steps, cluster_id=cluster)
    onset = int(rng.integers(50, max(51, n_steps // 3)))
    if label == 22:  ca, cb = CH["Mot.SV"], CH["Pmp.SV"]
    else:            ca, cb = CH["Pres.SV"], CH["Temp.SV"]
    for t in range(onset, n_steps):
        prog = (t - onset) / max(1, n_steps - onset)
        seq[t, ca] += severity * 0.5 * prog
        seq[t, cb] -= severity * 0.4 * prog
    return seq.astype(np.float32)


def _gradual_segment(severity, cluster, n_steps, rng):
    """Label 21 — Paris-law gradual bearing wear (real m6b at low severity, long).
    Breakdown is intentionally beyond the test window; CUSUM is the detector."""
    return plib.generate_bearing_wear(max(0.10, severity * 0.4),
                                      cluster_id=cluster, n_steps=max(1000, n_steps))


def _raw_fault_segment(label, severity, cluster, n_steps, rng):
    if label in (7, 8, 9, 10, 11, 12):
        return _compound_segment(label, severity, cluster, n_steps, rng)
    if label in (13, 14, 15, 16, 17):
        seq = _fault_via_m6b(label, severity, cluster, n_steps, rng)
        return _apply_masking(seq[:n_steps] if seq.shape[0] >= n_steps else seq, label, rng)
    if label in (22, 23):
        return _groupE_segment(label, severity, cluster, n_steps, rng)
    if label == 21:
        return _gradual_segment(severity, cluster, n_steps, rng)
    return _fault_via_m6b(label, severity, cluster, n_steps, rng)


def _amplify_to_breakdown(fault_seg, label, cluster, rng):
    """Apply a BOUNDED rising severity envelope (<=1.5x) to the generator's
    DEVIATION from baseline, so the fault carries to its physical breakdown
    level along the generator's own trajectory shape. Preserves all signatures
    and couplings (amplitude-only). Returns amplified seg."""
    base = plib.make_baseline(fault_seg.shape[0], cluster_id=cluster)
    # Match length defensively
    n = min(fault_seg.shape[0], base.shape[0])
    fault_seg = fault_seg[:n]; base = base[:n]
    dev = fault_seg - base
    env = np.linspace(1.0, ENVELOPE_MAX, n).reshape(-1, 1)
    return (base + env * dev).astype(np.float32)


def _find_breakdown(seq, label):
    """First step at which the ground-truth signal crosses the physical
    destructive level. Returns step index or None (no in-window breakdown)."""
    crit = BREAKDOWN_CRITERIA.get(label)
    if crit is None:
        return None
    ch, direction, level = crit
    col = seq[:, CH[ch]]
    if direction == "above":
        hits = np.where(col >= level)[0]
    else:
        hits = np.where(col <= level)[0]
    return int(hits[0]) if len(hits) else None


def compose_narrative(label, mode, rng):
    """Build the full switch-on -> startup -> steady -> fault -> breakdown
    sequence. Returns dict with the array, per-window clusters, and markers."""
    pw = PHASE_WINDOWS[mode]
    fault_cluster = FAULT_CLUSTER.get(label, 1)

    n_start = pw["startup"]    * WINDOW_SIZE
    n_trans = pw["transition"] * WINDOW_SIZE
    n_steady = pw["steady"]    * WINDOW_SIZE
    n_fault_max = pw["fault_max"] * WINDOW_SIZE

    # Phase 1 — startup (cluster 2 baseline + startup vibration character)
    startup = plib.make_baseline(n_start, cluster_id=2)
    startup = _startup_character(startup, rng)

    # Phase 2 — transition: blend startup-end -> steady (fault cluster) baseline
    steady_base = plib.make_baseline(n_steady, cluster_id=fault_cluster)
    a = startup[-1]; b = steady_base[0]
    trans = np.array([a + (b - a) * (i / max(1, n_trans - 1)) for i in range(n_trans)],
                     dtype=np.float32)
    trans += rng.normal(0, 0.015, trans.shape).astype(np.float32)

    # Phase 3 — steady normal (model must read NORMAL here)
    steady = steady_base

    prefix = np.vstack([startup, trans, steady]).astype(np.float32)
    t_inject = prefix.shape[0]   # fault-injection step (T1 origin)

    # Phase 4 — fault, amplified to physical breakdown
    if label == 0:
        full = prefix
        clusters = (["startup"] * n_start + ["steady_state"] * n_trans +
                    [CLUSTER_NAMES[fault_cluster]] * n_steady)
        return {"seq": full, "clusters": clusters, "t_inject": None,
                "breakdown": None, "fault_cluster": fault_cluster}

    fault_raw = _raw_fault_segment(label, TERMINAL_SEVERITY, fault_cluster,
                                   n_fault_max, rng)
    if label != 21:   # gradual wear is left un-amplified (CUSUM-detected, slow)
        fault_raw = _amplify_to_breakdown(fault_raw, label, fault_cluster, rng)

    bd_local = _find_breakdown(fault_raw, label)
    if bd_local is not None:
        fault_seg = fault_raw[:bd_local + 1]
        breakdown_step = t_inject + bd_local
    else:
        fault_seg = fault_raw
        breakdown_step = None   # slow fault, no in-window breakdown

    full = np.vstack([prefix, fault_seg]).astype(np.float32)
    full = np.clip(full, 0.0, 8.8)

    # Per-window cluster stream
    cstart = ["startup"] * n_start
    ctrans = ["steady_state"] * n_trans
    csteady = [CLUSTER_NAMES[fault_cluster]] * n_steady
    cfault = [CLUSTER_NAMES[fault_cluster]] * fault_seg.shape[0]
    clusters = cstart + ctrans + csteady + cfault

    return {"seq": full, "clusters": clusters, "t_inject": t_inject,
            "breakdown": breakdown_step, "fault_cluster": fault_cluster}


# =============================================================================
# CLIENT — server-driven (reuses module_12b v3.1 endpoint conventions)
# =============================================================================
class PumpClient:
    def __init__(self, base_url, timeout=30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self):
        r = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status(); return r.json()

    def reset_state(self, reason="M12_stage5_boundary", reset_zt=False):
        r = self.session.post(f"{self.base_url}/api/acknowledge", json={
            "pump_id": "PUMP-0032", "action_taken": reason,
            "operator_id": "M12_stage5_rig",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reset_zt": reset_zt,
        }, timeout=self.timeout)
        r.raise_for_status(); return r.json()

    def test_reset_latch(self):
        try:
            r = self.session.post(f"{self.base_url}/api/test_reset_latch",
                                  json={}, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def anomaly_detect(self, window, cluster="steady_state"):
        t0 = time.perf_counter()
        r = self.session.post(f"{self.base_url}/api/anomaly_detect", json={
            "window": window, "pump_id": "PUMP-0032", "cluster": cluster,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }, timeout=self.timeout)
        lat = time.perf_counter() - t0
        r.raise_for_status(); return r.json(), lat


def do_warmup(client, n_windows, cluster="steady_state", seed=99):
    """Adapt theta_t to a realistic steady baseline (sigma=0.045), per v3.1."""
    import random
    base = {"startup":[0.85,0.80,0.75,0.20,0.72,0.60,0.65,0.55],
            "steady_state":[0.45,0.42,0.50,0.95,0.55,0.80,0.52,0.90],
            "high_load":[0.55,0.52,0.65,0.98,0.68,0.72,0.62,0.98],
            "cooldown":[0.30,0.28,0.40,0.45,0.38,0.35,0.28,0.35]}.get(
            cluster, [0.45,0.42,0.50,0.95,0.55,0.80,0.52,0.90])
    rng = random.Random(seed)
    for _ in range(n_windows):
        win = [[max(0.0, min(3.0, b + rng.gauss(0, WARMUP_NOISE_SIGMA))) for b in base]
               for _ in range(WINDOW_SIZE)]
        try: client.anomaly_detect(win, cluster=cluster)
        except Exception: pass


# =============================================================================
# SCORER — dual timer + regime transitions + classification + normal-phase FPR
# =============================================================================
SEV_RANK = {"NORMAL": 0, "WATCH": 1, "WARN": 2, "DANGER": 3}


def score_sequence(client, narrative, target_label, realtime=False):
    """Stream the composed narrative window-by-window; return a result row."""
    seq = narrative["seq"]; clusters = narrative["clusters"]
    t_inject = narrative["t_inject"]; breakdown = narrative["breakdown"]

    # Per-sequence latch isolation (does NOT touch detector cadence — v3.1).
    try: client.test_reset_latch()
    except Exception: pass

    n_steps = seq.shape[0]
    n_windows = n_steps // WINDOW_SIZE

    first_watch = first_warn = first_danger = None     # step index (window end)
    first_correct = None
    normal_phase_fire = False                          # alert BEFORE t_inject
    normal_phase_worst = "NORMAL"                      # worst alert in pre-inject run
    normal_phase_first_fire_s = None                   # when it first deviated
    confidences = []; correct_any = False
    lat_list = []; ok7 = True
    sA_max = 0.0; cusum_max = 0.0; first_cusum_watch = None
    regime_trace = []   # (window_end_step, alert, label_int)

    for k in range(n_windows):
        s0 = k * WINDOW_SIZE
        win = np.nan_to_num(seq[s0:s0 + WINDOW_SIZE], nan=0.0).tolist()
        cl = clusters[s0] if s0 < len(clusters) else "steady_state"
        try:
            pred, lat = client.anomaly_detect(win, cluster=cl)
        except Exception as e:
            log(f"      win {k} err: {e}"); continue
        lat_list.append(lat)
        win_end = (k + 1) * WINDOW_SIZE        # pump-time (s) at window end

        for f in MANDATORY_FIELDS:
            if pred.get(f) in (None, ""): ok7 = False

        alert = pred.get("raw_alert_state") or pred.get("alert_state", "NORMAL")
        sA = float(pred.get("score_A", 0.0) or 0.0)
        cu = float(pred.get("cusum_Sn", 0.0) or 0.0)
        sA_max = max(sA_max, sA); cusum_max = max(cusum_max, cu)
        lbl = pred.get("fault_label_int")
        lbl = int(lbl) if lbl is not None else -1
        conf = float(pred.get("confidence_pct", 0.0) or 0.0)
        regime_trace.append((win_end, alert, lbl))

        if cu >= 2.0 and first_cusum_watch is None:
            first_cusum_watch = win_end

        # Normal-phase false-fire cross-check (alert before fault injection)
        pre_inject = (t_inject is not None and win_end <= t_inject) or (target_label == 0)
        if pre_inject and alert != "NORMAL":
            normal_phase_fire = True
            if SEV_RANK.get(alert, 0) > SEV_RANK.get(normal_phase_worst, 0):
                normal_phase_worst = alert
            if normal_phase_first_fire_s is None:
                normal_phase_first_fire_s = win_end

        # Regime first-crossings (only count AFTER fault injection)
        after_inject = (t_inject is None) or (win_end > t_inject)
        if after_inject:
            if alert in ("WATCH", "WARN", "DANGER") and first_watch is None:
                first_watch = win_end
            if alert in ("WARN", "DANGER") and first_warn is None:
                first_warn = win_end
            if alert == "DANGER" and first_danger is None:
                first_danger = win_end
            if lbl == target_label and target_label != 0:
                correct_any = True
                if first_correct is None:
                    first_correct = win_end
                confidences.append(conf)

        if realtime:
            time.sleep(WINDOW_SIZE * SEC_PER_STEP)

    # ── Timer 1 — detection latency (seconds after injection) ────────────────
    def lat_after(x): return None if (x is None or t_inject is None) else (x - t_inject)
    t1_watch  = lat_after(first_watch)
    t1_warn   = lat_after(first_warn)
    t1_danger = lat_after(first_danger)
    t1_correct = lat_after(first_correct)

    # ── Timer 2 — breakdown lead-time (gap from first DANGER to breakdown) ───
    margin_req = max(LEAD_MARGIN_FLOOR_S, LEAD_MARGIN_S.get(target_label, 60))
    if breakdown is None:
        # No in-window breakdown (slow fault). Score on CUSUM WATCH presence.
        t2_gap = None
        if target_label == 21:
            t2_verdict = PASS if first_cusum_watch is not None else FAIL
            t2_note = "slow fault: CUSUM WATCH lead (breakdown beyond window)"
        elif target_label == 0:
            t2_verdict = "N/A"; t2_note = "normal — no breakdown"
        else:
            t2_verdict = PASS if (first_danger or first_warn) else FAIL
            t2_note = "no in-window breakdown; scored on alert presence"
    else:
        if first_danger is None:
            t2_gap = -1; t2_verdict = FAIL
            t2_note = "DANGER never fired before breakdown"
        else:
            t2_gap = breakdown - first_danger
            t2_verdict = PASS if t2_gap >= margin_req else FAIL
            t2_note = (f"DANGER {first_danger}s; breakdown {breakdown}s; "
                       f"gap {t2_gap}s vs req {margin_req}s")

    detected = first_watch is not None or sA_max > 0.05
    classified = correct_any

    # ── Classification integrity gate (per spec: wrong class = FAIL) ─────────
    # A safe alert must name the RIGHT fault. Graded:
    #   correct class emitted during fault run        -> PASS
    #   fault detected but NEVER correctly classified -> FAIL (misclassified)
    #   not detected at all                           -> FAIL (can't classify)
    #   normal (label 0): N/A (no class to predict)
    if target_label == 0:
        classify_gate = "N/A"
    elif classified:
        classify_gate = PASS
    else:
        classify_gate = FAIL

    # ── Normal-phase integrity gate (graded) ─────────────────────────────────
    # The pump must read NORMAL through switch-on -> startup -> transition ->
    # steady, right up to fault injection. Graded verdict per spec:
    #   NORMAL kept        -> PASS          (model does not fire on healthy data)
    #   reached WATCH      -> PARTIAL_FAIL  (over-sensitive but not alarming)
    #   reached WARN/DANGER-> FAIL          (false alarm on good data — unsafe)
    if normal_phase_worst == "NORMAL":
        normal_gate = PASS
    elif normal_phase_worst == "WATCH":
        normal_gate = "PARTIAL_FAIL"
    else:  # WARN or DANGER
        normal_gate = FAIL

    return {
        "label": target_label,
        "label_name": LABEL_NAME.get(target_label, f"label_{target_label}"),
        "fault_cluster": narrative["fault_cluster"],
        "n_windows": n_windows,
        "t_inject_step": t_inject,
        "breakdown_step": breakdown,
        # Timer 1
        "t1_watch_s": t1_watch, "t1_warn_s": t1_warn,
        "t1_danger_s": t1_danger, "t1_correct_label_s": t1_correct,
        # Timer 2
        "lead_margin_req_s": margin_req,
        "t2_breakdown_lead_s": t2_gap,
        "t2_verdict": t2_verdict, "t2_note": t2_note,
        # detection / classification
        "detected": detected, "classified": classified,
        "classify_gate": classify_gate,
        "mean_conf_pct": round(float(np.mean(confidences)), 2) if confidences else None,
        "normal_phase_false_fire": normal_phase_fire,
        "normal_phase_worst": normal_phase_worst,
        "normal_phase_gate": normal_gate,
        "normal_phase_first_fire_s": normal_phase_first_fire_s,
        # diagnostics
        "score_A_max": round(sA_max, 6), "cusum_max": round(cusum_max, 6),
        "first_cusum_watch_s": first_cusum_watch,
        "seven_field_complete": ok7,
        "latency_p95": round(float(np.percentile(lat_list, 95)), 4) if lat_list else 0.0,
        "regime_trace": regime_trace,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--reps", type=int, default=None,
                    help="sequences per label (overrides mode default)")
    ap.add_argument("--realtime", action="store_true",
                    help="insert real 1 s/window delay (true-speed demo)")
    ap.add_argument("--labels", type=str, default="all",
                    help="comma-separated label ints, or 'all'")
    args = ap.parse_args()

    reps = args.reps or {"smoke": 1, "quick": 3, "full": 5}[args.mode]

    log("=" * 72)
    log("  PumpSmart v14.2 — M12 Stage 5 : Honest Validation Rig + Lead-Time")
    log(f"  Mode: {args.mode}  |  reps/label: {reps}  |  URL: {args.base_url}")
    log(f"  Fast-forward: {not args.realtime}  |  warmup {WARMUP_WINDOWS}×σ={WARMUP_NOISE_SIGMA}")
    log(f"  Physics: m6b_physics_lib real M5 generators | breakdown = physical level")
    log("=" * 72)

    # Init physics lib (same config the live builder uses)
    with open(MODEL_DIR / "M3_normalization_config.json", encoding="utf-8") as f:
        norm_cfg = json.load(f)
    phys_path = MODEL_DIR / "M5_physics_config.json"
    phys_cfg = (json.load(open(phys_path, encoding="utf-8")) if phys_path.exists()
                else {"physics_constants": {}})
    plib.init_lib(norm_cfg, phys_cfg, seed=args.seed)

    client = PumpClient(args.base_url)

    log("\nSTEP 1 — Server health + model contract")
    try:
        h = client.health()
        log(f"  status={h.get('status')} arch={h.get('arch_version')}")
        log(f"  M4 q={h.get('m4_threshold_locked')}  "
            f"M7 classes={h.get('xgb_n_classes', h.get('m7_n_classes','?'))}")
        nclass = h.get("xgb_n_classes") or h.get("m7_n_classes")
        if nclass is not None and int(nclass) != 24:
            log(f"  ⚠ WARNING: server M7 has {nclass} classes (expected 24). "
                f"Confirm model_registry loaded M7_xgboost_classifier_cpu.json (v3).")
    except Exception as e:
        log(f"  FATAL: server not reachable — {e}"); sys.exit(1)

    labels = (list(range(0, 24)) if args.labels == "all"
              else [int(x) for x in args.labels.split(",")])
    rng = np.random.default_rng(args.seed)

    log(f"\nSTEP 2 — Running {len(labels)} labels × {reps} reps "
        f"({len(labels)*reps} narratives)")
    # WARM UP ONCE for the whole run. Per-sequence reset_state() was resetting
    # RollingState θ_t → θ_initial (1.881) every sequence; re-warming each time
    # left θ_t fragile and contaminated the normal-phase gate. The probe proved
    # a single 432-window warmup adapts θ_t → ~0.178 and a normal window then
    # reads NORMAL/WATCH (not DANGER). Between sequences we release ONLY the
    # latch (test_reset_latch) — detector cadence (CUSUM/rolling/θ_t/z_t) is
    # left intact, exactly as module_12b's 21/24 baseline does.
    log(f"  One-time warmup: {WARMUP_WINDOWS} windows × σ={WARMUP_NOISE_SIGMA} (steady_state)")
    do_warmup(client, WARMUP_WINDOWS, cluster="steady_state")
    try:
        th = client.health().get("rolling_state", {}).get("theta_t")
        log(f"  θ_t after warmup → {th}")
    except Exception:
        pass

    rows = []
    t0 = time.time()
    for lbl in labels:
        for rep in range(reps):
            narrative = compose_narrative(lbl, args.mode, rng)
            res = score_sequence(client, narrative, lbl, realtime=args.realtime)
            res["rep"] = rep
            rows.append(res)
        ln = LABEL_NAME.get(lbl, f"label_{lbl}")
        log(f"  ✓ L{lbl:02d} {ln:<32s} ({(time.time()-t0)/60:.1f}m)")

    df = pd.DataFrame(rows)

    # ── Aggregate per label ──────────────────────────────────────────────────
    log("\nSTEP 3 — Aggregating per-label results")
    agg = []
    for lbl in labels:
        sub = df[df["label"] == lbl]
        det = float(sub["detected"].mean())
        cls = float(sub["classified"].mean()) if lbl != 0 else None
        t1w = sub["t1_watch_s"].dropna()
        t1d = sub["t1_danger_s"].dropna()
        t1c = sub["t1_correct_label_s"].dropna()
        t2 = sub["t2_breakdown_lead_s"].dropna()
        t2pass = (sub["t2_verdict"] == PASS).mean() if lbl != 0 else None
        ff = float(sub["normal_phase_false_fire"].mean())
        # Normal-phase gate roll-up: worst verdict across reps governs.
        gates_seen = list(sub["normal_phase_gate"])
        if any(g == FAIL for g in gates_seen):
            ng = FAIL
        elif any(g == "PARTIAL_FAIL" for g in gates_seen):
            ng = "PARTIAL_FAIL"
        else:
            ng = PASS
        worst_state = max(sub["normal_phase_worst"],
                          key=lambda s: SEV_RANK.get(s, 0))
        # Classification gate roll-up: FAIL if ANY rep misclassified (strict),
        # else PASS. N/A for normal.
        if lbl == 0:
            cg = "N/A"
        else:
            cgates = list(sub["classify_gate"])
            cg = FAIL if any(g == FAIL for g in cgates) else PASS
        agg.append({
            "label": lbl, "name": LABEL_NAME.get(lbl, f"label_{lbl}"),
            "group": ("A" if lbl < 7 else "B" if lbl < 13 else "C" if lbl < 18
                      else "D" if lbl < 22 else "E"),
            "detect_rate": round(det, 3),
            "classify_rate": round(cls, 3) if cls is not None else None,
            "t1_watch_s_med": round(float(t1w.median()), 1) if len(t1w) else None,
            "t1_danger_s_med": round(float(t1d.median()), 1) if len(t1d) else None,
            "t1_correct_s_med": round(float(t1c.median()), 1) if len(t1c) else None,
            "t2_lead_s_med": round(float(t2.median()), 1) if len(t2) else None,
            "lead_req_s": max(LEAD_MARGIN_FLOOR_S, LEAD_MARGIN_S.get(lbl, 60)),
            "t2_pass_rate": round(float(t2pass), 3) if t2pass is not None else None,
            "mean_conf_pct": round(float(sub["mean_conf_pct"].dropna().mean()), 1)
                             if sub["mean_conf_pct"].notna().any() else None,
            "normal_false_fire_rate": round(ff, 3),
            "normal_phase_gate": ng,
            "normal_phase_worst": worst_state,
            "classify_gate": cg,
        })
    adf = pd.DataFrame(agg)

    # ── Terminal report (the jury reads this) ────────────────────────────────
    print("\n" + "═" * 100)
    print("  M12 STAGE 5 — HONEST VALIDATION RESULTS  (per fault class)")
    print("  time units = pump-seconds @ 1 Hz | T1 = latency after fault injection (smaller=better)")
    print("  T2 = DANGER→breakdown lead margin (larger=better; PASS needs ≥ required margin)")
    print("  NORMgate = healthy-data integrity: NORMAL→PASS, WATCH→PARTIAL, WARN/DANGER→FAIL")
    print("  CLSgate  = correct fault classification: right class→PASS, wrong/none→FAIL")
    print("═" * 110)
    hdr = (f"{'L':>3} {'fault':<24}{'grp':>4}{'det':>6}{'cls':>6}"
           f"{'T1wat':>7}{'T1dng':>7}{'T1cor':>7}"
           f"{'T2lead':>8}{'req':>5}{'T2ok':>6}{'CLS':>6}{'NORMgate':>13}")
    print(hdr); print("─" * 110)

    def _f(x):
        return f"{x:.0f}" if isinstance(x, (int, float)) and x == x else "·"

    def _pct(x):
        return f"{x*100:.0f}%" if isinstance(x, (int, float)) and x == x else "·"

    def _gate_disp(g, worst):
        m = {PASS: "✅PASS", "PARTIAL_FAIL": "⚠ PARTIAL", FAIL: "❌FAIL"}
        tag = m.get(g, g)
        return f"{tag}({worst[:4]})" if g not in (PASS, "N/A") else tag

    def _cls_disp(g):
        return {PASS: "✅", FAIL: "❌", "N/A": "·"}.get(g, g)

    for _, r in adf.iterrows():
        print(f"{r['label']:>3} {str(r['name'])[:24]:<24}{r['group']:>4}"
              f"{_pct(r['detect_rate']):>6}{_pct(r['classify_rate']):>6}"
              f"{_f(r['t1_watch_s_med']):>7}{_f(r['t1_danger_s_med']):>7}"
              f"{_f(r['t1_correct_s_med']):>7}{_f(r['t2_lead_s_med']):>8}"
              f"{r['lead_req_s']:>5}{_pct(r['t2_pass_rate']):>6}"
              f"{_cls_disp(r['classify_gate']):>6}"
              f"{_gate_disp(r['normal_phase_gate'], r['normal_phase_worst']):>13}")
    print("═" * 110)

    # ── Headline metrics ─────────────────────────────────────────────────────
    fault_df = adf[adf["label"] != 0]
    overall_detect = float(fault_df["detect_rate"].mean())
    overall_classify = float(fault_df["classify_rate"].dropna().mean())
    t2_overall = fault_df["t2_pass_rate"].dropna()
    overall_t2 = float(t2_overall.mean()) if len(t2_overall) else 0.0
    normal_ff = float(adf[adf["label"] == 0]["normal_false_fire_rate"].iloc[0]) \
        if (adf["label"] == 0).any() else None

    # Normal-phase integrity gate rollup across ALL labels' pre-injection runs.
    ng_all = list(adf["normal_phase_gate"])
    n_pass = sum(g == PASS for g in ng_all)
    n_partial = sum(g == "PARTIAL_FAIL" for g in ng_all)
    n_fail = sum(g == FAIL for g in ng_all)
    if n_fail > 0:
        normal_gate_overall = FAIL
    elif n_partial > 0:
        normal_gate_overall = "PARTIAL_FAIL"
    else:
        normal_gate_overall = PASS

    # Classification integrity gate rollup (faults only; wrong class = FAIL).
    cg_all = [g for g in adf["classify_gate"] if g != "N/A"]
    cg_pass = sum(g == PASS for g in cg_all)
    cg_fail = sum(g == FAIL for g in cg_all)
    classify_gate_overall = FAIL if cg_fail > 0 else PASS

    print("\n  HEADLINE METRICS")
    print(f"    Overall detection rate (faults)   : {overall_detect*100:.1f}%")
    print(f"    Overall classification rate        : {overall_classify*100:.1f}%")
    print(f"    Breakdown lead-time PASS rate      : {overall_t2*100:.1f}%")
    if normal_ff is not None:
        print(f"    Normal-data false-fire rate        : {normal_ff*100:.1f}%")
    print(f"    CLASSIFICATION INTEGRITY GATE      : {classify_gate_overall}")
    print(f"      (faults correctly classified: {cg_pass} PASS / {cg_fail} FAIL "
          f"of {len(cg_all)} fault classes)")
    if cg_fail:
        bad_c = adf[adf["classify_gate"] == FAIL][["label", "name"]]
        print(f"      ❌ MISCLASSIFIED — labels: {list(bad_c['label'])} "
              f"(detected but M7 never named the correct fault)")
    print(f"    NORMAL-PHASE INTEGRITY GATE        : {normal_gate_overall}")
    print(f"      (healthy-data runs: {n_pass} PASS / {n_partial} PARTIAL / {n_fail} FAIL "
          f"out of {len(ng_all)} labels)")
    if n_fail:
        bad = adf[adf["normal_phase_gate"] == FAIL][["label", "name", "normal_phase_worst"]]
        print(f"      ❌ FALSE ALARM on healthy data — labels: "
              f"{list(bad['label'])} (fired WARN/DANGER before any fault)")
    if n_partial:
        par = adf[adf["normal_phase_gate"] == "PARTIAL_FAIL"][["label"]]
        print(f"      ⚠ over-sensitive (reached WATCH) — labels: {list(par['label'])}")
    print("\n  PER-GROUP DETECTION / CLASSIFICATION")
    for grp in ["A", "B", "C", "D", "E"]:
        g = fault_df[fault_df["group"] == grp]
        if len(g):
            print(f"    Group {grp}: detect {g['detect_rate'].mean()*100:5.1f}%  "
                  f"classify {g['classify_rate'].dropna().mean()*100:5.1f}%")

    print("\n  C-26 HONESTY DISCLAIMER")
    print("    Trained on CIRA-anchored physics-synthetic data for 110 kW 7-stage")
    print("    centrifugal pump @ 2980 RPM, 40 bar, 45 m³/h. Synthetic 5-fold seq")
    print("    macro F1 = 0.9965 ± 0.0005. Real-world F1 expected 0.65–0.85 (C-26)")
    print("    until the active-learning loop completes its first retrain on ≥50")
    print("    operator-confirmed real faults. Advisory only — verify physically.")
    print("═" * 100)

    # ── Save artifacts ────────────────────────────────────────────────────────
    log("\nSTEP 4 — Saving artifacts")
    per_seq_cols = [c for c in df.columns if c != "regime_trace"]
    df[per_seq_cols].to_csv(OUTPUT_DIR / "M12_stage5_per_sequence.csv",
                            index=False, encoding="utf-8")
    adf.to_csv(OUTPUT_DIR / "M12_stage5_per_label.csv",
               index=False, encoding="utf-8")

    summary = {
        "version": "stage5_v1.0",
        "mode": args.mode, "reps_per_label": reps,
        "base_url": args.base_url, "seed": args.seed,
        "fast_forward": not args.realtime,
        "warmup_windows": WARMUP_WINDOWS, "warmup_sigma": WARMUP_NOISE_SIGMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "headline": {
            "overall_detection_rate": round(overall_detect, 4),
            "overall_classification_rate": round(overall_classify, 4),
            "breakdown_lead_pass_rate": round(overall_t2, 4),
            "normal_false_fire_rate": round(normal_ff, 4) if normal_ff is not None else None,
            "normal_phase_integrity_gate": normal_gate_overall,
            "normal_phase_counts": {"PASS": n_pass, "PARTIAL_FAIL": n_partial, "FAIL": n_fail},
            "classification_integrity_gate": classify_gate_overall,
            "classification_counts": {"PASS": cg_pass, "FAIL": cg_fail},
        },
        "per_label": adf.to_dict(orient="records"),
        "lead_margin_policy_s": {str(k): max(LEAD_MARGIN_FLOOR_S, v)
                                 for k, v in LEAD_MARGIN_S.items()},
        "breakdown_criteria": {str(k): v for k, v in BREAKDOWN_CRITERIA.items()},
        "c26_disclaimer": ("Synthetic-domain results; real-world F1 expected "
                           "0.65–0.85 per C-26 until active-learning first retrain."),
        "physics_provenance": ("All sequences composed from m6b_physics_lib M5 "
                               "generators (Paris-Erdogan, orifice-discharge, "
                               "first-order thermal, Rayleigh-Plesset, ISO 1940). "
                               "Amplification = bounded ≤1.5× envelope on deviation "
                               "only; breakdown = physical destructive-level crossing."),
    }
    with open(OUTPUT_DIR / "M12_stage5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1 — detection latency (T1) per fault
        fdf = adf[adf["label"] != 0].copy()
        fig, ax = plt.subplots(figsize=(15, 5))
        x = np.arange(len(fdf))
        ax.bar(x - 0.25, fdf["t1_watch_s_med"].fillna(0), 0.25, label="→WATCH", color="#FFC107")
        ax.bar(x, fdf["t1_danger_s_med"].fillna(0), 0.25, label="→DANGER", color="#F44336")
        ax.bar(x + 0.25, fdf["t1_correct_s_med"].fillna(0), 0.25, label="→correct label", color="#2196F3")
        ax.set_xticks(x); ax.set_xticklabels(fdf["label"], fontsize=8)
        ax.set_xlabel("Fault label"); ax.set_ylabel("Detection latency after injection (s)")
        ax.set_title("M12 Stage 5 — Timer 1: detection latency (smaller = better)")
        ax.legend(); plt.tight_layout()
        plt.savefig(PLOTS_DIR / "stage5_detection_latency.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 2 — breakdown lead-time (T2) vs required margin
        fig, ax = plt.subplots(figsize=(15, 5))
        lead = fdf["t2_lead_s_med"].fillna(0)
        ax.bar(x, lead, color="#4CAF50", label="achieved lead (s)")
        ax.plot(x, fdf["lead_req_s"], "r--o", markersize=4, label="required margin (s)")
        ax.set_xticks(x); ax.set_xticklabels(fdf["label"], fontsize=8)
        ax.set_xlabel("Fault label"); ax.set_ylabel("DANGER→breakdown lead (s)")
        ax.set_title("M12 Stage 5 — Timer 2: breakdown lead-time vs required margin (larger = better)")
        ax.legend(); plt.tight_layout()
        plt.savefig(PLOTS_DIR / "stage5_breakdown_lead_time.png", dpi=150, bbox_inches="tight")
        plt.close()
        log("  Plots saved: stage5_detection_latency.png, stage5_breakdown_lead_time.png")
    except Exception as e:
        log(f"  WARNING: plotting skipped — {e}")

    # ── Report.md ──────────────────────────────────────────────────────────────
    rep = [
        f"# M12 Stage 5 — Honest Validation Rig Report",
        f"**Date:** {date.today()}  |  **Mode:** {args.mode}  |  reps/label: {reps}",
        f"**Drive:** server ({args.base_url}) | fast-forward: {not args.realtime}",
        "",
        "## Headline",
        f"- Overall detection (faults): **{overall_detect*100:.1f}%**",
        f"- Overall classification: **{overall_classify*100:.1f}%**",
        f"- Breakdown lead-time PASS: **{overall_t2*100:.1f}%**",
        f"- Normal-data false-fire: **{(normal_ff or 0)*100:.1f}%**",
        f"- **Normal-phase integrity gate: {normal_gate_overall}** "
        f"({n_pass} PASS / {n_partial} PARTIAL / {n_fail} FAIL of {len(ng_all)} labels)",
        f"- **Classification integrity gate: {classify_gate_overall}** "
        f"({cg_pass} PASS / {cg_fail} FAIL of {len(cg_all)} fault classes)",
        "",
        "## Classification integrity gate (correct fault named)",
        "Detection alone is insufficient — a safe alert must name the RIGHT "
        "fault so the operator takes the correct action. Per fault class: M7 "
        "emits the correct label during the fault run = **PASS**; detected but "
        "never correctly classified, or not detected = **FAIL** (misclassified). "
        "Normal (label 0) is N/A. Any misclassified fault fails this gate and "
        "blocks M11 readiness.",
        "",
        "## Normal-phase integrity gate (no false fire on healthy data)",
        "Every test runs switch-on → startup → transition → steady BEFORE any "
        "fault is injected. The model must stay NORMAL throughout this healthy "
        "run. Graded verdict: NORMAL kept = **PASS**; reached WATCH = "
        "**PARTIAL_FAIL** (over-sensitive); reached WARN/DANGER = **FAIL** "
        "(false alarm on good data — unsafe). This guarantees the model does not "
        "fire randomly on correct data.",
        "",
        "## Dual-timer design",
        "- **Timer 1 (detection latency):** from fault injection to first "
        "WATCH/WARN/DANGER and to first-correct M7 label. Smaller is better.",
        "- **Timer 2 (breakdown lead-time):** from pump switch-on to physical "
        "breakdown; scored gap = DANGER→breakdown. PASS needs ≥ per-fault margin "
        "(≥60 s floor; larger for high-mechanical-damage faults).",
        "",
        "## Per-label results",
        adf.to_markdown(index=False),
        "",
        "## Physics provenance (chemical-engineering panel)",
        "All sequences composed from `m6b_physics_lib` M5-faithful generators: "
        "Paris-Erdogan crack growth (bearing), orifice-discharge leak (seal), "
        "first-order thermal (overloading), Rayleigh-Plesset (cavitation), "
        "ISO 1940 unbalance (impeller), M2 thermal coupling r=0.9793. "
        "Amplification toward breakdown is a bounded ≤1.5× envelope applied to "
        "the generator's deviation amplitude only — all signatures, phases and "
        "couplings preserved. Breakdown = physical destructive-level crossing.",
        "",
        "## C-26 disclaimer",
        "Synthetic-domain results. Real-world F1 expected 0.65–0.85 per C-26 "
        "until active-learning first retrain (≥50 confirmed real faults). "
        "Advisory only — verify physically.",
    ]
    with open(REPORT_DIR / f"{SCRIPT_NAME}_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    log(f"  Report: {REPORT_DIR / (SCRIPT_NAME + '_report.md')}")

    # ── Paste update ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12_stage5_version        : stage5_v1.0 (dual-timer lead-time)")
    print(f"M12_stage5_mode           : {args.mode} ({reps}/label)")
    print(f"M12_stage5_detect_rate    : {overall_detect*100:.1f}%")
    print(f"M12_stage5_classify_rate  : {overall_classify*100:.1f}%")
    print(f"M12_stage5_lead_pass_rate : {overall_t2*100:.1f}%")
    print(f"M12_stage5_normal_FF      : {(normal_ff or 0)*100:.1f}%")
    print(f"M12_stage5_normal_gate    : {normal_gate_overall} "
          f"({n_pass}P/{n_partial}PF/{n_fail}F)")
    print(f"M12_stage5_classify_gate  : {classify_gate_overall} "
          f"({cg_pass}P/{cg_fail}F)")
    print(f"M12_stage5_lead_floor_s   : {LEAD_MARGIN_FLOOR_S}")
    print(f"M12_stage5_C26            : real-world 0.65-0.85 (synthetic above)")
    _ready = (overall_t2 >= 0.8 and overall_detect >= 0.8
              and normal_gate_overall != FAIL
              and classify_gate_overall != FAIL)
    print(f"Status for M11            : {'READY' if _ready else 'REVIEW'}")
    print("══ END PASTE UPDATE ══")
    print("═" * 60)
    print(f"\n📦 M12 Stage 5 done. Artifacts in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()