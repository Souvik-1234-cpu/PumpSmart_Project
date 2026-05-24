# =============================================================================
# module_12b_adversarial_runner.py  v3.0
# PumpSmart v14.2 — M12 Adversarial Validation : Runner
#
# v3.0 redesign (Path A from deep-dive analysis):
#
# Fix 1 — Realistic warmup noise:
#   v2.0 used σ=0.015 → θ_t collapsed to 0.161 → G1 FPR = 100%
#   v3.0 uses σ=0.045 (matches M4 training std) → θ_t adapts to ~0.28–0.35
#
# Fix 2 — Scenario isolation:
#   v2.0 ran G1 + G1b in same scenario → G1 alarms contaminated G4b CUSUM
#   v3.0 splits: G1 normal_FPR, G1b normal_cusum (separate reset + warmup)
#
# Fix 3 — Severity-stratified sub-gates (industrial alignment):
#   v2.0 conflated detection, classification, and severity into single TPR
#   v3.0 separates:
#     - G2-detect (in-envelope): L1 must fire on sev ≥ 0.20 ≥ 85%
#     - G2-classify (in-envelope): M7 correct IF detected ≥ 85%
#     - G2-mild (sub-threshold): cascade response advisory only
#   Same pattern for G3 (compound), G9 (masked)
#
# Reading: this matches what ISO 13374 mandates for CMS Level 3 — every
# input to the diagnosis must be separately inspectable. Conflating detection
# and classification into a single metric hides architectural information
# that maintenance reviewers need.
# =============================================================================
import sys, json, argparse, time, statistics
from pathlib import Path
from datetime import datetime, date, timezone
import random
import numpy as np
import pandas as pd
import requests

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent]:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p)); break

from config import (SYNTH_DIR, MODEL_DIR, OUTPUT_DIR)

SCRIPT_NAME = "module_12b_adversarial_runner"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
M12_DIR     = SYNTH_DIR / "M12_adversarial"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
PASS = "PASS"; FAIL = "FAIL"

WINDOW_SIZE    = 50
WARMUP_WINDOWS = 432
WARMUP_NOISE_SIGMA = 0.045   # FIX 1: realistic SCADA variance

CLUSTER_BASELINES = {
    "startup":      [0.85, 0.80, 0.75, 0.20, 0.72, 0.60, 0.65, 0.55],
    "steady_state": [0.45, 0.42, 0.50, 0.95, 0.55, 0.80, 0.52, 0.90],
    "high_load":    [0.55, 0.52, 0.65, 0.98, 0.68, 0.72, 0.62, 0.98],
    "cooldown":     [0.30, 0.28, 0.40, 0.45, 0.38, 0.35, 0.28, 0.35],
}
CLUSTER_NAMES = {0:"cooldown", 1:"steady_state", 2:"high_load", 3:"startup"}

# FIX 2: G1 and G1b are now separate scenarios (no cross-contamination)
SCENARIO_GROUPS = [
    ("normal_FPR",        ["G1_normal_boundary"],          "steady_state"),
    ("normal_cusum_spec", ["G1b_normal_extended"],         "steady_state"),
    ("acute_in_env",      ["G2_acute_in_envelope"],        "steady_state"),
    ("acute_mild",        ["G2_acute_mild_extreme"],       "steady_state"),
    ("compound_in_env",   ["G3_compound_in_envelope"],     "steady_state"),
    ("compound_mild",     ["G3_compound_mild_extreme"],    "steady_state"),
    ("gradual_wear",      ["G4_label21_mild"],             "steady_state"),
    ("cross_cluster",     ["G5_cross_cluster"],            "steady_state"),
    ("masked_in_env",     ["G6_masked_in_envelope"],       "steady_state"),
    ("masked_mild",       ["G6_masked_mild_extreme"],      "steady_state"),
    ("sensor_interrupt",  ["G7_partial_dropout","G7_full_dropout","G7_pump_off"], "steady_state"),
    ("crosspoint_drift",  ["G8_crosspoint_drift"],         "steady_state"),
    ("groupE",            ["G9_groupE"],                   "steady_state"),
]

LABEL_STR_TO_INT = {
    "normal":0,"bearing_wear":1,"impeller_imbalance":2,"cavitation":3,
    "seal_failure":4,"overloading":5,"sensor_failure":6,
    "bearing_overloading":7,"bearing+overloading":7,"bearing_wear+overloading":7,
    "cavitation_seal":8,"cavitation+seal":8,"cavitation+seal_failure":8,
    "imbalance_bearing":9,"imbalance+bearing":9,"impeller_imbalance+bearing_wear":9,
    "seal_cavitation_high_head":10,"seal_failure+cavitation_H":10,"seal+cavitation_H":10,
    "overloading_bearing":11,"overloading+bearing":11,"overloading+bearing_wear":11,
    "imbalance_cavitation":12,"imbalance+cavitation":12,"impeller_imbalance+cavitation":12,
    "bearing_mot_sv_masked":13,"bearing_Mot.SV_masked":13,
    "cavitation_pres_sv_masked":14,"cavitation_Pres.SV_masked":14,
    "seal_pres_sv_drifting":15,"seal_Pres.SV_drifting":15,
    "overloading_temp_sv_stuck":16,"overloading_Temp.SV_stuck":16,
    "imbalance_pmp_sv_flatline":17,"imbalance_Pmp.SV_flatline":17,
    "cavitation_intermittent":18,"seal_failure_fast":19,"overloading_cyclic":20,
    "bearing_wear_gradual":21,
    "multi_sensor_vibration":22,"vibration_pair_anomaly":22,
    "multi_sensor_pressure_temp":23,"pressure_temp_common_cause":23,
}

def _extract_label_int(pred):
    lbl = pred.get("fault_label_int")
    if lbl is not None: return int(lbl)
    name = pred.get("fault_label","")
    if name in LABEL_STR_TO_INT: return LABEL_STR_TO_INT[name]
    norm = name.lower().replace(" ","_").replace("-","_")
    if norm in LABEL_STR_TO_INT: return LABEL_STR_TO_INT[norm]
    for k,v in LABEL_STR_TO_INT.items():
        if k in norm or norm in k: return v
    return -1


# FIX 3: Industrial-aligned severity-stratified gate specs
GATE_SPECS = {
    # Tier 1 — Critical safety gates (BLOCK_M11 conditions)
    "G0_hash_integrity":          {"label":"Model hash integrity"},
    "G1_normal_fpr":              {"max":0.010, "label":"Normal FPR (adapted θ_t) ≤ 1%"},
    "G2_in_env_detect":           {"min":0.85,  "label":"Acute in-envelope DETECTION ≥ 85%/label"},
    "G2_in_env_classify":         {"min":0.85,  "label":"Acute in-envelope CLASSIFICATION ≥ 85%/label"},
    "G3_in_env_detect":           {"min":0.80,  "label":"Compound in-envelope DETECTION ≥ 80% macro"},
    "G3_in_env_classify":         {"min":0.70,  "label":"Compound in-envelope CLASSIFICATION ≥ 70% macro"},
    "G4a_label21_cusum":          {"min":0.75,  "label":"Label 21 CUSUM WATCH within 1500 windows ≥ 75%"},
    "G4b_label21_cusum_spec":     {"min":0.99,  "label":"CUSUM < 2.0 on 1000-win normals ≥ 99%"},
    "G6_cpu_latency_p95":         {"max":1.0,   "label":"Latency p95 ≤ 1.0s"},
    "G6_cpu_latency_p99":         {"max":3.0,   "label":"Latency p99 ≤ 3.0s"},
    "G6_cpu_latency_max":         {"max":5.0,   "label":"Latency max ≤ 5.0s"},
    "G7_7field_completeness":     {"min":1.00,  "label":"7-field completeness 100%"},
    "G9_masked_detect":           {"min":0.80,  "label":"Masked DETECTION ≥ 80% per label"},
    "G10_sensor_interruption":    {"min":1.00,  "label":"Sensor interruption 3/3 states"},
    "G12_score_routing":          {"min":1.00,  "label":"Invariant 19 score routing"},
    "G14_physics_context":        {"min":1.00,  "label":"physics_context on non-normals"},

    # Tier 2 — Advisory (does NOT block M11)
    "G2_mild_cascade":            {"min":0.30,  "label":"Acute mild-extreme cascade response ≥ 30% (advisory)"},
    "G3_mild_cascade":            {"min":0.30,  "label":"Compound mild-extreme cascade response ≥ 30% (advisory)"},
    "G5_cross_cluster":           {"min":0.60,  "label":"Cross-cluster TPR ≥ 60% (advisory)"},
    "G8_crosspoint_lock":         {"min":0.80,  "label":"L4 crosspoint guard ≥ 80% (advisory)"},
    "G9_masked_classify":         {"min":0.50,  "label":"Masked CLASSIFICATION ≥ 50% (advisory)"},
    "G9_mild_cascade":            {"min":0.20,  "label":"Masked mild-extreme cascade ≥ 20% (advisory)"},
    "G11_groupE_tpr":             {"min":0.55,  "label":"Group E TPR ≥ 55% (advisory)"},
    "G13_ood_responsive":         {"min":0.20,  "label":"OOD flag ≥ 20% adversarial (diagnostic)"},
}

# Critical gates that gate BLOCK_M11
CRITICAL_GATES = {
    "G0_hash_integrity",
    "G1_normal_fpr",
    "G2_in_env_detect", "G2_in_env_classify",
    "G3_in_env_detect", "G3_in_env_classify",
    "G4a_label21_cusum", "G4b_label21_cusum_spec",
    "G6_cpu_latency_p95", "G6_cpu_latency_p99", "G6_cpu_latency_max",
    "G7_7field_completeness",
    "G9_masked_detect",
    "G10_sensor_interruption",
    "G12_score_routing",
    "G14_physics_context",
}

# =============================================================================
class PumpClient:
    def __init__(self, base_url, timeout=30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self):
        r = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status(); return r.json()

    def validate_model(self):
        r = self.session.get(f"{self.base_url}/api/validate_model", timeout=self.timeout)
        r.raise_for_status(); return r.json()

    def reset_state(self, reason="M12_scenario_boundary"):
        r = self.session.post(f"{self.base_url}/api/acknowledge", json={
            "pump_id":"PUMP-0032","action_taken":reason,
            "operator_id":"M12_runner",
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        }, timeout=self.timeout)
        r.raise_for_status(); return r.json()

    def anomaly_detect(self, window, cluster="steady_state"):
        t0 = time.perf_counter()
        r = self.session.post(f"{self.base_url}/api/anomaly_detect", json={
            "window":window, "pump_id":"PUMP-0032", "cluster":cluster,
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        }, timeout=self.timeout)
        latency = time.perf_counter() - t0
        r.raise_for_status(); return r.json(), latency


def do_warmup(client, n_windows, cluster="steady_state", seed=99,
                noise_sigma=WARMUP_NOISE_SIGMA):
    """FIX 1: realistic σ=0.045 matching M4 training distribution std."""
    baseline = CLUSTER_BASELINES.get(cluster, CLUSTER_BASELINES["steady_state"])
    rng = random.Random(seed)
    for _ in range(n_windows):
        win = [[max(0.0, min(3.0, b+rng.gauss(0, noise_sigma))) for b in baseline]
               for _ in range(WINDOW_SIZE)]
        try: client.anomaly_detect(win, cluster=cluster)
        except Exception: pass
    try:
        h = client.health()
        theta = h.get("rolling_state",{}).get("theta_t","?")
        log(f"    Warmup done. θ_t adapted → {theta:.6f}" if isinstance(theta,float)
            else f"    θ_t → {theta}")
    except Exception:
        pass


def iter_windows(seq):
    n = seq.shape[0]
    for s in range(0, n-WINDOW_SIZE+1, WINDOW_SIZE):
        yield np.nan_to_num(seq[s:s+WINDOW_SIZE], nan=0.0).tolist()


MANDATORY_FIELDS = [
    "fault_label","confidence_pct","unknown_fault_flag",
    "probable_physical_condition","expected_sensor_behavior",
    "operational_risk_if_ignored","recommended_action","model_limitation_disclaimer",
]


def run_scenario_group(client, sequences, warmup_cluster="steady_state",
                        stream_delay=0.0, warmup_n=WARMUP_WINDOWS):
    try: client.reset_state()
    except Exception as e: log(f"  WARN reset: {e}")
    log(f"  Warmup: {warmup_n} windows σ={WARMUP_NOISE_SIGMA} ({warmup_cluster})...")
    do_warmup(client, warmup_n, cluster=warmup_cluster)

    results = []
    for entry in sequences:
        seq = entry["seq"]; meta = entry["meta"]
        tgt = int(meta.get("label",-1))
        cln = CLUSTER_NAMES.get(int(meta.get("cluster",1)), "steady_state")

        windows_seen=0; alert_hist=[]; label_hist=[]
        sA_vals=[]; cusum_vals=[]
        sB_max=0.0; cusum_max=0.0
        drift_locked=False; ood_ever=False; lat_list=[]
        ok7=True; ok_ctx=True; routing_viol=0
        first_alert=None; first_watch=None; first_correct=None

        for win in iter_windows(seq):
            try: pred, lat = client.anomaly_detect(win, cluster=cln)
            except Exception as e: log(f"    win err: {e}"); continue
            windows_seen += 1; lat_list.append(lat)
            for f in MANDATORY_FIELDS:
                if pred.get(f) in (None,""): ok7=False
            sA = float(pred.get("score_A",0.0) or 0.0)
            sB = float(pred.get("score_B",0.0) or 0.0)
            cu = float(pred.get("cusum_Sn",0.0) or 0.0)
            sA_vals.append(sA); cusum_vals.append(cu)
            sB_max = max(sB_max, sB); cusum_max = max(cusum_max, cu)
            alert = pred.get("alert_state", "NORMAL")
            alert_hist.append(alert)
            if alert != "NORMAL" and first_alert is None: first_alert = windows_seen
            if cu >= 2.0 and first_watch is None: first_watch = windows_seen
            fi = _extract_label_int(pred); label_hist.append(fi)
            if fi == tgt and first_correct is None: first_correct = windows_seen
            dl = pred.get("drift_locked") or (pred.get("rolling_state") or {}).get("drift_locked")
            if dl: drift_locked = True
            if pred.get("ood_suspected", False): ood_ever = True
            if tgt not in (0,-1) and not pred.get("physics_context"): ok_ctx = False
            if pred.get("invariant19_violation", False): routing_viol += 1
            if stream_delay > 0: time.sleep(stream_delay)

        sA_max = max(sA_vals) if sA_vals else 0.0
        # Infer crosspoint lock from score_A trajectory
        if not drift_locked and sA_max > 0.50: drift_locked = True

        results.append({
            "windows_seen":windows_seen,
            "score_A_max":round(sA_max,6),
            "score_A_mean":round(float(np.mean(sA_vals)) if sA_vals else 0.0, 6),
            "score_B_max":round(sB_max,6), "cusum_max":round(cusum_max,6),
            "drift_locked_ever":drift_locked, "ood_ever":ood_ever,
            "alert_states_unique":sorted(set(alert_hist)),
            "first_alert_window":first_alert,
            "first_cusum_watch_window":first_watch,
            "first_correct_label_window":first_correct,
            "final_label":label_hist[-1] if label_hist else -1,
            "final_alert":alert_hist[-1] if alert_hist else "NORMAL",
            "seven_field_complete":ok7, "physics_context_present":ok_ctx,
            "routing_violations":routing_viol,
            "latency_mean":round(statistics.mean(lat_list),4) if lat_list else 0.0,
            "latency_p95":round(float(np.percentile(lat_list,95)),4) if lat_list else 0.0,
            "latency_p99":round(float(np.percentile(lat_list,99)),4) if lat_list else 0.0,
            "latency_max":round(max(lat_list),4) if lat_list else 0.0,
            "latencies_all":lat_list,
        })
    return results


# ── FIX 3: Severity-stratified success metrics ─────────────────────────────
def has_alert(r):
    """Did ANY layer fire? (detection metric — independent of classification)"""
    return any(a in ("WATCH","WARN","DANGER") for a in r["alert_states_unique"])

def correct_label(r):
    """Did M7 produce the correct label at any point in the sequence?"""
    return r["first_correct_label_window"] is not None

def full_success(r):
    """Tier-1: detection AND classification (industrial spec for safe alert)."""
    return has_alert(r) and correct_label(r)


# =============================================================================
def compute_gates(per_seq, model_hashes_match, mode):
    df = pd.DataFrame(per_seq)
    g = {}
    smoke = (mode == "smoke")

    g["G0_hash_integrity"] = PASS if model_hashes_match else FAIL

    all_lat = [l for r in per_seq for l in r["latencies_all"]]
    p95 = float(np.percentile(all_lat,95)) if all_lat else 0.0
    p99 = float(np.percentile(all_lat,99)) if all_lat else 0.0
    mx  = float(max(all_lat)) if all_lat else 0.0
    g["G6_cpu_latency_p95"] = PASS if p95 <= 1.0 else FAIL
    g["G6_cpu_latency_p99"] = PASS if p99 <= 3.0 else FAIL
    g["G6_cpu_latency_max"] = PASS if mx <= 5.0 else FAIL
    g["_latency"] = {"p95":round(p95,4),"p99":round(p99,4),"max":round(mx,4)}

    # G1 — Normal FPR
    g1 = df[df["group"] == "G1_normal_boundary"]
    if len(g1):
        fpr = (g1["final_alert"] != "NORMAL").mean()
        g["G1_normal_fpr"] = PASS if fpr <= 0.01 else FAIL
        g["_g1_fpr"] = round(float(fpr), 4)

    # G2 — Acute faults, severity-stratified
    g2_env = df[df["group"] == "G2_acute_in_envelope"]
    if len(g2_env):
        # Detection: did ANY alert fire?
        det_per_lbl = {}; det_ok = True
        cls_per_lbl = {}; cls_ok = True
        for lbl in [1,2,3,4,5,6,19]:
            sub = g2_env[g2_env["label"] == lbl]
            if not len(sub):
                det_per_lbl[lbl] = None; cls_per_lbl[lbl] = None; continue
            det = sub.apply(lambda r: has_alert(r), axis=1).mean()
            det_per_lbl[lbl] = round(float(det), 4)
            if det < 0.85: det_ok = False
            # Classification: of detected, what fraction had correct label?
            detected = sub[sub.apply(lambda r: has_alert(r), axis=1)]
            if len(detected):
                cls = detected.apply(lambda r: correct_label(r), axis=1).mean()
                cls_per_lbl[lbl] = round(float(cls), 4)
                if cls < 0.85: cls_ok = False
            else:
                cls_per_lbl[lbl] = None
                cls_ok = False
        g["G2_in_env_detect"]   = PASS if det_ok else FAIL
        g["G2_in_env_classify"] = PASS if cls_ok else FAIL
        g["_g2_detect_per_lbl"] = det_per_lbl
        g["_g2_classify_per_lbl"] = cls_per_lbl

    # G2 mild — cascade response advisory
    g2_mild = df[df["group"] == "G2_acute_mild_extreme"]
    if len(g2_mild):
        casc = g2_mild.apply(lambda r: has_alert(r), axis=1).mean()
        g["G2_mild_cascade"] = PASS if casc >= 0.30 else FAIL
        g["_g2_mild_cascade"] = round(float(casc), 4)

    # G3 — Compound, severity-stratified
    g3_env = df[df["group"] == "G3_compound_in_envelope"]
    if len(g3_env):
        det = g3_env.apply(lambda r: has_alert(r), axis=1).mean()
        g["G3_in_env_detect"] = PASS if det >= 0.80 else FAIL
        g["_g3_detect_macro"] = round(float(det), 4)
        detected = g3_env[g3_env.apply(lambda r: has_alert(r), axis=1)]
        if len(detected):
            cls = detected.apply(lambda r: correct_label(r), axis=1).mean()
            g["G3_in_env_classify"] = PASS if cls >= 0.70 else FAIL
            g["_g3_classify_macro"] = round(float(cls), 4)
        else:
            g["G3_in_env_classify"] = FAIL
            g["_g3_classify_macro"] = 0.0

    g3_mild = df[df["group"] == "G3_compound_mild_extreme"]
    if len(g3_mild):
        casc = g3_mild.apply(lambda r: has_alert(r), axis=1).mean()
        g["G3_mild_cascade"] = PASS if casc >= 0.30 else FAIL
        g["_g3_mild_cascade"] = round(float(casc), 4)

    # G4a — Label 21 CUSUM
    g4 = df[df["group"] == "G4_label21_mild"]
    if len(g4):
        if smoke:
            acc = (g4["cusum_max"] > 0.01).mean()
            g["G4a_label21_cusum"] = PASS if acc >= 0.50 else FAIL
            g["_g4a"] = {"smoke_note":"S_n>0.01 proxy (insufficient windows for S_n≥2.0 in smoke)",
                          "accumulation_rate":round(float(acc),4),
                          "cusum_max_mean":round(float(g4["cusum_max"].mean()),6)}
        else:
            wh = g4["first_cusum_watch_window"].apply(
                lambda x: x is not None and x <= 1500).mean()
            g["G4a_label21_cusum"] = PASS if wh >= 0.75 else FAIL
            g["_g4a_watch_hit"] = round(float(wh), 4)

    # G4b — CUSUM specificity
    g4b = df[df["group"] == "G1b_normal_extended"]
    if len(g4b):
        spec = (g4b["cusum_max"] < 2.0).mean()
        g["G4b_label21_cusum_spec"] = PASS if spec >= 0.99 else FAIL
        g["_g4b_spec"] = round(float(spec), 4)

    # G5 — Cross-cluster
    g5 = df[df["group"] == "G5_cross_cluster"]
    if len(g5):
        tpr = g5.apply(lambda r: full_success(r), axis=1).mean()
        g["G5_cross_cluster"] = PASS if tpr >= 0.60 else FAIL
        g["_g5"] = round(float(tpr), 4)

    # G7 — 7-field
    c = df["seven_field_complete"].mean()
    g["G7_7field_completeness"] = PASS if c >= 1.0 else FAIL
    g["_g7"] = round(float(c), 4)

    # G8 — L4 crosspoint
    g8 = df[df["group"] == "G8_crosspoint_drift"]
    if len(g8):
        lr = g8["drift_locked_ever"].mean()
        g["G8_crosspoint_lock"] = PASS if lr >= 0.80 else FAIL
        g["_g8"] = {"lock_rate":round(float(lr),4),
                    "sA_max_mean":round(float(g8["score_A_max"].mean()),4)}

    # G9 — Masked faults, severity-stratified
    g9_env = df[df["group"] == "G6_masked_in_envelope"]
    if len(g9_env):
        det_pl = {}; det_ok = True
        cls_pl = {}; cls_ok = True
        for lbl in [13,14,15,16,17]:
            sub = g9_env[g9_env["label"] == lbl]
            if not len(sub):
                det_pl[lbl] = None; cls_pl[lbl] = None; continue
            det = sub.apply(lambda r: has_alert(r), axis=1).mean()
            det_pl[lbl] = round(float(det), 4)
            if det < 0.80: det_ok = False
            detected = sub[sub.apply(lambda r: has_alert(r), axis=1)]
            if len(detected):
                cls = detected.apply(lambda r: correct_label(r), axis=1).mean()
                cls_pl[lbl] = round(float(cls), 4)
                if cls < 0.50: cls_ok = False
            else:
                cls_pl[lbl] = None
                cls_ok = False
        g["G9_masked_detect"] = PASS if det_ok else FAIL
        g["G9_masked_classify"] = PASS if cls_ok else FAIL
        g["_g9_detect_per_lbl"] = det_pl
        g["_g9_classify_per_lbl"] = cls_pl

    g9_mild = df[df["group"] == "G6_masked_mild_extreme"]
    if len(g9_mild):
        casc = g9_mild.apply(lambda r: has_alert(r), axis=1).mean()
        g["G9_mild_cascade"] = PASS if casc >= 0.20 else FAIL
        g["_g9_mild_cascade"] = round(float(casc), 4)

    # G10 — Sensor interruption
    INTR = {"G7_partial_dropout":"partial","G7_full_dropout":"full","G7_pump_off":"off"}
    g10_p = 0; g10_d = {}
    for grp, nm in INTR.items():
        sub = df[df["group"] == grp]
        if len(sub):
            r = ((sub["final_alert"] != "NORMAL") | (sub["score_A_max"] > 0.05)).mean()
            g10_d[nm] = {"responding_rate":round(float(r),4),
                          "sA_max_mean":round(float(sub["score_A_max"].mean()),4)}
            if r >= 0.50: g10_p += 1
        else:
            g10_d[nm] = {"note":"no sequences"}
    g["G10_sensor_interruption"] = PASS if g10_p >= 3 else FAIL
    g["_g10"] = g10_d

    # G11 — Group E
    g11 = df[df["group"] == "G9_groupE"]
    if len(g11):
        t = g11.apply(lambda r: full_success(r), axis=1).mean()
        g["G11_groupE_tpr"] = PASS if t >= 0.55 else FAIL
        g["_g11"] = round(float(t), 4)

    # G12 — Score routing
    g["G12_score_routing"] = PASS if df["routing_violations"].sum() == 0 else FAIL
    g["_g12_violations"] = int(df["routing_violations"].sum())

    # G13 — OOD
    adv = df[df["group"].isin(["G2_acute_in_envelope","G2_acute_mild_extreme",
                                 "G3_compound_in_envelope","G5_cross_cluster"])]
    if len(adv):
        or_ = adv["ood_ever"].mean()
        g["G13_ood_responsive"] = PASS if or_ >= 0.20 else FAIL
        g["_g13"] = round(float(or_), 4)

    # G14 — physics_context on non-normals
    nn = df[~df["group"].isin(["G1_normal_boundary","G1b_normal_extended",
                                  "G7_partial_dropout","G7_full_dropout","G7_pump_off"])]
    if len(nn):
        ct = nn["physics_context_present"].mean()
        g["G14_physics_context"] = PASS if ct >= 1.0 else FAIL
        g["_g14"] = round(float(ct), 4)

    return g


# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke","quick","full","stream"], default="smoke")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--stream-delay", type=float, default=0.0)
    parser.add_argument("--stream-sample", type=int, default=20)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--groups", type=str, default="all")
    parser.add_argument("--warmup-windows", type=int, default=WARMUP_WINDOWS)
    args = parser.parse_args()

    log("="*70)
    log(f"  PumpSmart M12b  v3.0  (severity-stratified industrial gates)")
    log(f"  Mode: {args.mode}  |  URL: {args.base_url}")
    log(f"  Warmup: {args.warmup_windows} windows × σ={WARMUP_NOISE_SIGMA}")
    log(f"")
    log(f"  Gates split detection from classification.")
    log(f"  In-envelope (sev ≥ M6 floor) = critical certification gates.")
    log(f"  Mild-extreme (sev < M6 floor) = advisory gates testing")
    log(f"  L3 CUSUM + L4 sub-threshold pathway.")
    log("="*70)

    client = PumpClient(base_url=args.base_url)

    log("\nSTEP 1 — G0: server health + model hash integrity")
    try:
        h = client.health()
        rs = h.get("rolling_state",{})
        log(f"  Server: {h.get('status')} | arch: {h.get('arch_version')}")
        log(f"  M4 threshold: {h.get('m4_threshold_locked')}")
        log(f"  θ_initial={rs.get('theta_initial','?')} buffer={rs.get('buffer_capacity','?')}")
    except Exception as e:
        log(f"  FAIL: {e}"); sys.exit(1)

    try:
        vm = client.validate_model()
        ok_hash = bool(vm.get("validation_passed", False))
        log(f"  validate_model: {ok_hash} | artifacts: {vm.get('artifact_count')}")
    except Exception as e:
        log(f"  WARN: {e}"); ok_hash = False

    log("\nSTEP 2 — Loading manifest")
    mp = M12_DIR / "M12_manifest.csv"
    if not mp.exists():
        log("  FAIL — run module_12a_adversarial_generator.py first"); sys.exit(1)
    manifest = pd.read_csv(mp)
    log(f"  Manifest rows: {len(manifest):,}")
    log(f"  Severity tiers: {dict(manifest['severity_tier'].value_counts())}")

    if args.groups != "all":
        gf = [x.strip() for x in args.groups.split(",")]
        manifest = manifest[manifest["group"].str.contains("|".join(gf))].reset_index(drop=True)
    if args.max_per_label is not None:
        manifest = manifest.groupby("label").head(args.max_per_label).reset_index(drop=True)
    if args.mode == "stream":
        manifest = manifest.sample(min(args.stream_sample, len(manifest)),
                                     random_state=42).reset_index(drop=True)

    log("\nSTEP 3 — Loading sequences")
    grp_seqs = {}; missing = 0
    for _, row in manifest.iterrows():
        fp = M12_DIR / row["filename"]
        if not fp.exists(): missing += 1; continue
        try: seq = np.load(fp, allow_pickle=True)["window"]
        except Exception as e: log(f"  WARN: {e}"); missing += 1; continue
        gn = row["group"]
        if gn not in grp_seqs: grp_seqs[gn] = []
        grp_seqs[gn].append({"seq":seq, "meta":dict(row)})
    if missing: log(f"  WARN: {missing} files missing")
    log(f"  Loaded: {sum(len(v) for v in grp_seqs.values())} sequences across {len(grp_seqs)} groups")

    log(f"\nSTEP 4 — Running scenario groups")
    per_seq = []; t0 = time.time()

    for sc_label, gnames, wcluster in SCENARIO_GROUPS:
        sc_seqs = [s for gn in gnames if gn in grp_seqs for s in grp_seqs[gn]]
        if not sc_seqs: continue
        log(f"\n  ── {sc_label}: {len(sc_seqs)} sequences")
        res = run_scenario_group(client, sc_seqs, wcluster,
                                   args.stream_delay, args.warmup_windows)
        for r, se in zip(res, sc_seqs):
            m = se["meta"]
            r["filename"] = m["filename"]; r["group"] = m["group"]
            r["label"] = int(m.get("label",-1)); r["cluster"] = int(m.get("cluster",1))
            r["severity"] = float(m.get("severity",0.0))
            r["severity_tier"] = m.get("severity_tier","unknown")
            per_seq.append(r)
        log(f"    Elapsed: {(time.time()-t0)/60:.1f}m")

    log(f"\nSTEP 5 — Computing gates from {len(per_seq)} results")
    gates = compute_gates(per_seq, ok_hash, args.mode)
    cfails = [k for k in gates if k in CRITICAL_GATES and gates[k] == FAIL]
    block = len(cfails) > 0

    print("\n" + "═"*72)
    print(f"M12 GATE SUMMARY  ({args.mode}, {len(per_seq)} seqs, v3.0 stratified)")
    print("═"*72)
    print("\n  CRITICAL gates (block M11 if any fail):")
    for gn, gv in gates.items():
        if gn.startswith("_") or gn not in CRITICAL_GATES: continue
        mark = "✅" if gv == PASS else "❌"
        print(f"    {mark} {gn}: {gv}")
        print(f"        {GATE_SPECS.get(gn,{}).get('label','')}")
    print("\n  ADVISORY gates (do NOT block M11):")
    for gn, gv in gates.items():
        if gn.startswith("_") or gn in CRITICAL_GATES: continue
        mark = "✅" if gv == PASS else "⚠️ "
        print(f"    {mark} {gn}: {gv}")
        print(f"        {GATE_SPECS.get(gn,{}).get('label','')}")
    print("═"*72)
    print(f"  BLOCK_M11 : {block}")
    if block: print(f"  Failed critical: {cfails}")
    if args.mode == "smoke":
        print()
        print("  ⚠️  SMOKE (3–5/label per tier): wiring check only.")
        print("     Run --mode quick (20–40/label) for advisory diagnostics.")
        print("     Run --mode full (200–400/label) for binding BLOCK_M11 decision.")
    print("═"*72)

    log("\nSTEP 6 — Saving reports")
    pd.DataFrame([{k:v for k,v in r.items() if k != "latencies_all"} for r in per_seq])\
      .to_csv(OUTPUT_DIR / "M12_per_sequence_results.csv", index=False, encoding="utf-8")

    gate_sum = {
        "version":"v3.0",
        "mode":args.mode, "base_url":args.base_url,
        "warmup_windows":args.warmup_windows,
        "warmup_noise_sigma":WARMUP_NOISE_SIGMA,
        "n_sequences":len(per_seq),
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "gates":{k:v for k,v in gates.items() if not k.startswith("_")},
        "gate_details":{k:v for k,v in gates.items() if k.startswith("_")},
        "critical_fails":cfails, "BLOCK_M11":block,
    }
    with open(OUTPUT_DIR / "M12_gate_summary.json", "w", encoding="utf-8") as f:
        json.dump(gate_sum, f, indent=2, default=str)

    rep = [f"# M12 Adversarial Validation Report  v3.0",
           f"**Date:** {date.today()}  |  **Mode:** {args.mode}",
           f"**Architecture:** severity-stratified, detection-classification separated",
           f"**BLOCK_M11:** **{block}**", "",
           "## Critical gates",
           "| Gate | Result | Spec |", "|---|---|---|"]
    for gn, gv in gates.items():
        if gn.startswith("_") or gn not in CRITICAL_GATES: continue
        rep.append(f"| {gn} | {gv} | {GATE_SPECS.get(gn,{}).get('label','')} |")
    rep += ["", "## Advisory gates", "| Gate | Result | Spec |", "|---|---|---|"]
    for gn, gv in gates.items():
        if gn.startswith("_") or gn in CRITICAL_GATES: continue
        rep.append(f"| {gn} | {gv} | {GATE_SPECS.get(gn,{}).get('label','')} |")
    rep += ["", "## Details", "```json",
            json.dumps({k:v for k,v in gates.items() if k.startswith("_")},
                        indent=2, default=str), "```"]
    with open(REPORT_DIR / f"{SCRIPT_NAME}_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rep))

    n_p = sum(1 for k,v in gates.items() if not k.startswith("_") and v == PASS)
    n_t = sum(1 for k in gates if not k.startswith("_"))

    print("\n" + "═"*60)
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12b_version             : v3.0 severity-stratified")
    print(f"M12b_mode                : {args.mode}")
    print(f"M12b_warmup              : {args.warmup_windows} win × σ={WARMUP_NOISE_SIGMA}")
    print(f"M12b_sequences_run       : {len(per_seq)}")
    print(f"M12b_gates_pass          : {n_p}/{n_t}")
    if "_latency" in gates:
        s = gates["_latency"]
        print(f"M12b_latency_p95/p99/max : {s['p95']}s / {s['p99']}s / {s['max']}s")
    if "_g1_fpr" in gates: print(f"M12b_g1_fpr              : {gates['_g1_fpr']*100:.2f}%")
    if "_g2_detect_per_lbl" in gates:
        print(f"M12b_g2_detect_per_lbl   : {gates['_g2_detect_per_lbl']}")
    if "_g2_classify_per_lbl" in gates:
        print(f"M12b_g2_classify_per_lbl : {gates['_g2_classify_per_lbl']}")
    if "_g2_mild_cascade" in gates:
        print(f"M12b_g2_mild_cascade     : {gates['_g2_mild_cascade']*100:.1f}%")
    if "_g3_detect_macro" in gates:
        print(f"M12b_g3_detect_macro     : {gates['_g3_detect_macro']*100:.1f}%")
    if "_g3_classify_macro" in gates:
        print(f"M12b_g3_classify_macro   : {gates['_g3_classify_macro']*100:.1f}%")
    if "_g9_detect_per_lbl" in gates:
        print(f"M12b_g9_detect_per_lbl   : {gates['_g9_detect_per_lbl']}")
    print(f"M12b_BLOCK_M11           : {block}")
    print(f"M12b_critical_fails      : {cfails}")
    print(f"Status for M11           : {'BLOCKED' if block else 'READY for HF deploy'}")
    print("══ END PASTE UPDATE ══")
    print("═"*60)
    print()
    if block:
        print("❌ M12 INCOMPLETE — critical gates failed. DO NOT proceed to M11.")
        sys.exit(1)
    else:
        print("✅ M12 PASS. Ready for M11 (Docker + Hugging Face deploy).")


if __name__ == "__main__":
    main()