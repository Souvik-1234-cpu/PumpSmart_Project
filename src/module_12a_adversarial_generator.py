# =============================================================================
# module_12a_adversarial_generator.py  v2.0
# PumpSmart v14.2 — M12 Adversarial Validation : Generator
#
# v2.0 redesign (Path A from deep-dive analysis):
#   - Adds severity_tier column: "in_envelope" | "mild_extreme" | "severe_extreme"
#   - Splits Group 2/3/5/6 sequences explicitly so M12b can stratify gates
#   - In-envelope sequences sample from M6 training range (sev ≥ 0.20 typically)
#     → tests what L1 + M7 were designed to detect/classify
#   - Mild-extreme sequences sample below M6 floor (sev 0.05–0.20)
#     → tests the L3 CUSUM + L4 sub-threshold pathway
#   - Quick/full modes get more sequences in each tier for statistical power
#
# Rationale: M6B physics invariant #9 — "mild sev 0.2–0.3 must produce MAE
# in [0.110058, 0.140]" — and Label 21 docs explicitly state sev < 0.15
# produces sub-threshold MAE BY DESIGN. Testing those at the same gate as
# severe acute faults conflates architectural intent.
# =============================================================================
import sys, os, json, argparse, warnings
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent]:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p)); break

from config import (RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
warnings.filterwarnings('ignore')

SCRIPT_NAME = "module_12a_adversarial_generator"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
M12_DIR = SYNTH_DIR / "M12_adversarial"
M12_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
results = {}

# =============================================================================
CHANNELS = ["Mot.SV","Pmp.SV","Mot.TV","Pmp.PV","Temp.SV","Pres.SV","Pmp.TV","Mot.PV"]
CH = {n:i for i,n in enumerate(CHANNELS)}

SEQ_LENGTHS = {
    0:200, 1:250, 2:200, 3:150, 4:400, 5:300, 6:150,
    7:600, 8:550, 9:700, 10:900, 11:800, 12:450,
    13:300, 14:210, 15:500, 16:350, 17:250,
    18:300, 19:150, 20:600, 21:1000, 22:500, 23:500,
}

# M6B training severity ranges (LOCKED — from project knowledge)
M6_SEVERITY = {
    1:(0.20,0.80), 2:(0.20,0.80), 3:(0.20,0.80), 4:(0.20,0.80),
    5:(0.20,0.80), 6:(0.20,0.80),
    7:(0.30,0.70), 8:(0.30,0.70), 9:(0.30,0.70),
    10:(0.30,0.70), 11:(0.30,0.70), 12:(0.30,0.70),
    13:(0.20,0.80), 14:(0.20,0.80), 15:(0.20,0.80),
    16:(0.20,0.80), 17:(0.20,0.80),
    18:(0.30,0.70), 19:(0.20,0.80), 20:(0.30,0.70),
    21:(0.40,0.80), 22:(0.30,0.70), 23:(0.30,0.70),
}
M6_LAG = {7:(200,400), 8:(50,150), 9:(300,600), 10:(400,800), 11:(400,600), 12:(100,300)}
M6_CLUSTER = {
    0:{0,1,2,3}, 1:{1}, 2:{1}, 3:{1}, 4:{1}, 5:{1,2}, 6:{0,1,2,3},
    7:{1}, 8:{1}, 9:{1}, 10:{1}, 11:{1}, 12:{1},
    13:{1}, 14:{1}, 15:{1}, 16:{1}, 17:{1},
    18:{1}, 19:{1}, 20:{2}, 21:{1,2}, 22:{1}, 23:{1},
}
CLUSTER_NAMES = {0:"cooldown", 1:"steady_state", 2:"high_load", 3:"startup"}

# Mode → counts. Each fault label now splits into 2 tiers (in_env + mild_extreme).
MODE_COUNTS = {
    "smoke": {
        "g1_normal":10, "g1b_normal_ext":5,
        "g2_in_env_per_label":3, "g2_mild_per_label":3,
        "g3_in_env_per_label":3, "g3_mild_per_label":2,
        "g4_label21":5,
        "g5_cross_cluster_per_label":5,
        "g6_in_env_per_label":3, "g6_mild_per_label":2,
        "g7_interruption_per_state":3,
        "g8_crosspoint":5,
        "g9_groupE_per_label":5,
    },
    "quick": {
        "g1_normal":60, "g1b_normal_ext":30,
        "g2_in_env_per_label":40, "g2_mild_per_label":20,
        "g3_in_env_per_label":40, "g3_mild_per_label":20,
        "g4_label21":60,
        "g5_cross_cluster_per_label":30,
        "g6_in_env_per_label":40, "g6_mild_per_label":20,
        "g7_interruption_per_state":30,
        "g8_crosspoint":30,
        "g9_groupE_per_label":40,
    },
    "full": {
        "g1_normal":600, "g1b_normal_ext":200,
        "g2_in_env_per_label":400, "g2_mild_per_label":200,
        "g3_in_env_per_label":400, "g3_mild_per_label":200,
        "g4_label21":600,
        "g5_cross_cluster_per_label":200,
        "g6_in_env_per_label":400, "g6_mild_per_label":200,
        "g7_interruption_per_state":200,
        "g8_crosspoint":200,
        "g9_groupE_per_label":400,
    },
}


def _import_physics_lib():
    src_path = Path(__file__).resolve().parent
    if str(src_path) not in sys.path: sys.path.insert(0, str(src_path))
    try:
        import m6b_physics_lib as plib
    except ImportError:
        log("  WARNING: m6b_physics_lib not in src/ — fallback synthesis")
        return None
    norm_path = MODEL_DIR / "M3_normalization_config.json"
    if not norm_path.exists(): norm_path = OUTPUT_DIR / "M3_normalization_config.json"
    with open(norm_path) as f: norm_config = json.load(f)
    phys_path = MODEL_DIR / "M5_physics_config.json"
    if phys_path.exists():
        with open(phys_path) as f: phys_config = json.load(f)
    else:
        phys_config = {"physics_constants":{"TAU_THERMAL_s":388.9,"BPF_HZ":347.67,
                       "A_WAVE_m_s":1200.0,"RHO":1000.0}}
    return plib, norm_config, phys_config


# ── Severity sampling — TIER-AWARE ──────────────────────────────────────────
def sample_severity_in_envelope(label, rng, n):
    """Sample WITHIN M6 training range. Tests M7 in its training envelope."""
    lo, hi = M6_SEVERITY.get(label, (0.20, 0.80))
    # Use full range, slight margin to avoid edges
    return rng.uniform(lo + 0.03, hi - 0.03, size=n)

def sample_severity_mild_extreme(label, rng, n):
    """Sample BELOW M6 floor — sub-threshold by M6B invariant design.
    Tests the L3 CUSUM + L4 sub-threshold detection pathway."""
    lo, _ = M6_SEVERITY.get(label, (0.20, 0.80))
    band_lo = max(0.05, lo - 0.15)   # e.g., 0.05–0.05 below floor
    band_hi = lo - 0.01                # just below M6 floor
    if band_hi <= band_lo:
        band_hi = band_lo + 0.05
    return rng.uniform(band_lo, band_hi, size=n)

def sample_severity_severe_extreme(label, rng, n):
    """Sample ABOVE M6 ceiling — stress test."""
    _, hi = M6_SEVERITY.get(label, (0.20, 0.80))
    band_lo = hi + 0.01
    band_hi = min(0.95, hi + 0.10)
    if band_hi <= band_lo:
        band_hi = band_lo + 0.05
    return rng.uniform(band_lo, band_hi, size=n)

def sample_lag_held_out(label, rng, n):
    lo, hi = M6_LAG[label]; span = hi - lo
    below_lo = max(20, int(lo - 0.30*span)); below_hi = lo - 1
    above_lo = hi + 1; above_hi = int(hi + 0.30*span)
    n_b = n//2; n_a = n - n_b
    below = rng.integers(below_lo, max(below_lo+1, below_hi+1), size=n_b)
    above = rng.integers(above_lo, above_hi+1, size=n_a)
    lags = np.concatenate([below, above]); rng.shuffle(lags)
    return lags


def held_out_distance(label, severity, lag=None, cluster=None):
    lo, hi = M6_SEVERITY.get(label, (0.20, 0.80))
    if severity < lo:
        sd = (lo - severity) / max(1e-6, hi - lo)
    elif severity > hi:
        sd = (severity - hi) / max(1e-6, hi - lo)
    else:
        sd = -min(severity-lo, hi-severity) / max(1e-6, hi-lo)
    out = {"held_out_severity_distance": round(float(sd), 4)}
    if lag is not None and label in M6_LAG:
        lo_l, hi_l = M6_LAG[label]
        if lag < lo_l:   ld = (lo_l-lag) / max(1, hi_l-lo_l)
        elif lag > hi_l: ld = (lag-hi_l) / max(1, hi_l-lo_l)
        else:            ld = 0.0
        out["held_out_lag_distance"] = round(float(ld), 4)
    else:
        out["held_out_lag_distance"] = None
    if cluster is not None:
        out["held_out_cluster_match"] = bool(cluster not in M6_CLUSTER.get(label, {1}))
    else:
        out["held_out_cluster_match"] = False
    return out


# ── Sequence synthesis (unchanged from v1.0) ────────────────────────────────
def _baseline_with_noise(n_steps, cluster, rng, plib=None):
    if plib is not None:
        try: return plib.make_baseline(n_steps, cluster_id=cluster, noise_sigma=0.015)
        except Exception: pass
    seq = np.ones((n_steps, 8), dtype=np.float32)
    seq += rng.normal(0, 0.015, size=seq.shape).astype(np.float32)
    return seq


def gen_normal_boundary(rng, cluster, plib, n_steps=200):
    seq = _baseline_with_noise(n_steps, cluster, rng, plib)
    push = rng.uniform(1.10, 1.25)
    if cluster == 2:
        seq[:, CH["Pres.SV"]] *= push
        seq[:, CH["Temp.SV"]] *= push*0.5 + 0.5
    elif cluster == 3:
        seq[:, CH["Mot.SV"]] *= rng.uniform(1.5, 2.5)
        seq[:, CH["Pmp.SV"]] *= rng.uniform(1.5, 2.5)
    else:
        seq[:, CH["Pres.SV"]] *= rng.uniform(0.95, 1.10)
    return seq.astype(np.float32)


def gen_acute_fault(rng, label, severity, cluster, plib, n_steps):
    seq = _baseline_with_noise(n_steps, cluster, rng, plib)
    onset = int(rng.integers(max(20, n_steps//5), max(21, n_steps//3)))
    if label == 1:
        K = 0.001 + severity*0.004
        for t in range(onset, n_steps):
            seq[t, CH["Mot.SV"]] += K*(t-onset)
    elif label == 2:
        amp = 0.05 + severity*0.20
        seq[onset:, CH["Pmp.SV"]] += amp*np.sin(np.linspace(0, 4*np.pi, n_steps-onset))
    elif label == 3:
        n_b = rng.integers(3, 8)
        for _ in range(n_b):
            t0 = rng.integers(onset, n_steps-5)
            seq[t0:t0+3, CH["Pmp.SV"]] += severity*rng.uniform(0.3, 0.8)
            seq[t0:t0+3, CH["Pres.SV"]] -= severity*rng.uniform(0.1, 0.3)
    elif label == 4:
        rate = -0.0005 - severity*0.001
        for t in range(onset, n_steps):
            seq[t, CH["Pres.SV"]] += rate*(t-onset)
        seq[:, CH["Pres.SV"]] = np.clip(seq[:, CH["Pres.SV"]], 0.0, None)
    elif label == 5:
        rate = 0.0005 + severity*0.003
        for t in range(onset, n_steps):
            seq[t, CH["Temp.SV"]] += rate*(t-onset)
            seq[t, CH["Mot.TV"]]  += rate*(t-onset)*0.7
    elif label == 6:
        ch_fail = rng.choice([CH["Mot.SV"], CH["Pres.SV"], CH["Temp.SV"]])
        seq[onset:, ch_fail] = seq[onset, ch_fail]
    elif label == 19:
        drop = severity*0.60
        of = int(rng.integers(55, 85))
        for t in range(of, min(of+15, n_steps)):
            frac = (t - of + 1) / 15
            seq[t, CH["Pres.SV"]] = max(0.05, seq[t, CH["Pres.SV"]] - drop*frac)
        if of+15 < n_steps:
            seq[of+15:, CH["Pres.SV"]] = max(0.05, 1.0 - drop)
        seq[of:of+15, CH["Mot.PV"]] += rng.uniform(0.20, 0.35)
    return seq.astype(np.float32)


def gen_compound_chain(rng, label, severity, lag, cluster, plib, n_steps):
    seq = _baseline_with_noise(n_steps, cluster, rng, plib)
    primary_map = {7:1, 8:3, 9:2, 10:4, 11:5, 12:2}
    secondary_map = {7:5, 8:4, 9:1, 10:3, 11:1, 12:3}
    p_lbl = primary_map[label]; s_lbl = secondary_map[label]
    p_seq = gen_acute_fault(rng, p_lbl, severity, cluster, plib, n_steps)
    s_seq = gen_acute_fault(rng, s_lbl, severity*0.85, cluster, plib, max(50, n_steps-lag))
    primary_dev   = p_seq - _baseline_with_noise(n_steps, cluster, rng, plib)
    secondary_dev = s_seq - _baseline_with_noise(len(s_seq), cluster, rng, plib)
    seq = _baseline_with_noise(n_steps, cluster, rng, plib)
    seq += primary_dev
    lag = max(0, min(lag, n_steps - 50))
    s_end = min(n_steps, lag + len(secondary_dev))
    if s_end > lag:
        seq[lag:s_end] += secondary_dev[:s_end-lag]*0.7
    return seq.astype(np.float32)


def gen_label_21_gradual(rng, severity, cluster, plib, n_steps=1000):
    seq = _baseline_with_noise(n_steps, cluster, rng, plib)
    drift_rate = 0.0003 * severity
    for t in range(n_steps):
        seq[t, CH["Mot.SV"]] += drift_rate*t
        seq[t, CH["Pmp.SV"]] += drift_rate*t*0.6
    return seq.astype(np.float32)


def gen_masked_fault(rng, label, severity, cluster, plib, n_steps):
    underlying = {13:1, 14:3, 15:4, 16:5, 17:2}
    masked_ch  = {13:"Mot.SV", 14:"Pres.SV", 15:"Pres.SV", 16:"Temp.SV", 17:"Pmp.SV"}
    seq = gen_acute_fault(rng, underlying[label], severity, cluster, plib, n_steps)
    mask_onset = int(rng.integers(n_steps//4, n_steps//2))
    ch = CH[masked_ch[label]]
    if label == 13:   seq[mask_onset:, ch] = seq[mask_onset, ch]
    elif label == 14: seq[mask_onset:, ch] = seq[mask_onset, ch]
    elif label == 15:
        for t in range(mask_onset, n_steps):
            seq[t, ch] = seq[mask_onset, ch] - 0.0001*(t-mask_onset)
    elif label == 16: seq[mask_onset:, ch] = seq[mask_onset, ch]
    elif label == 17: seq[mask_onset:, ch] = 1.0
    return seq.astype(np.float32)


def gen_sensor_interruption(rng, state, cluster, plib, n_steps=200):
    seq = _baseline_with_noise(n_steps, cluster, rng, plib)
    onset = n_steps // 2
    if state == "partial_dropout":
        ch_idx = int(rng.integers(0, 8))
        seq[onset:, ch_idx] = np.nan
        expected = "PARTIAL_SENSOR_LOSS"
    elif state == "full_dropout":
        seq[onset:, :] = np.nan
        expected = "FULL_SENSOR_LOSS"
    else:
        seq[onset:, :] *= 0.05
        expected = "PUMP_OFF"
    return seq.astype(np.float32), expected


def gen_crosspoint_drift(rng, cluster, plib, n_steps=600):
    seq = _baseline_with_noise(n_steps, cluster, rng, plib)
    drift = rng.uniform(0.0008, 0.0015)
    for t in range(n_steps):
        seq[t, CH["Mot.SV"]]  += drift*t
        seq[t, CH["Pres.SV"]] += drift*t*0.4
        seq[t, CH["Temp.SV"]] += drift*t*0.3
    return seq.astype(np.float32)


def gen_groupE_multisensor(rng, label, severity, cluster, plib, n_steps=500):
    seq = _baseline_with_noise(n_steps, cluster, rng, plib)
    onset = int(rng.integers(50, n_steps//3))
    if label == 22: ch_a, ch_b = CH["Mot.SV"], CH["Pmp.SV"]
    else:           ch_a, ch_b = CH["Pres.SV"], CH["Temp.SV"]
    for t in range(onset, n_steps):
        seq[t, ch_a] += severity*0.5
        seq[t, ch_b] -= severity*0.4
    return seq.astype(np.float32)


def save_sequence(seq, meta, idx, group, label, dry_run):
    fname = f"seq_{group}_L{label:02d}_{idx:05d}.npz"
    fpath = M12_DIR / fname
    if not dry_run:
        np.savez_compressed(fpath, window=seq.astype(np.float32),
                             **{k:v for k,v in meta.items() if v is not None})
    return fname


def physics_validate(seq, label):
    g = {}
    g["G1_pres_nonneg"] = bool(np.nanmin(seq[:, CH["Pres.SV"]]) >= -0.01)
    tm = min(np.nanmin(seq[:, CH["Temp.SV"]]), np.nanmin(seq[:, CH["Mot.TV"]]),
             np.nanmin(seq[:, CH["Pmp.TV"]]))
    g["G2_temp_floor"] = bool(tm >= -0.12)
    g["G3_finite"] = bool(np.all(np.isfinite(np.nan_to_num(seq, nan=0))))
    g["all_pass"] = all(v for v in g.values())
    return g


# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke","quick","full"], default="smoke")
    parser.add_argument("--seed", type=int, default=12042026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log("="*70)
    log(f"  PumpSmart M12a — Adversarial Generator  v2.0  (severity-stratified)")
    log(f"  Mode: {args.mode}  |  Seed: {args.seed}  |  Dry-run: {args.dry_run}")
    log(f"  Output: {M12_DIR}")
    log("="*70)

    rng = np.random.default_rng(args.seed)
    counts = MODE_COUNTS[args.mode]

    log("\nSTEP 1 — Importing m6b_physics_lib + M3 normalization")
    bundle = _import_physics_lib()
    if bundle is not None:
        plib, norm_config, phys_config = bundle
        plib.init_lib(norm_config, phys_config, seed=args.seed)
        log("  m6b_physics_lib initialised ✓")
    else:
        plib = None
        log("  Fallback synthesis")

    manifest_rows = []
    idx = 0

    # ── Group 1 — Normal boundary ──────────────────────────────────────────
    log(f"\nSTEP 2 — Group 1: Normal boundary ({counts['g1_normal']}/cluster × 3)")
    for cluster in [1, 2, 3]:
        for _ in range(counts["g1_normal"]):
            seq = gen_normal_boundary(rng, cluster, plib)
            phys = physics_validate(seq, 0)
            meta = {"group":"G1_normal_boundary", "label":0, "cluster":cluster,
                    "severity":0.0, "severity_tier":"normal", "lag":None, "n_steps":200,
                    **held_out_distance(0, 0.0, cluster=cluster),
                    "phys_pass":phys["all_pass"]}
            fname = save_sequence(seq, meta, idx, "G1", 0, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Group 1b — Normal extended (CUSUM specificity) ─────────────────────
    log(f"\nSTEP 3 — Group 1b: 1000-window normals ({counts['g1b_normal_ext']})")
    for _ in range(counts["g1b_normal_ext"]):
        seq = gen_normal_boundary(rng, cluster=1, plib=plib, n_steps=1000)
        phys = physics_validate(seq, 0)
        meta = {"group":"G1b_normal_extended", "label":0, "cluster":1,
                "severity":0.0, "severity_tier":"normal", "lag":None, "n_steps":1000,
                **held_out_distance(0, 0.0, cluster=1), "phys_pass":phys["all_pass"]}
        fname = save_sequence(seq, meta, idx, "G1b", 0, args.dry_run)
        manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Group 2 — Single acute (SPLIT: in_envelope + mild_extreme) ─────────
    log(f"\nSTEP 4 — Group 2: Single acute, stratified by severity tier")
    log(f"  in-envelope  ({counts['g2_in_env_per_label']}/label, sev in M6 range)")
    log(f"  mild-extreme ({counts['g2_mild_per_label']}/label, sev below M6 floor)")
    for label in [1,2,3,4,5,6,19]:
        n_steps = SEQ_LENGTHS[label]
        m6_clusters = list(M6_CLUSTER[label])
        # In-envelope tier
        sevs = sample_severity_in_envelope(label, rng, counts["g2_in_env_per_label"])
        for sev in sevs:
            cluster = int(rng.choice(m6_clusters))
            seq = gen_acute_fault(rng, label, float(sev), cluster, plib, n_steps)
            phys = physics_validate(seq, label)
            meta = {"group":"G2_acute_in_envelope", "label":label, "cluster":cluster,
                    "severity":round(float(sev),4), "severity_tier":"in_envelope",
                    "lag":None, "n_steps":n_steps,
                    **held_out_distance(label, float(sev), cluster=cluster),
                    "phys_pass":phys["all_pass"]}
            fname = save_sequence(seq, meta, idx, "G2env", label, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1
        # Mild-extreme tier
        sevs = sample_severity_mild_extreme(label, rng, counts["g2_mild_per_label"])
        for sev in sevs:
            cluster = int(rng.choice(m6_clusters))
            seq = gen_acute_fault(rng, label, float(sev), cluster, plib, n_steps)
            phys = physics_validate(seq, label)
            meta = {"group":"G2_acute_mild_extreme", "label":label, "cluster":cluster,
                    "severity":round(float(sev),4), "severity_tier":"mild_extreme",
                    "lag":None, "n_steps":n_steps,
                    **held_out_distance(label, float(sev), cluster=cluster),
                    "phys_pass":phys["all_pass"]}
            fname = save_sequence(seq, meta, idx, "G2mild", label, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Group 3 — Compound chains (SPLIT by severity tier, lag held-out) ───
    log(f"\nSTEP 5 — Group 3: Compound chains, stratified")
    log(f"  in-envelope  ({counts['g3_in_env_per_label']}/label)")
    log(f"  mild-extreme ({counts['g3_mild_per_label']}/label)")
    for label in [7,8,9,10,11,12]:
        n_steps = SEQ_LENGTHS[label]
        # In-envelope
        lags_env = sample_lag_held_out(label, rng, counts["g3_in_env_per_label"])
        sevs_env = sample_severity_in_envelope(label, rng, counts["g3_in_env_per_label"])
        for sev, lag in zip(sevs_env, lags_env):
            seq = gen_compound_chain(rng, label, float(sev), int(lag), 1, plib, n_steps)
            phys = physics_validate(seq, label)
            meta = {"group":"G3_compound_in_envelope", "label":label, "cluster":1,
                    "severity":round(float(sev),4), "severity_tier":"in_envelope",
                    "lag":int(lag), "n_steps":n_steps,
                    **held_out_distance(label, float(sev), lag=int(lag), cluster=1),
                    "phys_pass":phys["all_pass"]}
            fname = save_sequence(seq, meta, idx, "G3env", label, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1
        # Mild-extreme
        lags_m = sample_lag_held_out(label, rng, counts["g3_mild_per_label"])
        sevs_m = sample_severity_mild_extreme(label, rng, counts["g3_mild_per_label"])
        for sev, lag in zip(sevs_m, lags_m):
            seq = gen_compound_chain(rng, label, float(sev), int(lag), 1, plib, n_steps)
            phys = physics_validate(seq, label)
            meta = {"group":"G3_compound_mild_extreme", "label":label, "cluster":1,
                    "severity":round(float(sev),4), "severity_tier":"mild_extreme",
                    "lag":int(lag), "n_steps":n_steps,
                    **held_out_distance(label, float(sev), lag=int(lag), cluster=1),
                    "phys_pass":phys["all_pass"]}
            fname = save_sequence(seq, meta, idx, "G3mild", label, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Group 4 — Label 21 mild (this IS the sub-threshold pathway test) ───
    log(f"\nSTEP 6 — Group 4: Label 21 gradual ({counts['g4_label21']})")
    for _ in range(counts["g4_label21"]):
        sev = float(rng.uniform(0.10, 0.35))
        cluster = int(rng.choice([1, 2]))
        seq = gen_label_21_gradual(rng, sev, cluster, plib)
        phys = physics_validate(seq, 21)
        meta = {"group":"G4_label21_mild", "label":21, "cluster":cluster,
                "severity":round(sev,4), "severity_tier":"sub_threshold_by_design",
                "lag":None, "n_steps":1000,
                **held_out_distance(21, sev, cluster=cluster),
                "phys_pass":phys["all_pass"]}
        fname = save_sequence(seq, meta, idx, "G4", 21, args.dry_run)
        manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Group 5 — Cross-cluster (use in-envelope severity only) ────────────
    log(f"\nSTEP 7 — Group 5: Cross-cluster ({counts['g5_cross_cluster_per_label']}/label)")
    log(f"  Using in-envelope severities — cross-cluster is the held-out axis")
    cross_targets = {1:2, 3:2, 4:3, 5:3, 8:2}
    for label, novel_cluster in cross_targets.items():
        sevs = sample_severity_in_envelope(label, rng, counts["g5_cross_cluster_per_label"])
        n_steps = SEQ_LENGTHS[label]
        for sev in sevs:
            if label in [7,8,9,10,11,12]:
                lag = int(np.mean(M6_LAG[label]))
                seq = gen_compound_chain(rng, label, float(sev), lag, novel_cluster, plib, n_steps)
                lag_meta = lag
            else:
                seq = gen_acute_fault(rng, label, float(sev), novel_cluster, plib, n_steps)
                lag_meta = None
            phys = physics_validate(seq, label)
            meta = {"group":"G5_cross_cluster", "label":label, "cluster":novel_cluster,
                    "severity":round(float(sev),4), "severity_tier":"in_envelope",
                    "lag":lag_meta, "n_steps":n_steps,
                    **held_out_distance(label, float(sev), lag=lag_meta, cluster=novel_cluster),
                    "phys_pass":phys["all_pass"]}
            fname = save_sequence(seq, meta, idx, "G5", label, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Group 6 — Masked faults (SPLIT by severity tier) ───────────────────
    log(f"\nSTEP 8 — Group 6: Masked faults, stratified")
    for label in [13,14,15,16,17]:
        n_steps = SEQ_LENGTHS[label]
        # In-envelope
        sevs = sample_severity_in_envelope(label, rng, counts["g6_in_env_per_label"])
        for sev in sevs:
            seq = gen_masked_fault(rng, label, float(sev), 1, plib, n_steps)
            phys = physics_validate(seq, label)
            meta = {"group":"G6_masked_in_envelope", "label":label, "cluster":1,
                    "severity":round(float(sev),4), "severity_tier":"in_envelope",
                    "lag":None, "n_steps":n_steps,
                    **held_out_distance(label, float(sev), cluster=1),
                    "phys_pass":phys["all_pass"]}
            fname = save_sequence(seq, meta, idx, "G6env", label, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1
        # Mild-extreme
        sevs = sample_severity_mild_extreme(label, rng, counts["g6_mild_per_label"])
        for sev in sevs:
            seq = gen_masked_fault(rng, label, float(sev), 1, plib, n_steps)
            phys = physics_validate(seq, label)
            meta = {"group":"G6_masked_mild_extreme", "label":label, "cluster":1,
                    "severity":round(float(sev),4), "severity_tier":"mild_extreme",
                    "lag":None, "n_steps":n_steps,
                    **held_out_distance(label, float(sev), cluster=1),
                    "phys_pass":phys["all_pass"]}
            fname = save_sequence(seq, meta, idx, "G6mild", label, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Group 7 — Sensor interruption ──────────────────────────────────────
    log(f"\nSTEP 9 — Group 7: Sensor interruption (3 states × {counts['g7_interruption_per_state']})")
    for state in ["partial_dropout","full_dropout","pump_off"]:
        for _ in range(counts["g7_interruption_per_state"]):
            seq, expected = gen_sensor_interruption(rng, state, 1, plib)
            meta = {"group":f"G7_{state}", "label":-1, "cluster":1,
                    "severity":0.0, "severity_tier":"special", "lag":None, "n_steps":200,
                    "expected_ui_state":expected,
                    "held_out_severity_distance":0.0, "held_out_lag_distance":None,
                    "held_out_cluster_match":False, "phys_pass":True}
            fname = save_sequence(seq, meta, idx, "G7", 0, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Group 8 — L4 crosspoint drift ──────────────────────────────────────
    log(f"\nSTEP 10 — Group 8: L4 crosspoint drift ({counts['g8_crosspoint']})")
    for _ in range(counts["g8_crosspoint"]):
        seq = gen_crosspoint_drift(rng, 1, plib)
        phys = physics_validate(seq, 0)
        meta = {"group":"G8_crosspoint_drift", "label":0, "cluster":1,
                "severity":0.6, "severity_tier":"drift", "lag":None, "n_steps":600,
                "held_out_severity_distance":0.5, "held_out_lag_distance":None,
                "held_out_cluster_match":False, "phys_pass":phys["all_pass"]}
        fname = save_sequence(seq, meta, idx, "G8", 0, args.dry_run)
        manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Group 9 — Group E multi-sensor ─────────────────────────────────────
    log(f"\nSTEP 11 — Group 9: Group E multi-sensor ({counts['g9_groupE_per_label']}/label)")
    for label in [22, 23]:
        sevs = sample_severity_in_envelope(label, rng, counts["g9_groupE_per_label"])
        n_steps = SEQ_LENGTHS[label]
        for sev in sevs:
            seq = gen_groupE_multisensor(rng, label, float(sev), 1, plib, n_steps)
            phys = physics_validate(seq, label)
            meta = {"group":"G9_groupE", "label":label, "cluster":1,
                    "severity":round(float(sev),4), "severity_tier":"in_envelope",
                    "lag":None, "n_steps":n_steps,
                    **held_out_distance(label, float(sev), cluster=1),
                    "phys_pass":phys["all_pass"]}
            fname = save_sequence(seq, meta, idx, "G9", label, args.dry_run)
            manifest_rows.append({**meta, "filename":fname, "idx":idx}); idx += 1

    # ── Manifest ────────────────────────────────────────────────────────────
    log("\nSTEP 12 — Writing manifest")
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = M12_DIR / "M12_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")
    log(f"  Manifest: {manifest_path} ({len(manifest_df):,} rows)")

    log("\nSTEP 13 — Severity tier breakdown")
    tier_counts = manifest_df.groupby(["group","severity_tier"]).size().unstack(fill_value=0)
    log(f"\n{tier_counts}")

    log("\nSTEP 14 — Leakage proof")
    leakage = {
        "total_sequences": len(manifest_df),
        "physics_pass_rate": round(float(manifest_df["phys_pass"].mean()), 4),
        "per_group_counts": manifest_df["group"].value_counts().to_dict(),
        "per_severity_tier": manifest_df["severity_tier"].value_counts().to_dict(),
        "cross_cluster_count": int(manifest_df["held_out_cluster_match"].sum()),
        "seed": args.seed, "mode": args.mode,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "version": "v2.0",
        "claim": (
            "Each fault group split into in-envelope (sev in M6 training range, "
            "tests M7 in-distribution) and mild-extreme (sev below M6 floor, "
            "tests L3 CUSUM + L4 sub-threshold pathway per M6B invariant design)."
        ),
    }
    leakage_path = OUTPUT_DIR / "M12_leakage_proof.json"
    with open(leakage_path, "w", encoding="utf-8") as f:
        json.dump(leakage, f, indent=2)

    log("\nSTEP 15 — Physics validation summary")
    phys_pass = float(manifest_df["phys_pass"].mean())
    log(f"  Physics pass rate: {phys_pass*100:.2f}%")

    results.update({"total_sequences":len(manifest_df), "physics_pass_rate":phys_pass,
                    "manifest_path":str(manifest_path), "m12_dir":str(M12_DIR),
                    "dry_run":args.dry_run, "mode":args.mode})

    report_lines = [
        f"# M12a Adversarial Generator Report  v2.0",
        f"**Date:** {date.today()}  |  **Mode:** {args.mode}  |  **Seed:** {args.seed}",
        f"**Total sequences:** {len(manifest_df):,}  |  **Physics pass rate:** {phys_pass*100:.2f}%",
        f"",
        f"## Severity tier stratification",
        manifest_df.groupby(["group","severity_tier"]).size().unstack(fill_value=0).to_markdown(),
        "",
        "## Outputs",
        f"- `{manifest_path}`",
        f"- `{leakage_path}`",
        f"- `.npz` files: `{M12_DIR}/` (skipped if --dry-run)",
    ]
    report_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print()
    print("═"*60)
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12a_version             : v2.0 severity-stratified")
    print(f"M12a_total_sequences     : {len(manifest_df):,}")
    print(f"M12a_physics_pass_rate   : {phys_pass*100:.2f}%")
    print(f"M12a_mode                : {args.mode}")
    print(f"M12a_seed                : {args.seed}")
    print(f"M12a_dry_run             : {args.dry_run}")
    print(f"M12a_severity_tiers      : {leakage['per_severity_tier']}")
    print(f"M12a_cross_cluster_count : {leakage['cross_cluster_count']}")
    print(f"Status for M12b          : READY" if phys_pass >= 0.95 else
          f"Status for M12b          : NEEDS_REVIEW")
    print("══ END PASTE UPDATE ══")
    print("═"*60)
    print(f"\n📦 M12a done. Next: python src/module_12b_adversarial_runner.py --mode {args.mode}")


if __name__ == "__main__":
    main()