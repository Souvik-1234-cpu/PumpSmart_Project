# src/probe_streaming_theta_chase.py
# INDUSTRIAL-VALID: every window posted is real M6B physics — NO synthetic noise.
#   Warmup  = real label-0 (normal) cluster-1 sequences  -> adapts theta_t like deployment
#   Stream  = real label-1 (bearing) cluster-1 sequence, severity 1.0, window-by-window
# Tests the C-25 Adaptive Threshold Paradox:
#   HYPOTHESIS: theta_t chases the ramping score_A, so the level path never crosses,
#   so a gradual in-envelope fault is never escalated.
# No model load, no prod patch — uses the live /api/anomaly_detect only.
import pickle, sys
from pathlib import Path
import numpy as np
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from config import SYNTH_DIR
    SEARCH = [Path(SYNTH_DIR), Path(".")]
except Exception:
    SEARCH = [Path("."), Path("data"), Path("synthetic")]

URL            = "http://localhost:8000/api/anomaly_detect"
PKL            = "M6B_combined_sequences.pkl"
WARMUP_N       = 432    # match the runner's warmup window count exactly
STRIDE_WARMUP  = 4      # slide stride across normal sequences during warmup
STRIDE_FAULT   = 4      # slide stride across the fault sequence
TARGET_CLUSTER = 1      # steady_state

def log(m): print(m, flush=True)

def post(win, cluster="steady_state"):
    r = requests.post(URL, json={"window": win.tolist(),
                                  "pump_id": "PUMP-0032", "cluster": cluster})
    return r.json()

def windows_from(seq, stride):
    T = seq.shape[0]
    return [seq[s:s+50, :] for s in range(0, T - 50 + 1, stride)]

# -- locate + load pkl -----------------------------------------------------
path = None
for d in SEARCH:
    for c in list(d.rglob(PKL)) + ([d / PKL] if (d / PKL).exists() else []):
        path = c; break
    if path: break
if not path:
    log(f"X {PKL} not found"); sys.exit(1)
log(f"Found pkl: {path}")
with open(path, "rb") as f:
    obj = pickle.load(f)
seqs, meta = obj["sequences"], obj["metadata"]
y = np.array([int(m["label"]) for m in meta])

# -- collect REAL label-0 cluster-1 windows for warmup ---------------------
normal_idx = [i for i in range(len(seqs))
              if y[i] == 0 and meta[i].get("cluster_id") == TARGET_CLUSTER]
if not normal_idx:
    log("X no cluster-1 label-0 (normal) sequences for warmup"); sys.exit(1)
warmup_windows = []
for i in normal_idx:
    s = np.asarray(seqs[i], dtype=np.float32)
    if s.shape[0] >= 50:
        warmup_windows.extend(windows_from(s, STRIDE_WARMUP))
    if len(warmup_windows) >= WARMUP_N:
        break
warmup_windows = warmup_windows[:WARMUP_N]
log(f"Warmup pool: {len(warmup_windows)} real normal windows "
    f"from {len(normal_idx)} cluster-1 label-0 sequences")

# -- select highest-severity in-envelope label-1 cluster-1 fault -----------
fidx = [i for i in range(len(seqs))
        if y[i] == 1 and meta[i].get("cluster_id") == TARGET_CLUSTER]
fidx.sort(key=lambda i: meta[i].get("severity", 0), reverse=True)
if not fidx:
    log("X no cluster-1 label-1 (bearing) sequence"); sys.exit(1)
fseq = np.asarray(seqs[fidx[0]], dtype=np.float32)
fsev = meta[fidx[0]].get("severity", 0)
log(f"Fault seq: real bearing_wear cluster=1 sev={fsev:.3f} steps={fseq.shape[0]}")

# -- 1. WARMUP on real normal data -----------------------------------------
log(f"\nWarmup: streaming {len(warmup_windows)} REAL normal windows...")
theta_w = None
for w in warmup_windows:
    theta_w = post(w)["adaptive_threshold"]
log(f"Warmup done. theta_t adapted -> {theta_w:.6f}")

# -- 2. STREAM the real fault sequence --------------------------------------
log(f"\nStreaming REAL bearing fault (stride={STRIDE_FAULT})...")
log(f"{'win':>4} {'score_A':>9} {'theta_t':>9} {'1.5theta':>9} {'d(sA-th)':>9} {'state':>7}  Mot.SV_mean")
log("-" * 74)
rows, escalated, peak_gap = [], False, -9.0
for start in range(0, fseq.shape[0] - 50 + 1, STRIDE_FAULT):
    win = fseq[start:start+50, :]
    d = post(win)
    sA, th, st = d["score_A"], d["adaptive_threshold"], d["alert_state"]
    gap = sA - th
    peak_gap = max(peak_gap, gap)
    if st in ("WARN", "DANGER"): escalated = True
    rows.append((start, sA, th, st))
    log(f"{start:>4} {sA:>9.5f} {th:>9.5f} {1.5*th:>9.5f} {gap:>+9.5f} {st:>7}  {float(win[:,0].mean()):.3f}")

# -- 3. verdict -------------------------------------------------------------
sA_arr = np.array([r[1] for r in rows]); th_arr = np.array([r[2] for r in rows])
log("\n" + "=" * 74)
log(f"  warmup theta_t (real normal): {theta_w:.5f}")
log(f"  score_A  min->max           : {sA_arr.min():.5f} -> {sA_arr.max():.5f}")
log(f"  theta_t  min->max           : {th_arr.min():.5f} -> {th_arr.max():.5f}   (did theta_t climb with the fault?)")
log(f"  peak gap (score_A-theta_t)  : {peak_gap:+.5f}   (must be > 0 for level-path WARN)")
log(f"  EVER escalated WARN/DANGER during stream: {escalated}")
log("=" * 74)
log("  theta_t climbed with score_A, peak gap <= 0, escalated=False")
log("    -> CONFIRMED C-25: L4 chases the ramp. Fix = crosspoint-guard re-baseline.")
log("  peak gap > 0 / escalated=True")
log("    -> theta_t did NOT chase; detection problem is elsewhere (runner criterion / feed).")
log("=" * 74)