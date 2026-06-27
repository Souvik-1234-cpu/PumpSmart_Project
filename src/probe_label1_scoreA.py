# src/probe_label1_scoreA.py
# Decisive probe: does score_A separate a real label-1 (bearing) window from normal?
# No model load, no prod patch — uses the live /api/anomaly_detect.
# v2: cluster-matched selection (both sequences from cluster_id==1 steady_state)
import json, pickle, sys
from pathlib import Path
import numpy as np
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from config import SYNTH_DIR
    SEARCH = [Path(SYNTH_DIR), Path(".")]
except Exception:
    SEARCH = [Path("."), Path("data"), Path("synthetic")]

URL = "http://localhost:8000/api/anomaly_detect"
PKL = "M6B_combined_sequences.pkl"

def log(m): print(m, flush=True)

# ── locate pkl ───────────────────────────────────────────────────────────
path = None
for d in SEARCH:
    for c in list(d.rglob(PKL)) + ([d / PKL] if (d / PKL).exists() else []):
        path = c; break
    if path: break
if not path:
    log(f"❌ {PKL} not found in {[str(s) for s in SEARCH]}"); sys.exit(1)
log(f"Found pkl: {path}")

with open(path, "rb") as f:
    obj = pickle.load(f)

# ── introspect structure ─────────────────────────────────────────────────
log(f"\nTop-level type: {type(obj)}")
if isinstance(obj, dict):
    log(f"Keys: {list(obj.keys())}")
    for k, v in obj.items():
        shape = getattr(v, "shape", None)
        log(f"  {k}: type={type(v).__name__} shape={shape} "
            f"{'len='+str(len(v)) if shape is None and hasattr(v,'__len__') else ''}")

# ── extract sequences + labels from metadata ─────────────────────────────
seqs = obj["sequences"]
meta = obj["metadata"]
log(f"\nmetadata[0] type: {type(meta[0])}")
if isinstance(meta[0], dict):
    log(f"metadata[0] keys: {list(meta[0].keys())}")
    log(f"metadata[0] sample: {meta[0]}")

LABEL_KEY = None
for k in ("label", "label_int", "fault_label", "y", "class", "class_int"):
    if isinstance(meta[0], dict) and k in meta[0]:
        LABEL_KEY = k; break
if LABEL_KEY is None:
    raise SystemExit(f"❌ no label key in metadata[0]={meta[0]}")
log(f"Using label key: '{LABEL_KEY}'")

y = np.array([int(m[LABEL_KEY]) for m in meta])
X = seqs
log(f"\nExtracted: n={len(X)}  labels present={sorted(set(y.tolist()))}")

# ── print cluster inventory so we can see what's available ───────────────
log("\nCluster inventory (label × cluster_id counts):")
from collections import Counter
cc = Counter((int(m[LABEL_KEY]), m.get("cluster_id", "?")) for m in meta)
for lbl in [0, 1]:
    counts = {k[1]: v for k, v in cc.items() if k[0] == lbl}
    log(f"  label {lbl}: {dict(sorted(counts.items()))}")

# ── helper: window from sequence ─────────────────────────────────────────
def window_from(seq, where="late"):
    T = seq.shape[0]
    if T < 50: raise SystemExit(f"❌ sequence too short ({T})")
    start = max(0, T - 50) if where == "late" else (T // 2 - 25)
    return seq[start:start + 50, :]

# ── helper: POST one window, return score_A ───────────────────────────────
def probe(name, win, cluster="steady_state"):
    r = requests.post(URL, json={"window": win.tolist(),
                                  "pump_id": "PUMP-0032", "cluster": cluster})
    d = r.json()
    log(f"\n── {name} ──")
    log(f"  window per-channel mean (raw)  : {np.round(win.mean(axis=0), 4).tolist()}")
    log(f"  window per-channel max  (raw)  : {np.round(win.max(axis=0), 4).tolist()}")
    log(f"  score_A={d['score_A']:.6f}  θ_t={d['adaptive_threshold']:.6f}  "
        f"state={d['alert_state']}  label={d['fault_label']}")
    return d["score_A"]

# ── cluster-matched selection ─────────────────────────────────────────────
# Try cluster_id == 1 (steady_state) first; fall back to any cluster if empty.
def pick(label, cluster_id, want="max_sev"):
    idx = [i for i in range(len(X))
           if y[i] == label and meta[i].get("cluster_id") == cluster_id]
    if not idx:
        return None, None
    if want == "max_sev":
        idx.sort(key=lambda i: meta[i].get("severity", 0), reverse=True)
    return np.asarray(X[idx[0]], dtype=np.float32), meta[idx[0]]

TARGET_CLUSTER = 1   # cluster_id==1  =  steady_state
normal_seq,  nm = pick(0, TARGET_CLUSTER, want="any")
bearing_seq, bm = pick(1, TARGET_CLUSTER, want="max_sev")

# fallback: search all clusters if cluster_id==1 is empty for either
if normal_seq is None:
    log(f"⚠️  no label-0 in cluster {TARGET_CLUSTER} — using any cluster")
    idx0 = [i for i in range(len(X)) if y[i] == 0]
    normal_seq, nm = np.asarray(X[idx0[0]], dtype=np.float32), meta[idx0[0]]
if bearing_seq is None:
    log(f"⚠️  no label-1 in cluster {TARGET_CLUSTER} — using highest-severity any cluster")
    idx1 = sorted([i for i in range(len(X)) if y[i] == 1],
                  key=lambda i: meta[i].get("severity", 0), reverse=True)
    bearing_seq, bm = np.asarray(X[idx1[0]], dtype=np.float32), meta[idx1[0]]

log(f"\nnormal  cluster={nm.get('cluster_id')} sev={nm.get('severity', 0):.3f} steps={nm.get('steps')}")
log(f"bearing cluster={bm.get('cluster_id')} sev={bm.get('severity', 0):.3f} steps={bm.get('steps')}")

# ── run probes ────────────────────────────────────────────────────────────
# Use "steady_state" for both (matches production warmup cluster).
# If sequences come from a different cluster, we'll see it in the raw means.
sA_norm = probe("LABEL 0 — normal (mid window)",   window_from(normal_seq,  "mid"))
sA_flt  = probe("LABEL 1 — bearing (late window)", window_from(bearing_seq, "late"))

log("\n══════════════════════════════════════════════════════")
log(f"  normal  score_A : {sA_norm:.6f}")
log(f"  bearing score_A : {sA_flt:.6f}")
log(f"  separation      : {sA_flt - sA_norm:+.6f}")
log("══════════════════════════════════════════════════════")
log("  CHECK uvicorn terminal for [PROBE] mae_per_ch lines.")
log("  Match each [PROBE] line to the score_A printed above.")
log("  Index 0 = Mot.SV (bearing primary channel).")
log("  Index order: [Mot.SV, Pmp.SV, Mot.TV, Pmp.PV, Temp.SV, Pres.SV, Pmp.TV, Mot.PV]")
log("══════════════════════════════════════════════════════")