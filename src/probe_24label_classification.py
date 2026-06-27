# src/probe_24label_classification.py
# DIRECT classification test — bypasses the runner and streaming entirely.
# Feeds the MATURED window of one real M6B sequence per label (0..23) straight
# to /api/anomaly_detect and prints true-vs-predicted. This is the confusion
# probe that answers: "Can M7 actually name the 24 fault classes, or not?"
#
# Why direct, not via runner: the runner slices sequences into non-overlapping
# 50-step windows and measures classification only on detection-escalated
# sequences, so it reports `None` for most labels. This probe removes that
# confound: 24 inputs, 24 predictions, one table. ~2 s, no warmup loop.
#
# Cluster-matched: each window is POSTed with the sequence's OWN cluster, so
# M7 sees it in the operating mode it was generated for (avoids the cross-
# cluster artifact that produced false negatives earlier in the session).
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

URL = "http://localhost:8000/api/anomaly_detect"
PKL = "M6B_combined_sequences.pkl"
CLUSTER_NAMES = {0: "cooldown", 1: "steady_state", 2: "high_load", 3: "startup"}
LABEL_NAMES = {
    0:"normal",1:"bearing_wear",2:"impeller_imbalance",3:"cavitation",
    4:"seal_failure",5:"overloading",6:"sensor_failure",
    7:"bearing+overloading",8:"cavitation+seal",9:"imbalance+bearing",
    10:"seal+cavitation_H",11:"overloading+bearing",12:"imbalance+cavitation",
    13:"bearing_Mot.SV_masked",14:"cavitation_Pres.SV_masked",
    15:"seal_Pres.SV_drifting",16:"overloading_Temp.SV_stuck",
    17:"imbalance_Pmp.SV_flatline",18:"cavitation_intermittent",
    19:"seal_failure_fast",20:"overloading_cyclic",21:"bearing_wear_gradual",
    22:"multi_sensor_vibration",23:"pressure_temp_common_cause",
}

def log(m): print(m, flush=True)

def post(win, cluster):
    r = requests.post(URL, json={"window": win.tolist(),
                                  "pump_id": "PUMP-0032", "cluster": cluster})
    return r.json()

# -- load pkl --------------------------------------------------------------
path = None
for d in SEARCH:
    for c in list(d.rglob(PKL)) + ([d / PKL] if (d / PKL).exists() else []):
        path = c; break
    if path: break
if not path:
    log(f"X {PKL} not found"); sys.exit(1)
log(f"Found pkl: {path}\n")
with open(path, "rb") as f:
    obj = pickle.load(f)
seqs, meta = obj["sequences"], obj["metadata"]
y = np.array([int(m["label"]) for m in meta])

def pick_matured(label):
    """Highest-severity sequence for this label; return its matured (late) window + cluster."""
    idx = [i for i in range(len(seqs)) if y[i] == label]
    if not idx:
        return None, None
    idx.sort(key=lambda i: meta[i].get("severity", 0), reverse=True)
    i = idx[0]
    s = np.asarray(seqs[i], dtype=np.float32)
    if s.shape[0] < 50:
        return None, None
    cl = CLUSTER_NAMES.get(int(meta[i].get("cluster_id", 1)), "steady_state")
    return s[-50:, :], cl   # last 50 steps = most matured fault state

# -- run 24-label sweep ----------------------------------------------------
log(f"{'lbl':>3} {'true_name':>26} {'pred':>3} {'pred_name':>26} {'conf%':>7} {'alert':>7}  hit")
log("-" * 92)
hits = 0; total = 0; confusion = []
for lbl in range(24):
    win, cl = pick_matured(lbl)
    if win is None:
        log(f"{lbl:>3} {LABEL_NAMES.get(lbl,'?'):>26}  (no sequence in pkl)")
        continue
    d = post(win, cl)
    pred = int(d.get("fault_label_int", -1))
    pname = d.get("fault_label", "?")
    conf = float(d.get("confidence_pct", 0))
    alert = d.get("alert_state", "?")
    hit = (pred == lbl)
    hits += int(hit); total += 1
    confusion.append((lbl, pred))
    mark = "OK" if hit else "XX"
    log(f"{lbl:>3} {LABEL_NAMES.get(lbl,'?'):>26} {pred:>3} {pname:>26} {conf:>7.2f} {alert:>7}  {mark}")

log("-" * 92)
log(f"\n  Exact-label accuracy: {hits}/{total} = {100.0*hits/max(1,total):.1f}%")
# how many collapsed to label 0 (the failure signature we suspect)
collapse0 = sum(1 for t, p in confusion if t != 0 and p == 0)
log(f"  Non-normal faults predicted as label 0 (normal): {collapse0}/{total-1 if total else 0}")
log("\n  READING:")
log("   - High accuracy (>=18/24) -> M7 classifies correctly on matured windows;")
log("     the dashboard 'normal' is a STREAMING/detection-window issue, not M7.")
log("   - Many faults -> label 0 -> serve-time feature pipeline (D1) is the real")
log("     defect: M7 is being fed features that collapse it to normal.")
log("   - Mixed -> note WHICH labels collapse; that points to the channel/feature.")
