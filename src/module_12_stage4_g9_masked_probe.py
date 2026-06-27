# =============================================================================
# src/module_12_stage4_g9_masked_probe.py
# PumpSmart v14.2 — M12 Stage 4 : G9 masked-fault DETECTION diagnostic probe
#
# WHY: v3.1 (reading raw_alert_state) restored G2/G3 but G9_masked_detect
# dropped to 0.0 for all of labels 13–17. Two competing hypotheses:
#   H1  The OLD 21/24 G9 PASS was INFLATED by the operator-UX latch holding a
#       transient detection across the sequence. raw_alert_state (instantaneous)
#       is the honest read, and masked-fault instantaneous detection is simply
#       weak — the latch was hiding a real architectural limitation.
#   H2  The per-sequence test_reset_latch (or some other v3.1 change) is actively
#       suppressing a detection that WOULD otherwise fire instantaneously.
#
# This probe runs the 5 masked-in-envelope sequences EXACTLY as the runner does
# (same warmup, same window stride) and dumps, per window:
#     raw_alert_state | alert_state(latched) | score_A | theta_t | cusum_Sn |
#     m7_label | is_latched
# plus a per-sequence summary: did raw EVER fire? did latched EVER fire?
#
# It runs three conditions back-to-back so we can compare directly:
#   COND A: with per-sequence test_reset_latch  (v3.1 behaviour)
#   COND B: WITHOUT per-sequence reset (latch carries — old behaviour)
#   COND C: raw vs latched gap quantified
#
# Stdlib + requests only. Needs uvicorn running + M12 manifest present.
# Run:  python src/module_12_stage4_g9_masked_probe.py
# =============================================================================
import sys, json, time
from pathlib import Path
from datetime import datetime, timezone
import random
import numpy as np
import pandas as pd
import requests

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent]:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p)); break
from config import SYNTH_DIR, OUTPUT_DIR

SCRIPT_NAME = "module_12_stage4_g9_masked_probe"
REPORT_DIR  = OUTPUT_DIR / "reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)
M12_DIR     = SYNTH_DIR / "M12_adversarial"
BASE_URL    = "http://localhost:8000"

WINDOW_SIZE = 50
WARMUP_WINDOWS = 432
WARMUP_NOISE_SIGMA = 0.045
CLUSTER_BASELINES = {
    "steady_state": [0.45, 0.42, 0.50, 0.95, 0.55, 0.80, 0.52, 0.90],
}
MASKED_GROUP = "G6_masked_in_envelope"
MASKED_LABELS = [13, 14, 15, 16, 17]

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

S = requests.Session()
def post(path, body, t=30):
    r = S.post(BASE_URL + path, json=body, timeout=t); r.raise_for_status(); return r.json()
def get(path, t=10):
    r = S.get(BASE_URL + path, timeout=t); r.raise_for_status(); return r.json()

def warmup(n=WARMUP_WINDOWS):
    base = CLUSTER_BASELINES["steady_state"]; rng = random.Random(99)
    for _ in range(n):
        win = [[max(0.0,min(3.0,b+rng.gauss(0,WARMUP_NOISE_SIGMA))) for b in base]
               for _ in range(WINDOW_SIZE)]
        try: post("/api/anomaly_detect", {"window":win,"pump_id":"PUMP-0032","cluster":"steady_state"})
        except Exception: pass

def reset_group():
    try: post("/api/acknowledge", {"pump_id":"PUMP-0032","action_taken":"probe_group",
              "operator_id":"probe","reset_zt":False})
    except Exception: pass

def reset_latch_only():
    try: post("/api/test_reset_latch", {})
    except Exception: pass

def iter_windows(seq):
    n = seq.shape[0]
    for s in range(0, n-WINDOW_SIZE+1, WINDOW_SIZE):
        yield np.nan_to_num(seq[s:s+WINDOW_SIZE], nan=0.0).tolist()

def load_masked():
    mp = M12_DIR / "M12_manifest.csv"
    man = pd.read_csv(mp)
    sub = man[man["group"] == MASKED_GROUP]
    seqs = []
    for _, row in sub.iterrows():
        fp = M12_DIR / row["filename"]
        if not fp.exists(): continue
        seq = np.load(fp, allow_pickle=True)["window"]
        seqs.append({"seq":seq, "label":int(row["label"]), "fname":row["filename"]})
    return seqs

def run_sequence(seq, cluster="steady_state"):
    """Return per-window rows + summary for ONE sequence."""
    rows = []
    raw_fired = False; latched_fired = False
    for win in iter_windows(seq):
        try: pred = post("/api/anomaly_detect", {"window":win,"pump_id":"PUMP-0032","cluster":cluster})
        except Exception as e:
            rows.append({"err":str(e)}); continue
        raw = pred.get("raw_alert_state") or "NONE"
        lat = pred.get("alert_state") or "NONE"
        rows.append({
            "raw_alert_state": raw,
            "latched_alert_state": lat,
            "score_A": round(float(pred.get("score_A",0) or 0),4),
            "cusum_Sn": round(float(pred.get("cusum_Sn",0) or 0),4),
            "m7_label": pred.get("fault_label_int"),
            "is_latched": pred.get("is_latched"),
        })
        if raw in ("WATCH","WARN","DANGER"): raw_fired = True
        if lat in ("WATCH","WARN","DANGER"): latched_fired = True
    return rows, raw_fired, latched_fired

def main():
    log("="*70)
    log("  G9 MASKED-FAULT DETECTION PROBE")
    log(f"  Server: {BASE_URL}")
    log("="*70)
    try:
        h = get("/health"); log(f"  Server healthy, arch={h.get('arch_version')}")
    except Exception as e:
        log(f"  FAIL: server unreachable: {e}"); sys.exit(1)

    masked = load_masked()
    log(f"  Loaded {len(masked)} masked sequences (labels {sorted(set(m['label'] for m in masked))})")

    out = {"conditions":{}}

    # ── COND A: v3.1 behaviour — warmup once, per-sequence latch-only reset ──
    log("\n=== COND A: v3.1 (per-sequence test_reset_latch) ===")
    reset_group(); warmup()
    condA = []
    for m in masked:
        reset_latch_only()
        rows, raw_f, lat_f = run_sequence(m["seq"])
        condA.append({"label":m["label"], "raw_fired":raw_f, "latched_fired":lat_f,
                      "n_windows":len(rows),
                      "raw_states":sorted(set(r.get("raw_alert_state","?") for r in rows)),
                      "latched_states":sorted(set(r.get("latched_alert_state","?") for r in rows)),
                      "score_A_max":max((r.get("score_A",0) for r in rows), default=0),
                      "cusum_max":max((r.get("cusum_Sn",0) for r in rows), default=0)})
        log(f"  L{m['label']}: raw_fired={raw_f} latched_fired={lat_f} "
            f"raw_states={condA[-1]['raw_states']} sA_max={condA[-1]['score_A_max']}")
    out["conditions"]["A_v31_per_seq_reset"] = condA

    # ── COND B: old behaviour — warmup once, NO per-sequence reset (latch carries) ──
    log("\n=== COND B: NO per-sequence reset (old latch-carry behaviour) ===")
    reset_group(); warmup()
    condB = []
    for m in masked:
        rows, raw_f, lat_f = run_sequence(m["seq"])
        condB.append({"label":m["label"], "raw_fired":raw_f, "latched_fired":lat_f,
                      "n_windows":len(rows),
                      "raw_states":sorted(set(r.get("raw_alert_state","?") for r in rows)),
                      "latched_states":sorted(set(r.get("latched_alert_state","?") for r in rows)),
                      "score_A_max":max((r.get("score_A",0) for r in rows), default=0),
                      "cusum_max":max((r.get("cusum_Sn",0) for r in rows), default=0)})
        log(f"  L{m['label']}: raw_fired={raw_f} latched_fired={lat_f} "
            f"raw_states={condB[-1]['raw_states']} latched_states={condB[-1]['latched_states']}")
    out["conditions"]["B_no_per_seq_reset"] = condB

    # ── Full per-window dump for ONE sequence per label (COND A) for forensics ─
    log("\n=== Per-window dump (1 seq/label, COND A) ===")
    reset_group(); warmup()
    dumps = {}
    seen = set()
    for m in masked:
        if m["label"] in seen: continue
        seen.add(m["label"])
        reset_latch_only()
        rows, raw_f, lat_f = run_sequence(m["seq"])
        dumps[m["label"]] = rows
        # show first 6 windows compactly
        log(f"\n  --- L{m['label']} ({m['fname']}) raw_fired={raw_f} ---")
        for i, r in enumerate(rows[:8]):
            log(f"    win{i:02d}: raw={r.get('raw_alert_state'):6s} "
                f"latched={r.get('latched_alert_state'):6s} "
                f"sA={r.get('score_A'):.4f} cusum={r.get('cusum_Sn'):.4f} "
                f"m7={r.get('m7_label')}")
    out["per_window_dumps"] = dumps

    # ── VERDICT ──────────────────────────────────────────────────────────────
    log("\n" + "="*70)
    log("VERDICT")
    log("="*70)
    a_raw = sum(1 for x in condA if x["raw_fired"])
    a_lat = sum(1 for x in condA if x["latched_fired"])
    b_raw = sum(1 for x in condB if x["raw_fired"])
    b_lat = sum(1 for x in condB if x["latched_fired"])
    log(f"  COND A (v3.1, per-seq reset):  raw_fired {a_raw}/{len(condA)}  latched_fired {a_lat}/{len(condA)}")
    log(f"  COND B (no per-seq reset):     raw_fired {b_raw}/{len(condB)}  latched_fired {b_lat}/{len(condB)}")
    log("")
    if a_raw == 0 and b_raw == 0:
        verdict = ("H1 CONFIRMED: masked faults NEVER trip the instantaneous detector "
                   "(raw_alert_state) regardless of reset. The old 21/24 G9 PASS was "
                   "INFLATED by the latch holding a transient/cross-sequence state. "
                   "raw_alert_state is the honest read. G9 masked-detect is a REAL "
                   "architectural weakness, not a v3.1 regression. Options: (a) accept "
                   "honest 19/24 and document; (b) the masked-fault DETECTION path needs "
                   "a genuine fix (e.g. CUSUM/drift sensitivity for frozen-channel "
                   "signatures), which is a model/threshold change, not a harness change.")
    elif a_raw < b_raw:
        verdict = ("H2 PARTIAL: per-sequence reset is SUPPRESSING masked detection "
                   "(COND B raw fires more than COND A). The test_reset_latch call is "
                   "interfering. FIX: remove per-sequence reset for masked groups, or "
                   "make detection rely on cross-sequence detector state as the baseline did.")
    elif a_raw > 0:
        verdict = ("raw_alert_state DOES fire for masked faults under v3.1. If the gate "
                   "still scored 0.0, the runner's has_alert / group filter has a bug — "
                   "compare alert_states_unique construction. Investigate runner scoring.")
    else:
        verdict = "Mixed — inspect per-window dumps."
    log("  " + verdict)
    out["verdict"] = verdict

    rp = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    rp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log(f"\n  Saved: {rp}")
    log("  Paste the VERDICT + COND A/B lines back for the fix decision.")

if __name__ == "__main__":
    main()
