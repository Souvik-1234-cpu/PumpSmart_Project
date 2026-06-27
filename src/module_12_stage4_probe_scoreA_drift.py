# =============================================================================
# src/module_12_stage4_probe_scoreA_drift.py
# PumpSmart v14.2 — M12 Stage 4 : does score_A (LEVEL, not delta) separate
# label-21 gradual wear from normal? Decides whether CUSUM can be re-wired onto
# a score_A-derived drift signal (Invariant 19-preserving: L3 = slow-drift).
#
# score_B proved unusable (probe: normal p99=0.0285 vs label21 p99=0.0260,
# overlapping). Label-21 is gradual wear → should manifest as a slow UPWARD
# creep in reconstruction error LEVEL (score_A), integrated over the full
# sequence — exactly what CUSUM is built for. This probe MEASURES that creep.
#
# RUN: python src/module_12_stage4_probe_scoreA_drift.py
# =============================================================================
import sys, json, time
from pathlib import Path
from datetime import datetime, timezone
import random
import numpy as np
import requests

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent]:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p)); break
from config import SYNTH_DIR, OUTPUT_DIR

BASE_URL = "http://localhost:8000"
WINDOW_SIZE = 50
WARMUP_WINDOWS = 100
WARMUP_SIGMA = 0.045
M12_DIR = SYNTH_DIR / "M12_adversarial"
REPORT_DIR = OUTPUT_DIR / "reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT = "module_12_stage4_probe_scoreA_drift"
STEADY = [0.45, 0.42, 0.50, 0.95, 0.55, 0.80, 0.52, 0.90]

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

class C:
    def __init__(s,u): s.u=u.rstrip("/"); s.s=requests.Session()
    def reset(s,reset_zt=True):
        s.s.post(f"{s.u}/api/acknowledge",json={"pump_id":"PUMP-0032",
            "action_taken":"probe","operator_id":"probe",
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "reset_zt":reset_zt},timeout=30).raise_for_status()
    def detect(s,win,cl="steady_state"):
        r=s.s.post(f"{s.u}/api/anomaly_detect",json={"window":win,
            "pump_id":"PUMP-0032","cluster":cl,
            "timestamp_utc":datetime.now(timezone.utc).isoformat()},timeout=30)
        r.raise_for_status(); return r.json()

def warm(c,n=WARMUP_WINDOWS,seed=99):
    rng=random.Random(seed)
    for _ in range(n):
        w=[[max(0.0,min(3.0,b+rng.gauss(0,WARMUP_SIGMA))) for b in STEADY]
           for _ in range(WINDOW_SIZE)]
        try: c.detect(w)
        except: pass

def iw(seq):
    n=seq.shape[0]
    for s in range(0,n-WINDOW_SIZE+1,WINDOW_SIZE):
        yield np.nan_to_num(seq[s:s+WINDOW_SIZE],nan=0.0).tolist()

def slope(y):
    n=len(y)
    if n<5: return 0.0
    x=np.arange(n); mx=x.mean(); my=np.mean(y)
    d=((x-mx)**2).sum() or 1e-9
    return float(((x-mx)*(y-my)).sum()/d)

def run_pool(c,seqs,name):
    """Per-sequence: warm fresh, stream, record score_A trajectory + its slope."""
    per_seq=[]
    for e in seqs:
        c.reset(reset_zt=True); warm(c)
        sA=[]
        for win in iw(e["seq"]):
            try: p=c.detect(win)
            except Exception as ex: log(f" err {ex}"); continue
            sA.append(float(p.get("score_A",0.0) or 0.0))
        if sA:
            per_seq.append({"sA_mean":float(np.mean(sA)),"sA_max":float(np.max(sA)),
                            "sA_last":sA[-1],"sA_slope":slope(np.array(sA)),
                            "n_win":len(sA)})
    return per_seq

def agg(ps,key):
    v=[r[key] for r in ps]
    if not v: return {}
    a=np.array(v)
    return {"mean":round(float(a.mean()),6),"p50":round(float(np.percentile(a,50)),6),
            "p95":round(float(np.percentile(a,95)),6),"max":round(float(a.max()),6),
            "min":round(float(a.min()),6)}

def main():
    log("="*64); log("score_A LEVEL/SLOPE separation probe (label-21 vs normal)"); log("="*64)
    c=C(BASE_URL)
    try: c.s.get(f"{BASE_URL}/health",timeout=10).raise_for_status(); log("server: healthy")
    except Exception as e: log(f"unreachable: {e}"); sys.exit(1)
    man=__import__("pandas").read_csv(M12_DIR/"M12_manifest.csv")
    def load(sub,cap):
        out=[]
        for _,r in man[man["group"].str.contains(sub)].head(cap).iterrows():
            fp=M12_DIR/r["filename"]
            if fp.exists():
                try: out.append({"seq":np.load(fp,allow_pickle=True)["window"]})
                except: pass
        return out
    nor=load("G1_normal",15) or load("normal",15)
    l21=load("G4_label21",5) or load("label21",5) or load("groupE",5)
    log(f"normal={len(nor)} label21={len(l21)}")

    log("running normal pool..."); nps=run_pool(c,nor,"normal")
    log("running label21 pool..."); lps=run_pool(c,l21,"label21")

    out={"normal":{k:agg(nps,k) for k in ["sA_mean","sA_max","sA_last","sA_slope"]},
         "label21":{k:agg(lps,k) for k in ["sA_mean","sA_max","sA_last","sA_slope"]}}
    with open(REPORT_DIR/f"{SCRIPT}_results.json","w",encoding="utf-8") as f:
        json.dump(out,f,indent=2,default=str)

    print("\n"+"="*60); print("══ PASTE THIS BACK ══")
    for metric in ["sA_mean","sA_max","sA_last","sA_slope"]:
        print(f"{metric:10s} normal={out['normal'][metric]}")
        print(f"{'':10s} label21={out['label21'][metric]}")
    # verdict
    n_lvl=out["normal"]["sA_last"].get("p95",0); l_lvl=out["label21"]["sA_last"].get("p50",0)
    n_slp=out["normal"]["sA_slope"].get("p95",0); l_slp=out["label21"]["sA_slope"].get("p50",0)
    print(f"\nLEVEL separation (label21 p50_last > normal p95_last?): {l_lvl} vs {n_lvl} -> {l_lvl>n_lvl}")
    print(f"SLOPE separation (label21 p50_slope > normal p95_slope?): {l_slp} vs {n_slp} -> {l_slp>n_slp}")
    print("If EITHER True -> CUSUM can re-wire onto score_A drift (option 1 works).")
    print("If BOTH False -> label-21 not live-separable; re-scope gate honestly.")
    print("══ END ══"); print("="*60)

if __name__=="__main__": main()
