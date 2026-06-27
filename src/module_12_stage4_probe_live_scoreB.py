# =============================================================================
# src/module_12_stage4_probe_live_scoreB.py
# PumpSmart v14.2 — M12 Stage 4 : LIVE score_B distribution probe
#
# PURPOSE
# -------
# G4b (CUSUM > H on normals) and G1 (FPR 100%, now CUSUM-driven WATCH) both
# trace to mu0_B being mis-centered for the LIVE warmed score_B. Offline k was
# tuned on an offline score_B whose normal center differs from what the live
# TCN-AE produces (~0.038 on the dashboard). This probe MEASURES the live
# distribution so mu0_B is re-centered on the real number — not an eyeballed one.
#
# WHAT IT DOES (online-faithful — hits the live /api route)
#   1. Warms the buffer (full 63+) with steady-state noise (σ=0.045, matches M4).
#   2. Streams a NORMAL pool and a LABEL-21 pool, recording per-window:
#        score_B, score_B_provisional, cusum_Sn, cusum_mu0_B, cusum_k
#   3. Computes, on NON-PROVISIONAL windows only:
#        - normal score_B: mean, p50, p95, p99, max
#        - label21 score_B: mean, p50, p95
#        - live evidence = score_B - mu0_B - k  (is normal evidence > 0? = G4b cause)
#   4. Recommends a re-centered mu0_B (and k if margin requires) that puts
#      normal evidence <= 0 at p99 while keeping label-21 evidence > 0.
#
# It does NOT write any config. It prints a paste-ready block + writes a JSON.
# You paste the block back; the centering fix is applied in a follow-up edit.
#
# RUN (server must be up with all current patches):
#   .venv\Scripts\Activate.ps1
#   python src/module_12_stage4_probe_live_scoreB.py
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
WARMUP_WINDOWS = 100          # > 63 so buffer is fully warmed before we measure
WARMUP_SIGMA = 0.045
M12_DIR = SYNTH_DIR / "M12_adversarial"
REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT = "module_12_stage4_probe_live_scoreB"

STEADY_BASELINE = [0.45, 0.42, 0.50, 0.95, 0.55, 0.80, 0.52, 0.90]

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

class C:
    def __init__(self, url): self.url=url.rstrip("/"); self.s=requests.Session()
    def reset(self, reset_zt=True):
        # full wipe so each pool starts clean; we re-warm explicitly after
        self.s.post(f"{self.url}/api/acknowledge", json={
            "pump_id":"PUMP-0032","action_taken":"probe_reset","operator_id":"probe",
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),"reset_zt":reset_zt},
            timeout=30).raise_for_status()
    def detect(self, win, cluster="steady_state"):
        r=self.s.post(f"{self.url}/api/anomaly_detect", json={
            "window":win,"pump_id":"PUMP-0032","cluster":cluster,
            "timestamp_utc":datetime.now(timezone.utc).isoformat()}, timeout=30)
        r.raise_for_status(); return r.json()

def warm(c, n=WARMUP_WINDOWS, seed=99):
    rng=random.Random(seed)
    for _ in range(n):
        win=[[max(0.0,min(3.0,b+rng.gauss(0,WARMUP_SIGMA))) for b in STEADY_BASELINE]
             for _ in range(WINDOW_SIZE)]
        try: c.detect(win)
        except Exception: pass

def iter_windows(seq):
    n=seq.shape[0]
    for s in range(0,n-WINDOW_SIZE+1,WINDOW_SIZE):
        yield np.nan_to_num(seq[s:s+WINDOW_SIZE],nan=0.0).tolist()

def stream_pool(c, seqs, label_name):
    """Warm once, then stream all sequences keeping the buffer warm between them
    (reset_zt=False) — matches the runner's online-faithful behavior."""
    log(f"  warming buffer for {label_name} ...")
    c.reset(reset_zt=True); warm(c)
    rows=[]
    for entry in seqs:
        for win in iter_windows(entry["seq"]):
            try: p=c.detect(win, cluster="steady_state")
            except Exception as e: log(f"   win err {e}"); continue
            rows.append({
                "score_B": float(p.get("score_B",0.0) or 0.0),
                "provisional": bool(p.get("score_B_provisional", False)),
                "cusum_Sn": float(p.get("cusum_Sn",0.0) or 0.0),
                "mu0_B": p.get("cusum_mu0_B"),
                "k": p.get("cusum_k"),
            })
        c.reset(reset_zt=False)   # scenario boundary: keep warmed buffer
    return rows

def stats(vals):
    if not vals: return {"n":0}
    a=np.array(vals,dtype=float)
    return {"n":len(a),"mean":round(float(a.mean()),6),"p50":round(float(np.percentile(a,50)),6),
            "p95":round(float(np.percentile(a,95)),6),"p99":round(float(np.percentile(a,99)),6),
            "max":round(float(a.max()),6)}

def main():
    log("="*64); log("LIVE score_B probe — re-centering mu0_B for G4b/G1"); log("="*64)
    c=C(BASE_URL)
    try: h=c.s.get(f"{BASE_URL}/health",timeout=10).json(); log(f"server: {h.get('status')}")
    except Exception as e: log(f"server unreachable: {e}"); sys.exit(1)

    man=__import__("pandas").read_csv(M12_DIR/"M12_manifest.csv")
    def load(group_substr, cap):
        rows=man[man["group"].str.contains(group_substr)].head(cap)
        out=[]
        for _,r in rows.iterrows():
            fp=M12_DIR/r["filename"]
            if fp.exists():
                try: out.append({"seq":np.load(fp,allow_pickle=True)["window"]})
                except Exception: pass
        return out

    normal_seqs=load("G1_normal", 30) or load("normal", 30)
    label21_seqs=load("G4_label21", 5) or load("label21", 5) or load("groupE", 5)
    log(f"normal seqs={len(normal_seqs)}  label21 seqs={len(label21_seqs)}")

    nrows=stream_pool(c, normal_seqs, "normal")
    lrows=stream_pool(c, label21_seqs, "label21")

    # warmed (non-provisional) only
    n_warm=[r["score_B"] for r in nrows if not r["provisional"]]
    l_warm=[r["score_B"] for r in lrows if not r["provisional"]]
    n_all=[r["score_B"] for r in nrows]
    mu0=next((r["mu0_B"] for r in nrows if r["mu0_B"] is not None), None)
    k  =next((r["k"]     for r in nrows if r["k"]     is not None), None)

    ns=stats(n_warm); ls=stats(l_warm)
    log(f"normal warmed score_B: {ns}")
    log(f"label21 warmed score_B: {ls}")
    log(f"provisional fraction normal: {1-len(n_warm)/max(len(n_all),1):.2f}")

    # live evidence on normal: is it >0? (that is the G4b cause)
    rec={}
    if mu0 is not None and k is not None and ns.get("n"):
        ev_mean = ns["mean"] - mu0 - k
        ev_p99  = ns["p99"]  - mu0 - k
        rec["current_mu0_B"]=mu0; rec["current_k"]=k
        rec["normal_evidence_mean"]=round(ev_mean,6)
        rec["normal_evidence_p99"]=round(ev_p99,6)
        # Re-center: put normal p99 evidence at ~0 (no accumulation on normal),
        # keep k as the M8p5 reference noise band. mu0_B* = normal_p99 - k
        mu0_star = round(ns["p99"] - k, 6)
        rec["recommended_mu0_B"]=mu0_star
        # verify label-21 still fires: evidence at label21 mean with new mu0
        if ls.get("n"):
            l_ev = round(ls["mean"] - mu0_star - k, 6)
            rec["label21_evidence_at_new_mu0"]=l_ev
            rec["label21_still_accumulates"]=bool(l_ev>0)
        rec["normal_evidence_p99_at_new_mu0"]=round(ns["p99"]-mu0_star-k,6)

    out={"script":SCRIPT,"ts":datetime.now(timezone.utc).isoformat(),
         "normal_scoreB":ns,"label21_scoreB":ls,
         "provisional_frac_normal":round(1-len(n_warm)/max(len(n_all),1),3),
         "recenter":rec}
    with open(REPORT_DIR/f"{SCRIPT}_results.json","w",encoding="utf-8") as f:
        json.dump(out,f,indent=2,default=str)

    print("\n"+"="*60)
    print("══ PASTE THIS BACK ══")
    print(f"normal_scoreB        : {ns}")
    print(f"label21_scoreB       : {ls}")
    print(f"current mu0_B / k     : {mu0} / {k}")
    print(f"normal_evidence mean/p99 (current): "
          f"{rec.get('normal_evidence_mean')} / {rec.get('normal_evidence_p99')}")
    print(f"  -> if p99 > 0, that is why S_n climbs on normals (G4b)")
    print(f"recommended_mu0_B     : {rec.get('recommended_mu0_B')}")
    print(f"label21_evidence@new  : {rec.get('label21_evidence_at_new_mu0')} "
          f"(accumulates={rec.get('label21_still_accumulates')})")
    print(f"normal_evidence_p99@new: {rec.get('normal_evidence_p99_at_new_mu0')} "
          f"(should be ~0 or slightly <0)")
    print("══ END ══"); print("="*60)

if __name__=="__main__":
    main()
