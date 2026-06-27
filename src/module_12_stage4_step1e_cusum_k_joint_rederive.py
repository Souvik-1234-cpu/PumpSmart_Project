# =============================================================================
# src/module_12_stage4_step1e_cusum_k_joint_rederive.py
# PumpSmart v14.2 — M12 Stage 4, Step 1e: joint CUSUM k re-derivation (G4a+G4b)
# =============================================================================
#
# WHY
# ---
# Step 4.1c derived k=0.230 to keep normal S_n < 2.0. After Option A relaxed G4b
# to "normal S_n < H=5.0" (no false ALARM), k=0.230 is OVER-tuned: smoke showed
# G4a FAIL with cusum_max_mean=0.0 — the high k suppressed label-21 entirely.
#
# With G4b at H=5.0 there is headroom for a LOWER k that satisfies BOTH:
#   G4a: label-21 slow drift crosses H (5.0) within the spec window in >=75%
#   G4b: normal S_n max stays < H (5.0) (no false ALARM)
#
# This step measures normal + label-21 score_B streams (serve-faithful, via the
# production TCN-AE), then SEARCHES k over a grid for the highest k that still
# satisfies G4a, and verifies G4b holds at that k. Highest-feasible-k is chosen
# (maximises normal specificity margin while preserving detection). If NO k
# satisfies both, that is surfaced (score_B cannot separate) — not masked.
#
# Persists the chosen k -> models/M8_threshold_config.json (main.py reads it),
# with .bak. mu0_B + quiet-decay semantics unchanged (Step 1b fix stands).
#
# RUN:  python src/module_12_stage4_step1e_cusum_k_joint_rederive.py
# =============================================================================

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import DEVICE, MODEL_DIR, SYNTH_DIR, OUTPUT_DIR
from datetime import date, datetime
import json, pickle, shutil, warnings, traceback
from collections import deque
warnings.filterwarnings("ignore")
import numpy as np
import torch

SCRIPT_NAME = "module_12_stage4_step1e_cusum_k_joint_rederive"
REPORT_DIR = OUTPUT_DIR / "reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)
PASS, FAIL = "PASS", "FAIL"


def _abspath(p, d):
    p = Path(p)
    if not p.is_absolute(): p = _ROOT / p
    if not p.exists():
        alt = _ROOT / d
        if alt.exists(): return alt
    return p

M4_PATH     = _abspath(MODEL_DIR / "lstm_ae_baseline_final.pth", "models/lstm_ae_baseline_final.pth")
M8_PATH     = _abspath(MODEL_DIR / "tcn_ae_level2_best.pth", "models/tcn_ae_level2_best.pth")
M8_CFG_PATH = _abspath(MODEL_DIR / "M8_threshold_config.json", "models/M8_threshold_config.json")
POLICY_PATH = _abspath(MODEL_DIR / "M8p5_cusum_runtime_policy.json", "models/M8p5_cusum_runtime_policy.json")
M6B_PATH    = _abspath(SYNTH_DIR / "M6B_combined_sequences.pkl", "data/synthetic/M6B_combined_sequences.pkl")

WINDOW_SIZE, ZT_BUF_LEN = 50, 63
H_ALARM = 5.0
CH_W = torch.tensor([2.5, 2.5, 0.3, 2.0, 0.5, 2.5, 0.3, 2.0], dtype=torch.float32)
SEED = 415
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


results = {"script": SCRIPT_NAME, "timestamp": datetime.now().isoformat(),
           "gates": {}, "evidence": {}, "derived": {}, "overall_status": "UNKNOWN"}
G = results["gates"]


def load_m6b():
    with open(M6B_PATH, "rb") as f: d = pickle.load(f)
    return d["sequences"], d.get("metadata", d.get("meta"))


def label_of(m):
    if isinstance(m, dict):
        for k in ("label", "label_int", "fault_label", "class"):
            if k in m: return int(m[k])
    return -1


def load_models():
    from app.runtime.model_registry import _M4LSTMAutoencoder, _load_tcn_ae
    m4 = _M4LSTMAutoencoder()
    m4.load_state_dict(torch.load(M4_PATH, map_location="cpu", weights_only=True), strict=True)
    m4.eval()
    for p in m4.parameters(): p.requires_grad_(False)
    with open(M8_CFG_PATH, encoding="utf-8") as f: cfg = json.load(f)
    m8 = _load_tcn_ae(M8_PATH, cfg)
    return m4.to(DEVICE), (m8.to(DEVICE) if m8 else None)


@torch.no_grad()
def zt_of(win, m4):
    x = torch.from_numpy(win).float().unsqueeze(0).to(DEVICE)
    enc = m4.encoder
    o1, _ = enc.lstm1(x); o2, (h, c) = enc.lstm2(o1); z = enc.bn(h[-1])
    return z.squeeze(0).cpu().numpy()


@torch.no_grad()
def scoreB_of_buf(buf, m8):
    x = torch.from_numpy(np.stack(buf)).float().unsqueeze(0).to(DEVICE)
    _a, b, _c = m8(x); return float(b)


def collect_scoreB(seqs, idxs, m4, m8, cap):
    """Return list of score_B sequences (one per source sequence), serve-faithful
    (adaptive stride to fill the 63 buffer like the live route on long streams)."""
    out = []
    for i in idxs[:cap]:
        arr = np.asarray(seqs[i], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < WINDOW_SIZE: continue
        buf, sbs = deque(maxlen=ZT_BUF_LEN), []
        stride = max(1, (arr.shape[0] - WINDOW_SIZE) // (ZT_BUF_LEN - 1))
        for s in range(0, arr.shape[0] - WINDOW_SIZE + 1, stride):
            buf.append(zt_of(arr[s:s+WINDOW_SIZE], m4))
            if len(buf) >= 5: sbs.append(scoreB_of_buf(list(buf), m8))
        if sbs: out.append(np.array(sbs))
    return out


def main():
    log("=" * 76)
    log(f"  {SCRIPT_NAME}  |  {date.today().isoformat()}")
    log("  Stage 4 Step 1e — joint CUSUM k re-derivation for H=5.0 (G4a+G4b)")
    log("=" * 76)

    if not M6B_PATH.exists():
        log("  SKIP — M6B absent."); results["overall_status"] = "SKIP_NO_M6B"; _finish(); return
    try:
        m4, m8 = load_models()
        if m8 is None: raise RuntimeError("TCN-AE failed to load")
        log("  models loaded")
    except Exception as e:
        log(f"  FATAL: {e}"); log(traceback.format_exc())
        results["overall_status"] = FAIL; _finish(); return

    pol = {"mu0_B": -0.00954, "lambda": 5.7302192538299934e-05}
    if POLICY_PATH.exists():
        with open(POLICY_PATH, encoding="utf-8") as f: j = json.load(f)
        pol["mu0_B"] = float(j["cusum_parameters"]["mu0_B"])
        pol["lambda"] = float(j["decay_policy"]["geometric_quiet_decay"]["lambda"])
    mu0_B, lam, decay = pol["mu0_B"], pol["lambda"], 1.0 - pol["lambda"]

    seqs, meta = load_m6b()
    by = {}
    for i, m in enumerate(meta): by.setdefault(label_of(m), []).append(i)

    log("\nStep 1 — collect normal + label-21 score_B streams")
    normal_streams = collect_scoreB(seqs, by.get(0, []), m4, m8, cap=40)
    lbl21_streams = collect_scoreB(seqs, by.get(21, []), m4, m8, cap=40)
    norm_all = np.concatenate(normal_streams) if normal_streams else np.array([0.0])
    log(f"  normal score_B: n={len(norm_all)} mean={norm_all.mean():.5f} "
        f"P99={np.percentile(norm_all,99):.5f}")
    log(f"  label-21 sequences: {len(lbl21_streams)}")
    results["evidence"]["normal_scoreB_mean"] = round(float(norm_all.mean()), 6)
    results["evidence"]["normal_scoreB_p99"] = round(float(np.percentile(norm_all, 99)), 6)

    def sim_max(stream, k):
        S, mx = 0.0, 0.0
        for sb in stream:
            e = sb - mu0_B - k
            S = (S + e) if e > 0 else S * decay
            if S < 0: S = 0.0
            mx = max(mx, S)
        return mx

    def normal_specificity(k):
        # fraction of normal sequences whose S_n stays < H
        ok = sum(1 for st in normal_streams if sim_max(st, k) < H_ALARM)
        return ok / max(len(normal_streams), 1)

    def label21_sensitivity(k):
        # fraction of label-21 sequences that cross H (repeat to ~1500 win window)
        fired = 0
        for st in lbl21_streams:
            S, crossed = 0.0, False
            for rep in range(20):
                for sb in st:
                    e = sb - mu0_B - k
                    S = (S + e) if e > 0 else S * decay
                    if S < 0: S = 0.0
                    if S >= H_ALARM: crossed = True; break
                if crossed: break
            if crossed: fired += 1
        return fired / max(len(lbl21_streams), 1)

    # ── Step 2 — grid search: highest k with G4a>=0.75 AND G4b>=0.99 ─────────
    log("\nStep 2 — grid search k for joint G4a(>=0.75) + G4b(>=0.99 normal<H)")
    grid = np.round(np.arange(0.00, 0.2601, 0.005), 4)
    feasible = []
    table = []
    for k in grid:
        spec = normal_specificity(k)      # G4b
        sens = label21_sensitivity(k)     # G4a
        table.append((float(k), round(sens, 3), round(spec, 3)))
        if sens >= 0.75 and spec >= 0.99:
            feasible.append(float(k))
    results["evidence"]["k_grid"] = table

    # log a few rows around the action
    log("    k      G4a(sens)  G4b(spec)")
    for k, sens, spec in table:
        if (sens >= 0.5 or spec < 1.0) or abs(k - round(k,2)) < 1e-9:
            log(f"    {k:.3f}    {sens:.3f}      {spec:.3f}")

    if feasible:
        k_star = max(feasible)            # highest feasible k = max specificity margin
        sens = label21_sensitivity(k_star); spec = normal_specificity(k_star)
        norm_max = max((sim_max(st, k_star) for st in normal_streams), default=0.0)
        log(f"\n  FEASIBLE k range: [{min(feasible):.3f}, {max(feasible):.3f}]")
        log(f"  chosen k* = {k_star:.4f}  (G4a sens={sens:.2%}, G4b spec={spec:.2%}, "
            f"normal S_n max={norm_max:.3f} < {H_ALARM})")
        results["derived"]["cusum_k"] = round(k_star, 6)
        results["evidence"]["g4a_sensitivity"] = round(sens, 4)
        results["evidence"]["g4b_specificity"] = round(spec, 4)
        results["evidence"]["normal_Sn_max"] = round(norm_max, 4)
        G["G1e_joint_feasible"] = PASS
    else:
        log("\n  NO k satisfies both G4a>=0.75 and G4b>=0.99.")
        log("  score_B cannot jointly separate label-21 from normal at H=5.0.")
        log("  This is a real finding — surfaced. Fallback: keep CUSUM advisory")
        log("  (G4a reframed advisory like G4b), label-21 primary detection = L1.")
        G["G1e_joint_feasible"] = FAIL
        # pick k that maximises G4a while keeping G4b (best-effort), for the record
        best = max(table, key=lambda r: (r[1], r[2]))
        results["derived"]["cusum_k_best_effort"] = best[0]
        results["evidence"]["best_effort"] = {"k": best[0], "g4a": best[1], "g4b": best[2]}

    # ── Persist chosen k (only if feasible) ──────────────────────────────────
    if G["G1e_joint_feasible"] == PASS:
        try:
            with open(M8_CFG_PATH, encoding="utf-8") as f: mc = json.load(f)
            bak = M8_CFG_PATH.with_suffix(f".json.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(M8_CFG_PATH, bak)
            mc["cusum_k"] = round(results["derived"]["cusum_k"], 6)
            mc["_cusum_k_calibrated_by"] = SCRIPT_NAME
            mc["_cusum_k_note"] = ("joint G4a(>=75%)+G4b(normal<H=5.0) feasible k; "
                                   "highest-feasible for max specificity margin")
            with open(M8_CFG_PATH, "w", encoding="utf-8") as f: json.dump(mc, f, indent=2)
            results["evidence"]["backup"] = str(bak)
            log(f"\n  cusum_k={results['derived']['cusum_k']:.4f} -> {M8_CFG_PATH.name} "
                f"(backup {bak.name})")
        except Exception as e:
            log(f"  WARN persist: {e}")

    results["overall_status"] = PASS if G["G1e_joint_feasible"] == PASS else FAIL
    _finish()


def _finish():
    out = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    with open(out, "w", encoding="utf-8") as f: json.dump(results, f, indent=2, default=str)
    g, d, ev, status = results["gates"], results["derived"], results["evidence"], results["overall_status"]
    print("\n" + "=" * 76)
    print("══ PASTE TEXT UPDATE ══")
    print(f"M12 Stage 4 Step 1e (joint CUSUM k re-derive): {status}")
    print(f"  G1e_joint_feasible: {g.get('G1e_joint_feasible','-')}")
    if d.get("cusum_k") is not None:
        print(f"  chosen cusum_k = {d['cusum_k']}  (G4a sens={ev.get('g4a_sensitivity')}, "
              f"G4b spec={ev.get('g4b_specificity')}, normal S_n max={ev.get('normal_Sn_max')})")
    elif d.get("cusum_k_best_effort") is not None:
        print(f"  NO feasible k. best-effort: {ev.get('best_effort')}")
    if status == PASS:
        print("  Persisted cusum_k -> M8_threshold_config.json.")
        print("  Next: RESTART server, re-run smoke. G1+G4a+G4b should all PASS.")
    else:
        print("  score_B cannot jointly separate — escalate G4a to advisory (reframe).")
    print("══ END PASTE UPDATE ══")
    print(f"  Report: {out}")
    if ev.get("backup"):
        print("  GitHub push: models/M8_threshold_config.json")
    print(f"  GitHub push: src/{SCRIPT_NAME}.py")
    print("=" * 76)


if __name__ == "__main__":
    main()
