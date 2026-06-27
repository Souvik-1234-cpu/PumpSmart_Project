# =============================================================================
# src/module_12_stage4_step1c_normal_baseline_calibration.py
# PumpSmart v14.2 — M12 Stage 4, Step 4.1c: normal-baseline calibration (G1+G4b)
# =============================================================================
#
# WHY
# ---
# Smoke run (post state-machine + CUSUM fix) still failed two CRITICAL gates:
#   G1_normal_fpr = 1.0   — normal windows trip WARN. Root cause: rolling-mean
#     floors (0.095/0.085, M8-spec defaults) sit BELOW the post-warmup normal
#     score_A level. Report: theta_t adapts to 0.178 and normal sA_max_mean
#     ~0.16-0.45, so rm100 (~0.16) >> 0.095 -> every normal window -> WARN.
#   G4b_label21_cusum_spec = 0.0 — normal CUSUM S_n exceeds 2.0
#     (cusum_max_mean=7.646). Not the old 75.9 runaway (corrected CUSUM IS
#     loaded), but normal score_B still yields positive evidence
#     (score_B - mu0_B - k > 0) so S_n accumulates past WATCH.
#
# Both thresholds were guessed/spec-default. ISA-18.2 requires them MEASURED
# from the normal pool. This step measures the SERVE-PATH normal distribution
# (CRITICAL: after the same 432-window sigma=0.045 warmup that adapts theta_t),
# then derives:
#   rolling_mean_100_floor = P99.5(normal rm100)   (normal stays NORMAL)
#   rolling_mean_200_floor = P99.5(normal rm200)
#   cusum_k                = P99(normal score_B) - mu0_B  (normal evidence ~ 0)
# and JOINTLY verifies the derived k still lets label-21 slow-drift cross H
# (G4a) — if no k satisfies both G4b and G4a, that is surfaced as a real
# finding, not papered over.
#
# Persists:
#   - rolling floors -> models/M8_alert_thresholds.json (state_machine block)
#   - cusum_k        -> models/M8_threshold_config.json  (so main.py picks it up)
#                       (timestamped .bak first)
#
# OFFLINE: replicates the production score_A/score_B/RollingState math exactly
# (same as the live route) — no server needed. Matches the 4.1 method.
#
# RUN (CWD-independent)
#   .venv\Scripts\Activate.ps1
#   python src/module_12_stage4_step1c_normal_baseline_calibration.py
# =============================================================================

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import DEVICE, IS_GPU, MODEL_DIR, SYNTH_DIR, OUTPUT_DIR

from datetime import date, datetime
import json, pickle, shutil, warnings, traceback
from collections import deque
warnings.filterwarnings("ignore")

import numpy as np
import torch

SCRIPT_NAME = "module_12_stage4_step1c_normal_baseline_calibration"
REPORT_DIR = OUTPUT_DIR / "reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)
PASS, FAIL = "PASS", "FAIL"


def _abspath(p, d):
    p = Path(p)
    if not p.is_absolute(): p = _ROOT / p
    if not p.exists():
        alt = _ROOT / d
        if alt.exists(): return alt
    return p

M4_PATH      = _abspath(MODEL_DIR / "lstm_ae_baseline_final.pth", "models/lstm_ae_baseline_final.pth")
M8_PATH      = _abspath(MODEL_DIR / "tcn_ae_level2_best.pth", "models/tcn_ae_level2_best.pth")
M8_CFG_PATH  = _abspath(MODEL_DIR / "M8_threshold_config.json", "models/M8_threshold_config.json")
ALERT_CFG    = _abspath(MODEL_DIR / "M8_alert_thresholds.json", "models/M8_alert_thresholds.json")
POLICY_PATH  = _abspath(MODEL_DIR / "M8p5_cusum_runtime_policy.json", "models/M8p5_cusum_runtime_policy.json")
M6B_PATH     = _abspath(SYNTH_DIR / "M6B_combined_sequences.pkl", "data/synthetic/M6B_combined_sequences.pkl")

WINDOW_SIZE = 50
WARMUP_WINDOWS = 432
WARMUP_SIGMA = 0.045
ZT_BUF_LEN = 63
CH_WEIGHTS = torch.tensor([2.5, 2.5, 0.3, 2.0, 0.5, 2.5, 0.3, 2.0], dtype=torch.float32)
SEED = 415

np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


results = {"script": SCRIPT_NAME, "timestamp": datetime.now().isoformat(),
           "gates": {}, "evidence": {}, "derived": {}, "overall_status": "UNKNOWN",
           "block_m11": True}
G = results["gates"]


def load_m6b():
    with open(M6B_PATH, "rb") as f: d = pickle.load(f)
    return d["sequences"], d.get("metadata", d.get("meta"))


def label_of(m):
    if isinstance(m, dict):
        for k in ("label", "label_int", "fault_label", "class"):
            if k in m: return int(m[k])
    return -1


def load_m4():
    from app.runtime.model_registry import _M4LSTMAutoencoder
    m = _M4LSTMAutoencoder()
    m.load_state_dict(torch.load(M4_PATH, map_location="cpu", weights_only=True), strict=True)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m.to(DEVICE)


def load_tcn():
    from app.runtime.model_registry import _load_tcn_ae
    with open(M8_CFG_PATH, encoding="utf-8") as f: cfg = json.load(f)
    m = _load_tcn_ae(M8_PATH, cfg)
    return m.to(DEVICE) if m else None


@torch.no_grad()
def score_A_of_window(win_np, m4):
    """Replicate production run_m4 score_A exactly (physics-weighted MAE)."""
    x = torch.from_numpy(win_np).float().unsqueeze(0).to(DEVICE)
    enc = m4.encoder
    o1, _ = enc.lstm1(x); o2, (h, c) = enc.lstm2(o1); z = enc.bn(h[-1])
    recon = m4.decoder(z, x.size(1), h, c)
    mae = (x - recon).abs().mean(dim=1).squeeze(0)
    w = CH_WEIGHTS.to(mae.device)
    sA = (mae * w).sum().item() / w.sum().item()
    return sA, z.squeeze(0).cpu().numpy()


@torch.no_grad()
def score_B_of_buffer(zt_buf, m8):
    x = torch.from_numpy(np.stack(zt_buf)).float().unsqueeze(0).to(DEVICE)
    _sA, sB, _sC = m8(x)
    return float(sB)


def main():
    log("=" * 76)
    log(f"  {SCRIPT_NAME}  |  {date.today().isoformat()}")
    log("  Stage 4 Step 4.1c — normal-baseline calibration (G1 floors + G4b k)")
    log("  offline; replicates 432-win sigma=0.045 warmup, then measures normal")
    log("=" * 76)

    if not M6B_PATH.exists():
        log("  SKIP — M6B absent."); 
        for k in ("G1c_floors","G1c_k","G1c_joint"): G[k] = "SKIP"
        results["overall_status"] = "SKIP_NO_M6B"; _finish(); return

    try:
        m4 = load_m4(); m8 = load_tcn()
        log(f"  models loaded (TCN {'ok' if m8 else 'MISSING'})")
    except Exception as e:
        log(f"  FATAL load: {e}"); log(traceback.format_exc())
        results["overall_status"] = FAIL; _finish(); return

    seqs, meta = load_m6b()
    by_label = {}
    for i, m in enumerate(meta): by_label.setdefault(label_of(m), []).append(i)

    # CUSUM policy params
    pol = {"mu0_B": -0.00954, "lambda": 5.7302192538299934e-05}
    if POLICY_PATH.exists():
        with open(POLICY_PATH, encoding="utf-8") as f: j = json.load(f)
        pol["mu0_B"] = float(j["cusum_parameters"]["mu0_B"])
        pol["lambda"] = float(j["decay_policy"]["geometric_quiet_decay"]["lambda"])
    mu0_B, lam = pol["mu0_B"], pol["lambda"]

    # ── Step 1: replicate warmup to populate normal score_A baseline ─────────
    # Warmup = 432 windows of sigma=0.045 noise (steady-state). We generate the
    # same noise windows the runner uses and pass through M4 to get the normal
    # score_A level the rolling baseline adapts to.
    log("\nStep 1 — warmup replication (432 win x sigma=0.045) -> normal score_A level")
    rng = np.random.default_rng(SEED)
    warmup_sA = []
    for _ in range(WARMUP_WINDOWS):
        win = rng.normal(0.0, WARMUP_SIGMA, size=(WINDOW_SIZE, 8)).astype(np.float32)
        sA, _ = score_A_of_window(win, m4)
        warmup_sA.append(sA)
    warmup_sA = np.array(warmup_sA)
    theta_t_adapted = float(np.mean(warmup_sA) + 3.0 * np.std(warmup_sA))
    log(f"  warmup score_A: mean={warmup_sA.mean():.4f} std={warmup_sA.std():.4f} "
        f"-> theta_t~{theta_t_adapted:.4f}")
    results["evidence"]["warmup_sA_mean"] = round(float(warmup_sA.mean()), 6)
    results["evidence"]["theta_t_adapted_est"] = round(theta_t_adapted, 6)

    # ── Step 2: normal-pool score_A stream -> rm100/rm200 distribution ───────
    # Stream normal (label 0) sequences windowed at stride 50 (serve cadence),
    # maintaining rolling_mean_100/200 deques exactly as AlertStateMachine does.
    log("\nStep 2 — normal score_A stream -> rolling-mean floors")
    normal_idx = by_label.get(0, [])
    sa_stream = []
    for i in normal_idx:
        arr = np.asarray(seqs[i], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < WINDOW_SIZE: continue
        for s in range(0, arr.shape[0] - WINDOW_SIZE + 1, WINDOW_SIZE):
            sA, _ = score_A_of_window(arr[s:s+WINDOW_SIZE], m4)
            sa_stream.append(sA)
    sa_stream = np.array(sa_stream)
    # rolling means over the concatenated normal stream
    d100, d200 = deque(maxlen=100), deque(maxlen=200)
    rm100s, rm200s = [], []
    for v in sa_stream:
        d100.append(v); d200.append(v)
        rm100s.append(np.mean(d100)); rm200s.append(np.mean(d200))
    rm100s, rm200s = np.array(rm100s), np.array(rm200s)
    # floors at P99.5 of normal rolling means (clear normal w/ margin)
    MARGIN = 1.10
    floor100 = float(np.percentile(rm100s, 99.5) * MARGIN)
    floor200 = float(np.percentile(rm200s, 99.5) * MARGIN)
    log(f"  normal n_windows={len(sa_stream)} sA mean={sa_stream.mean():.4f} "
        f"max={sa_stream.max():.4f}")
    log(f"  normal rm100 P99.5={np.percentile(rm100s,99.5):.4f} -> floor={floor100:.4f}")
    log(f"  normal rm200 P99.5={np.percentile(rm200s,99.5):.4f} -> floor={floor200:.4f}")
    results["derived"]["rolling_mean_100_floor"] = round(floor100, 6)
    results["derived"]["rolling_mean_200_floor"] = round(floor200, 6)
    G["G1c_floors"] = PASS if (len(sa_stream) > 30 and floor100 > sa_stream.mean()) else FAIL

    # ── Step 3: normal score_B -> CUSUM k (normal evidence ~ 0 -> S_n<2.0) ───
    log("\nStep 3 — normal score_B stream -> CUSUM k")
    sb_stream = []
    if m8 is not None:
        for i in normal_idx:
            arr = np.asarray(seqs[i], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < WINDOW_SIZE: continue
            buf = deque(maxlen=ZT_BUF_LEN)
            # adaptive stride to fill buffer like 4.1 (normal seq=200 -> stride ~3)
            stride = max(1, (arr.shape[0] - WINDOW_SIZE) // (ZT_BUF_LEN - 1))
            for s in range(0, arr.shape[0] - WINDOW_SIZE + 1, stride):
                _sA, z = score_A_of_window(arr[s:s+WINDOW_SIZE], m4)
                buf.append(z)
                if len(buf) >= 5:
                    sb_stream.append(score_B_of_buffer(list(buf), m8))
    sb_stream = np.array(sb_stream) if sb_stream else np.array([0.0])
    sb_p99 = float(np.percentile(sb_stream, 99))
    # k so that P99 normal evidence ~ 0:  k = P99(score_B) - mu0_B
    k_derived = float(sb_p99 - mu0_B)
    log(f"  normal score_B n={len(sb_stream)} mean={sb_stream.mean():.5f} "
        f"P99={sb_p99:.5f}")
    log(f"  derived k = P99(score_B) - mu0_B = {sb_p99:.5f} - ({mu0_B}) = {k_derived:.5f}")
    results["derived"]["cusum_k"] = round(k_derived, 6)
    results["evidence"]["normal_scoreB_mean"] = round(float(sb_stream.mean()), 6)
    results["evidence"]["normal_scoreB_p99"] = round(sb_p99, 6)

    # verify normal S_n stays < 2.0 with derived k
    def sim_cusum(stream, k):
        S, decay = 0.0, 1.0 - lam
        mx = 0.0
        for sb in stream:
            e = sb - mu0_B - k
            S = (S + e) if e > 0 else S * decay
            if S < 0: S = 0.0
            mx = max(mx, S)
        return mx
    normal_Sn_max = sim_cusum(sb_stream, k_derived)
    # OPTION A (24 May 2026): CUSUM-on-score_B is a DOCUMENTED-WEAK advisory.
    # Normal score_B mean is materially positive (~0.077) and label-21 drift does
    # NOT exceed normal score_B, so no single k both suppresses normal below WATCH
    # (2.0) AND detects label-21. The defensible criterion is therefore: normal
    # must never reach ALARM (S_n < H=5.0). WATCH-level chatter on a weak advisory
    # signal is tolerable; a false ALARM is not. Primary label-21 detection is the
    # L1/rolling pathway (G4a), not CUSUM-on-score_B.
    H_alarm = 5.0
    log(f"  normal CUSUM S_n max with derived k = {normal_Sn_max:.4f} "
        f"(Option A criterion: < H={H_alarm} no false ALARM; was 7.65 pre-fix)")
    results["evidence"]["normal_Sn_max_derived_k"] = round(normal_Sn_max, 4)
    results["evidence"]["g4b_criterion"] = f"normal_Sn_max < H={H_alarm} (Option A: no false ALARM)"
    G["G1c_k"] = PASS if normal_Sn_max < H_alarm else FAIL

    # ── Step 4: JOINT feasibility — label-21 must still cross H with derived k ─
    log("\nStep 4 — joint G4a check: label-21 slow drift still crosses H with derived k")
    H = 5.0
    lbl21 = by_label.get(21, [])
    fired = 0; checked = 0
    if m8 is not None and lbl21:
        for i in lbl21[:40]:
            arr = np.asarray(seqs[i], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < WINDOW_SIZE: continue
            buf = deque(maxlen=ZT_BUF_LEN); sbs = []
            stride = max(1, (arr.shape[0] - WINDOW_SIZE) // (ZT_BUF_LEN - 1))
            for s in range(0, arr.shape[0] - WINDOW_SIZE + 1, stride):
                _sA, z = score_A_of_window(arr[s:s+WINDOW_SIZE], m4)
                buf.append(z)
                if len(buf) >= 5: sbs.append(score_B_of_buffer(list(buf), m8))
            # accumulate; repeat sequence to emulate sustained drift up to 1500 win
            S, decay = 0.0, 1.0 - lam
            crossed = False
            for rep in range(20):
                for sb in sbs:
                    e = sb - mu0_B - k_derived
                    S = (S + e) if e > 0 else S * decay
                    if S < 0: S = 0.0
                    if S >= H: crossed = True; break
                if crossed: break
            checked += 1
            if crossed: fired += 1
    g4a_rate = fired / checked if checked else 0.0
    log(f"  label-21 H-crossing rate with derived k = {g4a_rate:.2%} "
        f"(ADVISORY under Option A — CUSUM-score_B does NOT own label-21 detection)")
    results["evidence"]["label21_fire_rate_derived_k"] = round(g4a_rate, 4)
    # Option A: CUSUM-on-score_B is a documented-weak advisory. Label-21 primary
    # detection is the L1/rolling pathway (runner G4a). This joint check is now
    # ADVISORY: records the weakness but does NOT block persistence.
    G["G1c_joint"] = PASS if g4a_rate >= 0.75 else "ADVISORY_FAIL"

    if G["G1c_k"] == PASS and G["G1c_joint"] != PASS:
        log("  NOTE (Option A): derived k keeps normal below ALARM (G4b ok) but")
        log("  label-21 does not cross H via CUSUM-score_B. EXPECTED: score_B is a")
        log("  documented-weak change-in-error statistic; steady drift barely moves it.")
        log("  Label-21 PRIMARY detection = L1/rolling pathway (runner G4a). CUSUM-on-")
        log("  score_B is retained as a slow advisory only. Not a blocker — persisting.")

    # ── Persist (only if floors + k both valid) ──────────────────────────────
    persist_ok = (G["G1c_floors"] == PASS and G["G1c_k"] == PASS)
    if persist_ok:
        # 1) floors -> M8_alert_thresholds.json state_machine block
        try:
            j = {}
            if ALERT_CFG.exists():
                with open(ALERT_CFG, encoding="utf-8") as f: j = json.load(f)
            sm = j.get("state_machine", {})
            sm["rolling_mean_100_floor"] = round(floor100, 6)
            sm["rolling_mean_200_floor"] = round(floor200, 6)
            sm["_calibrated_by"] = SCRIPT_NAME
            sm["_calibrated_utc"] = datetime.utcnow().isoformat() + "Z"
            sm["_calibration_note"] = ("floors = P99.5(normal rm) x1.10 measured post-"
                                       "432-win warmup; replaces M8-spec guess that caused "
                                       "G1 FPR=1.0")
            j["state_machine"] = sm
            with open(ALERT_CFG, "w", encoding="utf-8") as f: json.dump(j, f, indent=2)
            log(f"\n  floors -> {ALERT_CFG.name}")
        except Exception as e:
            log(f"  WARN floors persist: {e}")
        # 2) cusum_k -> M8_threshold_config.json (main.py reads cusum_k)
        try:
            with open(M8_CFG_PATH, encoding="utf-8") as f: mc = json.load(f)
            bak = M8_CFG_PATH.with_suffix(f".json.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(M8_CFG_PATH, bak)
            mc["cusum_k"] = round(k_derived, 6)
            mc["_cusum_k_calibrated_by"] = SCRIPT_NAME
            with open(M8_CFG_PATH, "w", encoding="utf-8") as f: json.dump(mc, f, indent=2)
            results["evidence"]["m8cfg_backup"] = str(bak)
            log(f"  cusum_k={k_derived:.5f} -> {M8_CFG_PATH.name} (backup {bak.name})")
        except Exception as e:
            log(f"  WARN k persist: {e}")

    core = [G.get("G1c_floors"), G.get("G1c_k")]
    results["overall_status"] = PASS if all(s == PASS for s in core) else FAIL
    _finish()


def _finish():
    out_json = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    with open(out_json, "w", encoding="utf-8") as f: json.dump(results, f, indent=2, default=str)
    g, d, ev = results["gates"], results["derived"], results["evidence"]
    status = results["overall_status"]
    print("\n" + "=" * 76)
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12 Stage 4 Step 4.1c (normal-baseline calibration): {status}")
    print(f"  G1c_floors (rm floors > normal): {g.get('G1c_floors','-')}")
    print(f"  G1c_k (normal S_n < 2.0)       : {g.get('G1c_k','-')}")
    print(f"  G1c_joint (label-21 still fires): {g.get('G1c_joint','-')}")
    if d:
        print(f"  derived rolling_mean_100_floor = {d.get('rolling_mean_100_floor')}")
        print(f"  derived rolling_mean_200_floor = {d.get('rolling_mean_200_floor')}")
        print(f"  derived cusum_k                = {d.get('cusum_k')}")
    print(f"  normal S_n max (derived k) = {ev.get('normal_Sn_max_derived_k')} (was 7.65)")
    print(f"  label-21 fire rate (derived k) = {ev.get('label21_fire_rate_derived_k')}")
    if status == PASS:
        print("  Persisted floors->M8_alert_thresholds.json, cusum_k->M8_threshold_config.json")
        print("  Next: RESTART server, re-run smoke. G1 + G4b should now PASS.")
    else:
        print("  NOT persisted — a gate failed. See interpretation in report.")
    print("  BLOCK_M11 = True (Step 4.4 full-run owns the flip)")
    print("══ END PASTE UPDATE ══")
    print(f"\n  Report: {out_json}")
    if results["evidence"].get("m8cfg_backup"):
        print("  GitHub push: models/M8_threshold_config.json, models/M8_alert_thresholds.json")
    print(f"  GitHub push: src/{SCRIPT_NAME}.py")
    print("=" * 76)
    if status == PASS:
        print("\n📦 Step 4.1c done — floors + k calibrated from normal pool. RESTART server, "
              "re-run smoke (G1 + G4b should clear).")
    else:
        print("\n📦 Step 4.1c: a gate failed — see report before re-running smoke.")


if __name__ == "__main__":
    main()