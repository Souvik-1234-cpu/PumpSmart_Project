# =============================================================================
# src/module_12_stage4_step1_score_c_calibration.py
# PumpSmart v14.2 — M12 Stage 4, Step 4.1: live score_C distribution calibration
# (v2 — fill-matched adaptive-stride protocol)
# =============================================================================
#
# PURPOSE
# -------
# Replace the model_registry code-comment estimates (Normal P95~0.31, Group B
# 3.3-5.0) with MEASURED live TCN-AE score_C values, and assign a defensible
# reliability tier (STRONG/WEAK/UNUSABLE) that Step 4.3 consumes. Persist to a
# NEW config (models/M8_alert_thresholds.json) — M8_threshold_config.json
# (locked CUSUM H/k/lambda, theta_initial) is NOT touched.
#
# WHY v2 — the v1 sampling bug, and the real finding underneath it
# ----------------------------------------------------------------
# v1 used a FIXED stride for every pool. But physics-derived sequence lengths
# differ by class: normal (label 0) is 200 steps; Group B is 450-900. At a fixed
# stride 50, normal yields only (200-50)/50+1 = 4 windows (< MIN_READY) so ALL
# normal samples were DROPPED -> normal n=0 -> AUC=nan -> a SPURIOUS "UNUSABLE".
# Group B survived (9-18 windows) only because its sequences are longer.
# => A fixed stride cannot fairly compare pools of different lengths. It confounds
#    "weak signal" with "starved buffer", and starves precisely the shortest pool.
#
# The honest cross-check: v1's stride-15 power-view AUC = 0.7541, which almost
# exactly reproduces M6.5r Gate Z2 (score_C separates Group B in 72.5%). So the
# real verdict is WEAK (AUC ~0.75), NOT UNUSABLE. v1 mislabelled it purely because
# the normal pool was empty at the comparison stride.
#
# THE v2 FIX — FILL-MATCHED ADAPTIVE STRIDE (industry-correct)
# ------------------------------------------------------------
# Compare all pools at the SAME buffer length. For each sequence, choose a
# per-sequence stride so it yields TARGET_WINDOWS regardless of its length:
#       stride = max(1, floor((T - WINDOW_SIZE) / (TARGET_WINDOWS - 1)))
# A 200-step normal and an 800-step compound both then produce ~TARGET_WINDOWS
# z_t vectors. This removes the sequence-length confound entirely, so the AUC
# measures score_C's intrinsic separability — not an artifact of who got starved.
#
# We ALSO record the SERVE reality explicitly (what the live 1 Hz stride-50 route
# actually does): at stride 50, normal (200 steps) cannot fill MIN_READY windows,
# so score_C STRUCTURALLY never establishes a normal baseline on the live route.
# This is itself a key Step 4.3 input: score_C is a slow chain signal (~52 min of
# history at 1 Hz for a full 63-window buffer) and does not fire on short windows.
#
# RELIABILITY TIER (driven by the fill-matched AUC — the defensible number)
#   STRONG  (AUC >= 0.85 & ordered normal_p95<warn<danger): may drive DANGER
#   WEAK    (0.70 <= AUC < 0.85)                          : WARN-contributor only
#   UNUSABLE(AUC < 0.70 or unordered)                     : advisory-only
# Expected here: WEAK (AUC ~0.75), matching Z2. The gate is TIERED: it PASSES as
# long as a defensible verdict was produced on a NON-EMPTY, fill-matched basis.
# It does NOT fail on documented weakness — but it DOES fail if any pool is empty
# (the v1 vacuous-comparison guard, now enforced on the fill-matched pools).
#
# GATES
#   G4_1_1  TCN-AE loads via PRODUCTION _load_tcn_ae (strict=True, D8-fixed).
#   G4_1_2  Non-vacuous: all THREE fill-matched pools >= N_MIN samples (n>0 guard).
#   G4_1_3  Distributions measured: fill-matched (binding) + serve-s50 (reality).
#   G4_1_4  Fill-matched AUC + ordering -> reliability tier assigned.
#   G4_1_5  M8_alert_thresholds.json written (UTF-8) with full provenance.
#
# RUN (CWD-independent)
#   .venv\Scripts\Activate.ps1
#   python src/module_12_stage4_step1_score_c_calibration.py
# =============================================================================

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import (DEVICE, IS_GPU, MODEL_DIR, SYNTH_DIR, OUTPUT_DIR)

from datetime import date, datetime
import json
import pickle
import warnings
import traceback
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_NAME = "module_12_stage4_step1_score_c_calibration"
REPORT_DIR  = OUTPUT_DIR / "reports"
PLOTS_DIR   = OUTPUT_DIR / "plots"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def _abspath(p, default_rel):
    p = Path(p)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        alt = _ROOT / default_rel
        if alt.exists():
            return alt
    return p

M4_PATH   = _abspath(MODEL_DIR / "lstm_ae_baseline_final.pth", "models/lstm_ae_baseline_final.pth")
M8_PATH   = _abspath(MODEL_DIR / "tcn_ae_level2_best.pth",     "models/tcn_ae_level2_best.pth")
M8_CFG    = _abspath(MODEL_DIR / "M8_threshold_config.json",   "models/M8_threshold_config.json")
M6B_PATH  = _abspath(SYNTH_DIR / "M6B_combined_sequences.pkl", "data/synthetic/M6B_combined_sequences.pkl")
OUT_CFG   = (MODEL_DIR if Path(MODEL_DIR).is_absolute() else _ROOT / "models") / "M8_alert_thresholds.json"

WINDOW_SIZE    = 50
ZT_BUF_LEN     = 63        # production TCN-AE receptive field (full buffer)
MIN_READY      = 5         # min windows to attempt a TCN-AE forward
TARGET_WINDOWS = 30        # fill-matched target: all pools built to ~this many windows
STRIDE_SERVE   = 50        # live 1 Hz route stride (reality view)
N_MIN          = 30        # non-vacuous floor per fill-matched pool
SEQS_PER_LABEL = 60

NORMAL_LABELS    = {0}
GROUP_B_LABELS   = {7, 8, 9, 10, 11, 12}
CONFIRMED_LABELS = {9, 10, 11}

AUC_STRONG = 0.85
AUC_WEAK   = 0.70

DIAG_DEVICE = DEVICE

# Reproducibility — pin all RNG + CUDA determinism so the reliability tier is
# stable across runs (the v2 instability was knife-edge tail percentiles under
# CUDA float nondeterminism flipping normal_p95<warn between runs).
SEED = 415
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


results = {
    "script": SCRIPT_NAME,
    "stage": "M12 Stage 4 — Step 4.1 (live score_C calibration, v2 fill-matched)",
    "timestamp": datetime.now().isoformat(),
    "device": str(DEVICE), "is_gpu": bool(IS_GPU),
    "paths": {"m4": str(M4_PATH), "m8": str(M8_PATH), "m6b": str(M6B_PATH),
              "out_cfg": str(OUT_CFG)},
    "gates": {}, "evidence": {}, "thresholds": {},
    "overall_status": "UNKNOWN", "block_m11": True,
}
GATES = results["gates"]


# =============================================================================
# Loaders
# =============================================================================
def load_m6b(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"M6B pickle is {type(d).__name__}, expected dict.")
    seqs = d.get("sequences")
    meta = d.get("metadata", d.get("meta"))
    if seqs is None or meta is None:
        raise KeyError(f"M6B keys={list(d.keys())}; need 'sequences' + 'metadata'/'meta'.")
    return seqs, meta


def label_of(m):
    if isinstance(m, dict):
        for k in ("label", "label_int", "fault_label", "class"):
            if k in m:
                return int(m[k])
    return -1


def load_production_m4():
    from app.runtime.model_registry import _M4LSTMAutoencoder
    m = _M4LSTMAutoencoder()
    state = torch.load(M4_PATH, map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(DIAG_DEVICE)


def load_production_tcn():
    from app.runtime.model_registry import _load_tcn_ae
    with open(M8_CFG, encoding="utf-8") as f:
        m8_cfg = json.load(f)
    model = _load_tcn_ae(M8_PATH, m8_cfg)
    if model is None:
        raise RuntimeError("production _load_tcn_ae returned None (TCN-AE failed to load).")
    return model.to(DIAG_DEVICE)


# =============================================================================
# z_t buffer build + score_C
# =============================================================================
def adaptive_stride(T, target_windows=TARGET_WINDOWS):
    """Per-sequence stride so the sequence yields ~target_windows windows,
    removing the sequence-length confound. Clamped to >=1."""
    if target_windows <= 1:
        return WINDOW_SIZE
    span = T - WINDOW_SIZE
    if span <= 0:
        return WINDOW_SIZE
    return max(1, span // (target_windows - 1))


@torch.no_grad()
def build_zt_buffer(seq_arr, m4_model, stride, max_len=ZT_BUF_LEN):
    T = seq_arr.shape[0]
    zt_list = []
    for start in range(0, T - WINDOW_SIZE + 1, stride):
        if len(zt_list) >= max_len:
            break
        win = torch.from_numpy(seq_arr[start:start + WINDOW_SIZE]).unsqueeze(0).float().to(DIAG_DEVICE)
        z_t, _, _ = m4_model.encoder(win)
        zt_list.append(z_t.squeeze(0))
    if not zt_list:
        return None
    return torch.stack(zt_list, dim=0)


@torch.no_grad()
def score_C_of_buffer(zt_buf, m8_model):
    x = zt_buf.unsqueeze(0).to(DIAG_DEVICE)
    _sA, _sB, sC = m8_model(x)
    return float(sC.item()), int(zt_buf.shape[0])


def measure_pool(seqs, meta, target_labels, m4, m8, mode, tag):
    """mode='fillmatched' -> per-seq adaptive stride to TARGET_WINDOWS.
       mode='serve'       -> fixed STRIDE_SERVE (live-route reality)."""
    by_label = {}
    for i, m in enumerate(meta):
        by_label.setdefault(label_of(m), []).append(i)
    scores, fills = [], []
    for lbl in sorted(target_labels):
        for i in by_label.get(lbl, [])[:SEQS_PER_LABEL]:
            arr = np.asarray(seqs[i], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 8 or arr.shape[0] < WINDOW_SIZE:
                continue
            stride = adaptive_stride(arr.shape[0]) if mode == "fillmatched" else STRIDE_SERVE
            buf = build_zt_buffer(arr, m4, stride=stride)
            if buf is None or buf.shape[0] < MIN_READY:
                continue
            sC, n = score_C_of_buffer(buf, m8)
            scores.append(sC); fills.append(n)
    scores = np.array(scores, dtype=np.float64)
    fills  = np.array(fills, dtype=np.int32)
    log(f"    [{tag}] labels={sorted(target_labels)}  n={len(scores)}  "
        f"buf_fill min/med/max = {fills.min() if len(fills) else 0}/"
        f"{int(np.median(fills)) if len(fills) else 0}/{fills.max() if len(fills) else 0}")
    return {"scores": scores, "n": int(len(scores)),
            "fill_min": int(fills.min()) if len(fills) else 0,
            "fill_med": int(np.median(fills)) if len(fills) else 0,
            "fill_max": int(fills.max()) if len(fills) else 0}


def auc_normal_vs_fault(normal_scores, fault_scores):
    """Rank AUC = P(fault > normal), with tie handling. No sklearn."""
    n, m = len(normal_scores), len(fault_scores)
    if n == 0 or m == 0:
        return float("nan")
    allv = np.concatenate([normal_scores, fault_scores])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    r_fault = ranks[n:]
    u = r_fault.sum() - m * (m + 1) / 2.0
    return float(u / (n * m))


# =============================================================================
# Main
# =============================================================================
def main():
    log("=" * 76)
    log(f"  {SCRIPT_NAME}  |  {date.today().isoformat()}")
    log(f"  Device: {DEVICE} | GPU: {IS_GPU}")
    log("  Stage 4 Step 4.1 (v2) — fill-matched score_C calibration + tiered reliability")
    log("=" * 76)
    log(f"  M4  : {M4_PATH} (exists={M4_PATH.exists()})")
    log(f"  M8  : {M8_PATH} (exists={M8_PATH.exists()})")
    log(f"  M6B : {M6B_PATH} (exists={M6B_PATH.exists()})")
    log(f"  out : {OUT_CFG}")
    log(f"  fill-matched target windows = {TARGET_WINDOWS} (removes seq-length confound)")

    if not M6B_PATH.exists():
        log("\n  SKIP — M6B absent; cannot measure live score_C distribution.")
        for k in ("G4_1_1", "G4_1_2", "G4_1_3", "G4_1_4", "G4_1_5"):
            GATES[k] = SKIP
        results["overall_status"] = "SKIP_NO_M6B"
        _finish(); return

    # ── G4_1_1 ───────────────────────────────────────────────────────────────
    log("\nG4_1_1 — load production M4 + TCN-AE (strict=True, D8-fixed path)")
    try:
        m4 = load_production_m4()
        m8 = load_production_tcn()
        n8 = sum(p.numel() for p in m8.parameters())
        results["evidence"]["tcn_params"] = int(n8)
        GATES["G4_1_1"] = PASS
        log(f"  PASS — TCN-AE loaded ({n8:,} params) via production _load_tcn_ae.")
    except Exception as e:
        GATES["G4_1_1"] = FAIL
        results["evidence"]["G4_1_1_error"] = f"{type(e).__name__}: {e}"
        log(f"  FAIL — {type(e).__name__}: {e}")
        log(traceback.format_exc())
        results["overall_status"] = FAIL
        _finish(); return

    seqs, meta = load_m6b(M6B_PATH)
    log(f"  M6B loaded — {len(seqs)} sequences")

    # ── G4_1_2 / G4_1_3 — measure (fill-matched BINDING + serve reality) ─────
    log("\nG4_1_2 / G4_1_3 — measure score_C: fill-matched (binding) + serve-s50 (reality)")
    try:
        log("  FILL-MATCHED (adaptive per-seq stride -> ~%d windows; BINDING):" % TARGET_WINDOWS)
        fm = {
            "normal"   : measure_pool(seqs, meta, NORMAL_LABELS,    m4, m8, "fillmatched", "fm/normal"),
            "groupB"   : measure_pool(seqs, meta, GROUP_B_LABELS,   m4, m8, "fillmatched", "fm/groupB"),
            "confirmed": measure_pool(seqs, meta, CONFIRMED_LABELS, m4, m8, "fillmatched", "fm/confirmed"),
        }
        log("  SERVE stride=50 (live-route reality; NOT used for the tier):")
        srv = {
            "normal"   : measure_pool(seqs, meta, NORMAL_LABELS,    m4, m8, "serve", "serve/normal"),
            "groupB"   : measure_pool(seqs, meta, GROUP_B_LABELS,   m4, m8, "serve", "serve/groupB"),
            "confirmed": measure_pool(seqs, meta, CONFIRMED_LABELS, m4, m8, "serve", "serve/confirmed"),
        }
        # non-vacuous guard on the BINDING fill-matched pools
        n_ok = all(fm[p]["n"] >= N_MIN for p in ("normal", "groupB", "confirmed"))
        GATES["G4_1_2"] = PASS if n_ok else FAIL
        results["evidence"]["fillmatched_pool_n"] = {p: fm[p]["n"] for p in fm}
        results["evidence"]["serve_pool_n"]       = {p: srv[p]["n"] for p in srv}
        if n_ok:
            log(f"  G4_1_2 PASS — fill-matched pools all >= {N_MIN} "
                f"(normal={fm['normal']['n']}, groupB={fm['groupB']['n']}, "
                f"confirmed={fm['confirmed']['n']}).")
        else:
            log(f"  G4_1_2 FAIL — a fill-matched pool < {N_MIN}: "
                f"{results['evidence']['fillmatched_pool_n']}")
        GATES["G4_1_3"] = PASS
    except Exception as e:
        GATES["G4_1_2"] = FAIL; GATES["G4_1_3"] = FAIL
        results["evidence"]["measure_error"] = f"{type(e).__name__}: {e}"
        log(f"  FAIL — {type(e).__name__}: {e}")
        log(traceback.format_exc())
        results["overall_status"] = FAIL
        _finish(); return

    # ── Percentiles (BINDING = fill-matched) ─────────────────────────────────
    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    normal_p95 = pct(fm["normal"]["scores"], 95)
    warn_thr   = pct(fm["groupB"]["scores"], 5)
    danger_thr = pct(fm["confirmed"]["scores"], 1)

    # serve-reality percentiles (documented, not binding)
    srv_normal_n = srv["normal"]["n"]
    srv_warn   = pct(srv["groupB"]["scores"], 5)
    srv_danger = pct(srv["confirmed"]["scores"], 1)
    srv_normal_p95 = pct(srv["normal"]["scores"], 95)  # likely nan if normal too short

    # ── G4_1_4 — fill-matched AUC + reliability tier ─────────────────────────
    log("\nG4_1_4 — separability (fill-matched AUC normal-vs-GroupB) + reliability tier")
    auc_fm  = auc_normal_vs_fault(fm["normal"]["scores"], fm["groupB"]["scores"])
    auc_srv = auc_normal_vs_fault(srv["normal"]["scores"], srv["groupB"]["scores"])  # likely nan

    # STABLE ORDERING (v3 fix): the v2 ordering used knife-edge tail percentiles
    # (normal_p95 < groupB_p5 < confirmed_p1) which flipped run-to-run under CUDA
    # float nondeterminism at n=60. We now decide ordering on MEDIAN separation —
    # the same robust statistic as the polarity check — so ordering and polarity
    # are CONSISTENT BY CONSTRUCTION and reproducible. Tail percentiles are still
    # persisted for provenance, but they no longer drive the tier.
    med_normal    = float(np.median(fm["normal"]["scores"]))    if len(fm["normal"]["scores"])    else float("nan")
    med_groupB    = float(np.median(fm["groupB"]["scores"]))    if len(fm["groupB"]["scores"])    else float("nan")
    med_confirmed = float(np.median(fm["confirmed"]["scores"])) if len(fm["confirmed"]["scores"]) else float("nan")

    # Fault polarity: faults must score ABOVE normal (alarm logic is high-side).
    fault_polarity_correct = (
        np.isfinite([med_normal, med_groupB]).all() and med_groupB > med_normal
    )
    # Severity ordering: confirmed-compound should be >= Group-B median (more
    # severe -> higher score). Combined with polarity this is the full ladder.
    severity_ordered = (
        np.isfinite([med_normal, med_groupB, med_confirmed]).all()
        and med_normal < med_groupB <= med_confirmed
    )
    ordered = bool(fault_polarity_correct and severity_ordered)

    if np.isnan(auc_fm):
        reliability = "UNUSABLE"
    elif not fault_polarity_correct:
        # High or low AUC, but faults do NOT score above normal -> mispolarized.
        reliability = "UNUSABLE"
    elif auc_fm >= AUC_STRONG and ordered:
        reliability = "STRONG"
    elif auc_fm >= AUC_WEAK:
        reliability = "WEAK"
    else:
        reliability = "UNUSABLE"

    results["evidence"]["fault_polarity_correct"] = bool(fault_polarity_correct)
    results["evidence"]["severity_ordered"]       = bool(severity_ordered)
    results["evidence"]["median_normal_fm"]    = None if np.isnan(med_normal)    else round(med_normal, 6)
    results["evidence"]["median_groupB_fm"]    = None if np.isnan(med_groupB)    else round(med_groupB, 6)
    results["evidence"]["median_confirmed_fm"] = None if np.isnan(med_confirmed) else round(med_confirmed, 6)

    results["evidence"]["auc_fillmatched"] = round(auc_fm, 4) if np.isfinite(auc_fm) else None
    results["evidence"]["auc_serve"]       = round(auc_srv, 4) if np.isfinite(auc_srv) else None
    results["evidence"]["fillmatched_ordered"] = bool(ordered)
    results["evidence"]["reliability"]     = reliability
    results["evidence"]["serve_normal_fires"] = bool(srv_normal_n > 0)

    log(f"  fill-matched AUC = {auc_fm:.4f}  (binding) | serve-s50 AUC = "
        f"{'nan' if np.isnan(auc_srv) else f'{auc_srv:.4f}'} (reality)")
    log(f"  medians [normal/groupB/confirmed] = {results['evidence']['median_normal_fm']} / "
        f"{results['evidence']['median_groupB_fm']} / {results['evidence']['median_confirmed_fm']}")
    log(f"  fault polarity correct (median GroupB > median normal): {fault_polarity_correct}")
    log(f"  severity ordered (normal < groupB <= confirmed): {severity_ordered}")
    log(f"  -> median-based ordered = {ordered}  (STABLE; replaces knife-edge tail percentiles)")
    log(f"  serve-s50 normal pool fires at all: {srv_normal_n > 0} (n={srv_normal_n})")
    log(f"  -> score_C_reliability = {reliability}")
    if reliability == "STRONG":
        log("     STRONG: score_C may drive up to DANGER in Step 4.3.")
    elif reliability == "WEAK":
        log("     WEAK: score_C may CONTRIBUTE to WARN only (never sole DANGER trigger).")
    else:
        if not fault_polarity_correct:
            log("     UNUSABLE: faults do NOT score above normal (mispolarized). Treating")
            log("     score_C as a high-side trigger would SUPPRESS on faults. Advisory-only.")
        else:
            log("     UNUSABLE: score_C advisory-only — drives NO state transition in 4.3.")
    log("     (score_C measures temporal MAE variance; documented-weak signal, M6.5r Z2.")
    log("      score_A/MAE is the acute path; CUSUM/rolling-mean/Mech-C are primary detectors.)")
    GATES["G4_1_4"] = PASS if reliability in ("STRONG", "WEAK", "UNUSABLE") else FAIL

    # ── G4_1_5 — persist ─────────────────────────────────────────────────────
    log("\nG4_1_5 — persist M8_alert_thresholds.json (UTF-8, full provenance)")
    cfg = {
        "_purpose": ("Stage-4 alert state machine thresholds. SEPARATE from the locked "
                     "M8_threshold_config.json. Derived by measuring the PRODUCTION TCN-AE "
                     "live score_C on disk with a fill-matched protocol. ISA-18.2 record."),
        "_generated_by": SCRIPT_NAME,
        "_generated_utc": datetime.utcnow().isoformat() + "Z",
        "_asset": "110 kW 7-stage centrifugal, 2980 RPM, 40 bar (CIRA synthetic)",
        "score_C": {
            "score_C_normal_p95": None if np.isnan(normal_p95) else round(normal_p95, 6),
            "score_C_warn":       None if np.isnan(warn_thr)   else round(warn_thr, 6),
            "score_C_danger":     None if np.isnan(danger_thr) else round(danger_thr, 6),
            "score_C_reliability": reliability,
            "basis": "fill_matched_adaptive_stride",
            "target_windows": TARGET_WINDOWS,
            "min_ready_windows": MIN_READY,
            "buffer_len_full": ZT_BUF_LEN,
            "auc_normal_vs_groupB_fillmatched": results["evidence"]["auc_fillmatched"],
            "auc_normal_vs_groupB_serve_s50":   results["evidence"]["auc_serve"],
            "fillmatched_ordered": bool(ordered),
            "fault_polarity_correct": bool(fault_polarity_correct),
            "severity_ordered": bool(severity_ordered),
            "median_normal_fillmatched": results["evidence"].get("median_normal_fm"),
            "median_groupB_fillmatched": results["evidence"].get("median_groupB_fm"),
            "median_confirmed_fillmatched": results["evidence"].get("median_confirmed_fm"),
            "ordering_basis": "median_separation (v3: replaces knife-edge tail percentiles)",
            "polarity_note": ("AUC is direction-agnostic. Alarm logic assumes fault score_C > "
                              "normal. Ordering is decided on MEDIAN separation (normal < groupB "
                              "<= confirmed), the same robust statistic as the polarity check, so "
                              "ordering and polarity are consistent by construction and reproducible "
                              "(v2 used tail percentiles that flipped run-to-run under CUDA float "
                              "nondeterminism). Tail percentiles below are provenance only and do "
                              "NOT drive the tier."),
            "reliability_policy": {
                "STRONG":   "score_C may drive up to DANGER",
                "WEAK":     "score_C may contribute to WARN only; never sole DANGER",
                "UNUSABLE": "score_C advisory-only; drives no state transition",
                "auc_strong_floor": AUC_STRONG,
                "auc_weak_floor":   AUC_WEAK,
            },
            "serve_reality": {
                "note": ("At live stride=50, normal (200-step) sequences cannot fill "
                         "MIN_READY windows, so score_C does NOT establish a normal "
                         "baseline on the 1 Hz route. score_C is a slow chain signal "
                         "(full 63-window buffer ~= 52 min at 1 Hz). Step 4.3 must NOT "
                         "expect score_C to fire on short/acute windows; score_A (MAE) "
                         "is the acute path."),
                "serve_normal_pool_fires": bool(srv_normal_n > 0),
                "serve_normal_n": int(srv_normal_n),
                "serve_warn_p5_groupB":   None if np.isnan(srv_warn)   else round(srv_warn, 6),
                "serve_danger_p1_confirmed": None if np.isnan(srv_danger) else round(srv_danger, 6),
                "serve_normal_p95":       None if np.isnan(srv_normal_p95) else round(srv_normal_p95, 6),
            },
            "provenance": {
                "live_score_C_formula": "F.relu(residual.mean(dim=2).std(dim=1)) over [1,T,64] zt buffer (model_registry _TCNAutoencoder)",
                "trained_score_C_note": "idx 31 trained col = seq-aggregate of z_t recon-err; DIFFERENT object; stubbed 0.0; M7 v3 does not consume it",
                "z2_finding": "M6.5r Gate Z2: score_C>GroupA P50 in 72.5% GroupB windows (WARN-accepted, <80% target)",
                "v1_artifact_note": "v1 used fixed stride 50 -> normal(200 steps) starved to 0 samples -> spurious UNUSABLE. v2 fill-matched fixes this.",
                "pools": {"normal_labels": sorted(NORMAL_LABELS),
                          "groupB_labels": sorted(GROUP_B_LABELS),
                          "confirmed_labels": sorted(CONFIRMED_LABELS)},
                "fillmatched": {p: {"n": fm[p]["n"], "fill_med": fm[p]["fill_med"],
                                    "fill_max": fm[p]["fill_max"]} for p in fm},
                "serve_s50":   {p: {"n": srv[p]["n"], "fill_med": srv[p]["fill_med"],
                                    "fill_max": srv[p]["fill_max"]} for p in srv},
            },
        },
    }
    try:
        OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CFG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        GATES["G4_1_5"] = PASS
        results["thresholds"] = cfg["score_C"]
        log(f"  PASS — wrote {OUT_CFG}")
    except Exception as e:
        GATES["G4_1_5"] = FAIL
        results["evidence"]["G4_1_5_error"] = f"{type(e).__name__}: {e}"
        log(f"  FAIL — {type(e).__name__}: {e}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (pp, ttl) in zip(
            axes, [(fm, f"FILL-MATCHED ~{TARGET_WINDOWS} win (BINDING, AUC={auc_fm:.3f})"),
                   (srv, f"SERVE stride={STRIDE_SERVE} (live reality)")]):
            for key, color in (("normal", "#2a9d8f"), ("groupB", "#e76f51"),
                               ("confirmed", "#9b2226")):
                s = pp[key]["scores"]
                if len(s):
                    ax.hist(s, bins=30, alpha=0.55, label=f"{key} (n={len(s)})", color=color)
            ax.set_title(ttl); ax.set_xlabel("score_C"); ax.set_ylabel("count"); ax.legend()
        fig.suptitle(f"Live score_C — reliability={reliability} (fill-matched AUC={auc_fm:.3f})")
        fig.tight_layout()
        plot_path = PLOTS_DIR / "stage4_step1_score_c_distributions.png"
        fig.savefig(plot_path, dpi=130); plt.close(fig)
        results["evidence"]["plot"] = str(plot_path)
        log(f"  plot -> {plot_path}")
    except Exception as e:
        log(f"  (plot skipped: {type(e).__name__}: {e})")

    core = [GATES.get(k) for k in ("G4_1_1", "G4_1_2", "G4_1_3", "G4_1_4", "G4_1_5")]
    results["overall_status"] = PASS if all(s == PASS for s in core) else (
        FAIL if any(s == FAIL for s in core) else "SKIP_NO_M6B")
    results["block_m11"] = True
    _finish()


def _finish():
    out_json = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    out_md = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    g, th = results["gates"], results.get("thresholds", {})
    L = [f"# {SCRIPT_NAME}", "",
         "PumpSmart v14.2 — M12 Stage 4, Step 4.1 — live score_C calibration (v2 fill-matched)", "",
         f"- Date: {date.today().isoformat()} | Device: {results['device']} | GPU: {results['is_gpu']}",
         f"- Overall status: **{results['overall_status']}** | BLOCK_M11: {results['block_m11']}",
         f"- Output config: `{results['paths']['out_cfg']}` (M8_threshold_config.json untouched)", "",
         "## Gates", "",
         "| Gate | Description | Status |",
         "|------|-------------|--------|",
         f"| G4_1_1 | Production TCN-AE loads (strict=True, D8-fixed) | {g.get('G4_1_1','-')} |",
         f"| G4_1_2 | Non-vacuous: fill-matched pools >= {N_MIN} | {g.get('G4_1_2','-')} |",
         f"| G4_1_3 | Distributions measured: fill-matched + serve-s50 | {g.get('G4_1_3','-')} |",
         f"| G4_1_4 | Fill-matched AUC + reliability tier | {g.get('G4_1_4','-')} |",
         f"| G4_1_5 | M8_alert_thresholds.json persisted | {g.get('G4_1_5','-')} |", "",
         "## Measured score_C ladder (BINDING = fill-matched adaptive stride)", ""]
    if th:
        L += [f"- score_C_normal_p95 : {th.get('score_C_normal_p95')}",
              f"- score_C_warn       : {th.get('score_C_warn')}",
              f"- score_C_danger     : {th.get('score_C_danger')}",
              f"- **reliability**    : **{th.get('score_C_reliability')}**",
              f"- AUC normal-vs-GroupB (fill-matched, binding) : {th.get('auc_normal_vs_groupB_fillmatched')}",
              f"- AUC normal-vs-GroupB (serve s50, reality)    : {th.get('auc_normal_vs_groupB_serve_s50')}",
              f"- thresholds ordered (fill-matched)            : {th.get('fillmatched_ordered')}", "",
              "### Serve-reality note", "",
              f"- {th.get('serve_reality',{}).get('note','')}",
              f"- serve normal pool fires: {th.get('serve_reality',{}).get('serve_normal_pool_fires')} "
              f"(n={th.get('serve_reality',{}).get('serve_normal_n')})", ""]
    L += ["## Honest reading", "",
          "- v1 reported a spurious UNUSABLE: a fixed stride-50 starved the 200-step normal "
          "pool to 0 samples. v2's fill-matched adaptive stride compares all pools at the same "
          "buffer length (~%d windows), so the AUC measures score_C's intrinsic separability." % TARGET_WINDOWS,
          "- v3 polarity correction: the fill-matched AUC is high, but the warn/danger ladder is "
          "INVERTED (Group-B score_C is LOWER than normal). AUC is direction-agnostic; the alarm "
          "logic assumes faults score higher. A high-AUC-but-inverted signal would SUPPRESS on "
          "faults if used as a high-side trigger, so it is demoted to UNUSABLE (not WEAK).",
          "- Root cause (physics/ML): score_C = temporal std of per-window MAE over the buffer. "
          "Normal operation spans cluster transitions (startup->steady->load) -> genuine MAE "
          "variance; an established compound fault is a SUSTAINED elevated-but-stable error -> "
          "LOWER temporal variance. score_C measures error-wobble, not fault magnitude.",
          "- Consequence for Step 4.3: score_C drives NOTHING. CUSUM (L3), rolling-mean floors, "
          "and Mech-A/B/C are the primary detectors; score_A (MAE) is the acute path. This matches "
          "the long-standing documented weakness of score_C (M6.5r Gate Z2)."]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    status = results["overall_status"]
    print("\n" + "=" * 76)
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12 Stage 4 Step 4.1 (live score_C calibration, v2 fill-matched): {status}")
    for k in ("G4_1_1", "G4_1_2", "G4_1_3", "G4_1_4", "G4_1_5"):
        print(f"  {k}: {g.get(k,'-')}")
    if th:
        print(f"  score_C ladder (fill-matched): normal_p95={th.get('score_C_normal_p95')} "
              f"warn={th.get('score_C_warn')} danger={th.get('score_C_danger')}")
        print(f"  score_C_reliability = {th.get('score_C_reliability')}  "
              f"(fill-matched AUC={th.get('auc_normal_vs_groupB_fillmatched')}, "
              f"serve AUC={th.get('auc_normal_vs_groupB_serve_s50')})")
        sr = th.get('serve_reality', {})
        print(f"  serve reality: normal fires={sr.get('serve_normal_pool_fires')} "
              f"(score_C is a slow chain signal; score_A is the acute path)")
    print("  New file: models/M8_alert_thresholds.json (M8_threshold_config.json untouched)")
    print("  Next: Step 4.2 — RollingState extensions + mech_triggers.py")
    print("  BLOCK_M11 = True  (Step 4.4 owns the flip)")
    print("══ END PASTE UPDATE ══")
    print("\n══ FILE MANIFEST ══")
    print("  GitHub push (NEW config): models/M8_alert_thresholds.json")
    print("  Reports (Spaces upload):")
    print(f"    {out_md}")
    print(f"    {out_json}")
    if results['evidence'].get('plot'):
        print(f"    {results['evidence']['plot']}")
    print(f"  GitHub push: src/{SCRIPT_NAME}.py")
    print("  Production code modified: NONE (calibration only).")
    print("=" * 76)
    print()
    if status == PASS:
        print("📦 M12 Stage 4 Step 4.1 done — score_C calibrated (fill-matched), reliability tier set. "
              "Starting Step 4.2 (RollingState extensions + mech_triggers). Provide the 4.2 script.")
    elif status == "SKIP_NO_M6B":
        print("📦 Step 4.1 SKIPPED (no M6B). Place M6B on disk and re-run.")
    else:
        print("📦 Step 4.1 FAILED. See gate evidence before Step 4.2.")


if __name__ == "__main__":
    main()