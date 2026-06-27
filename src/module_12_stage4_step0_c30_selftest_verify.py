# =============================================================================
# src/module_12_stage4_step0_c30_selftest_verify.py
# PumpSmart v14.2 — M12 Stage 4, Step 4.0: C-30 startup self-test verification
# (v2 — diagnose-then-verify; repo-root paths; compare-to-disk; conditional regen)
# =============================================================================
#
# WHY v2 (lessons from the v1 run, 24 May 2026)
# ---------------------------------------------
# v1 surfaced three failures. Two were bugs in v1 itself; one was a REAL finding:
#
#   * G4_0_1 FileNotFoundError 'models\lstm_ae_baseline_final.pth'
#       -> CWD bug. model_registry uses MODEL_DIR = Path("models") (relative to
#          CWD). Run from src/ -> resolves to src\models\... (absent). FIX: this
#          script resolves the repo ROOT (parents[1]) and loads every artifact
#          by absolute path, independent of CWD.
#
#   * G4_0_2 selftest RAISED, 20/20 windows, max_diff ~1.28 (flat, systematic)
#       -> REAL finding, NOT a production feature-path defect. The embedded
#          _REFERENCES in feature_builder_selftest.py are STALE relative to the
#          current Stage-2 build_m7_features. A flat ~1.28 across all 20 windows
#          is a systematic reference/offset mismatch, not float noise.
#          PROTOCOL: do NOT blindly regenerate. First PROVE the live builder is
#          correct against the persisted training matrix on disk
#          (M6B_feature_matrix_v3.csv — every row built by the current builder).
#          Only if the builder matches disk do we regenerate the stale reference.
#
#   * G4_0_3 'no module-level build_m7_features to patch'
#       -> the selftest does a function-local `from app.runtime import
#          feature_builder as fb` and calls fb.build_m7_features. FIX: poison the
#          DEFINITION site app.runtime.feature_builder.build_m7_features.
#
# GATE PLAN (diagnose -> disk-truth -> conditional regen -> verify -> poison)
# --------------------------------------------------------------------------
#   G4_0_1   Production M4 loads (repo-root path) & forward() -> recon-only
#            Tensor [B,50,8]  (the contract Stage-3's 4-tuple M4 violated).
#
#   G4_0_2a  DIAGNOSIS: per-column max_diff of the live builder vs the embedded
#            reference at the 16 checked indices [0-10,12-16,24]. Localises which
#            columns diverge and by how much. Diagnostic only.
#
#   G4_0_2b  DISK TRUTH: rebuild N rows from M6B_feature_matrix_v3.csv via the
#            production builder + a real M4 pass; compare at the 16 bit-exact
#            indices within 1e-5. Compare-to-persisted-artifact gate (Stage 1.5
#            lesson). PASS => live builder correct; v1 divergence is a stale
#            reference, safe to regenerate.
#
#   G4_0_2c  CONDITIONAL REGEN + VERIFY: only if G4_0_2b PASSES, regenerate the
#            embedded _REFERENCES in feature_builder_selftest.py against the
#            current builder, then run production run_startup_selftest clean ->
#            must NOT raise. If G4_0_2b FAILS, ABORT (real defect) — no regen.
#
#   G4_0_3   POISON: corrupt a bit-exact column at the DEFINITION site and
#            confirm run_startup_selftest RAISES. A guard that cannot fail is
#            vacuous (Stage 1.5 self-referential-gate lesson).
#
# ROBUSTNESS (Souvik, 24 May 2026): focus local (M6B + v3 matrix on disk).
# If M6B absent -> LOUD explicit SKIP (never silent pass). Production-without-M6B
# C-30 behaviour is an M11 (full-deployment) item — logged in the report.
#
# This step modifies ONE production artifact and ONLY on a proven-correct
# builder: app/runtime/feature_builder_selftest.py (embedded _REFERENCES).
# A timestamped .bak is written first. No model is retrained.
#
# RUN (CWD-independent)
#   .venv\Scripts\Activate.ps1
#   python src/module_12_stage4_step0_c30_selftest_verify.py
#
# OUTPUTS
#   outputs/reports/module_12_stage4_step0_c30_selftest_verify_report.md
#   outputs/reports/module_12_stage4_step0_c30_selftest_verify_results.json
# =============================================================================

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import (DEVICE, IS_GPU, MODEL_DIR, SYNTH_DIR, OUTPUT_DIR)

from datetime import date, datetime
import json
import pickle
import shutil
import re
import importlib
import warnings
import traceback
warnings.filterwarnings("ignore")

import numpy as np
import torch

SCRIPT_NAME = "module_12_stage4_step0_c30_selftest_verify"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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


M4_PATH        = _abspath(MODEL_DIR / "lstm_ae_baseline_final.pth", "models/lstm_ae_baseline_final.pth")
M6B_PATH       = _abspath(SYNTH_DIR / "M6B_combined_sequences.pkl", "data/synthetic/M6B_combined_sequences.pkl")
V3_MATRIX_PATH = _abspath(SYNTH_DIR / "M6B_feature_matrix_v3.csv",  "data/synthetic/M6B_feature_matrix_v3.csv")
SELFTEST_PATH  = _ROOT / "app" / "runtime" / "feature_builder_selftest.py"

CHECK_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 24]
TOL = 1e-05
N_DISK_ROWS = 40


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


results = {
    "script": SCRIPT_NAME,
    "stage": "M12 Stage 4 — Step 4.0 (C-30 selftest verification, v2)",
    "timestamp": datetime.now().isoformat(),
    "device": str(DEVICE), "is_gpu": bool(IS_GPU),
    "paths": {"m4": str(M4_PATH), "m6b": str(M6B_PATH),
              "v3_matrix": str(V3_MATRIX_PATH), "selftest": str(SELFTEST_PATH)},
    "m6b_present": M6B_PATH.exists(), "v3_present": V3_MATRIX_PATH.exists(),
    "gates": {}, "evidence": {}, "overall_status": "UNKNOWN", "block_m11": True,
}
GATES = results["gates"]


def save_results():
    out = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return out


def save_report():
    out = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    g, ev = results["gates"], results["evidence"]
    L = [f"# {SCRIPT_NAME}", "",
         "PumpSmart v14.2 — M12 Stage 4, Step 4.0 — C-30 startup self-test verification (v2)", "",
         f"- Date: {date.today().isoformat()}",
         f"- Device: {results['device']} | GPU: {results['is_gpu']}",
         f"- M4: {results['paths']['m4']}",
         f"- M6B present: {results['m6b_present']} | v3 matrix present: {results['v3_present']}",
         f"- Overall status: **{results['overall_status']}**",
         f"- BLOCK_M11: {results['block_m11']}", "",
         "## Gate results", "",
         "| Gate | Description | Status |",
         "|------|-------------|--------|",
         f"| G4_0_1  | Production M4 forward() returns recon-only Tensor [B,50,8] | {g.get('G4_0_1','-')} |",
         f"| G4_0_2a | DIAGNOSIS: per-column live-vs-reference divergence | {g.get('G4_0_2a','-')} |",
         f"| G4_0_2b | DISK TRUTH: builder matches M6B_feature_matrix_v3.csv (1e-5) | {g.get('G4_0_2b','-')} |",
         f"| G4_0_2c | Regenerate stale reference + clean selftest passes | {g.get('G4_0_2c','-')} |",
         f"| G4_0_3  | POISON: corrupted bit-exact column makes selftest RAISE | {g.get('G4_0_3','-')} |",
         "", "## Evidence", ""]
    for k, v in ev.items():
        L.append(f"- **{k}**: {v}")
    L += ["", "## Interpretation", ""]
    if results["overall_status"] == PASS:
        L.append("- The live builder matches the persisted v3 training matrix at all 16 "
                 "bit-exact columns, so the v1 selftest divergence was a STALE embedded "
                 "reference (not a feature-path defect). The reference was regenerated and "
                 "the C-30 guard now passes clean and fails on poison. Feature path verified. "
                 "**Safe to proceed to Step 4.1 (live score_C calibration).**")
    elif results["overall_status"] == "SKIP_NO_M6B":
        L.append("- M6B absent; identity gates could not run. Expected HF-Spaces condition. "
                 "Re-run locally for the PASS. Production-without-M6B C-30 is an M11 item.")
    else:
        L.append("- A gate FAILED. If G4_0_2b failed, the live builder does NOT match the "
                 "persisted v3 matrix — a REAL bit-exact break, not a stale reference; the "
                 "reference was NOT regenerated. Investigate the diverging columns in G4_0_2a.")
    L += ["", "## Carry-forward to M11 (full deployment)", "",
          "- HF Spaces ships `models/` without `data/synthetic/M6B_combined_sequences.pkl`; "
          "the production registry SKIPs the C-30 selftest there. M11 should ship a small "
          "frozen reference bundle (e.g. `models/M6B_selftest_refset.npz`, ~20 windows) so the "
          "guard runs in production too — negligible size — or accept the documented SKIP and "
          "rely on this build-time (local) verification."]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out


def load_m6b(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"M6B pickle is {type(d).__name__}, expected dict.")
    seqs = d.get("sequences")
    meta = d.get("metadata", d.get("meta"))
    if seqs is None or meta is None:
        raise KeyError(f"M6B keys={list(d.keys())}; need 'sequences' + 'metadata'/'meta'.")
    if len(seqs) != len(meta):
        raise ValueError(f"len(sequences)={len(seqs)} != len(meta)={len(meta)}.")
    return seqs, meta


@torch.no_grad()
def m4_forward(window_np, m4_model):
    x = torch.from_numpy(np.asarray(window_np, dtype=np.float32)).float().unsqueeze(0)
    recon = m4_model(x)
    mae = torch.mean(torch.abs(x - recon), dim=1).squeeze(0)
    enc = m4_model.encoder
    o1, _ = enc.lstm1(x); o2, (h, c) = enc.lstm2(o1); z = enc.bn(h[-1])
    return mae.cpu().numpy(), z.squeeze(0).cpu().numpy()


def main():
    log("=" * 76)
    log(f"  {SCRIPT_NAME}  |  {date.today().isoformat()}")
    log(f"  Device: {DEVICE} | GPU: {IS_GPU}")
    log("  Stage 4 Step 4.0 (v2) — diagnose-then-verify the C-30 self-test")
    log("=" * 76)
    log(f"  repo root : {_ROOT}")
    log(f"  M4 path   : {M4_PATH}  (exists={M4_PATH.exists()})")
    log(f"  M6B path  : {M6B_PATH}  (exists={M6B_PATH.exists()})")
    log(f"  v3 matrix : {V3_MATRIX_PATH}  (exists={V3_MATRIX_PATH.exists()})")

    if not M6B_PATH.exists():
        log("\n" + "!" * 76)
        log("  SKIP — M6B_combined_sequences.pkl absent. C-30 identity gates need it.")
        log("         Expected HF-Spaces condition; production SKIPs gracefully.")
        log("         Re-run locally for the verification PASS. (M11 deployment item.)")
        log("!" * 76)
        for k in ("G4_0_1", "G4_0_2a", "G4_0_2b", "G4_0_2c", "G4_0_3"):
            GATES[k] = SKIP
        results["overall_status"] = "SKIP_NO_M6B"
        _finish(); return

    # ── G4_0_1 ───────────────────────────────────────────────────────────────
    log("\nG4_0_1 — load PRODUCTION M4 (repo-root path) & check recon-only contract")
    m4_model = None
    try:
        from app.runtime.model_registry import _M4LSTMAutoencoder
        m4_model = _M4LSTMAutoencoder()
        state = torch.load(M4_PATH, map_location="cpu", weights_only=True)
        m4_model.load_state_dict(state, strict=True)
        m4_model.eval()
        for p in m4_model.parameters():
            p.requires_grad_(False)
        n_params = sum(p.numel() for p in m4_model.parameters())
        with torch.no_grad():
            out = m4_model(torch.zeros(1, 50, 8, dtype=torch.float32))
        shape_ok = isinstance(out, torch.Tensor) and tuple(out.shape) == (1, 50, 8)
        results["evidence"]["m4_params"]    = int(n_params)
        results["evidence"]["m4_out_type"]  = type(out).__name__
        results["evidence"]["m4_out_shape"] = (tuple(out.shape) if isinstance(out, torch.Tensor) else "n/a")
        if shape_ok:
            GATES["G4_0_1"] = PASS
            log(f"  PASS — recon-only Tensor [1,50,8], {n_params:,} params "
                f"(the contract Stage-3's 4-tuple M4 violated).")
        else:
            GATES["G4_0_1"] = FAIL
            log(f"  FAIL — forward() returned {type(out).__name__}; need recon-only [1,50,8].")
    except Exception as e:
        GATES["G4_0_1"] = FAIL
        results["evidence"]["G4_0_1_error"] = f"{type(e).__name__}: {e}"
        log(f"  FAIL — {type(e).__name__}: {e}")
        log(traceback.format_exc())

    if m4_model is None:
        results["overall_status"] = FAIL
        _finish(); return

    try:
        seqs, meta = load_m6b(M6B_PATH)
        results["evidence"]["m6b_n_sequences"] = len(seqs)
        log(f"  M6B loaded — {len(seqs)} sequences")
    except Exception as e:
        results["evidence"]["m6b_load_error"] = f"{type(e).__name__}: {e}"
        log(f"  M6B load error: {type(e).__name__}: {e}")
        results["overall_status"] = FAIL
        _finish(); return

    from app.runtime import feature_builder as fb

    # ── G4_0_2a — per-column divergence diagnosis ──────────────────────────────
    log("\nG4_0_2a — DIAGNOSIS: per-column divergence (live builder vs embedded ref)")
    try:
        from app.runtime import feature_builder_selftest as st
        refs = list(getattr(st, "_REFERENCES", []))
        if not refs:
            raise RuntimeError("feature_builder_selftest._REFERENCES empty/missing.")
        per_col_max = {idx: 0.0 for idx in CHECK_IDX}
        n_ref = 0
        for ref in refs:
            win = np.asarray(ref["window"], dtype=np.float32)
            mae, z = m4_forward(win, m4_model)
            fv = fb.build_m7_features(mae, win, z)
            got = np.array([float(fv[i]) for i in CHECK_IDX])
            exp = np.array(ref["expected"], dtype=np.float64)
            for j, idx in enumerate(CHECK_IDX):
                per_col_max[idx] = max(per_col_max[idx], abs(got[j] - exp[j]))
            n_ref += 1
        worst = sorted(per_col_max.items(), key=lambda kv: kv[1], reverse=True)
        results["evidence"]["per_column_max_diff"] = {str(k): round(v, 6) for k, v in per_col_max.items()}
        results["evidence"]["n_reference_windows"] = n_ref
        GATES["G4_0_2a"] = PASS
        log(f"  Computed per-column max_diff over {n_ref} reference windows.")
        log("  Worst-diverging columns (idx: max_diff):")
        for idx, d in worst[:6]:
            log(f"      idx {idx:>2} : {d:.4e}")
        n_div = sum(1 for _, d in per_col_max.items() if d >= TOL)
        log(f"  {n_div}/{len(CHECK_IDX)} checked columns diverge >= {TOL:.0e} "
            f"(many cols sharing a flat offset -> stale-reference signature).")
    except Exception as e:
        GATES["G4_0_2a"] = FAIL
        results["evidence"]["G4_0_2a_error"] = f"{type(e).__name__}: {e}"
        log(f"  FAIL — {type(e).__name__}: {e}")

    # ── G4_0_2b — disk truth ───────────────────────────────────────────────────
    log("\nG4_0_2b — DISK TRUTH: live builder vs M6B_feature_matrix_v3.csv (16 cols, 1e-5)")
    if not V3_MATRIX_PATH.exists():
        GATES["G4_0_2b"] = FAIL
        results["evidence"]["v3_matrix"] = "ABSENT — cannot prove builder on disk"
        log("  FAIL — v3 matrix absent; cannot run disk-truth gate. Do NOT regenerate.")
    else:
        try:
            import pandas as pd
            df_head = pd.read_csv(V3_MATRIX_PATH, nrows=1)
            cols = list(df_head.columns)
            label_col = next((c for c in cols if c.lower() in
                              ("label_int", "label", "y", "class")), cols[-1])
            feat_cols = [c for c in cols if c != label_col][:33]
            df = pd.read_csv(V3_MATRIX_PATH, usecols=feat_cols)
            disk = df.to_numpy(dtype=np.float64)
            disk_mae = disk[:, 0:8]

            rng = np.random.default_rng(40)
            idxs = rng.choice(len(seqs), size=min(N_DISK_ROWS, len(seqs)), replace=False)
            ANCHOR_TOL = 5e-3   # training mae was GPU float16 (Stage 1.5 anchor method)
            checked, unmatched, max_abs_diff = 0, 0, 0.0
            for i in idxs:
                arr = np.asarray(seqs[i], dtype=np.float32)
                if arr.shape[0] < 50:
                    continue
                win = arr[:50]
                mae, z = m4_forward(win, m4_model)
                fv = fb.build_m7_features(mae, win, z)
                d = np.max(np.abs(disk_mae - mae[None, :]), axis=1)
                jmin = int(np.argmin(d))
                if d[jmin] > ANCHOR_TOL:
                    unmatched += 1
                    continue
                got = np.array([fv[k] for k in CHECK_IDX], dtype=np.float64)
                exp = disk[jmin, CHECK_IDX]
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(got - exp))))
                checked += 1
            results["evidence"]["disk_truth_checked_rows"] = checked
            results["evidence"]["disk_truth_unmatched"]   = unmatched
            results["evidence"]["disk_truth_max_abs_diff"] = round(max_abs_diff, 12)
            if checked == 0:
                GATES["G4_0_2b"] = FAIL
                log(f"  FAIL — 0 rows matched (anchor {ANCHOR_TOL:.0e}); vacuous. "
                    f"unmatched={unmatched}.")
            elif max_abs_diff < TOL:
                GATES["G4_0_2b"] = PASS
                log(f"  PASS — {checked} rows matched, max_abs_diff {max_abs_diff:.2e} < {TOL:.0e}.")
                log("         Builder IS correct vs v3 matrix => v1 divergence is a STALE reference.")
            else:
                GATES["G4_0_2b"] = FAIL
                log(f"  FAIL — {checked} rows matched but max_abs_diff {max_abs_diff:.2e} "
                    f">= {TOL:.0e}. REAL bit-exact break — do NOT regenerate.")
        except Exception as e:
            GATES["G4_0_2b"] = FAIL
            results["evidence"]["G4_0_2b_error"] = f"{type(e).__name__}: {e}"
            log(f"  FAIL — {type(e).__name__}: {e}")
            log(traceback.format_exc())

    # ── G4_0_2c — conditional regen + verify ───────────────────────────────────
    log("\nG4_0_2c — regenerate stale reference (only if G4_0_2b PASS) + clean selftest")
    if GATES.get("G4_0_2b") != PASS:
        GATES["G4_0_2c"] = SKIP
        log("  SKIP — builder not proven correct on disk; refusing to regenerate "
            "(would mask a real defect). This is the protocol guard.")
    else:
        try:
            rng = np.random.default_rng(415)
            take = sorted(rng.choice(len(seqs), size=min(20, len(seqs)), replace=False).tolist())
            new_refs = []
            for si in take:
                arr = np.asarray(seqs[si], dtype=np.float32)
                if arr.shape[0] < 50:
                    continue
                win = arr[:50]
                mae, z = m4_forward(win, m4_model)
                fv = fb.build_m7_features(mae, win, z)
                new_refs.append({"seq_idx": int(si),
                                 "window": win.astype(float).tolist(),
                                 "expected": [float(fv[k]) for k in CHECK_IDX]})
            if not new_refs:
                raise RuntimeError("no sequences >= 50 steps to build references.")
            src = SELFTEST_PATH.read_text(encoding="utf-8")
            bak = SELFTEST_PATH.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(SELFTEST_PATH, bak)
            results["evidence"]["selftest_backup"] = str(bak)
            ref_literal = "_REFERENCES = " + json.dumps(new_refs)
            new_src, n_sub = re.subn(
                r"_REFERENCES\s*=\s*\[.*?\]\s*(?=\n\n|\ndef )",
                ref_literal + "\n", src, count=1, flags=re.DOTALL)
            if n_sub != 1:
                raise RuntimeError(f"could not locate _REFERENCES block (subs={n_sub}). "
                                   f"Backup at {bak}.")
            SELFTEST_PATH.write_text(new_src, encoding="utf-8")
            results["evidence"]["references_regenerated"] = len(new_refs)
            log(f"  Regenerated {len(new_refs)} references; backup -> {bak.name}")
            from app.runtime import feature_builder_selftest as st2
            importlib.reload(st2)
            st2.run_startup_selftest(m4_model, seqs, meta)
            GATES["G4_0_2c"] = PASS
            log("  PASS — production run_startup_selftest clean after regeneration.")
        except RuntimeError as e:
            GATES["G4_0_2c"] = FAIL
            results["evidence"]["G4_0_2c_error"] = f"RuntimeError: {e}"
            log(f"  FAIL — selftest still raised after regen: {e}")
        except Exception as e:
            GATES["G4_0_2c"] = FAIL
            results["evidence"]["G4_0_2c_error"] = f"{type(e).__name__}: {e}"
            log(f"  FAIL — {type(e).__name__}: {e}")
            log(traceback.format_exc())

    # ── G4_0_3 — poison at definition site ─────────────────────────────────────
    log("\nG4_0_3 — POISON: corrupt a bit-exact column at the definition site -> must RAISE")
    if GATES.get("G4_0_2c") != PASS:
        GATES["G4_0_3"] = SKIP
        log("  SKIP — clean selftest not green; poison test only meaningful on a passing guard.")
    else:
        import app.runtime.feature_builder as fbmod
        from app.runtime import feature_builder_selftest as st3
        original = fbmod.build_m7_features
        try:
            def _poisoned(*a, **k):
                fv = np.array(original(*a, **k), dtype=np.float32, copy=True)
                fv[0] = fv[0] + 10.0   # idx 0 (mae_MotSV) — a CHECK_IDX column
                return fv
            fbmod.build_m7_features = _poisoned
            raised, txt = False, ""
            try:
                st3.run_startup_selftest(m4_model, seqs, meta)
            except Exception as e:
                raised, txt = True, f"{type(e).__name__}: {e}"
            if raised:
                GATES["G4_0_3"] = PASS
                results["evidence"]["poison"] = f"raised as required — {txt.split(':')[0]}"
                log(f"  PASS — poison triggered a raise ({txt.split(':')[0]}). Guard has teeth.")
            else:
                GATES["G4_0_3"] = FAIL
                results["evidence"]["poison"] = "did NOT raise — guard is vacuous"
                log("  FAIL — poisoned features did NOT raise; guard is vacuous.")
        finally:
            fbmod.build_m7_features = original
            log("  (restored original build_m7_features)")

    core = [GATES.get(k) for k in ("G4_0_1", "G4_0_2b", "G4_0_2c", "G4_0_3")]
    if all(s == PASS for s in core):
        results["overall_status"] = PASS
    elif any(s == FAIL for s in core):
        results["overall_status"] = FAIL
    else:
        results["overall_status"] = "SKIP_NO_M6B"
    results["block_m11"] = True
    _finish()


def _finish():
    out_json = save_results()
    out_md = save_report()
    g, status = results["gates"], results["overall_status"]
    print("\n" + "=" * 76)
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12 Stage 4 Step 4.0 (C-30 selftest verification, v2): {status}")
    print(f"  G4_0_1  M4 recon-only contract        : {g.get('G4_0_1','-')}")
    print(f"  G4_0_2a per-column divergence diag     : {g.get('G4_0_2a','-')}")
    print(f"  G4_0_2b builder == v3 matrix (disk)    : {g.get('G4_0_2b','-')}")
    print(f"  G4_0_2c stale reference regenerated    : {g.get('G4_0_2c','-')}")
    print(f"  G4_0_3  poison selftest raises         : {g.get('G4_0_3','-')}")
    if status == PASS:
        print("  Finding: v1 divergence was a STALE C-30 reference; builder proven")
        print("           correct vs M6B_feature_matrix_v3.csv. Reference regenerated.")
        print("  Stage 3 §11.5 caveat: CLOSED — C-30 guard verified genuine.")
        print("  Next: Step 4.1 — live score_C distribution calibration.")
    elif status == "SKIP_NO_M6B":
        print("  M6B absent — verification skipped (expected on HF Spaces).")
    else:
        print("  FAILED. If G4_0_2b failed -> REAL bit-exact break (not stale ref).")
        print("  Reference NOT regenerated. Investigate G4_0_2a columns.")
    print("  Config decision: new M8_alert_thresholds.json (M8_threshold_config.json untouched)")
    print("  BLOCK_M11 = True  (Step 4.4 owns the flip)")
    print("══ END PASTE UPDATE ══")
    print("\n══ FILE MANIFEST ══")
    print("  Reports (Spaces upload):")
    print(f"    {out_md}")
    print(f"    {out_json}")
    if results["evidence"].get("references_regenerated"):
        print("  Production artifact MODIFIED (reference block only, backup written):")
        print(f"    {SELFTEST_PATH}")
        print(f"    backup: {results['evidence'].get('selftest_backup','-')}")
        print("  GitHub push: app/runtime/feature_builder_selftest.py (regenerated refs)")
    else:
        print("  Production artifacts modified: NONE")
    print(f"  GitHub push: src/{SCRIPT_NAME}.py")
    print("=" * 76)
    print()
    if status == PASS:
        print("📦 M12 Stage 4 Step 4.0 done — C-30 guard verified, stale reference fixed. "
              "Starting Step 4.1 (live score_C calibration). Provide the Step 4.1 script.")
    elif status == "SKIP_NO_M6B":
        print("📦 Step 4.0 SKIPPED (no M6B locally). Place M6B on disk and re-run.")
    else:
        print("📦 Step 4.0 FAILED. See G4_0_2a / G4_0_2b evidence before Step 4.1.")


if __name__ == "__main__":
    main()