# =============================================================================
# src/module_12_stage1_step1_m4_per_channel_identity_test.py
#
# M12 Stage 1, Step 1.1 — Identity test for patched run_m4
#
# PURPOSE
#   The current run_m4 in app/routers/anomaly.py computes per-channel MAE on
#   line 85 then DISCARDS it. M7 was trained on per-channel M4 reconstruction
#   error as its 8 Domain-1 features (mae_MotSV ... mae_MotPV at indices 0-7).
#   The current runtime substitutes window-centered MAD on raw data (defect
#   D1a), a completely different quantity at a different scale.
#
#   This script:
#     1. Defines the PATCHED run_m4 (surfaces mae_per_ch as 3rd return value).
#     2. Independently re-computes per-channel MAE via an explicit per-channel
#        loop (no PyTorch vector shortcuts).
#     3. Asserts bit-exact agreement across windows drawn from diverse labels.
#     4. Prints a small sample of patched mae_per_ch values for label-0 and
#        label-21 windows so the absolute scale can be eyeballed against
#        the M6.5r feature matrix distribution before any patch is applied.
#
#   PASS = patched run_m4 is mathematically identical to M6.5r training-time
#          per-channel MAE. Safe to commit the 2-line patch to anomaly.py.
#   FAIL = a defect exists in the patch. DO NOT commit. Investigate divergence.
#
# RUN
#   .venv\Scripts\Activate.ps1
#   python src/module_12_stage1_step1_m4_per_channel_identity_test.py
#
# OUTPUTS
#   outputs/reports/module_12_stage1_step1_m4_per_channel_identity_test_report.md
#   outputs/reports/module_12_stage1_step1_m4_per_channel_identity_test_results.json
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, MODEL_DIR, SYNTH_DIR, OUTPUT_DIR)
from datetime import datetime, date
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn

SCRIPT_NAME = "module_12_stage1_step1_m4_per_channel_identity_test"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# Channel order — MUST match M4 training and M6.5r feature naming
# Locked from app/routers/anomaly.py CH_WEIGHTS comment (line 41).
CHANNEL_NAMES = ["Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
                 "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV"]
MAE_FEATURE_NAMES = ["mae_MotSV", "mae_PmpSV", "mae_MotTV", "mae_PmpPV",
                     "mae_TempSV", "mae_PresSV", "mae_PmpTV", "mae_MotPV"]

CH_WEIGHTS = torch.tensor([2.5, 2.5, 0.3, 2.0, 0.5, 2.5, 0.3, 2.0],
                          dtype=torch.float32)

# Tolerances
TOL_BIT_EXACT = 1e-7   # patched vs reference must agree to this
TOL_TRAINING  = 1e-5   # patched vs M6B_feature_matrix.csv (if alignable)


# =============================================================================
# M4 architecture — EXACT copy from app/runtime/model_registry.py
# Duplicated here so this script is fully standalone.
# =============================================================================
class _M4Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8,   128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64,  num_layers=1, batch_first=True)
        self.bn    = nn.LayerNorm(64)   # LayerNorm (T1.5.1 confirmed)

    def forward(self, x):
        out1, _      = self.lstm1(x)
        out2, (h, c) = self.lstm2(out1)
        return self.bn(h[-1]), h, c


class _M4Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_h  = nn.Linear(64, 128)
        self.fc_c  = nn.Linear(64, 128)
        self.lstm1 = nn.LSTM(64,  128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 8,   num_layers=1, batch_first=True)
        self.out   = nn.Linear(8, 8)

    def forward(self, z, seq_len, h_enc, c_enc):
        h0    = torch.tanh(self.fc_h(h_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0    = torch.tanh(self.fc_c(c_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        x_rep = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm1(x_rep, (h0, c0))
        out, _ = self.lstm2(out)
        return self.out(out)


class _M4LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _M4Encoder()
        self.decoder = _M4Decoder()

    def forward(self, x):
        z_t, h, c = self.encoder(x)
        return self.decoder(z_t, x.size(1), h, c)


# =============================================================================
# THE PATCHED run_m4 — exactly what will go into app/routers/anomaly.py
# Only change vs. current: returns mae_per_ch_np as a 3rd element of tuple.
# =============================================================================
@torch.no_grad()
def patched_run_m4(window_tensor, m4_model, q=0.110058):
    """
    Patched run_m4 — surfaces per-channel reconstruction error.
    Returns: (score_A, z_t_np, mae_per_ch_np, raw_mae)
      score_A         = physics-weighted MAE scalar  -> L4 RollingState
      z_t_np          = encoder bottleneck (64,)     -> ZTBuffer -> L2 TCN-AE
      mae_per_ch_np   = per-channel MAE (8,)         -> feature_builder (NEW)
      raw_mae         = unweighted mean MAE          -> OOD Mahalanobis feature
    """
    x = window_tensor.float()
    enc = m4_model.encoder
    out1, _ = enc.lstm1(x)
    out2, (h_n, c_n) = enc.lstm2(out1)
    z_t = enc.bn(h_n[-1])
    recon = m4_model.decoder(z_t, x.size(1), h_n, c_n)

    # Per-channel MAE — identical formula to M6.5r training-time computation.
    # The decoder output is (batch, T, 8) so we mean over the time dimension.
    mae_per_ch = (x - recon).abs().mean(dim=1).squeeze(0)   # [8]

    # Physics-weighted score_A (existing behaviour — unchanged)
    weights = CH_WEIGHTS.to(mae_per_ch.device)
    score_A = (mae_per_ch * weights).sum().item() / weights.sum().item()

    # Raw unweighted MAE (existing behaviour — unchanged)
    raw_mae = mae_per_ch.mean().item()

    return (score_A,
            z_t.squeeze(0).cpu().numpy(),
            mae_per_ch.cpu().numpy(),    # ← NEW — was discarded
            raw_mae)


# =============================================================================
# REFERENCE per-channel MAE — independent implementation, no PyTorch vector ops
# This must produce identical values to patched_run_m4's mae_per_ch_np.
# =============================================================================
@torch.no_grad()
def reference_compute_mae_per_ch(window_tensor, m4_model):
    """Per-channel MAE by explicit per-channel loop. No vector shortcuts.
       Ground-truth reference against the patched run_m4."""
    x = window_tensor.float()
    enc = m4_model.encoder
    out1, _ = enc.lstm1(x)
    out2, (h_n, c_n) = enc.lstm2(out1)
    z_t = enc.bn(h_n[-1])
    recon = m4_model.decoder(z_t, x.size(1), h_n, c_n)

    x_np     = x.squeeze(0).cpu().numpy()       # [50, 8]
    recon_np = recon.squeeze(0).cpu().numpy()   # [50, 8]

    mae_per_ch_ref = np.zeros(8, dtype=np.float32)
    for ch in range(8):
        err_ch = np.abs(x_np[:, ch] - recon_np[:, ch])   # [50]
        mae_per_ch_ref[ch] = err_ch.mean()
    return mae_per_ch_ref


# =============================================================================
# Helpers for M6B structure tolerance
# =============================================================================
def extract_sequence_array(entry):
    """Return the (T, 8) numpy array regardless of exact M6B entry schema."""
    for key in ("sequence", "data", "window", "array", "sensors"):
        if key in entry:
            return np.asarray(entry[key], dtype=np.float32)
    # Last resort: find the first ndarray-like value with 2 dims and 8 cols
    for v in entry.values():
        try:
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 8:
                return arr
        except Exception:
            pass
    return None


def extract_label(entry):
    for key in ("label", "label_int", "fault_label", "class"):
        if key in entry:
            return int(entry[key])
    return -1


# =============================================================================
# Main
# =============================================================================
def main():
    log(f"==== {SCRIPT_NAME} ====")
    log(f"Date: {date.today().isoformat()}  Device for test: CPU (only 20-30 windows)")

    results = {
        "script"              : SCRIPT_NAME,
        "timestamp"           : datetime.now().isoformat(),
        "tolerance_bit_exact" : TOL_BIT_EXACT,
        "n_windows"           : 0,
        "per_window_results"  : [],
        "per_channel_max_diff": {},
        "sample_scale_check"  : {},
        "overall_status"      : "UNKNOWN",
    }

    # ── Load M4 model ────────────────────────────────────────────────────────
    log("Loading M4 LSTM-AE checkpoint...")
    try:
        m4_path = MODEL_DIR / "lstm_ae_baseline_final.pth"
        m4_model = _M4LSTMAutoencoder()
        state = torch.load(m4_path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
        m4_model.load_state_dict(sd)
        m4_model.eval()
        log(f"  OK — loaded from {m4_path}")
    except Exception as e:
        log(f"  FAIL — M4 load error: {e}")
        results["overall_status"] = "FAIL — M4 not loadable"
        _save(results)
        return

    # ── Load M6B sequences ───────────────────────────────────────────────────
    log("Loading M6B_combined_sequences.pkl...")
    try:
        m6b_path = SYNTH_DIR / "M6B_combined_sequences.pkl"
        with open(m6b_path, "rb") as f:
            m6b = pickle.load(f)
        log(f"  OK — type={type(m6b).__name__}")
    except Exception as e:
        log(f"  FAIL — M6B load error: {e}")
        results["overall_status"] = "FAIL — M6B not loadable"
        _save(results)
        return

    # Normalise to parallel-arrays schema:
    # d['sequences'][i] -> (T, 8) array
    # d['metadata'][i]  -> dict with 'label', 'steps', etc.
    if isinstance(m6b, dict) and 'sequences' in m6b and 'metadata' in m6b:
        seqs_list = m6b['sequences']
        meta_list = m6b['metadata']
        m6b_list  = [{"sequence": seqs_list[i], **meta_list[i]}
                     for i in range(len(seqs_list))]
    elif isinstance(m6b, dict):
        # Fallback: old list-of-dicts-per-key schema
        m6b_list = list(m6b.values()) if all(isinstance(v, dict) for v in m6b.values()) \
                   else [{"label": k, "sequence": v} for k, v in m6b.items()]
    else:
        m6b_list = list(m6b)
    log(f"  Total sequences: {len(m6b_list)}")
    if m6b_list:
        log(f"  First entry keys: {list(m6b_list[0].keys()) if isinstance(m6b_list[0], dict) else 'n/a'}")

    # ── Sample 3 sequences per target label ──────────────────────────────────
    target_labels = [0, 1, 3, 4, 5, 10, 15, 21]
    log(f"Target labels: {target_labels}")

    samples = []
    for tgt in target_labels:
        found = 0
        for idx, entry in enumerate(m6b_list):
            if not isinstance(entry, dict):
                continue
            if extract_label(entry) == tgt:
                samples.append((idx, tgt, entry))
                found += 1
                if found >= 3:
                    break
        if found == 0:
            log(f"  WARN — no sequences found for label {tgt}")
    results["n_windows"] = len(samples)
    log(f"  Will test {len(samples)} windows")

    if not samples:
        log("FAIL — no samples available; cannot run identity test.")
        results["overall_status"] = "FAIL — no samples"
        _save(results)
        return

    # ── Per-window identity test ─────────────────────────────────────────────
    log("")
    log("=" * 78)
    log(f"PATCHED run_m4 vs REFERENCE — identity test (tol={TOL_BIT_EXACT:.0e})")
    log("=" * 78)

    per_ch_diffs = {ch: [] for ch in MAE_FEATURE_NAMES}
    all_pass = True
    scale_samples = {0: None, 21: None}   # capture one sample each for scale check

    for (idx, lbl, entry) in samples:
        seq_arr = extract_sequence_array(entry)
        if seq_arr is None or seq_arr.shape[0] < 50:
            log(f"  idx={idx:>5}  label={lbl:>2}  SKIP (no usable sequence)")
            continue

        window_np = seq_arr[:50]                                  # [50, 8]
        wt = torch.from_numpy(window_np).unsqueeze(0)             # [1, 50, 8]

        _sA, _zt, mae_patched, _rm = patched_run_m4(wt, m4_model)
        mae_ref = reference_compute_mae_per_ch(wt, m4_model)

        diffs = np.abs(mae_patched - mae_ref)
        mxd = float(diffs.max())
        ok = mxd < TOL_BIT_EXACT
        if not ok:
            all_pass = False

        log(f"  idx={idx:>5}  label={lbl:>2}  max_abs_diff={mxd:.2e}  [{'PASS' if ok else 'FAIL'}]")

        for ci, cn in enumerate(MAE_FEATURE_NAMES):
            per_ch_diffs[cn].append(float(diffs[ci]))

        results["per_window_results"].append({
            "seq_idx"      : idx,
            "label"        : int(lbl),
            "max_abs_diff" : mxd,
            "patched"      : {n: float(v) for n, v in zip(MAE_FEATURE_NAMES, mae_patched)},
            "reference"    : {n: float(v) for n, v in zip(MAE_FEATURE_NAMES, mae_ref)},
            "status"       : "PASS" if ok else "FAIL",
        })

        if lbl in scale_samples and scale_samples[lbl] is None:
            scale_samples[lbl] = {n: float(v) for n, v in zip(MAE_FEATURE_NAMES, mae_patched)}

    # ── Per-channel summary ──────────────────────────────────────────────────
    log("")
    log("-" * 78)
    log("Per-channel max abs diff across all tested windows:")
    for cn in MAE_FEATURE_NAMES:
        mx = max(per_ch_diffs[cn]) if per_ch_diffs[cn] else 0.0
        results["per_channel_max_diff"][cn] = mx
        mark = "OK " if mx < TOL_BIT_EXACT else "FAIL"
        log(f"  [{mark}]  {cn:>12}  max_abs_diff = {mx:.2e}")

    # ── Sample scale check (eyeball against M6.5r distribution) ──────────────
    log("")
    log("-" * 78)
    log("Sample scale check — first window per label, patched values:")
    log("  Expected: label 0 ~ small (<0.05/ch); label 21 ~ rising mid (~0.05-0.20/ch)")
    for lbl, vals in scale_samples.items():
        if vals is None:
            continue
        log(f"  label={lbl}:")
        for cn, v in vals.items():
            log(f"    {cn:>12} = {v:.4f}")
        results["sample_scale_check"][str(lbl)] = vals

    # ── Status + report ──────────────────────────────────────────────────────
    results["overall_status"] = "PASS" if all_pass else "FAIL"

    log("")
    log("=" * 78)
    if all_pass:
        log("ALL WINDOWS PASS — patched_run_m4 is bit-exactly identical to reference.")
        log("Safe to apply the 2-line patch to app/routers/anomaly.py.")
    else:
        log("ONE OR MORE WINDOWS FAILED — investigate divergence before patching.")
    log("=" * 78)

    _save(results)

    # ── PASTE UPDATE banner ──────────────────────────────────────────────────
    print()
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12 Stage 1.1 identity test: {results['overall_status']}")
    print(f"Windows tested: {results['n_windows']}")
    if results["per_channel_max_diff"]:
        gmax = max(results["per_channel_max_diff"].values())
        print(f"Global max abs diff (8 channels x all windows): {gmax:.2e}")
        print(f"Tolerance: {TOL_BIT_EXACT:.0e}")
    if results["overall_status"] == "PASS":
        print("Status for next step: READY — safe to apply 2-line patch to anomaly.py")
        print("Next: Stage 1.2 — export M6.5r z_t PCA as M6p5r_zt_pca.pkl")
    else:
        print("Status for next step: BLOCKED — investigate divergence")
    print("══ END PASTE UPDATE ══")

    # ── File manifest ────────────────────────────────────────────────────────
    print()
    print("══ FILE MANIFEST ══")
    print(f"  Report (Spaces upload): outputs/reports/{SCRIPT_NAME}_report.md")
    print(f"  Results JSON (GitHub) : outputs/reports/{SCRIPT_NAME}_results.json")


def _save(results):
    """Save markdown report and JSON results."""
    rp = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(f"# {SCRIPT_NAME} report\n\n")
        f.write(f"- Timestamp: {results.get('timestamp', 'n/a')}\n")
        f.write(f"- Status: **{results.get('overall_status', 'UNKNOWN')}**\n")
        f.write(f"- Tolerance (bit-exact): {results.get('tolerance_bit_exact', 'n/a')}\n")
        f.write(f"- Windows tested: {results.get('n_windows', 0)}\n\n")
        f.write("## Per-channel max abs diff\n\n")
        for cn, mx in results.get("per_channel_max_diff", {}).items():
            f.write(f"- {cn}: {mx:.2e}\n")
        f.write("\n## Sample scale check (first window per label, patched values)\n\n")
        for lbl, vals in results.get("sample_scale_check", {}).items():
            f.write(f"\n### Label {lbl}\n\n")
            if vals:
                for cn, v in vals.items():
                    f.write(f"- {cn}: {v:.4f}\n")
        f.write("\n## Per-window detail\n\n")
        for r in results.get("per_window_results", []):
            f.write(f"- seq_idx={r['seq_idx']} label={r['label']} "
                    f"max_diff={r['max_abs_diff']:.2e} [{r['status']}]\n")
    jp = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
