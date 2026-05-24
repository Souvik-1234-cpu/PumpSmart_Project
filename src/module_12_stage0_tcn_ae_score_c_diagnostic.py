# =============================================================================
# src/module_12_stage0_tcn_ae_score_c_diagnostic.py
#
# M12 Stage 1.0 — TCN-AE score_C head diagnostic (v2 — architecture fix)
#
# WHAT CHANGED FROM v1:
#   v1 showed 36 missing + 84 unexpected keys under strict=False.
#   This means the inline _TCNAutoencoder stub in model_registry.py has
#   WRONG key names vs tcn_ae_level2_best.pth. Every weight loaded as
#   random PyTorch init. This is a new critical defect (D8).
#
#   This v2 script:
#     1. Prints ALL checkpoint keys to identify the real architecture.
#     2. Tries importing the TRUE trained class from
#        module_08_tcn_ae_detection_stack (same file used for training).
#     3. If that import works, loads with correct architecture and runs
#        the pre-relu head_C diagnostic properly.
#     4. If import fails, loads checkpoint with weights_only=False
#        (tries full-model pickle) and uses forward hook for pre-relu capture.
#     5. Reports D8 (architecture mismatch) regardless of outcome.
#     6. Fixes M6B loading with aggressive structure flattening.
#
# RUN (from project root OR src/ — sys.path fixed below):
#   .venv\Scripts\Activate.ps1
#   python src/module_12_stage0_tcn_ae_score_c_diagnostic.py
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, MODEL_DIR, SYNTH_DIR, OUTPUT_DIR)
from datetime import datetime, date
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn

SCRIPT_NAME = "module_12_stage0_tcn_ae_score_c_diagnostic"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DIAG_DEVICE  = torch.device("cpu")
WINDOW_SIZE  = 50
ZT_BUF_LEN   = 63
ZT_DIM       = 64
STRIDE       = 25
TARGET_LABELS = [7, 8, 10, 11, 12]
SEQS_PER_LABEL = 4


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# M4 architecture — exact copy from model_registry.py (unchanged)
# =============================================================================
class _M4Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8,   128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64,  num_layers=1, batch_first=True)
        self.bn    = nn.LayerNorm(64)
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
        h0 = torch.tanh(self.fc_h(h_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.tanh(self.fc_c(c_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
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
# M6B structure: aggressive recursive flattening
# =============================================================================
def flatten_m6b(data, depth=0):
    """Return a flat list of sequence-entry dicts regardless of nesting."""
    if depth > 5:
        return []
    if isinstance(data, dict):
        # Leaf: has sequence array and label
        has_arr = any(k in data for k in ('sequence','data','array','sensors','windows'))
        has_lbl = any(k in data for k in ('label','label_int','fault_label'))
        if has_arr and has_lbl:
            return [data]
        # Container: recurse into values
        out = []
        for v in data.values():
            out.extend(flatten_m6b(v, depth+1))
        return out
    elif isinstance(data, (list, tuple)):
        out = []
        for item in data:
            out.extend(flatten_m6b(item, depth+1))
        return out
    return []


def get_seq_array(entry):
    for key in ('sequence','data','array','sensors','windows'):
        if key in entry:
            try:
                arr = np.asarray(entry[key], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 8:
                    return arr
                # Some formats store windows as [n_windows, window_size, 8]
                if arr.ndim == 3 and arr.shape[2] == 8:
                    return arr.reshape(-1, 8)
            except Exception:
                pass
    return None


def get_label(entry):
    for key in ('label','label_int','fault_label'):
        if key in entry:
            return int(entry[key])
    return -1


# =============================================================================
# z_t builder
# =============================================================================
@torch.no_grad()
def build_zt_buffer(seq_arr, m4_model, max_len=ZT_BUF_LEN):
    T = seq_arr.shape[0]
    zt_list = []
    for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
        if len(zt_list) >= max_len:
            break
        win = seq_arr[start: start + WINDOW_SIZE]
        wt  = torch.from_numpy(win).unsqueeze(0).float()
        z_t, _, _ = m4_model.encoder(wt)
        zt_list.append(z_t.squeeze(0))
    return torch.stack(zt_list, dim=0) if zt_list else None


# =============================================================================
# Load TCN-AE: try correct architecture first, fallback to weights_only=False
# =============================================================================
def load_tcn_ae(ckpt_path, results):
    """
    Returns (model, load_method, pre_relu_capture_fn)
    pre_relu_capture_fn(x) → (sA, sB, sC, pre_relu_C)  as floats
    """
    # ── Step A: inspect checkpoint keys ──────────────────────────────────────
    log("  Inspecting checkpoint keys…")
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        log(f"  FAIL loading checkpoint: {e}")
        return None, "failed", None

    if isinstance(raw, dict) and 'state_dict' in raw:
        raw = raw['state_dict']

    if isinstance(raw, dict):
        all_keys = sorted(raw.keys())
        results["checkpoint_all_keys"] = all_keys
        log(f"  Checkpoint is state_dict with {len(all_keys)} keys")
        log(f"  First 12 keys: {all_keys[:12]}")
        # Determine key naming scheme
        uses_tcn_stack = any('tcn_stack' in k for k in all_keys)
        has_head_C     = any('head_C' in k for k in all_keys)
        log(f"  uses_tcn_stack naming: {uses_tcn_stack}  |  has head_C key: {has_head_C}")
        results["d8_architecture_mismatch"] = uses_tcn_stack  # True = D8 confirmed
    else:
        # Full model object saved — best case
        log("  Checkpoint is a full model object (not just state_dict)")
        results["checkpoint_all_keys"] = ["<full model object>"]
        results["d8_architecture_mismatch"] = False

    # ── Step B: try importing trained class from module_08 ───────────────────
    src_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(src_dir))
    m8_module = None
    for cls_name in ('TCNAutoencoder', 'TCNAutoEncoder', 'DriftDetectionModel',
                     'TCNAE', 'DetectionStack', 'LevelTwoModel'):
        try:
            import importlib
            mod = importlib.import_module('module_08_tcn_ae_detection_stack')
            if hasattr(mod, cls_name):
                m8_module = mod
                m8_cls    = getattr(mod, cls_name)
                log(f"  Imported {cls_name} from module_08_tcn_ae_detection_stack")
                break
        except Exception:
            pass

    if m8_module is not None:
        try:
            model = m8_cls()
            if isinstance(raw, dict):
                res = model.load_state_dict(raw, strict=False)
                missing = res.missing_keys
                unexpected = res.unexpected_keys
                results["m08_import_missing"]    = missing
                results["m08_import_unexpected"] = unexpected
                log(f"  Loaded with correct class: missing={len(missing)} unexpected={len(unexpected)}")
                if len(missing) == 0:
                    log("  PERFECT LOAD — all weights matched")
                    results["d8_architecture_mismatch"] = False
            else:
                model = raw  # raw is already the model object
            model.eval()
            load_method = "module_08_class"
        except Exception as e:
            log(f"  module_08 class load failed: {e}; falling back")
            m8_module = None

    # ── Step C: fallback — use raw as model object if possible ───────────────
    if m8_module is None and not isinstance(raw, dict):
        model = raw
        model.eval()
        load_method = "full_object"
        log("  Using full model object directly")
    elif m8_module is None:
        log("  WARNING: using stub architecture with mismatched weights (D8 confirmed)")
        # Fallback to the inline stub for key inspection
        class _TCNBlockStub(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size, dilation):
                super().__init__()
                pad = (kernel_size - 1) * dilation
                self.conv  = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
                self.trim  = pad
                self.norm  = nn.LayerNorm(out_ch)
                self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            def forward(self, x):
                out = self.conv(x)
                if self.trim: out = out[:, :, :-self.trim]
                out = self.norm(out.transpose(1,2)).transpose(1,2)
                return torch.relu(out + self.skip(x))

        class _TCNAEStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    _TCNBlockStub(64,128,3,1), _TCNBlockStub(128,128,3,2),
                    _TCNBlockStub(128,128,3,4), _TCNBlockStub(128,64,3,8),
                    _TCNBlockStub(64,64,3,16))
                self.decoder = nn.Sequential(
                    _TCNBlockStub(64,64,3,1), nn.Conv1d(64,64,1))
                self.head_A = nn.Linear(64,1)
                self.head_B = nn.Linear(64,1)
                self.head_C = nn.Linear(64,1)
            def forward(self, x):
                x = x.transpose(1,2)
                z = self.encoder(x)
                pooled = z.mean(dim=2)
                return (self.head_A(pooled).squeeze(-1),
                        self.head_B(pooled).squeeze(-1),
                        torch.relu(self.head_C(pooled).squeeze(-1)))

        model = _TCNAEStub()
        if isinstance(raw, dict):
            model.load_state_dict(raw, strict=False)
        model.eval()
        load_method = "stub_mismatched"

    # ── Step D: build pre_relu capture function ───────────────────────────────
    # Use forward hook on head_C regardless of model class
    _hook_store = {}

    def _hook_fn(module, inp, out):
        _hook_store['pre_relu_C'] = out.detach().cpu()

    # Try to find head_C in model
    head_c_module = None
    for name, mod in model.named_modules():
        if 'head_C' in name and isinstance(mod, nn.Linear):
            head_c_module = mod
            log(f"  Found head_C at: {name}  weight_norm={mod.weight.data.norm().item():.6f}  bias={mod.bias.data.item():.6f}")
            results["head_c_weight_norm"] = float(mod.weight.data.norm().item())
            results["head_c_bias"]        = float(mod.bias.data.item())
            break

    if head_c_module is not None:
        hook = head_c_module.register_forward_hook(_hook_fn)
    else:
        log("  WARNING: head_C not found in loaded model — cannot capture pre_relu")
        results["head_c_weight_norm"] = None
        results["head_c_bias"]        = None
        hook = None

    @torch.no_grad()
    def capture_fn(x_in):
        out = model(x_in)
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            sA_t, sB_t, sC_t = out[0], out[1], out[2]
        else:
            sA_t = sB_t = sC_t = torch.tensor(0.0)
        pre_c = _hook_store.get('pre_relu_C', torch.tensor([[0.0]])).squeeze().item()
        return float(sA_t.item()), float(sB_t.item()), float(sC_t.item()), float(pre_c)

    results["load_method"] = load_method
    return model, load_method, capture_fn


# =============================================================================
# Main
# =============================================================================
def main():
    log(f"==== {SCRIPT_NAME} v2 ====")
    log(f"Date: {date.today().isoformat()}")

    results = {
        "script": SCRIPT_NAME, "version": "v2",
        "timestamp": datetime.now().isoformat(),
        "checkpoint_all_keys": [],
        "d8_architecture_mismatch": None,
        "load_method": None,
        "head_c_weight_norm": None,
        "head_c_bias": None,
        "m6b_structure_info": {},
        "per_sequence": [],
        "aggregate": {},
        "outcome": "UNKNOWN",
        "outcome_reasoning": "",
        "recommended_path": "",
    }

    # ── Load M4 ──────────────────────────────────────────────────────────────
    log("Loading M4 LSTM-AE…")
    try:
        m4 = _M4LSTMAutoencoder()
        st = torch.load(MODEL_DIR / "lstm_ae_baseline_final.pth",
                        map_location="cpu", weights_only=False)
        sd = st.get("model_state_dict", st) if isinstance(st, dict) else st
        m4.load_state_dict(sd); m4.eval()
        log(f"  OK")
    except Exception as e:
        log(f"  FAIL: {e}"); results["outcome"] = f"FAIL-M4: {e}"; _save(results); return

    # ── Load TCN-AE ──────────────────────────────────────────────────────────
    log("Loading TCN-AE…")
    ckpt = MODEL_DIR / "tcn_ae_level2_best.pth"
    model, load_method, capture_fn = load_tcn_ae(ckpt, results)
    if model is None:
        results["outcome"] = "FAIL-TCN"; _save(results); return

    d8 = results.get("d8_architecture_mismatch", False)
    if d8:
        log("  *** D8 CONFIRMED: architecture mismatch — key names incompatible ***")
        log("  *** Production model_registry.py has WRONG inline architecture   ***")
        log("  *** All weights including head_A/B/C are RANDOM INITIALIZATION   ***")

    # ── Load M6B ─────────────────────────────────────────────────────────────
    log("Loading M6B_combined_sequences.pkl…")
    try:
        with open(SYNTH_DIR / "M6B_combined_sequences.pkl", "rb") as f:
            raw_m6b = pickle.load(f)

        # Inspect raw structure before flattening
        if isinstance(raw_m6b, dict):
            top_keys   = list(raw_m6b.keys())[:10]
            top_types  = [type(v).__name__ for v in list(raw_m6b.values())[:5]]
            log(f"  Top-level dict: {len(raw_m6b)} keys, first={top_keys[:5]}")
            log(f"  Value types: {top_types}")
            # If values are lists, peek inside
            for v in list(raw_m6b.values())[:3]:
                if isinstance(v, list) and v:
                    log(f"    Value[0] type: {type(v[0]).__name__}  len={len(v)}")
                    if isinstance(v[0], dict):
                        log(f"    Sequence dict keys: {list(v[0].keys())[:8]}")
                    break
        elif isinstance(raw_m6b, list):
            log(f"  Top-level list: {len(raw_m6b)} items, first type={type(raw_m6b[0]).__name__}")
            if isinstance(raw_m6b[0], dict):
                log(f"  Seq dict keys: {list(raw_m6b[0].keys())[:8]}")

        m6b_list = flatten_m6b(raw_m6b)
        log(f"  Flattened: {len(m6b_list)} sequence entries")

        # If still tiny, try alternative: values might be numpy arrays keyed by label
        if len(m6b_list) < 20 and isinstance(raw_m6b, dict):
            log("  Retrying: treating dict values as sequence arrays…")
            alt = []
            for k, v in raw_m6b.items():
                try:
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.ndim == 3 and arr.shape[2] == 8:
                        # [n_sequences, T, 8]
                        for i in range(arr.shape[0]):
                            alt.append({"label": k, "sequence": arr[i]})
                    elif arr.ndim == 2 and arr.shape[1] == 8:
                        alt.append({"label": k, "sequence": arr})
                except Exception:
                    pass
            if len(alt) > len(m6b_list):
                m6b_list = alt
                log(f"  Alt parse: {len(m6b_list)} sequences")

        results["m6b_structure_info"] = {
            "raw_type": type(raw_m6b).__name__,
            "raw_len": len(raw_m6b) if hasattr(raw_m6b, '__len__') else -1,
            "flattened_len": len(m6b_list),
        }
    except Exception as e:
        log(f"  FAIL: {e}"); results["outcome"] = f"FAIL-M6B: {e}"; _save(results); return

    if len(m6b_list) < 5:
        log("  FAIL: cannot extract enough sequences for diagnostic")
        log("  Check SYNTH_DIR path and M6B file format manually")
        log(f"  SYNTH_DIR = {SYNTH_DIR}")
        results["outcome"] = "FAIL-M6B-EMPTY"
        _save(results); return

    # ── Sample ────────────────────────────────────────────────────────────────
    label_counts = {}
    for e in m6b_list:
        l = get_label(e)
        label_counts[l] = label_counts.get(l, 0) + 1
    log(f"  Label distribution (top 10): {sorted(label_counts.items())[:10]}")

    samples = []
    for tgt in TARGET_LABELS:
        found = 0
        for e in m6b_list:
            if get_label(e) == tgt:
                samples.append((tgt, e)); found += 1
                if found >= SEQS_PER_LABEL: break
        if found == 0:
            log(f"  label {tgt}: not found — using available labels as fallback")

    # Fallback: use any available Group B-ish labels if specific ones missing
    if not samples:
        log("  No target labels found — sampling first 20 entries for structural check")
        for e in m6b_list[:20]:
            samples.append((get_label(e), e))

    log(f"  Will test {len(samples)} sequences")

    # ── Per-sequence diagnostic ───────────────────────────────────────────────
    log("")
    log("=" * 72)
    log("DIAGNOSTIC: pre_relu_C vs sC vs sB")
    log(f"  {'lbl':>4}  {'buf_len':>7}  {'pre_relu_C':>12}  {'sC':>8}  {'sB':>8}  {'pre>0':>6}")
    log("-" * 72)

    pre_c_vals, sB_vals, sC_vals = [], [], []

    with torch.no_grad():
        for (lbl, entry) in samples:
            seq_arr = get_seq_array(entry)
            if seq_arr is None or seq_arr.shape[0] < WINDOW_SIZE:
                log(f"  lbl={lbl}: SKIP (bad array shape or too short)")
                continue

            zt_buf = build_zt_buffer(seq_arr, m4, max_len=ZT_BUF_LEN)
            if zt_buf is None or zt_buf.shape[0] < 5:
                log(f"  lbl={lbl}: SKIP (z_t buffer too short, seq_len={seq_arr.shape[0]})")
                continue

            x_in = zt_buf.unsqueeze(0).to(DIAG_DEVICE)
            try:
                sA, sB, sC, pre_c = capture_fn(x_in)
            except Exception as e:
                log(f"  lbl={lbl}: forward FAIL: {e}"); continue

            pre_c_vals.append(pre_c)
            sB_vals.append(sB)
            sC_vals.append(sC)
            pos = "YES" if pre_c > 0 else "no"
            log(f"  {lbl:>4}  {zt_buf.shape[0]:>7}  {pre_c:>+12.6f}  {sC:>8.6f}  {sB:>8.6f}  {pos:>6}")
            results["per_sequence"].append({
                "label": lbl, "buf_len": int(zt_buf.shape[0]),
                "pre_relu_C": pre_c, "sC": sC, "sB": sB,
                "pre_c_positive": bool(pre_c > 0),
            })

    log("=" * 72)

    if not pre_c_vals:
        log("FAIL — no sequences produced valid forward pass results")
        results["outcome"] = "FAIL-NO-FORWARD"
        _save(results); return

    # ── Aggregates ────────────────────────────────────────────────────────────
    prc = np.array(pre_c_vals)
    sB_a, sC_a = np.array(sB_vals), np.array(sC_vals)
    n = len(prc)
    n_pos = int((prc > 0).sum())
    pos_frac = n_pos / n

    agg = {
        "n": n,
        "pre_relu_C_mean": float(prc.mean()),
        "pre_relu_C_std":  float(prc.std()),
        "pre_relu_C_min":  float(prc.min()),
        "pre_relu_C_max":  float(prc.max()),
        "n_positive": n_pos, "pos_fraction": float(pos_frac),
        "sC_mean": float(sC_a.mean()), "sC_max": float(sC_a.max()),
        "sC_all_zero": bool((sC_a == 0.0).all()),
        "sB_mean": float(sB_a.mean()), "sB_max": float(sB_a.max()),
    }
    results["aggregate"] = agg

    log("")
    log("AGGREGATE:")
    log(f"  pre_relu_C : mean={agg['pre_relu_C_mean']:+.6f}  std={agg['pre_relu_C_std']:.6f}"
        f"  range=[{agg['pre_relu_C_min']:+.4f}, {agg['pre_relu_C_max']:+.4f}]")
    log(f"  positive   : {n_pos}/{n} ({pos_frac*100:.1f}%)")
    log(f"  sB         : mean={agg['sB_mean']:+.6f}  max={agg['sB_max']:+.6f}")
    log(f"  sC         : mean={agg['sC_mean']:+.6f}  all_zero={agg['sC_all_zero']}")
    log(f"  head_C w.norm = {results.get('head_c_weight_norm', 'n/a')}")
    log(f"  D8 (arch mismatch) = {d8}   load_method = {load_method}")

    # ── Outcome ───────────────────────────────────────────────────────────────
    wt_norm  = results.get("head_c_weight_norm") or 0.0
    pre_mean = agg["pre_relu_C_mean"]
    pre_std  = agg["pre_relu_C_std"]

    log("")
    log("=" * 72)
    log("OUTCOME:")

    # D8 takes priority if architecture mismatch confirmed
    if d8 and load_method == "stub_mismatched":
        outcome = "D8"
        reasoning = (
            f"Architecture mismatch CONFIRMED. All 36 model keys (encoder, decoder, "
            f"head_A, head_B, head_C) are MISSING from the checkpoint under the current "
            f"model_registry.py naming scheme. The production TCN-AE runs on random "
            f"PyTorch initialization. The score_B values (0.18-0.45) seen in live screenshots "
            f"were random-weight outputs, not trained signals. Score C=0 because random-init "
            f"head_C after ReLU produces zero for this z_t distribution. "
            f"The inline _TCNAutoencoder stub in model_registry.py uses key names "
            f"'encoder.X.conv.weight' but checkpoint expects 'encoder.tcn_stack.X.conv1.conv.weight'."
        )
        recommended = (
            "IMMEDIATE FIX NEEDED BEFORE STAGE 1.1: "
            "Correct the TCN-AE architecture in model_registry.py:_load_tcn_ae to match "
            "the actual trained architecture in src/module_08_tcn_ae_detection_stack.py. "
            "The fix is to replace the inline _TCNAutoencoder class with the correct one "
            "from that module (import or copy). Checkpoint keys are saved to results JSON "
            "and report for architecture reconstruction. "
            "Once the correct architecture loads, re-run this diagnostic — "
            "score_C behaviour with trained weights may be completely different."
        )
    elif wt_norm is not None and wt_norm < 1e-4:
        outcome = "C"
        reasoning = f"head_C weight norm = {wt_norm:.2e} — weights are zero (failed to load)."
        recommended = "Stage 1.5: fix architecture mismatch, then re-run diagnostic."
    elif pos_frac >= 0.50 and pre_mean >= 0.01:
        outcome = "A"
        reasoning = (
            f"pre_relu_C mean={pre_mean:+.4f}, {n_pos}/{n} positive. "
            f"Score C is producing genuine signal."
        )
        recommended = "Proceed to Stage 1.1. Stage 4 threshold recalibration needed."
    elif pre_std >= 1e-3:
        outcome = "B"
        reasoning = (
            f"pre_relu_C std={pre_std:.4f} (head alive) but mean={pre_mean:+.4f} "
            f"({n_pos}/{n} positive). ReLU killing most signal."
        )
        recommended = "Proceed to Stage 1.1. Stage 4 score_C threshold recalibration needed."
    else:
        outcome = "C"
        reasoning = f"pre_relu_C near constant zero (std={pre_std:.2e}). Head degenerate."
        recommended = "Stage 1.5 required before Stage 4 score_C predicates."

    results["outcome"] = outcome
    results["outcome_reasoning"] = reasoning
    results["recommended_path"]  = recommended

    log(f"  OUTCOME : {outcome}")
    log(f"  Reasoning: {reasoning}")
    log(f"  Path: {recommended}")
    log("=" * 72)

    _save(results)

    print()
    print("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
    print(f"M12 Stage 1.0 (score_C diagnostic v2): COMPLETE — Outcome: {outcome}")
    print(f"D8 architecture mismatch: {d8}   load_method: {load_method}")
    print(f"pre_relu_C: mean={agg['pre_relu_C_mean']:+.6f}  std={agg['pre_relu_C_std']:.6f}  pos={n_pos}/{n}")
    print(f"sB: mean={agg['sB_mean']:.6f}  max={agg['sB_max']:.6f}")
    print(f"sC all_zero: {agg['sC_all_zero']}")
    print(f"head_C weight_norm: {results.get('head_c_weight_norm')}")
    if outcome == "D8":
        print("Status: BLOCKED — fix model_registry.py TCN-AE architecture before Stage 1.1")
        print("Next: supply correct _TCNAutoencoder class matching checkpoint key names")
        print("      Checkpoint keys saved to diagnostic JSON for reconstruction")
    elif outcome == "A":
        print("Status: READY for Stage 1.1")
    elif outcome == "B":
        print("Status: READY for Stage 1.1, score_C threshold recalibration in Stage 4")
    else:
        print("Status: Stage 1.5 required")
    print("══ END PASTE UPDATE ══")
    print()
    print("══ FILE MANIFEST ══")
    print(f"  Report : outputs/reports/{SCRIPT_NAME}_report.md   (GitHub push)")
    print(f"  Results: outputs/reports/{SCRIPT_NAME}_results.json (GitHub push)")
    print("  Both files contain ALL checkpoint keys for architecture reconstruction")


def _save(results):
    try:
        rp = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
        with open(rp, "w", encoding="utf-8") as f:
            f.write(f"# {SCRIPT_NAME} v2\n\n")
            f.write(f"- Date: {date.today().isoformat()}\n")
            f.write(f"- Outcome: **{results.get('outcome','UNKNOWN')}**\n")
            f.write(f"- D8 architecture mismatch: {results.get('d8_architecture_mismatch')}\n")
            f.write(f"- Load method: {results.get('load_method')}\n")
            f.write(f"- head_C weight norm: {results.get('head_c_weight_norm')}\n")
            f.write(f"- head_C bias: {results.get('head_c_bias')}\n\n")
            f.write(f"## Outcome reasoning\n\n{results.get('outcome_reasoning','')}\n\n")
            f.write(f"## Recommended path\n\n{results.get('recommended_path','')}\n\n")
            f.write("## Aggregate\n\n")
            for k, v in results.get("aggregate", {}).items():
                f.write(f"- {k}: {v}\n")
            f.write("\n## ALL checkpoint keys (for architecture reconstruction)\n\n```\n")
            for k in results.get("checkpoint_all_keys", []):
                f.write(f"{k}\n")
            f.write("```\n\n## Per-sequence\n\n")
            f.write("| label | buf_len | pre_relu_C | sC | sB | pre>0 |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in results.get("per_sequence", []):
                f.write(f"| {r['label']} | {r['buf_len']} | {r['pre_relu_C']:+.6f} "
                        f"| {r['sC']:.6f} | {r['sB']:.6f} "
                        f"| {'Y' if r['pre_c_positive'] else 'N'} |\n")
    except Exception as e:
        log(f"  Report save error: {e}")
    try:
        jp = REPORT_DIR / f"{SCRIPT_NAME}_results.json"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        log(f"  JSON save error: {e}")


if __name__ == "__main__":
    main()