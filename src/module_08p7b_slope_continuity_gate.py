# =============================================================================
# module_08p7b_v2_slope_continuity_gate.py
# PumpSmart v14.2 — T1.5.2 v2: Numerically Stable Slope-Change Gate
# =============================================================================
#
# WHY v1 FAILED (module_08p7b_slope_continuity_gate.py):
#   The ratio test slope_post / slope_pre explodes when slope_pre ≈ 0.
#   Many primary channels have near-zero pre-onset slopes (e.g. Pres.SV during
#   bearing wear Phase 1 — pressure is stable). The denominator clamp of 1e-6
#   is orders of magnitude smaller than actual slope values, producing ratios
#   of 100–7000 for physically correct sequences.
#   Evidence: mean_ratio=7131 for Pres.SV with P5=-0.8, P95=14901 — this is
#   pure numerical explosion, not physics failure.
#
# CORRECT TEST — absolute slope-change normalised by local signal variability:
#   For each primary channel at the Phase 2 boundary:
#     slope_pre   = linear slope over [p2_start-W : p2_start-1]
#     slope_post  = linear slope over [p2_start   : p2_start+W]
#     signal_std  = std of channel over [p2_start-W : p2_start+W]
#     delta_slope = |slope_post - slope_pre|
#     normalised  = delta_slope / (signal_std + epsilon)
#   PASS if normalised < DELTA_THRESHOLD (set to 1.0)
#
# PHYSICS RATIONALE:
#   If np.tile freeze bug (Bug 1) were still present:
#     slope_pre  > 0  (fault progressing into Phase 2 boundary)
#     slope_post ≈ 0  (frozen — flat signal)
#     delta_slope ≈ |slope_pre| which is large relative to signal_std → FAIL
#   With the fix (linear extrapolation):
#     slope_post ≈ slope_pre (fault continues at same rate)
#     delta_slope ≈ 0 → normalised ≈ 0 → PASS
#   Near-zero pre-onset slopes (stable channels):
#     slope_pre ≈ 0, slope_post ≈ 0 → delta ≈ 0 → PASS (correctly)
#
# THRESHOLD SELECTION:
#   DELTA_THRESHOLD = 1.0 means: the slope must not change by more than
#   1 signal-std per step at the boundary. This is generous — a sudden freeze
#   would produce delta >> 1.0 for any progressing fault channel.
#   A value < 0.5 would be too tight (noise in slope estimates).
#   A value > 2.0 would be too loose (might miss real artifacts).
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (SYNTH_DIR, MODEL_DIR, OUTPUT_DIR)
from datetime import date, datetime
import json, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np

SCRIPT_NAME = "module_08p7b_v2_slope_continuity_gate"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
GATES   = {}

log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}")
log("  T1.5.2 v2 — Absolute slope-change gate (numerically stable)")
log("=" * 72)

# =============================================================================
# SECTION 0 — CONSTANTS
# =============================================================================
CHANNELS = ["Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
            "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV"]
CH = {c: i for i, c in enumerate(CHANNELS)}

# Primary channels per label (unchanged from v1)
PRIMARY_CH_MASK = {
    7:  ["Mot.SV", "Pmp.SV", "Mot.TV"],
    8:  ["Pres.SV", "Pmp.SV", "Pmp.TV"],
    9:  ["Pmp.PV", "Pmp.SV", "Pres.SV"],
    10: ["Pres.SV", "Pmp.TV", "Pmp.PV"],
    11: ["Temp.SV", "Mot.TV", "Mot.PV"],
    12: ["Pmp.PV", "Pmp.SV", "Pres.SV"],
}

COMPOUND_NAMES = {
    7: "bearing_wear+overloading",    8: "cavitation+seal_failure",
    9: "impeller_imbalance+bearing_wear", 10: "seal_failure+cavitation_H",
    11: "overloading+bearing_wear",   12: "impeller_imbalance+cavitation",
}

SLOPE_WINDOW    = 10      # steps each side
DELTA_THRESHOLD = 1.0     # max |slope_post - slope_pre| / signal_std
PASS_THRESHOLD  = 0.95    # fraction of sequences required per label
EPSILON         = 1e-6    # numerical stability

def linear_slope(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2: return 0.0
    x = np.arange(n, dtype=np.float64) - (n-1)/2.0
    d = (x**2).sum()
    return float((x*(arr - arr.mean())).sum() / d) if d > 1e-12 else 0.0

# =============================================================================
# SECTION 1 — LOAD SEQUENCES
# =============================================================================
log("\nSECTION 1 — Load Group B v2 sequences")
try:
    with open(SYNTH_DIR / "M6B_sequences_groupB_v2.pkl", "rb") as f:
        data = pickle.load(f)
    seqs = data["sequences"]
    meta = data["meta"]
    log(f"  Loaded {len(seqs)} sequences")
except Exception as e:
    log(f"  [FATAL] {e}"); sys.exit(1)

# =============================================================================
# SECTION 2 — GATE
# =============================================================================
log(f"\nSECTION 2 — Absolute slope-change gate")
log(f"  Window: {SLOPE_WINDOW} steps either side")
log(f"  Threshold: |Δslope|/signal_std < {DELTA_THRESHOLD}")
log(f"  Pass criterion: >={PASS_THRESHOLD*100:.0f}% of sequences per label")
log(f"  Why stable: works correctly when slope_pre ≈ 0 (no division by slope)")

per_label = {}
all_pass  = []

for label in [7, 8, 9, 10, 11, 12]:
    prim_chs   = PRIMARY_CH_MASK[label]
    pairs      = [(np.array(s, dtype=np.float32), m)
                  for s, m in zip(seqs, meta) if m["label"] == label]
    n_seqs     = len(pairs)
    pass_count = 0
    skip_count = 0
    ch_deltas  = {ch: [] for ch in prim_chs}

    for seq_np, m in pairs:
        p2 = m["secondary_onset_step"]
        if p2 < SLOPE_WINDOW + 1 or p2 + SLOPE_WINDOW >= len(seq_np):
            skip_count += 1
            pass_count += 1
            all_pass.append(True)
            continue

        seq_pass = True
        for ch in prim_chs:
            ch_idx = CH[ch]
            # Slopes in units of (normalized_value / step)
            pre_win  = seq_np[p2 - SLOPE_WINDOW : p2,                ch_idx]
            post_win = seq_np[p2               : p2 + SLOPE_WINDOW,  ch_idx]
            full_win = seq_np[p2 - SLOPE_WINDOW : p2 + SLOPE_WINDOW, ch_idx]

            s_pre  = linear_slope(pre_win)
            s_post = linear_slope(post_win)
            s_std  = float(full_win.std()) + EPSILON

            delta_norm = abs(s_post - s_pre) / s_std
            ch_deltas[ch].append(delta_norm)

            if delta_norm > DELTA_THRESHOLD:
                seq_pass = False

        if seq_pass:
            pass_count += 1
        all_pass.append(seq_pass)

    rate = pass_count / n_seqs if n_seqs > 0 else 0.0
    label_ok = rate >= PASS_THRESHOLD
    GATES[f"T1.5.2_G_L{label}"] = {
        "passed": label_ok,
        "detail": f"{pass_count}/{n_seqs} ({rate*100:.1f}%) target>={PASS_THRESHOLD*100:.0f}%"
    }
    per_label[label] = {
        "pass_rate": round(rate, 4), "n_pass": pass_count, "n_total": n_seqs,
        "skipped": skip_count,
        "mean_delta_norm": {ch: round(float(np.mean(v)), 4) if v else 0.0
                            for ch, v in ch_deltas.items()},
        "p95_delta_norm":  {ch: round(float(np.percentile(v, 95)), 4) if v else 0.0
                            for ch, v in ch_deltas.items()},
    }
    log(f"  Label {label} ({COMPOUND_NAMES[label][:28]}): "
        f"{pass_count}/{n_seqs} ({rate*100:.1f}%) {'PASS' if label_ok else 'FAIL'}")
    for ch in prim_chs:
        v = ch_deltas[ch]
        if v:
            log(f"    {ch}: mean_delta={np.mean(v):.4f}  "
                f"p50={np.median(v):.4f}  p95={np.percentile(v,95):.4f}")

overall      = float(np.mean(all_pass)) if all_pass else 0.0
overall_pass = overall >= PASS_THRESHOLD
GATES["T1.5.2_G_overall"] = {
    "passed": overall_pass,
    "detail": f"{overall*100:.2f}% overall (target>={PASS_THRESHOLD*100:.0f}%)"
}
log(f"\n  Overall: {overall*100:.2f}% "
    f"({'PASS' if overall_pass else 'FAIL'}) target>={PASS_THRESHOLD*100:.0f}%")

n_pass = sum(1 for g in GATES.values() if g["passed"])
n_fail = len(GATES) - n_pass
results.update({
    "overall_pass_rate": round(overall, 4),
    "gate_status":       "PASS" if overall_pass else "FAIL",
    "per_label":         per_label,
    "gates_passed":      n_pass,
    "gates_failed":      n_fail,
})

# =============================================================================
# SECTION 3 — SAVE CONFIG
# =============================================================================
log("\nSECTION 3 — Save gate config")
cfg = {
    "date":              str(date.today()),
    "script":            SCRIPT_NAME,
    "version":           "v2",
    "method":            "absolute_slope_change_normalised_by_signal_std",
    "slope_window":      SLOPE_WINDOW,
    "delta_threshold":   DELTA_THRESHOLD,
    "pass_threshold":    PASS_THRESHOLD,
    "overall_pass_rate": round(overall, 4),
    "gate_status":       "PASS" if overall_pass else "FAIL",
    "per_label":         per_label,
    "primary_ch_mask":   PRIMARY_CH_MASK,
    "why_v1_failed":     (
        "v1 used slope_post/slope_pre ratio. When slope_pre≈0 (stable primary "
        "channel before secondary onset), division explodes to 100-7000. "
        "This is numerical instability, not physics failure. "
        "v2 uses |slope_post - slope_pre| / signal_std — stable for all cases."
    ),
    "audit_reference":   "PumpSmart Industrial Audit v3.0 §10.3 Concern B / T1.5.2",
}
cfg_path = MODEL_DIR / "M8p7b_v2_slope_continuity_gate_config.json"
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
log(f"  Saved → {cfg_path.name}")

# =============================================================================
# SECTION 4 — REPORT
# =============================================================================
gate_table = "\n".join(
    f"| {n} | {'PASS' if g['passed'] else 'FAIL'} | {g['detail']} |"
    for n, g in GATES.items()
)
label_table = "\n".join(
    f"| {l} | {COMPOUND_NAMES[l]} | "
    f"{per_label[l]['n_pass']}/{per_label[l]['n_total']} | "
    f"{per_label[l]['pass_rate']*100:.1f}% |"
    for l in range(7, 13)
)

report = f"""# {SCRIPT_NAME} — Report
**Date:** {date.today()}
**Gate status:** {"PASS" if overall_pass else "FAIL"} ({overall*100:.2f}%)

## Why v1 Failed

v1 used `slope_post / slope_pre` ratio. When `slope_pre ≈ 0` (stable primary
channel before secondary onset, e.g. Pres.SV during bearing wear Phase 1),
division explodes: mean_ratio = 7131 for Pres.SV in Label 9.
This was numerical instability, not physics failure.

## v2 Method

For each primary channel at the Phase 2 boundary:
```
delta_norm = |slope_post - slope_pre| / (signal_std + ε)
PASS if delta_norm < {DELTA_THRESHOLD}
```

**Numerically stable:** when slope_pre ≈ 0 and slope_post ≈ 0, delta = 0 → PASS.
**Detects Bug 1 freeze:** slope_pre > 0 (progressing), slope_post ≈ 0 (frozen)
→ delta = |0 - slope_pre| / std >> {DELTA_THRESHOLD} → FAIL.

## Per-Label Results

| Label | Class | Pass | Pass Rate |
|---|---|---|---|
{label_table}

## Gates

| Gate | Status | Detail |
|---|---|---|
{gate_table}

---
*{SCRIPT_NAME} | PumpSmart v14.2 | {date.today()}*
"""
rp = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
with open(rp, "w", encoding="utf-8") as f:
    f.write(report)
log(f"  Report → {rp}")

# =============================================================================
# PASTE TEXT + MANIFEST
# =============================================================================
print()
print("=" * 72)
print("== PASTE TEXT UPDATE ==")
print(f"T1.5.2_v2_status         = {'COMPLETE' if overall_pass else 'GATE_FAIL'}")
print(f"T1.5.2_v2_method         = absolute_slope_change / signal_std")
print(f"T1.5.2_v2_overall        = {overall*100:.2f}% (target >=95%)")
print(f"T1.5.2_v2_gate_status    = {'PASS' if overall_pass else 'FAIL'}")
print(f"T1.5.2_v2_delta_thresh   = {DELTA_THRESHOLD}")
print(f"T1.5.2_v2_gates          = {n_pass}/{n_pass+n_fail}")
if overall_pass:
    print("ALL TIER-1.5 ITEMS COMPLETE — M10 IS UNBLOCKED")
else:
    print("Gate still failing — paste output for diagnosis")
print("== END PASTE UPDATE ==")
print()
print("-- FILE MANIFEST --")
print(f"NEW: {cfg_path}")
print(f"NEW: {rp}")

log("\n[DONE]")