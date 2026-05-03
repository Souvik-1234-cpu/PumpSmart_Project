# =============================================================================
# module_06B_patch_label19_seal_fast.py
# PumpSmart v14.2 — Surgical patch: Label 19 (seal_failure_fast) regeneration
# =============================================================================
#
# ROOT CAUSE (confirmed 2026-05-02):
#   generate_seal_failure_fast() had onset = rng.integers(20, 50).
#   M6B dispatcher embeds CIRA spike seed into steps 0–49.
#   Generator onset (20–50) fell inside spike seed window → drop applied
#   inside the spike window which gets overwritten → "hold at minimum"
#   locked in spike seed noise value (0.9654) not orifice physics (0.48).
#   Result: Pres.SV drop = 0.035 units instead of 0.35–0.52 units.
#   LSTM-AE produces near-normal z_t → score_A ≈ normal → TPR = 0%.
#
# FIX:
#   onset = rng.integers(55, 85) — forces onset into physics extension
#   window (steps 50+), after spike seed window ends.
#   frac reaches 1.0 at last drop step (off-by-one corrected).
#   target_min computed from max_drop, held explicitly for steps after drop.
#
# SCOPE: Label 19 ONLY (800 sequences).
#   Replaces positions 1200–1999 in M6B_sequences_groupD.pkl.
#   Replaces corresponding entries in z_t_sequences_groupD.pkl.
#   All other labels (18, 20, 21) and all other pkl files: UNTOUCHED.
#
# SEQUENCE: Run this → then re-run module_08_tcn_ae_detection_stack.py
#
# Pump: 110 kW | 7-stage | 40 bar | 2980 RPM | 45 m³/h | 450 m head
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, IS_GPU, SYNTH_DIR, MODEL_DIR, OUTPUT_DIR)
from datetime import date, datetime
import json, math, pickle, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')

SCRIPT_NAME = "module_06B_patch_label19_seal_fast"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}

log("=" * 72)
log(f"  PumpSmart — Label 19 Patch | v14.2 | {date.today()}")
log(f"  Device: {DEVICE} | GPU: {IS_GPU}")
log("=" * 72)

# =============================================================================
# SECTION 0 — CONSTANTS (must match m6b_physics_lib.py exactly)
# =============================================================================
SEED        = 42 + 1900   # offset from original M6B seed to get different but
                           # reproducible sequences
LABEL_19    = 19
N_SEQ_19    = 800
SEQ_STEPS   = 150         # Label 19 locked sequence length
WINDOW_SIZE = 50          # M4 LSTM-AE window — LOCKED
N_WINDOWS   = SEQ_STEPS // WINDOW_SIZE  # = 3

# M6B locked channel order (NEVER change)
CH = {
    "Mot.SV": 0, "Pmp.SV": 1, "Mot.TV": 2, "Pmp.PV": 3,
    "Temp.SV": 4, "Pres.SV": 5, "Pmp.TV": 6, "Mot.PV": 7
}
CLUSTER_ID_STEADY = 1   # steady_state cluster

# Physics constants (IEC 60534 / orifice discharge)
Cd    = 0.61       # orifice discharge coefficient
RHO   = 1000.0     # fluid density kg/m³
dP_NOM = 40e5      # 40 bar nominal discharge pressure in Pa

# SCADA noise std (M5 locked)
NOISE_STD = {
    "Mot.SV": 0.035, "Pmp.SV": 0.040, "Mot.TV": 0.008, "Pmp.PV": 0.012,
    "Temp.SV": 0.010, "Pres.SV": 0.015, "Pmp.TV": 0.008, "Mot.PV": 0.012,
}

log(f"  Label 19 sequences to regenerate: {N_SEQ_19}")
log(f"  Sequence length: {SEQ_STEPS} steps = {N_WINDOWS} windows × {WINDOW_SIZE}")

# =============================================================================
# SECTION 1 — LOAD M3 NORMALIZATION CONFIG
# =============================================================================
log("\nSECTION 1 — Loading M3 normalization config")

try:
    norm_path = OUTPUT_DIR / "M3_normalization_config.json"
    if not norm_path.exists():
        norm_path = MODEL_DIR / "M3_normalization_config.json"
    with open(norm_path) as f:
        norm_config = json.load(f)
    log(f"  ✓ Loaded: {norm_path.name}")
except Exception as e:
    log(f"  [ERROR] Cannot load M3 config: {e}")
    raise

TEMP_CHANNELS = {"Mot.TV", "Pmp.TV", "Temp.SV"}
CHANNEL_TO_M3_KEY = {
    "Mot.SV": "X_ACR_Mot.SV", "Pmp.SV": "X_ACR_Pmp.SV",
    "Mot.TV": "X_ACR_Mot.TV", "Pmp.PV": "X_ACR_Pmp.PV",
    "Temp.SV": "X_Temp.SV",   "Pres.SV": "X_Pres.SV",
    "Pmp.TV": "X_ACR_Pmp.TV", "Mot.PV": "X_ACR_Mot.PV",
}

def get_cluster_mean(cluster_id, channel):
    """P*/a* channels → 1.0. T* channels → cluster-relative mean."""
    if channel not in TEMP_CHANNELS:
        return 1.0
    m3_key = CHANNEL_TO_M3_KEY.get(channel)
    if m3_key is None:
        return 1.0
    try:
        ch_data = norm_config[str(cluster_id)][m3_key]
        T_mean = float(ch_data["mean"])
        T_min  = float(ch_data["p2_5"])
        T_max  = float(ch_data["p97_5"])
        denom  = T_max - T_min
        return (T_mean - T_min) / denom if denom > 1e-6 else 1.0
    except (KeyError, TypeError):
        return 1.0

def make_baseline(n_steps, cluster_id=1, rng=None, noise_sigma=0.015):
    """All 8 channels at cluster normalized baseline ± Gaussian noise."""
    seq = np.zeros((n_steps, 8), dtype=np.float32)
    for ch_name, ch_idx in CH.items():
        mean_val = get_cluster_mean(cluster_id, ch_name)
        noise    = rng.normal(0, noise_sigma, size=n_steps).astype(np.float32)
        seq[:, ch_idx] = mean_val + noise
    return seq

# =============================================================================
# SECTION 2 — FIXED generate_seal_failure_fast()
# =============================================================================
log("\nSECTION 2 — Defining fixed generator")

def generate_seal_failure_fast_fixed(rng, cluster_id=1):
    """
    Label 19: Catastrophic seal blowout — turbulent orifice discharge.
    Physics: Q_leak = Cd × A_orifice × sqrt(2 × dP / rho)  [IEC 60534]
    NOT Hagen-Poiseuille — seal blowout is turbulent, not laminar.

    FIX vs original:
      (1) onset = rng.integers(55, 85) — forces onset AFTER spike seed window
          (steps 0–49 are spike seed territory; physics extension is steps 50+)
      (2) frac = (t - onset + 1) / drop_steps — reaches 1.0 at final drop step
          (original had off-by-one: frac never reached 1.0)
      (3) target_min computed from max_drop then held explicitly — prevents
          "hold at minimum" from locking in spike seed noise value

    Physics validation:
      A_frac = 0.001–0.004 → A_ori = 1e-7 to 4e-7 m²
      Q_leak at A_frac=0.002: 0.61 × 2e-7 × √(8e6) = 3.45e-4 m³/s
      max_drop = min(0.8, 3.45e-4 × 1500) = min(0.8, 0.517) = 0.517
      target_min = 1.0 - 0.517 = 0.483 ← physically correct
      Pres.SV drops from ~1.0 to ~0.48 in 10–20 steps = catastrophic blowout
      For 40 bar pump: 0.48 × cluster_mean_bar ≈ 19 bar residual — realistic
      for a damaged mechanical seal at full process pressure.

    Secondary signal: Mot.PV current spike (motor overload from fluid loss)
    """
    seq = make_baseline(SEQ_STEPS, cluster_id, rng=rng)

    # ── Seal blowout drop magnitude — severity-based (physics-verified) ─────
    # Direct parameterisation: max_drop = severity * 0.60
    # Physics basis (40-bar 7-stage pump, IEC 60534 orifice discharge):
    #   Minor seal damage  (sev=0.20): drop=0.12 → Pres.SV min ≈ 0.88
    #   Moderate blowout   (sev=0.50): drop=0.30 → Pres.SV min ≈ 0.70
    #   Catastrophic fail  (sev=0.80): drop=0.48 → Pres.SV min ≈ 0.52
    # Using severity directly avoids A_ref version mismatch between
    # m6b_physics_lib.py versions (A_ref=1e-4 vs 1e-6 discrepancy confirmed).
    severity_local = float(rng.uniform(0.20, 0.80))
    max_drop       = float(severity_local * 0.60)

    # target_min is the Pres.SV value after complete blowout
    target_min = float(max(0.05, 1.0 - max_drop))

    # ── FIX 1: onset AFTER spike seed window ─────────────────────────────────
    # Spike seed occupies steps 0–49. Physics extension = steps 50–149.
    # Onset at 55–85 ensures the orifice drop is in the physics window.
    onset      = int(rng.integers(55, 85))
    drop_steps = int(rng.integers(10, 21))   # ≤20 step catastrophic drop (spec)

    # ── FIX 2: frac reaches 1.0 at final drop step ───────────────────────────
    for t in range(onset, min(onset + drop_steps, SEQ_STEPS)):
        frac = (t - onset + 1) / drop_steps   # +1: 1/N, 2/N, ..., N/N
        seq[t, CH["Pres.SV"]] = float(max(
            target_min,
            seq[t, CH["Pres.SV"]] - max_drop * frac
        ))

    # ── FIX 3: hold at target_min explicitly ─────────────────────────────────
    for t in range(min(onset + drop_steps, SEQ_STEPS), SEQ_STEPS):
        seq[t, CH["Pres.SV"]] = target_min + float(
            rng.normal(0, NOISE_STD["Pres.SV"]))

    # ── Secondary: motor current spike (fluid momentum loss → current rise) ──
    t_sec_end = min(onset + 15, SEQ_STEPS)
    seq[onset:t_sec_end, CH["Mot.PV"]] += float(
        rng.uniform(0.20, 0.35))

    # ── Add SCADA noise to all channels ──────────────────────────────────────
    for ch_name, ch_idx in CH.items():
        if ch_idx not in [CH["Pres.SV"], CH["Mot.PV"]]:
            seq[:, ch_idx] += rng.normal(
                0, NOISE_STD[ch_name], size=SEQ_STEPS).astype(np.float32)

    return seq.astype(np.float32), onset, max_drop, target_min

# =============================================================================
# SECTION 3 — VALIDATE FIX ON 5 SAMPLES BEFORE FULL GENERATION
# =============================================================================
log("\nSECTION 3 — Pre-generation validation (5 samples)")

rng_test = np.random.default_rng(SEED)
validation_pass = True

log(f"  {'Seq':>4}  {'onset':>6}  {'max_drop':>9}  {'target_min':>11}  "
    f"{'Pres_min':>9}  {'Pres_min_step':>14}  {'Physics_OK':>11}")
log(f"  {'-'*70}")

for i in range(5):
    seq_v, onset_v, mdrop_v, tmin_v = generate_seal_failure_fast_fixed(rng_test)
    pres   = seq_v[:, CH["Pres.SV"]]
    pmin   = float(pres.min())
    pstep  = int(pres.argmin())
    # Physics check: Pres.SV must drop meaningfully below 0.85
    ok = pmin < 0.85 and pstep >= 50 and (pres[:50].mean() > pmin + 0.05)
    if not ok:
        validation_pass = False
    log(f"  {i:>4}  {onset_v:>6}  {mdrop_v:>9.4f}  {tmin_v:>11.4f}  "
        f"{pmin:>9.4f}  {pstep:>14}  "
        f"{'✓ PASS' if ok else '✗ FAIL'}")

if not validation_pass:
    log("\n  [FATAL] Pre-generation validation FAILED.")
    log("  Check generate_seal_failure_fast_fixed() physics.")
    raise RuntimeError("Label 19 patch validation failed — not safe to proceed.")

log(f"\n  Pre-generation validation: PASS ✓")
results['label19_validation_pass'] = True

# =============================================================================
# SECTION 4 — GENERATE 800 FIXED LABEL 19 SEQUENCES
# =============================================================================
log("\nSECTION 4 — Generating 800 fixed Label 19 sequences")

rng_gen = np.random.default_rng(SEED)
new_sequences = []
new_metadata  = []

pres_min_vals = []
drop_magnitudes = []

for i in range(N_SEQ_19):
    severity = float(rng_gen.uniform(0.20, 0.80))  # match original severity range
    seq_i, onset_i, mdrop_i, tmin_i = generate_seal_failure_fast_fixed(
        rng_gen, cluster_id=CLUSTER_ID_STEADY)

    new_sequences.append(seq_i)
    new_metadata.append({
        'label':        LABEL_19,
        'fault_name':   'seal_failure_fast',
        'group':        'D',
        'severity':     severity,
        'cluster_id':   CLUSTER_ID_STEADY,
        'cluster_name': 'steady_state',
        'steps':        SEQ_STEPS,
        'source':       'physics_synthetic_variant_patched_v2',
        'arch_version': 'v14.2',
        'patch_note':   'onset_fixed_post_spike_seed_window',
        'onset_step':   onset_i,
        'max_drop':     round(mdrop_i, 4),
        'target_min':   round(tmin_i, 4),
    })

    pres_min_vals.append(float(seq_i[:, CH["Pres.SV"]].min()))
    drop_magnitudes.append(mdrop_i)

    if (i + 1) % 200 == 0:
        log(f"  Generated {i+1}/{N_SEQ_19}")

pres_min_arr = np.array(pres_min_vals)
log(f"\n  Generation complete: {len(new_sequences)} sequences")
log(f"  Pres.SV min stats: mean={pres_min_arr.mean():.4f} "
    f"std={pres_min_arr.std():.4f} "
    f"min={pres_min_arr.min():.4f} max={pres_min_arr.max():.4f}")
log(f"  Drop magnitude: mean={np.mean(drop_magnitudes):.4f} "
    f"min={np.min(drop_magnitudes):.4f} max={np.max(drop_magnitudes):.4f}")

# Physics gate: ≥95% sequences must have Pres.SV min < 0.85
gate_pres_drop = float(np.mean(pres_min_arr < 0.85))
log(f"  Gate (Pres.SV min < 0.85): {gate_pres_drop:.2%} (target ≥95%)")
if gate_pres_drop < 0.95:
    log("  [FATAL] Physics gate failed — seal blowout physics not firing correctly")
    raise RuntimeError(f"Pres.SV drop gate failed: {gate_pres_drop:.2%} < 95%")

results['label19_generated']   = N_SEQ_19
results['label19_pres_min_mean'] = round(float(pres_min_arr.mean()), 4)
results['label19_gate_pres_drop'] = round(gate_pres_drop, 4)

# =============================================================================
# SECTION 5 — LOAD M4 LSTM-AE FOR z_t EXPORT
# =============================================================================
log("\nSECTION 5 — Loading frozen M4 LSTM-AE")

class LSTMAEEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(8, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm2 = torch.nn.LSTM(128, 64, num_layers=1, batch_first=True)
        self.bn    = torch.nn.LayerNorm(64)
    def forward(self, x):
        out1, _          = self.lstm1(x)
        out2, (h_n, c_n) = self.lstm2(out1)
        return self.bn(h_n[-1]), h_n, c_n

class LSTMAEDecoder(torch.nn.Module):
    def __init__(self, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.fc_h  = torch.nn.Linear(64, 128)
        self.fc_c  = torch.nn.Linear(64, 128)
        self.lstm1 = torch.nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm2 = torch.nn.LSTM(128,  8, num_layers=1, batch_first=True)
        self.out   = torch.nn.Linear(8, 8)
    def forward(self, z, h_n, c_n):
        h0    = torch.tanh(self.fc_h(h_n[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0    = torch.tanh(self.fc_c(c_n[-1])).unsqueeze(0).repeat(2, 1, 1)
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm1(z_rep, (h0, c0))
        out, _ = self.lstm2(out)
        return self.out(out)

class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, seq_len=50):
        super().__init__()
        self.encoder = LSTMAEEncoder()
        self.decoder = LSTMAEDecoder(seq_len=seq_len)
    def forward(self, x):
        z, h_n, c_n = self.encoder(x)
        return self.decoder(z, h_n, c_n)
    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

try:
    m4_model = LSTMAutoencoder(seq_len=WINDOW_SIZE)
    ckpt = torch.load(MODEL_DIR / "lstm_ae_baseline_final.pth", map_location='cpu')
    m4_model.load_state_dict(ckpt)
    m4_model.eval()
    for p in m4_model.parameters():
        p.requires_grad_(False)
    m4_model.to(DEVICE)
    log(f"  ✓ M4 LSTM-AE loaded → {DEVICE}")
    log(f"  Params: {sum(p.numel() for p in m4_model.parameters()):,} (FROZEN)")
except Exception as e:
    log(f"  [FATAL] M4 load error: {e}"); raise

# =============================================================================
# SECTION 6 — EXPORT z_t FOR NEW LABEL 19 SEQUENCES
# =============================================================================
log("\nSECTION 6 — Exporting z_t for 800 new sequences")

new_zt_records = []
m4_model.eval()

for i, seq in enumerate(new_sequences):
    # Slide WINDOW_SIZE windows over the 150-step sequence
    n_win    = SEQ_STEPS // WINDOW_SIZE  # = 3
    z_seq    = []
    mae_seq  = []

    for w in range(n_win):
        window = seq[w*WINDOW_SIZE:(w+1)*WINDOW_SIZE]   # (50, 8)
        x_t    = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            z    = m4_model.encode(x_t).squeeze(0).cpu().numpy()    # (64,)
            recon = m4_model(x_t).squeeze(0).cpu().numpy()           # (50, 8)

        mae_ch = np.mean(np.abs(recon - window), axis=0)             # (8,)
        z_seq.append(z)
        mae_seq.append(mae_ch)

    new_zt_records.append({
        'z_t': np.array(z_seq,   dtype=np.float32),   # (3, 64)
        'mae': np.array(mae_seq, dtype=np.float32),   # (3, 8)
    })

    if (i + 1) % 200 == 0:
        log(f"  z_t exported: {i+1}/{N_SEQ_19}")

# Validate z_t norms — should NOT be uniform ~-0.018 anymore
zt_norms = [np.linalg.norm(r['z_t']) for r in new_zt_records[:20]]
log(f"\n  z_t norm sample (first 5): {[round(n,3) for n in zt_norms[:5]]}")
log(f"  z_t mean sample (first 5): "
    f"{[round(float(r['z_t'].mean()),4) for r in new_zt_records[:5]]}")

# Check MAE on Pres.SV (ch5) — should be HIGH post-blowout
mae_pres = [r['mae'][:, CH["Pres.SV"]].mean() for r in new_zt_records[:20]]
log(f"  Pres.SV MAE mean (first 5): {[round(m,4) for m in mae_pres[:5]]}")
log(f"  (original was ~0.006 — should now be substantially higher)")

results['label19_zt_exported'] = len(new_zt_records)

# =============================================================================
# SECTION 7 — SURGICAL REPLACEMENT IN groupD pkl FILES
# =============================================================================
log("\nSECTION 7 — Patching M6B_sequences_groupD.pkl (Label 19 positions 1200–1999)")

grpD_path = SYNTH_DIR / "M6B_sequences_groupD.pkl"
zt_D_path = SYNTH_DIR / "z_t_sequences_groupD.pkl"

# ── Backup originals ──────────────────────────────────────────────────────────
backup_grpD = SYNTH_DIR / "M6B_sequences_groupD_backup_pre_label19_patch.pkl"
backup_ztD  = SYNTH_DIR / "z_t_sequences_groupD_backup_pre_label19_patch.pkl"

try:
    import shutil
    if not backup_grpD.exists():
        shutil.copy2(grpD_path, backup_grpD)
        log(f"  ✓ Backup created: {backup_grpD.name}")
    else:
        log(f"  Backup already exists: {backup_grpD.name} — skipping")

    if not backup_ztD.exists():
        shutil.copy2(zt_D_path, backup_ztD)
        log(f"  ✓ Backup created: {backup_ztD.name}")
    else:
        log(f"  Backup already exists: {backup_ztD.name} — skipping")
except Exception as e:
    log(f"  [FATAL] Cannot create backup: {e}"); raise

# ── Load existing groupD pkl ──────────────────────────────────────────────────
log("\n  Loading existing M6B_sequences_groupD.pkl...")
try:
    with open(grpD_path, "rb") as f:
        grpD = pickle.load(f)
    seqs_D = grpD['sequences']
    meta_D = grpD['metadata']
    log(f"  Loaded: {len(seqs_D)} sequences")

    # Verify Label 19 positions
    labels_in_pkl = [m['label'] for m in meta_D]
    lbl19_positions = [i for i, l in enumerate(labels_in_pkl) if l == LABEL_19]
    assert len(lbl19_positions) == N_SEQ_19, \
        f"Expected {N_SEQ_19} Label 19 positions, found {len(lbl19_positions)}"
    assert lbl19_positions[0] == 1200, \
        f"Label 19 start position: expected 1200, got {lbl19_positions[0]}"
    log(f"  Label 19 positions confirmed: {lbl19_positions[0]}–{lbl19_positions[-1]}")
except Exception as e:
    log(f"  [FATAL] Cannot load/verify groupD pkl: {e}"); raise

# ── Replace Label 19 sequences and metadata ───────────────────────────────────
log("  Replacing 800 Label 19 sequences...")
for j, pos in enumerate(lbl19_positions):
    seqs_D[pos] = new_sequences[j]
    meta_D[pos] = new_metadata[j]

# Verify replacement
pres_after = [seqs_D[pos][:, CH["Pres.SV"]].min()
              for pos in lbl19_positions[:10]]
log(f"  Pres.SV min after patch (first 10): {[round(p,4) for p in pres_after]}")

# ── Save patched groupD pkl ───────────────────────────────────────────────────
grpD_patched = {'sequences': seqs_D, 'metadata': meta_D}
try:
    with open(grpD_path, "wb") as f:
        pickle.dump(grpD_patched, f, protocol=4)
    log(f"  ✓ Saved patched: {grpD_path.name} "
        f"({grpD_path.stat().st_size / 1e6:.1f} MB)")
except Exception as e:
    log(f"  [FATAL] Cannot save patched groupD: {e}"); raise

# ── Load and patch z_t_sequences_groupD.pkl ──────────────────────────────────
log("\n  Loading existing z_t_sequences_groupD.pkl...")
try:
    with open(zt_D_path, "rb") as f:
        zt_D = pickle.load(f)

    # z_t pkl is list[dict{z_t, mae}] — same length as sequences
    if isinstance(zt_D, list):
        log(f"  z_t pkl: list format, {len(zt_D)} entries")
        for j, pos in enumerate(lbl19_positions):
            zt_D[pos] = new_zt_records[j]
    elif isinstance(zt_D, dict) and 'sequences' in zt_D:
        zt_list = zt_D['sequences']
        log(f"  z_t pkl: dict format, {len(zt_list)} entries")
        for j, pos in enumerate(lbl19_positions):
            zt_list[pos] = new_zt_records[j]
        zt_D['sequences'] = zt_list
    else:
        log(f"  [WARN] Unknown z_t pkl format: {type(zt_D)}. Attempting list replacement.")
        zt_D = list(zt_D) if not isinstance(zt_D, list) else zt_D
        for j, pos in enumerate(lbl19_positions):
            zt_D[pos] = new_zt_records[j]

    with open(zt_D_path, "wb") as f:
        pickle.dump(zt_D, f, protocol=4)
    log(f"  ✓ Saved patched z_t: {zt_D_path.name} "
        f"({zt_D_path.stat().st_size / 1e6:.1f} MB)")
except Exception as e:
    log(f"  [FATAL] Cannot patch z_t pkl: {e}"); raise

results['groupD_patched'] = True
results['zt_groupD_patched'] = True

# =============================================================================
# SECTION 8 — POST-PATCH VERIFICATION
# =============================================================================
log("\nSECTION 8 — Post-patch verification")

# Reload and re-check
with open(grpD_path, "rb") as f:
    verify = pickle.load(f)
v_seqs = verify['sequences']
v_meta = verify['metadata']

v_lbl19 = [i for i, m in enumerate(v_meta) if m['label'] == LABEL_19]
v_pres  = [v_seqs[i][:, CH["Pres.SV"]].min() for i in v_lbl19]
v_pres_arr = np.array(v_pres)

log(f"  Label 19 count in patched pkl: {len(v_lbl19)} (expected {N_SEQ_19})")
log(f"  Pres.SV min: mean={v_pres_arr.mean():.4f} "
    f"std={v_pres_arr.std():.4f} "
    f"min={v_pres_arr.min():.4f} max={v_pres_arr.max():.4f}")
log(f"  Fraction below 0.85: {np.mean(v_pres_arr < 0.85):.2%} (target ≥95%)")

# Check other labels are intact
label_dist = {}
for m in v_meta:
    label_dist[m['label']] = label_dist.get(m['label'], 0) + 1
log(f"  Label distribution after patch: {label_dist}")
assert label_dist == {18: 1200, 19: 800, 20: 1200, 21: 2000}, \
    f"Label distribution mismatch: {label_dist}"
log("  Label distribution: INTACT ✓")

# Verify other labels untouched (sample Label 21)
lbl21_idx = [i for i, m in enumerate(v_meta) if m['label'] == 21]
lbl21_motSV = v_seqs[lbl21_idx[0]][:, CH["Mot.SV"]]
assert len(lbl21_motSV) == 20 * WINDOW_SIZE, \
    f"Label 21 sequence length corrupted: {len(lbl21_motSV)}"
log("  Label 21 sequences: INTACT ✓")

post_gate = float(np.mean(v_pres_arr < 0.85))
GATE_PASS = post_gate >= 0.95
results['post_patch_gate_pass'] = GATE_PASS
results['post_patch_pres_min_mean'] = round(float(v_pres_arr.mean()), 4)
results['post_patch_pres_drop_rate'] = round(post_gate, 4)

# =============================================================================
# SECTION 9 — REPORT
# =============================================================================
log("\nSECTION 9 — Saving report")

report = f"""# PumpSmart — Label 19 Patch Report
**Date:** {date.today()}  
**Script:** {SCRIPT_NAME}  
**Arch:** v14.2

## Root Cause
`generate_seal_failure_fast()` had `onset = rng.integers(20, 50)`.
M6B dispatcher embeds CIRA spike seed into steps 0–49.
Onset inside spike window → drop applied inside spike data →
"hold at minimum" locked in spike noise value (0.9654) not orifice drop (0.48).
Pres.SV drop magnitude: **0.035 units** (was) vs **0.35–0.52 units** (expected).
LSTM-AE z_t ≈ normal → score_A ≈ normal → M8 TPR = 0%.

## Fix Applied
1. `onset = rng.integers(55, 85)` — forces onset after spike seed window
2. `frac = (t - onset + 1) / drop_steps` — reaches 1.0 at final step
3. `target_min = 1.0 - max_drop` held explicitly in post-drop window

## Results
| Metric | Before | After |
|--------|--------|-------|
| Pres.SV min mean | 0.9738 | {results['post_patch_pres_min_mean']} |
| Fraction < 0.85 | ~0% | {results['post_patch_pres_drop_rate']:.2%} |
| Physics gate pass | FAIL | {'PASS ✓' if GATE_PASS else 'FAIL ✗'} |

## Files Modified
- `data/synthetic/M6B_sequences_groupD.pkl` — Label 19 positions 1200–1999 replaced
- `data/synthetic/z_t_sequences_groupD.pkl` — z_t for same positions replaced

## Backups Created
- `data/synthetic/M6B_sequences_groupD_backup_pre_label19_patch.pkl`
- `data/synthetic/z_t_sequences_groupD_backup_pre_label19_patch.pkl`

## Next Step
Re-run `module_08_tcn_ae_detection_stack.py`
Expected: Label 19 TPR substantially above 0%.
"""

rpt_path = REPORT_DIR / f"{SCRIPT_NAME}_report.md"
with open(rpt_path, 'w', encoding='utf-8') as f:
    f.write(report)
log(f"  ✓ Report: {rpt_path}")

# =============================================================================
# SECTION 10 — PASTE TEXT UPDATE
# =============================================================================
print("\n" + "═"*72)
print("══ PASTE TEXT UPDATE ══")
print("═"*72)
print(f"""
LABEL19_PATCH ({date.today()})
label19_patch_applied              : True
label19_onset_fix                  : rng.integers(55,85) [was 20,50]
label19_pres_min_mean_before       : ~0.9738
label19_pres_min_mean_after        : {results['post_patch_pres_min_mean']}
label19_pres_drop_gate             : {results['post_patch_pres_drop_rate']:.2%} ≥95%
label19_gate_pass                  : {results['post_patch_gate_pass']}
label19_files_patched              : M6B_sequences_groupD.pkl, z_t_sequences_groupD.pkl
label19_backups_created            : True
NEXT_ACTION                        : Re-run module_08_tcn_ae_detection_stack.py
""")
print("═"*72)

log(f"\n{'='*72}")
log(f"  Label 19 Patch COMPLETE")
log(f"  Physics gate: {'PASS ✓' if GATE_PASS else 'FAIL ✗'}")
log(f"  Pres.SV min mean: {results['post_patch_pres_min_mean']}")
log(f"  Next: Run module_08_tcn_ae_detection_stack.py")
log(f"{'='*72}")