# ═══════════════════════════════════════════════════════════════════════════════
# src/m6b_physics_lib.py
# PumpSmart — Unified Physics Generation Library (M6B canonical)
# Single source of truth for ALL fault generation in M6B Steps 0, 0b, 1, 2, 3
# and M12 validation suite.
#
# Fixes applied vs M6A / fragmented Step 0/0b generators:
#   F1: Label 1 — Temp.SV* now coupled to Mot.TV* via _tcoup (r=0.9793, M2 locked)
#   F2: Label 2 — AM envelope uses abs(sin) — vibration amplitude non-negative
#   F3: Label 3 — M5-faithful cavitation: severity-dependent t_onset,
#                 mean_drop=sev*0.6, noise_amp=sev*0.3, Bernoulli spike scaling
#   F4: Label 5 — Pres.SV* affinity law Q-H shift (M5: PresSV=(Q/Q_BEP)^2*(1-sev*0.1))
#   F5: Label 6 — dropout subtype added (channel→0.0, cable cut / I/O card failure)
#
# All generators operate in NORMALIZED space (P*, a*, ΔT*)
# All channel indices use M6B locked order: Mot.SV=0, Pmp.SV=1, Mot.TV=2,
#   Pmp.PV=3, Temp.SV=4, Pres.SV=5, Pmp.TV=6, Mot.PV=7
# All cluster means read via get_cluster_mean() — NEVER hardcoded
# All winsorization via apply_winsorization() — cluster-conditional, NEVER global sigma
# ═══════════════════════════════════════════════════════════════════════════════

import math
import json
import numpy as np
from pathlib import Path

# ── M6B locked channel order ───────────────────────────────────────────────────
CHANNELS = ["Mot.SV", "Pmp.SV", "Mot.TV", "Pmp.PV",
            "Temp.SV", "Pres.SV", "Pmp.TV", "Mot.PV"]
CH   = {c: i for i, c in enumerate(CHANNELS)}
N_CH = 8

# ── M3 column name mapping ─────────────────────────────────────────────────────
CHANNEL_TO_M3_KEY = {
    "Mot.SV":  "X_ACR_Mot.SV",
    "Pmp.SV":  "X_ACR_Pmp.SV",
    "Mot.TV":  "X_ACR_Mot.TV",
    "Pmp.PV":  "X_ACR_Pmp.PV",
    "Temp.SV": "X_Temp.SV",
    "Pres.SV": "X_Pres.SV",
    "Pmp.TV":  "X_ACR_Pmp.TV",
    "Mot.PV":  "X_ACR_Mot.PV",
}

# Temperature channels (use ΔT* normalization, not P*/a*)
TEMP_CHANNELS = {"Mot.TV", "Pmp.TV", "Temp.SV"}

# M5 SCADA noise std (from M5 physics engine — LOCKED)
NOISE_STD = {
    "Mot.SV":  0.035,
    "Pmp.SV":  0.040,
    "Mot.TV":  0.008,
    "Pmp.PV":  0.012,
    "Temp.SV": 0.010,
    "Pres.SV": 0.015,
    "Pmp.TV":  0.008,
    "Mot.PV":  0.012,
}

# Sensor failure subtypes — M5 canonical (4 subtypes including dropout)
SENSOR_SUBTYPES = ["flatline", "spike", "drift", "dropout"]

# Module-level state — populated by init_lib()
_norm_config  = None
_phys_config  = None
_rng          = None
TAU_THERMAL_s = 388.9
BPF_HZ        = 347.67
A_WAVE_m_s    = 1200.0
RHO           = 1000.0


def init_lib(norm_config: dict, phys_config: dict, seed: int = 42):
    """
    Must be called once before using any generation function.
    Loads normalization config and physics constants into module state.
    """
    global _norm_config, _phys_config, _rng
    global TAU_THERMAL_s, BPF_HZ, A_WAVE_m_s, RHO
    _norm_config = norm_config
    _phys_config = phys_config
    _rng         = np.random.default_rng(seed)
    _pc          = phys_config.get("physics_constants", phys_config)
    TAU_THERMAL_s = float(_pc.get("TAU_THERMAL_s", 388.9))
    BPF_HZ        = float(_pc.get("BPF_HZ",        347.67))
    A_WAVE_m_s    = float(_pc.get("A_WAVE_m_s",    1200.0))
    RHO           = float(_pc.get("RHO",            1000.0))


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_cluster_mean(cluster_id: int, channel: str, fallback: float = 1.0) -> float:
    """
    P*/a* channels → 1.0 by definition (normalized to cluster mean).
    T* channels → (T_mean - T_min) / (T_max - T_min) from M3 config.
    """
    if channel not in TEMP_CHANNELS:
        return 1.0
    m3_key = CHANNEL_TO_M3_KEY.get(channel)
    if m3_key is None or _norm_config is None:
        return fallback
    try:
        ch_data = _norm_config[str(cluster_id)][m3_key]
        T_mean  = float(ch_data["mean"])
        T_min   = float(ch_data["p2_5"])
        T_max   = float(ch_data["p97_5"])
        denom   = T_max - T_min
        return (T_mean - T_min) / denom if denom > 1e-6 else fallback
    except (KeyError, TypeError, ValueError):
        return fallback


def apply_winsorization(seq: np.ndarray, cluster_id: int) -> np.ndarray:
    """
    Cluster-conditional winsorization per C-18.
    High-load (3): ceiling 2.0× (fault-sensitive).
    All others: ceiling 3.0×.
    NEVER global sigma winsorization.
    """
    cluster_ceilings = {0: 3.0, 1: 3.0, 2: 3.0, 3: 2.0}
    ceil_mult = cluster_ceilings.get(cluster_id, 3.0)
    for ch_name, ch_idx in CH.items():
        mean_val = get_cluster_mean(cluster_id, ch_name, fallback=1.0)
        seq[:, ch_idx] = np.clip(seq[:, ch_idx], 0.0, ceil_mult * mean_val)
    return seq


def make_baseline(n_steps: int, cluster_id: int = 1,
                  noise_sigma: float = 0.015) -> np.ndarray:
    """All 8 channels at cluster normalized baseline ± Gaussian noise."""
    seq = np.zeros((n_steps, N_CH), dtype=np.float32)
    for ch_name, ch_idx in CH.items():
        mean_val = get_cluster_mean(cluster_id, ch_name, fallback=1.0)
        noise    = _rng.normal(0, noise_sigma, size=n_steps).astype(np.float32)
        seq[:, ch_idx] = mean_val + noise
    return seq


def scada_noise(ch_name: str, size: int) -> np.ndarray:
    """M5-calibrated SCADA noise per channel."""
    return _rng.normal(0, NOISE_STD.get(ch_name, 0.015), size=size).astype(np.float32)


def _tcoup(mot_tv_arr: np.ndarray, temp_ref: float,
           r: float = 0.9793) -> np.ndarray:
    """
    M5 EQ15: enforce M2 thermal coupling r=0.9793 between Mot.TV* and Temp.SV*.
    TempSV* = temp_ref + r * (MotTV* - mot_tv_baseline)
    """
    mot_tv_baseline = get_cluster_mean(1, "Mot.TV", fallback=0.6)
    return temp_ref + r * (mot_tv_arr - mot_tv_baseline)


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT GENERATION FUNCTIONS — M5 FAITHFUL
# ═══════════════════════════════════════════════════════════════════════════════

def generate_bearing_wear(severity: float = 0.5, cluster_id: int = 1,
                          n_steps: int = 250) -> np.ndarray:
    """
    Label 1 — bearing_wear.
    Physics: Paris-Erdogan crack growth → Mot.SV* exponential rise
             Palmgren bearing heat → Mot.TV* Euler integration
             M2 coupling r=0.9793 → Temp.SV* tracks Mot.TV* (F1 FIX)
             Shaft coupling lag 5-15s → Pmp.SV* sympathetic rise
    Clusters: startup(2), steady_state(1), high_load(3)
    """
    assert n_steps >= 200, "bearing_wear needs >= 200 steps for Paris law evolution"
    PRE = 50
    seq = make_baseline(n_steps, cluster_id=cluster_id)

    motSV_baseline  = get_cluster_mean(cluster_id, "Mot.SV",  fallback=1.0)
    motTV_baseline  = get_cluster_mean(cluster_id, "Mot.TV",  fallback=1.0)
    tempSV_baseline = get_cluster_mean(cluster_id, "Temp.SV", fallback=1.0)
    pmpSV_baseline  = get_cluster_mean(cluster_id, "Pmp.SV",  fallback=1.0)

    # ── Paris-Erdogan crack growth: da/dN = C·ΔK^m ────────────────────────
    C_paris  = 1.8e-4
    m_paris  = 2.8
    dK_base  = severity * 3.5
    crack    = np.zeros(n_steps, dtype=np.float64)
    for t in range(PRE, n_steps):
        dt = t - PRE
        crack[t] = crack[t-1] + C_paris * (dK_base ** m_paris) * (1.0 + 0.003 * dt)
    crack[PRE + min(150, n_steps - PRE):] = crack[PRE + min(150, n_steps - PRE) - 1]
    c_max   = crack.max()
    crack_n = crack / c_max * (1.5 * severity) if c_max > 0 else crack

    # ── Mot.SV* — Paris exponential ────────────────────────────────────────
    for t in range(n_steps):
        bpf_jitter = 0.015 * severity * math.sin(2 * math.pi * t / 50.0)
        seq[t, CH["Mot.SV"]] = (motSV_baseline + crack_n[t]
                                 + bpf_jitter + scada_noise("Mot.SV", 1)[0])

    # ── Mot.TV* — Euler integration of Palmgren bearing heat ───────────────
    thermal_lag = int(_rng.integers(20, 41))
    tau_steps   = TAU_THERMAL_s / 50.0
    MotTV_arr   = np.zeros(n_steps, dtype=np.float64)
    for t in range(n_steps):
        t_src            = max(0, t - thermal_lag)
        bearing_heat     = crack_n[t_src] * 0.35
        temp_rise        = bearing_heat * (1.0 - math.exp(-t / max(1.0, tau_steps)))
        MotTV_arr[t]     = motTV_baseline + temp_rise
        seq[t, CH["Mot.TV"]] = float(MotTV_arr[t]) + scada_noise("Mot.TV", 1)[0]

    # ── Temp.SV* — M2 coupling r=0.9793 (F1 FIX — was missing) ───────────
    TempSV_arr = _tcoup(MotTV_arr, temp_ref=tempSV_baseline, r=0.9793)
    for t in range(n_steps):
        seq[t, CH["Temp.SV"]] = float(TempSV_arr[t]) + scada_noise("Temp.SV", 1)[0]

    # ── Pmp.SV* — sympathetic rise with shaft coupling lag ─────────────────
    pmp_lag = int(_rng.integers(5, 16))
    for t in range(n_steps):
        t_src = max(0, t - pmp_lag)
        seq[t, CH["Pmp.SV"]] = (pmpSV_baseline + 0.30 * crack_n[t_src]
                                  + scada_noise("Pmp.SV", 1)[0])

    seq = apply_winsorization(seq, cluster_id)
    return seq.astype(np.float32)


def generate_impeller_imbalance(severity: float = 0.5, cluster_id: int = 1,
                                n_steps: int = 200) -> np.ndarray:
    """
    Label 2 — impeller_imbalance.
    Physics: ISO 1940 rotating unbalance F=m·e·ω² drives BOTH Pmp.PV and Pmp.SV
             at BPF. abs(sin) envelope — vibration amplitude always ≥ 0 (F2 FIX).
    Clusters: steady_state(1), high_load(3)
    """
    PRE = 50
    seq = make_baseline(n_steps, cluster_id=cluster_id)

    pmpPV_baseline  = get_cluster_mean(cluster_id, "Pmp.PV",  fallback=1.0)
    pmpSV_baseline  = get_cluster_mean(cluster_id, "Pmp.SV",  fallback=1.0)
    presSV_baseline = get_cluster_mean(cluster_id, "Pres.SV", fallback=1.0)
    motPV_baseline  = get_cluster_mean(cluster_id, "Mot.PV",  fallback=1.0)

    for t in range(PRE, n_steps):
        dt        = t - PRE
        progress  = dt / (n_steps - PRE)
        bpf_phase = 2 * math.pi * dt / 50.0

        # F2 FIX: abs(sin) — AM envelope always positive (ISO 1940 unbalance)
        pmpPV_envelope = severity * 0.7 * progress
        seq[t, CH["Pmp.PV"]] = (pmpPV_baseline
                                  + pmpPV_envelope * (1.0 + 0.3 * abs(math.sin(bpf_phase)))
                                  + scada_noise("Pmp.PV", 1)[0])

        pmpSV_envelope = severity * 0.9 * progress
        seq[t, CH["Pmp.SV"]] = (pmpSV_baseline
                                  + pmpSV_envelope * (1.0 + 0.4 * abs(math.sin(bpf_phase)))
                                  + scada_noise("Pmp.SV", 1)[0])

        # Pressure pulsation at BPF (Euler: dP = ρ·ω·r·v_u)
        pres_osc = 0.12 * severity * progress * math.sin(bpf_phase + math.pi / 4)
        seq[t, CH["Pres.SV"]] = (presSV_baseline + pres_osc
                                   + scada_noise("Pres.SV", 1)[0])

        # Mot.PV* sympathetic via shaft coupling (20-step lag)
        t_src    = max(PRE, t - 20)
        prog_src = (t_src - PRE) / (n_steps - PRE)
        bpf_lag  = 2 * math.pi * (t_src - PRE) / 50.0
        seq[t, CH["Mot.PV"]] = (motPV_baseline
                                  + severity * 0.25 * prog_src
                                    * (1.0 + 0.3 * abs(math.sin(bpf_lag)))
                                  + scada_noise("Mot.PV", 1)[0])

    seq = apply_winsorization(seq, cluster_id)
    return seq.astype(np.float32)


def generate_cavitation(severity: float = 0.5, cluster_id: int = 2,
                        n_steps: int = 150) -> np.ndarray:
    """
    Label 3 — cavitation. LOCKED to startup cluster (cluster_id=2).
    Physics: M5 canonical (F3 FIX — full faithful reproduction):
      NPSHa = (P_suc - P_vap)/(ρg) + v²/2g - h_friction → marginal at startup
      Pres.SV*: mean_drop = severity*0.6*progress, noise = severity*0.30
      Pmp.SV*: p_spike = 0.05+0.35*sev, A_spike = sev*3.5 (Rayleigh-Plesset)
      t_onset: severity-dependent — high severity cavitates earlier
    """
    assert cluster_id == 2, "Cavitation MUST be startup cluster (cluster_id=2)"
    P_VAPOUR = 0.023   # bar at ~20°C

    # Severity-dependent onset (M5 canonical)
    t_onset = int((1.0 - severity) * 0.5 * n_steps)
    t_onset = max(10, min(t_onset, 60))

    seq = make_baseline(n_steps, cluster_id=cluster_id)

    presSV_baseline = get_cluster_mean(cluster_id, "Pres.SV", fallback=1.0)
    pmpSV_baseline  = get_cluster_mean(cluster_id, "Pmp.SV",  fallback=1.0)
    pmpTV_baseline  = get_cluster_mean(cluster_id, "Pmp.TV",  fallback=1.0)

    # ── Pres.SV* — erratic decline (M5: mean_drop=sev*0.6, noise=sev*0.30) ─
    PresSV = np.full(n_steps, presSV_baseline, dtype=np.float64)
    for t in range(t_onset, n_steps):
        progress    = (t - t_onset) / (n_steps - t_onset)
        mean_drop   = severity * 0.6 * progress
        noise_amp   = severity * 0.30
        PresSV[t]   = max(presSV_baseline - mean_drop
                          + _rng.normal(0, noise_amp), 0.05)
    PresSV = np.clip(PresSV, 0.02, 3.0)

    # ── Pmp.SV* — Rayleigh-Plesset spikes (M5: p_spike, A_spike, Bernoulli) ─
    p_spike = min(0.05 + severity * 0.35, 0.60)
    A_spike = severity * 3.5
    PmpSV   = np.full(n_steps, pmpSV_baseline, dtype=np.float64)
    for t in range(t_onset, n_steps):
        dP_bar      = max(presSV_baseline - PresSV[t], 0.0)
        bernoulli_v = math.sqrt(max(2.0 * dP_bar * 1e5 / RHO, 0.0))
        spike_scale = min(bernoulli_v / 10.0, 1.0)
        if _rng.random() < p_spike:
            PmpSV[t] = pmpSV_baseline + A_spike * spike_scale * float(_rng.uniform(0.3, 1.0))
        else:
            PmpSV[t] = pmpSV_baseline + 0.10 * float(_rng.random())
    PmpSV = np.clip(PmpSV, 0.5, 8.8)

    # ── Pmp.TV* — cumulative thermal from bubble collapse ─────────────────
    k_cav_thermal  = severity * 0.003
    cum_energy     = np.cumsum(np.maximum(PmpSV - pmpSV_baseline, 0.0))
    PmpTV          = np.clip(pmpTV_baseline + k_cav_thermal * cum_energy, 0.0, 1.4)

    # ── Apply to sequence ─────────────────────────────────────────────────
    for t in range(n_steps):
        seq[t, CH["Pres.SV"]] = float(PresSV[t]) + scada_noise("Pres.SV", 1)[0]
        seq[t, CH["Pmp.SV"]]  = float(PmpSV[t])  + scada_noise("Pmp.SV",  1)[0]
        seq[t, CH["Pmp.TV"]]  = float(PmpTV[t])  + scada_noise("Pmp.TV",  1)[0]

    seq = apply_winsorization(seq, cluster_id)
    return seq.astype(np.float32)


def generate_seal_failure(severity: float = 0.5, cluster_id: int = 1,
                          n_steps: int = 400) -> np.ndarray:
    """
    Label 4 — seal_failure.
    Physics: Q_leak = Cd·A_eff·√(2ΔP/ρ) — orifice discharge (n_steps > 200)
             M5 linear model: PresSV = 1 - sev*0.006*t  (n_steps <= 200)
             Reason: orifice model needs 300+ steps to develop detectable signal.
             At 200 steps, linear M5 canonical gives 60% drop at sev=1.0 — detectable.
    Coupled: Pmp.TV* N-S viscous rise, Pmp.PV* axial thrust, Mot.TV* UNCHANGED (r=-0.013)
    Clusters: steady_state(1), high_load(3)
    """
    PRE = 50
    seq = make_baseline(n_steps, cluster_id=cluster_id)

    presSV_baseline = get_cluster_mean(cluster_id, "Pres.SV", fallback=1.0)
    pmpTV_baseline  = get_cluster_mean(cluster_id, "Pmp.TV",  fallback=1.0)
    pmpSV_baseline  = get_cluster_mean(cluster_id, "Pmp.SV",  fallback=1.0)
    pmpPV_baseline  = get_cluster_mean(cluster_id, "Pmp.PV",  fallback=1.0)

    pres_traj = np.full(n_steps, presSV_baseline, dtype=np.float64)

    if n_steps <= 200:
        # ── M5 canonical linear model (detectable at 200 steps) ──────────
        # PresSV = clip(1.0 - severity*0.006*t, 0.15, ceiling)
        # At sev=0.6, t=200: 1.0 - 0.6*0.006*200 = 0.28 — clearly anomalous
        for t in range(n_steps):
            pres_traj[t] = float(np.clip(
                presSV_baseline - severity * 0.006 * t, 0.15, 3.0))
    else:
        # ── Orifice discharge model (physically rigorous for 400 steps) ──
        # Q_leak = Cd * A_eff(t) * sqrt(2*P_head/rho)
        Cd        = 0.61
        k_growth  = 0.018 * severity
        A_initial = 0.003 * severity
        cum_leak  = 0.0
        for t in range(PRE, n_steps):
            dt        = t - PRE
            A_eff     = A_initial * math.exp(k_growth * dt)
            P_head    = max(0.05, presSV_baseline - cum_leak)
            Q_step    = Cd * A_eff * math.sqrt(2.0 * P_head)
            cum_leak += Q_step * 0.15
            pres_traj[t] = max(0.05, presSV_baseline - cum_leak)

    # ── Apply Pres.SV* ─────────────────────────────────────────────────────
    for t in range(n_steps):
        seq[t, CH["Pres.SV"]] = max(0.0, float(pres_traj[t])
                                    + scada_noise("Pres.SV", 1)[0])

    # ── Pmp.TV* — N-S viscous dissipation → decoupling ─────────────────────
    for t in range(PRE, n_steps):
        dt = t - PRE
        r_thermal       = max(0.30, 0.97 - 0.0022 * dt)
        decoupling_noise = (0.97 - r_thermal) * 0.12
        seq[t, CH["Pmp.TV"]] = (pmpTV_baseline
                                  + scada_noise("Pmp.TV", 1)[0]
                                  + float(_rng.normal(0, decoupling_noise)))

    # ── Pmp.PV* — axial thrust rises as pressure drops ─────────────────────
    for t in range(n_steps):
        pres_drop = max(0.0, presSV_baseline - pres_traj[t])
        seq[t, CH["Pmp.PV"]] = (pmpPV_baseline + 0.15 * severity * pres_drop
                                  + scada_noise("Pmp.PV", 1)[0])

    # ── Pmp.SV* — slight linear rise (M5: 0.12*sev*t/T) ───────────────────
    for t in range(n_steps):
        seq[t, CH["Pmp.SV"]] = (pmpSV_baseline
                                  + 0.12 * severity * (t / n_steps)
                                  + scada_noise("Pmp.SV", 1)[0])

    # Mot.TV* intentionally UNCHANGED — thermal decoupling r=-0.013 confirmed M5

    seq = apply_winsorization(seq, cluster_id)
    return seq.astype(np.float32)


def generate_overloading(severity: float = 0.5, cluster_id: int = 2,
                         n_steps: int = 300) -> np.ndarray:
    """
    Label 5 — overloading. LOCKED to startup cluster (cluster_id=2 in M6B).
    Physics: Cp·m·dT/dt = Q_friction − Q_ambient (first-order thermal)
             Temp.SV* monotonic rise (C-04 invariant — rate-of-change pattern)
             Mot.TV* coupled (r=0.997, M5 confirmed)
             Mot.SV*, Pmp.SV* STABLE — pure noise (C-04)
             Pres.SV* affinity law Q-H shift (F4 FIX — M5 canonical)
    """
    PRE   = 50
    ONSET = n_steps - PRE

    seq = make_baseline(n_steps, cluster_id=cluster_id)

    tempSV_baseline = get_cluster_mean(cluster_id, "Temp.SV", fallback=0.5)
    motTV_baseline  = get_cluster_mean(cluster_id, "Mot.TV",  fallback=0.4)
    motSV_baseline  = get_cluster_mean(cluster_id, "Mot.SV",  fallback=1.0)
    pmpSV_baseline  = get_cluster_mean(cluster_id, "Pmp.SV",  fallback=1.0)

    # ── Temp.SV* — first-order thermal response ───────────────────────────
    tau_steps      = ONSET / 3.0     # 95% of rise complete by end of onset
    delta_T        = 0.50 * severity
    T_target       = tempSV_baseline + delta_T
    T_current      = tempSV_baseline

    temp_traj = np.full(n_steps, tempSV_baseline, dtype=np.float64)
    for t in range(PRE, n_steps):
        T_current      += (T_target - T_current) / max(0.1, tau_steps)
        temp_traj[t]    = T_current

    for t in range(n_steps):
        seq[t, CH["Temp.SV"]] = float(temp_traj[t]) + scada_noise("Temp.SV", 1)[0]

    # ── Mot.TV* — coupled r=0.997, lag 5-15 steps ─────────────────────────
    mot_lag = int(_rng.integers(5, 16))
    for t in range(n_steps):
        t_src = max(0, t - mot_lag)
        seq[t, CH["Mot.TV"]] = (motTV_baseline
                                  + 0.94 * (temp_traj[t_src] - tempSV_baseline)
                                  + scada_noise("Mot.TV", 1)[0])

    # ── Mot.SV*, Pmp.SV* — STABLE (C-04 invariant) ────────────────────────
    for t in range(n_steps):
        seq[t, CH["Mot.SV"]] = motSV_baseline + scada_noise("Mot.SV", 1)[0]
        seq[t, CH["Pmp.SV"]] = pmpSV_baseline + scada_noise("Pmp.SV", 1)[0]

    # ── Pres.SV* — affinity law Q-H shift (F4 FIX — M5 canonical) ────────
    # M5: PresSV = (Q/Q_BEP)^2 * (1 - severity*0.1)
    # Q = Q_BEP * (1 + severity*0.25) at overload → operating point right on Q-H
    Q_ratio = 1.0 + severity * 0.25
    H_ratio = float(np.clip(Q_ratio**2 * (1.0 - severity * 0.10), 0.5, 2.0))
    for t in range(n_steps):
        seq[t, CH["Pres.SV"]] = H_ratio + scada_noise("Pres.SV", 1)[0]

    seq = apply_winsorization(seq, cluster_id)
    return seq.astype(np.float32)


def generate_sensor_failure(severity: float = 0.5, cluster_id: int = 1,
                             n_steps: int = 150, fail_type: str = None,
                             fail_channel: str = None):
    """
    Label 6 — sensor_failure.
    M5 canonical — 4 subtypes (F5 FIX: dropout added):
      flatline : channel holds last value (wire break / signal loss)
      spike    : random large excursions ±3-5σ (EMI / connector arc)
      drift    : monotonic calibration offset (thermocouple/transducer degradation)
      dropout  : channel → 0.0 (cable cut / SCADA I/O card failure)
    All 7 other channels remain at normal baseline.
    Clusters: all 4
    Returns: (seq, fail_type, fail_channel)
    """
    PRE = 50
    seq = make_baseline(n_steps, cluster_id=cluster_id)

    if fail_type is None:
        fail_type = _rng.choice(SENSOR_SUBTYPES)
    if fail_channel is None:
        fail_channel = _rng.choice(CHANNELS)

    ch_idx   = CH[fail_channel]
    baseline = get_cluster_mean(cluster_id, fail_channel, fallback=1.0)

    if fail_type == "flatline":
        # Wire break / SCADA signal loss → holds last value
        stuck_val = float(seq[PRE - 1, ch_idx])
        for t in range(PRE, n_steps):
            seq[t, ch_idx] = stuck_val + float(_rng.normal(0, 0.002))

    elif fail_type == "spike":
        # EMI / connector arc → random large excursions ±3-5σ
        sigma_ch = NOISE_STD.get(fail_channel, 0.015) * 10.0
        for t in range(PRE, n_steps):
            if _rng.random() < 0.30:
                spike = float(_rng.choice([-1.0, 1.0])) * float(_rng.uniform(3, 5)) * sigma_ch
                seq[t, ch_idx] = baseline + spike
            else:
                seq[t, ch_idx] = baseline + scada_noise(fail_channel, 1)[0]

    elif fail_type == "drift":
        # Calibration offset / thermocouple Seebeck degradation → monotonic drift
        # drift_rate calibrated: mean shift > 0.22 for all sev >= 0.3
        direction  = float(_rng.choice([-1.0, 1.0]))
        drift_rate = direction * severity * 0.016
        max_drift  = 0.75   # never hits winsorization ceiling
        for t in range(PRE, n_steps):
            dt      = t - PRE
            clamped = float(np.clip(drift_rate * dt, -max_drift, max_drift))
            seq[t, ch_idx] = baseline + clamped + scada_noise(fail_channel, 1)[0]

    elif fail_type == "dropout":
        # Cable cut / SCADA I/O card failure → hard zero (F5 FIX)
        # Physically distinct from flatline: dropout = absolute zero, not last value
        # On 40 bar pump: Pres.SV* dropout to 0.0 = safety-critical event
        on = int(_rng.integers(PRE, min(PRE + 20, n_steps - 10)))
        seq[on:, ch_idx] = 0.0 + float(_rng.normal(0, 0.002))

    seq[:, ch_idx] = np.clip(seq[:, ch_idx], 0.0, 8.8)
    seq = apply_winsorization(seq, cluster_id)
    return seq.astype(np.float32), fail_type, fail_channel


# ═══════════════════════════════════════════════════════════════════════════════
# NORMAL SEQUENCE HELPER (Label 0)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_normal_from_real(norm_df, n_target: int = 2000,
                               n_steps: int = 200,
                               m3_norm_cols: list = None) -> tuple:
    """
    Sample real CIRA windows from normalised_data.csv in M6B channel order.
    Falls back to physics baseline for any shortfall.
    Returns: (sequences_list, meta_list)
    """
    WIN = 50
    sequences, meta_list = [], []

    M3_NORM_COLS = m3_norm_cols or [
        "X_ACR_Mot.SV_norm", "X_ACR_Pmp.SV_norm", "X_ACR_Mot.TV_norm",
        "X_ACR_Pmp.PV_norm", "X_Temp.SV_norm",    "X_Pres.SV_norm",
        "X_ACR_Pmp.TV_norm", "X_ACR_Mot.PV_norm",
    ]

    seg_col = "segment_id" if "segment_id" in norm_df.columns else None
    missing = [c for c in M3_NORM_COLS if c not in norm_df.columns]

    if not missing and norm_df is not None:
        attempts = 0
        while len(sequences) < n_target and attempts < n_target * 20:
            attempts += 1
            try:
                if seg_col:
                    seg_id = _rng.choice(norm_df[seg_col].unique())
                    seg    = norm_df[norm_df[seg_col] == seg_id]
                else:
                    seg = norm_df
                if len(seg) < WIN:
                    continue
                start  = int(_rng.integers(0, len(seg) - WIN))
                window = seg.iloc[start:start+WIN][M3_NORM_COLS].values.astype(np.float32)
                seq    = np.zeros((n_steps, N_CH), dtype=np.float32)
                for r in range(n_steps // WIN):
                    noise_r = _rng.normal(0, 0.015 * 0.4, (WIN, N_CH)).astype(np.float32)
                    seq[r*WIN:(r+1)*WIN] = window + noise_r
                if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
                    continue
                seq = np.clip(seq, 0.0, 8.8)
                if "operating_mode" in seg.columns:
                    mode       = seg.iloc[start]["operating_mode"]
                    cluster_id = {"cooldown":0,"steady_state":1,
                                  "startup":2,"high_load":3}.get(mode, 1)
                elif "cluster_id" in seg.columns:
                    cluster_id = int(seg.iloc[start]["cluster_id"])
                else:
                    cluster_id = 1
                sequences.append(seq)
                meta_list.append({"cluster_id": cluster_id, "source": "real_cira"})
            except Exception:
                continue

    # Fill remainder with physics baseline
    cluster_cycle = [0, 1, 2, 3]
    while len(sequences) < n_target:
        cid = cluster_cycle[len(sequences) % 4]
        seq = make_baseline(n_steps, cluster_id=cid, noise_sigma=0.015)
        seq = apply_winsorization(seq, cid)
        sequences.append(seq.astype(np.float32))
        meta_list.append({"cluster_id": cid, "source": "physics_baseline"})

    return sequences[:n_target], meta_list[:n_target]