# src/patch_m6a_plots.py
# Regenerates M6A plots from saved data — no rerunning of generator needed
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
from config import SYNTH_DIR, PLOTS_DIR, OUTPUT_DIR

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["Mot_PV","Mot_SV","Mot_TV","Pmp_PV","Pmp_SV","Pmp_TV","Temp_SV","Pres_SV"]
I = {ch: i for i, ch in enumerate(CHANNELS)}
FAULT_TYPES = ["bearing_wear","impeller_imbalance","cavitation",
               "seal_failure","overloading","sensor_failure"]

print("Loading saved sequences...")
with open(SYNTH_DIR / "M6_sequences.pkl", "rb") as f:
    sequences_arr = pickle.load(f)
meta_df = pd.read_csv(SYNTH_DIR / "M6_sequence_meta.csv")
print(f"  Loaded: {sequences_arr.shape}")

# ── PLOT 1: Label Distribution ────────────────────────────────
dist = meta_df["fault_type"].value_counts().to_dict()
labels_plot = ["normal"] + FAULT_TYPES
counts_plot = [dist.get(ft, 0) for ft in labels_plot]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ['#27ae60','#e74c3c','#c0392b','#2980b9','#8e44ad','#e67e22','#16a085']
bars = ax.bar(labels_plot, counts_plot, color=colors)
for bar, cnt in zip(bars, counts_plot):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            str(cnt), ha='center', fontsize=9, fontweight='bold')
ax.axhline(1200, color='k', linestyle='--', alpha=0.4, label='Target (1200)')
ax.set_title("M6A Label Distribution — 8400 Total Sequences", fontsize=12)
ax.set_ylabel("Sequence Count")
ax.set_ylim(0, 1400)
ax.legend()
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "M6A_label_distribution.png", dpi=150)
plt.close()
print("  ✅ Label distribution plot saved")

# ── PLOT 2: Fault Signature Grid ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
colors_ch = plt.cm.tab10(np.linspace(0, 1, 8))
for ax, ft in zip(axes.flat, FAULT_TYPES):
    ft_idx = np.where(meta_df["fault_type"].values == ft)[0]
    if len(ft_idx) == 0:
        ax.set_title(f"{ft}\nNO DATA"); continue
    seq = sequences_arr[ft_idx[0]]   # first sequence of this type
    for ci, ch in enumerate(CHANNELS):
        ax.plot(seq[:, ci], alpha=0.8, linewidth=1.0,
                label=ch, color=colors_ch[ci])
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.set_title(ft.replace("_"," ").title(), fontsize=10, fontweight='bold')
    ax.set_xlabel("Timestep (0–200)")
    ax.set_ylabel("Normalized Value")
    ax.set_ylim(-0.1, 4.0)
axes[0,0].legend(fontsize=6, ncol=2, loc='upper left')
plt.suptitle("M6A — One Sample per Fault Class (Physics Progression visible)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "M6A_fault_signatures_grid.png", dpi=150)
plt.close()
print("  ✅ Fault signatures grid saved")

# ── PLOT 3: Coupling Fidelity Scatter (THE MISSING ONE) ───────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("M6A Coupling Fidelity", fontsize=13, fontweight='bold')

coupling_configs = [
    ("bearing_wear",  "Mot_SV", "Mot_TV",  "#e74c3c", "r ≥ 0.70"),
    ("overloading",   "Mot_TV", "Temp_SV", "#e67e22", "r ≥ 0.85"),
    ("seal_failure",  "Pres_SV","Pmp_TV",  "#8e44ad", "informational"),
]

for ax, (ft, ch_x, ch_y, color, gate_label) in zip(axes, coupling_configs):
    ft_idx = np.where(meta_df["fault_type"].values == ft)[0]
    if len(ft_idx) == 0:
        ax.set_title(f"{ft}\nNO DATA"); continue

    # Sample up to 200 sequences for scatter
    sample_idx = ft_idx[:200]
    x_vals, y_vals, r_vals = [], [], []
    for idx in sample_idx:
        seq = sequences_arr[idx]
        x_mean = seq[:, I[ch_x]].mean()
        y_mean = seq[:, I[ch_y]].mean()
        r, _   = pearsonr(seq[:, I[ch_x]], seq[:, I[ch_y]])
        x_vals.append(x_mean)
        y_vals.append(y_mean)
        r_vals.append(r)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    r_vals = np.array(r_vals)
    mean_r = r_vals.mean()

    sc = ax.scatter(x_vals, y_vals, c=r_vals, cmap='RdYlGn',
                    vmin=-1, vmax=1, alpha=0.75, s=30, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Pearson r')
    ax.set_xlabel(f"{ch_x} (mean per seq)")
    ax.set_ylabel(f"{ch_y} (mean per seq)")
    ax.set_title(f"{ft}\n{ch_x}↔{ch_y}  |  mean r={mean_r:.3f}  ({gate_label})",
                 fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "M6A_coupling_fidelity.png", dpi=150)
plt.close()
print("  ✅ Coupling fidelity scatter saved (FIXED)")

# ── PLOT 4: Severity Distribution ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, ft in zip(axes.flat, FAULT_TYPES):
    ft_df = meta_df[meta_df["fault_type"] == ft]
    if len(ft_df) == 0:
        continue
    ax.hist(ft_df["severity"].values, bins=30, color='#3498db',
            edgecolor='white', linewidth=0.5)
    ax.set_title(ft.replace("_"," ").title(), fontsize=9)
    ax.set_xlabel("Severity"); ax.set_ylabel("Count")
    ax.axvline(ft_df["severity"].mean(), color='red',
               linestyle='--', linewidth=1.2,
               label=f'mean={ft_df["severity"].mean():.2f}')
    ax.legend(fontsize=7)
plt.suptitle("M6A — Severity Distribution per Fault Class", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "M6A_severity_distribution.png", dpi=150)
plt.close()
print("  ✅ Severity distribution saved (BONUS plot)")

print(f"\nAll plots saved to: {PLOTS_DIR}")
print("Upload to Spaces: M6A_label_distribution.png, M6A_fault_signatures_grid.png,")
print("                  M6A_coupling_fidelity.png, M6A_severity_distribution.png")