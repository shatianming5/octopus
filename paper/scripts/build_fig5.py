#!/usr/bin/env python3
"""
Paper-Native High-Typography Figure 5 (1:1 True-Print 6.50in Scale, Inter Typography):
  (a) Cross-Branch Read Pareto Frontier (Base 4B & RL LoRA, N=366 Mounts)
  (b) 1D RoPE Position Rules (Peer-Behind vs. Peer-Ahead EM)
  (c) Live Search Semantic Action Taxonomy (Mounted vs. Refused Blind Audit)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.scale import FuncScale

ROOT = Path(__file__).resolve().parent.parent
for ttf in (ROOT / "scripts/fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(ttf))

plt.rcParams.update({
    "font.family": "Inter",
    "axes.edgecolor": "#3F3F46",
    "axes.linewidth": 0.85,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": "#3F3F46",
    "ytick.color": "#3F3F46",
    "grid.color": "#E4E4E7",
    "grid.linestyle": "--",
    "grid.linewidth": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(6.50, 2.26), dpi=300)
outer = gridspec.GridSpec(
    1, 3, width_ratios=[1.25, 0.88, 0.88], wspace=0.31,
    left=0.068, right=0.992, top=0.815, bottom=0.225
)

# ==============================================================================
# PANEL (a): Cross-Branch Read Pareto Frontier (Base & RL Adapter, N=366 Mounts)
# ==============================================================================
gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.09)

base_data = [
    ("none",    0.3,  0.0,  0.8, 63.0, 57.1, 68.7, "#71717A", "o", 30,  2.2, 63.5, "left"),
    ("K1",      2.2,  0.8,  3.7, 94.6, 90.9, 97.9, "#3B6998", "s", 32,  5.6, 89.6, "left"),
    ("K3",      7.7,  4.9, 10.6, 97.3, 95.0, 99.2, "#C87524", "D", 32,  9.2, 93.4, "left"),
    ("K2",      3.8,  2.2,  5.9, 97.3, 95.0, 99.2, "#1D7A67", "*", 95,  4.2,100.3, "left"),
    ("replay", 56.0, 49.5, 62.4, 99.5, 98.3,100.0, "#7054B3", "^", 34, 46.0, 94.6, "right"),
    ("text",   57.9, 51.6, 64.0, 99.5, 98.3,100.0, "#C83E3E", "v", 34, 62.0,100.3, "right"),
]

rl_data = [
    ("none",    0.5,  0.0,  1.4, 62.5, 56.1, 68.6, "#71717A", "o", 30,  2.4, 63.0, "left"),
    ("K1",      2.5,  1.1,  4.1, 94.3, 90.5, 97.7, "#3B6998", "s", 32,  5.6, 89.6, "left"),
    ("K3",      7.1,  4.3, 10.1, 96.7, 93.9, 99.2, "#C87524", "D", 32,  8.8, 93.4, "left"),
    ("K2",      2.7,  1.1,  4.7, 96.5, 93.6, 99.0, "#1D7A67", "*", 95,  3.5,100.3, "left"),
    ("replay", 55.2, 48.9, 61.0, 99.2, 98.1,100.0, "#7054B3", "^", 34, 46.0, 94.6, "right"),
    ("text",   57.9, 51.6, 64.3, 99.5, 98.3,100.0, "#C83E3E", "v", 34, 62.0,100.3, "right"),
]

def fwd(x):
    return np.sqrt(np.maximum(x, 0.0))

def inv(y):
    return np.square(np.maximum(y, 0.0))

def fwd_y(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= 88.0, (y - 54.0) / 34.0 * 0.28, 0.28 + (y - 88.0) / 13.5 * 0.72)

def inv_y(u):
    u = np.asarray(u, dtype=float)
    return np.where(u <= 0.28, 54.0 + (u / 0.28) * 34.0, 88.0 + ((u - 0.28) / 0.72) * 13.5)

for idx, (sub_t, dataset) in enumerate([("Base (4B)", base_data), ("RL (LoRA)", rl_data)]):
    ax = fig.add_subplot(gs_a[idx])
    ax.set_xscale(FuncScale(ax, (fwd, inv)))
    ax.set_yscale(FuncScale(ax, (fwd_y, inv_y)))
    ax.set_xlim(0.0, 68.0)
    ax.set_ylim(54.0, 102.0)

    ax.axvspan(0.0, 5.0, ymin=float(fwd_y(90.0)), ymax=1.0, color="#D8F3EC", alpha=0.55, zorder=0)
    ax.axvline(5.0, color="#1D7A67", linestyle=":", linewidth=0.8, alpha=0.65, zorder=1)
    ax.grid(True, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for name, x, xlo, xhi, y, ylo, yhi, col, mk, sz, tx, ty, ha in dataset:
        ax.errorbar(
            x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
            fmt="none", ecolor=col, elinewidth=0.85, capsize=1.5, alpha=0.82, zorder=3
        )
        ax.scatter(
            [x], [y], s=sz, c=col, marker=mk,
            edgecolors="#FFFFFF" if mk != "*" else "#0C4A3E",
            linewidths=0.55 if mk != "*" else 0.75, zorder=4
        )
        is_oct = (name == "K2")
        lbl = "Octopus" if is_oct else name
        ax.annotate(
            lbl, xy=(x, y), xytext=(tx, ty), textcoords="data",
            ha=ha, va="center", fontsize=7.2 if is_oct else 6.8,
            fontweight="bold" if is_oct else 600, color="#0B4F41" if is_oct else col,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.65, alpha=0.7, shrinkA=1, shrinkB=2) if name != "none" else None,
            zorder=5
        )

    ax.set_xticks([0, 4, 20, 60])
    ax.set_xticklabels(["0", "4", "20", "60"], fontsize=7.0)
    ax.set_title(sub_t, fontsize=7.7, fontweight=600, pad=3, color="#27272A")
    ax.set_yticks([60, 80, 90, 95, 100])
    if idx == 0:
        ax.set_ylabel("4-Way Rank Acc. (%)", fontsize=7.9, fontweight=600, color="#18181B")
        ax.set_yticklabels(["60", "80", "90", "95", "100"], fontsize=7.1)
    else:
        ax.set_yticklabels([])
        ax.text(0.52, 0.21, "92.2% Δlogπ\n29× < ΔNLL",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6.6, fontweight="bold", color="#0B4F41",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#EEF8F5", edgecolor="#5BA392", lw=0.7))

fig.text(0.242, 0.935, "(a) Retrieval vs. Hijack (21× Drop)", ha="center", va="top",
         fontsize=8.6, fontweight="bold", color="#18181B")
fig.text(0.242, 0.055, "Task Hijack Rate (%) [-]", ha="center", va="bottom",
         fontsize=7.7, fontweight=600, color="#18181B")

# ==============================================================================
# PANEL (b): Peer-Behind vs. Peer-Ahead EM Retrieval & Hijack Asymmetry
# ==============================================================================
ax2 = fig.add_subplot(outer[1])
ax2.set_xlim(-0.52, 2.52)
ax2.set_ylim(0.0, 1.18)
ax2.grid(True, axis="y", zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

groups = ["Behind\n(48%)", "Ahead\n(52%)", "Pooled\n(947)"]
x_g = np.arange(len(groups))
w = 0.245

em_k1 = np.array([0.624, 0.459, 0.531])
em_k3 = np.array([0.588, 0.624, 0.608])
em_k2 = np.array([0.624, 0.630, 0.627])

ax2.axhline(0.760, color="#C83E3E", linestyle="--", linewidth=1.1,
            label="Text (0.76, 63% Hijack)", zorder=2)

ax2.bar(x_g - w, em_k1, width=w, color="#A6CCF0", edgecolor="#38628C", linewidth=0.65,
        label="K1: Unshifted", zorder=3)
ax2.bar(x_g,     em_k3, width=w, color="#F6CE95", edgecolor="#A35D22", linewidth=0.65,
        label="K3: RoPE Copy", zorder=3)
ax2.bar(x_g + w, em_k2, width=w, color="#1D7A67", edgecolor="#0C4A3E", linewidth=0.75,
        label="K2: Octopus (0-Copy)", zorder=4)

# Clean high-contrast value callouts on key bars without clutter
ax2.text(1.0 - w, 0.459 + 0.02, ".459", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#A62626")
ax2.text(1.0 + w, 0.630 + 0.02, ".630", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#0B4F41")
ax2.text(2.0 + w, 0.627 + 0.02, ".627", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#0B4F41")

ax2.annotate(
    "K1 Ahead\n-0.165 EM",
    xy=(1.0 - w, 0.44), xytext=(0.62, 0.19),
    ha="center", va="center", fontsize=6.7, fontweight="bold", color="#A62626",
    bbox=dict(boxstyle="round,pad=0.18", facecolor="#FEF2F2", edgecolor="#E69A9A", lw=0.7),
    arrowprops=dict(arrowstyle="->", color="#C83E3E", lw=0.85), zorder=5
)

ax2.set_xticks(x_g)
ax2.set_xticklabels(groups, fontsize=7.3, fontweight=600, color="#18181B")
ax2.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax2.set_yticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"], fontsize=7.1)
ax2.set_ylabel("Exact-Match (EM)", fontsize=7.9, fontweight=600, color="#18181B")
ax2.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#D4D4D8",
           fontsize=6.4, handlelength=1.15, handletextpad=0.4, borderpad=0.24, labelspacing=0.18)
fig.text(0.588, 0.935, "(b) 1D RoPE Position Rules", ha="center", va="top",
         fontsize=8.6, fontweight="bold", color="#18181B")

# ==============================================================================
# PANEL (c): Live Search Semantic Action Taxonomy (Mounted vs. Refused Audit)
# ==============================================================================
ax3 = fig.add_subplot(outer[2])
ax3.set_xlim(-0.55, 2.55)
ax3.set_ylim(0.0, 72.0)
ax3.grid(True, axis="y", zorder=0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

cats = ["On-Task\n(OWN)", "Hijack\n(S_TASK)", "Root\nClose"]
x_c = np.arange(len(cats))
wc = 0.29

v_ref = np.array([56.4, 32.7,  7.3])
v_mnt = np.array([48.3, 31.7, 18.3])

ax3.bar(x_c - wc/2, v_ref, width=wc, color="#E2E0DA", edgecolor="#6E6A63", linewidth=0.7,
        label="Refused (0 KV, n=55)", zorder=3)
ax3.bar(x_c + wc/2, v_mnt, width=wc, color="#1D7A67", edgecolor="#0C4A3E", linewidth=0.75,
        label="Mounted (K2, n=60)", zorder=4)

for i in range(len(cats)):
    dx_l = -0.06 if i == 1 else -0.02
    dx_r =  0.07 if i == 1 else  0.03
    ax3.text(x_c[i] - wc/2 + dx_l, v_ref[i] + 1.2, f"{v_ref[i]:.1f}", ha="center", va="bottom",
             fontsize=6.5, color="#52525B", fontweight=600)
    ax3.text(x_c[i] + wc/2 + dx_r, v_mnt[i] + 1.2, f"{v_mnt[i]:.1f}", ha="center", va="bottom",
             fontsize=6.7, color="#0B4F41", fontweight="bold")

ax3.annotate(
    "ΔHijack = -1.1%\n(0 Extra Drift)",
    xy=(1.0 + wc/2, 37.5), xytext=(1.42, 48.0),
    ha="center", va="center", fontsize=6.7, fontweight="bold", color="#0B4F41",
    bbox=dict(boxstyle="round,pad=0.18", facecolor="#EEF8F5", edgecolor="#5BA392", lw=0.7),
    arrowprops=dict(arrowstyle="->", color="#1D7A67", lw=0.85), zorder=5
)

ax3.set_xticks(x_c)
ax3.set_xticklabels(cats, fontsize=7.3, fontweight=600, color="#18181B")
ax3.set_yticks([0, 20, 40, 60])
ax3.set_yticklabels(["0", "20", "40", "60"], fontsize=7.1)
ax3.set_ylabel("Blind Audit Share (%)", fontsize=7.9, fontweight=600, color="#18181B")
ax3.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#D4D4D8",
           fontsize=6.4, handlelength=1.15, handletextpad=0.4, borderpad=0.24, labelspacing=0.20)
fig.text(0.865, 0.935, "(c) Live Semantic Audit", ha="center", va="top",
         fontsize=8.6, fontweight="bold", color="#18181B")

out_pdf = ROOT / "figures/fig5_systems_compute.pdf"
out_svg = ROOT / "figures/fig5_systems_compute.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
print("Saved true-print Figure 5:", out_pdf)
