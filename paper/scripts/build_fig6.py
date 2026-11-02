#!/usr/bin/env python3
"""
Paper-Native High-Typography Figure 6 (1:1 True-Print 6.50in Scale, Inter Typography, 2x2 Grid):
  (a) Depth-1 (D_max=1, 55 steps, 25,812 eps): Fork Rate vs. Root Rewinds
  (b) Unlimited-Depth (D_max=inf, branch<=10): Three Closing Contract Regimes (33,075 eps)
  (c) Unlimited-Depth (D_max=inf, 41 steps, 20,098 eps): Deep Trees (d>=3) & Reproduction Number mu
  (d) Unlimited-Depth (D_max=inf, 41 steps): Root vs. Branch Rewind Crossover
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

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

def smooth(y, box=3):
    out = np.copy(y)
    for i in range(len(y)):
        lo, hi = max(0, i - box // 2), min(len(y), i + box // 2 + 1)
        out[i] = np.mean(y[lo:hi])
    return out

fig = plt.figure(figsize=(6.50, 2.94), dpi=300)
gs = gridspec.GridSpec(
    2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
    wspace=0.47, hspace=0.58, left=0.072, right=0.935, top=0.915, bottom=0.115
)

# ==============================================================================
# DATA 1: v6d (D_max=1, 55 steps, 25,812 graded episodes)
# ==============================================================================
steps_v6d = np.arange(1, 56, dtype=float)
fork_v6d = np.array([
    0.27, 0.34, 0.28, 0.42, 0.36, 0.47, 0.48, 0.43, 0.36, 0.29,
    0.24, 0.24, 0.26, 0.27, 0.34, 0.38, 0.31, 0.42, 0.46, 0.46,
    0.56, 0.63, 0.67, 0.76, 0.82, 0.74, 0.83, 0.88, 0.89, 0.90,
    0.90, 0.91, 0.91, 0.93, 0.92, 0.96, 0.95, 0.94, 0.94, 0.95,
    0.92, 0.99, 0.94, 0.97, 0.98, 0.95, 0.97, 0.96, 0.98, 0.99,
    0.95, 0.98, 0.98, 0.99, 0.98
]) * 100.0

main_rw_v6d = np.array([
    0.360, 0.439, 0.274, 0.467, 0.490, 0.444, 0.312, 0.364, 0.501, 0.481,
    0.335, 0.613, 0.338, 0.431, 0.426, 0.393, 0.464, 0.402, 0.418, 0.518,
    0.416, 0.438, 0.336, 0.332, 0.285, 0.377, 0.218, 0.365, 0.341, 0.277,
    0.381, 0.288, 0.236, 0.253, 0.292, 0.410, 0.454, 0.182, 0.251, 0.243,
    0.203, 0.181, 0.199, 0.092, 0.093, 0.145, 0.096, 0.079, 0.075, 0.077,
    0.044, 0.018, 0.073, 0.038, 0.026
])

br_rw_v6d = np.array([
    0.272, 0.416, 0.372, 0.364, 0.418, 0.539, 0.546, 0.527, 0.566, 0.610,
    0.411, 0.599, 0.309, 0.632, 0.545, 0.498, 0.487, 0.420, 0.520, 0.493,
    0.548, 0.618, 0.530, 0.629, 0.503, 0.602, 0.382, 0.625, 0.630, 0.405,
    0.542, 0.456, 0.590, 0.558, 0.524, 0.608, 0.749, 0.465, 0.499, 0.490,
    0.545, 0.519, 0.562, 0.396, 0.298, 0.432, 0.378, 0.308, 0.362, 0.329,
    0.332, 0.241, 0.294, 0.248, 0.239
])

# ==============================================================================
# DATA 2: Unlimited-Depth Trials v7 (12 steps), v8 (17 steps), v9 (41 steps)
# ==============================================================================
steps_v7 = np.arange(1, 13, dtype=float)
notool_v7 = np.array([6, 8, 4, 9, 10, 16, 26, 36, 53, 55, 60, 75], dtype=float)

steps_v8 = np.arange(1, 18, dtype=float)
fork_v8 = np.array([
    23.9, 28.2, 19.6, 14.0, 8.9, 6.0, 2.9, 0.5, 2.2, 1.9,
    1.3, 2.4, 0.9, 2.0, 0.0, 1.3, 1.2
], dtype=float)

steps_v9 = np.arange(1, 42, dtype=float)
fork_v9 = np.array([
    22.8, 25.4, 17.4, 16.9, 22.2, 19.0, 15.3, 25.7, 20.1, 18.7,
    19.6, 24.0, 25.3, 20.4, 35.6, 35.9, 48.9, 43.2, 53.7, 49.7,
    59.5, 60.7, 60.9, 74.7, 67.1, 78.1, 81.1, 79.6, 81.4, 86.2,
    91.9, 91.6, 94.0, 94.5, 95.4, 97.1, 95.8, 96.7, 98.0, 98.0, 99.2
], dtype=float)

notool_v9 = np.array([
    0.6, 0.4, 1.2, 1.0, 0.9, 0.7, 3.6, 1.6, 1.0, 1.6,
    1.9, 1.8, 2.1, 0.0, 1.1, 1.2, 1.0, 2.7, 2.1, 1.7,
    1.0, 1.4, 1.4, 1.1, 1.1, 1.5, 0.2, 0.2, 0.0, 0.3,
    1.0, 0.2, 0.3, 0.2, 0.6, 0.2, 0.0, 0.2, 0.3, 0.3, 0.2
], dtype=float)

mu_v9 = np.array([
    0.471, 0.573, 0.485, 0.439, 0.456, 0.442, 0.522, 0.510, 0.455, 0.439,
    0.352, 0.357, 0.369, 0.319, 0.386, 0.433, 0.341, 0.368, 0.357, 0.464,
    0.394, 0.441, 0.432, 0.437, 0.441, 0.522, 0.512, 0.561, 0.549, 0.557,
    0.578, 0.580, 0.528, 0.649, 0.611, 0.600, 0.622, 0.692, 0.621, 0.670, 0.650
], dtype=float)

cap10_v9 = np.array([
    2.6, 6.2, 2.3, 1.3, 2.0, 1.9, 1.5, 5.8, 2.8, 2.5,
    1.4, 2.4, 1.4, 1.3, 2.9, 3.5, 3.5, 4.5, 5.3, 8.9,
    4.7, 8.9, 6.5, 11.0, 7.4, 15.3, 13.9, 17.8, 16.8, 20.9,
    21.9, 25.3, 16.8, 37.0, 30.2, 28.0, 30.9, 42.4, 39.9, 41.0, 38.4
], dtype=float)

depth3_v9 = np.array([
    5.8, 11.3, 3.9, 3.2, 5.2, 3.3, 3.1, 7.8, 4.2, 3.1,
    2.9, 3.4, 4.2, 1.3, 5.7, 8.1, 6.4, 6.3, 7.8, 11.4,
    8.7, 13.5, 11.8, 16.3, 11.8, 20.1, 22.9, 24.7, 25.3, 28.1,
    35.5, 31.6, 24.9, 44.6, 40.1, 36.5, 40.9, 51.4, 41.3, 49.7, 45.3
], dtype=float)

rw_main_v9 = np.array([
    22.3, 28.0, 22.5, 29.3, 27.1, 24.8, 25.5, 27.6, 27.3, 27.0,
    27.8, 25.6, 27.4, 26.3, 22.2, 24.5, 24.1, 23.9, 15.9, 23.7,
    20.1, 21.9, 16.3, 17.6, 17.6, 25.1, 18.0, 22.7, 13.6, 17.8,
    15.4, 12.2, 12.6, 21.3, 14.1, 11.6, 10.2, 18.9, 14.5, 14.9, 15.5
], dtype=float)

rw_br_v9 = np.array([
    4.4, 6.9, 3.2, 3.8, 4.0, 4.4, 2.6, 11.3, 6.6, 5.8,
    6.9, 5.0, 10.5, 7.2, 12.6, 14.1, 19.9, 14.5, 20.8, 18.6,
    16.1, 17.3, 21.2, 28.2, 21.9, 32.6, 29.3, 34.6, 25.9, 33.2,
    37.6, 31.9, 27.3, 44.3, 34.9, 32.4, 35.1, 45.1, 43.9, 40.6, 38.4
], dtype=float)

rw_any_v9 = np.array([
    25.1, 33.3, 24.5, 31.5, 29.7, 26.0, 28.1, 34.2, 31.1, 31.6,
    32.1, 28.0, 35.4, 32.2, 30.6, 34.5, 39.2, 34.4, 33.2, 35.4,
    31.4, 35.2, 32.8, 38.4, 34.2, 45.8, 39.0, 48.1, 31.4, 41.0,
    44.1, 35.7, 30.6, 49.1, 38.3, 36.0, 37.1, 48.4, 46.4, 42.6, 40.4
], dtype=float)

# ==============================================================================
# PANEL (a): Depth-1 Fork Rate vs. Root Rewinds
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 56.5)
ax1.set_ylim(0, 112)
ax1.grid(True, zorder=0)
ax1.spines["top"].set_visible(False)

ax1.plot(steps_v6d, fork_v6d, color="#1D7A67", linewidth=0.7, alpha=0.22, zorder=2)
l1a, = ax1.plot(steps_v6d, smooth(fork_v6d, 3), color="#1D7A67", linestyle="-", linewidth=1.9,
                label="Fork Rate (27%→98%)", zorder=4)
ax1.fill_between(steps_v6d, 0, smooth(fork_v6d, 3), color="#1D7A67", alpha=0.08, zorder=1)

ax1_r = ax1.twinx()
ax1_r.set_ylim(0.0, 0.72)
ax1_r.spines["top"].set_visible(False)
ax1_r.plot(steps_v6d, main_rw_v6d, color="#C83E3E", linewidth=0.65, alpha=0.20, zorder=2)
l2a, = ax1_r.plot(steps_v6d, smooth(main_rw_v6d, 5), color="#C83E3E", linestyle="-", linewidth=1.65,
                  label="Root Rewind/Ep (15×↓)", zorder=4)
l3a, = ax1_r.plot(steps_v6d, smooth(br_rw_v6d, 5), color="#C87524", linestyle="--", linewidth=1.35,
                  label="Branch Rewind/Br", zorder=3)

ax1.set_xticks([1, 15, 30, 45, 55])
ax1.set_xticklabels(["1", "15", "30", "45", "55"], fontsize=7.2)
ax1.set_yticks([0, 50, 100])
ax1.set_yticklabels(["0", "50", "100"], fontsize=7.2, color="#0B4F41")
ax1_r.set_yticks([0.0, 0.3, 0.6])
ax1_r.set_yticklabels(["0.0", "0.3", "0.6"], fontsize=7.2, color="#A62626")
ax1.set_xlabel("Training Step (Depth-1, 25.8K Eps)", fontsize=7.6, fontweight=600, labelpad=2.5)
ax1.set_ylabel("Fork Rate (%)", fontsize=7.8, fontweight=600, color="#0B4F41")
ax1_r.set_ylabel("Rewinds / Thread", fontsize=7.6, fontweight=600, color="#A62626")
ax1.set_title("(a) Depth-1 Fork vs. Root Rewinds", fontsize=8.4, fontweight="bold", pad=5, color="#18181B")
ax1.legend([l1a, l2a, l3a], [l1a.get_label(), l2a.get_label(), l3a.get_label()],
           loc="lower left", bbox_to_anchor=(0.015, 0.02), frameon=True, facecolor="#FFFFFF",
           edgecolor="#D4D4D8", fontsize=6.4, handlelength=1.25, borderpad=0.24, labelspacing=0.18)

# ==============================================================================
# PANEL (b): Unlimited-Depth Closing Contract Regimes
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 42.5)
ax2.set_ylim(0, 148)
ax2.grid(True, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2.plot(steps_v9, fork_v9, color="#1D7A67", linewidth=0.7, alpha=0.22, zorder=2)
l1b, = ax2.plot(steps_v9, smooth(fork_v9, 3), color="#1D7A67", linestyle="-", linewidth=1.95,
                label="Ours Fork (23%→99.2%)", zorder=5)
ax2.fill_between(steps_v9, 0, smooth(fork_v9, 3), color="#1D7A67", alpha=0.08, zorder=1)

l2b, = ax2.plot(steps_v9, smooth(notool_v9, 3), color="#2A9D8F", linestyle=":", linewidth=1.35,
                label="Ours No-Tool (0.2%)", zorder=4)
l3b, = ax2.plot(steps_v7, notool_v7, color="#C87524", linestyle="--", marker="o", markersize=2.6,
                linewidth=1.5, label="Fallback No-Tool (75% @12)", zorder=4)
ax2.scatter([12], [75.0], color="#A35D22", marker="X", s=38, edgecolor="#18181B", linewidth=0.6, zorder=6)

l4b, = ax2.plot(steps_v8, fork_v8, color="#C83E3E", linestyle="-.", marker="v", markersize=2.6,
                linewidth=1.5, label="No-Remind Fork (1% @17)", zorder=4)
ax2.scatter([17], [1.2], color="#C83E3E", marker="X", s=38, edgecolor="#18181B", linewidth=0.6, zorder=6)

ax2.text(29.5, 29.0, "97.4% Reminded\nRoots → done", ha="center", va="center",
         fontsize=6.5, fontweight="bold", color="#0B4F41",
         bbox=dict(boxstyle="round,pad=0.20", facecolor="#EEF8F5", edgecolor="#5BA392", lw=0.7))

ax2.set_xticks([1, 12, 17, 30, 41])
ax2.set_xticklabels(["1", "12", "17", "30", "41"], fontsize=7.2)
ax2.set_yticks([0, 50, 100])
ax2.set_yticklabels(["0", "50", "100"], fontsize=7.2)
ax2.set_xlabel("Training Step (Unlim. Depth, ≤10 Br.)", fontsize=7.6, fontweight=600, labelpad=2.5)
ax2.set_ylabel("Episode Rate (%)", fontsize=7.8, fontweight=600, color="#18181B")
ax2.set_title("(b) Closing Contract Regimes", fontsize=8.4, fontweight="bold", pad=5, color="#18181B")
ax2.legend([l1b, l2b, l3b, l4b], [l1b.get_label(), l2b.get_label(), l3b.get_label(), l4b.get_label()],
           loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True, facecolor="#FFFFFF",
           edgecolor="#D4D4D8", fontsize=6.1, handlelength=1.2, borderpad=0.22, labelspacing=0.15, framealpha=0.94)

# ==============================================================================
# PANEL (c): Multi-Hop Depth (d>=3) & Sub-Critical Reproduction mu
# ==============================================================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlim(0, 42.5)
ax3.set_ylim(0, 62)
ax3.grid(True, zorder=0)
ax3.spines["top"].set_visible(False)

ax3.plot(steps_v9, depth3_v9, color="#1D7A67", linewidth=0.7, alpha=0.22, zorder=2)
l1c, = ax3.plot(steps_v9, smooth(depth3_v9, 3), color="#1D7A67", linestyle="-", linewidth=1.9,
                label="Depth d ≥ 3 (51.4%)", zorder=4)
ax3.fill_between(steps_v9, 0, smooth(depth3_v9, 3), color="#1D7A67", alpha=0.09, zorder=1)

ax3.plot(steps_v9, cap10_v9, color="#3B6998", linewidth=0.7, alpha=0.22, zorder=2)
l2c, = ax3.plot(steps_v9, smooth(cap10_v9, 3), color="#3B6998", linestyle="--", linewidth=1.6,
                label="10-Branch Cap (42.4%)", zorder=4)

ax3_r = ax3.twinx()
ax3_r.set_ylim(0.0, 1.15)
ax3_r.spines["top"].set_visible(False)
ax3_r.axhline(1.0, color="#C83E3E", linestyle=":", linewidth=1.05, zorder=2)
ax3_r.text(1.5, 1.025, "Critical μ = 1.0", fontsize=6.5, fontweight="bold", color="#A62626")

ax3_r.plot(steps_v9, mu_v9, color="#7054B3", linewidth=0.65, alpha=0.22, zorder=2)
l3c, = ax3_r.plot(steps_v9, smooth(mu_v9, 3), color="#7054B3", linestyle="-.", linewidth=1.6,
                  label="Sub-Branch μ (< 0.69)", zorder=3)

ax3.set_xticks([1, 10, 20, 30, 41])
ax3.set_xticklabels(["1", "10", "20", "30", "41"], fontsize=7.2)
ax3.set_yticks([0, 30, 60])
ax3.set_yticklabels(["0", "30", "60"], fontsize=7.2, color="#0B4F41")
ax3_r.set_yticks([0.0, 0.5, 1.0])
ax3_r.set_yticklabels(["0.0", "0.5", "1.0"], fontsize=7.2, color="#583E99")
ax3.set_xlabel("Training Step (Unlim. Depth, 20.1K Eps)", fontsize=7.6, fontweight=600, labelpad=2.5)
ax3.set_ylabel("Tree Share (%)", fontsize=7.8, fontweight=600, color="#0B4F41")
ax3_r.set_ylabel("Branch Ratio μ", fontsize=7.6, fontweight=600, color="#583E99")
ax3.set_title("(c) Deep Trees (d ≥ 3) & Sub-Critical μ", fontsize=8.4, fontweight="bold", pad=5, color="#18181B")
ax3.legend([l1c, l2c, l3c], [l1c.get_label(), l2c.get_label(), l3c.get_label()],
           loc="upper left", bbox_to_anchor=(0.01, 0.86), frameon=True, facecolor="#FFFFFF",
           edgecolor="#D4D4D8", fontsize=6.3, handlelength=1.25, borderpad=0.22, labelspacing=0.16, framealpha=0.94)

# ==============================================================================
# PANEL (d): Root vs. Branch Rewind Share Crossover
# ==============================================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlim(0, 42.5)
ax4.set_ylim(0, 58)
ax4.grid(True, zorder=0)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

ax4.plot(steps_v9, rw_br_v9, color="#1D7A67", linewidth=0.7, alpha=0.22, zorder=2)
l1d, = ax4.plot(steps_v9, smooth(rw_br_v9, 3), color="#1D7A67", linestyle="-", linewidth=1.95,
                label="Branch Rewind (3.2%→45.1%)", zorder=5)
ax4.fill_between(steps_v9, 0, smooth(rw_br_v9, 3), color="#1D7A67", alpha=0.08, zorder=1)

ax4.plot(steps_v9, rw_main_v9, color="#C83E3E", linewidth=0.7, alpha=0.22, zorder=2)
l2d, = ax4.plot(steps_v9, smooth(rw_main_v9, 3), color="#C83E3E", linestyle="-", linewidth=1.7,
                label="Root Rewind (29.3%→10.2%)", zorder=4)

ax4.plot(steps_v9, rw_any_v9, color="#71717A", linewidth=0.65, alpha=0.20, zorder=2)
l3d, = ax4.plot(steps_v9, smooth(rw_any_v9, 3), color="#52525B", linestyle="--", linewidth=1.25,
                label="Any Rewind (25.1%→48.4%)", zorder=3)

ax4.scatter([19], [19.2], color="#C87524", s=34, edgecolor="#18181B", linewidth=0.8, zorder=6)
ax4.annotate(
    "Step 19 Crossover",
    xy=(19, 19.2), xytext=(28.5, 6.2),
    ha="center", va="center", fontsize=6.5, fontweight="bold", color="#8C4A15",
    bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFF8EB", edgecolor="#D89B52", lw=0.7),
    arrowprops=dict(arrowstyle="->", color="#C87524", lw=0.85), zorder=6
)

ax4.set_xticks([1, 10, 19, 30, 41])
ax4.set_xticklabels(["1", "10", "19", "30", "41"], fontsize=7.2)
ax4.set_yticks([0, 20, 40])
ax4.set_yticklabels(["0", "20", "40"], fontsize=7.2)
ax4.set_xlabel("Training Step (Unlim. Depth, ≤10 Br.)", fontsize=7.6, fontweight=600, labelpad=2.5)
ax4.set_ylabel("Rewind Share (%)", fontsize=7.8, fontweight=600, color="#18181B")
ax4.set_title("(d) Root vs. Branch Rewind Share", fontsize=8.4, fontweight="bold", pad=5, color="#18181B")
ax4.legend([l1d, l2d, l3d], [l1d.get_label(), l2d.get_label(), l3d.get_label()],
           loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True, facecolor="#FFFFFF",
           edgecolor="#D4D4D8", fontsize=6.3, handlelength=1.25, borderpad=0.22, labelspacing=0.16, framealpha=0.94)

out_pdf = ROOT / "figures/fig6_pareto_rl.pdf"
out_svg = ROOT / "figures/fig6_pareto_rl.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
print("Saved true-print Figure 6:", out_pdf)
