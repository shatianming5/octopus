#!/usr/bin/env python3
"""
Paper-Native High-Typography Figure 4 (1:1 True-Print 6.50in Scale, Inter Typography):
  (a) BrowseComp-Plus (830 Queries) Evidence Recall Funnel & End-to-End Accuracy
  (b) OpenResearcher Validation (500 Queries) Checkpoint Scaling Curve
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
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": "#3F3F46",
    "ytick.color": "#3F3F46",
    "grid.color": "#E4E4E7",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Exact 1:1 LaTeX \textwidth (6.50 in) so fontsize=8.5 prints at true 8.5pt on paper!
fig = plt.figure(figsize=(6.50, 2.34), dpi=300)
gs = gridspec.GridSpec(
    1, 2, width_ratios=[1.05, 0.95], wspace=0.25,
    left=0.072, right=0.988, top=0.875, bottom=0.205
)

# ==============================================================================
# PANEL (a): BrowseComp-Plus (830 Queries) Retrieval Funnel & Final Accuracy
# ==============================================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(-0.52, 3.54)
ax1.set_ylim(0, 77.0)
ax1.grid(True, axis="y", zorder=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

stages = [
    "Search\nRecall",
    "Opened\nRecall",
    "Gold Doc\nOpened",
    "Final\nAcc. (%)",
]
x_pos = np.arange(len(stages))
w = 0.255

v_base  = np.array([25.9,  6.4,  9.7, 24.6])
v_linrl = np.array([30.5, 14.6, 21.1, 32.1])
v_oct   = np.array([49.5, 29.0, 37.7, 42.6])

ax1.bar(x_pos - w, v_base, width=w, color="#E2E0DA", edgecolor="#6E6A63", linewidth=0.7,
        label="Base Linear-Big (4B, 100 Calls)", zorder=3)
ax1.bar(x_pos, v_linrl, width=w, color="#F6CE95", edgecolor="#A35D22", linewidth=0.7,
        label="Linear-Big RL (4B, 143K Cap)", zorder=3)
ax1.bar(x_pos + w, v_oct, width=w, color="#1D7A67", edgecolor="#0C4A3E", linewidth=0.75,
        label="Octopus RL (4B, 32K/Branch)", zorder=4)

# External 9B CLM+SCR RL reference line on Final Accuracy (42.5%)
ax1.plot([2.56, 3.44], [42.5, 42.5], color="#C83E3E", linestyle="--", linewidth=1.25, zorder=5)
ax1.text(2.85, 48.2, "9B CLM+SCR (42.5%)", ha="center", va="bottom",
         fontsize=6.9, fontweight="bold", color="#A62626",
         bbox=dict(boxstyle="round,pad=0.20", facecolor="#FEF2F2", edgecolor="#F3B4B4", lw=0.7))

for i in range(len(stages)):
    ax1.text(x_pos[i] - w - 0.02, v_base[i] + 1.1, f"{v_base[i]:.1f}", ha="center", va="bottom",
             fontsize=6.6, fontweight=500, color="#52525B")
    ax1.text(x_pos[i], v_linrl[i] + 1.1, f"{v_linrl[i]:.1f}", ha="center", va="bottom",
             fontsize=6.6, fontweight=600, color="#8C4A15")
    y_oct_lbl = v_oct[i] + (1.8 if i == 3 else 1.2)
    ax1.text(x_pos[i] + w + 0.03, y_oct_lbl, f"{v_oct[i]:.1f}", ha="center", va="bottom",
             fontsize=7.1, fontweight="bold", color="#0B4F41")

ax1.set_xticks(x_pos)
ax1.set_xticklabels(stages, fontsize=7.5, fontweight=600, color="#18181B")
ax1.set_yticks([0, 15, 30, 45, 60, 75])
ax1.set_yticklabels(["0", "15", "30", "45", "60", "75"], fontsize=7.4)
ax1.set_ylabel("BrowseComp-Plus (%)", fontsize=8.1, fontweight=600, color="#18181B")
ax1.set_title("(a) BrowseComp-Plus Funnel (830Q)", fontsize=8.8, fontweight="bold", pad=6, color="#18181B")
ax1.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#D4D4D8",
           fontsize=6.8, handlelength=1.25, handletextpad=0.45, borderpad=0.30, labelspacing=0.25)

# ==============================================================================
# PANEL (b): OpenResearcher Validation (500 Queries) Checkpoint Trajectory
# ==============================================================================
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(-2.0, 58.0)
ax2.set_ylim(38.5, 53.8)
ax2.grid(True, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

steps_v6d = np.array([0, 10, 20, 30, 40, 55], dtype=float)
acc_v6d   = np.array([41.6, 46.2, 48.2, 48.4, 49.2, 49.6])
ci_v6d    = np.array([1.5,  1.5,  1.6,  1.7,  1.6,  1.6])

steps_v5  = np.array([0, 30, 40, 50, 55], dtype=float)
acc_v5    = np.array([41.6, 43.4, 45.3, 45.3, 45.2])

ax2.axhline(40.0, color="#71717A", linestyle=":", linewidth=1.15,
            label="Base Linear (40.0%)", zorder=2)
ax2.axhline(43.2, color="#C87524", linestyle="-.", linewidth=1.25,
            label="143K Linear RL (43.2%)", zorder=2)

ax2.plot(steps_v5, acc_v5, color="#7054B3", linestyle="--", marker="s", markersize=3.8,
         linewidth=1.55, label="Process Gate (45.2%)", zorder=3)

ax2.plot(steps_v6d, acc_v6d, color="#1D7A67", linestyle="-", marker="o", markersize=4.2,
         linewidth=2.0, label="Pure Outcome (49.6%)", zorder=4)
ax2.fill_between(steps_v6d, acc_v6d - ci_v6d, acc_v6d + ci_v6d, color="#1D7A67", alpha=0.14, zorder=2)

ax2.scatter([55.0], [51.2], color="#1D7A67", marker="*", s=115, edgecolors="#0C4A3E", linewidths=0.8,
            label="Final Eval (51.2%)", zorder=5)

ax2.annotate(
    "51.2% (+11.2 pts)",
    xy=(54.5, 51.25), xytext=(33.0, 52.1),
    ha="center", va="center", fontsize=7.2, fontweight="bold", color="#0B4F41",
    bbox=dict(boxstyle="round,pad=0.22", facecolor="#EEF8F5", edgecolor="#5BA392", lw=0.8),
    arrowprops=dict(arrowstyle="->", color="#1D7A67", lw=0.9), zorder=5
)

ax2.set_xticks([0, 10, 20, 30, 40, 55])
ax2.set_xticklabels(["0", "10", "20", "30", "40", "55"], fontsize=7.4)
ax2.set_yticks([40, 44, 48, 52])
ax2.set_yticklabels(["40", "44", "48", "52"], fontsize=7.4)
ax2.set_xlabel("DAG-GRPO Step (NVIDIA H200)", fontsize=8.0, fontweight=600, color="#18181B", labelpad=3)
ax2.set_ylabel("Val Accuracy (%)", fontsize=8.1, fontweight=600, color="#18181B")
ax2.set_title("(b) OpenResearcher Val Scaling (500Q)", fontsize=8.8, fontweight="bold", pad=6, color="#18181B")
ax2.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#D4D4D8",
           fontsize=6.7, handlelength=1.35, handletextpad=0.45, borderpad=0.28, labelspacing=0.22)

out_pdf = ROOT / "figures/fig4_mechanism_pressure.pdf"
out_svg = ROOT / "figures/fig4_mechanism_pressure.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
print("Saved true-print Figure 4:", out_pdf)
