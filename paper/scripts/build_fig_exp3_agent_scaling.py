#!/usr/bin/env python3
"""
Publication-Grade Asymmetric Emergence Figure 10 (Section 4.3):
  - Matches Figure 8 (build_fig_exp1_pareto.py) & Figure 9 (build_fig_exp2_efficiency.py) visual grammar 1:1:
      * Strict 3-family editorial palette:
          1. Monochromatic Octopus Teal Ramp (#065F46 -> #0D9488 -> #38A398 -> #78C2B8) + Steel-Petrol (#2B5B84)
          2. Muted Burnt Terracotta (#C2410C / #B91C1C / #FCEAE6 blush) for baselines, failure & root detour burden
          3. Cool Slate (#0F172A / #334155 / #64748B / #EEF2F6 / #F8FAFC) for typography & structure
      * Asymmetric Master Composition:
          - Left (41% width, full height): Hero Panel (a) Collective Scaling & Causal Fork Knockout
          - Right (59% width, 3-tier shared-X stack + dedicated Direct Tail-Label margin):
              (b) Step-15 Recursive Phase Transition vs. Collapse
              (c) Zero-Reward Commander & Peer KV Mesh
              (d) Depth-Cascaded Backtracking Shield
      * Zero boxed legends, zero dual-axis spine clutter, zero UI sticker pills, strict 3-tier typography,
        and exhaustive pixel-level zero-overlap geometry audit (including vertical phase-transition spines).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

ROOT = Path(__file__).resolve().parent.parent
for ttf in (ROOT / "scripts/fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(ttf))

plt.rcParams.update({
    "font.family": "Inter",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.85,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Unified Editorial Color System (1:1 with Figure 8 & Figure 9)
TEAL_DEEP = "#065F46"   # Primary deep emerald-teal (d>=3 / Turn-1 / d>=1 / Val)
TEAL_MAIN = "#0D9488"   # Signature Octopus teal (BCP / d>=4 / Worker Reads / d>=2)
TEAL_MID  = "#38A398"   # Medium seafoam teal (d>=5 / d>=3)
TEAL_SOFT = "#78C2B8"   # Soft mist teal (d>=4 sub-workers)
PETROL    = "#2B5B84"   # Deep steel-petrol (Peer KV Mesh)
TERRA     = "#C2410C"   # Muted terracotta (1-Agent GRPO / Naive RL / Root burden)
TERRA_KO  = "#B91C1C"   # Deep crimson-terracotta (Same-weight Fork-Off knockout)
SLATE_900 = "#0F172A"
SLATE_700 = "#334155"
SLATE_600 = "#475569"
SLATE_500 = "#64748B"
GRID_COL  = "#EEF2F6"
ZONE_BG   = "#F8FAFC"


def smooth_preserve_ends(y):
    """3-point [0.25, 0.50, 0.25] interior smoothing preserving exact Step-1 & Step-43 endpoints."""
    out = np.copy(y)
    for i in range(1, len(y) - 1):
        out[i] = 0.25 * y[i - 1] + 0.50 * y[i] + 0.25 * y[i + 1]
    return out


def solve_tail_positions(targets, min_gap, y_lo, y_hi):
    """Deterministic repulsion solver keeping direct tail labels strictly separated within [y_lo, y_hi]."""
    n = len(targets)
    order = np.argsort(targets)
    ys = np.array(targets, dtype=float)[order]
    for _ in range(80):
        for i in range(1, n):
            if ys[i] - ys[i - 1] < min_gap:
                mid = 0.5 * (ys[i] + ys[i - 1])
                ys[i - 1] = mid - 0.5 * min_gap
                ys[i] = mid + 0.5 * min_gap
        if ys[0] < y_lo:
            ys[0] = y_lo
            for i in range(1, n):
                ys[i] = max(ys[i], ys[i - 1] + min_gap)
        if ys[-1] > y_hi:
            ys[-1] = y_hi
            for i in range(n - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - min_gap)
    out = np.zeros(n, dtype=float)
    out[order] = ys
    return out


fig = plt.figure(figsize=(6.80, 4.02), dpi=300)
fig.patch.set_facecolor("#FFFFFF")

# Asymmetric Golden Grid:
#   - Left Hero Panel (a): [0.068 .. 0.406], full vertical span [0.128 .. 0.866]
#   - Right 3-Tier Stack (b, c, d): [0.468 .. 0.808], aligned [0.128 .. 0.866],
#     with [0.812 .. 0.996] dedicated exclusively to Direct Tail Labels.
ax1 = fig.add_axes([0.064, 0.128, 0.356, 0.738])
ax2 = fig.add_axes([0.486, 0.660, 0.368, 0.206])
ax3 = fig.add_axes([0.486, 0.394, 0.368, 0.206])
ax4 = fig.add_axes([0.486, 0.128, 0.368, 0.206])

hdr_artists = []
tail_artists = []

# ==============================================================================
# PANEL (a): HERO — Collective Scaling & Causal Fork Knockout
# ==============================================================================
ax1.set_facecolor("#FFFFFF")
ax1.set_xlim(0.50, 7.95)
ax1.set_ylim(20.5, 62.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

for yt in [25, 35, 45, 55]:
    ax1.axhline(yt, color=GRID_COL, linewidth=0.65, zorder=0.5)
for xt in [1, 3, 5, 7]:
    ax1.axvline(xt, color=GRID_COL, linewidth=0.65, zorder=0.5)

val_lin_react = 40.0
val_lin_rl    = 43.2
bcp_lin_react = 26.0
bcp_lin_rl    = 32.1

x_val  = np.array([1.32, 1.37, 1.75, 2.12, 2.33, 2.70])
y_val  = np.array([41.6, 46.2, 48.2, 48.4, 49.2, 51.2])
se_val = np.array([1.55, 1.52, 1.60, 1.68, 1.60, 1.60])

x_bcp  = np.array([1.41, 4.18, 7.41])
y_bcp  = np.array([26.3, 42.6, 49.6])
se_bcp = np.array([1.53, 1.72, 1.74])

x_ko, y_ko = 1.06, 28.8

# Horizontal reference ceilings for Single-Agent GRPO (143K)
ax1.plot([0.96, 2.70], [val_lin_rl, val_lin_rl], color=TERRA,
         linestyle=(0, (3.0, 2.0)), linewidth=0.90, alpha=0.72, zorder=1.8)
ax1.plot([0.96, 7.41], [bcp_lin_rl, bcp_lin_rl], color=TERRA,
         linestyle=(0, (3.0, 2.0)), linewidth=0.90, alpha=0.72, zorder=1.8)

# Subtle +/- 1 SE ribbons
ax1.fill_between(x_val, y_val - se_val, y_val + se_val,
                 facecolor=TEAL_DEEP, alpha=0.09, edgecolor="none", zorder=1.5)
ax1.fill_between(x_bcp, y_bcp - se_bcp, y_bcp + se_bcp,
                 facecolor=TEAL_MAIN, alpha=0.12, edgecolor="none", zorder=1.4)

# Scaling curves (both in Octopus Teal family!)
ax1.plot(x_val, y_val, color=TEAL_DEEP, linestyle="-", linewidth=1.75, zorder=3.6)
ax1.plot(x_bcp, y_bcp, color=TEAL_MAIN, linestyle="-", linewidth=2.05, zorder=3.7)

# Figure-8/9 signature luminous teal halo on the two final Octopus endpoints
for r_mult in np.linspace(4.8, 1.5, 8):
    a_ring = 0.028 * ((5.2 - r_mult) / 3.7) ** 1.25
    ax1.scatter([x_bcp[-1]], [y_bcp[-1]], s=46.0 * (r_mult ** 1.04),
                facecolors="#14B8A6", edgecolors="none", alpha=a_ring, zorder=3.4)
    ax1.scatter([x_val[-1]], [y_val[-1]], s=36.0 * (r_mult ** 1.02),
                facecolors=TEAL_MAIN, edgecolors="none", alpha=a_ring * 0.85, zorder=3.4)

# Smooth quadratic Bezier arc for Same-Weight v45 Fork-Off Knockout
t_bz = np.linspace(0.0, 1.0, 42)
p0_bz = np.array([7.24, 48.8])
p1_bz = np.array([4.15, 36.2])
p2_bz = np.array([1.26, 29.15])
ko_bx = (1 - t_bz) ** 2 * p0_bz[0] + 2 * (1 - t_bz) * t_bz * p1_bz[0] + t_bz ** 2 * p2_bz[0]
ko_by = (1 - t_bz) ** 2 * p0_bz[1] + 2 * (1 - t_bz) * t_bz * p1_bz[1] + t_bz ** 2 * p2_bz[1]
ax1.plot(ko_bx[:-2], ko_by[:-2], color=TERRA_KO, linestyle=(0, (3.2, 1.8)), linewidth=1.25, zorder=3.3)
ax1.annotate(
    "", xy=(ko_bx[-1], ko_by[-1]), xytext=(ko_bx[-4], ko_by[-4]),
    arrowprops=dict(arrowstyle="-|>,head_width=0.20,head_length=0.27",
                    color=TERRA_KO, lw=1.15), zorder=3.4
)

a1_markers = []
for xv, yv in [(0.82, val_lin_react), (0.82, bcp_lin_react)]:
    ax1.scatter([xv], [yv], s=22, facecolor="#E2E8F0", edgecolor=SLATE_600,
                linewidth=1.05, marker="o", zorder=4.2)
    a1_markers.append((f"react_{yv:.0f}", xv, yv, 2.5))

for xv, yv in [(0.96, val_lin_rl), (0.96, bcp_lin_rl)]:
    ax1.scatter([xv], [yv], s=24, facecolor="#FED7AA", edgecolor=TERRA,
                linewidth=1.05, marker="s", zorder=4.4)
    a1_markers.append((f"linrl_{yv:.0f}", xv, yv, 2.6))

ax1.scatter([x_ko], [y_ko], s=30, facecolor="#FEE2E2", edgecolor=TERRA_KO,
            linewidth=1.20, marker="D", zorder=4.9)
a1_markers.append(("bcp_ko", x_ko, y_ko, 3.0))

ax1.scatter([x_val[0]], [y_val[0]], s=22, facecolor="#FFFFFF", edgecolor=TEAL_DEEP,
            linewidth=1.15, marker="o", zorder=4.5)
a1_markers.append(("val_0", x_val[0], y_val[0], 2.5))
for idx in [1, 2, 4]:
    ax1.scatter([x_val[idx]], [y_val[idx]], s=19, facecolor=TEAL_DEEP, edgecolor="#022C22",
                linewidth=0.75, marker="o", zorder=4.6)
    a1_markers.append((f"val_{idx}", x_val[idx], y_val[idx], 2.3))
ax1.scatter([x_val[-1]], [y_val[-1]], s=42, facecolor=TEAL_DEEP, edgecolor="#022C22",
            linewidth=1.10, marker="o", zorder=5.0)
a1_markers.append(("val_final", x_val[-1], y_val[-1], 3.5))

ax1.scatter([x_bcp[0]], [y_bcp[0]], s=23, facecolor="#FFFFFF", edgecolor=TEAL_MAIN,
            linewidth=1.15, marker="D", zorder=4.5)
a1_markers.append(("bcp_0", x_bcp[0], y_bcp[0], 2.5))
ax1.scatter([x_bcp[1]], [y_bcp[1]], s=27, facecolor=TEAL_MAIN, edgecolor=TEAL_DEEP,
            linewidth=0.90, marker="D", zorder=4.8)
a1_markers.append(("bcp_d1", x_bcp[1], y_bcp[1], 2.7))
ax1.scatter([x_bcp[2]], [y_bcp[2]], s=46, facecolor=TEAL_MAIN, edgecolor=TEAL_DEEP,
            linewidth=1.15, marker="D", zorder=5.1)
a1_markers.append(("bcp_v45", x_bcp[2], y_bcp[2], 3.7))

# Vertical gain arrows matching Figure 8 & Figure 9a
ax1.annotate(
    "", xy=(2.70, 49.9), xytext=(2.70, val_lin_rl + 0.45),
    arrowprops=dict(arrowstyle="-|>,head_width=0.20,head_length=0.28",
                    color=TEAL_DEEP, lw=1.20), zorder=3.8
)
ax1.annotate(
    "", xy=(7.41, 48.1), xytext=(7.41, bcp_lin_rl + 0.45),
    arrowprops=dict(arrowstyle="-|>,head_width=0.21,head_length=0.30",
                    color=TEAL_MAIN, lw=1.30), zorder=3.8
)

a1_texts = []
a1_texts.append(("lbl_val", ax1.text(
    3.12, 53.2, "OpenResearcher Val · 51.2%",
    ha="left", va="center", fontsize=5.45, fontweight=700, color=TEAL_DEEP, zorder=6
)))
a1_texts.append(("gain_val", ax1.text(
    2.88, 46.5, "+8.0%",
    ha="left", va="center", fontsize=5.55, fontweight=700, color=TEAL_DEEP, zorder=6
)))
a1_texts.append(("lbl_bcp", ax1.text(
    4.86, 49.2, "BrowseComp-Plus · 49.6%",
    ha="center", va="center", fontsize=5.45, fontweight=700, color="#0F766E", zorder=6
)))
a1_texts.append(("gain_bcp", ax1.text(
    7.20, 39.8, "+17.5%",
    ha="right", va="center", fontsize=5.75, fontweight=700, color="#0F766E", zorder=6
)))
a1_texts.append(("ko_lbl", ax1.text(
    5.48, 34.4, "Step 45, Fork Off (−21.7%)",
    ha="center", va="center", fontsize=5.15, fontweight=600, color=TERRA_KO, zorder=6
)))
# Pure borderless editorial callout (zero cheap UI sticker box!)
a1_texts.append(("rescue_note", ax1.text(
    4.58, 22.8, "32K-wall dead Qs: 0% → 31.8% rescued\nHard multi-hop (Q3): 3.4% → 34.1% (10×)",
    ha="center", va="center", fontsize=5.05, fontweight=500, color=SLATE_600,
    linespacing=1.15, zorder=6
)))

# Borderless clean legend matching Figure 9a
h_oct_v = Line2D([0], [0], color=TEAL_DEEP, lw=1.65, marker="o", markersize=3.8,
                 markerfacecolor=TEAL_DEEP, markeredgecolor="#022C22")
h_oct_b = Line2D([0], [0], color=TEAL_MAIN, lw=1.80, marker="D", markersize=3.6,
                 markerfacecolor=TEAL_MAIN, markeredgecolor=TEAL_DEEP)
h_ko    = Line2D([0], [0], color=TERRA_KO, lw=1.20, ls="--", marker="D", markersize=3.6,
                 markerfacecolor="#FEE2E2", markeredgecolor=TERRA_KO)
h_grpo1 = Line2D([0], [0], color=TERRA, lw=0.95, ls="--", marker="s", markersize=3.8,
                 markerfacecolor="#FED7AA", markeredgecolor=TERRA)
h_reac1 = Line2D([0], [0], color="none", marker="o", markersize=3.8,
                 markerfacecolor="#E2E8F0", markeredgecolor=SLATE_600)

leg1 = ax1.legend(
    [(h_oct_v, h_oct_b), h_ko, h_grpo1, h_reac1],
    ["Octopus (4B, Ours)", "Step 45, Fork Off (1 Agent)", "Single-Agent GRPO (143K)", "Single-Agent ReAct (32K)"],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.28)},
    loc="upper left", bbox_to_anchor=(0.005, 0.995),
    frameon=False, fontsize=5.15, labelcolor=SLATE_700,
    handlelength=1.70, handletextpad=0.36, borderpad=0.15, labelspacing=0.22
)
for t_idx, txt_obj in enumerate(leg1.get_texts()):
    if t_idx == 0:
        txt_obj.set_fontweight(700)
        txt_obj.set_color(SLATE_900)

ax1.set_xticks([1, 3, 5, 7])
ax1.set_xticklabels(["1", "3", "5", "7"], fontsize=5.75, color=SLATE_700)
ax1.set_yticks([25, 35, 45, 55])
ax1.set_yticklabels(["25%", "35%", "45%", "55%"], fontsize=5.75, color=SLATE_700)
ax1.set_xlabel("Deployed Agents / Episode", fontsize=6.25, fontweight=600, color=SLATE_900, labelpad=3.2)
ax1.set_ylabel("Held-Out Accuracy (%)", fontsize=6.25, fontweight=600, color=SLATE_900, labelpad=2.8)

pos1 = ax1.get_position()
c1 = 0.5 * (pos1.x0 + pos1.x1)
hdr_artists.append(fig.text(
    c1, 0.950, "(a) Collective Scaling & Causal Fork Knockout",
    fontsize=6.95, fontweight=700, color=SLATE_900, ha="center", va="center"
))
hdr_artists.append(fig.text(
    c1, 0.910, "500Q OpenResearcher Val & 830Q BrowseComp-Plus (±1 SE)",
    fontsize=5.25, fontweight=400, color=SLATE_500, ha="center", va="center"
))

# ==============================================================================
# SHARED 43-STEP EMPIRICAL SERIES (23,001 tree.json traces + Steps 44-45 pass completion)
# ==============================================================================
steps_v9 = np.arange(1, 44, dtype=float)
steps_v8 = np.arange(1, 18, dtype=float)

fork_v8 = np.array([
    23.9, 28.2, 19.6, 14.0, 8.9, 6.0, 2.9, 0.5, 2.2, 1.9,
    1.3, 2.4, 0.9, 2.0, 0.0, 1.3, 1.2
], dtype=float)

mu_pct_v9 = 100.0 * np.array([
    0.471, 0.573, 0.485, 0.439, 0.456, 0.442, 0.522, 0.510, 0.455, 0.439,
    0.352, 0.357, 0.369, 0.319, 0.386, 0.433, 0.341, 0.368, 0.357, 0.464,
    0.394, 0.441, 0.432, 0.437, 0.441, 0.522, 0.512, 0.561, 0.549, 0.557,
    0.578, 0.580, 0.528, 0.649, 0.611, 0.600, 0.622, 0.692, 0.621, 0.670, 0.650,
    0.644, 0.679
], dtype=float)

cap10_v9 = np.array([
    2.6, 6.2, 2.3, 1.3, 2.0, 1.9, 1.5, 5.8, 2.8, 2.5,
    1.4, 2.4, 1.4, 1.3, 2.9, 3.5, 3.5, 4.5, 5.3, 8.9,
    4.7, 8.9, 6.5, 11.0, 7.4, 15.3, 13.9, 17.8, 16.8, 20.9,
    21.9, 25.3, 16.8, 37.0, 30.2, 28.0, 30.9, 42.4, 39.9, 41.0, 38.4,
    37.8, 51.9
], dtype=float)

depth3_v9 = np.array([
    5.8, 11.3, 3.9, 3.2, 5.2, 3.3, 3.1, 7.8, 4.2, 3.1,
    2.9, 3.4, 4.2, 1.3, 5.7, 8.1, 6.4, 6.3, 7.8, 11.4,
    8.7, 13.5, 11.8, 16.3, 11.8, 20.1, 22.9, 24.7, 25.3, 28.1,
    35.5, 31.6, 24.9, 44.6, 40.1, 36.5, 40.9, 51.4, 41.3, 49.7, 45.3,
    44.8, 59.3
], dtype=float)

depth4_v9 = np.array([
    2.5, 5.9, 1.6, 1.6, 3.2, 1.6, 1.5, 3.9, 2.3, 1.5,
    0.7, 0.8, 1.1, 0.7, 2.9, 4.2, 3.2, 2.5, 2.8, 6.6,
    2.7, 6.2, 4.7, 7.9, 6.1, 11.7, 12.9, 15.5, 14.6, 17.2,
    18.3, 17.9, 13.5, 29.3, 22.4, 21.3, 25.0, 34.7, 23.2, 33.3, 27.2,
    27.3, 39.1
], dtype=float)

depth5_v9 = np.array([
    0.4, 3.1, 1.2, 1.0, 1.2, 0.7, 1.5, 1.2, 1.3, 0.7,
    0.2, 0.3, 0.4, 0.0, 1.0, 1.4, 1.9, 1.4, 2.5, 3.1,
    1.7, 2.4, 1.8, 4.4, 2.6, 5.4, 5.9, 7.8, 7.7, 7.7,
    7.0, 7.8, 7.2, 16.3, 11.9, 11.0, 12.8, 21.4, 12.0, 19.0, 13.7,
    14.7, 21.5
], dtype=float)

turn1_fork_v9 = np.array([
    2.6, 1.6, 2.8, 1.3, 4.6, 4.0, 4.1, 8.2, 9.4, 8.3,
    11.0, 12.4, 13.0, 12.5, 24.5, 25.0, 39.2, 34.7, 44.9, 40.9,
    49.2, 50.4, 52.8, 64.6, 57.7, 66.8, 71.8, 71.6, 74.3, 79.7,
    83.8, 86.7, 91.6, 90.4, 93.2, 94.2, 93.5, 94.6, 96.1, 96.4, 97.2,
    93.4, 98.0
], dtype=float)

work_open_v9 = np.array([
    49.2, 56.9, 35.7, 40.5, 40.9, 31.1, 27.1, 53.0, 41.2, 40.8,
    31.0, 41.0, 51.7, 40.7, 55.9, 57.3, 61.0, 65.1, 70.0, 72.4,
    75.3, 76.4, 75.3, 81.5, 78.9, 80.4, 85.2, 84.0, 86.0, 85.6,
    89.2, 89.5, 88.2, 89.9, 91.6, 91.3, 92.0, 90.9, 92.4, 91.4, 92.8,
    92.8, 92.4
], dtype=float)

read_ep_v9 = np.array([
    17.3, 21.8, 12.3, 13.1, 15.3, 11.9, 9.7, 21.8, 14.5, 12.3,
    16.0, 13.7, 19.6, 17.8, 22.9, 23.4, 34.1, 28.2, 41.7, 37.4,
    35.8, 41.0, 40.9, 52.9, 43.6, 53.9, 56.4, 54.0, 53.2, 62.5,
    66.8, 61.4, 51.7, 69.6, 66.3, 70.9, 71.0, 74.3, 74.6, 74.5, 76.7,
    66.9, 85.9
], dtype=float)

rw_main_v9 = np.array([
    22.3, 28.0, 22.5, 29.3, 27.1, 24.8, 25.5, 27.6, 27.3, 27.0,
    27.8, 25.6, 27.4, 26.3, 22.2, 24.5, 24.1, 23.9, 15.9, 23.7,
    20.1, 21.9, 16.3, 17.6, 17.6, 25.1, 18.0, 22.7, 13.6, 17.8,
    15.4, 12.2, 12.6, 21.3, 14.1, 11.6, 10.2, 18.9, 14.5, 14.9, 15.5,
    11.7, 19.2
], dtype=float)

rw_br_v9 = np.array([
    4.4, 6.9, 3.2, 3.8, 4.0, 4.4, 2.6, 11.3, 6.6, 5.8,
    6.9, 5.0, 10.5, 7.2, 12.6, 14.1, 19.9, 14.5, 20.8, 18.6,
    16.1, 17.3, 21.2, 28.2, 21.9, 32.6, 29.3, 34.6, 25.9, 33.2,
    37.6, 31.9, 27.3, 44.3, 34.9, 32.4, 35.1, 45.1, 43.9, 40.6, 38.4,
    39.2, 47.5
], dtype=float)

rw_br2_v9 = np.array([
    1.8, 2.6, 1.4, 1.6, 0.9, 1.4, 0.5, 3.9, 2.7, 1.8,
    1.0, 1.3, 1.8, 0.0, 2.9, 3.5, 2.6, 3.1, 3.9, 4.0,
    2.7, 6.5, 5.9, 7.2, 5.0, 12.0, 10.0, 13.7, 10.7, 12.3,
    13.6, 14.3, 11.7, 21.6, 17.4, 13.7, 18.5, 24.3, 23.7, 22.8, 21.1,
    22.6, 27.9
], dtype=float)

rw_br3_v9 = np.array([
    0.8, 1.3, 0.2, 0.3, 0.0, 0.5, 0.5, 1.2, 0.7, 0.5,
    0.2, 0.3, 1.1, 0.0, 1.9, 1.9, 0.3, 0.5, 1.1, 1.4,
    1.0, 2.1, 1.8, 2.4, 1.4, 3.3, 3.2, 5.1, 4.0, 4.6,
    6.3, 4.2, 5.7, 10.3, 6.9, 7.1, 8.8, 10.4, 11.5, 11.3, 10.7,
    11.1, 11.8
], dtype=float)

rw_br4_v9 = np.array([
    0.1, 0.4, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.6, 0.2,
    0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.0, 0.0, 1.1, 0.6,
    0.0, 0.3, 0.5, 0.3, 0.4, 0.3, 0.6, 2.0, 1.4, 1.7,
    2.1, 0.6, 1.8, 3.4, 3.1, 1.8, 3.0, 3.5, 3.6, 3.3, 3.0,
    3.3, 4.7
], dtype=float)


def style_stack_axis(ax, y_max, y_ticks, y_labels, show_xticks=False):
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0.2, 45.4)
    ax.set_ylim(0.0, y_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for yt in y_ticks:
        if yt > 0:
            ax.axhline(yt, color=GRID_COL, linewidth=0.65, zorder=0.5)
    for xt in [1, 15, 30, 45]:
        ax.axvline(xt, color=GRID_COL, linewidth=0.65, zorder=0.5)
    # Single clean neutral plateau column (Steps 1-14) + vertically aligned Step-15 spine
    ax.axvspan(0.2, 14.5, facecolor=ZONE_BG, edgecolor="none", alpha=0.95, zorder=0.3)
    ax.axvline(14.5, color=SLATE_600, linestyle=(0, (2.6, 1.8)), linewidth=0.85, alpha=0.85, zorder=2.0)
    # Subtle 1-epoch pass completion strip (Steps 43-45)
    ax.axvspan(43.0, 45.0, facecolor="#F1F5F9", edgecolor="none", alpha=0.85, zorder=0.4)
    ax.axvline(45.0, color="#94A3B8", linestyle=(0, (2.0, 1.8)), linewidth=0.75, zorder=1.6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=5.45, color=SLATE_700)
    if show_xticks:
        ax.set_xticks([1, 15, 19, 30, 45])
        ax.set_xticklabels(["1", "15", "19", "30", "45"], fontsize=5.65, color=SLATE_700)
    else:
        ax.set_xticks([1, 15, 30, 45])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=1.8, color="#CBD5E1")


def attach_tail_labels(ax, items, min_gap, y_lo, y_hi):
    """Draw crisp 2-segment leader elbows from (43.2, y_end) to Direct Tail Labels at x = 46.1."""
    ys_raw = [it[0] for it in items]
    ys_lbl = solve_tail_positions(ys_raw, min_gap=min_gap, y_lo=y_lo, y_hi=y_hi)
    for (y_end, label, col, fw), y_l in zip(items, ys_lbl):
        ax.plot([43.3, 44.5, 45.7], [y_end, y_l, y_l], color=col, lw=0.68, alpha=0.62,
                clip_on=False, zorder=4.0)
        t_obj = ax.text(
            46.1, y_l, label, ha="left", va="center",
            fontsize=5.12, fontweight=fw, color=col, clip_on=False, zorder=6.0
        )
        tail_artists.append((label, t_obj))


# ==============================================================================
# PANEL (b): TOP TIER — Step-15 Recursive Phase Transition vs. Collapse
# Unified [0%, 100%] <-> mu in [0.0, 1.0] natural scale!
# ==============================================================================
style_stack_axis(ax2, y_max=114.0, y_ticks=[0, 50, 100], y_labels=["0%", "50%", "100%"])

# Blush runaway band above mu = 1.0 (100%), matching Figure 8 & Figure 9b dominated zone 1:1
ax2.axhspan(100.0, 114.0, facecolor="#FCEAE6", edgecolor="none", alpha=0.72, zorder=0.8)
ax2.axhline(100.0, color="#D97757", linestyle=(0, (3.2, 2.0)), linewidth=0.95, zorder=2.4)

s_d3  = smooth_preserve_ends(depth3_v9)
s_d4  = smooth_preserve_ends(depth4_v9)
s_d5  = smooth_preserve_ends(depth5_v9)
s_c10 = smooth_preserve_ends(cap10_v9)
s_mu  = smooth_preserve_ends(mu_pct_v9)
s_v8  = smooth_preserve_ends(fork_v8)

ax2.fill_between(steps_v9, 0, s_d3, facecolor=TEAL_MAIN, alpha=0.055, edgecolor="none", zorder=1.0)

ax2.plot(steps_v8, s_v8, color=TERRA, linestyle=(0, (2.8, 1.6)), linewidth=1.40, zorder=4.4)
ax2.scatter([17.0], [s_v8[-1]], s=16, color=TERRA, marker="x", linewidth=1.10, zorder=5.0)

ax2.plot(steps_v9, s_mu,  color=SLATE_700, linestyle=(0, (2.0, 1.5)), linewidth=1.30, zorder=3.9)
ax2.plot(steps_v9, s_c10, color=SLATE_500, linestyle=(0, (3.6, 1.8)), linewidth=1.20, zorder=4.0)
ax2.plot(steps_v9, s_d3,  color=TEAL_DEEP, linestyle="-", linewidth=1.80, zorder=4.5)
ax2.plot(steps_v9, s_d4,  color=TEAL_MAIN, linestyle="-", linewidth=1.48, zorder=4.3)
ax2.plot(steps_v9, s_d5,  color=TEAL_MID,  linestyle="-", linewidth=1.32, zorder=4.2)

for y_e, c_e in [(s_mu[-1], SLATE_700), (s_d3[-1], TEAL_DEEP), (s_c10[-1], SLATE_500),
                 (s_d4[-1], TEAL_MAIN), (s_d5[-1], TEAL_MID)]:
    ax2.scatter([43.0], [y_e], s=11, facecolor=c_e, edgecolor="#FFFFFF", linewidth=0.5, zorder=5.0)

a2_texts = []
a2_texts.append(("runaway_b", ax2.text(
    29.5, 106.8, "Runaway threshold (μ = 1.0)",
    ha="center", va="center", fontsize=5.00, fontweight=600, color="#9A3412", zorder=6
)))
a2_texts.append(("plat_b", ax2.text(
    7.4, 80.5, "14-step\nplateau",
    ha="center", va="center", fontsize=5.00, fontweight=500, color=SLATE_500,
    linespacing=1.0, zorder=6
)))
a2_texts.append(("ign_b", ax2.text(
    22.2, 80.5, "Step 15\nignition",
    ha="center", va="center", fontsize=5.05, fontweight=700, color=SLATE_700,
    linespacing=1.0, zorder=6
)))
ax2.annotate(
    "", xy=(14.8, 80.5), xytext=(18.3, 80.5),
    arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.22",
                    color=SLATE_700, lw=0.80), zorder=4.5
)
# Centered cleanly inside plateau [4.2 .. 13.6], zero contact with x = 14.5 vertical spine!
a2_texts.append(("v8_lbl", ax2.text(
    8.8, 22.8, "Naive RL → 0%",
    ha="center", va="center", fontsize=4.85, fontweight=600, color=TERRA, zorder=6
)))

attach_tail_labels(
    ax2,
    [
        (s_mu[-1],  "μ = 0.68 (< 1.0)", SLATE_700, 600),
        (s_d3[-1],  "d ≥ 3  59.3%",     TEAL_DEEP, 700),
        (s_c10[-1], "10-W  51.9%",      SLATE_500, 500),
        (s_d4[-1],  "d ≥ 4  39.1%",     TEAL_MAIN, 600),
        (s_d5[-1],  "d ≥ 5  21.5%",     "#287D74", 600),
    ],
    min_gap=15.8, y_lo=10.0, y_hi=92.5
)

ax2.set_ylabel("Share / Ratio μ", fontsize=5.80, fontweight=600, color=SLATE_900, labelpad=2.5)

pos2 = ax2.get_position()
hdr_artists.append(fig.text(
    pos2.x0, 0.950, "(b) Step-15 Recursive Phase Transition vs. Collapse",
    fontsize=6.95, fontweight=700, color=SLATE_900, ha="left", va="center"
))
hdr_artists.append(fig.text(
    pos2.x0, 0.910, "24,041 episodes · Unlimited depth, ≤10 workers · 54× d ≥ 5 surge",
    fontsize=5.25, fontweight=400, color=SLATE_500, ha="left", va="center"
))

# ==============================================================================
# PANEL (c): MIDDLE TIER — Zero-Reward Commander & Peer KV Mesh
# ==============================================================================
style_stack_axis(ax3, y_max=108.0, y_ticks=[0, 50, 100], y_labels=["0%", "50%", "100%"])

s_t1    = smooth_preserve_ends(turn1_fork_v9)
s_wopen = smooth_preserve_ends(work_open_v9)
s_rep   = smooth_preserve_ends(read_ep_v9)

ax3.fill_between(steps_v9, 0, s_t1, facecolor=TEAL_MAIN, alpha=0.055, edgecolor="none", zorder=1.0)

ax3.plot(steps_v9, s_t1,    color=TEAL_DEEP, linestyle="-", linewidth=1.85, zorder=4.5)
ax3.plot(steps_v9, s_wopen, color=TEAL_MAIN, linestyle="-", linewidth=1.50, zorder=4.3)
ax3.plot(steps_v9, s_rep,   color=PETROL,    linestyle=(0, (3.6, 1.6)), linewidth=1.55, zorder=4.4)

for y_e, c_e in [(s_t1[-1], TEAL_DEEP), (s_wopen[-1], TEAL_MAIN), (s_rep[-1], PETROL)]:
    ax3.scatter([43.0], [y_e], s=11, facecolor=c_e, edgecolor="#FFFFFF", linewidth=0.5, zorder=5.0)

a3_texts = []
# Cleanly nested in the open bottom-right pocket [x = 21..42, y = 7..31], far from x=14.5 & all curves
a3_texts.append(("zero_note", ax3.text(
    31.5, 19.0, "0 shaping reward · 23.9 KV mounts/ep\n96.4% lateral Worker ↔ Worker",
    ha="center", va="center", fontsize=4.95, fontweight=500, color=SLATE_600,
    linespacing=1.15, zorder=6
)))

attach_tail_labels(
    ax3,
    [
        (s_t1[-1],    "Turn-1 Fork  98.0%",  TEAL_DEEP, 700),
        (s_wopen[-1], "Worker Open  92.8%",  TEAL_MAIN, 600),
        (s_rep[-1],   "Peer KV Mesh  85.9%", PETROL,    600),
    ],
    min_gap=15.5, y_lo=56.0, y_hi=99.5
)

ax3.set_ylabel("Share (%)", fontsize=5.80, fontweight=600, color=SLATE_900, labelpad=2.5)

pos3 = ax3.get_position()
hdr_artists.append(fig.text(
    pos3.x0, 0.627, "(c) Zero-Reward Commander & Peer-to-Peer KV Mesh",
    fontsize=6.60, fontweight=700, color=SLATE_900, ha="left", va="center"
))

# ==============================================================================
# PANEL (d): BOTTOM TIER — Depth-Cascaded Backtracking Shield
# ==============================================================================
style_stack_axis(ax4, y_max=54.0, y_ticks=[0, 20, 40], y_labels=["0%", "20%", "40%"], show_xticks=True)

s_rm  = smooth_preserve_ends(rw_main_v9)
s_rb  = smooth_preserve_ends(rw_br_v9)
s_rb2 = smooth_preserve_ends(rw_br2_v9)
s_rb3 = smooth_preserve_ends(rw_br3_v9)
s_rb4 = smooth_preserve_ends(rw_br4_v9)

ax4.fill_between(steps_v9, 0, s_rb, facecolor=TEAL_MAIN, alpha=0.055, edgecolor="none", zorder=1.0)

ax4.plot(steps_v9, s_rm,  color=TERRA,     linestyle="-", linewidth=1.65, zorder=4.35)
ax4.plot(steps_v9, s_rb,  color=TEAL_DEEP, linestyle="-", linewidth=1.85, zorder=4.5)
ax4.plot(steps_v9, s_rb2, color=TEAL_MAIN, linestyle="-", linewidth=1.48, zorder=4.3)
ax4.plot(steps_v9, s_rb3, color=TEAL_MID,  linestyle="-", linewidth=1.35, zorder=4.2)
ax4.plot(steps_v9, s_rb4, color=TEAL_SOFT, linestyle="-", linewidth=1.25, zorder=4.1)

for y_e, c_e in [(s_rb[-1], TEAL_DEEP), (s_rb2[-1], TEAL_MAIN), (s_rm[-1], TERRA),
                 (s_rb3[-1], TEAL_MID), (s_rb4[-1], TEAL_SOFT)]:
    ax4.scatter([43.0], [y_e], s=11, facecolor=c_e, edgecolor="#FFFFFF", linewidth=0.5, zorder=5.0)

cross_x = 19.0
cross_y = 0.5 * (s_rm[18] + s_rb[18])
ax4.scatter([cross_x], [cross_y], s=28, facecolor="#FEF3C7", edgecolor=TERRA,
            linewidth=1.10, zorder=5.2)

a4_markers = [("crossover", cross_x, cross_y, 3.0)]
a4_texts = []
# Left-aligned at x = 15.8 so it sits 100% to the RIGHT of the Step-15 vertical dashed line!
a4_texts.append(("cross_lbl", ax4.text(
    15.8, 43.2, "Step 19 crossover\n(91.5% worker rewinds)",
    ha="left", va="center", fontsize=4.95, fontweight=500, color=SLATE_600,
    linespacing=1.10, zorder=6
)))
ax4.annotate(
    "", xy=(19.0, cross_y + 2.3), xytext=(19.0, 36.2),
    arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.22",
                    color=SLATE_500, lw=0.80), zorder=4.5
)

attach_tail_labels(
    ax4,
    [
        (s_rb[-1],  "Workers d ≥ 1  47.5%", TEAL_DEEP, 700),
        (s_rb2[-1], "Workers d ≥ 2  27.9%", TEAL_MAIN, 600),
        (s_rm[-1],  "Root d = 0 (10.2% min)", TERRA,   600),
        (s_rb3[-1], "Workers d ≥ 3  11.8%", "#287D74", 600),
        (s_rb4[-1], "Workers d ≥ 4   4.7%", "#4A8F87", 500),
    ],
    min_gap=7.6, y_lo=3.8, y_hi=49.5
)

ax4.set_xlabel("RL Training Step (1-Epoch Recursive Pass)", fontsize=6.25, fontweight=600, color=SLATE_900, labelpad=3.2)
ax4.set_ylabel("Rewind (%)", fontsize=5.80, fontweight=600, color=SLATE_900, labelpad=2.5)

pos4 = ax4.get_position()
hdr_artists.append(fig.text(
    pos4.x0, 0.361, "(d) Depth-Cascaded Backtracking & Detour Shielding",
    fontsize=6.60, fontweight=700, color=SLATE_900, ha="left", va="center"
))

# ==============================================================================
# EXHAUSTIVE PIXEL-LEVEL ZERO-OVERLAP GEOMETRY AUDIT
# ==============================================================================
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
PT_TO_PX = fig.dpi / 72.0
fig_w_px = fig.get_figwidth() * fig.dpi
fig_h_px = fig.get_figheight() * fig.dpi


def seg_intersects_bbox(p0, p1, bb, pad=1.2):
    x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
    if max(p0[0], p1[0]) < x0 or min(p0[0], p1[0]) > x1:
        return False
    if max(p0[1], p1[1]) < y0 or min(p0[1], p1[1]) > y1:
        return False
    ts = np.linspace(0.0, 1.0, 25)
    xs = p0[0] + ts * (p1[0] - p0[0])
    ys = p0[1] + ts * (p1[1] - p0[1])
    return bool(np.any((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)))


errors = []

for i, art in enumerate(hdr_artists):
    bi = art.get_window_extent(renderer)
    if bi.x0 < 3.0 or bi.x1 > fig_w_px - 3.0 or bi.y1 > fig_h_px - 2.0:
        errors.append(f"[HEADER] artist {i} clips figure edge ({bi.x0:.1f}..{bi.x1:.1f})")
    for j in range(i + 1, len(hdr_artists)):
        if bi.overlaps(hdr_artists[j].get_window_extent(renderer)):
            errors.append(f"[HEADER] overlap {i} vs {j}")

for i, (ni, ti) in enumerate(tail_artists):
    bi = ti.get_window_extent(renderer)
    if bi.x1 > fig_w_px - 2.0:
        errors.append(f"[TAIL] '{ni}' clips right figure edge ({bi.x1:.1f} > {fig_w_px - 2.0:.1f})")
    for j in range(i + 1, len(tail_artists)):
        nj, tj = tail_artists[j]
        bj = tj.get_window_extent(renderer)
        if bi.expanded(1.01, 1.04).overlaps(bj):
            errors.append(f"[TAIL] overlap '{ni}' vs '{nj}'")
    for h_idx, h_art in enumerate(hdr_artists):
        if bi.expanded(1.01, 1.05).overlaps(h_art.get_window_extent(renderer)):
            errors.append(f"[TAIL] '{ni}' overlaps header {h_idx}")


def audit_panel(tag, ax_main, texts, markers, curves, leg_obj=None):
    ax_bb = ax_main.get_window_extent(renderer)
    bbs = []
    for name, art in texts:
        bb = art.get_window_extent(renderer)
        bbs.append((name, bb))
        if (bb.x0 < ax_bb.x0 + 1.0 or bb.x1 > ax_bb.x1 - 1.0 or
                bb.y0 < ax_bb.y0 + 1.0 or bb.y1 > ax_bb.y1 - 1.0):
            errors.append(f"[{tag}] Text '{name}' escapes plot spine!")
    if leg_obj is not None:
        bbs.append(("LEGEND", leg_obj.get_window_extent(renderer)))

    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            ni, bi = bbs[i]
            nj, bj = bbs[j]
            if bi.expanded(1.02, 1.05).overlaps(bj):
                errors.append(f"[{tag}] Text/Legend overlap: '{ni}' vs '{nj}'")

    m_px = []
    for mname, mx, my, mr_pt in markers:
        px, py = ax_main.transData.transform((mx, my))
        m_px.append((mname, px, py, mr_pt * PT_TO_PX))
    for i in range(len(m_px)):
        for j in range(i + 1, len(m_px)):
            n1, x1, y1, r1 = m_px[i]
            n2, x2, y2, r2 = m_px[j]
            dist = float(np.hypot(x1 - x2, y1 - y2)) - (r1 + r2)
            if dist < 1.0 * PT_TO_PX:
                errors.append(f"[{tag}] Markers too close ({dist/PT_TO_PX:.2f}pt): '{n1}' vs '{n2}'")

    for tname, tbb in bbs:
        for mname, px, py, mr in m_px:
            if (tbb.x0 - mr - 1.5 <= px <= tbb.x1 + mr + 1.5 and
                    tbb.y0 - mr - 1.5 <= py <= tbb.y1 + mr + 1.5):
                errors.append(f"[{tag}] Text '{tname}' collides with marker '{mname}'")

    for cname, ax_c, cx, cy in curves:
        pts = ax_c.transData.transform(np.column_stack([cx, cy]))
        for tname, tbb in bbs:
            for k in range(len(pts) - 1):
                if seg_intersects_bbox(pts[k], pts[k + 1], tbb, pad=1.2):
                    errors.append(f"[{tag}] Text '{tname}' intersects curve '{cname}' at seg {k}")
                    break


audit_panel(
    "PanelA", ax1, a1_texts, a1_markers,
    [("val_curve", ax1, x_val, y_val),
     ("bcp_curve", ax1, x_bcp, y_bcp),
     ("ko_arc", ax1, ko_bx, ko_by),
     ("val_ceil", ax1, np.array([0.96, 2.70]), np.array([val_lin_rl, val_lin_rl])),
     ("bcp_ceil", ax1, np.array([0.96, 7.41]), np.array([bcp_lin_rl, bcp_lin_rl]))],
    leg_obj=leg1
)

audit_panel(
    "PanelB", ax2, a2_texts, [],
    [("d3", ax2, steps_v9, s_d3),
     ("d4", ax2, steps_v9, s_d4),
     ("d5", ax2, steps_v9, s_d5),
     ("cap10", ax2, steps_v9, s_c10),
     ("v8", ax2, steps_v8, s_v8),
     ("mu", ax2, steps_v9, s_mu),
     ("mu_lim", ax2, np.array([0.2, 45.4]), np.array([100.0, 100.0])),
     ("step15_vline", ax2, np.array([14.5, 14.5]), np.array([0.0, 114.0]))]
)

audit_panel(
    "PanelC", ax3, a3_texts, [],
    [("turn1", ax3, steps_v9, s_t1),
     ("wopen", ax3, steps_v9, s_wopen),
     ("read_ep", ax3, steps_v9, s_rep),
     ("step15_vline", ax3, np.array([14.5, 14.5]), np.array([0.0, 108.0]))]
)

audit_panel(
    "PanelD", ax4, a4_texts, a4_markers,
    [("rw_br", ax4, steps_v9, s_rb),
     ("rw_br2", ax4, steps_v9, s_rb2),
     ("rw_br3", ax4, steps_v9, s_rb3),
     ("rw_br4", ax4, steps_v9, s_rb4),
     ("rw_main", ax4, steps_v9, s_rm),
     ("step15_vline", ax4, np.array([14.5, 14.5]), np.array([0.0, 54.0]))]
)

out_pdf = ROOT / "figures/fig_exp3_agent_scaling.pdf"
out_svg = ROOT / "figures/fig_exp3_agent_scaling.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)

if errors:
    print("GEOMETRY AUDIT FAILED:")
    for e in errors:
        print("  -", e)
    raise SystemExit(1)

print("ZERO-OVERLAP GEOMETRY AUDIT PASSED! Saved:", out_pdf)
