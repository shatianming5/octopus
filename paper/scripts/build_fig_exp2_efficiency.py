#!/usr/bin/env python3
"""
Publication-Grade Two-Panel Efficiency & Cross-Branch Sharing Figure (Figure 9, Section 4.2):
  - Matches Figure 8 (scripts/build_fig_exp1_pareto.py) visual grammar 1:1:
      * Inter typography, #334155 0.85lw spines, #EEF2F6 0.65lw major gridlines
      * Single blush-terracotta (#FCEAE6) dominated quadrant with #D97757 dashed crosshairs
      * Dynamic [N] citation resolution from main.bbl
  - Genius Anti-Overlap Structural Geometry:
      * Panel (a): Consolidates the 3 statistically indistinguishable 0-shot untrained variants
        (100-Turn ReAct, 32K ReAct, 0-Shot Fork) into 1 crisp 0-Shot Base trajectory wrapped in a
        shaded min-max 0-Shot envelope band, paired with deterministic micro-track stage dodging
        (dx = -0.115, 0.000, +0.115) inside subtle stage columns so zero markers/errorbars ever stack.
      * Panel (b): Eliminates ghost 0-shot double-stamps (1 crisp jewel marker + +/-1 SE per paradigm)
        and replaces the right-crushing Sqrt X-scale with a Piecewise-Linear Broken X-Axis
        ([0% - 10.5%] // [48% - 64%]) that collapses the empty 11%-47% desert into a clean '//' break,
        giving true linear separation to both low-hijack KV methods and high-hijack full-context methods.
  - Exhaustive pixel-level geometry audit INCLUDING marker-vs-marker & errorbar-vs-errorbar clearance.
"""
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.scale import FuncScale
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
for ttf in (ROOT / "scripts/fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(ttf))

# Dynamically resolve exact [N] citation indices from main.bbl
cite_map = {
    "wu2023autogen": "6",
    "shao2026clm": "18",
    "yao2025cacheblend": "29",
    "rodionov2025hogwild": "32",
    "ye2025kvcomm": "34",
}
bbl_path = ROOT / "main.bbl"
if bbl_path.exists():
    bbl_txt = bbl_path.read_text(encoding="utf-8", errors="ignore")
    keys = re.findall(r"\\bibitem(?:\[.*?\])?\{([^}]+)\}", bbl_txt, re.S)
    for idx, k in enumerate(keys, 1):
        if k in cite_map:
            cite_map[k] = str(idx)

C_AG  = f"[{cite_map['wu2023autogen']}]"
C_CLM = f"[{cite_map['shao2026clm']}]"
C_CB  = f"[{cite_map['yao2025cacheblend']}]"
C_KVC = f"[{cite_map['rodionov2025hogwild']}, {cite_map['ye2025kvcomm']}]"

plt.rcParams.update({
    "font.family": "Inter",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.85,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(6.55, 2.98), dpi=300)
fig.patch.set_facecolor("#FFFFFF")

gs = gridspec.GridSpec(
    1, 2, width_ratios=[1.05, 1.18], wspace=0.245,
    left=0.068, right=0.988, top=0.812, bottom=0.198
)

hdr_artists = []

def binom_se(pct: float, n: float = 830.0) -> float:
    return float(np.sqrt(pct * (100.0 - pct) / n))


# ==============================================================================
# PANEL (a): Evidence Conversion Across Search Pipeline (BrowseComp-Plus, N=830)
# 3 distinct regimes (with shaded min-max envelope across all 3 0-shot arms)
# + deterministic micro-track stage dodging so zero markers ever stack!
# ==============================================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#FFFFFF")
ax1.set_xlim(-0.36, 3.86)
ax1.set_ylim(2.0, 72.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

stages_base = np.array([0.0, 1.0, 2.0, 3.0])

# Subtle cool-slate vertical stage column bands visually grouping the 3 dodged micro-tracks per stage
for sx in stages_base:
    ax1.axvspan(sx - 0.215, sx + 0.215, facecolor="#F8FAFC", edgecolor="none", alpha=0.95, zorder=0.2)
    ax1.axvline(sx, color="#EEF2F6", linewidth=0.65, zorder=0.4)

for yt in [10, 20, 30, 40, 50, 60, 70]:
    ax1.axhline(yt, color="#EEF2F6", linewidth=0.65, zorder=0.5)

# Exact 0-shot arms from Table 3 (100-Turn ReAct, 32K ReAct, 0-Shot Fork D=1 & D=inf v0):
z0_100t  = np.array([25.9,  6.4,  9.7, 24.6])
z0_32k   = np.array([28.6,  7.5, 12.2, 26.0])
z0_fork1 = np.array([32.2,  8.7, 12.5, 24.8])
z0_fork0 = np.array([30.5,  8.6, 12.5, 26.3])
z0_lo    = np.minimum.reduce([z0_100t, z0_32k, z0_fork1, z0_fork0])
z0_hi    = np.maximum.reduce([z0_100t, z0_32k, z0_fork1, z0_fork0])

DX_TRACK = 0.162
x_z0   = stages_base - DX_TRACK
x_grpo = stages_base
x_octo = stages_base + DX_TRACK

# Shaded 0-shot min-max range band across all untrained 4B variants (ReAct 100T / 32K / 0-Shot Fork)
ax1.fill_between(
    x_z0, z0_lo, z0_hi,
    facecolor="#CBD5E1", edgecolor="#94A3B8", linewidth=0.55, linestyle=(0, (2.2, 1.8)),
    alpha=0.38, zorder=2.2
)

arms_a = [
    ("base_0shot", "0-Shot Base (ReAct / Fork range)", x_z0,   z0_32k,
     "#475569", "#E2E8F0", "#334155", "-", 1.15, "o", 25, 3.4, None),
    ("grpo_143k",  "Single-Agent GRPO (143K Flat)",    x_grpo, np.array([30.5, 14.6, 21.1, 32.1]),
     "#C2410C", "#FED7AA", "#C2410C", "-", 1.38, "s", 28, 4.2, None),
    ("octopus",    "Octopus (4B, Ours)",               x_octo, np.array([58.2, 36.0, 43.3, 49.6]),
     "#0D9488", "#0D9488", "#064E3B", "-", 1.88, "o", 46, 5.5, np.array([1.39, 1.14, 1.47, 1.74])),
]

# Subtle Figure-8-style luminous halo on Octopus final accuracy point (x_octo[3], 49.6)
for r_mult in np.linspace(5.0, 1.5, 9):
    a_ring = 0.030 * ((5.4 - r_mult) / 3.9) ** 1.25
    ax1.scatter([x_octo[3]], [49.6], s=46.0 * (r_mult ** 1.05),
                facecolors="#14B8A6", edgecolors="none", alpha=a_ring, zorder=3.9)

seg_a_list = []
marker_a_info = []

for key, label, xs, ys, lc, fc, ec, ls, lw, mk, sz, z, custom_se in arms_a:
    is_ours = (key == "octopus")
    ses = custom_se if custom_se is not None else np.array([binom_se(float(v)) for v in ys])

    ax1.plot(xs, ys, color=lc, linestyle=ls, linewidth=lw,
             alpha=0.96 if is_ours else 0.88, zorder=z)
    for k in range(3):
        seg_a_list.append((f"{key}_seg{k}", xs[k], ys[k], xs[k + 1], ys[k + 1]))

    ax1.errorbar(
        xs, ys, yerr=ses, fmt="none", ecolor=ec,
        elinewidth=0.85, capsize=1.7, alpha=0.84, zorder=z + 0.1
    )
    if is_ours:
        ax1.scatter(xs, ys, s=sz * 1.48, facecolor="#FFFFFF", edgecolor="none",
                    alpha=0.95, zorder=z + 0.15)
    ax1.scatter(
        xs, ys, s=sz, facecolor=fc, edgecolor=ec, marker=mk,
        linewidth=1.35 if is_ours else 1.00, zorder=z + 0.3
    )
    for k in range(4):
        seg_a_list.append((f"{key}_eb{k}", xs[k], ys[k] - ses[k], xs[k], ys[k] + ses[k]))
        marker_a_info.append((f"{key}_{k}", float(xs[k]), float(ys[k]), sz))

# External 9B BM25 Context-Editing Reference at Stage 4: CLM + SCR (9B) [18] = 42.5%
X_CLM_REF = 3.53
Y_CLM_REF = 42.5
ax1.plot([x_octo[3] + 0.08, X_CLM_REF], [Y_CLM_REF, Y_CLM_REF],
         color="#1D4ED8", linestyle=(0, (2.0, 1.5)), linewidth=0.90, alpha=0.75, zorder=4.0)
seg_a_list.append(("clm_ref_tick", float(x_octo[3] + 0.08), Y_CLM_REF, X_CLM_REF, Y_CLM_REF))
ax1.scatter([X_CLM_REF], [Y_CLM_REF], s=32, facecolor="#93C5FD", edgecolor="none", alpha=0.50, zorder=4.6)
ax1.scatter([X_CLM_REF], [Y_CLM_REF], s=32, facecolor="none", edgecolor="#1D4ED8",
            linewidth=1.15, alpha=0.92, zorder=4.7)
marker_a_info.append(("clm_9b", X_CLM_REF, Y_CLM_REF, 32))

t1_list = []
t_clm = ax1.text(
    X_CLM_REF, 39.6, f"CLM+SCR\n(9B) {C_CLM}",
    fontsize=5.05, fontweight=500, color="#1D4ED8",
    ha="center", va="top", linespacing=1.10, zorder=7
)
t1_list.append(("CLM_9B", t_clm))

# Precision horizontal datum tick + vertical gain shaft at Gold Doc Opened (21.1% -> 43.3%, +22.2%)
ax1.plot([x_grpo[2] + 0.03, x_octo[2] + 0.03], [21.1, 21.1],
         color="#C2410C", linestyle=":", linewidth=0.80, alpha=0.65, zorder=3.8)
ax1.annotate(
    "", xy=(x_octo[2], 43.3 - 0.8), xytext=(x_octo[2], 21.1 + 0.3),
    arrowprops=dict(
        arrowstyle="-|>,head_length=0.40,head_width=0.23",
        color="#0D9488", lw=1.45, shrinkA=2.0, shrinkB=3.6
    ),
    zorder=5.0
)
seg_a_list.append(("gain_arrow_x2", float(x_octo[2]), 22.0, float(x_octo[2]), 41.3))
t_g22 = ax1.text(
    x_octo[2] - 0.08, 31.8, "+22.2%",
    fontsize=5.70, fontweight=700, color="#0F766E",
    ha="right", va="center", zorder=7
)
t1_list.append(("GAIN_22.2", t_g22))

# Precision horizontal datum tick + vertical gain shaft at Test Accuracy (32.1% -> 49.6%, +17.5%)
ax1.plot([x_grpo[3] + 0.03, x_octo[3] + 0.03], [32.1, 32.1],
         color="#C2410C", linestyle=":", linewidth=0.80, alpha=0.65, zorder=3.8)
ax1.annotate(
    "", xy=(x_octo[3], 49.6 - 0.8), xytext=(x_octo[3], 32.1 + 0.3),
    arrowprops=dict(
        arrowstyle="-|>,head_length=0.38,head_width=0.22",
        color="#0D9488", lw=1.45, shrinkA=2.0, shrinkB=3.8
    ),
    zorder=5.0
)
seg_a_list.append(("gain_arrow_x3", float(x_octo[3]), 33.0, float(x_octo[3]), 47.6))
t_g17 = ax1.text(
    x_octo[3] - 0.08, 40.5, "+17.5%",
    fontsize=5.65, fontweight=700, color="#0F766E",
    ha="right", va="center", zorder=7
)
t1_list.append(("GAIN_17.5", t_g17))

# Clean compact legend in upper-right concave white space (3 crisp rows, zero clutter)
h_octo = Line2D([0], [0], color="#0D9488", lw=1.85, marker="o", markersize=5.2,
                markerfacecolor="#0D9488", markeredgecolor="#064E3B", markeredgewidth=1.15)
h_grpo = Line2D([0], [0], color="#C2410C", lw=1.38, marker="s", markersize=4.3,
                markerfacecolor="#FED7AA", markeredgecolor="#C2410C", markeredgewidth=0.95)
h_base = (
    Patch(facecolor="#CBD5E1", edgecolor="#94A3B8", linestyle=(0, (2.0, 1.6)), linewidth=0.55, alpha=0.45),
    Line2D([0], [0], color="#475569", lw=1.15, marker="o", markersize=4.1,
           markerfacecolor="#E2E8F0", markeredgecolor="#334155", markeredgewidth=0.95)
)

leg1 = ax1.legend(
    [h_octo, h_grpo, h_base],
    ["Octopus (4B, Ours)", "Single-Agent GRPO (143K)", "0-Shot Base (ReAct / Fork)"],
    loc="upper right", bbox_to_anchor=(0.995, 0.996),
    frameon=True, facecolor="#FFFFFF", edgecolor="none", framealpha=0.92,
    fontsize=5.35, handlelength=1.75, handletextpad=0.48,
    labelspacing=0.32, borderpad=0.22
)
leg1.set_zorder(8)
for idx_l, txt_l in enumerate(leg1.get_texts()):
    if idx_l == 0:
        txt_l.set_fontweight(700)
        txt_l.set_color("#0F172A")
    else:
        txt_l.set_fontweight(500)
        txt_l.set_color("#334155")
    t1_list.append((f"LEG_{idx_l}", txt_l))

ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(
    ["Search\nRecall", "Evidence\nOpen Rec.", "Gold Open\nRecall", "Test\nAccuracy"],
    fontsize=6.05, color="#1E293B", linespacing=1.06
)
ax1.set_yticks([10, 20, 30, 40, 50, 60, 70])
ax1.set_yticklabels(["10%", "20%", "30%", "40%", "50%", "60%", "70%"], fontsize=6.4, color="#1E293B")
ax1.set_xlabel("Deep Search Pipeline Stage", fontsize=6.9, fontweight=600, color="#0F172A", labelpad=4.0)
ax1.set_ylabel("BrowseComp-Plus Rate (%)", fontsize=6.9, fontweight=600, color="#0F172A", labelpad=3.5)

pos1 = ax1.get_position()
c1 = 0.5 * (pos1.x0 + pos1.x1)
hdr_artists.append(("HDR_A_TITLE", fig.text(
    c1, 0.936, "(a) Evidence Conversion Across Search Pipeline",
    fontsize=7.30, fontweight=700, color="#0F172A", ha="center", va="center"
)))
hdr_artists.append(("HDR_A_SUB", fig.text(
    c1, 0.876, "830-query BrowseComp-Plus (±1 SE)  ·  Qwen3.5-4B, Sparse BM25",
    fontsize=5.55, fontweight=400, color="#64748B", ha="center", va="center"
)))


# ==============================================================================
# PANEL (b): Cross-Branch Sharing: Retrieval vs. Off-Task Hijacking
# Genius Broken Piecewise-Linear X-Axis:
#   - Regime 1 (Low-Hijack Selective KV, x in [0%, 10.5%]): 58% of axis width, linear
#   - Axis Break Gap (x in [10.5%, 47.5%], zero data points): 6% break notch '//'
#   - Regime 2 (Full-Context Pollution, x in [47.5%, 64.0%]): 36% of axis width, linear
#   - 1 crisp jewel-ring marker + +/-1 SE per paradigm (RL-Trained, N=366 mounts)
# ==============================================================================
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#FFFFFF")

B_L_MAX = 10.5
B_R_MIN = 46.8
B_R_MAX = 64.8
U_L = 0.51
U_R = 0.57

def fwd_broken(x):
    xa = np.asarray(x, dtype=float)
    out = np.empty_like(xa, dtype=float)
    m_l = xa <= B_L_MAX
    m_r = xa >= B_R_MIN
    m_m = (~m_l) & (~m_r)
    out[m_l] = (xa[m_l] / B_L_MAX) * U_L
    out[m_m] = U_L + ((xa[m_m] - B_L_MAX) / (B_R_MIN - B_L_MAX)) * (U_R - U_L)
    out[m_r] = U_R + ((xa[m_r] - B_R_MIN) / (B_R_MAX - B_R_MIN)) * (1.0 - U_R)
    return out

def inv_broken(u):
    ua = np.asarray(u, dtype=float)
    out = np.empty_like(ua, dtype=float)
    m_l = ua <= U_L
    m_r = ua >= U_R
    m_m = (~m_l) & (~m_r)
    out[m_l] = (ua[m_l] / U_L) * B_L_MAX
    out[m_m] = B_L_MAX + ((ua[m_m] - U_L) / (U_R - U_L)) * (B_R_MIN - B_L_MAX)
    out[m_r] = B_R_MIN + ((ua[m_r] - U_R) / (1.0 - U_R)) * (B_R_MAX - B_R_MIN)
    return out

ax2.set_xscale(FuncScale(ax2, (fwd_broken, inv_broken)))
X2_MIN, X2_MAX = -0.35, B_R_MAX
Y2_MIN, Y2_MAX = -4.2, 97.5
ax2.set_xlim(X2_MIN, X2_MAX)
ax2.set_ylim(Y2_MIN, Y2_MAX)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

for yt in [0, 20, 40, 60, 80]:
    ax2.axhline(yt, color="#EEF2F6", linewidth=0.65, zorder=0.5)
for xt in [0, 3, 6, 9, 50, 56, 62]:
    ax2.axvline(xt, color="#EEF2F6", linewidth=0.65, zorder=0.5)

X2_OURS = 2.7
Y2_OURS = 61.2

# Exact Figure-8-style blush-terracotta Dominated Quadrant + #D97757 dashed crosshairs
ax2.fill_between(
    [X2_OURS, B_L_MAX], [Y2_MIN, Y2_MIN], [Y2_OURS, Y2_OURS],
    facecolor="#FCEAE6", alpha=0.62, zorder=1.0
)
ax2.fill_between(
    [B_R_MIN, X2_MAX], [Y2_MIN, Y2_MIN], [Y2_OURS, Y2_OURS],
    facecolor="#FCEAE6", alpha=0.62, zorder=1.0
)
ax2.plot([X2_OURS, X2_OURS], [Y2_MIN, Y2_OURS], color="#D97757", linestyle=(0, (4.0, 2.6)),
         linewidth=0.95, alpha=0.82, zorder=2.2)
ax2.plot([X2_OURS, B_L_MAX], [Y2_OURS, Y2_OURS], color="#D97757", linestyle=(0, (4.0, 2.6)),
         linewidth=0.95, alpha=0.82, zorder=2.2)
ax2.plot([B_R_MIN, X2_MAX], [Y2_OURS, Y2_OURS], color="#D97757", linestyle=(0, (4.0, 2.6)),
         linewidth=0.95, alpha=0.82, zorder=2.2)

# Clean white vertical break corridor + crisp '//' diagonal break slashes on bottom spine
ax2.axvspan(B_L_MAX, B_R_MIN, facecolor="#FFFFFF", edgecolor="none", zorder=2.6)
ax2.axvline(B_L_MAX, color="#CBD5E1", linestyle=(0, (2.2, 2.0)), linewidth=0.75, zorder=2.7)
ax2.axvline(B_R_MIN, color="#CBD5E1", linestyle=(0, (2.2, 2.0)), linewidth=0.75, zorder=2.7)

# Mask out the bottom spine segment inside the break gap and draw crisp '//' break slashes in axes coords
u_min = float(fwd_broken(X2_MIN))
u_max = float(fwd_broken(X2_MAX))
def u_to_ax(u_val: float) -> float:
    return (u_val - u_min) / (u_max - u_min)

ax_brk_l = u_to_ax(U_L)
ax_brk_r = u_to_ax(U_R)
ax2.plot(
    [ax_brk_l + 0.004, ax_brk_r - 0.004], [0.0, 0.0],
    transform=ax2.transAxes, color="#FFFFFF", linewidth=2.4, solid_capstyle="butt",
    clip_on=False, zorder=6.0
)
slash_dx, slash_dy = 0.011, 0.024
for cx_b in (ax_brk_l + 0.006, ax_brk_r - 0.006):
    ax2.plot(
        [cx_b - slash_dx, cx_b + slash_dx], [-slash_dy, slash_dy],
        transform=ax2.transAxes, color="#334155", linewidth=0.90,
        clip_on=False, zorder=6.5
    )

t2_list = []

# Subtle Figure-8-style top regime headers
t_reg_l = ax2.text(
    0.05, 95.2, "SELECTIVE KV  (0% – 10%)",
    fontsize=5.30, fontweight=600, color="#64748B", ha="left", va="top", zorder=4
)
t_reg_r = ax2.text(
    56.0, 95.2, "FULL-CONTEXT COPY",
    fontsize=5.30, fontweight=600, color="#64748B", ha="center", va="top", zorder=4
)
t2_list.extend([("HDR_REG_L", t_reg_l), ("HDR_REG_R", t_reg_r)])

t_dom2 = ax2.text(
    63.6, 0.2,
    "Dominated by Octopus\n(Higher Hijack & Lower EM)",
    fontsize=5.10, fontweight=600, color="#9A3412", alpha=0.88,
    ha="right", va="bottom", linespacing=1.10, zorder=3.5,
    bbox=dict(facecolor="#FDF2EF", edgecolor="none", pad=1.0, alpha=0.92)
)
t2_list.append(("DOM_CAPTION_B", t_dom2))

# Exact Run 2 pooled ('all', 366 mounts, 947 probes) RL-Trained data from repo/docs/kv-read-probe.md:
# (key, x_rl, x_se_lo, x_se_hi, y_rl, y_se_lo, y_se_hi, fc, ec, mk, sz_rl)
paradigms_b = [
    ("none",    0.5, 0.26, 0.46,  3.1, 0.66, 0.71, "#CBD5E1", "#475569", "o", 38),
    ("K1",      2.5, 0.71, 0.82, 52.1, 2.86, 2.81, "#93C5FD", "#1D4ED8", "o", 40),
    ("K3",      7.1, 1.43, 1.53, 60.1, 2.65, 2.50, "#FDE68A", "#B45309", "s", 40),
    ("K2",      2.7, 0.82, 1.02, 61.2, 2.65, 2.45, "#0D9488", "#064E3B", "o", 70),
    ("replay", 55.2, 3.21, 2.96, 71.5, 2.86, 2.55, "#DDD6FE", "#6D28D9", "s", 40),
    ("text",   57.9, 3.21, 3.27, 76.0, 2.55, 2.30, "#FECACA", "#DC2626", "s", 40),
]

# Luminous halo on Octopus hero point (2.7, 61.2) matching Figure 8
for r_mult in np.linspace(5.2, 1.5, 10):
    a_ring = 0.032 * ((5.6 - r_mult) / 4.1) ** 1.25
    ax2.scatter([X2_OURS], [Y2_OURS], s=70.0 * (r_mult ** 1.06),
                facecolors="#14B8A6", edgecolors="none", alpha=a_ring, zorder=3.6)

seg_b_list = []
marker_b_info = []

for key, xr, dx_lo, dx_hi, yr, dy_lo, dy_hi, fc, ec, mk, sz_r in paradigms_b:
    is_ours = (key == "K2")

    ax2.errorbar(
        [xr], [yr], xerr=[[dx_lo], [dx_hi]], yerr=[[dy_lo], [dy_hi]],
        fmt="none", ecolor=ec, elinewidth=0.90, capsize=1.9, alpha=0.84, zorder=4.4
    )
    seg_b_list.append((f"{key}_eb_h", xr - dx_lo, yr, xr + dx_hi, yr))
    seg_b_list.append((f"{key}_eb_v", xr, yr - dy_lo, xr, yr + dy_hi))

    if is_ours:
        ax2.scatter([xr], [yr], s=sz_r * 1.54, facecolor="#FFFFFF", edgecolor="none",
                    alpha=0.96, zorder=5.0)
        ax2.scatter([xr], [yr], s=sz_r, facecolor=fc, edgecolor=ec, marker=mk,
                    linewidth=1.55, zorder=5.4)
    else:
        ax2.scatter([xr], [yr], s=sz_r, facecolor=fc, edgecolor="none", marker=mk,
                    alpha=0.64, zorder=4.7)
        ax2.scatter([xr], [yr], s=sz_r, facecolor="none", edgecolor=ec, marker=mk,
                    linewidth=1.20, alpha=0.94, zorder=4.8)
    marker_b_info.append((key, xr, yr, sz_r))

# Clean, spacious Figure-8-style single-point labels with external [N] citations
# 1. Octopus (Ours) — cleanly above (2.7, 61.2)
t_octo = ax2.text(
    0.15, 67.2, "Octopus (Ours)\n0-Copy · 2.7%",
    fontsize=6.25, fontweight=700, color="#0F766E",
    ha="left", va="bottom", linespacing=1.14, zorder=8
)
t2_list.append(("K2_LBL", t_octo))

# 2. KVCOMM / Hogwild! [32, 34] — cleanly above (7.1, 60.1) in wide-open upper-middle space
t_k3 = ax2.text(
    7.45, 64.8, f"KVCOMM /\nHogwild! {C_KVC}",
    fontsize=5.30, fontweight=500, color="#334155",
    ha="center", va="bottom", linespacing=1.08, zorder=7
)
t2_list.append(("K3_LBL", t_k3))

# 3. Unshifted Zero-Copy — clean 1-line label directly right of (2.5, 52.1) & x=2.7 crosshair
t_k1 = ax2.text(
    3.62, 52.1, "Unshifted Zero-Copy",
    fontsize=5.35, fontweight=500, color="#334155",
    ha="left", va="center", zorder=7
)
t2_list.append(("K1_LBL", t_k1))

# 4. No Sibling Read — cleanly above (0.5, 3.1), strictly left of x=2.7 crosshair
t_none = ax2.text(
    -0.18, 8.0, "No Sibling\nRead",
    fontsize=5.10, fontweight=500, color="#475569",
    ha="left", va="bottom", linespacing=1.06, zorder=7
)
t2_list.append(("NONE_LBL", t_none))

# 5. Transcript Copy (AutoGen [6]) — clean 1-line label above (57.9, 76.0) top errorbar cap (78.30)
t_text = ax2.text(
    56.0, 80.6, f"Transcript Copy (AutoGen {C_AG})",
    fontsize=5.15, fontweight=500, color="#1E293B",
    ha="center", va="bottom", zorder=7
)
t2_list.append(("TEXT_LBL", t_text))

# 6. Full-Layer Splice (CacheBlend [29]) — clean 1-line label strictly between y=61.2 crosshair and 68.64 bottom cap (zero leader line!)
t_rep = ax2.text(
    55.6, 63.0, f"Full-Layer Splice (CacheBlend {C_CB})",
    fontsize=5.10, fontweight=500, color="#1E293B",
    ha="center", va="bottom", zorder=7
)
t2_list.append(("REPLAY_LBL", t_rep))

# Register crosshair segments in seg_b_list so no label ever touches dashed crosshairs
seg_b_list.append(("crosshair_v", X2_OURS, Y2_MIN, X2_OURS, Y2_OURS))
seg_b_list.append(("crosshair_hl", X2_OURS, Y2_OURS, B_L_MAX, Y2_OURS))
seg_b_list.append(("crosshair_hr", B_R_MIN, Y2_OURS, X2_MAX, Y2_OURS))

ax2.set_xticks([0, 3, 6, 9, 50, 56, 62])
ax2.set_xticklabels(["0%", "3%", "6%", "9%", "50%", "56%", "62%"], fontsize=6.25, color="#1E293B")
ax2.set_yticks([0, 20, 40, 60, 80])
ax2.set_yticklabels(["0%", "20%", "40%", "60%", "80%"], fontsize=6.4, color="#1E293B")
ax2.minorticks_off()
ax2.set_xlabel("Off-Task Hijack Rate (%, Broken Linear Scale)",
               fontsize=6.9, fontweight=600, color="#0F172A", labelpad=4.0)
ax2.set_ylabel("Sibling Fact Exact Match (%)",
               fontsize=6.9, fontweight=600, color="#0F172A", labelpad=3.5)

pos2 = ax2.get_position()
c2 = 0.5 * (pos2.x0 + pos2.x1)
hdr_artists.append(("HDR_B_TITLE", fig.text(
    c2, 0.936, "(b) Cross-Branch Sharing: Retrieval vs. Hijacking",
    fontsize=7.30, fontweight=700, color="#0F172A", ha="center", va="center"
)))
hdr_artists.append(("HDR_B_SUB", fig.text(
    c2, 0.876, "366 live mounts, 947 probes (±1 SE)  ·  ● 0 Extra KV   ■ +100% KV Copy",
    fontsize=5.50, fontweight=400, color="#64748B", ha="center", va="center"
)))


# ==============================================================================
# EXHAUSTIVE PIXEL-LEVEL GEOMETRY AUDIT (INCLUDING MARKER-VS-MARKER & EB-VS-EB!)
# ==============================================================================
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
dpi_scale = fig.dpi / 72.0

def seg_intersects_box(p1, p2, bb, pad=1.0):
    x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
    ts = np.linspace(0.0, 1.0, 60)
    px = p1[0] + ts * (p2[0] - p1[0])
    py = p1[1] + ts * (p2[1] - p1[1])
    return bool(np.any((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)))

def segs_intersect(a1, a2, b1, b2) -> bool:
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2 = ccw(a1, a2, b1), ccw(a1, a2, b2)
    d3, d4 = ccw(b1, b2, a1), ccw(b1, b2, a2)
    return (d1 * d2 < 0.0) and (d3 * d4 < 0.0)

issues = []

# 0. Header text overlap audit
for i in range(len(hdr_artists)):
    n1, a1 = hdr_artists[i]
    b1 = a1.get_window_extent(renderer=renderer)
    for j in range(i + 1, len(hdr_artists)):
        n2, a2 = hdr_artists[j]
        b2 = a2.get_window_extent(renderer=renderer)
        if b1.overlaps(b2):
            issues.append(f"[HDR] Text overlap: {n1} vs {n2}")

def audit_panel(tag: str, ax, t_list, m_list, s_list, min_marker_gap_pt: float = 2.0):
    ax_bb = ax.get_window_extent(renderer=renderer)

    # 1. STRICT MARKER-VS-MARKER CLEARANCE AUDIT
    for i in range(len(m_list)):
        m1, mx1, my1, sz1 = m_list[i]
        p1 = ax.transData.transform((mx1, my1))
        r1_px = (np.sqrt(sz1) * 0.54) * dpi_scale
        for j in range(i + 1, len(m_list)):
            m2, mx2, my2, sz2 = m_list[j]
            p2 = ax.transData.transform((mx2, my2))
            r2_px = (np.sqrt(sz2) * 0.54) * dpi_scale
            dist_px = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            gap_pt = (dist_px - (r1_px + r2_px)) / dpi_scale
            if gap_pt < min_marker_gap_pt:
                issues.append(f"[{tag}] MARKER OVERLAP/CROWDING: {m1} vs {m2} (gap={gap_pt:.2f} pt < {min_marker_gap_pt} pt)")

    # 2. STRICT ERRORBAR-VS-ERRORBAR INTERSECTION AUDIT
    eb_only = [s for s in s_list if "_eb" in s[0]]
    for i in range(len(eb_only)):
        s1, x1a, y1a, x1b, y1b = eb_only[i]
        base1 = s1.split("_eb")[0]
        pa1, pb1 = ax.transData.transform((x1a, y1a)), ax.transData.transform((x1b, y1b))
        for j in range(i + 1, len(eb_only)):
            s2, x2a, y2a, x2b, y2b = eb_only[j]
            base2 = s2.split("_eb")[0]
            if base1 == base2:
                continue
            pa2, pb2 = ax.transData.transform((x2a, y2a)), ax.transData.transform((x2b, y2b))
            if segs_intersect(pa1, pb1, pa2, pb2):
                issues.append(f"[{tag}] ERRORBAR CROSSING: {s1} vs {s2}")

    # 3. TEXT-VS-TEXT, TEXT-VS-SPINE, TEXT-VS-MARKER, TEXT-VS-SEGMENT AUDIT
    bboxes = [(n, t.get_window_extent(renderer=renderer)) for n, t in t_list]
    for i in range(len(bboxes)):
        n1, b1 = bboxes[i]
        if (b1.x0 < ax_bb.x0 + 1.0 or b1.x1 > ax_bb.x1 + 2.0 or
                b1.y0 < ax_bb.y0 + 1.0 or b1.y1 > ax_bb.y1 + 2.0):
            issues.append(f"[{tag}] Text out of axes bounds: {n1}")
        for j in range(i + 1, len(bboxes)):
            n2, b2 = bboxes[j]
            if b1.expanded(1.03, 1.06).overlaps(b2):
                issues.append(f"[{tag}] Text-vs-text overlap: {n1} vs {n2}")
        for mk_name, mx, my, sz in m_list:
            px, py = ax.transData.transform((mx, my))
            rad = (np.sqrt(sz) * 0.56 + 1.4) * dpi_scale
            if (b1.x0 - rad <= px <= b1.x1 + rad) and (b1.y0 - rad <= py <= b1.y1 + rad):
                issues.append(f"[{tag}] Text hits marker: {n1} vs {mk_name}")
        for s_name, sx0, sy0, sx1, sy1 in s_list:
            if n1.startswith("LEG_"):
                continue
            if n1 == "CLM_9B" and s_name == "clm_leader":
                continue
            if n1 == "REPLAY_LBL" and s_name == "replay_leader":
                continue
            p1 = ax.transData.transform((sx0, sy0))
            p2 = ax.transData.transform((sx1, sy1))
            if seg_intersects_box(p1, p2, b1, pad=0.8 * dpi_scale):
                issues.append(f"[{tag}] Text hits segment: {n1} vs {s_name}")

audit_panel("PanelA", ax1, t1_list, marker_a_info, seg_a_list, min_marker_gap_pt=1.8)
audit_panel("PanelB", ax2, t2_list, marker_b_info, seg_b_list, min_marker_gap_pt=3.5)

if issues:
    print("GEOMETRY AUDIT FAILED:")
    for it in issues:
        print("  -", it)
    raise SystemExit(1)

print("GEOMETRY AUDIT PASSED: 0 marker overlaps, 0 errorbar crossings, 0 text collisions!")

out_pdf = ROOT / "figures/fig_exp2_efficiency.pdf"
fig.savefig(out_pdf)
print("Saved:", out_pdf)
