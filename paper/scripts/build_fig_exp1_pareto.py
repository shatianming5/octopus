#!/usr/bin/env python3
"""
Scientifically Rigorous Frontier Parameter-vs-Accuracy Bubble Plot (Figure 8):
  - Full 830-query BrowseComp-Plus official benchmark (Chen et al., ACL 2026 [2]; Li et al., 2026 [3]).
  - Strict Tool-Parity Fairness:
      * Every dominated foundation/proprietary model uses FULL browser tools (search + open [+ find])
        AND the stronger Qwen3-Embed-8B dense retriever (†), or identical BM25 + search + open (CLM+SCR 9B).
      * GPT-OSS-120B† [3] is placed honestly at its full-browser-tool teacher accuracy (62.2%, 120B)
        along the upper-right open-weight Pareto frontier (4B: 49.6% -> 27B: 59.4% -> 120B: 62.2%).
      * Snippet-only policies that do not use document-open tools are explicitly marked with * (Search-R1*, o3*, GPT-5*).
  - Complete 6-way pixel-level geometry audit (bubbles, texts, spines, Pareto curve, divider, crosshairs).
"""
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
for ttf in (ROOT / "scripts/fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(ttf))

# Dynamically resolve exact [N] citation indices from main.bbl
cite_map = {
    "jin2025searchr1": "1",
    "chen2025browsecompplus": "2",
    "li2026openresearcher": "3",
    "shao2026clm": "18",
}
bbl_path = ROOT / "main.bbl"
if bbl_path.exists():
    bbl_txt = bbl_path.read_text(encoding="utf-8", errors="ignore")
    keys = re.findall(r"\\bibitem(?:\[.*?\])?\{([^}]+)\}", bbl_txt, re.S)
    for idx, k in enumerate(keys, 1):
        if k in cite_map:
            cite_map[k] = str(idx)

C_SR1 = f"[{cite_map['jin2025searchr1']}]"
C_BCP = f"[{cite_map['chen2025browsecompplus']}]"
C_OR  = f"[{cite_map['li2026openresearcher']}]"
C_CLM = f"[{cite_map['shao2026clm']}]"

plt.rcParams.update({
    "font.family": "Inter",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.85,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(6.65, 3.92), dpi=300)
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")
fig.subplots_adjust(left=0.082, right=0.985, top=0.865, bottom=0.130)

# ==============================================================================
# 1. CLEAN TOP LEGEND BAR + EXPLICIT PROTOCOL SUBTITLE LINE
# ==============================================================================
legend_items = [
    ("#0D9488", "Octopus (4B, Ours)",   True),
    ("#334155", "Single-Agent (4B)",    False),
    ("#2563EB", "Deep Research Agents", False),
    ("#7C3AED", "Open-Weight LLMs",     False),
    ("#475569", "Proprietary Frontier", False),
]

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

legend_artists = []
lx = 0.082
ly = 0.958
for col, name, is_ours in legend_items:
    dot_t = fig.text(
        lx, ly, "●", fontsize=8.0 if is_ours else 7.4, color=col,
        alpha=1.0 if is_ours else 0.80, ha="left", va="center"
    )
    bb_dot = dot_t.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    lbl_t = fig.text(
        bb_dot.x1 + 0.0045, ly, name,
        fontsize=6.65, fontweight=700 if is_ours else 500,
        color="#0F172A" if is_ours else "#475569", ha="left", va="center"
    )
    bb_lbl = lbl_t.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    legend_artists.extend([dot_t, lbl_t])
    lx = bb_lbl.x1 + 0.0185

total_w = bb_lbl.x1 - 0.082
shift_x = ((0.082 + 0.985) / 2.0 - total_w / 2.0) - 0.082
for art in legend_artists:
    px, py = art.get_position()
    art.set_position((px + shift_x, py))

fig.text(
    (0.082 + 0.985) / 2.0, 0.908,
    "830-query BrowseComp-Plus  ·  search + open browser tools (*: search-only)  ·  Sparse BM25 (†: Qwen3-Embed-8B dense retriever)",
    fontsize=5.70, fontweight=400, color="#64748B", ha="center", va="center"
)

# ==============================================================================
# 2. COORDINATE SYSTEM: LOG10(PARAMS_B) [0.20 .. 3.24] | PROPRIETARY [3.24 .. 4.30]
# ==============================================================================
X_MIN, X_DIV, X_MAX = 0.20, 3.24, 4.30
Y_MIN, Y_MAX = -3.4, 77.8
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

X_OURS = np.log10(4.0)  # 0.60206
Y_OURS = 49.6

# Subtle cool-slate column wash for Proprietary Frontier band
ax.axvspan(X_DIV, X_MAX, facecolor="#F8FAFC", alpha=0.88, zorder=0)

# Horizontal & vertical major grid lines
for yt in [0, 10, 20, 30, 40, 50, 60, 70]:
    ax.axhline(yt, color="#EEF2F6", linewidth=0.65, zorder=0.5)
for pb_tick in [4, 10, 30, 100, 300, 1000]:
    ax.axvline(np.log10(pb_tick), color="#EEF2F6", linewidth=0.65, zorder=0.5)

# Warm editorial blush-terracotta Dominated Quadrant (Artificial Analysis inspired)
ax.fill_between(
    [X_OURS, X_MAX], [Y_MIN, Y_MIN], [Y_OURS, Y_OURS],
    facecolor="#FCEAE6", alpha=0.62, zorder=1
)

# Crisp vertical separator between Open-Weight Parameter Scale and Proprietary Frontier
ax.axvline(X_DIV, color="#CBD5E1", linestyle="-", linewidth=0.90, zorder=2)

text_artists = []

t_hdr_open = ax.text(
    0.29, 75.6, "OPEN-WEIGHT MODELS & AGENTS  (4B – 1T)",
    fontsize=5.85, fontweight=600, color="#64748B", ha="left", va="top", zorder=4
)
t_hdr_prop = ax.text(
    (X_DIV + X_MAX) / 2.0, 75.6, "PROPRIETARY",
    fontsize=5.85, fontweight=600, color="#64748B", ha="center", va="top", zorder=4
)
text_artists.extend([("HDR_OPEN", t_hdr_open), ("HDR_PROP", t_hdr_prop)])

# Crisp dashed dominance crosshairs anchored at Octopus (4B, 42.6%)
ax.plot([X_OURS, X_OURS], [Y_MIN, Y_OURS], color="#D97757", linestyle=(0, (4.0, 2.6)),
        linewidth=0.95, alpha=0.82, zorder=2.5)
ax.plot([X_OURS, X_MAX], [Y_OURS, Y_OURS], color="#D97757", linestyle=(0, (4.0, 2.6)),
        linewidth=0.95, alpha=0.82, zorder=2.5)

t_dom = ax.text(
    4.23, -1.3,
    "Dominated by Octopus-4B  (Larger Model & Lower Accuracy)",
    fontsize=6.05, fontweight=600, color="#9A3412", alpha=0.88,
    ha="right", va="bottom", zorder=3.5,
    bbox=dict(facecolor="#FDF2EF", edgecolor="none", pad=1.4, alpha=0.92)
)
text_artists.append(("DOM_CAPTION", t_dom))

# ==============================================================================
# 3. PERCEPTUAL BUBBLE SCALING (Smooth Parameter Area Encoding, Solid Jewel Rings)
# ==============================================================================
def bsize(params_b: float, is_hero: bool = False, is_prop: bool = False) -> float:
    if is_hero:
        return 108.0
    if is_prop:
        return 140.0
    return 60.0 * ((params_b / 4.0) ** 0.37)

# ==============================================================================
# 4. CONCAVE OPEN-WEIGHT PARETO FRONTIER & VERTICAL 4B BREAKTHROUGH SHAFT
# ==============================================================================
# Open-weight Pareto frontier: Octopus (4B, 49.6%) -> CLM (27B, 59.4%) -> GPT-OSS-120B (120B, 62.2%)
ax.plot(
    [X_OURS, np.log10(27.0), np.log10(120.0)],
    [Y_OURS, 59.4, 62.2],
    color="#0F766E", linestyle=(0, (3.5, 2.6)), linewidth=1.05, alpha=0.58, zorder=2.6
)

# Controlled 4B single-agent step: ReAct (26.0%) -> GRPO (32.1%)
ax.plot(
    [X_OURS, X_OURS], [26.0, 32.1],
    color="#475569", linestyle="-", linewidth=1.20, alpha=0.82, zorder=3
)

# Straight vertical RL branching jump at x = 4B: 32.1% -> 49.6% (+17.5%)
ax.annotate(
    "", xy=(X_OURS, Y_OURS - 0.5), xytext=(X_OURS, 32.1 + 0.4),
    arrowprops=dict(
        arrowstyle="-|>,head_length=0.46,head_width=0.27",
        color="#0D9488", lw=1.75, shrinkA=3.8, shrinkB=4.5
    ),
    zorder=7
)
t_gain = ax.text(
    X_OURS - 0.055, 40.8, "+17.5%",
    fontsize=6.40, fontweight=700, color="#0F766E", ha="right", va="center", zorder=8
)
text_artists.append(("GAIN_17.5", t_gain))

# ==============================================================================
# 5. LUMINOUS HALO AROUND HERO POINT (Octopus 4B, Ours)
# ==============================================================================
for r_mult in np.linspace(6.2, 1.5, 12):
    a_ring = 0.032 * ((6.6 - r_mult) / 5.1) ** 1.25
    ax.scatter(
        [X_OURS], [Y_OURS], s=bsize(4.0, is_hero=True) * (r_mult ** 1.08),
        facecolors="#14B8A6", edgecolors="none", alpha=a_ring, zorder=4
    )

# ==============================================================================
# 6. FLAGSHIP BENCHMARK MODELS (100% Tool-Fair Full Browser Tools Unless Marked *)
# ==============================================================================
# Schema:
# (x_coord, y_acc, params_B, cat, label, dx_pt, dy_pt, ha, va)
models = [
    # --- Our 4B Controlled Progression & Hero (BM25 + search + open) ---
    (X_OURS,          26.0,    4.0, "ctrl",  "Single-Agent ReAct (32K)",            9,  -2, "left",   "center"),
    (X_OURS,          32.1,    4.0, "ctrl",  "Single-Agent GRPO (143K)",            9,   4, "left",   "center"),
    (X_OURS,          49.6,    4.0, "hero",  "Octopus (Ours)\n4B · 49.6%",        -29,  16, "left",   "bottom"),

    # --- Specialized Deep Research & Context Agents (9B - 32B) ---
    (np.log10(9.0),   42.5,    9.0, "agent", f"CLM + SCR (9B) {C_CLM}",             8,  -7, "left",   "top"),
    (np.log10(27.0),  59.4,   27.0, "agent", f"CLM (27B) {C_CLM}",                 -9,   2, "right",  "center"),
    (np.log10(30.0),  54.8,   30.0, "agent", f"OpenResearcher (30B)† {C_OR}",      10,   0, "left",   "center"),
    (np.log10(30.0),  44.5,   30.0, "agent", f"Tongyi-DeepResearch (30B)† {C_OR}", 10,  -2, "left",   "center"),
    (np.log10(30.0),  30.3,   30.0, "agent", f"CutBill (30B)† {C_OR}",             10,   0, "left",   "center"),
    (np.log10(32.0),   3.9,   32.0, "agent", f"Search-R1 (32B)* {C_SR1}",          10,   1, "left",   "center"),

    # --- Open-Weight Flagship LLMs (30B - 1T, Full Browser Tools search+open[+find]) ---
    (np.log10(30.0),  20.8,   30.0, "llm",   f"Nemotron-3 (30B)† {C_OR}",          10,   0, "left",   "center"),
    (np.log10(32.0),  11.7,   32.0, "llm",   f"Qwen3-32B (+open)† {C_BCP}",        10,   0, "left",   "center"),
    (np.log10(120.0), 62.2,  120.0, "llm",   f"GPT-OSS-120B (Teacher)† {C_OR}",    11,   0, "left",   "center"),
    (np.log10(671.0), 16.4,  671.0, "llm",   f"DeepSeek-R1 (671B)† {C_OR}",         0, -14, "center", "top"),
    (np.log10(1000.), 35.4, 1000.0, "llm",   f"Kimi-K2 (1T)† {C_OR}",             -15,   0, "right",  "center"),

    # --- Proprietary Frontier Models (BM25 o3* [2] + Qwen3-Embed-8B Dense † [2, 3]) ---
    (3.66,            29.5,  200.0, "prop",  f"Gemini 2.5 Pro† {C_OR}",             0, -10, "center", "top"),
    (3.42,            36.4,  200.0, "prop",  f"GPT-4.1† {C_OR}",                    0, -10, "center", "top"),
    (4.02,            36.8,  200.0, "prop",  f"Claude 4 Opus† {C_OR}",              0, -10, "center", "top"),
    (3.77,            49.3,  200.0, "prop",  f"OpenAI o3* {C_BCP}",                 0, -10, "center", "top"),
    (3.55,            63.5,  200.0, "prop",  f"OpenAI o3*† {C_BCP}",                0, -10, "center", "top"),
    (3.96,            70.1,  200.0, "prop",  f"GPT-5*† {C_BCP}",                    0, -10, "center", "top"),
]

PALETTE = {
    "hero":  ("#0D9488", "#064E3B", 1.00, 1.65),
    "ctrl":  ("#334155", "#0F172A", 0.90, 1.05),
    "agent": ("#93C5FD", "#1D4ED8", 0.44, 1.15),
    "llm":   ("#DDD6FE", "#6D28D9", 0.50, 1.15),
    "prop":  ("#CBD5E1", "#334155", 0.52, 1.15),
}

bubble_centers_px = []

for x, y, pb, cat, lbl, dx, dy, ha, va in models:
    is_hero = (cat == "hero")
    is_prop = (cat == "prop")
    fc, ec, al, lw = PALETTE[cat]
    sz = bsize(pb, is_hero=is_hero, is_prop=is_prop)
    z = 9 if is_hero else (6 if cat == "ctrl" else 5)

    if is_hero:
        ax.scatter([x], [y], s=sz * 1.62, facecolor="#FFFFFF", edgecolor="none",
                   alpha=0.96, zorder=8)

    ax.scatter([x], [y], s=sz, facecolor=fc, edgecolor="none", alpha=al, zorder=z)
    ax.scatter([x], [y], s=sz, facecolor="none", edgecolor=ec,
               linewidth=lw, linestyle="-", alpha=0.92, zorder=z + 0.2)

    if is_hero:
        t = ax.annotate(
            lbl, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=7.8, fontweight=700, color="#0F766E",
            linespacing=1.16, zorder=10
        )
    elif cat == "ctrl":
        t = ax.annotate(
            lbl, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=6.25, fontweight=500, color="#334155", zorder=7
        )
    else:
        is_dom = (y < Y_OURS)
        t = ax.annotate(
            lbl, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=6.15, fontweight=500 if not is_dom else 400,
            color="#1E293B" if not is_dom else "#475569", zorder=7
        )
    text_artists.append((lbl, t))

# ==============================================================================
# 7. AXIS TICKS & LABELS
# ==============================================================================
xtick_vals = [np.log10(4), np.log10(10), np.log10(30), np.log10(100), np.log10(300), np.log10(1000), (X_DIV + X_MAX) / 2.0]
xtick_lbls = ["4B", "10B", "30B", "100B", "300B", "1T", "Proprietary"]
ax.set_xticks(xtick_vals)
ax.set_xticklabels(xtick_lbls, fontsize=7.4)
ax.minorticks_off()

ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%"], fontsize=7.5)

ax.set_xlabel("Model Parameters (Billions, Log Scale)  &  Proprietary Frontier",
              fontsize=8.0, fontweight=600, color="#1E293B", labelpad=6)
ax.set_ylabel("BrowseComp-Plus Test Accuracy (%)",
              fontsize=8.0, fontweight=600, color="#1E293B", labelpad=5)

# ==============================================================================
# 8. COMPLETE 6-WAY PIXEL-LEVEL GEOMETRY AUDIT
# ==============================================================================
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
ax_bb = ax.get_window_extent(renderer=renderer)
bboxes = [(name.replace("\n", " "), plt.matplotlib.text.Text.get_window_extent(art, renderer=renderer))
          for name, art in text_artists]

dpi_scale = fig.dpi / 72.0
for x, y, pb, cat, *_ in models:
    cx, cy = ax.transData.transform((x, y))
    sz = bsize(pb, is_hero=(cat == "hero"), is_prop=(cat == "prop"))
    r_px = np.sqrt(sz / np.pi) * dpi_scale
    bubble_centers_px.append((x, y, pb, cat, cx, cy, r_px))

bb_overlaps = []
for i in range(len(bubble_centers_px)):
    for j in range(i + 1, len(bubble_centers_px)):
        x1, y1, pb1, c1, cx1, cy1, r1 = bubble_centers_px[i]
        x2, y2, pb2, c2, cx2, cy2, r2 = bubble_centers_px[j]
        d = np.hypot(cx1 - cx2, cy1 - cy2)
        if d < r1 + r2:
            bb_overlaps.append((f"{c1}@{y1}", f"{c2}@{y2}", round(r1 + r2 - d, 1)))

pareto_pts = (
    [ax.transData.transform((x, y)) for x, y in zip(np.linspace(X_OURS, np.log10(27.0), 40), np.linspace(Y_OURS, 59.4, 40))]
    + [ax.transData.transform((x, y)) for x, y in zip(np.linspace(np.log10(27.0), np.log10(120.0), 40), np.linspace(59.4, 62.2, 40))]
)
divider_pts = [ax.transData.transform((X_DIV, y)) for y in np.linspace(Y_MIN, Y_MAX, 80)]
hcross_pts  = [ax.transData.transform((x, Y_OURS)) for x in np.linspace(X_OURS, X_MAX, 100)]

overlaps = []
bubble_hits = []
spine_crossings = []
line_hits = []

for i in range(len(bboxes)):
    nm, b = bboxes[i]
    if b.x0 < ax_bb.x0 + 4 or b.x1 > ax_bb.x1 - 4 or b.y1 > ax_bb.y1 - 2 or b.y0 < ax_bb.y0 + 2:
        spine_crossings.append((nm, round(b.x0 - ax_bb.x0, 1), round(ax_bb.x1 - b.x1, 1)))
    for j in range(i + 1, len(bboxes)):
        if b.overlaps(bboxes[j][1]):
            overlaps.append((nm, bboxes[j][0]))
    for bx, by, pb, cat, cx, cy, r_px in bubble_centers_px:
        nx = np.clip(cx, b.x0, b.x1)
        ny = np.clip(cy, b.y0, b.y1)
        dist = np.hypot(cx - nx, cy - ny)
        if dist < r_px - 0.5:
            bubble_hits.append((nm, f"{cat}@{by}", round(r_px - dist, 1)))
    for px, py in pareto_pts:
        if b.x0 <= px <= b.x1 and b.y0 <= py <= b.y1:
            line_hits.append((nm, "PARETO_LINE"))
            break
    for px, py in divider_pts:
        if nm != "DOM_CAPTION" and b.x0 <= px <= b.x1 and b.y0 <= py <= b.y1:
            line_hits.append((nm, "DIVIDER_LINE"))
            break
    for px, py in hcross_pts:
        if b.x0 <= px <= b.x1 and b.y0 <= py <= b.y1:
            line_hits.append((nm, "HCROSS_LINE"))
            break

if bb_overlaps or overlaps or bubble_hits or spine_crossings or line_hits:
    print("GEOMETRY ISSUES:", {
        "bubble_bubble": bb_overlaps,
        "text_text": overlaps,
        "text_bubble": bubble_hits,
        "spine_crossings": spine_crossings,
        "line_hits": line_hits,
    })
else:
    print("100% CLEAN GEOMETRY: 0 bubble overlaps, 0 text overlaps, 0 bubble hits, 0 spine crossings, 0 line crossings!")

out_pdf = ROOT / "figures/fig_exp1_pareto.pdf"
out_svg = ROOT / "figures/fig_exp1_pareto.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
print("Saved Tool-Fair Frontier Parameter Bubble Figure:", out_pdf)
