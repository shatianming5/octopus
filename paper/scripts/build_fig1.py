#!/usr/bin/env python3
"""
Publication-grade Figure 1 (System Overview: Topological Comparison & Octopus 3-Layer Co-Design)
Completely redesigned to replace generic stacked boxes with:
- Left (a): Three distinct failure topologies:
    (1) Clogged Linear Pipe hitting a 32K brick wall + mid-context edit crack invalidating Radix Cache
    (2) Stateless Multi-Agent O(N^2) Fan duplicating shared prefixes across isolated sessions
    (3) Full-Layer KV Replay trajectory collision where Sibling A hijacks Reader B's trajectory path (57.9%)
- Right (b): Octopus Co-Designed Architecture:
    - Top: An organic multi-branch execution tree with circular cursor nodes (c_0 -> c_1 -> fork into 3 branches,
      curved rewind arc on Branch 1, zero-copy KV mount beam from Branch 1 to Branch 2, and synthesis join)
    - Middle: Vertical memory pointers dropping into the H200 Shared Radix KV Page Pool (ref-counted blocks)
      and Private Gated DeltaNet Recurrent State Bank (524 resident states, 0 new pages on mount)
    - Bottom: Packed 1D DAG-GRPO GPU sequence buffer with cross-branch backward gradient flow
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Polygon
from matplotlib.path import Path
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(11.4, 5.05), dpi=300)
ax.set_xlim(0, 114)
ax.set_ylim(0, 50.5)
ax.axis("off")

def rbox(x, y, w, h, fc="#FFFFFF", ec="#CBD5E1", lw=1.0, r=0.7, z=2, ls="-", alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha, zorder=z)
    ax.add_patch(p)
    return p

def arrow(x1, y1, x2, y2, col="#475569", lw=1.2, style="-|>", rad=0.0, ls="-", z=5, ms=9.5):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        color=col, linewidth=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z)
    ax.add_patch(a)
    return a

def node(x, y, rad, fc="#0F766E", ec="#064E3B", txt="", tc="#FFFFFF", fs=7.2, lw=1.1, z=6):
    c = Circle((x, y), rad, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(c)
    if txt:
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc, zorder=z+1)
    return c

def bezier_curve(p0, p1, p2, p3, col="#0F766E", lw=1.8, ls="-", z=4):
    verts = [p0, p1, p2, p3]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor="none", edgecolor=col, lw=lw, linestyle=ls, zorder=z)
    ax.add_patch(patch)

# ==============================================================================
# LEFT COLUMN (a): Prior Agentic Execution Paradigms (3 Distinct Visual Topologies)
# ==============================================================================
rbox(0.8, 3.6, 42.0, 45.6, fc="#FAFAFA", ec="#CBD5E1", lw=1.15, r=1.0, z=1)
ax.text(21.8, 47.4, "Prior Agentic Execution Paradigms", ha="center", va="center",
        fontsize=10.2, fontweight="bold", color="#1E293B")

# ------------------------------------------------------------------------------
# 1. TOP LEFT: Clogged Single-Thread Pipe & Mid-Context Edit Crack
# ------------------------------------------------------------------------------
rbox(2.0, 34.0, 39.6, 11.8, fc="#FFFFFF", ec="#FECACA", lw=0.95, r=0.65, z=2)
ax.text(3.2, 44.3, "1. Single-Thread Linear / In-Place Edit: Context Pollution & Cache Invalidation",
        ha="left", va="center", fontsize=8.0, fontweight="bold", color="#991B1B")

# Draw a physical pipe/tunnel getting clogged with red dead-end blocks
ax.plot([3.2, 30.8], [41.6, 41.6], color="#64748B", lw=1.2, zorder=3)
ax.plot([3.2, 30.8], [38.2, 38.2], color="#64748B", lw=1.2, zorder=3)

pipe_Segs = [
    (3.4, 3.8, "#E2E8F0", "#475569", "Prompt", "#1E293B"),
    (7.5, 4.8, "#FEE2E2", "#DC2626", "Dead End 1", "#991B1B"),
    (12.6, 5.2, "#FECACA", "#DC2626", "Edit / Fold", "#991B1B"),
    (18.1, 6.2, "#FCA5A5", "#B91C1C", "Dead End 2", "#7F1D1D"),
    (24.6, 5.8, "#EF4444", "#991B1B", "Noise Bloat", "#FFFFFF"),
]
for sx, sw, sfc, sec, stxt, stc in pipe_Segs:
    ax.add_patch(Rectangle((sx, 38.45), sw, 2.9, facecolor=sfc, edgecolor=sec, lw=0.75, zorder=4))
    ax.text(sx + sw/2, 39.9, stxt, ha="center", va="center", fontsize=6.6, fontweight="bold", color=stc, zorder=5)

# Zigzag crack at "Edit / Fold" showing broken prefix & recurrent state
crack_x = [15.2, 14.8, 15.5, 14.9, 15.3]
crack_y = [42.1, 40.7, 39.8, 38.9, 37.7]
ax.plot(crack_x, crack_y, color="#DC2626", lw=1.6, zorder=6)
ax.text(15.2, 36.0, "Mid-Context Edit Breaks Radix Cache\n& Corrupts Recurrent State ($42\\%$–$76\\%$ Re-Prefill)",
        ha="center", va="center", fontsize=6.7, fontweight="bold", color="#B91C1C")

# Red Brick Wall at x=31.0
for wy in np.arange(37.6, 42.2, 0.9):
    ax.add_patch(Rectangle((31.0, wy), 1.3, 0.8, facecolor="#DC2626", edgecolor="#FFFFFF", lw=0.4, zorder=5))
ax.text(36.8, 40.3, "$B_{\\mathrm{thread}}$ Wall!\nStalls Early\n($21\\%$ Recall)", ha="center", va="center",
        fontsize=7.0, fontweight="bold", color="#991B1B")

# ------------------------------------------------------------------------------
# 2. MIDDLE LEFT: Stateless Multi-Agent Chat (O(N^2) Duplicated Fan)
# ------------------------------------------------------------------------------
rbox(2.0, 19.2, 39.6, 13.8, fc="#FFFFFF", ec="#FDE68A", lw=0.95, r=0.65, z=2)
ax.text(3.2, 31.5, "2. Stateless Multi-Agent Chat: $\\mathcal{O}(L)$ Re-Tokenization & $\\mathcal{O}(N^2)$ KV Bloat",
        ha="left", va="center", fontsize=8.0, fontweight="bold", color="#92400E")

# Client box on left sending 3 redundant fans of full text strings
rbox(3.2, 21.8, 7.2, 7.6, fc="#FEF3C7", ec="#D97706", lw=0.9, r=0.45, z=3)
ax.text(6.8, 26.8, "App String\nOrchestrator", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#92400E")
ax.text(6.8, 23.4, "Serializes State\nto Text Every Turn", ha="center", va="center", fontsize=6.1, color="#78350F")

# Three isolated GPUKV tracks duplicating the exact same trunk + cross-transcripts
track_ys = [27.8, 24.6, 21.4]
for idx, ty in enumerate(track_ys):
    arrow(10.6, 25.6, 16.8, ty + 0.85, col="#D97706", lw=1.0, ls="--", ms=8)
    # Duplicated trunk cells + transcript copies
    for c_i, c_col in enumerate(["#CBD5E1", "#CBD5E1", "#CBD5E1", "#FCD34D", "#F59E0B", "#EF4444"]):
        ax.add_patch(Rectangle((17.2 + c_i * 2.15, ty), 1.95, 1.7, facecolor=c_col, edgecolor="#475569", lw=0.5, zorder=4))
    ax.text(30.6, ty + 0.85, f"Sess.{idx+1} Copy", ha="left", va="center", fontsize=6.5, fontweight="bold", color="#92400E")

ax.text(13.6, 29.5, "$\\mathcal{O}(L)$ Wire Text", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#B45309")
rbox(35.2, 21.2, 5.6, 8.4, fc="#FEE2E2", ec="#DC2626", lw=0.9, r=0.4, z=3)
ax.text(38.0, 25.4, "$\\mathcal{O}(N^2)$\nKV Pool\nBloat &\nQueue\nStall", ha="center", va="center",
        fontsize=6.8, fontweight="bold", color="#991B1B")

# ------------------------------------------------------------------------------
# 3. BOTTOM LEFT: Full-Layer KV Replay / Text Injection (Trajectory Hijacking!)
# ------------------------------------------------------------------------------
rbox(2.0, 4.8, 39.6, 13.4, fc="#FFFFFF", ec="#FECDD3", lw=0.95, r=0.65, z=2)
ax.text(3.2, 16.7, "3. Full-Layer KV Replay / Text Injection: Catastrophic Control-Flow Hijacking",
        ha="left", va="center", fontsize=8.0, fontweight="bold", color="#9F1239")

# Sibling A path (top amber trajectory)
ax.plot([4.0, 34.0], [13.6, 13.6], color="#D97706", lw=2.2, zorder=3)
node(5.5, 13.6, 1.0, fc="#F59E0B", ec="#92400E", txt="$A_1$", fs=6.8)
node(14.5, 13.6, 1.0, fc="#F59E0B", ec="#92400E", txt="$A_2$", fs=6.8)
node(26.5, 13.6, 1.0, fc="#F59E0B", ec="#92400E", txt="$A_3$", fs=6.8)
ax.text(35.0, 13.6, "Task $A$ Goal", ha="left", va="center", fontsize=6.9, fontweight="bold", color="#92400E")

# Reader B intended path (bottom blue dashed line)
ax.plot([4.0, 14.5], [7.8, 7.8], color="#2563EB", lw=2.2, zorder=3)
ax.plot([14.5, 34.0], [7.8, 7.8], color="#93C5FD", lw=1.4, ls="--", zorder=2)
node(5.5, 7.8, 1.0, fc="#2563EB", ec="#1E3A8A", txt="$B_1$", fs=6.8)
node(14.5, 7.8, 1.1, fc="#E11D48", ec="#881337", txt="$B_2$", fs=6.8)
ax.text(35.0, 7.8, "Task $B$ Abandoned!", ha="left", va="center", fontsize=6.8, fontweight="bold", color="#64748B")

# Full-layer injection arrow from A_2 to B_2 overwriting recurrent state h_B
arrow(14.5, 12.4, 14.5, 9.1, col="#E11D48", lw=1.6, ms=10)
ax.text(13.8, 10.7, "All Layers\n($\\mathbf{h}_A^{\\mathrm{GDN}} \\to \\mathbf{h}_B$)", ha="right", va="center",
        fontsize=6.5, fontweight="bold", color="#BE123C")

# Hijacked trajectory swerving from B_2 up onto A's path!
bezier_curve((15.6, 7.8), (19.5, 7.8), (20.5, 12.6), (25.2, 13.2), col="#E11D48", lw=2.3, ls="-", z=5)
arrow(24.0, 13.0, 25.3, 13.3, col="#E11D48", lw=2.0, ms=10, z=6)
rbox(18.5, 8.7, 15.0, 3.1, fc="#FFE4E6", ec="#E11D48", lw=0.9, r=0.4, z=6)
ax.text(26.0, 10.25, "Reader $B$ Hijacked into Task $A$!\n($55.2\\%$–$57.9\\%$ Hijack Rate)",
        ha="center", va="center", fontsize=6.7, fontweight="bold", color="#9F1239", zorder=7)

ax.text(21.8, 1.5, "(a) Prior Paradigms: Linear Clogging, $\\mathcal{O}(N^2)$ Bloat & Control-Flow Hijacking",
        ha="center", va="center", fontsize=9.0, fontweight="bold", color="#111827")

# ==============================================================================
# RIGHT COLUMN (b): Octopus Co-Designed Branching Tree, Radix Pool & DAG-GRPO
# ==============================================================================
rbox(44.2, 3.6, 69.0, 45.6, fc="#F8FAFC", ec="#0F766E", lw=1.25, r=1.0, z=1)
ax.text(78.7, 47.4, "Octopus: Co-Designed Branching Execution Tree, Radix KV Pool & DAG-GRPO",
        ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0F766E")

# ------------------------------------------------------------------------------
# LAYER 1 (TOP RIGHT): Visual Branching Execution Tree with Cursors, Rewind & Read
# ------------------------------------------------------------------------------
rbox(45.5, 28.0, 66.4, 17.8, fc="#FFFFFF", ec="#99F6E4", lw=1.0, r=0.7, z=2)
ax.text(46.8, 44.5, "Layer 1 (Execution): Cursor-Addressed Branching Tree $\\mathcal{G} = (\\mathcal{V}, \\mathcal{E}_{\\mathrm{tree}}, \\mathcal{E}_{\\mathrm{mount}})$ ($32\\mathrm{K}$/Thread Cap)",
        ha="left", va="center", fontsize=8.0, fontweight="bold", color="#0F766E")

# Root trunk c_0 -> c_1
ax.plot([48.5, 56.5], [34.8, 34.8], color="#0F766E", lw=2.6, zorder=4)
node(48.5, 34.8, 1.3, fc="#0F766E", ec="#064E3B", txt="$c_0$", fs=7.6)
node(56.5, 34.8, 1.3, fc="#0F766E", ec="#064E3B", txt="$c_{\\mathrm{fork}}$", fs=6.8)
ax.text(52.5, 36.4, "$\\mathcal{O}(\\Delta)$ Step", ha="center", va="center", fontsize=6.6, fontweight="bold", color="#0F766E")

# Fork curves from c_fork (56.5, 34.8) into 3 parallel branches:
# Branch A (Top, y=39.8): c_A1 -> dead-end X -> rewind arc back to c_A1 -> c_A2 (Gold Evidence!)
bezier_curve((57.8, 35.2), (61.0, 35.5), (61.5, 39.8), (65.0, 39.8), col="#D97706", lw=2.0)
ax.plot([65.0, 73.5], [39.8, 39.8], color="#94A3B8", lw=1.5, ls="--", zorder=3)
node(65.0, 39.8, 1.1, fc="#F59E0B", ec="#92400E", txt="$c_{A1}$", fs=6.7)
node(73.5, 39.8, 1.0, fc="#FEE2E2", ec="#DC2626", txt="$\\times$", tc="#DC2626", fs=8.2)
ax.text(73.5, 41.4, "Dead End", ha="center", va="center", fontsize=6.2, color="#B91C1C")

# Rewind curved arc from X (73.5, 39.8) back to c_A1 (65.0, 39.8) and continuing to c_A2 (83.5, 39.8)
arrow(72.6, 40.6, 65.9, 40.6, col="#DC2626", lw=1.25, rad=0.38, ms=8.0, z=7)
ax.text(69.2, 42.5, "rewind($c_{A1}$)", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#B91C1C")
bezier_curve((66.1, 39.6), (72.0, 38.0), (77.5, 38.0), (83.5, 39.8), col="#D97706", lw=2.0)
node(83.5, 39.8, 1.2, fc="#F59E0B", ec="#92400E", txt="$c_{A2}^\\star$", fs=6.9)
ax.text(83.5, 41.7, "Span $A$ $[s_A, e_A]$", ha="center", va="center", fontsize=6.6, fontweight="bold", color="#92400E")

# Branch B (Middle, y=34.8): c_B1 -> mounts Span A in 0-copy -> c_B2 -> joins Root!
ax.plot([57.8, 92.0], [34.8, 34.8], color="#2563EB", lw=2.2, zorder=4)
node(68.5, 34.8, 1.1, fc="#2563EB", ec="#1E3A8A", txt="$c_{B1}$", fs=6.7)
node(83.5, 34.8, 1.2, fc="#1D4ED8", ec="#1E3A8A", txt="$c_{B2}$", fs=6.7)
node(92.5, 34.8, 1.1, fc="#0F766E", ec="#064E3B", txt="done", fs=6.2)

# Zero-Copy Cross-Branch KV Mount beam from c_A2 (83.5, 39.8) to c_B2 (83.5, 34.8)
arrow(83.5, 38.5, 83.5, 36.1, col="#D97706", lw=1.9, ms=9.5, z=7)
rbox(74.2, 36.2, 8.5, 1.9, fc="#FEF3C7", ec="#D97706", lw=0.8, r=0.3, z=6)
ax.text(78.45, 37.15, "read($A$) $0$-Copy", ha="center", va="center", fontsize=6.3, fontweight="bold", color="#92400E", zorder=7)

# Branch C (Bottom, y=29.8): Parallel hypothesis branch exploring & joining
bezier_curve((57.8, 34.4), (61.0, 34.0), (61.5, 29.8), (66.5, 29.8), col="#0F766E", lw=1.9)
ax.plot([66.5, 81.0], [29.8, 29.8], color="#0F766E", lw=1.9, zorder=4)
node(66.5, 29.8, 1.05, fc="#0D9488", ec="#064E3B", txt="$c_{C1}$", fs=6.7)
node(81.0, 29.8, 1.05, fc="#0D9488", ec="#064E3B", txt="$c_{C2}$", fs=6.7)
bezier_curve((82.1, 29.8), (87.0, 29.8), (88.5, 34.4), (91.2, 34.6), col="#0F766E", lw=1.5, ls="--")
bezier_curve((84.8, 39.8), (88.0, 39.8), (89.0, 35.2), (91.2, 35.0), col="#D97706", lw=1.5, ls="--")

# Right callout box inside Layer 1: Emergent Tree Compression
rbox(95.2, 29.2, 15.4, 14.6, fc="#F0FDFA", ec="#0F766E", lw=0.95, r=0.5, z=3)
ax.text(102.9, 41.8, "Tree Scaling\non H200:", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#065F46")
ax.text(102.9, 38.2, "• $143.5\\mathrm{K}$ Tree Tok\n  ($4.4\\times$ Thread Cap)", ha="center", va="center", fontsize=6.7, color="#0F766E")
ax.text(102.9, 34.6, "• $1.86\\times$ Crit. Path\n  Compression", ha="center", va="center", fontsize=6.7, color="#0F766E")
ax.text(102.9, 31.1, "• $15\\times$ Fewer Root\n  Rewinds ($98\\%$ Fork)", ha="center", va="center", fontsize=6.7, fontweight="bold", color="#065F46")

# ------------------------------------------------------------------------------
# LAYER 2 (MIDDLE RIGHT): H200 Shared Radix KV Page Pool & Attention-Only Mount Gate
# ------------------------------------------------------------------------------
rbox(45.5, 14.8, 66.4, 12.0, fc="#FFFFFF", ec="#93C5FD", lw=1.0, r=0.7, z=2)
ax.text(46.8, 25.4, "Layer 2 (Memory): H200 Radix KV Page Pool ($89.59\\%$ Hit) & Layer-Selective Mount (K2)",
        ha="left", va="center", fontsize=8.2, fontweight="bold", color="#1E3A8A")

# Vertical dashed memory pointers dropping from c_fork, c_A2, c_B2 into Layer 2!
ax.plot([56.5, 56.5], [34.3, 22.8], color="#64748B", lw=1.0, ls=":", zorder=3)
ax.plot([83.5, 73.0], [39.9, 22.8], color="#D97706", lw=1.1, ls=":", zorder=3)

# Left of Layer 2: Physical Radix KV Page Pool (1.94M Tok, O(N))
rbox(46.8, 16.0, 33.5, 8.0, fc="#F8FAFC", ec="#64748B", lw=0.9, r=0.45, z=3)
ax.text(63.55, 22.7, "H200 Physical Radix KV Pool ($1.94\\mathrm{M}$ Tok, $14.56\\mathrm{B}$ Reused, $0$ Burst Timeouts)",
        ha="center", va="center", fontsize=7.0, fontweight="bold", color="#1E293B")

# Draw contiguous physical page slots
slots = [
    ("#CBD5E1", "#475569", "Trunk #0\nref=3"),
    ("#CBD5E1", "#475569", "Trunk #1\nref=3"),
    ("#FCD34D", "#B45309", "Span A #4\nref:1→2"),
    ("#FCD34D", "#B45309", "Span A #5\n0-Copy!"),
    ("#93C5FD", "#1D4ED8", "Branch B\nref=1"),
    ("#99F6E4", "#0F766E", "Branch C\nref=1"),
]
for s_i, (sfc, sec, stxt) in enumerate(slots):
    sx = 47.6 + s_i * 5.35
    ax.add_patch(Rectangle((sx, 16.8), 4.95, 4.4, facecolor=sfc, edgecolor=sec, lw=0.8, zorder=4))
    ax.text(sx + 2.47, 19.0, stxt, ha="center", va="center", fontsize=6.1, fontweight="bold", color="#1E293B", zorder=5)

# Right of Layer 2: Hybrid Layer-Selective Gate (24 GDN Private vs. 8 Attn Mounted)
rbox(81.8, 16.0, 28.8, 8.0, fc="#FFFBEB", ec="#D97706", lw=0.95, r=0.45, z=3)
ax.text(96.2, 22.7, "Attention-Only Mount (K2: $21\\times$ Less Hijack)",
        ha="center", va="center", fontsize=7.1, fontweight="bold", color="#92400E")
rbox(82.8, 19.3, 26.8, 2.2, fc="#EFF6FF", ec="#3B82F6", lw=0.75, r=0.3, z=4)
ax.text(96.2, 20.4, "24 GDN Recurrent Layers: Private $\\mathbf{h}_B$ ($524$ States/H200)",
        ha="center", va="center", fontsize=6.5, fontweight="bold", color="#1E40AF", zorder=5)
rbox(82.8, 16.6, 26.8, 2.2, fc="#FEF3C7", ec="#D97706", lw=0.75, r=0.3, z=4)
ax.text(96.2, 17.7, "8 Full-Attn Layers: Mount $\\mathbf{K}_A$ @ Monotonic $s(c_B')$ ($2.7\\%$ Hijack)",
        ha="center", va="center", fontsize=6.5, fontweight="bold", color="#92400E", zorder=5)

# ------------------------------------------------------------------------------
# LAYER 3 (BOTTOM RIGHT): Packed 1D DAG-GRPO Sequence & Cross-Branch Autograd
# ------------------------------------------------------------------------------
rbox(45.5, 4.8, 66.4, 8.8, fc="#FDF4FF", ec="#D8B4FE", lw=1.0, r=0.7, z=2)
ax.text(46.8, 12.3, "Layer 3 (Optimization): $4\\times\\mathrm{H200}$ Tree-and-Mount DAG-GRPO ($1.12\\mathrm{M}$ Valid Tok/Step, $\\mathrm{KL}=4.4\\times 10^{-4}$)",
        ha="left", va="center", fontsize=8.0, fontweight="bold", color="#581C87")

# Draw packed 1D GPU sequence buffer with backward gradient arc from Reader B to Source A
pack_blocks = [
    (47.0, 9.5, "#E2E8F0", "#475569", "Trunk $u_0$ ($1\\times$)", "#1E293B"),
    (57.0, 10.5, "#FEF3C7", "#D97706", "Source $u_A$ ($m=1$)", "#92400E"),
    (68.0, 10.5, "#F1F5F9", "#94A3B8", "Pre-Rw ($m=1$)", "#475569"),
    (79.0, 10.5, "#DCFCE7", "#16A34A", "Post-Rw ($m=1$)", "#166534"),
    (90.0, 11.5, "#DBEAFE", "#2563EB", "Reader $u_B$ ($m=1$)", "#1E40AF"),
]
for px, pw, pfc, pec, ptxt, ptc in pack_blocks:
    ax.add_patch(Rectangle((px, 5.6), pw, 2.9, facecolor=pfc, edgecolor=pec, lw=0.85, zorder=4))
    ax.text(px + pw/2, 7.05, ptxt, ha="center", va="center", fontsize=6.4, fontweight="bold", color=ptc, zorder=5)

# Reverse Autograd Arc from Reader u_B (95.0, 8.6) curving UPWARD back to Source u_A (62.5, 8.6)
arrow(95.0, 8.6, 62.5, 8.6, col="#E11D48", lw=1.5, rad=0.09, ms=9.0, z=7)
ax.text(78.8, 10.7, "Cross-Branch Autograd $\\partial \\mathcal{L}_B / \\partial \\mathbf{K}_A^{(\\ell)} \\neq 0$",
        ha="center", va="center", fontsize=6.6, fontweight="bold", color="#BE123C", zorder=8)

rbox(102.2, 5.6, 8.6, 4.8, fc="#ECFDF5", ec="#0F766E", lw=0.9, r=0.4, z=4)
ax.text(106.5, 8.0, "Pure Outcome\n$R = 2y - 1$\nLOO $\\hat{A}(\\tau_i)$", ha="center", va="center",
        fontsize=6.5, fontweight="bold", color="#065F46", zorder=5)

ax.text(78.7, 1.5, "(b) Octopus: Zero-Copy Cursor Tree, Attention-Only KV Mount & Packed DAG-GRPO",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#111827")

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
out_pdf = "./figures/fig1_overview.pdf"
out_svg = "./figures/fig1_overview.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
print("Saved Figure 1:", out_pdf)
