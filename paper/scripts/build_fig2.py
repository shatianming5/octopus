#!/usr/bin/env python3
"""
Publication-grade Figure 2 (Server-Resident Cursor & Zero-Copy Cross-Branch KV Mounting)
Completely redesigned so (a), (b), and (c) have three distinct engineering visual forms:
- Panel (a): Radix Tree Graph mapped onto a Contiguous Physical GPU Page Frame Bank (with ref-count badges)
- Panel (b): 32-Layer Hybrid Architecture Cross-Section (3x Gated DeltaNet + 1x Full-Attention stack)
             with dual vertical state buses (Blocked GDN Highway vs. Mounted Attention Port)
- Panel (c): 1D RoPE Coordinate Number Lines (0..260) with Causal Attention Arcs & Monotonic Jump Arc
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

fig, ax = plt.subplots(figsize=(11.4, 4.75), dpi=300)
ax.set_xlim(0, 114)
ax.set_ylim(0, 47.5)
ax.axis("off")

def rbox(x, y, w, h, fc="#FFFFFF", ec="#CBD5E1", lw=1.0, r=0.7, z=2, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p

def arrow(x1, y1, x2, y2, col="#475569", lw=1.2, style="-|>", rad=0.0, ls="-", z=5, ms=9.5):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        color=col, linewidth=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z)
    ax.add_patch(a)
    return a

def bezier(p0, p1, p2, p3, col="#0F766E", lw=1.6, ls="-", z=4):
    verts = [p0, p1, p2, p3]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor="none", edgecolor=col, lw=lw, linestyle=ls, zorder=z))

# ==============================================================================
# PANEL (a): Radix Tree Graph -> Physical GPU Page Frame Bank (Zero-Copy Lending)
# ==============================================================================
rbox(0.8, 3.6, 35.5, 42.6, fc="#FAFAFA", ec="#CBD5E1", lw=1.15, r=1.0, z=1)
ax.text(18.55, 44.4, "Radix Tree & Physical KV Page Bank", ha="center", va="center",
        fontsize=9.6, fontweight="bold", color="#1E3A8A")

# --- Top Half of (a): Logical Radix Cursor Tree ---
rbox(2.0, 24.4, 33.1, 18.4, fc="#FFFFFF", ec="#CBD5E1", lw=0.9, r=0.6, z=2)
ax.text(3.2, 41.4, "Logical Radix Cursor Tree (Virtual Paths)", ha="left", va="center",
        fontsize=7.7, fontweight="bold", color="#334155")

# Root node (#0 Shared Trunk)
rbox(12.8, 36.8, 11.5, 3.2, fc="#E2E8F0", ec="#475569", lw=1.0, r=1.4, z=4)
ax.text(18.55, 38.4, "Root Trunk $c_0$ $[0..79]$", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#1E293B", zorder=5)

# Left branch: Cursor c_A (Sibling Source)
bezier((16.0, 36.8), (14.0, 34.5), (10.0, 34.5), (8.5, 32.4), col="#D97706", lw=1.8)
rbox(2.8, 29.0, 11.8, 3.4, fc="#FEF3C7", ec="#D97706", lw=1.05, r=0.8, z=4)
ax.text(8.7, 30.7, "Span $A$ $[s_A, e_A]$\nCursor $c_A$ (Source)", ha="center", va="center",
        fontsize=6.9, fontweight="bold", color="#92400E", zorder=5)

# Right branch: Cursor c_B' (Reader with Negative Placeholder -M -> Delta_B)
bezier((21.0, 36.8), (23.0, 34.5), (26.5, 34.5), (28.0, 33.4), col="#2563EB", lw=1.8)
rbox(21.2, 30.2, 12.8, 3.2, fc="#FFFBEB", ec="#D97706", lw=1.0, r=0.8, ls="--", z=4)
ax.text(27.6, 31.8, "Mount Slot $-M_A$ (Virt.)", ha="center", va="center",
        fontsize=6.9, fontweight="bold", color="#B45309", zorder=5)

arrow(27.6, 30.1, 27.6, 28.3, col="#2563EB", lw=1.4, ms=8, z=5)
rbox(21.2, 25.1, 12.8, 3.1, fc="#DBEAFE", ec="#2563EB", lw=1.0, r=0.8, z=4)
ax.text(27.6, 26.65, "Cursor $c_B'$: Delta $\\Delta_B$", ha="center", va="center",
        fontsize=6.9, fontweight="bold", color="#1E40AF", zorder=5)

# --- Bottom Half of (a): Contiguous Physical GPU Page Frame Bank ---
rbox(2.0, 8.6, 33.1, 14.2, fc="#0F172A", ec="#334155", lw=1.05, r=0.6, z=2)
ax.text(3.2, 21.3, "Physical H200 KV Page Frame Array ($1.94\\mathrm{M}$ Tok Pool)",
        ha="left", va="center", fontsize=7.5, fontweight="bold", color="#F8FAFC", zorder=3)

page_frames = [
    ("P0", "#CBD5E1", "#475569", "Trunk", "ref=2", "#1E293B"),
    ("P1", "#CBD5E1", "#475569", "Trunk", "ref=2", "#1E293B"),
    ("P2", "#334155", "#64748B", "Free",  "ref=0", "#94A3B8"),
    ("P3", "#FCD34D", "#D97706", "Span A","1→2!",  "#78350F"),
    ("P4", "#FCD34D", "#D97706", "Span A","1→2!",  "#78350F"),
    ("P5", "#334155", "#64748B", "Free",  "ref=0", "#94A3B8"),
    ("P6", "#93C5FD", "#2563EB", "Delta B","ref=1","#1E3A8A"),
]
for idx, (pid, pfc, pec, plbl, pref, ptc) in enumerate(page_frames):
    px = 3.0 + idx * 4.45
    ax.add_patch(Rectangle((px, 12.2), 4.05, 7.4, facecolor=pfc, edgecolor=pec, lw=0.9, zorder=4))
    ax.text(px + 2.02, 18.2, pid, ha="center", va="center", fontsize=6.8, fontweight="bold", color=ptc, zorder=5)
    ax.text(px + 2.02, 15.8, plbl, ha="center", va="center", fontsize=6.2, fontweight="bold", color=ptc, zorder=5)
    # Ref-count pill at bottom of page frame
    pill_fc = "#FEF08A" if "1→2" in pref else ("#E2E8F0" if "ref=2" in pref else "#1E293B")
    pill_tc = "#991B1B" if "1→2" in pref else ("#1E293B" if "ref=2" in pref else "#CBD5E1")
    ax.add_patch(Rectangle((px + 0.25, 12.55), 3.55, 1.8, facecolor=pill_fc, edgecolor="none", zorder=5))
    ax.text(px + 2.02, 13.45, pref, ha="center", va="center", fontsize=6.0, fontweight="bold", color=pill_tc, zorder=6)

ax.text(18.55, 10.2, "$0$ New Pages Allocated on Mount $\\cdot$ Freed Only When $\\mathrm{ref} \\to 0$",
        ha="center", va="center", fontsize=7.0, fontweight="bold", color="#5EEAD4", zorder=4)

# Pointers from Virtual Radix Nodes to Physical Page Frames (Both c_A and -M_A point to P3/P4!)
arrow(8.7, 28.9, 17.5, 19.8, col="#F59E0B", lw=1.4, rad=0.14, ms=9, z=7)
arrow(21.1, 31.6, 20.5, 19.8, col="#F59E0B", lw=1.5, rad=0.18, ls="--", ms=9, z=7)
arrow(31.8, 25.0, 31.6, 19.8, col="#60A5FA", lw=1.3, rad=-0.06, ms=8.5, z=7)

rbox(2.0, 4.6, 33.1, 3.1, fc="#ECFDF5", ec="#0F766E", lw=0.9, r=0.45, z=3)
ax.text(18.55, 6.15, "Invariant: $N_{\\mathrm{avail}} + N_{\\mathrm{evict}} + N_{\\mathrm{prot}} = N_{\\mathrm{pool}}$ ($0$ Leaks)",
        ha="center", va="center", fontsize=7.3, fontweight="bold", color="#065F46")

ax.text(18.55, 1.5, "(a) Zero-Copy Radix Page Lending", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#111827")

# ==============================================================================
# PANEL (b): 32-Layer Hybrid Cross-Section (3x GDN Private vs. 1x Full-Attn Mount)
# ==============================================================================
rbox(37.8, 3.6, 37.0, 42.6, fc="#FAFAFA", ec="#CBD5E1", lw=1.15, r=1.0, z=1)
ax.text(56.3, 44.4, "Hybrid Layer-Selective Mount (K2)", ha="center", va="center",
        fontsize=9.6, fontweight="bold", color="#1E3A8A")

# Top header badges for the two vertical buses:
# Left Bus: Sibling A's Mounted KV Pages
rbox(39.2, 37.8, 14.5, 4.8, fc="#FEF3C7", ec="#D97706", lw=1.0, r=0.5, z=3)
ax.text(46.45, 40.9, "Sibling $A$ KV Bus", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#92400E")
ax.text(46.45, 38.9, "Physical Pages $\\mathbf{K}_A, \\mathbf{V}_A$", ha="center", va="center", fontsize=6.8, color="#78350F")

# Right Bus: Reader B's Private Recurrent Stream
rbox(58.5, 37.8, 15.0, 4.8, fc="#DBEAFE", ec="#2563EB", lw=1.0, r=0.5, z=3)
ax.text(66.0, 40.9, "Reader $B$ State Bus", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#1E40AF")
ax.text(66.0, 38.9, "Private $\\mathbf{h}_B^{\\mathrm{GDN}}$ & Tokens $\\Delta_B$", ha="center", va="center", fontsize=6.8, color="#1E3A8A")

# Draw the repeating Hybrid Block Stack in the center (x=45.5 to 67.5, y=15.0 to 35.8)
rbox(44.8, 14.2, 23.4, 22.2, fc="#FFFFFF", ec="#64748B", lw=1.05, r=0.6, z=2)
ax.text(56.5, 35.1, "Hybrid Block ($\\times 8$ in Qwen3.5: $24$ GDN + $8$ Attn)",
        ha="center", va="center", fontsize=7.1, fontweight="bold", color="#334155", zorder=4)

# Draw 3 Gated DeltaNet Recurrent Slices (y=31.2, 27.6, 24.0)
gdn_ys = [30.8, 27.2, 23.6]
for g_i, gy in enumerate(gdn_ys):
    rbox(46.2, gy, 20.6, 2.9, fc="#EFF6FF", ec="#3B82F6", lw=0.85, r=0.35, z=4)
    ax.text(56.5, gy + 1.45, f"Gated DeltaNet Layer $4k+{g_i+1}$: $\\mathbf{{h}}_B \\leftarrow f(\\mathbf{{h}}_B, \\Delta_B)$",
            ha="center", va="center", fontsize=6.6, fontweight="bold", color="#1E40AF", zorder=5)
    # Blocked tap from Left Bus (x=41.5) trying to enter GDN layer:
    ax.plot([41.5, 44.2], [gy + 1.45, gy + 1.45], color="#DC2626", lw=1.0, ls=":", zorder=4)
    c_x = Circle((44.8, gy + 1.45), 0.75, facecolor="#FEE2E2", edgecolor="#DC2626", lw=0.9, zorder=6)
    ax.add_patch(c_x)
    ax.text(44.8, gy + 1.45, "$\\times$", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#DC2626", zorder=7)
    # Right Bus feeds private h_B into every GDN layer
    arrow(70.8, gy + 1.45, 67.0, gy + 1.45, col="#2563EB", lw=1.1, ms=7.5, z=5)

# Draw 1 Full-Attention Layer Slice at the bottom of the block (y=15.6)
rbox(46.2, 15.4, 20.6, 6.8, fc="#FEF3C7", ec="#D97706", lw=1.15, r=0.45, z=4)
ax.text(56.5, 20.8, "Full-Attention Layer $4k+4$ ($\\ell \\in \\mathcal{L}_{\\mathrm{mount}}$)",
        ha="center", va="center", fontsize=7.0, fontweight="bold", color="#92400E", zorder=5)

# Inside the Full-Attention Layer, show the concatenated KV tape: [K_trunk || K_A (0-Copy) || K_B]
tape_items = [
    (47.2, 5.4, "#E2E8F0", "#475569", "$\\mathbf{K}_{\\mathrm{trunk}}$", "#1E293B"),
    (53.0, 6.8, "#FDE047", "#B45309", "$\\mathbf{K}_A^{(\\ell)}$ ($0$-Copy)", "#78350F"),
    (60.2, 5.6, "#93C5FD", "#1D4ED8", "$\\mathbf{K}_B^{(\\ell)}$", "#1E3A8A"),
]
for tx, tw, tfc, tec, ttxt, ttc in tape_items:
    ax.add_patch(Rectangle((tx, 16.2), tw, 3.2, facecolor=tfc, edgecolor=tec, lw=0.8, zorder=5))
    ax.text(tx + tw/2, 17.8, ttxt, ha="center", va="center", fontsize=6.3, fontweight="bold", color=ttc, zorder=6)

# Vertical Left Bus line (x=41.5) dropping from Sibling A (37.8) straight into the Full-Attention port (18.8)!
ax.plot([41.5, 41.5], [37.8, 18.8], color="#D97706", lw=2.0, zorder=4)
arrow(41.5, 18.8, 46.0, 18.8, col="#D97706", lw=1.8, ms=9.5, z=6)
ax.text(40.7, 28.5, "Bypass\n24 GDN\nLayers", ha="right", va="center", fontsize=6.4, fontweight="bold", color="#B91C1C", rotation=90)

# Vertical Right Bus line (x=70.8) dropping from Reader B (37.8) through all layers
ax.plot([70.8, 70.8], [37.8, 11.2], color="#2563EB", lw=2.0, zorder=4)
arrow(70.8, 18.8, 67.0, 18.8, col="#2563EB", lw=1.3, ms=8, z=5)

# Output synthesis bar below stack
arrow(56.5, 14.1, 56.5, 12.2, col="#0F766E", lw=1.4, ms=9, z=5)
rbox(39.2, 8.6, 34.3, 3.4, fc="#F0FDFA", ec="#0F766E", lw=0.95, r=0.45, z=3)
ax.text(56.35, 10.3, "Associative Lookup in $\\mathcal{L}_{\\mathrm{mount}}$ ($97.3\\%$ Rank) + Private $\\mathbf{h}_B$ Control Flow",
        ha="center", va="center", fontsize=7.0, fontweight="bold", color="#065F46")

rbox(39.2, 4.6, 34.3, 3.1, fc="#ECFDF5", ec="#0F766E", lw=0.9, r=0.45, z=3)
ax.text(56.35, 6.15, "Task Hijacking Slashed $21\\times$: $57.9\\%$ (Text) / $55.2\\%$ (Replay) $\\to \\mathbf{2.7\\%}$",
        ha="center", va="center", fontsize=7.1, fontweight="bold", color="#065F46")

ax.text(56.3, 1.5, "(b) Hybrid Layer-Selective Mounting", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#111827")

# ==============================================================================
# PANEL (c): 1D RoPE Coordinate Number Lines & Causal Attention Arcs (e_A > p_B)
# ==============================================================================
rbox(76.3, 3.6, 36.9, 42.6, fc="#FAFAFA", ec="#CBD5E1", lw=1.15, r=1.0, z=1)
ax.text(94.75, 44.4, "1D RoPE Coordinate Axis ($e_A > p_B$)", ha="center", va="center",
        fontsize=9.6, fontweight="bold", color="#1E3A8A")

# Helper to draw a 1D coordinate number line with ticks at 0, 80, 120, 200, 260
def draw_axis(y_axis, col="#475569"):
    ax.plot([78.2, 111.2], [y_axis, y_axis], color=col, lw=1.2, zorder=3)
    arrow(110.8, y_axis, 111.8, y_axis, col=col, lw=1.2, ms=7, z=3)
    ticks = [(78.8, "$0$"), (86.8, "$s_A=80$"), (93.2, "$p_B=120$"), (103.0, "$e_A=200$"), (109.8, "$260$")]
    for tx, tlbl in ticks:
        ax.plot([tx, tx], [y_axis - 0.45, y_axis + 0.45], color=col, lw=0.9, zorder=3)
        ax.text(tx, y_axis - 1.25, tlbl, ha="center", va="center", fontsize=6.0, color="#334155")

# ------------------------------------------------------------------------------
# 1. TOP RIGHT: K1 Unshifted Positions (Causal Inversion Collapse!)
# ------------------------------------------------------------------------------
rbox(77.5, 30.6, 34.5, 12.2, fc="#FFFFFF", ec="#FECACA", lw=0.95, r=0.6, z=2)
ax.text(78.6, 41.4, "1. Unshifted Positions (K1): Causal Distance Inversion when $e_A > p_B$",
        ha="left", va="center", fontsize=7.4, fontweight="bold", color="#991B1B")

draw_axis(33.4, col="#64748B")
# Prefix c_B [0..120]
ax.add_patch(Rectangle((78.8, 33.7), 14.4, 1.7, facecolor="#DBEAFE", edgecolor="#2563EB", lw=0.8, zorder=4))
ax.text(86.0, 34.55, "Reader Prefix $c_B$ $[0..120]$", ha="center", va="center", fontsize=6.1, fontweight="bold", color="#1E40AF", zorder=5)
# Mounted Span A [80..200] extending BEYOND p_B=120!
ax.add_patch(Rectangle((86.8, 35.7), 16.2, 1.7, facecolor="#FEF3C7", edgecolor="#D97706", lw=0.85, zorder=4))
ax.text(94.9, 36.55, "Mounted Span $A$ $[80..200]$", ha="center", va="center", fontsize=6.1, fontweight="bold", color="#92400E", zorder=5)
# Query q_B stuck at p_B = 120!
ax.add_patch(Circle((93.2, 34.55), 0.8, facecolor="#DC2626", edgecolor="#FFFFFF", lw=0.8, zorder=6))
ax.text(93.2, 34.55, "$q_B$", ha="center", va="center", fontsize=5.9, fontweight="bold", color="#FFFFFF", zorder=7)

# Red Inverted Attention Arc pointing FORWARD into the "Future" (120 -> 195)!
arrow(93.5, 35.3, 102.0, 37.5, col="#DC2626", lw=1.4, rad=-0.38, ms=8.5, z=7)
ax.text(98.5, 39.4, "Keys in “Future” ($\\mathrm{pos} > p_B$): $-16.5\\%$ EM Drop!",
        ha="center", va="center", fontsize=6.4, fontweight="bold", color="#B91C1C", zorder=8)

# ------------------------------------------------------------------------------
# 2. MIDDLE RIGHT: K3 RoPE Key Relocation Copy (O(N^2) Page Allocation)
# ------------------------------------------------------------------------------
rbox(77.5, 17.6, 34.5, 12.0, fc="#FFFFFF", ec="#FDE68A", lw=0.95, r=0.6, z=2)
ax.text(78.6, 28.2, "2. RoPE Relocation Copy (K3): Mutates Keys & Allocates Private Pages",
        ha="left", va="center", fontsize=7.5, fontweight="bold", color="#92400E")

draw_axis(21.4, col="#64748B")
ax.add_patch(Rectangle((78.8, 21.7), 14.4, 1.8, facecolor="#DBEAFE", edgecolor="#2563EB", lw=0.8, zorder=4))
ax.text(86.0, 22.6, "Prefix $c_B$ $[0..120]$", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#1E40AF", zorder=5)
# Copied & Rotated Span A placed at [121..241]
ax.add_patch(Rectangle((93.2, 21.7), 11.5, 1.8, facecolor="#FEE2E2", edgecolor="#DC2626", lw=0.9, ls="--", zorder=4))
ax.text(98.95, 22.6, "$\\mathbf{R}_{\\Delta\\theta}\\,\\mathrm{Copy}(\\mathbf{K}_A)$", ha="center", va="center",
        fontsize=6.2, fontweight="bold", color="#991B1B", zorder=5)
ax.add_patch(Rectangle((104.9, 21.7), 5.0, 1.8, facecolor="#93C5FD", edgecolor="#1D4ED8", lw=0.8, zorder=4))
ax.text(107.4, 22.6, "$\\Delta_B$", ha="center", va="center", fontsize=6.3, fontweight="bold", color="#1E3A8A", zorder=5)

ax.text(94.75, 25.4, "Breaks $0$-Copy! Allocates $\\mathcal{O}(N^2)$ Private KV Pages & Re-Rotates Keys",
        ha="center", va="center", fontsize=6.5, fontweight="bold", color="#B45309", zorder=6)

# ------------------------------------------------------------------------------
# 3. BOTTOM RIGHT: K2 Octopus Monotonic Continuation (Zero-Copy Leapfrog!)
# ------------------------------------------------------------------------------
rbox(77.5, 4.6, 34.5, 12.0, fc="#ECFDF5", ec="#0F766E", lw=1.15, r=0.6, z=2)
ax.text(78.6, 15.2, "3. Octopus Monotonic Continuation (K2, Ours): $0$-Copy Coordinate Jump",
        ha="left", va="center", fontsize=7.5, fontweight="bold", color="#065F46")

draw_axis(8.4, col="#0F766E")
# Prefix c_B [0..120]
ax.add_patch(Rectangle((78.8, 8.7), 14.4, 1.8, facecolor="#DBEAFE", edgecolor="#2563EB", lw=0.8, zorder=4))
ax.text(86.0, 9.6, "Prefix $c_B$ $[0..120]$", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#1E40AF", zorder=5)
# Zero-Copy Span A untouched at [80..200]
ax.add_patch(Rectangle((86.8, 10.7), 16.2, 1.8, facecolor="#FEF3C7", edgecolor="#D97706", lw=0.9, zorder=4))
ax.text(94.9, 11.6, "$0$-Copy $\\mathbf{K}_A$ @ Native $[80..200]$", ha="center", va="center",
        fontsize=6.2, fontweight="bold", color="#92400E", zorder=5)

# Delta_B jumps monotonically to s(c_B') = e_A + 1 = 201!
ax.add_patch(Rectangle((103.2, 8.7), 6.8, 1.8, facecolor="#0F766E", edgecolor="#064E3B", lw=0.9, zorder=5))
ax.text(106.6, 9.6, "$\\Delta_B$ @ $201+$", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#FFFFFF", zorder=6)

# Green Monotonic Jump Arc from p_B=120 to s(c_B')=201 AND Causal Attention Arc back to K_A!
arrow(105.5, 10.6, 96.5, 12.5, col="#0F766E", lw=1.4, rad=0.25, ms=8.5, z=7)
ax.text(94.75, 13.5, "All Mounted Keys Strictly in Causal Past ($0.630$ Ahead EM, $0$-Copy)",
        ha="center", va="center", fontsize=6.4, fontweight="bold", color="#065F46", zorder=8)

ax.text(94.75, 1.5, "(c) Monotonic Position Coordinate Jump", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#111827")

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
out_pdf = "./figures/fig2_cursor_mount.pdf"
out_svg = "./figures/fig2_cursor_mount.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
print("Saved Figure 2:", out_pdf)
