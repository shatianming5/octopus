#!/usr/bin/env python3
"""
Publication-grade Figure 3 (Tree-and-Mount DAG-GRPO Training Kernel)
Completely redesigned so (a), (b), and (c) have three distinct compiler/kernel visuals:
- Panel (a): 2D Branching Tree compiled into a 1D Contiguous GPU Packed Buffer with
             Forward GDN Boundary Hook Arcs (top) and Reverse Cross-Branch Autograd Arc (bottom)
- Panel (b): 2D Block-Sparse Tree & Cross-Branch Attention Mask Matrix (with Transitive Mounts)
- Panel (c): 2D Token-Segment Single-Ownership Grid (proving Sum_l m_{l,t} = 1 per reply) +
             Group Leave-One-Out Tree Outcome Credit Broadcast
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

fig, ax = plt.subplots(figsize=(11.4, 4.65), dpi=300)
ax.set_xlim(0, 114)
ax.set_ylim(0, 46.5)
ax.axis("off")

def rbox(x, y, w, h, fc="#FFFFFF", ec="#CBD5E1", lw=1.0, r=0.7, z=2, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p

def arrow(x1, y1, x2, y2, col="#475569", lw=1.2, style="-|>", rad=0.0, ls="-", z=5, ms=9.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        color=col, linewidth=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z)
    ax.add_patch(a)
    return a

def cnode(x, y, rad, fc="#0F766E", ec="#064E3B", txt="", tc="#FFFFFF", fs=6.8, lw=1.0, z=5):
    c = Circle((x, y), rad, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(c)
    if txt:
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc, zorder=z+1)
    return c

# ==============================================================================
# PANEL (a): Tree-to-1D Packed GPU Buffer Compilation & Cross-Branch Autograd
# ==============================================================================
rbox(0.8, 3.6, 36.2, 41.6, fc="#FAFAFA", ec="#CBD5E1", lw=1.15, r=1.0, z=1)
ax.text(18.9, 43.4, "Tree-to-1D Packed GPU DAG & Autograd", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#1E3A8A")

# --- Top Half of (a): 2D Episode Execution DAG G ---
rbox(2.0, 26.8, 33.8, 15.0, fc="#FFFFFF", ec="#CBD5E1", lw=0.9, r=0.6, z=2)
ax.text(3.1, 40.5, "1. Branched Episode DAG $\\mathcal{G} = (\\mathcal{V}, \\mathcal{E}_{\\mathrm{tree}}, \\mathcal{E}_{\\mathrm{mount}})$",
        ha="left", va="center", fontsize=7.3, fontweight="bold", color="#334155")

# Trunk u_0 at left (5.5, 33.2)
cnode(5.5, 33.2, 1.4, fc="#E2E8F0", ec="#475569", txt="$u_0$", tc="#1E293B", fs=7.6)
ax.text(5.5, 30.7, "Trunk", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#475569")

# Branch 1 (Top): Source u_A (15.5, 37.2)
arrow(6.9, 33.9, 14.0, 36.9, col="#D97706", lw=1.4, ms=8)
cnode(15.5, 37.2, 1.35, fc="#FEF3C7", ec="#D97706", txt="$u_A$", tc="#92400E", fs=7.4)
ax.text(11.2, 37.8, "Source", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#92400E")

# Branch 2 (Middle): Reader u_B1 (15.5, 33.2) -> u_B2 (27.5, 33.2)
arrow(7.0, 33.2, 14.0, 33.2, col="#0F766E", lw=1.4, ms=8)
cnode(15.5, 33.2, 1.3, fc="#CCFBF1", ec="#0F766E", txt="$u_{B1}$", tc="#065F46", fs=7.0)
arrow(16.9, 33.2, 25.9, 33.2, col="#2563EB", lw=1.4, ms=8)
cnode(27.5, 33.2, 1.4, fc="#DBEAFE", ec="#2563EB", txt="$u_{B2}$", tc="#1E40AF", fs=7.2)
ax.text(31.8, 33.2, "Reader", ha="left", va="center", fontsize=6.2, fontweight="bold", color="#1E40AF")

# Mount edge from u_A (15.5, 37.2) to u_B2 (27.5, 33.2)
arrow(16.9, 36.9, 26.1, 34.0, col="#D97706", lw=1.55, ls="--", ms=8.5, z=6)
ax.text(22.5, 36.8, "Mount $\\mathbf{K}_A^{(\\ell)}$", ha="center", va="center", fontsize=6.4, fontweight="bold", color="#B45309")

# Branch 3 (Bottom): Pre-Rewind l_pre (15.5, 29.0) -> Rewind to Post-Rewind l_post (27.5, 29.0)
arrow(6.9, 32.5, 14.0, 29.4, col="#94A3B8", lw=1.2, ls=":", ms=7.5)
cnode(15.5, 29.0, 1.25, fc="#FEE2E2", ec="#DC2626", txt="$l_{\\mathrm{pre}}$", tc="#991B1B", fs=6.7)
arrow(16.9, 29.0, 26.0, 29.0, col="#16A34A", lw=1.35, ms=8)
ax.text(21.4, 30.1, "rewind", ha="center", va="center", fontsize=6.1, fontweight="bold", color="#166534")
cnode(27.5, 29.0, 1.3, fc="#DCFCE7", ec="#16A34A", txt="$l_{\\mathrm{post}}$", tc="#166534", fs=6.7)

# Compiler Funnel arrow down from 2D Tree to 1D GPU Packed Sequence
arrow(18.9, 26.6, 18.9, 23.6, col="#4F46E5", lw=1.8, ms=10, z=6)
rbox(11.5, 24.3, 14.8, 1.9, fc="#EEF2FF", ec="#4F46E5", lw=0.8, r=0.35, z=7)
ax.text(18.9, 25.25, "Pack into $1\\times$ GPU Micro-Batch", ha="center", va="center",
        fontsize=6.5, fontweight="bold", color="#3730A3", zorder=8)

# --- Bottom Half of (a): 1D Packed GPU Sequence Buffer + Forward Hooks & Backward Autograd ---
rbox(2.0, 4.6, 33.8, 18.8, fc="#FFFFFF", ec="#9333EA", lw=1.0, r=0.6, z=2)
ax.text(3.1, 22.1, "2. 1D Packed FlashAttention Buffer ($0$ Prefix Re-Computation)",
        ha="left", va="center", fontsize=7.3, fontweight="bold", color="#581C87")

# Forward GDN Recurrent Boundary Hook Arcs ABOVE the 1D tape (staying cleanly below y=18.8)
arrow(6.2, 16.1, 12.2, 16.1, col="#2563EB", lw=1.15, rad=-0.26, ms=7.0, z=6)
arrow(6.2, 16.1, 23.8, 16.1, col="#2563EB", lw=1.15, rad=-0.20, ms=7.0, z=6)
arrow(6.2, 16.1, 29.2, 16.1, col="#16A34A", lw=1.1, rad=-0.17, ls="--", ms=7.0, z=6)
ax.text(18.9, 20.1, "GDN Recurrent Boundary Hooks $\\mathbf{h}_{\\mathrm{end}}(u_0)$ (Span $A$ Bypassed)",
        ha="center", va="center", fontsize=6.3, fontweight="bold", color="#1E40AF", zorder=7)

# The 1D Contiguous GPU Token Tape at y=12.4..16.0
tape_segs = [
    (3.2,  5.8, "#E2E8F0", "#475569", "$u_0$\n($1\\times$)", "#1E293B"),
    (9.2,  6.2, "#FEF3C7", "#D97706", "$u_A$\nSource", "#92400E"),
    (15.6, 5.2, "#FEE2E2", "#DC2626", "$l_{\\mathrm{pre}}$\nAband.", "#991B1B"),
    (21.0, 5.4, "#CCFBF1", "#0F766E", "$u_{B1}$\nPrefix", "#065F46"),
    (26.6, 4.8, "#DCFCE7", "#16A34A", "$l_{\\mathrm{post}}$\nRew.", "#166534"),
    (31.6, 3.4, "#DBEAFE", "#2563EB", "$u_{B2}$\nRead", "#1E40AF"),
]
for sx, sw, sfc, sec, stxt, stc in tape_segs:
    ax.add_patch(Rectangle((sx, 12.3), sw, 3.7, facecolor=sfc, edgecolor=sec, lw=0.85, zorder=4))
    ax.text(sx + sw/2, 14.15, stxt, ha="center", va="center", fontsize=6.1, fontweight="bold", color=stc, zorder=5)

# Reverse Cross-Branch Autograd Arc BELOW the 1D tape from u_B2 (33.3, 12.2) back into u_A (12.3, 12.2)!
arrow(33.3, 12.2, 12.3, 12.2, col="#E11D48", lw=1.7, rad=-0.20, ms=9.5, z=7)
rbox(4.2, 5.2, 29.4, 3.3, fc="#FFF1F2", ec="#E11D48", lw=0.9, r=0.4, z=6)
ax.text(18.9, 6.85, "Cross-Branch Autograd $\\partial \\mathcal{L}_{B2} / \\partial \\mathbf{K}_A^{(\\ell)} \\neq 0$\n($\\|\\nabla_X\\|=1.9200$ Verified $\\cdot$ Rollout $|\\Delta\\log\\pi_\\theta| = 0.0038$)",
        ha="center", va="center", fontsize=6.3, fontweight="bold", color="#9F1239", zorder=7)

ax.text(18.9, 1.5, "(a) Tree-to-1D Packed DAG & Autograd", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#111827")

# ==============================================================================
# PANEL (b): 2D Block-Sparse Tree-and-Mount Attention Mask (l in L_mount)
# ==============================================================================
rbox(38.4, 3.6, 37.2, 41.6, fc="#FAFAFA", ec="#CBD5E1", lw=1.15, r=1.0, z=1)
ax.text(57.0, 43.4, "2D Block-Sparse Attention Mask ($\\ell \\in \\mathcal{L}_{\\mathrm{mount}}$)",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1E3A8A")

tokens = ["$t_0$", "$t_1$", "$a_0$", "$a_1$", "$b_0$", "$b_1$", "$b_2$", "$c_0$"]
n_tok = len(tokens)
gx0, gy0 = 47.6, 37.2
cw, ch = 3.25, 3.25

ax.text(gx0 + 4.0 * cw, gy0 + 3.0, "Key / Value Token Stream ($\\mathbf{K}^{(\\ell)}, \\mathbf{V}^{(\\ell)}$)",
        ha="center", va="center", fontsize=7.3, fontweight="bold", color="#334155")

for j, tok in enumerate(tokens):
    ax.text(gx0 + (j + 0.5) * cw, gy0 + 1.2, tok, ha="center", va="center",
            fontsize=7.8, fontweight="bold", color="#1E293B")

row_groups = [
    ("Trunk $u_0$",  [0, 1], "#E2E8F0", "#475569"),
    ("Source $u_A$", [2, 3], "#FEF3C7", "#92400E"),
    ("Node $u_{B1}$",[4],    "#CCFBF1", "#065F46"),
    ("Read $u_{B2}$",[5, 6], "#E0E7FF", "#1E40AF"),
    ("Trans $u_C$",  [7],    "#DBEAFE", "#1D4ED8"),
]

for glbl, r_indices, gfc, gec in row_groups:
    y_top = gy0 - r_indices[0] * ch
    y_bot = gy0 - (r_indices[-1] + 1) * ch
    rbox(39.6, y_bot + 0.15, 7.4, (y_top - y_bot) - 0.3, fc=gfc, ec=gec, lw=0.85, r=0.35, z=3)
    ax.text(43.3, 0.5 * (y_top + y_bot), glbl, ha="center", va="center",
            fontsize=7.1, fontweight="bold", color=gec)

mask_grid = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 2, 0, 0, 0, 0, 0],
    [1, 1, 2, 2, 0, 0, 0, 0],
    [1, 1, 0, 0, 3, 0, 0, 0],
    [1, 1, 5, 5, 3, 4, 0, 0],
    [1, 1, 5, 5, 3, 4, 4, 0],
    [1, 1, 6, 6, 3, 6, 6, 7],
]

cell_styles = {
    0: ("#F8FAFC", "#E2E8F0", "", "#64748B"),
    1: ("#E2E8F0", "#94A3B8", "$\\checkmark$", "#334155"),
    2: ("#FEF3C7", "#D97706", "$\\checkmark$", "#92400E"),
    3: ("#CCFBF1", "#0F766E", "$\\checkmark$", "#065F46"),
    4: ("#E0E7FF", "#4F46E5", "$\\checkmark$", "#3730A3"),
    5: ("#FDE047", "#B45309", "$\\bigstar$", "#991B1B"),
    6: ("#93C5FD", "#1D4ED8", "$\\bigstar$", "#1E3A8A"),
    7: ("#DBEAFE", "#2563EB", "$\\checkmark$", "#1E40AF"),
}

for r in range(n_tok):
    for c in range(n_tok):
        st = mask_grid[r][c]
        fc, ec, sym, tc = cell_styles[st]
        x_c = gx0 + c * cw
        y_c = gy0 - (r + 1) * ch
        lw = 1.1 if st in (5, 6) else 0.55
        ax.add_patch(Rectangle((x_c + 0.08, y_c + 0.08), cw - 0.16, ch - 0.16,
                               facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
        if sym:
            ax.text(x_c + 0.5 * cw, y_c + 0.5 * ch, sym, ha="center", va="center",
                    fontsize=7.6, fontweight="bold", color=tc, zorder=4)

leg_items = [
    ("#E2E8F0", "#64748B", "Shared Trunk"),
    ("#E0E7FF", "#4F46E5", "Self Causal"),
    ("#FDE047", "#B45309", "$\\bigstar$ Mount $A$"),
    ("#93C5FD", "#1D4ED8", "$\\bigstar$ Transitive"),
]
for idx, (lfc, lec, ltxt) in enumerate(leg_items):
    lx = 39.6 + idx * 8.65
    ax.add_patch(Rectangle((lx, 5.4), 1.7, 1.7, facecolor=lfc, edgecolor=lec, linewidth=0.8, zorder=3))
    ax.text(lx + 2.0, 6.25, ltxt, ha="left", va="center", fontsize=6.9, fontweight="bold", color="#334155")

ax.text(57.0, 1.5, "(b) 2D Tree & Cross-Branch Attention Mask", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#111827")

# ==============================================================================
# PANEL (c): 2D Token-Segment Single-Ownership Matrix & LOO Tree Credit
# ==============================================================================
rbox(77.0, 3.6, 36.2, 41.6, fc="#FAFAFA", ec="#CBD5E1", lw=1.15, r=1.0, z=1)
ax.text(95.1, 43.4, "Single-Ownership Grid & LOO Credit", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#1E3A8A")

# --- Top Half of (c): 2D Ownership Matrix (Rows = Exported Leaves, Cols = Tree Segments) ---
rbox(78.2, 20.6, 33.8, 21.2, fc="#FFFFFF", ec="#94A3B8", lw=0.95, r=0.6, z=2)
ax.text(95.1, 40.3, "1. Single-Ownership Loss Mask $m_{l,t} \\in \\{0, 1\\}$ Across Leaves",
        ha="center", va="center", fontsize=7.6, fontweight="bold", color="#1E293B")

col_headers = ["$u_0$", "$u_A$", "$\\mathcal{M}(A)$", "$u_B$", "$l_{\\mathrm{pre}}$", "$l_{\\mathrm{post}}$"]
col_x0 = 84.8
col_w = 4.25
for c_i, chdr in enumerate(col_headers):
    ax.text(col_x0 + (c_i + 0.5) * col_w, 37.8, chdr, ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#1E293B")

leaf_rows = [
    ("Leaf $A$",        [1, 1, -1, -1, -1, -1], "#FEF3C7", "#D97706"),
    ("Leaf $B$",        [0, -1, 0,  1, -1, -1], "#DBEAFE", "#2563EB"),
    ("Leaf $l_{\\mathrm{pre}}$", [0, -1, -1, -1, 1, -1], "#FEE2E2", "#DC2626"),
    ("Leaf $l_{\\mathrm{post}}$",[0, -1, -1, -1, 0,  1], "#DCFCE7", "#16A34A"),
]

row_y0 = 36.4
row_h = 2.85
for r_i, (rlbl, rvals, rfc, rec) in enumerate(leaf_rows):
    ry = row_y0 - (r_i + 1) * row_h
    ax.text(79.0, ry + 0.5 * row_h, rlbl, ha="left", va="center", fontsize=7.0, fontweight="bold", color="#334155")
    for c_i, val in enumerate(rvals):
        cx = col_x0 + c_i * col_w
        if val == 1:
            ax.add_patch(Rectangle((cx + 0.15, ry + 0.2), col_w - 0.3, row_h - 0.4,
                                   facecolor=rfc, edgecolor=rec, lw=1.0, zorder=4))
            ax.text(cx + 0.5 * col_w, ry + 0.5 * row_h, "$m=1$", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="#1E293B", zorder=5)
        elif val == 0:
            ax.add_patch(Rectangle((cx + 0.15, ry + 0.2), col_w - 0.3, row_h - 0.4,
                                   facecolor="#F1F5F9", edgecolor="#94A3B8", lw=0.75, ls="--", zorder=3))
            ax.text(cx + 0.5 * col_w, ry + 0.5 * row_h, "$0$", ha="center", va="center",
                    fontsize=6.5, color="#64748B", zorder=4)
        else:
            ax.add_patch(Rectangle((cx + 0.15, ry + 0.2), col_w - 0.3, row_h - 0.4,
                                   facecolor="#FAFAFA", edgecolor="#E2E8F0", lw=0.5, zorder=2))
            ax.text(cx + 0.5 * col_w, ry + 0.5 * row_h, "—", ha="center", va="center",
                    fontsize=6.2, color="#CBD5E1", zorder=3)

# Column sum row at bottom of grid proving exact single ownership!
sum_y = row_y0 - 4 * row_h - 2.4
ax.plot([79.0, 110.4], [sum_y + 2.15, sum_y + 2.15], color="#64748B", lw=0.8, zorder=4)
ax.text(79.0, sum_y + 1.0, "$\\sum_l m_{l,t}$:", ha="left", va="center", fontsize=7.0, fontweight="bold", color="#065F46")
col_sums = ["$1\\times$", "$1\\times$", "$0$ (RO)", "$1\\times$", "$1\\times$", "$1\\times$"]
for c_i, csum in enumerate(col_sums):
    cx = col_x0 + c_i * col_w
    fc_s = "#F1F5F9" if "0" in csum else "#ECFDF5"
    ec_s = "#94A3B8" if "0" in csum else "#0F766E"
    tc_s = "#64748B" if "0" in csum else "#065F46"
    ax.add_patch(Rectangle((cx + 0.15, sum_y + 0.1), col_w - 0.3, 1.8, facecolor=fc_s, edgecolor=ec_s, lw=0.85, zorder=4))
    ax.text(cx + 0.5 * col_w, sum_y + 1.0, csum, ha="center", va="center", fontsize=6.2, fontweight="bold", color=tc_s, zorder=5)

ax.text(95.1, 21.5, "Every generated reply is trained $1\\times$; mounted spans $\\mathcal{M}(A)$ are read-only ($0$)",
        ha="center", va="center", fontsize=6.4, fontweight="bold", color="#475569")

# --- Bottom Half of (c): Group of G Branched Trees & Leave-One-Out Credit ---
rbox(78.2, 4.6, 33.8, 14.8, fc="#ECFDF5", ec="#0F766E", lw=0.95, r=0.6, z=2)
ax.text(95.1, 17.9, "2. Pure Outcome Leave-One-Out Tree Advantage ($R_i = 2y_i - 1$)",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#065F46")

# Draw 3 miniature rollout trees on the left (tau_1, tau_2, tau_G)
tree_specs = [
    (81.5, 13.8, "$\\tau_1$: $y=1$", "$R_1=+1$", "#DCFCE7", "#16A34A"),
    (81.5, 10.4, "$\\tau_2$: $y=0$", "$R_2=-1$", "#FEE2E2", "#DC2626"),
    (81.5,  7.0, "$\\tau_G$: $y=1$", "$R_G=+1$", "#DCFCE7", "#16A34A"),
]
for tx, ty, tlbl, trw, tfc, tec in tree_specs:
    rbox(tx - 2.4, ty - 1.2, 11.2, 2.5, fc=tfc, ec=tec, lw=0.85, r=0.35, z=3)
    ax.text(tx + 3.2, ty + 0.05, f"{tlbl} ({trw})", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#1E293B", zorder=4)
    arrow(90.5, ty + 0.05, 94.2, 10.4, col="#0F766E", lw=1.0, ms=7.5, z=4)

rbox(94.5, 6.2, 16.2, 8.4, fc="#FFFFFF", ec="#0F766E", lw=1.0, r=0.5, z=3)
ax.text(102.6, 12.6, "LOO Tree Advantage", ha="center", va="center", fontsize=7.3, fontweight="bold", color="#065F46", zorder=4)
ax.text(102.6, 9.8, "$\\hat{A}(\\tau_i) = R_i - \\frac{1}{G-1}\\sum_{j\\neq i} R_j$",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0F766E", zorder=4)
ax.text(102.6, 7.3, "Broadcast to all $m_{l,t}=1$ tokens", ha="center", va="center", fontsize=6.3, fontweight="bold", color="#B45309", zorder=4)

ax.text(95.1, 1.5, "(c) Token-Segment Ownership Matrix & LOO Credit", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#111827")

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
out_pdf = "./figures/fig3_dag_grpo.pdf"
out_svg = "./figures/fig3_dag_grpo.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
print("Saved Figure 3:", out_pdf)
