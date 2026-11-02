"""Shared palette and drawing helpers for the method figures (Fig. 1-3).

Figures are drawn at true print size (6.5 in = \\textwidth), 1 data unit = 0.05 in.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "hatch.linewidth": 0.5,
})

INK = "#1F2937"
MUTED = "#6B7280"
FAINT = "#D1D5DB"
TRUNK = "#64748B"
A = "#E39B2D"      # source branch / mounted KV
A_LT = "#FBE7C6"
B = "#2F6FDE"      # reader branch
B_LT = "#D6E4FB"
C = "#159A8C"      # third branch
C_LT = "#CDEFEA"
BAD = "#D9363E"
BAD_LT = "#FBDADB"
GRAD = "#B02AB8"   # backward gradient
OK = "#1A9A5A"

UNIT = 0.05  # inches per data unit


def canvas(w, h):
    fig = plt.figure(figsize=(w * UNIT, h * UNIT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def bezier(P, n=80):
    P = np.asarray(P, float)
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
    d = 3 * (1 - t) ** 2 * (P[1] - P[0]) + 6 * (1 - t) * t * (P[2] - P[1]) + 3 * t ** 2 * (P[3] - P[2])
    return pts, d


def tentacle(ax, P, w0, w1, color, alpha=1.0, z=3, ec="none", ls="-", lw=0.6, t0=0.0, t1=1.0):
    """Filled stroke along a cubic Bezier whose width tapers from w0 to w1."""
    pts, d = bezier(P)
    lo, hi = int(t0 * (len(pts) - 1)), int(t1 * (len(pts) - 1)) + 1
    pts, d = pts[lo:hi], d[lo:hi]
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    w = np.linspace(w0, w1, len(pts))[:, None] / 2
    poly = np.vstack([pts + nrm * w, (pts - nrm * w)[::-1]])
    ax.add_patch(Polygon(poly, closed=True, fc=color, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=z, joinstyle="round"))
    return pts


def point_on(P, t):
    pts, _ = bezier(P, 201)
    return pts[int(round(t * 200))]


def rbox(ax, x, y, w, h, fc="white", ec=FAINT, lw=0.6, r=0.6, z=2, ls="-", hatch=None, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       ls=ls, hatch=hatch, alpha=alpha, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, p, q, color=INK, lw=0.8, rad=0.0, ls="-", z=6, ms=6, style="-|>", shrinkA=0, shrinkB=0):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=shrinkA, shrinkB=shrinkB,
                        capstyle="round", joinstyle="round")
    ax.add_patch(a)
    return a


def cursor(ax, x, y, color, r=0.95, z=8, fc="white", lw=0.9):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=color, lw=lw, zorder=z))


def text(ax, x, y, s, size=6.5, color=INK, ha="center", va="center", weight="normal", z=9, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=z, **kw)


def panel(ax, x, y, s):
    text(ax, x, y, s, size=8, weight="bold", ha="left", va="top")


def save(fig, name):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", name)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".svg")
    print("saved", os.path.normpath(out + ".pdf"))
