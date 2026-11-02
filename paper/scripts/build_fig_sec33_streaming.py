#!/usr/bin/env python3
"""
Section 3.3 Standalone Method Figure (4-Column Cross-Branch Attention & Position Jump).
High-density print-calibrated W=820x320 canvas so matrices, token strips, mascots,
and 7.7pt-9.5pt labels fill each column cleanly with zero sparse white voids.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_pdf import render_svg_to_pdf, figma_rich_text, svg_mascot

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig_sec33_streaming.svg"
PDF_OUT = ROOT / "figures/fig_sec33_streaming.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig_sec33_streaming_check.png")

W, H = 820, 320

INK        = "#1E293B"
MUTED      = "#64748B"

TILE_MASK  = "#ECEAE4"
TILE_MINT  = "#8FD9C1"
TILE_DIAG  = "#1B7A62"
TILE_VIOL  = "#C4B5FD"
TILE_VDIAG = "#8B5CF6"
TILE_CORAL = "#EE6C4D"
TILE_ERR   = "#FAD4D4"

GOOD_TXT   = "#0F6E56"
BAD_TXT    = "#B9382C"
VIOL_TXT   = "#5B21B6"
VIOL_STR   = "#6D28D9"


def curved_arc(x_from: float, y_from: float, x_to: float, y_to: float, h_arc: float,
               col: str = "#334155", dash: str = "", sw: float = 1.55) -> str:
    mx = 0.5 * (x_from + x_to)
    my = min(y_from, y_to) - h_arc
    tx, ty = x_to - mx, y_to - my
    L = math.hypot(tx, ty) or 1.0
    ux, uy = tx / L, ty / L
    px, py = -uy, ux
    hl, hw = 5.8, 2.9
    bx, by = x_to - ux * hl, y_to - uy * hl
    p1x, p1y = bx + px * hw, by + py * hw
    p2x, p2y = bx - px * hw, by - py * hw
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<g>'
        f'<path d="M {x_from:.1f} {y_from:.1f} Q {mx:.1f} {my:.1f} {bx:.1f} {by:.1f}" '
        f'fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"{d_attr}/>'
        f'<polygon points="{x_to:.1f},{y_to:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" fill="{col}"/>'
        f'</g>'
    )


out: list[str] = []
p = out.append

p(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# Soft mint highlight card behind Column (d) Ours
p('<rect x="613" y="4" width="203" height="312" rx="13" fill="#F2FBF8" stroke="#94D3BC" stroke-width="1.5"/>')

N = 7
TS = 20.2
TG = 2.5
STEP = TS + TG
MAT_W = N * TS + (N - 1) * TG  # 156.4px
MY0 = 42.0


def draw_matrix(cx: float, grid: list[list[str]]) -> None:
    x0 = cx - MAT_W / 2.0
    p(f'  <rect x="{x0 - 6.3:.1f}" y="{MY0 - 6.3:.1f}" width="{MAT_W + 12.6:.1f}" height="{MAT_W + 12.6:.1f}" rx="9" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3"/>')
    for r in range(N):
        for c in range(N):
            code = grid[r][c]
            x = x0 + c * STEP
            y = MY0 + r * STEP
            fill = {
                "M": TILE_MASK,
                "C": TILE_MINT,
                "D": TILE_DIAG,
                "V": TILE_VIOL,
                "W": TILE_VDIAG,
                "Q": TILE_CORAL,
                "X": TILE_ERR,
            }[code]
            p(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{TS:.1f}" height="{TS:.1f}" rx="3.8" fill="{fill}"/>')
            if code == "X":
                cx_t, cy_t = x + TS / 2.0, y + TS / 2.0
                d = 4.2
                p(f'  <line x1="{cx_t-d:.1f}" y1="{cy_t-d:.1f}" x2="{cx_t+d:.1f}" y2="{cy_t+d:.1f}" stroke="{BAD_TXT}" stroke-width="1.9" stroke-linecap="round"/>')
                p(f'  <line x1="{cx_t+d:.1f}" y1="{cy_t-d:.1f}" x2="{cx_t-d:.1f}" y2="{cy_t+d:.1f}" stroke="{BAD_TXT}" stroke-width="1.9" stroke-linecap="round"/>')


TW, TH = 22.8, 20.5
T_STEP = (MAT_W - TW) / 5.0  # 26.72px -> 6 slots span exact MAT_W = 156.4px


def strip_tile(x: float, y: float, fill: str, stroke: str = "#334155", dash: str = "") -> None:
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    p(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{TW:.1f}" height="{TH:.1f}" rx="4.2" fill="{fill}" stroke="{stroke}" stroke-width="1.3"{d_attr}/>')


def dots_3(cx_d: float, cy_d: float, col: str = MUTED) -> None:
    for dx in (-4.8, 0.0, 4.8):
        p(f'  <circle cx="{cx_d + dx:.1f}" cy="{cy_d:.1f}" r="1.55" fill="{col}"/>')


COL_CX = [101.0, 304.0, 507.0, 714.5]
SY0 = 225.0

# ==============================================================================
# (a) Flat Text Copy
# ==============================================================================
cx = COL_CX[0]
sx = cx - MAT_W / 2.0
p('<g id="Col a">')
p(figma_rich_text(cx, 24, "(a) Flat Text Copy", size=16.0, weight=700, fill=INK))
grid_a = [["M"] * N for _ in range(N)]
for r in range(N):
    for c in range(r + 1):
        grid_a[r][c] = "D" if r == c else "C"
grid_a[N - 1][N - 1] = "Q"
draw_matrix(cx, grid_a)

strip_tile(sx + 0 * T_STEP, SY0, TILE_MINT)
strip_tile(sx + 1 * T_STEP, SY0, TILE_MINT)
dots_3(sx + 2 * T_STEP + TW / 2, SY0 + TH / 2)
strip_tile(sx + 3 * T_STEP, SY0, TILE_VIOL, VIOL_STR)
strip_tile(sx + 4 * T_STEP, SY0, TILE_MINT)
strip_tile(sx + 5 * T_STEP, SY0, TILE_CORAL, "#B9382C")
qx = sx + 5 * T_STEP + TW / 2
p(curved_arc(qx, SY0 - 1, sx + 0 * T_STEP + TW / 2, SY0 - 1, 17))
p(curved_arc(qx, SY0 - 1, sx + 3 * T_STEP + TW / 2, SY0 - 1, 11.5))
p(curved_arc(qx, SY0 - 1, sx + 4 * T_STEP + TW / 2, SY0 - 1, 7))
p(figma_rich_text(cx, SY0 + 36, "Appends sibling text", size=13.5, weight=600, fill=MUTED))

p(svg_mascot("octo_09_brainwash", cx - 72, 293, 36, 44))
p(figma_rich_text(cx - 48, 286, "✗ Steps GDN", size=14.6, weight=700, fill=BAD_TXT, anchor="start"))
p(figma_rich_text(cx - 48, 306, "Overwrites memory.", size=13.2, weight=500, fill=INK, anchor="start"))
p('</g>')

# ==============================================================================
# (b) Unshifted Mount
# ==============================================================================
cx = COL_CX[1]
sx = cx - MAT_W / 2.0
p('<g id="Col b">')
p(figma_rich_text(cx, 24, "(b) Unshifted Mount", size=16.0, weight=700, fill=INK))
grid_b = [["M"] * N for _ in range(N)]
for r in range(4):
    for c in range(r + 1):
        grid_b[r][c] = "D" if r == c else "C"
grid_b[3][3] = "Q"
grid_b[3][4] = "X"
grid_b[3][5] = "X"
grid_b[3][6] = "X"
for r in range(4, N):
    for c in range(4, r + 1):
        grid_b[r][c] = "W" if r == c else "V"
draw_matrix(cx, grid_b)

strip_tile(sx + 0 * T_STEP, SY0, TILE_MINT)
strip_tile(sx + 1 * T_STEP, SY0, TILE_MINT)
strip_tile(sx + 2 * T_STEP, SY0, TILE_CORAL, "#B9382C")
dots_3(sx + 3 * T_STEP + TW / 2, SY0 + TH / 2)
strip_tile(sx + 4 * T_STEP, SY0, TILE_VIOL, VIOL_STR)
strip_tile(sx + 5 * T_STEP, SY0, TILE_VIOL, VIOL_STR)
qx_b = sx + 2 * T_STEP + TW / 2
p(curved_arc(qx_b, SY0 - 1, sx + 0 * T_STEP + TW / 2, SY0 - 1, 10.5))
p(curved_arc(qx_b, SY0 - 1, sx + 4 * T_STEP + TW / 2, SY0 - 1, 14.5, col=BAD_TXT, dash="3.4 2.3", sw=1.6))
p(figma_rich_text(cx, SY0 + 36, "Query behind sibling", size=13.5, weight=600, fill=BAD_TXT))

p(svg_mascot("octo_12_rope_future", cx - 72, 294, 40, 40))
p(figma_rich_text(cx - 47, 286, "✗ RoPE Phase Flip", size=14.6, weight=700, fill=BAD_TXT, anchor="start"))
p(figma_rich_text(cx - 47, 306, "Keys fall in future.", size=13.2, weight=500, fill=INK, anchor="start"))
p('</g>')

# ==============================================================================
# (c) Copy & Rotate
# ==============================================================================
cx = COL_CX[2]
sx = cx - MAT_W / 2.0
p('<g id="Col c">')
p(figma_rich_text(cx, 24, "(c) Copy & Rotate", size=16.0, weight=700, fill=INK))
grid_c = [["M"] * N for _ in range(N)]
for r in range(2, N):
    for c in range(2, r + 1):
        if c in (2, 3):
            grid_c[r][c] = "W" if r == c else "V"
        else:
            grid_c[r][c] = "D" if r == c else "C"
grid_c[N - 1][N - 1] = "Q"
draw_matrix(cx, grid_c)

strip_tile(sx + 0 * T_STEP, SY0, "#F1EFEA", "#A8A29E", dash="3.0 2.0")
strip_tile(sx + 1 * T_STEP, SY0, TILE_VIOL, VIOL_STR)
strip_tile(sx + 2 * T_STEP, SY0, TILE_VIOL, VIOL_STR)
strip_tile(sx + 3 * T_STEP, SY0, TILE_MINT)
strip_tile(sx + 4 * T_STEP, SY0, TILE_MINT)
strip_tile(sx + 5 * T_STEP, SY0, TILE_CORAL, "#B9382C")
qx_c = sx + 5 * T_STEP + TW / 2
p(curved_arc(qx_c, SY0 - 1, sx + 1 * T_STEP + TW / 2, SY0 - 1, 16))
p(curved_arc(qx_c, SY0 - 1, sx + 2 * T_STEP + TW / 2, SY0 - 1, 12.5))
p(curved_arc(qx_c, SY0 - 1, sx + 3 * T_STEP + TW / 2, SY0 - 1, 9))
p(figma_rich_text(cx, SY0 + 36, "Allocates rotated copy", size=13.5, weight=600, fill=MUTED))

p(svg_mascot("octo_13_rope_copy", cx - 72, 294, 40, 38))
p(figma_rich_text(cx - 47, 286, "✗ Copies KV", size=14.6, weight=700, fill=BAD_TXT, anchor="start"))
p(figma_rich_text(cx - 47, 306, "Duplicates memory.", size=13.2, weight=500, fill=INK, anchor="start"))
p('</g>')

# ==============================================================================
# (d) Octopus Jump (Ours)
# ==============================================================================
cx = COL_CX[3]
sx = cx - MAT_W / 2.0
p('<g id="Col d">')
p(figma_rich_text(cx, 24, "(d) Octopus Jump", size=16.3, weight=700, fill=GOOD_TXT))
grid_d = [["M"] * N for _ in range(N)]
grid_d[0][0] = "W"
grid_d[1][0] = "V"
grid_d[1][1] = "W"
for r in range(4, N):
    grid_d[r][0] = "V"
    grid_d[r][1] = "V"
    for c in range(4, r + 1):
        grid_d[r][c] = "D" if r == c else "C"
grid_d[N - 1][N - 1] = "Q"
draw_matrix(cx, grid_d)

strip_tile(sx + 0 * T_STEP, SY0, TILE_VIOL, VIOL_STR)
strip_tile(sx + 1 * T_STEP, SY0, TILE_VIOL, VIOL_STR)
strip_tile(sx + 2 * T_STEP, SY0, "#E6F7F1", "#1B7A62", dash="3.2 2.0")
p(figma_rich_text(sx + 2 * T_STEP + TW / 2, SY0 + 14.8, "+Δ", size=11.8, weight=700, fill=GOOD_TXT))
strip_tile(sx + 3 * T_STEP, SY0, TILE_MINT)
strip_tile(sx + 4 * T_STEP, SY0, TILE_MINT)
strip_tile(sx + 5 * T_STEP, SY0, TILE_CORAL, "#B9382C")
qx_d = sx + 5 * T_STEP + TW / 2
p(curved_arc(qx_d, SY0 - 1, sx + 0 * T_STEP + TW / 2, SY0 - 1, 17, col=VIOL_STR))
p(curved_arc(qx_d, SY0 - 1, sx + 1 * T_STEP + TW / 2, SY0 - 1, 13.5, col=VIOL_STR))
p(curved_arc(qx_d, SY0 - 1, sx + 3 * T_STEP + TW / 2, SY0 - 1, 9))
p(figma_rich_text(cx, SY0 + 36, "0-copy + position jump", size=13.5, weight=700, fill=GOOD_TXT))

p(svg_mascot("octo_k2", cx - 72, 294, 40, 40))
p(figma_rich_text(cx - 47, 286, "✓ 0-Copy · Valid", size=14.6, weight=700, fill=GOOD_TXT, anchor="start"))
p(figma_rich_text(cx - 47, 306, "No hijack · 0 pages.", size=13.2, weight=600, fill=INK, anchor="start"))
p('</g>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved high-density Section 3.3 Streaming Figure: {PDF_OUT}")
