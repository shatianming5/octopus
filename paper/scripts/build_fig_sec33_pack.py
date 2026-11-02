#!/usr/bin/env python3
"""
Section 3.3 Standalone Figure 5: Tree-to-1D Contiguous Sequence Packing & Recurrent Boundary Hooks.
Print-calibrated on compact W=820 canvas (0.571 pt/px) so all labels print at 7.8pt-10.0pt
with bold, spacious, crystal-clear legibility.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_pdf import render_svg_to_pdf, figma_rich_text, svg_mascot

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig_sec33_pack.svg"
PDF_OUT = ROOT / "figures/fig_sec33_pack.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig_sec33_pack_check.png")

W, H = 820, 248

INK        = "#1E293B"
MUTED      = "#64748B"

BLUE_FILL  = "#CFE3FA"
BLUE_STR   = "#3B6E9C"
BLUE_TXT   = "#1E3A5F"

APRI_FILL  = "#FDE2C4"
APRI_STR   = "#C86D2D"
APRI_TXT   = "#7C2D12"

MINT_FILL  = "#CBECE4"
MINT_HERO  = "#9FE0C9"
MINT_STR   = "#1B6B55"
MINT_TXT   = "#0B4F3F"

VIOL_STR   = "#6D28D9"
VIOL_TXT   = "#4C1D95"

STONE_FILL = "#F1EFEA"
STONE_STR  = "#A8A29E"
STONE_TXT  = "#78716C"


def arrow(x1: float, y1: float, x2: float, y2: float, col: str, sw: float = 1.8, dash: str = "") -> str:
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return ""
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    hl, hw = 7.2, 3.7
    bx, by = x2 - ux * hl, y2 - uy * hl
    lx1, ly1 = x2 - ux * (hl * 0.72), y2 - uy * (hl * 0.72)
    p1x, p1y = bx + px * hw, by + py * hw
    p2x, p2y = bx - px * hw, by - py * hw
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<g>'
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{lx1:.1f}" y2="{ly1:.1f}" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"{d_attr}/>'
        f'<polygon points="{x2:.1f},{y2:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" fill="{col}"/>'
        f'</g>'
    )


def state_arc(x1: float, x2: float, y_base: float, h_arc: float, col: str = MINT_STR, sw: float = 1.85) -> str:
    c1x = x1 + (x2 - x1) * 0.22
    c2x = x2 - (x2 - x1) * 0.22
    cy = y_base - h_arc
    tx, ty = x2 - c2x, y_base - cy
    L = math.hypot(tx, ty) or 1.0
    ux, uy = tx / L, ty / L
    px, py = -uy, ux
    hl, hw = 6.6, 3.3
    bx, by = x2 - ux * hl, y_base - uy * hl
    p1x, p1y = bx + px * hw, by + py * hw
    p2x, p2y = bx - px * hw, by - py * hw
    return (
        f'<g>'
        f'<path d="M {x1:.1f} {y_base:.1f} C {c1x:.1f} {cy:.1f}, {c2x:.1f} {cy:.1f}, {bx:.1f} {by:.1f}" '
        f'fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"/>'
        f'<polygon points="{x2:.1f},{y_base:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" fill="{col}"/>'
        f'</g>'
    )


def node_pill(x: float, y: float, w: float, h: float, fill: str, stroke: str, txt: str, tcol: str,
              size: float = 15.2, dashed: bool = False, rx: float = 8.0) -> str:
    d_attr = ' stroke-dasharray="4.2 2.6"' if dashed else ""
    return (
        f'<g>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.55"{d_attr}/>'
        f'{figma_rich_text(x + w/2, y + h/2 + 5.3, txt, size=size, weight=700, fill=tcol)}'
        f'</g>'
    )


out: list[str] = []
p = out.append

p(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ==============================================================================
# 1. SECTION TITLES (17.6px = 10.0pt printed!)
# ==============================================================================
p(figma_rich_text(154, 25, "Episode Execution Tree", size=17.6, weight=700, fill=INK))
p(figma_rich_text(574, 25, "Octopus 1D Contiguous Packed Pass", size=17.6, weight=700, fill=MINT_TXT))

# ==============================================================================
# 2. LEFT ZONE: BRANCHED EPISODE TREE (x = 10..298)
# ==============================================================================
p('<g id="Episode Execution Tree">')
p(node_pill(10,  116, 82, 38, BLUE_FILL,  BLUE_STR,  "Trunk u_{0}", BLUE_TXT, size=14.8))
p(node_pill(116, 52,  92, 38, APRI_FILL,  APRI_STR,  "Source u_{A}",APRI_TXT, size=14.8))
p(node_pill(116, 116, 76, 38, MINT_FILL,  MINT_STR,  "u_{B1}",      MINT_TXT, size=15.4))
p(node_pill(214, 116, 84, 38, MINT_HERO,  MINT_STR,  "Read u_{B2}", MINT_TXT, size=14.8))
p(node_pill(116, 180, 76, 38, STONE_FILL, STONE_STR, "l_{pre}",     STONE_TXT, size=15.4, dashed=True))
p(node_pill(214, 180, 84, 38, MINT_FILL,  MINT_STR,  "l_{post}",    MINT_TXT, size=15.4))

# Tree edges from Trunk u_0
p(arrow(92, 130, 114, 71,  APRI_STR,  1.8))
p(arrow(92, 135, 114, 135, MINT_STR,  1.8))
p(arrow(92, 140, 114, 199, STONE_STR, 1.7, dash="3.8 2.4"))

# Continuations
p(arrow(192, 135, 212, 135, MINT_STR, 1.8))
p(arrow(192, 199, 212, 199, MINT_STR, 1.8))
p(figma_rich_text(203, 173, "rewind", size=13.6, weight=700, fill=MINT_TXT))

# 0-copy mount edge u_A -> u_B2
p(f'<path d="M 208 71 L 256 71 L 256 108" fill="none" stroke="{VIOL_STR}" stroke-width="1.8" stroke-dasharray="4.2 2.6" stroke-linecap="round"/>')
p(f'<polygon points="256,115 252.2,107.8 259.8,107.8" fill="{VIOL_STR}"/>')
p(figma_rich_text(246, 62, "mount", size=13.8, weight=700, fill=VIOL_TXT))

# Packing arrow into Right Container
p(arrow(304, 135, 331, 135, MINT_STR, 2.2))
p('</g>')

# ==============================================================================
# 3. RIGHT ZONE: 1D CONTIGUOUS PACKED STRIP + BOUNDARY HOOKS + MASCOT
# ==============================================================================
p('<g id="Octopus 1D Contiguous Packed Pass">')
p('<rect x="336" y="40" width="474" height="198" rx="13" fill="#F3FBF8" stroke="#A7D7C5" stroke-width="1.5"/>')

p(figma_rich_text(352, 66, "✓ 1× prefix · Zero padding", size=14.8, weight=700, fill=MINT_TXT, anchor="start"))
p(figma_rich_text(794, 66, "State hooks h_{end}", size=14.8, weight=700, fill=MINT_STR, anchor="end"))

# Chunky 1D Contiguous Strip (x = 352..720, width = 368, y = 150, h = 43)
SY = 150.0
SH = 43.0
segs = [
    (70, BLUE_FILL,  BLUE_STR,  BLUE_TXT,  "u_{0} (1×)", 14.4),
    (58, APRI_FILL,  APRI_STR,  APRI_TXT,  "u_{A}",      15.4),
    (58, STONE_FILL, STONE_STR, STONE_TXT, "l_{pre}",    15.2),
    (60, MINT_FILL,  MINT_STR,  MINT_TXT,  "u_{B1}",     15.2),
    (60, MINT_FILL,  MINT_STR,  MINT_TXT,  "l_{post}",   15.2),
    (62, MINT_HERO,  MINT_STR,  MINT_TXT,  "u_{B2}",     15.2),
]

cur_x = 352.0
bounds = [cur_x]
for sw_px, sfill, sstr, stxt, slbl, fsz in segs:
    p(f'<rect x="{cur_x:.1f}" y="{SY:.1f}" width="{sw_px:.1f}" height="{SH:.1f}" rx="5.5" fill="{sfill}" stroke="{sstr}" stroke-width="1.5"/>')
    p(figma_rich_text(cur_x + sw_px/2, SY + 26.8, slbl, size=fsz, weight=700, fill=stxt))
    cur_x += sw_px
    bounds.append(cur_x)

# cu_seqlens boundary ticks
for bx in bounds:
    p(f'<line x1="{bx:.1f}" y1="{SY + SH + 3:.1f}" x2="{bx:.1f}" y2="{SY + SH + 10:.1f}" stroke="#64748B" stroke-width="1.5" stroke-linecap="round"/>')

p(figma_rich_text(536, 224, "cu_seqlens · Single contiguous MLP & QKV pass", size=14.2, weight=600, fill=MUTED))

# Recurrent State Boundary Hooks h_end(p(u))
p(f'<circle cx="{bounds[1]:.1f}" cy="{SY:.1f}" r="3.8" fill="{MINT_STR}"/>')
p(f'<circle cx="{bounds[4]:.1f}" cy="{SY:.1f}" r="3.8" fill="{MINT_STR}"/>')

p(state_arc(bounds[1], bounds[2] - 1, SY - 2, 34))
p(state_arc(bounds[1], bounds[3] - 1, SY - 2, 47))
p(state_arc(bounds[1], bounds[4] - 2, SY - 2, 60))
p(state_arc(bounds[4] + 2, bounds[5] - 1, SY - 2, 30))

p(figma_rich_text(451, 143, "skip u_{A}", size=12.4, weight=700, fill=VIOL_TXT))

# Block-packing Octopus mascot
p(svg_mascot("octo_dag", 765, 160, 72, 72))
p('</g>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved bold print-calibrated Figure 5 (Pack): {PDF_OUT}")
