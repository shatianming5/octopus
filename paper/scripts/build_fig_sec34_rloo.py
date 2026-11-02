#!/usr/bin/env python3
"""
Section 3.4 Standalone Figure 7: Single-Ownership Prefix Deduplication & Group RLOO Tree Credit.
Print-calibrated on compact W=840 canvas (0.557 pt/px) so every label prints at 7.7pt-9.8pt.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_pdf import render_svg_to_pdf, figma_rich_text, svg_mascot

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig_sec34_rloo.svg"
PDF_OUT = ROOT / "figures/fig_sec34_rloo.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig_sec34_rloo_check.png")

W, H = 840, 276

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

VIOL_FILL  = "#EDE9FE"
VIOL_STR   = "#6D28D9"
VIOL_TXT   = "#4C1D95"

BAD_SOFT   = "#FADBD8"
BAD_STR    = "#B9382C"
BAD_TXT    = "#991B1B"


def arrow(x1: float, y1: float, x2: float, y2: float, col: str, sw: float = 1.7, dash: str = "") -> str:
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return ""
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    hl, hw = 6.8, 3.5
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


def mini_seg(x: float, y: float, w: float, h: float, fill: str, stroke: str, lbl: str, tcol: str,
             size: float = 14.0, dashed: bool = False, rx: float = 6.0) -> str:
    d_attr = ' stroke-dasharray="3.8 2.4"' if dashed else ""
    return (
        f'<g>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.45"{d_attr}/>'
        f'{figma_rich_text(x + w/2, y + h/2 + 4.9, lbl, size=size, weight=700, fill=tcol)}'
        f'</g>'
    )


def mini_fork_icon(cx: float, cy: float, col: str) -> str:
    return (
        f'<g stroke="{col}" stroke-width="1.6" fill="{col}" stroke-linecap="round">'
        f'<line x1="{cx-5.5:.1f}" y1="{cy:.1f}" x2="{cx+4.5:.1f}" y2="{cy-5.5:.1f}"/>'
        f'<line x1="{cx-5.5:.1f}" y1="{cy:.1f}" x2="{cx+4.5:.1f}" y2="{cy+5.5:.1f}"/>'
        f'<circle cx="{cx-5.5:.1f}" cy="{cy:.1f}" r="2.1"/>'
        f'<circle cx="{cx+4.5:.1f}" cy="{cy-5.5:.1f}" r="2.1"/>'
        f'<circle cx="{cx+4.5:.1f}" cy="{cy+5.5:.1f}" r="2.1"/>'
        f'</g>'
    )


out: list[str] = []
p = out.append

p(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# 1. HEADERS (17.4px = 9.7pt printed!)
p(figma_rich_text(196, 25, "Single-Ownership Deduplication", size=17.4, weight=700, fill=INK))
p(figma_rich_text(618, 25, "Group RLOO Tree Credit", size=17.4, weight=700, fill=INK))

# 2. LEFT TOP: NAIVE LEAF FLATTENING
p('<g id="Naive Leaf Flattening">')
p('<rect x="10" y="38" width="374" height="108" rx="11" fill="#FDF8F8" stroke="#E8CACA" stroke-width="1.35"/>')
p(figma_rich_text(24,  58, "Naive Flattening", size=15.0, weight=700, fill=BAD_TXT, anchor="start"))
p(figma_rich_text(370, 58, "✗ 3× trunk bias", size=13.8, weight=700, fill=BAD_STR, anchor="end"))

p(mini_seg(22, 88, 90, 32, BAD_SOFT, BAD_STR, "u_{0} (3×)", BAD_TXT, size=14.2))
p(arrow(112, 100, 140, 77,  APRI_STR, 1.55))
p(arrow(112, 104, 140, 104, BAD_STR,  1.55))
p(arrow(112, 108, 140, 131, BAD_STR,  1.55))

p(mini_seg(142, 65,   104, 24, APRI_FILL, APRI_STR, "u_{A} (1×)",   APRI_TXT, size=13.6))
p(mini_seg(142, 92,   104, 24, BAD_SOFT,  BAD_STR,  "M(A) (1×)",    BAD_TXT,  size=13.6))
p(arrow(246, 104, 266, 104, MINT_STR, 1.5))
p(mini_seg(268, 92,   102, 24, MINT_FILL, MINT_STR, "u_{B} (1×)",   MINT_TXT, size=13.6))
p(mini_seg(142, 119,  104, 24, BAD_SOFT,  BAD_STR,  "l_{pre} (2×)", BAD_TXT,  size=13.6))
p(arrow(246, 131, 266, 131, MINT_STR, 1.5))
p(mini_seg(268, 119,  102, 24, MINT_FILL, MINT_STR, "l_{post} (1×)",MINT_TXT, size=13.6))
p('</g>')

# 3. LEFT BOTTOM: OCTOPUS SINGLE-OWNERSHIP MASK
p('<g id="Octopus Single-Ownership">')
p('<rect x="10" y="158" width="374" height="108" rx="11" fill="#F3FBF8" stroke="#A7D7C5" stroke-width="1.45"/>')
p(figma_rich_text(24,  178, "Single-Ownership (m_{l,t})", size=15.0, weight=700, fill=MINT_TXT, anchor="start"))
p(figma_rich_text(370, 178, "✓ Exact 1×", size=14.0, weight=700, fill=MINT_STR, anchor="end"))

p(mini_seg(22, 208, 90, 32, BLUE_FILL, BLUE_STR, "u_{0} (1×)", BLUE_TXT, size=14.2))
p(arrow(112, 220, 140, 197, APRI_STR, 1.55))
p(arrow(112, 224, 140, 224, VIOL_STR, 1.55, dash="3.4 2.2"))
p(arrow(112, 228, 140, 251, MINT_STR, 1.55))

p(mini_seg(142, 185,  104, 24, APRI_FILL, APRI_STR, "u_{A} (1×)",   APRI_TXT, size=13.6))
p(mini_seg(142, 212,  104, 24, VIOL_FILL, VIOL_STR, "M(A) (m=0)",   VIOL_TXT, size=13.4, dashed=True))
p(arrow(246, 224, 266, 224, MINT_STR, 1.5))
p(mini_seg(268, 212,  102, 24, MINT_HERO, MINT_STR, "u_{B} (1×)",   MINT_TXT, size=13.6))
p(mini_seg(142, 239,  104, 24, MINT_FILL, MINT_STR, "l_{pre} (1×)", MINT_TXT, size=13.6))
p(arrow(246, 251, 266, 251, MINT_STR, 1.5))
p(mini_seg(268, 239,  102, 24, MINT_FILL, MINT_STR, "l_{post} (1×)",MINT_TXT, size=13.6))
p('</g>')

# 4. RIGHT HALF: GROUP RLOO TREE CREDIT FLOW
p('<g id="Group RLOO Pipeline">')
p(mini_seg(400, 114, 78, 38, BLUE_FILL, BLUE_STR, "Prompt q", BLUE_TXT, size=14.6, rx=8))

p('<rect x="498" y="38" width="158" height="190" rx="11" fill="#FAFAFA" stroke="#CBD5E1" stroke-width="1.35" stroke-dasharray="4.5 2.8"/>')
p(figma_rich_text(577, 56, "G Rollout Trees", size=13.6, weight=700, fill=MUTED))

p(arrow(478, 128, 504, 84,  BLUE_STR, 1.6))
p(arrow(478, 133, 504, 133, BLUE_STR, 1.6))
p(arrow(478, 138, 504, 196, BLUE_STR, 1.6))

trees = [
    (67,  "#F3FBF8", MINT_STR, "τ_{1}", MINT_TXT, MINT_HERO, MINT_STR, "+1", MINT_TXT),
    (116, "#FDF8F8", BAD_STR,  "τ_{2}", BAD_TXT,  BAD_SOFT,  BAD_STR,  "−1", BAD_TXT),
    (179, "#F3FBF8", MINT_STR, "τ_{G}", MINT_TXT, MINT_HERO, MINT_STR, "+1", MINT_TXT),
]
for ty, tfill, tstr, tlbl, ttcol, rfill, rstr, rlbl, rtcol in trees:
    tcy = ty + 17.0
    p(f'<rect x="506" y="{ty}" width="66" height="34" rx="7.5" fill="{tfill}" stroke="{tstr}" stroke-width="1.4"/>')
    p(mini_fork_icon(521, tcy, tstr))
    p(figma_rich_text(548, tcy + 5.0, tlbl, size=14.6, weight=700, fill=ttcol))
    p(arrow(572, tcy, 588, tcy, tstr, 1.55))
    p(mini_seg(590, ty + 2, 58, 30, rfill, rstr, f"R={rlbl}", rtcol, size=13.6, rx=7))
    p(arrow(648, tcy, 672, 133 if ty == 116 else (108 if ty < 100 else 158), tstr, 1.6))

for dx_dot in (539.0, 619.0):
    for dy_dot in (158.0, 164.5, 171.0):
        p(f'<circle cx="{dx_dot:.1f}" cy="{dy_dot:.1f}" r="1.55" fill="{MUTED}"/>')

# RLOO Baseline Card + Balance-Scale Mascot
p('<rect x="674" y="38" width="156" height="190" rx="12" fill="#F3FBF8" stroke="#1B6B55" stroke-width="1.5"/>')
p(svg_mascot("octo_11_layer3_loo", 752, 86, 68, 64))
p(figma_rich_text(752, 141, "RLOO Baseline", size=15.4, weight=700, fill=MINT_TXT))
p(figma_rich_text(752, 166, "Â_{i} = R_{i} − mean(R_{≠i})", size=13.6, weight=700, fill=INK))
p('<rect x="688" y="182" width="128" height="31" rx="7.5" fill="#FFFFFF" stroke="#A7D7C5" stroke-width="1.3"/>')
p(figma_rich_text(752, 202.5, "✓ No critic", size=14.0, weight=700, fill=MINT_STR))

# Broadcast feedback arrow
p(f'<path d="M 752 228 L 752 246 L 391 246" fill="none" stroke="{MINT_STR}" stroke-width="2.05" stroke-linecap="round" stroke-linejoin="round"/>')
p(f'<polygon points="384,246 391.5,242.2 391.5,249.8" fill="{MINT_STR}"/>')
p(figma_rich_text(572, 267, "Broadcast Â_{i} to all 1× owned tokens", size=14.2, weight=700, fill=MINT_TXT))
p('</g>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved bold print-calibrated Figure 7 (RLOO): {PDF_OUT}")
