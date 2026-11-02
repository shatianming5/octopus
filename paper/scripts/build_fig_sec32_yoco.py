#!/usr/bin/env python3
"""
Section 3.2 Standalone Method Figure (Octopus Layer-Selective KV Mounting).
Print-calibrated on compact W=810 canvas so all typography renders at 8.0pt-9.8pt on the page.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_pdf import render_svg_to_pdf, figma_rich_text, svg_mascot

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig_sec32_yoco.svg"
PDF_OUT = ROOT / "figures/fig_sec32_yoco.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig_sec32_yoco_check.png")

W, H = 816, 368

INK        = "#1E293B"
MUTED      = "#64748B"
ARROW_COL  = "#334155"

MINT_HERO  = "#A7E3CE"
MINT_SOFT  = "#CBECE4"
MINT_TINT  = "#F2FBF8"
MINT_STR   = "#1B6B55"
MINT_TXT   = "#0B4F3F"

BLUE_SOFT  = "#DCEBFA"
BLUE_MLP   = "#E8F1FC"
BLUE_STR   = "#3B6E9C"
BLUE_TXT   = "#1E3A5F"

VIOL_SOFT  = "#EDE9FE"
VIOL_PILL  = "#DDD6FE"
VIOL_STR   = "#6D28D9"
VIOL_TXT   = "#4C1D95"

APRI_SOFT  = "#FFF3E4"
APRI_STR   = "#C86D2D"
APRI_TXT   = "#7C2D12"

BAD_SOFT   = "#FDE8E8"
BAD_STR    = "#B9382C"
BAD_TXT    = "#991B1B"


def arrow(x1: float, y1: float, x2: float, y2: float, col: str = ARROW_COL, sw: float = 1.85, dash: str = "") -> str:
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return ""
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    hl, hw = 7.4, 3.8
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


def block_bar(x: float, y: float, w: float, h: float, fill: str, stroke: str, label: str, txt_col: str = INK, size: float = 16.2) -> str:
    return (
        f'<g>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.45"/>'
        f'{figma_rich_text(x + w/2, y + h/2 + 5.6, label, size=size, weight=700, fill=txt_col)}'
        f'</g>'
    )


out: list[str] = []
p = out.append

p(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# Center Stack Coordinates
CX = 424.0
BW = 248.0
BX = CX - BW / 2.0          # 300..548
CONT_X = BX - 18.0          # 282..566
CONT_W = BW + 36.0          # 284

# 1. TOP OUTPUT BAR
p(arrow(CX, 58, CX, 42, MINT_STR, 1.9))
p(block_bar(BX + 20, 8, BW - 40, 32, MINT_HERO, MINT_STR, "Reader B Output", txt_col=MINT_TXT, size=16.0))

# 2. UPPER MACRO-BLOCK: 1x GATED FULL-ATTENTION
p(f'<rect x="{CONT_X:.1f}" y="58" width="{CONT_W:.1f}" height="108" rx="12" fill="{MINT_TINT}" stroke="#A7D7C5" stroke-width="1.4"/>')
p(block_bar(BX, 70,  BW, 35, BLUE_MLP,  BLUE_STR, "SwiGLU MLP", txt_col=BLUE_TXT))
p(block_bar(BX, 119, BW, 35, VIOL_SOFT, VIOL_STR, "Gated Full-Attention", txt_col=VIOL_TXT))
p(arrow(CX, 119, CX, 107, ARROW_COL, 1.85))

p(figma_rich_text(CONT_X + CONT_W + 14, 108, "1× Attention Layer", size=15.8, weight=700, fill=VIOL_TXT, anchor="start"))
p(figma_rich_text(CONT_X + CONT_W + 14, 129, "Associative KV lookup", size=13.8, weight=500, fill=MUTED, anchor="start"))

# 3. LOWER MACRO-BLOCK: 3x GATED DELTANET
p(f'<rect x="{CONT_X:.1f}" y="194" width="{CONT_W:.1f}" height="108" rx="12" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.4"/>')
p(block_bar(BX, 206, BW, 35, BLUE_MLP,  BLUE_STR, "SwiGLU MLP", txt_col=BLUE_TXT))
p(block_bar(BX, 255, BW, 35, MINT_SOFT, MINT_STR, "Gated DeltaNet", txt_col=MINT_TXT))
p(arrow(CX, 255, CX, 243, ARROW_COL, 1.85))

# Spine Q_B
p(arrow(CX, 206, CX, 156, ARROW_COL, 1.9))
p(figma_rich_text(CX + 18, 185, "Q_{B}", size=15.4, weight=700, fill=INK))

# Private recurrent state h_B self-loop
p(f'<path d="M {CX:.1f} 312 L {BX - 14:.1f} 312 L {BX - 14:.1f} 273" fill="none" stroke="{MINT_STR}" stroke-width="1.85" stroke-linecap="round"/>')
p(arrow(BX - 14, 273, BX - 2, 273, MINT_STR, 1.85))
p(figma_rich_text(BX - 17, 258, "h_{B}", size=15.4, weight=700, fill=MINT_TXT))

p(figma_rich_text(CONT_X + CONT_W + 14, 244, "3× Recurrent Layers", size=15.8, weight=700, fill=MINT_TXT, anchor="start"))
p(figma_rich_text(CONT_X + CONT_W + 14, 265, "Private working memory", size=13.8, weight=500, fill=MUTED, anchor="start"))

# Outer hybrid repeat bracket
BR_X = 748.0
p(f'<path d="M {BR_X:.1f} 64 L {BR_X + 7:.1f} 64 L {BR_X + 7:.1f} 296 L {BR_X:.1f} 296" fill="none" stroke="#94A3B8" stroke-width="1.65" stroke-linecap="round"/>')
p(figma_rich_text(BR_X + 13, 177, "×8", size=16.5, weight=700, fill=INK, anchor="start"))
p(figma_rich_text(BR_X + 13, 197, "Blocks", size=13.8, weight=600, fill=MUTED, anchor="start"))

# 4. BOTTOM INPUT BAR
p(arrow(CX, 326, CX, 292, ARROW_COL, 1.9))
p(block_bar(BX - 20, 328, BW + 40, 33, BLUE_SOFT, BLUE_STR, "Reader B Tokens (Monotonic Jump)", txt_col=BLUE_TXT, size=15.2))

# 5. LEFT SIDE: SIBLING KV MOUNT PILLAR + MASCOTS
p(svg_mascot("octo_10_layer2_mount", 32, 136, 58, 50))
p(f'<rect x="68" y="104" width="102" height="62" rx="9" fill="{APRI_SOFT}" stroke="{APRI_STR}" stroke-width="1.45"/>')
p(figma_rich_text(119, 131, "Sibling A", size=15.0, weight=700, fill=APRI_TXT))
p(figma_rich_text(119, 152, "Pinned KV", size=13.6, weight=600, fill=APRI_STR))

KV_X, KV_Y, KV_W, KV_H = 190.0, 66.0, 48.0, 140.0
p(arrow(170, 136, KV_X - 3, 136, VIOL_STR, 1.8, dash="4.0 2.5"))
p(f'<rect x="{KV_X:.1f}" y="{KV_Y:.1f}" width="{KV_W:.1f}" height="{KV_H:.1f}" rx="9" fill="{VIOL_PILL}" stroke="{VIOL_STR}" stroke-width="1.5"/>')
p(f'<text x="{KV_X + KV_W/2 + 5.2:.1f}" y="{KV_Y + KV_H/2:.1f}" font-family="Inter" font-size="14.8" font-weight="700" fill="{VIOL_TXT}" text-anchor="middle" transform="rotate(-90 {KV_X + KV_W/2 + 5.2:.1f} {KV_Y + KV_H/2:.1f})">0-Copy KV Mount</text>')

p(arrow(KV_X + KV_W + 2, 128, BX - 3, 128, VIOL_STR, 1.9))
p(arrow(KV_X + KV_W + 2, 145, BX - 3, 145, VIOL_STR, 1.9))
p(figma_rich_text(266, 119, "K_{A}", size=14.8, weight=700, fill=VIOL_TXT))
p(figma_rich_text(266, 163, "V_{A}", size=14.8, weight=700, fill=VIOL_TXT))

p(svg_mascot("octo_09_brainwash", 38, 272, 44, 52))
p(f'<rect x="68" y="251" width="160" height="44" rx="9" fill="{BAD_SOFT}" stroke="{BAD_STR}" stroke-width="1.4" stroke-dasharray="4.2 2.6"/>')
p(figma_rich_text(148, 270, "✗ Skipped by GDN", size=14.2, weight=700, fill=BAD_TXT))
p(figma_rich_text(148, 287, "No persona hijack", size=12.8, weight=600, fill=BAD_STR))

p(f'<line x1="230" y1="273" x2="262" y2="273" stroke="{BAD_STR}" stroke-width="1.75" stroke-dasharray="3.6 2.4"/>')
p(f'<line x1="262" y1="263" x2="262" y2="283" stroke="{BAD_STR}" stroke-width="2.5" stroke-linecap="round"/>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved print-calibrated Section 3.2 Figure: {PDF_OUT}")
