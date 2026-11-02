#!/usr/bin/env python3
"""
Section 3.3/3.4 Standalone Figure 6: Contiguous KV Gather, Rectangular Bottom-Right Causal
FlashAttention & Cross-Branch LoRA Gradient.
Print-calibrated on compact W=820 canvas (0.571 pt/px) with zero empty voids so every label
prints at 7.8pt-10.0pt clearly matching the LaTeX text.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_pdf import render_svg_to_pdf, figma_rich_text, svg_mascot

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig_sec33_gather.svg"
PDF_OUT = ROOT / "figures/fig_sec33_gather.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig_sec33_gather_check.png")

W, H = 820, 254

INK        = "#1E293B"
MUTED      = "#64748B"

BLUE_FILL  = "#CFE3FA"
BLUE_TILE  = "#9AC5F4"
BLUE_STR   = "#3B6E9C"
BLUE_TXT   = "#1E3A5F"

APRI_FILL  = "#FDE2C4"
APRI_STR   = "#C86D2D"
APRI_TXT   = "#7C2D12"

MINT_FILL  = "#CBECE4"
MINT_HERO  = "#9FE0C9"
MINT_TILE  = "#8FD9C1"
MINT_DIAG  = "#1B7A62"
MINT_STR   = "#1B6B55"
MINT_TXT   = "#0B4F3F"

VIOL_FILL  = "#EDE9FE"
VIOL_DEEP  = "#DDD6FE"
VIOL_TILE  = "#C4B5FD"
VIOL_STR   = "#6D28D9"
VIOL_TXT   = "#4C1D95"

TILE_MASK  = "#ECEAE4"
CORAL      = "#EE6C4D"
GRAD_STR   = "#C84B31"


def arrow(x1: float, y1: float, x2: float, y2: float, col: str, sw: float = 1.85, dash: str = "") -> str:
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


def pill(x: float, y: float, w: float, h: float, fill: str, stroke: str, lbl: str, tcol: str, size: float = 15.2) -> str:
    return (
        f'<g>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8.5" fill="{fill}" stroke="{stroke}" stroke-width="1.55"/>'
        f'{figma_rich_text(x + w/2, y + h/2 + 5.3, lbl, size=size, weight=700, fill=tcol)}'
        f'</g>'
    )


out: list[str] = []
p = out.append

p(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ==============================================================================
# 1. HEADERS (17.4px = 9.9pt printed!)
# ==============================================================================
p(figma_rich_text(134, 25, "Source LoRA Projection", size=17.4, weight=700, fill=INK))
p(figma_rich_text(534, 25, "Contiguous KV Gather & Bottom-Right Mask", size=17.4, weight=700, fill=INK))

# ==============================================================================
# 2. COMPACT PRODUCERS: SOURCE u_A -> LoRA theta -> K_mount, READER u_B2 -> Q
# ==============================================================================
p('<g id="Producers">')
# Source u_A (x=8..104) -> LoRA theta (x=122..206, exact center cx=164)
p(pill(8,   46, 96, 40, APRI_FILL, APRI_STR, "Source u_{A}", APRI_TXT, size=14.8))
p(arrow(104, 66, 120, 66, APRI_STR, 1.85))
p(pill(122, 46, 84, 40, VIOL_FILL, VIOL_STR, "LoRA θ",       VIOL_TXT, size=15.4))

# Forward 0-copy mount K_A, V_A -> K_mount
p(arrow(206, 66, 338, 66, VIOL_STR, 1.9, dash="4.4 2.6"))
p(figma_rich_text(272, 56, "K_{A}, V_{A}", size=14.4, weight=700, fill=VIOL_TXT))

# Reader u_B2 placed cleanly right of cx=164 (x=184..286, 20px clear of LoRA grad line!)
p(pill(184, 132, 102, 40, MINT_HERO, MINT_STR, "Reader u_{B2}", MINT_TXT, size=14.8))
p(arrow(286, 152, 310, 152, MINT_STR, 1.85))
p(figma_rich_text(298, 141, "Q", size=14.4, weight=700, fill=MINT_TXT))
p('</g>')

# ==============================================================================
# 3. CENTER: CONTIGUOUS GATHER HEADER + 3x8 BOTTOM-RIGHT CAUSAL MATRIX
# ==============================================================================
p('<g id="Gather Matrix">')
TW, TH = 32.0, 31.0
COL_X = [344.0, 380.0,   426.0, 462.0, 498.0,   544.0, 580.0, 616.0]
ROW_Y = [101.0, 136.5, 172.0]

# Gathered Key Header Bar [ K_mount || K_parent || K_self ]
p(pill(344, 46, 68,  40, VIOL_DEEP, VIOL_STR, "K_{mount}",  VIOL_TXT, size=14.6))
p('<line x1="417.5" y1="58" x2="417.5" y2="74" stroke="#64748B" stroke-width="1.7" stroke-linecap="round"/>')
p('<line x1="421.0" y1="58" x2="421.0" y2="74" stroke="#64748B" stroke-width="1.7" stroke-linecap="round"/>')
p(pill(426, 46, 104, 40, BLUE_FILL, BLUE_STR, "K_{parent}", BLUE_TXT, size=14.8))
p('<line x1="535.5" y1="58" x2="535.5" y2="74" stroke="#64748B" stroke-width="1.7" stroke-linecap="round"/>')
p('<line x1="539.0" y1="58" x2="539.0" y2="74" stroke="#64748B" stroke-width="1.7" stroke-linecap="round"/>')
p(pill(544, 46, 104, 40, MINT_FILL, MINT_STR, "K_{self}",   MINT_TXT, size=14.8))

# Matrix backdrop tray
p('<rect x="337" y="94" width="318" height="116" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3"/>')

# Left query row labels & bracket
p('<path d="M 318 104 L 313 104 L 313 200 L 318 200" fill="none" stroke="#1B6B55" stroke-width="1.6" stroke-linecap="round"/>')
for r_idx, ry in enumerate(ROW_Y):
    p(figma_rich_text(333, ry + 20.4, f"q_{{{r_idx}}}", size=14.0, weight=700, fill=MINT_TXT, anchor="end"))

for r_idx, ry in enumerate(ROW_Y):
    for c_idx, cx in enumerate(COL_X):
        if c_idx < 2:
            fill, stroke = VIOL_TILE, "#7C3AED"
        elif c_idx < 5:
            fill, stroke = BLUE_TILE, BLUE_STR
        else:
            s_col = c_idx - 5
            if s_col < r_idx:
                fill, stroke = MINT_TILE, MINT_STR
            elif s_col == r_idx:
                fill, stroke = (CORAL, "#B9382C") if r_idx == 2 else (MINT_DIAG, "#0F5132")
            else:
                fill, stroke = TILE_MASK, "#D6D3CD"
        sw = 1.25 if fill != TILE_MASK else 0.95
        p(f'<rect x="{cx:.1f}" y="{ry:.1f}" width="{TW:.1f}" height="{TH:.1f}" rx="4.5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

p(figma_rich_text(654, 231, "Bottom-right causal=True", size=14.2, weight=700, fill=MINT_TXT, anchor="end"))
p('</g>')

# ==============================================================================
# 4. COMPACT CROSS-BRANCH BACKWARD GRADIENT INTO LoRA theta
# ==============================================================================
p('<g id="Cross-Branch LoRA Gradient">')
p(f'<path d="M 378 210 L 378 226 L 164 226 L 164 94" fill="none" stroke="{GRAD_STR}" stroke-width="2.15" stroke-linecap="round" stroke-linejoin="round"/>')
p(f'<polygon points="164,87 160.0,94.8 168.0,94.8" fill="{GRAD_STR}"/>')
p(figma_rich_text(272, 245, "∂L_{B} / ∂K_{A}, V_{A} → updates LoRA θ", size=14.2, weight=700, fill=GRAD_STR))
p('</g>')

# ==============================================================================
# 5. SNUG RIGHT TAKEAWAY CARD + MONOCLE MATRIX OCTOPUS (NO WASTED WIDTH!)
# ==============================================================================
p('<g id="Right Takeaway">')
p('<rect x="668" y="46" width="144" height="196" rx="12" fill="#F3FBF8" stroke="#A7D7C5" stroke-width="1.45"/>')
p(svg_mascot("octo_14_mask_matrix", 740, 94, 76, 72))

p(figma_rich_text(740, 154, "✓ Native Flash", size=14.8, weight=700, fill=MINT_TXT))
p(figma_rich_text(740, 173, "No 2D mask", size=13.6, weight=600, fill=MUTED))
p(figma_rich_text(740, 203, "✓ LoRA KV Grad", size=14.6, weight=700, fill=VIOL_TXT))
p(figma_rich_text(740, 222, "On-policy mount", size=13.4, weight=600, fill=MUTED))
p('</g>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved bold print-calibrated Figure 6 (Gather): {PDF_OUT}")
