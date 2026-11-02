#!/usr/bin/env python3
"""
Section 3.1 Standalone Method Figure (Octopus-branded Paged Radix KV Pool & Cursor Lending).
Print-calibrated on W=880 canvas so all typography renders at 7.3pt-9.7pt on the printed page.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_pdf import render_svg_to_pdf, figma_rich_text, svg_mascot

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig_sec31_paged.svg"
PDF_OUT = ROOT / "figures/fig_sec31_paged.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig_sec31_paged_check.png")

W, H = 880, 326

# Unified Octopus Palette
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
VIOL_DEEP  = "#DDD6FE"
VIOL_STR   = "#6D28D9"
VIOL_TXT   = "#4C1D95"

STONE_FILL = "#F1EFEA"
STONE_STR  = "#A8A29E"
STONE_TXT  = "#78716C"


def arrow(x1: float, y1: float, x2: float, y2: float, col: str, sw: float = 1.65, dash: str = "") -> str:
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


def page_strip(x: float, y: float, cells: list[str], fill: str, stroke: str, txt_col: str,
               cw: float = 47.0, ch: float = 30.0, fsize: float = 13.4,
               dashed: bool = False, mount_ring: bool = False) -> str:
    tw = cw * len(cells)
    d_attr = ' stroke-dasharray="4.2 2.6"' if dashed else ""
    els = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{tw:.1f}" height="{ch:.1f}" rx="6.5" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.45"{d_attr}/>'
    ]
    for i in range(1, len(cells)):
        gx = x + i * cw
        els.append(
            f'<line x1="{gx:.1f}" y1="{y + 4:.1f}" x2="{gx:.1f}" y2="{y + ch - 4:.1f}" '
            f'stroke="{stroke}" stroke-opacity="0.34" stroke-width="1.1"{d_attr}/>'
        )
    for i, label in enumerate(cells):
        if label:
            els.append(figma_rich_text(x + (i + 0.5) * cw, y + ch / 2 + 4.8, label, size=fsize, weight=700, fill=txt_col))
    if mount_ring:
        els.append(
            f'<rect x="{x - 2:.1f}" y="{y - 2:.1f}" width="{tw + 4:.1f}" height="{ch + 4:.1f}" rx="8" '
            f'fill="none" stroke="{VIOL_STR}" stroke-width="1.8" stroke-dasharray="4.2 2.6"/>'
        )
    return "\n".join(els)


out: list[str] = []
p = out.append

p(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ==============================================================================
# 1. TOP HEADERS WITH OCTOPUS MASCOTS & INLINE LEGEND CHIPS
# ==============================================================================
p('<g id="Headers">')
p(svg_mascot("octo_fork", 34, 37, 48, 48))
p(figma_rich_text(64, 31, "Branch A (Worker)", size=17.2, weight=700, fill=INK, anchor="start"))
p(figma_rich_text(64, 51, "fork → clue → done", size=13.8, weight=500, fill=MUTED, anchor="start"))

p(figma_rich_text(440, 29, "Shared GPU Radix KV Pool", size=18.0, weight=700, fill=MINT_TXT))

chips = [
    (264, BLUE_FILL,  BLUE_STR,  BLUE_TXT,  "Shared (fork)"),
    (382, VIOL_DEEP,  VIOL_STR,  VIOL_TXT,  "0-copy (read)"),
    (504, STONE_FILL, STONE_STR, STONE_TXT, "Freed (rewind)"),
]
for cx0, cf, cs, ct, clbl in chips:
    p(f'<rect x="{cx0}" y="41" width="12" height="12" rx="3.2" fill="{cf}" stroke="{cs}" stroke-width="1.25"/>')
    p(figma_rich_text(cx0 + 16, 51.6, clbl, size=13.0, weight=600, fill=ct, anchor="start"))

p(svg_mascot("octo_radix", 846, 37, 50, 48))
p(figma_rich_text(815, 31, "Branch B (Reader)", size=17.2, weight=700, fill=INK, anchor="end"))
p(figma_rich_text(815, 51, "rewind → read(A) → done", size=13.8, weight=500, fill=VIOL_TXT, anchor="end"))
p('</g>')

# ==============================================================================
# 2. CENTER SHARED GPU RADIX KV POOL (x = 296..584, y = 68..314)
# ==============================================================================
p('<g id="GPU Radix KV Pool">')
POOL_X, POOL_Y, POOL_W, POOL_H = 296.0, 68.0, 288.0, 246.0
p(f'<rect x="{POOL_X}" y="{POOL_Y}" width="{POOL_W}" height="{POOL_H}" rx="12" fill="#F7FAF9" stroke="#CBD5E1" stroke-width="1.35"/>')

PY0 = 80.0
PDY = 38.0
PX_CELLS = 338.0  # 3 * 51px = 153px -> 338..491; badge 499..574

phys_pages = [
    ("P0", ["Query",  "Luzon", "fork"],   BLUE_FILL,  BLUE_STR,  BLUE_TXT,  False, "ref = 2", BLUE_FILL,  BLUE_TXT),
    ("P1", ["Detour", "Noise", "cut"],    STONE_FILL, STONE_STR, STONE_TXT, True,  "freed",   STONE_FILL, STONE_TXT),
    ("P2", ["Clue 1", "Boeing","747"],    APRI_FILL,  APRI_STR,  APRI_TXT,  False, "0-copy",  VIOL_DEEP,  VIOL_TXT),
    ("P3", ["Search", "24",    "read"],   MINT_FILL,  MINT_STR,  MINT_TXT,  False, "ref = 1", MINT_FILL,  MINT_TXT),
    ("P4", ["Clue 2", "Ash",   "done"],   APRI_FILL,  APRI_STR,  APRI_TXT,  False, "0-copy",  VIOL_DEEP,  VIOL_TXT),
    ("P5", ["Answer", "24",    "done"],   MINT_HERO,  MINT_STR,  MINT_TXT,  False, "ref = 1", MINT_HERO,  MINT_TXT),
]

def phys_cy(idx: int) -> float:
    return PY0 + idx * PDY + 15.0

for idx, (pid, cells, cf, cs, ct, is_dash, badge, bf, bt) in enumerate(phys_pages):
    py = PY0 + idx * PDY
    p(figma_rich_text(318, py + 20.0, pid, size=13.6, weight=700, fill=STONE_TXT if is_dash else INK))
    p(page_strip(PX_CELLS, py, cells, cf, cs, ct, cw=51.0, ch=30.0, fsize=12.4, dashed=is_dash))
    d_b = ' stroke-dasharray="3.5 2.2"' if is_dash else ""
    b_str = VIOL_STR if badge == "0-copy" else cs
    p(f'<rect x="498" y="{py + 3.0:.1f}" width="75" height="24" rx="12" fill="{bf}" stroke="{b_str}" stroke-width="1.25"{d_b}/>')
    p(figma_rich_text(535.5, py + 19.6, badge, size=12.8, weight=700, fill=bt))

p(svg_mascot("octo_17_rewind_cut", 600, phys_cy(1) + 4, 28, 28))
p('</g>')

# ==============================================================================
# 3. LEFT: BRANCH A WINDOW + CURSOR c_A TABLE
# ==============================================================================
p('<g id="Branch A">')
LA_X = 8.0
LA_Y0 = 114.0
LDY = 43.0
TA_X, TA_W = 172.0, 88.0

p(figma_rich_text(LA_X + 70, 96, "Logical Window", size=14.8, weight=700, fill=INK))
p(figma_rich_text(TA_X + TA_W / 2, 96, "Cursor c_{A}", size=14.8, weight=700, fill=INK))

p(f'<rect x="{TA_X}" y="{LA_Y0 - 6}" width="{TA_W}" height="{3 * LDY + 3}" rx="9" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3"/>')

rows_A = [
    (["Query",  "Luzon", "fork"], BLUE_FILL, BLUE_STR, BLUE_TXT, "P0", "fork", 0, BLUE_STR),
    (["Clue 1", "Boeing","747"],  APRI_FILL, APRI_STR, APRI_TXT, "P2", "pin",  2, APRI_STR),
    (["Clue 2", "Ash",   "done"], APRI_FILL, APRI_STR, APRI_TXT, "P4", "pin",  4, APRI_STR),
]

for r_i, (cells, cf, cs, ct, p_id, mode, target_p, acol) in enumerate(rows_A):
    ry = LA_Y0 + r_i * LDY
    rcy = ry + 15.0
    p(page_strip(LA_X, ry, cells, cf, cs, ct, cw=47.0, ch=30.0, fsize=12.2))
    p(arrow(LA_X + 143, rcy, TA_X + 4, rcy, "#64748B", 1.45))
    p(f'<rect x="{TA_X + 7}" y="{ry + 1.5}" width="{TA_W - 14}" height="27" rx="6" fill="#FFFFFF" stroke="{cs}" stroke-width="1.25"/>')
    p(figma_rich_text(TA_X + 24, ry + 19.8, p_id, size=13.4, weight=700, fill=ct))
    p(figma_rich_text(TA_X + 59, ry + 19.6, mode, size=12.6, weight=600, fill=MUTED))
    p(arrow(TA_X + TA_W + 2, rcy, POOL_X - 3, phys_cy(target_p), acol, 1.7))
p('</g>')

# ==============================================================================
# 4. RIGHT: CURSOR c'_B TABLE + BRANCH B WINDOW
# ==============================================================================
p('<g id="Branch B">')
LB_X = 731.0
LB_Y0 = 98.0
TB_X, TB_W = 618.0, 92.0

p(figma_rich_text(TB_X + TB_W / 2, 81, "Cursor c'_{B}", size=14.8, weight=700, fill=INK))
p(figma_rich_text(LB_X + 70, 81, "Logical Window", size=14.8, weight=700, fill=INK))

p(f'<rect x="{TB_X}" y="{LB_Y0 - 6}" width="{TB_W}" height="{4 * LDY + 3}" rx="9" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3"/>')

rows_B = [
    (["Query",  "Luzon", "fork"],   BLUE_FILL, BLUE_STR, BLUE_TXT, False, "P0",   "fork", [0],    BLUE_STR, False),
    (["Search", "24",    "read"],   MINT_FILL, MINT_STR, MINT_TXT, False, "P3",   "own",  [3],    MINT_STR, False),
    (["Clue 1", "Boeing","Clue 2"], APRI_FILL, APRI_STR, APRI_TXT, True,  "P2,4", "read", [2, 4], VIOL_STR, True),
    (["Answer", "24",    "done"],   MINT_HERO, MINT_STR, MINT_TXT, False, "P5",   "own",  [5],    MINT_STR, False),
]

for r_i, (cells, cf, cs, ct, m_ring, p_id, mode, targets, acol, is_mnt) in enumerate(rows_B):
    ry = LB_Y0 + r_i * LDY
    rcy = ry + 15.0
    p(page_strip(LB_X, ry, cells, cf, cs, ct, cw=47.0, ch=30.0, fsize=12.2, mount_ring=m_ring))
    p(arrow(LB_X - 3, rcy, TB_X + TB_W - 4, rcy, VIOL_STR if is_mnt else "#64748B", 1.45))
    efill = VIOL_FILL if is_mnt else "#FFFFFF"
    estr  = VIOL_STR if is_mnt else cs
    etxt  = VIOL_TXT if is_mnt else ct
    p(f'<rect x="{TB_X + 7}" y="{ry + 1.5}" width="{TB_W - 14}" height="27" rx="6" fill="{efill}" stroke="{estr}" stroke-width="1.35"/>')
    p(figma_rich_text(TB_X + 27, ry + 19.8, p_id, size=13.0, weight=700, fill=etxt))
    p(figma_rich_text(TB_X + 66, ry + 19.6, mode, size=12.4, weight=700 if is_mnt else 600, fill=VIOL_TXT if is_mnt else MUTED))
    for t_idx in targets:
        p(arrow(TB_X - 2, rcy, POOL_X + POOL_W + 3, phys_cy(t_idx), acol, 1.85 if is_mnt else 1.65, dash="4.4 2.6" if is_mnt else ""))

p(arrow(TB_X - 2, LB_Y0 + 18, POOL_X + POOL_W + 3, phys_cy(1) - 6, STONE_STR, 1.45, dash="3.8 2.4"))
p('</g>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved print-calibrated Section 3.1 Figure: {PDF_OUT}")
