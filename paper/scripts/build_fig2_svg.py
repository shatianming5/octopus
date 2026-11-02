#!/usr/bin/env python3
"""
Builds v4 Editorial Edition of Figure 2 (`figures/fig2_cursor_mount.{svg,pdf}`):
Designed to match Figure 1 v5's exact visual DNA (Anthropic / Distill editorial illustration style):
- Canvas: 1126 x 424 (@ 6.50 in = 468 pt \textwidth -> 1 px = 0.4156 pt, matching Figure 1 v5's 1126px grid!)
- Restrained semantic palette matching Figure 1 v5:
  - Calm Warm Ivory (#FAF9F6 / #D5D0C7) & Vermeer Porcelain-Teal (#F7FBFA / #6BA396) panels
  - Recessed dashed wells (#EEF5F3 / #4E766D & #F1EFEA / #6B6760) instead of nested thick-bordered boxes
  - Warm Stone (#E5E2DC / #B0ABA1) neutrals so Vermeer Mint (#A8E2CD), Soft Lavender (#EAE4FA, Mounted KV),
    Warm Peach (#FBD6B6, Source Span A), and Slate Blue (#BDD9F2, Trunk/Cursor) pop effortlessly.
- True visual metaphors instead of stacked text banners:
  - (a) Virtual-to-Physical PagedAttention memory bank with converging zero-copy page pointers onto Span A (P2-P3)
  - (b) Architectural 3x GDN + 1x Full-Attn layer stack with Lavender `read` bypass bus & spliced `[K0 | KA | KB]` slots
  - (c) Aligned 1D RoPE coordinate timeline (`0 .. p_fork .. p_B .. e_A .. 201+`) with vertical timing guides & jump arc
"""
from pathlib import Path
from svg_to_pdf import format_rich_tspans, render_svg_to_pdf

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig2_cursor_mount.svg"
PDF_OUT = ROOT / "figures/fig2_cursor_mount.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig2_v4_editorial.png")

W, H = 1126, 424
FONT = "Inter"

# Exact Figure 1 v5 Editorial Palette
C_PANEL_A_BG, C_PANEL_A_STR = "#FAF9F6", "#D5D0C7"
C_PANEL_B_BG, C_PANEL_B_STR = "#F7FBFA", "#6BA396"
C_WELL_TEAL,  C_WELL_TDASH  = "#EEF5F3", "#5B8278"
C_WELL_STONE, C_WELL_SDASH  = "#F2EFE9", "#8C867E"

C_INK    = "#18181B"
C_SUBHDR = "#2E2C29"
C_MUTED  = "#524E49"
C_STROKE = "#524E49"

BLUE_FILL,  BLUE_STR,  BLUE_TXT  = "#BDD9F2", "#4B749F", "#142E49"
MINT_FILL,  MINT_STR,  MINT_TXT  = "#A8E2CD", "#2A7A60", "#0B382B"
MINT_SOFT                        = "#DCF2E9"
LAV_FILL,   LAV_STR,   LAV_TXT   = "#EAE4FA", "#7054B3", "#3F2A75"
PEACH_FILL, PEACH_STR, PEACH_TXT = "#FBD6B6", "#B56B36", "#47230A"
GREY_FILL,  GREY_STR,  GREY_TXT  = "#E5E2DC", "#B0ABA1", "#4F4B45"
ROSE_FILL,  ROSE_STR,  ROSE_TXT  = "#F9E5E5", "#B85454", "#6E2222"

out: list[str] = []


def p(s: str) -> None:
    out.append(s)


def text_el(x: float, y: float, content: str, size: float = 16.0, weight: int = 600,
            fill: str = C_INK, anchor: str = "middle", ls: str = "-0.22") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}">'
        f'{format_rich_tspans(content, size)}</text>'
    )


def pill_el(x: float, y: float, w: float, h: float, fill: str, stroke: str,
            label: str | None = None, txt_col: str = C_INK, rx: float = 5.5,
            sw: float = 1.4, size: float = 15.8, weight: int = 600, dash: str | None = None) -> str:
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d_attr}/>'
    ]
    if label:
        parts.append(text_el(x + w / 2, y + h / 2 + size * 0.35, label, size=size, weight=weight, fill=txt_col))
    return "\n".join(parts)


def chevron(x: float, y: float, direction: str, color: str = C_STROKE, sw: float = 1.5, s: float = 3.7) -> str:
    if direction == "right":
        d = f"M {x - s*1.15:.1f} {y - s:.1f} L {x:.1f} {y:.1f} L {x - s*1.15:.1f} {y + s:.1f}"
    elif direction == "left":
        d = f"M {x + s*1.15:.1f} {y - s:.1f} L {x:.1f} {y:.1f} L {x + s*1.15:.1f} {y + s:.1f}"
    elif direction == "down":
        d = f"M {x - s:.1f} {y - s*1.15:.1f} L {x:.1f} {y:.1f} L {x + s:.1f} {y - s*1.15:.1f}"
    else:
        d = f"M {x - s:.1f} {y + s*1.15:.1f} L {x:.1f} {y:.1f} L {x + s:.1f} {y + s*1.15:.1f}"
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'


def ortho_h(x1: float, y1: float, x2: float, y2: float, x_mid: float, r: float = 5.5) -> str:
    if abs(y2 - y1) < 1.0:
        return f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}"
    rr = min(r, abs(y2 - y1) / 2.0, abs(x_mid - x1), abs(x2 - x_mid))
    sy = 1.0 if y2 > y1 else -1.0
    sx1 = 1.0 if x_mid >= x1 else -1.0
    sx2 = 1.0 if x2 >= x_mid else -1.0
    return (
        f"M {x1:.1f} {y1:.1f} "
        f"L {x_mid - sx1*rr:.1f} {y1:.1f} "
        f"Q {x_mid:.1f} {y1:.1f} {x_mid:.1f} {y1 + sy*rr:.1f} "
        f"L {x_mid:.1f} {y2 - sy*rr:.1f} "
        f"Q {x_mid:.1f} {y2:.1f} {x_mid + sx2*rr:.1f} {y2:.1f} "
        f"L {x2:.1f} {y2:.1f}"
    )


def kv_page_card(x: float, y: float, w: float, h: float, fill: str, stroke: str,
                 title: str, txt_col: str, bar_col: str, dashed: bool = False, highlight: bool = False) -> str:
    """Mini physical KV memory page card with internal tensor-row stripes (matching Figure 1 mini-windows!)."""
    d_attr = ' stroke-dasharray="4.0 3.0"' if dashed else ""
    sw = 1.75 if highlight else 1.4
    parts = ['<g>']
    if highlight:
        parts.append(f'  <rect x="{x-2.5:.1f}" y="{y-2.5:.1f}" width="{w+5:.1f}" height="{h+5:.1f}" rx="7.5" fill="none" stroke="{LAV_STR}" stroke-width="1.35" stroke-dasharray="3.5 2.5"/>')
    parts.append(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5.5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d_attr}/>')
    # Subtle tensor slot stripes inside upper half of page card
    for k in range(2):
        by = y + 8 + k * 8.5
        parts.append(f'  <rect x="{x+7:.1f}" y="{by:.1f}" width="{w-14:.1f}" height="4.8" rx="2.2" fill="{bar_col}" opacity="0.65"/>')
    parts.append(f'  {text_el(x + w/2, y + h - 13, title, size=14.8, weight=700 if highlight else 600, fill=txt_col)}')
    parts.append('</g>')
    return "\n".join(parts)


p(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ==============================================================================
# PANEL (a): Zero-copy radix page lending (x=1, w=361)
# ==============================================================================
PA_X, PA_Y, PA_W, PA_H = 1, 2, 361, 420
p('<g id="Panel A - Zero-Copy Radix Page Lending">')
p(f'  <rect x="{PA_X}" y="{PA_Y}" width="{PA_W}" height="{PA_H}" rx="10" fill="{C_PANEL_B_BG}" stroke="{C_PANEL_B_STR}" stroke-width="1.45"/>')
p(f'  {text_el(PA_X + 16, 32, "(a) Zero-copy radix lending", size=20.8, weight=700, fill=C_INK, anchor="start", ls="-0.34")}')

p(f'  {text_el(PA_X + 16, 61, "Logical cursor DAG", size=17.0, weight=600, fill=C_SUBHDR, anchor="start")}')
p(f'  {text_el(PA_X + PA_W - 16, 61, "O(Δ) delta wire", size=14.6, weight=600, fill="#3D7065", anchor="end")}')

# Top recessed well: Logical Cursor DAG
p(f'  <rect x="16" y="72" width="330" height="124" rx="6.5" fill="{C_WELL_TEAL}" stroke="{C_WELL_TDASH}" stroke-width="1.4" stroke-dasharray="4.6 3.4"/>')

# Shared trunk cursor -> fork
p(pill_el(26, 117, 72, 34, BLUE_FILL, BLUE_STR, "c_{fork}", txt_col=BLUE_TXT, rx=6, sw=1.45, size=16.4, weight=700))

# Orthogonal fork branches
p(f'  <path d="{ortho_h(98, 134, 122, 101, 110, 5.5)}" fill="none" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'  {chevron(122, 101, "right", C_STROKE, 1.5, 3.5)}')
p(f'  <path d="{ortho_h(98, 134, 122, 167, 110, 5.5)}" fill="none" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'  {chevron(122, 167, "right", C_STROKE, 1.5, 3.5)}')

# Upper branch: Sibling A cursor span
p(pill_el(124, 84, 186, 33, PEACH_FILL, PEACH_STR, "c_{A} : Span [s_{A}, e_{A})", txt_col=PEACH_TXT, rx=5.5, sw=1.45, size=15.6, weight=700))

# Lower branch: Mounted placeholder -M_A (centered at x=157 -> aligned directly over P2!) -> Reader delta c'_B
p(pill_el(124, 151, 66, 33, LAV_FILL, LAV_STR, "−M_{A}", txt_col=LAV_TXT, rx=5.5, sw=1.5, size=15.8, weight=700, dash="4.2 2.8"))
p(f'  <line x1="190" y1="167.5" x2="213" y2="167.5" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'  {chevron(213, 167.5, "right", C_STROKE, 1.5, 3.5)}')
p(pill_el(215, 151, 97, 33, MINT_FILL, MINT_STR, "c\'_{B} (Δ_{B})", txt_col=MINT_TXT, rx=5.5, sw=1.45, size=15.6, weight=700))

# Lavender zero-copy `read` mount link between c_A and -M_A (exact Figure 1 `read` aesthetic!)
p(f'  <line x1="157" y1="117" x2="157" y2="149" stroke="{LAV_STR}" stroke-width="1.75" stroke-dasharray="4.0 2.8"/>')
p(f'  {chevron(157, 149, "down", LAV_STR, 1.65, 3.5)}')
p(f'  {text_el(167, 138, "read (0-copy)", size=14.6, weight=700, fill=LAV_TXT, anchor="start")}')

# Bottom section: Physical GPU KV Page Pool (header labels placed left & right of central pointer corridor!)
p(f'  <rect x="16" y="232" width="330" height="148" rx="6.5" fill="{C_WELL_TEAL}" stroke="#9BBAB1" stroke-width="1.35"/>')
p(f'  {text_el(29, 254, "Physical KV pool", size=14.8, weight=600, fill=C_SUBHDR, anchor="start")}')
p(f'  {text_el(334, 254, "1.94M tok", size=14.4, weight=600, fill="#3D7065", anchor="end")}')

# 6 Physical KV Page Cards inside pool (x0 = 29, w = 46, gap = 6.4 -> P2 center = 156.8, P3 center = 209.2)
PY = 272
PW, PH, PG = 46.0, 60.0, 6.4
pages = [
    ("P0", BLUE_FILL,  BLUE_STR,  BLUE_TXT,  "#4B749F", False, False),
    ("P1", BLUE_FILL,  BLUE_STR,  BLUE_TXT,  "#4B749F", False, False),
    ("P2", PEACH_FILL, PEACH_STR, PEACH_TXT, "#B56B36", False, True),
    ("P3", PEACH_FILL, PEACH_STR, PEACH_TXT, "#B56B36", False, True),
    ("P4", MINT_FILL,  MINT_STR,  MINT_TXT,  "#2A7A60", False, False),
    ("P5", GREY_FILL,  GREY_STR,  GREY_TXT,  "#B0ABA1", True,  False),
]
for idx, (pttl, pfc, pec, ptc, pbc, pdash, phl) in enumerate(pages):
    px = 29.0 + idx * (PW + PG)
    p(kv_page_card(px, PY, PW, PH, pfc, pec, pttl, ptc, pbc, dashed=pdash, highlight=phl))

# Group labels under physical pages
p(f'  {text_el(29 + PW + PG/2, 356, "Trunk (ref=2)", size=13.8, weight=600, fill=BLUE_TXT)}')
p(f'  {text_el(29 + 3*PW + 2.5*PG, 356, "Span A (1→2)", size=14.2, weight=700, fill=LAV_TXT)}')
p(f'  {text_el(29 + 4.5*PW + 4*PG, 356, "Δ_{B}", size=14.0, weight=700, fill=MINT_TXT)}')
p(f'  {text_el(29 + 5.5*PW + 5*PG, 356, "Free", size=13.8, weight=500, fill=GREY_TXT)}')

# Crisp Figure-1-style orthogonal rounded-corner pointers: BOTH c_A and -M_A map to Span A (P2, P3) with ZERO crossings!
X_P2 = 29.0 + 2 * (PW + PG) + PW / 2       # 156.8
X_P3 = 29.0 + 3 * (PW + PG) + PW / 2       # 209.2
# Pointer 1: straight vertical drop from -M_A bottom (156.8, 184) -> P2 top (156.8, 266)
p(f'  <line x1="{X_P2:.1f}" y1="184" x2="{X_P2:.1f}" y2="266" stroke="{LAV_STR}" stroke-width="1.7" stroke-dasharray="4.2 2.8"/>')
p(f'  {chevron(X_P2, 266, "down", LAV_STR, 1.65, 3.5)}')
# Pointer 2: orthogonal right-channel route from c_A right edge (310, 100.5) -> x=332 -> y=214 -> X_P3 (209.2) -> P3 top (266)
p(
    f'  <path d="M 310 100.5 L 326 100.5 Q 332 100.5 332 106.5 L 332 208 Q 332 214 326 214 '
    f'L {X_P3 + 6:.1f} 214 Q {X_P3:.1f} 214 {X_P3:.1f} 220 L {X_P3:.1f} 266" '
    f'fill="none" stroke="{PEACH_STR}" stroke-width="1.6"/>'
)
p(f'  {chevron(X_P3, 266, "down", PEACH_STR, 1.6, 3.5)}')

p(f'  {text_el(PA_X + PA_W/2, 403, "0 new pages allocated · freed when ref → 0", size=15.0, weight=600, fill=C_MUTED)}')
p('</g>')

# ==============================================================================
# PANEL (b): Layer-selective hybrid KV mount (x=374, w=378)
# ==============================================================================
PB_X, PB_Y, PB_W, PB_H = 374, 2, 378, 420
p('<g id="Panel B - Layer-Selective Hybrid Mount">')
p(f'  <rect x="{PB_X}" y="{PB_Y}" width="{PB_W}" height="{PB_H}" rx="10" fill="{C_PANEL_B_BG}" stroke="{C_PANEL_B_STR}" stroke-width="1.45"/>')
p(f'  {text_el(PB_X + 16, 32, "(b) Layer-selective hybrid mount", size=20.8, weight=700, fill=C_INK, anchor="start", ls="-0.34")}')

p(f'  {text_el(PB_X + 16, 61, "Hybrid decoder stack (×8 blocks)", size=17.0, weight=600, fill=C_SUBHDR, anchor="start")}')

# Left Source: Stacked Sibling A Physical KV Page Icon (x=392, y=102)
p('  <g id="Sibling A KV Stack">')
p(f'    <rect x="398" y="96" width="88" height="64" rx="6" fill="#F6C89E" stroke="{PEACH_STR}" stroke-width="1.3"/>')
p(f'    <rect x="391" y="103" width="88" height="64" rx="6" fill="{PEACH_FILL}" stroke="{PEACH_STR}" stroke-width="1.5"/>')
p(f'    <rect x="401" y="113" width="68" height="5" rx="2.5" fill="{PEACH_STR}" opacity="0.55"/>')
p(f'    <rect x="401" y="122" width="52" height="5" rx="2.5" fill="{PEACH_STR}" opacity="0.55"/>')
p(f'    {text_el(435, 146, "Sibling A", size=14.8, weight=700, fill=PEACH_TXT)}')
p(f'    {text_el(435, 161, "K_{A}, V_{A}", size=13.8, weight=600, fill=PEACH_TXT)}')
p('  </g>')

# Repeating Hybrid Block Card on Right (x=502, y=100, w=238, h=272)
p(f'  <rect x="502" y="100" width="238" height="272" rx="8" fill="#FFFFFF" stroke="#8AB3A8" stroke-width="1.45"/>')

# Reader B residual stream entry at top of stack
p(f'  {text_el(604, 86, "Reader B stream (h_{B}, Δ_{B})", size=14.8, weight=600, fill=BLUE_TXT)}')
p(f'  <line x1="604" y1="92" x2="604" y2="114" stroke="{C_STROKE}" stroke-width="1.55"/>')
p(f'  {chevron(604, 114, "down", C_STROKE, 1.5, 3.3)}')

# 3 Sleek Calm Recurrent GDN Layer Bars (y = 117, 158, 199)
gdn_layers = [
    (117, "GDN 4k+1", "h_{B} → h\'_{B}"),
    (158, "GDN 4k+2", "skip Span A"),
    (199, "GDN 4k+3", "private state"),
]
for idx, (gy, gl_l, gl_r) in enumerate(gdn_layers):
    p(f'  <rect x="515" y="{gy}" width="178" height="32" rx="5.5" fill="{GREY_FILL}" stroke="{GREY_STR}" stroke-width="1.35"/>')
    p(f'  {text_el(526, gy + 21, gl_l, size=14.4, weight=700, fill="#33312E", anchor="start")}')
    p(f'  {text_el(683, gy + 21, gl_r, size=13.2, weight=500, fill=GREY_TXT, anchor="end")}')
    if idx < 2:
        p(f'  <line x1="604" y1="{gy + 32}" x2="604" y2="{gy + 41}" stroke="{C_STROKE}" stroke-width="1.45"/>')

# Right bracket for 24 GDN layers
p(f'  <path d="M 699 120 L 705 120 L 705 228 L 699 228" fill="none" stroke="{GREY_STR}" stroke-width="1.35" stroke-linecap="round"/>')
p(f'  {text_el(722, 169, "24", size=14.6, weight=700, fill=GREY_TXT)}')
p(f'  {text_el(722, 185, "GDN", size=12.8, weight=600, fill=GREY_TXT)}')

# Residual link into Full-Attention layer
p(f'  <line x1="604" y1="231" x2="604" y2="251" stroke="{C_STROKE}" stroke-width="1.55"/>')
p(f'  {chevron(604, 251, "down", C_STROKE, 1.5, 3.3)}')

# 1 Featured Full-Attention Layer (y = 253, h = 84) with visual spliced KV capsules!
p(f'  <rect x="515" y="253" width="178" height="84" rx="6.5" fill="{MINT_SOFT}" stroke="{MINT_STR}" stroke-width="1.6"/>')
p(f'  {text_el(604, 276, "Full-Attn 4k+4", size=15.6, weight=700, fill=MINT_TXT)}')

# 3 Mini Spliced KV Capsules inside Full-Attention: [K_0] [K_A 0-copy] [K_B]
p(pill_el(523, 289, 40, 36, BLUE_FILL, BLUE_STR, "K_{0}", txt_col=BLUE_TXT, rx=4.5, sw=1.3, size=14.4, weight=700))
p(pill_el(568, 289, 72, 36, LAV_FILL,  LAV_STR,  "K_{A}", txt_col=LAV_TXT,  rx=4.5, sw=1.55, size=15.0, weight=700, dash="3.6 2.4"))
p(pill_el(645, 289, 40, 36, MINT_FILL, MINT_STR, "K_{B}", txt_col=MINT_TXT, rx=4.5, sw=1.3, size=14.4, weight=700))

# Right bracket for 8 Full-Attn layers
p(f'  <path d="M 699 256 L 705 256 L 705 334 L 699 334" fill="none" stroke="{MINT_STR}" stroke-width="1.4" stroke-linecap="round"/>')
p(f'  {text_el(722, 292, "8", size=14.8, weight=700, fill=MINT_TXT)}')
p(f'  {text_el(722, 308, "Attn", size=12.8, weight=600, fill=MINT_TXT)}')

# Output residual arrow
p(f'  <line x1="604" y1="337" x2="604" y2="361" stroke="{MINT_STR}" stroke-width="1.6"/>')
p(f'  {chevron(604, 361, "down", MINT_STR, 1.55, 3.4)}')

# Lavender `read` Bypass Bus from Sibling A KV (435, 167) -> bypassing 3 GDN bars -> directly into Full-Attn (512, 297)
p(f'  <path d="M 435 167 L 435 289 Q 435 297 443 297 L 512 297" fill="none" stroke="{LAV_STR}" stroke-width="2.0" stroke-dasharray="4.8 3.2"/>')
p(f'  {chevron(512, 297, "right", LAV_STR, 1.8, 4.0)}')

# Subtle bypass indicator next to GDN layers
p(pill_el(395, 198, 80, 26, "#FFFFFF", "#C4B5FD", "skip GDN", txt_col=LAV_TXT, rx=13, sw=1.3, size=13.8, weight=600))
p(f'  {text_el(475, 287, "0-copy", size=14.2, weight=700, fill=LAV_TXT)}')

p(f'  {text_el(PB_X + PB_W/2, 403, "97.3% retrieval acc. · 2.7% hijack (21×↓ vs. text)", size=15.0, weight=600, fill=MINT_TXT)}')
p('</g>')

# ==============================================================================
# PANEL (c): Monotonic RoPE position jump (x=764, w=361)
# ==============================================================================
PC_X, PC_Y, PC_W, PC_H = 764, 2, 361, 420
p('<g id="Panel C - Monotonic RoPE Position Jump">')
p(f'  <rect x="{PC_X}" y="{PC_Y}" width="{PC_W}" height="{PC_H}" rx="10" fill="{C_PANEL_A_BG}" stroke="{C_PANEL_A_STR}" stroke-width="1.45"/>')
p(f'  {text_el(PC_X + 16, 32, "(c) Monotonic RoPE jump", size=20.8, weight=700, fill=C_INK, anchor="start", ls="-0.34")}')

p(f'  {text_el(PC_X + 16, 61, "1D token position axis (e_{A} > p_{B})", size=17.0, weight=600, fill=C_SUBHDR, anchor="start")}')

# Shared Coordinate Ruler & Vertical Alignment Guides!
X0    = 784.0   # pos = 0
XFORK = 846.0   # p_fork = 80
XPB   = 916.0   # p_B = 120
XEA   = 1010.0  # e_A = 200
XEND  = 1104.0  # 201+

# Faint vertical timing guides through all 3 tracks
for gx in (X0, XFORK, XPB, XEA):
    p(f'  <line x1="{gx:.1f}" y1="90" x2="{gx:.1f}" y2="378" stroke="#DEDAD2" stroke-width="1.2" stroke-dasharray="3.2 2.8"/>')

# Top Ruler Axis
p(f'  <line x1="{X0:.1f}" y1="90" x2="{XEND:.1f}" y2="90" stroke="#78736B" stroke-width="1.4"/>')
p(f'  {chevron(XEND, 90, "right", "#78736B", 1.4, 3.2)}')
for gx, glbl in [(X0, "0"), (XFORK, "p_{fork}"), (XPB, "p_{B}"), (XEA, "e_{A}"), (XEND - 12, "201+")]:
    p(f'  <line x1="{gx:.1f}" y1="86" x2="{gx:.1f}" y2="94" stroke="#78736B" stroke-width="1.4"/>')
    p(f'  {text_el(gx, 81, glbl, size=13.8, weight=600, fill=C_MUTED)}')

# --- Track 1: Unshifted (Rule K1) (y = 104..186) ---
p(f'  {text_el(X0, 114, "Unshifted (Rule K1)", size=15.4, weight=700, fill=C_SUBHDR, anchor="start")}')
p(f'  {text_el(XEND, 114, "0.459 EM", size=14.6, weight=700, fill=ROSE_STR, anchor="end")}')

# c_B [0..p_B] + unshifted query q_B right after p_B
p(pill_el(X0, 123, XPB - X0, 24, BLUE_FILL, BLUE_STR, "c_{B}", txt_col=BLUE_TXT, rx=4.5, sw=1.35, size=14.4, weight=700))
p(pill_el(XPB + 2, 123, 38, 24, ROSE_FILL, ROSE_STR, "q_{B}", txt_col=ROSE_TXT, rx=4.5, sw=1.45, size=14.2, weight=700))
# Span A sitting at [p_fork .. e_A] -> extends PAST q_B into its "future"!
p(pill_el(XFORK, 152, XEA - XFORK, 24, PEACH_FILL, PEACH_STR, "Span A [s_{A}..e_{A})", txt_col=PEACH_TXT, rx=4.5, sw=1.35, size=13.8, weight=600))
# Inverted phase arrow starting cleanly from outer right edge of q_B (956) -> pointing FORWARD to e_A (1004)!
p(f'  <path d="M 956 135 L 999 135 Q 1004 135 1004 140 L 1004 150" fill="none" stroke="{ROSE_STR}" stroke-width="1.55"/>')
p(f'  {chevron(1004, 150, "down", ROSE_STR, 1.5, 3.2)}')
p(f'  {text_el(1015, 143, "q_{B} < k_{A}", size=13.8, weight=700, fill=ROSE_STR, anchor="start")}')

# --- Track 2: RoPE Copy (Rule K3) (y = 198..266) ---
p(f'  {text_el(X0, 206, "RoPE re-rotation (Rule K3)", size=15.4, weight=700, fill=C_SUBHDR, anchor="start")}')
p(f'  {text_el(XEND, 206, "O(N^{2}) copy", size=14.6, weight=600, fill=PEACH_STR, anchor="end")}')

p(pill_el(X0, 216, XPB - X0, 26, BLUE_FILL, BLUE_STR, "c_{B}", txt_col=BLUE_TXT, rx=4.5, sw=1.35, size=14.4, weight=700))
p(pill_el(XPB + 3, 216, 125, 26, GREY_FILL, PEACH_STR, "R_{Δθ} Copy(K_{A})", txt_col=PEACH_TXT, rx=4.5, sw=1.4, size=13.8, weight=600, dash="4.0 2.6"))
p(pill_el(XPB + 132, 216, 56, 26, GREY_FILL, GREY_STR, "Δ_{B}", txt_col=GREY_TXT, rx=4.5, sw=1.3, size=14.2, weight=600))

# --- Track 3: Octopus Monotonic Jump (Rule K2) (y = 264..376) ---
p(f'  <rect x="776" y="258" width="336" height="118" rx="7" fill="#EEF8F5" stroke="#76AF9F" stroke-width="1.4"/>')
p(f'  {text_el(X0, 279, "Octopus monotonic jump (K2)", size=15.6, weight=700, fill=MINT_TXT, anchor="start")}')
p(f'  {text_el(XEND, 279, "0.630 EM", size=14.8, weight=700, fill="#1D7A67", anchor="end")}')

# c_B at [0..p_B], 0-copy K_A untouched at native [p_fork..e_A], and Δ_B jumping monotonically to e_A+1 (201+)!
p(pill_el(X0, 305, XPB - X0, 25, BLUE_FILL, BLUE_STR, "c_{B}", txt_col=BLUE_TXT, rx=4.5, sw=1.35, size=14.4, weight=700))
p(pill_el(XFORK, 337, XEA - XFORK, 25, LAV_FILL, LAV_STR, "0-copy K_{A} (native pos)", txt_col=LAV_TXT, rx=4.5, sw=1.5, size=13.8, weight=700, dash="4.0 2.6"))
p(pill_el(XEA + 4, 305, XEND - XEA - 4, 25, MINT_FILL, MINT_STR, "Δ_{B} @ 201+", txt_col=MINT_TXT, rx=4.5, sw=1.55, size=14.2, weight=700))

# Signature Monotonic Jump Arc from p_B (916) vaulting cleanly to e_A+1 (1014)!
p(f'  <path d="M {XPB+3:.1f} 317 Q {(XPB+XEA)/2:.1f} 288 {XEA+2:.1f} 313" fill="none" stroke="#1D7A67" stroke-width="1.85"/>')
p(f'  {chevron(XEA + 2, 313, "right", "#1D7A67", 1.7, 3.6)}')
p(f'  {text_el((XPB+XEA)/2, 291, "jump", size=13.4, weight=700, fill="#1D7A67")}')

p(f'  {text_el(PC_X + PC_W/2, 403, "s(c\'_{B}) = max(p_{B}, e_{A} + 1) · causal q_{B} > k", size=15.0, weight=600, fill=MINT_TXT)}')
p('</g>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved Editorial v4 Figure 2: {PDF_OUT}")
