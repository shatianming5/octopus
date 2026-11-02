#!/usr/bin/env python3
"""
Builds v4 Editorial Edition of Figure 3 (`figures/fig3_dag_grpo.{svg,pdf}`):
Designed to match Figure 1 v5's exact visual DNA (Anthropic / Distill editorial illustration style):
- Canvas: 1126 x 456 (@ 6.50 in = 468 pt \textwidth -> 1 px = 0.4156 pt, matching Figure 1 v5's 1126px grid!)
- Eliminates double-nested white boxes, rainbow box soup, and 30x checkmark/star matrix clutter:
  - (a) Recessed DAG well + recessed 1D GPU token tray with graceful Bezier GDN state hooks & LoRA autograd curve
  - (b) Perfectly square (1:1) 8x8 geometric block-sparse attention matrix + FlashAttention `[K_mount | K_parent | K_self]` bar
  - (c) Horizontal leaf timeline tracks (solid owned capsules vs. hollow dashed m=0 capsules) + mini branching-tree RLOO credit
"""
from pathlib import Path
from svg_to_pdf import format_rich_tspans, render_svg_to_pdf

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig3_dag_grpo.svg"
PDF_OUT = ROOT / "figures/fig3_dag_grpo.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig3_v4_editorial.png")

W, H = 1126, 456
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
LAV_DEEP                         = "#DCD3FA"
PEACH_FILL, PEACH_STR, PEACH_TXT = "#FBD6B6", "#B56B36", "#47230A"
GREY_FILL,  GREY_STR,  GREY_TXT  = "#E5E2DC", "#B0ABA1", "#4F4B45"
ROSE_FILL,  ROSE_STR,  ROSE_TXT  = "#F9E5E5", "#B85454", "#6E2222"

out: list[str] = []


def p(s: str) -> None:
    out.append(s)


def text_el(x: float, y: float, content: str, size: float = 15.6, weight: int = 600,
            fill: str = C_INK, anchor: str = "middle", ls: str = "-0.22") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}">'
        f'{format_rich_tspans(content, size)}</text>'
    )


def pill_el(x: float, y: float, w: float, h: float, fill: str, stroke: str,
            label: str | None = None, sublabel: str | None = None,
            txt_col: str = C_INK, rx: float = 5.5,
            sw: float = 1.4, size: float = 15.2, sub_size: float = 13.4,
            weight: int = 700, dash: str | None = None) -> str:
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d_attr}/>'
    ]
    if label and sublabel:
        parts.append(text_el(x + w / 2, y + h * 0.38 + size * 0.33, label, size=size, weight=weight, fill=txt_col))
        parts.append(text_el(x + w / 2, y + h * 0.77 + sub_size * 0.28, sublabel, size=sub_size, weight=600, fill=txt_col))
    elif label:
        parts.append(text_el(x + w / 2, y + h / 2 + size * 0.35, label, size=size, weight=weight, fill=txt_col))
    return "\n".join(parts)


def chevron(x: float, y: float, direction: str, color: str = C_STROKE, sw: float = 1.5, s: float = 3.6) -> str:
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


def mini_tree_icon(x: float, y: float, col: str) -> str:
    """Tiny 3-node fork glyph inside rollout tree cards."""
    return (
        f'<g>'
        f'  <line x1="{x:.1f}" y1="{y:.1f}" x2="{x+9:.1f}" y2="{y-5:.1f}" stroke="{col}" stroke-width="1.35"/>'
        f'  <line x1="{x:.1f}" y1="{y:.1f}" x2="{x+9:.1f}" y2="{y+5:.1f}" stroke="{col}" stroke-width="1.35"/>'
        f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="{col}"/>'
        f'  <circle cx="{x+9:.1f}" cy="{y-5:.1f}" r="2.1" fill="{col}"/>'
        f'  <circle cx="{x+9:.1f}" cy="{y+5:.1f}" r="2.1" fill="{col}"/>'
        f'</g>'
    )


p(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ==============================================================================
# PANEL (a): Tree-to-1D packed DAG (x=1, w=375)
# ==============================================================================
PA_X, PA_Y, PA_W, PA_H = 1, 2, 375, 452
p('<g id="Panel A - Tree to 1D Packed DAG">')
p(f'  <rect x="{PA_X}" y="{PA_Y}" width="{PA_W}" height="{PA_H}" rx="10" fill="{C_PANEL_A_BG}" stroke="{C_PANEL_A_STR}" stroke-width="1.45"/>')
p(f'  {text_el(PA_X + 16, 32, "(a) Tree-to-1D packed DAG", size=20.8, weight=700, fill=C_INK, anchor="start", ls="-0.34")}')

p(f'  {text_el(PA_X + 16, 60, "Episode execution DAG", size=16.8, weight=600, fill=C_SUBHDR, anchor="start")}')

# Recessed DAG Well (exact Figure 1 DAG well aesthetic!)
p(f'  <rect x="16" y="71" width="345" height="142" rx="6.5" fill="{C_WELL_STONE}" stroke="{C_WELL_SDASH}" stroke-width="1.35" stroke-dasharray="4.6 3.4"/>')

# DAG Nodes
p(pill_el(28,  125, 60,  34, BLUE_FILL,  BLUE_STR,  "u_{0}",          txt_col=BLUE_TXT,  rx=6,   sw=1.45, size=16.4))
p(pill_el(118, 84,  106, 32, PEACH_FILL, PEACH_STR, "u_{A} (Source)", txt_col=PEACH_TXT, rx=5.5, sw=1.45, size=15.0))
p(pill_el(118, 126, 64,  32, MINT_SOFT,  MINT_STR,  "u_{B1}",         txt_col=MINT_TXT,  rx=5.5, sw=1.4,  size=15.2))
p(pill_el(244, 126, 104, 32, MINT_FILL,  MINT_STR,  "u_{B2} (Read)",  txt_col=MINT_TXT,  rx=5.5, sw=1.5,  size=15.0))
p(pill_el(118, 168, 64,  32, GREY_FILL,  GREY_STR,  "l_{pre}",        txt_col=GREY_TXT,  rx=5.5, sw=1.35, size=15.0))
p(pill_el(244, 168, 104, 32, MINT_SOFT,  MINT_STR,  "l_{post}",       txt_col=MINT_TXT,  rx=5.5, sw=1.4,  size=15.0))

# Orthogonal Fork & Continuation Wires
p(f'  <path d="{ortho_h(88, 142, 116, 100, 102, 5.5)}" fill="none" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'  {chevron(116, 100, "right", C_STROKE, 1.5, 3.4)}')
p(f'  <line x1="88" y1="142" x2="116" y2="142" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'  {chevron(116, 142, "right", C_STROKE, 1.5, 3.4)}')
p(f'  <path d="{ortho_h(88, 142, 116, 184, 102, 5.5)}" fill="none" stroke="#8C867E" stroke-width="1.45" stroke-dasharray="3.6 2.6"/>')
p(f'  {chevron(116, 184, "right", "#8C867E", 1.45, 3.4)}')

p(f'  <line x1="182" y1="142" x2="242" y2="142" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'  {chevron(242, 142, "right", C_STROKE, 1.5, 3.4)}')

p(f'  <line x1="182" y1="184" x2="242" y2="184" stroke="{MINT_STR}" stroke-width="1.5"/>')
p(f'  {chevron(242, 184, "right", MINT_STR, 1.5, 3.4)}')
p(f'  {text_el(213, 176, "rewind", size=13.2, weight=700, fill="#2A7A60")}')

# Lavender dashed `read` mount edge from u_A to u_B2
p(f'  <path d="M 224 100 L 290 100 Q 296 100 296 106 L 296 124" fill="none" stroke="{LAV_STR}" stroke-width="1.8" stroke-dasharray="4.2 2.8"/>')
p(f'  {chevron(296, 124, "down", LAV_STR, 1.65, 3.5)}')
p(f'  {text_el(286, 93, "mount K_{A}", size=14.2, weight=700, fill=LAV_TXT)}')

# Clean Topological Pack Transition
p(f'  <line x1="56" y1="220" x2="56" y2="244" stroke="{C_STROKE}" stroke-width="1.6"/>')
p(f'  {chevron(56, 244, "down", C_STROKE, 1.55, 3.6)}')
p(f'  {text_el(70, 237, "1D packed GPU pass (1× prefix)", size=15.4, weight=600, fill=C_SUBHDR, anchor="start")}')

# GDN Boundary Hook Arcs above 1D Tape (non-adjacent parent state routing: u0->uB1, u0->lpost, uB1->uB2 skipping uA!)
p(f'  {text_el(188, 269, "GDN boundary hooks h_{end}(p(u))  (skip u_{A})", size=14.2, weight=600, fill=BLUE_TXT)}')
# Hook 1: u_0 (51) -> u_B1 (213)
p(f'  <path d="M 52 313 Q 132 287 212 312" fill="none" stroke="{BLUE_STR}" stroke-width="1.45"/>')
p(f'  {chevron(212, 312, "down", BLUE_STR, 1.4, 3.0)}')
# Hook 2: u_0 (43) -> l_post (273) [rewind restores h_end(u_0)]
p(f'  <path d="M 43 313 Q 158 273 273 312" fill="none" stroke="{MINT_STR}" stroke-width="1.4" stroke-dasharray="3.6 2.6"/>')
p(f'  {chevron(273, 312, "down", MINT_STR, 1.4, 3.0)}')
# Hook 3: u_B1 (222) -> u_B2 (330) [continues h_end(u_B1), skips l_post & mounted u_A]
p(f'  <path d="M 222 313 Q 276 291 330 312" fill="none" stroke="#2A7A60" stroke-width="1.45"/>')
p(f'  {chevron(330, 312, "down", "#2A7A60", 1.4, 3.0)}')

# Recessed 1D GPU Token Tray holding 6 sleek capsules (x=16, y=314, w=345, h=52)
p(f'  <rect x="16" y="314" width="345" height="52" rx="7" fill="#EAE6DE" stroke="#C9C3B8" stroke-width="1.25"/>')
tape = [
    ("u_{0}",    "1×",    BLUE_FILL,  BLUE_STR,  BLUE_TXT),
    ("u_{A}",    "Src",   PEACH_FILL, PEACH_STR, PEACH_TXT),
    ("l_{pre}",  "Prune", GREY_FILL,  GREY_STR,  GREY_TXT),
    ("u_{B1}",   "Fork",  MINT_SOFT,  MINT_STR,  MINT_TXT),
    ("l_{post}", "Rew",   MINT_SOFT,  MINT_STR,  MINT_TXT),
    ("u_{B2}",   "Read",  MINT_FILL,  MINT_STR,  MINT_TXT),
]
for idx, (t_top, t_bot, t_fc, t_ec, t_tc) in enumerate(tape):
    tx = 21.0 + idx * 56.4
    p(pill_el(tx, 319, 52.8, 42, t_fc, t_ec, t_top, t_bot, txt_col=t_tc, rx=4.5, sw=1.35, size=14.8, sub_size=12.8))

# Cross-Branch LoRA Autograd Curve below Tape (from u_B2 center x=329.4 back to u_A center x=103.8)
p(f'  <path d="M 329 368 Q 216 402 104 369" fill="none" stroke="{LAV_STR}" stroke-width="1.95"/>')
p(f'  {chevron(104, 369, "up", LAV_STR, 1.75, 3.7)}')
p(f'  {text_el(216, 405, "∂L_{B2} / ∂K_{A}, V_{A} → LoRA θ", size=14.8, weight=700, fill=LAV_TXT)}')

p(f'  {text_el(PA_X + PA_W/2, 434, "Matches SGLang rollout (|Δlog π_{θ}| = 0.0038)", size=14.6, weight=600, fill=C_MUTED)}')
p('</g>')

# ==============================================================================
# PANEL (b): Tree & mount attention mask (x=388, w=350)
# ==============================================================================
PB_X, PB_Y, PB_W, PB_H = 388, 2, 350, 452
p('<g id="Panel B - 2D Attention Mask">')
p(f'  <rect x="{PB_X}" y="{PB_Y}" width="{PB_W}" height="{PB_H}" rx="10" fill="{C_PANEL_B_BG}" stroke="{C_PANEL_B_STR}" stroke-width="1.45"/>')
p(f'  {text_el(PB_X + 16, 32, "(b) Tree & mount mask", size=20.8, weight=700, fill=C_INK, anchor="start", ls="-0.34")}')
p(f'  {text_el(PB_X + 16, 60, "Block-sparse visibility (ℓ : L_{mount})", size=16.6, weight=600, fill=C_SUBHDR, anchor="start")}')

# Perfectly Square 8x8 Matrix (28.0 x 28.0 px squares -> 224 x 224 px!)
tokens = ["t_{0}", "t_{1}", "a_{0}", "a_{1}", "b_{0}", "b_{1}", "b_{2}", "c_{0}"]
GX0, GY0 = 491.0, 96.0
CS = 28.0

# Subtle white tray behind 8x8 matrix
p(f'  <rect x="{GX0 - 3:.1f}" y="{GY0 - 3:.1f}" width="{8*CS + 6:.1f}" height="{8*CS + 6:.1f}" rx="6" fill="#FFFFFF" stroke="#C5DBD4" stroke-width="1.3"/>')

for j, tok in enumerate(tokens):
    p(text_el(GX0 + (j + 0.5) * CS, GY0 - 10, tok, size=14.6, weight=600, fill=C_SUBHDR))

# Sleek Right-Aligned Row Labels + Vertical Color Accent Bars (zero chunky row boxes!)
row_groups = [
    ("Trunk u_{0}",  [0, 1], BLUE_STR,  BLUE_TXT),
    ("Source u_{A}", [2, 3], PEACH_STR, PEACH_TXT),
    ("Node u_{B1}",  [4],    MINT_STR,  MINT_TXT),
    ("Read u_{B2}",  [5, 6], LAV_STR,   LAV_TXT),
    ("Trans. u_{C}", [7],    MINT_STR,  MINT_TXT),
]
for glbl, r_idx, bar_col, txt_col in row_groups:
    yt = GY0 + r_idx[0] * CS + 3.0
    yb = GY0 + (r_idx[-1] + 1) * CS - 3.0
    ym = 0.5 * (yt + yb)
    p(f'  <rect x="{GX0 - 11:.1f}" y="{yt:.1f}" width="3.8" height="{yb - yt:.1f}" rx="1.9" fill="{bar_col}"/>')
    p(text_el(GX0 - 17, ym + 5.0, glbl, size=14.4, weight=600, fill=txt_col, anchor="end"))

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

# Pure geometric color-block cells (NO noisy checkmarks/stars spam!)
for r in range(8):
    for c in range(8):
        st = mask_grid[r][c]
        xc = GX0 + c * CS + 1.5
        yc = GY0 + r * CS + 1.5
        s_in = CS - 3.0
        cx_m, cy_m = xc + s_in / 2, yc + s_in / 2
        if st == 0:
            p(f'  <rect x="{xc:.1f}" y="{yc:.1f}" width="{s_in:.1f}" height="{s_in:.1f}" rx="3.2" fill="#F6F9F8" stroke="#E3ECE9" stroke-width="0.9"/>')
        elif st == 1:
            p(f'  <rect x="{xc:.1f}" y="{yc:.1f}" width="{s_in:.1f}" height="{s_in:.1f}" rx="3.5" fill="{BLUE_FILL}" stroke="{BLUE_STR}" stroke-width="1.15"/>')
        elif st in (2, 3, 4, 7):
            p(f'  <rect x="{xc:.1f}" y="{yc:.1f}" width="{s_in:.1f}" height="{s_in:.1f}" rx="3.5" fill="{MINT_FILL}" stroke="{MINT_STR}" stroke-width="1.15"/>')
        elif st == 5:
            # Direct 0-Copy Mount A (Lavender + crisp center dot)
            p(f'  <rect x="{xc:.1f}" y="{yc:.1f}" width="{s_in:.1f}" height="{s_in:.1f}" rx="3.5" fill="{LAV_DEEP}" stroke="{LAV_STR}" stroke-width="1.5"/>')
            p(f'  <circle cx="{cx_m:.1f}" cy="{cy_m:.1f}" r="3.1" fill="{LAV_TXT}"/>')
        elif st == 6:
            # Transitive Mount (Warm Peach + crisp hollow ring)
            p(f'  <rect x="{xc:.1f}" y="{yc:.1f}" width="{s_in:.1f}" height="{s_in:.1f}" rx="3.5" fill="{PEACH_FILL}" stroke="{PEACH_STR}" stroke-width="1.5"/>')
            p(f'  <circle cx="{cx_m:.1f}" cy="{cy_m:.1f}" r="3.2" fill="#FFFFFF" stroke="{PEACH_STR}" stroke-width="1.6"/>')

# Eq. (3) Bottom-Right FlashAttention Gather Strip (y = 338)
p(f'  {text_el(404, 355, "Gather:", size=14.0, weight=600, fill=C_MUTED, anchor="start")}')
p(pill_el(460, 337, 82, 26, LAV_FILL,  LAV_STR,  "K_{mount}",  txt_col=LAV_TXT,  rx=4.5, sw=1.35, size=13.8))
p(pill_el(546, 337, 82, 26, BLUE_FILL, BLUE_STR, "K_{parent}", txt_col=BLUE_TXT, rx=4.5, sw=1.3,  size=13.8))
# Pure SVG causal triangle inside K_self pill (avoids U+25E2 Type 3 font fallback!)
p(pill_el(632, 337, 90, 26, MINT_FILL, MINT_STR, None, txt_col=MINT_TXT, rx=4.5, sw=1.35, size=13.8))
p(f'  {text_el(669, 354.8, "K_{self}", size=13.8, weight=700, fill=MINT_TXT)}')
p(f'  <polygon points="698,356 710,356 710,344" fill="{MINT_TXT}"/>')

# Clean 2x2 Swatch Legend (y = 386..434, aligned to Panel A/C bottom baseline)
leg = [
    (404, 386, BLUE_FILL,  BLUE_STR,  "none",   "Shared trunk"),
    (568, 386, MINT_FILL,  MINT_STR,  "none",   "Self-causal"),
    (404, 416, LAV_DEEP,   LAV_STR,   "dot",    "0-copy mount"),
    (568, 416, PEACH_FILL, PEACH_STR, "ring",   "Transitive mount"),
]
for lx, ly, lfc, lec, lmk, ltxt in leg:
    p(f'  <rect x="{lx}" y="{ly}" width="17" height="17" rx="3.5" fill="{lfc}" stroke="{lec}" stroke-width="1.35"/>')
    if lmk == "dot":
        p(f'  <circle cx="{lx+8.5:.1f}" cy="{ly+8.5:.1f}" r="2.6" fill="{LAV_TXT}"/>')
    elif lmk == "ring":
        p(f'  <circle cx="{lx+8.5:.1f}" cy="{ly+8.5:.1f}" r="2.7" fill="#FFFFFF" stroke="{PEACH_STR}" stroke-width="1.4"/>')
    p(text_el(lx + 24, ly + 13.8, ltxt, size=14.4, weight=600, fill=C_SUBHDR, anchor="start"))
p('</g>')

# ==============================================================================
# PANEL (c): Single-ownership & RLOO (x=750, w=375)
# ==============================================================================
PC_X, PC_Y, PC_W, PC_H = 750, 2, 375, 452
p('<g id="Panel C - Single-Ownership & RLOO">')
p(f'  <rect x="{PC_X}" y="{PC_Y}" width="{PC_W}" height="{PC_H}" rx="10" fill="{C_PANEL_B_BG}" stroke="{C_PANEL_B_STR}" stroke-width="1.45"/>')
p(f'  {text_el(PC_X + 16, 32, "(c) Single-ownership & RLOO", size=20.8, weight=700, fill=C_INK, anchor="start", ls="-0.34")}')

p(f'  {text_el(PC_X + 16, 60, "Single-ownership loss mask (m_{l,t})", size=16.6, weight=600, fill=C_SUBHDR, anchor="start")}')

# Upper Recessed Well: Leaf Timeline Tracks
p(f'  <rect x="765" y="71" width="345" height="198" rx="6.5" fill="{C_WELL_TEAL}" stroke="#9BBAB1" stroke-width="1.35"/>')

cols = ["u_{0}", "u_{A}", "M(A)", "u_{B}", "l_{pre}", "l_{post}"]
CX0 = 852.0
COL_W = 41.0
for ci, chdr in enumerate(cols):
    gx = CX0 + (ci + 0.5) * COL_W
    p(text_el(gx, 92, chdr, size=14.4, weight=700, fill=C_SUBHDR))
    p(f'  <line x1="{gx:.1f}" y1="99" x2="{gx:.1f}" y2="227" stroke="#DCE7E3" stroke-width="1.1"/>')

leaf_rows = [
    ("Leaf A",        [1,  1, -1, -1, -1, -1], PEACH_FILL, PEACH_STR, PEACH_TXT),
    ("Leaf B",        [0, -1,  0,  1, -1, -1], MINT_FILL,  MINT_STR,  MINT_TXT),
    ("Leaf l_{pre}",  [0, -1, -1, -1,  1, -1], GREY_FILL,  "#78716C", "#292524"),
    ("Leaf l_{post}", [0, -1, -1, -1,  0,  1], MINT_FILL,  MINT_STR,  MINT_TXT),
]
for ri, (rlbl, rvals, rfc, rec, rtc) in enumerate(leaf_rows):
    ry = 115.0 + ri * 30.0
    p(text_el(776, ry + 5.0, rlbl, size=14.4, weight=600, fill=C_SUBHDR, anchor="start"))
    # Delicate horizontal leaf track wire terminating cleanly on u_0 and l_post column centers
    p(f'  <line x1="{CX0 + 0.5*COL_W:.1f}" y1="{ry:.1f}" x2="{CX0 + 5.5*COL_W:.1f}" y2="{ry:.1f}" stroke="#CBDAD5" stroke-width="1.2"/>')
    for ci, val in enumerate(rvals):
        gx = CX0 + (ci + 0.5) * COL_W
        if val == 1:
            p(pill_el(gx - 17, ry - 11, 34, 22, rfc, rec, "1", txt_col=rtc, rx=5, sw=1.45, size=14.2, weight=700))
        elif val == 0:
            p(pill_el(gx - 17, ry - 11, 34, 22, "#F6F8F7", "#9CA3AF", "0", txt_col="#6B7280", rx=5, sw=1.25, size=13.8, weight=600, dash="3.0 2.2"))
        else:
            p(f'  <circle cx="{gx:.1f}" cy="{ry:.1f}" r="2.2" fill="#B8C9C3"/>')

# Hairline + Sum Row
p(f'  <line x1="776" y1="229" x2="1098" y2="229" stroke="#B6CCC5" stroke-width="1.2"/>')
p(text_el(776, 252, "∑ m_{l,t}", size=14.6, weight=700, fill=MINT_TXT, anchor="start"))
sums = ["1×", "1×", "0", "1×", "1×", "1×"]
for ci, stxt in enumerate(sums):
    gx = CX0 + (ci + 0.5) * COL_W
    is_ro = (stxt == "0")
    p(pill_el(
        gx - 17, 236, 34, 23,
        LAV_FILL if is_ro else MINT_SOFT,
        LAV_STR if is_ro else MINT_STR,
        stxt,
        txt_col=LAV_TXT if is_ro else MINT_TXT,
        rx=5, sw=1.35, size=13.8, weight=700
    ))

# Lower Section: Group RLOO Tree Credit
p(f'  {text_el(PC_X + 16, 294, "Group RLOO tree credit (G rollouts)", size=16.6, weight=600, fill=C_SUBHDR, anchor="start")}')
p(f'  <rect x="765" y="305" width="345" height="133" rx="6.5" fill="{C_WELL_TEAL}" stroke="{C_WELL_TDASH}" stroke-width="1.35" stroke-dasharray="4.6 3.4"/>')

trees = [
    (317, "τ_{1} : R = +1", MINT_SOFT, MINT_STR, MINT_TXT, "#2A7A60"),
    (356, "τ_{2} : R = −1", GREY_FILL, GREY_STR, GREY_TXT, "#78736B"),
    (395, "τ_{G} : R = +1", MINT_SOFT, MINT_STR, MINT_TXT, "#2A7A60"),
]
for ty, tlbl, tfc, tec, ttc, ic_col in trees:
    p(f'  <rect x="777" y="{ty}" width="118" height="30" rx="5.5" fill="{tfc}" stroke="{tec}" stroke-width="1.35"/>')
    p(mini_tree_icon(787, ty + 15, ic_col))
    p(text_el(848, ty + 20, tlbl, size=14.2, weight=700, fill=ttc))
    p(f'  <path d="{ortho_h(895, ty + 15, 919, 371, 907, 5.0)}" fill="none" stroke="{C_STROKE}" stroke-width="1.35"/>')

p(f'  {chevron(920, 371, "right", C_STROKE, 1.5, 3.5)}')

# Clean RLOO Baseline Card
p(f'  <rect x="922" y="318" width="176" height="106" rx="7" fill="#FFFFFF" stroke="#8AB3A8" stroke-width="1.4"/>')
p(f'  {text_el(1010, 342, "RLOO baseline", size=14.6, weight=600, fill="#3D7065")}')
p(f'  {text_el(1010, 372, "Â_{i} = R_{i} − mean_{j≠i}(R_{j})", size=14.8, weight=700, fill=C_INK)}')
p(pill_el(936, 389, 148, 25, MINT_SOFT, MINT_STR, "→ all m_{l,t} = 1 tokens", txt_col=MINT_TXT, rx=12.5, sw=1.25, size=13.4, weight=700))
p('</g>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved Editorial v4 Figure 3: {PDF_OUT}")
