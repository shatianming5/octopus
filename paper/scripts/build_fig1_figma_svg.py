#!/usr/bin/env python3
"""
Builds v5 Paper-Native High-Typography Edition (`fig_octopus_figma_v5.svg`):
- True-print academic typography scale (`1126 x 468` canvas @ `\textwidth` = `468pt`):
  - Panel titles `(a)` / `(b)`: `22.8px` Bold (~`9.5pt`, visually matching the `10pt` bold `Figure 1:` caption)
  - Primitives `fork` / `rewind` / `read`: `20.5px` Bold (~`8.5pt` Inter = `9.4pt` Times x-height)
  - Zone headers & `Query` / `Clue 1..3` / `Answer`: `19.0–19.2px` Bold/SemiBold (matching the `9pt` Abstract)
  - Secondary 1D labels (`Failed detour`, `Dead-end doc`, `Tool trace`): `16.4px` SemiBold
  - Legend & `fork · rewind · read`: `14.8–15.4px` SemiBold
- Paper-native floating dual panels `(a) Flat long context` and `(b) Octopus context virtualization` on pure `#FFFFFF`,
  aligned flush left/right with the LaTeX textblock (`\textwidth`).
- Preserves 100% of v1's beloved `1.122 : 1` landscape DAG aspect ratio (`376 x 335`) and `1.617 : 1` mini-windows (`43.0 x 26.6`).
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_pdf import render_svg_to_pdf, svg_mascot

ROOT = Path(__file__).resolve().parent.parent
ART = Path("/tmp/octopus-fig-preview")
SVG_OUT = ROOT / "figures/fig1_overview.svg"
PDF_OUT = ROOT / "figures/fig1_overview.pdf"
PNG_CHECK = ART / "scratch/fig1_overview_octo_check.png"

W, H = 1126, 468
FONT = "Inter"

# Paper-native high-clarity editorial palette
C_PANEL_A_BG  = "#FAF9F6"
C_PANEL_A_STR = "#D5D0C7"
C_PANEL_B_BG  = "#F7FBFA"
C_PANEL_B_STR = "#6BA396"

C_WIN_BG      = "#F1EFEA"
C_WIN_B_BG    = "#EEF5F3"
C_DAG_BG      = "#F2F6F5"

C_INK         = "#18181B"
C_SUBHDR      = "#2E2C29"
C_STROKE      = "#524E49"
C_WIN_DASH    = "#6B6760"
C_WIN_B_DASH  = "#4E766D"

BLUE_FILL,  BLUE_STR,  BLUE_TXT  = "#BDD9F2", "#4B749F", "#142E49"
MINT_FILL,  MINT_FLAP, MINT_STR, MINT_TXT = "#A8E2CD", "#7DC5AC", "#2A7A60", "#0B382B"
LAV_FILL,   LAV_STR,   LAV_TXT   = "#EAE4FA", "#7054B3", "#3F2A75"
PEACH_FILL, PEACH_STR, PEACH_TXT = "#FBD6B6", "#B56B36", "#47230A"
GREY_FILL,  GREY_FLAP, GREY_STR, GREY_TXT = "#E5E2DC", "#D3CEC5", "#B0ABA1", "#4F4B45"

out = []

def p(s: str):
    out.append(s)

def text_el(x, y, content, size=19.0, weight=600, fill=C_INK, anchor="middle", letter_spacing="-0.25", node_id=None):
    id_attr = f' id="{escape(node_id)}"' if node_id else ""
    return (
        f'<text{id_attr} x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" letter-spacing="{letter_spacing}">'
        f'{escape(content)}</text>'
    )

def speech_pill(x, y, w, h, fill, stroke, label=None, txt_col=BLUE_TXT, font_size=19.2, r=4.6, sw=1.45, tail_x=10.0, tail_w=6.8, tail_h=4.6, gid=None):
    tx0 = x + tail_x
    tx1 = tx0 + tail_w
    d = (
        f"M {x+r:.1f} {y:.1f} "
        f"L {x+w-r:.1f} {y:.1f} "
        f"Q {x+w:.1f} {y:.1f} {x+w:.1f} {y+r:.1f} "
        f"L {x+w:.1f} {y+h-r:.1f} "
        f"Q {x+w:.1f} {y+h:.1f} {x+w-r:.1f} {y+h:.1f} "
        f"L {tx1:.1f} {y+h:.1f} "
        f"L {tx0-1.5:.1f} {y+h+tail_h:.1f} "
        f"L {tx0:.1f} {y+h:.1f} "
        f"L {x+r:.1f} {y+h:.1f} "
        f"Q {x:.1f} {y+h:.1f} {x:.1f} {y+h-r:.1f} "
        f"L {x:.1f} {y+r:.1f} "
        f"Q {x:.1f} {y:.1f} {x+r:.1f} {y:.1f} Z"
    )
    g_open = f'<g id="{escape(gid)}">' if gid else "<g>"
    parts = [g_open, f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>']
    if label:
        pad_x = 10.5 if font_size >= 18 else 9.5
        parts.append(text_el(x + pad_x, y + h/2 + font_size*0.35, label, size=font_size, weight=700 if font_size >= 18 else 600, fill=txt_col, anchor="start"))
    parts.append("</g>")
    return "\n".join(parts)

def doc_pill(x, y, w, h, fill, flap_fill, stroke, label=None, txt_col=MINT_TXT, font_size=19.2, r=4.4, fold=6.8, sw=1.45, gid=None):
    d_body = (
        f"M {x+r:.1f} {y:.1f} "
        f"L {x+w-fold:.1f} {y:.1f} "
        f"L {x+w:.1f} {y+fold:.1f} "
        f"L {x+w:.1f} {y+h-r:.1f} "
        f"Q {x+w:.1f} {y+h:.1f} {x+w-r:.1f} {y+h:.1f} "
        f"L {x+r:.1f} {y+h:.1f} "
        f"Q {x:.1f} {y+h:.1f} {x:.1f} {y+h-r:.1f} "
        f"L {x:.1f} {y+r:.1f} "
        f"Q {x:.1f} {y:.1f} {x+r:.1f} {y:.1f} Z"
    )
    d_flap = (
        f"M {x+w-fold:.1f} {y:.1f} "
        f"L {x+w-fold:.1f} {y+fold:.1f} "
        f"L {x+w:.1f} {y+fold:.1f} Z"
    )
    g_open = f'<g id="{escape(gid)}">' if gid else "<g>"
    parts = [
        g_open,
        f'<path d="{d_body}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>',
        f'<path d="{d_flap}" fill="{flap_fill}" stroke="{stroke}" stroke-width="{sw*0.9:.2f}" stroke-linejoin="round"/>',
    ]
    if label:
        parts.append(text_el(x + w/2 - 1, y + h/2 + font_size*0.35, label, size=font_size, weight=700 if font_size >= 18 else 600, fill=txt_col, anchor="middle"))
    parts.append("</g>")
    return "\n".join(parts)

def rect_pill(x, y, w, h, fill, stroke, label=None, txt_col=GREY_TXT, r=4.3, sw=1.4, size=16.4, weight=600, gid=None):
    g_open = f'<g id="{escape(gid)}">' if gid else "<g>"
    parts = [g_open, f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>']
    if label:
        parts.append(text_el(x + w/2, y + h/2 + size*0.35, label, size=size, weight=weight, fill=txt_col, anchor="middle"))
    parts.append("</g>")
    return "\n".join(parts)

def context_window_frame(x, y, w, h, bg=C_WIN_BG, dash_col=C_WIN_DASH, gid="Context Window Frame"):
    r = 6.5
    sx = x + 25.0
    gap_l = sx - 11.5
    gap_r = sx + 11.5
    d_border = (
        f"M {gap_r:.1f} {y+h:.1f} "
        f"L {x+w-r:.1f} {y+h:.1f} "
        f"Q {x+w:.1f} {y+h:.1f} {x+w:.1f} {y+h-r:.1f} "
        f"L {x+w:.1f} {y+r:.1f} "
        f"Q {x+w:.1f} {y:.1f} {x+w-r:.1f} {y:.1f} "
        f"L {x+r:.1f} {y:.1f} "
        f"Q {x:.1f} {y:.1f} {x:.1f} {y+r:.1f} "
        f"L {x:.1f} {y+h-r:.1f} "
        f"Q {x:.1f} {y+h:.1f} {x+r:.1f} {y+h:.1f} "
        f"L {gap_l:.1f} {y+h:.1f}"
    )
    sy = y + h
    return "\n".join([
        f'<g id="{escape(gid)}">',
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{bg}"/>',
        f'  <path d="{d_border}" fill="none" stroke="{dash_col}" stroke-width="1.55" stroke-dasharray="4.8 3.5" stroke-linecap="round"/>',
        f'  <g id="{escape(gid)} - Scissors">',
        f'    <circle cx="{sx-5.4:.1f}" cy="{sy-3.7:.1f}" r="2.8" fill="none" stroke="#4A4742" stroke-width="1.45"/>',
        f'    <circle cx="{sx-5.4:.1f}" cy="{sy+3.7:.1f}" r="2.8" fill="none" stroke="#4A4742" stroke-width="1.45"/>',
        f'    <line x1="{sx-2.6:.1f}" y1="{sy-2.3:.1f}" x2="{sx+7.8:.1f}" y2="{sy+3.6:.1f}" stroke="#4A4742" stroke-width="1.45" stroke-linecap="round"/>',
        f'    <line x1="{sx-2.6:.1f}" y1="{sy+2.3:.1f}" x2="{sx+7.8:.1f}" y2="{sy-3.6:.1f}" stroke="#4A4742" stroke-width="1.45" stroke-linecap="round"/>',
        f'  </g>',
        f'</g>'
    ])

def ortho_h(x1, y1, x2, y2, x_mid, r=5.0):
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

def chevron_el(x, y, direction, color=C_STROKE, sw=1.45, s=3.6):
    if direction == "right":
        d = f"M {x - s*1.15:.1f} {y - s:.1f} L {x:.1f} {y:.1f} L {x - s*1.15:.1f} {y + s:.1f}"
    elif direction == "left":
        d = f"M {x + s*1.15:.1f} {y - s:.1f} L {x:.1f} {y:.1f} L {x + s*1.15:.1f} {y + s:.1f}"
    elif direction == "up":
        d = f"M {x - s:.1f} {y + s*1.15:.1f} L {x:.1f} {y:.1f} L {x + s:.1f} {y + s*1.15:.1f}"
    else:
        d = f"M {x - s:.1f} {y - s*1.15:.1f} L {x:.1f} {y:.1f} L {x + s:.1f} {y - s*1.15:.1f}"
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'

def neural_icon(cx, cy, gid="Model 3-4-3 Icon"):
    dx = 9.8
    cols = [
        (cx - dx, [cy - 9.4, cy, cy + 9.4], [BLUE_FILL, PEACH_FILL, BLUE_FILL]),
        (cx,      [cy - 14.1, cy - 4.7, cy + 4.7, cy + 14.1], [MINT_FILL, GREY_FILL, BLUE_FILL, MINT_FILL]),
        (cx + dx, [cy - 9.4, cy, cy + 9.4], [LAV_FILL, BLUE_FILL, PEACH_FILL]),
    ]
    lines = [f'<g id="{escape(gid)}">']
    for c in range(2):
        for y1 in cols[c][1]:
            for y2 in cols[c+1][1]:
                lines.append(f'  <line x1="{cols[c][0]:.1f}" y1="{y1:.1f}" x2="{cols[c+1][0]:.1f}" y2="{y2:.1f}" stroke="#B0ABA2" stroke-width="0.95"/>')
    for x_c, ys, fills in cols:
        for y_d, fl in zip(ys, fills):
            lines.append(f'  <circle cx="{x_c:.1f}" cy="{y_d:.1f}" r="3.2" fill="{fl}" stroke="#4A4742" stroke-width="1.25"/>')
    lines.append("</g>")
    return "\n".join(lines)

# ==============================================================================
# BUILD PAPER-NATIVE HIGH-TYPOGRAPHY SVG (1126 x 468)
# ==============================================================================
p(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none">')
p('<g id="Figure 1 - Paper Native BigType v5">')
p(f'  <rect id="Pure White Paper Backing" x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ==============================================================================
# 1. PANEL (a): FLAT LONG CONTEXT (318 x 464)
# ==============================================================================
LC_X, LC_Y, LC_W, LC_H = 1, 2, 318, 464
p('  <g id="Panel A - Flat Long Context">')
p(f'    <rect id="Panel A Surface" x="{LC_X}" y="{LC_Y}" width="{LC_W}" height="{LC_H}" rx="10" fill="{C_PANEL_A_BG}" stroke="{C_PANEL_A_STR}" stroke-width="1.45"/>')
p(f'    {text_el(LC_X+15, LC_Y+31, "(a) Flat long context", size=21.8, weight=700, fill=C_INK, anchor="start", letter_spacing="-0.35")}')
p(f'    {text_el(LC_X+15, LC_Y+63, "Context window", size=19.0, weight=600, fill=C_SUBHDR, anchor="start")}')

LW_X, LW_Y, LW_W, LW_H = LC_X + 15, LC_Y + 75, 150, 369
p(context_window_frame(LW_X, LW_Y, LW_W, LW_H, bg=C_WIN_BG, dash_col=C_WIN_DASH, gid="Left 1D Context Window"))

IX = LW_X + 8.0
IW = LW_W - 16.0  # 134px
cur_y = LW_Y + 11.0
left_row_cys = []

def add_1d_row(svg_snippet, h_row, gap_after=6.8):
    global cur_y
    p(svg_snippet)
    left_row_cys.append(cur_y + h_row / 2.0)
    cur_y += h_row + gap_after

# 11 Plump, High-Typography Rows filling LW_H=369 cleanly down to the scissors
add_1d_row(speech_pill(IX, cur_y, IW, 30.0, BLUE_FILL, BLUE_STR, "Query", font_size=19.2, gid="1D - Query"), 30.0, gap_after=8.6)
add_1d_row(speech_pill(IX, cur_y, IW, 26.5, GREY_FILL, GREY_STR, "Failed detour", txt_col=GREY_TXT, font_size=16.4, tail_h=4.0, sw=1.35), 26.5, gap_after=7.6)

w3 = (IW - 8.0) / 3.0
add_1d_row(
    f'<g id="1D - Tool Row 1">'
    f'{rect_pill(IX, cur_y, w3, 16.0, GREY_FILL, GREY_STR)}'
    f'{rect_pill(IX+w3+4, cur_y, w3, 16.0, GREY_FILL, GREY_STR)}'
    f'{rect_pill(IX+2*(w3+4), cur_y, w3, 16.0, GREY_FILL, GREY_STR)}'
    f'</g>', 16.0, gap_after=6.8
)
add_1d_row(doc_pill(IX, cur_y, IW, 26.5, GREY_FILL, GREY_FLAP, GREY_STR, "Dead-end doc", txt_col=GREY_TXT, font_size=16.4, fold=6.0, sw=1.35), 26.5, gap_after=7.0)

# Buried Clue 1
add_1d_row(doc_pill(IX, cur_y, IW, 29.0, MINT_FILL, MINT_FLAP, MINT_STR, "Clue 1", font_size=19.2, gid="1D - Buried Clue 1"), 29.0, gap_after=7.0)

add_1d_row(rect_pill(IX, cur_y, IW, 26.0, GREY_FILL, GREY_STR, "Tool trace", size=16.4), 26.0, gap_after=6.8)
add_1d_row(
    f'<g>{rect_pill(IX, cur_y, 55, 16.0, GREY_FILL, GREY_STR)}{rect_pill(IX+59, cur_y, IW-59, 16.0, GREY_FILL, GREY_STR)}</g>',
    16.0, gap_after=7.0
)

# Buried Clue 2
add_1d_row(doc_pill(IX, cur_y, IW, 29.0, MINT_FILL, MINT_FLAP, MINT_STR, "Clue 2", font_size=19.2, gid="1D - Buried Clue 2"), 29.0, gap_after=7.0)

add_1d_row(speech_pill(IX, cur_y, IW, 18.5, GREY_FILL, GREY_STR, tail_h=3.6, sw=1.35), 18.5, gap_after=7.8)

# Buried Clue 3
add_1d_row(doc_pill(IX, cur_y, IW, 29.0, MINT_FILL, MINT_FLAP, MINT_STR, "Clue 3", font_size=19.2, gid="1D - Buried Clue 3"), 29.0, gap_after=7.0)

add_1d_row(
    f'<g>{rect_pill(IX, cur_y, 62, 17.5, GREY_FILL, GREY_STR)}{rect_pill(IX+66, cur_y, IW-66, 17.5, GREY_FILL, GREY_STR)}</g>',
    17.5, gap_after=0.0
)

# 1D Collector Comb -> 3-4-3 Neural Icon -> Answer
L_SPINE_X = LW_X + LW_W + 8.5
L_ANS_W, L_ANS_H = 82, 44
L_ANS_X = LC_X + LC_W - 12 - L_ANS_W  # 50.5px neural zone
L_MODEL_Y = LW_Y + LW_H / 2.0
L_ANS_Y = L_MODEL_Y - L_ANS_H / 2.0
L_MODEL_X = (L_SPINE_X + L_ANS_X) / 2.0 - 1.0

p('    <g id="1D - Diluted Collector Comb">')
for idx, ry in enumerate(left_row_cys):
    xs = LW_X + LW_W - 6.0
    if idx == 0:
        p(f'      <path d="M {xs:.1f} {ry:.1f} L {L_SPINE_X-4:.1f} {ry:.1f} Q {L_SPINE_X:.1f} {ry:.1f} {L_SPINE_X:.1f} {ry+4:.1f} L {L_SPINE_X:.1f} {L_MODEL_Y:.1f}" fill="none" stroke="#858078" stroke-width="1.3"/>')
    elif idx == len(left_row_cys) - 1:
        p(f'      <path d="M {xs:.1f} {ry:.1f} L {L_SPINE_X-4:.1f} {ry:.1f} Q {L_SPINE_X:.1f} {ry:.1f} {L_SPINE_X:.1f} {ry-4:.1f} L {L_SPINE_X:.1f} {L_MODEL_Y:.1f}" fill="none" stroke="#858078" stroke-width="1.3"/>')
    else:
        p(f'      <line x1="{xs:.1f}" y1="{ry:.1f}" x2="{L_SPINE_X:.1f}" y2="{ry:.1f}" stroke="#858078" stroke-width="1.25"/>')
p(f'      <line x1="{L_SPINE_X:.1f}" y1="{L_MODEL_Y:.1f}" x2="{L_MODEL_X-16.0:.1f}" y2="{L_MODEL_Y:.1f}" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'      {chevron_el(L_MODEL_X-16.0, L_MODEL_Y, "right")}')
p(f'      <line x1="{L_MODEL_X+16.0:.1f}" y1="{L_MODEL_Y:.1f}" x2="{L_ANS_X-3:.1f}" y2="{L_MODEL_Y:.1f}" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'      {chevron_el(L_ANS_X-3, L_MODEL_Y, "right")}')
p('    </g>')

p(neural_icon(L_MODEL_X, L_MODEL_Y, gid="1D - Model Icon"))
p(rect_pill(L_ANS_X, L_ANS_Y, L_ANS_W, L_ANS_H, "#FFFFFF", C_WIN_DASH, "Answer", txt_col="#18181B", r=5.5, sw=1.5, size=19.0, weight=700, gid="1D - Answer"))
# Subtle Octopus mascot in Panel (a): overwhelmed by noisy flat context
p(f'    {svg_mascot("octo_08_chat_retok", L_ANS_X + L_ANS_W/2.0 - 1.0, L_ANS_Y - 52.0, 74, 74)}')
p('  </g>')

# ==============================================================================
# 2. PANEL (b): OCTOPUS CONTEXT VIRTUALIZATION (794 x 464, 12px white gap)
# ==============================================================================
RC_X, RC_Y, RC_W, RC_H = 331, 2, 794, 464
p('  <g id="Panel B - Octopus Context Virtualization">')
p(f'    <rect id="Panel B Surface" x="{RC_X}" y="{RC_Y}" width="{RC_W}" height="{RC_H}" rx="10" fill="{C_PANEL_B_BG}" stroke="{C_PANEL_B_STR}" stroke-width="1.55"/>')
p(f'    {text_el(RC_X+16, RC_Y+31, "(b) Octopus context virtualization", size=21.8, weight=700, fill=C_INK, anchor="start", letter_spacing="-0.35")}')

# Crisp 3-Swatch Micro-Legend right-aligned on Panel B's top header row (14.5px SemiBold!)
LEG_Y = RC_Y + 19.0
p('    <g id="Panel B - SubWindow Legend">')
s1_x = 738.0
p(f'      <rect x="{s1_x:.1f}" y="{LEG_Y:.1f}" width="16" height="11" rx="2.5" fill="#FFFFFF" stroke="{MINT_STR}" stroke-width="1.4" stroke-dasharray="2.6 1.8"/>')
p(f'      <rect x="{s1_x+2.6:.1f}" y="{LEG_Y+3.5:.1f}" width="10.8" height="4.0" rx="1.2" fill="{MINT_FILL}" stroke="{MINT_STR}" stroke-width="0.85"/>')
p(f'      {text_el(s1_x + 20.0, LEG_Y + 10.3, "Verified clue", size=14.5, weight=600, fill="#1E5643", anchor="start", letter_spacing="-0.15")}')

s2_x = 860.0
p(f'      <rect x="{s2_x:.1f}" y="{LEG_Y:.1f}" width="16" height="11" rx="2.5" fill="#FFFFFF" stroke="{LAV_STR}" stroke-width="1.4" stroke-dasharray="2.6 1.8"/>')
p(f'      <rect x="{s2_x+2.6:.1f}" y="{LEG_Y+3.5:.1f}" width="10.8" height="4.0" rx="1.2" fill="{LAV_FILL}" stroke="{LAV_STR}" stroke-width="0.85"/>')
p(f'      {text_el(s2_x + 20.0, LEG_Y + 10.3, "Mounted KV", size=14.5, weight=600, fill=LAV_TXT, anchor="start", letter_spacing="-0.15")}')

s3_x = 980.0
p(f'      <rect x="{s3_x:.1f}" y="{LEG_Y:.1f}" width="16" height="11" rx="2.5" fill="#ECEAE4" stroke="#B5B0A6" stroke-width="1.3" stroke-dasharray="2.6 1.8"/>')
p(f'      <rect x="{s3_x+2.6:.1f}" y="{LEG_Y+3.5:.1f}" width="10.8" height="4.0" rx="1.2" fill="#DCD8D0" stroke="#A6A098" stroke-width="0.85"/>')
p(f'      {text_el(s3_x + 20.0, LEG_Y + 10.3, "Rewound detour", size=14.5, weight=500, fill="#4F4B45", anchor="start", letter_spacing="-0.15")}')
p('    </g>')

p(f'    {text_el(RC_X+16, RC_Y+63, "Virtualized sub-window DAG", size=19.0, weight=600, fill=C_SUBHDR, anchor="start")}')

# Zone A: Sleek Landscape 5-Stage Sub-Window DAG Container (376 x 335 = 1.122 : 1 — exact v1 ratio!)
DAG_X, DAG_Y, DAG_W, DAG_H = RC_X + 16, RC_Y + 75, 376, 335
p(f'    <rect id="DAG Outer Container" x="{DAG_X}" y="{DAG_Y}" width="{DAG_W}" height="{DAG_H}" rx="8" fill="{C_DAG_BG}" stroke="#84A39C" stroke-width="1.5" stroke-dasharray="4.8 3.5"/>')

# Sleek 1.617 : 1 mini-windows (43.0 x 26.6)
MW, MH = 43.0, 26.6
COL_GAP = (DAG_W - 22.0 - 5 * MW) / 4.0  # 34.75px
COL_XS = [DAG_X + 11.0 + c * (MW + COL_GAP) for c in range(5)]

def lane_top(lane_float: float) -> float:
    base = DAG_Y + 11.0
    step = (DAG_H - 22.0 - MH) / 7.0
    return base + lane_float * step

dag_nodes = {
    "root":   (0, 3.5, "active",   [("blue", 1.0), ("peach", 0.78)]),

    "d1_0":   (1, 1.0, "active",   [("blue", 1.0), ("grey", 0.68)]),
    "d1_1":   (1, 3.5, "active",   [("blue", 1.0), ("peach", 0.78)]),
    "d1_2":   (1, 6.0, "active",   [("blue", 1.0), ("grey", 0.68)]),

    "d2_0":   (2, 0.0, "active",   [("blue", 1.0), ("peach", 0.78)]),
    "d2_1":   (2, 1.0, "verified", [("blue", 0.68), ("mint", 1.0)]),
    "d2_2":   (2, 2.0, "active",   [("blue", 1.0), ("peach", 0.78)]),
    "d2_3":   (2, 3.0, "dead",     [("grey", 1.0), ("grey", 0.68)]),
    "d2_4":   (2, 4.0, "active",   [("blue", 1.0), ("peach", 0.78)]),
    "d2_5":   (2, 5.0, "dead",     [("grey", 1.0), ("grey", 0.68)]),
    "d2_6":   (2, 6.0, "active",   [("blue", 1.0), ("peach", 0.78)]),
    "d2_7":   (2, 7.0, "dead",     [("grey", 1.0), ("grey", 0.68)]),

    "d3_0":   (3, 0.0, "dead",     [("grey", 1.0), ("grey", 0.68)]),
    "d3_2":   (3, 2.0, "verified", [("lav",  0.82), ("mint", 1.0)]),
    "d3_4":   (3, 4.0, "active",   [("blue", 1.0), ("peach", 0.78)]),
    "d3_5":   (3, 5.0, "cue",      [("blue", 0.70), ("lav",  0.90)]),
    "d3_6":   (3, 6.0, "active",   [("blue", 1.0), ("peach", 0.78)]),

    "d4_3":   (4, 3.0, "dead",     [("grey", 1.0), ("grey", 0.68)]),
    "d4_4":   (4, 4.0, "dead",     [("grey", 1.0), ("grey", 0.68)]),
    "d4_6":   (4, 6.0, "verified", [("lav",  0.82), ("mint", 1.0)]),
    "d4_7":   (4, 7.0, "dead",     [("grey", 1.0), ("grey", 0.68)]),
}

def nbox(nid):
    c, lf, kind, _ = dag_nodes[nid]
    nx = COL_XS[c]
    ny = lane_top(lf)
    return {"left": nx, "right": nx + MW, "top": ny, "bottom": ny + MH, "cx": nx + MW/2.0, "cy": ny + MH/2.0, "kind": kind}

dag_edges = [
    ("root", "d1_0"), ("root", "d1_1"), ("root", "d1_2"),
    ("d1_0", "d2_0"), ("d1_0", "d2_1"), ("d1_0", "d2_2"),
    ("d1_1", "d2_3"), ("d1_1", "d2_4"),
    ("d1_2", "d2_5"), ("d1_2", "d2_6"), ("d1_2", "d2_7"),
    ("d2_0", "d3_0"), ("d2_2", "d3_2"), ("d2_4", "d3_4"), ("d2_6", "d3_5"), ("d2_6", "d3_6"),
    ("d3_4", "d4_3"), ("d3_4", "d4_4"), ("d3_6", "d4_6"), ("d3_6", "d4_7"),
]

p('    <g id="DAG Fork Edges">')
for u_id, v_id in dag_edges:
    u, v = nbox(u_id), nbox(v_id)
    is_dead = (v["kind"] == "dead")
    xm = (u["right"] + v["left"]) / 2.0
    col_s = "#B5B0A6" if is_dead else C_STROKE
    sw_s = "1.25" if is_dead else "1.45"
    p(f'      <path d="{ortho_h(u["right"]+1, u["cy"], v["left"]-2, v["cy"], xm, 4.2)}" fill="none" stroke="{col_s}" stroke-width="{sw_s}"/>')
    if not is_dead:
        p(f'      {chevron_el(v["left"]-2, v["cy"], "right", C_STROKE, 1.35, 2.6)}')

r_b, d1_top_b = nbox("root"), nbox("d1_0")
p(f'      {svg_mascot("octo_fork", DAG_X + 27.5, d1_top_b["top"] - 9.0, 38, 40)}')
p(f'      {text_el(DAG_X + 50.5, d1_top_b["top"] - 5.5, "fork", size=20.5, weight=700, fill="#18181B", anchor="start")}')
p('    </g>')

BAR_STYLES = {
    "blue":  (BLUE_FILL,  BLUE_STR),
    "peach": (PEACH_FILL, PEACH_STR),
    "mint":  (MINT_FILL,  MINT_STR),
    "lav":   (LAV_FILL,   LAV_STR),
    "grey":  ("#DCD8D0",  GREY_STR),
}

p('    <g id="DAG Sub-Windows">')
for nid, (c, lf, kind, bars) in dag_nodes.items():
    nb = nbox(nid)
    nx, ny = nb["left"], nb["top"]
    if kind == "dead":
        bg, bd, op = "#ECEAE4", "#BEB9B0", ' opacity="0.72"'
    elif kind == "verified":
        bg, bd, op = "#FFFFFF", MINT_STR, ""
    elif kind == "cue":
        bg, bd, op = "#FFFFFF", LAV_STR, ""
    else:
        bg, bd, op = "#FFFFFF", C_WIN_DASH, ""

    p(f'      <g id="SubWindow - {nid}"{op}>')
    p(f'        <rect x="{nx:.1f}" y="{ny:.1f}" width="{MW:.1f}" height="{MH:.1f}" rx="3.6" fill="{bg}" stroke="{bd}" stroke-width="1.35" stroke-dasharray="3.1 2.2"/>')
    by0 = ny + 5.8
    for b_idx, (b_kind, w_frac) in enumerate(bars):
        bf, bs = BAR_STYLES[b_kind]
        bw = (MW - 9.0) * w_frac
        by = by0 + b_idx * 8.5
        p(f'        <rect x="{nx+4.5:.1f}" y="{by:.1f}" width="{bw:.1f}" height="5.1" rx="1.6" fill="{bf}" stroke="{bs}" stroke-width="1.05"/>')
    p('      </g>')
p('    </g>')

p('    <g id="DAG Rewind Quarantines">')
for did, (_, _, kind, _) in dag_nodes.items():
    if kind == "dead":
        db = nbox(did)
        rx, ry = db["right"] + 2.0, db["cy"]
        p(f'      <path d="M {rx:.1f} {ry-2.3:.1f} C {rx+7.8:.1f} {ry-2.3:.1f} {rx+7.8:.1f} {ry+4.2:.1f} {rx+0.5:.1f} {ry+4.2:.1f}" fill="none" stroke="#807B73" stroke-width="1.25" stroke-linecap="round"/>')
        p(f'      {chevron_el(rx+0.5, ry+4.2, "left", "#807B73", 1.25, 2.2)}')
w_dead_lbl = nbox("d2_3")
p(f'      {text_el(w_dead_lbl["right"] + 13.0, w_dead_lbl["cy"] + 6.5, "rewind", size=20.2, weight=700, fill="#3F3C37", anchor="start")}')
p('    </g>')

# ==============================================================================
# Zone B: Bounded Active Window (122 x 222, large 19.2px Bold labels, 66px bridge!)
# ==============================================================================
ACT_X = DAG_X + DAG_W + 66
ACT_Y = DAG_Y + 84
ACT_W = 122
ACT_H = 222

p(f'    {text_el(ACT_X, ACT_Y - 11.5, "Active window", size=19.0, weight=600, fill=C_SUBHDR, anchor="start")}')
p(context_window_frame(ACT_X, ACT_Y, ACT_W, ACT_H, bg=C_WIN_B_BG, dash_col=C_WIN_B_DASH, gid="Right Bounded Active Window"))

AIX = ACT_X + 8.5
AIW = ACT_W - 17.0  # 105px
act_q_y  = ACT_Y + 12.0
act_c1_y = act_q_y + 30.0 + 11.5
act_c2_y = act_c1_y + 29.0 + 8.5
act_c3_y = act_c2_y + 29.0 + 8.5

p('    <g id="Active Window Mounted Tokens">')
p(speech_pill(AIX, act_q_y, AIW, 30.0, BLUE_FILL, BLUE_STR, "Query", font_size=19.2, gid="Active - Query"))
p(doc_pill(AIX, act_c1_y, AIW, 29.0, MINT_FILL, MINT_FLAP, MINT_STR, "Clue 1", font_size=19.2, gid="Active - Clue 1"))
p(doc_pill(AIX, act_c2_y, AIW, 29.0, MINT_FILL, MINT_FLAP, MINT_STR, "Clue 2", font_size=19.2, gid="Active - Clue 2"))
p(doc_pill(AIX, act_c3_y, AIW, 29.0, MINT_FILL, MINT_FLAP, MINT_STR, "Clue 3", font_size=19.2, gid="Active - Clue 3"))
p('    </g>')

# ==============================================================================
# Zero-Copy `read` Connectors (`read` centered cleanly in the 66px bridge above top purple line!)
# ==============================================================================
c1_src = nbox("d2_1")
c2_src = nbox("d3_2")
cue_src = nbox("d3_5")
c3_src = nbox("d4_6")

p('    <g id="Zero-Copy Read Connectors">')
p(f'      <circle cx="{c2_src["cx"]:.1f}" cy="{c1_src["cy"]:.1f}" r="2.4" fill="{LAV_STR}"/>')
p(f'      <path d="M {c2_src["cx"]:.1f} {c1_src["cy"]:.1f} L {c2_src["cx"]:.1f} {c2_src["top"]-2.5:.1f}" fill="none" stroke="{LAV_STR}" stroke-width="1.4" stroke-dasharray="3.3 2.4"/>')
p(f'      {chevron_el(c2_src["cx"], c2_src["top"]-2.5, "down", LAV_STR, 1.35, 2.5)}')

p(f'      <path d="M {cue_src["right"]+2:.1f} {cue_src["cy"]:.1f} L {c3_src["cx"]-5:.1f} {cue_src["cy"]:.1f} Q {c3_src["cx"]:.1f} {cue_src["cy"]:.1f} {c3_src["cx"]:.1f} {cue_src["cy"]+5:.1f} L {c3_src["cx"]:.1f} {c3_src["top"]-2.5:.1f}" fill="none" stroke="{LAV_STR}" stroke-width="1.4" stroke-dasharray="3.3 2.4"/>')
p(f'      {chevron_el(c3_src["cx"], c3_src["top"]-2.5, "down", LAV_STR, 1.35, 2.5)}')

XB0, XB1 = DAG_X + DAG_W, ACT_X
t1 = XB0 + (XB1 - XB0) * 0.79
t2 = XB0 + (XB1 - XB0) * 0.34
t3 = XB0 + (XB1 - XB0) * 0.56

for src_b, dst_cy, xt in [
    (c1_src, act_c1_y + 14.5, t1),
    (c2_src, act_c2_y + 14.5, t2),
    (c3_src, act_c3_y + 14.5, t3),
]:
    p(f'      <path d="{ortho_h(src_b["right"]+2, src_b["cy"], AIX-2.5, dst_cy, xt, 5.2)}" fill="none" stroke="{LAV_STR}" stroke-width="1.55" stroke-dasharray="3.8 2.7"/>')
    p(f'      {chevron_el(AIX-2.5, dst_cy, "right", LAV_STR, 1.45, 3.0)}')

p(f'      {text_el((XB0 + XB1)/2.0, c1_src["cy"] - 9.5, "read", size=20.5, weight=700, fill=LAV_TXT)}')
p(f'      {svg_mascot("octo_radix", (XB0 + XB1)/2.0, c1_src["cy"] - 47.0, 42, 38)}')
p('    </g>')

# ==============================================================================
# Zone C: Clean 4-Prong Collector -> 3-4-3 Neural Icon -> Answer + Context Action
# ==============================================================================
R_SPINE_X = ACT_X + ACT_W + 8.5
R_OUT_W = 142
R_OUT_X = RC_X + RC_W - 12 - R_OUT_W  # 51.5px neural zone
R_ANS_Y = ACT_Y + 42
R_ANS_H = 44
R_ACT_Y = R_ANS_Y + R_ANS_H + 15
R_ACT_H = 66

R_MODEL_Y = (R_ANS_Y + R_ANS_H/2.0 + R_ACT_Y + R_ACT_H/2.0) / 2.0
R_MODEL_X = (R_SPINE_X + R_OUT_X) / 2.0 - 2.0

act_cys = [act_q_y + 15.0, act_c1_y + 14.5, act_c2_y + 14.5, act_c3_y + 14.5]
p('    <g id="2D - Clean Active Collector and Action Loop">')
for idx, ry in enumerate(act_cys):
    xs = AIX + AIW + 1.5
    if idx == 0:
        p(f'      <path d="M {xs:.1f} {ry:.1f} L {R_SPINE_X-4:.1f} {ry:.1f} Q {R_SPINE_X:.1f} {ry:.1f} {R_SPINE_X:.1f} {ry+4:.1f} L {R_SPINE_X:.1f} {R_MODEL_Y:.1f}" fill="none" stroke="{C_STROKE}" stroke-width="1.45"/>')
    elif idx == len(act_cys) - 1:
        p(f'      <path d="M {xs:.1f} {ry:.1f} L {R_SPINE_X-4:.1f} {ry:.1f} Q {R_SPINE_X:.1f} {ry:.1f} {R_SPINE_X:.1f} {ry-4:.1f} L {R_SPINE_X:.1f} {R_MODEL_Y:.1f}" fill="none" stroke="{C_STROKE}" stroke-width="1.45"/>')
    else:
        p(f'      <line x1="{xs:.1f}" y1="{ry:.1f}" x2="{R_SPINE_X:.1f}" y2="{ry:.1f}" stroke="{C_STROKE}" stroke-width="1.45"/>')

p(f'      <line x1="{R_SPINE_X:.1f}" y1="{R_MODEL_Y:.1f}" x2="{R_MODEL_X-15.5:.1f}" y2="{R_MODEL_Y:.1f}" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'      {chevron_el(R_MODEL_X-15.5, R_MODEL_Y, "right")}')

x_split = R_MODEL_X + 17.5
p(f'      <line x1="{R_MODEL_X+13.8:.1f}" y1="{R_MODEL_Y:.1f}" x2="{x_split:.1f}" y2="{R_MODEL_Y:.1f}" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'      <path d="{ortho_h(x_split, R_MODEL_Y, R_OUT_X-3, R_ANS_Y+R_ANS_H/2.0, x_split+2.5, 4.5)}" fill="none" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'      {chevron_el(R_OUT_X-3, R_ANS_Y+R_ANS_H/2.0, "right")}')
p(f'      <path d="{ortho_h(x_split, R_MODEL_Y, R_OUT_X-3, R_ACT_Y+R_ACT_H/2.0, x_split+2.5, 4.5)}" fill="none" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'      {chevron_el(R_OUT_X-3, R_ACT_Y+R_ACT_H/2.0, "right")}')

# Swiss-grid alignment: Y_LOOP aligns to LW_Y + LW_H (y = 446)!
Y_LOOP = LW_Y + LW_H
X_RET = DAG_X + DAG_W / 2.0 + 6.0
X_ACT_CX = R_OUT_X + R_OUT_W / 2.0
RL = 7.0
d_loop = (
    f"M {X_ACT_CX:.1f} {R_ACT_Y+R_ACT_H+1:.1f} "
    f"L {X_ACT_CX:.1f} {Y_LOOP-RL:.1f} "
    f"Q {X_ACT_CX:.1f} {Y_LOOP:.1f} {X_ACT_CX-RL:.1f} {Y_LOOP:.1f} "
    f"L {X_RET+RL:.1f} {Y_LOOP:.1f} "
    f"Q {X_RET:.1f} {Y_LOOP:.1f} {X_RET:.1f} {Y_LOOP-RL:.1f} "
    f"L {X_RET:.1f} {DAG_Y+DAG_H+3.5:.1f}"
)
p(f'      <path d="{d_loop}" fill="none" stroke="{C_STROKE}" stroke-width="1.5"/>')
p(f'      {chevron_el(X_RET, DAG_Y+DAG_H+3.5, "up")}')
p('    </g>')

p(neural_icon(R_MODEL_X, R_MODEL_Y, gid="2D - Model Icon"))
p(rect_pill(R_OUT_X, R_ANS_Y, R_OUT_W, R_ANS_H, "#FFFFFF", C_WIN_B_DASH, "Answer", txt_col="#18181B", r=5.5, sw=1.5, size=19.0, weight=700, gid="2D - Answer"))
# Subtle Octopus hero mascot in Panel (b): cool branching & KV-key boss above Answer
p(f'    {svg_mascot("octo_21_title_boss", R_OUT_X + R_OUT_W/2.0, R_ANS_Y - 52.0, 82, 80)}')

p('    <g id="2D - Context Action">')
p(f'      <rect x="{R_OUT_X:.1f}" y="{R_ACT_Y:.1f}" width="{R_OUT_W:.1f}" height="{R_ACT_H:.1f}" rx="5.5" fill="{PEACH_FILL}" stroke="{PEACH_STR}" stroke-width="1.5"/>')
p(f'      {text_el(R_OUT_X+R_OUT_W/2.0, R_ACT_Y+27.0, "Context action", size=18.5, weight=700, fill=PEACH_TXT)}')
p(f'      {text_el(R_OUT_X+R_OUT_W/2.0, R_ACT_Y+49.5, "fork · rewind · read", size=15.2, weight=600, fill="#6E411D")}')
p('    </g>')

p('  </g>')
p('</g>')
p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved Figure 1 with subtle Octopus mascots: {PDF_OUT}")
