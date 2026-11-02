#!/usr/bin/env python3
"""
Builds Paper-Native High-Typography Figure 8 (`figures/fig_search_rollout_case.{svg,pdf}`):
- Exact 1100 x 446 canvas (@ 6.50 in = 468 pt \textwidth -> 1 px = 0.4255 pt, printed height = 189.8 pt):
  - Panel titles `(a)/(b)`: 19.8px Bold (~8.4pt Inter)
  - Node & turn headers: 14.6-15.4px Bold (~6.2-6.6pt Inter)
  - Sublabels & badges: 13.6-14.8px SemiBold/Bold (~5.8-6.3pt Inter)
- Removes redundant top query banner & cartoon clip-art; 100% pure Inter 4.0 CID TrueType PDF (0 Type 3 fonts).
"""
from pathlib import Path
from svg_to_pdf import format_rich_tspans, render_svg_to_pdf

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig_search_rollout_case.svg"
PDF_OUT = ROOT / "figures/fig_search_rollout_case.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig8_v2_check.png")

W, H = 1100, 446
FONT = "Inter"

C_TEAL_BG,  C_TEAL_STR  = "#F7FBFA", "#6BA396"
C_STONE_BG, C_STONE_STR = "#FAF9F6", "#D5D0C7"
C_INK       = "#18181B"
C_MUTED     = "#3F3F46"
C_STROKE    = "#524E49"

BLUE_FILL,  BLUE_STR,  BLUE_TXT  = "#BDD9F2", "#4B749F", "#142E49"
BLUE_SOFT,  BLUE_SSTR            = "#EEF5FC", "#5B84B1"
MINT_FILL,  MINT_STR,  MINT_TXT  = "#D8F3EC", "#2A7A60", "#0B382B"
MINT_DEEP                        = "#A8E2CD"
PEACH_FILL, PEACH_STR, PEACH_TXT = "#FBD6B6", "#B56B36", "#47230A"
AMBER_FILL, AMBER_STR, AMBER_TXT = "#FEF3C7", "#C87524", "#78350F"
CORAL_FILL, CORAL_STR, CORAL_TXT = "#FEE2E2", "#C83E3E", "#7F1D1D"
CORAL_SOFT                       = "#FEF2F2"
LAV_FILL,   LAV_STR,   LAV_TXT   = "#EAE4FA", "#7054B3", "#3F2A75"

out: list[str] = []


def p(s: str) -> None:
    out.append(s)


def text_el(x: float, y: float, content: str, size: float = 15.0, weight: int = 600,
            fill: str = C_INK, anchor: str = "middle", ls: str = "-0.22") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}">'
        f'{format_rich_tspans(content, size)}</text>'
    )


def box_el(x: float, y: float, w: float, h: float, fill: str, stroke: str,
           lines: list[tuple[str, float, int, str]] | None = None,
           rx: float = 6.5, sw: float = 1.45, dash: str | None = None) -> str:
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d_attr}/>'
    ]
    if lines:
        n = len(lines)
        if n == 1:
            txt, sz, wt, col = lines[0]
            parts.append(text_el(x + w / 2, y + h / 2 + sz * 0.35, txt, size=sz, weight=wt, fill=col))
        elif n == 2:
            (t1, s1, w1, c1), (t2, s2, w2, c2) = lines
            parts.append(text_el(x + w / 2, y + h * 0.36 + s1 * 0.34, t1, size=s1, weight=w1, fill=c1))
            parts.append(text_el(x + w / 2, y + h * 0.74 + s2 * 0.30, t2, size=s2, weight=w2, fill=c2))
        elif n == 3:
            (t1, s1, w1, c1), (t2, s2, w2, c2), (t3, s3, w3, c3) = lines
            parts.append(text_el(x + w / 2, y + h * 0.25 + s1 * 0.34, t1, size=s1, weight=w1, fill=c1))
            parts.append(text_el(x + w / 2, y + h * 0.54 + s2 * 0.32, t2, size=s2, weight=w2, fill=c2))
            parts.append(text_el(x + w / 2, y + h * 0.82 + s3 * 0.28, t3, size=s3, weight=w3, fill=c3))
    return "\n".join(parts)


def chevron(x: float, y: float, direction: str, color: str = C_STROKE, sw: float = 1.65, s: float = 3.8) -> str:
    if direction == "right":
        d = f"M {x - s*1.15:.1f} {y - s:.1f} L {x:.1f} {y:.1f} L {x - s*1.15:.1f} {y + s:.1f}"
    elif direction == "left":
        d = f"M {x + s*1.15:.1f} {y - s:.1f} L {x:.1f} {y:.1f} L {x + s*1.15:.1f} {y + s:.1f}"
    elif direction == "down":
        d = f"M {x - s:.1f} {y - s*1.15:.1f} L {x:.1f} {y:.1f} L {x + s:.1f} {y - s*1.15:.1f}"
    else:
        d = f"M {x - s:.1f} {y + s*1.15:.1f} L {x:.1f} {y:.1f} L {x + s:.1f} {y + s*1.15:.1f}"
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'


p(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ==============================================================================
# PANEL (a): Linear-Big RL (143K Cap): Detour Pollution (x=1, w=362)
# ==============================================================================
PA_X, PA_Y, PA_W, PA_H = 1, 2, 362, 442
p('<g id="Panel A - Linear-Big RL">')
p(f'  <rect x="{PA_X}" y="{PA_Y}" width="{PA_W}" height="{PA_H}" rx="10" fill="{C_STONE_BG}" stroke="{C_STONE_STR}" stroke-width="1.5"/>')
p(f'  {text_el(PA_X + 14, 31, "(a) Linear-Big RL (143K)", size=19.8, weight=700, fill=C_INK, anchor="start", ls="-0.32")}')
p(box_el(272, 12, 78, 25, CORAL_FILL, CORAL_STR, [("✗ y = 0", 14.4, 700, CORAL_TXT)], rx=12.5, sw=1.35))

steps_a = [
    (48,  "#EEF8F5",  MINT_STR,  None, [
        ('Turn 1–2: search("Luzon cooling")', 15.0, 700, MINT_TXT),
        ("✓ Finds 1991 Mt. Pinatubo (2.1K tok)", 14.0, 600, "#1D7A67"),
    ]),
    (112, CORAL_SOFT, CORAL_STR, "4.5 3.0", [
        ("Turn 3–5: open(1840291) [+8.2K]", 15.0, 700, CORAL_TXT),
        ("✗ SO_{2} aerosols dead-end (12.4K)", 14.0, 600, "#991B1B"),
    ]),
    (176, CORAL_SOFT, CORAL_STR, "4.5 3.0", [
        ("Turn 6–9: open(3910442) [+8.5K]", 15.0, 700, CORAL_TXT),
        ('✗ Noise: "2" stalls & "9" airports (24.8K)', 14.0, 600, "#991B1B"),
    ]),
    (240, CORAL_SOFT, CORAL_STR, "4.5 3.0", [
        ("Turn 10–14: open(7102938) [+8.4K]", 15.0, 700, CORAL_TXT),
        ("✗ Wrong volcano ash clouds (40.8K)", 14.0, 600, "#991B1B"),
    ]),
    (304, CORAL_FILL, CORAL_STR, None, [
        ('Turn 15: done("2")  [✗ Wrong]', 15.2, 700, CORAL_TXT),
        ("Misses doc 5544925 · h_{t} polluted", 14.2, 700, "#991B1B"),
    ]),
]

for idx, (sy, sfc, sec, sdash, slines) in enumerate(steps_a):
    p(box_el(14, sy, 336, 53, sfc, sec, slines, rx=7, sw=1.55 if idx == 4 else 1.4, dash=sdash))
    if idx < 4:
        col_a = "#71717A" if idx == 0 else CORAL_STR
        p(f'  <line x1="182" y1="{sy + 53}" x2="182" y2="{sy + 63}" stroke="{col_a}" stroke-width="1.8"/>')
        p(f'  {chevron(182, sy + 63, "down", col_a, 1.65, 3.3)}')

p(box_el(14, 369, 336, 34, CORAL_FILL, CORAL_STR, [("40.8K noisy tok · 21.1% gold recall", 15.2, 700, CORAL_TXT)], rx=7, sw=1.45))
p(f'  {text_el(182, 427, "Cannot prune 8K dead-ends or branch", size=14.4, weight=600, fill=C_MUTED)}')
p('</g>')

# ==============================================================================
# PANEL (b): Octopus Unlimited-Depth DAG (x=375, w=724)
# ==============================================================================
PB_X, PB_Y, PB_W, PB_H = 375, 2, 724, 442
p('<g id="Panel B - Octopus Unlimited-Depth DAG">')
p(f'  <rect x="{PB_X}" y="{PB_Y}" width="{PB_W}" height="{PB_H}" rx="10" fill="{C_TEAL_BG}" stroke="{C_TEAL_STR}" stroke-width="1.5"/>')
p(f'  {text_el(PB_X + 14, 31, "(b) Octopus unlimited-depth DAG (d = 0→5, 10 branches)", size=19.6, weight=700, fill=C_INK, anchor="start", ls="-0.32")}')
p(box_el(950, 12, 136, 25, MINT_FILL, MINT_STR, [("✓ y = 1 · R = +1", 14.4, 700, MINT_TXT)], rx=12.5, sw=1.35))

# 4 Depth Columns (y=48, h=308 -> bottom=356)
cols_b = [
    (388, 136, "#FFFFFF", "#CBD5E1", MINT_FILL,  MINT_STR,  MINT_TXT,  "d = 0 → 1"),
    (532, 152, "#FFFFFF", "#CBD5E1", BLUE_FILL,  BLUE_STR,  BLUE_TXT,  "d = 2 forks"),
    (692, 224, "#FFFDF5", "#E5C17B", AMBER_FILL, AMBER_STR, AMBER_TXT, "d = 3 → 4"),
    (924, 162, "#F3FAF7", "#7DC5AC", MINT_FILL,  MINT_STR,  MINT_TXT,  "d = 5 & join"),
]
for cx, cw, bg_c, bd_c, h_fc, h_ec, h_tc, h_txt in cols_b:
    p(f'  <rect x="{cx}" y="48" width="{cw}" height="308" rx="7" fill="{bg_c}" stroke="{bd_c}" stroke-width="1.25" stroke-dasharray="4.5 3.2"/>')
    p(box_el(cx + 8, 55, cw - 16, 25, h_fc, h_ec, [(h_txt, 14.4, 700, h_tc)], rx=12.5, sw=1.3))

# Col 1 Nodes (d=0 -> d=1)
p(box_el(396, 94, 120, 62, MINT_FILL, MINT_STR, [
    ("d=0 Root (c_{0})", 15.0, 700, MINT_TXT),
    ("fork0 + wait", 13.8, 600, "#1D7A67"),
], rx=7, sw=1.5))

p(box_el(396, 180, 120, 62, BLUE_FILL, BLUE_STR, [
    ("d=1 branch-0", 15.0, 700, BLUE_TXT),
    ("fork(0.0, 0.1)", 13.8, 600, BLUE_TXT),
], rx=7, sw=1.5))

p(box_el(396, 272, 120, 62, LAV_FILL, LAV_STR, [
    ("Clean 1.8K root", 14.4, 700, LAV_TXT),
    ("0 root rewinds", 13.8, 600, LAV_TXT),
], rx=7, sw=1.35))

# Col 2 Nodes (d=2)
p(box_el(540, 100, 136, 70, PEACH_FILL, PEACH_STR, [
    ("branch-0.0 (S)", 15.0, 700, PEACH_TXT),
    ("1991 accident", 13.8, 600, PEACH_TXT),
    ("[KV pages K_{S}]", 13.8, 700, "#9A3412"),
], rx=7, sw=1.55))

p(box_el(540, 238, 136, 62, BLUE_FILL, BLUE_STR, [
    ("branch-0.1", 15.0, 700, BLUE_TXT),
    ("Boeing ash query", 13.8, 600, BLUE_TXT),
], rx=7, sw=1.5))

# Col 3 Nodes (d=3 -> d=4)
p(box_el(700, 106, 120, 52, BLUE_SOFT, BLUE_SSTR, [
    ("br-0.0.0", 14.8, 700, BLUE_TXT),
    ('Tests "2"', 13.8, 600, BLUE_TXT),
], rx=6.5, sw=1.45))

p(box_el(828, 90, 80, 28, CORAL_FILL, CORAL_STR, [("✗ 8K doc", 13.8, 700, CORAL_TXT)], rx=14, sw=1.4, dash="3.8 2.5"))
p(box_el(700, 166, 208, 27, CORAL_SOFT, CORAL_STR, [("rewind(k=1) · restores h_{1}", 13.8, 700, "#991B1B")], rx=6, sw=1.3))

p(box_el(700, 236, 82, 62, BLUE_SOFT, BLUE_SSTR, [
    ("br-0.1.0", 14.4, 700, BLUE_TXT),
    ("d = 3", 13.8, 600, BLUE_TXT),
], rx=6.5, sw=1.45))

p(box_el(792, 224, 116, 76, AMBER_FILL, AMBER_STR, [
    ("br-0.1.0.0 (R)", 14.6, 700, AMBER_TXT),
    ("0-copy read(S)", 13.8, 700, "#9A3412"),
    ("GDN skips K_{S}", 13.4, 600, AMBER_TXT),
], rx=6.5, sw=1.6))

p(box_el(700, 310, 208, 28, MINT_FILL, MINT_STR, [("OWN · 0% hijack · 0 pages", 13.8, 700, MINT_TXT)], rx=6, sw=1.35))

# Col 4 Nodes (d=5 & Root Join)
p(box_el(932, 104, 146, 54, "#EEF8F5", MINT_STR, [
    ("br-0.0.0.0.0", 14.6, 700, MINT_TXT),
    ("Verify report", 13.8, 600, "#1D7A67"),
], rx=6.5, sw=1.45))

p(box_el(932, 174, 146, 74, MINT_DEEP, "#1D7A67", [
    ("br-0.1.0.0.0 ★", 14.8, 700, MINT_TXT),
    ("open(5544925)", 13.8, 700, MINT_TXT),
    ("✓ Gold USGS hit", 13.6, 700, "#0B4F41"),
], rx=6.5, sw=1.7))

p(box_el(932, 266, 146, 72, "#1D7A67", "#0C4A3E", [
    ("Root c_{join} → done", 14.8, 700, "#FFFFFF"),
    ("Close gate (m=0)", 13.6, 600, "#E6F7F3"),
    ("Verified · R = +1", 13.8, 700, "#FFFFFF"),
], rx=7, sw=1.6))

# Edges across DAG in Panel (b)
# Root -> branch-0
p(f'  <line x1="456" y1="156" x2="456" y2="178" stroke="#1D7A67" stroke-width="1.9"/>')
p(f'  {chevron(456, 178, "down", "#1D7A67", 1.75, 3.6)}')

# branch-0 -> branch-0.0 (top) & branch-0.1 (bottom)
p(f'  <path d="M 516 202 L 523 202 Q 528 202 528 197 L 528 140 Q 528 135 533 135 L 538 135" fill="none" stroke="{PEACH_STR}" stroke-width="1.9"/>')
p(f'  {chevron(538, 135, "right", PEACH_STR, 1.75, 3.6)}')

p(f'  <path d="M 516 222 L 523 222 Q 528 222 528 227 L 528 264 Q 528 269 533 269 L 538 269" fill="none" stroke="{BLUE_STR}" stroke-width="1.9"/>')
p(f'  {chevron(538, 269, "right", BLUE_STR, 1.75, 3.6)}')

# branch-0.0 -> br-0.0.0
p(f'  <line x1="676" y1="132" x2="698" y2="132" stroke="{BLUE_STR}" stroke-width="1.85"/>')
p(f'  {chevron(698, 132, "right", BLUE_STR, 1.7, 3.6)}')

# br-0.0.0 -> 8K dead-end & rewind loop back
p(f'  <path d="M 800 106 Q 812 104 826 104" fill="none" stroke="{CORAL_STR}" stroke-width="1.6" stroke-dasharray="3.5 2.5"/>')
p(f'  <path d="M 845 90 Q 795 76 756 104" fill="none" stroke="{CORAL_STR}" stroke-width="1.75"/>')
p(f'  {chevron(756, 104, "down", CORAL_STR, 1.65, 3.3)}')

# br-0.0.0 -> br-0.0.0.0.0 (d=5)
p(f'  <line x1="820" y1="134" x2="930" y2="134" stroke="{MINT_STR}" stroke-width="1.85"/>')
p(f'  {chevron(930, 134, "right", MINT_STR, 1.7, 3.6)}')

# branch-0.1 -> br-0.1.0 -> br-0.1.0.0 (R)
p(f'  <line x1="676" y1="268" x2="698" y2="268" stroke="{BLUE_STR}" stroke-width="1.85"/>')
p(f'  {chevron(698, 268, "right", BLUE_STR, 1.7, 3.6)}')
p(f'  <line x1="782" y1="265" x2="790" y2="265" stroke="{BLUE_STR}" stroke-width="1.85"/>')
p(f'  {chevron(790, 265, "right", BLUE_STR, 1.7, 3.5)}')

# Cross-branch 0-Copy KV Mount Edge from branch-0.0 (S) bottom (608, 170) to br-0.1.0.0 (R) top (850, 222)
p(f'  <path d="M 608 170 L 608 197 Q 608 205 616 205 L 842 205 Q 850 205 850 213 L 850 222" fill="none" stroke="{AMBER_STR}" stroke-width="2.2" stroke-dasharray="5.0 3.0"/>')
p(f'  {chevron(850, 222, "down", AMBER_STR, 1.85, 3.8)}')
p(box_el(618, 194, 70, 21, AMBER_FILL, AMBER_STR, [("0-copy", 13.4, 700, AMBER_TXT)], rx=10.5, sw=1.25))

# br-0.1.0.0 (R) -> br-0.1.0.0.0 (d=5 gold hit)
p(f'  <path d="M 908 248 L 916 248 Q 920 248 920 243 L 920 216 Q 920 211 925 211 L 930 211" fill="none" stroke="#1D7A67" stroke-width="2.0"/>')
p(f'  {chevron(930, 211, "right", "#1D7A67", 1.8, 3.6)}')

# d=5 gold hit -> Root c_join -> done
p(f'  <line x1="1005" y1="248" x2="1005" y2="264" stroke="#0C4A3E" stroke-width="2.1"/>')
p(f'  {chevron(1005, 264, "down", "#0C4A3E", 1.85, 3.8)}')

# Bottom Summary Metric Pills
p(box_el(388, 369, 232, 34, BLUE_SOFT,  BLUE_STR,  [("199.0K tree tok (6.1× 32K)", 14.8, 700, BLUE_TXT)], rx=7, sw=1.4))
p(box_el(628, 369, 222, 34, AMBER_FILL, AMBER_STR, [("Max branch 16.4K ≤ 32K", 14.8, 700, AMBER_TXT)], rx=7, sw=1.4))
p(box_el(858, 369, 228, 34, MINT_FILL,  MINT_STR,  [("Crit. path 9.9K (1.94×↓)", 14.8, 700, MINT_TXT)], rx=7, sw=1.45))

p(f'  {text_el(737, 427, "Sub-critical tree (μ = 0.67 < 1) · clean root + sibling 0-copy KV mount", size=14.4, weight=600, fill="#0B4F41")}')
p('</g>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved Figure 8 SVG & PDF: {PDF_OUT}")
