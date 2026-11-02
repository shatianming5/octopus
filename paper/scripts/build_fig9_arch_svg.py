#!/usr/bin/env python3
"""
Builds Paper-Native High-Typography Figure 9 (`figures/fig_arch_hybrid_gdn_mamba.{svg,pdf}`):
- Exact 1100 x 688 canvas (@ 6.50 in = 468 pt \textwidth -> 1 px = 0.4255 pt, printed height = 292.7 pt):
  - Panel titles `(a)/(b)`: 19.2px Bold (~8.2pt Inter vs old 4.0pt!)
  - Zone headers: 15.0-15.2px Bold (~6.4-6.5pt Inter vs old 3.1pt!)
  - Module pills & hook labels: 13.4-14.4px Bold/SemiBold (~5.7-6.1pt Inter vs old 2.6-2.9pt!)
- Modular Swiss-grid architecture blueprints for (a) Qwen3.5 (24 GDN + 8 Gated Attn) and
  (b) Granite-4.0-H (Mamba-2 SSD + GQA Attn + MoE), 100% pure Inter 4.0 CID TrueType PDF (0 Type 3 fonts).
"""
from pathlib import Path
from svg_to_pdf import format_rich_tspans, render_svg_to_pdf

ROOT = Path(__file__).resolve().parent.parent
SVG_OUT = ROOT / "figures/fig_arch_hybrid_gdn_mamba.svg"
PDF_OUT = ROOT / "figures/fig_arch_hybrid_gdn_mamba.pdf"
PNG_CHECK = Path("/tmp/octopus-fig-preview/scratch/fig9_v2_check.png")

W, H = 1100, 688
FONT = "Inter"

C_TEAL_BG,  C_TEAL_STR  = "#F7FBFA", "#6BA396"
C_STONE_BG, C_STONE_STR = "#FAF9F6", "#D5D0C7"
C_INK       = "#18181B"
C_MUTED     = "#3F3F46"
C_STROKE    = "#524E49"

BLUE_FILL,  BLUE_STR,  BLUE_TXT  = "#BDD9F2", "#4B749F", "#142E49"
BLUE_SOFT,  BLUE_SSTR            = "#EEF5FC", "#5B84B1"
MINT_FILL,  MINT_STR,  MINT_TXT  = "#D8F3EC", "#2A7A60", "#0B382B"
PEACH_FILL, PEACH_STR, PEACH_TXT = "#FBD6B6", "#B56B36", "#47230A"
AMBER_FILL, AMBER_STR, AMBER_TXT = "#FEF3C7", "#C87524", "#78350F"
CORAL_FILL, CORAL_STR, CORAL_TXT = "#FEE2E2", "#C83E3E", "#7F1D1D"
LAV_FILL,   LAV_STR,   LAV_TXT   = "#EAE4FA", "#7054B3", "#3F2A75"
PINK_FILL,  PINK_STR,  PINK_TXT  = "#FCE7F3", "#9D174D", "#831843"
GREY_FILL,  GREY_STR,  GREY_TXT  = "#F4F4F5", "#8C857B", "#3F3F46"

out: list[str] = []


def p(s: str) -> None:
    out.append(s)


def text_el(x: float, y: float, content: str, size: float = 14.5, weight: int = 600,
            fill: str = C_INK, anchor: str = "middle", ls: str = "-0.20") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}">'
        f'{format_rich_tspans(content, size)}</text>'
    )


def box_el(x: float, y: float, w: float, h: float, fill: str, stroke: str,
           lines: list[tuple[str, float, int, str]] | None = None,
           rx: float = 6.0, sw: float = 1.4, dash: str | None = None) -> str:
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
            parts.append(text_el(x + w / 2, y + h * 0.36 + s1 * 0.33, t1, size=s1, weight=w1, fill=c1))
            parts.append(text_el(x + w / 2, y + h * 0.74 + s2 * 0.30, t2, size=s2, weight=w2, fill=c2))
        elif n == 3:
            (t1, s1, w1, c1), (t2, s2, w2, c2), (t3, s3, w3, c3) = lines
            parts.append(text_el(x + w / 2, y + h * 0.23 + s1 * 0.34, t1, size=s1, weight=w1, fill=c1))
            parts.append(text_el(x + w / 2, y + h * 0.53 + s2 * 0.32, t2, size=s2, weight=w2, fill=c2))
            parts.append(text_el(x + w / 2, y + h * 0.82 + s3 * 0.28, t3, size=s3, weight=w3, fill=c3))
    return "\n".join(parts)


def up_arrow(x: float, y_bot: float, y_top: float, col: str = C_STROKE, sw: float = 1.55) -> str:
    s = 3.2
    return (
        f'<line x1="{x:.1f}" y1="{y_bot:.1f}" x2="{x:.1f}" y2="{y_top:.1f}" stroke="{col}" stroke-width="{sw}"/>'
        f'<path d="M {x - s:.1f} {y_top + s*1.15:.1f} L {x:.1f} {y_top:.1f} L {x + s:.1f} {y_top + s*1.15:.1f}" '
        f'fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


p(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none">')
p(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ==============================================================================
# CARD (a): Qwen3.5 Hybrid Blueprint (24 GDN + 8 Gated Full-Attn) — y=2, h=336
# ==============================================================================
YA = 2
p('<g id="Panel A - Qwen3.5 Hybrid Blueprint">')
p(f'  <rect x="1" y="{YA}" width="1098" height="336" rx="10" fill="{C_TEAL_BG}" stroke="{C_TEAL_STR}" stroke-width="1.5"/>')
p(f'  {text_el(15, YA + 27, "(a) Qwen3.5 hybrid blueprint (24 GDN + 8 Gated Attn)", size=19.2, weight=700, fill=C_INK, anchor="start", ls="-0.30")}')
p(box_el(742, YA + 8, 344, 25, LAV_FILL, LAV_STR, [("CONV = 3 · 16K-tiled · 10 LoRA targets", 13.8, 700, LAV_TXT)], rx=12.5, sw=1.3))

# --- Zone A1: Left Hybrid Stack (x=13, y=42, w=238, h=284) ---
Z1X, Z1Y = 13, YA + 40
p(f'  <rect x="{Z1X}" y="{Z1Y}" width="238" height="284" rx="7" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.35"/>')
p(f'  {text_el(Z1X + 119, Z1Y + 20, "Hybrid stack (×8 blocks)", size=15.0, weight=700, fill=C_INK)}')

p(box_el(Z1X + 12, Z1Y + 30, 214, 28, PEACH_FILL, PEACH_STR, [("RMSNorm (16K-tiled)", 13.8, 700, PEACH_TXT)], rx=6, sw=1.35))
p(up_arrow(Z1X + 119, Z1Y + 67, Z1Y + 59))
p(box_el(Z1X + 12, Z1Y + 68, 214, 62, "#FFFBEB", AMBER_STR, [
    ("1× Gated Full-Attn", 14.2, 700, AMBER_TXT),
    ("0-copy mount + MLP", 13.5, 700, "#1D7A67"),
], rx=6, sw=1.4, dash="4.2 2.8"))
p(up_arrow(Z1X + 119, Z1Y + 139, Z1Y + 131))
p(box_el(Z1X + 12, Z1Y + 140, 214, 62, BLUE_SOFT, BLUE_SSTR, [
    ("3× Gated DeltaNet", 14.2, 700, BLUE_TXT),
    ("✗ Skip mount · private h_{t}", 13.5, 700, "#991B1B"),
], rx=6, sw=1.4, dash="4.2 2.8"))
p(up_arrow(Z1X + 119, Z1Y + 211, Z1Y + 203))
p(box_el(Z1X + 12, Z1Y + 212, 214, 28, AMBER_FILL, AMBER_STR, [("M-RoPE · Rule K2 s(c'_{B})", 13.6, 700, AMBER_TXT)], rx=6, sw=1.35))
p(up_arrow(Z1X + 119, Z1Y + 249, Z1Y + 241))
p(box_el(Z1X + 12, Z1Y + 250, 214, 26, BLUE_FILL, BLUE_STR, [("Cursor c_{B} / 1D tree T_{tree}", 13.6, 700, BLUE_TXT)], rx=6, sw=1.35))

# --- Zone A2-Top: Gated Full-Attention (8L) (x=261, y=42, w=402, h=168) ---
Z2X = 261
p(f'  <rect x="{Z2X}" y="{Z1Y}" width="402" height="168" rx="7" fill="#FFFDF5" stroke="{AMBER_STR}" stroke-width="1.4"/>')
p(f'  {text_el(Z2X + 12, Z1Y + 20, "Gated Full-Attention (8 layers)", size=15.0, weight=700, fill=AMBER_TXT, anchor="start")}')

p(box_el(Z2X + 11, Z1Y + 29, 380, 28, LAV_FILL, LAV_STR, [("o_proj [+LoRA ΔW_{O}] ← × σ(Gate)", 13.8, 700, LAV_TXT)], rx=5.5, sw=1.35))
p(up_arrow(Z2X + 201, Z1Y + 66, Z1Y + 58))
p(box_el(Z2X + 11, Z1Y + 67, 380, 48, MINT_FILL, MINT_STR, [
    ("0-copy KV mount c_{A}[s:e) · Rule K2", 14.0, 700, MINT_TXT),
    ("q/k_norm + M-RoPE · grad ∂L_{B}/∂K_{A},V_{A}", 13.4, 600, "#1D7A67"),
], rx=5.5, sw=1.45))

p(up_arrow(Z2X + 82,  Z1Y + 125, Z1Y + 116))
p(up_arrow(Z2X + 216, Z1Y + 125, Z1Y + 116))
p(up_arrow(Z2X + 334, Z1Y + 125, Z1Y + 116))
p(box_el(Z2X + 11,  Z1Y + 126, 142, 33, LAV_FILL, LAV_STR, [("q_proj (+Gate)", 13.4, 700, LAV_TXT)], rx=5.5, sw=1.3))
p(box_el(Z2X + 159, Z1Y + 126, 113, 33, LAV_FILL, LAV_STR, [("k_proj [+LoRA]", 13.4, 700, LAV_TXT)], rx=5.5, sw=1.3))
p(box_el(Z2X + 278, Z1Y + 126, 113, 33, LAV_FILL, LAV_STR, [("v_proj [+LoRA]", 13.4, 700, LAV_TXT)], rx=5.5, sw=1.3))

# --- Zone A2-Bottom: 16K-Tiled SwiGLU / MoE (x=261, y=218, w=402, h=108) ---
M1Y = Z1Y + 176
p(f'  <rect x="{Z2X}" y="{M1Y}" width="402" height="108" rx="7" fill="#F3FAF7" stroke="{MINT_STR}" stroke-width="1.4"/>')
p(f'  {text_el(Z2X + 12, M1Y + 20, "16K-tiled SwiGLU / MoE (seq_tiles.py)", size=15.0, weight=700, fill="#0B4F41", anchor="start")}')
p(box_el(Z2X + 11,  M1Y + 30, 186, 32, LAV_FILL, LAV_STR, [("gate/up_proj [+LoRA]", 13.6, 700, LAV_TXT)], rx=5.5, sw=1.3))
p(box_el(Z2X + 205, M1Y + 30, 186, 32, LAV_FILL, LAV_STR, [("down_proj [+LoRA]", 13.6, 700, LAV_TXT)], rx=5.5, sw=1.3))
p(box_el(Z2X + 11,  M1Y + 70, 380, 29, AMBER_FILL, AMBER_STR, [("TILE = 16,384 · saves ~35 GB in 262K tree", 13.6, 700, AMBER_TXT)], rx=5.5, sw=1.35))

# --- Zone A3: Gated DeltaNet (24L) (x=673, y=42, w=413, h=284) ---
Z3X = 673
p(f'  <rect x="{Z3X}" y="{Z1Y}" width="413" height="284" rx="7" fill="{BLUE_SOFT}" stroke="{BLUE_STR}" stroke-width="1.45"/>')
p(f'  {text_el(Z3X + 12, Z1Y + 20, "Gated DeltaNet (24 layers · private h_{t})", size=15.0, weight=700, fill=BLUE_TXT, anchor="start")}')

p(box_el(Z3X + 12, Z1Y + 29, 389, 28, LAV_FILL, LAV_STR, [("out_proj [+LoRA] ← RMSNorm(core) × SiLU(z)", 13.6, 700, LAV_TXT)], rx=5.5, sw=1.35))
p(up_arrow(Z3X + 206, Z1Y + 66, Z1Y + 58))
p(box_el(Z3X + 12, Z1Y + 67, 389, 66, "#FFFFFF", "#2A7A60", [
    ("Chunk Gated Delta Rule (fla tree_recurrence)", 14.2, 700, BLUE_TXT),
    ("fork clones h_{end} · rewind(k) restores h_{k}", 13.5, 600, "#0B4F41"),
    ("✗ Skips mounted spans (2.7% hijack)", 13.8, 700, "#991B1B"),
], rx=6, sw=1.55))

p(up_arrow(Z3X + 115, Z1Y + 142, Z1Y + 134))
p(up_arrow(Z3X + 314, Z1Y + 142, Z1Y + 134))
p(box_el(Z3X + 12,  Z1Y + 143, 206, 46, PINK_FILL, PINK_STR, [
    ("tree_conv (CONV = 3)", 13.8, 700, PINK_TXT),
    ("3 ancestor tokens", 13.2, 600, PINK_TXT),
], rx=5.5, sw=1.35))
p(box_el(Z3X + 226, Z1Y + 143, 175, 46, PEACH_FILL, PEACH_STR, [
    ("Gates β = σ(b)", 13.8, 700, PEACH_TXT),
    ("decay g(a, b_{dt})", 13.2, 600, PEACH_TXT),
], rx=5.5, sw=1.35))

p(up_arrow(Z3X + 88,  Z1Y + 198, Z1Y + 190))
p(up_arrow(Z3X + 225, Z1Y + 198, Z1Y + 190))
p(box_el(Z3X + 12,  Z1Y + 199, 152, 34, LAV_FILL,  LAV_STR,  [("in_proj_qkv [+LoRA]", 13.3, 700, LAV_TXT)], rx=5.5, sw=1.3))
p(box_el(Z3X + 170, Z1Y + 199, 108, 34, GREY_FILL, GREY_STR, [("in_proj_b,a", 13.3, 600, GREY_TXT)], rx=5.5, sw=1.25))
p(box_el(Z3X + 284, Z1Y + 199, 117, 34, LAV_FILL,  LAV_STR,  [("in_proj_z [+LoRA]", 13.3, 700, LAV_TXT)], rx=5.5, sw=1.3))

p(box_el(Z3X + 12, Z1Y + 242, 389, 33, LAV_FILL, LAV_STR, [("SGLang bridge: fuses qkv + z → in_proj_qkvz", 13.6, 700, LAV_TXT)], rx=5.5, sw=1.35))
p('</g>')

# ==============================================================================
# CARD (b): Granite-4.0-H Hybrid Blueprint (Mamba-2 + GQA + MoE) — y=350, h=336
# ==============================================================================
YB = 350
p('<g id="Panel B - Granite-4.0-H Hybrid Blueprint">')
p(f'  <rect x="1" y="{YB}" width="1098" height="336" rx="10" fill="{C_STONE_BG}" stroke="{C_STONE_STR}" stroke-width="1.5"/>')
p(f'  {text_el(15, YB + 27, "(b) Granite-4.0-H hybrid blueprint (Mamba-2 + GQA + MoE)", size=19.2, weight=700, fill=C_INK, anchor="start", ls="-0.30")}')
p(box_el(732, YB + 8, 354, 25, LAV_FILL, LAV_STR, [("Transitive X→A→C · 98.5% agree · 5 LoRA", 13.8, 700, LAV_TXT)], rx=12.5, sw=1.3))

# --- Zone B1: Left Hybrid Stack (x=13, y=390, w=238, h=284) ---
B1Y = YB + 40
p(f'  <rect x="{Z1X}" y="{B1Y}" width="238" height="284" rx="7" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.35"/>')
p(f'  {text_el(Z1X + 119, B1Y + 20, "Hybrid stack (×L blocks)", size=15.0, weight=700, fill=C_INK)}')

p(box_el(Z1X + 12, B1Y + 30, 214, 28, CORAL_FILL, CORAL_STR, [("RMSNorm (patched 1× res.)", 13.6, 700, CORAL_TXT)], rx=6, sw=1.35))
p(up_arrow(Z1X + 119, B1Y + 67, B1Y + 59))
p(box_el(Z1X + 12, B1Y + 68, 214, 62, "#FFFBEB", AMBER_STR, [
    ("GQA Full-Attention", 14.2, 700, AMBER_TXT),
    ("Transitive 0-copy mount", 13.5, 700, "#1D7A67"),
], rx=6, sw=1.4, dash="4.2 2.8"))
p(up_arrow(Z1X + 119, B1Y + 139, B1Y + 131))
p(box_el(Z1X + 12, B1Y + 140, 214, 62, BLUE_SOFT, BLUE_SSTR, [
    ("Mamba-2 SSD Mixer", 14.2, 700, BLUE_TXT),
    ("✗ Skip mount · SSM h_{t}", 13.5, 700, "#991B1B"),
], rx=6, sw=1.4, dash="4.2 2.8"))
p(up_arrow(Z1X + 119, B1Y + 211, B1Y + 203))
p(box_el(Z1X + 12, B1Y + 212, 214, 28, AMBER_FILL, AMBER_STR, [("RoPE · Rule K2 s(c'_{B})", 13.6, 700, AMBER_TXT)], rx=6, sw=1.35))
p(up_arrow(Z1X + 119, B1Y + 249, B1Y + 241))
p(box_el(Z1X + 12, B1Y + 250, 214, 26, BLUE_FILL, BLUE_STR, [("Cursor c_{B} / 1D tree T_{tree}", 13.6, 700, BLUE_TXT)], rx=6, sw=1.35))

# --- Zone B2-Top: GQA Full-Attention (Transitive Mount) (x=261, h=156) ---
p(f'  <rect x="{Z2X}" y="{B1Y}" width="402" height="156" rx="7" fill="#FFFDF5" stroke="{AMBER_STR}" stroke-width="1.4"/>')
p(f'  {text_el(Z2X + 12, B1Y + 20, "GQA Full-Attn (transitive mount)", size=15.0, weight=700, fill=AMBER_TXT, anchor="start")}')

p(box_el(Z2X + 11, B1Y + 28, 380, 27, LAV_FILL, LAV_STR, [("o_proj [+Rank-64 LoRA ΔW_{O}]", 13.8, 700, LAV_TXT)], rx=5.5, sw=1.35))
p(up_arrow(Z2X + 201, B1Y + 63, B1Y + 56))
p(box_el(Z2X + 11, B1Y + 64, 380, 46, MINT_FILL, MINT_STR, [
    ("Transitive 0-copy [KV_{X}  ||  KV_{A}  ||  KV_{C}]", 13.9, 700, MINT_TXT),
    ("Recursive segment(src, s_{0}, s_{1}) · 98.5% agree", 13.3, 600, "#1D7A67"),
], rx=5.5, sw=1.45))

p(up_arrow(Z2X + 72,  B1Y + 118, B1Y + 111))
p(up_arrow(Z2X + 201, B1Y + 118, B1Y + 111))
p(up_arrow(Z2X + 330, B1Y + 118, B1Y + 111))
p(box_el(Z2X + 11,  B1Y + 119, 122, 29, LAV_FILL, LAV_STR, [("q_proj [+LoRA]", 13.4, 700, LAV_TXT)], rx=5.5, sw=1.3))
p(box_el(Z2X + 140, B1Y + 119, 122, 29, LAV_FILL, LAV_STR, [("k_proj [+LoRA]", 13.4, 700, LAV_TXT)], rx=5.5, sw=1.3))
p(box_el(Z2X + 269, B1Y + 119, 122, 29, LAV_FILL, LAV_STR, [("v_proj [+LoRA]", 13.4, 700, LAV_TXT)], rx=5.5, sw=1.3))

# --- Zone B2-Bottom: Parallel Shared MLP + Sparse MoE (x=261, h=120) ---
M2Y = B1Y + 164
p(f'  <rect x="{Z2X}" y="{M2Y}" width="402" height="120" rx="7" fill="#F3FAF7" stroke="{MINT_STR}" stroke-width="1.4"/>')
p(f'  {text_el(Z2X + 12, M2Y + 20, "Parallel Shared MLP + Sparse MoE", size=15.0, weight=700, fill="#0B4F41", anchor="start")}')

p(box_el(Z2X + 11,  M2Y + 28, 186, 46, GREY_FILL, GREY_STR, [
    ("Shared MLP", 13.8, 700, C_INK),
    ("Stateless (no LoRA)", 13.0, 600, GREY_TXT),
], rx=5.5, sw=1.3))
p(box_el(Z2X + 205, M2Y + 28, 186, 46, MINT_FILL, MINT_STR, [
    ("Top-K Sparse MoE", 13.8, 700, MINT_TXT),
    ("1D batched T_{tree}", 13.0, 600, "#1D7A67"),
], rx=5.5, sw=1.35))
p(box_el(Z2X + 11, M2Y + 82, 380, 30, LAV_FILL, LAV_STR, [("LoRA filter: [q/k/v/o_proj, out_proj] + 1× norm", 13.5, 700, LAV_TXT)], rx=5.5, sw=1.35))

# --- Zone B3: Mamba-2 SSD Mixer (mamba2_varlen.py) (x=673, h=284) ---
p(f'  <rect x="{Z3X}" y="{B1Y}" width="413" height="284" rx="7" fill="{BLUE_SOFT}" stroke="{BLUE_STR}" stroke-width="1.45"/>')
p(f'  {text_el(Z3X + 12, B1Y + 20, "Mamba-2 SSD Mixer (mamba2_varlen.py)", size=15.0, weight=700, fill=BLUE_TXT, anchor="start")}')

p(box_el(Z3X + 12, B1Y + 29, 389, 28, LAV_FILL, LAV_STR, [("out_proj [+Patched LoRA H·P → D] ← RMSNormGated", 13.4, 700, LAV_TXT)], rx=5.5, sw=1.35))
p(up_arrow(Z3X + 206, B1Y + 66, B1Y + 58))
p(box_el(Z3X + 12, B1Y + 67, 389, 66, "#FFFFFF", "#2A7A60", [
    ("SSD → fla chunk_simple_gla (tree_recurrence)", 14.0, 700, BLUE_TXT),
    ("q=C, k=B, v=dt·x, g=A·dt · init from h_{parent}", 13.3, 600, "#0B4F41"),
    ("✗ Skips mounted KV spans (100% isolated)", 13.7, 700, "#991B1B"),
], rx=6, sw=1.55))

p(up_arrow(Z3X + 115, B1Y + 142, B1Y + 134))
p(up_arrow(Z3X + 314, B1Y + 142, B1Y + 134))
p(box_el(Z3X + 12,  B1Y + 143, 206, 46, PINK_FILL, PINK_STR, [
    ("tree_conv [x, B, C]", 13.8, 700, PINK_TXT),
    ("d_conv - 1 ancestors", 13.2, 600, PINK_TXT),
], rx=5.5, sw=1.35))
p(box_el(Z3X + 226, B1Y + 143, 175, 46, PEACH_FILL, PEACH_STR, [
    ("Step gate dt", 13.8, 700, PEACH_TXT),
    ("softplus(dt + b_{dt})", 13.2, 600, PEACH_TXT),
], rx=5.5, sw=1.35))

p(up_arrow(Z3X + 206, B1Y + 198, B1Y + 190))
p(box_el(Z3X + 12, B1Y + 199, 389, 34, AMBER_FILL, AMBER_STR, [("5-way split [x, B, C | dt | z] ← Linear (in_proj)", 13.6, 700, AMBER_TXT)], rx=5.5, sw=1.35))
p(box_el(Z3X + 12, B1Y + 242, 389, 33, MINT_FILL, MINT_STR, [("Varlen SSD-to-GLA replaces HF fallback on trees", 13.5, 700, MINT_TXT)], rx=5.5, sw=1.35))
p('</g>')

p('</svg>')

SVG_OUT.write_text("\n".join(out), encoding="utf-8")
render_svg_to_pdf(SVG_OUT, PDF_OUT, W, H, PNG_CHECK)
print(f"Saved Figure 9 SVG & PDF: {PDF_OUT}")
