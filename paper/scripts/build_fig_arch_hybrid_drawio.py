#!/usr/bin/env python3
"""
Generate Draw.io (.drawio) and vector PDF for Figure 11:
Per-Architecture Octopus Integration Blueprints (Part II):
  (a) Qwen3.5 (Dense & MoE Hybrid: 24 Gated DeltaNet + 8 Gated Full-Attention Layers)
  (b) Granite-4.0-H (Mamba-2 SSD State-Space + Full-Attention + MoE/Shared MLP Hybrid)
Styled in the exact modular block-diagram aesthetic as Figure 10 and the reference diagrams.
"""

import base64
import io
import subprocess
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DRAWIO_PATH = FIG_DIR / "fig_arch_hybrid_gdn_mamba.drawio"
PDF_PATH = FIG_DIR / "fig_arch_hybrid_gdn_mamba.pdf"
DRAWIO_BIN = "/Applications/draw.io.app/Contents/MacOS/draw.io"


def b64_img(name: str, max_px: int = 110) -> str:
    p = FIG_DIR / f"{name}.png"
    img = Image.open(p).convert("RGBA")
    img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    q = img.quantize(colors=64, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO()
    q.save(buf, format="PNG", optimize=True)
    return "data:image/png," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_xml() -> str:
    icons = {
        k: b64_img(k)
        for k in [
            "octo_21_title_boss",
            "octo_12_rope_future",
            "octo_17_rewind_cut",
            "octo_18_h200_turbo",
        ]
    }

    def img_cell(cid, key, x, y, w, h):
        return (
            f'<mxCell id="{cid}" value="" style="shape=image;verticalLabelPosition=bottom;'
            f'labelBackgroundColor=default;verticalAlign=top;aspect=fixed;imageAspect=0;image={icons[key]};" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def box(cid, val, x, y, w, h, style):
        return (
            f'<mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def circle_op(cid, sym, cx, cy, r=11):
        return (
            f'<mxCell id="{cid}" value="&lt;b&gt;{sym}&lt;/b&gt;" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;'
            f'fillColor=#FFFFFF;strokeColor=#1E293B;strokeWidth=1.8;fontFamily=Helvetica;fontSize=13.5;fontColor=#0F172A;'
            f'align=center;verticalAlign=middle;" vertex="1" parent="1">'
            f'<mxGeometry x="{cx - r}" y="{cy - r}" width="{2 * r}" height="{2 * r}" as="geometry"/></mxCell>'
        )

    def line_pts(cid, pts, color="#1E293B", width=1.8, dashed=False, arrow=True):
        dash_s = "dashed=1;dashPattern=5 4;" if dashed else ""
        arr_s = "endArrow=block;endFill=1;endSize=5;" if arrow else "endArrow=none;"
        pts_xml = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts[1:-1])
        arr_tag = f'<Array as="points">{pts_xml}</Array>' if len(pts) > 2 else ""
        return (
            f'<mxCell id="{cid}" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;'
            f'jettySize=auto;html=1;strokeColor={color};strokeWidth={width};{dash_s}{arr_s}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
            f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/>'
            f'{arr_tag}</mxGeometry></mxCell>'
        )

    S_CARD = "rounded=1;arcSize=5;fillColor=#F1F5F9;strokeColor=#1E293B;strokeWidth=2.2;whiteSpace=wrap;html=1;"
    S_DEC_BOX = "rounded=1;arcSize=6;fillColor=#DBEAFE;strokeColor=#1E293B;strokeWidth=1.9;whiteSpace=wrap;html=1;"
    S_SUB_ATTN = "rounded=1;arcSize=8;fillColor=#FEF9C3;strokeColor=#B45309;strokeWidth=1.5;dashed=1;whiteSpace=wrap;html=1;"
    S_SUB_REC = "rounded=1;arcSize=8;fillColor=#E0F2FE;strokeColor=#1D4ED8;strokeWidth=1.5;dashed=1;whiteSpace=wrap;html=1;"
    S_ATTN_BOX = "rounded=1;arcSize=6;fillColor=#FEF3C7;strokeColor=#1E293B;strokeWidth=1.9;whiteSpace=wrap;html=1;"
    S_MLP_BOX = "rounded=1;arcSize=6;fillColor=#DCFCE7;strokeColor=#1E293B;strokeWidth=1.9;whiteSpace=wrap;html=1;"
    S_REC_BOX = "rounded=1;arcSize=6;fillColor=#E0F2FE;strokeColor=#1E40AF;strokeWidth=1.9;whiteSpace=wrap;html=1;"

    S_NORM = "rounded=1;arcSize=45;fillColor=#FFEDD5;strokeColor=#1E293B;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_EMB = "rounded=1;arcSize=45;fillColor=#DBEAFE;strokeColor=#1E293B;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_ROPE = "rounded=1;arcSize=35;fillColor=#FEF08A;strokeColor=#1E293B;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10.5;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_ATTN_PILL = "rounded=1;arcSize=45;fillColor=#FDE68A;strokeColor=#B45309;strokeWidth=1.6;fontFamily=Helvetica;fontSize=11.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_REC_PILL = "rounded=1;arcSize=45;fillColor=#BAE6FD;strokeColor=#1D4ED8;strokeWidth=1.6;fontFamily=Helvetica;fontSize=11.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_MLP_PILL = "rounded=1;arcSize=45;fillColor=#BBF7D0;strokeColor=#1E293B;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_LIN = "rounded=1;arcSize=26;fillColor=#E0E7FF;strokeColor=#1E293B;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10.5;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_IFACE = "rounded=1;arcSize=18;fillColor=#DCFCE7;strokeColor=#0F766E;strokeWidth=1.8;fontFamily=Helvetica;fontSize=10.5;fontColor=#064E3B;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_KERNEL = "rounded=1;arcSize=16;fillColor=#DBEAFE;strokeColor=#1D4ED8;strokeWidth=1.8;fontFamily=Helvetica;fontSize=10.5;fontColor=#1E3A8A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_CONV = "rounded=1;arcSize=24;fillColor=#FCE7F3;strokeColor=#9D174D;strokeWidth=1.6;fontFamily=Helvetica;fontSize=10.5;fontColor=#831843;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_TXT = "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=12.0;fontColor=#0F172A;align=center;verticalAlign=middle;"
    S_TITLE_LEFT = "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=13.0;fontColor=#0F172A;align=left;verticalAlign=middle;"
    S_BADGE_LORA = "rounded=1;arcSize=28;fillColor=#EDE9FE;strokeColor=#6D28D9;strokeWidth=1.4;fontFamily=Helvetica;fontSize=10.0;fontColor=#4C1D95;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"

    cells = []

    def build_hybrid_stack(prefix, y0, is_granite=False):
        cx = 162
        sx = 282
        cells.append(box(f"{prefix}_out", "&lt;b&gt;Output / Pair-Deduplicated LM Head&lt;/b&gt;", cx - 125, y0 + 44, 250, 20, S_TXT))
        fn_lbl = (
            "&lt;b&gt;RMSNorm&lt;/b&gt; &lt;font style=&quot;font-size:10px;color:#0F766E;&quot;&gt;(16K-Tiled)&lt;/font&gt;"
            if not is_granite else
            "&lt;b&gt;RMSNorm&lt;/b&gt; &lt;font style=&quot;font-size:10px;color:#DC2626;&quot;&gt;[Patched 1× Residual]&lt;/font&gt;"
        )
        cells.append(box(f"{prefix}_fnorm", fn_lbl, cx - 102, y0 + 70, 204, 26, S_NORM))
        cells.append(line_pts(f"{prefix}_e_out", [(cx, y0 + 70), (cx, y0 + 64)]))

        blk_l = "Hybrid Block × 8" if not is_granite else "Hybrid Stack × L"
        blk_r = "(32L: 24 GDN + 8 Attn)" if not is_granite else "(Mamba-2 + Full-Attn)"
        cells.append(box(f"{prefix}_dlbl_l", f"&lt;b style=&quot;font-size:11.5px;&quot;&gt;{blk_l}&lt;/b&gt;", 32, y0 + 98, 122, 18, "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=11.5;fontColor=#0F172A;align=right;verticalAlign=middle;"))
        cells.append(box(f"{prefix}_dlbl_r", f"&lt;b style=&quot;font-size:11px;color:#0F766E;&quot;&gt;{blk_r}&lt;/b&gt;", 170, y0 + 98, 142, 18, "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=11.0;fontColor=#0F766E;align=left;verticalAlign=middle;"))
        cells.append(box(f"{prefix}_dbox", "", 34, y0 + 120, 274, 268, S_DEC_BOX))
        cells.append(line_pts(f"{prefix}_e_fn", [(cx, y0 + 120), (cx, y0 + 96)]))

        cells.append(box(f"{prefix}_sub_at", "", 42, y0 + 128, 258, 118, S_SUB_ATTN))
        cells.append(box(f"{prefix}_sat_lbl_l", "&lt;b style=&quot;font-size:9.5px;color:#92400E;&quot;&gt;Full-Attn (×1)&lt;/b&gt;", 46, y0 + 130, 104, 14, S_TITLE_LEFT))
        cells.append(box(f"{prefix}_sat_lbl_r", "&lt;b style=&quot;font-size:9.5px;color:#0F766E;&quot;&gt;[0-Copy Mount]&lt;/b&gt;", 174, y0 + 130, 120, 14, S_TITLE_LEFT))
        cells.append(circle_op(f"{prefix}_a_add2", "+", cx, y0 + 156, r=9))
        ffn_top = "&lt;b&gt;MLP (16K-Tiled)&lt;/b&gt; / &lt;b&gt;MoE&lt;/b&gt;" if not is_granite else "&lt;b&gt;MoE + Shared MLP&lt;/b&gt;"
        cells.append(box(f"{prefix}_a_ffn", ffn_top, cx - 88, y0 + 170, 176, 22, S_MLP_PILL))
        cells.append(circle_op(f"{prefix}_a_add1", "+", cx, y0 + 204, r=9))
        at_pill = "&lt;b&gt;Gated Full-Attn&lt;/b&gt; + RMSNorm" if not is_granite else "&lt;b&gt;GQA Full-Attn&lt;/b&gt; + RMSNorm"
        cells.append(box(f"{prefix}_a_at", at_pill, cx - 88, y0 + 218, 176, 22, S_ATTN_PILL))
        cells.append(line_pts(f"{prefix}_ea1", [(cx, y0 + 218), (cx, y0 + 213)], arrow=False))
        cells.append(line_pts(f"{prefix}_ea2", [(cx, y0 + 195), (cx, y0 + 192)], arrow=False))
        cells.append(line_pts(f"{prefix}_ea3", [(cx, y0 + 170), (cx, y0 + 165)], arrow=False))
        cells.append(line_pts(f"{prefix}_ea_sk1", [(cx, y0 + 243), (sx, y0 + 243), (sx, y0 + 204), (cx + 9, y0 + 204)]))
        cells.append(line_pts(f"{prefix}_ea_sk2", [(cx, y0 + 195), (sx, y0 + 195), (sx, y0 + 156), (cx + 9, y0 + 156)]))

        cells.append(box(f"{prefix}_sub_rc", "", 42, y0 + 256, 258, 124, S_SUB_REC))
        rc_hdr_l = "GDN Layer (×3)" if not is_granite else "Mamba-2 (×m)"
        rc_hdr_r = "[Skip Mount · h_t]" if not is_granite else "[Skip Mount · SSM]"
        cells.append(box(f"{prefix}_src_lbl_l", f"&lt;b style=&quot;font-size:9.5px;color:#1E40AF;&quot;&gt;{rc_hdr_l}&lt;/b&gt;", 46, y0 + 258, 106, 14, S_TITLE_LEFT))
        cells.append(box(f"{prefix}_src_lbl_r", f"&lt;b style=&quot;font-size:9.5px;color:#DC2626;&quot;&gt;{rc_hdr_r}&lt;/b&gt;", 174, y0 + 258, 122, 14, S_TITLE_LEFT))
        cells.append(circle_op(f"{prefix}_r_add2", "+", cx, y0 + 284, r=9))
        cells.append(box(f"{prefix}_r_ffn", ffn_top, cx - 88, y0 + 298, 176, 22, S_MLP_PILL))
        cells.append(circle_op(f"{prefix}_r_add1", "+", cx, y0 + 332, r=9))
        rc_pill = "&lt;b&gt;Gated DeltaNet&lt;/b&gt; + RMSNorm" if not is_granite else "&lt;b&gt;Mamba-2 Mixer&lt;/b&gt; + RMSNorm"
        cells.append(box(f"{prefix}_r_rc", rc_pill, cx - 88, y0 + 348, 176, 24, S_REC_PILL))
        cells.append(line_pts(f"{prefix}_er0", [(cx, y0 + 275), (cx, y0 + 240)]))
        cells.append(line_pts(f"{prefix}_er1", [(cx, y0 + 348), (cx, y0 + 341)], arrow=False))
        cells.append(line_pts(f"{prefix}_er2", [(cx, y0 + 323), (cx, y0 + 320)], arrow=False))
        cells.append(line_pts(f"{prefix}_er3", [(cx, y0 + 298), (cx, y0 + 293)], arrow=False))
        cells.append(line_pts(f"{prefix}_er_sk1", [(cx, y0 + 376), (sx, y0 + 376), (sx, y0 + 332), (cx + 9, y0 + 332)]))
        cells.append(line_pts(f"{prefix}_er_sk2", [(cx, y0 + 323), (sx, y0 + 323), (sx, y0 + 284), (cx + 9, y0 + 284)]))

        cells.append(box(f"{prefix}_rope", "&lt;b&gt;M-RoPE / RoPE&lt;/b&gt; &lt;font style=&quot;font-size:10px;color:#92400E;&quot;&gt;(Rule K2 s(c&apos;&lt;sub&gt;B&lt;/sub&gt;))&lt;/font&gt;", cx - 102, y0 + 396, 204, 24, S_ROPE))
        cells.append(line_pts(f"{prefix}_e_rp", [(cx, y0 + 396), (cx, y0 + 372)]))
        cells.append(box(f"{prefix}_emb", "&lt;b&gt;Embedding&lt;/b&gt; &lt;font style=&quot;font-size:10px;&quot;&gt;(1D Prefix-Tree T&lt;sub&gt;tree&lt;/sub&gt;)&lt;/font&gt;", cx - 102, y0 + 426, 204, 24, S_EMB))
        cells.append(line_pts(f"{prefix}_e_em", [(cx, y0 + 426), (cx, y0 + 420)], arrow=False))
        cells.append(box(f"{prefix}_in", "&lt;b&gt;Input Tokens (Cursor c&lt;sub&gt;B&lt;/sub&gt; / Tree T&lt;sub&gt;tree&lt;/sub&gt;)&lt;/b&gt;", cx - 125, y0 + 454, 250, 18, S_TXT))
        cells.append(line_pts(f"{prefix}_e_in", [(cx, y0 + 454), (cx, y0 + 450)], arrow=False))

    # =========================================================================
    # PANEL (A): QWEN3.5 (DENSE & MoE HYBRID: 24 GDN + 8 FULL-ATTN) — y0 = 10
    # =========================================================================
    yA = 10
    cells.append(box("card_a", "", 12, yA, 1576, 476, S_CARD))
    cells.append(img_cell("ic_a", "octo_21_title_boss", 18, yA + 6, 30, 30))
    cells.append(box(
        "hdr_a",
        "&lt;b style=&quot;font-size:15.5px;color:#0F172A;&quot;&gt;(a) Qwen3.5 (Dense &amp; MoE Hybrid) Architecture &amp; Octopus Integration Blueprint&lt;/b&gt; "
        "&lt;font style=&quot;font-size:12.5px;color:#0F766E;&quot;&gt;&lt;b&gt;(24 Gated DeltaNet + 8 Full-Attention · 2.7% Hijack · 16K-Tiled)&lt;/b&gt;&lt;/font&gt;",
        52, yA + 8, 1140, 28, S_TITLE_LEFT
    ))
    cells.append(box("tag_a", "&lt;b&gt;qwen3_5 / qwen3_5_moe · CONV_CONTEXT = 3 · 10 LoRA Targets&lt;/b&gt;", 1202, yA + 9, 370, 26, S_BADGE_LORA))

    build_hybrid_stack("ha", yA, is_granite=False)

    # Panel (a) Middle-Top: Gated Full-Attention (8 Layers) (x=324..904, y=yA+54..yA+286)
    ax0, ay0 = 324, yA + 54
    acx = ax0 + 290  # 614
    cells.append(box("ha_at_title", "&lt;b style=&quot;color:#92400E;&quot;&gt;Gated Full-Attention (8 Layers)&lt;/b&gt;", ax0, ay0 - 20, 240, 18, S_TITLE_LEFT))
    cells.append(box("ha_at_out", "&lt;b&gt;Output&lt;/b&gt;", acx - 35, ay0 - 20, 70, 18, S_TXT))
    cells.append(box("ha_at_box", "", ax0, ay0, 580, 232, S_ATTN_BOX))

    cells.append(box("ha_oproj", "&lt;b&gt;Linear (o_proj)&lt;/b&gt; &lt;font style=&quot;color:#4C1D95;&quot;&gt;&lt;b&gt;[+LoRA ΔW&lt;sub&gt;O&lt;/sub&gt; · 16K-Tiled]&lt;/b&gt;&lt;/font&gt;", ax0 + 80, ay0 + 12, 420, 26, S_LIN))
    cells.append(line_pts("ha_e_op_out", [(acx, ay0 + 12), (acx, ay0 - 2)]))

    cells.append(circle_op("ha_sgate", "×", acx, ay0 + 54, r=10))
    cells.append(line_pts("ha_e_sg_op", [(acx, ay0 + 44), (acx, ay0 + 38)]))
    cells.append(box("ha_sig_pill", "&lt;b&gt;σ(Gate)&lt;/b&gt; (sigmoid_gate)", ax0 + 20, ay0 + 42, 148, 24, S_NORM))
    cells.append(line_pts("ha_e_sig_x", [(ax0 + 168, ay0 + 54), (acx - 10, ay0 + 54)]))

    cells.append(box(
        "ha_iface",
        "&lt;b&gt;Attention Interface (All 8 Full-Attn Layers Mount):&lt;/b&gt;&lt;br&gt;"
        "&lt;font style=&quot;font-size:10px;&quot;&gt;0-Copy Splices &lt;b&gt;c&lt;sub&gt;A&lt;/sub&gt;[s:e)&lt;/b&gt; at Native Pos · Rule K2 Jump &lt;b&gt;s(c&apos;&lt;sub&gt;B&lt;/sub&gt;)&lt;/b&gt; · LoRA Grad &lt;b&gt;∂L&lt;sub&gt;B&lt;/sub&gt;/∂K&lt;sub&gt;A&lt;/sub&gt;,V&lt;sub&gt;A&lt;/sub&gt;&lt;/b&gt;&lt;/font&gt;",
        ax0 + 150, ay0 + 74, 414, 42, S_IFACE
    ))
    cells.append(line_pts("ha_e_if_sg", [(acx, ay0 + 74), (acx, ay0 + 64)]))

    qx, kx, vx = ax0 + 210, ax0 + 356, ax0 + 496
    cells.append(box("ha_qnorm", "&lt;b&gt;q_norm + M-RoPE&lt;/b&gt;", qx - 66, ay0 + 130, 132, 26, S_NORM))
    cells.append(box("ha_knorm", "&lt;b&gt;k_norm + M-RoPE&lt;/b&gt;", kx - 66, ay0 + 130, 132, 26, S_NORM))

    cells.append(box("ha_qlin", "&lt;b&gt;Linear (q_proj → Q, Gate)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9.5px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;Q&lt;/sub&gt; · 2×head_dim]&lt;/font&gt;", ax0 + 24, ay0 + 170, 210, 36, S_LIN))
    cells.append(box("ha_klin", "&lt;b&gt;Linear (k_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9.5px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;K&lt;/sub&gt;]&lt;/font&gt;", kx - 66, ay0 + 170, 132, 36, S_LIN))
    cells.append(box("ha_vlin", "&lt;b&gt;Linear (v_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9.5px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;V&lt;/sub&gt;]&lt;/font&gt;", vx - 66, ay0 + 170, 132, 36, S_LIN))

    cells.append(line_pts("ha_e_ql_sig", [(ax0 + 94, ay0 + 170), (ax0 + 94, ay0 + 66)]))
    cells.append(line_pts("ha_e_ql_qn", [(qx, ay0 + 170), (qx, ay0 + 156)]))
    cells.append(line_pts("ha_e_kl_kn", [(kx, ay0 + 170), (kx, ay0 + 156)]))
    cells.append(line_pts("ha_e_qn_if", [(qx, ay0 + 130), (qx, ay0 + 116)]))
    cells.append(line_pts("ha_e_kn_if", [(kx, ay0 + 130), (kx, ay0 + 116)]))
    cells.append(line_pts("ha_e_vl_if", [(vx, ay0 + 170), (vx, ay0 + 116)]))
    cells.append(line_pts("ha_e_in_k", [(kx, ay0 + 224), (kx, ay0 + 206)]))
    cells.append(line_pts("ha_e_in_q", [(kx, ay0 + 216), (ax0 + 129, ay0 + 216), (ax0 + 129, ay0 + 206)]))
    cells.append(line_pts("ha_e_in_v", [(kx, ay0 + 216), (vx, ay0 + 216), (vx, ay0 + 206)]))

    # Panel (a) Middle-Bottom: 16K Sequence-Tiled SwiGLU MLP / Sparse MoE (x=324..904, y=yA+312..yA+468)
    my0 = yA + 312
    cells.append(box("ha_mlp_title", "&lt;b style=&quot;color:#065F46;&quot;&gt;16K Sequence-Tiled SwiGLU MLP / MoE (seq_tiles.py)&lt;/b&gt;", ax0, my0 - 20, 440, 18, S_TITLE_LEFT))
    cells.append(box("ha_mlp_box", "", ax0, my0, 580, 156, S_MLP_BOX))

    cells.append(box("ha_down", "&lt;b&gt;Linear (down_proj)&lt;/b&gt; &lt;font style=&quot;color:#4C1D95;&quot;&gt;&lt;b&gt;[+LoRA ΔW&lt;sub&gt;down&lt;/sub&gt; · Checkpointed 16K Tile Concat]&lt;/b&gt;&lt;/font&gt;", ax0 + 70, my0 + 12, 440, 26, S_LIN))
    cells.append(circle_op("ha_mmul", "×", acx, my0 + 54, r=9))
    cells.append(line_pts("ha_e_mm_dn", [(acx, my0 + 45), (acx, my0 + 38)]))

    hgx, hux = ax0 + 145, ax0 + 435
    cells.append(box("ha_msilu", "&lt;b&gt;Fused SiLU (torch.compile)&lt;/b&gt;", hgx - 85, my0 + 66, 170, 22, S_NORM))
    cells.append(box("ha_mgate", "&lt;b&gt;Gate: gate_proj [+LoRA]&lt;/b&gt;", hgx - 95, my0 + 98, 190, 26, S_LIN))
    cells.append(box("ha_mup", "&lt;b&gt;Up: up_proj [+LoRA]&lt;/b&gt;", hux - 95, my0 + 98, 190, 26, S_LIN))
    cells.append(line_pts("ha_e_mg_ms", [(hgx, my0 + 98), (hgx, my0 + 88)]))
    cells.append(line_pts("ha_e_ms_mm", [(hgx, my0 + 66), (hgx, my0 + 54), (acx - 9, my0 + 54)]))
    cells.append(line_pts("ha_e_mu_mm", [(hux, my0 + 98), (hux, my0 + 54), (acx + 9, my0 + 54)]))
    cells.append(box(
        "ha_tile_bar",
        "&lt;b&gt;tiled(fn, TILE=16,384):&lt;/b&gt; Splits 262K-token tree into 16K views (saves ~35 GB activation peak in backward!)",
        ax0 + 20, my0 + 128, 540, 22, S_ROPE
    ))

    # Panel (a) Right: Gated DeltaNet (24 Layers) (x=922..1568, y=yA+54..yA+468)
    gx0, gy0 = 922, yA + 54
    gcx = gx0 + 323  # 1245
    cells.append(box("ha_gdn_title", "&lt;b style=&quot;color:#1E40AF;&quot;&gt;Gated DeltaNet (24 Layers: Private h&lt;sub&gt;t&lt;/sub&gt;)&lt;/b&gt;", gx0, gy0 - 20, 275, 18, S_TITLE_LEFT))
    cells.append(box("ha_gdn_out", "&lt;b&gt;Output&lt;/b&gt;", gcx - 35, gy0 - 20, 70, 18, S_TXT))
    cells.append(box("ha_gdn_box", "", gx0, gy0, 646, 414, S_REC_BOX))

    cells.append(box("ha_gout", "&lt;b&gt;Linear (out_proj)&lt;/b&gt; &lt;font style=&quot;color:#4C1D95;&quot;&gt;&lt;b&gt;[+Rank-64 LoRA ΔW&lt;sub&gt;out&lt;/sub&gt; · 16K-Tiled]&lt;/b&gt;&lt;/font&gt;", gx0 + 103, gy0 + 14, 440, 28, S_LIN))
    cells.append(line_pts("ha_e_go_out", [(gcx, gy0 + 14), (gcx, gy0 - 2)]))

    cells.append(box(
        "ha_gnorm",
        "&lt;b&gt;Qwen3_5RMSNormGated (Fused fp32):&lt;/b&gt; RMSNorm(core) ⊗ SiLU(z)",
        gx0 + 83, gy0 + 56, 480, 30, S_NORM
    ))
    cells.append(line_pts("ha_e_gn_go", [(gcx, gy0 + 56), (gcx, gy0 + 42)]))

    cells.append(box(
        "ha_gdelta",
        "&lt;b style=&quot;font-size:11.5px;color:#1E3A8A;&quot;&gt;Chunk Gated Delta Rule (gdn_varlen.py · fla tree_recurrence · SEG=32K)&lt;/b&gt;&lt;br&gt;"
        "&lt;font style=&quot;font-size:10px;color:#0F172A;&quot;&gt;• &lt;b&gt;fork:&lt;/b&gt; Clones parent state &lt;b&gt;h&lt;sub&gt;end&lt;/sub&gt;(c&lt;sub&gt;fork&lt;/sub&gt;)&lt;/b&gt;  "
        "• &lt;b&gt;rewind(k):&lt;/b&gt; Restores exact checkpoint &lt;b&gt;h&lt;sub&gt;k&lt;/sub&gt;&lt;/b&gt;&lt;/font&gt;&lt;br&gt;"
        "&lt;font style=&quot;font-size:10px;color:#DC2626;&quot;&gt;• &lt;b&gt;read_thread (Mount): [✗ Skips Mounted Sibling Spans Completely]&lt;/b&gt; → h&lt;sub&gt;B&lt;/sub&gt; 100% Private (2.7% Hijack!)&lt;/font&gt;",
        gx0 + 24, gy0 + 104, 598, 66, S_KERNEL
    ))
    cells.append(line_pts("ha_e_gd_gn", [(gcx - 40, gy0 + 104), (gcx - 40, gy0 + 86)]))

    qkv_x, ba_x, zx = gx0 + 165, gx0 + 392, gx0 + 560
    cells.append(box("ha_gsplit", "&lt;b&gt;Split [Q, K, V] + Q/K L2-Norm&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9.5px;&quot;&gt;GVA Repeat (num_v_heads / num_k_heads)&lt;/font&gt;", qkv_x - 135, gy0 + 188, 270, 34, S_ROPE))
    cells.append(box("ha_gconv", "&lt;b&gt;Causal Conv1d + SiLU (tree_conv)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;&quot;&gt;Gathers &lt;b&gt;CONV_CONTEXT = 3&lt;/b&gt; Ancestor Tokens!&lt;/font&gt;", qkv_x - 135, gy0 + 236, 270, 36, S_CONV))
    cells.append(line_pts("ha_e_gc_gs", [(qkv_x, gy0 + 236), (qkv_x, gy0 + 222)]))
    cells.append(line_pts("ha_e_gs_gd", [(qkv_x, gy0 + 188), (qkv_x, gy0 + 170)]))

    cells.append(box("ha_gba_op", "&lt;b&gt;Gates: β = σ(b)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9.5px;&quot;&gt;g = −e&lt;sup&gt;A&lt;/sup&gt;·softplus(a+b&lt;sub&gt;dt&lt;/sub&gt;)&lt;/font&gt;", ba_x - 82, gy0 + 206, 174, 40, S_NORM))
    cells.append(line_pts("ha_e_ba_gd", [(ba_x + 5, gy0 + 206), (ba_x + 5, gy0 + 170)]))

    cells.append(box("ha_pqkv", "&lt;b&gt;Linear (in_proj_qkv)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9.5px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;qkv&lt;/sub&gt;]&lt;/font&gt;", qkv_x - 115, gy0 + 292, 230, 36, S_LIN))
    cells.append(box("ha_pba", "&lt;b&gt;Linear (in_proj_b, in_proj_a)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9.5px;color:#64748B;&quot;&gt;(Unadapted Gate Scalars)&lt;/font&gt;", ba_x - 82, gy0 + 292, 174, 36, S_LIN))
    cells.append(box("ha_pz", "&lt;b&gt;Linear (in_proj_z)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9.5px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;z&lt;/sub&gt;]&lt;/font&gt;", zx - 68, gy0 + 292, 136, 36, S_LIN))

    cells.append(line_pts("ha_e_pq_gc", [(qkv_x, gy0 + 292), (qkv_x, gy0 + 272)]))
    cells.append(line_pts("ha_e_pb_op", [(ba_x + 5, gy0 + 292), (ba_x + 5, gy0 + 246)]))
    cells.append(line_pts("ha_e_pz_gn", [(zx, gy0 + 292), (zx, gy0 + 71), (gx0 + 563, gy0 + 71)]))

    cells.append(box(
        "ha_sgl_fuse",
        "&lt;b style=&quot;color:#4C1D95;&quot;&gt;SGLang LoRA Bridge (apply_sglang_lora_targets.py):&lt;/b&gt; Fuses PEFT &lt;code&gt;in_proj_qkv + in_proj_z&lt;/code&gt; "
        "into SGLang &lt;code&gt;in_proj_qkvz&lt;/code&gt; via &lt;b&gt;A = [A&lt;sub&gt;qkv&lt;/sub&gt;, A&lt;sub&gt;qkv&lt;/sub&gt;, A&lt;sub&gt;qkv&lt;/sub&gt;, A&lt;sub&gt;z&lt;/sub&gt;]&lt;/b&gt;, &lt;b&gt;B = [B&lt;sub&gt;qkv&lt;/sub&gt;; B&lt;sub&gt;z&lt;/sub&gt;]&lt;/b&gt;",
        gx0 + 20, gy0 + 340, 606, 42, S_BADGE_LORA
    ))
    cells.append(box("ha_gin_lbl", "&lt;b&gt;Input: 16K-Tiled RMSNorm Hidden States&lt;/b&gt;", gcx - 160, gy0 + 388, 320, 20, S_TXT))

    # =========================================================================
    # PANEL (B): GRANITE-4.0-H (MAMBA-2 SSM + ATTENTION + MoE HYBRID) — y0 = 504
    # =========================================================================
    yB = 504
    cells.append(box("card_b", "", 12, yB, 1576, 476, S_CARD))
    cells.append(img_cell("ic_b", "octo_18_h200_turbo", 18, yB + 6, 30, 30))
    cells.append(box(
        "hdr_b",
        "&lt;b style=&quot;font-size:15.5px;color:#0F172A;&quot;&gt;(b) Granite-4.0-H (Mamba-2 SSM + Attention + MoE Hybrid) Integration Blueprint&lt;/b&gt; "
        "&lt;font style=&quot;font-size:12.5px;color:#0F766E;&quot;&gt;&lt;b&gt;(Transitive X ⇝ A ⇝ C Mounts · 98.5% Agree · fla Simple-GLA)&lt;/b&gt;&lt;/font&gt;",
        52, yB + 8, 1130, 28, S_TITLE_LEFT
    ))
    cells.append(box("tag_b", "&lt;b&gt;granitemoehybrid · CONV_CONTEXT = d_conv−1 · 5 LoRA Targets&lt;/b&gt;", 1192, yB + 9, 380, 26, S_BADGE_LORA))

    build_hybrid_stack("hb", yB, is_granite=True)

    # Panel (b) Middle-Top: GQA Full-Attention (x=324..904, y=yB+54..yB+256)
    by0 = yB + 54
    cells.append(box("hb_at_title", "&lt;b style=&quot;color:#92400E;&quot;&gt;GQA Full-Attention (Transitive Mount)&lt;/b&gt;", ax0, by0 - 20, 250, 18, S_TITLE_LEFT))
    cells.append(box("hb_at_out", "&lt;b&gt;Output&lt;/b&gt;", acx - 35, by0 - 20, 70, 18, S_TXT))
    cells.append(box("hb_at_box", "", ax0, by0, 580, 202, S_ATTN_BOX))

    cells.append(box("hb_oproj", "&lt;b&gt;Linear (o_proj)&lt;/b&gt; &lt;font style=&quot;color:#4C1D95;&quot;&gt;&lt;b&gt;[+Rank-64 LoRA ΔW&lt;sub&gt;O&lt;/sub&gt;]&lt;/b&gt;&lt;/font&gt;", ax0 + 80, by0 + 14, 420, 28, S_LIN))
    cells.append(line_pts("hb_e_op_out", [(acx, by0 + 14), (acx, by0 - 2)]))

    cells.append(box(
        "hb_iface",
        "&lt;b&gt;Attention Interface (Recursive Transitive Mount segment(src, s0, s1)):&lt;/b&gt;&lt;br&gt;"
        "&lt;font style=&quot;font-size:10px;&quot;&gt;When C mounts A[s&lt;sub&gt;0&lt;/sub&gt;:s&lt;sub&gt;1&lt;/sub&gt;) which mounted X, C inherits &lt;b&gt;[KV&lt;sub&gt;X&lt;/sub&gt; ∥ KV&lt;sub&gt;A&lt;/sub&gt; ∥ KV&lt;sub&gt;C&lt;/sub&gt;]&lt;/b&gt; in 0-Copy (&lt;b&gt;98.5% Agree&lt;/b&gt;)&lt;/font&gt;",
        ax0 + 24, by0 + 56, 532, 46, S_IFACE
    ))
    cells.append(line_pts("hb_e_if_op", [(acx, by0 + 56), (acx, by0 + 42)]))

    bqx, bkx, bvx = ax0 + 110, ax0 + 290, ax0 + 470
    cells.append(box("hb_qlin", "&lt;b&gt;Linear (q_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;Q&lt;/sub&gt;]&lt;/font&gt;", bqx - 72, by0 + 122, 144, 36, S_LIN))
    cells.append(box("hb_klin", "&lt;b&gt;Linear (k_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;K&lt;/sub&gt;]&lt;/font&gt;", bkx - 72, by0 + 122, 144, 36, S_LIN))
    cells.append(box("hb_vlin", "&lt;b&gt;Linear (v_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;V&lt;/sub&gt;]&lt;/font&gt;", bvx - 72, by0 + 122, 144, 36, S_LIN))

    cells.append(line_pts("hb_e_q_if", [(bqx, by0 + 122), (bqx, by0 + 102)]))
    cells.append(line_pts("hb_e_k_if", [(bkx, by0 + 122), (bkx, by0 + 102)]))
    cells.append(line_pts("hb_e_v_if", [(bvx, by0 + 122), (bvx, by0 + 102)]))
    cells.append(line_pts("hb_e_in_k", [(bkx, by0 + 194), (bkx, by0 + 158)]))
    cells.append(line_pts("hb_e_in_q", [(bkx, by0 + 178), (bqx, by0 + 178), (bqx, by0 + 158)]))
    cells.append(line_pts("hb_e_in_v", [(bkx, by0 + 178), (bvx, by0 + 178), (bvx, by0 + 158)]))

    # Panel (b) Middle-Bottom: MoE + Shared MLP & SGLang LoRA Resolution (x=324..904, y=yB+280..yB+468)
    bmy0 = yB + 280
    cells.append(box("hb_mlp_title", "&lt;b style=&quot;color:#065F46;&quot;&gt;Parallel Sparse MoE + Shared MLP &amp; LoRA Target Resolution&lt;/b&gt;", ax0, bmy0 - 20, 480, 18, S_TITLE_LEFT))
    cells.append(box("hb_mlp_box", "", ax0, bmy0, 580, 188, S_MLP_BOX))

    cells.append(circle_op("hb_fsum", "+", acx, bmy0 + 22, r=10))
    cells.append(box("hb_smlp", "&lt;b&gt;Shared MLP:&lt;/b&gt; input_linear → SiLU → output_linear&lt;br&gt;&lt;font style=&quot;font-size:9.5px;color:#64748B;&quot;&gt;(Stateless Token-Wise · Excluded from LoRA)&lt;/font&gt;", ax0 + 20, bmy0 + 42, 256, 40, S_LIN))
    cells.append(box("hb_rmoe", "&lt;b&gt;Sparse Top-K MoE:&lt;/b&gt; Router → Experts [1..E]&lt;br&gt;&lt;font style=&quot;font-size:9.5px;color:#065F46;&quot;&gt;(Stateless · Batched over 1D Tree T&lt;sub&gt;tree&lt;/sub&gt;)&lt;/font&gt;", ax0 + 304, bmy0 + 42, 256, 40, S_IFACE))
    cells.append(line_pts("hb_e_sm_s", [(ax0 + 148, bmy0 + 42), (ax0 + 148, bmy0 + 22), (acx - 10, bmy0 + 22)]))
    cells.append(line_pts("hb_e_rm_s", [(ax0 + 432, bmy0 + 42), (ax0 + 432, bmy0 + 22), (acx + 10, bmy0 + 22)]))

    cells.append(box(
        "hb_lora_rule",
        "&lt;b style=&quot;color:#4C1D95;&quot;&gt;Automatic SGLang LoRA Resolution (lora_targets.py + apply_sglang_lora_targets.py):&lt;/b&gt;&lt;br&gt;"
        "• &lt;b&gt;Served Targets:&lt;/b&gt; &lt;code&gt;[q_proj, k_proj, v_proj, o_proj, out_proj]&lt;/code&gt; (patches Mamba &lt;code&gt;out_proj&lt;/code&gt; dim &lt;code&gt;H·P → D&lt;/code&gt;)&lt;br&gt;"
        "• &lt;b&gt;Filtered Out:&lt;/b&gt; &lt;code&gt;in_proj&lt;/code&gt; (SGLang Nemotron-H 2-of-5 partition layout) &amp; &lt;code&gt;input/output_linear&lt;/code&gt;&lt;br&gt;"
        "• &lt;b&gt;Norm Patch (apply_sglang_granite_norm.py):&lt;/b&gt; Fixes SGLang v0.5.19 double-residual final RMSNorm.",
        ax0 + 16, bmy0 + 90, 548, 88, "rounded=1;arcSize=14;fillColor=#FFFFFF;strokeColor=#0F766E;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10.0;fontColor=#0F172A;align=left;verticalAlign=middle;spacingLeft=8;spacingRight=6;whiteSpace=wrap;html=1;"
    ))

    # Panel (b) Right: Mamba-2 SSD Mixer Sub-Module (mamba2_varlen.py) (x=922..1568, y=yB+54..yB+468)
    gyB = yB + 54
    cells.append(box("hb_m2_title", "&lt;b style=&quot;color:#1E40AF;&quot;&gt;Mamba-2 SSD Mixer (mamba2_varlen.py)&lt;/b&gt;", gx0, gyB - 20, 275, 18, S_TITLE_LEFT))
    cells.append(box("hb_m2_out", "&lt;b&gt;Output&lt;/b&gt;", gcx - 35, gyB - 20, 70, 18, S_TXT))
    cells.append(box("hb_m2_box", "", gx0, gyB, 646, 414, S_REC_BOX))

    cells.append(box("hb_mout", "&lt;b&gt;Linear (out_proj)&lt;/b&gt; &lt;font style=&quot;color:#4C1D95;&quot;&gt;&lt;b&gt;[+Patched SGLang LoRA ΔW&lt;sub&gt;out&lt;/sub&gt;: H·P → D]&lt;/b&gt;&lt;/font&gt;", gx0 + 93, gyB + 14, 460, 28, S_LIN))
    cells.append(line_pts("hb_e_mo_out", [(gcx, gyB + 14), (gcx, gyB - 2)]))

    cells.append(box("hb_mnorm", "&lt;b&gt;RMSNormGated:&lt;/b&gt; RMSNorm(y + D · x) ⊗ SiLU(z)", gx0 + 93, gyB + 56, 460, 28, S_NORM))
    cells.append(line_pts("hb_e_mn_mo", [(gcx, gyB + 56), (gcx, gyB + 42)]))

    cells.append(box(
        "hb_mscan",
        "&lt;b style=&quot;font-size:11.5px;color:#1E3A8A;&quot;&gt;Mamba-2 SSD Chunk Scan → fla chunk_simple_gla (tree_recurrence)&lt;/b&gt;&lt;br&gt;"
        "&lt;font style=&quot;font-size:10px;color:#0F172A;&quot;&gt;• &lt;b&gt;Exact SSD-to-GLA Mapping:&lt;/b&gt; q = repeat(C), k = repeat(B), v = dt · x, decay g = A · dt + skip D · x&lt;/font&gt;&lt;br&gt;"
        "&lt;font style=&quot;font-size:10px;color:#0F766E;&quot;&gt;• &lt;b&gt;Tree State:&lt;/b&gt; Starts from parent SSM state &lt;b&gt;h&lt;sub&gt;parent&lt;/sub&gt;&lt;/b&gt; on fork/rewind; &lt;/font&gt;"
        "&lt;font style=&quot;font-size:10px;color:#DC2626;&quot;&gt;&lt;b&gt;[✗ Skips Mounted KV Spans]&lt;/b&gt;&lt;/font&gt;",
        gx0 + 24, gyB + 100, 598, 66, S_KERNEL
    ))
    cells.append(line_pts("hb_e_ms_mn", [(gcx - 30, gyB + 100), (gcx - 30, gyB + 84)]))

    xbc_x, dt_x, mzx = gx0 + 175, gx0 + 400, gx0 + 560
    cells.append(box(
        "hb_mconv",
        "&lt;b&gt;Causal Conv1d + SiLU on [x, B, C] (tree_conv)&lt;/b&gt;&lt;br&gt;"
        "&lt;font style=&quot;font-size:9.5px;&quot;&gt;Gathers &lt;b&gt;CONV_CONTEXT = mamba_d_conv − 1&lt;/b&gt; Ancestor Tokens!&lt;/font&gt;",
        xbc_x - 145, gyB + 190, 290, 42, S_CONV
    ))
    cells.append(box(
        "hb_mdt",
        "&lt;b&gt;Discretization Gate:&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9.5px;&quot;&gt;dt = clamp(softplus(dt + b&lt;sub&gt;dt&lt;/sub&gt;))&lt;/font&gt;",
        dt_x - 75, gyB + 190, 170, 42, S_NORM
    ))
    cells.append(line_pts("hb_e_mc_ms", [(xbc_x, gyB + 190), (xbc_x, gyB + 166)]))
    cells.append(line_pts("hb_e_dt_ms", [(dt_x + 10, gyB + 190), (dt_x + 10, gyB + 166)]))

    cells.append(box(
        "hb_msplit",
        "&lt;b&gt;Split 5 Partitions:&lt;/b&gt; [x, B, C] (Conv Path) · [dt] (Decay Step) · [z] (Output Gate)",
        gx0 + 44, gyB + 252, 558, 30, S_ROPE
    ))
    cells.append(line_pts("hb_e_sp_mc", [(xbc_x, gyB + 252), (xbc_x, gyB + 232)]))
    cells.append(line_pts("hb_e_sp_dt", [(dt_x + 10, gyB + 252), (dt_x + 10, gyB + 232)]))
    cells.append(line_pts("hb_e_sp_z", [(mzx, gyB + 252), (mzx, gyB + 70), (gx0 + 553, gyB + 70)]))

    cells.append(box(
        "hb_minproj",
        "&lt;b&gt;Linear (in_proj)&lt;/b&gt; &lt;font style=&quot;font-size:10px;color:#64748B;&quot;&gt;(Fused 5-Partition Input Projection)&lt;/font&gt;",
        gx0 + 93, gyB + 298, 460, 30, S_LIN
    ))
    cells.append(line_pts("hb_e_ip_sp", [(gcx, gyB + 298), (gcx, gyB + 282)]))

    cells.append(box(
        "hb_mhook_note",
        "&lt;b style=&quot;color:#064E3B;&quot;&gt;Why patch_mamba2 Routes to fla chunk_simple_gla:&lt;/b&gt; Stock HF torch fallback ignores &lt;code&gt;cu_seq_lens_q&lt;/code&gt; "
        "and lacks per-node initial states; our SSD-to-GLA tree kernel guarantees &lt;b&gt;100% bit-exact branch isolation&lt;/b&gt;.",
        gx0 + 20, gyB + 340, 606, 42, "rounded=1;arcSize=18;fillColor=#CCFBF1;strokeColor=#0F766E;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10.0;fontColor=#064E3B;align=center;verticalAlign=middle;spacingLeft=6;spacingRight=6;whiteSpace=wrap;html=1;"
    ))
    cells.append(box("hb_min_lbl", "&lt;b&gt;Input: RMSNorm Hidden States&lt;/b&gt;", gcx - 140, gyB + 388, 280, 20, S_TXT))

    body = "\n".join(cells)
    return f"""<mxfile host="Electron" modified="2026-10-26T23:51:00.000Z" agent="OctopusDrawioGenerator" version="24.7.5">
  <diagram id="arch-hybrid" name="Page-1">
    <mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="992" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""


def main():
    DRAWIO_PATH.write_text(build_xml(), encoding="utf-8")
    subprocess.run(
        [
            DRAWIO_BIN,
            "--export",
            "--format",
            "pdf",
            "--crop",
            "--border",
            "4",
            "--output",
            str(PDF_PATH),
            str(DRAWIO_PATH),
        ],
        check=True,
    )
    print(f"Exported: {PDF_PATH}")


if __name__ == "__main__":
    main()
