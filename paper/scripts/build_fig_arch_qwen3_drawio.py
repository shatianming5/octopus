#!/usr/bin/env python3
"""
Generate Draw.io (.drawio) and vector PDF for Figure 10:
Per-Architecture Octopus Integration Blueprints (Part I):
  (a) Qwen3 (Dense) — Pure Full-Attention Transformer
  (b) Qwen3 (MoE)   — Pure Full-Attention Sparse Mixture-of-Experts
Styled after the clean modular block-diagram aesthetic (Macro Decoder Stack on left,
Zoomed-in Attention / MLP / MoE sub-modules + Octopus Serving & Prefix-Tree Hooks on right).
"""

import base64
import io
import subprocess
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DRAWIO_PATH = FIG_DIR / "fig_arch_qwen3_dense_moe.drawio"
PDF_PATH = FIG_DIR / "fig_arch_qwen3_dense_moe.pdf"
DRAWIO_BIN = "/Applications/draw.io.app/Contents/MacOS/draw.io"


def b64_img(name: str, max_px: int = 110) -> str:
    p = FIG_DIR / f"{name}.png"
    img = Image.open(p).convert("RGBA")
    img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    q = img.quantize(colors=64, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO()
    q.save(buf, format="PNG", optimize=True)
    # Note: no ';base64' because Draw.io splits style attributes on ';'
    return "data:image/png," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_xml() -> str:
    icons = {
        k: b64_img(k)
        for k in [
            "octo_10_layer2_mount",
            "octo_12_rope_future",
            "octo_radix",
            "octo_dag",
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

    def circle_op(cid, sym, cx, cy, r=12):
        return (
            f'<mxCell id="{cid}" value="&lt;b&gt;{sym}&lt;/b&gt;" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;'
            f'fillColor=#FFFFFF;strokeColor=#1E293B;strokeWidth=1.8;fontFamily=Helvetica;fontSize=14;fontColor=#0F172A;'
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

    # Common styles with whiteSpace=wrap;html=1; so nothing ever overflows
    S_CARD = "rounded=1;arcSize=5;fillColor=#F1F5F9;strokeColor=#1E293B;strokeWidth=2.2;whiteSpace=wrap;html=1;"
    S_DEC_BOX = "rounded=1;arcSize=8;fillColor=#DBEAFE;strokeColor=#1E293B;strokeWidth=1.9;whiteSpace=wrap;html=1;"
    S_ATTN_BOX = "rounded=1;arcSize=7;fillColor=#FEF3C7;strokeColor=#1E293B;strokeWidth=1.9;whiteSpace=wrap;html=1;"
    S_MLP_BOX = "rounded=1;arcSize=7;fillColor=#DCFCE7;strokeColor=#1E293B;strokeWidth=1.9;whiteSpace=wrap;html=1;"
    S_HOOK_BOX = "rounded=1;arcSize=7;fillColor=#CCFBF1;strokeColor=#0F766E;strokeWidth=1.9;whiteSpace=wrap;html=1;"

    S_NORM = "rounded=1;arcSize=45;fillColor=#FFEDD5;strokeColor=#1E293B;strokeWidth=1.6;fontFamily=Helvetica;fontSize=12.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_EMB = "rounded=1;arcSize=45;fillColor=#DBEAFE;strokeColor=#1E293B;strokeWidth=1.6;fontFamily=Helvetica;fontSize=12.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_ROPE = "rounded=1;arcSize=40;fillColor=#FEF08A;strokeColor=#1E293B;strokeWidth=1.6;fontFamily=Helvetica;fontSize=11.5;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_ATTN_PILL = "rounded=1;arcSize=45;fillColor=#FDE68A;strokeColor=#1E293B;strokeWidth=1.6;fontFamily=Helvetica;fontSize=12.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_MLP_PILL = "rounded=1;arcSize=45;fillColor=#BBF7D0;strokeColor=#1E293B;strokeWidth=1.6;fontFamily=Helvetica;fontSize=12.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_LIN = "rounded=1;arcSize=28;fillColor=#E0E7FF;strokeColor=#1E293B;strokeWidth=1.6;fontFamily=Helvetica;fontSize=11.5;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_IFACE = "rounded=1;arcSize=18;fillColor=#DCFCE7;strokeColor=#0F766E;strokeWidth=1.8;fontFamily=Helvetica;fontSize=11.0;fontColor=#064E3B;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_TXT = "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=12.5;fontColor=#0F172A;align=center;verticalAlign=middle;"
    S_TITLE_LEFT = "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=13.5;fontColor=#0F172A;align=left;verticalAlign=middle;"
    S_BADGE_LORA = "rounded=1;arcSize=35;fillColor=#EDE9FE;strokeColor=#6D28D9;strokeWidth=1.4;fontFamily=Helvetica;fontSize=10.5;fontColor=#4C1D95;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
    S_BADGE_TEAL = "rounded=1;arcSize=24;fillColor=#FFFFFF;strokeColor=#0F766E;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10.5;fontColor=#064E3B;align=left;verticalAlign=middle;spacingLeft=8;spacingRight=6;whiteSpace=wrap;html=1;"

    cells = []

    def build_decoder_stack(prefix, y0, is_moe=False):
        cx = 165
        sx = 285
        cells.append(box(f"{prefix}_out", "&lt;b&gt;Output / Pair-Deduplicated LM Head&lt;/b&gt;", cx - 125, y0 + 48, 250, 22, S_TXT))
        cells.append(box(f"{prefix}_fnorm", "&lt;b&gt;RMSNorm&lt;/b&gt; &lt;font style=&quot;font-size:10.5px;&quot;&gt;(Final Norm)&lt;/font&gt;", cx - 95, y0 + 82, 190, 30, S_NORM))
        cells.append(line_pts(f"{prefix}_e_out", [(cx, y0 + 82), (cx, y0 + 70)]))

        dec_l = "Decoder Layer × L"
        dec_r = "(L=36, Qwen3-4B)" if not is_moe else "(Sparse MoE)"
        cells.append(box(f"{prefix}_dlbl_l", f"&lt;b&gt;{dec_l}&lt;/b&gt;", 36, y0 + 120, 122, 20, "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=12.0;fontColor=#0F172A;align=right;verticalAlign=middle;"))
        cells.append(box(f"{prefix}_dlbl_r", f"&lt;b&gt;{dec_r}&lt;/b&gt;", 172, y0 + 120, 136, 20, "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=11.5;fontColor=#0F766E;align=left;verticalAlign=middle;"))
        cells.append(box(f"{prefix}_dbox", "", 42, y0 + 144, 266, 214, S_DEC_BOX))
        cells.append(line_pts(f"{prefix}_e_fnorm", [(cx, y0 + 144), (cx, y0 + 112)]))

        cells.append(circle_op(f"{prefix}_add2", "+", cx, y0 + 166, r=11))
        cells.append(line_pts(f"{prefix}_e_add2_top", [(cx, y0 + 155), (cx, y0 + 144)], arrow=False))

        ffn_lbl = "&lt;b&gt;SwiGLU MLP&lt;/b&gt; &lt;font style=&quot;font-size:10.5px;color:#4C1D95;&quot;&gt;[+LoRA]&lt;/font&gt;" if not is_moe else "&lt;b&gt;Sparse MoE&lt;/b&gt; &lt;font style=&quot;font-size:10.5px;color:#065F46;&quot;&gt;[Top-K Experts]&lt;/font&gt;"
        cells.append(box(f"{prefix}_ffn", ffn_lbl, cx - 92, y0 + 188, 184, 28, S_MLP_PILL))
        cells.append(line_pts(f"{prefix}_e_ffn_add2", [(cx, y0 + 188), (cx, y0 + 177)]))

        cells.append(box(f"{prefix}_norm2", "&lt;b&gt;RMSNorm&lt;/b&gt;", cx - 92, y0 + 226, 184, 26, S_NORM))
        cells.append(line_pts(f"{prefix}_e_n2_ffn", [(cx, y0 + 226), (cx, y0 + 216)], arrow=False))

        cells.append(circle_op(f"{prefix}_add1", "+", cx, y0 + 272, r=11))
        cells.append(line_pts(f"{prefix}_e_add1_n2", [(cx, y0 + 261), (cx, y0 + 252)], arrow=False))
        cells.append(line_pts(f"{prefix}_skip2", [(cx, y0 + 258), (sx, y0 + 258), (sx, y0 + 166), (cx + 11, y0 + 166)]))

        cells.append(box(f"{prefix}_attn", "&lt;b&gt;Attention&lt;/b&gt; &lt;font style=&quot;font-size:10.5px;color:#065F46;&quot;&gt;[Mount / Own]&lt;/font&gt;", cx - 92, y0 + 292, 184, 28, S_ATTN_PILL))
        cells.append(line_pts(f"{prefix}_e_at_add1", [(cx, y0 + 292), (cx, y0 + 283)]))

        cells.append(box(f"{prefix}_norm1", "&lt;b&gt;RMSNorm&lt;/b&gt;", cx - 92, y0 + 328, 184, 25, S_NORM))
        cells.append(line_pts(f"{prefix}_e_n1_at", [(cx, y0 + 328), (cx, y0 + 320)], arrow=False))
        cells.append(line_pts(f"{prefix}_skip1", [(cx, y0 + 362), (sx, y0 + 362), (sx, y0 + 272), (cx + 11, y0 + 272)]))

        cells.append(box(f"{prefix}_rope", "&lt;b&gt;RotaryEmbedding&lt;/b&gt; &lt;font style=&quot;font-size:10px;color:#92400E;&quot;&gt;(K2 Jump s(c&apos;&lt;sub&gt;B&lt;/sub&gt;))&lt;/font&gt;", cx - 105, y0 + 370, 210, 28, S_ROPE))
        cells.append(line_pts(f"{prefix}_e_rope_n1", [(cx, y0 + 370), (cx, y0 + 353)]))

        cells.append(box(f"{prefix}_emb", "&lt;b&gt;Embedding&lt;/b&gt; &lt;font style=&quot;font-size:10px;&quot;&gt;(1D Prefix-Tree T&lt;sub&gt;tree&lt;/sub&gt;)&lt;/font&gt;", cx - 105, y0 + 410, 210, 28, S_EMB))
        cells.append(line_pts(f"{prefix}_e_emb_rope", [(cx, y0 + 410), (cx, y0 + 398)]))

        cells.append(box(f"{prefix}_in", "&lt;b&gt;Input Tokens (Cursor c&lt;sub&gt;B&lt;/sub&gt; / Tree T&lt;sub&gt;tree&lt;/sub&gt;)&lt;/b&gt;", cx - 125, y0 + 448, 250, 20, S_TXT))
        cells.append(line_pts(f"{prefix}_e_in_emb", [(cx, y0 + 448), (cx, y0 + 438)]))

    # =========================================================================
    # PANEL (A): QWEN3 (DENSE) — y0 = 10
    # =========================================================================
    yA = 10
    cells.append(box("card_a", "", 12, yA, 1576, 476, S_CARD))
    cells.append(img_cell("ic_a1", "octo_10_layer2_mount", 18, yA + 6, 30, 30))
    cells.append(box(
        "hdr_a",
        "&lt;b style=&quot;font-size:15.5px;color:#0F172A;&quot;&gt;(a) Qwen3 (Dense) Architecture &amp; Octopus Integration Blueprint&lt;/b&gt; "
        "&lt;font style=&quot;font-size:12.5px;color:#0F766E;&quot;&gt;&lt;b&gt;(Pure Full-Attention · Sparse 9-Layer KV Mount · 96.9% Agreement)&lt;/b&gt;&lt;/font&gt;",
        52, yA + 8, 1160, 28, S_TITLE_LEFT
    ))
    cells.append(box("tag_a", "&lt;b&gt;family = None · CONV_CONTEXT = 0 · 0 Recurrent State&lt;/b&gt;", 1220, yA + 9, 352, 26, S_BADGE_LORA))

    build_decoder_stack("da", yA, is_moe=False)

    # Panel (a) Middle: Attention Zoomed-in Box (x=334..934, y=yA+54..yA+438)
    ax0, ay0 = 334, yA + 54
    acx = ax0 + 300  # 634
    cells.append(box("da_at_title", "&lt;b&gt;Attention (Layer-Selective KV Mount)&lt;/b&gt;", ax0, ay0 - 22, 250, 20, S_TITLE_LEFT))
    cells.append(box("da_at_out_lbl", "&lt;b&gt;Output&lt;/b&gt;", acx - 35, ay0 - 22, 70, 20, S_TXT))
    cells.append(box("da_at_box", "", ax0, ay0, 600, 384, S_ATTN_BOX))

    cells.append(line_pts("da_at_e_out", [(acx, ay0 + 22), (acx, ay0 - 2)]))
    cells.append(box("da_oproj", "&lt;b&gt;Linear (o_proj)&lt;/b&gt; &lt;font style=&quot;color:#4C1D95;&quot;&gt;&lt;b&gt;[+Rank-64 LoRA ΔW&lt;sub&gt;O&lt;/sub&gt;]&lt;/b&gt;&lt;/font&gt;", ax0 + 60, ay0 + 22, 480, 34, S_LIN))

    cells.append(box(
        "da_iface",
        "&lt;b style=&quot;font-size:12px;color:#064E3B;&quot;&gt;Attention Interface (TritonAttnBackend / FlashAttention Varlen)&lt;/b&gt;&lt;br&gt;"
        "&lt;font style=&quot;font-size:10.5px;color:#0F766E;&quot;&gt;• &lt;b&gt;Mounted (ℓ ∈ OCTOPUS_MOUNT_LAYERS, 9 Layers):&lt;/b&gt; Reads &lt;b&gt;[Mounted KV c&lt;sub&gt;A&lt;/sub&gt;[s:e) ∥ Own Path KV]&lt;/b&gt; (0-Copy)&lt;/font&gt;&lt;br&gt;"
        "&lt;font style=&quot;font-size:10.5px;color:#B45309;&quot;&gt;• &lt;b&gt;Private (ℓ ∉ OCTOPUS_MOUNT_LAYERS, 27 Layers):&lt;/b&gt; &lt;b&gt;hide_mounts / own_calls&lt;/b&gt; → Own Path KV Only&lt;/font&gt;",
        ax0 + 22, ay0 + 76, 556, 76, S_IFACE
    ))
    cells.append(line_pts("da_e_if_op", [(acx, ay0 + 76), (acx, ay0 + 56)]))

    qx, kx, vx = ax0 + 110, ax0 + 300, ax0 + 480
    cells.append(box(
        "da_mrope",
        "&lt;b&gt;1D RoPE (Rule K2 Monotonic Jump):&lt;/b&gt; s(c&apos;&lt;sub&gt;B&lt;/sub&gt;) = max(p&lt;sub&gt;B&lt;/sub&gt;, e&lt;sub&gt;A&lt;/sub&gt;+1) · Native Keys c&lt;sub&gt;A&lt;/sub&gt; Unrotated",
        ax0 + 28, ay0 + 172, 376, 36, S_ROPE
    ))

    cells.append(box("da_qnorm", "&lt;b&gt;RMSNorm&lt;/b&gt; (q_norm)", qx - 72, ay0 + 228, 144, 30, S_NORM))
    cells.append(box("da_knorm", "&lt;b&gt;RMSNorm&lt;/b&gt; (k_norm)", kx - 72, ay0 + 228, 144, 30, S_NORM))

    cells.append(box("da_qlin", "&lt;b&gt;Linear (q_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;Q&lt;/sub&gt;]&lt;/font&gt;", qx - 72, ay0 + 286, 144, 40, S_LIN))
    cells.append(box("da_klin", "&lt;b&gt;Linear (k_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;K&lt;/sub&gt; · ∂L&lt;sub&gt;B&lt;/sub&gt;/∂K&lt;sub&gt;A&lt;/sub&gt;]&lt;/font&gt;", kx - 78, ay0 + 286, 156, 40, S_LIN))
    cells.append(box("da_vlin", "&lt;b&gt;Linear (v_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;V&lt;/sub&gt; · ∂L&lt;sub&gt;B&lt;/sub&gt;/∂V&lt;sub&gt;A&lt;/sub&gt;]&lt;/font&gt;", vx - 78, ay0 + 286, 156, 40, S_LIN))

    cells.append(line_pts("da_e_ql_qn", [(qx, ay0 + 286), (qx, ay0 + 258)]))
    cells.append(line_pts("da_e_kl_kn", [(kx, ay0 + 286), (kx, ay0 + 258)]))
    cells.append(line_pts("da_e_qn_mr", [(qx, ay0 + 228), (qx, ay0 + 208)]))
    cells.append(line_pts("da_e_kn_mr", [(kx, ay0 + 228), (kx, ay0 + 208)]))
    cells.append(line_pts("da_e_mr_q", [(qx, ay0 + 172), (qx, ay0 + 152)]))
    cells.append(line_pts("da_e_mr_k", [(kx, ay0 + 172), (kx, ay0 + 152)]))
    cells.append(line_pts("da_e_vl_if", [(vx, ay0 + 286), (vx, ay0 + 152)]))

    cells.append(line_pts("da_e_in_k", [(kx, ay0 + 398), (kx, ay0 + 326)]))
    cells.append(line_pts("da_e_in_q", [(kx, ay0 + 356), (qx, ay0 + 356), (qx, ay0 + 326)]))
    cells.append(line_pts("da_e_in_v", [(kx, ay0 + 356), (vx, ay0 + 356), (vx, ay0 + 326)]))
    cells.append(box("da_at_in_lbl", "&lt;b&gt;Input Hidden States&lt;/b&gt;", kx - 90, ay0 + 398, 180, 22, S_TXT))

    # Panel (a) Top-Right: MLP Sub-Module (x=956..1568, y=yA+54..yA+252)
    mx0, my0 = 956, yA + 54
    mcx = mx0 + 306  # 1262
    cells.append(box("da_mlp_title", "&lt;b&gt;MLP (SwiGLU FFN)&lt;/b&gt;", mx0, my0 - 22, 220, 20, S_TITLE_LEFT))
    cells.append(box("da_mlp_out_lbl", "&lt;b&gt;Output&lt;/b&gt;", mcx - 35, my0 - 22, 70, 20, S_TXT))
    cells.append(box("da_mlp_box", "", mx0, my0, 612, 198, S_MLP_BOX))

    cells.append(box("da_down", "&lt;b&gt;Linear (down_proj)&lt;/b&gt; &lt;font style=&quot;color:#4C1D95;&quot;&gt;&lt;b&gt;[+Rank-64 LoRA ΔW&lt;sub&gt;down&lt;/sub&gt;]&lt;/b&gt;&lt;/font&gt;", mx0 + 116, my0 + 16, 380, 30, S_LIN))
    cells.append(line_pts("da_e_down_out", [(mcx, my0 + 16), (mcx, my0 - 2)]))

    cells.append(circle_op("da_mul", "×", mcx, my0 + 68, r=11))
    cells.append(line_pts("da_e_mul_down", [(mcx, my0 + 57), (mcx, my0 + 46)]))

    gx, ux = mx0 + 150, mx0 + 462
    cells.append(box("da_silu", "&lt;b&gt;SiLU&lt;/b&gt;", gx - 65, my0 + 82, 130, 26, S_NORM))
    cells.append(box("da_gate", "&lt;b&gt;Gate: Linear (gate_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;gate&lt;/sub&gt;]&lt;/font&gt;", gx - 95, my0 + 124, 190, 38, S_LIN))
    cells.append(box("da_up", "&lt;b&gt;Up: Linear (up_proj)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;color:#4C1D95;&quot;&gt;[+LoRA ΔW&lt;sub&gt;up&lt;/sub&gt;]&lt;/font&gt;", ux - 95, my0 + 124, 190, 38, S_LIN))

    cells.append(line_pts("da_e_gate_silu", [(gx, my0 + 124), (gx, my0 + 108)]))
    cells.append(line_pts("da_e_silu_mul", [(gx, my0 + 82), (gx, my0 + 68), (mcx - 11, my0 + 68)]))
    cells.append(line_pts("da_e_up_mul", [(ux, my0 + 124), (ux, my0 + 68), (mcx + 11, my0 + 68)]))
    cells.append(line_pts("da_e_min_g", [(mcx, my0 + 180), (gx, my0 + 180), (gx, my0 + 162)]))
    cells.append(line_pts("da_e_min_u", [(mcx, my0 + 180), (ux, my0 + 180), (ux, my0 + 162)]))
    cells.append(line_pts("da_e_min_stem", [(mcx, my0 + 206), (mcx, my0 + 180)], arrow=False))
    cells.append(box("da_mlp_in_lbl", "&lt;b&gt;Input&lt;/b&gt;", mcx - 40, my0 + 204, 80, 20, S_TXT))

    # Panel (a) Bottom-Right: Octopus Runtime & Training Integration Card (x=956..1568, y=yA+284..yA+468)
    hx0, hy0 = 956, yA + 284
    cells.append(box("da_hk_box", "", hx0, hy0, 612, 184, S_HOOK_BOX))
    cells.append(img_cell("ic_a2", "octo_radix", hx0 + 10, hy0 + 6, 26, 26))
    cells.append(box(
        "da_hk_hdr",
        "&lt;b style=&quot;font-size:13px;color:#064E3B;&quot;&gt;How Qwen3 (Dense) Hooks into Octopus (Serving &amp; DAG-GRPO)&lt;/b&gt;",
        hx0 + 42, hy0 + 6, 560, 24, S_TITLE_LEFT
    ))
    cells.append(box(
        "da_hk_b1",
        "• &lt;b&gt;Zero Recurrent State (model_hooks.py):&lt;/b&gt; &lt;code&gt;family=None&lt;/code&gt;, &lt;code&gt;CONV_CONTEXT=0&lt;/code&gt;. "
        "&lt;b&gt;fork&lt;/b&gt; (+K Radix ref) and &lt;b&gt;rewind(k)&lt;/b&gt; (cursor reset c&lt;sub&gt;k&lt;/sub&gt;) are pure O(1) PagedAttention pointer ops.",
        hx0 + 12, hy0 + 36, 588, 42, S_BADGE_TEAL
    ))
    cells.append(box(
        "da_hk_b2",
        "• &lt;b&gt;Sparse Retrieval-Layer Mount (kv_mount.py):&lt;/b&gt; &lt;code&gt;OCTOPUS_MOUNT_LAYERS&lt;/code&gt; splices sibling KV c&lt;sub&gt;A&lt;/sub&gt;[s:e) into "
        "9 retrieval layers while &lt;code&gt;hide_mounts&lt;/code&gt; keeps 27 layers on B&apos;s own path (&lt;b&gt;96.9% agreement&lt;/b&gt;).",
        hx0 + 12, hy0 + 84, 588, 42, S_BADGE_TEAL
    ))
    cells.append(box(
        "da_hk_b3",
        "• &lt;b&gt;Prefix-Tree FlashAttention &amp; LoRA (prefix_tree.py):&lt;/b&gt; Runs 1D deduplicated &lt;code&gt;attn_calls&lt;/code&gt; vs. "
        "&lt;code&gt;own_calls&lt;/code&gt; (32K key chunks); resolves 7 LoRA targets (&lt;code&gt;q/k/v/o/gate/up/down_proj&lt;/code&gt;).",
        hx0 + 12, hy0 + 132, 588, 42, S_BADGE_TEAL
    ))

    # =========================================================================
    # PANEL (B): QWEN3 (MoE) — y0 = 504
    # =========================================================================
    yB = 504
    cells.append(box("card_b", "", 12, yB, 1576, 476, S_CARD))
    cells.append(img_cell("ic_b1", "octo_dag", 18, yB + 6, 30, 30))
    cells.append(box(
        "hdr_b",
        "&lt;b style=&quot;font-size:15.5px;color:#0F172A;&quot;&gt;(b) Qwen3 (MoE) Architecture &amp; Octopus Integration Blueprint&lt;/b&gt; "
        "&lt;font style=&quot;font-size:12.5px;color:#0F766E;&quot;&gt;&lt;b&gt;(Sparse Top-K MoE · Stateless 1D Prefix-Tree Grouped GEMM · 0 Expert State)&lt;/b&gt;&lt;/font&gt;",
        52, yB + 8, 1180, 28, S_TITLE_LEFT
    ))
    cells.append(box("tag_b", "&lt;b&gt;Stateless MoE FFN · 1D Packed Tree Routing&lt;/b&gt;", 1240, yB + 9, 332, 26, S_BADGE_LORA))

    build_decoder_stack("db", yB, is_moe=True)

    # Panel (b) Middle-Top: Attention Sub-Module (x=334..910, y=yB+54..yB+252)
    bx0, by0 = 334, yB + 54
    bcx = bx0 + 288  # 622
    cells.append(box("db_at_title", "&lt;b&gt;Attention (All State Here)&lt;/b&gt;", bx0, by0 - 22, 230, 20, S_TITLE_LEFT))
    cells.append(box("db_at_out", "&lt;b&gt;Output&lt;/b&gt;", bcx - 35, by0 - 22, 70, 20, S_TXT))
    cells.append(box("db_at_box", "", bx0, by0, 576, 198, S_ATTN_BOX))

    cells.append(box("db_oproj", "&lt;b&gt;Linear (o_proj)&lt;/b&gt; &lt;font style=&quot;color:#4C1D95;&quot;&gt;&lt;b&gt;[+LoRA ΔW&lt;sub&gt;O&lt;/sub&gt;]&lt;/b&gt;&lt;/font&gt;", bx0 + 68, by0 + 14, 440, 28, S_LIN))
    cells.append(line_pts("db_e_op_out", [(bcx, by0 + 14), (bcx, by0 - 2)]))

    cells.append(box(
        "db_iface",
        "&lt;b&gt;Attention Interface:&lt;/b&gt; 0-Copy KV Mount (&lt;code&gt;OCTOPUS_MOUNT_LAYERS&lt;/code&gt;) + Rule K2 Jump s(c&apos;&lt;sub&gt;B&lt;/sub&gt;)",
        bx0 + 24, by0 + 54, 528, 34, S_IFACE
    ))
    cells.append(line_pts("db_e_if_op", [(bcx, by0 + 54), (bcx, by0 + 42)]))

    bqx, bkx, bvx = bx0 + 106, bx0 + 288, bx0 + 470
    cells.append(box("db_qnorm", "&lt;b&gt;RMSNorm&lt;/b&gt; (Q)", bqx - 68, by0 + 102, 136, 26, S_NORM))
    cells.append(box("db_knorm", "&lt;b&gt;RMSNorm&lt;/b&gt; (K)", bkx - 68, by0 + 102, 136, 26, S_NORM))
    cells.append(box("db_qlin", "&lt;b&gt;Linear (Q)&lt;/b&gt; [+LoRA]", bqx - 68, by0 + 142, 136, 28, S_LIN))
    cells.append(box("db_klin", "&lt;b&gt;Linear (K)&lt;/b&gt; [+LoRA]", bkx - 68, by0 + 142, 136, 28, S_LIN))
    cells.append(box("db_vlin", "&lt;b&gt;Linear (V)&lt;/b&gt; [+LoRA]", bvx - 68, by0 + 142, 136, 28, S_LIN))

    cells.append(line_pts("db_e_ql_qn", [(bqx, by0 + 142), (bqx, by0 + 128)]))
    cells.append(line_pts("db_e_kl_kn", [(bkx, by0 + 142), (bkx, by0 + 128)]))
    cells.append(line_pts("db_e_qn_if", [(bqx, by0 + 102), (bqx, by0 + 88)]))
    cells.append(line_pts("db_e_kn_if", [(bkx, by0 + 102), (bkx, by0 + 88)]))
    cells.append(line_pts("db_e_vl_if", [(bvx, by0 + 142), (bvx, by0 + 88)]))
    cells.append(line_pts("db_e_in_k", [(bkx, by0 + 190), (bkx, by0 + 170)]))
    cells.append(line_pts("db_e_in_q", [(bkx, by0 + 182), (bqx, by0 + 182), (bqx, by0 + 170)]))
    cells.append(line_pts("db_e_in_v", [(bkx, by0 + 182), (bvx, by0 + 182), (bvx, by0 + 170)]))

    # Panel (b) Middle-Bottom: Single Expert MLP (x=334..910, y=yB+280..yB+458)
    em_y0 = yB + 280
    cells.append(box("db_mlp_title", "&lt;b&gt;Expert MLP (Per Expert)&lt;/b&gt;", bx0, em_y0 - 20, 230, 20, S_TITLE_LEFT))
    cells.append(box("db_mlp_out", "&lt;b&gt;Output&lt;/b&gt;", bcx - 35, em_y0 - 20, 70, 20, S_TXT))
    cells.append(box("db_mlp_box", "", bx0, em_y0, 576, 178, S_MLP_BOX))

    cells.append(box("db_down", "&lt;b&gt;Linear (down_proj)&lt;/b&gt; · Expert Down Projection", bx0 + 108, em_y0 + 14, 360, 28, S_LIN))
    cells.append(line_pts("db_e_dn_out", [(bcx, em_y0 + 14), (bcx, em_y0 - 2)]))
    cells.append(circle_op("db_mul", "×", bcx, em_y0 + 60, r=10))
    cells.append(line_pts("db_e_mul_dn", [(bcx, em_y0 + 50), (bcx, em_y0 + 42)]))

    bgx, bux = bx0 + 140, bx0 + 436
    cells.append(box("db_silu", "&lt;b&gt;SiLU&lt;/b&gt;", bgx - 60, em_y0 + 74, 120, 24, S_NORM))
    cells.append(box("db_gate", "&lt;b&gt;Gate: Linear (gate_proj)&lt;/b&gt;", bgx - 85, em_y0 + 112, 170, 30, S_LIN))
    cells.append(box("db_up", "&lt;b&gt;Up: Linear (up_proj)&lt;/b&gt;", bux - 85, em_y0 + 112, 170, 30, S_LIN))
    cells.append(line_pts("db_e_gt_sl", [(bgx, em_y0 + 112), (bgx, em_y0 + 98)]))
    cells.append(line_pts("db_e_sl_ml", [(bgx, em_y0 + 74), (bgx, em_y0 + 60), (bcx - 10, em_y0 + 60)]))
    cells.append(line_pts("db_e_up_ml", [(bux, em_y0 + 112), (bux, em_y0 + 60), (bcx + 10, em_y0 + 60)]))
    cells.append(line_pts("db_e_min_g", [(bcx, em_y0 + 158), (bgx, em_y0 + 158), (bgx, em_y0 + 142)]))
    cells.append(line_pts("db_e_min_u", [(bcx, em_y0 + 158), (bux, em_y0 + 158), (bux, em_y0 + 142)]))
    cells.append(line_pts("db_e_min_st", [(bcx, em_y0 + 174), (bcx, em_y0 + 158)], arrow=False))

    # Panel (b) Right: Sparse MoE Router + Octopus Integration (x=930..1568, y=yB+54..yB+458)
    rx0, ry0 = 930, yB + 54
    rcx = rx0 + 319  # 1249
    cells.append(box("db_moe_title", "&lt;b&gt;MoE (1D Packed Tree Router)&lt;/b&gt;", rx0, ry0 - 22, 260, 20, S_TITLE_LEFT))
    cells.append(box("db_moe_out", "&lt;b&gt;Output&lt;/b&gt;", rcx - 35, ry0 - 22, 70, 20, S_TXT))
    cells.append(box("db_moe_box", "", rx0, ry0, 638, 250, S_MLP_BOX))

    cells.append(circle_op("db_moe_sum", "+", rcx, ry0 + 28, r=11))
    cells.append(line_pts("db_e_msum_out", [(rcx, ry0 + 17), (rcx, ry0 - 2)]))

    w1x, w2x = rx0 + 190, rx0 + 350
    cells.append(circle_op("db_mw1", "×", w1x, ry0 + 62, r=10))
    cells.append(circle_op("db_mw2", "×", w2x, ry0 + 62, r=10))
    cells.append(line_pts("db_e_mw1_s", [(w1x, ry0 + 52), (w1x, ry0 + 28), (rcx - 11, ry0 + 28)]))
    cells.append(line_pts("db_e_mw2_s", [(w2x, ry0 + 52), (w2x, ry0 + 28), (rcx + 11, ry0 + 28)]))

    ex_xs = [rx0 + 95, rx0 + 190, rx0 + 270, rx0 + 350, rx0 + 465]
    ex_labels = ["MLP 1", "MLP 2 ★", "MLP 3", "MLP 4 ★", "MLP E"]
    for idx, (ex, elbl) in enumerate(zip(ex_xs, ex_labels)):
        active = "★" in elbl
        st = (
            "rounded=1;arcSize=25;fillColor=#86EFAC;strokeColor=#065F46;strokeWidth=2.0;fontFamily=Helvetica;fontSize=11.5;fontColor=#064E3B;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
            if active else
            "rounded=1;arcSize=25;fillColor=#BBF7D0;strokeColor=#1E293B;strokeWidth=1.4;fontFamily=Helvetica;fontSize=11.0;fontColor=#0F172A;align=center;verticalAlign=middle;whiteSpace=wrap;html=1;"
        )
        cells.append(box(f"db_exp_{idx}", f"&lt;b&gt;{elbl}&lt;/b&gt;", ex - 34, ry0 + 92, 68, 68, st))

    cells.append(box("db_exp_dots", "&lt;b style=&quot;font-size:16px;&quot;&gt;⋯⋯&lt;/b&gt;", rx0 + 392, ry0 + 114, 36, 24, S_TXT))
    cells.append(box("db_exp_lbl", "&lt;b&gt;Sparse&lt;br&gt;Experts&lt;/b&gt;", rx0 + 6, ry0 + 108, 52, 36, S_TXT))

    cells.append(line_pts("db_e_ex1_w1", [(w1x, ry0 + 92), (w1x, ry0 + 72)]))
    cells.append(line_pts("db_e_ex3_w2", [(w2x, ry0 + 92), (w2x, ry0 + 72)]))

    cells.append(box("db_rgate", "&lt;b&gt;Router Gate: Linear (gate)&lt;/b&gt;", rcx - 110, ry0 + 186, 220, 32, S_LIN))
    cells.append(box(
        "db_topk",
        "&lt;b&gt;Top-K Selection (K=8)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;&quot;&gt;Batched over T&lt;sub&gt;tree&lt;/sub&gt;&lt;/font&gt;",
        rx0 + 455, ry0 + 182, 168, 38, S_ROPE
    ))
    cells.append(line_pts("db_e_rg_ex1", [(rcx - 30, ry0 + 186), (w1x, ry0 + 160)]))
    cells.append(line_pts("db_e_rg_ex3", [(rcx + 30, ry0 + 186), (w2x, ry0 + 160)]))
    cells.append(line_pts("db_e_rg_tk", [(rcx + 110, ry0 + 202), (rx0 + 455, ry0 + 202)], dashed=True))
    cells.append(line_pts("db_e_tk_w2", [(rx0 + 539, ry0 + 182), (rx0 + 539, ry0 + 62), (w2x + 10, ry0 + 62)], dashed=True))
    cells.append(line_pts("db_e_min_rg", [(rcx, ry0 + 232), (rcx, ry0 + 218)]))
    cells.append(box("db_moe_in", "&lt;b&gt;Input: 1D Deduplicated Prefix-Tree Tokens (T&lt;sub&gt;tree&lt;/sub&gt;)&lt;/b&gt;", rcx - 180, ry0 + 232, 360, 17, "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize=11.0;fontColor=#0F172A;align=center;verticalAlign=middle;"))

    mhy0 = yB + 312
    cells.append(box("db_hk_box", "", rx0, mhy0, 638, 158, S_HOOK_BOX))
    cells.append(box(
        "db_hk_hdr",
        "&lt;b style=&quot;font-size:13px;color:#064E3B;&quot;&gt;How Qwen3 (MoE) Hooks into Octopus (Stateless Expert Scaling)&lt;/b&gt;",
        rx0 + 14, mhy0 + 4, 610, 22, S_TITLE_LEFT
    ))
    cells.append(box(
        "db_hk_b1",
        "• &lt;b&gt;Zero MoE State on fork / rewind / read_thread:&lt;/b&gt; Top-K routing is strictly token-wise and stateless across "
        "time; branching &amp; KV mounting operate purely on the &lt;b&gt;Attention Radix Cache&lt;/b&gt; with 0 expert state copies.",
        rx0 + 12, mhy0 + 28, 614, 40, S_BADGE_TEAL
    ))
    cells.append(box(
        "db_hk_b2",
        "• &lt;b&gt;High-Density Grouped GEMM on Parallel Branches:&lt;/b&gt; Spawning W parallel branches and packing deduplicated "
        "&lt;code&gt;T_tree&lt;/code&gt; tokens turns memory-bound single-branch expert calls into high-utilization batched H200 GEMMs.",
        rx0 + 12, mhy0 + 72, 614, 40, S_BADGE_TEAL
    ))
    cells.append(box(
        "db_hk_b3",
        "• &lt;b&gt;Router-Preserving LoRA Adaptation:&lt;/b&gt; Adapts attention projections (&lt;code&gt;q/k/v/o_proj&lt;/code&gt;) while keeping "
        "Top-K router logits &amp; expert balance unperturbed during multi-branch DAG-GRPO reinforcement learning.",
        rx0 + 12, mhy0 + 116, 614, 36, S_BADGE_TEAL
    ))

    body = "\n".join(cells)
    return f"""<mxfile host="Electron" modified="2026-10-26T23:50:00.000Z" agent="OctopusDrawioGenerator" version="24.7.5">
  <diagram id="arch-qwen3" name="Page-1">
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
