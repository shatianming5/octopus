#!/usr/bin/env python3
"""
Rebuilds Figure 1 (Main Framework Overview: fig1_overview.drawio & fig1_overview.pdf) in Draw.io
with:
1. Authentic Octopus Tentacle DAG edges (dual-tone curved tentacle arms with source/target + graduated suction cups)
2. Explicit 32-Layer Hybrid Architecture Stack (8x [3 Gated DeltaNet Recurrent Layers + 1 Full-Attention Layer]
   = 24 GDN + 8 Full-Attention Layers) in Layer 2
3. Curling Octopus Tentacle Cross-Branch Autograd in Layer 3
"""
import base64
import io
import json
import math
import subprocess
from pathlib import Path
from PIL import Image

FIG_DIR = Path("./figures")
ART_DIR = Path("/tmp/octopus-fig-preview")

def b64_icon(name: str, max_px: int = 110) -> str:
    p = FIG_DIR / f"{name}.png"
    img = Image.open(p).convert("RGBA")
    img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    q = img.quantize(colors=64, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO()
    q.save(buf, format="PNG", optimize=True)
    return "data:image/png," + base64.b64encode(buf.getvalue()).decode("ascii")

ICONS = {
    name: b64_icon(name)
    for name in [
        "octopus_cool",
        "octopus_work",
        "octo_07_wall32k",
        "octo_08_chat_retok",
        "octo_09_brainwash",
        "octo_10_layer2_mount",
        "octo_11_layer3_loo",
        "octo_21_title_boss",
        "octo_fork",
        "octo_17_rewind_cut",
        "octo_k2",
    ]
}

def img_cell(cid: str, icon_name: str, x: float, y: float, w: float, h: float) -> str:
    data = ICONS[icon_name]
    return (
        f'<mxCell id="{cid}" value="" '
        f'style="shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;'
        f'verticalAlign=top;aspect=fixed;imageAspect=0;image={data};" '
        f'x="{x:.1f}" y="{y:.1f}" w="{w:.1f}" h="{h:.1f}"/>'
    )

from tentacle_drawio import tentacle_edge

CELLS = f"""
<mxStyle name="panelA" value="rounded=1;arcSize=4;fillColor=#FAFAFA;strokeColor=#CBD5E1;strokeWidth=1.6;fontFamily=Comic Sans MS;"/>
<mxStyle name="panelB" value="rounded=1;arcSize=4;fillColor=#F4FBF9;strokeColor=#0D9488;strokeWidth=2.0;fontFamily=Comic Sans MS;"/>
<mxStyle name="pillRed" value="rounded=1;arcSize=50;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fillColor=#FEE2E2;strokeColor=#FCA5A5;fontColor=#991B1B;"/>
<mxStyle name="pillTeal" value="rounded=1;arcSize=50;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fillColor=#CCFBF1;strokeColor=#5EEAD4;fontColor=#0F766E;"/>
<mxStyle name="boxNode" value="rounded=1;arcSize=22;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.5;strokeWidth=1.5;"/>
<mxStyle name="cNode" value="ellipse;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.5;strokeWidth=1.6;"/>

<mxCell id="zone_a" value="" style="panelA;" x="10" y="10" w="306" h="358"/>
<mxCell id="za_hdr" value="&lt;font color=&quot;#7F1D1D&quot;&gt;Prior Agentic Paradigms&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=12.0;" x="46" y="15" w="166" h="22"/>
<mxCell id="za_tag" value="Stateless Pipe" style="pillRed;" x="214" y="16" w="92" h="20"/>
<mxCell id="za_div0" value="" style="line;strokeColor=#CBD5E1;strokeWidth=1.2;" x="20" y="39" w="286" h="4"/>

<mxCell id="s1_hdr" value="&lt;font color=&quot;#991B1B&quot;&gt;1. Linear Append / Context Edit&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.2;" x="44" y="46" w="168" h="18"/>
<mxCell id="s1_tag" value="O(L²) · 32K Wall" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fontColor=#B91C1C;" x="212" y="46" w="94" h="18"/>
<mxCell id="s1_b1" value="Prompt" style="boxNode;fillColor=#E2E8F0;strokeColor=#475569;fontColor=#1E293B;" x="20" y="70" w="54" h="28"/>
<mxCell id="s1_b2" value="Detour" style="boxNode;fillColor=#FEE2E2;strokeColor=#DC2626;fontColor=#991B1B;" x="88" y="70" w="56" h="28"/>
<mxCell id="s1_b3" value="Edit ⚡" style="boxNode;fillColor=#FEF3C7;strokeColor=#D97706;dashed=1;fontColor=#92400E;" x="158" y="70" w="60" h="28"/>
<mxCell id="s1_b4" value="32K Full!" style="boxNode;fillColor=#FCA5A5;strokeColor=#B91C1C;fontColor=#7F1D1D;" x="232" y="70" w="74" h="28"/>
<mxCell id="s1_e1" value="" style="strokeColor=#64748B;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" source="s1_b1" target="s1_b2"/>
<mxCell id="s1_e2" value="" style="strokeColor=#64748B;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" source="s1_b2" target="s1_b3"/>
<mxCell id="s1_e3" value="" style="strokeColor=#DC2626;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" source="s1_b3" target="s1_b4"/>
<mxCell id="s1_note" value="• Breaks KV cache (41.8% re-prefill) &amp;amp; corrupts h&lt;sub&gt;t&lt;/sub&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fontColor=#475569;" x="20" y="103" w="286" h="18"/>
<mxCell id="za_div1" value="" style="line;dashed=1;strokeColor=#CBD5E1;strokeWidth=1.2;" x="20" y="127" w="286" h="4"/>

<mxCell id="s2_hdr" value="&lt;font color=&quot;#92400E&quot;&gt;2. Stateless Multi-Agent API&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.2;" x="44" y="135" w="164" h="18"/>
<mxCell id="s2_tag" value="O(L) Wire · O(N²) KV" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fontColor=#B45309;" x="198" y="135" w="108" h="18"/>
<mxCell id="s2_app" value="Chat App&lt;br&gt;&lt;font style=&quot;font-size:8.2px;&quot; color=&quot;#B45309&quot;&gt;Re-sends Text&lt;/font&gt;" style="boxNode;fillColor=#FFFBEB;strokeColor=#D97706;fontColor=#92400E;" x="20" y="160" w="74" h="56"/>
<mxCell id="s2_r1" value="[Trunk] [Copy A]  S1" style="boxNode;arcSize=18;fillColor=#F1F5F9;strokeColor=#94A3B8;fontColor=#334155;fontSize=9.0;" x="112" y="160" w="120" h="24"/>
<mxCell id="s2_r2" value="[Trunk] [Re-Tok]  SN" style="boxNode;arcSize=18;fillColor=#FEE2E2;strokeColor=#EF4444;fontColor=#991B1B;fontSize=9.0;" x="112" y="192" w="120" h="24"/>
<mxCell id="s2_e1" value="" style="dashed=1;strokeColor=#D97706;strokeWidth=1.5;exitX=1;exitY=0.3;entryX=0;entryY=0.5;" source="s2_app" target="s2_r1"/>
<mxCell id="s2_e2" value="" style="dashed=1;strokeColor=#D97706;strokeWidth=1.5;exitX=1;exitY=0.7;entryX=0;entryY=0.5;" source="s2_app" target="s2_r2"/>
<mxCell id="s2_kv" value="O(N²) KV&lt;br&gt;&lt;font style=&quot;font-size:8.2px;&quot;&gt;99% Full&lt;/font&gt;" style="boxNode;fillColor=#FEF2F2;strokeColor=#DC2626;fontColor=#991B1B;" x="240" y="160" w="66" h="56"/>
<mxCell id="za_div2" value="" style="line;dashed=1;strokeColor=#CBD5E1;strokeWidth=1.2;" x="20" y="226" w="286" h="4"/>

<mxCell id="s3_hdr" value="&lt;font color=&quot;#9F1239&quot;&gt;3. All-Layer KV / Text Injection&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.2;" x="43" y="234" w="172" h="18"/>
<mxCell id="s3_tag" value="57.9% Hijack" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fontColor=#E11D48;" x="212" y="234" w="94" h="18"/>
<mxCell id="s3_a1" value="A₁" style="cNode;fillColor=#F1F5F9;strokeColor=#334155;fontColor=#1E293B;" x="22" y="264" w="28" h="24"/>
<mxCell id="s3_a2" value="A₂" style="cNode;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="96" y="264" w="28" h="24"/>
<mxCell id="s3_a3" value="A₃" style="cNode;fillColor=#F1F5F9;strokeColor=#334155;fontColor=#1E293B;" x="184" y="264" w="28" h="24"/>
<mxCell id="s3_ta" value="Task A" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.8;fontColor=#1E293B;" x="248" y="264" w="56" h="24"/>
<mxCell id="s3_ea1" value="" style="strokeColor=#334155;strokeWidth=1.8;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" source="s3_a1" target="s3_a2"/>
<mxCell id="s3_ea2" value="" style="strokeColor=#334155;strokeWidth=1.8;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" source="s3_a2" target="s3_a3"/>
<mxCell id="s3_ea3" value="" style="strokeColor=#334155;strokeWidth=1.8;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" source="s3_a3" target="s3_ta"/>
<mxCell id="s3_b1" value="B₁" style="cNode;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E40AF;" x="22" y="324" w="28" h="24"/>
<mxCell id="s3_b2" value="B₂" style="cNode;fillColor=#FFE4E6;strokeColor=#E11D48;fontColor=#9F1239;" x="96" y="324" w="28" h="24"/>
<mxCell id="s3_tb" value="Task B Lost" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.8;fontColor=#94A3B8;" x="238" y="324" w="70" h="24"/>
<mxCell id="s3_eb1" value="" style="strokeColor=#2563EB;strokeWidth=1.8;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" source="s3_b1" target="s3_b2"/>
<mxCell id="s3_eb2" value="" style="dashed=1;strokeColor=#CBD5E1;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" source="s3_b2" target="s3_tb"/>
<mxCell id="s3_einj" value="" style="strokeColor=#E11D48;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" source="s3_a2" target="s3_b2"/>
<mxCell id="s3_inj_t" value="Overwrites h&lt;sub&gt;B&lt;/sub&gt;" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.8;fontColor=#BE123C;" x="16" y="296" w="76" h="18"/>
<mxCell id="s3_ehij" value="" style="curved=1;strokeColor=#E11D48;strokeWidth=2.2;exitX=1;exitY=0;entryX=0;entryY=1;" source="s3_b2" target="s3_a3">
  <mxGeometry relative="1" as="geometry"><Array as="points"><mxPoint x="142" y="296"/></Array></mxGeometry>
</mxCell>
<mxCell id="s3_hij_t" value="B Hijacked!" style="rounded=1;arcSize=25;fillColor=#FFE4E6;strokeColor=#E11D48;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.8;fontColor=#9F1239;" x="164" y="296" w="74" h="18"/>

<!-- ================= RIGHT PANEL (b): OCTOPUS TENTACLE-DAG & 24-GDN/8-ATTN STACK ================= -->
<mxCell id="zone_b" value="" style="panelB;" x="326" y="10" w="468" h="358"/>
<mxCell id="zb_hdr" value="&lt;font color=&quot;#0F766E&quot;&gt;Octopus: Cursor DAG, Hybrid Mount &amp;amp; RL&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=11.4;" x="366" y="15" w="292" h="22"/>
<mxCell id="zb_tag" value="0-Copy &amp;amp; DAG-GRPO" style="pillTeal;" x="660" y="16" w="124" h="20"/>
<mxCell id="zb_div0" value="" style="line;strokeColor=#99F6E4;strokeWidth=1.2;" x="336" y="39" w="448" h="4"/>

<!-- LAYER 1: Execution Layer -->
<mxCell id="l1_hdr" value="&lt;font color=&quot;#0F766E&quot;&gt;Layer 1 (Execution): Cursor Execution DAG&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.5;" x="336" y="44" w="256" h="18"/>
<mxCell id="l1_tag" value="fork · rewind · read (O(Δ) Wire)" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fontColor=#047857;" x="596" y="44" w="188" h="18"/>

<mxCell id="c0" value="c₀" style="cNode;fillColor=#CCFBF1;strokeColor=#0F766E;fontColor=#064E3B;" x="342" y="108" w="32" h="24"/>
<mxCell id="cfork" value="c&lt;sub&gt;fork&lt;/sub&gt;" style="cNode;fillColor=#99F6E4;strokeColor=#0F766E;fontColor=#064E3B;" x="408" y="108" w="42" h="24"/>
<mxCell id="ca1" value="c&lt;sub&gt;A1&lt;/sub&gt;" style="cNode;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="482" y="72" w="36" h="22"/>
<mxCell id="dead" value="×" style="cNode;fillColor=#FEE2E2;strokeColor=#DC2626;fontColor=#991B1B;" x="548" y="61" w="22" h="22"/>
<mxCell id="rw_t" value="rewind" style="text;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;fontColor=#DC2626;" x="572" y="61" w="38" h="14"/>
<mxCell id="ca2" value="c&lt;sub&gt;A2&lt;/sub&gt; (Span A)" style="boxNode;arcSize=50;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;fontSize=9.0;" x="616" y="71" w="80" h="22"/>
<mxCell id="cb1" value="c&lt;sub&gt;B1&lt;/sub&gt;" style="cNode;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E40AF;" x="512" y="111" w="36" h="22"/>
<mxCell id="cb2" value="c&lt;sub&gt;B2&lt;/sub&gt;" style="cNode;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E40AF;" x="640" y="111" w="36" h="22"/>
<mxCell id="mnt_t" value="0-Copy read" style="rounded=1;arcSize=30;fillColor=#FEF3C7;strokeColor=#D97706;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.0;fontColor=#92400E;" x="574" y="96" w="62" h="14"/>
<mxCell id="cc1" value="c&lt;sub&gt;C1&lt;/sub&gt;" style="cNode;fillColor=#CCFBF1;strokeColor=#0D9488;fontColor=#065F46;" x="496" y="141" w="36" h="22"/>
<mxCell id="cc2" value="c&lt;sub&gt;C2&lt;/sub&gt;" style="cNode;fillColor=#CCFBF1;strokeColor=#0D9488;fontColor=#065F46;" x="600" y="141" w="36" h="22"/>
<mxCell id="done" value="done" style="boxNode;arcSize=50;fillColor=#0F766E;strokeColor=#064E3B;fontColor=#FFFFFF;fontSize=9.5;" x="730" y="110" w="48" h="24"/>

{ tentacle_edge("t_c0_cf", "c0", "cfork", (1.0, 0.5), (0.0, 0.5), (370, 120), (390, 120), (408, 120), outer_col="#0F766E", width=4.6) }
{ tentacle_edge("t_cf_ca1", "cfork", "ca1", (0.85, 0.15), (0.0, 0.5), (442, 112), (460, 84), (482, 83), outer_col="#EA580C", width=4.5) }
{ tentacle_edge("t_ca1_dead", "ca1", "dead", (1.0, 0.25), (0.0, 0.5), (516, 78), (533, 68), (548, 72), outer_col="#94A3B8", width=3.2, dashed=True) }
{ tentacle_edge("t_dead_rw", "dead", "ca1", (0.2, 0.0), (0.5, 0.0), (548, 66), (525, 55), (504, 72), outer_col="#E11D48", width=3.5) }
{ tentacle_edge("t_ca1_ca2", "ca1", "ca2", (1.0, 0.75), (0.0, 0.5), (516, 87), (566, 94), (616, 82), outer_col="#D97706", width=4.4) }

{ tentacle_edge("t_cf_cb1", "cfork", "cb1", (1.0, 0.5), (0.0, 0.5), (448, 120), (481, 122), (512, 122), outer_col="#2563EB", width=4.5) }
{ tentacle_edge("t_cb1_cb2", "cb1", "cb2", (1.0, 0.5), (0.0, 0.5), (546, 122), (594, 122), (640, 122), outer_col="#2563EB", width=4.4) }

{ tentacle_edge("t_mnt_ab", "ca2", "cb2", (0.5, 1.0), (0.5, 0.0), (656, 92), (658, 102), (658, 111), outer_col="#D97706", width=3.8, dashed=True) }

{ tentacle_edge("t_cf_cc1", "cfork", "cc1", (0.85, 0.85), (0.0, 0.5), (442, 128), (468, 152), (496, 152), outer_col="#0D9488", width=4.5) }
{ tentacle_edge("t_cc1_cc2", "cc1", "cc2", (1.0, 0.5), (0.0, 0.5), (530, 152), (566, 152), (600, 152), outer_col="#0D9488", width=4.4) }

{ tentacle_edge("t_ca2_done", "ca2", "done", (1.0, 0.5), (0.3, 0.0), (694, 82), (726, 88), (744, 110), outer_col="#D97706", width=3.5, dashed=True) }
{ tentacle_edge("t_cb2_done", "cb2", "done", (1.0, 0.5), (0.0, 0.5), (674, 122), (703, 122), (730, 122), outer_col="#2563EB", width=4.2) }
{ tentacle_edge("t_cc2_done", "cc2", "done", (1.0, 0.5), (0.3, 1.0), (634, 152), (696, 152), (744, 134), outer_col="#0D9488", width=3.5, dashed=True) }

<mxCell id="zb_div1" value="" style="line;dashed=1;strokeColor=#99F6E4;strokeWidth=1.2;" x="336" y="168" w="448" h="4"/>

<!-- LAYER 2: Radix Pool & Layer-Selective Hybrid KV Mount -->
<mxCell id="l2_hdr" value="&lt;font color=&quot;#1E40AF&quot;&gt;Layer 2 (Memory): Radix Pool &amp;amp; Hybrid KV Mount&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.8;" x="366" y="173" w="290" h="18"/>
<mxCell id="l2_tag" value="Attn Mount · GDN Skips Span" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.8;fontColor=#1E40AF;" x="634" y="173" w="150" h="18"/>

<mxCell id="pg0" value="Page 0–1&lt;br&gt;&lt;font style=&quot;font-size:8.0px;&quot; color=&quot;#0F766E&quot;&gt;Trunk (ref=3)&lt;/font&gt;" style="boxNode;fillColor=#E2E8F0;strokeColor=#475569;fontColor=#1E293B;fontSize=8.8;" x="336" y="194" w="66" h="56"/>
<mxCell id="pg4" value="Page 4–5&lt;br&gt;&lt;font style=&quot;font-size:8.0px;&quot; color=&quot;#B91C1C&quot;&gt;Span A (0-Copy)&lt;/font&gt;" style="boxNode;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;fontSize=8.8;" x="405" y="194" w="74" h="56"/>
<mxCell id="pg7" value="Page 7&lt;br&gt;&lt;font style=&quot;font-size:8.0px;&quot; color=&quot;#1E40AF&quot;&gt;Delta B (ref=1)&lt;/font&gt;" style="boxNode;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E3A8A;fontSize=8.8;" x="482" y="194" w="68" h="56"/>

<!-- 32-Layer Hybrid Architecture Stack Box (×8 Blocks: 24 GDN Layers + 8 Full-Attention Layers) -->
<mxCell id="hyb_box" value="" style="rounded=1;arcSize=8;fillColor=#FFFFFF;strokeColor=#0D9488;strokeWidth=1.4;" x="554" y="191" w="230" h="62"/>
<mxCell id="gdn_l1" value="GDN 4k+1: Step Δ&lt;sub&gt;B&lt;/sub&gt; &lt;font color=&quot;#DC2626&quot;&gt;[✗ Skip A]&lt;/font&gt;" style="rounded=1;arcSize=22;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.2;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E40AF;" x="558" y="194" w="177" h="12"/>
<mxCell id="gdn_l2" value="GDN 4k+2: Step Δ&lt;sub&gt;B&lt;/sub&gt; &lt;font color=&quot;#DC2626&quot;&gt;[✗ Skip A]&lt;/font&gt;" style="rounded=1;arcSize=22;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.2;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E40AF;" x="558" y="207" w="177" h="12"/>
<mxCell id="gdn_l3" value="GDN 4k+3: Step Δ&lt;sub&gt;B&lt;/sub&gt; &lt;font color=&quot;#DC2626&quot;&gt;[✗ Skip A]&lt;/font&gt;" style="rounded=1;arcSize=22;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.2;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E40AF;" x="558" y="220" w="177" h="12"/>
<mxCell id="att_l4" value="Attn 4k+4: [K₀ ∥ K&lt;sub&gt;A&lt;/sub&gt; ∥ K&lt;sub&gt;B&lt;/sub&gt;] &lt;font color=&quot;#047857&quot;&gt;[⚡Mount]&lt;/font&gt;" style="rounded=1;arcSize=22;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.5;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="558" y="234" w="177" h="16"/>
<mxCell id="hyb_gdn_p" value="24 GDN&lt;br&gt;(×8×3)" style="rounded=1;arcSize=18;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.0;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E3A8A;" x="738" y="194" w="42" h="38"/>
<mxCell id="hyb_att_p" value="8 Attn" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.2;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="738" y="234" w="42" h="16"/>
{ tentacle_edge("t_pg4_att", "pg4", "att_l4", (0.85, 1.0), (0.0, 0.5), (464, 248), (512, 256), (558, 242), outer_col="#D97706", width=3.8) }

<mxCell id="zb_div2" value="" style="line;dashed=1;strokeColor=#99F6E4;strokeWidth=1.2;" x="336" y="256" w="448" h="4"/>

<!-- LAYER 3: RL Packed Tree-and-Mount DAG-GRPO -->
<mxCell id="l3_hdr" value="&lt;font color=&quot;#581C87&quot;&gt;Layer 3 (RL): Packed Tree-and-Mount DAG-GRPO&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.5;" x="336" y="261" w="285" h="18"/>
<mxCell id="l3_tag" value="LoRA θ · R = 2y − 1 · 8.1× Tok" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fontColor=#6B21A8;" x="618" y="261" w="166" h="18"/>

<mxCell id="t1" value="Trunk u₀&lt;br&gt;&lt;font style=&quot;font-size:8.2px;&quot; color=&quot;#0F766E&quot;&gt;1× Shared&lt;/font&gt;" style="boxNode;fillColor=#E2E8F0;strokeColor=#475569;fontColor=#1E293B;fontSize=9.0;" x="336" y="314" w="78" h="40"/>
<mxCell id="t2" value="Source u&lt;sub&gt;A&lt;/sub&gt; (m=1)&lt;br&gt;&lt;font style=&quot;font-size:8.2px;&quot; color=&quot;#92400E&quot;&gt;Retains K&lt;sub&gt;A&lt;/sub&gt;(θ)&lt;/font&gt;" style="boxNode;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;fontSize=8.8;" x="420" y="314" w="90" h="40"/>
<mxCell id="t3" value="Rewind l&lt;sub&gt;post&lt;/sub&gt;&lt;br&gt;&lt;font style=&quot;font-size:8.2px;&quot; color=&quot;#065F46&quot;&gt;GDN Hook&lt;/font&gt;" style="boxNode;fillColor=#CCFBF1;strokeColor=#0D9488;fontColor=#065F46;fontSize=9.0;" x="516" y="314" w="84" h="40"/>
<mxCell id="t4" value="Reader u&lt;sub&gt;B&lt;/sub&gt; (m=1)&lt;br&gt;&lt;font style=&quot;font-size:8.2px;&quot; color=&quot;#1E40AF&quot;&gt;Mounts K&lt;sub&gt;A&lt;/sub&gt;(θ)&lt;/font&gt;" style="boxNode;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E40AF;fontSize=8.8;" x="606" y="314" w="86" h="40"/>
<mxCell id="t5" value="LOO Credit&lt;br&gt;&lt;font style=&quot;font-size:8.5px;&quot;&gt;Â&lt;sub&gt;i&lt;/sub&gt; = R&lt;sub&gt;i&lt;/sub&gt; − μ&lt;sub&gt;R&lt;/sub&gt;&lt;/font&gt;" style="boxNode;fillColor=#ECFDF5;strokeColor=#059669;fontColor=#065F46;fontSize=9.0;" x="698" y="308" w="86" h="46"/>

{ tentacle_edge("t_grad", "t4", "t2", (0.5, 0.0), (0.5, 0.0), (649, 314), (556, 286), (465, 314), outer_col="#E11D48", width=4.4) }

<mxCell id="grad_t" value="On-Policy Mount Edge (KL = 4.45e−4)" style="rounded=1;arcSize=30;fillColor=#FFF1F2;strokeColor=#FECDD3;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;fontColor=#BE123C;" x="466" y="278" w="184" h="16"/>

<mxCell id="cap_a" value="(a) Prior Paradigms: Context Pollution &amp;amp; Hijacking" style="text;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.5;fontColor=#0F172A;" x="10" y="372" w="306" h="20"/>
<mxCell id="cap_b" value="(b) Octopus: Cursor Execution DAG, Hybrid KV Mount &amp;amp; DAG-GRPO" style="text;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.5;fontColor=#0F172A;" x="326" y="372" w="468" h="20"/>

{img_cell("ic_cool", "octopus_cool", 334, 10, 30, 28)}
{img_cell("ic_work", "octo_21_title_boss", 380, 66, 44, 42)}
{img_cell("ic_07", "octo_07_wall32k", 16, 41, 26, 27)}
{img_cell("ic_08", "octo_08_chat_retok", 16, 130, 26, 26)}
{img_cell("ic_09", "octo_09_brainwash", 17, 228, 23, 30)}
{img_cell("ic_10", "octo_10_layer2_mount", 336, 170, 28, 23)}
{img_cell("ic_11", "octo_11_layer3_loo", 360, 280, 38, 32)}
"""

clean_xml = "\n".join(
    line for line in CELLS.strip().splitlines() if not line.strip().startswith("<!--")
)

mcp_script = "/opt/homebrew/lib/node_modules/@next-ai-drawio/mcp-server/dist/index.js"
drawio_out = FIG_DIR / "fig1_overview.drawio"
pdf_out = FIG_DIR / "fig1_overview.pdf"

p = subprocess.Popen(
    ["/opt/homebrew/bin/node", mcp_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

def rpc(req):
    p.stdin.write(json.dumps(req) + "\n")
    p.stdin.flush()
    if "id" not in req:
        return None
    while True:
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("EOF from MCP server: " + p.stderr.read())
        if line.strip().startswith("{"):
            msg = json.loads(line)
            if msg.get("id") == req["id"]:
                return msg

rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "agy", "version": "1.0"}}})
rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "start_session", "arguments": {}}})
r_create = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "create_new_diagram", "arguments": {"xml": clean_xml, "page_name": "Figure 1 Octopus Tentacle-DAG Overview", "screenshot": False}}})
print("create_new_diagram:", r_create.get("result", {}).get("content", [{}])[0].get("text", "")[:250])
r_export = rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "export_diagram", "arguments": {"path": str(drawio_out), "format": "drawio"}}})
print("export_diagram:", r_export.get("result", {}).get("content", [{}])[0].get("text", "")[:200])
p.terminate()

raw_pdf = Path("/tmp/fig1_tentacle_raw.pdf")
subprocess.run([
    "/Applications/draw.io.app/Contents/MacOS/draw.io",
    "--export",
    "--format", "pdf",
    "--crop",
    "--output", str(raw_pdf),
    str(drawio_out),
], check=True)

subprocess.run([
    "/Library/TeX/texbin/pdfcrop",
    "--margins", "2",
    str(raw_pdf),
    str(pdf_out),
], check=True)
raw_pdf.unlink(missing_ok=True)

subprocess.run([
    "pdftoppm",
    "-png",
    "-r", "220",
    str(pdf_out),
    str(ART_DIR / "fig1_tentacle_v2"),
], check=True)
print("Exported fig1_overview.pdf and preview PNG successfully!")
