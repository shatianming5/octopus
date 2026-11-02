#!/usr/bin/env python3
"""
Generate figures/fig2_cursor_mount.drawio and figures/fig2_cursor_mount.pdf via Draw.io MCP + Draw.io CLI:
- Panel (a): Radix Cursor Tree with Octopus Tentacle Edges + H200 Physical KV Page Pool
- Panel (b): Explicit 32-Layer Hybrid Architecture Stack (×8 Blocks: 3x GDN Layers 4k+1..3 [24L Private h_B, Skip A]
             + 1x Full-Attention Layer 4k+4 [8L 0-Copy Mount]) with Octopus Tentacle Bypass Arm
- Panel (c): 1D RoPE Coordinate Rules (Unshifted vs. RoPE Copy vs. Octopus Monotonic Jump)
"""
import base64
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(".")
FIG_DIR = ROOT / "figures"

def b64_img(name: str) -> str:
    p = FIG_DIR / f"{name}.png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png,{data}"

def img_cell(cid: str, name: str, x: float, y: float, w: float, h: float) -> str:
    uri = b64_img(name)
    return (
        f'<mxCell id="{cid}" value="" '
        f'style="shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;'
        f'verticalAlign=top;aspect=fixed;imageAspect=0;image={uri};" '
        f'x="{x}" y="{y}" w="{w}" h="{h}"/>'
    )

from tentacle_drawio import tentacle_edge

CELLS = f"""
<mxStyle name="panelTeal" value="rounded=1;arcSize=4;fillColor=#F4FBF9;strokeColor=#0D9488;strokeWidth=2;fontFamily=Comic Sans MS;"/>
<mxStyle name="panelStone" value="rounded=1;arcSize=4;fillColor=#FAF8F5;strokeColor=#CBD5E1;strokeWidth=1.8;fontFamily=Comic Sans MS;"/>
<mxStyle name="cTeal" value="ellipse;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.8;fillColor=#CCFBF1;strokeColor=#0F766E;strokeWidth=1.6;fontColor=#064E3B;"/>
<mxStyle name="bAmber" value="rounded=1;arcSize=24;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.5;fontColor=#92400E;"/>
<mxStyle name="bBlue" value="rounded=1;arcSize=24;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=1.5;fontColor=#1E40AF;"/>
<mxStyle name="bTeal" value="rounded=1;arcSize=24;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;fillColor=#CCFBF1;strokeColor=#0D9488;strokeWidth=1.5;fontColor=#065F46;"/>
<mxStyle name="bSlate" value="rounded=1;arcSize=24;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;fillColor=#E2E8F0;strokeColor=#475569;strokeWidth=1.5;fontColor=#1E293B;"/>
<mxStyle name="bRose" value="rounded=1;arcSize=24;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;fillColor=#FFE4E6;strokeColor=#E11D48;strokeWidth=1.5;fontColor=#9F1239;"/>
<mxStyle name="pillTeal" value="rounded=1;arcSize=45;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.8;fillColor=#CCFBF1;strokeColor=#0D9488;fontColor=#065F46;"/>
<mxStyle name="pillAmber" value="rounded=1;arcSize=45;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.8;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;"/>

<!-- ================= PANEL (a): ZERO-COPY RADIX PAGE LENDING ================= -->
<mxCell id="pa" value="" style="panelTeal;" x="8" y="8" w="252" h="256"/>
<mxCell id="pa_hdr" value="&lt;font color=&quot;#0F766E&quot;&gt;Radix Cursor &amp;amp; Page Pool&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.2;" x="18" y="13" w="156" h="20"/>
<mxCell id="pa_pill" value="0-Copy Lend" style="pillTeal;" x="178" y="14" w="74" h="18"/>
<mxCell id="pa_div0" value="" style="line;strokeColor=#99F6E4;strokeWidth=1.2;" x="16" y="36" w="236" h="4"/>

<mxCell id="pa_s1" value="&lt;font color=&quot;#0F766E&quot;&gt;1. Radix Cursor Tree&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.8;" x="18" y="40" w="136" h="16"/>
<mxCell id="pa_s1_r" value="O(Δ) Render" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=1;fontSize=8.2;fontColor=#047857;" x="160" y="40" w="92" h="16"/>

<mxCell id="a_c0" value="c₀" style="cTeal;" x="18" y="80" w="32" h="24"/>
<mxCell id="a_cf" value="c&lt;sub&gt;fork&lt;/sub&gt;" style="cTeal;fillColor=#99F6E4;" x="68" y="80" w="40" h="24"/>
<mxCell id="a_ca" value="c&lt;sub&gt;A&lt;/sub&gt; [s&lt;sub&gt;A&lt;/sub&gt;, e&lt;sub&gt;A&lt;/sub&gt;]" style="bAmber;arcSize=45;" x="134" y="58" w="114" h="24"/>
<mxCell id="a_ma" value="−M&lt;sub&gt;A&lt;/sub&gt;" style="bAmber;arcSize=45;dashed=1;" x="134" y="102" w="48" h="24"/>
<mxCell id="a_cb" value="c'&lt;sub&gt;B&lt;/sub&gt; (Δ&lt;sub&gt;B&lt;/sub&gt;)" style="bBlue;arcSize=45;" x="196" y="102" w="54" h="24"/>
<mxCell id="a_0cp" value="0-Copy" style="pillAmber;fontSize=7.2;" x="164" y="84" w="46" h="15"/>

{ tentacle_edge("ta_0_f", "a_c0", "a_cf", (1.0, 0.5), (0.0, 0.5), (48, 92), (59, 87), (68, 92), outer_col="#0F766E", inner_col="#5EEAD4", cup_fill="#CCFBF1", cup_stroke="#0F766E", width=4.2, n_cups=2, cup_side=1.0) }
{ tentacle_edge("ta_f_a", "a_cf", "a_ca", (0.85, 0.15), (0.0, 0.5), (102, 83), (116, 68), (134, 70), outer_col="#EA580C", inner_col="#FDBA74", cup_fill="#FFEDD5", cup_stroke="#C2410C", width=4.0, n_cups=2, cup_side=-1.0) }
{ tentacle_edge("ta_f_m", "a_cf", "a_ma", (0.85, 0.85), (0.0, 0.5), (102, 101), (116, 114), (134, 114), outer_col="#2563EB", inner_col="#93C5FD", cup_fill="#DBEAFE", cup_stroke="#1D4ED8", width=4.0, n_cups=2, cup_side=1.0) }
{ tentacle_edge("ta_m_b", "a_ma", "a_cb", (1.0, 0.5), (0.0, 0.5), (180, 114), (188, 110), (196, 114), outer_col="#2563EB", inner_col="#93C5FD", cup_fill="#DBEAFE", cup_stroke="#1D4ED8", width=3.8, n_cups=1, cup_side=1.0) }
{ tentacle_edge("ta_lend", "a_ca", "a_ma", (0.22, 1.0), (0.5, 0.0), (158, 82), (144, 92), (158, 102), outer_col="#D97706", inner_col="#FDE047", cup_fill="#FEF3C7", cup_stroke="#B45309", width=3.6, n_cups=2, cup_side=-1.0) }

<mxCell id="pa_n1" value="• Placeholder −M&lt;sub&gt;A&lt;/sub&gt; links Span A in O(1)" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.8;fontColor=#334155;" x="18" y="129" w="232" h="14"/>
<mxCell id="pa_div1" value="" style="line;dashed=1;strokeColor=#99F6E4;strokeWidth=1.2;" x="16" y="145" w="236" h="4"/>

<mxCell id="pa_s2" value="&lt;font color=&quot;#1E40AF&quot;&gt;2. H200 Physical KV Pool&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.8;" x="18" y="150" w="152" h="16"/>
<mxCell id="pa_s2_r" value="1.94M Tok" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.2;fontColor=#1E40AF;" x="174" y="150" w="78" h="16"/>

<mxCell id="a_p0" value="Page 0–1&lt;br&gt;&lt;font style=&quot;font-size:7.4px;&quot; color=&quot;#0F766E&quot;&gt;Trunk (ref=2)&lt;/font&gt;" style="bSlate;" x="18" y="170" w="74" h="42"/>
<mxCell id="a_p3" value="Page 3–4&lt;br&gt;&lt;font style=&quot;font-size:7.4px;&quot; color=&quot;#B91C1C&quot;&gt;Span A (1→2!)&lt;/font&gt;" style="bAmber;" x="96" y="170" w="78" h="42"/>
<mxCell id="a_p6" value="Page 6&lt;br&gt;&lt;font style=&quot;font-size:7.4px;&quot; color=&quot;#1E40AF&quot;&gt;Delta B (ref=1)&lt;/font&gt;" style="bBlue;" x="178" y="170" w="74" h="42"/>

<mxCell id="a_inv" value="N&lt;sub&gt;avail&lt;/sub&gt; + N&lt;sub&gt;evict&lt;/sub&gt; + N&lt;sub&gt;prot&lt;/sub&gt; = N&lt;sub&gt;pool&lt;/sub&gt; [0 New]" style="bTeal;fontSize=7.8;" x="18" y="218" w="234" h="22"/>
<mxCell id="a_n2" value="• Freed once when last holder ref → 0" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.8;fontColor=#334155;" x="18" y="243" w="232" h="14"/>

<!-- ================= PANEL (b): LAYER-SELECTIVE HYBRID KV MOUNT ================= -->
<mxCell id="pb" value="" style="panelTeal;" x="268" y="8" w="264" h="256"/>
<mxCell id="pb_hdr" value="&lt;font color=&quot;#0F766E&quot;&gt;Layer-Selective Hybrid Mount&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.8;" x="278" y="13" w="174" h="20"/>
<mxCell id="pb_pill" value="21×↓ Hijack" style="pillTeal;" x="454" y="14" w="70" h="18"/>
<mxCell id="pb_div0" value="" style="line;strokeColor=#99F6E4;strokeWidth=1.2;" x="276" y="36" w="248" h="4"/>

<mxCell id="pb_s1" value="&lt;font color=&quot;#0F766E&quot;&gt;1. ×8 Hybrid Blocks (32 Layers)&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.6;" x="278" y="40" w="176" h="16"/>
<mxCell id="pb_s1_r" value="ℓ ∈ L&lt;sub&gt;mount&lt;/sub&gt;" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.2;fontColor=#047857;" x="454" y="40" w="68" h="16"/>

<mxCell id="b_bus_a" value="Sibling A KV&lt;br&gt;&lt;font style=&quot;font-size:7.1px;&quot;&gt;Pages K&lt;sub&gt;A&lt;/sub&gt;, V&lt;sub&gt;A&lt;/sub&gt;&lt;/font&gt;" style="bAmber;" x="278" y="58" w="116" h="28"/>
<mxCell id="b_bus_b" value="Reader B State&lt;br&gt;&lt;font style=&quot;font-size:7.1px;&quot;&gt;Private h&lt;sub&gt;B&lt;/sub&gt;, Δ&lt;sub&gt;B&lt;/sub&gt;&lt;/font&gt;" style="bBlue;" x="406" y="58" w="116" h="28"/>

<!-- Repeating Hybrid Block Frame (3x GDN Layers + 1x Full-Attention Layer) -->
<mxCell id="b_blk" value="" style="rounded=1;arcSize=6;fillColor=#FFFFFF;strokeColor=#0D9488;strokeWidth=1.4;" x="278" y="92" w="244" h="96"/>

<mxCell id="b_g1" value="🔒 GDN Layer 4k+1: h&lt;sub&gt;B&lt;/sub&gt; ← f(h&lt;sub&gt;B&lt;/sub&gt;, Δ&lt;sub&gt;B&lt;/sub&gt;) &lt;font color=&quot;#DC2626&quot;&gt;[✗ Skip A]&lt;/font&gt;" style="rounded=1;arcSize=22;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.1;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E40AF;" x="292" y="96" w="198" h="16"/>
<mxCell id="b_g2" value="🔒 GDN Layer 4k+2: h&lt;sub&gt;B&lt;/sub&gt; ← f(h&lt;sub&gt;B&lt;/sub&gt;, Δ&lt;sub&gt;B&lt;/sub&gt;) &lt;font color=&quot;#DC2626&quot;&gt;[✗ Skip A]&lt;/font&gt;" style="rounded=1;arcSize=22;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.1;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E40AF;" x="292" y="115" w="198" h="16"/>
<mxCell id="b_g3" value="🔒 GDN Layer 4k+3: h&lt;sub&gt;B&lt;/sub&gt; ← f(h&lt;sub&gt;B&lt;/sub&gt;, Δ&lt;sub&gt;B&lt;/sub&gt;) &lt;font color=&quot;#DC2626&quot;&gt;[✗ Skip A]&lt;/font&gt;" style="rounded=1;arcSize=22;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.1;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E40AF;" x="292" y="134" w="198" h="16"/>
<mxCell id="b_tag_gdn" value="24&lt;br&gt;GDN" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=6.3;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E3A8A;" x="493" y="96" w="24" h="54"/>

<mxCell id="b_a4" value="⚡ Full-Attn Layer 4k+4: [K₀ ∥ K&lt;sub&gt;A&lt;/sub&gt; (0-Copy) ∥ K&lt;sub&gt;B&lt;/sub&gt;]&lt;br&gt;&lt;font style=&quot;font-size:6.8px;&quot; color=&quot;#047857&quot;&gt;0-Copy Spliced Pages · Monotonic Jump s(c'&lt;sub&gt;B&lt;/sub&gt;)&lt;/font&gt;" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.2;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="292" y="153" w="198" h="31"/>
<mxCell id="b_tag_att" value="8&lt;br&gt;Attn" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=6.3;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="493" y="153" w="24" h="31"/>

{ tentacle_edge("tb_bypass", "b_bus_a", "b_a4", (0.12, 1.0), (0.0, 0.5), (290, 86), (274, 134), (292, 168), outer_col="#D97706", inner_col="#FDE047", cup_fill="#FEF3C7", cup_stroke="#B45309", width=3.6, n_cups=3, cup_side=-1.0) }
<mxCell id="tb_state" value="" style="strokeColor=#2563EB;strokeWidth=2.0;exitX=0.72;exitY=1;entryX=0.88;entryY=0;" source="b_bus_b" target="b_g1"/>

<mxCell id="pb_div1" value="" style="line;dashed=1;strokeColor=#99F6E4;strokeWidth=1.2;" x="276" y="191" w="248" h="4"/>
<mxCell id="pb_s2" value="&lt;font color=&quot;#0F766E&quot;&gt;2. Cross-Branch Probe Fidelity&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;" x="302" y="195" w="168" h="16"/>
<mxCell id="pb_s2_r" value="N = 366" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.0;fontColor=#047857;" x="472" y="195" w="50" h="16"/>

<mxCell id="b_res" value="97.3% Retrieval Rank · 2.7% Task Hijack" style="bTeal;fontSize=8.0;" x="278" y="218" w="244" h="22"/>
<mxCell id="b_n2" value="• vs. 57.9% Text / 55.2% All-Layer Replay" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.8;fontColor=#334155;" x="278" y="243" w="244" h="14"/>

<!-- ================= PANEL (c): 1D ROPE POSITION RULES ================= -->
<mxCell id="pc" value="" style="panelStone;" x="540" y="8" w="252" h="256"/>
<mxCell id="pc_hdr" value="&lt;font color=&quot;#1E293B&quot;&gt;1D RoPE Position Rules&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.0;" x="550" y="13" w="150" h="20"/>
<mxCell id="pc_pill" value="Ahead e&lt;sub&gt;A&lt;/sub&gt; &amp;gt; p&lt;sub&gt;B&lt;/sub&gt;" style="pillAmber;" x="702" y="14" w="82" h="18"/>
<mxCell id="pc_div0" value="" style="line;strokeColor=#CBD5E1;strokeWidth=1.2;" x="548" y="36" w="236" h="4"/>

<mxCell id="pc_s1" value="&lt;font color=&quot;#991B1B&quot;&gt;1. Unshifted Pos.&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.6;" x="572" y="40" w="110" h="16"/>
<mxCell id="pc_s1_r" value="−16.5% EM Drop" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.0;fontColor=#DC2626;" x="684" y="40" w="100" h="16"/>
<mxCell id="c_u1" value="c&lt;sub&gt;B&lt;/sub&gt; [0..120]" style="bBlue;fontSize=7.8;" x="550" y="59" w="86" h="22"/>
<mxCell id="c_u2" value="Span A [80..200] &amp;gt; p&lt;sub&gt;B&lt;/sub&gt;!" style="bRose;fontSize=7.8;" x="642" y="59" w="142" h="22"/>
<mxCell id="c_un" value="• Keys in &quot;future&quot; invert causal distance (0.459)" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.5;fontColor=#475569;" x="550" y="83" w="234" h="14"/>
<mxCell id="pc_div1" value="" style="line;dashed=1;strokeColor=#CBD5E1;strokeWidth=1.2;" x="548" y="99" w="236" h="4"/>

<mxCell id="pc_s2" value="&lt;font color=&quot;#92400E&quot;&gt;2. RoPE Copy&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.6;" x="572" y="103" w="100" h="16"/>
<mxCell id="pc_s2_r" value="O(N²) Page Copy" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.0;fontColor=#B45309;" x="676" y="103" w="108" h="16"/>
<mxCell id="c_r1" value="c&lt;sub&gt;B&lt;/sub&gt; [0..120]" style="bBlue;fontSize=7.8;" x="550" y="122" w="80" h="22"/>
<mxCell id="c_r2" value="R&lt;sub&gt;Δθ&lt;/sub&gt; Copy(K&lt;sub&gt;A&lt;/sub&gt;)" style="bAmber;dashed=1;fontSize=7.8;" x="634" y="122" w="96" h="22"/>
<mxCell id="c_r3" value="Δ&lt;sub&gt;B&lt;/sub&gt;" style="bBlue;fontSize=7.8;" x="734" y="122" w="50" h="22"/>
<mxCell id="c_rn" value="• Mutates keys &amp;amp; allocates session-private pages" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.5;fontColor=#475569;" x="550" y="146" w="234" h="14"/>
<mxCell id="pc_div2" value="" style="line;dashed=1;strokeColor=#CBD5E1;strokeWidth=1.2;" x="548" y="162" w="236" h="4"/>

<mxCell id="pc_s3" value="&lt;font color=&quot;#0F766E&quot;&gt;3. Monotonic Jump&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.6;" x="572" y="166" w="110" h="16"/>
<mxCell id="pc_s3_r" value="0.630 EM · 0-Copy" style="text;align=right;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.0;fontColor=#047857;" x="680" y="166" w="104" h="16"/>
<mxCell id="c_m1" value="c&lt;sub&gt;B&lt;/sub&gt; [0..120]" style="bBlue;fontSize=7.8;" x="550" y="185" w="76" h="22"/>
<mxCell id="c_m2" value="0-Copy K&lt;sub&gt;A&lt;/sub&gt;" style="bAmber;fontSize=7.8;" x="630" y="185" w="76" h="22"/>
<mxCell id="c_m3" value="Δ&lt;sub&gt;B&lt;/sub&gt; @ 201+" style="bTeal;fillColor=#0F766E;fontColor=#FFFFFF;fontSize=7.8;" x="710" y="185" w="74" h="22"/>
<mxCell id="c_meq" value="s(c'&lt;sub&gt;B&lt;/sub&gt;) = max(pos&lt;sub&gt;next&lt;/sub&gt;, pos&lt;sub&gt;end&lt;/sub&gt;(A)+1) [Causal]" style="bTeal;fontSize=7.5;" x="550" y="218" w="234" h="22"/>
<mxCell id="c_mn" value="• Native K&lt;sub&gt;A&lt;/sub&gt; pos + monotonic jump (0 New Pages)" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.5;fontColor=#334155;" x="550" y="243" w="234" h="14"/>

<mxCell id="cap_pa" value="(a) Zero-Copy Radix Page Lending" style="text;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fontColor=#0F172A;" x="8" y="268" w="252" h="18"/>
<mxCell id="cap_pb" value="(b) Layer-Selective Hybrid KV Mount" style="text;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fontColor=#0F172A;" x="268" y="268" w="264" h="18"/>
<mxCell id="cap_pc" value="(c) Monotonic Position Coordinate Jump" style="text;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;fontColor=#0F172A;" x="540" y="268" w="252" h="18"/>

{img_cell("ic_rad", "octo_radix", 74, 56, 26, 24)}
{img_cell("ic_mnt", "octo_10_layer2_mount", 278, 194, 22, 20)}
{img_cell("ic_u1", "octo_09_brainwash", 548, 38, 20, 20)}
{img_cell("ic_u2", "octo_08_chat_retok", 548, 101, 20, 20)}
{img_cell("ic_u3", "octopus_cool", 548, 164, 20, 20)}
"""

clean_xml = "\n".join(
    line for line in CELLS.strip().splitlines() if not line.strip().startswith("<!--")
)

mcp_script = "/opt/homebrew/lib/node_modules/@next-ai-drawio/mcp-server/dist/index.js"
drawio_out = FIG_DIR / "fig2_cursor_mount.drawio"
pdf_out = FIG_DIR / "fig2_cursor_mount.pdf"

p = subprocess.Popen(
    ["/opt/homebrew/bin/node", mcp_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

def send(msg):
    p.stdin.write(json.dumps(msg) + "\n")
    p.stdin.flush()
    if "id" not in msg:
        return None
    while True:
        line = p.stdout.readline()
        if not line:
            return None
        try:
            obj = json.loads(line)
            if obj.get("id") == msg["id"]:
                return obj
        except Exception:
            continue

send({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "fig2-gen", "version": "1.0"},
    },
})
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
send({"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "start_session", "arguments": {}}})
res = send({
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "create_new_diagram", "arguments": {"xml": clean_xml, "page_name": "Figure 3 Octopus Mount", "screenshot": False}},
})
print("create_new_diagram:", res["result"]["content"][0]["text"][:200])

res2 = send({
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {"name": "export_diagram", "arguments": {"path": str(drawio_out), "format": "drawio"}},
})
print("export_diagram:", res2["result"]["content"][0]["text"][:200])
p.terminate()

raw_pdf = Path("/tmp/fig2_raw.pdf")
subprocess.run(
    [
        "/Applications/draw.io.app/Contents/MacOS/draw.io",
        "--export",
        "--format",
        "pdf",
        "--crop",
        "--output",
        str(raw_pdf),
        str(drawio_out),
    ],
    check=True,
)
subprocess.run(["pdfcrop", str(raw_pdf), str(pdf_out)], check=True)
subprocess.run(
    [
        "pdftoppm",
        "-png",
        "-r",
        "200",
        str(pdf_out),
        "/tmp/octopus-fig-preview/fig2_tentacle_v2",
    ],
    check=True,
)
print("Exported fig2_cursor_mount.pdf and preview PNG successfully!")
