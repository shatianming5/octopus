#!/usr/bin/env python3
"""
Builds the Method Framework Figure (Figure 2 in Section 3.1: fig_tree_execution.drawio & fig_tree_execution.pdf)
in Draw.io with high font density, crisp vector curved edges, and zero collisions:
1. Left Panel: Unlimited-Depth Octopus Cursor Execution DAG G = (V, E_tree, E_mount) (d=0 -> d=5).
2. Right Panel:
   - Card ①: Parallel fork & Galton-Watson Deep Tree (99.2% fork, 51.4% d>=3)
   - Card ②: Tactical rewind Crossover & Red-Green Commit Gate (10x branch rewind, 85% SWE pass)
   - Card ③: Explicit 32-Layer Hybrid Architecture Stack (24 GDN Recurrent + 8 Full-Attention)
"""
import base64
import io
import json
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
        "octo_21_title_boss",
        "octo_fork",
        "octo_19_wait_sleep",
        "octo_radix",
        "octo_15_search_king",
        "octo_17_rewind_cut",
        "octo_10_layer2_mount",
        "octo_20_commit_gate",
        "octo_16_tea_smug",
        "octo_credit",
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
<mxStyle name="dagPanel" value="rounded=1;arcSize=3;fillColor=#F4FBF9;strokeColor=#0D9488;strokeWidth=2;fontFamily=Comic Sans MS;"/>
<mxStyle name="bandBox" value="rounded=1;arcSize=4;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1.2;dashed=1;fontFamily=Comic Sans MS;"/>
<mxStyle name="bandHdr" value="rounded=1;arcSize=40;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;strokeWidth=1.1;"/>
<mxStyle name="cTeal" value="ellipse;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.2;fillColor=#CCFBF1;strokeColor=#0F766E;strokeWidth=1.8;fontColor=#064E3B;"/>
<mxStyle name="cBlue" value="ellipse;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.2;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=1.8;fontColor=#1E3A8A;"/>
<mxStyle name="cAmber" value="rounded=1;arcSize=45;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.6;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.8;fontColor=#92400E;"/>
<mxStyle name="cDead" value="ellipse;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.6;fillColor=#FEE2E2;strokeColor=#DC2626;strokeWidth=1.8;dashed=1;fontColor=#991B1B;"/>
<mxStyle name="cDone" value="rounded=1;arcSize=45;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.8;fillColor=#0F766E;strokeColor=#064E3B;strokeWidth=1.8;fontColor=#FFFFFF;"/>
<mxStyle name="badgeTeal" value="rounded=1;arcSize=38;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.5;fillColor=#CCFBF1;strokeColor=#0D9488;fontColor=#065F46;"/>
<mxStyle name="badgeAmber" value="rounded=1;arcSize=38;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.5;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;"/>
<mxStyle name="badgeRose" value="rounded=1;arcSize=38;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.5;fillColor=#FEE2E2;strokeColor=#DC2626;fontColor=#991B1B;"/>
<mxStyle name="badgeBlue" value="rounded=1;arcSize=38;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.5;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E40AF;"/>
<mxStyle name="badgePurple" value="rounded=1;arcSize=38;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;fillColor=#EDE9FE;strokeColor=#7C3AED;fontColor=#4C1D95;"/>
<mxStyle name="sideCard" value="rounded=1;arcSize=5;fillColor=#FFFFFF;strokeWidth=1.6;fontFamily=Comic Sans MS;"/>

<mxCell id="left_panel" value="" style="dagPanel;" x="8" y="8" w="526" h="416"/>
<mxCell id="lp_title" value="&lt;font color=&quot;#0F766E&quot;&gt;Unlimited-Depth Cursor Execution DAG  G = (V, E&lt;sub&gt;tree&lt;/sub&gt;, E&lt;sub&gt;mount&lt;/sub&gt;)&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=10.4;" x="42" y="11" w="342" h="22"/>
<mxCell id="lp_tag" value="D&lt;sub&gt;max&lt;/sub&gt;=∞ · W&lt;sub&gt;max&lt;/sub&gt;=10 · 51.4% d≥3" style="badgeTeal;fontSize=7.8;" x="382" y="12" w="144" h="21"/>
<mxCell id="lp_div" value="" style="line;strokeColor=#99F6E4;strokeWidth=1.2;" x="16" y="36" w="510" h="4"/>

<mxCell id="b_d0" value="" style="bandBox;fillColor=#F0FDFA;" x="16" y="42" w="104" h="298"/>
<mxCell id="bh_d0" value="d = 0 (Root)" style="bandHdr;fillColor=#CCFBF1;strokeColor=#0D9488;fontColor=#065F46;" x="21" y="46" w="94" h="20"/>

<mxCell id="b_d1" value="" style="bandBox;fillColor=#F8FAFC;" x="125" y="42" w="118" h="298"/>
<mxCell id="bh_d1" value="d = 1 (Forks)" style="bandHdr;fillColor=#DBEAFE;strokeColor=#3B82F6;fontColor=#1E3A8A;" x="130" y="46" w="108" h="20"/>

<mxCell id="b_d23" value="" style="bandBox;fillColor=#FFFBEB;" x="248" y="42" w="166" h="298"/>
<mxCell id="bh_d23" value="d = 2 → d = 5 (Deep Tree)" style="bandHdr;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="252" y="46" w="136" h="20"/>

<mxCell id="b_join" value="" style="bandBox;fillColor=#F0FDF4;" x="419" y="42" w="107" h="298"/>
<mxCell id="bh_join" value="Join &amp;amp; Close" style="bandHdr;fillColor=#CCFBF1;strokeColor=#0D9488;fontColor=#065F46;" x="424" y="46" w="97" h="20"/>

<!-- DAG NODES & LABELS (High Font Density, Zero Collisions) -->
<mxCell id="c0" value="c₀" style="cTeal;" x="18" y="195" w="34" h="30"/>
<mxCell id="cfork" value="c&lt;sub&gt;fork&lt;/sub&gt;" style="cTeal;fillColor=#99F6E4;" x="68" y="195" w="44" h="30"/>
<mxCell id="lbl_fork0" value="① fork(A,B,C)&lt;br&gt;&lt;font style=&quot;font-size:7.6px;&quot;&gt;O(1) ref+3, 32K/br&lt;/font&gt;" style="badgeTeal;" x="18" y="118" w="86" h="32"/>
<mxCell id="lbl_wait" value="wait(A,B,C)&lt;br&gt;&lt;font style=&quot;font-size:7.5px;&quot;&gt;Root Paused · 0 Bloat&lt;/font&gt;" style="badgePurple;" x="18" y="256" w="86" h="32"/>

<mxCell id="ca1" value="c&lt;sub&gt;A1&lt;/sub&gt;" style="cBlue;" x="132" y="94" w="40" h="28"/>
<mxCell id="caf" value="sub&lt;br&gt;fork" style="cAmber;fontSize=8.8;" x="186" y="92" w="48" h="32"/>
<mxCell id="ca21" value="c&lt;sub&gt;A2.1&lt;/sub&gt; (Span A)&lt;br&gt;&lt;font style=&quot;font-size:7.8px;&quot;&gt;Pages K&lt;sub&gt;A&lt;/sub&gt;, V&lt;sub&gt;A&lt;/sub&gt;&lt;/font&gt;" style="cAmber;" x="253" y="69" w="94" h="34"/>
<mxCell id="ca3" value="c&lt;sub&gt;A3..5&lt;/sub&gt; (d≥3)" style="rounded=1;arcSize=45;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.8;fillColor=#99F6E4;strokeColor=#0F766E;strokeWidth=1.8;fontColor=#064E3B;" x="355" y="72" w="57" h="28"/>

<mxCell id="ca22" value="c&lt;sub&gt;A2.2&lt;/sub&gt;" style="cBlue;" x="272" y="133" w="44" h="28"/>
<mxCell id="ca_dead" value="✗" style="cDead;" x="325" y="106" w="25" h="25"/>
<mxCell id="ca22_p" value="c'&lt;sub&gt;A2.2&lt;/sub&gt;" style="cTeal;" x="362" y="133" w="48" h="28"/>
<mxCell id="lbl_subrw" value="Br. Rewind 10×↑" style="badgeRose;fontSize=8.0;" x="304" y="164" w="84" h="18"/>

<mxCell id="cb1" value="c&lt;sub&gt;B1&lt;/sub&gt;" style="cBlue;" x="132" y="195" w="40" h="30"/>
<mxCell id="cb_dead" value="✗ 8K" style="rounded=1;arcSize=45;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.4;fillColor=#FEE2E2;strokeColor=#DC2626;strokeWidth=1.6;dashed=1;fontColor=#991B1B;" x="188" y="168" w="42" h="24"/>
<mxCell id="cb2" value="c'&lt;sub&gt;B2&lt;/sub&gt; (Reader)&lt;br&gt;&lt;font style=&quot;font-size:7.9px;&quot;&gt;Clean h&lt;sub&gt;B1&lt;/sub&gt; + Note&lt;/font&gt;" style="rounded=1;arcSize=42;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.4;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=1.8;fontColor=#1E3A8A;" x="260" y="192" w="106" h="35"/>
<mxCell id="lbl_rw1" value="② rewind(k=1)" style="badgeRose;fontSize=8.1;" x="130" y="136" w="78" h="19"/>
<mxCell id="lbl_mount" value="③ 0-Copy read(A2.1)&lt;br&gt;&lt;font style=&quot;font-size:7.6px;&quot;&gt;Attn [Mount] · Recurrent [Skip]&lt;/font&gt;" style="badgeAmber;" x="176" y="234" w="144" h="31"/>

<mxCell id="cc1" value="c&lt;sub&gt;C1&lt;/sub&gt;" style="cTeal;" x="132" y="279" w="40" h="28"/>
<mxCell id="cc2" value="c&lt;sub&gt;C2&lt;/sub&gt; (open Doc)&lt;br&gt;&lt;font style=&quot;font-size:7.9px;&quot;&gt;Gold Evid. ✓&lt;/font&gt;" style="rounded=1;arcSize=42;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.4;fillColor=#CCFBF1;strokeColor=#0F766E;strokeWidth=1.8;fontColor=#064E3B;" x="260" y="275" w="106" h="35"/>
<mxCell id="lbl_commit" value="④ done(evid) · 93.1% Close" style="badgeTeal;fontSize=8.0;" x="158" y="315" w="132" h="19"/>

<mxCell id="cjoin" value="c&lt;sub&gt;join&lt;/sub&gt; (d=0)" style="rounded=1;arcSize=45;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.0;fillColor=#99F6E4;strokeColor=#0F766E;strokeWidth=1.8;fontColor=#064E3B;" x="427" y="132" w="64" h="30"/>
<mxCell id="cremind" value="Close Remind&lt;br&gt;&lt;font style=&quot;font-size:7.8px;&quot;&gt;m=0 · 97.4%&lt;/font&gt;" style="cAmber;" x="425" y="198" w="94" h="33"/>
<mxCell id="cdone" value="done ★&lt;br&gt;&lt;font style=&quot;font-size:8.0px;&quot;&gt;R = 2y − 1&lt;/font&gt;" style="cDone;" x="429" y="266" w="86" h="36"/>

<!-- CLEAN CURVED EDGES ACROSS THE DAG (Zero Collisions) -->
{ tentacle_edge("te_c0_cf", "c0", "cfork", (1.0, 0.5), (0.0, 0.5), (52, 210), (60, 210), (68, 210), outer_col="#0F766E", width=4.6) }
{ tentacle_edge("te_cf_ca1", "cfork", "ca1", (0.92, 0.15), (0.0, 0.5), (108, 199), (118, 134), (132, 108), outer_col="#EA580C", width=4.5) }
{ tentacle_edge("te_ca1_caf", "ca1", "caf", (1.0, 0.5), (0.0, 0.5), (172, 108), (179, 108), (186, 108), outer_col="#D97706", width=4.2) }
{ tentacle_edge("te_caf_21", "caf", "ca21", (1.0, 0.3), (0.0, 0.5), (234, 101), (244, 86), (253, 86), outer_col="#D97706", width=4.2) }
{ tentacle_edge("te_21_a3", "ca21", "ca3", (1.0, 0.5), (0.0, 0.5), (347, 86), (351, 86), (355, 86), outer_col="#0F766E", width=4.0) }

{ tentacle_edge("te_caf_22", "caf", "ca22", (1.0, 0.7), (0.0, 0.5), (234, 115), (254, 145), (272, 147), outer_col="#2563EB", width=4.2) }
{ tentacle_edge("te_22_dead", "ca22", "ca_dead", (0.85, 0.15), (0.0, 0.65), (310, 137), (318, 128), (325, 122), outer_col="#DC2626", width=3.0, dashed=True) }
{ tentacle_edge("te_22_rw", "ca_dead", "ca22", (0.2, 0.0), (0.4, 0.0), (330, 106), (308, 96), (290, 133), outer_col="#E11D48", width=3.4) }
{ tentacle_edge("te_22_ok", "ca22", "ca22_p", (1.0, 0.65), (0.0, 0.65), (316, 151), (339, 153), (362, 151), outer_col="#0D9488", width=4.0) }

{ tentacle_edge("te_cf_cb1", "cfork", "cb1", (1.0, 0.5), (0.0, 0.5), (112, 210), (122, 210), (132, 210), outer_col="#2563EB", width=4.5) }
{ tentacle_edge("te_cb1_dead", "cb1", "cb_dead", (0.85, 0.15), (0.0, 0.65), (166, 199), (178, 189), (188, 184), outer_col="#DC2626", width=3.0, dashed=True) }
{ tentacle_edge("te_cb_rw", "cb_dead", "cb1", (0.2, 0.0), (0.45, 0.0), (196, 168), (170, 155), (150, 195), outer_col="#E11D48", width=3.4) }
{ tentacle_edge("te_cb1_cb2", "cb1", "cb2", (1.0, 0.65), (0.0, 0.55), (172, 214), (216, 215), (260, 211), outer_col="#2563EB", width=4.4) }

{ tentacle_edge("te_mount", "ca21", "cb2", (0.09, 1.0), (0.14, 0.0), (261, 103), (240, 150), (275, 192), outer_col="#D97706", width=4.2, dashed=True) }

{ tentacle_edge("te_cf_cc1", "cfork", "cc1", (0.92, 0.85), (0.0, 0.5), (108, 220), (118, 270), (132, 293), outer_col="#0D9488", width=4.5) }
{ tentacle_edge("te_cc1_cc2", "cc1", "cc2", (1.0, 0.5), (0.0, 0.5), (172, 293), (216, 293), (260, 293), outer_col="#0D9488", width=4.4) }

{ tentacle_edge("te_a3_j", "ca3", "cjoin", (1.0, 0.5), (0.5, 0.0), (412, 86), (436, 98), (459, 132), outer_col="#0F766E", width=3.5, dashed=True) }
{ tentacle_edge("te_22p_j", "ca22_p", "cjoin", (1.0, 0.5), (0.0, 0.5), (410, 147), (419, 147), (427, 147), outer_col="#0F766E", width=3.5, dashed=True) }
{ tentacle_edge("te_cb2_j", "cb2", "cjoin", (1.0, 0.4), (0.0, 0.75), (366, 206), (404, 192), (427, 155), outer_col="#2563EB", width=3.6, dashed=True) }
{ tentacle_edge("te_cc2_j", "cc2", "cjoin", (1.0, 0.3), (0.2, 1.0), (366, 286), (414, 240), (440, 162), outer_col="#0D9488", width=3.6, dashed=True) }
{ tentacle_edge("te_j_rem", "cjoin", "cremind", (0.5, 1.0), (0.4, 0.0), (459, 162), (461, 180), (463, 198), outer_col="#D97706", width=4.0) }
{ tentacle_edge("te_rem_dn", "cremind", "cdone", (0.5, 1.0), (0.5, 0.0), (472, 231), (472, 248), (472, 266), outer_col="#0F766E", width=4.4) }

<mxCell id="leg_box" value="" style="rounded=1;arcSize=6;fillColor=#FFFFFF;strokeColor=#99F6E4;strokeWidth=1.3;" x="16" y="345" w="510" h="71"/>
<mxCell id="leg_t1" value="&lt;b&gt;&lt;font color=&quot;#0F766E&quot;&gt;DAG Legend:&lt;/font&gt;&lt;/b&gt;  &lt;font color=&quot;#0F766E&quot;&gt;&lt;b&gt;●──▸&lt;/b&gt; Tree Edge E&lt;sub&gt;tree&lt;/sub&gt;&lt;/font&gt;   &lt;font color=&quot;#D97706&quot;&gt;&lt;b&gt;● ⇢ ▸&lt;/b&gt; 0-Copy KV Mount E&lt;sub&gt;mount&lt;/sub&gt;&lt;/font&gt;   &lt;font color=&quot;#DC2626&quot;&gt;&lt;b&gt;✗ ↶&lt;/b&gt; Rewind Rollback (Pruned m=1)&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.8;fontColor=#1E293B;" x="22" y="348" w="498" h="18"/>
<mxCell id="bar_tree" value="Episode Tree Explores 143.5K Effective Tok (4.4× of 32K Cap Across ≤10 Br.)" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.8;fillColor=#DBEAFE;strokeColor=#3B82F6;fontColor=#1E3A8A;" x="22" y="370" w="334" h="19"/>
<mxCell id="bar_crit" value="Critical Path: 18.4K (1.86× Shorter!)" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.8;fillColor=#CCFBF1;strokeColor=#0D9488;fontColor=#065F46;" x="360" y="370" w="160" h="19"/>
<mxCell id="leg_t2" value="• Single-Ownership Mask m&lt;sub&gt;l,t&lt;/sub&gt; ∈ {{0,1}}: Every generated token counted 1× · On-Policy Mount Edge (|Δ log π&lt;sub&gt;θ&lt;/sub&gt;| = 0.0038, KL = 4.45e−4)" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.2;fontColor=#334155;" x="22" y="393" w="498" h="18"/>

<!-- ================= RIGHT PANEL: SYSTEM CALLS + 24-GDN / 8-ATTN LAYERED ARCHITECTURE DIAGRAM ================= -->
<mxCell id="sc1" value="" style="sideCard;strokeColor=#0D9488;fillColor=#F0FDFA;" x="542" y="8" w="252" h="92"/>
<mxCell id="sc1_hdr" value="&lt;font color=&quot;#0F766E&quot;&gt;① Parallel fork &amp;amp; Deep Tree&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=9.2;" x="572" y="10" w="156" h="18"/>
<mxCell id="sc1_pill" value="99.2% Fork" style="badgeTeal;fontSize=7.5;" x="728" y="11" w="60" h="17"/>
<mxCell id="sc1_b" value="• &lt;b&gt;O(1) Radix Fork:&lt;/b&gt; Shares KV prefix (ref+K),&lt;br&gt;  clones GDN state h&lt;sub&gt;t&lt;/sub&gt;, fresh 32K cap/branch.&lt;br&gt;• &lt;b&gt;Sub-Critical Galton–Watson (μ=0.65 &amp;lt; 1):&lt;/b&gt;&lt;br&gt;  5.5 br/ep; &lt;b&gt;51.4%&lt;/b&gt; grow depth &lt;b&gt;d ≥ 3&lt;/b&gt; naturally." style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.8;fontColor=#1E293B;" x="548" y="28" w="242" h="48"/>
<mxCell id="sc1_m1" value="d=1: 48.6%" style="badgeBlue;fontSize=7.4;" x="548" y="78" w="75" h="17"/>
<mxCell id="sc1_m2" value="d=2: 33.8%" style="badgeAmber;fontSize=7.4;" x="628" y="78" w="75" h="17"/>
<mxCell id="sc1_m3" value="d≥3: 51.4% ★" style="badgeTeal;fontSize=7.4;" x="708" y="78" w="80" h="17"/>

<mxCell id="sc2" value="" style="sideCard;strokeColor=#DC2626;fillColor=#FFF1F2;" x="542" y="106" w="252" h="92"/>
<mxCell id="sc2_hdr" value="&lt;font color=&quot;#991B1B&quot;&gt;② State Rollback &amp;amp; Close Contract&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.2;" x="572" y="108" w="164" h="18"/>
<mxCell id="sc2_pill" value="97.4% done" style="badgeRose;fontSize=7.2;" x="734" y="109" w="54" h="17"/>
<mxCell id="sc2_b" value="• &lt;b&gt;Exact State Rollback:&lt;/b&gt; Resets cursor c&lt;sub&gt;k&lt;/sub&gt;, restores&lt;br&gt;  GDN h&lt;sub&gt;k&lt;/sub&gt; &amp;amp; 32K cap (Root ↓2.9×, Br. ↑10×).&lt;br&gt;• &lt;b&gt;Close Reminder (m=0) &amp;amp; 2K Reserve:&lt;/b&gt; Converts&lt;br&gt;  97.4% roots &amp;amp; 93.1% br. into valid done (0.2% drift)." style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontSize=7.7;fontColor=#1E293B;" x="548" y="126" w="242" h="48"/>
<mxCell id="sc2_m1" value="Root Rw: 29%→10%" style="badgeTeal;fontSize=7.4;" x="548" y="176" w="115" h="17"/>
<mxCell id="sc2_m2" value="Br. Rw: 3.2%→45.1% (10×↑)" style="badgeRose;fontSize=7.4;" x="667" y="176" w="121" h="17"/>

<!-- CARD 3: LAYER-SELECTIVE HYBRID KV MOUNT ARCHITECTURE DIAGRAM -->
<mxCell id="sc3" value="" style="sideCard;strokeColor=#D97706;fillColor=#FFFBEB;" x="542" y="204" w="252" h="220"/>
<mxCell id="sc3_hdr" value="&lt;font color=&quot;#92400E&quot;&gt;③ Layer-Selective Hybrid KV Mount&lt;/font&gt;" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.3;" x="572" y="206" w="170" h="18"/>
<mxCell id="sc3_pill" value="21×↓ Hijack" style="badgeAmber;fontSize=7.1;" x="742" y="207" w="48" h="17"/>

<!-- Input Buses at top of Card 3 -->
<mxCell id="bus_a" value="Sibling A KV&lt;br&gt;&lt;font style=&quot;font-size:7.3px;&quot;&gt;Pages K&lt;sub&gt;A&lt;/sub&gt;, V&lt;sub&gt;A&lt;/sub&gt;&lt;/font&gt;" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.0;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="548" y="227" w="114" h="27"/>
<mxCell id="bus_b" value="Reader B State&lt;br&gt;&lt;font style=&quot;font-size:7.3px;&quot;&gt;Private h&lt;sub&gt;B&lt;/sub&gt;, Δ&lt;sub&gt;B&lt;/sub&gt;&lt;/font&gt;" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=8.0;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E40AF;" x="674" y="227" w="114" h="27"/>

<!-- Repeating Hybrid Block Container (x8 Blocks = 24 GDN + 8 Full-Attn) -->
<mxCell id="blk_frame" value="" style="rounded=1;arcSize=6;fillColor=#FFFFFF;strokeColor=#64748B;strokeWidth=1.3;" x="548" y="259" w="240" h="138"/>
<mxCell id="blk_hdr" value="× 8 Hybrid Blocks in Qwen3.5 (32 Layers)" style="text;align=left;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.8;fontColor=#334155;" x="562" y="261" w="188" h="15"/>

<!-- 3 Stacked Gated DeltaNet Recurrent Layers (24 Layers Total) -->
<mxCell id="L_gdn1" value="🔒 GDN Layer 4k+1: h&lt;sub&gt;B&lt;/sub&gt; ← f(h&lt;sub&gt;B&lt;/sub&gt;, Δ&lt;sub&gt;B&lt;/sub&gt;) &lt;font color=&quot;#DC2626&quot;&gt;[✗ Skip A]&lt;/font&gt;" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.5;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E40AF;" x="562" y="278" w="196" h="20"/>
<mxCell id="L_gdn2" value="🔒 GDN Layer 4k+2: h&lt;sub&gt;B&lt;/sub&gt; ← f(h&lt;sub&gt;B&lt;/sub&gt;, Δ&lt;sub&gt;B&lt;/sub&gt;) &lt;font color=&quot;#DC2626&quot;&gt;[✗ Skip A]&lt;/font&gt;" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.5;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E40AF;" x="562" y="301" w="196" h="20"/>
<mxCell id="L_gdn3" value="🔒 GDN Layer 4k+3: h&lt;sub&gt;B&lt;/sub&gt; ← f(h&lt;sub&gt;B&lt;/sub&gt;, Δ&lt;sub&gt;B&lt;/sub&gt;) &lt;font color=&quot;#DC2626&quot;&gt;[✗ Skip A]&lt;/font&gt;" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.5;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E40AF;" x="562" y="324" w="196" h="20"/>
<mxCell id="tag_24gdn" value="24&lt;br&gt;GDN" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=6.8;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#1E3A8A;" x="760" y="278" w="24" h="66"/>

<!-- 1 Full-Attention Layer Slice (8 Layers Total) -->
<mxCell id="L_att4" value="⚡ Full-Attn Layer 4k+4 (ℓ ∈ L&lt;sub&gt;mount&lt;/sub&gt;): 0-Copy&lt;br&gt;&lt;font style=&quot;font-size:7.2px;&quot; color=&quot;#047857&quot;&gt;[K&lt;sub&gt;trunk&lt;/sub&gt; ∥ K&lt;sub&gt;A&lt;/sub&gt; (0-Copy) ∥ K&lt;sub&gt;B&lt;/sub&gt;] @ Monotonic s(c'&lt;sub&gt;B&lt;/sub&gt;)&lt;/font&gt;" style="rounded=1;arcSize=18;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=7.5;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="562" y="348" w="196" h="44"/>
<mxCell id="tag_8att" value="8&lt;br&gt;Attn" style="rounded=1;arcSize=20;align=center;verticalAlign=middle;fontFamily=Comic Sans MS;fontStyle=1;fontSize=6.8;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#92400E;" x="760" y="348" w="24" h="44"/>

{ tentacle_edge("te_bus_a", "bus_a", "L_att4", (0.12, 1.0), (0.0, 0.5), (562, 254), (552, 315), (562, 370), outer_col="#D97706", width=3.8) }

<mxCell id="eb_bus" value="" style="strokeColor=#2563EB;strokeWidth=2.0;exitX=0.75;exitY=1;entryX=0.88;entryY=0;" source="bus_b" target="L_gdn1"/>

<mxCell id="sc3_m1" value="🔒 24 GDN: Private h&lt;sub&gt;B&lt;/sub&gt;" style="badgeBlue;fontSize=7.0;" x="548" y="400" w="98" h="18"/>
<mxCell id="sc3_m2" value="⚡ 8 Attn: 97.3% Rank · 2.7% Hijack" style="badgeTeal;fontSize=6.8;" x="650" y="400" w="138" h="18"/>

{img_cell("ic_boss", "octo_21_title_boss", 14, 10, 26, 24)}
{img_cell("ic_fork0", "octo_fork", 48, 154, 26, 26)}
{img_cell("ic_wait", "octo_19_wait_sleep", 48, 290, 28, 28)}
{img_cell("ic_subf", "octo_radix", 196, 65, 26, 25)}
{img_cell("ic_king", "octo_15_search_king", 389, 44, 25, 24)}
{img_cell("ic_rw_b", "octo_17_rewind_cut", 210, 132, 24, 24)}
{img_cell("ic_mnt", "octo_10_layer2_mount", 322, 238, 28, 26)}
{img_cell("ic_gate", "octo_20_commit_gate", 368, 284, 28, 26)}
{img_cell("ic_tea", "octo_16_tea_smug", 484, 166, 26, 26)}
{img_cell("ic_cred", "octo_credit", 484, 304, 28, 26)}

{img_cell("ic_sc1", "octo_fork", 548, 9, 22, 20)}
{img_cell("ic_sc2", "octo_17_rewind_cut", 548, 107, 22, 20)}
{img_cell("ic_sc3", "octo_k2", 548, 206, 22, 20)}
"""

clean_xml = "\n".join(
    line for line in CELLS.strip().splitlines() if not line.strip().startswith("<!--")
)

mcp_script = "/opt/homebrew/lib/node_modules/@next-ai-drawio/mcp-server/dist/index.js"
drawio_out = FIG_DIR / "fig_tree_execution.drawio"
pdf_out = FIG_DIR / "fig_tree_execution.pdf"

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
            raise RuntimeError("EOF from MCP server")
        if line.strip().startswith("{"):
            msg = json.loads(line)
            if msg.get("id") == req["id"]:
                return msg

rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "antigravity", "version": "1.0"}}})
rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "start_session", "arguments": {}}})
rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "create_new_diagram", "arguments": {"xml": clean_xml}}})
rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "export_diagram", "arguments": {"path": str(drawio_out)}}})
p.terminate()

subprocess.run(
    [
        "/Applications/draw.io.app/Contents/MacOS/draw.io",
        "--export",
        "--format",
        "pdf",
        "--crop",
        "--output",
        str(pdf_out),
        str(drawio_out),
    ],
    check=True,
)
subprocess.run(
    ["pdftoppm", "-png", "-r", "240", str(pdf_out), str(ART_DIR / "fig_tree_v38")],
    check=True,
)
print("Exported:", pdf_out)
