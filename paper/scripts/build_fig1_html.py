#!/usr/bin/env python3
"""
Generates Figure 1 (Page-1 Teaser Overview) with:
1. Clean, non-boxy 3-column visual teaser layout:
   (a) Prior Stateless / All-Layer Paradigms
   (b) Octopus Co-Designed Branching Runtime, K2 Mount & DAG-GRPO
   (c) Headline Accuracy & H200 Systems Gains
2. 100% rounded 'Comic Sans MS' typography for all text, numbers, and symbols.
3. ZERO italics anywhere (all math/variables rendered in crisp upright Comic Sans MS).
"""
import subprocess
from pathlib import Path

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Octopus Figure 1 Teaser</title>
<style>
  @page {
    size: 960px 340px;
    margin: 0;
  }
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: "Comic Sans MS", "Chalkboard SE", "Comic Neue", sans-serif !important;
    font-style: normal !important;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }
  body {
    width: 960px;
    height: 340px;
    background: #FFFFFF;
    color: #1E293B;
    padding: 4px 6px 2px 6px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
  }
  .teaser-grid {
    display: grid;
    grid-template-columns: 286px 418px 230px;
    gap: 10px;
    height: 308px;
  }
  .col-card {
    border-radius: 12px;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
  }
  .col-a {
    background: #FAFAFA;
    border: 1.8px solid #CBD5E1;
  }
  .col-b {
    background: #F4FBF9;
    border: 2px solid #0D9488;
  }
  .col-c {
    background: #F8FAFC;
    border: 1.8px solid #94A3B8;
  }
  .col-header {
    font-size: 13px;
    font-weight: 700;
    padding-bottom: 4px;
    border-bottom: 1.5px dashed #CBD5E1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    white-space: nowrap;
  }
  .tag {
    font-size: 10px;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 999px;
  }
  .node {
    position: absolute;
    transform: translate(-50%, -50%);
    border-radius: 999px;
    font-size: 10.5px;
    font-weight: 700;
    padding: 2px 7px;
    line-height: 1.1;
    white-space: nowrap;
    text-align: center;
    border: 1.6px solid #1E293B;
    background: #FFFFFF;
    color: #1E293B;
  }
  .lbl {
    position: absolute;
    transform: translate(-50%, -50%);
    white-space: nowrap;
    line-height: 1.15;
    text-align: center;
    font-size: 10px;
    font-weight: 700;
    color: #1E293B;
  }
  .caption-row {
    display: grid;
    grid-template-columns: 286px 418px 230px;
    gap: 10px;
  }
  .col-cap {
    text-align: center;
    font-size: 11.5px;
    font-weight: 700;
    color: #0F172A;
    white-space: nowrap;
  }
  sub, sup {
    font-size: 0.74em;
    line-height: 0;
  }
</style>
</head>
<body>

<div class="teaser-grid">
  <!-- ===================================================================== -->
  <!-- COLUMN (a): PRIOR PARADIGMS (CLEAN VISUAL FLOW, NO BOXY CLUTTER)       -->
  <!-- ===================================================================== -->
  <div class="col-card col-a">
    <div class="col-header" style="color:#991B1B; border-color:#FECACA;">
      <span>Prior Agentic Paradigms</span>
      <span class="tag" style="background:#FEE2E2; color:#991B1B;">Stateless Pipe</span>
    </div>

    <!-- Top visual: Linear Pipe + Mid-Edit Cache Break -->
    <div style="padding: 2px 0;">
      <div style="font-size:11px; font-weight:700; color:#7F1D1D; margin-bottom:5px; display:flex; justify-content:space-between;">
        <span>1. Linear Append / Context Edit</span>
        <span style="color:#B91C1C;">O(L²) &amp; 32K Wall</span>
      </div>
      <div style="display:flex; align-items:center; gap:3px;">
        <div style="background:#E2E8F0; border:1.5px solid #475569; border-radius:7px; padding:5px 6px; font-size:10px; font-weight:700;">Prompt</div>
        <span style="font-weight:700; color:#64748B;">→</span>
        <div style="background:#FEE2E2; border:1.5px solid #DC2626; border-radius:7px; padding:5px 6px; font-size:10px; font-weight:700; color:#991B1B;">Detour</div>
        <span style="font-weight:700; color:#64748B;">→</span>
        <div style="background:#FEF3C7; border:1.5px dashed #D97706; border-radius:7px; padding:5px 6px; font-size:10px; font-weight:700; color:#92400E;">Edit ⚡</div>
        <span style="font-weight:700; color:#64748B;">→</span>
        <div style="background:#FCA5A5; border:1.5px solid #B91C1C; border-radius:7px; padding:5px 6px; font-size:10px; font-weight:700; color:#7F1D1D; flex:1; text-align:center;">Full!</div>
      </div>
      <div style="font-size:9.8px; color:#475569; margin-top:4px; padding-left:2px;">
        • Mid-edit breaks KV cache (41.8% re-prefill) &amp; h<sub>t</sub>
      </div>
    </div>

    <div style="border-top:1.2px dashed #CBD5E1;"></div>

    <!-- Middle visual: Multi-Agent O(N^2) KV Duplication -->
    <div style="padding: 2px 0;">
      <div style="font-size:11px; font-weight:700; color:#92400E; margin-bottom:4px; display:flex; justify-content:space-between;">
        <span>2. Multi-Agent Chat Sessions</span>
        <span style="color:#B45309;">O(L) Wire · O(N²) KV</span>
      </div>
      <div style="display:flex; align-items:center; justify-content:space-between; gap:6px;">
        <div style="background:#FFFBEB; border:1.5px solid #D97706; border-radius:8px; padding:5px 7px; font-size:9.8px; font-weight:700; color:#92400E; text-align:center;">
          Text Chat<br><span style="font-size:9px; color:#B45309;">Re-Tokenize</span>
        </div>
        <span style="font-size:12px; font-weight:700; color:#D97706;">⇉</span>
        <div style="display:flex; flex-direction:column; gap:3px; flex:1; font-size:9.2px; font-weight:700;">
          <div style="background:#F1F5F9; border:1.2px solid #94A3B8; border-radius:5px; padding:2px 5px; display:flex; justify-content:space-between;">
            <span>[Trunk] [Copy A]</span><span style="color:#DC2626;">Dup 1</span>
          </div>
          <div style="background:#F1F5F9; border:1.2px solid #94A3B8; border-radius:5px; padding:2px 5px; display:flex; justify-content:space-between;">
            <span>[Trunk] [Copy A]</span><span style="color:#DC2626;">Dup N</span>
          </div>
        </div>
      </div>
    </div>

    <div style="border-top:1.2px dashed #CBD5E1;"></div>

    <!-- Bottom visual: All-Layer Hijacking -->
    <div style="padding: 2px 0;">
      <div style="font-size:11px; font-weight:700; color:#9F1239; margin-bottom:2px; display:flex; justify-content:space-between;">
        <span>3. All-Layer KV / Text Injection</span>
        <span style="color:#E11D48;">57.9% Hijack</span>
      </div>
      <div style="position:relative; width:264px; height:64px;">
        <svg width="264" height="64" viewBox="0 0 264 64" style="position:absolute; top:0; left:0;">
          <defs>
            <marker id="a-dk" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#334155"/>
            </marker>
            <marker id="a-rd" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#E11D48"/>
            </marker>
          </defs>
          <line x1="12" y1="15" x2="205" y2="15" stroke="#334155" stroke-width="2" marker-end="url(#a-dk)"/>
          <line x1="12" y1="50" x2="86" y2="50" stroke="#2563EB" stroke-width="2"/>
          <line x1="86" y1="50" x2="205" y2="50" stroke="#CBD5E1" stroke-width="1.6" stroke-dasharray="4,3"/>
          <line x1="86" y1="25" x2="86" y2="39" stroke="#E11D48" stroke-width="2" marker-end="url(#a-rd)"/>
          <path d="M 98 50 C 125 50, 135 17, 162 15" stroke="#E11D48" stroke-width="2.2" fill="none" marker-end="url(#a-rd)"/>
        </svg>
        <div class="node" style="left:22px; top:15px; background:#F1F5F9;">A<sub>1</sub></div>
        <div class="node" style="left:86px; top:15px; background:#FEF3C7; border-color:#D97706;">A<sub>2</sub></div>
        <div class="node" style="left:176px; top:15px; background:#F1F5F9;">A<sub>3</sub></div>
        <div class="lbl" style="left:236px; top:15px; font-size:9.8px;">Task A</div>

        <div class="node" style="left:22px; top:50px; background:#DBEAFE; border-color:#2563EB;">B<sub>1</sub></div>
        <div class="node" style="left:86px; top:50px; background:#FFE4E6; border-color:#E11D48; color:#9F1239;">B<sub>2</sub></div>
        <div class="lbl" style="left:195px; top:38px; font-size:9.2px; color:#BE123C; background:#FFE4E6; padding:1px 5px; border-radius:4px;">B Hijacked!</div>
      </div>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- COLUMN (b): OCTOPUS CO-DESIGNED ARCHITECTURE                          -->
  <!-- ===================================================================== -->
  <div class="col-card col-b">
    <div class="col-header" style="color:#0F766E; border-color:#99F6E4;">
      <span>Octopus: Co-Designed Runtime &amp; RL</span>
      <span class="tag" style="background:#CCFBF1; color:#0F766E;">0-Copy Cursor &amp; K2 Mount</span>
    </div>

    <!-- Top: Clean Branching Cursor Tree -->
    <div>
      <div style="font-size:11px; font-weight:700; color:#0F766E; display:flex; justify-content:space-between;">
        <span>1. Cursor Branching Tree (fork · rewind · read)</span>
        <span style="color:#047857;">O(Δ) Wire</span>
      </div>
      <div style="position:relative; width:394px; height:96px;">
        <svg width="394" height="96" viewBox="0 0 394 96" style="position:absolute; top:0; left:0;">
          <defs>
            <marker id="b-tl" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0F766E"/>
            </marker>
            <marker id="b-bl" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563EB"/>
            </marker>
            <marker id="b-gd" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#D97706"/>
            </marker>
            <marker id="b-rd" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#DC2626"/>
            </marker>
          </defs>
          <line x1="22" y1="52" x2="78" y2="52" stroke="#0F766E" stroke-width="2.2" marker-end="url(#b-tl)"/>
          <path d="M 102 46 C 122 42, 130 20, 154 20" stroke="#D97706" stroke-width="2" fill="none"/>
          <line x1="154" y1="20" x2="222" y2="20" stroke="#94A3B8" stroke-width="1.6" stroke-dasharray="4,3"/>
          <path d="M 218 12 C 202 2, 176 2, 162 12" stroke="#DC2626" stroke-width="1.6" fill="none" marker-end="url(#b-rd)"/>
          <path d="M 164 24 C 198 36, 250 36, 286 22" stroke="#D97706" stroke-width="2" fill="none"/>

          <line x1="106" y1="52" x2="348" y2="52" stroke="#2563EB" stroke-width="2.2" marker-end="url(#b-bl)"/>
          <line x1="298" y1="30" x2="298" y2="41" stroke="#D97706" stroke-width="2.4" marker-end="url(#b-gd)"/>

          <path d="M 102 58 C 122 62, 130 82, 158 82" stroke="#0D9488" stroke-width="2" fill="none"/>
          <line x1="158" y1="82" x2="264" y2="82" stroke="#0D9488" stroke-width="2"/>

          <path d="M 314 20 C 338 20, 344 44, 352 48" stroke="#D97706" stroke-width="1.6" stroke-dasharray="3,2" fill="none"/>
          <path d="M 278 82 C 320 82, 338 60, 352 56" stroke="#0D9488" stroke-width="1.6" stroke-dasharray="3,2" fill="none"/>
        </svg>

        <div class="node" style="left:22px; top:52px; background:#CCFBF1; border-color:#0F766E;">c<sub>0</sub></div>
        <div class="node" style="left:96px; top:52px; background:#CCFBF1; border-color:#0F766E;">c<sub>fork</sub></div>

        <div class="node" style="left:156px; top:20px; background:#FEF3C7; border-color:#D97706;">c<sub>A1</sub></div>
        <div class="lbl" style="left:190px; top:5px; font-size:9px; color:#DC2626; background:#F4FBF9; padding:0 2px;">rewind</div>
        <div class="node" style="left:228px; top:20px; background:#FEE2E2; border-color:#DC2626; color:#991B1B; padding:1px 5px;">×</div>
        <div class="node" style="left:298px; top:20px; background:#FEF3C7; border-color:#D97706;">c<sub>A2</sub> (Span A)</div>

        <div class="node" style="left:192px; top:52px; background:#DBEAFE; border-color:#2563EB;">c<sub>B1</sub></div>
        <div class="lbl" style="left:252px; top:38px; font-size:9px; color:#92400E; background:#FEF3C7; border:1px solid #F59E0B; border-radius:4px; padding:0 4px;">0-Copy read</div>
        <div class="node" style="left:298px; top:52px; background:#DBEAFE; border-color:#2563EB;">c<sub>B2</sub></div>

        <div class="node" style="left:168px; top:82px; background:#CCFBF1; border-color:#0F766E;">c<sub>C1</sub></div>
        <div class="node" style="left:264px; top:82px; background:#CCFBF1; border-color:#0F766E;">c<sub>C2</sub></div>

        <div class="node" style="left:370px; top:52px; background:#0F766E; color:#FFF;">done</div>
      </div>
    </div>

    <div style="border-top:1.2px dashed #99F6E4;"></div>

    <!-- Middle: Layer-Selective K2 Mount & Radix KV Page Pool -->
    <div>
      <div style="font-size:11px; font-weight:700; color:#1E40AF; margin-bottom:4px; display:flex; justify-content:space-between;">
        <span>2. Layer-Selective KV Mount (K2)</span>
        <span style="color:#0F766E;">0 New Pages · Monotonic Pos</span>
      </div>
      <div style="display:grid; grid-template-columns: 168px 220px; gap:6px;">
        <div style="background:#FFFFFF; border:1.5px solid #64748B; border-radius:8px; padding:4px 6px; display:flex; flex-direction:column; justify-content:space-between;">
          <div style="font-size:9.8px; font-weight:700; color:#334155; text-align:center;">Radix KV Page Table</div>
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:4px; text-align:center; font-size:9.2px; font-weight:700; margin-top:2px;">
            <div style="background:#E2E8F0; border-radius:5px; padding:3px 2px;">Trunk<br><span style="color:#0F766E;">ref = 3</span></div>
            <div style="background:#FEF3C7; border:1.2px solid #D97706; border-radius:5px; padding:3px 2px; color:#92400E;">Span A<br><span style="color:#B91C1C;">0-Copy!</span></div>
          </div>
        </div>
        <div style="display:flex; flex-direction:column; gap:4px; justify-content:center; font-size:9.6px; font-weight:700;">
          <div style="background:#EFF6FF; border:1.4px solid #60A5FA; border-radius:6px; padding:3px 6px; display:flex; justify-content:space-between;">
            <span style="color:#1E40AF;">24 GDN Layers: Private h<sub>B</sub></span>
            <span style="color:#DC2626;">Skip A</span>
          </div>
          <div style="background:#FEF3C7; border:1.4px solid #D97706; border-radius:6px; padding:3px 6px; display:flex; justify-content:space-between;">
            <span style="color:#92400E;">8 Attn Layers: [K<sub>0</sub> ∥ K<sub>A</sub> ∥ K<sub>B</sub>]</span>
            <span style="color:#047857;">Mount A</span>
          </div>
        </div>
      </div>
    </div>

    <div style="border-top:1.2px dashed #99F6E4;"></div>

    <!-- Bottom: Packed DAG-GRPO -->
    <div>
      <div style="font-size:11px; font-weight:700; color:#581C87; margin-bottom:3px; display:flex; justify-content:space-between;">
        <span>3. Packed Tree-and-Mount DAG-GRPO</span>
        <span style="color:#6B21A8;">Cross-Branch ∇K<sub>A</sub> ≠ 0</span>
      </div>
      <div style="display:grid; grid-template-columns: repeat(4, 1fr) 88px; gap:4px; text-align:center; font-size:9.2px; font-weight:700;">
        <div style="background:#E2E8F0; border:1.3px solid #475569; border-radius:6px; padding:3px 2px;">Trunk u<sub>0</sub><br><span style="font-size:8.5px; color:#0F766E;">1× Shared</span></div>
        <div style="background:#FEF3C7; border:1.3px solid #D97706; border-radius:6px; padding:3px 2px; color:#92400E;">Source u<sub>A</sub><br><span style="font-size:8.5px; color:#B91C1C;">Gets ∇K<sub>A</sub></span></div>
        <div style="background:#CCFBF1; border:1.3px solid #0D9488; border-radius:6px; padding:3px 2px; color:#065F46;">Rewind l<sub>post</sub><br><span style="font-size:8.5px;">GDN Hook</span></div>
        <div style="background:#DBEAFE; border:1.3px solid #2563EB; border-radius:6px; padding:3px 2px; color:#1E40AF;">Reader u<sub>B</sub><br><span style="font-size:8.5px;">Mounts K<sub>A</sub></span></div>
        <div style="background:#ECFDF5; border:1.4px solid #059669; border-radius:6px; padding:3px 2px; color:#065F46;">Outcome<br><span style="font-size:8.8px;">R = 2y − 1</span></div>
      </div>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- COLUMN (c): HEADLINE GAINS (CLEAN VISUAL BAR CHARTS IN COMIC SANS)     -->
  <!-- ===================================================================== -->
  <div class="col-card col-c">
    <div class="col-header" style="color:#0F172A; border-color:#CBD5E1;">
      <span>Headline Gains (4B)</span>
      <span class="tag" style="background:#DBEAFE; color:#1E40AF;">H200</span>
    </div>

    <!-- Chart 1: Deep Search Accuracy -->
    <div style="background:#FFFFFF; border:1.4px solid #CBD5E1; border-radius:8px; padding:5px 7px;">
      <div style="font-size:10.2px; font-weight:700; color:#0F172A; margin-bottom:4px; display:flex; justify-content:space-between;">
        <span>Deep Search Acc. (%)</span>
        <span style="color:#0D9488;">+10.5%</span>
      </div>
      <div style="display:flex; flex-direction:column; gap:4px; font-size:9.5px; font-weight:700;">
        <div>
          <div style="display:flex; justify-content:space-between;"><span>Linear-Big RL (4B)</span><span>43.2%</span></div>
          <div style="width:100%; height:7px; background:#E2E8F0; border-radius:4px; margin-top:1px;">
            <div style="width:72%; height:100%; background:#94A3B8; border-radius:4px;"></div>
          </div>
        </div>
        <div>
          <div style="display:flex; justify-content:space-between; color:#0F766E;"><span>Octopus RL (4B)</span><span>51.2% (52.1%)</span></div>
          <div style="width:100%; height:7px; background:#E2E8F0; border-radius:4px; margin-top:1px;">
            <div style="width:92%; height:100%; background:#0D9488; border-radius:4px;"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Chart 2: Cross-Branch Task Hijacking -->
    <div style="background:#FFFFFF; border:1.4px solid #CBD5E1; border-radius:8px; padding:5px 7px;">
      <div style="font-size:10.2px; font-weight:700; color:#0F172A; margin-bottom:4px; display:flex; justify-content:space-between;">
        <span>Task Hijack Rate ↓</span>
        <span style="color:#DC2626;">21× Lower</span>
      </div>
      <div style="display:flex; flex-direction:column; gap:4px; font-size:9.5px; font-weight:700;">
        <div>
          <div style="display:flex; justify-content:space-between; color:#991B1B;"><span>Text / All-Layer</span><span>57.9%</span></div>
          <div style="width:100%; height:7px; background:#E2E8F0; border-radius:4px; margin-top:1px;">
            <div style="width:88%; height:100%; background:#EF4444; border-radius:4px;"></div>
          </div>
        </div>
        <div>
          <div style="display:flex; justify-content:space-between; color:#0F766E;"><span>Octopus K2 (Ours)</span><span>2.7%</span></div>
          <div style="width:100%; height:7px; background:#E2E8F0; border-radius:4px; margin-top:1px;">
            <div style="width:6%; height:100%; background:#0D9488; border-radius:4px;"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom 2x2 Key Systems Badges -->
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:5px; text-align:center;">
      <div style="background:#FEF3C7; border:1.4px solid #F59E0B; border-radius:8px; padding:4px 2px;">
        <div style="font-size:12px; font-weight:700; color:#92400E;">98.3% ↓</div>
        <div style="font-size:8.8px; font-weight:700; color:#78350F;">Wire Traffic</div>
      </div>
      <div style="background:#CCFBF1; border:1.4px solid #0D9488; border-radius:8px; padding:4px 2px;">
        <div style="font-size:12px; font-weight:700; color:#0F766E;">89.6%</div>
        <div style="font-size:8.8px; font-weight:700; color:#115E59;">KV Cache Hit</div>
      </div>
    </div>
  </div>
</div>

<div class="caption-row">
  <div class="col-cap">(a) Prior Stateless / Replay Failure Modes</div>
  <div class="col-cap">(b) Octopus: Zero-Copy Cursor Tree, K2 Mount &amp; DAG-GRPO</div>
  <div class="col-cap">(c) Accuracy &amp; Systems Gains</div>
</div>

</body>
</html>
"""

html_path = Path("./figures/fig1_overview.html")
raw_pdf = Path("./figures/fig1_raw.pdf")
out_pdf = Path("./figures/fig1_overview.pdf")

html_path.write_text(HTML_CONTENT, encoding="utf-8")

chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([
    chrome_bin,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--virtual-time-budget=2000",
    f"--print-to-pdf={raw_pdf}",
    f"file://{html_path}"
], check=True)

subprocess.run([
    "/Library/TeX/texbin/pdfcrop",
    "--margins", "2",
    str(raw_pdf),
    str(out_pdf)
], check=True)

raw_pdf.unlink(missing_ok=True)
print("Successfully rendered Comic Sans MS Teaser Figure 1:", out_pdf)
