#!/usr/bin/env python3
"""
Reusable helper to render a standalone SVG file (with exact W x H pixel canvas)
into a 100% vector PDF (CID TrueType embedded local Inter 4.0 fonts) and an optional
2x Retina preview PNG via headless Chromium / Playwright, plus rich subscript/superscript
`<tspan>` formatting (`_{...}` and `^{...}`) that preserves literal code underscores (`read_thread`).
"""
import base64
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = ROOT / "scripts/fonts"
FIGURES_DIR = ROOT / "figures"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

_TOKEN_RE = re.compile(r"(_\{[^{}]+\}|\^\{[^{}]+\}|\{[^{}|]+\|[^{}]+\})")


@lru_cache(maxsize=64)
def _load_mascot_b64(name: str) -> tuple[str, int, int]:
    p = FIGURES_DIR / (name if name.endswith(".png") else f"{name}.png")
    with Image.open(p) as im:
        w, h = im.size
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return b64, w, h


def svg_mascot(name: str, cx: float, cy: float, max_w: float, max_h: float) -> str:
    """Embed an Octopus RGBA PNG mascot centered at (cx, cy) preserving exact aspect ratio."""
    b64, ow, oh = _load_mascot_b64(name)
    scale = min(max_w / ow, max_h / oh)
    w = ow * scale
    h = oh * scale
    x = cx - w / 2.0
    y = cy - h / 2.0
    uri = f"data:image/png;base64,{b64}"
    return (
        f'<image id="mascot-{escape(name)}" x="{x:.2f}" y="{y:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}" preserveAspectRatio="xMidYMid meet" '
        f'href="{uri}" xlink:href="{uri}"/>'
    )


@lru_cache(maxsize=64)
def _get_pil_font(weight: int, size_tenths: int) -> ImageFont.FreeTypeFont:
    if weight >= 700:
        fname = "Inter-Bold.ttf"
    elif weight >= 600:
        fname = "Inter-SemiBold.ttf"
    elif weight >= 500:
        fname = "Inter-Medium.ttf"
    else:
        fname = "Inter-Regular.ttf"
    # Measure at 10x resolution and divide by 10 for sub-pixel float accuracy
    return ImageFont.truetype(str(FONTS_DIR / fname), size_tenths)


def _measure_run(text: str, size: float, weight: int, ls: float) -> float:
    if not text:
        return 0.0
    fnt = _get_pil_font(weight, max(10, int(round(size * 10))))
    raw_w = fnt.getlength(text) / 10.0
    return raw_w + ls * max(0, len(text) - 1)


def figma_rich_text(
    x: float,
    y: float,
    content: str,
    size: float = 16.0,
    weight: int = 600,
    fill: str = "#18181B",
    anchor: str = "middle",
    ls: float = 0.0,
    italic: bool = False,
) -> str:
    """
    Emit Figma-native flat `<text>` elements supporting:
      - `_{sub}` subscripts (exact pixel x/y offset, no `<tspan dy>` flattening in Figma!)
      - `^{sup}` superscripts
      - `{#HEX|text}` or `{bold|text}` or `{#HEX:700|text}` inline color/weight spans!
    Uses natural Inter glyph metrics (ls=0.0 by default) so PIL, Chromium PDF, and Figma match 1:1.
    """
    # Collapse consecutive spaces just like SVG/HTML text layout so PIL advance widths match 1:1
    content = re.sub(r" {2,}", " ", content)
    style_attr = ""
    ls_attr = f' letter-spacing="{ls:.2f}px"' if abs(ls) > 1e-6 else ""
    if "_{" not in content and "^{" not in content and "{" not in content:
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Inter" font-size="{size:.1f}" '
            f'font-weight="{weight}"{style_attr} fill="{fill}" text-anchor="{anchor}"'
            f'{ls_attr}>{escape(content)}</text>'
        )

    parts = _TOKEN_RE.split(content)
    runs: list[tuple[str, float, int, str, float, float]] = []
    # Each run: (txt, run_size, run_weight, run_fill, dy, run_ls)
    for tok in parts:
        if not tok:
            continue
        if tok.startswith("_{") and tok.endswith("}"):
            txt = tok[2:-1]
            r_size = round(size * 0.74, 1)
            runs.append((txt, r_size, weight, fill, size * 0.23, ls * 0.7))
        elif tok.startswith("^{") and tok.endswith("}"):
            txt = tok[2:-1]
            r_size = round(size * 0.73, 1)
            runs.append((txt, r_size, weight, fill, -size * 0.30, ls * 0.7))
        elif tok.startswith("{") and tok.endswith("}") and "|" in tok:
            spec, txt = tok[1:-1].split("|", 1)
            r_fill = fill
            r_weight = weight
            if ":" in spec:
                c_part, w_part = spec.split(":", 1)
                if c_part:
                    r_fill = c_part
                if w_part.isdigit():
                    r_weight = int(w_part)
            elif spec == "bold":
                r_weight = 700
            elif spec == "reg":
                r_weight = 400
            elif spec.startswith("#"):
                r_fill = spec
            runs.append((txt, size, r_weight, r_fill, 0.0, ls))
        else:
            runs.append((tok, size, weight, fill, 0.0, ls))

    widths = [_measure_run(t, rs, rw, rls) for (t, rs, rw, _, _, rls) in runs]
    # Add tiny optical kerning gap around script transitions
    total_w = sum(widths)
    if anchor == "middle":
        cur_x = x - total_w / 2.0
    elif anchor == "end":
        cur_x = x - total_w
    else:
        cur_x = x

    els: list[str] = ["<g>"]
    for (t, rs, rw, rf, dy, rls), w_px in zip(runs, widths):
        stripped = t.lstrip(" ")
        if stripped:
            n_lead = len(t) - len(stripped)
            lead_w = _measure_run(" " * n_lead, rs, rw, rls) + (rls if n_lead > 0 else 0.0) if n_lead > 0 else 0.0
            rls_attr = f' letter-spacing="{rls:.2f}px"' if abs(rls) > 1e-6 else ""
            els.append(
                f'  <text x="{cur_x + lead_w:.2f}" y="{y + dy:.2f}" font-family="Inter" '
                f'font-size="{rs:.1f}" font-weight="{rw}"{style_attr} fill="{rf}" '
                f'text-anchor="start"{rls_attr}>{escape(stripped.rstrip(" "))}</text>'
            )
        cur_x += w_px
    els.append("</g>")
    return "\n".join(els)


def format_rich_tspans(content: str, size: float) -> str:
    """Convert `_{sub}` and `^{sup}` into exact-baseline `<tspan>` elements while leaving plain `_` intact."""
    if "_{" not in content and "^{" not in content:
        return escape(content)
    parts = _TOKEN_RE.split(content)
    cur_dy = 0.0
    out: list[str] = []
    for tok in parts:
        if not tok:
            continue
        if tok.startswith("_{") and tok.endswith("}"):
            txt = tok[2:-1]
            target_dy = round(size * 0.22, 2)
            fsize = round(size * 0.76, 1)
        elif tok.startswith("^{") and tok.endswith("}"):
            txt = tok[2:-1]
            target_dy = round(-size * 0.30, 2)
            fsize = round(size * 0.75, 1)
        else:
            txt = tok
            target_dy = 0.0
            fsize = size

        delta = target_dy - cur_dy
        cur_dy = target_dy
        attrs = []
        if abs(delta) > 0.01:
            attrs.append(f'dy="{delta:.2f}"')
        if abs(fsize - size) > 0.05:
            attrs.append(f'font-size="{fsize:.1f}"')
        if attrs:
            out.append(f'<tspan {" ".join(attrs)}>{escape(txt)}</tspan>')
        else:
            out.append(escape(txt))
    return "".join(out)


def render_svg_to_pdf(svg_path: Path, pdf_path: Path, width: int, height: int, png_path: Path | None = None) -> None:
    svg_text = svg_path.read_text(encoding="utf-8")
    f_reg = (FONTS_DIR / "Inter-Regular.ttf").as_uri()
    f_med = (FONTS_DIR / "Inter-Medium.ttf").as_uri()
    f_semi = (FONTS_DIR / "Inter-SemiBold.ttf").as_uri()
    f_bold = (FONTS_DIR / "Inter-Bold.ttf").as_uri()

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @font-face {{
    font-family: 'Inter';
    font-style: normal;
    font-weight: 400;
    src: url('{f_reg}') format('truetype');
  }}
  @font-face {{
    font-family: 'Inter';
    font-style: normal;
    font-weight: 500;
    src: url('{f_med}') format('truetype');
  }}
  @font-face {{
    font-family: 'Inter';
    font-style: normal;
    font-weight: 600;
    src: url('{f_semi}') format('truetype');
  }}
  @font-face {{
    font-family: 'Inter';
    font-style: normal;
    font-weight: 700;
    src: url('{f_bold}') format('truetype');
  }}
  @page {{
    size: {width}px {height}px;
    margin: 0;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0;
    padding: 0;
    width: {width}px;
    height: {height}px;
    overflow: hidden;
    background: #FFFFFF;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  svg {{
    display: block;
    width: {width}px;
    height: {height}px;
  }}
</style>
</head>
<body>
{svg_text}
</body>
</html>
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        tmp.write(html)
        tmp_html = Path(tmp.name)

    try:
        with sync_playwright() as p:
            launch_kwargs = {"headless": True}
            if Path(CHROME_BIN).exists():
                launch_kwargs["executable_path"] = CHROME_BIN
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=2,
            )
            page = context.new_page()
            page.goto(tmp_html.as_uri(), wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(120)
            if png_path is not None:
                page.screenshot(path=str(png_path), full_page=False)
            page.pdf(
                path=str(pdf_path),
                width=f"{width}px",
                height=f"{height}px",
                print_background=True,
                margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"},
            )
            browser.close()
    finally:
        tmp_html.unlink(missing_ok=True)
