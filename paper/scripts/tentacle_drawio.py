#!/usr/bin/env python3
"""
Clean Draw.io Vector Curved Edge Helper for Figure 1, Figure 2, and Figure 3.
Replaces tentacle rendering with crisp, publication-grade curved Draw.io <mxCell> edges.
"""


def tentacle_edge(
    cid: str,
    src_id: str,
    dst_id: str,
    exit_xy,
    entry_xy,
    p0,
    p1,
    p2,
    outer_col: str = "#0F766E",
    inner_col: str = "#5EEAD4",
    cup_fill: str = "#CCFBF1",
    cup_stroke: str = "#0F766E",
    width: float = 2.2,
    n_cups: int = 0,
    cup_side: float = 1.0,
    dashed: bool = False,
    arrow: str = "classic",
) -> str:
    dash_style = "dashed=1;dashPattern=4 3;" if dashed else ""
    ex, ey = exit_xy
    nx_in, ny_in = entry_xy
    sw = max(1.8, min(2.4, width * 0.48))
    return (
        f'<mxCell id="{cid}" value="" '
        f'style="curved=1;strokeColor={outer_col};strokeWidth={sw:.1f};{dash_style}'
        f'exitX={ex:.2f};exitY={ey:.2f};entryX={nx_in:.2f};entryY={ny_in:.2f};'
        f'endArrow={arrow};endFill=1;endSize=4.5;strokeLinecap=round;" '
        f'source="{src_id}" target="{dst_id}">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<Array as="points"><mxPoint x="{p1[0]:.1f}" y="{p1[1]:.1f}"/></Array>'
        f'</mxGeometry></mxCell>'
    )
