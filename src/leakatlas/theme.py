"""One editorial theme for every LeakAtlas view. Rationale and sources: docs/design_principles.md.

Dark ink on a warm-white surface. Exactly one accent, reserved for "at risk". Scores and flag rates use a
truncated cividis ramp (CVD-safe, one perceptual direction). Defenses are never told apart by hue
(6 > the 3-hue all-pairs limit): they get small multiples and direct labels instead.
"""

import holoviews as hv
from bokeh.palettes import cividis

SURFACE = "#fcfcfb"
INK = "#16161a"
INK_2 = "#55555e"      # secondary text, axis labels
MUTED = "#9a9aa3"      # non-focus marks, notes
RULE = "#e4e4e7"       # grid, dividers
GHOST = "#d4d4d8"      # de-emphasized marks (other defenses, never-flagged density)
ACCENT = "#c2410c"     # AT RISK only: flagged records, the FPR threshold, top-exposure call-outs
# cividis cut at 85%: its last yellows are ~1.2:1 on SURFACE (validator WARN), too faint for marks
SEQ = cividis(256)[: int(256 * 0.85)]
FONT = "Inter, -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"

DEFENSE_LABEL = {
    "baseline": "Undefended (overfit)", "aof": "Anti-overfitting", "dp_eps1": "DP-SGD ε=1",
    "dp_eps4": "DP-SGD ε=4", "dp_eps8": "DP-SGD ε=8", "miashield": "MIAShield",
}

CSS = f"""
:root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED}; --rule:{RULE}; --accent:{ACCENT}; }}
body, .bk-root, .pn-wrapper {{ background:{SURFACE}; color:{INK}; font-family:{FONT}; }}
.la-masthead h1 {{ font-size:28px; font-weight:700; letter-spacing:-0.01em; margin:0 0 4px 0; color:{INK}; }}
.la-masthead p {{ font-size:14px; color:{INK_2}; margin:0; max-width:980px; line-height:1.45; }}
.la-view {{ border-top:1px solid {RULE}; padding-top:14px; margin-top:22px; }}
.la-kicker {{ font-size:11px; font-weight:650; letter-spacing:0.08em; text-transform:uppercase; color:{MUTED}; margin:0; }}
.la-takeaway {{ font-size:20px; font-weight:650; line-height:1.3; color:{INK}; margin:2px 0 4px 0; max-width:1100px; }}
.la-takeaway em {{ font-style:normal; color:{ACCENT}; }}
.la-note {{ font-size:12px; color:{INK_2}; margin:0 0 8px 0; max-width:1100px; line-height:1.45; }}
.la-card {{ font-size:13px; color:{INK}; line-height:1.5; }}
.la-card table {{ border-collapse:collapse; }}
.la-card td {{ padding:1px 10px 1px 0; border-bottom:1px solid {RULE}; }}
.la-card td:first-child {{ color:{INK_2}; }}
.la-risk {{ color:{ACCENT}; font-weight:650; }}
"""


def plot_defaults() -> None:
    """Bokeh-level defaults: recessive solid grid, no chart junk, readable 12px+ text."""
    common = dict(fontsize={"title": 13, "labels": 12, "xticks": 11, "yticks": 11},
                  toolbar="above", default_tools=["pan", "wheel_zoom", "reset"],
                  bgcolor=SURFACE, show_frame=False)
    hv.opts.defaults(
        hv.opts.Scatter(**common, size=5, line_color=None),
        hv.opts.Points(**common, size=5),
        hv.opts.Curve(**common, line_width=2, color=INK),
        hv.opts.Area(**common, line_width=0),
        hv.opts.Image(**common),
        hv.opts.RGB(**common),
        hv.opts.HexTiles(**common),
        hv.opts.ErrorBars(**common, line_width=1.5, color=INK),
        hv.opts.Spread(**common, line_width=0),
        hv.opts.Segments(**common),
        hv.opts.Labels(text_font_size="11px", text_color=INK_2),
    )


def style_bokeh(plot, element) -> None:
    """Hook applied to every figure: solid hairline grid, quiet axes, theme font."""
    p = plot.state
    p.background_fill_color = SURFACE
    p.border_fill_color = SURFACE
    p.outline_line_color = None
    for ax in p.axis:
        ax.axis_line_color = MUTED
        ax.major_tick_line_color = MUTED
        ax.minor_tick_line_color = None
        ax.axis_label_text_color = INK_2
        ax.axis_label_text_font_style = "normal"
        ax.major_label_text_color = INK_2
        ax.axis_label_text_font = FONT
        ax.major_label_text_font = FONT
    for g in p.grid:
        g.grid_line_color = RULE
        g.grid_line_dash = []
    if p.title is not None:
        p.title.text_color = INK
        p.title.text_font = FONT
        p.title.text_font_style = "bold"
