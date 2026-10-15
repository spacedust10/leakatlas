"""One editorial theme for every LeakAtlas view. Rationale and sources: docs/design_principles.md.

Forensic console: warm light ink on a near-black surface. Exactly one accent, reserved for "at risk".
Scores and flag rates use cividis trimmed from the dark end (CVD-safe, one perceptual direction). Defenses are never told apart by hue
(6 > the 3-hue all-pairs limit): they get small multiples and direct labels instead.
"""

import holoviews as hv
from bokeh.palettes import cividis

# Contrast on SURFACE (WCAG): INK 15.2:1, INK_2 8.4:1, MUTED 4.1:1, ACCENT 7.3:1.
SURFACE = "#0f1115"    # page and plot background
PANEL = "#15181e"      # raised panels (sidebar, view cards)
INK = "#e9e6df"        # primary text, emphasized marks
INK_2 = "#a7adb8"      # secondary text, axis labels
MUTED = "#6f7682"      # non-focus marks, notes
RULE = "#262a33"       # grid, dividers
GHOST = "#3a3f4b"      # de-emphasized marks (other defenses, bands)
ACCENT = "#ff7a45"     # AT RISK only. dataviz validator: worst CVD dE 11.7 vs the cividis anchors on SURFACE
# cividis from step 91: the first step at >= 3:1 on SURFACE (its dark blues vanish on near-black)
SEQ = cividis(256)[91:]
DENSITY = ["#1b1e25", "#2c313b", "#4a505c", "#7d8491"]  # record-density raster, faint -> dense
# Fira Sans for prose, Fira Code for numbers/labels (tabular figures). System fallbacks keep it offline-safe.
FONT = "'Fira Sans', -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'Fira Code', ui-monospace, 'SF Mono', Menlo, monospace"
FONT_IMPORT = ("@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600"
               "&family=Fira+Sans:wght@400;500;600;700&display=swap');")

DEFENSE_LABEL = {
    "baseline": "Undefended (overfit)", "aof": "Anti-overfitting", "dp_eps1": "DP-SGD ε=1",
    "dp_eps4": "DP-SGD ε=4", "dp_eps8": "DP-SGD ε=8", "miashield": "MIAShield",
}

CSS = FONT_IMPORT + f"""
:root {{ --surface:{SURFACE}; --panel:{PANEL}; --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED}; --rule:{RULE};
        --accent:{ACCENT}; --font:{FONT}; --mono:{MONO}; }}
body, .bk-root, .pn-wrapper {{ background:{SURFACE}; color:{INK}; font-family:{FONT}; font-size:15px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}

/* type scale 11 / 12 / 13 / 15 / 19 / 26 */
.la-brand {{ font-family:{MONO}; font-size:26px; font-weight:600; letter-spacing:-0.02em; color:{INK}; margin:0; }}
.la-brand b {{ color:{ACCENT}; font-weight:600; }}
.la-thesis {{ font-size:13px; line-height:1.5; color:{INK_2}; margin:6px 0 0 0; }}
.la-label {{ font-family:{MONO}; font-size:11px; font-weight:500; letter-spacing:0.08em; text-transform:uppercase;
            color:{MUTED}; margin:18px 0 6px 0; }}
.la-kicker {{ font-family:{MONO}; font-size:12px; font-weight:500; letter-spacing:0.06em; text-transform:uppercase;
             color:{MUTED}; margin:0; }}
.la-kicker b {{ color:{ACCENT}; font-weight:600; margin-right:8px; }}
.la-takeaway {{ font-size:19px; font-weight:600; line-height:1.35; color:{INK}; margin:4px 0 6px 0; max-width:72ch; }}
.la-takeaway em {{ font-style:normal; color:{ACCENT}; font-family:{MONO}; }}
.la-view {{ background:{PANEL}; border:1px solid {RULE}; border-radius:8px; padding:16px 20px 12px 20px;
            margin:0 0 16px 0; }}
.la-note {{ font-size:12px; color:{INK_2}; margin:0 0 10px 0; max-width:96ch; line-height:1.5; }}

/* verdict ledger: one typographic line of facts, not tiles */
.la-ledger {{ font-family:{MONO}; font-size:13px; color:{INK_2}; line-height:1.9; border-top:1px solid {RULE};
             border-bottom:1px solid {RULE}; padding:8px 0; margin:0; }}
.la-ledger b {{ color:{INK}; font-weight:600; }}
.la-ledger .risk {{ color:{ACCENT}; font-weight:600; }}
.la-ledger .sep {{ color:{GHOST}; margin:0 10px; }}

/* section index */
.la-index a {{ display:block; font-family:{MONO}; font-size:12px; color:{INK_2}; text-decoration:none; padding:5px 8px;
              border-left:2px solid transparent; transition:color .15s ease-out, border-color .15s ease-out; }}
.la-index a:hover, .la-index a:focus-visible {{ color:{INK}; border-left-color:{ACCENT}; outline:none; }}
.la-index a span {{ color:{MUTED}; margin-right:8px; }}

/* case file (selected record) */
.la-case {{ font-size:12px; color:{INK_2}; line-height:1.55; border:1px solid {RULE}; border-radius:6px; padding:10px; }}
.la-case b {{ color:{INK}; font-family:{MONO}; }}
.la-case .risk {{ color:{ACCENT}; font-weight:600; }}

.la-card {{ font-size:13px; color:{INK}; line-height:1.5; }}
.la-card table {{ border-collapse:collapse; font-family:{MONO}; font-size:12px; }}
.la-card td {{ padding:2px 12px 2px 0; border-bottom:1px solid {RULE}; }}
.la-card td:first-child {{ color:{MUTED}; }}
.la-risk {{ color:{ACCENT}; font-weight:600; }}
"""


def plot_defaults() -> None:
    """Bokeh-level defaults: recessive solid grid, no chart junk, readable 12px+ text."""
    common = dict(fontsize={"title": 13, "labels": 12, "xticks": 11, "yticks": 11},
                  toolbar="above", default_tools=["pan", "wheel_zoom", "reset"],
                  bgcolor=PANEL, show_frame=False)
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
    p.background_fill_color = PANEL   # plots sit on view cards
    p.border_fill_color = PANEL
    p.outline_line_color = None
    if p.toolbar is not None:  # tools appear on hover only; no library logo competing with the data
        p.toolbar.logo = None
        p.toolbar.autohide = True
    for ax in p.axis:
        ax.axis_line_color = MUTED
        ax.major_tick_line_color = MUTED
        ax.minor_tick_line_color = None
        ax.axis_label_text_color = INK_2
        ax.axis_label_text_font_style = "normal"
        ax.major_label_text_color = INK_2
        ax.axis_label_text_font = FONT
        ax.major_label_text_font = MONO
    for g in p.grid:
        g.grid_line_color = RULE
        g.grid_line_dash = []
    if p.title is not None:
        p.title.text_color = INK
        p.title.text_font = FONT
        p.title.text_font_style = "bold"
