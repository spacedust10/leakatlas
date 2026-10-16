"""One editorial theme for every LeakAtlas view, in a dark and a light mode.
Rationale and sources: docs/design_principles.md.

Exactly one accent, reserved for "at risk". Scores and flag rates use cividis, trimmed so every step keeps
>= 3:1 contrast on the mode's surface (CVD-safe, one perceptual direction). Defenses are never told apart by
hue (6 > the 3-hue all-pairs limit): they get small multiples and direct labels instead.

Plot colours come from a per-session Palette (module state is shared by every browser session); page CSS
reads the same palette through CSS custom properties set on the app root, which inherit into Panel's
shadow roots.
"""

from dataclasses import dataclass

import holoviews as hv
from bokeh.palettes import cividis


@dataclass(frozen=True)
class Palette:
    name: str
    SURFACE: str   # page background
    PANEL: str     # raised panels (sidebar, view cards) and plot backgrounds
    INK: str       # primary text, emphasized marks
    INK_2: str     # secondary text, axis labels
    MUTED: str     # non-focus marks, notes
    RULE: str      # grid, dividers
    GHOST: str     # de-emphasized marks (other defenses, bands)
    ACCENT: str    # AT RISK only
    SEQ: tuple     # sequential ramp for scores / flag rates
    DENSITY: tuple  # record-density raster, faint -> dense
    IMG_CMAP: str  # Fashion-MNIST thumbnails


# Dark: contrast on SURFACE (WCAG) INK 15.2, INK_2 8.4, MUTED 4.1, ACCENT 7.3. dataviz validator: ACCENT worst
# CVD dE 11.7 vs the cividis anchors. cividis from step 91 = first step at >= 3:1 on near-black.
DARK = Palette("dark", "#0f1115", "#15181e", "#e9e6df", "#a7adb8", "#6f7682", "#262a33", "#3a3f4b",
               "#ff7a45", tuple(cividis(256)[91:]), ("#1b1e25", "#2c313b", "#4a505c", "#7d8491"), "gray")
# Light: contrast on SURFACE INK 16.3, INK_2 7.9, MUTED 4.8, ACCENT 4.7. ACCENT worst CVD dE 11.5 vs cividis.
# cividis cut at 85%: its last yellows are ~1.2:1 on a light surface.
LIGHT = Palette("light", "#f4f3ef", "#fbfbf9", "#16161a", "#4a4a52", "#6b6b74", "#e2e1dc", "#cfcfd4",
                "#c2410c", tuple(cividis(256)[:217]), ("#ececee", "#c9c9cf", "#9a9aa3", "#55555e"), "gray_r")
PALETTES = {"dark": DARK, "light": LIGHT}


def palette(name: str) -> Palette:
    return PALETTES.get(name, DARK)


def css_vars(p: Palette) -> dict:
    """Custom properties for the app root; the stylesheet below only references these."""
    return {"--surface": p.SURFACE, "--panel": p.PANEL, "--ink": p.INK, "--ink2": p.INK_2, "--muted": p.MUTED,
            "--rule": p.RULE, "--ghost": p.GHOST, "--accent": p.ACCENT, "background": p.SURFACE}


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
body, .bk-root, .pn-wrapper {{ font-family:{FONT}; font-size:15px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}

/* type scale 11 / 12 / 13 / 15 / 19 / 26 */
.la-brand {{ font-family:{MONO}; font-size:26px; font-weight:600; letter-spacing:-0.02em; color:var(--ink); margin:0; }}
.la-brand b {{ color:var(--accent); font-weight:600; }}
.la-thesis {{ font-size:13px; line-height:1.5; color:var(--ink2); margin:6px 0 0 0; }}
.la-label {{ font-family:{MONO}; font-size:11px; font-weight:500; letter-spacing:0.08em; text-transform:uppercase;
            color:var(--muted); margin:18px 0 6px 0; }}
.la-kicker {{ font-family:{MONO}; font-size:12px; font-weight:500; letter-spacing:0.06em; text-transform:uppercase;
             color:var(--muted); margin:0; }}
.la-takeaway {{ font-size:19px; font-weight:600; line-height:1.35; color:var(--ink); margin:4px 0 6px 0; max-width:72ch; }}
.la-takeaway em {{ font-style:normal; color:var(--accent); font-family:{MONO}; }}
.la-view {{ background:var(--panel); border:1px solid var(--rule); border-radius:8px; padding:16px 20px 12px 20px;
            margin:0 0 16px 0; }}
.la-note {{ font-size:12px; color:var(--ink2); margin:0 0 10px 0; max-width:96ch; line-height:1.5; }}

/* verdict ledger: one typographic line of facts, not tiles */
.la-ledger {{ font-family:{MONO}; font-size:13px; color:var(--ink2); line-height:1.9; border-top:1px solid var(--rule);
             border-bottom:1px solid var(--rule); padding:8px 0; margin:0; }}
.la-ledger b {{ color:var(--ink); font-weight:600; }}
.la-ledger .risk {{ color:var(--accent); font-weight:600; }}
.la-ledger .sep {{ color:var(--ghost); margin:0 10px; }}

/* case file (selected record) */
.la-case {{ font-size:12px; color:var(--ink2); line-height:1.55; border:1px solid var(--rule); border-radius:6px; padding:10px; }}
.la-case b {{ color:var(--ink); font-family:{MONO}; }}
.la-case .risk {{ color:var(--accent); font-weight:600; }}

/* explanations: native <details>, keyboard-accessible, collapsible */
.la-primer, .la-explain {{ border:1px solid var(--rule); border-radius:8px; background:var(--panel); margin:0 0 12px 0; }}
.la-primer {{ padding:12px 18px; margin:10px 0 16px 0; }}
.la-explain {{ padding:8px 14px; margin:2px 0 12px 0; background:transparent; }}
.la-primer summary, .la-explain summary {{ cursor:pointer; font-family:{MONO}; font-size:12px; letter-spacing:0.04em;
    color:var(--ink2); list-style-position:outside; }}
.la-primer summary:focus-visible, .la-explain summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.la-primer summary b {{ color:var(--ink); font-weight:600; }}
.la-primer p, .la-explain p {{ font-size:13.5px; line-height:1.6; color:var(--ink2); margin:8px 0; max-width:96ch; }}
.la-primer p b, .la-explain p b {{ color:var(--ink); font-weight:600; }}
.la-explain h4 {{ font-family:{MONO}; font-size:11px; font-weight:500; letter-spacing:0.08em; text-transform:uppercase;
    color:var(--muted); margin:12px 0 2px 0; }}
.la-explain .privacy {{ border-left:2px solid var(--accent); padding-left:10px; }}

.la-card {{ font-size:13px; color:var(--ink); line-height:1.5; }}
.la-card table {{ border-collapse:collapse; font-family:{MONO}; font-size:12px; }}
.la-card td {{ padding:2px 12px 2px 0; border-bottom:1px solid var(--rule); }}
.la-card td:first-child {{ color:var(--muted); }}
.la-risk {{ color:var(--accent); font-weight:600; }}
"""


def plot_defaults() -> None:
    """Mode-independent Bokeh defaults (these are process-global, so no colours here)."""
    common = dict(fontsize={"title": 13, "labels": 12, "xticks": 11, "yticks": 11},
                  toolbar="above", default_tools=["pan", "wheel_zoom", "reset"], show_frame=False)
    hv.opts.defaults(
        hv.opts.Scatter(**common, size=5, line_color=None),
        hv.opts.Points(**common, size=5),
        hv.opts.Curve(**common, line_width=2),
        hv.opts.Area(**common, line_width=0),
        hv.opts.Image(**common),
        hv.opts.RGB(**common),
        hv.opts.HexTiles(**common),
        hv.opts.ErrorBars(**common, line_width=1.5),
        hv.opts.Spread(**common, line_width=0),
        hv.opts.Segments(**common),
        hv.opts.Labels(text_font_size="11px"),
    )


def hook(p: Palette):
    """Figure hook for one palette: card-coloured background, solid hairline grid, quiet axes, theme font."""
    def style_bokeh(plot, element) -> None:
        f = plot.state
        f.background_fill_color = p.PANEL   # plots sit on view cards
        f.border_fill_color = p.PANEL
        f.outline_line_color = None
        if f.toolbar is not None:  # tools appear on hover only; no library logo competing with the data
            f.toolbar.logo = None
            f.toolbar.autohide = True
        for ax in f.axis:
            ax.axis_line_color = p.MUTED
            ax.major_tick_line_color = p.MUTED
            ax.minor_tick_line_color = None
            ax.axis_label_text_color = p.INK_2
            ax.axis_label_text_font_style = "normal"
            ax.major_label_text_color = p.INK_2
            ax.axis_label_text_font = FONT
            ax.major_label_text_font = MONO
        for g in f.grid:
            g.grid_line_color = p.RULE
            g.grid_line_dash = []
        if f.title is not None:
            f.title.text_color = p.INK
            f.title.text_font = FONT
            f.title.text_font_style = "bold"
    return style_bokeh
