"""LeakAtlas dashboard: which training records leak, whose they are, and whether a cheap proxy saw it coming.

Run:  PYTHONPATH=src uv run panel serve src/leakatlas/viewer.py --show

Reads only data/leakatlas.duckdb (read-only). One control row drives every view; record selections
(box-select or click in the Atlas, Proxy-vs-Truth or Stability views) highlight the same records in every
record-level view and open the Record Inspector. ROC and Trade-off are configuration-level views: they follow
the controls and mark the current configuration.
"""

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ponytail: `panel serve` without an installed pkg

import datashader as ds
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from holoviews.operation.datashader import rasterize
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from leakatlas import theme
from leakatlas.data import LOADERS, FMNIST_MEAN, FMNIST_STD
from leakatlas.defenses import load_config, make_split
from leakatlas.proxies import VIEW
from leakatlas.store import FPR_GRID, connect

hv.extension("bokeh")
pn.extension(raw_css=[theme.CSS], design="native", sizing_mode="stretch_width")
pn.pane.HoloViews.param.linked_axes.default = False  # panes share dim names; never link their axes
theme.plot_defaults()
HOOKS = [theme.style_bokeh]
DEFENSES = list(theme.DEFENSE_LABEL)
LN10 = np.log(10)
RECORD_SQL = VIEW.split(" AS\n", 1)[1]  # record_view's SELECT, run as a subquery (dashboard is read-only)
CONFIGS = {d: load_config(Path(__file__).resolve().parents[2] / "configs" / f"{d}.yaml") for d in ("adult", "fmnist")}
CON = connect(read_only=True)


def sql(q, params=()):
    return CON.cursor().execute(q, list(params)).df()


# ----------------------------------------------------------------------------------------------- data access
@functools.lru_cache(maxsize=48)
def cell(dataset: str, size: int, defense: str) -> pd.DataFrame:
    df = sql(f"""SELECT seed, record_id, member, score_rmia, log_grad_norm, log_loss_ratio_rec, p_true,
                        sex, race, age_band, class_name
                 FROM ({RECORD_SQL}) v WHERE dataset = ? AND train_size = ? AND defense = ?""",
             (dataset, size, defense))
    df["gn10"] = df.log_grad_norm / LN10            # display in log10 (Wilke: name the variable, state the base)
    df["lr10"] = df.log_loss_ratio_rec / LN10
    return df


@functools.lru_cache(maxsize=48)
def thresholds(dataset: str, size: int, defense: str) -> pd.DataFrame:
    return sql("""SELECT r.seed, t.fpr, t.threshold FROM thresholds t JOIN runs r USING (run_id)
                  WHERE t.attack = 'rmia' AND r.dataset = ? AND r.train_size = ? AND r.defense = ?""",
               (dataset, size, defense))


@functools.lru_cache(maxsize=4)
def runs(dataset: str) -> pd.DataFrame:
    return sql("""SELECT r.*, a.auc, a.tpr_at_1 FROM runs r JOIN attack_results a USING (run_id)
                  WHERE a.attack = 'rmia' AND r.dataset = ?""", (dataset,))


FPR_AXIS = np.logspace(-4, 0, 200)


def tail(s):
    """Tail-stretched RMIA score, -log10(1 - s): the at-risk region (s > 0.99) gets most of the axis.
    Monotone, so ranks and thresholds are unchanged; ticks are labelled in raw score."""
    return -np.log10(np.clip(1 - np.asarray(s, float), 1e-4, 1))


SCORE_TICKS = [(float(tail(v)), lab) for v, lab in [(0, "0"), (0.9, "0.9"), (0.99, "0.99"), (0.999, "0.999"), (0.9999, "0.9999")]]
SCORE_LABEL = "RMIA attack score (tail-stretched)"


@functools.lru_cache(maxsize=48)
def roc(dataset: str, size: int, defense: str):
    """Per seed: TPR interpolated on a common log-FPR grid, and AUC (target ROC = evaluation only)."""
    df = cell(dataset, size, defense)
    curves, aucs = [], []
    for _, g in df.groupby("seed"):
        f, t, _ = roc_curve(g.member, g.score_rmia)
        curves.append(np.interp(FPR_AXIS, f, t))
        aucs.append(roc_auc_score(g.member, g.score_rmia))
    return np.array(curves), np.array(aucs)


def tpr_at(dataset, size, defense, fpr):
    curves, _ = roc(dataset, size, defense)
    return np.array([np.interp(fpr, FPR_AXIS, c) for c in curves])


def records(dataset, size, defense, fpr, seed) -> pd.DataFrame:
    """One row per audited record: flags per seed at the chosen FPR, plus proxy/score for the chosen seed
    (or the mean over seeds when seed == 'all')."""
    df = cell(dataset, size, defense).copy()
    thr = thresholds(dataset, size, defense)
    thr = thr[np.isclose(thr.fpr, fpr)].set_index("seed").threshold
    df["flag"] = df.score_rmia >= df.seed.map(thr)
    agg = df.groupby("record_id").agg(
        member=("member", "first"), k=("flag", "sum"), flag_rate=("flag", "mean"),
        sex=("sex", "first"), race=("race", "first"), age_band=("age_band", "first"),
        class_name=("class_name", "first"))
    src = df if seed == "all" else df[df.seed == int(seed)]
    vals = src.groupby("record_id")[["score_rmia", "gn10", "lr10", "p_true"]].mean()
    out = agg.join(vals).reset_index()
    out["st"] = tail(out.score_rmia)
    out["flag_now"] = out.flag_rate > 0 if seed == "all" else out.record_id.map(
        df[df.seed == int(seed)].set_index("record_id").flag)
    return out


@functools.lru_cache(maxsize=2)
def dataset_obj(name):
    return LOADERS[name]()


# ------------------------------------------------------------------------------------------------- state
class State(param.Parameterized):
    dataset = param.Selector(objects=["adult", "fmnist"], default="fmnist")
    size = param.Selector(objects=[5000, 15000], default=15000)
    defense = param.Selector(objects=["baseline", "aof", "dp", "miashield"], default="baseline")
    eps = param.Selector(objects=[1, 4, 8], default=8, label="ε (DP-SGD)")
    seed = param.Selector(objects=["all", "0", "1", "2"], default="all")
    fpr = param.Selector(objects={f"{100 * f:g}%": f for f in FPR_GRID}, default=0.01, label="FPR threshold")
    attribute = param.Selector(objects=["race", "age_band", "sex"], default="race", label="Subgroup (Adult)")
    selected = param.List(default=[], precedence=-1)

    @param.depends("dataset", watch=True, on_init=True)
    def _sizes(self):
        sizes = CONFIGS[self.dataset].sizes
        self.param.size.objects = sizes
        if self.size not in sizes:
            self.size = sizes[-1]
        self.selected = []

    @property
    def key(self):
        return f"dp_eps{self.eps}" if self.defense == "dp" else self.defense

    def frame(self):
        return records(self.dataset, self.size, self.key, self.fpr, self.seed)


S = State()
CTRL = ["dataset", "size", "defense", "eps", "seed", "fpr"]


def fmt_fpr(f):
    return f"{100 * f:g}%"


def section(n, name, takeaway, note, *body):
    return pn.Column(
        pn.pane.HTML(f'<p class="la-kicker">{n:02d} · {name}</p>'),
        pn.pane.HTML(pn.bind(lambda *_: f'<p class="la-takeaway">{takeaway()}</p>', *[S.param[c] for c in CTRL + ["attribute", "selected"]])),
        pn.pane.HTML(f'<p class="la-note">{note}</p>'), *body, css_classes=["la-view"])


def select_box(member_filter=None, xcol="gn10", ycol="st", frame_fn=None):
    """Return BoundsXY/Tap callbacks that turn a box or click into selected record ids."""
    def ids(bounds=None, x=None, y=None):
        df = (frame_fn or S.frame)()
        if member_filter is not None:
            df = df[df.member == member_filter]
        df = df.dropna(subset=[xcol, ycol])
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            hit = df[(df[xcol].between(x0, x1)) & (df[ycol].between(y0, y1))]
            S.selected = hit.record_id.astype(int).tolist()[:5000]
        elif x is not None and len(df):
            sx, sy = (df[xcol].max() - df[xcol].min()) or 1, (df[ycol].max() - df[ycol].min()) or 1
            d = ((df[xcol] - x) / sx) ** 2 + ((df[ycol] - y) / sy) ** 2
            S.selected = [int(df.record_id.iloc[int(np.argmin(d.values))])]
    return ids


def attach(dm, cb):
    hv.streams.BoundsXY(source=dm).add_subscriber(lambda bounds: cb(bounds=bounds) if bounds else None)
    hv.streams.Tap(source=dm).add_subscriber(lambda x, y: cb(x=x, y=y) if x is not None else None)
    return dm


def highlight(xcol, ycol, member_filter=None, rows=None):
    """Hollow ink rings on the selected records (selection is not 'risk', so it never uses the accent)."""
    def draw(**_):
        df = S.frame()
        df = df[df.record_id.isin(S.selected)]
        if member_filter is not None:
            df = df[df.member == member_filter]
        if rows is not None:
            df = rows(df)
        return hv.Points(df, [xcol, ycol]).opts(size=13, fill_alpha=0, line_color=theme.INK, line_width=2)
    return hv.DynamicMap(draw, streams=[hv.streams.Params(S, CTRL + ["selected"])])


def member_rows(df, member: bool):
    return df[df.member == member]


# --------------------------------------------------------------------------------- 1. Record Atlas (hero)
def atlas_panel(member: bool):
    params = hv.streams.Params(S, CTRL)
    base = hv.DynamicMap(lambda **_: hv.Points(member_rows(S.frame(), member), ["gn10", "st"]),
                         streams=[params])
    density = rasterize(base, aggregator=ds.count(), width=420, height=300).opts(
        cmap=["#ececee", "#c9c9cf", "#9a9aa3", "#55555e"], cnorm="eq_hist", colorbar=False, alpha=0.9,
        tools=["box_select", "tap"], active_tools=["box_select"], hooks=HOOKS, height=360, responsive=True,
        xlabel="Proxy A: gradient norm at final epoch (log10)", ylabel=SCORE_LABEL, yticks=SCORE_TICKS,
        title=("Training records (members)" if member else "Held-out records (non-members)"), ylim=(-0.1, 4.1))
    attach(density, select_box(member))

    def flagged(**_):
        df = member_rows(S.frame(), member)
        df = df[df.k > 0]
        pts = hv.Points(df, ["gn10", "st"], ["flag_rate", "k", "record_id", "score_rmia"]).opts(
            color="flag_rate", cmap=theme.SEQ, clim=(0, 1), size=7, line_color=theme.INK if member else None,
            line_width=1.2 if member else 0, colorbar=member, clabel="share of seeds flagged",
            tools=["hover"], hooks=HOOKS)
        thr = thresholds(S.dataset, S.size, S.key)
        y = float(tail(thr[np.isclose(thr.fpr, S.fpr)].threshold.median()))
        x0 = float(member_rows(S.frame(), member).gn10.min())
        line = hv.HLine(y).opts(color=theme.ACCENT, line_width=1.5)
        lab = hv.Text(x0, y, f" {fmt_fpr(S.fpr)} FPR threshold (median over seeds, set on reference models)",
                      halign="left", valign="bottom").opts(text_color=theme.ACCENT, text_font_size="11px")
        out = pts * line * lab
        if member and len(df):
            # call-outs stacked in the empty lower-left, leader lines to each record (no label collisions)
            top = df.sort_values(["k", "st"], ascending=False).head(5).assign(rank=range(1, 6)[:min(5, len(df))])
            top = top.sort_values("gn10")
            ly = np.linspace(y - 0.6, y - 2.2, len(top))
            span = float(member_rows(S.frame(), member).gn10.max()) - x0
            lx = x0 + 0.03 * span
            out = out * hv.Segments([(lx + 0.33 * span, yy, gx, gy) for yy, gx, gy in zip(ly, top.gn10, top.st)]).opts(
                color=theme.ACCENT, line_width=0.8, alpha=0.7)
            out = out * hv.Points(top, ["gn10", "st"]).opts(size=15, fill_alpha=0, line_color=theme.ACCENT, line_width=2)
            out = out * hv.Labels({"x": [lx] * len(top), "y": ly,
                                   "text": [f"#{n}  record {r} · {int(k)}/3 seeds" for n, r, k in
                                            zip(top["rank"], top.record_id, top.k)]},
                                  ["x", "y"], "text").opts(text_color=theme.ACCENT, text_font_size="10px", text_align="left")
        return out
    return density * hv.DynamicMap(flagged, streams=[hv.streams.Params(S, CTRL)]) * highlight("gn10", "st", member)


def atlas_takeaway():
    df = S.frame().query("member")
    n3, n = int((df.k == 3).sum()), len(df)
    rho = stats.spearmanr(df.gn10, df.score_rmia).statistic
    if abs(rho) < 0.1:
        where = f"the gradient norm barely orders exposure (Spearman ρ = {rho:+.2f})"
    else:
        where = f"exposure concentrates at {'large' if rho > 0 else 'small'} gradient norms (Spearman ρ = {rho:+.2f})"
    return (f"<em>{n3:,}</em> of {n:,} training records are flagged in all three seeds at {fmt_fpr(S.fpr)} FPR; {where}.")


# ------------------------------------------------------------------------------------------ 2. Log-log ROC
def roc_panels(**_):
    plots = []
    for d in DEFENSES:
        curves, aucs = roc(S.dataset, S.size, d)
        lo, med, hi = (np.clip(a, 1e-4, 1) for a in (curves.min(0), np.median(curves, 0), curves.max(0)))
        cur = d == S.key
        ink = theme.INK if cur else theme.INK_2
        band = hv.Area((FPR_AXIS, lo, hi), vdims=["lo", "hi"]).opts(color=theme.GHOST, alpha=0.8)
        line = hv.Curve((FPR_AXIS, med)).opts(color=ink, line_width=2.5 if cur else 1.5)
        chance = hv.Curve((FPR_AXIS, FPR_AXIS)).opts(color=theme.MUTED, line_width=1, line_dash="dotted")
        marks = []
        for f in (0.001, 0.01):
            t = float(np.median([np.interp(f, FPR_AXIS, c) for c in curves]))
            marks.append(hv.Points([(f, max(t, 1e-4))]).opts(color=ink, size=7))
            marks.append(hv.Text(f, max(t, 1e-4), f" {100 * t:.1f}% @ {fmt_fpr(f)}", halign="left",
                                 valign="top").opts(text_font_size="10px", text_color=ink))
        vline = hv.VLine(S.fpr).opts(color=theme.ACCENT, line_width=1, alpha=0.6)
        title = f"{theme.DEFENSE_LABEL[d]}{'  ◀ current' if cur else ''}\nAUC {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f}"
        plots.append(hv.Overlay([band, chance, line, vline, *marks]).opts(
            logx=True, logy=True, xlim=(1e-4, 1), ylim=(1e-4, 1), title=title, height=250, responsive=True,
            xlabel="FPR (log10)", ylabel="TPR (log10)", hooks=HOOKS, fontsize={"title": 10}))
    return hv.Layout(plots).cols(3).opts(shared_axes=False)


def roc_takeaway():
    t = tpr_at(S.dataset, S.size, S.key, S.fpr).mean()
    b = tpr_at(S.dataset, S.size, "baseline", S.fpr).mean()
    if S.key == "baseline":
        return (f"Undefended, RMIA identifies <em>{100 * t:.1f}%</em> of members at {fmt_fpr(S.fpr)} FPR "
                f"({t / S.fpr:.1f}× chance); the defense panels show how far each pulls this back to chance.")
    return (f"Under {theme.DEFENSE_LABEL[S.key]}, RMIA identifies <em>{100 * t:.1f}%</em> of members at "
            f"{fmt_fpr(S.fpr)} FPR, against {100 * b:.1f}% undefended (chance = {fmt_fpr(S.fpr)}).")


# ------------------------------------------------------------------------------------------------ layout
def controls():
    w = pn.Param(S, parameters=CTRL + ["attribute"], show_name=False, default_layout=pn.Row,
                 widgets={
                          "seed": {"type": pn.widgets.RadioButtonGroup, "button_type": "default", "width": 170},
                          "defense": {"type": pn.widgets.RadioButtonGroup, "button_type": "default", "width": 330},
                          "eps": {"width": 110}, "dataset": {"width": 110}, "size": {"width": 100},
                          "attribute": {"width": 150}, "fpr": {"type": pn.widgets.DiscreteSlider, "width": 220}})
    return pn.Column(w, css_classes=["la-controls"], styles={"position": "sticky", "top": "0", "z-index": "10", "background": theme.SURFACE,
                                "border-bottom": f"1px solid {theme.RULE}", "padding": "6px 0"})


def masthead():
    n = int(sql("SELECT count(*) n FROM record_scores").n[0])
    r = int(sql("SELECT count(*) n FROM runs").n[0])
    return pn.pane.HTML(
        '<div class="la-masthead"><h1>LeakAtlas</h1><p>Which training records does a model leak through membership '
        'inference, whose records are they, and could a cheap, attack-free signal have warned us? Every number '
        f'below is queried live from {n:,} record×run audit rows over {r} target models (RMIA via ML Privacy '
        'Meter; thresholds set on reference models only). Change the seed, defense, ε or FPR and watch whether a '
        'record stays at risk: one run is never the answer.</p></div>')


def build():
    return pn.Column(
        masthead(), controls(),
        section(1, "Record Atlas", atlas_takeaway,
                "Each training record is a point: x = gradient norm at the final epoch (Proxy A), y = RMIA score. "
                "Grey = density of all records; coloured = records flagged in ≥1 seed (colour = share of seeds), "
                "ink ring = training member; orange = the threshold and the five most exposed records.",
                pn.Row(pn.pane.HoloViews(atlas_panel(True)), pn.pane.HoloViews(atlas_panel(False)))),
        section(2, "Log-scale ROC", roc_takeaway,
                "One panel per defense (log-log, the LiRA convention). Line = median over 3 seeds, grey band = "
                "seed-to-seed range, dotted = chance. Orange line = the FPR control.",
                pn.pane.HoloViews(hv.DynamicMap(roc_panels, streams=[hv.streams.Params(S, CTRL)]))),
        pn.pane.HTML('<p class="la-note">Source: data/leakatlas.duckdb · regenerate with <code>make reproduce</code> · '
                     'design notes: docs/design_principles.md</p>'),
        styles={"max-width": "1280px", "margin": "0 auto", "padding": "24px 28px"})


build().servable(title="LeakAtlas")
