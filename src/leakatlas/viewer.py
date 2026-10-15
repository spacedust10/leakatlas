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
        pn.pane.HTML('<p class="la-note">Source: data/leakatlas.duckdb · regenerate with <code>make reproduce</code> · '
                     'design notes: docs/design_principles.md</p>'),
        styles={"max-width": "1280px", "margin": "0 auto", "padding": "24px 28px"})


build().servable(title="LeakAtlas")
