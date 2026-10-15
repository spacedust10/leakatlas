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
pn.extension(raw_css=[theme.CSS], design="native", theme="dark", sizing_mode="stretch_width")
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
        cmap=theme.DENSITY, cnorm="eq_hist", colorbar=False, alpha=0.9,
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


# --------------------------------------------------------------------------------- 3. Subgroup ridgelines
XS = np.linspace(0, 4, 200)


def lambda_pct():
    from bokeh.models import CustomJSTickFormatter
    return CustomJSTickFormatter(code="return (100 * tick).toFixed(1) + '%'")


def ridge_facet(defense):
    df = records("adult", S.size, defense, S.fpr, S.seed).query("member")
    attr = S.attribute
    groups = [g for g, n in df[attr].value_counts().items() if n >= 20]
    groups = sorted(groups, key=lambda g: df.loc[df[attr] == g, "flag_rate"].mean())
    rng = np.random.default_rng(0)
    areas, dots, bars, labels = [], [], [], []
    for i, g in enumerate(groups):
        sub = df[df[attr] == g]
        dens = stats.gaussian_kde(sub.st)(XS) if sub.st.std() > 0 else np.zeros_like(XS)
        dens = 0.85 * dens / dens.max() if dens.max() > 0 else dens
        cur = defense == S.key
        areas.append(hv.Area((XS, i + dens, np.full_like(XS, i)), "score_t", ["ridge", "ridge0"]).opts(
            color=theme.MUTED if cur else theme.GHOST, alpha=0.75))
        areas.append(hv.Curve((XS, i + dens), "score_t", "ridge").opts(color=theme.INK if cur else theme.INK_2, line_width=1))
        sel = sub[sub.record_id.isin(S.selected)]
        if len(sel):
            areas.append(hv.Spikes(sel.st.to_numpy(), "score_t").opts(position=i, spike_length=0.3, color=theme.INK, line_width=1.5))
        v = sub.flag_rate.to_numpy()
        boot = v[rng.integers(0, len(v), (500, len(v)))].mean(1)  # percentile bootstrap over records
        lo, hi = np.percentile(boot, [2.5, 97.5])
        dots.append((v.mean(), i))
        bars.append((lo, i, hi, i))
        labels.append((4.0, i + 0.3, f"{g}  (n={len(v)})"))
    ridge = hv.Overlay(areas + [hv.Labels(labels, ["score_t", "ridge"], "text").opts(text_align="right", text_font_size="10px", text_color=theme.INK_2)]).opts(
        title=theme.DEFENSE_LABEL[defense], xlabel=SCORE_LABEL, xticks=SCORE_TICKS, yaxis=None,
        height=60 + 46 * len(groups), xlim=(0, 4.05), responsive=True, hooks=HOOKS, fontsize={"title": 11})
    xmax = max(0.02, max(b[2] for b in bars) * 1.15) if bars else 0.05
    dot = (hv.Segments(bars, ["rate_lo", "row", "rate_hi", "row_b"]).opts(color=theme.INK, line_width=1.5)
           * hv.Points(dots, ["rate_lo", "row"]).opts(color=theme.ACCENT, size=7)).opts(
        xlim=(0, xmax), ylim=(-0.3, len(groups) - 0.2), yaxis=None, xlabel=f"flag rate @ {fmt_fpr(S.fpr)} (95% CI)",
        height=60 + 46 * len(groups), width=170, hooks=HOOKS, title=" ", xformatter=lambda_pct(), xticks=3,
        toolbar=None)
    return pn.Row(pn.pane.HoloViews(ridge, sizing_mode="stretch_width", linked_axes=False),
                  pn.pane.HoloViews(dot, sizing_mode="fixed", width=190, linked_axes=False))


def ridgelines(*_):
    if S.dataset != "adult":
        return pn.pane.HTML('<p class="la-note">Subgroup analysis uses Adult\'s sex, race and age band. '
                            'Switch the dataset to <b>adult</b> to see it.</p>')
    return pn.GridBox(*[ridge_facet(d) for d in DEFENSES], ncols=2)


def ridge_takeaway():
    if S.dataset != "adult":
        return "Subgroup exposure is measured on Adult only."
    df = S.frame().query("member")
    rates = df.groupby(S.attribute).agg(rate=("flag_rate", "mean"), n=("flag_rate", "size")).query("n >= 20").sort_values("rate")
    lo, hi = rates.iloc[0], rates.iloc[-1]
    ratio = f"{hi.rate / lo.rate:.1f}× as often as" if lo.rate > 0 else "while none of"
    return (f"Under {theme.DEFENSE_LABEL[S.key]}, <em>{rates.index[-1]}</em> members are flagged {ratio} "
            f"{rates.index[0]} members ({100 * hi.rate:.2f}% vs {100 * lo.rate:.2f}%).")


# ---------------------------------------------------------------------------------- 4. Proxy vs truth
def proxy_density():
    params = hv.streams.Params(S, CTRL)
    base = hv.DynamicMap(lambda **_: hv.Points(member_rows(S.frame(), True), ["lr10", "st"]), streams=[params])
    dens = rasterize(base, aggregator=ds.count(), width=360, height=280).opts(
        cmap=list(theme.SEQ), cnorm="log", colorbar=True, clabel="records per pixel (log)", tools=["box_select", "tap"],
        active_tools=["box_select"], responsive=True, height=340, hooks=HOOKS, ylim=(-0.1, 4.1), yticks=SCORE_TICKS,
        xlabel="Proxy B: record loss / median held-out loss (log10)", ylabel=SCORE_LABEL,
        title="Per record (training records)")
    attach(dens, select_box(True, "lr10"))
    return dens * highlight("lr10", "st", True)


def model_level(**_):
    r = pd.concat([runs(d) for d in ("adult", "fmnist")])
    g = r.groupby(["dataset", "train_size", "defense"]).agg(lr=("loss_ratio", "mean"), tpr=("tpr_at_1", "mean")).reset_index()
    g["mlr10"] = np.log10(g.lr)
    rho = stats.spearmanr(g.mlr10, g.tpr).statistic
    slope, icpt = np.polyfit(g.mlr10, g.tpr, 1)
    xs = np.linspace(g.mlr10.min(), g.mlr10.max(), 50)
    cur = g[(g.dataset == S.dataset) & (g.train_size == S.size) & (g.defense == S.key)]
    pts = hv.Points(g, ["mlr10", "tpr"], ["dataset", "train_size", "defense"]).opts(
        color=theme.INK_2, size=8, tools=["hover"], marker="circle")
    fit = hv.Curve((xs, slope * xs + icpt)).opts(color=theme.INK, line_width=1.5)
    ring = hv.Points(cur, ["mlr10", "tpr"]).opts(size=16, fill_alpha=0, line_color=theme.INK, line_width=2)
    lab = hv.Text(g.mlr10.min(), g.tpr.max(), f"Spearman ρ = {rho:.2f}  (n = {len(g)} configurations)",
                  halign="left", valign="top").opts(text_font_size="12px", text_color=theme.INK)
    names = hv.Labels({"x": cur.mlr10, "y": cur.tpr, "t": ["  current"] * len(cur)}, ["x", "y"], "t").opts(
        text_align="left", text_font_size="10px")
    return (pts * fit * ring * lab * names).opts(
        xlabel="Model-level loss ratio, test / train (log10)", ylabel="RMIA TPR @ 1% FPR (mean of 3 seeds)",
        title="Per configuration (both datasets)", height=340, responsive=True, hooks=HOOKS)


def proxy_takeaway():
    r = pd.concat([runs(d) for d in ("adult", "fmnist")])
    g = r.groupby(["dataset", "train_size", "defense"]).agg(lr=("loss_ratio", "mean"), tpr=("tpr_at_1", "mean"))
    rho_m = stats.spearmanr(g.lr, g.tpr).statistic
    df = S.frame().query("member")
    rho_r = stats.spearmanr(df.lr10, df.score_rmia).statistic
    return (f"Across configurations the loss ratio tracks attack success (ρ = <em>{rho_m:.2f}</em>); per record, "
            f"under {theme.DEFENSE_LABEL[S.key]} its rank correlation with the attack score is {rho_r:+.2f}.")


# ---------------------------------------------------------------------------------- 5. Stability strip
def jitter(df):
    # ponytail: seeded vertical jitter, not a packed beeswarm; swap in a packing pass if overlap hides structure
    return df.assign(yj=df.k + (np.random.default_rng(0).random(len(df)) - 0.5) * 0.7)


def stability_frame():
    df = S.frame().query("member and k > 0")
    return jitter(df) if len(df) else df.assign(yj=[])


def stability():
    def draw(**_):
        df = stability_frame()
        if not len(df):
            return hv.Points([], ["st", "yj"])
        return hv.Points(df, ["st", "yj"], ["k", "record_id", "flag_rate", "score_rmia"]).opts(
            color="flag_rate", cmap=theme.SEQ, clim=(0, 1), size=5, tools=["hover", "box_select", "tap"],
            active_tools=["box_select"])
    dm = hv.DynamicMap(draw, streams=[hv.streams.Params(S, CTRL)])
    attach(dm, select_box(True, "st", "yj", frame_fn=stability_frame))
    hl = hv.DynamicMap(lambda **_: hv.Points(stability_frame()[lambda d: d.record_id.isin(S.selected)], ["st", "yj"]).opts(
        size=11, fill_alpha=0, line_color=theme.INK, line_width=2), streams=[hv.streams.Params(S, CTRL + ["selected"])])
    return (dm * hl).opts(yticks=[(1, "1 of 3 seeds"), (2, "2 of 3"), (3, "all 3 seeds")], ylim=(0.4, 3.6),
                          xlabel=SCORE_LABEL + ", mean over seeds", xticks=SCORE_TICKS, xlim=(-0.1, 4.1), ylabel="", height=300, responsive=True,
                          hooks=HOOKS, title="Training records flagged at least once")


def stability_takeaway():
    df = S.frame().query("member")
    n = df.k.value_counts()
    n1, n3, n0 = int(n.get(1, 0)), int(n.get(3, 0)), int(n.get(0, 0))
    ever = max(1, int((df.k > 0).sum()))
    return (f"<em>{n3:,}</em> records are flagged in every seed and {n1:,} in only one: {100 * n3 / ever:.0f}% of "
            f"flagged records are reproducibly at risk ({n0:,} are never flagged).")


# --------------------------------------------------------------------------------- 6. Record inspector
def neighbors(rid, k=5):
    d = dataset_obj("fmnist")
    split = make_split(len(d.y), CONFIGS["fmnist"], S.size)
    pool = split.members[split.members != rid]
    flat = d.x[pool].flatten(1).numpy()
    q = d.x[rid].flatten().numpy()
    return pool[np.argsort(((flat - q) ** 2).sum(1))[:k]]


def image(rid, title, w=150):
    d = dataset_obj("fmnist")
    img = (d.x[rid, 0].numpy() * FMNIST_STD + FMNIST_MEAN)[::-1]
    return hv.Image(img, bounds=(0, 0, 28, 28)).opts(cmap="gray", xaxis=None, yaxis=None, frame_width=w,
                                                      frame_height=w, title=title, toolbar=None, fontsize={"title": 9})


def inspector(*_):
    if not S.selected:
        return pn.pane.HTML('<p class="la-note">Click a point (or box-select) in the Atlas, Proxy or Stability view.</p>')
    rid = int(S.selected[0])
    per_def = []
    for d in DEFENSES:
        r = records(S.dataset, S.size, d, S.fpr, S.seed).set_index("record_id").loc[rid]
        per_def.append((theme.DEFENSE_LABEL[d], float(r.st), int(r.k), float(r.p_true)))
    cur = S.frame().set_index("record_id").loc[rid]
    risk = f'<span class="la-risk">at risk: flagged in {int(cur.k)}/3 seeds</span>' if cur.k else "not flagged"
    who = (f"{cur.sex}, {cur.race}, age {cur.age_band}" if S.dataset == "adult" else f"class: {cur.class_name}")
    head = (f'<div class="la-card"><b>Record {rid}</b> · {"training member" if cur.member else "held-out"} · {who}<br>'
            f'{risk} under {theme.DEFENSE_LABEL[S.key]} · model confidence in true label {100 * cur.p_true:.1f}%'
            f'{"" if len(S.selected) == 1 else f" · {len(S.selected):,} records selected (showing the first)"}</div>')
    xs = list(range(len(per_def)))
    base_y = per_def[0][1]
    segs = hv.Segments([(0, base_y, i, y) for i, (_, y, _, _) in enumerate(per_def)]).opts(color=theme.GHOST, line_width=1.5)
    pts = hv.Points([(i, y, k) for i, (_, y, k, _) in enumerate(per_def)], vdims=["k"]).opts(
        color="k", cmap=theme.SEQ, clim=(0, 3), size=10, line_color=theme.INK, line_width=1, tools=["hover"])
    risky = hv.Points([(i, y) for i, (_, y, k, _) in enumerate(per_def) if k > 0]).opts(
        size=17, fill_alpha=0, line_color=theme.ACCENT, line_width=2)
    short = ["Undefended", "Anti-overfit", "DP ε=1", "DP ε=4", "DP ε=8", "MIAShield"]
    slope = (segs * pts * risky).opts(xticks=list(enumerate(short)),
                                      ylim=(-0.1, 4.1), yticks=SCORE_TICKS, xlim=(-0.4, len(xs) - 0.6), ylabel=SCORE_LABEL,
                                      xlabel="", height=280, width=760, hooks=HOOKS,
                                      title="Attack score under each defense (orange ring = flagged)")
    if S.dataset == "fmnist":
        nn = neighbors(rid)
        visual = pn.Row(image(rid, "this record", 150),
                        *[image(int(j), f"neighbour {i + 1}", 90) for i, j in enumerate(nn)])
    else:
        row = dataset_obj("adult").raw.filter(dataset_obj("adult").raw["record_id"] == rid).to_dicts()[0]
        visual = pn.pane.HTML('<div class="la-card"><table>' + "".join(
            f"<tr><td>{k.replace('_', ' ')}</td><td>{v}</td></tr>" for k, v in row.items() if k != "record_id")
                              + "</table></div>", width=340)
    return pn.Column(pn.pane.HTML(head), visual, pn.pane.HoloViews(slope, sizing_mode="fixed", width=780))


def inspector_takeaway():
    if not S.selected:
        return "Select any record to see whether its risk survives each defense."
    rid = int(S.selected[0])
    ks = {d: int(records(S.dataset, S.size, d, S.fpr, S.seed).set_index("record_id").loc[rid].k) for d in DEFENSES}
    kept = [theme.DEFENSE_LABEL[d] for d in DEFENSES[1:] if ks[d] > 0]
    return (f"Record {rid} is flagged in {ks['baseline']}/3 undefended seeds; "
            + (f"it stays flagged under <em>{', '.join(kept)}</em>." if kept else "no defense leaves it flagged."))


# ---------------------------------------------------------------------------------- 7. Defense trade-off
T95 = stats.t.ppf(0.975, 2)  # 3 seeds -> 2 dof


def tradeoff_frame():
    r = runs(S.dataset)
    r = r[r.train_size == S.size]
    rows = []
    for d in DEFENSES:
        g = r[r.defense == d]
        t = tpr_at(S.dataset, S.size, d, S.fpr)
        acc = g.test_acc.to_numpy()
        cost = g.train_sec.mean() + g.ref_train_sec.iloc[0]
        rows.append(dict(defense=d, label=theme.DEFENSE_LABEL[d], tpr=t.mean(), tpr_ci=T95 * t.std(ddof=1) / np.sqrt(len(t)),
                         acc=acc.mean(), acc_ci=T95 * acc.std(ddof=1) / np.sqrt(len(acc)), cost=cost))
    df = pd.DataFrame(rows)
    df["pareto"] = [not ((df.acc >= a) & (df.tpr <= t) & ((df.acc > a) | (df.tpr < t))).any() for a, t in zip(df.acc, df.tpr)]
    return df


def tradeoff(**_):
    df = tradeoff_frame()
    df["size_px"] = 8 + 22 * np.sqrt(df.cost / df.cost.max())   # marker AREA ∝ cost
    front = df[df.pareto].sort_values("tpr")
    eb_y = hv.ErrorBars(df, "tpr", ["acc", "acc_ci"]).opts(color=theme.INK_2)
    eb_x = hv.ErrorBars([(t, a, e) for t, a, e in zip(df.tpr, df.acc, df.tpr_ci)], horizontal=True).opts(color=theme.INK_2)
    pts = hv.Points(df, ["tpr", "acc"], ["label", "cost", "size_px"]).opts(
        size="size_px", color=theme.MUTED, line_color=theme.INK, line_width=1, tools=["hover"])
    cur = hv.Points(df[df.defense == S.key], ["tpr", "acc"]).opts(size=30, fill_alpha=0, line_color=theme.INK, line_width=2)
    frontier = hv.Curve(front, "tpr", "acc").opts(color=theme.INK_2, line_width=1.2)
    xr = max(df.tpr.max() + df.tpr_ci.max(), S.fpr) - min(df.tpr.min() - df.tpr_ci.max(), S.fpr)
    yr = (df.acc.max() - df.acc.min()) or 0.01
    order = df.sort_values("acc", ascending=False).reset_index(drop=True)
    ly = list(order.acc)
    for i in range(1, len(ly)):  # dodge: keep stacked labels at least 6% of the y-range apart
        if abs(ly[i - 1] - ly[i]) < 0.06 * yr and abs(order.tpr[i - 1] - order.tpr[i]) < 0.15 * xr:
            ly[i] = ly[i - 1] - 0.06 * yr
    labels = hv.Labels({"x": order.tpr + 0.045 * xr, "y": ly, "t": list(order.label)}, ["x", "y"], "t").opts(
        text_align="left", text_font_size="11px", text_color=theme.INK)
    chance = hv.VLine(S.fpr).opts(color=theme.ACCENT, line_width=1, alpha=0.7)
    ctext = hv.Text(S.fpr, df.acc.min() - df.acc_ci.max(), f" chance = {fmt_fpr(S.fpr)}", halign="left",
                    valign="bottom").opts(text_color=theme.ACCENT, text_font_size="10px")
    return (eb_y * eb_x * frontier * pts * cur * labels * chance * ctext).opts(
        xlabel=f"Privacy risk: RMIA TPR @ {fmt_fpr(S.fpr)} FPR (mean, 95% CI over 3 seeds)",
        ylabel="Utility: test accuracy (mean, 95% CI)", height=420, responsive=True, hooks=HOOKS,
        title="Marker area ∝ training cost (target + reference models); thin line = Pareto front")


def tradeoff_takeaway():
    df = tradeoff_frame()
    base = df[df.defense == "baseline"].iloc[0]
    best = df[df.pareto & (df.defense != "baseline")].sort_values("acc", ascending=False)
    if not len(best):
        return "No defense improves on the undefended model's trade-off here."
    b = best.iloc[0]
    return (f"<em>{b.label}</em> is the most accurate defense on the Pareto front: {100 * b.acc:.1f}% accuracy "
            f"(vs {100 * base.acc:.1f}% undefended) with TPR {100 * b.tpr:.1f}% at {fmt_fpr(S.fpr)} FPR "
            f"(vs {100 * base.tpr:.1f}%).")


# ------------------------------------------------------------------------------------------------ layout
GROUPS = [("Scope", ["dataset", "size"]), ("Defense", ["defense", "eps"]), ("Evidence", ["seed", "fpr"]),
          ("Subgroup", ["attribute"])]
WIDGETS = {"seed": {"type": pn.widgets.RadioButtonGroup, "button_type": "default"},
           "defense": {"type": pn.widgets.RadioButtonGroup, "button_type": "default"},
           "fpr": {"type": pn.widgets.DiscreteSlider}}


def controls():
    """Controls grouped by the question they answer, each group under a visible label."""
    col = []
    for label, params in GROUPS:
        col.append(pn.pane.HTML(f'<p class="la-label">{label}</p>', margin=0))
        col.append(pn.Param(S, parameters=params, show_name=False, default_layout=pn.Column,
                            widgets={k: v for k, v in WIDGETS.items() if k in params}, margin=0))
    return pn.Column(*col, css_classes=["la-controls"], margin=0)


def brand():
    n = int(sql("SELECT count(*) n FROM record_scores").n[0])
    r = int(sql("SELECT count(*) n FROM runs").n[0])
    return pn.pane.HTML(
        '<p class="la-brand">Leak<b>Atlas</b></p><p class="la-thesis">Which training records a model leaks through '
        'membership inference, whose they are, and whether a cheap attack-free signal saw it coming. '
        f'Live from {n:,} record×run rows over {r} target models.</p>', margin=0)


def sidebar():
    return pn.Column(brand(), controls(), sizing_mode="fixed", width=300, margin=0,
                     styles={"position": "sticky", "top": "0", "height": "100vh", "overflow-y": "auto",
                             "background": theme.PANEL, "border-right": f"1px solid {theme.RULE}",
                             "padding": "22px 20px 28px 20px", "box-sizing": "border-box"})


def build():
    main = pn.Column(
        section(1, "Record Atlas", atlas_takeaway,
                "Each training record is a point: x = gradient norm at the final epoch (Proxy A), y = RMIA score. "
                "Grey = density of all records; coloured = records flagged in ≥1 seed (colour = share of seeds), "
                "ink ring = training member; orange = the threshold and the five most exposed records.",
                pn.Row(pn.pane.HoloViews(atlas_panel(True)), pn.pane.HoloViews(atlas_panel(False)))),
        section(2, "Log-scale ROC", roc_takeaway,
                "One panel per defense (log-log, the LiRA convention). Line = median over 3 seeds, grey band = "
                "seed-to-seed range, dotted = chance. Orange line = the FPR control.",
                pn.pane.HoloViews(hv.DynamicMap(roc_panels, streams=[hv.streams.Params(S, CTRL)]))),
        section(3, "Subgroup ridgelines", ridge_takeaway,
                "Adult training records. Ridges = distribution of attack scores per subgroup (sorted by flag rate); "
                "right column = flag rate with 95% percentile-bootstrap CI over records (seeds averaged per record). "
                "Groups under 20 members are omitted. Ink ticks = selected records.",
                pn.panel(pn.bind(ridgelines, *[S.param[c] for c in CTRL + ["attribute", "selected"]]))),
        section(4, "Proxy vs truth", proxy_takeaway,
                "Left: per-record loss ratio (Proxy B) against the attack score; box-select to pick records. "
                "Right: one point per configuration, model-level test/train loss ratio against attack TPR, with an "
                "OLS fit; ring = current configuration.",
                pn.Row(pn.pane.HoloViews(proxy_density()),
                       pn.pane.HoloViews(hv.DynamicMap(model_level, streams=[hv.streams.Params(S, CTRL)])))),
        section(5, "Stability strip", stability_takeaway,
                "Training records flagged in at least one seed, one row per number of seeds that flag them. "
                "Flags that appear in only one seed are what a single run would wrongly call a finding.",
                pn.pane.HoloViews(stability())),
        section(6, "Record inspector", inspector_takeaway,
                "The selected record, its nearest training neighbours (Fashion-MNIST) or feature row (Adult), and "
                "its attack score under every defense.",
                pn.panel(pn.bind(inspector, *[S.param[c] for c in CTRL + ["selected"]]))),
        section(7, "Defense trade-off", tradeoff_takeaway,
                "One point per defense at the current dataset and size. ε is written next to each DP-SGD point.",
                pn.pane.HoloViews(hv.DynamicMap(tradeoff, streams=[hv.streams.Params(S, CTRL)]))),
        pn.pane.HTML('<p class="la-note">Source: data/leakatlas.duckdb · regenerate with <code>make reproduce</code> · '
                     'design notes: docs/design_principles.md</p>'),
        styles={"padding": "22px 32px 40px 32px", "max-width": "1400px"}, margin=0)
    return pn.Row(sidebar(), main, sizing_mode="stretch_width", margin=0, styles={"background": theme.SURFACE})


if pn.state.location is not None:  # shareable, reproducible views: every control (and the selection) lives in the URL
    pn.state.location.sync(S, {c: c for c in CTRL + ["attribute"]})
app = build()
if (rid := pn.state.session_args.get("record")) is not None:  # e.g. ?record=698 opens that record's card
    S.selected = [int(rid[0])]
app.servable(title="LeakAtlas")
