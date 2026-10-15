"""Phase 3: can attack-free proxies predict per-record membership exposure?

Usage: uv run python -m leakatlas.proxies   (writes results/phase3_*.md)

Proxy A  log_grad_norm      : log of the per-sample loss-gradient norm at the final weights (Tobaben et al.)
Proxy B  log_loss_ratio_rec : log(record loss / median non-member loss of the same run), the per-record
                              version of Jebreel et al.'s test/train loss ratio. The defender knows its own
                              hold-out. Logs avoid float underflow; rankings equal those of the raw ratios.
Truth    flag             : RMIA flags the record at the reference-calibrated 1% FPR threshold.

Integrity: proxy direction and top-q cutoff are chosen on seed 0 x dev records only (record_id even);
evaluation runs once on seeds 1-2 x test records (record_id odd).
"""

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from leakatlas.analysis import DEFENSE_ORDER
from leakatlas.store import connect, query, replace_table

RESULTS = Path(__file__).resolve().parents[2] / "results"
PROXIES = ["log_grad_norm", "log_loss_ratio_rec"]
Q_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]  # flag the top-q fraction of records by proxy
CELL = ["dataset", "train_size", "defense"]
SUBGROUPS = ["sex", "race", "age_band"]
VIEW = """
CREATE OR REPLACE VIEW record_view AS
WITH thr AS (SELECT run_id, fpr, threshold FROM thresholds WHERE attack = 'rmia'),
     med AS (SELECT run_id, median(log_loss) FILTER (WHERE NOT member) AS med_out FROM record_scores GROUP BY run_id)
SELECT r.dataset, r.defense, r.epsilon, r.train_size, r.seed, s.*,
       s.log_loss - med.med_out AS log_loss_ratio_rec,
       s.score_rmia >= t1.threshold AS flag,
       s.record_id % 2 = 0 AS dev,
       rec.sex, rec.race, rec.age_band, rec.class_name
FROM record_scores s JOIN runs r USING (run_id) JOIN med USING (run_id)
JOIN thr t1 ON t1.run_id = s.run_id AND t1.fpr = 0.01
JOIN records rec ON rec.dataset = r.dataset AND rec.record_id = s.record_id
"""


def create_view(con) -> None:
    con.execute(VIEW)


def members(con) -> pl.DataFrame:
    return query(con, "SELECT * FROM record_view WHERE member")


def _top_q(x: np.ndarray, q: float) -> np.ndarray:
    return x >= np.quantile(x, 1 - q)


def tune(dev: pl.DataFrame, proxy: str) -> dict:
    """Pick sign (+1: high proxy = exposed) and q maximizing F1 against flags. Seed 0, dev records only."""
    y = dev["flag"].to_numpy()
    if y.sum() == 0:
        return {"sign": None, "q": None, "dev_f1": None}
    best = max(((f1(_top_q(s * dev[proxy].to_numpy(), q), y), -q, s) for s in (1, -1) for q in Q_GRID))
    return {"sign": best[2], "q": -best[1], "dev_f1": best[0]}


def f1(pred, y):
    tp = (pred & y).sum()
    return 0.0 if tp == 0 else 2 * tp / (pred.sum() + y.sum())


def evaluate(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for key, cell in df.group_by(CELL, maintain_order=True):
        for proxy in PROXIES:
            t = tune(cell.filter((pl.col("seed") == 0) & pl.col("dev")), proxy)
            for seed in (1, 2):
                test = cell.filter((pl.col("seed") == seed) & ~pl.col("dev"))
                y, s = test["flag"].to_numpy(), test[proxy].to_numpy()
                row = dict(zip(CELL, key)) | {"proxy": proxy, "seed": seed, "n": len(y), "base_rate": y.mean()} | t
                rho = stats.spearmanr(s, test["score_rmia"].to_numpy()).statistic
                if t["sign"] is None or y.sum() == 0:
                    rows.append(row | {"spearman": rho, "precision": None, "recall": None, "lift": None})
                    continue
                pred = _top_q(t["sign"] * s, t["q"])
                prec = (pred & y).sum() / pred.sum()
                rows.append(row | {
                    "spearman": stats.spearmanr(s, test["score_rmia"].to_numpy()).statistic,  # raw, unoriented
                    "precision": prec, "recall": (pred & y).sum() / y.sum(), "lift": prec / y.mean()})
    return pl.DataFrame(rows, infer_schema_length=None)


def model_level(con) -> pl.DataFrame:
    """Spearman between Jebreel's model-level loss ratio and RMIA TPR@1% FPR across all target runs."""
    df = query(con, """SELECT r.dataset, r.loss_ratio, a.tpr_at_1, a.auc FROM runs r
                       JOIN attack_results a USING (run_id) WHERE a.attack = 'rmia'""")
    rows = []
    for name, g in [("pooled", df)] + [(d, df.filter(pl.col("dataset") == d)) for d in ("adult", "fmnist")]:
        for metric in ("tpr_at_1", "auc"):
            r = stats.spearmanr(g["loss_ratio"].to_numpy(), g[metric].to_numpy())
            rows.append({"scope": name, "metric": metric, "n_runs": len(g), "spearman": r.statistic, "p": r.pvalue})
    return pl.DataFrame(rows)


def stability(df: pl.DataFrame) -> pl.DataFrame:
    """Per training record: number of seeds (of 3) in which it is flagged; observed vs independent-chance."""
    per_rec = df.group_by(CELL + ["record_id"]).agg(pl.col("flag").sum().alias("k"))
    rate = df.group_by(CELL + ["seed"]).agg(pl.col("flag").mean().alias("p"))
    chance = rate.group_by(CELL).agg(pl.col("p").product().alias("chance_3of3_rate"))
    out = per_rec.group_by(CELL).agg([(pl.col("k") == i).sum().alias(f"flagged_{i}of3") for i in range(4)])
    n = pl.sum_horizontal(f"flagged_{i}of3" for i in range(4))
    return out.join(chance, on=CELL).with_columns(
        (pl.col("chance_3of3_rate") * n).alias("expected_3of3_if_independent"))


def subgroups(df: pl.DataFrame, n_boot: int = 2000) -> pl.DataFrame:
    """Adult: per-record flag rate (mean over seeds) per subgroup with 95% BCa bootstrap CI, plus the
    max-min gap between groups of each attribute with a percentile bootstrap CI."""
    adult = df.filter(pl.col("dataset") == "adult")
    per_rec = adult.group_by(CELL + ["record_id"] + SUBGROUPS).agg(
        pl.col("flag").mean().alias("rate"), pl.col("score_rmia").median().alias("score")
    )
    rng = np.random.default_rng(0)
    rows = []
    for key, cell in per_rec.group_by(CELL, maintain_order=True):
        for attr in SUBGROUPS:
            groups = {g: c["rate"].to_numpy() for (g,), c in cell.group_by(attr)}
            groups = {g: v for g, v in groups.items() if len(v) >= 20}
            means = {g: v.mean() for g, v in groups.items()}
            gap = stats.bootstrap(tuple(groups.values()), lambda *gs, axis=-1: np.ptp([g.mean(axis) for g in gs], axis=0),
                                  n_resamples=n_boot, method="percentile", random_state=rng, vectorized=True)
            for g, v in groups.items():
                ci = (stats.bootstrap((v,), np.mean, n_resamples=n_boot, random_state=rng).confidence_interval
                      if v.std() > 0 else (v.mean(), v.mean()))
                rows.append(dict(zip(CELL, key)) | {
                    "attribute": attr, "group": g, "n": len(v), "flag_rate": means[g], "ci_lo": ci[0], "ci_hi": ci[1],
                    "median_score": float(np.median(cell.filter(pl.col(attr) == g)["score"].to_numpy())),
                    "gap": float(np.ptp(list(means.values()))), "gap_lo": gap.confidence_interval[0],
                    "gap_hi": gap.confidence_interval[1]})
    return pl.DataFrame(rows)


def _order(df):
    """Deterministic row order (group_by output order is not guaranteed), so reruns are byte-identical."""
    ties = [c for c in ("proxy", "attribute", "group") if c in df.columns]
    return df.with_columns(pl.col("defense").replace_strict({d: i for i, d in enumerate(DEFENSE_ORDER)}).alias("_o")
                           ).sort(["dataset", "train_size", "_o", *ties]).drop("_o")


def _md(df: pl.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cell = lambda c, v: "–" if v is None else fmt.get(c, "{:.4f}" if isinstance(v, float) else "{}").format(v)
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(cell(c, v) for c, v in zip(df.columns, r)) + " |" for r in df.iter_rows()]
    return "\n".join(rows)


def main():
    con = connect()
    create_view(con)
    df = members(con)
    RESULTS.mkdir(exist_ok=True)
    ev = evaluate(df)
    summary = _order(ev.group_by(CELL + ["proxy", "sign", "q"]).agg(
        pl.col("base_rate").mean(), pl.col("spearman").mean().alias("spearman"),
        pl.col("spearman").std().alias("spearman_sd"), pl.col("precision").mean(), pl.col("recall").mean(),
        pl.col("lift").mean()))
    ml, st, sg = model_level(con), _order(stability(df)), _order(subgroups(df))
    for name, title, frame in [
        ("phase3_proxy_eval", "Proxy evaluation on held-out seeds 1-2 x test records (tuned on seed 0 x dev)", summary),
        ("phase3_model_level", "Model-level loss ratio vs RMIA success across target runs", ml),
        ("phase3_stability", "Per-record flag stability across 3 seeds (training records)", st),
        ("phase3_subgroups", "Adult subgroup flag rates (members) with 95% bootstrap CIs", sg)]:
        (RESULTS / f"{name}.md").write_text(f"# {title}\n\nGenerated by `leakatlas.proxies` from data/leakatlas.duckdb.\n\n{_md(frame)}\n")
        replace_table(con, name, frame)  # summary.md queries these tables
        print(f"wrote results/{name}.md ({len(frame)} rows)")
    con.close()


if __name__ == "__main__":
    main()
