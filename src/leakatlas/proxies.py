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
from leakatlas.store import connect, query

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
