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
