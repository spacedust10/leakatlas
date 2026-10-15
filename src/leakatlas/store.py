"""DuckDB persistence. One row per record x run in `record_scores`; runs/attack results/thresholds
are small side tables. Writes are idempotent: rows of a (dataset, train_size, defense) cell are
replaced as a unit."""

import tempfile
from pathlib import Path

import duckdb
import polars as pl
from pydantic import BaseModel

from leakatlas.data import DATA_DIR

DB_PATH = DATA_DIR / "leakatlas.duckdb"
FPR_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]  # dashboard FPR control snaps to these


class RunRow(BaseModel):
    run_id: str
    dataset: str
    defense: str
    epsilon: float | None
    train_size: int
    seed: int
    test_acc: float
    train_acc: float
    train_loss: float     # mean loss on members
    test_loss: float      # mean loss on non-members
    loss_ratio: float     # test_loss / train_loss (Jebreel et al.)
    train_sec: float      # this target model
    ref_train_sec: float  # all reference models of the cell (shared by the 3 seeds)
    epochs_run: int
    eps_spent: float | None
    delta: float | None
    offline_a: float
    num_refs_used: int


class AttackRow(BaseModel):
    run_id: str
    attack: str
    auc: float
    tpr_at_0p1: float          # evaluation: from the target ROC
    tpr_at_1: float
    shadow_tpr_at_1: float     # deployable: threshold calibrated on references only
    shadow_fpr_at_1: float     # realized FPR of that threshold on the target


SCHEMA = """
CREATE TABLE IF NOT EXISTS records (dataset VARCHAR, record_id BIGINT, sex VARCHAR, race VARCHAR,
    age_band VARCHAR, age DOUBLE, class_name VARCHAR);
CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR, dataset VARCHAR, defense VARCHAR, epsilon DOUBLE,
    train_size BIGINT, seed BIGINT, test_acc DOUBLE, train_acc DOUBLE, train_loss DOUBLE, test_loss DOUBLE,
    loss_ratio DOUBLE, train_sec DOUBLE, ref_train_sec DOUBLE, epochs_run BIGINT, eps_spent DOUBLE,
    delta DOUBLE, offline_a DOUBLE, num_refs_used BIGINT);
CREATE TABLE IF NOT EXISTS attack_results (run_id VARCHAR, attack VARCHAR, auc DOUBLE, tpr_at_0p1 DOUBLE,
    tpr_at_1 DOUBLE, shadow_tpr_at_1 DOUBLE, shadow_fpr_at_1 DOUBLE);
CREATE TABLE IF NOT EXISTS thresholds (run_id VARCHAR, attack VARCHAR, fpr DOUBLE, threshold DOUBLE);
CREATE TABLE IF NOT EXISTS record_scores (run_id VARCHAR, record_id BIGINT, member BOOLEAN, loss DOUBLE,
    log_loss DOUBLE, p_true DOUBLE, log_grad_norm DOUBLE, score_loss DOUBLE, score_rmia DOUBLE);
"""


def _insert(con, table: str, df: pl.DataFrame) -> None:
    # ponytail: Parquet hop instead of DuckDB's polars scan, which needs pyarrow (not a dependency).
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.parquet"
        df.write_parquet(path)
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM read_parquet(?)", [str(path)])


def query(con, sql: str, params=None) -> pl.DataFrame:
    """Run SQL and return a polars DataFrame (via numpy, no pyarrow)."""
    cols = con.execute(sql, params or []).fetchnumpy()
    return pl.DataFrame([pl.Series(k, v.tolist() if hasattr(v, "mask") else v) for k, v in cols.items()])


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def replace_cell(con, run_ids: list[str], runs: list[RunRow], attacks: list[AttackRow],
                 thresholds: pl.DataFrame, scores: pl.DataFrame) -> None:
    runs_df = pl.DataFrame([r.model_dump() for r in runs])
    attacks_df = pl.DataFrame([a.model_dump() for a in attacks])
    con.execute("BEGIN")
    for t in ("runs", "attack_results", "thresholds", "record_scores"):
        con.execute(f"DELETE FROM {t} WHERE run_id IN (SELECT unnest(?))", [run_ids])
    for t, df in (("runs", runs_df), ("attack_results", attacks_df), ("thresholds", thresholds),
                  ("record_scores", scores)):
        _insert(con, t, df)
    con.execute("COMMIT")


def replace_records(con, dataset: str, df: pl.DataFrame) -> None:
    con.execute("DELETE FROM records WHERE dataset = ?", [dataset])
    _insert(con, "records", df)
