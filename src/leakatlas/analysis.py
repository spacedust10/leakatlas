"""Aggregate queries over the stored audit. Every reported number comes from here.

Usage: uv run python -m leakatlas.analysis   (prints the per-configuration table)
"""

import polars as pl

from leakatlas.store import connect, query

DEFENSE_ORDER = ["baseline", "aof", "dp_eps1", "dp_eps4", "dp_eps8", "miashield"]


def config_table(con=None) -> pl.DataFrame:
    """Utility, attack success (mean ± std over seeds) and training cost per configuration."""
    con = con or connect(read_only=True)
    df = query(con, """
        SELECT r.dataset, r.defense, r.train_size, r.seed, r.test_acc, r.loss_ratio, r.eps_spent,
               r.train_sec, r.ref_train_sec, a.attack, a.auc, a.tpr_at_0p1, a.tpr_at_1,
               a.shadow_tpr_at_1, a.shadow_fpr_at_1
        FROM runs r JOIN attack_results a USING (run_id)""")
    stats = ["test_acc", "auc", "tpr_at_0p1", "tpr_at_1", "shadow_tpr_at_1", "shadow_fpr_at_1", "loss_ratio"]
    out = df.group_by("dataset", "defense", "train_size", "attack").agg(
        [pl.col(c).mean().alias(c) for c in stats]
        + [pl.col(c).std().alias(f"{c}_sd") for c in stats]
        + [pl.col("eps_spent").mean(), pl.col("train_sec").mean().alias("target_sec"),
           pl.col("ref_train_sec").first().alias("refs_sec"), pl.len().alias("seeds")])
    order = {d: i for i, d in enumerate(DEFENSE_ORDER)}
    return out.with_columns(pl.col("defense").replace_strict(order).alias("_o")).sort(
        "dataset", "train_size", "_o", "attack").drop("_o")


def markdown(df: pl.DataFrame) -> str:
    pm = lambda c, d=3, pct=False: (lambda r: (f"{100 * r[c]:.{d}f} ± {100 * r[c + '_sd']:.{d}f}" if pct
                                               else f"{r[c]:.{d}f} ± {r[c + '_sd']:.{d}f}"))
    head = ("| dataset | S | defense | ε spent (RDP) | attack | test acc | AUC | TPR@0.1% FPR (%) | TPR@1% FPR (%) "
            "| shadow-thr TPR / realized FPR @1% (%) | loss ratio | target s | refs s |")
    rows = [head, "|" + "---|" * 13]
    for r in df.iter_rows(named=True):
        eps = "–" if r["eps_spent"] is None else f"{r['eps_spent']:.2f}"
        rows.append(
            f"| {r['dataset']} | {r['train_size']} | {r['defense']} | {eps} | {r['attack']} | "
            f"{pm('test_acc')(r)} | {pm('auc')(r)} | {pm('tpr_at_0p1', 2, True)(r)} | {pm('tpr_at_1', 2, True)(r)} | "
            f"{100 * r['shadow_tpr_at_1']:.2f} / {100 * r['shadow_fpr_at_1']:.2f} | {pm('loss_ratio', 2)(r)} | "
            f"{r['target_sec']:.1f} | {r['refs_sec']:.0f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(markdown(config_table()))
