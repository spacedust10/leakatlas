"""Dataset loading and preprocessing for LeakAtlas.

Adult: UCI Machine Learning Repository id 2 (Becker & Kohavi, 1996), CC BY 4.0.
Fashion-MNIST: Xiao, Rasul & Vollgraf (2017), MIT license, via torchvision.
"""

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torchvision.datasets import FashionMNIST

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]
ADULT_NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
AGE_BANDS = [(0, 30, "<30"), (30, 45, "30-44"), (45, 60, "45-59"), (60, 200, "60+")]
FMNIST_MEAN, FMNIST_STD = 0.2860, 0.3530


@dataclass
class Dataset:
    """All records of one dataset. Splitting into target/shadow pools happens later."""

    name: str
    x: torch.Tensor  # float32 features or [N,1,28,28] images
    y: torch.Tensor  # int64 labels
    groups: pl.DataFrame  # per-record attributes kept for analysis (subgroups / image label)
    num_classes: int
    source_sha256: str


def _download(url: str, dest: Path) -> bytes:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return dest.read_bytes()


def load_adult() -> Dataset:
    raw = _download(ADULT_URL, DATA_DIR / "raw" / "adult.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        train_txt = z.read("adult.data").decode()
        # adult.test has a junk first line and labels ending in "."
        test_txt = z.read("adult.test").decode().split("\n", 1)[1]
    frames = [
        pl.read_csv(io.StringIO(t), has_header=False, new_columns=ADULT_COLS, infer_schema_length=0)
        for t in (train_txt, test_txt)
    ]
    df = pl.concat(frames).with_columns(pl.all().str.strip_chars())
    # Drop blank lines and the 3,620 rows with missing ("?") values.
    df = df.filter(pl.all_horizontal(pl.all().is_not_null() & (pl.all() != "?") & (pl.all() != "")))
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in ADULT_NUMERIC])
    df = df.with_columns(pl.col("income").str.strip_suffix(".").eq(">50K").cast(pl.Int64).alias("label"))
    age_band = pl.lit(None, dtype=pl.String)
    for lo, hi, name in reversed(AGE_BANDS):
        age_band = pl.when((pl.col("age") >= lo) & (pl.col("age") < hi)).then(pl.lit(name)).otherwise(age_band)
    df = df.with_columns(age_band.alias("age_band"))

    # Standardize numerics, one-hot categoricals. Sensitive attributes stay as model inputs,
    # as in the standard Adult setup; they are also kept separately for the subgroup analysis.
    cat_cols = [c for c in ADULT_COLS if c not in ADULT_NUMERIC and c != "income"]
    num = df.select(ADULT_NUMERIC).to_numpy()
    num = (num - num.mean(0)) / num.std(0)
    cat = df.select(cat_cols).to_dummies().to_numpy().astype(np.float64)
    x = torch.tensor(np.hstack([num, cat]), dtype=torch.float32)
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.int64)
    groups = df.select("sex", "race", "age_band", "age").with_row_index("record_id")
    return Dataset("adult", x, y, groups, 2, hashlib.sha256(raw).hexdigest())


LOADERS = {"adult": load_adult}


if __name__ == "__main__":
    for name, fn in LOADERS.items():
        d = fn()
        print(name, tuple(d.x.shape), "classes", d.num_classes, "pos/class balance",
              np.bincount(d.y.numpy()).tolist(), "sha256", d.source_sha256[:16])
        if name == "adult":
            for g in ("sex", "race", "age_band"):
                print("  ", d.groups[g].value_counts().sort("count", descending=True).rows())
