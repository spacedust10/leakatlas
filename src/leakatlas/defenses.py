"""Data split, training recipes for every defense, and the training driver.

Usage: uv run python -m leakatlas.defenses configs/adult.yaml
Trained models are cached under data/runs/ and skipped on re-run (resumable).
"""

import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from opacus import PrivacyEngine
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from leakatlas.data import DATA_DIR, LOADERS, Dataset
from leakatlas.models import build_model, seed_everything

warnings.filterwarnings("ignore", module="opacus")
RUNS_DIR = DATA_DIR / "runs"


class ExperimentConfig(BaseModel):
    dataset: str
    device: str
    sizes: list[int]
    target_seeds: list[int]
    num_references: int
    defender_val: int
    batch_size: int
    lr: float
    defenses: dict[str, dict]


def load_config(path: str | Path) -> ExperimentConfig:
    cfg = ExperimentConfig(**yaml.safe_load(Path(path).read_text()))
    assert cfg.num_references % 2 == 0, "references are trained in complementary pairs"
    return cfg


@dataclass
class Split:
    """Index sets into the full dataset. Fixed once per dataset with seed 0."""

    members: np.ndarray       # target training set (size S)
    non_members: np.ndarray   # same-distribution held-out records (size S)
    shadow: np.ndarray        # attacker's pool for this size (size 2S)
    ref_keep: np.ndarray      # [R, 2S] bool: reference r trained on shadow[j]
    val: np.ndarray           # defender's early-stopping set
    population: np.ndarray    # RMIA z-samples, OUT for every model

    @property
    def audit(self) -> np.ndarray:
        return np.concatenate([self.members, self.non_members])


def make_split(n: int, cfg: ExperimentConfig, size: int) -> Split:
    big = max(cfg.sizes)
    perm = np.random.default_rng(0).permutation(n)
    target, shadow, rest = perm[:2 * big], perm[2 * big:4 * big], perm[4 * big:]
    assert len(rest) > cfg.defender_val, "dataset too small for the configured sizes"
    shadow = shadow[:2 * size]
    keep = np.zeros((cfg.num_references, 2 * size), bool)
    for p in range(cfg.num_references // 2):  # Privacy Meter pairing: complementary halves
        half = np.random.default_rng(1000 + p).permutation(2 * size)[:size]
        keep[2 * p, half] = True
        keep[2 * p + 1] = ~keep[2 * p]
    return Split(target[:size], target[big:big + size], shadow, keep,
                 rest[:cfg.defender_val], rest[cfg.defender_val:])
