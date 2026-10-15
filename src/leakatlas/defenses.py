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


class MIAShield(nn.Module):
    """Ensemble on K disjoint subsets; the exact-signature oracle (ESE) drops the sub-model
    that trained on the query (Jarin & Eshete, PoPETs 2023). forward() returns log-probs."""

    def __init__(self, subs: list[nn.Module], subsets: list[torch.Tensor]):
        super().__init__()
        self.subs = nn.ModuleList(subs)
        self.signatures = {row.numpy().tobytes(): k for k, xs in enumerate(subsets) for row in xs}

    def oracle(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.signatures.get(r.numpy().tobytes(), -1) for r in x.cpu()])

    def forward(self, x, excl):
        logp = torch.stack([F.log_softmax(m(x), -1) for m in self.subs], 1)       # [B, K, C]
        keep = torch.arange(len(self.subs), device=x.device)[None] != excl[:, None]
        # log-mean-exp over kept sub-models: no log(0) when a probability underflows
        return torch.logsumexp(logp.masked_fill(~keep[..., None], float("-inf")), 1) - keep.sum(1, keepdim=True).log()


def log_probs(model: nn.Module, x: torch.Tensor, device, bs: int = 4096) -> torch.Tensor:
    """Served log-probabilities on CPU, applying MIAShield's oracle when present."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i:i + bs]
            if isinstance(model, MIAShield):
                out.append(model(xb.to(device), model.oracle(xb).to(device)).cpu())
            else:
                out.append(F.log_softmax(model(xb.to(device)), -1).cpu())
    return torch.cat(out)


def _sgd_train(model, x, y, epochs, lr, bs, device, seed, weight_decay=0.0, label_smoothing=0.0,
               val=None, patience=None):
    g = seed_everything(seed)
    dl = DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True, generator=g)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    best, best_state, bad, ran = float("inf"), None, 0, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            F.cross_entropy(model(xb.to(device)), yb.to(device), label_smoothing=label_smoothing).backward()
            opt.step()
        ran += 1
        if val is not None:  # early stopping on unsmoothed validation loss, restore best weights
            vl = F.nll_loss(log_probs(model, val[0], device), val[1]).item()
            if vl < best:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif (bad := bad + 1) >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs_run": ran}


def train_one(name: str, p: dict, d: Dataset, idx: np.ndarray, split: Split, cfg: ExperimentConfig, seed: int):
    dev = torch.device(cfg.device)
    x, y = d.x[idx], d.y[idx]
    new = lambda: build_model(d.name, d.x.shape[1], d.num_classes).to(dev)
    lr, bs = p.get("lr", cfg.lr), cfg.batch_size
    if name == "baseline":
        return _sgd_train(new(), x, y, p["epochs"], lr, bs, dev, seed)
    if name == "aof":
        val = (d.x[split.val], d.y[split.val])
        return _sgd_train(new(), x, y, p["epochs"], lr, bs, dev, seed, p["weight_decay"],
                          p["label_smoothing"], val, p["patience"])
    if name.startswith("dp_"):
        g = seed_everything(seed)
        model = new()
        dl = DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True, generator=g)
        engine = PrivacyEngine(accountant="rdp")
        delta = 1.0 / len(x)
        model, opt, dl = engine.make_private_with_epsilon(
            module=model, optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
            data_loader=dl, target_epsilon=p["epsilon"], target_delta=delta, epochs=p["epochs"],
            max_grad_norm=p["max_grad_norm"])
        for _ in range(p["epochs"]):
            model.train()
            for xb, yb in dl:
                if len(xb) == 0:  # Poisson sampling can yield an empty batch
                    continue
                opt.zero_grad()
                F.cross_entropy(model(xb.to(dev)), yb.to(dev)).backward()
                opt.step()
        eps_spent = engine.get_epsilon(delta)
        model._close()  # strip Opacus grad-sample hooks so the plain module works with torch.func
        return model._module, {"epochs_run": p["epochs"], "eps_spent": eps_spent,
                               "delta": delta, "noise_multiplier": opt.noise_multiplier}
    if name == "miashield":
        parts = np.array_split(np.random.default_rng(seed).permutation(len(idx)), p["k"])
        subs = [_sgd_train(new(), x[s], y[s], p["epochs"], lr, bs, dev, seed * 100 + k)[0]
                for k, s in enumerate(parts)]
        return MIAShield(subs, [x[s] for s in parts]), {"epochs_run": p["epochs"]}
    raise ValueError(f"unknown defense {name}")


def model_path(cfg: ExperimentConfig, size: int, defense: str, role: str) -> Path:
    return RUNS_DIR / cfg.dataset / f"S{size}" / defense / f"{role}.pt"


def train_all(cfg: ExperimentConfig, only: str | None = None):
    d = LOADERS[cfg.dataset]()
    for size in cfg.sizes:
        split = make_split(len(d.y), cfg, size)
        for name, p in cfg.defenses.items():
            if only and name != only:
                continue
            jobs = [(f"ref{r}", split.shadow[split.ref_keep[r]], 100 + r) for r in range(cfg.num_references)]
            jobs += [(f"target_seed{s}", split.members, s) for s in cfg.target_seeds]
            for role, idx, seed in jobs:
                path = model_path(cfg, size, name, role)
                if path.exists():
                    continue
                t = time.perf_counter()
                model, info = train_one(name, p, d, idx, split, cfg, seed)
                info |= {"train_sec": time.perf_counter() - t, "seed": seed, "train_size": len(idx)}
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.cpu(), path)
                path.with_suffix(".json").write_text(json.dumps(info))
                print(f"{cfg.dataset} S{size} {name:9s} {role:13s} {info['train_sec']:6.1f}s "
                      f"{json.dumps({k: v for k, v in info.items() if k in ('eps_spent', 'epochs_run')})}", flush=True)


if __name__ == "__main__":
    # Call through the importable module so pickled classes (MIAShield) resolve as leakatlas.defenses.*
    from leakatlas import defenses
    defenses.train_all(defenses.load_config(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
