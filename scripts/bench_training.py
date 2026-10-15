"""Phase 1 timing benchmark: seconds per epoch (standard and DP-SGD) and per-sample
gradient-norm pass, per dataset and device. Output feeds the compute projection.

Usage: uv run python scripts/bench_training.py
"""

import json
import time
import warnings

import torch
from opacus import PrivacyEngine
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, TensorDataset

from leakatlas.data import LOADERS
from leakatlas.models import build_model, seed_everything

warnings.filterwarnings("ignore")
SIZES = {"adult": 10_000, "fmnist": 15_000}   # larger of the two planned training-set sizes
BATCH = {"adult": 256, "fmnist": 128}
EPOCHS = 3


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def epoch_time(model, loader, opt, dev):
    loss_fn = nn.CrossEntropyLoss()
    times = []
    for _ in range(EPOCHS + 1):  # first epoch is warm-up
        t = time.perf_counter()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sync(dev)
        times.append(time.perf_counter() - t)
    return sum(times[1:]) / EPOCHS


def grad_norm_time(model, x, y, dev, chunk=1024):
    params = {k: v.detach() for k, v in model.named_parameters()}
    loss_fn = nn.CrossEntropyLoss()

    def loss_one(p, xi, yi):
        return loss_fn(functional_call(model, p, (xi.unsqueeze(0),)), yi.unsqueeze(0))

    per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))
    t = time.perf_counter()
    for i in range(0, len(x), chunk):
        g = per_sample(params, x[i:i + chunk].to(dev), y[i:i + chunk].to(dev))
        torch.sqrt(sum((v.flatten(1) ** 2).sum(1) for v in g.values())).cpu()
    sync(dev)
    return time.perf_counter() - t


def main():
    devices = [torch.device("cpu")] + ([torch.device("mps")] if torch.backends.mps.is_available() else [])
    out = []
    for name, loader_fn in LOADERS.items():
        d = loader_fn()
        n = SIZES[name]
        x, y = d.x[:n], d.y[:n]
        for dev in devices:
            g = seed_everything(0)
            row = {"dataset": name, "device": dev.type, "train_size": n, "batch": BATCH[name]}
            model = build_model(name, d.x.shape[1] if name == "adult" else 0, d.num_classes).to(dev)
            dl = DataLoader(TensorDataset(x, y), batch_size=BATCH[name], shuffle=True, generator=g)
            row["sec_per_epoch_std"] = round(epoch_time(model, dl, torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9), dev), 3)
            row["sec_grad_norm_pass_2N"] = round(grad_norm_time(model, torch.cat([x, x]), torch.cat([y, y]), dev), 3)
            try:
                model = build_model(name, d.x.shape[1] if name == "adult" else 0, d.num_classes).to(dev)
                opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
                dl = DataLoader(TensorDataset(x, y), batch_size=BATCH[name], shuffle=True, generator=seed_everything(0))
                model, opt, dl = PrivacyEngine().make_private(module=model, optimizer=opt, data_loader=dl,
                                                              noise_multiplier=1.0, max_grad_norm=1.0)
                row["sec_per_epoch_dp"] = round(epoch_time(model, dl, opt, dev), 3)
            except Exception as e:  # report, don't hide, device incompatibilities
                row["sec_per_epoch_dp"] = None
                row["dp_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            print(json.dumps(row), flush=True)
            out.append(row)
    return out


if __name__ == "__main__":
    main()
